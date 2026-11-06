//! App state and the chat logic (sending, streaming, regenerating). Screens are in `ui/`.

use crate::engine::{self, gpu::GpuMonitor, library::Library, server::Server, Task};
use crate::llm::{self, Ev};
use crate::store::{Chat, ChatStore, Message, Paths, Settings, ThemeChoice};
use crate::{personas, providers, secrets::Secrets, ui};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Engine,
    Anatomy,
    Settings,
}

pub struct Job {
    pub id: u64,
    pub chat: String,
    pub msg: String,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
}

/// A reply waiting for the local server to finish loading its model.
pub struct Pending {
    pub chat: String,
    pub msg: String,
    pub model: String,
}

#[derive(Clone)]
pub enum ModelList {
    Loading,
    Ready(Vec<(String, Option<u64>)>),
    Failed(String),
}

pub struct ChatState {
    pub chats: Vec<Chat>,
    pub current: Option<String>,
    pub input: String,
    pub job: Option<Job>,
    pub pending: Option<Pending>,
    pub tx: Sender<(u64, Ev)>,
    pub rx: Receiver<(u64, Ev)>,
    next_job: u64,
    pub md: egui_commonmark::CommonMarkCache,
    pub model_lists: Arc<Mutex<HashMap<String, ModelList>>>,
    pub search: String,
    pub renaming: Option<(String, String)>,
    pub focus_input: bool,
}

pub struct App {
    pub paths: Paths,
    pub settings: Settings,
    pub secrets: Secrets,
    pub store: ChatStore,
    pub tab: Tab,
    pub chat: ChatState,
    pub library: Library,
    pub server: Server,
    pub gpu: GpuMonitor,
    pub engine_ui: ui::engine::EngineUi,
    pub anatomy: ui::anatomy::AnatomyUi,
    pub settings_ui: ui::settings::SettingsUi,
    applied_theme: Option<ThemeChoice>,
    pub toast: Option<(String, Instant)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        crate::theme::install(&cc.egui_ctx);
        let paths = Paths::new();
        let settings = Settings::load(&paths);
        let secrets = Secrets::new(&paths);
        let store = ChatStore::new(&paths);
        let chats = store.load_all();
        let (tx, rx) = channel();
        let mut app = App {
            chat: ChatState {
                current: None,
                chats,
                input: String::new(),
                job: None,
                pending: None,
                tx,
                rx,
                next_job: 1,
                md: Default::default(),
                model_lists: Default::default(),
                search: String::new(),
                renaming: None,
                focus_input: true,
            },
            library: Library::new(),
            server: Server::new(),
            gpu: GpuMonitor::new(),
            engine_ui: Default::default(),
            anatomy: Default::default(),
            settings_ui: Default::default(),
            tab: Tab::Chat,
            applied_theme: None,
            toast: None,
            paths,
            settings,
            secrets,
            store,
        };
        app.rescan(&cc.egui_ctx);
        app
    }

    pub fn save_settings(&self) {
        self.settings.save(&self.paths);
    }

    pub fn rescan(&mut self, ctx: &egui::Context) {
        let mut dirs = vec![self.paths.models.clone()];
        dirs.extend(self.settings.model_dirs.iter().cloned());
        let ollama = if self.settings.show_ollama { self.settings.ollama_dir.clone().or_else(engine::library::find_ollama_dir) } else { None };
        self.library.scan(&dirs, ollama, ctx);
        if !self.settings.show_ollama {
            self.library.models.retain(|m| m.source == engine::library::Source::Folder);
        }
    }

    pub fn notify(&mut self, s: impl Into<String>) {
        self.toast = Some((s.into(), Instant::now()));
    }

    /// The folder with llama-server: the user's own build, else the installed one.
    pub fn bin_dir(&self) -> Option<std::path::PathBuf> {
        if let Some(d) = &self.settings.llama_custom_dir
            && engine::server::bin_dir_valid(d) {
                return Some(d.clone());
            }
        self.settings.llama.as_ref().map(|b| b.dir.clone()).filter(|d| engine::server::bin_dir_valid(d))
    }

    pub fn build_label(&self) -> String {
        if self.settings.llama_custom_dir.as_ref().is_some_and(|d| engine::server::bin_dir_valid(d)) {
            return "own build".into();
        }
        self.settings.llama.as_ref().map(|b| format!("{} · {}", b.tag, b.variant)).unwrap_or_else(|| "not installed".into())
    }

    /// Starts llama-server with the model's saved settings.
    pub fn start_model(&mut self, key: &str) -> Result<(), String> {
        let bin = self.bin_dir().ok_or("llama.cpp isn't installed yet: open the Engine Room and install it.")?;
        let m = self.library.get(key).cloned().ok_or("That model is no longer in the library.")?;
        let preset = self.settings.preset(key);
        let display = short_model_name(&m.name);
        self.server.start(engine::server::Launch {
            bin_dir: &bin,
            model_key: key,
            model_name: &display,
            model: m.path(),
            mmproj: m.mmproj.as_deref(),
            preset: &preset,
            port: self.settings.port,
        })
    }

    // --- chat ------------------------------------------------------------------------------------

    pub fn current_chat(&self) -> Option<&Chat> {
        let id = self.chat.current.as_ref()?;
        self.chat.chats.iter().find(|c| &c.id == id)
    }

    pub fn chat_mut(&mut self, id: &str) -> Option<&mut Chat> {
        self.chat.chats.iter_mut().find(|c| c.id == id)
    }

    pub fn save_chat(&self, id: &str) {
        if let Some(c) = self.chat.chats.iter().find(|c| c.id == id) {
            self.store.save(c);
        }
    }

    pub fn new_chat(&mut self) {
        // Reuse an untouched empty chat rather than piling them up.
        if let Some(c) = self.chat.chats.iter().find(|c| c.messages.is_empty()) {
            self.chat.current = Some(c.id.clone());
        } else {
            let c = Chat { persona_id: self.settings.persona.clone(), ..Default::default() };
            self.chat.current = Some(c.id.clone());
            self.chat.chats.insert(0, c);
        }
        self.chat.input.clear();
        self.chat.focus_input = true;
    }

    pub fn open_chat(&mut self, id: &str) {
        self.chat.current = Some(id.to_string());
        if let Some(c) = self.chat.chats.iter().find(|c| c.id == id) {
            // A chat brings back the behaviour and model it used.
            self.settings.persona = c.persona_id.clone();
            if let (Some(p), Some(m)) = (&c.provider_id, &c.model)
                && providers::ALL.iter().any(|x| x.id == p) {
                    self.settings.provider = p.clone();
                    self.settings.models.insert(p.clone(), m.clone());
                }
        }
        self.chat.focus_input = true;
    }

    pub fn delete_chat(&mut self, id: &str) {
        if self.chat.job.as_ref().is_some_and(|j| j.chat == id) {
            self.stop();
        }
        self.store.delete(id);
        self.chat.chats.retain(|c| c.id != id);
        if self.chat.current.as_deref() == Some(id) {
            self.chat.current = None;
        }
    }

    pub fn busy(&self) -> bool {
        self.chat.job.is_some() || self.chat.pending.is_some()
    }

    pub fn send(&mut self, ctx: &egui::Context) {
        let text = self.chat.input.trim().to_string();
        if text.is_empty() || self.busy() {
            return;
        }
        if self.current_chat().is_none() {
            self.new_chat();
        }
        let id = self.chat.current.clone().unwrap();
        let persona = self.settings.persona.clone();
        let chat = self.chat_mut(&id).unwrap();
        if chat.messages.is_empty() {
            chat.title = title_from(&text);
        }
        chat.persona_id = persona;
        chat.messages.push(Message { role: "user".into(), content: text, ..Default::default() });
        self.chat.input.clear();
        self.reply(&id, ctx);
    }

    /// Adds an empty reply to the chat and starts filling it.
    fn reply(&mut self, chat_id: &str, ctx: &egui::Context) {
        let provider = self.settings.provider.clone();
        let model_id = self.settings.models.get(&provider).cloned().unwrap_or_default();
        let model_label = if provider == providers::PC {
            self.library.get(&model_id).map(|m| short_model_name(&m.name)).unwrap_or_else(|| "local model".into())
        } else {
            model_id.clone()
        };
        let persona_name = self.persona_name(&self.settings.persona.clone());
        let msg = Message { role: "assistant".into(), model: Some(model_label), persona: persona_name, ..Default::default() };
        let msg_id = msg.id.clone();
        let chat = self.chat_mut(chat_id).unwrap();
        chat.provider_id = Some(provider.clone());
        chat.model = Some(model_id.clone());
        chat.updated_at = crate::store::now_ms();
        chat.messages.push(msg);
        self.bump(chat_id);
        self.save_chat(chat_id);

        if provider == providers::PC {
            if model_id.is_empty() || self.library.get(&model_id).is_none() {
                self.fail(chat_id, &msg_id, "Pick a model on this PC first (the model menu above, or the Engine Room).");
                return;
            }
            if !self.server.is_ready_with(&model_id) {
                let loading = self.server.model_key.as_deref() == Some(model_id.as_str()) && self.server.state() == engine::server::State::Loading;
                if !loading
                    && let Err(e) = self.start_model(&model_id) {
                        self.fail(chat_id, &msg_id, &e);
                        return;
                    }
                self.chat.pending = Some(Pending { chat: chat_id.to_string(), msg: msg_id, model: model_id });
                return;
            }
        }
        self.dispatch(chat_id, &msg_id, ctx);
    }

    fn bump(&mut self, chat_id: &str) {
        if let Some(i) = self.chat.chats.iter().position(|c| c.id == chat_id) {
            let c = self.chat.chats.remove(i);
            self.chat.chats.insert(0, c);
        }
    }

    fn fail(&mut self, chat_id: &str, msg_id: &str, err: &str) {
        if let Some(m) = self.chat_mut(chat_id).and_then(|c| c.messages.iter_mut().find(|m| m.id == msg_id)) {
            m.error = Some(err.to_string());
        }
        self.save_chat(chat_id);
    }

    pub fn persona_name(&self, id: &str) -> Option<String> {
        if id == "none" {
            return None;
        }
        personas::BUILT_IN
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.to_string())
            .or_else(|| self.settings.custom_personas.iter().find(|p| p.id == id).map(|p| p.name.clone()))
    }

    pub fn system_prompt(&self, persona: &str) -> String {
        if persona == "none" {
            return String::new();
        }
        let mut s = personas::BUILT_IN
            .iter()
            .find(|p| p.id == persona)
            .map(|p| p.instructions.to_string())
            .or_else(|| self.settings.custom_personas.iter().find(|p| p.id == persona).map(|p| p.instructions.clone()))
            .unwrap_or_default();
        if !self.settings.about_me.trim().is_empty() {
            s.push_str("\n\nAbout the user (use when relevant):\n");
            s.push_str(self.settings.about_me.trim());
        }
        s
    }

    /// Sends the history up to `msg_id` and streams into it.
    fn dispatch(&mut self, chat_id: &str, msg_id: &str, ctx: &egui::Context) {
        let Some(chat) = self.chat.chats.iter().find(|c| c.id == chat_id) else { return };
        let provider = providers::get(&self.settings.provider);
        let mut messages = vec![];
        let sys = self.system_prompt(&chat.persona_id);
        if !sys.is_empty() {
            messages.push(json!({"role": "system", "content": sys}));
        }
        for m in chat.messages.iter().take_while(|m| m.id != msg_id) {
            if m.role == "assistant" && (m.content.is_empty() || m.error.is_some()) {
                continue;
            }
            messages.push(json!({"role": m.role, "content": m.content}));
        }
        let (base_url, model, local) = if provider.id == providers::PC {
            (self.server.base_url(), self.server.model_name.clone(), true)
        } else if provider.id == "custom" {
            (self.settings.custom_base_url.clone(), self.settings.models.get("custom").cloned().unwrap_or_default(), false)
        } else {
            (provider.base_url.to_string(), self.settings.models.get(provider.id).cloned().unwrap_or_default(), provider.id == "ollama")
        };
        if model.is_empty() {
            self.fail(chat_id, msg_id, "Choose a model first.");
            return;
        }
        if base_url.is_empty() {
            self.fail(chat_id, msg_id, "Set the server address for the custom provider in Settings.");
            return;
        }
        let key = if provider.needs_key { self.secrets.get(provider.id) } else { String::new() };
        if provider.needs_key && key.is_empty() && provider.id != "custom" {
            self.fail(chat_id, msg_id, &format!("Add your {} API key in Settings.", provider.short));
            return;
        }
        let id = self.chat.next_job;
        self.chat.next_job += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.chat.job = Some(Job { id, chat: chat_id.into(), msg: msg_id.into(), cancel: cancel.clone(), started: Instant::now() });
        llm::stream(llm::Request { base_url, key, model, messages, local }, id, self.chat.tx.clone(), cancel, ctx.clone());
    }

    pub fn stop(&mut self) {
        if let Some(j) = self.chat.job.take() {
            j.cancel.store(true, Ordering::Relaxed);
            let took = j.started.elapsed().as_millis() as u64;
            if let Some(m) = self.chat_mut(&j.chat).and_then(|c| c.messages.iter_mut().find(|m| m.id == j.msg)) {
                m.stopped = true;
                m.took = took;
            }
            self.save_chat(&j.chat);
        }
        if let Some(p) = self.chat.pending.take() {
            // Nothing was written; drop the empty reply.
            if let Some(c) = self.chat_mut(&p.chat) {
                c.messages.retain(|m| m.id != p.msg);
            }
            self.save_chat(&p.chat);
        }
    }

    /// Replaces the last reply with a new one.
    pub fn regenerate(&mut self, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let Some(id) = self.chat.current.clone() else { return };
        let Some(c) = self.chat_mut(&id) else { return };
        if c.messages.last().is_some_and(|m| m.role == "assistant") {
            c.messages.pop();
        }
        if c.messages.last().is_some_and(|m| m.role == "user") {
            self.reply(&id, ctx);
        }
    }

    /// Puts a user message back in the composer and drops it and everything after it.
    pub fn edit_message(&mut self, msg_id: &str) {
        if self.busy() {
            return;
        }
        let Some(id) = self.chat.current.clone() else { return };
        let Some(c) = self.chat_mut(&id) else { return };
        if let Some(i) = c.messages.iter().position(|m| m.id == msg_id) {
            let text = c.messages[i].content.clone();
            c.messages.truncate(i);
            self.chat.input = text;
            self.chat.focus_input = true;
            self.save_chat(&id);
        }
    }

    /// Applies streamed events and starts replies that were waiting for the server.
    fn pump(&mut self, ctx: &egui::Context) {
        while let Ok((job, ev)) = self.chat.rx.try_recv() {
            let Some(j) = self.chat.job.as_ref().filter(|j| j.id == job) else { continue };
            let (chat_id, msg_id, started) = (j.chat.clone(), j.msg.clone(), j.started);
            let mut finished = false;
            if let Some(m) = self.chat_mut(&chat_id).and_then(|c| c.messages.iter_mut().find(|m| m.id == msg_id)) {
                match ev {
                    Ev::Content(s) => m.content.push_str(&s),
                    Ev::Reasoning(s) => m.reasoning.push_str(&s),
                    Ev::Usage { prompt, output } => {
                        m.prompt_tokens = prompt;
                        m.output_tokens = output;
                    }
                    Ev::Speed { gen_tps } => m.tps = Some(gen_tps),
                    Ev::Done => {
                        m.took = started.elapsed().as_millis() as u64;
                        finished = true;
                    }
                    Ev::Error(e) => {
                        m.error = Some(e);
                        m.took = started.elapsed().as_millis() as u64;
                        finished = true;
                    }
                }
            }
            if finished {
                self.chat.job = None;
                self.save_chat(&chat_id);
            }
        }
        if let Some(p) = &self.chat.pending {
            match self.server.state() {
                engine::server::State::Ready if self.server.model_key.as_deref() == Some(p.model.as_str()) => {
                    let p = self.chat.pending.take().unwrap();
                    self.dispatch(&p.chat, &p.msg, ctx);
                }
                engine::server::State::Failed(e) => {
                    let p = self.chat.pending.take().unwrap();
                    self.fail(&p.chat, &p.msg, &e);
                }
                engine::server::State::Stopped => {
                    let p = self.chat.pending.take().unwrap();
                    self.fail(&p.chat, &p.msg, "The model was stopped before it finished loading.");
                }
                _ => ctx.request_repaint_after(std::time::Duration::from_millis(200)),
            }
        }
    }

    /// Fetches a provider's model list in the background.
    pub fn fetch_models(&mut self, provider_id: &str, ctx: &egui::Context) {
        let p = providers::get(provider_id);
        let base = if p.id == "custom" { self.settings.custom_base_url.clone() } else { p.base_url.to_string() };
        if base.is_empty() {
            return;
        }
        let key = if p.needs_key { self.secrets.get(p.id) } else { String::new() };
        let lists = self.chat.model_lists.clone();
        let id = p.id.to_string();
        lists.lock().unwrap().insert(id.clone(), ModelList::Loading);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let r = match llm::list_models(&base, &key) {
                Ok(l) => ModelList::Ready(l),
                Err(e) => ModelList::Failed(e),
            };
            lists.lock().unwrap().insert(id, r);
            ctx.request_repaint();
        });
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if self.applied_theme != Some(self.settings.theme) {
            ctx.set_theme(match self.settings.theme {
                ThemeChoice::System => egui::ThemePreference::System,
                ThemeChoice::Light => egui::ThemePreference::Light,
                ThemeChoice::Dark => egui::ThemePreference::Dark,
            });
            self.applied_theme = Some(self.settings.theme);
        }
    }

    pub fn start_task(&mut self, task: &Arc<Task>, ctx: &egui::Context, f: impl FnOnce(&Arc<Task>) -> Result<(), String> + Send + 'static) {
        engine::spawn(task, ctx, f);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.apply_theme(&ctx);
        self.pump(&ctx);
        self.server.poll();
        if self.server.state() == engine::server::State::Loading {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        ui::frame(self, ui);
    }

    fn on_exit(&mut self) {
        self.server.stop();
        self.save_settings();
    }
}


pub fn title_from(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or(text).trim();
    let mut t: String = line.chars().take(48).collect();
    if line.chars().count() > 48 {
        t.push('…');
    }
    t
}

/// `hf.co/user/Some-Model-GGUF:Q4_K_M` → `Some-Model-GGUF:Q4_K_M`; file names lose `.gguf`.
pub fn short_model_name(name: &str) -> String {
    let n = name.rsplit('/').next().unwrap_or(name);
    n.trim_end_matches(".gguf").to_string()
}
