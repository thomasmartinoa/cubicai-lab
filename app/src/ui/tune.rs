//! Fine-tune Studio: build a dataset (drafted by another AI, taken from chats or written by hand),
//! train a LoRA adapter on this GPU with Unsloth, then compare the model before and after and keep
//! the adapter on for chatting.

use crate::app::App;
use crate::engine::dataset::{self, Dataset, Example};
use crate::engine::server::State;
use crate::engine::train::{self, Adapter, EnvInfo, Live, Params};
use crate::engine::{library, xray, Task};
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, Kind};
use egui::{pos2, vec2, RichText, Sense, Stroke};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq)]
pub enum View {
    Data,
    Train,
    Try,
}

pub struct Draft {
    pub task: Arc<Task>,
    pub dataset: String,
    pub out: Arc<Mutex<Option<Vec<Example>>>>,
}

pub struct Run {
    pub task: Arc<Task>,
    pub live: Arc<Mutex<Live>>,
    pub out: Arc<Mutex<Option<Adapter>>>,
    pub dir: std::path::PathBuf,
}

pub enum Answer {
    Waiting,
    Asking(Receiver<Result<String, String>>),
    Done(String),
    Failed(String),
}

pub struct Trial {
    pub adapter: String,
    pub prompt: String,
    /// (adapter scale, answer); 0 is the base model.
    pub answers: Vec<(f32, Answer)>,
    pub started: bool,
}

pub struct TuneUi {
    pub view: View,
    pub datasets: Vec<Dataset>,
    pub adapters: Vec<Adapter>,
    pub dataset: Option<String>,
    pub draft_provider: String,
    pub draft_count: usize,
    pub draft: Option<Draft>,
    pub env: Option<Result<EnvInfo, String>>,
    pub env_task: Option<(Arc<Task>, Arc<Mutex<Option<Result<EnvInfo, String>>>>)>,
    pub model_key: Option<String>,
    pub base_repo: String,
    pub adapter_name: String,
    pub params: Params,
    pub run: Option<Run>,
    pub adapter: Option<String>,
    pub try_prompt: String,
    pub trial: Option<Trial>,
    pub error: Option<String>,
    pub build: crate::ui::build::BuildUi,
}

impl TuneUi {
    pub fn new(datasets: Vec<Dataset>, adapters: Vec<Adapter>) -> TuneUi {
        TuneUi {
            view: View::Data,
            dataset: datasets.first().map(|d| d.id.clone()),
            adapter: adapters.first().map(|a| a.id.clone()),
            datasets,
            adapters,
            draft_provider: "gemini".into(),
            draft_count: 20,
            draft: None,
            env: None,
            env_task: None,
            model_key: None,
            base_repo: String::new(),
            adapter_name: String::new(),
            params: Params::default(),
            run: None,
            try_prompt: String::new(),
            trial: None,
            error: None,
            build: Default::default(),
        }
    }

    pub fn training(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.task.is_done())
    }

    pub fn busy(&self) -> bool {
        self.training()
            || self.draft.as_ref().is_some_and(|d| !d.task.is_done())
            || self.env_task.as_ref().is_some_and(|(t, _)| !t.is_done())
            || self.build.busy()
            || self.trial.as_ref().is_some_and(|t| t.answers.iter().any(|(_, a)| matches!(a, Answer::Waiting | Answer::Asking(_))))
    }

    /// Stops everything now (the app is closing).
    pub fn stop(&mut self) {
        for t in [self.run.as_ref().map(|r| &r.task), self.draft.as_ref().map(|d| &d.task), self.env_task.as_ref().map(|e| &e.0), self.build.job.as_ref().map(|j| &j.task)].into_iter().flatten() {
            t.stop();
        }
    }

    fn current(&mut self) -> Option<&mut Dataset> {
        let id = self.dataset.clone()?;
        self.datasets.iter_mut().find(|d| d.id == id)
    }
}

/// Advances drafting, set-up, training and trials; called every frame from the app loop.
pub fn pump(app: &mut App, ctx: &egui::Context) {
    let t = &mut app.tune;
    if let Some(d) = &t.draft
        && d.task.is_done()
    {
        let d = t.draft.take().unwrap();
        if let Some(new) = d.out.lock().unwrap().take()
            && let Some(ds) = t.datasets.iter_mut().find(|x| x.id == d.dataset)
        {
            crate::engine::datakit::merge(ds, new, vec![]);
            let _ = app.datasets.save(ds);
        }
        if let Some(e) = d.task.error().filter(|e| e != "Stopped") {
            t.error = Some(e);
        }
    }
    if let Some((task, out)) = &t.env_task
        && task.is_done()
    {
        t.env = out.lock().unwrap().take().or_else(|| task.error().map(Err));
        t.env_task = None;
    }
    if let Some(r) = &t.run
        && r.task.is_done()
    {
        let done = r.out.lock().unwrap().take();
        if let Some(a) = done {
            t.adapter = Some(a.id.clone());
            t.adapters.insert(0, a);
            t.view = View::Try;
        } else if let Some(e) = r.task.error().filter(|e| e != "Stopped") {
            t.error = Some(e);
        }
        // Keep the finished run on screen (its curve and log) until the next one starts.
    }
    crate::ui::build::pump(app);
    pump_trial(app, ctx);
}

fn pump_trial(app: &mut App, ctx: &egui::Context) {
    let Some(tr) = app.tune.trial.as_mut() else { return };
    if tr.answers.iter().all(|(_, a)| matches!(a, Answer::Done(_) | Answer::Failed(_))) {
        return;
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(250));
    let Some(a) = app.tune.adapters.iter().find(|a| a.id == tr.adapter).cloned() else { return };
    let gguf = app.adapters.gguf(&a.id);
    let ready = app.server.is_ready_with(&a.model_key) && app.server.lora.as_deref() == Some(gguf.as_path());
    if !tr.started {
        tr.started = true;
        if !ready {
            let steer = app.preset_steer_file(&a.model_key).ok().flatten();
            if let Err(e) = app.start_model_with(&a.model_key, steer.as_deref(), Some(&gguf)) {
                for (_, ans) in &mut app.tune.trial.as_mut().unwrap().answers {
                    *ans = Answer::Failed(e.clone());
                }
            }
        }
        return;
    }
    if !ready {
        if let State::Failed(e) = app.server.state() {
            for (_, ans) in &mut tr.answers {
                *ans = Answer::Failed(e.clone());
            }
        }
        return;
    }
    let root = xray::root_of(&app.server.base_url());
    for (scale, ans) in &mut tr.answers {
        match ans {
            Answer::Waiting => {
                let (tx, rx) = channel();
                let (root, scale, ctx) = (root.clone(), *scale, ctx.clone());
                let mut messages = vec![];
                if !a.system.trim().is_empty() {
                    messages.push(json!({"role": "system", "content": a.system.trim()}));
                }
                messages.push(json!({"role": "user", "content": tr.prompt.trim()}));
                std::thread::spawn(move || {
                    let r = xray::apply_template(&root, &messages).and_then(|p| train::complete_with_lora(&root, &p, 200, scale));
                    let _ = tx.send(r);
                    ctx.request_repaint();
                });
                *ans = Answer::Asking(rx);
                // One at a time: llama-server runs a single slot.
                return;
            }
            Answer::Asking(rx) => {
                if let Ok(r) = rx.try_recv() {
                    *ans = match r {
                        Ok(s) => Answer::Done(s),
                        Err(e) => Answer::Failed(e),
                    };
                }
                return;
            }
            _ => {}
        }
    }
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    if app.tune.busy() {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(300));
    }
    egui::Panel::left("tune-left")
        .resizable(false)
        .exact_size(300.0)
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(16))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| sidebar(app, ui, &p));
        });
    egui::CentralPanel::no_frame().show(ui, |ui| {
        egui::ScrollArea::vertical().auto_shrink(false).id_salt("tune-main").show(ui, |ui| {
            egui::Frame::new().inner_margin(16).show(ui, |ui| {
                let mut view = app.tune.view;
                widgets::segmented(ui, &mut view, &[(View::Data, "1  Data"), (View::Train, "2  Train"), (View::Try, "3  Try it")]);
                app.tune.view = view;
                ui.add_space(12.0);
                if let Some(e) = app.tune.error.clone() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(e).color(p.error));
                        if ui.small_button("Dismiss").clicked() {
                            app.tune.error = None;
                        }
                    });
                    ui.add_space(8.0);
                }
                match view {
                    View::Data => data_view(app, ui, &p),
                    View::Train => train_view(app, ui, &p),
                    View::Try => try_view(app, ui, &p),
                }
            });
        });
    });
}

fn label(ui: &mut egui::Ui, p: &Palette, text: &str) {
    ui.label(RichText::new(text).font(theme::medium(13.0)).color(p.on_variant));
}

fn sidebar(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Fine-tune").font(theme::heading(20.0)));
    widgets::hint(ui, "Teach a model new habits: collect example conversations, train a small adapter on your GPU, and switch it on.");
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        label(ui, p, "Datasets");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("New").clicked() {
                let d = Dataset::new(&format!("Dataset {}", app.tune.datasets.len() + 1));
                let _ = app.datasets.save(&d);
                app.tune.dataset = Some(d.id.clone());
                app.tune.datasets.insert(0, d);
                app.tune.view = View::Data;
            }
        });
    });
    if app.tune.datasets.is_empty() {
        widgets::hint(ui, "None yet: press New.");
    }
    let mut pick = None;
    for d in &app.tune.datasets {
        let sel = app.tune.dataset.as_deref() == Some(d.id.as_str());
        let text = format!("{}  ·  {}", d.name, d.valid_count());
        if ui.add(egui::Button::selectable(sel, text).min_size(vec2(ui.available_width(), 26.0))).clicked() {
            pick = Some(d.id.clone());
        }
    }
    if let Some(id) = pick {
        app.tune.dataset = Some(id);
        app.tune.view = View::Data;
    }
    ui.add_space(14.0);
    label(ui, p, "Adapters");
    if app.tune.adapters.is_empty() {
        widgets::hint(ui, "Trained adapters appear here.");
    }
    let mut pick = None;
    for a in &app.tune.adapters {
        let sel = app.tune.adapter.as_deref() == Some(a.id.as_str());
        let in_chat = app.settings.preset(&a.model_key).lora.as_deref() == Some(a.id.as_str());
        let text = format!("{}{}", a.name, if in_chat { "  · in chat" } else { "" });
        if ui.add(egui::Button::selectable(sel, text).min_size(vec2(ui.available_width(), 26.0))).on_hover_text(format!("{} on {}", a.dataset, a.model_name)).clicked() {
            pick = Some(a.id.clone());
        }
    }
    if let Some(id) = pick {
        app.tune.adapter = Some(id);
        app.tune.view = View::Try;
    }
}

// --- 1 Data ------------------------------------------------------------------------------------

fn data_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(ds) = app.tune.current().cloned() else {
        empty(ui, "Start with examples", "Make a dataset (New, on the left): conversations showing exactly how the model should answer.");
        return;
    };
    let mut d = ds.clone();
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut d.name).font(theme::heading(18.0)).desired_width(280.0));
        let held = if d.held_count() > 0 { format!(" · {} held back", d.held_count()) } else { String::new() };
        let pairs = if d.pair_count() > 0 { format!(" · {}", crate::ui::arena::plural(d.pair_count(), "pair")) } else { String::new() };
        widgets::hint(ui, format!("{} to train on{held}{pairs} · about {} tokens", d.train_count(), widgets::fmt_count(d.approx_tokens() as u64)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::button(ui, Kind::Danger, "Delete").clicked() {
                app.datasets.delete(&d.id);
                app.tune.datasets.retain(|x| x.id != d.id);
                app.tune.dataset = app.tune.datasets.first().map(|x| x.id.clone());
            }
            crate::ui::build::header_tools(app, ui, &mut d);
        });
    });
    if !app.tune.datasets.iter().any(|x| x.id == d.id) {
        return;
    }
    ui.add_space(6.0);
    crate::ui::build::versions(app, ui, p, &mut d);
    label(ui, p, "System prompt (optional, sent with every example)");
    ui.add(egui::TextEdit::multiline(&mut d.system).desired_rows(2).desired_width(f32::INFINITY).hint_text("You are…"));
    ui.add_space(10.0);
    crate::ui::build::panel(app, ui, p, &mut d);
    ui.add_space(12.0);
    crate::ui::build::examples(app, ui, p, &mut d);
    if d != ds {
        d.ensure_ids();
        let _ = app.datasets.save(&d);
        if let Some(x) = app.tune.current() {
            *x = d;
        }
    }
}

/// Drafting by another AI from a brief (the "AI draft" source).
pub fn draft_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    label(ui, p, "What should the model learn?");
    ui.add(
        egui::TextEdit::multiline(&mut d.brief)
            .desired_rows(3)
            .desired_width(f32::INFINITY)
            .hint_text("e.g. Answer as a cheerful pirate, in two sentences at most, and always end with a nautical tip."),
    );
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Ask");
        let cur = crate::providers::get(&app.tune.draft_provider);
        egui::ComboBox::from_id_salt("draft-provider").selected_text(cur.short).show_ui(ui, |ui| {
            for pr in crate::providers::ALL.iter().filter(|pr| pr.id != crate::providers::PC) {
                ui.selectable_value(&mut app.tune.draft_provider, pr.id.to_string(), pr.short);
            }
        });
        let model = app.settings.models.get(&app.tune.draft_provider).cloned().unwrap_or_else(|| "no model chosen".into());
        widgets::hint(ui, model);
        ui.add_space(8.0);
        ui.label("for");
        ui.add(egui::DragValue::new(&mut app.tune.draft_count).range(5..=50));
        ui.label("examples");
    });
    ui.add_space(4.0);
    match &app.tune.draft {
        Some(dr) if !dr.task.is_done() => {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new());
                ui.label(dr.task.status());
                if ui.small_button("Stop").clicked() {
                    dr.task.stop();
                }
            });
        }
        _ => {
            if widgets::button_ext(ui, Kind::Primary, "Draft", !d.brief.trim().is_empty() && !app.tune.build.busy()).clicked() {
                let _ = app.datasets.snapshot(d, "Before drafting");
                start_draft(app, d, ui.ctx());
            }
        }
    }
    widgets::hint(ui, "Uses the provider's model from Settings. Review what comes back: the model learns exactly these replies. The brief also guides grading, documents and rewrites.");
}

fn start_draft(app: &mut App, d: &Dataset, ctx: &egui::Context) {
    let messages = dataset::draft_messages(&d.brief, &d.system, app.tune.draft_count, &d.examples);
    let req = match app.provider_request(&app.tune.draft_provider.clone(), messages) {
        Ok(r) => r,
        Err(e) => {
            app.tune.error = Some(e);
            return;
        }
    };
    let task = Task::new("Asking for examples…");
    let out: Arc<Mutex<Option<Vec<Example>>>> = Default::default();
    let slot = out.clone();
    let want = app.tune.draft_count;
    app.start_task(&task, ctx, move |t| {
        let cancel = AtomicBool::new(false);
        let mut got: Vec<Example> = vec![];
        // Big asks come back in batches of up to 25, so replies aren't cut off.
        while got.len() < want {
            if t.cancel.load(Ordering::Relaxed) {
                break;
            }
            let n = (want - got.len()).min(25);
            t.set_status(format!("Asking for {n} examples ({} so far)…", got.len()));
            let mut r = req.clone();
            if let Some(m) = r.messages.last_mut() {
                let text = m["content"].as_str().unwrap_or("").replacen(&format!("Write {want} "), &format!("Write {n} "), 1);
                m["content"] = json!(text);
            }
            let reply = crate::llm::collect(&r, &cancel)?;
            let mut batch = dataset::parse_draft(&reply)?;
            batch.retain(|e| !got.iter().any(|g| g.turns == e.turns));
            if batch.is_empty() {
                break;
            }
            got.extend(batch);
        }
        got.truncate(want);
        *slot.lock().unwrap() = Some(got);
        Ok(())
    });
    app.tune.draft = Some(Draft { task, dataset: d.id.clone(), out });
}

// --- 2 Train -----------------------------------------------------------------------------------

fn train_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    env_card(app, ui, p);
    ui.add_space(10.0);
    // While (or after) training, its progress and controls come first.
    let has_run = app.tune.run.is_some();
    if has_run {
        actions(app, ui);
        if let Some(r) = &app.tune.run {
            ui.add_space(10.0);
            run_card(ui, p, r);
        }
        ui.add_space(10.0);
    }

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "What to train");
        egui::Grid::new("train-what").num_columns(2).spacing(vec2(10.0, 8.0)).show(ui, |ui| {
            ui.label("Dataset");
            let name = app.tune.current().map(|d| format!("{} ({} examples, {} pairs)", d.name, d.train_count(), d.pair_count())).unwrap_or_else(|| "Make one in step 1".into());
            egui::ComboBox::from_id_salt("train-ds").selected_text(name).width(320.0).show_ui(ui, |ui| {
                for d in &app.tune.datasets {
                    if ui.selectable_label(app.tune.dataset.as_deref() == Some(d.id.as_str()), &d.name).clicked() {
                        app.tune.dataset = Some(d.id.clone());
                    }
                }
            });
            ui.end_row();

            ui.label("Model");
            let key = app.tune.model_key.clone().or_else(|| app.server.model_key.clone());
            let name = key.as_ref().and_then(|k| app.library.get(k)).map(|m| crate::app::short_model_name(&m.name)).unwrap_or_else(|| "Pick a model".into());
            egui::ComboBox::from_id_salt("train-model").selected_text(name).width(320.0).height(420.0).show_ui(ui, |ui| {
                for m in &app.library.models {
                    if ui.selectable_label(key.as_deref() == Some(m.key.as_str()), crate::app::short_model_name(&m.name)).clicked() {
                        app.tune.model_key = Some(m.key.clone());
                        let full = app.library.info(&m.key).and_then(|i| i.gguf.get_str("general.name").map(str::to_string)).unwrap_or_else(|| m.name.clone());
                        app.tune.base_repo = train::suggest_base(&full).or_else(|| train::suggest_base(&m.name)).unwrap_or("").to_string();
                    }
                }
            });
            ui.end_row();

            ui.label("Trains on");
            ui.add(egui::TextEdit::singleline(&mut app.tune.base_repo).desired_width(320.0).hint_text("Hugging Face repo, e.g. unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"));
            ui.end_row();

            ui.label("Adapter name");
            let default_name = app.tune.current().map(|d| d.name.clone()).unwrap_or_default();
            ui.add(egui::TextEdit::singleline(&mut app.tune.adapter_name).desired_width(320.0).hint_text(default_name));
            ui.end_row();
        });
        widgets::hint(ui, "Training uses the original Hugging Face weights (Unsloth's 4-bit copy of the same model, downloaded once, about 5.5 GB for 8B). The adapter is then converted for your GGUF.");
    });
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Settings");
        ui.horizontal(|ui| {
            ui.label("Learn from");
            let pr = &mut app.tune.params;
            let mut dpo = pr.dpo();
            widgets::segmented(ui, &mut dpo, &[(false, "Examples"), (true, "Preferences (DPO)")]);
            pr.method = if dpo { "dpo".into() } else { "sft".into() };
            if dpo {
                ui.add_space(12.0);
                ui.label("Beta");
                ui.add(egui::Slider::new(&mut pr.beta, 0.01..=0.5).logarithmic(true));
            }
        });
        widgets::hint(ui, if app.tune.params.dpo() {
            "DPO: from each pair the model learns to prefer the better reply over the worse one, without copying either word for word. Lower beta lets it move further from how it started. A smaller learning rate (around 5e-6 to 5e-5) suits it."
        } else {
            "The model learns to give the examples' replies."
        });
        ui.add_space(4.0);
        let pr = &mut app.tune.params;
        ui.spacing_mut().slider_width = 160.0;
        egui::Grid::new("train-params").num_columns(4).spacing(vec2(12.0, 8.0)).show(ui, |ui| {
            ui.label("Rank");
            ui.add(egui::Slider::new(&mut pr.rank, 4..=128).logarithmic(true));
            ui.label("Epochs");
            ui.add(egui::Slider::new(&mut pr.epochs, 1.0..=10.0).step_by(0.5));
            ui.end_row();
            ui.label("Alpha");
            ui.add(egui::Slider::new(&mut pr.alpha, 4.0..=256.0).logarithmic(true));
            ui.label("Learning rate");
            ui.add(egui::Slider::new(&mut pr.lr, 1e-5..=1e-3).logarithmic(true).custom_formatter(|v, _| format!("{v:.0e}")));
            ui.end_row();
            ui.label("Max length");
            ui.add(egui::Slider::new(&mut pr.max_seq, 256..=4096).logarithmic(true));
            ui.label("Batch");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut pr.batch).range(1..=16));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut pr.grad_accum).range(1..=32));
            });
            ui.end_row();
        });
        if !pr.dpo() {
            widgets::switch_row(ui, "Learn from replies only", "The user's messages are context, not something to imitate.", &mut pr.replies_only);
        }
        let dpo = app.tune.params.dpo();
        let examples = app.tune.current().map(|d| if dpo { d.pair_count() } else { d.train_count() }).unwrap_or(0);
        let steps = ((examples as f32 * app.tune.params.epochs) / (app.tune.params.batch * app.tune.params.grad_accum).max(1) as f32).ceil();
        widgets::hint(ui, format!("{examples} {} × {} epochs ≈ {steps:.0} steps. On 8 GB, keep 8B models at max length 1024 and batch 2{}.", if dpo { "pairs" } else { "examples" }, app.tune.params.epochs, if dpo { " (DPO holds two replies per pair: try batch 1)" } else { "" }));
    });
    if !has_run {
        ui.add_space(10.0);
        actions(app, ui);
    }
}

/// Start, or Stop while training, with the reason when it can't start.
fn actions(app: &mut App, ui: &mut egui::Ui) {
    let mut start = false;
    let env_ok = matches!(&app.tune.env, Some(Ok(i)) if i.cuda);
    let dpo = app.tune.params.dpo();
    let examples = app.tune.current().map(|d| if dpo { d.pair_count() } else { d.train_count() }).unwrap_or(0);
    let key = app.tune.model_key.clone().or_else(|| app.server.model_key.clone());
    let arch_ok = key.as_ref().and_then(|k| app.library.info(k)).is_some_and(|i| ["llama", "qwen2", "qwen3"].contains(&i.gguf.arch()));
    ui.horizontal(|ui| {
        if app.tune.training() {
            let r = app.tune.run.as_ref().unwrap();
            if widgets::button(ui, Kind::Tonal, "Stop and keep").on_hover_text("Finishes the current step and saves what it learned so far").clicked() {
                let _ = std::fs::write(r.dir.join("STOP"), b"");
            }
            if widgets::button(ui, Kind::Ghost, "Stop now").clicked() {
                r.task.stop();
            }
        } else {
            let ready = env_ok && examples >= 5 && key.is_some() && arch_ok && !app.tune.base_repo.trim().is_empty();
            start = widgets::button_ext(ui, Kind::Primary, "Start training", ready).clicked();
            let why = if !env_ok {
                "Set up the trainer first."
            } else if examples < 5 && dpo {
                "Preference training needs at least 5 pairs (vote in the Arena, or import prompt/chosen/rejected)."
            } else if examples < 5 {
                "The dataset needs at least 5 usable examples."
            } else if key.is_none() {
                "Pick the model to train."
            } else if !arch_ok {
                "Adapters can be converted for Llama, Mistral and Qwen 2/3 models."
            } else if app.tune.base_repo.trim().is_empty() {
                "Name the Hugging Face model to train on."
            } else {
                "Stops the local model while training: the GPU is needed."
            };
            widgets::hint(ui, why);
        }
    });
    if start && let Some(k) = key {
        start_training(app, &k, ui.ctx());
    }
}

fn env_card(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    if app.tune.env.is_none() && app.tune.env_task.is_none() && app.train_env.exists() {
        // Look once, in the background (importing torch takes a few seconds).
        start_env(app, false, ui.ctx());
    }
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Trainer");
        match (&app.tune.env_task, &app.tune.env) {
            (Some((t, _)), _) => {
                ui.label(RichText::new(t.status()).small());
                ui.add(egui::ProgressBar::new(t.fraction()).desired_height(6.0));
                if ui.small_button("Stop").clicked() {
                    t.stop();
                }
            }
            (None, Some(Ok(i))) => {
                let gpu = if i.cuda { format!("{} ({})", i.gpu, widgets::fmt_bytes(i.vram)) } else { "no GPU visible to PyTorch".into() };
                ui.label(format!("Unsloth {} · PyTorch {} · {gpu}", if i.unsloth.is_empty() { "missing" } else { &i.unsloth }, i.torch));
                ui.horizontal(|ui| {
                    if ui.small_button("Check again").clicked() {
                        start_env(app, false, ui.ctx());
                    }
                    if ui.small_button("Update").on_hover_text("Reinstalls the latest Unsloth").clicked() {
                        start_env(app, true, ui.ctx());
                    }
                });
            }
            (None, Some(Err(e))) => {
                ui.label(RichText::new(e).small().color(p.error));
                if widgets::button(ui, Kind::Primary, "Set up the trainer").clicked() {
                    start_env(app, true, ui.ctx());
                }
            }
            (None, None) => {
                widgets::hint(ui, format!(
                    "Training runs in a private Python environment with Unsloth and PyTorch (about 5-8 GB, in {}). Nothing is installed system-wide.",
                    app.train_env.dir.display()
                ));
                if widgets::button(ui, Kind::Primary, "Set up the trainer").clicked() {
                    start_env(app, true, ui.ctx());
                }
            }
        }
    });
}

fn start_env(app: &mut App, install: bool, ctx: &egui::Context) {
    let task = Task::new(if install { "Setting up" } else { "Checking the trainer" });
    let out: Arc<Mutex<Option<Result<EnvInfo, String>>>> = Default::default();
    let slot = out.clone();
    let env = train::Env::new(app.train_env.dir.clone());
    app.start_task(&task, ctx, move |t| {
        let r = if install { env.setup(t) } else { env.check() };
        *slot.lock().unwrap() = Some(r);
        Ok(())
    });
    app.tune.env_task = Some((task, out));
}

fn start_training(app: &mut App, key: &str, ctx: &egui::Context) {
    let Some(ds) = app.tune.current().cloned() else { return };
    let Some(model) = app.library.get(key).cloned() else { return };
    // The GPU is needed: stop our server and ask Ollama to let go.
    app.server.stop();
    crate::ui::engine::unload_ollama(ctx);
    let id = crate::store::new_id();
    let dir = app.adapters.run_dir(&id);
    let meta = Adapter {
        id: id.clone(),
        name: if app.tune.adapter_name.trim().is_empty() { ds.name.clone() } else { app.tune.adapter_name.trim().into() },
        dataset: ds.name.clone(),
        dataset_id: ds.id.clone(),
        examples: if app.tune.params.dpo() { ds.pair_count() } else { ds.train_count() },
        base_repo: app.tune.base_repo.trim().into(),
        model_key: key.into(),
        model_name: crate::app::short_model_name(&model.name),
        params: app.tune.params.clone(),
        created: crate::store::now_ms(),
        system: ds.system.clone(),
        ..Default::default()
    };
    let (python, hf) = (app.train_env.python(), app.secrets.get("hf"));
    let cached = app.library.info(key);
    let adapters_dir = app.adapters.dir.clone();
    let task = Task::new("Starting the trainer");
    let live: Arc<Mutex<Live>> = Default::default();
    let out: Arc<Mutex<Option<Adapter>>> = Default::default();
    let (live2, slot, dir2) = (live.clone(), out.clone(), dir.clone());
    app.start_task(&task, ctx, move |t| {
        std::fs::create_dir_all(&dir2).map_err(|e| e.to_string())?;
        let data = dir2.join("data.jsonl");
        std::fs::write(&data, if meta.params.dpo() { ds.pairs_jsonl() } else { ds.to_jsonl() }).map_err(|e| e.to_string())?;
        let cfg = train::config_json(&meta.base_repo, &data, &dir2, &meta.params);
        std::fs::write(dir2.join("train.json"), serde_json::to_vec_pretty(&cfg).unwrap()).map_err(|e| e.to_string())?;
        let started = std::time::Instant::now();
        let peft = train::run(&python, &dir2, &hf, &live2, t)?;
        t.set_status("Converting the adapter for llama.cpp".into());
        let info = match cached {
            Some(i) => i,
            None => Arc::new(library::read_info(&model)?),
        };
        let store = train::Adapters::new(adapters_dir);
        let c = store.convert(&meta.id, &peft, &info.gguf)?;
        let mut meta = meta;
        meta.lora_params = c.params;
        meta.lora_pairs = c.pairs;
        meta.losses = live2.lock().unwrap().losses.clone();
        meta.secs = started.elapsed().as_secs_f32();
        store.save(&meta)?;
        *slot.lock().unwrap() = Some(meta);
        Ok(())
    });
    app.tune.error = None;
    app.tune.run = Some(Run { task, live, out, dir });
}

fn run_card(ui: &mut egui::Ui, p: &Palette, r: &Run) {
    widgets::card(ui, |ui| {
        let l = r.live.lock().unwrap();
        let state = if !r.task.is_done() {
            l.status.clone()
        } else if let Some(e) = r.task.error() {
            e
        } else {
            "Done: the adapter is ready to try.".into()
        };
        widgets::card_title(ui, "Training");
        ui.label(RichText::new(state).small().color(if r.task.error().is_some() { p.error } else { p.on_bg }));
        if l.total > 0 {
            widgets::hint(ui, format!("step {} of {}", l.step, l.total));
            ui.add(egui::ProgressBar::new(l.step as f32 / l.total as f32).desired_height(6.0));
        }
        ui.add_space(6.0);
        loss_chart(ui, p, &l.losses, 160.0);
        ui.add_space(6.0);
        let tail: Vec<&String> = l.log.iter().rev().take(6).collect();
        for line in tail.into_iter().rev() {
            ui.label(RichText::new(line).monospace().small().color(p.on_variant));
        }
    });
}

/// Training loss by step, with the first and latest value labelled.
/// A centred moving average of the loss curve (window about a tenth of the run, shrinking at the
/// ends so the first and last points stay put). Empty when the run is too short to need one.
pub fn smoothed(losses: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let half = losses.len() / 20;
    if half == 0 {
        return vec![];
    }
    (0..losses.len())
        .map(|i| {
            let k = half.min(i).min(losses.len() - 1 - i);
            let w = &losses[i - k..=i + k];
            (losses[i].0, w.iter().map(|x| x.1).sum::<f32>() / w.len() as f32)
        })
        .collect()
}

/// "44 s", "7 min", "1 h 12 min".
pub fn took(secs: f32) -> String {
    let s = secs.round() as u64;
    match s {
        0..90 => format!("{s} s"),
        90..5400 => format!("{} min", (s + 30) / 60),
        _ => format!("{} h {} min", s / 3600, (s % 3600 + 30) / 60),
    }
}

pub fn loss_chart(ui: &mut egui::Ui, p: &Palette, losses: &[(u32, f32)], h: f32) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, p.low);
    if losses.len() < 2 {
        widgets::text_at(&painter, rect.center(), egui::Align2::CENTER_CENTER, "the loss curve appears here", egui::FontId::proportional(12.0), p.on_variant);
        return;
    }
    let (lo, hi) = losses.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &(_, l)| (a.min(l), b.max(l)));
    let span = (hi - lo).max(1e-6);
    let last = losses.last().unwrap().0.max(1) as f32;
    let first = losses[0].0 as f32;
    let inner = rect.shrink2(vec2(10.0, 16.0));
    let pt = |s: u32, l: f32| pos2(inner.left() + (s as f32 - first) / (last - first).max(1.0) * inner.width(), inner.bottom() - (l - lo) / span * inner.height());
    // Raw points faint, a centred moving average on top. Centred so it doesn't lag behind the loss
    // (a trailing average ends well above the real final loss on a short run); short runs, where
    // there's nothing to smooth, get just the raw line.
    let raw: Vec<_> = losses.iter().map(|&(s, l)| pt(s, l)).collect();
    let smooth = smoothed(losses);
    if smooth.is_empty() {
        painter.add(egui::Shape::line(raw, Stroke::new(2.0, p.primary)));
    } else {
        painter.add(egui::Shape::line(raw, Stroke::new(1.0, p.primary.gamma_multiply(0.35))));
        painter.add(egui::Shape::line(smooth.iter().map(|&(s, l)| pt(s, l)).collect(), Stroke::new(2.0, p.primary)));
    }
    let f = egui::FontId::proportional(11.0);
    widgets::text_at(&painter, pos2(rect.left() + 6.0, rect.top() + 3.0), egui::Align2::LEFT_TOP, &format!("loss {:.3}", losses[0].1), f.clone(), p.on_variant);
    let (s, l) = *losses.last().unwrap();
    widgets::text_at(&painter, pos2(rect.right() - 6.0, rect.top() + 3.0), egui::Align2::RIGHT_TOP, &format!("step {s}: {l:.3}"), f, p.on_bg);
}

// --- 3 Try it ----------------------------------------------------------------------------------

fn try_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(a) = app.tune.adapter.as_ref().and_then(|id| app.tune.adapters.iter().find(|a| &a.id == id)).cloned() else {
        empty(ui, "Nothing trained yet", "Train an adapter in step 2, then compare the model with and without it here.");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(&a.name).font(theme::heading(20.0)));
        let fin = a.losses.last().map(|l| format!(" · final loss {:.3}", l.1)).unwrap_or_default();
        widgets::hint(ui, format!(
            "{} · {} examples · rank {} adapting {} weights ({} numbers) · {}{fin}",
            a.model_name,
            a.examples,
            a.params.rank,
            a.lora_pairs,
            widgets::fmt_count(a.lora_params),
            took(a.secs)
        ));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::button(ui, Kind::Danger, "Delete").clicked() {
                app.adapters.delete(&a.id);
                app.tune.adapters.retain(|x| x.id != a.id);
                app.tune.adapter = app.tune.adapters.first().map(|x| x.id.clone());
                for preset in app.settings.presets.values_mut() {
                    if preset.lora.as_deref() == Some(a.id.as_str()) {
                        preset.lora = None;
                    }
                }
                app.save_settings();
            }
        });
    });
    if !app.tune.adapters.iter().any(|x| x.id == a.id) {
        return;
    }
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "How it learned");
        loss_chart(ui, p, &a.losses, 120.0);
    });
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Before and after");
        widgets::hint(ui, "The same prompt with the adapter off and on. No restart between them: llama-server switches the adapter per request.");
        if !a.system.trim().is_empty() {
            widgets::hint(ui, format!("System prompt from the dataset: {}", a.system.trim()));
        }
        let running = app.tune.trial.as_ref().is_some_and(|t| t.answers.iter().any(|(_, x)| matches!(x, Answer::Waiting | Answer::Asking(_))));
        ui.horizontal(|ui| {
            ui.label("Prompt");
            ui.add_enabled(!running, egui::TextEdit::singleline(&mut app.tune.try_prompt).desired_width(f32::INFINITY).hint_text("Ask something the dataset covers"));
        });
        if widgets::button_ext(ui, Kind::Primary, "Compare", !running && !app.tune.try_prompt.trim().is_empty() && !app.tune.training()).clicked() {
            app.tune.trial = Some(Trial {
                adapter: a.id.clone(),
                prompt: app.tune.try_prompt.clone(),
                answers: vec![(0.0, Answer::Waiting), (1.0, Answer::Waiting)],
                started: false,
            });
        }
        let Some(tr) = app.tune.trial.as_ref().filter(|t| t.adapter == a.id) else { return };
        ui.add_space(8.0);
        ui.columns(tr.answers.len(), |cols| {
            for ((scale, ans), col) in tr.answers.iter().zip(cols.iter_mut()) {
                let tint = if *scale > 0.0 { p.primary } else { p.outline };
                widgets::surface(col, p.low, Stroke::new(1.0, tint.gamma_multiply(0.7)), 8.0, 10, |ui| {
                    ui.set_width(ui.available_width());
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        ui.label(RichText::new(if *scale > 0.0 { "With the adapter" } else { "Without (the base model)" }).font(theme::medium(14.0)).color(tint));
                        match ans {
                            Answer::Waiting => widgets::hint(ui, "waiting for the model…"),
                            Answer::Asking(_) => widgets::hint(ui, "writing…"),
                            Answer::Done(s) => ui.label(s),
                            Answer::Failed(e) => ui.label(RichText::new(e).small().color(p.error)),
                        };
                    });
                });
            }
        });
    });
    ui.add_space(10.0);
    crate::ui::arena::report_card(app, ui, p, &a);
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Use while chatting");
        let mut preset = app.settings.preset(&a.model_key);
        let on = preset.lora.as_deref() == Some(a.id.as_str());
        widgets::hint(ui, if on {
            "On: this model loads with the adapter wherever it runs (chat, X-ray, Steering)."
        } else {
            "Loads the adapter with this model wherever it runs. Use the dataset's system prompt in chat for the closest match."
        });
        ui.horizontal(|ui| {
            let mut changed = false;
            if !on && widgets::button(ui, Kind::Primary, "Use in chat").clicked() {
                preset.lora = Some(a.id.clone());
                changed = true;
            }
            if on && widgets::button(ui, Kind::Ghost, "Stop using").clicked() {
                preset.lora = None;
                changed = true;
            }
            if changed {
                app.settings.presets.insert(a.model_key.clone(), preset);
                app.save_settings();
                if app.server.model_key.as_deref() == Some(a.model_key.as_str())
                    && app.server.state() != State::Stopped
                    && let Err(e) = app.start_model(&a.model_key)
                {
                    app.tune.error = Some(e);
                }
            }
        });
    });
}

fn empty(ui: &mut egui::Ui, title: &str, text: &str) {
    ui.add_space(60.0);
    ui.vertical_centered(|ui| {
        widgets::cube_mark(ui, 54.0, false);
        ui.add_space(8.0);
        ui.label(RichText::new(title).font(theme::heading(22.0)));
        ui.add_space(4.0);
        widgets::hint(ui, text);
    });
}
