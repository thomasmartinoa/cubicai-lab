//! The Arena: pick 2–4 models, ask them the same prompts, and see the answers side by side with
//! their speed, length and differences. Vote (blind if you like), let an AI judge score them,
//! keep prompt sets, and watch a leaderboard of your own picks.

use crate::app::App;
use crate::engine::arena::{self, Answer, Judge, PromptSet, Run, SetPrompt, Vote, Who};
use crate::engine::server::State;
use crate::llm::{self, Ev};
use crate::providers;
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, Kind};
use egui::{RichText, Stroke};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

pub const MAX_CONTESTANTS: usize = 4;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum View {
    Setup,
    Run,
    Board,
    Set,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RunView {
    Answers,
    Vote,
}

/// What "Add" adds.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source {
    Library,
    Adapter,
    Provider,
}

/// The comparison being set up.
pub struct Draft {
    pub contestants: Vec<Who>,
    pub prompt: String,
    pub set: Option<String>,
    pub system: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub blind: bool,
    pub judge_on: bool,
    pub judge: Judge,
    pub source: Source,
    pub provider: String,
    pub model: String,
    pub library_key: Option<String>,
    pub adapter: Option<String>,
    pub scale: f32,
}

impl Default for Draft {
    fn default() -> Self {
        Draft {
            contestants: vec![],
            prompt: String::new(),
            set: None,
            system: String::new(),
            temperature: 0.7,
            max_tokens: 0,
            blind: false,
            judge_on: false,
            judge: Judge::default(),
            source: Source::Provider,
            provider: "ollama".into(),
            model: String::new(),
            library_key: None,
            adapter: None,
            scale: 1.0,
        }
    }
}

struct Job {
    item: usize,
    c: usize,
    started: Instant,
    cancel: Arc<AtomicBool>,
}

/// Where the local lane is: handing the GPU over or loading a model before it can ask.
enum Local {
    Idle,
    Freeing { c: usize, rx: Receiver<Result<(), String>> },
    Loading { c: usize },
}

type JudgeResult = (usize, Vec<usize>, Result<String, String>);

/// A run being answered (and judged) right now.
pub struct Live {
    pub run: String,
    /// Events with the moment they arrived (stamped off the UI thread, so frames don't round them).
    tx: Sender<(u64, Ev, Instant)>,
    rx: Receiver<(u64, Ev, Instant)>,
    next_job: u64,
    jobs: HashMap<u64, Job>,
    local: Local,
    /// The Ollama contestant whose model is loaded and warm.
    ollama_ready: Option<usize>,
    /// The model our server ran before, to bring back afterwards.
    restore: Option<String>,
    judging: Option<(Receiver<JudgeResult>, Arc<AtomicBool>)>,
    judge_after: bool,
}

pub struct ArenaUi {
    pub view: View,
    pub run_view: RunView,
    pub runs: Vec<Run>,
    pub sets: Vec<PromptSet>,
    pub open: Option<String>,
    pub set: Option<String>,
    pub draft: Draft,
    /// Who judges report cards.
    pub report_judge: Judge,
    pub live: Option<Live>,
    pub diff: bool,
    pub thinking: bool,
    pub vote_at: usize,
    pub paste: String,
    pub error: Option<String>,
}

impl ArenaUi {
    pub fn new(runs: Vec<Run>, sets: Vec<PromptSet>) -> ArenaUi {
        ArenaUi {
            view: if runs.is_empty() { View::Setup } else { View::Run },
            run_view: RunView::Answers,
            open: runs.first().map(|r| r.id.clone()),
            set: None,
            runs,
            sets,
            draft: Draft::default(),
            report_judge: Judge::default(),
            live: None,
            diff: false,
            thinking: false,
            vote_at: 0,
            paste: String::new(),
            error: None,
        }
    }

    pub fn busy(&self) -> bool {
        self.live.is_some()
    }

    pub fn stop(&mut self) {
        if let Some(l) = &self.live {
            for j in l.jobs.values() {
                j.cancel.store(true, Ordering::Relaxed);
            }
            if let Some((_, c)) = &l.judging {
                c.store(true, Ordering::Relaxed);
            }
        }
    }

    fn run_mut(&mut self, id: &str) -> Option<&mut Run> {
        self.runs.iter_mut().find(|r| r.id == id)
    }

    /// What a live run is doing: Some(false) answering, Some(true) judging.
    pub fn phase(&self, run_id: &str) -> Option<bool> {
        let l = self.live.as_ref().filter(|l| l.run == run_id)?;
        Some(l.judging.is_some() || (l.judge_after && l.jobs.is_empty() && self.runs.iter().any(|r| r.id == run_id && r.finished())))
    }

    pub fn current(&self) -> Option<&Run> {
        let id = self.open.as_ref()?;
        self.runs.iter().find(|r| &r.id == id)
    }
}

// --- running -------------------------------------------------------------------------------------

/// Starts answering what's left of a run (a new one, or one that was stopped).
pub fn start(app: &mut App, run_id: &str, judge_after: bool) {
    if app.arena.live.is_some() {
        return;
    }
    let Some(run) = app.arena.runs.iter().find(|r| r.id == run_id) else { return };
    let uses_gpu = run.contestants.iter().any(|w| w.local());
    let restore = if uses_gpu && app.server.state() != State::Stopped { app.server.model_key.clone() } else { None };
    let (tx, rx) = channel();
    app.arena.live = Some(Live {
        run: run_id.to_string(),
        tx,
        rx,
        next_job: 1,
        jobs: HashMap::new(),
        local: Local::Idle,
        ollama_ready: None,
        restore,
        judging: None,
        judge_after,
    });
}

/// Stops the live run: answers being written keep what they have and are marked stopped.
pub fn stop(app: &mut App) {
    let Some(live) = app.arena.live.take() else { return };
    for j in live.jobs.values() {
        j.cancel.store(true, Ordering::Relaxed);
        if let Some(a) = app.arena.run_mut(&live.run).and_then(|r| r.items.get_mut(j.item)).and_then(|i| i.answers.get_mut(j.c)) {
            a.stopped = true;
            a.took = j.started.elapsed().as_millis() as u64;
        }
    }
    if let Some((_, c)) = &live.judging {
        c.store(true, Ordering::Relaxed);
    }
    save(app, &live.run);
    restore_server(app, live.restore);
}

fn save(app: &mut App, run_id: &str) {
    if let Some(r) = app.arena.runs.iter().find(|r| r.id == run_id)
        && let Err(e) = app.arena_store.save_run(r)
    {
        app.arena.error = Some(format!("Couldn't save the run: {e}"));
    }
}

fn restore_server(app: &mut App, key: Option<String>) {
    let Some(k) = key else { return };
    let preset = app.settings.preset(&k);
    let as_saved = app.server.model_key.as_deref() == Some(k.as_str())
        && app.server.state() != State::Stopped
        && app.server.lora == app.preset_lora_path(&k, &preset)
        && app.server.steer == app.preset_steer_path(&k, &preset);
    if !as_saved && let Err(e) = app.start_model(&k) {
        app.notify(e);
    }
}

/// Advances the live run; called every frame from the app loop.
pub fn pump(app: &mut App, ctx: &egui::Context) {
    let Some(live) = app.arena.live.as_mut() else { return };
    ctx.request_repaint_after(std::time::Duration::from_millis(200));
    let run_id = live.run.clone();
    // Streamed events into the answers.
    let mut finished = vec![];
    let events: Vec<(u64, Ev, Instant)> = live.rx.try_iter().collect();
    for (job, ev, at) in events {
        let Some(j) = app.arena.live.as_ref().and_then(|l| l.jobs.get(&job)) else { continue };
        let (item, c, started) = (j.item, j.c, j.started);
        let Some(a) = app.arena.runs.iter_mut().find(|r| r.id == run_id).and_then(|r| r.items.get_mut(item)).and_then(|i| i.answers.get_mut(c)) else { continue };
        let ms = at.saturating_duration_since(started).as_millis() as u64;
        match ev {
            Ev::Content(s) => {
                a.first_ms.get_or_insert(ms);
                a.text.push_str(&s);
            }
            Ev::Reasoning(s) => {
                a.first_ms.get_or_insert(ms);
                a.reasoning.push_str(&s);
            }
            Ev::Usage { prompt, output } => {
                a.prompt_tokens = prompt;
                a.output_tokens = output;
            }
            Ev::Speed { gen_tps } => a.tps = Some(gen_tps),
            Ev::Done => {
                a.done = true;
                a.took = ms;
                finished.push((job, c));
            }
            Ev::Error(e) => {
                a.done = true;
                a.error = Some(e);
                a.took = ms;
                finished.push((job, c));
            }
        }
    }
    if !finished.is_empty() {
        app.gpu.refresh();
        for (job, c) in finished {
            app.arena.live.as_mut().unwrap().jobs.remove(&job);
            let vram = vram_of(app, &run_id, c);
            if let Some(r) = app.arena.run_mut(&run_id) {
                for it in &mut r.items {
                    if it.answers[c].done && it.answers[c].vram.is_none() && it.answers[c].error.is_none() {
                        it.answers[c].vram = vram;
                    }
                }
            }
        }
        save(app, &run_id);
    }
    let Some(run) = app.arena.runs.iter().find(|r| r.id == run_id).cloned() else {
        app.arena.live = None;
        return;
    };
    // Cloud contestants: each answers its prompts in turn, all of them at once.
    for c in (0..run.contestants.len()).filter(|&c| !run.contestants[c].local()) {
        let live = app.arena.live.as_ref().unwrap();
        if live.jobs.values().any(|j| j.c == c) {
            continue;
        }
        if let Some(i) = arena::next_for(&run, c) {
            ask(app, &run, i, c, ctx);
        }
    }
    pump_local(app, &run, ctx);
    pump_judge(app, &run_id, ctx);
    // Everything answered (and judged): hand the GPU back.
    let live = app.arena.live.as_ref().unwrap();
    let run = app.arena.runs.iter().find(|r| r.id == run_id).unwrap();
    if live.jobs.is_empty() && run.finished() && live.judging.is_none() && !(live.judge_after && needs_judging(run)) {
        let live = app.arena.live.take().unwrap();
        save(app, &run_id);
        restore_server(app, live.restore);
    }
}

fn needs_judging(run: &Run) -> bool {
    run.judge.is_some() && run.items.iter().any(|i| i.verdict.is_none() && i.answers.iter().filter(|a| a.ok()).count() >= 2)
}

/// The GPU memory the contestant's model holds now (our server's process, or Ollama's).
fn vram_of(app: &App, run_id: &str, c: usize) -> Option<u64> {
    let run = app.arena.runs.iter().find(|r| r.id == run_id)?;
    let who = run.contestants.get(c)?;
    let procs = &app.gpu.info.processes;
    if who.is_ollama() {
        // Ollama's runner is a llama-server too (/usr/lib/ollama/llama-server): tell it from ours by its path.
        let ours = app.server.pid();
        let used: u64 = procs
            .iter()
            .filter(|p| Some(p.pid) != ours && (p.name.to_lowercase().contains("ollama") || crate::engine::gpu::process_exe(p.pid).contains("ollama")))
            .map(|p| p.used)
            .sum();
        return (used > 0).then_some(used);
    }
    if who.server_setup().is_some() {
        let pid = app.server.pid()?;
        return procs.iter().find(|p| p.pid == pid).map(|p| p.used).filter(|u| *u > 0);
    }
    None
}

/// What our server must run for a contestant: (model key, steering file, adapter file).
fn wanted(app: &App, who: &Who) -> Option<(String, Option<std::path::PathBuf>, Option<std::path::PathBuf>)> {
    match who {
        Who::Library { key, .. } => {
            let preset = app.settings.preset(key);
            Some((key.clone(), app.preset_steer_path(key, &preset), app.preset_lora_path(key, &preset)))
        }
        Who::Adapter { id, model_key, .. } => Some((model_key.clone(), None, Some(app.adapters.gguf(id)))),
        Who::Provider { .. } => None,
    }
}

fn server_has(app: &App, w: &(String, Option<std::path::PathBuf>, Option<std::path::PathBuf>)) -> bool {
    app.server.is_ready_with(&w.0) && app.server.steer == w.1 && app.server.lora == w.2
}

fn fail_contestant(app: &mut App, run_id: &str, c: usize, e: &str) {
    if let Some(r) = app.arena.run_mut(run_id) {
        for it in &mut r.items {
            if !it.answers[c].done {
                it.answers[c] = Answer { done: true, error: Some(e.to_string()), ..Default::default() };
            }
        }
    }
    save(app, run_id);
}

/// The GPU lane: one local answer at a time, handing the GPU between Ollama and our server.
fn pump_local(app: &mut App, run: &Run, ctx: &egui::Context) {
    let live = app.arena.live.as_mut().unwrap();
    if live.jobs.values().any(|j| run.contestants[j.c].local()) {
        return;
    }
    match &live.local {
        Local::Freeing { c, rx } => {
            let c = *c;
            match rx.try_recv() {
                Ok(Ok(())) => {
                    live.local = Local::Idle;
                    if run.contestants[c].is_ollama() {
                        live.ollama_ready = Some(c);
                    } else if let Some(w) = wanted(app, &run.contestants[c]) {
                        let r = if let Who::Library { key, .. } = &run.contestants[c] {
                            let k = key.clone();
                            app.preset_steer_file(&k).and_then(|s| app.start_model_with(&k, s.as_deref(), w.2.as_deref()))
                        } else {
                            app.start_model_with(&w.0, None, w.2.as_deref())
                        };
                        match r {
                            Ok(()) => app.arena.live.as_mut().unwrap().local = Local::Loading { c },
                            Err(e) => fail_contestant(app, &run.id, c, &e),
                        }
                    }
                }
                Ok(Err(e)) => {
                    live.local = Local::Idle;
                    fail_contestant(app, &run.id, c, &e);
                }
                Err(_) => {}
            }
            return;
        }
        Local::Loading { c } => {
            let c = *c;
            match app.server.state() {
                State::Ready => live.local = Local::Idle,
                State::Failed(e) => {
                    live.local = Local::Idle;
                    fail_contestant(app, &run.id, c, &e);
                }
                State::Stopped => {
                    live.local = Local::Idle;
                    fail_contestant(app, &run.id, c, "The model was stopped before it finished loading.");
                }
                State::Loading => {}
            }
            return;
        }
        Local::Idle => {}
    }
    let Some((i, c)) = arena::next_local(run) else { return };
    let who = &run.contestants[c];
    let ready = if who.is_ollama() {
        live.ollama_ready == Some(c)
    } else {
        wanted(app, who).is_some_and(|w| server_has(app, &w))
    };
    if ready {
        ask(app, run, i, c, ctx);
        return;
    }
    // Hand the GPU over first: our server off for Ollama, Ollama's models out for ours.
    let keep = match who {
        Who::Provider { model, .. } => {
            app.server.stop();
            Some(model.clone())
        }
        _ => None,
    };
    let live = app.arena.live.as_mut().unwrap();
    live.ollama_ready = None;
    let (tx, rx) = channel();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let _ = tx.send(arena::hand_gpu_to_ollama(keep.as_deref()));
        ctx.request_repaint();
    });
    live.local = Local::Freeing { c, rx };
}

/// Starts one answer streaming.
fn ask(app: &mut App, run: &Run, item: usize, c: usize, ctx: &egui::Context) {
    let messages = run.messages(item);
    let req = match &run.contestants[c] {
        Who::Library { .. } => Ok(llm::Request { base_url: app.server.base_url(), model: app.server.model_name.clone(), messages, local: true, ..Default::default() }),
        Who::Adapter { scale, .. } => {
            let mut r = llm::Request { base_url: app.server.base_url(), model: app.server.model_name.clone(), messages, local: true, ..Default::default() };
            r.extra.insert("lora".into(), json!([{"id": 0, "scale": scale}]));
            Ok(r)
        }
        Who::Provider { provider, model } => app.request_for(provider, Some(model), messages),
    };
    let fresh = Answer::default();
    let Some(a) = app.arena.run_mut(&run.id).and_then(|r| r.items.get_mut(item)).and_then(|i| i.answers.get_mut(c)) else { return };
    *a = fresh;
    let mut req = match req {
        Ok(r) => r,
        Err(e) => {
            *a = Answer { done: true, error: Some(e), ..Default::default() };
            return;
        }
    };
    req.extra.extend(run.sampling());
    let live = app.arena.live.as_mut().unwrap();
    let job = live.next_job;
    live.next_job += 1;
    let cancel = Arc::new(AtomicBool::new(false));
    live.jobs.insert(job, Job { item, c, started: Instant::now(), cancel: cancel.clone() });
    let (tx, rx) = channel();
    let stamped = live.tx.clone();
    std::thread::spawn(move || {
        for (j, ev) in rx {
            if stamped.send((j, ev, Instant::now())).is_err() {
                break;
            }
        }
    });
    llm::stream(req, job, tx, cancel, ctx.clone());
}

/// Judges every answered prompt that has no verdict yet, one request after another.
fn pump_judge(app: &mut App, run_id: &str, ctx: &egui::Context) {
    let live = app.arena.live.as_mut().unwrap();
    if let Some((rx, _)) = &live.judging {
        let mut done = false;
        let mut got = vec![];
        loop {
            match rx.try_recv() {
                Ok(r) => got.push(r),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    done = true;
                    break;
                }
            }
        }
        if done {
            live.judging = None;
            live.judge_after = false;
        }
        if !got.is_empty() {
            if let Some(r) = app.arena.run_mut(run_id) {
                for (i, shown, reply) in got {
                    let parsed = reply.and_then(|t| arena::parse_verdict(&t, &shown));
                    arena::apply_verdict(&mut r.items[i], &shown, parsed);
                }
            }
            save(app, run_id);
        }
        return;
    }
    if !live.judge_after || !live.jobs.is_empty() {
        return;
    }
    let Some(run) = app.arena.runs.iter().find(|r| r.id == run_id).cloned() else { return };
    if !run.finished() {
        return;
    }
    let Some(judge) = run.judge.clone() else {
        app.arena.live.as_mut().unwrap().judge_after = false;
        return;
    };
    let mut asks = vec![];
    for i in 0..run.items.len() {
        if run.items[i].verdict.is_none()
            && let Some((messages, shown)) = arena::judge_messages(&run, i, &judge.rubric)
        {
            match app.request_for(&judge.provider, Some(&judge.model), messages) {
                Ok(mut req) => {
                    req.extra.insert("temperature".into(), json!(0));
                    asks.push((i, shown, req));
                }
                Err(e) => {
                    app.arena.error = Some(format!("The judge can't start: {e}"));
                    app.arena.live.as_mut().unwrap().judge_after = false;
                    return;
                }
            }
        }
    }
    if asks.is_empty() {
        app.arena.live.as_mut().unwrap().judge_after = false;
        return;
    }
    // A judge on Ollama needs the GPU: with our server still holding it, Ollama runs the judge on
    // the CPU (a report card's judging took minutes instead of seconds). The model the user had
    // running comes back when the run ends.
    if judge.provider == "ollama" && app.server.state() != State::Stopped {
        app.server.stop();
    }
    let live = app.arena.live.as_mut().unwrap();
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    live.judging = Some((rx, cancel.clone()));
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        for (i, shown, req) in asks {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let r = llm::collect(&req, &cancel);
            if tx.send((i, shown, r)).is_err() {
                break;
            }
            ctx.request_repaint();
        }
        ctx.request_repaint();
    });
}

// --- the screen ------------------------------------------------------------------------------------

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    egui::Panel::left("arena-left")
        .resizable(false)
        .exact_size(290.0)
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(16))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| sidebar(app, ui, &p));
        });
    egui::CentralPanel::no_frame().show(ui, |ui| {
        egui::ScrollArea::vertical().auto_shrink(false).id_salt("arena-main").show(ui, |ui| {
            egui::Frame::new().inner_margin(16).show(ui, |ui| {
                if let Some(e) = app.arena.error.clone() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(e).color(p.error));
                        if ui.small_button("Dismiss").clicked() {
                            app.arena.error = None;
                        }
                    });
                    ui.add_space(8.0);
                }
                match app.arena.view {
                    View::Setup => setup_view(app, ui, &p),
                    View::Run => run_view(app, ui, &p),
                    View::Board => board_view(app, ui, &p),
                    View::Set => set_view(app, ui, &p),
                }
            });
        });
    });
}

fn sidebar(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Arena").font(theme::heading(20.0)));
    widgets::hint(ui, "The same prompts to several models, side by side.");
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        if widgets::button(ui, Kind::Primary, "New comparison").clicked() {
            app.arena.view = View::Setup;
        }
        if widgets::button(ui, Kind::Tonal, "Leaderboard").clicked() {
            app.arena.view = View::Board;
        }
    });
    ui.add_space(14.0);
    section(ui, p, "Prompt sets");
    let mut open_set = None;
    for s in &app.arena.sets {
        let sel = app.arena.view == View::Set && app.arena.set.as_deref() == Some(s.id.as_str());
        if row(ui, p, sel, &s.name, &plural(s.prompts.len(), "prompt")).clicked() {
            open_set = Some(s.id.clone());
        }
    }
    if widgets::button(ui, Kind::Ghost, "+ New set").clicked() {
        let s = PromptSet::new("New prompt set");
        let _ = app.arena_store.save_set(&s);
        open_set = Some(s.id.clone());
        app.arena.sets.insert(0, s);
    }
    if let Some(id) = open_set {
        app.arena.set = Some(id);
        app.arena.view = View::Set;
    }
    ui.add_space(14.0);
    section(ui, p, "Comparisons");
    if app.arena.runs.is_empty() {
        widgets::hint(ui, "None yet.");
    }
    let live = app.arena.live.as_ref().map(|l| l.run.clone());
    let mut open = None;
    for r in &app.arena.runs {
        let sel = app.arena.view == View::Run && app.arena.open.as_deref() == Some(r.id.as_str());
        let state = if live.as_deref() == Some(r.id.as_str()) {
            format!("running · {}/{}", r.answered(), r.total())
        } else if r.report_for.is_some() {
            format!("report card · {}", plural(r.items.len(), "prompt"))
        } else {
            format!("{} × {} models", plural(r.items.len(), "prompt"), r.contestants.len())
        };
        if row(ui, p, sel, &r.name, &state).clicked() {
            open = Some(r.id.clone());
        }
    }
    if let Some(id) = open {
        app.arena.open = Some(id);
        app.arena.view = View::Run;
        app.arena.vote_at = 0;
    }
}

pub fn plural(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

fn section(ui: &mut egui::Ui, p: &Palette, text: &str) {
    ui.label(RichText::new(text).font(theme::medium(13.0)).color(p.on_variant));
    ui.add_space(4.0);
}

fn row(ui: &mut egui::Ui, p: &Palette, selected: bool, title: &str, detail: &str) -> egui::Response {
    let w = ui.available_width();
    let fill = if selected { p.secondary_container } else { egui::Color32::TRANSPARENT };
    let r = widgets::surface(ui, fill, Stroke::NONE, 6.0, egui::Margin::symmetric(8, 6), |ui| {
        ui.set_width(w - 16.0);
        ui.label(RichText::new(title).color(p.on_bg));
        ui.label(RichText::new(detail).small().color(p.on_variant));
    });
    r.response.interact(egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A model chooser for a provider: its model list (fetched once) and a box for any id.
pub(crate) fn model_choice(app: &mut App, ui: &mut egui::Ui, provider: &str, model: &mut String, salt: &str) {
    let lists = app.chat.model_lists.lock().unwrap().get(provider).cloned();
    if lists.is_none() {
        let p = providers::get(provider);
        if !p.needs_key || !app.secrets.get(provider).is_empty() {
            app.fetch_models(provider, ui.ctx());
        }
    }
    let label = if model.is_empty() { "Model".to_string() } else { model.clone() };
    egui::ComboBox::from_id_salt(("arena-model", salt, provider)).selected_text(label).width(260.0).height(420.0).show_ui(ui, |ui| {
        ui.add(egui::TextEdit::singleline(model).hint_text("model id"));
        ui.separator();
        match &lists {
            Some(crate::app::ModelList::Ready(list)) => {
                for (id, _) in list.iter().take(400) {
                    if ui.selectable_label(id == model, id).clicked() {
                        *model = id.clone();
                    }
                }
            }
            Some(crate::app::ModelList::Failed(e)) => {
                ui.colored_label(pal(ui.ctx()).error, e);
            }
            Some(crate::app::ModelList::Loading) => {
                ui.label("Loading models…");
            }
            None => {
                for s in providers::get(provider).suggested {
                    if ui.selectable_label(*s == model, *s).clicked() {
                        *model = s.to_string();
                    }
                }
            }
        }
    });
}

fn provider_choice(ui: &mut egui::Ui, provider: &mut String, salt: &str) -> bool {
    let before = provider.clone();
    egui::ComboBox::from_id_salt(("arena-provider", salt)).selected_text(providers::get(provider).short).width(130.0).show_ui(ui, |ui| {
        for pr in providers::ALL.iter().filter(|x| x.id != providers::PC) {
            ui.selectable_value(provider, pr.id.to_string(), pr.short);
        }
    });
    *provider != before
}

fn setup_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("New comparison").font(theme::heading(22.0)));
    widgets::hint(ui, "Two to four models answer the same prompts with the same settings. Cloud models answer at once; models on this PC take turns on the GPU so each gets it to itself.");
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Who's competing");
        let mut remove = None;
        let mut add_quants = None;
        let names: Vec<String> = app.library.models.iter().map(|m| crate::app::short_model_name(&m.name)).collect();
        for (i, w) in app.arena.draft.contestants.iter().enumerate() {
            ui.horizontal(|ui| {
                widgets::badge(ui, &arena::letter(i).to_string(), p.on_primary_container, p.primary_container);
                ui.label(w.label());
                ui.label(RichText::new(kind_text(w)).small().color(p.on_variant));
                if let Who::Library { name, .. } = w {
                    let sib = arena::quant_siblings(name, &names);
                    if !sib.is_empty() && ui.small_button(format!("+ other quants ({})", sib.len())).on_hover_text("Add the same model's other quantisations: quality against speed and memory").clicked() {
                        add_quants = Some(sib);
                    }
                }
                if ui.small_button("Remove").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            app.arena.draft.contestants.remove(i);
        }
        if let Some(sib) = add_quants {
            for i in sib {
                let m = &app.library.models[i];
                let w = Who::Library { key: m.key.clone(), name: crate::app::short_model_name(&m.name) };
                if app.arena.draft.contestants.len() < MAX_CONTESTANTS && !app.arena.draft.contestants.contains(&w) {
                    app.arena.draft.contestants.push(w);
                }
            }
        }
        if app.arena.draft.contestants.len() < MAX_CONTESTANTS {
            ui.add_space(6.0);
            add_row(app, ui);
        } else {
            widgets::hint(ui, "Four at most, so the answers stay readable side by side.");
        }
    });
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "What to ask");
        let sets: Vec<(String, String, usize)> = app.arena.sets.iter().map(|s| (s.id.clone(), s.name.clone(), s.prompts.len())).collect();
        ui.horizontal(|ui| {
            let d = &mut app.arena.draft;
            if widgets::chip(ui, d.set.is_none(), "One prompt").clicked() {
                d.set = None;
            }
            if widgets::chip(ui, d.set.is_some(), "A prompt set").clicked() && d.set.is_none() {
                d.set = sets.first().map(|s| s.0.clone()).or(Some(String::new()));
            }
        });
        ui.add_space(6.0);
        let d = &mut app.arena.draft;
        if let Some(set) = d.set.as_mut() {
            if sets.is_empty() {
                widgets::hint(ui, "No prompt sets yet: make one with “+ New set” on the left.");
            } else {
                let name = sets.iter().find(|s| &s.0 == set).map(|s| format!("{} ({} prompts)", s.1, s.2)).unwrap_or("Choose a set".into());
                egui::ComboBox::from_id_salt("arena-set").selected_text(name).width(320.0).show_ui(ui, |ui| {
                    for (id, n, c) in &sets {
                        ui.selectable_value(set, id.clone(), format!("{n} ({c} prompts)"));
                    }
                });
            }
        } else {
            ui.add(egui::TextEdit::multiline(&mut d.prompt).desired_rows(3).desired_width(f32::INFINITY).hint_text("Ask something…"));
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("System prompt");
            ui.add(egui::TextEdit::singleline(&mut d.system).desired_width(f32::INFINITY).hint_text("optional, the same for everyone"));
        });
    });
    ui.add_space(10.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "How");
        let d = &mut app.arena.draft;
        ui.horizontal(|ui| {
            ui.label("Temperature");
            ui.add(egui::Slider::new(&mut d.temperature, 0.0..=1.5).step_by(0.05));
            ui.add_space(16.0);
            ui.label("Length limit");
            ui.add(egui::DragValue::new(&mut d.max_tokens).range(0..=32768).speed(16).custom_formatter(|v, _| if v == 0.0 { "none".into() } else { format!("{v:.0} tokens") }));
        });
        ui.add_space(4.0);
        widgets::switch_row(ui, "Blind", "Answers show as A, B, … in a shuffled order until you vote, so names can't sway you.", &mut d.blind);
        widgets::switch_row(ui, "AI judge", "Another model scores every answer out of 10 against your rubric and picks the best.", &mut d.judge_on);
        if d.judge_on {
            ui.horizontal(|ui| {
                ui.label("Judge");
                let mut prov = d.judge.provider.clone();
                if provider_choice(ui, &mut prov, "judge") {
                    d.judge.model.clear();
                }
                d.judge.provider = prov;
            });
            let (prov, mut model) = (app.arena.draft.judge.provider.clone(), app.arena.draft.judge.model.clone());
            ui.horizontal(|ui| {
                ui.label("Model");
                model_choice(app, ui, &prov, &mut model, "judge");
            });
            let d = &mut app.arena.draft;
            d.judge.model = model;
            ui.label("Rubric");
            ui.add(egui::TextEdit::multiline(&mut d.judge.rubric).desired_rows(2).desired_width(f32::INFINITY));
        }
    });
    ui.add_space(12.0);

    let d = &app.arena.draft;
    let prompts: Vec<(String, String)> = match &d.set {
        Some(id) => app.arena.sets.iter().find(|s| &s.id == id).map(|s| s.prompts.iter().filter(|x| !x.text.trim().is_empty()).map(|x| (x.text.clone(), x.reference.clone())).collect()).unwrap_or_default(),
        None if !d.prompt.trim().is_empty() => vec![(d.prompt.trim().to_string(), String::new())],
        None => vec![],
    };
    let why_not = if d.contestants.len() < 2 {
        Some("Add at least two contestants.")
    } else if prompts.is_empty() {
        Some("Write a prompt or choose a set with prompts in it.")
    } else if d.judge_on && d.judge.model.trim().is_empty() {
        Some("Choose the judge's model.")
    } else if app.arena.live.is_some() {
        Some("Another comparison is running.")
    } else if app.tune.training() && d.contestants.iter().any(|w| w.local()) {
        Some("Training has the GPU right now.")
    } else {
        None
    };
    ui.horizontal(|ui| {
        if widgets::button_ext(ui, Kind::Primary, "Start", why_not.is_none()).clicked() {
            let d = &app.arena.draft;
            let name = match &d.set {
                Some(id) => app.arena.sets.iter().find(|s| &s.id == id).map(|s| s.name.clone()).unwrap_or_default(),
                None => crate::app::title_from(&d.prompt),
            };
            let mut run = Run::new(&name, d.contestants.clone(), prompts);
            run.system = d.system.clone();
            run.temperature = d.temperature;
            run.max_tokens = d.max_tokens;
            run.blind = d.blind;
            run.judge = d.judge_on.then(|| d.judge.clone());
            let id = run.id.clone();
            let judge = run.judge.is_some();
            app.arena.runs.insert(0, run);
            save(app, &id);
            app.arena.open = Some(id.clone());
            app.arena.view = View::Run;
            app.arena.run_view = if app.arena.draft.blind { RunView::Vote } else { RunView::Answers };
            app.arena.vote_at = 0;
            start(app, &id, judge);
        }
        if let Some(w) = why_not {
            widgets::hint(ui, w);
        }
    });
}

fn kind_text(w: &Who) -> &'static str {
    match w {
        Who::Library { .. } => "llama.cpp on this PC",
        Who::Adapter { .. } => "fine-tuned adapter",
        Who::Provider { provider, .. } if provider == "ollama" => "Ollama on this PC",
        Who::Provider { .. } => "cloud",
    }
}

fn add_row(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        let d = &mut app.arena.draft;
        widgets::segmented(ui, &mut d.source, &[(Source::Provider, "Provider"), (Source::Library, "Model file"), (Source::Adapter, "Adapter")]);
    });
    ui.add_space(4.0);
    let mut add: Option<Who> = None;
    ui.horizontal(|ui| match app.arena.draft.source {
        Source::Provider => {
            let mut prov = app.arena.draft.provider.clone();
            if provider_choice(ui, &mut prov, "add") {
                app.arena.draft.model.clear();
            }
            app.arena.draft.provider = prov.clone();
            let mut model = app.arena.draft.model.clone();
            model_choice(app, ui, &prov, &mut model, "add");
            app.arena.draft.model = model.clone();
            if widgets::button_ext(ui, Kind::Tonal, "Add", !model.trim().is_empty()).clicked() {
                add = Some(Who::Provider { provider: prov, model: model.trim().to_string() });
            }
        }
        Source::Library => {
            let key = app.arena.draft.library_key.clone();
            let label = key.as_ref().and_then(|k| app.library.get(k)).map(|m| crate::app::short_model_name(&m.name)).unwrap_or("Choose a model file".into());
            let mut pick = None;
            egui::ComboBox::from_id_salt("arena-lib").selected_text(label).width(340.0).height(420.0).show_ui(ui, |ui| {
                if app.library.models.is_empty() {
                    ui.label("No models yet: add some in the Engine Room.");
                }
                for m in &app.library.models {
                    if ui.selectable_label(key.as_deref() == Some(m.key.as_str()), format!("{}   {}", crate::app::short_model_name(&m.name), widgets::fmt_bytes(m.size))).clicked() {
                        pick = Some(m.key.clone());
                    }
                }
            });
            if pick.is_some() {
                app.arena.draft.library_key = pick;
            }
            let chosen = app.arena.draft.library_key.as_ref().and_then(|k| app.library.get(k));
            if widgets::button_ext(ui, Kind::Tonal, "Add", chosen.is_some()).clicked()
                && let Some(m) = chosen
            {
                add = Some(Who::Library { key: m.key.clone(), name: crate::app::short_model_name(&m.name) });
            }
        }
        Source::Adapter => {
            if app.tune.adapters.is_empty() {
                widgets::hint(ui, "No adapters yet: train one in Fine-tune.");
                return;
            }
            let id = app.arena.draft.adapter.clone().or_else(|| app.tune.adapters.first().map(|a| a.id.clone()));
            let label = id.as_ref().and_then(|i| app.tune.adapters.iter().find(|a| &a.id == i)).map(|a| a.name.clone()).unwrap_or_default();
            let mut pick = id.clone();
            egui::ComboBox::from_id_salt("arena-adapter").selected_text(label).width(240.0).show_ui(ui, |ui| {
                for a in &app.tune.adapters {
                    ui.selectable_value(&mut pick, Some(a.id.clone()), format!("{} (on {})", a.name, a.model_name));
                }
            });
            app.arena.draft.adapter = pick.clone();
            let d = &mut app.arena.draft;
            if widgets::chip(ui, d.scale == 0.0, "Base model").clicked() {
                d.scale = 0.0;
            }
            if widgets::chip(ui, d.scale == 1.0, "Tuned").clicked() {
                d.scale = 1.0;
            }
            if let Some(a) = pick.and_then(|i| app.tune.adapters.iter().find(|a| a.id == i))
                && widgets::button(ui, Kind::Tonal, "Add").clicked()
            {
                add = Some(Who::Adapter { id: a.id.clone(), name: a.name.clone(), model_key: a.model_key.clone(), scale: app.arena.draft.scale });
            }
        }
    });
    if let Some(w) = add
        && !app.arena.draft.contestants.contains(&w)
    {
        app.arena.draft.contestants.push(w);
    }
}

fn secs(ms: f64) -> String {
    if ms < 1000.0 { format!("{ms:.0} ms") } else { format!("{:.1} s", ms / 1000.0) }
}

fn run_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(run) = app.arena.current().cloned() else {
        empty(ui, "No comparison open", "Start one with “New comparison”.");
        return;
    };
    let live = app.arena.live.as_ref().is_some_and(|l| l.run == run.id);
    let judging = live && app.arena.live.as_ref().is_some_and(|l| l.judging.is_some());
    ui.horizontal(|ui| {
        ui.label(RichText::new(&run.name).font(theme::heading(20.0)));
        widgets::hint(ui, format!(
            "{} · temperature {:.2}{}{}",
            plural(run.items.len(), "prompt"),
            run.temperature,
            if run.max_tokens > 0 { format!(" · up to {} tokens", run.max_tokens) } else { String::new() },
            run.judge.as_ref().map(|j| format!(" · judged by {}", j.model)).unwrap_or_default()
        ));
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if live {
            if widgets::button(ui, Kind::Ghost, "Stop").clicked() {
                stop(app);
            }
            let status = if judging { "Judging…".to_string() } else { format!("Answering · {} of {}", run.answered(), run.total()) };
            ui.add(egui::Spinner::new());
            widgets::hint(ui, status);
        } else {
            if !run.finished() && widgets::button_ext(ui, Kind::Primary, "Continue", app.arena.live.is_none()).clicked() {
                start(app, &run.id, run.judge.is_some());
            }
            if run.finished() && run.judge.is_some() && widgets::button_ext(ui, Kind::Tonal, if needs_judging(&run) { "Judge" } else { "Judge again" }, app.arena.live.is_none()).clicked() {
                if !needs_judging(&run)
                    && let Some(r) = app.arena.run_mut(&run.id)
                {
                    for it in &mut r.items {
                        it.verdict = None;
                        it.answers.iter_mut().for_each(|a| a.score = None);
                    }
                }
                start(app, &run.id, true);
            }
            if widgets::button_ext(ui, Kind::Tonal, "Run again", app.arena.live.is_none()).on_hover_text("The same models and prompts as a new comparison").clicked() {
                let mut again = Run::new(&run.name, run.contestants.clone(), run.items.iter().map(|i| (i.prompt.clone(), i.reference.clone())).collect());
                again.system = run.system.clone();
                again.temperature = run.temperature;
                again.max_tokens = run.max_tokens;
                again.blind = run.blind;
                again.judge = run.judge.clone();
                again.report_for = run.report_for.clone();
                let id = again.id.clone();
                let judge = again.judge.is_some();
                app.arena.runs.insert(0, again);
                save(app, &id);
                app.arena.open = Some(id.clone());
                app.arena.vote_at = 0;
                start(app, &id, judge);
            }
            if run.items.len() > 1 || run.items.iter().any(|i| !i.reference.is_empty()) {
                if widgets::button(ui, Kind::Ghost, "Save prompts as a set").clicked() {
                    let mut s = PromptSet::new(&run.name);
                    s.prompts = run.items.iter().map(|i| SetPrompt { text: i.prompt.clone(), reference: i.reference.clone() }).collect();
                    let _ = app.arena_store.save_set(&s);
                    app.arena.sets.insert(0, s);
                    app.notify("Saved as a prompt set");
                }
            } else if widgets::button(ui, Kind::Ghost, "Add to a set").on_hover_text("Adds this prompt to the newest prompt set (or a new one)").clicked() {
                let prompt = run.items[0].prompt.clone();
                if app.arena.sets.is_empty() {
                    app.arena.sets.insert(0, PromptSet::new("My prompts"));
                }
                let s = &mut app.arena.sets[0];
                s.add_lines(&prompt.replace('\n', " "));
                let _ = app.arena_store.save_set(s);
                let name = s.name.clone();
                app.notify(format!("Added to “{name}”"));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::button(ui, Kind::Danger, "Delete").clicked() {
                    app.arena_store.delete_run(&run.id);
                    app.arena.runs.retain(|r| r.id != run.id);
                    app.arena.open = app.arena.runs.first().map(|r| r.id.clone());
                }
            });
        }
    });
    if !app.arena.runs.iter().any(|r| r.id == run.id) {
        return;
    }
    ui.add_space(10.0);
    let hidden = run.blind && !run.all_voted();
    if !hidden {
        summary_card(ui, p, &run);
        ui.add_space(10.0);
    }
    ui.horizontal(|ui| {
        let mut v = if hidden { RunView::Vote } else { app.arena.run_view };
        if hidden {
            widgets::hint(ui, "Blind: names show once every prompt has your vote.");
            if widgets::button(ui, Kind::Ghost, "Reveal now").clicked()
                && let Some(r) = app.arena.run_mut(&run.id)
            {
                r.blind = false;
                save(app, &run.id);
            }
        } else {
            widgets::segmented(ui, &mut v, &[(RunView::Answers, "Answers"), (RunView::Vote, "Vote")]);
            app.arena.run_view = v;
            if v == RunView::Answers {
                ui.add_space(12.0);
                ui.checkbox(&mut app.arena.diff, "Highlight what differs from A");
                ui.checkbox(&mut app.arena.thinking, "Show thinking");
            }
        }
    });
    ui.add_space(8.0);
    if hidden || app.arena.run_view == RunView::Vote {
        vote_view(app, ui, p, &run, hidden);
    } else {
        answers_view(app, ui, p, &run);
    }
}

fn summary_card(ui: &mut egui::Ui, p: &Palette, run: &Run) {
    let s = run.summary();
    widgets::card(ui, |ui| {
        egui::Grid::new(("arena-summary", &run.id)).num_columns(9).spacing([18.0, 6.0]).striped(false).show(ui, |ui| {
            for h in ["", "Model", "First token", "Speed", "Length", "Time", "GPU memory", "Judge", "Your picks"] {
                ui.label(RichText::new(h).small().color(p.on_variant));
            }
            ui.end_row();
            let best_speed = s.iter().filter_map(|x| x.speed).fold(0.0, f64::max);
            let best_first = s.iter().filter_map(|x| x.first_ms).fold(f64::INFINITY, f64::min);
            let best_score = s.iter().filter_map(|x| x.score).fold(0.0, f64::max);
            for (c, (w, x)) in run.contestants.iter().zip(&s).enumerate() {
                widgets::badge(ui, &arena::letter(c).to_string(), p.on_primary_container, p.primary_container);
                ui.label(RichText::new(w.label()).color(p.on_bg));
                let star = |good: bool, t: String| if good && run.contestants.len() > 1 { RichText::new(t).color(p.good) } else { RichText::new(t) };
                ui.label(x.first_ms.map(|f| star(f == best_first, secs(f))).unwrap_or(RichText::new("–")));
                ui.label(x.speed.map(|v| star(v == best_speed, format!("{v:.1} tok/s"))).unwrap_or(RichText::new("–")));
                ui.label(x.tokens.map(|t| format!("{}{t:.0} tokens", if x.estimated { "~" } else { "" })).unwrap_or("–".into()));
                ui.label(x.took.map(secs).unwrap_or("–".into()));
                ui.label(x.vram.map(widgets::fmt_bytes).unwrap_or("–".into()));
                ui.label(x.score.map(|v| star(v == best_score, format!("{v:.1} / 10"))).unwrap_or(RichText::new("–")));
                let fails = if x.failed > 0 { format!(" · {} failed", x.failed) } else { String::new() };
                ui.label(format!("{}{fails}", x.votes));
                ui.end_row();
            }
        });
        if run.items.len() > 1 && run.judge.is_some() {
            let (agree, both) = arena::agreement(std::slice::from_ref(run));
            if both > 0 {
                widgets::hint(ui, format!("The judge picked what you picked on {agree} of {both}."));
            }
        }
    });
}

fn answers_view(app: &mut App, ui: &mut egui::Ui, p: &Palette, run: &Run) {
    let n = run.contestants.len();
    let mut vote: Option<(usize, Vote)> = None;
    let mut again: Option<(usize, usize)> = None;
    for (i, it) in run.items.iter().enumerate() {
        widgets::surface(ui, p.low, Stroke::NONE, 6.0, 10, |ui| {
            ui.set_width(ui.available_width());
            if run.items.len() > 1 {
                ui.label(RichText::new(format!("Prompt {}", i + 1)).small().color(p.on_variant));
            }
            ui.label(RichText::new(it.prompt.trim()).color(p.on_bg));
        });
        ui.add_space(6.0);
        ui.columns(n, |cols| {
            for (c, col) in cols.iter_mut().enumerate() {
                let a = &it.answers[c];
                let picked = it.vote.as_ref().is_some_and(|v| v.best == Some(c) && !v.all_bad);
                let judged = it.verdict.as_ref().is_some_and(|v| v.best == Some(c));
                let edge = if picked { p.primary } else { p.outline_variant };
                widgets::surface(col, p.container, Stroke::new(1.0, edge), 8.0, 10, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal_wrapped(|ui| {
                        widgets::badge(ui, &arena::letter(c).to_string(), p.on_primary_container, p.primary_container);
                        ui.label(RichText::new(run.contestants[c].label()).font(theme::medium(14.0)));
                        if let Some(s) = a.score {
                            widgets::badge(ui, &format!("{s:.0}/10{}", if judged { " · judge's pick" } else { "" }), p.on_bg, p.high);
                        }
                    });
                    widgets::hint(ui, metrics(a));
                    if app.arena.thinking && !a.reasoning.is_empty() {
                        egui::CollapsingHeader::new(RichText::new("Thinking").small()).id_salt(("think", &run.id, i, c)).default_open(false).show(ui, |ui| {
                            ui.label(RichText::new(a.reasoning.trim()).small().color(p.on_variant));
                        });
                    }
                    if app.arena.diff && c > 0 && a.done && it.answers[0].done {
                        diff_text(ui, p, &it.answers[0].text, &a.text);
                    } else if !a.text.is_empty() {
                        egui_commonmark::CommonMarkViewer::new().show(ui, &mut app.chat.md, &a.text);
                    } else if !a.done && !a.stopped {
                        widgets::hint(ui, if a.reasoning.is_empty() { "waiting…" } else { "thinking…" });
                    }
                    if let Some(e) = &a.error {
                        ui.label(RichText::new(e).small().color(p.error));
                    }
                    if a.stopped {
                        widgets::hint(ui, "Stopped");
                    }
                    if app.arena.live.is_none() {
                        ui.horizontal(|ui| {
                            if a.ok() && n > 1 && !picked && ui.small_button("Best").on_hover_text("Your pick for this prompt").clicked() {
                                vote = Some((i, Vote { best: Some(c), all_bad: false, at: crate::store::now_ms() }));
                            }
                            if a.done && ui.small_button("Ask again").clicked() {
                                again = Some((i, c));
                            }
                        });
                    }
                });
            }
        });
        if let Some(v) = &it.verdict {
            ui.add_space(4.0);
            let text = match (&v.error, v.best) {
                (Some(e), _) => format!("Judge: {e}"),
                (None, Some(b)) => format!("Judge: {} is best. {}", arena::letter(b), v.why),
                (None, None) => format!("Judge: a tie. {}", v.why),
            };
            widgets::hint(ui, text);
        }
        ui.add_space(14.0);
    }
    if let Some((i, v)) = vote
        && let Some(r) = app.arena.run_mut(&run.id)
    {
        r.items[i].vote = Some(v);
        save(app, &run.id);
    }
    if let Some((i, c)) = again
        && let Some(r) = app.arena.run_mut(&run.id)
    {
        r.items[i].answers[c] = Answer::default();
        r.items[i].vote = None;
        r.items[i].verdict = None;
        r.items[i].answers.iter_mut().for_each(|a| a.score = None);
        save(app, &run.id);
        start(app, &run.id, false);
    }
}

fn metrics(a: &Answer) -> String {
    if !a.done && !a.stopped {
        return match a.first_ms {
            Some(f) => format!("first token after {}", secs(f as f64)),
            None => String::new(),
        };
    }
    let mut parts = vec![];
    if let Some(f) = a.first_ms {
        parts.push(format!("first token {}", secs(f as f64)));
    }
    if let Some(s) = a.speed() {
        parts.push(format!("{s:.1} tok/s"));
    }
    let (t, est) = a.tokens();
    if t > 0 {
        parts.push(format!("{}{t} tokens", if est { "~" } else { "" }));
    }
    parts.push(secs(a.took as f64));
    parts.join(" · ")
}

/// The answer with the words A doesn't have highlighted.
fn diff_text(ui: &mut egui::Ui, p: &Palette, base: &str, other: &str) {
    let mut job = egui::text::LayoutJob::default();
    let font = egui::TextStyle::Body.resolve(ui.style());
    for (text, same) in arena::shared_runs(base, other) {
        let fmt = if same {
            egui::TextFormat { font_id: font.clone(), color: p.on_variant, ..Default::default() }
        } else {
            egui::TextFormat { font_id: font.clone(), color: p.on_bg, background: p.primary_container, ..Default::default() }
        };
        job.append(&text, 0.0, fmt);
    }
    job.wrap.max_width = ui.available_width();
    ui.label(job);
    let sim = arena::similarity(base, other);
    widgets::hint(ui, format!("{:.0}% the same words as A", sim * 100.0));
}

fn vote_view(app: &mut App, ui: &mut egui::Ui, p: &Palette, run: &Run, blind: bool) {
    let n = run.items.len();
    let at = app.arena.vote_at.min(n.saturating_sub(1));
    let it = &run.items[at];
    let voted = run.items.iter().filter(|i| i.vote.is_some()).count();
    ui.horizontal(|ui| {
        if ui.add_enabled(at > 0, egui::Button::new("‹")).clicked() {
            app.arena.vote_at = at - 1;
        }
        ui.label(format!("Prompt {} of {n}", at + 1));
        if ui.add_enabled(at + 1 < n, egui::Button::new("›")).clicked() {
            app.arena.vote_at = at + 1;
        }
        widgets::hint(ui, format!("{voted} voted"));
    });
    ui.add_space(6.0);
    widgets::surface(ui, p.low, Stroke::NONE, 6.0, 10, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(it.prompt.trim()).color(p.on_bg));
    });
    ui.add_space(6.0);
    let show_names = !blind || it.vote.is_some();
    let order: Vec<usize> = if blind { it.order.clone() } else { (0..run.contestants.len()).collect() };
    let mut pick: Option<Vote> = None;
    ui.columns(order.len(), |cols| {
        for (k, (&c, col)) in order.iter().zip(cols.iter_mut()).enumerate() {
            let a = &it.answers[c];
            let picked = it.vote.as_ref().is_some_and(|v| v.best == Some(c) && !v.all_bad);
            widgets::surface(col, p.container, Stroke::new(1.0, if picked { p.primary } else { p.outline_variant }), 8.0, 10, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    let tag = if blind { arena::letter(k) } else { arena::letter(c) };
                    widgets::badge(ui, &tag.to_string(), p.on_primary_container, p.primary_container);
                    if show_names {
                        ui.label(RichText::new(run.contestants[c].label()).font(theme::medium(14.0)));
                    }
                });
                if a.text.is_empty() && !a.done {
                    widgets::hint(ui, "still answering…");
                } else {
                    egui_commonmark::CommonMarkViewer::new().show(ui, &mut app.chat.md, &a.text);
                }
                if let Some(e) = &a.error {
                    ui.label(RichText::new(e).small().color(p.error));
                }
                if show_names {
                    widgets::hint(ui, metrics(a));
                }
                if a.ok() && widgets::button(ui, if picked { Kind::Primary } else { Kind::Tonal }, format!("{} is best", if blind { arena::letter(k) } else { arena::letter(c) })).clicked() {
                    pick = Some(Vote { best: Some(c), all_bad: false, at: crate::store::now_ms() });
                }
            });
        }
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if widgets::button(ui, Kind::Ghost, "Tie").clicked() {
            pick = Some(Vote { best: None, all_bad: false, at: crate::store::now_ms() });
        }
        if widgets::button(ui, Kind::Ghost, "All bad").clicked() {
            pick = Some(Vote { best: None, all_bad: true, at: crate::store::now_ms() });
        }
        if let Some(v) = &it.verdict
            && show_names
        {
            widgets::hint(ui, match v.best {
                Some(b) => format!("Judge picked {}: {}", run.contestants[b].label(), v.why),
                None => format!("Judge: {}", v.error.clone().unwrap_or_else(|| format!("a tie. {}", v.why))),
            });
        }
    });
    if let Some(v) = pick
        && let Some(r) = app.arena.run_mut(&run.id)
    {
        r.items[at].vote = Some(v);
        // On to the next prompt without a vote.
        if let Some(next) = (0..n).map(|d| (at + 1 + d) % n).find(|&j| r.items[j].vote.is_none()) {
            app.arena.vote_at = next;
        }
        save(app, &run.id);
    }
}

fn board_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Leaderboard").font(theme::heading(22.0)));
    widgets::hint(ui, "Your votes from every comparison as Elo ratings (everyone starts at 1000; a pick beats each other answer, a tie or “all bad” draws), with the AI judge's average score beside them.");
    ui.add_space(10.0);
    let board = arena::leaderboard(&app.arena.runs);
    if board.is_empty() {
        empty(ui, "No votes yet", "Vote for the best answer in a comparison and the standings show up here.");
        return;
    }
    widgets::card(ui, |ui| {
        egui::Grid::new("arena-board").num_columns(6).spacing([22.0, 8.0]).show(ui, |ui| {
            for h in ["", "Model", "Rating", "Won · tied · lost", "Win rate", "Judge's average"] {
                ui.label(RichText::new(h).small().color(p.on_variant));
            }
            ui.end_row();
            for (i, s) in board.iter().enumerate() {
                ui.label(RichText::new(format!("{}", i + 1)).font(theme::heading(16.0)).color(if i == 0 { p.primary } else { p.on_variant }));
                ui.label(RichText::new(&s.label).color(p.on_bg));
                ui.label(if s.games() > 0 { format!("{:.0}", s.elo) } else { "–".into() });
                ui.label(format!("{} · {} · {}", s.wins, s.ties, s.losses));
                ui.label(if s.games() > 0 { format!("{:.0}%", 100.0 * (s.wins as f64 + 0.5 * s.ties as f64) / s.games() as f64) } else { "–".into() });
                ui.label(s.score.map(|x| format!("{x:.1} over {}", s.judged)).unwrap_or("–".into()));
                ui.end_row();
            }
        });
    });
    let (agree, both) = arena::agreement(&app.arena.runs);
    if both > 0 {
        ui.add_space(8.0);
        widgets::hint(ui, format!("The AI judge agreed with you on {agree} of {both} prompts ({:.0}%).", 100.0 * agree as f64 / both as f64));
    }
}

fn set_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(id) = app.arena.set.clone() else { return };
    let Some(idx) = app.arena.sets.iter().position(|s| s.id == id) else {
        app.arena.view = View::Setup;
        return;
    };
    let mut changed = false;
    let mut delete = false;
    let mut use_it = false;
    {
        let s = &mut app.arena.sets[idx];
        ui.horizontal(|ui| {
            changed |= ui.add(egui::TextEdit::singleline(&mut s.name).font(theme::heading(18.0)).desired_width(320.0)).changed();
            widgets::hint(ui, plural(s.prompts.len(), "prompt"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::button(ui, Kind::Danger, "Delete set").clicked() {
                    delete = true;
                }
                if widgets::button_ext(ui, Kind::Primary, "Compare with this set", !s.prompts.is_empty()).clicked() {
                    use_it = true;
                }
            });
        });
        ui.add_space(8.0);
        widgets::card(ui, |ui| {
            widgets::card_title(ui, "Add prompts");
            widgets::hint(ui, "One per line.");
            ui.add(egui::TextEdit::multiline(&mut app.arena.paste).desired_rows(3).desired_width(f32::INFINITY).hint_text("What's the difference between TCP and UDP?\nWrite a haiku about rust (the metal)."));
            if widgets::button_ext(ui, Kind::Tonal, "Add", !app.arena.paste.trim().is_empty()).clicked() {
                s.add_lines(&app.arena.paste);
                app.arena.paste.clear();
                changed = true;
            }
        });
        ui.add_space(8.0);
        let mut remove = None;
        for (i, pr) in s.prompts.iter_mut().enumerate() {
            widgets::surface(ui, p.low, Stroke::NONE, 6.0, 10, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}", i + 1)).color(p.on_variant));
                    changed |= ui.add(egui::TextEdit::multiline(&mut pr.text).desired_rows(1).desired_width(ui.available_width() - 80.0)).changed();
                    if ui.small_button("Remove").clicked() {
                        remove = Some(i);
                    }
                });
                egui::CollapsingHeader::new(RichText::new(if pr.reference.trim().is_empty() { "Reference answer (optional, for the judge)" } else { "Reference answer" }).small())
                    .id_salt(("ref", i))
                    .show(ui, |ui| {
                        changed |= ui.add(egui::TextEdit::multiline(&mut pr.reference).desired_rows(2).desired_width(f32::INFINITY)).changed();
                    });
            });
            ui.add_space(4.0);
        }
        if let Some(i) = remove {
            s.prompts.remove(i);
            changed = true;
        }
    }
    if changed {
        let _ = app.arena_store.save_set(&app.arena.sets[idx]);
    }
    if use_it {
        app.arena.draft.set = Some(id.clone());
        app.arena.view = View::Setup;
    }
    if delete {
        app.arena_store.delete_set(&id);
        app.arena.sets.remove(idx);
        if app.arena.draft.set.as_deref() == Some(id.as_str()) {
            app.arena.draft.set = None;
        }
        app.arena.set = None;
        app.arena.view = View::Setup;
    }
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

// --- the report card (Fine-tune's Try it) -------------------------------------------------------------

/// After training: the base and the tuned model answer the held-back questions, a judge compares
/// each pair blind against the held-back reply.
pub fn report_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, a: &crate::engine::train::Adapter) {
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Report card");
        widgets::hint(ui, "The base model and the tuned one answer the questions you held back from training. A judge compares each pair without knowing which is which, against the reply you held back.");
        let ds = app.tune.datasets.iter().find(|d| d.id == a.dataset_id).map(|d| d.name.clone());
        let held = app.tune.datasets.iter().find(|d| d.id == a.dataset_id).map(|d| d.held_prompts()).unwrap_or_default();
        let latest = app.arena.runs.iter().find(|r| r.report_for.as_deref() == Some(a.id.as_str())).cloned();
        if let Some(r) = &latest {
            ui.add_space(4.0);
            report_view(app, ui, p, r);
        }
        ui.add_space(6.0);
        if held.is_empty() {
            widgets::hint(ui, match ds {
                None => "Its dataset is gone, so there are no held-back questions to ask.".to_string(),
                Some(name) => format!("No held-back questions in “{name}”. In step 1, Overview → Test split → Hold back 10%, then train again so the questions are new to the model."),
            });
            return;
        }
        let live = app.arena.live.is_some();
        ui.horizontal(|ui| {
            ui.label("Judge");
            let mut prov = app.arena.report_judge.provider.clone();
            egui::ComboBox::from_id_salt("report-judge").selected_text(providers::get(&prov).short).show_ui(ui, |ui| {
                for pr in providers::ALL.iter().filter(|pr| pr.id != providers::PC) {
                    ui.selectable_value(&mut prov, pr.id.to_string(), pr.short);
                }
            });
            if prov != app.arena.report_judge.provider {
                app.arena.report_judge.provider = prov.clone();
                app.arena.report_judge.model.clear();
            }
            let mut model = app.arena.report_judge.model.clone();
            model_choice(app, ui, &prov, &mut model, "report-judge");
            app.arena.report_judge.model = model;
        });
        let n = held.len().min(arena::REPORT_MAX);
        ui.horizontal(|ui| {
            let label = if latest.is_some() { "Make it again" } else { "Make the report card" };
            if widgets::button_ext(ui, Kind::Primary, label, !live && !app.tune.training()).clicked() {
                let j = app.arena.report_judge.clone();
                let run = arena::report_run(a, held.clone(), &j.provider, &j.model);
                let id = run.id.clone();
                app.arena.runs.insert(0, run);
                save(app, &id);
                start(app, &id, true);
            }
            widgets::hint(ui, format!(
                "{} · {} answers, one model at a time on the GPU{}",
                plural(n, "question"),
                n * 2,
                if live && app.arena.phase(latest.as_ref().map(|r| r.id.as_str()).unwrap_or("")).is_none() { " · waiting for the Arena's comparison to finish" } else { "" }
            ));
        });
    });
}

fn report_view(app: &mut App, ui: &mut egui::Ui, p: &Palette, r: &Run) {
    let rep = arena::report(r);
    match app.arena.phase(&r.id) {
        Some(judging) => {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new());
                ui.label(if judging {
                    format!("Judging: {} of {}", r.items.iter().filter(|i| i.verdict.is_some()).count(), r.items.len())
                } else {
                    format!("Answering: {} of {}", r.answered(), r.total())
                });
                if ui.small_button("Stop").clicked() {
                    stop(app);
                }
            });
        }
        None if rep.judged == 0 => {
            widgets::hint(ui, if r.finished() { "Answered but not judged (the judge failed or was stopped)." } else { "Stopped before the end." });
        }
        None => {}
    }
    if rep.judged > 0 {
        let (t, b, n) = (rep.tuned_wins, rep.base_wins, rep.judged);
        let (headline, colour) = match t.cmp(&b) {
            std::cmp::Ordering::Greater => (format!("Tuned is better: it won {t} of {n}"), p.good),
            std::cmp::Ordering::Less => (format!("The base model is better: it won {b} of {n}"), p.error),
            std::cmp::Ordering::Equal => (format!("No clear difference: {t} wins each of {n}"), p.on_bg),
        };
        ui.label(RichText::new(headline).font(theme::heading(18.0)).color(colour));
        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(560.0), 12.0), egui::Sense::hover());
        let total = rep.judged as f32;
        let mut x = rect.left();
        for (n, c) in [(rep.tuned_wins, p.primary), (rep.ties, p.outline_variant), (rep.base_wins, p.outline)] {
            let w = rect.width() * n as f32 / total;
            if w > 0.0 {
                ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(x, rect.top()), egui::vec2(w, rect.height())), 0.0, c);
            }
            x += w;
        }
        widgets::hint(ui, format!("tuned better {} · {} · base better {}", rep.tuned_wins, plural(rep.ties, "tie"), rep.base_wins));
        let mut facts = vec![];
        if let (Some(b), Some(t)) = (rep.base_score, rep.tuned_score) {
            facts.push(format!("Judge's average: tuned {t:.1}, base {b:.1} (of 10)"));
        }
        if rep.tuned_overlap > 0.0 || rep.base_overlap > 0.0 {
            facts.push(format!("Wording shared with the held-back replies: tuned {:.0}%, base {:.0}%", rep.tuned_overlap * 100.0, rep.base_overlap * 100.0));
        }
        for f in facts {
            ui.label(RichText::new(f).small());
        }
        if rep.judged < rep.items {
            widgets::hint(ui, format!("{} not judged (an answer failed or the judge did).", rep.items - rep.judged));
        }
    }
    if widgets::button(ui, Kind::Ghost, "See every answer in the Arena").clicked() {
        app.arena.open = Some(r.id.clone());
        app.arena.view = View::Run;
        app.arena.run_view = RunView::Answers;
        app.tab = crate::app::Tab::Arena;
    }
}

