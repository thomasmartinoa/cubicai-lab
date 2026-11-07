//! The dataset builder in Fine-tune's Data step: add examples from many places, clean them up,
//! let an AI grade and improve them, see the set at a glance, hold some back for testing, and go
//! back to an earlier version. Long jobs run in the background and land as they come.

use crate::app::App;
use crate::engine::datakit::{self, Change, Found, Op, TopicMap};
use crate::engine::dataset::{self, Dataset, Example, Pair, Turn};
use crate::engine::{xray, Task};
use crate::theme::Palette;
use crate::widgets::{self, Kind};
use egui::{pos2, vec2, Color32, RichText, Sense, Stroke};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Add,
    Clean,
    Improve,
    Overview,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source {
    Draft,
    Docs,
    Hf,
    File,
    Chats,
    Arena,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Filter {
    All,
    Weak,
    Ungraded,
    Held,
    Pairs,
}

/// What a background job hands back, a piece at a time.
pub enum Out {
    Add { examples: Vec<Example>, pairs: Vec<Pair>, system: Option<String> },
    Grades(Vec<(String, u8, String)>),
    Changes(Vec<(String, Change)>),
    Tokens(Vec<(String, u32)>),
}

pub struct Job {
    pub task: Arc<Task>,
    pub dataset: String,
    pub out: Arc<Mutex<Vec<Out>>>,
    /// Added and changed so far, for the summary when it ends.
    pub added: usize,
    pub changed: usize,
}

pub struct BuildUi {
    pub tab: Tab,
    pub source: Source,
    pub filter: Filter,
    pub search: String,
    pub page: usize,
    pub job: Option<Job>,
    pub docs: Vec<std::path::PathBuf>,
    pub per_piece: usize,
    pub hf_name: String,
    pub hf_limit: usize,
    pub dups: Option<(String, usize, Vec<(usize, usize, f32)>)>,
    pub dup_threshold: f32,
    pub min_reply: usize,
    pub max_tokens: usize,
    pub found: Option<(String, Found)>,
    pub ai_count: usize,
    pub style: String,
    pub topics: Option<(String, usize, TopicMap)>,
    pub versions_open: bool,
}

impl Default for BuildUi {
    fn default() -> Self {
        BuildUi {
            tab: Tab::Add,
            source: Source::Draft,
            filter: Filter::All,
            search: String::new(),
            page: 0,
            job: None,
            docs: vec![],
            per_piece: 4,
            hf_name: String::new(),
            hf_limit: 500,
            dups: None,
            dup_threshold: 0.8,
            min_reply: 0,
            max_tokens: 0,
            found: None,
            ai_count: 20,
            style: String::new(),
            topics: None,
            versions_open: false,
        }
    }
}

impl BuildUi {
    pub fn busy(&self) -> bool {
        self.job.as_ref().is_some_and(|j| !j.task.is_done())
    }
}

// --- background jobs --------------------------------------------------------------------------------

fn start_job(app: &mut App, d: &Dataset, label: &str, ctx: &egui::Context, work: impl FnOnce(&Arc<Task>, &dyn Fn(Out)) -> Result<(), String> + Send + 'static) {
    let mut c = label.chars();
    let lower = c.next().map(|f| f.to_lowercase().chain(c).collect::<String>()).unwrap_or_default();
    let _ = app.datasets.snapshot(d, &format!("Before {lower}"));
    let task = Task::new(label);
    let out: Arc<Mutex<Vec<Out>>> = Default::default();
    let sink = out.clone();
    let c2 = ctx.clone();
    app.start_task(&task, ctx, move |t| {
        let push = |o: Out| {
            sink.lock().unwrap().push(o);
            c2.request_repaint();
        };
        work(t, &push)
    });
    app.tune.build.job = Some(Job { task, dataset: d.id.clone(), out, added: 0, changed: 0 });
}

/// Applies what jobs have handed back so far; called every frame.
pub fn pump(app: &mut App) {
    let Some(job) = app.tune.build.job.as_mut() else { return };
    let got: Vec<Out> = std::mem::take(&mut *job.out.lock().unwrap());
    let done = job.task.is_done();
    if !got.is_empty()
        && let Some(d) = app.tune.datasets.iter_mut().find(|x| x.id == job.dataset)
    {
        for o in got {
            match o {
                Out::Add { examples, pairs, system } => {
                    if d.system.trim().is_empty()
                        && let Some(s) = system
                    {
                        d.system = s;
                    }
                    let (e, p) = datakit::merge(d, examples, pairs);
                    job.added += e + p;
                }
                Out::Grades(g) => {
                    for (id, score, why) in g {
                        if let Some(e) = d.examples.iter_mut().find(|e| e.id == id) {
                            e.score = Some(score);
                            e.note = why;
                            job.changed += 1;
                        }
                    }
                }
                Out::Changes(c) => {
                    for (id, ch) in c {
                        match ch {
                            Change::Replace(mut n) => {
                                if let Some(e) = d.examples.iter_mut().find(|e| e.id == id) {
                                    n.id = id;
                                    n.held = e.held;
                                    *e = n;
                                    job.changed += 1;
                                }
                            }
                            Change::Add(n) => {
                                let (e, _) = datakit::merge(d, vec![n], vec![]);
                                job.added += e;
                            }
                        }
                    }
                }
                Out::Tokens(t) => {
                    for (id, n) in t {
                        if let Some(e) = d.examples.iter_mut().find(|e| e.id == id) {
                            e.tokens = Some(n);
                            job.changed += 1;
                        }
                    }
                }
            }
        }
        let _ = app.datasets.save(d);
    }
    if done {
        let job = app.tune.build.job.take().unwrap();
        let mut msg = vec![];
        if job.added > 0 {
            msg.push(format!("added {}", job.added));
        }
        if job.changed > 0 {
            msg.push(format!("updated {}", job.changed));
        }
        match job.task.error().filter(|e| e != "Stopped") {
            Some(e) => app.tune.error = Some(if msg.is_empty() { e } else { format!("{e} (before that: {})", msg.join(", ")) }),
            None if !msg.is_empty() => app.notify(format!("{}: {}", job.task.status(), msg.join(", "))),
            None => {}
        }
    }
}

/// A request to the drafting provider, ready to have its messages swapped per batch.
fn ai_request(app: &mut App) -> Result<crate::llm::Request, String> {
    let p = app.tune.draft_provider.clone();
    let mut r = app.provider_request(&p, vec![])?;
    r.extra.insert("temperature".into(), json!(0.4));
    Ok(r)
}

/// Runs `op` over `targets` in batches through the drafting provider.
pub(crate) fn start_op(app: &mut App, d: &Dataset, op: Op, targets: Vec<Example>, ctx: &egui::Context) {
    let req = match ai_request(app) {
        Ok(r) => r,
        Err(e) => {
            app.tune.error = Some(e);
            return;
        }
    };
    let ds = d.clone();
    let label = op.label();
    start_job(app, d, label, ctx, move |t, push| {
        let cancel = AtomicBool::new(false);
        let size = 8;
        let total = targets.len().div_ceil(size);
        let mut failures = 0;
        for (b, batch) in targets.chunks(size).enumerate() {
            if t.cancel.load(Ordering::Relaxed) {
                break;
            }
            t.set_status(format!("{label}: batch {} of {total}", b + 1));
            t.progress.store(((b * 1000) / total.max(1)) as u32, Ordering::Relaxed);
            let refs: Vec<&Example> = batch.iter().collect();
            let mut r = req.clone();
            r.messages = datakit::op_messages(&op, &ds, &refs);
            let parsed = crate::llm::collect(&r, &cancel).and_then(|reply| datakit::parse_op(&reply, batch.len()));
            match parsed {
                Ok(entries) => {
                    let changes: Vec<(String, Change)> = entries.iter().filter_map(|(i, v)| Some((batch[*i].id.clone(), datakit::apply_op(&op, &batch[*i], v)?))).collect();
                    push(Out::Changes(changes));
                }
                Err(e) => {
                    failures += 1;
                    if failures >= 3 {
                        return Err(e);
                    }
                }
            }
        }
        t.set_status(label.to_string());
        Ok(())
    });
}

pub(crate) fn start_grading(app: &mut App, d: &Dataset, targets: Vec<Example>, ctx: &egui::Context) {
    let req = match ai_request(app) {
        Ok(r) => r,
        Err(e) => {
            app.tune.error = Some(e);
            return;
        }
    };
    let ds = d.clone();
    start_job(app, d, "Grading", ctx, move |t, push| {
        let cancel = AtomicBool::new(false);
        let size = 10;
        let total = targets.len().div_ceil(size);
        let mut failures = 0;
        for (b, batch) in targets.chunks(size).enumerate() {
            if t.cancel.load(Ordering::Relaxed) {
                break;
            }
            t.set_status(format!("Grading: batch {} of {total}", b + 1));
            t.progress.store(((b * 1000) / total.max(1)) as u32, Ordering::Relaxed);
            let mut r = req.clone();
            r.messages = datakit::grade_messages(&ds, &batch.iter().collect::<Vec<_>>());
            match crate::llm::collect(&r, &cancel).and_then(|reply| datakit::parse_grades(&reply, batch.len())) {
                Ok(g) => push(Out::Grades(g.into_iter().map(|(i, s, why)| (batch[i].id.clone(), s, why)).collect())),
                Err(e) => {
                    failures += 1;
                    if failures >= 3 {
                        return Err(e);
                    }
                }
            }
        }
        t.set_status("Grading".into());
        Ok(())
    });
}

pub(crate) fn start_docs(app: &mut App, d: &Dataset, ctx: &egui::Context) {
    let req = match ai_request(app) {
        Ok(r) => r,
        Err(e) => {
            app.tune.error = Some(e);
            return;
        }
    };
    let files = std::mem::take(&mut app.tune.build.docs);
    let per = app.tune.build.per_piece;
    let ds = d.clone();
    start_job(app, d, "Reading documents", ctx, move |t, push| {
        let cancel = AtomicBool::new(false);
        let mut pieces = vec![];
        for f in &files {
            t.set_status(format!("Reading {}", f.file_name().unwrap_or_default().to_string_lossy()));
            let text = datakit::read_document(f)?;
            let title = f.file_stem().unwrap_or_default().to_string_lossy().to_string();
            pieces.extend(datakit::chunks(&text, 3500).into_iter().map(|c| (title.clone(), c)));
        }
        let total = pieces.len();
        for (k, (title, chunk)) in pieces.iter().enumerate() {
            if t.cancel.load(Ordering::Relaxed) {
                break;
            }
            t.set_status(format!("Writing questions: piece {} of {total} ({title})", k + 1));
            t.progress.store(((k * 1000) / total.max(1)) as u32, Ordering::Relaxed);
            let mut r = req.clone();
            r.messages = datakit::doc_messages(title, chunk, per, &ds);
            let reply = crate::llm::collect(&r, &cancel)?;
            if let Ok(mut ex) = dataset::parse_draft(&reply) {
                ex.iter_mut().for_each(|e| e.source = "doc".into());
                push(Out::Add { examples: ex, pairs: vec![], system: None });
            }
        }
        t.set_status("Documents".into());
        Ok(())
    });
}

pub(crate) fn start_hf(app: &mut App, d: &Dataset, ctx: &egui::Context) {
    let Some(name) = datakit::hf_name(&app.tune.build.hf_name) else {
        app.tune.error = Some("Give the dataset as owner/name, e.g. HuggingFaceH4/no_robots.".into());
        return;
    };
    let (limit, token) = (app.tune.build.hf_limit, app.secrets.get("hf"));
    start_job(app, d, "Importing from Hugging Face", ctx, move |t, push| {
        t.set_status(format!("Looking up {name}"));
        let (config, split) = datakit::hf_split(datakit::HF_ROWS, &name, &token)?;
        t.set_status(format!("Reading {name} ({split})"));
        let (rows, total) = datakit::hf_rows(datakit::HF_ROWS, &name, &config, &split, limit, &token, &|| t.cancel.load(Ordering::Relaxed))?;
        let mut got = datakit::Imported::default();
        for r in &rows {
            datakit::read_record(r, &mut got, "hf");
        }
        if got.examples.is_empty() && got.pairs.is_empty() {
            return Err(format!("{name} has rows, but none in a shape the Lab reads ({} skipped).", got.skipped));
        }
        t.set_status(format!("{name}: {} of {} rows read", rows.len(), total.map(|n| n.to_string()).unwrap_or("?".into())));
        push(Out::Add { examples: got.examples, pairs: got.pairs, system: got.system });
        Ok(())
    });
}

pub(crate) fn start_tokens(app: &mut App, d: &Dataset, ctx: &egui::Context) {
    let root = xray::root_of(&app.server.base_url());
    let ds = d.clone();
    start_job(app, d, "Counting tokens", ctx, move |t, push| {
        let todo: Vec<&Example> = ds.examples.iter().filter(|e| e.valid()).collect();
        let total = todo.len();
        let mut batch = vec![];
        for (k, e) in todo.into_iter().enumerate() {
            if t.cancel.load(Ordering::Relaxed) {
                break;
            }
            let mut m = vec![];
            if !ds.system.trim().is_empty() {
                m.push(json!({"role": "system", "content": ds.system.trim()}));
            }
            m.extend(e.turns.iter().map(|x| json!({"role": x.role, "content": x.content.trim()})));
            let text = xray::apply_template(&root, &m)?;
            let n = crate::engine::steer::tokenize(&root, &text)?.len() as u32;
            batch.push((e.id.clone(), n));
            if batch.len() >= 25 || k + 1 == total {
                t.set_status(format!("Counting tokens: {} of {total}", k + 1));
                t.progress.store((((k + 1) * 1000) / total.max(1)) as u32, Ordering::Relaxed);
                push(Out::Tokens(std::mem::take(&mut batch)));
            }
        }
        t.set_status("Token counts".into());
        Ok(())
    });
}

// --- the panel ----------------------------------------------------------------------------------------

/// The builder above the examples. `d` is this frame's copy of the dataset (saved by the caller when changed).
pub fn panel(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    if let Some(j) = app.tune.build.job.as_ref().filter(|j| !j.task.is_done()) {
        widgets::surface(ui, p.primary_container, Stroke::NONE, 6.0, 10, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().color(p.on_primary_container));
                ui.label(RichText::new(j.task.status()).color(p.on_primary_container));
                if j.added + j.changed > 0 {
                    ui.label(RichText::new(format!("· {} so far", j.added + j.changed)).small().color(p.on_primary_container));
                }
                if ui.small_button("Stop").clicked() {
                    j.task.stop();
                }
            });
        });
        ui.add_space(8.0);
    }
    let mut tab = app.tune.build.tab;
    widgets::segmented(ui, &mut tab, &[(Tab::Add, "Add"), (Tab::Clean, "Clean"), (Tab::Improve, "Improve"), (Tab::Overview, "Overview")]);
    app.tune.build.tab = tab;
    ui.add_space(8.0);
    match tab {
        Tab::Add => add_card(app, ui, p, d),
        Tab::Clean => clean_card(app, ui, p, d),
        Tab::Improve => improve_card(app, ui, p, d),
        Tab::Overview => overview_card(app, ui, p, d),
    }
}

fn provider_row(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("AI");
        let cur = crate::providers::get(&app.tune.draft_provider);
        egui::ComboBox::from_id_salt("build-provider").selected_text(cur.short).show_ui(ui, |ui| {
            for pr in crate::providers::ALL.iter().filter(|pr| pr.id != crate::providers::PC) {
                ui.selectable_value(&mut app.tune.draft_provider, pr.id.to_string(), pr.short);
            }
        });
        let prov = app.tune.draft_provider.clone();
        let mut model = app.settings.models.get(&prov).cloned().unwrap_or_default();
        crate::ui::arena::model_choice(app, ui, &prov, &mut model, "build");
        // The pick is the provider's model everywhere it's used (as in chat).
        if app.settings.models.get(&prov).cloned().unwrap_or_default() != model {
            app.settings.models.insert(prov, model);
            app.save_settings();
        }
    });
}

fn add_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    let busy = app.tune.build.busy() || app.tune.draft.as_ref().is_some_and(|x| !x.task.is_done());
    widgets::card(ui, |ui| {
        let mut s = app.tune.build.source;
        widgets::segmented(ui, &mut s, &[(Source::Draft, "AI draft"), (Source::Docs, "Documents"), (Source::Hf, "Hugging Face"), (Source::File, "File"), (Source::Chats, "Chats"), (Source::Arena, "Arena")]);
        app.tune.build.source = s;
        ui.add_space(8.0);
        match s {
            Source::Draft => crate::ui::tune::draft_card(app, ui, p, d),
            Source::Docs => {
                widgets::hint(ui, "Your own material (PDF, Markdown, text, HTML) cut into pieces; an AI writes questions each piece answers, with answers drawn only from it. Good for teaching a model your notes, manuals or docs.");
                ui.horizontal(|ui| {
                    if widgets::button(ui, Kind::Tonal, "Choose files…").clicked()
                        && let Some(files) = rfd::FileDialog::new().add_filter("Documents", &["pdf", "md", "markdown", "txt", "html", "htm", "rst", "org"]).pick_files()
                    {
                        app.tune.build.docs = files;
                    }
                    for f in &app.tune.build.docs {
                        widgets::badge(ui, &f.file_name().unwrap_or_default().to_string_lossy(), p.on_bg, p.high);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Questions per piece");
                    ui.add(egui::DragValue::new(&mut app.tune.build.per_piece).range(1..=12));
                    widgets::hint(ui, "(a piece is about 3,500 characters, roughly a page)");
                });
                provider_row(app, ui);
                if widgets::button_ext(ui, Kind::Primary, "Write questions", !busy && !app.tune.build.docs.is_empty()).clicked() {
                    start_docs(app, d, ui.ctx());
                }
            }
            Source::Hf => {
                widgets::hint(ui, "Any dataset with a viewer on huggingface.co, read through its rows API: chat messages, ShareGPT, Alpaca, prompt/response or question/answer, and prompt/chosen/rejected as preference pairs.");
                ui.horizontal(|ui| {
                    ui.label("Dataset");
                    ui.add(egui::TextEdit::singleline(&mut app.tune.build.hf_name).desired_width(320.0).hint_text("HuggingFaceH4/no_robots"));
                    ui.label("first");
                    ui.add(egui::DragValue::new(&mut app.tune.build.hf_limit).range(10..=5000).speed(10));
                    ui.label("rows");
                });
                widgets::hint(ui, "Gated datasets use the Hugging Face token from Settings.");
                if widgets::button_ext(ui, Kind::Primary, "Import", !busy && !app.tune.build.hf_name.trim().is_empty()).clicked() {
                    start_hf(app, d, ui.ctx());
                }
            }
            Source::File => {
                widgets::hint(ui, "JSON or JSON Lines in the usual shapes: OpenAI chat messages, ShareGPT (from/value), Alpaca (instruction/input/output), prompt/response, question/answer, and prompt/chosen/rejected for preference pairs.");
                if widgets::button_ext(ui, Kind::Primary, "Choose files…", !busy).clicked()
                    && let Some(files) = rfd::FileDialog::new().add_filter("Data", &["jsonl", "json", "ndjson"]).pick_files()
                {
                    let _ = app.datasets.snapshot(d, "Before importing files");
                    let mut msg = vec![];
                    for f in files {
                        let name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
                        match std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| datakit::parse_records(&t, "file")) {
                            Ok(got) => {
                                if d.system.trim().is_empty()
                                    && let Some(s) = got.system
                                {
                                    d.system = s;
                                }
                                let (e, pr) = datakit::merge(d, got.examples, got.pairs);
                                msg.push(format!("{name}: {e} examples, {pr} pairs{}", if got.skipped > 0 { format!(" ({} skipped)", got.skipped) } else { String::new() }));
                            }
                            Err(e) => msg.push(format!("{name}: {e}")),
                        }
                    }
                    app.notify(msg.join(" · "));
                }
            }
            Source::Chats => {
                widgets::hint(ui, "Your Lab chats, each as one conversation (failed replies left out).");
                ui.horizontal(|ui| {
                    let chats: Vec<(String, String)> = app.chat.chats.iter().filter(|c| c.messages.len() >= 2).map(|c| (c.id.clone(), c.title.clone())).collect();
                    egui::ComboBox::from_id_salt("import-chat").selected_text("Add a chat…").width(260.0).height(400.0).show_ui(ui, |ui| {
                        for (id, title) in &chats {
                            if ui.selectable_label(false, if title.is_empty() { "Untitled" } else { title }).clicked()
                                && let Some(e) = app.chat.chats.iter().find(|c| &c.id == id).and_then(dataset::from_chat)
                            {
                                datakit::merge(d, vec![e], vec![]);
                            }
                        }
                    });
                    if widgets::button(ui, Kind::Ghost, "Add all").clicked() {
                        let all: Vec<Example> = app.chat.chats.iter().filter_map(dataset::from_chat).collect();
                        let (n, _) = datakit::merge(d, all, vec![]);
                        app.notify(format!("Added {n} chats"));
                    }
                });
            }
            Source::Arena => {
                let (ex, pairs) = datakit::from_arena(&app.arena.runs);
                widgets::hint(ui, "Your votes in the Arena: each prompt with the answer you picked becomes an example, and your pick against each other answer a preference pair (for DPO training).");
                widgets::hint(ui, format!("Available: {} examples and {} pairs from {} comparisons.", ex.len(), pairs.len(), app.arena.runs.len()));
                if widgets::button_ext(ui, Kind::Primary, "Add my picks", !ex.is_empty() || !pairs.is_empty()).clicked() {
                    let (e, pr) = datakit::merge(d, ex, pairs);
                    app.notify(format!("Added {e} examples and {pr} pairs from the Arena"));
                }
            }
        }
    });
}

fn clean_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    let busy = app.tune.build.busy();
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Duplicates");
        ui.horizontal(|ui| {
            ui.label("Near duplicates from");
            ui.add(egui::Slider::new(&mut app.tune.build.dup_threshold, 0.5..=0.98).step_by(0.01).custom_formatter(|v, _| format!("{:.0}% alike", v * 100.0)));
            if widgets::button(ui, Kind::Tonal, "Find").clicked() {
                app.tune.build.dups = Some((d.id.clone(), d.examples.len(), datakit::duplicates(&d.examples, app.tune.build.dup_threshold)));
            }
        });
        let dups = app.tune.build.dups.clone().filter(|x| x.0 == d.id && x.1 == d.examples.len());
        if let Some((_, _, list)) = dups {
            let exact = list.iter().filter(|x| x.2 >= 1.0).count();
            if list.is_empty() {
                widgets::hint(ui, "No duplicates.");
            } else {
                widgets::hint(ui, format!("{} duplicates: {exact} exact, {} near. The first copy of each is kept.", list.len(), list.len() - exact));
                for (late, early, sim) in list.iter().take(4) {
                    widgets::hint(ui, format!("#{} repeats #{} ({:.0}%): {}", late + 1, early + 1, sim * 100.0, d.examples[*late].first("user").chars().take(90).collect::<String>()));
                }
                if widgets::button(ui, Kind::Danger, format!("Remove {}", list.len())).clicked() {
                    let _ = app.datasets.snapshot(d, "Before removing duplicates");
                    let drop: std::collections::HashSet<usize> = list.iter().map(|x| x.0).collect();
                    let mut i = 0;
                    d.examples.retain(|_| {
                        i += 1;
                        !drop.contains(&(i - 1))
                    });
                    app.tune.build.dups = None;
                }
            }
        }
    });
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Length");
        if app.tune.build.max_tokens == 0 {
            app.tune.build.max_tokens = app.tune.params.max_seq as usize;
        }
        ui.horizontal(|ui| {
            ui.label("Replies shorter than");
            ui.add(egui::DragValue::new(&mut app.tune.build.min_reply).range(0..=2000).suffix(" characters"));
            ui.label("or conversations longer than");
            ui.add(egui::DragValue::new(&mut app.tune.build.max_tokens).range(0..=32768).speed(16).suffix(" tokens"));
        });
        let out = datakit::out_of_length(&d.examples, app.tune.build.min_reply, app.tune.build.max_tokens);
        let counted = d.examples.iter().filter(|e| e.tokens.is_some()).count();
        widgets::hint(ui, format!(
            "{} outside the limits. Training cuts anything longer than its max length ({} tokens now).{}",
            out.len(),
            app.tune.params.max_seq,
            if counted < d.examples.len() { " Uncounted examples are estimated at ~4 characters a token." } else { "" }
        ));
        if !out.is_empty() && widgets::button(ui, Kind::Danger, format!("Remove {}", out.len())).clicked() {
            let _ = app.datasets.snapshot(d, "Before the length filter");
            let drop: std::collections::HashSet<usize> = out.into_iter().collect();
            let mut i = 0;
            d.examples.retain(|_| {
                i += 1;
                !drop.contains(&(i - 1))
            });
        }
        ui.add_space(6.0);
        let ready = app.server.state() == crate::engine::server::State::Ready;
        ui.horizontal(|ui| {
            if widgets::button_ext(ui, Kind::Tonal, "Count tokens exactly", ready && !busy).clicked() {
                start_tokens(app, d, ui.ctx());
            }
            widgets::hint(ui, if ready {
                format!("with {}'s tokenizer and chat template", app.server.model_name)
            } else {
                "Start a model in the Engine Room (ideally the one you'll train) to count with its own tokenizer.".into()
            });
        });
        if counted > 0 {
            let toks: Vec<u32> = d.examples.iter().filter_map(|e| e.tokens).collect();
            let over = toks.iter().filter(|&&t| t > app.tune.params.max_seq).count();
            widgets::hint(ui, format!("{counted} counted: longest {} tokens, average {:.0}{}", toks.iter().max().unwrap_or(&0), toks.iter().sum::<u32>() as f32 / toks.len() as f32, if over > 0 { format!(", {over} longer than the max length") } else { String::new() }));
        }
    });
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Personal details");
        widgets::hint(ui, "Email addresses, phone numbers and API keys or tokens, replaced with [email], [phone] and [api key] so a model never learns them.");
        ui.horizontal(|ui| {
            if widgets::button(ui, Kind::Tonal, "Scan").clicked() {
                app.tune.build.found = Some((d.id.clone(), datakit::scan(d)));
            }
            if let Some((_, f)) = app.tune.build.found.clone().filter(|x| x.0 == d.id) {
                if f.total() == 0 {
                    widgets::hint(ui, "Nothing found.");
                } else {
                    ui.label(format!("{} emails · {} phone numbers · {} keys", f.emails, f.phones, f.keys));
                    if widgets::button(ui, Kind::Primary, "Replace them").clicked() {
                        let _ = app.datasets.snapshot(d, "Before replacing personal details");
                        let done = datakit::scrub_dataset(d);
                        app.notify(format!("Replaced {}", done.total()));
                        app.tune.build.found = Some((d.id.clone(), Found::default()));
                    }
                }
            }
        });
    });
    let _ = p;
}

fn improve_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    let busy = app.tune.build.busy();
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Grade with an AI");
        widgets::hint(ui, "Another model grades each example from 1 (harmful to train on) to 5 (exactly right) against the brief, with a reason. Ten examples per request.");
        provider_row(app, ui);
        let ungraded: Vec<Example> = d.examples.iter().filter(|e| e.valid() && e.score.is_none()).cloned().collect();
        let mut hist = [0usize; 5];
        for s in d.examples.iter().filter_map(|e| e.score) {
            hist[(s.clamp(1, 5) - 1) as usize] += 1;
        }
        ui.horizontal(|ui| {
            if widgets::button_ext(ui, Kind::Primary, format!("Grade {} ungraded", ungraded.len()), !busy && !ungraded.is_empty()).clicked() {
                start_grading(app, d, ungraded.clone(), ui.ctx());
            }
            let all: Vec<Example> = d.examples.iter().filter(|e| e.valid()).cloned().collect();
            if hist.iter().sum::<usize>() > 0 && widgets::button_ext(ui, Kind::Ghost, "Grade all again", !busy).clicked() {
                start_grading(app, d, all, ui.ctx());
            }
        });
        if hist.iter().sum::<usize>() > 0 {
            ui.horizontal(|ui| {
                for (k, n) in hist.iter().enumerate().rev() {
                    let c = if k <= 1 { p.error } else if k == 2 { p.on_variant } else { p.good };
                    widgets::badge(ui, &format!("{} ★ {n}", k + 1), c, p.high);
                }
            });
            let weak: Vec<Example> = d.examples.iter().filter(|e| e.score.is_some_and(|s| s <= 2)).cloned().collect();
            if !weak.is_empty() {
                ui.horizontal(|ui| {
                    if widgets::button_ext(ui, Kind::Tonal, format!("Fix {} weak replies", weak.len()), !busy).on_hover_text("Rewrites the replies graded 1 or 2, using the reviewer's reason").clicked() {
                        start_op(app, d, Op::Fix, weak.clone(), ui.ctx());
                    }
                    if widgets::button(ui, Kind::Danger, format!("Drop {} weak", weak.len())).clicked() {
                        let _ = app.datasets.snapshot(d, "Before dropping weak examples");
                        d.examples.retain(|e| !e.score.is_some_and(|s| s <= 2));
                    }
                });
            }
        }
    });
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "More variety");
        let singles: Vec<Example> = d.examples.iter().filter(|e| e.valid() && !e.held && e.source != "reword").cloned().collect();
        ui.horizontal(|ui| {
            ui.label("For");
            ui.add(egui::DragValue::new(&mut app.tune.build.ai_count).range(1..=500));
            ui.label("examples (spread through the set):");
        });
        let picked = spread(&singles, app.tune.build.ai_count);
        ui.horizontal(|ui| {
            if widgets::button_ext(ui, Kind::Tonal, "Reword the questions", !busy && !picked.is_empty()).on_hover_text("New examples: the same request in other words, with the same reply, so the model copes with how people really ask").clicked() {
                start_op(app, d, Op::Reword, picked.clone(), ui.ctx());
            }
            let short: Vec<Example> = picked.iter().filter(|e| e.turns.len() == 2).cloned().collect();
            if widgets::button_ext(ui, Kind::Tonal, "Extend into conversations", !busy && !short.is_empty()).on_hover_text("New examples: the conversation carried on with a follow-up, so the model keeps its style past the first reply").clicked() {
                start_op(app, d, Op::Extend, short, ui.ctx());
            }
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Restyle every reply as");
            ui.add(egui::TextEdit::singleline(&mut app.tune.build.style).desired_width(280.0).hint_text("e.g. a calm teacher, short and warm"));
            let all: Vec<Example> = d.examples.iter().filter(|e| e.valid()).cloned().collect();
            if widgets::button_ext(ui, Kind::Tonal, format!("Restyle {}", all.len()), !busy && !app.tune.build.style.trim().is_empty() && !all.is_empty()).clicked() {
                start_op(app, d, Op::Restyle(app.tune.build.style.clone()), all, ui.ctx());
            }
        });
        widgets::hint(ui, "Uses the AI chosen above. A copy of the set is kept first (Versions).");
    });
}

/// `n` examples spread evenly through the list.
fn spread(list: &[Example], n: usize) -> Vec<Example> {
    if list.len() <= n {
        return list.to_vec();
    }
    (0..n).map(|k| list[k * list.len() / n].clone()).collect()
}

const CLUSTER_COLOURS: [Color32; 8] = [
    Color32::from_rgb(0xFF, 0xB5, 0x47),
    Color32::from_rgb(0x6F, 0x9B, 0xFF),
    Color32::from_rgb(0x5F, 0xD0, 0x9A),
    Color32::from_rgb(0xE8, 0x6A, 0x8C),
    Color32::from_rgb(0xB3, 0x8B, 0xF5),
    Color32::from_rgb(0x4F, 0xC8, 0xD8),
    Color32::from_rgb(0xD9, 0xC2, 0x5A),
    Color32::from_rgb(0xA0, 0xA8, 0xC0),
];

fn overview_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "At a glance");
        let mut by_source: Vec<(String, usize)> = vec![];
        for e in d.examples.iter().filter(|e| e.valid()) {
            match by_source.iter_mut().find(|x| x.0 == e.source) {
                Some(x) => x.1 += 1,
                None => by_source.push((e.source.clone(), 1)),
            }
        }
        by_source.sort_by_key(|x| std::cmp::Reverse(x.1));
        ui.horizontal_wrapped(|ui| {
            ui.label("From");
            for (s, n) in &by_source {
                widgets::badge(ui, &format!("{} {n}", source_name(s)), p.on_bg, p.high);
            }
            if d.pair_count() > 0 {
                widgets::badge(ui, &format!("preference pairs {}", d.pair_count()), p.on_primary_container, p.primary_container);
            }
        });
        ui.add_space(6.0);
        let lens: Vec<f32> = d.examples.iter().filter(|e| e.valid()).map(|e| e.tokens.map(|t| t as f32).unwrap_or(e.chars() as f32 / 4.0)).collect();
        if !lens.is_empty() {
            let counted = d.examples.iter().filter(|e| e.valid() && e.tokens.is_some()).count() == lens.len();
            ui.label(RichText::new(format!("Length in tokens{}", if counted { "" } else { " (~ estimated)" })).small().color(p.on_variant));
            histogram(ui, p, &lens, app.tune.params.max_seq as f32);
        }
    });
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Topics");
        widgets::hint(ui, "What the users ask, grouped by shared words and laid out so similar requests sit together. Big clusters you didn't intend, or a topic you expected and can't find, are worth fixing before training.");
        let texts: Vec<(usize, String)> = d.examples.iter().enumerate().filter(|(_, e)| e.valid()).map(|(i, e)| (i, e.turns.iter().filter(|t| t.role == "user").map(|t| t.content.as_str()).collect::<Vec<_>>().join(" "))).take(3000).collect();
        if widgets::button_ext(ui, Kind::Tonal, "Map the topics", texts.len() >= 2).clicked() {
            let only: Vec<String> = texts.iter().map(|t| t.1.clone()).collect();
            app.tune.build.topics = Some((d.id.clone(), d.examples.len(), datakit::topic_map(&only, datakit::topic_count(only.len()))));
        }
        if let Some((_, _, m)) = app.tune.build.topics.as_ref().filter(|x| x.0 == d.id && x.1 == d.examples.len()) {
            topic_plot(ui, p, m, &texts);
        }
    });
    ui.add_space(8.0);
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Test split");
        widgets::hint(ui, "Held-back examples aren't trained on. After training, the report card asks the base and the tuned model their questions and compares the answers with these replies.");
        ui.horizontal(|ui| {
            if widgets::button(ui, Kind::Tonal, "Hold back 10%").clicked() {
                let _ = app.datasets.snapshot(d, "Before a new test split");
                let n = datakit::hold_out(d, 0.1);
                app.notify(if n == 0 { "Too few single-exchange examples to spare any (10 or more needed)".to_string() } else { format!("Holding back {n} for testing") });
            }
            if d.held_count() > 0 {
                widgets::hint(ui, format!("{} held back ({} usable as report-card questions), {} to train on", d.held_count(), d.held_prompts().len(), d.train_count()));
                if widgets::button(ui, Kind::Ghost, "Put them back").clicked() {
                    d.examples.iter_mut().for_each(|e| e.held = false);
                }
            }
        });
    });
}

fn source_name(s: &str) -> &str {
    match s {
        "ai" => "AI draft",
        "chat" => "chats",
        "hand" => "by hand",
        "doc" => "documents",
        "file" => "files",
        "hf" => "Hugging Face",
        "arena" => "Arena",
        "reword" => "reworded",
        "extend" => "extended",
        "" => "other",
        x => x,
    }
}

fn histogram(ui: &mut egui::Ui, p: &Palette, values: &[f32], limit: f32) {
    let max = values.iter().copied().fold(0f32, f32::max).max(limit.min(values.iter().copied().fold(0f32, f32::max) * 1.1)).max(1.0);
    const BINS: usize = 24;
    let mut bins = [0usize; BINS];
    for v in values {
        bins[((v / max * BINS as f32) as usize).min(BINS - 1)] += 1;
    }
    let top = *bins.iter().max().unwrap_or(&1) as f32;
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width().min(640.0), 70.0), Sense::hover());
    let w = rect.width() / BINS as f32;
    for (i, n) in bins.iter().enumerate() {
        let h = (*n as f32 / top) * (rect.height() - 14.0);
        let x = rect.left() + i as f32 * w;
        let over = (i as f32 + 0.5) * max / BINS as f32 > limit;
        let r = egui::Rect::from_min_max(pos2(x + 1.0, rect.bottom() - 14.0 - h), pos2(x + w - 1.0, rect.bottom() - 14.0));
        ui.painter().rect_filled(r, 0.0, if over { p.error.gamma_multiply(0.8) } else { p.primary.gamma_multiply(0.85) });
    }
    if limit < max {
        let x = rect.left() + limit / max * rect.width();
        ui.painter().line_segment([pos2(x, rect.top()), pos2(x, rect.bottom() - 14.0)], Stroke::new(1.0, p.error));
    }
    let small = egui::TextStyle::Small.resolve(ui.style());
    widgets::text_at(ui.painter(), pos2(rect.left(), rect.bottom()), egui::Align2::LEFT_BOTTOM, "0", small.clone(), p.on_variant);
    widgets::text_at(ui.painter(), pos2(rect.right(), rect.bottom()), egui::Align2::RIGHT_BOTTOM, &format!("{max:.0}"), small, p.on_variant);
    resp.on_hover_text(format!("Red: longer than the max length ({limit:.0}), cut in training"));
}

fn topic_plot(ui: &mut egui::Ui, p: &Palette, m: &TopicMap, texts: &[(usize, String)]) {
    ui.horizontal_wrapped(|ui| {
        for (k, (label, size)) in m.clusters.iter().enumerate() {
            let c = CLUSTER_COLOURS[k % CLUSTER_COLOURS.len()];
            let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 0.0, c);
            ui.label(RichText::new(format!("{label} ({size})")).small());
            ui.add_space(8.0);
        }
    });
    let side = ui.available_width().min(640.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(side, 300.0), Sense::hover());
    ui.painter().add(widgets::cut_shape(rect, 8.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let inner = rect.shrink(14.0);
    let at = |q: [f32; 2]| pos2(inner.center().x + q[0] * inner.width() / 2.0, inner.center().y - q[1] * inner.height() / 2.0);
    let mut nearest: Option<(f32, usize)> = None;
    for (i, q) in m.points.iter().enumerate() {
        // A small fixed nudge per point, so identical questions don't hide behind each other.
        let j = |k: usize| (((i * 7919 + k * 104729) % 1000) as f32 / 1000.0 - 0.5) * 0.04;
        let pt = at([q[0] + j(0), q[1] + j(1)]);
        ui.painter().circle_filled(pt, 3.0, CLUSTER_COLOURS[m.cluster[i] % CLUSTER_COLOURS.len()].gamma_multiply(0.85));
        if let Some(h) = resp.hover_pos() {
            let dist = h.distance(pt);
            if dist < 12.0 && nearest.is_none_or(|n| dist < n.0) {
                nearest = Some((dist, i));
            }
        }
    }
    if let Some((_, i)) = nearest {
        resp.on_hover_text(texts.get(i).map(|t| t.1.chars().take(200).collect::<String>()).unwrap_or_default());
    }
}

// --- the examples list ----------------------------------------------------------------------------------

const PAGE: usize = 25;

pub fn examples(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    ui.horizontal(|ui| {
        let b = &mut app.tune.build;
        let before = (b.filter, b.search.clone());
        widgets::segmented(ui, &mut b.filter, &[(Filter::All, "All"), (Filter::Weak, "Weak"), (Filter::Ungraded, "Ungraded"), (Filter::Held, "Held back"), (Filter::Pairs, "Pairs")]);
        ui.add(egui::TextEdit::singleline(&mut b.search).desired_width(200.0).hint_text("Search"));
        if before != (b.filter, b.search.clone()) {
            b.page = 0;
        }
        if widgets::button(ui, Kind::Tonal, "+ Example").clicked() {
            d.examples.push(Example::pair("", "", "hand"));
            d.ensure_ids();
            b.filter = Filter::All;
            b.search.clear();
            b.page = usize::MAX;
        }
    });
    ui.add_space(6.0);
    if app.tune.build.filter == Filter::Pairs {
        pairs_list(app, ui, p, d);
        return;
    }
    let q = app.tune.build.search.trim().to_lowercase();
    let shown: Vec<usize> = (0..d.examples.len())
        .rev()
        .filter(|&i| {
            let e = &d.examples[i];
            let keep = match app.tune.build.filter {
                Filter::All | Filter::Pairs => true,
                Filter::Weak => e.score.is_some_and(|s| s <= 2),
                Filter::Ungraded => e.score.is_none(),
                Filter::Held => e.held,
            };
            keep && (q.is_empty() || e.text().to_lowercase().contains(&q))
        })
        .collect();
    let pages = shown.len().div_ceil(PAGE).max(1);
    // A new example is added at the end, so it's shown first: page 0.
    let page = if app.tune.build.page == usize::MAX { 0 } else { app.tune.build.page.min(pages - 1) };
    app.tune.build.page = page;
    if pages > 1 {
        ui.horizontal(|ui| {
            if ui.add_enabled(page > 0, egui::Button::new("‹")).clicked() {
                app.tune.build.page = page - 1;
            }
            ui.label(format!("{}–{} of {}", page * PAGE + 1, ((page + 1) * PAGE).min(shown.len()), shown.len()));
            if ui.add_enabled(page + 1 < pages, egui::Button::new("›")).clicked() {
                app.tune.build.page = page + 1;
            }
        });
    }
    let mut remove = None;
    for &i in shown.iter().skip(page * PAGE).take(PAGE) {
        let e = &mut d.examples[i];
        let valid = e.valid();
        widgets::surface(ui, p.container, Stroke::new(1.0, if valid { p.outline_variant } else { p.error.gamma_multiply(0.6) }), 8.0, 10, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                widgets::hint(ui, format!("#{}  ·  {}", i + 1, source_name(&e.source)));
                if let Some(s) = e.score {
                    let c = if s <= 2 { p.error } else if s == 3 { p.on_variant } else { p.good };
                    widgets::badge(ui, &format!("{s} ★"), c, p.high).on_hover_text(if e.note.is_empty() { "AI grade" } else { e.note.as_str() });
                }
                if e.held {
                    widgets::badge(ui, "held back for testing", p.on_primary_container, p.primary_container);
                }
                if let Some(t) = e.tokens {
                    widgets::hint(ui, format!("{t} tokens"));
                }
                if !valid {
                    ui.label(RichText::new("needs a user message and a reply").small().color(p.error));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Remove").clicked() {
                        remove = Some(i);
                    }
                    if e.turns.len() >= 2 && ui.small_button("+ turn").on_hover_text("Continue the conversation").clicked() {
                        e.turns.push(Turn { role: "user".into(), content: String::new() });
                        e.turns.push(Turn { role: "assistant".into(), content: String::new() });
                    }
                });
            });
            if !e.note.is_empty() && e.score.is_some_and(|s| s <= 3) {
                ui.label(RichText::new(&e.note).small().color(p.on_variant));
            }
            for t in &mut e.turns {
                let who = if t.role == "user" { "User" } else { "Model" };
                ui.horizontal_top(|ui| {
                    ui.add_sized([52.0, 18.0], egui::Label::new(RichText::new(who).small().color(if t.role == "user" { p.on_variant } else { p.primary })));
                    if ui.add(egui::TextEdit::multiline(&mut t.content).desired_rows(1).desired_width(f32::INFINITY)).changed() {
                        // An edited example's count and grade are out of date.
                    }
                });
            }
        });
        ui.add_space(6.0);
    }
    if d.examples.is_empty() {
        widgets::hint(ui, "No examples yet. Aim for at least 30, ideally 100 or more.");
    } else if shown.is_empty() {
        widgets::hint(ui, "Nothing matches.");
    }
    if let Some(i) = remove {
        d.examples.remove(i);
    }
}

fn pairs_list(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    widgets::hint(ui, "Preference pairs teach a model which of two replies you prefer (DPO training in step 2). They come from your Arena votes or from files with prompt/chosen/rejected.");
    if d.pairs.is_empty() {
        widgets::hint(ui, "No pairs yet.");
        return;
    }
    let q = app.tune.build.search.trim().to_lowercase();
    let mut remove = None;
    for (i, pr) in d.pairs.iter_mut().enumerate().rev().filter(|(_, x)| q.is_empty() || format!("{} {} {}", x.prompt, x.chosen, x.rejected).to_lowercase().contains(&q)).take(60) {
        widgets::surface(ui, p.container, Stroke::new(1.0, p.outline_variant), 8.0, 10, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                widgets::hint(ui, format!("pair {}  ·  {}", i + 1, source_name(&pr.source)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Remove").clicked() {
                        remove = Some(i);
                    }
                });
            });
            for (label, text, c) in [("User", &mut pr.prompt, p.on_variant), ("Better", &mut pr.chosen, p.good), ("Worse", &mut pr.rejected, p.error)] {
                ui.horizontal_top(|ui| {
                    ui.add_sized([52.0, 18.0], egui::Label::new(RichText::new(label).small().color(c)));
                    ui.add(egui::TextEdit::multiline(text).desired_rows(1).desired_width(f32::INFINITY));
                });
            }
        });
        ui.add_space(6.0);
    }
    if let Some(i) = remove {
        d.pairs.remove(i);
    }
}

// --- versions and export -------------------------------------------------------------------------------------

pub fn header_tools(app: &mut App, ui: &mut egui::Ui, d: &mut Dataset) {
    if widgets::button(ui, Kind::Ghost, "Versions").on_hover_text("Copies kept before each bulk change").clicked() {
        app.tune.build.versions_open = !app.tune.build.versions_open;
    }
    if widgets::button(ui, Kind::Ghost, "Export").on_hover_text("The training examples as chat JSONL (and the pairs, if any)").clicked()
        && let Some(path) = rfd::FileDialog::new().set_file_name(format!("{}.jsonl", d.name.replace(['/', '\\'], "-"))).add_filter("JSON Lines", &["jsonl"]).save_file()
    {
        let mut msg = match std::fs::write(&path, d.to_jsonl()) {
            Ok(()) => format!("Saved {} examples", d.train_count()),
            Err(e) => format!("Couldn't save: {e}"),
        };
        if d.pair_count() > 0 {
            let pp = path.with_file_name(format!("{}.pairs.jsonl", path.file_stem().unwrap_or_default().to_string_lossy()));
            if std::fs::write(&pp, d.pairs_jsonl()).is_ok() {
                msg += &format!(" and {} pairs", d.pair_count());
            }
        }
        app.notify(msg);
    }
}

pub fn versions(app: &mut App, ui: &mut egui::Ui, p: &Palette, d: &mut Dataset) {
    if !app.tune.build.versions_open {
        return;
    }
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Versions");
        let list = app.datasets.versions(&d.id);
        if list.is_empty() {
            widgets::hint(ui, "None yet: a copy is kept before each bulk change (imports, cleaning, AI rewrites, the test split).");
        }
        for v in list.iter().take(dataset::MAX_VERSIONS) {
            ui.horizontal(|ui| {
                ui.label(&v.label);
                widgets::hint(ui, format!("{} · {} examples{}", ago(v.at), v.dataset.examples.len(), if v.dataset.pairs.is_empty() { String::new() } else { format!(", {} pairs", v.dataset.pairs.len()) }));
                if ui.small_button("Restore").clicked() {
                    let _ = app.datasets.snapshot(d, "Before restoring a version");
                    let mut back = v.dataset.clone();
                    back.id = d.id.clone();
                    back.ensure_ids();
                    *d = back;
                    app.notify(format!("Restored “{}”", v.label));
                }
            });
        }
        let _ = p;
    });
    ui.add_space(8.0);
}

fn ago(at: u64) -> String {
    let s = crate::store::now_ms().saturating_sub(at) / 1000;
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        _ => format!("{} days ago", s / 86400),
    }
}
