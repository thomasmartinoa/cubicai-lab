//! Steering Lab: describe a concept by its two opposite ends, let the model show where it keeps
//! that difference (layer by layer), and turn it into a dial: replies side by side at several
//! strengths, then keep one for chatting.

use crate::app::App;
use crate::engine::server::State;
use crate::engine::steer::{self, Built, Concept, Vector};
use crate::engine::{xray, Task};
use crate::store::SteerSetting;
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, Kind};
use egui::{pos2, vec2, Color32, Rect, RichText, Sense, Stroke};
use serde_json::json;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

const TOWARDS: Color32 = Color32::from_rgb(0xE8, 0x9A, 0x3C);
const AWAY: Color32 = Color32::from_rgb(0x5B, 0x8D, 0xEF);

pub enum ItemState {
    Waiting,
    Starting,
    Asking(Receiver<Result<String, String>>),
    Done(String),
    Failed(String),
}

pub struct Item {
    pub strength: f32,
    /// The baked vector the server must run with (none for strength 0).
    pub file: Option<PathBuf>,
    pub state: ItemState,
}

impl Item {
    fn finished(&self) -> bool {
        matches!(self.state, ItemState::Done(_) | ItemState::Failed(_))
    }
}

pub struct Compare {
    pub vector: String,
    pub model_key: String,
    pub prompt: String,
    pub items: Vec<Item>,
    /// The server is back to the model's own settings.
    pub restored: bool,
}

impl Compare {
    pub fn running(&self) -> bool {
        !self.restored
    }
}

pub struct SteerUi {
    pub vectors: Vec<Vector>,
    pub model_key: Option<String>,
    pub concept: Concept,
    pub preset_ix: Option<usize>,
    pub selected: Option<String>,
    /// Layer shown in the scatter.
    pub layer: Option<usize>,
    pub build: Option<(Arc<Task>, Arc<Mutex<Option<Built>>>)>,
    pub build_when_ready: bool,
    pub prompt: String,
    pub strengths: Vec<f32>,
    pub compare: Option<Compare>,
    pub chat_strength: f32,
    pub error: Option<String>,
}

impl Default for SteerUi {
    fn default() -> Self {
        SteerUi {
            vectors: vec![],
            model_key: None,
            concept: steer::presets().remove(0),
            preset_ix: Some(0),
            selected: None,
            layer: None,
            build: None,
            build_when_ready: false,
            prompt: "Tell me about your day in two sentences.".into(),
            strengths: vec![-1.0, 0.0, 1.0],
            compare: None,
            chat_strength: 1.0,
            error: None,
        }
    }
}

impl SteerUi {
    pub fn building(&self) -> bool {
        self.build.as_ref().is_some_and(|(t, _)| !t.is_done()) || self.build_when_ready
    }

    pub fn busy(&self) -> bool {
        self.building() || self.compare.as_ref().is_some_and(Compare::running)
    }

    /// Stops a build and whatever comparison step is in flight (the server is restored by `pump`).
    pub fn stop(&mut self) {
        if let Some((t, _)) = &self.build {
            t.stop();
        }
        self.build_when_ready = false;
        if let Some(c) = &mut self.compare {
            for it in c.items.iter_mut().filter(|i| !i.finished()) {
                it.state = ItemState::Failed("Stopped".into());
            }
        }
    }

    pub fn vector(&self, id: &str) -> Option<&Vector> {
        self.vectors.iter().find(|v| v.id == id)
    }
}

/// The model the lab works with: the picked one, else the running one, else the chat's.
fn model_key(app: &App) -> Option<String> {
    app.steer.model_key.clone().or_else(|| app.server.model_key.clone()).or_else(|| app.settings.models.get("pc").cloned())
}

fn server_ready_with(app: &App, key: &str) -> bool {
    app.server.is_ready_with(key)
}

fn start_build(app: &mut App, key: &str, ctx: &egui::Context) {
    let Some(m) = app.library.get(key).cloned() else {
        app.steer.error = Some("That model is no longer in the library.".into());
        return;
    };
    let task = Task::new("Starting");
    let out: Arc<Mutex<Option<Built>>> = Default::default();
    let slot = out.clone();
    let root = xray::root_of(&app.server.base_url());
    let concept = app.steer.concept.clone();
    let (k, name) = (key.to_string(), crate::app::short_model_name(&m.name));
    app.start_task(&task, ctx, move |t| {
        let b = steer::build(&root, &k, &name, &m.parts, &concept, t)?;
        *slot.lock().unwrap() = Some(b);
        Ok(())
    });
    app.steer.build = Some((task, out));
}

/// Advances builds and comparisons; called every frame from the app loop, so both carry on while
/// another tab is open.
pub fn pump(app: &mut App, ctx: &egui::Context) {
    // Building: the server tokenises, so it must run the model first.
    if app.steer.build_when_ready
        && let Some(key) = model_key(app)
    {
        if server_ready_with(app, &key) {
            app.steer.build_when_ready = false;
            start_build(app, &key, ctx);
        } else if let State::Failed(e) = app.server.state() {
            app.steer.build_when_ready = false;
            app.steer.error = Some(e);
        }
    }
    if let Some((t, out)) = &app.steer.build
        && t.is_done()
    {
        let (err, built) = (t.error(), out.lock().unwrap().take());
        app.steer.build = None;
        if let Some(b) = built {
            match app.steer_store.save(&b) {
                Ok(()) => {
                    app.steer.selected = Some(b.vector.id.clone());
                    app.steer.layer = None;
                    app.steer.vectors.insert(0, b.vector);
                }
                Err(e) => app.steer.error = Some(format!("Couldn't save the vector: {e}")),
            }
        }
        if let Some(e) = err.filter(|e| e != "Stopped") {
            app.steer.error = Some(e);
        }
    }
    pump_compare(app, ctx);
}

fn pump_compare(app: &mut App, ctx: &egui::Context) {
    let Some(c) = app.steer.compare.as_mut() else { return };
    if c.restored {
        return;
    }
    let key = c.model_key.clone();
    ctx.request_repaint_after(std::time::Duration::from_millis(250));
    if let Some(i) = c.items.iter().position(|i| !i.finished()) {
        let file = c.items[i].file.clone();
        let on_server = app.server.is_ready_with(&key) && app.server.steer == file;
        match &c.items[i].state {
            ItemState::Waiting if on_server => c.items[i].state = ItemState::Starting,
            ItemState::Waiting => {
                c.items[i].state = ItemState::Starting;
                if let Err(e) = app.start_model_with(&key, file.as_deref()) {
                    let c = app.steer.compare.as_mut().unwrap();
                    for it in c.items.iter_mut().filter(|i| !i.finished()) {
                        it.state = ItemState::Failed(e.clone());
                    }
                }
            }
            ItemState::Starting => {
                if on_server {
                    let (tx, rx) = channel();
                    let root = xray::root_of(&app.server.base_url());
                    let messages = vec![json!({"role": "user", "content": c.prompt.trim()})];
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        let r = xray::apply_template(&root, &messages).and_then(|p| steer::complete(&root, &p, 160));
                        let _ = tx.send(r);
                        ctx.request_repaint();
                    });
                    c.items[i].state = ItemState::Asking(rx);
                } else if let State::Failed(e) = app.server.state() {
                    // e.g. llama.cpp refused the vector; the next strength gets a fresh start.
                    c.items[i].state = ItemState::Failed(e);
                } else if app.server.state() == State::Stopped {
                    c.items[i].state = ItemState::Failed("The server stopped.".into());
                }
            }
            ItemState::Asking(rx) => {
                if let Ok(r) = rx.try_recv() {
                    c.items[i].state = match r {
                        Ok(s) => ItemState::Done(s),
                        Err(e) => ItemState::Failed(e),
                    };
                }
            }
            ItemState::Done(_) | ItemState::Failed(_) => {}
        }
        return;
    }
    // All answered: back to the model's own settings (its chat steering, if any).
    c.restored = true;
    let preset = app.settings.preset(&key);
    let wanted = app.preset_steer_path(&key, &preset);
    if app.server.model_key.as_deref() == Some(key.as_str())
        && app.server.steer != wanted
        && app.server.state() != State::Stopped
        && let Err(e) = app.start_model(&key)
    {
        app.notify(e);
    }
}

fn start_compare(app: &mut App, v: &Vector) {
    let mut items = vec![];
    let mut strengths = app.steer.strengths.clone();
    strengths.sort_by(f32::total_cmp);
    strengths.dedup();
    for s in strengths {
        let file = if s == 0.0 {
            None
        } else {
            match app.steer_store.bake(v, s) {
                Ok(p) => Some(p),
                Err(e) => {
                    app.steer.error = Some(e);
                    return;
                }
            }
        };
        items.push(Item { strength: s, file, state: ItemState::Waiting });
    }
    // Start with whatever the server already runs, to save a restart.
    if let Some(i) = items.iter().position(|it| app.server.is_ready_with(&v.model_key) && app.server.steer == it.file) {
        items.swap(0, i);
    }
    app.steer.error = None;
    app.steer.compare = Some(Compare { vector: v.id.clone(), model_key: v.model_key.clone(), prompt: app.steer.prompt.clone(), items, restored: false });
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let ctx = ui.ctx().clone();
    if app.steer.building() {
        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
    egui::Panel::left("steer-left")
        .resizable(false)
        .exact_size(320.0)
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(16))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| controls(app, ui, &p));
        });
    egui::CentralPanel::no_frame().show(ui, |ui| {
        egui::ScrollArea::vertical().auto_shrink(false).id_salt("steer-main").show(ui, |ui| {
            egui::Frame::new().inner_margin(16).show(ui, |ui| main(app, ui, &p));
        });
    });
}

fn label(ui: &mut egui::Ui, p: &Palette, text: &str) {
    ui.label(RichText::new(text).font(theme::medium(13.0)).color(p.on_variant));
}

fn controls(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Steering").font(theme::heading(20.0)));
    widgets::hint(ui, "Describe a trait by its two ends. The lab finds where the model keeps that difference and turns it into a dial you can apply while it writes.");
    ui.add_space(10.0);

    label(ui, p, "Model");
    let key = model_key(app);
    let name = key.as_ref().and_then(|k| app.library.get(k)).map(|m| crate::app::short_model_name(&m.name)).unwrap_or_else(|| "Pick a model".into());
    egui::ComboBox::from_id_salt("steer-model").selected_text(name).width(ui.available_width() - 4.0).height(480.0).show_ui(ui, |ui| {
        for m in &app.library.models {
            if ui.selectable_label(key.as_deref() == Some(m.key.as_str()), crate::app::short_model_name(&m.name)).clicked() {
                app.steer.model_key = Some(m.key.clone());
                if app.steer.selected.as_ref().and_then(|s| app.steer.vector(s)).is_some_and(|v| v.model_key != m.key) {
                    app.steer.selected = None;
                }
            }
        }
    });
    let arch = key.as_ref().and_then(|k| app.library.info(k)).map(|i| i.gguf.arch().to_string());
    let supported = arch.as_deref().is_none_or(|a| cubic_gguf::forward::SUPPORTED.contains(&a));
    if !supported {
        ui.label(RichText::new(format!("Building vectors needs a Llama, Mistral or Qwen 2/3 model; this one is {}.", arch.unwrap_or_default())).small().color(p.error));
    }
    ui.add_space(12.0);

    label(ui, p, "Concept");
    ui.horizontal_wrapped(|ui| {
        for (i, c) in steer::presets().into_iter().enumerate() {
            if widgets::chip(ui, app.steer.preset_ix == Some(i), &c.name).clicked() {
                app.steer.concept = c;
                app.steer.preset_ix = Some(i);
            }
        }
    });
    ui.add_space(6.0);
    let before = app.steer.concept.clone();
    egui::Grid::new("steer-concept").num_columns(2).spacing(vec2(8.0, 6.0)).show(ui, |ui| {
        ui.label("Name");
        ui.add(egui::TextEdit::singleline(&mut app.steer.concept.name).desired_width(f32::INFINITY));
        ui.end_row();
        ui.label("Towards");
        ui.add(egui::TextEdit::multiline(&mut app.steer.concept.positive).desired_rows(2).desired_width(f32::INFINITY));
        ui.end_row();
        ui.label("Away from");
        ui.add(egui::TextEdit::multiline(&mut app.steer.concept.negative).desired_rows(2).desired_width(f32::INFINITY));
        ui.end_row();
    });
    if app.steer.concept != before {
        app.steer.preset_ix = None;
    }
    widgets::hint(ui, format!("Each end goes into \"Act as if you are …\" for {} everyday questions.", steer::contrast_pairs(&app.steer.concept).len()));
    ui.add_space(10.0);

    if let Some((t, _)) = &app.steer.build {
        ui.label(RichText::new(t.status()).small());
        ui.add(egui::ProgressBar::new(t.fraction()).desired_height(6.0));
        if widgets::button(ui, Kind::Tonal, "Stop").clicked() {
            app.steer.stop();
        }
    } else if app.steer.build_when_ready {
        widgets::hint(ui, "Starting the model so it can read the prompts…");
        if widgets::button(ui, Kind::Tonal, "Stop").clicked() {
            app.steer.stop();
        }
    } else {
        let can = key.is_some() && supported && !app.steer.compare.as_ref().is_some_and(Compare::running);
        if widgets::button_ext(ui, Kind::Primary, "Build vector", can).clicked()
            && let Some(k) = key.clone()
        {
            app.steer.error = None;
            if server_ready_with(app, &k) {
                start_build(app, &k, ui.ctx());
            } else {
                match app.start_model(&k) {
                    Ok(()) => app.steer.build_when_ready = true,
                    Err(e) => app.steer.error = Some(e),
                }
            }
        }
        widgets::hint(ui, "Runs the model layer by layer on the CPU: about a minute for an 8B model.");
    }
    if let Some(e) = &app.steer.error {
        ui.add_space(6.0);
        ui.label(RichText::new(e).color(p.error));
    }
    ui.add_space(14.0);

    label(ui, p, "Vectors for this model");
    let mine: Vec<Vector> = app.steer.vectors.iter().filter(|v| Some(&v.model_key) == key.as_ref()).cloned().collect();
    if mine.is_empty() {
        widgets::hint(ui, "None yet.");
    }
    let in_use = key.as_ref().and_then(|k| app.settings.preset(k).steer).map(|s| s.vector);
    for v in mine {
        let selected = app.steer.selected.as_deref() == Some(v.id.as_str());
        let peak = v.separation.iter().copied().fold(0.0f32, f32::max);
        let text = format!("{}{}", v.name, if in_use.as_deref() == Some(v.id.as_str()) { "  · in chat" } else { "" });
        let r = ui.add(egui::Button::selectable(selected, RichText::new(text)).min_size(vec2(ui.available_width(), 26.0)));
        if r.clicked() {
            app.steer.selected = Some(v.id.clone());
            app.steer.layer = None;
        }
        r.on_hover_text(format!("Towards {}\nAway from {}\nPeak separation {peak:.1}", v.positive, v.negative));
    }
}

fn main(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(v) = app.steer.selected.as_ref().and_then(|s| app.steer.vector(s)).cloned() else {
        ui.add_space(80.0);
        ui.vertical_centered(|ui| {
            widgets::cube_mark(ui, 54.0, app.steer.busy());
            ui.add_space(8.0);
            ui.label(RichText::new("Give the model a dial").font(theme::heading(22.0)));
            ui.add_space(4.0);
            widgets::hint(ui, "Build a vector from a concept, see which layers carry it, then compare replies pushed towards either end.");
        });
        return;
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(&v.name).font(theme::heading(20.0)));
        widgets::hint(ui, format!("{} · {} prompts, {} tokens, built in {:.0} s", v.model_name, v.proj_pos.first().map(|x| x.len() * 2).unwrap_or(0), v.tokens, v.build_secs));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::button(ui, Kind::Danger, "Delete").clicked() {
                app.steer_store.delete(&v.id);
                app.steer.vectors.retain(|x| x.id != v.id);
                app.steer.selected = None;
                for preset in app.settings.presets.values_mut() {
                    if preset.steer.as_ref().is_some_and(|s| s.vector == v.id) {
                        preset.steer = None;
                    }
                }
                app.save_settings();
            }
        });
    });
    ui.horizontal(|ui| {
        swatch(ui, TOWARDS);
        widgets::hint(ui, format!("towards {}", v.positive));
        ui.add_space(10.0);
        swatch(ui, AWAY);
        widgets::hint(ui, format!("away from it: {}", v.negative));
    });
    ui.add_space(10.0);

    let mut edited = v.clone();
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Where the model keeps it");
        widgets::hint(ui, "Bars: how cleanly each layer tells the two ends apart. Line: how big the difference is. The highlighted window is what gets steered.");
        ui.add_space(6.0);
        let layer = app.steer.layer.unwrap_or((v.layer_start + v.layer_end) / 2);
        let mut picked = Some(layer);
        layer_chart(ui, p, &v, &mut picked);
        app.steer.layer = picked;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Steer layers");
            let n = v.separation.len().max(2);
            ui.add(egui::DragValue::new(&mut edited.layer_start).range(1..=edited.layer_end));
            ui.label("to");
            ui.add(egui::DragValue::new(&mut edited.layer_end).range(edited.layer_start..=n - 1));
            let (a, b) = cubic_gguf::steer::best_range(&v.separation);
            if (a, b) != (v.layer_start, v.layer_end) && ui.small_button("Suggested").clicked() {
                (edited.layer_start, edited.layer_end) = (a, b);
            }
        });
        ui.add_space(8.0);
        if let Some(l) = app.steer.layer.filter(|&l| l < v.proj_pos.len()) {
            ui.label(RichText::new(format!("Layer {l}: every prompt along the direction")).strong());
            scatter(ui, p, &v.proj_pos[l], &v.proj_neg[l]);
            widgets::hint(ui, format!("Separation {:.1}: {}", v.separation[l], match v.separation[l] {
                d if d >= 4.0 => "the two ends don't overlap at all",
                d if d >= 2.0 => "clearly apart",
                d if d >= 1.0 => "apart, with some overlap",
                _ => "mostly mixed together",
            }));
        }
    });
    if edited != v {
        app.steer_store.update(&edited);
        if let Some(x) = app.steer.vectors.iter_mut().find(|x| x.id == v.id) {
            *x = edited.clone();
        }
    }
    let v = edited;
    ui.add_space(12.0);

    compare_card(app, ui, p, &v);
    ui.add_space(12.0);
    chat_card(app, ui, &v);
}

fn swatch(ui: &mut egui::Ui, c: Color32) {
    let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
    ui.painter().add(widgets::cut_shape(r, 2.0, c, Stroke::NONE));
}

fn layer_chart(ui: &mut egui::Ui, p: &Palette, v: &Vector, selected: &mut Option<usize>) {
    let n = v.separation.len();
    if n == 0 {
        return;
    }
    let h = 150.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h + 18.0), Sense::click());
    let plot = Rect::from_min_max(rect.min, pos2(rect.right(), rect.top() + h));
    let painter = ui.painter_at(rect);
    painter.rect_filled(plot, 0.0, p.low);
    let max_d = v.separation.iter().copied().fold(1.0f32, f32::max);
    let max_n = v.norms.iter().copied().fold(f32::EPSILON, f32::max);
    let bw = plot.width() / n as f32;
    let hover = resp.hover_pos().filter(|h| plot.contains(*h)).map(|h| (((h.x - plot.left()) / bw) as usize).min(n - 1));
    for l in 0..n {
        let x0 = plot.left() + l as f32 * bw;
        let in_window = l >= v.layer_start && l <= v.layer_end;
        if in_window {
            painter.rect_filled(Rect::from_min_max(pos2(x0, plot.top()), pos2(x0 + bw, plot.bottom())), 0.0, p.primary.gamma_multiply(0.10));
        }
        let bh = (v.separation[l].max(0.0) / max_d) * (h - 10.0);
        let bar = Rect::from_min_max(pos2(x0 + bw * 0.15, plot.bottom() - bh), pos2(x0 + bw * 0.85, plot.bottom()));
        let fill = if in_window { p.primary } else { p.outline_variant };
        painter.rect_filled(bar, 0.0, fill);
        if *selected == Some(l) || hover == Some(l) {
            painter.rect_stroke(Rect::from_min_max(pos2(x0, plot.top()), pos2(x0 + bw, plot.bottom())), 0.0, Stroke::new(1.0, p.on_bg), egui::StrokeKind::Inside);
        }
        if n <= 24 || l % 4 == 0 || l == n - 1 {
            widgets::text_at(&painter, pos2(x0 + bw / 2.0, plot.bottom() + 3.0), egui::Align2::CENTER_TOP, &l.to_string(), egui::FontId::proportional(10.0), p.on_variant);
        }
    }
    let line: Vec<egui::Pos2> = v.norms.iter().enumerate().map(|(l, nm)| pos2(plot.left() + (l as f32 + 0.5) * bw, plot.bottom() - nm / max_n * (h - 10.0))).collect();
    painter.add(egui::Shape::line(line, Stroke::new(2.0, p.on_bg.gamma_multiply(0.85))));
    if let Some(l) = hover {
        let t = format!("layer {l} · separation {:.1} · size {:.2}", v.separation[l], v.norms.get(l).copied().unwrap_or(0.0));
        widgets::text_at(&painter, pos2(plot.right() - 6.0, plot.top() + 4.0), egui::Align2::RIGHT_TOP, &t, egui::FontId::proportional(12.0), p.on_bg);
        if resp.clicked() {
            *selected = Some(l);
        }
    }
}

fn scatter(ui: &mut egui::Ui, p: &Palette, pos: &[f32], neg: &[f32]) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 64.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, p.low);
    let all = pos.iter().chain(neg);
    let (lo, hi) = all.clone().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    if !lo.is_finite() {
        return;
    }
    let span = (hi - lo).max(1e-6);
    let x = |v: f32| rect.left() + 12.0 + (v - lo) / span * (rect.width() - 24.0);
    for (vals, row, c) in [(neg, 0.68, AWAY), (pos, 0.32, TOWARDS)] {
        let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
        painter.vline(x(mean), rect.y_range(), Stroke::new(1.0, c.gamma_multiply(0.6)));
        for (i, &v) in vals.iter().enumerate() {
            let jitter = ((i * 37) % 11) as f32 - 5.0;
            painter.circle_filled(pos2(x(v), rect.top() + rect.height() * row + jitter * 1.4), 3.5, c);
        }
    }
}

fn compare_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, v: &Vector) {
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Compare strengths");
        widgets::hint(ui, "The same prompt, answered greedily at each strength (minus pushes away). Every strength restarts the model, about 5 s each.");
        ui.add_space(6.0);
        let running = app.steer.compare.as_ref().is_some_and(Compare::running);
        ui.horizontal(|ui| {
            ui.label("Prompt");
            ui.add_enabled(!running, egui::TextEdit::singleline(&mut app.steer.prompt).desired_width(f32::INFINITY));
        });
        ui.horizontal(|ui| {
            ui.label("Strengths");
            let mut remove = None;
            let n = app.steer.strengths.len();
            for (i, s) in app.steer.strengths.iter_mut().enumerate() {
                ui.add_enabled(!running, egui::DragValue::new(s).range(-3.0..=3.0).speed(0.05).fixed_decimals(1));
                if n > 1 && !running && ui.small_button("x").on_hover_text("Remove").clicked() {
                    remove = Some(i);
                }
            }
            if let Some(i) = remove {
                app.steer.strengths.remove(i);
            }
            if n < 5 && !running && ui.small_button("Add").clicked() {
                let next = app.steer.strengths.iter().copied().fold(0.0f32, f32::max) + 0.5;
                app.steer.strengths.push(next.min(3.0));
            }
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if running {
                if widgets::button(ui, Kind::Tonal, "Stop").clicked() {
                    app.steer.stop();
                }
            } else {
                let can = app.bin_dir().is_some() && !app.steer.building() && !app.steer.prompt.trim().is_empty();
                if widgets::button_ext(ui, Kind::Primary, "Run comparison", can).on_disabled_hover_text("Needs llama.cpp installed and a prompt").clicked() {
                    start_compare(app, v);
                }
            }
        });
        let Some(c) = app.steer.compare.as_ref().filter(|c| c.vector == v.id) else { return };
        ui.add_space(8.0);
        let n = c.items.len();
        ui.columns(n, |cols| {
            for (it, col) in c.items.iter().zip(cols.iter_mut()) {
                let tint = if it.strength > 0.0 { TOWARDS } else if it.strength < 0.0 { AWAY } else { p.outline };
                widgets::surface(col, p.low, Stroke::new(1.0, tint.gamma_multiply(0.7)), 8.0, 10, |ui| {
                    ui.set_width(ui.available_width());
                    // Columns are justified; replies read better ragged.
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        let head = match it.strength {
                            0.0 => "0 · unsteered".to_string(),
                            s => format!("{s:+.1}"),
                        };
                        ui.label(RichText::new(head).font(theme::medium(14.0)).color(tint));
                        match &it.state {
                            ItemState::Waiting => widgets::hint(ui, "waiting"),
                            ItemState::Starting => widgets::hint(ui, "starting the model…"),
                            ItemState::Asking(_) => widgets::hint(ui, "writing…"),
                            ItemState::Done(s) => ui.label(s),
                            ItemState::Failed(e) => ui.label(RichText::new(e).small().color(p.error)),
                        };
                    });
                });
            }
        });
    });
}

fn chat_card(app: &mut App, ui: &mut egui::Ui, v: &Vector) {
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Use while chatting");
        let mut preset = app.settings.preset(&v.model_key);
        let active = preset.steer.as_ref().filter(|s| s.vector == v.id).map(|s| s.strength);
        match active {
            Some(s) => widgets::hint(ui, format!("On at {s:+.1}: every reply from this model is steered (shown in the Engine Room too).")),
            None => widgets::hint(ui, "Keeps the vector on whenever this model runs, in chat and the X-ray."),
        };
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Strength");
            ui.spacing_mut().slider_width = 180.0;
            ui.add(egui::Slider::new(&mut app.steer.chat_strength, -2.0..=2.0).step_by(0.1));
        });
        ui.horizontal(|ui| {
            let label = if active.is_some() { "Update" } else { "Use in chat" };
            let busy = app.steer.compare.as_ref().is_some_and(Compare::running);
            let mut changed = false;
            if widgets::button_ext(ui, Kind::Primary, label, !busy).clicked() {
                preset.steer = Some(SteerSetting { vector: v.id.clone(), strength: app.steer.chat_strength });
                changed = true;
            }
            if active.is_some() && widgets::button(ui, Kind::Ghost, "Stop steering").clicked() {
                preset.steer = None;
                changed = true;
            }
            if changed {
                app.settings.presets.insert(v.model_key.clone(), preset);
                app.save_settings();
                if app.server.model_key.as_deref() == Some(v.model_key.as_str()) && app.server.state() != State::Stopped
                    && let Err(e) = app.start_model(&v.model_key)
                {
                    app.steer.error = Some(e);
                }
            }
        });
    });
}
