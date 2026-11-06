//! X-ray: watch a reply being written token by token with the model's confidence on each one,
//! see the alternatives it weighed, fork the reply at any token, and see what the sampler
//! settings do to each choice.

use crate::app::App;
use crate::engine::server::State;
use crate::engine::xray::{self, BranchState, Sampling, Step, Tree, XEv};
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, Kind};
use egui::{vec2, Color32, Rect, RichText, Sense, Stroke};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
pub enum Colour {
    Confidence,
    Uncertainty,
}

pub struct XrayUi {
    pub system: String,
    pub prompt: String,
    /// History loaded from a chat (replaces system + prompt when set).
    pub from_chat: Option<(String, Vec<Value>)>,
    pub sampling: Sampling,
    pub tree: Option<Tree>,
    pub branch: usize,
    /// Position in the current branch's text.
    pub selected: Option<usize>,
    pub colour: Colour,
    pub model_key: Option<String>,
    /// Run as soon as the server is ready.
    pub run_when_ready: bool,
    pub error: Option<String>,
    tx: Sender<(usize, XEv)>,
    rx: Receiver<(usize, XEv)>,
    prompt_rx: Option<Receiver<Result<String, String>>>,
    running: Vec<(usize, Arc<AtomicBool>)>,
}

impl Default for XrayUi {
    fn default() -> Self {
        let (tx, rx) = channel();
        XrayUi {
            system: String::new(),
            prompt: "Write one sentence about a lighthouse keeper.".into(),
            from_chat: None,
            sampling: Sampling::default(),
            tree: None,
            branch: 0,
            selected: None,
            colour: Colour::Confidence,
            model_key: None,
            run_when_ready: false,
            error: None,
            tx,
            rx,
            prompt_rx: None,
            running: vec![],
        }
    }
}

impl XrayUi {
    pub fn busy(&self) -> bool {
        !self.running.is_empty() || self.prompt_rx.is_some()
    }

    pub fn stop(&mut self) {
        for (b, c) in self.running.drain(..) {
            c.store(true, Ordering::Relaxed);
            if let Some(br) = self.tree.as_mut().and_then(|t| t.branches.get_mut(b))
                && br.state == BranchState::Running {
                    br.state = BranchState::Stopped;
                }
        }
        self.prompt_rx = None;
    }

    /// Loads a chat's history (everything before the reply) to replay under the X-ray.
    pub fn load_chat(&mut self, title: String, messages: Vec<Value>) {
        self.stop();
        self.from_chat = Some((title, messages));
        self.tree = None;
        self.selected = None;
        self.error = None;
    }

    fn messages(&self) -> Vec<Value> {
        if let Some((_, m)) = &self.from_chat {
            return m.clone();
        }
        let mut m = vec![];
        if !self.system.trim().is_empty() {
            m.push(json!({"role": "system", "content": self.system.trim()}));
        }
        m.push(json!({"role": "user", "content": self.prompt.trim()}));
        m
    }

    /// Starts a fresh tree: template the messages on the server, then generate the main branch.
    fn run(&mut self, base_url: String, ctx: &egui::Context) {
        self.stop();
        self.tree = None;
        self.selected = None;
        self.branch = 0;
        self.error = None;
        let (tx, rx) = channel();
        let messages = self.messages();
        let root = xray::root_of(&base_url);
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(xray::apply_template(&root, &messages));
            ctx2.request_repaint();
        });
        self.prompt_rx = Some(rx);
    }

    fn generate(&mut self, base_url: &str, b: usize, ctx: &egui::Context) {
        let Some(t) = &self.tree else { return };
        let cancel = Arc::new(AtomicBool::new(false));
        let mut s = self.sampling.clone();
        // A fork carries on for what's left of the length.
        s.max_tokens = s.max_tokens.saturating_sub(t.path(b).len() as u32).max(1);
        xray::generate(xray::root_of(base_url), t.continuation(b), s, b, self.tx.clone(), cancel.clone(), ctx.clone());
        self.running.push((b, cancel));
    }

    fn pump(&mut self, base_url: &str, ctx: &egui::Context) {
        if let Some(rx) = &self.prompt_rx
            && let Ok(r) = rx.try_recv() {
                self.prompt_rx = None;
                match r {
                    Ok(prompt) => {
                        self.tree = Some(Tree::new(prompt));
                        self.generate(base_url, 0, ctx);
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        while let Ok((b, ev)) = self.rx.try_recv() {
            let Some(t) = self.tree.as_mut() else { continue };
            let Some(br) = t.branches.get_mut(b) else { continue };
            match ev {
                XEv::Steps(s) => br.steps.extend(s),
                XEv::Done { gen_tps } => {
                    br.state = BranchState::Done;
                    br.gen_tps = gen_tps;
                    self.running.retain(|(r, _)| *r != b);
                }
                XEv::Error(e) => {
                    br.state = BranchState::Failed(e);
                    self.running.retain(|(r, _)| *r != b);
                }
            }
        }
    }
}

const HOT: Color32 = Color32::from_rgb(0xE8, 0x4A, 0x5F);
const COOL: Color32 = Color32::from_rgb(0x5B, 0x8D, 0xEF);

fn token_bg(step: &Step, mode: Colour, p: &Palette) -> Color32 {
    match mode {
        Colour::Confidence => {
            // Sure tokens fade into the page; doubtful ones glow.
            let doubt = (1.0 - step.chosen.p).clamp(0.0, 1.0).powf(0.8);
            widgets::lerp_color(p.low, HOT, doubt * 0.85)
        }
        Colour::Uncertainty => {
            let h = (step.entropy() / (xray::N_PROBS as f32 + 1.0).log2()).clamp(0.0, 1.0);
            widgets::lerp_color(p.low, COOL, h.powf(0.7) * 0.9)
        }
    }
}

/// A token's text with its invisible parts made visible.
fn visible(text: &str) -> String {
    let t = text.replace('\n', "\\n").replace('\t', "\\t");
    if t.is_empty() { "(empty)".into() } else if t.trim().is_empty() { t.replace(' ', "\u{b7}") } else { t }
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let ctx = ui.ctx().clone();
    let base = app.server.base_url();
    app.xray.pump(&base, &ctx);
    if app.xray.run_when_ready && app.server.state() == State::Ready {
        app.xray.run_when_ready = false;
        app.xray.run(base.clone(), &ctx);
    }
    if app.xray.run_when_ready && matches!(app.server.state(), State::Failed(_) | State::Stopped) {
        app.xray.run_when_ready = false;
    }

    egui::Panel::left("xray-left")
        .resizable(false)
        .exact_size(310.0)
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(16))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| controls(app, ui, &p));
        });
    let selected_step = app.xray.selected;
    egui::Panel::right("xray-right")
        .resizable(false)
        .exact_size(360.0)
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(16))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                if selected_step.is_some() {
                    step_detail(app, ui, &p);
                } else {
                    branches(app, ui, &p);
                }
            });
        });
    egui::CentralPanel::no_frame().show(ui, |ui| {
        egui::Frame::new().inner_margin(16).show(ui, |ui| reply(app, ui, &p));
    });
}

fn controls(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("X-ray").font(theme::heading(20.0)));
    widgets::hint(ui, "Watch the model choose every token: how sure it was, what else it considered, and what happens if it had picked differently.");
    ui.add_space(10.0);

    // Model: whatever llama-server runs; start one here if none is.
    let state = app.server.state();
    let running = app.server.model_key.clone();
    ui.label(RichText::new("Model").font(theme::medium(13.0)).color(p.on_variant));
    let key = app.xray.model_key.clone().or(running.clone()).or_else(|| app.settings.models.get("pc").cloned());
    let label = key.as_ref().and_then(|k| app.library.get(k)).map(|m| crate::app::short_model_name(&m.name)).unwrap_or_else(|| "Pick a model".into());
    egui::ComboBox::from_id_salt("xray-model").selected_text(label).width(ui.available_width() - 4.0).height(480.0).show_ui(ui, |ui| {
        for m in &app.library.models {
            if ui.selectable_label(key.as_deref() == Some(m.key.as_str()), crate::app::short_model_name(&m.name)).clicked() {
                app.xray.model_key = Some(m.key.clone());
            }
        }
    });
    let wanted = app.xray.model_key.clone().or(running.clone());
    let ready = state == State::Ready && wanted.is_some() && running == wanted;
    match (&state, ready) {
        (_, true) => widgets::hint(ui, format!("Running on this PC at {}", app.server.base_url())),
        (State::Loading, _) => widgets::hint(ui, "Loading the model…"),
        (State::Failed(e), _) => ui.label(RichText::new(e).small().color(p.error)),
        _ => widgets::hint(ui, "Starts on your GPU when you run."),
    };
    ui.add_space(10.0);

    if let Some((title, msgs)) = app.xray.from_chat.clone() {
        ui.label(RichText::new("From chat").font(theme::medium(13.0)).color(p.on_variant));
        ui.label(RichText::new(&title).strong());
        widgets::hint(ui, format!("{} messages before the reply are replayed.", msgs.len()));
        if ui.small_button("Use my own prompt instead").clicked() {
            app.xray.from_chat = None;
        }
    } else {
        ui.label(RichText::new("System (optional)").font(theme::medium(13.0)).color(p.on_variant));
        ui.add(egui::TextEdit::multiline(&mut app.xray.system).desired_rows(2).desired_width(f32::INFINITY).hint_text("You are…"));
        ui.add_space(6.0);
        ui.label(RichText::new("Prompt").font(theme::medium(13.0)).color(p.on_variant));
        ui.add(egui::TextEdit::multiline(&mut app.xray.prompt).desired_rows(5).desired_width(f32::INFINITY));
    }
    ui.add_space(10.0);

    ui.label(RichText::new("Sampling").font(theme::medium(13.0)).color(p.on_variant));
    ui.spacing_mut().slider_width = 120.0;
    let s = &mut app.xray.sampling;
    egui::Grid::new("xray-sampling").num_columns(2).spacing(vec2(10.0, 6.0)).show(ui, |ui| {
        ui.label("Temperature");
        ui.add(egui::Slider::new(&mut s.temperature, 0.0..=2.0).step_by(0.05));
        ui.end_row();
        ui.label("Top-k");
        ui.add(egui::Slider::new(&mut s.top_k, 0..=100));
        ui.end_row();
        ui.label("Top-p");
        ui.add(egui::Slider::new(&mut s.top_p, 0.0..=1.0).step_by(0.01));
        ui.end_row();
        ui.label("Min-p");
        ui.add(egui::Slider::new(&mut s.min_p, 0.0..=0.5).step_by(0.01));
        ui.end_row();
        ui.label("Seed");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut s.seed).range(-1..=i64::from(u32::MAX)));
            if s.seed == -1 {
                widgets::hint(ui, "random");
            }
        });
        ui.end_row();
        ui.label("Max tokens");
        ui.add(egui::Slider::new(&mut s.max_tokens, 16..=2048).logarithmic(true));
        ui.end_row();
    });
    widgets::hint(ui, "Temperature 0 always takes the favourite. The lens on the right shows what each setting does to a step.");
    ui.add_space(12.0);

    ui.horizontal(|ui| {
        if app.xray.busy() || app.xray.run_when_ready {
            if widgets::button(ui, Kind::Tonal, "Stop").clicked() {
                app.xray.stop();
                app.xray.run_when_ready = false;
            }
        } else {
            let can = wanted.is_some() || state == State::Ready;
            if widgets::button_ext(ui, Kind::Primary, "Run", can).clicked() {
                if ready || (state == State::Ready && app.xray.model_key.is_none()) {
                    app.xray.run(app.server.base_url(), ui.ctx());
                } else if let Some(k) = wanted.clone() {
                    match app.start_model(&k) {
                        Ok(()) => app.xray.run_when_ready = true,
                        Err(e) => app.xray.error = Some(e),
                    }
                }
            }
        }
    });
    if let Some(e) = &app.xray.error {
        ui.add_space(6.0);
        ui.label(RichText::new(e).color(p.error));
    }
}

fn reply(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(tree) = app.xray.tree.as_ref() else {
        ui.add_space(80.0);
        ui.vertical_centered(|ui| {
            widgets::cube_mark(ui, 54.0, app.xray.busy() || app.xray.run_when_ready);
            ui.add_space(8.0);
            ui.label(RichText::new("See how a reply is chosen").font(theme::heading(22.0)));
            ui.add_space(4.0);
            widgets::hint(ui, "Run a prompt: every token is coloured by how sure the model was. Click a token to see its rivals and fork the reply there.");
        });
        return;
    };
    let b = app.xray.branch.min(tree.branches.len() - 1);
    let path = tree.path(b);
    let offset = tree.offset(b);
    let br = &tree.branches[b];

    // Header: branch, state, colour mode.
    let mut colour = app.xray.colour;
    ui.horizontal(|ui| {
        let title = if b == 0 { "Main reply".to_string() } else { format!("Fork {b}") };
        ui.label(RichText::new(title).font(theme::heading(18.0)));
        match &br.state {
            BranchState::Running => widgets::hint(ui, format!("writing… {} tokens", path.len())),
            BranchState::Done => widgets::hint(ui, format!("{} tokens{}", path.len(), br.gen_tps.map(|t| format!(" · {t:.0} tok/s")).unwrap_or_default())),
            BranchState::Stopped => widgets::hint(ui, "stopped"),
            BranchState::Failed(_) => ui.label(RichText::new("failed").small().color(p.error)),
        };
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            widgets::segmented(ui, &mut colour, &[(Colour::Confidence, "Confidence"), (Colour::Uncertainty, "Uncertainty")]);
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(tree.text(b));
            }
        });
    });
    app.xray.colour = colour;
    if let BranchState::Failed(e) = &br.state {
        ui.label(RichText::new(e).color(p.error));
    }
    ui.add_space(4.0);
    timeline(ui, p, &path, offset, colour, &mut app.xray.selected);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let (lo, hi, l1, l2) = match colour {
            Colour::Confidence => (p.low, HOT, "sure", "unsure"),
            Colour::Uncertainty => (p.low, COOL, "one clear choice", "many close rivals"),
        };
        widgets::hint(ui, l1);
        let (r, _) = ui.allocate_exact_size(vec2(120.0, 10.0), Sense::hover());
        for i in 0..24 {
            let x0 = r.left() + r.width() * i as f32 / 24.0;
            ui.painter().rect_filled(Rect::from_min_size(egui::pos2(x0, r.top()), vec2(r.width() / 24.0 + 0.5, r.height())), 0.0, widgets::lerp_color(lo, hi, i as f32 / 23.0));
        }
        widgets::hint(ui, l2);
        widgets::hint(ui, "  ·  underlined: forked here  ·  outlined: the alternative this fork took");
    });
    ui.add_space(8.0);

    // Which positions have forks hanging off them (in this branch's own path).
    let mut forked: Vec<usize> = vec![];
    for i in 0..path.len() {
        let (ob, os) = tree.locate(b, i);
        if tree.forks_of(ob).iter().any(|&(child, s)| s == os && child != b) {
            forked.push(i);
        }
    }

    let mut clicked = None;
    egui::ScrollArea::vertical().auto_shrink(false).id_salt("xray-text").show(ui, |ui| {
        ui.spacing_mut().item_spacing = vec2(0.0, 3.0);
        ui.horizontal_wrapped(|ui| {
            let font = egui::FontId::proportional(16.0);
            for (i, s) in path.iter().enumerate() {
                let text = s.chosen.text();
                let pieces: Vec<&str> = text.split('\n').collect();
                for (k, piece) in pieces.iter().enumerate() {
                    if k > 0 {
                        ui.end_row();
                    }
                    if piece.is_empty() && pieces.len() > 1 {
                        continue;
                    }
                    let galley = ui.painter().layout_no_wrap(piece.to_string(), font.clone(), p.on_bg);
                    let size = vec2(galley.size().x.max(4.0), galley.size().y + 4.0);
                    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
                    let selected = app.xray.selected == Some(i);
                    let painter = ui.painter();
                    painter.rect_filled(rect, 0.0, token_bg(s, colour, p));
                    if resp.hovered() || selected {
                        painter.rect_stroke(rect, 0.0, Stroke::new(1.5, p.on_bg), egui::StrokeKind::Inside);
                    }
                    if i == offset && b != 0 {
                        painter.rect_stroke(rect, 0.0, Stroke::new(1.5, p.primary), egui::StrokeKind::Inside);
                    }
                    if forked.contains(&i) {
                        painter.hline(rect.x_range(), rect.bottom() - 1.0, Stroke::new(2.0, p.primary));
                    }
                    painter.galley(rect.min + vec2((rect.width() - galley.size().x) / 2.0, 2.0), galley, p.on_bg);
                    if resp.clicked() {
                        clicked = Some(i);
                    }
                    resp.on_hover_ui(|ui| candidates_tooltip(ui, p, s));
                }
            }
            if br.state == BranchState::Running {
                let (r, _) = ui.allocate_exact_size(vec2(9.0, 18.0), Sense::hover());
                let a = 0.4 + 0.6 * (ui.input(|i| i.time) as f32 * 5.0).sin().abs();
                ui.painter().rect_filled(r.shrink2(vec2(1.0, 2.0)), 0.0, p.primary.gamma_multiply(a));
                ui.ctx().request_repaint();
            }
        });
    });
    if let Some(i) = clicked {
        app.xray.selected = if app.xray.selected == Some(i) { None } else { Some(i) };
    }
}

fn candidates_tooltip(ui: &mut egui::Ui, p: &Palette, s: &Step) {
    ui.set_max_width(260.0);
    ui.label(RichText::new(format!("{:.1}% sure · {:.2} bits", s.chosen.p * 100.0, s.entropy())).font(theme::medium(13.0)));
    for c in s.top.iter().take(6) {
        ui.horizontal(|ui| {
            let mine = c.id == s.chosen.id;
            let (r, _) = ui.allocate_exact_size(vec2(70.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 0.0, p.high);
            ui.painter().rect_filled(Rect::from_min_size(r.min, vec2(r.width() * c.p, r.height())), 0.0, if mine { p.primary } else { p.secondary });
            ui.label(RichText::new(visible(&c.text())).monospace().color(if mine { p.primary } else { p.on_bg }));
            widgets::hint(ui, format!("{:.1}%", c.p * 100.0));
        });
    }
    widgets::hint(ui, "Click to fork here.");
}

/// A strip of the whole reply: bar height = confidence, colour = the chosen mode. Click to select.
fn timeline(ui: &mut egui::Ui, p: &Palette, path: &[&Step], offset: usize, colour: Colour, selected: &mut Option<usize>) {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 46.0), Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, p.lowest);
    if path.is_empty() {
        return;
    }
    let n = path.len().max(60) as f32;
    let bw = rect.width() / n;
    for (i, s) in path.iter().enumerate() {
        let x = rect.left() + i as f32 * bw;
        let h = (rect.height() - 4.0) * s.chosen.p.clamp(0.02, 1.0);
        let bar = Rect::from_min_max(egui::pos2(x, rect.bottom() - h), egui::pos2(x + (bw - 1.0).max(1.0), rect.bottom()));
        let c = match colour {
            Colour::Confidence => widgets::lerp_color(p.outline, HOT, (1.0 - s.chosen.p).powf(0.8)),
            Colour::Uncertainty => widgets::lerp_color(p.outline, COOL, (s.entropy() / 3.46).clamp(0.0, 1.0)),
        };
        painter.rect_filled(bar, 0.0, c);
    }
    if offset > 0 {
        let x = rect.left() + offset as f32 * bw;
        painter.vline(x, rect.y_range(), Stroke::new(1.5, p.primary));
    }
    if let Some(i) = *selected {
        let x = rect.left() + (i as f32 + 0.5) * bw;
        painter.vline(x, rect.y_range(), Stroke::new(1.0, p.on_bg));
    }
    if let Some(pos) = resp.interact_pointer_pos()
        && resp.clicked() {
            let i = ((pos.x - rect.left()) / bw) as usize;
            if i < path.len() {
                *selected = Some(i);
            }
        }
    resp.on_hover_text("Each bar is a token: taller = more sure. Click one to inspect it.");
}

fn step_detail(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let Some(tree) = app.xray.tree.as_ref() else { return };
    let b = app.xray.branch;
    let path = tree.path(b);
    let Some(i) = app.xray.selected.filter(|&i| i < path.len()) else {
        app.xray.selected = None;
        return;
    };
    let s = path[i].clone();
    let (owner, owner_step) = tree.locate(b, i);
    let lens = xray::lens(&s.top.iter().map(|c| c.p).collect::<Vec<_>>(), &app.xray.sampling);

    if widgets::button(ui, Kind::Ghost, "Back to branches").clicked() {
        app.xray.selected = None;
        return;
    }
    ui.add_space(6.0);
    ui.label(RichText::new(format!("Token {}", i + 1)).font(theme::heading(18.0)));
    ui.label(RichText::new(visible(&s.chosen.text())).monospace().size(18.0).color(p.primary));
    egui::Grid::new("xray-step").num_columns(2).spacing(vec2(12.0, 4.0)).show(ui, |ui| {
        ui.label(RichText::new("Probability").color(p.on_variant));
        ui.label(format!("{:.2}%", s.chosen.p * 100.0));
        ui.end_row();
        ui.label(RichText::new("Rank").color(p.on_variant));
        ui.label(match s.rank() {
            Some(0) => "the favourite".to_string(),
            Some(r) => format!("#{} (the sampler picked an underdog)", r + 1),
            None => format!("outside the top {}", xray::N_PROBS),
        });
        ui.end_row();
        ui.label(RichText::new("Uncertainty").color(p.on_variant));
        ui.label(format!("{:.2} bits", s.entropy()));
        ui.end_row();
        ui.label(RichText::new("Token id").color(p.on_variant));
        ui.label(s.chosen.id.to_string());
        ui.end_row();
    });
    widgets::hint(ui, "1 bit ≈ a coin toss between two tokens; 0 = no doubt at all.");
    ui.add_space(10.0);

    ui.label(RichText::new("What it weighed").font(theme::heading(15.0)));
    widgets::hint(ui, "Left bar: the model's own odds. Right bar: the odds after your sampling settings (the lens). Fork to continue from any of them.");
    ui.add_space(4.0);
    let mut fork = None;
    for (k, c) in s.top.iter().enumerate() {
        let mine = c.id == s.chosen.id;
        ui.horizontal(|ui| {
            let text = RichText::new(visible(&c.text())).monospace().color(if mine { p.primary } else { p.on_bg });
            ui.add_sized([86.0, 18.0], egui::Label::new(text).truncate());
            bar(ui, p, c.p, if mine { p.primary } else { p.secondary }, 64.0);
            widgets::hint(ui, format!("{:>5.1}%", c.p * 100.0));
            let after = lens.get(k).copied().unwrap_or(0.0);
            bar(ui, p, after, if after == 0.0 { p.outline } else { p.good }, 50.0);
            if !mine && ui.add_enabled(!app.xray.busy(), egui::Button::new("Fork").small()).clicked() {
                fork = Some(c.clone());
            }
        });
    }
    let cut = lens.iter().filter(|x| **x == 0.0).count();
    if cut > 0 {
        widgets::hint(ui, format!("{cut} of these can never be picked with your settings."));
    }
    if let Some(alt) = fork {
        let base = app.server.base_url();
        let ctx = ui.ctx().clone();
        if let Some(t) = app.xray.tree.as_mut() {
            let nb = t.fork(owner, owner_step, alt);
            app.xray.branch = nb;
            app.xray.selected = None;
            app.xray.generate(&base, nb, &ctx);
        }
    }
}

fn bar(ui: &mut egui::Ui, p: &Palette, v: f32, c: Color32, w: f32) {
    let (r, _) = ui.allocate_exact_size(vec2(w, 10.0), Sense::hover());
    ui.painter().rect_filled(r, 0.0, p.high);
    ui.painter().rect_filled(Rect::from_min_size(r.min, vec2(r.width() * v.clamp(0.0, 1.0), r.height())), 0.0, c);
}

fn branches(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Branches").font(theme::heading(18.0)));
    let Some(tree) = app.xray.tree.as_ref() else {
        widgets::hint(ui, "Every fork you make shows up here as its own version of the reply.");
        return;
    };
    widgets::hint(ui, "Click a token in the reply and pick another candidate to fork. Forks of forks nest.");
    ui.add_space(6.0);
    let mut open = None;
    // Depth-first, children under their parent.
    let mut stack = vec![(0usize, 0usize)];
    while let Some((b, depth)) = stack.pop() {
        let br = &tree.branches[b];
        let selected = app.xray.branch == b;
        let label = match br.parent {
            None => "Main reply".to_string(),
            Some((pb, ps)) => {
                let was = tree.branches[pb].steps.get(ps).map(|s| visible(&s.chosen.text())).unwrap_or_default();
                let now = br.steps.first().map(|s| visible(&s.chosen.text())).unwrap_or_default();
                format!("\"{was}\" -> \"{now}\"")
            }
        };
        // What this branch wrote itself (its parent's text is shared).
        let own: Vec<u8> = br.steps.iter().flat_map(|s| s.chosen.bytes.iter().copied()).collect();
        let snippet: String = String::from_utf8_lossy(&own).trim().chars().take(90).collect();
        let indent = (depth as f32 * 14.0).min(70.0);
        let width = ui.available_width() - indent - 16.0;
        ui.horizontal(|ui| {
            ui.add_space(indent);
            ui.vertical(|ui| {
            let resp = widgets::surface(ui, if selected { p.secondary_container } else { p.container }, Stroke::new(1.0, p.outline_variant), 6.0, 8, |ui| {
                ui.set_width(width);
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(RichText::new(&label).font(theme::medium(13.5))).truncate());
                    let state = match &br.state {
                        BranchState::Running => "writing".to_string(),
                        BranchState::Done => format!("{} tokens", tree.path(b).len()),
                        BranchState::Stopped => "stopped".into(),
                        BranchState::Failed(_) => "failed".into(),
                    };
                    widgets::hint(ui, state);
                });
                ui.add(egui::Label::new(RichText::new(snippet).small().color(p.on_variant)).wrap());
            });
            if resp.response.interact(Sense::click()).clicked() {
                open = Some(b);
            }
            });
        });
        let mut kids = tree.forks_of(b);
        kids.sort_by_key(|k| std::cmp::Reverse(k.1));
        for (k, _) in kids {
            stack.push((k, depth + 1));
        }
    }
    if let Some(b) = open {
        app.xray.branch = b;
        app.xray.selected = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::mock_http;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    fn tok(id: i64, t: &str, p: f32, alts: &[(i64, &str, f32)]) -> String {
        let e = |id: i64, t: &str, p: f32| json!({"id": id, "token": t, "bytes": t.as_bytes(), "logprob": p.ln()});
        let mut top = vec![e(id, t, p)];
        top.extend(alts.iter().map(|&(i, t, p)| e(i, t, p)));
        format!("data: {}\n\n", json!({"content": t, "tokens": [id], "stop": false, "completion_probabilities": [{"id": id, "token": t, "bytes": t.as_bytes(), "logprob": p.ln(), "top_logprobs": top}]}))
    }

    fn settle(x: &mut XrayUi, base: &str, ctx: &egui::Context, until: impl Fn(&XrayUi) -> bool) {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(10) {
            x.pump(base, ctx);
            if until(x) {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out; error {:?}", x.error);
    }

    #[test]
    fn run_fork_and_continue() {
        let prompts = Arc::new(Mutex::new(vec![]));
        let p2 = prompts.clone();
        let port = mock_http(move |req: &str| {
            if req.starts_with("POST /apply-template") {
                let body = r#"{"prompt":"<u>hi<a>"}"#;
                return format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
            let body: Value = serde_json::from_str(&req[req.find("\r\n\r\n").unwrap() + 4..]).unwrap();
            p2.lock().unwrap().push(body["prompt"].clone());
            let forked = body["prompt"].as_array().unwrap().len() > 1;
            let mut out = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_string();
            if forked {
                out += &tok(9, " friend", 0.8, &[]);
            } else {
                out += &tok(1, "Hello", 0.9, &[(2, "Hi", 0.05)]);
                out += &tok(3, " there", 0.4, &[(4, " world", 0.35)]);
            }
            out += &format!("data: {}\n\n", json!({"stop": true, "content": "", "timings": {"predicted_per_second": 64.0}}));
            out
        });
        let base = format!("http://127.0.0.1:{port}/v1");
        let ctx = egui::Context::default();
        let mut x = XrayUi { prompt: "hi".into(), ..Default::default() };
        x.run(base.clone(), &ctx);
        assert!(x.busy());
        settle(&mut x, &base, &ctx, |x| x.tree.as_ref().is_some_and(|t| t.branches[0].state == BranchState::Done));
        assert!(!x.busy());
        let t = x.tree.as_ref().unwrap();
        assert_eq!(t.text(0), "Hello there");
        assert_eq!(t.branches[0].gen_tps, Some(64.0));

        // Fork " there" → " world" and let it continue.
        let alt = t.branches[0].steps[1].top[1].clone();
        let nb = x.tree.as_mut().unwrap().fork(0, 1, alt);
        x.generate(&base, nb, &ctx);
        settle(&mut x, &base, &ctx, |x| x.tree.as_ref().unwrap().branches[nb].state == BranchState::Done);
        let t = x.tree.as_ref().unwrap();
        assert_eq!(t.text(nb), "Hello world friend");
        assert_eq!(t.text(0), "Hello there", "the main reply is untouched");
        let sent = prompts.lock().unwrap().clone();
        assert_eq!(sent, vec![json!(["<u>hi<a>"]), json!(["<u>hi<a>", 1, 4])]);
    }

    #[test]
    fn chat_history_replaces_the_prompt_and_stop_marks_branches() {
        let mut x = XrayUi::default();
        let history = vec![json!({"role": "system", "content": "Be brief"}), json!({"role": "user", "content": "Why is the sky blue?"})];
        x.load_chat("Sky".into(), history.clone());
        assert_eq!(x.messages(), history);
        x.from_chat = None;
        x.system = "  ".into();
        assert_eq!(x.messages(), vec![json!({"role": "user", "content": x.prompt.trim()})]);

        let mut t = Tree::new("p".into());
        t.branches[0].state = BranchState::Running;
        x.tree = Some(t);
        x.running.push((0, Arc::new(AtomicBool::new(false))));
        let cancel = x.running[0].1.clone();
        x.stop();
        assert!(cancel.load(Ordering::Relaxed));
        assert_eq!(x.tree.as_ref().unwrap().branches[0].state, BranchState::Stopped);
        assert!(!x.busy());
    }

    #[test]
    fn token_text_is_made_visible() {
        assert_eq!(visible("\n"), "\\n");
        assert_eq!(visible("  "), "\u{b7}\u{b7}");
        assert_eq!(visible(""), "(empty)");
        assert_eq!(visible(" cat"), " cat");
    }
}
