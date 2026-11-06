//! Anatomy: a model drawn from its file. Every layer is a row of tiles sized by bytes, coloured by
//! part, precision or how spread its weights are; click a tensor to see its histogram, a map of
//! its whole matrix and the actual numbers under the cursor.

use crate::app::{short_model_name, App};
use crate::engine::library::{self, Info, Model, Source};
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, fmt_bytes, fmt_count, lerp_color, Kind};
use cubic_gguf::layout::{base_name, describe, local_of, part_of};
use cubic_gguf::stats::{self, Overview, Progress, Stats};
use cubic_gguf::{Part, TensorInfo};
use egui::{pos2, vec2, Color32, Rect, RichText, ScrollArea, Sense, Stroke, TextureHandle};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Part,
    Precision,
    Spread,
}

#[derive(Default)]
pub struct AnatomyUi {
    /// A model key to open (set by the Engine Room).
    pub request: Option<String>,
    loading: Option<Arc<Mutex<Option<Result<Loaded, String>>>>>,
    loaded: Option<Loaded>,
    error: Option<String>,
    color: ColorMode,
    selected: Option<usize>,
    hovered: Option<usize>,
    scan: Option<Scan>,
    detail: Option<Detail>,
    zoom: Option<Zoom>,
}

#[derive(Clone)]
struct Loaded {
    key: String,
    name: String,
    maps: Arc<Vec<memmap2::Mmap>>,
    info: Arc<Info>,
}

impl Loaded {
    fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = self.info.gguf.tensor_start(t) as usize;
        let map = &self.maps[t.file.min(self.maps.len() - 1)];
        let end = (start + t.n_bytes as usize).min(map.len());
        &map[start.min(end)..end]
    }
}

struct Scan {
    progress: Arc<Progress>,
    stats: Arc<Mutex<Vec<Option<Stats>>>>,
    done: Arc<AtomicBool>,
}

struct Detail {
    tensor: usize,
    slice: u64,
    progress: Arc<Progress>,
    quick: Arc<Mutex<Option<Stats>>>,
    overview: Arc<Mutex<Option<Option<Overview>>>>,
    vector: Option<Vec<(f32, f32)>>,
    texture: Option<TextureHandle>,
    p99: f32,
}

struct Zoom {
    tensor: usize,
    slice: u64,
    row0: u64,
    col0: usize,
    n: usize,
    values: Vec<f32>,
    absmax: f32,
    texture: TextureHandle,
}

impl Drop for Detail {
    fn drop(&mut self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Scan {
    fn drop(&mut self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }
}

impl AnatomyUi {
    #[allow(dead_code)]
    pub fn scan_done(&self) -> bool {
        self.scan.as_ref().is_some_and(|s| s.done.load(Ordering::Relaxed))
    }

    #[allow(dead_code)]
    pub fn detail_done(&self) -> bool {
        self.detail.as_ref().is_some_and(|d| d.overview.lock().unwrap().is_some() || d.vector.is_some())
    }

    /// Selects a tensor by name (screenshots, links from elsewhere).
    #[allow(dead_code)]
    pub fn select_named(&mut self, name: &str) {
        if let Some(l) = &self.loaded {
            self.selected = l.info.gguf.tensors.iter().position(|t| t.name == name);
            self.detail = None;
            self.zoom = None;
        }
    }
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    if let Some(key) = app.anatomy.request.take()
        && let Some(m) = app.library.get(&key).cloned() {
            open(app, m, ui.ctx());
        }
    if let Some(slot) = &app.anatomy.loading {
        let r = slot.lock().unwrap().take();
        if let Some(r) = r {
            app.anatomy.loading = None;
            match r {
                Ok(l) => {
                    start_scan(&mut app.anatomy, &l, ui.ctx());
                    app.anatomy.loaded = Some(l);
                    app.anatomy.error = None;
                }
                Err(e) => app.anatomy.error = Some(e),
            }
        }
    }

    egui::Panel::top("anatomy-bar")
        .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 10)))
        .show(ui, |ui| toolbar(app, ui, &p));

    let Some(l) = app.anatomy.loaded.clone() else {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.25);
            if app.anatomy.loading.is_some() {
                ui.spinner();
                ui.label("Opening…");
            } else {
                widgets::cube_mark(ui, 56.0, false);
                ui.add_space(10.0);
                ui.label(RichText::new("See inside a model").font(theme::heading(24.0)));
                ui.label(RichText::new("Pick one above. Nothing runs: everything here is read straight from the file.").color(p.on_variant));
            }
            if let Some(e) = &app.anatomy.error {
                ui.colored_label(p.error, e);
            }
        });
        return;
    };

    egui::Panel::right("anatomy-detail")
        .resizable(true)
        .default_size(470.0)
        .size_range(360.0..=760.0)
        .frame(egui::Frame::new().fill(p.low).inner_margin(egui::Margin::same(16)))
        .show(ui, |ui| {
            ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match app.anatomy.selected {
                Some(t) => tensor_detail(&mut app.anatomy, &l, ui, &p, t),
                None => model_card(&mut app.anatomy, &l, ui, &p),
            });
        });
    egui::CentralPanel::no_frame().show(ui, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Frame::new().inner_margin(egui::Margin { left: 16, right: 16, top: 4, bottom: 16 }).show(ui, |ui| {
                summary(&app.anatomy, &l, ui, &p);
                ui.add_space(10.0);
                blueprint(&mut app.anatomy, &l, ui, &p);
            });
        });
    });
}

fn toolbar(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.horizontal(|ui| {
        let current = app.anatomy.loaded.as_ref().map(|l| short_model_name(&l.name)).unwrap_or_else(|| "Choose a model".into());
        let mut pick = None;
        egui::ComboBox::from_id_salt("anatomy-model").selected_text(current).width(320.0).height(520.0).show_ui(ui, |ui| {
            for m in &app.library.models {
                let sel = app.anatomy.loaded.as_ref().is_some_and(|l| l.key == m.key);
                if ui.selectable_label(sel, format!("{}   {}", short_model_name(&m.name), fmt_bytes(m.size))).clicked() {
                    pick = Some(m.clone());
                }
            }
        });
        if widgets::button(ui, Kind::Ghost, "Open file…").clicked()
            && let Some(f) = rfd::FileDialog::new().add_filter("GGUF model", &["gguf"]).pick_file() {
                let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
                pick = Some(Model {
                    key: f.to_string_lossy().to_string(),
                    name: f.file_name().unwrap_or_default().to_string_lossy().to_string(),
                    parts: vec![f],
                    size,
                    mmproj: None,
                    source: Source::Folder,
                });
            }
        if let Some(m) = pick {
            open(app, m, ui.ctx());
        }
        if app.anatomy.loaded.is_some() {
            ui.add_space(16.0);
            ui.label(RichText::new("Colour by").color(p.on_variant));
            widgets::segmented(ui, &mut app.anatomy.color, &[(ColorMode::Part, "Part"), (ColorMode::Precision, "Precision"), (ColorMode::Spread, "Spread")]);
            if let Some(s) = &app.anatomy.scan
                && !s.done.load(Ordering::Relaxed) {
                    ui.add_space(8.0);
                    ui.spinner();
                    ui.label(RichText::new(format!("Sampling weights {:.0}%", s.progress.done.load(Ordering::Relaxed) as f32 / 10.0)).small().color(p.on_variant));
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
                }
        }
    });
}

fn open(app: &mut App, m: Model, ctx: &egui::Context) {
    app.anatomy.selected = None;
    app.anatomy.detail = None;
    app.anatomy.zoom = None;
    app.anatomy.scan = None;
    app.anatomy.loaded = None;
    let cached = app.library.info(&m.key);
    let slot: Arc<Mutex<Option<Result<Loaded, String>>>> = Default::default();
    app.anatomy.loading = Some(slot.clone());
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let r = (|| {
            let mut maps = vec![];
            for p in &m.parts {
                let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
                maps.push(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?);
            }
            let info = match cached {
                Some(i) => i,
                None => Arc::new(library::read_info(&m)?),
            };
            Ok(Loaded { key: m.key.clone(), name: m.name.clone(), maps: Arc::new(maps), info })
        })();
        *slot.lock().unwrap() = Some(r);
        ctx.request_repaint();
    });
}

/// Samples every tensor's spread in the background (a few rows each, so it's quick even for big files).
fn start_scan(a: &mut AnatomyUi, l: &Loaded, ctx: &egui::Context) {
    let n = l.info.gguf.tensors.len();
    let progress = Arc::new(Progress::default());
    let stats = Arc::new(Mutex::new(vec![None; n]));
    let done = Arc::new(AtomicBool::new(false));
    let (pr, st, dn, l, ctx) = (progress.clone(), stats.clone(), done.clone(), l.clone(), ctx.clone());
    std::thread::spawn(move || {
        // Biggest matrices are what people look at; norms are cheap anyway.
        for (i, t) in l.info.gguf.tensors.iter().enumerate() {
            if pr.cancel.load(Ordering::Relaxed) {
                return;
            }
            let s = stats::sample_stats(t, l.bytes(t), 8);
            st.lock().unwrap()[i] = s;
            pr.done.store((i * 1000 / n.max(1)) as u32, Ordering::Relaxed);
            if i % 16 == 0 {
                ctx.request_repaint();
            }
        }
        dn.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
    a.scan = Some(Scan { progress, stats, done });
}

// --- colours ---------------------------------------------------------------------------------------

const ATTN: Color32 = Color32::from_rgb(0x5B, 0x8D, 0xEF);
const ROUTER: Color32 = Color32::from_rgb(0xB4, 0x8E, 0xF5);
const OTHER: Color32 = Color32::from_rgb(0x5F, 0xC8, 0x9A);
const EMBED: Color32 = Color32::from_rgb(0x8E, 0x9B, 0xD6);

fn part_color(part: Part, p: &Palette) -> Color32 {
    match part {
        Part::Attention => ATTN,
        Part::Ffn => p.primary,
        Part::Experts => Color32::from_rgb(0xE0, 0x86, 0x2A),
        Part::SharedExpert => Color32::from_rgb(0xFF, 0xD3, 0x80),
        Part::Router => ROUTER,
        Part::Norm => p.outline,
        Part::Other => OTHER,
    }
}

fn part_label(part: Part) -> &'static str {
    match part {
        Part::Other => "Recurrent / other",
        other => other.label(),
    }
}

/// Bits per weight → red (2) … amber (4.5) … green (6.5) … blue (16+).
fn bpw_color(bits: f64) -> Color32 {
    let stops: [(f64, Color32); 6] = [
        (1.5, Color32::from_rgb(0xD9, 0x3F, 0x4C)),
        (3.0, Color32::from_rgb(0xF2, 0x7A, 0x3A)),
        (4.5, Color32::from_rgb(0xFF, 0xB5, 0x47)),
        (6.6, Color32::from_rgb(0x7F, 0xD8, 0x6E)),
        (8.5, Color32::from_rgb(0x3F, 0xB8, 0xAF)),
        (16.0, Color32::from_rgb(0x6C, 0x8E, 0xF0)),
    ];
    ramp(&stops, bits)
}

fn ramp(stops: &[(f64, Color32)], x: f64) -> Color32 {
    if x <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        if x <= w[1].0 {
            let t = ((x - w[0].0) / (w[1].0 - w[0].0)) as f32;
            return lerp_color(w[0].1, w[1].1, t);
        }
    }
    stops[stops.len() - 1].1
}

/// Magnitude 0..1 → dark navy → cobalt → amber → near white.
fn heat(t: f32) -> Color32 {
    let stops: [(f64, Color32); 5] = [
        (0.0, Color32::from_rgb(0x0B, 0x0E, 0x1E)),
        (0.3, Color32::from_rgb(0x2B, 0x3B, 0x8F)),
        (0.62, Color32::from_rgb(0xC0, 0x5A, 0x8A)),
        (0.85, Color32::from_rgb(0xFF, 0xB5, 0x47)),
        (1.0, Color32::from_rgb(0xFF, 0xF4, 0xD6)),
    ];
    ramp(&stops, t.clamp(0.0, 1.0) as f64)
}

/// Signed -1..1 → blue … dark … amber.
fn diverge(t: f32, mid: Color32) -> Color32 {
    let t = t.clamp(-1.0, 1.0);
    if t < 0.0 {
        lerp_color(mid, Color32::from_rgb(0x5B, 0x8D, 0xEF), (-t).powf(0.7))
    } else {
        lerp_color(mid, Color32::from_rgb(0xFF, 0xB5, 0x47), t.powf(0.7))
    }
}

fn tensor_color(a: &AnatomyUi, l: &Loaded, i: usize, p: &Palette, medians: &HashMap<String, f64>) -> Color32 {
    let t = &l.info.gguf.tensors[i];
    match a.color {
        ColorMode::Part if !t.name.starts_with("blk.") => EMBED,
        ColorMode::Part => part_color(part_of(local_of(&t.name)), p),
        ColorMode::Precision => match t.ty.bits_per_weight() {
            Some(b) => bpw_color(b),
            None => p.outline,
        },
        ColorMode::Spread => {
            let s = a.scan.as_ref().and_then(|s| s.stats.lock().unwrap()[i].clone());
            match s {
                Some(s) => {
                    let base = base_name(local_of(&t.name)).to_string();
                    let med = medians.get(&base).copied().unwrap_or(s.std).max(1e-12);
                    diverge(((s.std / med).log2()) as f32, p.highest)
                }
                None => p.lowest,
            }
        }
    }
}

/// Median std per tensor kind (attn_q, ffn_down…) across layers, the baseline "Spread" compares to.
fn spread_medians(a: &AnatomyUi, l: &Loaded) -> HashMap<String, f64> {
    let mut by: HashMap<String, Vec<f64>> = HashMap::new();
    if let Some(s) = &a.scan {
        let st = s.stats.lock().unwrap();
        for (i, t) in l.info.gguf.tensors.iter().enumerate() {
            if let Some(s) = &st[i] {
                by.entry(base_name(local_of(&t.name)).to_string()).or_default().push(s.std);
            }
        }
    }
    by.into_iter()
        .map(|(k, mut v)| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (k, v[v.len() / 2])
        })
        .collect()
}

// --- summary and blueprint -------------------------------------------------------------------------

fn summary(a: &AnatomyUi, l: &Loaded, ui: &mut egui::Ui, p: &Palette) {
    let info = &l.info;
    let an = &info.anatomy;
    let h = &an.hyper;
    widgets::card(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(short_model_name(&l.name)).font(theme::heading(20.0)));
            widgets::badge(ui, &h.arch, p.on_primary_container, p.primary_container);
            if info.ollama_packaging {
                widgets::badge(ui, "vision/audio inside", p.on_variant, p.highest);
            }
        });
        let mut facts = vec![format!("{} parameters", fmt_count(an.params)), fmt_bytes(an.bytes), format!("{} layers", an.layers.len())];
        if h.experts > 0 {
            facts.insert(1, format!("{} active per token", fmt_count(info.active_params)));
            facts.push(format!("{} experts, {} per token", h.experts, h.experts_used));
        }
        if h.heads > 0 {
            facts.push(format!("{} query heads / {} KV heads", h.heads, h.heads_kv));
        }
        if h.embd > 0 {
            facts.push(format!("width {}", h.embd));
        }
        if h.vocab > 0 {
            facts.push(format!("{} vocabulary", fmt_count(h.vocab)));
        }
        ui.label(RichText::new(facts.join("  ·  ")).color(p.on_variant));
        ui.add_space(6.0);
        // Where the bytes are, by type.
        let segs: Vec<(String, f64, Color32)> = an
            .type_mix
            .iter()
            .map(|(t, b, _)| (t.name(), *b as f64, t.bits_per_weight().map(bpw_color).unwrap_or(p.outline)))
            .collect();
        widgets::stacked_bar(ui, 12.0, an.bytes as f64, &segs, None);
        let total = an.bytes.max(1) as f64;
        widgets::legend(ui, &segs.iter().filter(|s| s.1 / total > 0.002).map(|(n, b, c)| (format!("{n} {:.0}%", b / total * 100.0), *c)).collect::<Vec<_>>());
        // What the layers are made of.
        let kinds = layer_kinds(l);
        let n_swa = kinds.iter().filter(|k| matches!(k, LayerKind::Window(_))).count();
        let n_rec = kinds.iter().filter(|k| matches!(k, LayerKind::Recurrent)).count();
        let n_shared = kinds.iter().filter(|k| matches!(k, LayerKind::Shared)).count();
        let n_full = kinds.iter().filter(|k| matches!(k, LayerKind::Full)).count();
        let mut mix = vec![];
        if n_full > 0 {
            mix.push(format!("{n_full} full-attention"));
        }
        if n_swa > 0 {
            mix.push(format!("{n_swa} sliding-window ({} tokens)", h.sliding_window));
        }
        if n_rec > 0 {
            mix.push(format!("{n_rec} linear/recurrent"));
        }
        if n_shared > 0 {
            mix.push(format!("{n_shared} reusing another layer's KV"));
        }
        if !mix.is_empty() {
            ui.label(RichText::new(format!("Layers: {}", mix.join(", "))).small().color(p.on_variant));
        }
        let _ = a;
    });
}

#[derive(Clone, Copy, PartialEq)]
enum LayerKind {
    Full,
    Window(u64),
    Recurrent,
    Shared,
    Plain,
}

fn layer_kinds(l: &Loaded) -> Vec<LayerKind> {
    let kv: HashMap<usize, Option<u64>> = l.info.kv.iter().map(|k| (k.layer, k.window)).collect();
    l.info
        .anatomy
        .layers
        .iter()
        .map(|layer| {
            if layer.recurrent {
                LayerKind::Recurrent
            } else if layer.has_attention {
                match kv.get(&layer.index) {
                    Some(Some(w)) => LayerKind::Window(*w),
                    Some(None) => LayerKind::Full,
                    None => LayerKind::Shared,
                }
            } else {
                LayerKind::Plain
            }
        })
        .collect()
}

fn blueprint(a: &mut AnatomyUi, l: &Loaded, ui: &mut egui::Ui, p: &Palette) {
    let info = &l.info;
    let an = &info.anatomy;
    let medians = if a.color == ColorMode::Spread { spread_medians(a, l) } else { HashMap::new() };
    let kinds = layer_kinds(l);
    // Legend for the current colouring.
    match a.color {
        ColorMode::Part => {
            let mut parts: Vec<Part> = vec![];
            for layer in &an.layers {
                for &i in &layer.tensors {
                    let pt = part_of(local_of(&info.gguf.tensors[i].name));
                    if !parts.contains(&pt) {
                        parts.push(pt);
                    }
                }
            }
            parts.sort();
            let mut items: Vec<(String, Color32)> = parts.iter().map(|pt| (part_label(*pt).to_string(), part_color(*pt, p))).collect();
            items.push(("Embeddings & head".into(), EMBED));
            widgets::legend(ui, &items);
        }
        ColorMode::Precision => {
            widgets::legend(ui, &[2.0, 3.0, 4.5, 5.5, 6.6, 8.5, 16.0].iter().map(|b| (format!("{b} bits"), bpw_color(*b))).collect::<Vec<_>>());
        }
        ColorMode::Spread => {
            widgets::legend(
                ui,
                &[
                    ("half as spread as usual for its kind".to_string(), diverge(-1.0, p.highest)),
                    ("typical".to_string(), diverge(0.0, p.highest)),
                    ("twice as spread".to_string(), diverge(1.0, p.highest)),
                ],
            );
        }
    }
    ui.add_space(6.0);
    let label_w = 34.0;
    let kind_w = 58.0;
    let row_h = 17.0;
    let gap = 3.0;
    let width = ui.available_width();
    let tiles_w = (width - label_w - kind_w - 8.0).max(200.0);
    // Tiles have a minimum width and gaps; pick the scale that lets the fullest row still fit.
    let fixed = |ts: &[usize]| -> f32 { ts.len() as f32 * 4.5 + 6.0 * 5.0 };
    let scale = an
        .layers
        .iter()
        .map(|x| (tiles_w - fixed(&x.tensors)).max(40.0) / x.bytes.max(1) as f32)
        .fold(f32::INFINITY, f32::min);
    let scale = if scale.is_finite() { scale } else { 1.0 };
    let mut hovered = None;
    let mut clicked = None;
    let pointer = ui.input(|i| i.pointer.hover_pos());

    // One row of tiles. Returns the row's rect.
    let mut draw_row = |ui: &mut egui::Ui, label: &str, kind: Option<LayerKind>, tensors: &[usize], scale: f32| {
        let (rect, resp) = ui.allocate_exact_size(vec2(width, row_h), Sense::click());
        let painter = ui.painter();
        painter.text(pos2(rect.left() + label_w - 6.0, rect.center().y), egui::Align2::RIGHT_CENTER, label, egui::FontId::monospace(11.0), p.on_variant);
        if let Some(k) = kind {
            let (txt, c) = match k {
                LayerKind::Full => ("full".to_string(), ATTN),
                LayerKind::Window(w) => (format!("win {}", fmt_count(w)), p.secondary),
                LayerKind::Recurrent => ("linear".to_string(), OTHER),
                LayerKind::Shared => ("shared".to_string(), p.outline),
                LayerKind::Plain => (String::new(), p.outline),
            };
            painter.text(pos2(rect.left() + label_w + 2.0, rect.center().y), egui::Align2::LEFT_CENTER, txt, egui::FontId::proportional(10.5), c);
        }
        let mut x = rect.left() + label_w + kind_w;
        let mut last_part = None;
        for &i in tensors {
            let t = &info.gguf.tensors[i];
            let part = part_of(local_of(&t.name));
            if last_part.is_some_and(|lp| lp != part) {
                x += 5.0;
            }
            last_part = Some(part);
            let w = (t.n_bytes as f32 * scale).max(3.0);
            let r = Rect::from_min_size(pos2(x, rect.top() + 1.0), vec2(w, row_h - 2.0));
            x += w + 1.5;
            if r.left() > rect.right() {
                continue;
            }
            let r = r.intersect(rect);
            let mut c = tensor_color(a, l, i, p, &medians);
            let is_sel = a.selected == Some(i);
            let is_hover = pointer.is_some_and(|ptr| r.expand2(vec2(0.75, 1.0)).contains(ptr));
            if is_hover {
                hovered = Some(i);
                c = lerp_color(c, Color32::WHITE, 0.25);
            }
            painter.rect_filled(r, 0.0, c);
            if is_sel {
                painter.rect_stroke(r.expand(1.5), 0.0, Stroke::new(2.0, p.on_bg), egui::StrokeKind::Outside);
            }
        }
        if resp.clicked() {
            clicked = hovered;
        }
        ui.add_space(gap);
    };

    let section = |ui: &mut egui::Ui, text: &str| {
        ui.label(RichText::new(text).font(theme::heading(14.0)).color(p.on_variant));
    };
    if !an.before.is_empty() {
        section(ui, "Input");
        let bytes: u64 = an.before.iter().map(|&i| info.gguf.tensors[i].n_bytes).sum();
        let s = scale.min((tiles_w - fixed(&an.before)) / bytes.max(1) as f32);
        draw_row(ui, "in", None, &an.before, s);
        if s < scale {
            widgets::hint(ui, "(drawn smaller than the layers: embeddings are much bigger than one layer)");
        }
    }
    section(ui, &format!("{} layers", an.layers.len()));
    for (li, layer) in an.layers.iter().enumerate() {
        draw_row(ui, &layer.index.to_string(), kinds.get(li).copied(), &layer.tensors, scale);
    }
    if !an.after.is_empty() {
        section(ui, "Output");
        let bytes: u64 = an.after.iter().map(|&i| info.gguf.tensors[i].n_bytes).sum();
        let s = scale.min((tiles_w - fixed(&an.after)) / bytes.max(1) as f32);
        draw_row(ui, "out", None, &an.after, s);
    }
    if !an.extra.is_empty() {
        section(ui, "Other parts (vision / audio towers, extras)");
        // Group extras in rows of at most 24 tensors.
        for (k, chunk) in an.extra.chunks(24).enumerate() {
            let bytes: u64 = chunk.iter().map(|&i| info.gguf.tensors[i].n_bytes).sum();
            let s = scale.min((tiles_w - fixed(chunk)) / bytes.max(1) as f32);
            draw_row(ui, &format!("+{k}"), None, chunk, s);
        }
    }

    a.hovered = hovered;
    if let Some(i) = hovered {
        let t = &info.gguf.tensors[i];
        let st = a.scan.as_ref().and_then(|s| s.stats.lock().unwrap()[i].clone());
        egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::unique("tensor-tip"), egui::PopupAnchor::Pointer).show(|ui| {
            ui.set_max_width(340.0);
            ui.label(RichText::new(&t.name).monospace().strong());
            let d = describe(&t.name);
            if !d.is_empty() {
                ui.label(d);
            }
            ui.label(RichText::new(format!("{}  ·  {}  ·  {}", t.shape_text(), t.ty, fmt_bytes(t.n_bytes))).small());
            if let Some(s) = st {
                ui.label(RichText::new(format!("spread (std) {:.4}", s.std)).small());
            }
        });
    }
    if let Some(i) = clicked
        && a.selected != Some(i) {
            a.selected = Some(i);
            a.detail = None;
            a.zoom = None;
        }
}

// --- right panel -----------------------------------------------------------------------------------

fn model_card(a: &mut AnatomyUi, l: &Loaded, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("How to read this").font(theme::heading(17.0)));
    ui.label("Each row is one layer, read top to bottom: a token's vector flows down through all of them. Tiles are the layer's weight tensors, as wide as their share of the bytes.");
    ui.add_space(4.0);
    widgets::hint(ui, "Hover a tile to see what it does; click it to look at its weights.");
    ui.add_space(12.0);
    layer_profile(a, l, ui, p);
    ui.add_space(12.0);
    ui.label(RichText::new("Hyperparameters").font(theme::heading(15.0)));
    let h = &l.info.anatomy.hyper;
    egui::Grid::new("hyper").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        let mut row = |k: &str, v: String| {
            if !v.is_empty() && v != "0" {
                ui.label(RichText::new(k).color(p.on_variant));
                ui.label(v);
                ui.end_row();
            }
        };
        row("Architecture", h.arch.clone());
        row("Name", h.name.clone());
        row("Hidden width", h.embd.to_string());
        row("Feed-forward width", h.ffn.to_string());
        row("Query heads", h.heads.to_string());
        row("KV heads", h.heads_kv.to_string());
        row("Head size", h.head_dim.to_string());
        row("Experts", h.experts.to_string());
        row("Experts per token", h.experts_used.to_string());
        row("Expert width", h.expert_ffn.to_string());
        row("Trained context", fmt_count(h.ctx_train));
        row("Sliding window", h.sliding_window.to_string());
        row("RoPE base", if h.rope_base > 0.0 { format!("{}", h.rope_base) } else { String::new() });
        row("Vocabulary", h.vocab.to_string());
        row("GGUF version", l.info.gguf.version.to_string());
    });
    ui.add_space(12.0);
    egui::CollapsingHeader::new(format!("All metadata ({} keys)", l.info.gguf.metadata.len())).show(ui, |ui| {
        egui::Grid::new("meta").num_columns(2).spacing([12.0, 2.0]).show(ui, |ui| {
            for (k, v) in &l.info.gguf.metadata {
                if k == "tokenizer.chat_template" {
                    continue;
                }
                ui.label(RichText::new(k).monospace().size(11.0).color(p.on_variant));
                ui.add(egui::Label::new(RichText::new(v.display()).size(11.5)).wrap());
                ui.end_row();
            }
        });
    });
    if let Some(t) = l.info.gguf.get_str("tokenizer.chat_template") {
        egui::CollapsingHeader::new("Chat template").show(ui, |ui| {
            ui.add(egui::Label::new(RichText::new(t).monospace().size(11.0)).wrap());
        });
    }
}

/// Spread of the main weight kinds, layer by layer: the model's fingerprint.
fn layer_profile(a: &AnatomyUi, l: &Loaded, ui: &mut egui::Ui, p: &Palette) {
    ui.label(RichText::new("Layer fingerprint").font(theme::heading(15.0)));
    let Some(scan) = &a.scan else { return };
    let st = scan.stats.lock().unwrap();
    let an = &l.info.anatomy;
    let kinds: [(&[&str], &str, Color32); 6] = [
        (&["attn_q"], "attention queries", ATTN),
        (&["attn_qkv"], "fused q/k/v (linear layers)", EMBED),
        (&["attn_output", "attn_out"], "attention out", ROUTER),
        (&["ssm_out"], "linear-attention out", OTHER),
        (&["ffn_up", "ffn_up_exps"], "feed-forward up", p.primary),
        (&["ffn_down", "ffn_down_exps"], "feed-forward down", Color32::from_rgb(0xE0, 0x86, 0x2A)),
    ];
    let mut series: Vec<(&str, Color32, Vec<Option<f64>>)> = vec![];
    for (bases, label, color) in kinds {
        let vals: Vec<Option<f64>> = an
            .layers
            .iter()
            .map(|layer| {
                layer
                    .tensors
                    .iter()
                    .find(|&&i| bases.contains(&base_name(local_of(&l.info.gguf.tensors[i].name))))
                    .and_then(|&i| st[i].as_ref().map(|s| s.std))
            })
            .collect();
        if vals.iter().filter(|v| v.is_some()).count() > 1 {
            series.push((label, color, vals));
        }
    }
    drop(st);
    if series.is_empty() {
        widgets::hint(ui, "Sampling…");
        return;
    }
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 150.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.add(widgets::cut_shape(rect, 6.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let inner = rect.shrink2(vec2(10.0, 12.0));
    let all: Vec<f64> = series.iter().flat_map(|s| s.2.iter().flatten().copied()).filter(|v| *v > 0.0).collect();
    let (lo, hi) = all.iter().fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v.ln()), hi.max(v.ln())));
    let span = (hi - lo).max(1e-6);
    let n = an.layers.len().max(2) as f32;
    for (_, color, vals) in &series {
        let pts: Vec<egui::Pos2> = vals
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                let v = (*v)?;
                if v <= 0.0 {
                    return None;
                }
                let x = inner.left() + i as f32 / (n - 1.0) * inner.width();
                let y = inner.bottom() - ((v.ln() - lo) / span) as f32 * inner.height();
                Some(pos2(x, y))
            })
            .collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.6, *color)));
    }
    if let Some(ptr) = resp.hover_pos() {
        let li = (((ptr.x - inner.left()) / inner.width()) * (n - 1.0)).round().clamp(0.0, n - 1.0) as usize;
        let x = inner.left() + li as f32 / (n - 1.0) * inner.width();
        painter.line_segment([pos2(x, inner.top()), pos2(x, inner.bottom())], Stroke::new(1.0, p.outline));
        resp.on_hover_ui(|ui| {
            ui.label(format!("Layer {li}"));
            for (label, color, vals) in &series {
                if let Some(Some(v)) = vals.get(li) {
                    ui.colored_label(*color, format!("{label}: {v:.4}"));
                }
            }
        });
    }
    widgets::legend(ui, &series.iter().map(|(l, c, _)| (l.to_string(), *c)).collect::<Vec<_>>());
    widgets::hint(ui, "Spread (standard deviation) of each kind of weight, log scale. Bumps show layers that were trained to act differently.");
}

fn tensor_detail(a: &mut AnatomyUi, l: &Loaded, ui: &mut egui::Ui, p: &Palette, ti: usize) {
    let ctx = ui.ctx().clone();
    let t = l.info.gguf.tensors[ti].clone();
    ui.horizontal(|ui| {
        if widgets::button(ui, Kind::Ghost, "Back to the model").clicked() {
            a.selected = None;
            a.detail = None;
            a.zoom = None;
        }
    });
    if a.selected.is_none() {
        return;
    }
    ui.add_space(6.0);
    ui.label(RichText::new(&t.name).font(theme::heading(17.0)));
    let d = describe(&t.name);
    if !d.is_empty() {
        ui.label(d);
    }
    ui.add_space(6.0);
    egui::Grid::new("tensor-facts").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
        let mut row = |k: &str, v: String| {
            ui.label(RichText::new(k).color(p.on_variant));
            ui.label(v);
            ui.end_row();
        };
        row("Shape", format!("{}  ({} numbers)", t.shape_text(), fmt_count(t.n_elements)));
        row("Stored as", format!("{}{}", t.ty, t.ty.bits_per_weight().map(|b| format!(" · {b:.2} bits each")).unwrap_or_default()));
        row("Size", fmt_bytes(t.n_bytes));
        row("Part", part_label(part_of(local_of(&t.name))).to_string());
    });
    let ex = t.ty.explain();
    if !ex.is_empty() {
        widgets::hint(ui, ex);
    }
    if !cubic_gguf::dequant::can_dequantize(t.ty) {
        ui.add_space(8.0);
        ui.colored_label(p.primary, format!("{} weights use lattice codebooks that this viewer doesn't decode yet; size and placement above are exact.", t.ty));
        return;
    }
    let (_, slices) = stats::slice_rows(&t);

    // (Re)start the detail job when the tensor or the slice changed.
    let slice = a.detail.as_ref().filter(|d| d.tensor == ti).map(|d| d.slice).unwrap_or(0);
    if slices > 1 {
        ui.add_space(6.0);
        let mut s = slice;
        ui.horizontal(|ui| {
            ui.label("Expert");
            ui.add(egui::Slider::new(&mut s, 0..=slices - 1));
            widgets::hint(ui, format!("of {slices}"));
        });
        if s != slice {
            a.detail = None;
            a.zoom = None;
            start_detail(a, l, ti, s, &ctx);
        }
    }
    if a.detail.as_ref().is_none_or(|d| d.tensor != ti) {
        start_detail(a, l, ti, 0, &ctx);
    }
    let d = a.detail.as_mut().unwrap();

    // Histogram.
    let stats = d.overview.lock().unwrap().as_ref().and_then(|o| o.as_ref().map(|o| o.stats.clone())).or_else(|| d.quick.lock().unwrap().clone());
    ui.add_space(10.0);
    ui.label(RichText::new("Value distribution").font(theme::heading(15.0)));
    match &stats {
        Some(s) => {
            histogram(ui, p, s);
            let seen = if s.rows_seen < s.rows_total { format!(" (sampled {} of {} rows)", fmt_count(s.rows_seen), fmt_count(s.rows_total)) } else { String::new() };
            widgets::hint(
                ui,
                format!("mean {:.5} · spread {:.5} · range {:.4} … {:.4} · zeros {:.2}%{seen}", s.mean, s.std, s.min, s.max, s.zeros * 100.0),
            );
        }
        None => {
            ui.spinner();
        }
    }

    // 1-D: plot the values.
    if t.dims.len() == 1 || t.n_rows() == 1 {
        if d.vector.is_none() {
            d.vector = stats::vector(&t, l.bytes(&t), 600);
        }
        if let Some(v) = &d.vector {
            ui.add_space(10.0);
            ui.label(RichText::new("Values, channel by channel").font(theme::heading(15.0)));
            vector_plot(ui, p, v);
            widgets::hint(ui, "For a norm, how much each channel of the hidden state is amplified. Spikes are the outlier channels models rely on.");
        }
        return;
    }

    // 2-D: the whole-matrix map.
    ui.add_space(10.0);
    ui.label(RichText::new("The whole matrix").font(theme::heading(15.0)));
    let ov = d.overview.lock().unwrap().clone();
    match ov {
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Reading every weight… {:.0}%", d.progress.done.load(Ordering::Relaxed) as f32 / 10.0));
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
        Some(None) => {
            ui.label("Couldn't read this tensor.");
        }
        Some(Some(o)) => {
            if d.texture.is_none() {
                let (tex, p99) = overview_texture(&ctx, &o);
                d.texture = Some(tex);
                d.p99 = p99;
            }
            let tex = d.texture.clone().unwrap();
            let w = ui.available_width();
            let aspect = (o.rows as f32 / o.cols as f32).clamp(0.3, 2.2);
            let img_w = w - 34.0;
            let img_h = img_w * aspect;
            let (rect, resp) = ui.allocate_exact_size(vec2(w, img_h + 30.0), Sense::click());
            let img = Rect::from_min_size(rect.min, vec2(img_w, img_h));
            ui.painter().image(tex.id(), img, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            // Row profile on the right, column profile below.
            let rmax = o.row_rms.iter().copied().fold(0f32, f32::max).max(1e-12);
            for (i, v) in o.row_rms.iter().enumerate() {
                let y0 = img.top() + i as f32 / o.h as f32 * img_h;
                let y1 = img.top() + (i + 1) as f32 / o.h as f32 * img_h;
                let len = v / rmax * 28.0;
                ui.painter().rect_filled(Rect::from_min_max(pos2(img.right() + 4.0, y0), pos2(img.right() + 4.0 + len, y1.max(y0 + 0.5))), 0.0, p.secondary.gamma_multiply(0.85));
            }
            let cmax = o.col_rms.iter().copied().fold(0f32, f32::max).max(1e-12);
            for (i, v) in o.col_rms.iter().enumerate() {
                let x0 = img.left() + i as f32 / o.w as f32 * img_w;
                let x1 = img.left() + (i + 1) as f32 / o.w as f32 * img_w;
                let len = v / cmax * 24.0;
                ui.painter().rect_filled(Rect::from_min_max(pos2(x0, img.bottom() + 4.0), pos2(x1.max(x0 + 0.5), img.bottom() + 4.0 + len)), 0.0, p.primary.gamma_multiply(0.85));
            }
            if let Some(z) = &a.zoom
                && z.tensor == ti {
                    let zr = Rect::from_min_size(
                        pos2(img.left() + z.col0 as f32 / o.cols as f32 * img_w, img.top() + z.row0 as f32 / o.rows as f32 * img_h),
                        vec2((z.n as f32 / o.cols as f32 * img_w).max(3.0), (z.n as f32 / o.rows as f32 * img_h).max(3.0)),
                    );
                    ui.painter().rect_stroke(zr, 0.0, Stroke::new(1.5, Color32::WHITE), egui::StrokeKind::Outside);
                }
            if let Some(ptr) = resp.hover_pos().filter(|ptr| img.contains(*ptr)) {
                let col = (((ptr.x - img.left()) / img_w) * o.cols as f32) as u64;
                let row = (((ptr.y - img.top()) / img_h) * o.rows as f32) as u64;
                resp.clone().on_hover_text(format!("output {row}, input {col}\nclick to see the actual numbers here"));
                if resp.clicked() {
                    let n = 48usize;
                    let row0 = row.saturating_sub(n as u64 / 2).min(o.rows.saturating_sub(n as u64));
                    let col0 = (col as usize).saturating_sub(n / 2).min((o.cols as usize).saturating_sub(n));
                    if let Some(values) = stats::window(&t, l.bytes(&t), slice, row0, col0, n, n) {
                        let absmax = values.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
                        let mid = p.highest;
                        let rgba: Vec<u8> = values.iter().flat_map(|v| diverge(v / absmax, mid).to_array()).collect();
                        let img = egui::ColorImage::from_rgba_unmultiplied([n, n], &rgba);
                        let texture = ctx.load_texture("zoom", img, egui::TextureOptions::NEAREST);
                        a.zoom = Some(Zoom { tensor: ti, slice, row0, col0, n, values, absmax, texture });
                    }
                }
            }
            widgets::hint(
                ui,
                format!(
                    "{} outputs × {} inputs, pooled to {}×{} cells of RMS magnitude. Bars: strength of each input (bottom) and output (right). Bright columns are the outlier features.",
                    o.rows, o.cols, o.h, o.w
                ),
            );
        }
    }
    // The numbers themselves.
    if let Some(z) = a.zoom.as_ref().filter(|z| z.tensor == ti && z.slice == slice) {
        ui.add_space(10.0);
        ui.label(RichText::new("Up close").font(theme::heading(15.0)));
        let w = ui.available_width().min(420.0);
        let (rect, resp) = ui.allocate_exact_size(vec2(w, w), Sense::hover());
        ui.painter().image(z.texture.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        if let Some(ptr) = resp.hover_pos() {
            let cx = (((ptr.x - rect.left()) / rect.width()) * z.n as f32) as usize;
            let cy = (((ptr.y - rect.top()) / rect.height()) * z.n as f32) as usize;
            if cx < z.n && cy < z.n {
                let v = z.values[cy * z.n + cx];
                resp.on_hover_text(format!("weight[{}, {}] = {v:+.6}", z.row0 + cy as u64, z.col0 + cx));
            }
        }
        widgets::hint(ui, format!("Outputs {}–{}, inputs {}–{}. Blue negative, amber positive, ±{:.4} at full colour.", z.row0, z.row0 + z.n as u64 - 1, z.col0, z.col0 + z.n - 1, z.absmax));
    }
}

fn start_detail(a: &mut AnatomyUi, l: &Loaded, ti: usize, slice: u64, ctx: &egui::Context) {
    let progress = Arc::new(Progress::default());
    let quick: Arc<Mutex<Option<Stats>>> = Default::default();
    let overview: Arc<Mutex<Option<Option<Overview>>>> = Default::default();
    let (pr, q, ov, l2, c) = (progress.clone(), quick.clone(), overview.clone(), l.clone(), ctx.clone());
    std::thread::spawn(move || {
        let t = &l2.info.gguf.tensors[ti];
        let data = l2.bytes(t);
        *q.lock().unwrap() = stats::sample_stats(t, data, 256);
        c.request_repaint();
        if t.dims.len() >= 2 && t.n_rows() > 1 {
            let o = stats::overview(t, data, slice, 512, 512, &pr);
            if pr.cancel.load(Ordering::Relaxed) {
                return;
            }
            *ov.lock().unwrap() = Some(o);
        }
        c.request_repaint();
    });
    a.detail = Some(Detail { tensor: ti, slice, progress, quick, overview, vector: None, texture: None, p99: 0.0 });
}

fn overview_texture(ctx: &egui::Context, o: &Overview) -> (TextureHandle, f32) {
    let mut sorted: Vec<f32> = o.rms.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = sorted.get(sorted.len() * 99 / 100).copied().unwrap_or(1.0).max(1e-12);
    let rgba: Vec<u8> = o.rms.iter().flat_map(|v| heat((v / p99).sqrt()).to_array()).collect();
    let img = egui::ColorImage::from_rgba_unmultiplied([o.w, o.h], &rgba);
    (ctx.load_texture("overview", img, egui::TextureOptions::LINEAR), p99)
}

fn histogram(ui: &mut egui::Ui, p: &Palette, s: &Stats) {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 120.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.add(widgets::cut_shape(rect, 6.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let inner = Rect::from_min_max(rect.min + vec2(8.0, 10.0), rect.max - vec2(8.0, 22.0));
    let max = s.hist.iter().copied().max().unwrap_or(1).max(1) as f32;
    let n = s.hist.len() as f32;
    let bw = inner.width() / n;
    for (i, &c) in s.hist.iter().enumerate() {
        // Square-root height so the tails stay visible.
        let h = (c as f32 / max).sqrt() * inner.height();
        let x = inner.left() + i as f32 * bw;
        let center = -s.range + (i as f32 + 0.5) / n * 2.0 * s.range;
        let col = diverge(center / s.range, p.outline);
        painter.rect_filled(Rect::from_min_max(pos2(x + 0.5, inner.bottom() - h), pos2(x + bw - 0.5, inner.bottom())), 0.0, col);
    }
    let zero_x = inner.center().x;
    painter.line_segment([pos2(zero_x, inner.top()), pos2(zero_x, inner.bottom())], Stroke::new(1.0, p.on_variant.gamma_multiply(0.5)));
    painter.text(inner.left_bottom() + vec2(0.0, 2.0), egui::Align2::LEFT_TOP, format!("{:.3}", -s.range), egui::FontId::proportional(10.0), p.on_variant);
    painter.text(inner.right_bottom() + vec2(0.0, 2.0), egui::Align2::RIGHT_TOP, format!("{:.3}", s.range), egui::FontId::proportional(10.0), p.on_variant);
    if let Some(ptr) = resp.hover_pos() {
        let i = (((ptr.x - inner.left()) / bw).floor().max(0.0) as usize).min(s.hist.len() - 1);
        let lo = -s.range + i as f32 / n * 2.0 * s.range;
        let hi = lo + 2.0 * s.range / n;
        resp.on_hover_text(format!("{lo:.4} … {hi:.4}: {} weights", s.hist[i]));
    }
}

fn vector_plot(ui: &mut egui::Ui, p: &Palette, v: &[(f32, f32)]) {
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 120.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.add(widgets::cut_shape(rect, 6.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let inner = rect.shrink2(vec2(8.0, 10.0));
    let lo = v.iter().map(|x| x.0).fold(0f32, f32::min);
    let hi = v.iter().map(|x| x.1).fold(0f32, f32::max);
    let span = (hi - lo).max(1e-9);
    let y = |val: f32| inner.bottom() - (val - lo) / span * inner.height();
    let n = v.len() as f32;
    for (i, (a, b)) in v.iter().enumerate() {
        let x = inner.left() + i as f32 / n * inner.width();
        painter.line_segment([pos2(x, y(*a)), pos2(x, y(*b).min(y(*a) - 0.8))], Stroke::new((inner.width() / n).max(1.0), p.primary));
    }
    painter.line_segment([pos2(inner.left(), y(0.0)), pos2(inner.right(), y(0.0))], Stroke::new(1.0, p.on_variant.gamma_multiply(0.5)));
    if let Some(ptr) = resp.hover_pos() {
        let i = (((ptr.x - inner.left()) / inner.width()) * n).clamp(0.0, n - 1.0) as usize;
        resp.on_hover_text(format!("{:.4} … {:.4}", v[i].0, v[i].1));
    }
}
