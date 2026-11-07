//! Cut-corner widgets shared by every screen: buttons, chips, cards, the switch, the cube mark,
//! stacked bars and small formatting helpers.

use crate::theme::{self, pal, Palette};
use egui::{
    pos2, vec2, Align2, Color32, CursorIcon, FontId, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2,
    WidgetText,
};

/// The eight corners of a rectangle with all four corners cut by `cut`.
pub fn cut_points(r: Rect, cut: f32) -> Vec<Pos2> {
    let c = cut.min(r.width() / 2.0).min(r.height() / 2.0).max(0.0);
    vec![
        pos2(r.left() + c, r.top()),
        pos2(r.right() - c, r.top()),
        pos2(r.right(), r.top() + c),
        pos2(r.right(), r.bottom() - c),
        pos2(r.right() - c, r.bottom()),
        pos2(r.left() + c, r.bottom()),
        pos2(r.left(), r.bottom() - c),
        pos2(r.left(), r.top() + c),
    ]
}

pub fn cut_shape(r: Rect, cut: f32, fill: Color32, stroke: Stroke) -> Shape {
    Shape::convex_polygon(cut_points(r, cut), fill, stroke)
}

/// A surface with cut corners behind whatever `add` draws.
pub fn surface<R>(
    ui: &mut Ui,
    fill: Color32,
    stroke: Stroke,
    cut: f32,
    margin: impl Into<egui::Margin>,
    add: impl FnOnce(&mut Ui) -> R,
) -> egui::InnerResponse<R> {
    let slot = ui.painter().add(Shape::Noop);
    let inner = egui::Frame::new().inner_margin(margin).show(ui, add);
    ui.painter().set(slot, cut_shape(inner.response.rect, cut, fill, stroke));
    inner
}

/// A settings-style card: raised tone, hairline edge, 10 px bevels.
pub fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let p = pal(ui.ctx());
    // The frame adds its margins outside the content, so the content gets what is left.
    let w = (ui.available_width() - 28.0).max(80.0);
    surface(ui, p.container, Stroke::new(1.0, p.outline_variant), 10.0, 14, |ui| {
        ui.set_width(w);
        add(ui)
    })
    .inner
}

pub fn card_title(ui: &mut Ui, text: &str) {
    ui.label(egui::RichText::new(text).font(theme::heading(17.0)).color(pal(ui.ctx()).on_bg));
    ui.add_space(4.0);
}

pub fn hint(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.label(egui::RichText::new(text.into()).small().color(pal(ui.ctx()).on_variant))
}

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Primary,
    Tonal,
    Ghost,
    #[allow(dead_code)]
    Danger,
}

/// A cut-corner button.
pub fn button(ui: &mut Ui, kind: Kind, text: impl Into<WidgetText>) -> Response {
    button_ext(ui, kind, text, true)
}

pub fn button_ext(ui: &mut Ui, kind: Kind, text: impl Into<WidgetText>, enabled: bool) -> Response {
    let p = *pal(ui.ctx());
    let font = theme::medium(14.0);
    let galley = text.into().into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, font);
    let size = vec2(galley.size().x + 26.0, 30.0f32.max(galley.size().y + 10.0));
    let (rect, resp) = ui.allocate_exact_size(size, if enabled { Sense::click() } else { Sense::hover() });
    let (fill, fg, stroke) = colors(&p, kind, enabled, resp.hovered(), resp.is_pointer_button_down_on());
    ui.painter().add(cut_shape(rect, 6.0, fill, stroke));
    let pos = rect.center() - galley.size() / 2.0;
    ui.painter().galley(pos, galley, fg);
    if enabled {
        resp.clone().on_hover_cursor(CursorIcon::PointingHand);
    }
    resp
}

fn colors(p: &Palette, kind: Kind, enabled: bool, hover: bool, down: bool) -> (Color32, Color32, Stroke) {
    if !enabled {
        return (Color32::TRANSPARENT, p.on_variant.gamma_multiply(0.7), Stroke::new(1.0, p.outline_variant));
    }
    let lift = |c: Color32| if down { c.gamma_multiply(0.85) } else if hover { c.gamma_multiply(1.12) } else { c };
    match kind {
        Kind::Primary => (lift(p.primary), p.on_primary, Stroke::NONE),
        Kind::Tonal => (lift(p.secondary_container), p.on_bg, Stroke::NONE),
        Kind::Ghost => (
            if hover { p.highest } else { Color32::TRANSPARENT },
            p.on_bg,
            Stroke::new(1.0, p.outline),
        ),
        Kind::Danger => (
            if hover { p.error_container } else { Color32::TRANSPARENT },
            p.error,
            Stroke::new(1.0, p.error.gamma_multiply(0.7)),
        ),
    }
}

/// A selectable chip (segmented choices, filters).
pub fn chip(ui: &mut Ui, selected: bool, text: impl Into<WidgetText>) -> Response {
    let p = *pal(ui.ctx());
    let galley = text.into().into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, theme::medium(13.0));
    let size = vec2(galley.size().x + 20.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let (fill, fg, stroke) = if selected {
        (p.primary_container, p.on_primary_container, Stroke::new(1.0, p.primary.gamma_multiply(0.8)))
    } else if resp.hovered() {
        (p.highest, p.on_bg, Stroke::new(1.0, p.outline))
    } else {
        (Color32::TRANSPARENT, p.on_variant, Stroke::new(1.0, p.outline_variant))
    };
    ui.painter().add(cut_shape(rect, 5.0, fill, stroke));
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
    resp.on_hover_cursor(CursorIcon::PointingHand)
}

/// Segmented choice among `options`; returns true when it changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for (v, label) in options {
            if chip(ui, *value == *v, *label).clicked() && *value != *v {
                *value = *v;
                changed = true;
            }
        }
    });
    changed
}

/// A cut-corner toggle switch.
pub fn switch(ui: &mut Ui, on: &mut bool) -> Response {
    let p = *pal(ui.ctx());
    let (rect, mut resp) = ui.allocate_exact_size(vec2(40.0, 22.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_with_time(resp.id, *on, 0.14);
    let track = if *on { p.primary } else { p.highest };
    ui.painter().add(cut_shape(rect, 5.0, track, Stroke::new(1.0, if *on { p.primary } else { p.outline })));
    let knob = Rect::from_min_size(pos2(egui::lerp(rect.left() + 3.0..=rect.right() - 19.0, t), rect.top() + 3.0), vec2(16.0, 16.0));
    ui.painter().add(cut_shape(knob, 4.0, if *on { p.on_primary } else { p.on_variant }, Stroke::NONE));
    resp.on_hover_cursor(CursorIcon::PointingHand)
}

/// A labelled row with a switch on the right.
pub fn switch_row(ui: &mut Ui, label: &str, detail: &str, on: &mut bool) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(label);
            if !detail.is_empty() {
                hint(ui, detail);
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            changed = switch(ui, on).changed();
        });
    });
    changed
}

/// The isometric cube mark. `phase` in 0..3 rolls the light around its faces.
pub fn cube(painter: &Painter, center: Pos2, r: f32, phase: f32, gap: Color32) {
    let p = theme::DARK.cube;
    let dx = r * 3f32.sqrt() / 2.0;
    let c = center;
    let top = pos2(c.x, c.y - r);
    let ur = pos2(c.x + dx, c.y - r / 2.0);
    let dr = pos2(c.x + dx, c.y + r / 2.0);
    let bottom = pos2(c.x, c.y + r);
    let dl = pos2(c.x - dx, c.y + r / 2.0);
    let ul = pos2(c.x - dx, c.y - r / 2.0);
    let step = phase.floor() as usize;
    let t = phase - phase.floor();
    let shade = |i: usize| lerp_color(p[(i + step) % 3], p[(i + step + 1) % 3], t);
    painter.add(Shape::convex_polygon(vec![top, ur, c, ul], shade(0), Stroke::NONE));
    painter.add(Shape::convex_polygon(vec![ul, c, bottom, dl], shade(1), Stroke::NONE));
    painter.add(Shape::convex_polygon(vec![c, ur, dr, bottom], shade(2), Stroke::NONE));
    let seam = Stroke::new((r * 0.07).max(1.0), gap);
    painter.line_segment([c, ul], seam);
    painter.line_segment([c, ur], seam);
    painter.line_segment([c, bottom], seam);
}

/// The cube as a widget; spins while `spinning`.
pub fn cube_mark(ui: &mut Ui, size: f32, spinning: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    let phase = if spinning {
        ui.ctx().request_repaint();
        (ui.input(|i| i.time) as f32 / 0.5) % 3.0
    } else {
        0.0
    };
    cube(ui.painter(), rect.center(), size / 2.0, phase, pal(ui.ctx()).bg);
    resp
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()), l(a.a(), b.a()))
}

/// A horizontal bar of segments (label, value, color) against `total`.
pub fn stacked_bar(ui: &mut Ui, height: f32, total: f64, segments: &[(String, f64, Color32)], limit: Option<f64>) -> Response {
    let p = *pal(ui.ctx());
    let w = ui.available_width();
    let (rect, resp) = ui.allocate_exact_size(vec2(w, height), Sense::hover());
    ui.painter().add(cut_shape(rect, 4.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let total = total.max(1.0);
    let mut x = rect.left();
    let inner = rect.shrink(1.0);
    for (_, v, c) in segments {
        let sw = (*v / total) as f32 * inner.width();
        let r = Rect::from_min_max(pos2(x, inner.top()), pos2((x + sw).min(inner.right()), inner.bottom()));
        if r.width() > 0.0 {
            ui.painter().rect_filled(r, 0.0, *c);
        }
        x += sw;
    }
    if let Some(l) = limit {
        let lx = inner.left() + (l / total).min(1.0) as f32 * inner.width();
        ui.painter().line_segment([pos2(lx, rect.top() - 3.0), pos2(lx, rect.bottom() + 3.0)], Stroke::new(2.0, p.on_bg));
    }
    resp
}

pub fn legend(ui: &mut Ui, items: &[(String, Color32)]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 12.0;
        for (label, c) in items {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let (r, _) = ui.allocate_exact_size(vec2(10.0, 10.0), Sense::hover());
                ui.painter().add(cut_shape(r, 2.5, *c, Stroke::NONE));
                ui.label(egui::RichText::new(label).small());
            });
        }
    });
}

pub fn badge(ui: &mut Ui, text: &str, fg: Color32, bg: Color32) -> Response {
    let galley = WidgetText::from(text).into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, FontId::new(11.0, theme::medium_family()));
    let (rect, resp) = ui.allocate_exact_size(galley.size() + vec2(10.0, 4.0), Sense::hover());
    ui.painter().add(cut_shape(rect, 3.0, bg, Stroke::NONE));
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, fg);
    resp
}

#[allow(dead_code)]
pub fn text_at(painter: &Painter, pos: Pos2, anchor: Align2, text: &str, font: FontId, color: Color32) {
    painter.text(pos, anchor, text, font, color);
}

pub fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{b:.0} B")
    }
}

pub fn fmt_count(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 {
        format!("{:.1}B", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.0}M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1}k", f / 1e3)
    } else {
        n.to_string()
    }
}

pub fn fmt_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms} ms")
    } else if ms < 60_000 {
        format!("{:.1} s", ms as f64 / 1000.0)
    } else {
        format!("{}m {:02}s", ms / 60_000, (ms / 1000) % 60)
    }
}

/// A page's one column: at most `max` wide (margins included) and centred, so a wide window doesn't
/// leave all its spare room on the right.
pub fn centred<R>(ui: &mut Ui, max: f32, add: impl FnOnce(&mut Ui) -> R) -> R {
    let avail = ui.available_width();
    let w = avail.min(max);
    let pad = ((avail - w) / 2.0).floor().max(0.0);
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.add_space(pad);
        ui.vertical(|ui| {
            ui.set_width(w);
            add(ui)
        })
        .inner
    })
    .inner
}
