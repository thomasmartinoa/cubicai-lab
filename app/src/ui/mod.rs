//! The window: a top bar with the brand mark, tabs and the server status, then the open screen.

pub mod anatomy;
pub mod chat;
pub mod engine;
pub mod phone;
pub mod settings;
pub mod steer;
pub mod tune;
pub mod xray;

use crate::app::{App, Tab};
use crate::engine::server::State;
use crate::theme::{self, pal};
use crate::widgets::{self, cut_shape};
use egui::{vec2, Color32, RichText, Sense, Stroke};

pub fn frame(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    // eframe hands over a bare Ui: paint the page colour first.
    ui.painter().rect_filled(ui.max_rect(), 0.0, p.bg);
    egui::Panel::top("top")
        .frame(egui::Frame::new().fill(p.lowest).inner_margin(egui::Margin::symmetric(16, 8)))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let spinning = app.busy() || app.xray.busy() || app.steer.busy() || app.tune.busy() || app.server.state() == State::Loading;
                // What fits decides what shows: the brand's words and the server's words give way first, then the tabs scroll.
                let tabs = [(Tab::Chat, "Chat"), (Tab::Engine, "Engine Room"), (Tab::Anatomy, "Anatomy"), (Tab::XRay, "X-ray"), (Tab::Steering, "Steering"), (Tab::Tune, "Fine-tune"), (Tab::Phone, "Phone"), (Tab::Settings, "Settings")];
                let gap = ui.spacing().item_spacing.x;
                let tabs_w: f32 = tabs.iter().map(|(_, l)| tab_size(ui, l).x + gap).sum();
                let brand_w = 24.0 + text_width(ui, "CubicAI", theme::heading(19.0)) + text_width(ui, "Lab", theme::heading(19.0)) + 3.0 * gap + 18.0;
                let (dot, text) = server_text(app, &p);
                let pill_full = text_width(ui, &text, egui::TextStyle::Small.resolve(ui.style())) + 9.0 + 3.0 * gap;
                let pill_dot = 9.0 + 2.0 * gap;
                let fit = HeaderFit::new(ui.available_width(), brand_w, 24.0 + gap + 8.0, tabs_w, pill_full, pill_dot);
                let (show_brand, show_pill_text) = (fit.brand, fit.pill_words);
                widgets::cube_mark(ui, 24.0, spinning);
                if show_brand {
                    ui.label(RichText::new("CubicAI").font(theme::heading(19.0)));
                    ui.label(RichText::new("Lab").font(theme::heading(19.0)).color(p.primary));
                    ui.add_space(18.0);
                } else {
                    ui.add_space(8.0);
                }
                let pill_w = if show_pill_text { pill_full } else { pill_dot };
                let room = (ui.available_width() - pill_w).max(0.0);
                let mut tab_row = |ui: &mut egui::Ui| {
                    for (tab, label) in tabs {
                        if tab_button(ui, app.tab == tab, label).clicked() {
                            app.tab = tab;
                        }
                    }
                };
                if !fit.scroll_tabs && tabs_w <= room {
                    tab_row(ui);
                } else {
                    // Too narrow for every tab: they scroll sideways (wheel or drag) instead of running under the status.
                    egui::ScrollArea::horizontal()
                        .max_width(room)
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                        .show(ui, |ui| ui.horizontal(|ui| tab_row(ui)));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    server_pill(app, ui, dot, &text, show_pill_text);
                });
            });
        });
    if let Some((msg, at)) = &app.toast {
        if at.elapsed().as_secs_f32() > 4.0 {
            app.toast = None;
        } else {
            let msg = msg.clone();
            egui::Panel::bottom("toast")
                .frame(egui::Frame::new().fill(p.primary_container).inner_margin(egui::Margin::symmetric(16, 6)))
                .show(ui, |ui| {
                    ui.label(RichText::new(msg).color(p.on_primary_container));
                });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
        }
    }
    egui::CentralPanel::no_frame().show(ui, |ui| match app.tab {
        Tab::Chat => chat::show(app, ui),
        Tab::Engine => engine::show(app, ui),
        Tab::Anatomy => anatomy::show(app, ui),
        Tab::XRay => xray::show(app, ui),
        Tab::Steering => steer::show(app, ui),
        Tab::Tune => tune::show(app, ui),
        Tab::Phone => phone::show(app, ui),
        Tab::Settings => settings::show(app, ui),
    });
}

fn tab_galley(ui: &egui::Ui, label: &str) -> std::sync::Arc<egui::Galley> {
    egui::WidgetText::from(RichText::new(label).font(egui::FontId::new(15.0, egui::FontFamily::Name("heading-medium".into())))).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::TextStyle::Button,
    )
}

fn tab_size(ui: &egui::Ui, label: &str) -> egui::Vec2 {
    tab_galley(ui, label).size() + vec2(22.0, 14.0)
}

fn tab_button(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response {
    let p = *pal(ui.ctx());
    let galley = tab_galley(ui, label);
    let (rect, resp) = ui.allocate_exact_size(galley.size() + vec2(22.0, 14.0), Sense::click());
    if selected {
        ui.painter().add(cut_shape(rect, 6.0, p.high, Stroke::new(1.0, p.outline_variant)));
        let bar = egui::Rect::from_min_max(egui::pos2(rect.left() + 8.0, rect.bottom() - 3.0), egui::pos2(rect.right() - 8.0, rect.bottom() - 1.0));
        ui.painter().rect_filled(bar, 0.0, p.primary);
    } else if resp.hovered() {
        ui.painter().add(cut_shape(rect, 6.0, p.low, Stroke::NONE));
    }
    let color = if selected { p.on_bg } else { p.on_variant };
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, color);
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// What the top bar shows at a width: the brand's words, the server's words, and whether the tabs must scroll.
#[derive(Debug, PartialEq)]
struct HeaderFit {
    brand: bool,
    pill_words: bool,
    scroll_tabs: bool,
}

impl HeaderFit {
    fn new(total: f32, brand_w: f32, cube_w: f32, tabs_w: f32, pill_full: f32, pill_dot: f32) -> HeaderFit {
        let brand = total >= brand_w + tabs_w + pill_dot;
        let used = if brand { brand_w } else { cube_w };
        let pill_words = total >= used + tabs_w + pill_full;
        let scroll_tabs = used + tabs_w + if pill_words { pill_full } else { pill_dot } > total;
        HeaderFit { brand, pill_words, scroll_tabs }
    }
}

fn server_text(app: &App, p: &theme::Palette) -> (Color32, String) {
    match app.server.state() {
        State::Ready => (p.good, format!("{} · ready", app.server.model_name)),
        State::Loading => (
            p.primary,
            format!("Loading {} · {:.0}s", app.server.model_name, app.server.started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0)),
        ),
        State::Failed(_) => (p.error, "Server stopped with an error".into()),
        State::Stopped => (p.outline, "No local model running".into()),
    }
}

fn text_width(ui: &egui::Ui, text: &str, font: egui::FontId) -> f32 {
    ui.fonts_mut(|f| f.layout_no_wrap(text.to_string(), font, Color32::WHITE).size().x)
}

/// The server's state on the right of the bar; only its dot when the window is narrow (the words show on hover).
fn server_pill(app: &mut App, ui: &mut egui::Ui, dot: Color32, text: &str, words: bool) {
    let p = *pal(ui.ctx());
    let resp = ui
        .horizontal(|ui| {
            if words {
                ui.label(RichText::new(text).small().color(p.on_variant));
            }
            let (r, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
            ui.painter().add(cut_shape(r, 2.0, dot, Stroke::NONE));
        })
        .response
        .interact(Sense::click());
    if resp.clicked() {
        app.tab = Tab::Engine;
    }
    resp.on_hover_text(if words { "Open the Engine Room".to_string() } else { format!("{text} · open the Engine Room") });
}

#[cfg(test)]
mod tests {
    use super::HeaderFit;

    #[test]
    fn the_top_bar_gives_way_in_order() {
        // Brand 150 (cube alone 40), tabs 640, server words 170, its dot 25.
        let fit = |w| HeaderFit::new(w, 150.0, 40.0, 640.0, 170.0, 25.0);
        assert_eq!(fit(1380.0), HeaderFit { brand: true, pill_words: true, scroll_tabs: false });
        // The words beside the dot go first (this is where they used to run over the tabs)...
        assert_eq!(fit(900.0), HeaderFit { brand: true, pill_words: false, scroll_tabs: false });
        // ...then the brand's words, then the tabs scroll instead of overlapping anything.
        assert_eq!(fit(760.0), HeaderFit { brand: false, pill_words: false, scroll_tabs: false });
        assert_eq!(fit(600.0), HeaderFit { brand: false, pill_words: false, scroll_tabs: true });
    }
}
