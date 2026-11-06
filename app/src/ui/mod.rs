//! The window: a top bar with the brand mark, tabs and the server status, then the open screen.

pub mod anatomy;
pub mod chat;
pub mod engine;
pub mod settings;
pub mod steer;
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
                let spinning = app.busy() || app.xray.busy() || app.steer.busy() || app.server.state() == State::Loading;
                widgets::cube_mark(ui, 24.0, spinning);
                ui.label(RichText::new("CubicAI").font(theme::heading(19.0)));
                ui.label(RichText::new("Lab").font(theme::heading(19.0)).color(p.primary));
                ui.add_space(18.0);
                for (tab, label) in [(Tab::Chat, "Chat"), (Tab::Engine, "Engine Room"), (Tab::Anatomy, "Anatomy"), (Tab::XRay, "X-ray"), (Tab::Steering, "Steering"), (Tab::Settings, "Settings")] {
                    if tab_button(ui, app.tab == tab, label).clicked() {
                        app.tab = tab;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    server_pill(app, ui);
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
        Tab::Settings => settings::show(app, ui),
    });
}

fn tab_button(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response {
    let p = *pal(ui.ctx());
    let galley = egui::WidgetText::from(RichText::new(label).font(egui::FontId::new(15.0, egui::FontFamily::Name("heading-medium".into())))).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::TextStyle::Button,
    );
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

fn server_pill(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let (dot, text) = match app.server.state() {
        State::Ready => (p.good, format!("{} · ready", app.server.model_name)),
        State::Loading => (
            p.primary,
            format!("Loading {} · {:.0}s", app.server.model_name, app.server.started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0)),
        ),
        State::Failed(_) => (p.error, "Server stopped with an error".into()),
        State::Stopped => (p.outline, "No local model running".into()),
    };
    let resp = ui
        .horizontal(|ui| {
            ui.label(RichText::new(text).small().color(p.on_variant));
            let (r, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
            ui.painter().add(cut_shape(r, 2.0, dot, Stroke::NONE));
        })
        .response
        .interact(Sense::click());
    if resp.clicked() {
        app.tab = Tab::Engine;
    }
    resp.on_hover_text("Open the Engine Room");
    let _ = Color32::TRANSPARENT;
}
