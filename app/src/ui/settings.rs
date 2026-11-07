//! Settings: look, about you, API keys, folders, the local server, and custom behaviours.

use crate::app::App;
use crate::providers;
use crate::store::{new_id, Persona, ThemeChoice};
use crate::theme::pal;
use crate::widgets::{self, Kind};
use egui::{RichText, ScrollArea};
use std::collections::HashMap;

#[derive(Default)]
pub struct SettingsUi {
    keys: HashMap<String, String>,
    loaded: bool,
    editing: Option<Persona>,
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    if !app.settings_ui.loaded {
        for pr in providers::ALL.iter().filter(|p| p.needs_key) {
            let k = app.secrets.get(pr.id);
            app.settings_ui.keys.insert(pr.id.to_string(), k);
        }
        let hf = app.secrets.get("hf");
        app.settings_ui.keys.insert("hf".into(), hf);
        app.settings_ui.loaded = true;
    }
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        widgets::centred(ui, 792.0, |ui| {
            egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
            let mut dirty = false;

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "Look");
                dirty |= widgets::segmented(ui, &mut app.settings.theme, &[(ThemeChoice::System, "System"), (ThemeChoice::Light, "Light"), (ThemeChoice::Dark, "Dark")]);
            });
            ui.add_space(12.0);

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "About you");
                widgets::hint(ui, "Added to every behaviour except None, so replies fit you.");
                dirty |= ui
                    .add(egui::TextEdit::multiline(&mut app.settings.about_me).hint_text("Name, what you do, how you like answers…").desired_rows(3).desired_width(f32::INFINITY))
                    .changed();
            });
            ui.add_space(12.0);

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "Cloud providers");
                widgets::hint(ui, format!("Keys are kept in {}.", app.secrets.where_stored()));
                ui.add_space(6.0);
                for pr in providers::ALL.iter().filter(|p| p.needs_key) {
                    ui.horizontal(|ui| {
                        ui.add_sized([150.0, 24.0], egui::Label::new(pr.short).truncate());
                        let v = app.settings_ui.keys.entry(pr.id.to_string()).or_default();
                        let r = ui.add(egui::TextEdit::singleline(v).password(true).hint_text("API key").desired_width(300.0));
                        if r.lost_focus() {
                            let v = v.clone();
                            app.secrets.set(pr.id, &v);
                            app.chat.model_lists.lock().unwrap().remove(pr.id);
                        }
                        if !pr.key_url.is_empty() {
                            ui.hyperlink_to("Get a key", pr.key_url);
                        }
                    });
                }
                ui.horizontal(|ui| {
                    ui.add_sized([150.0, 24.0], egui::Label::new("Custom server URL"));
                    dirty |= ui
                        .add(egui::TextEdit::singleline(&mut app.settings.custom_base_url).hint_text("https://host/v1").desired_width(300.0))
                        .changed();
                });
            });
            ui.add_space(12.0);

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "Models and folders");
                ui.label(format!("Downloads go to {}", crate::engine::server::path_display(&app.paths.models)));
                ui.horizontal(|ui| {
                    if widgets::button(ui, Kind::Ghost, "Open folder").clicked() {
                        let _ = open::that(&app.paths.models);
                    }
                    if widgets::button(ui, Kind::Ghost, "Add another folder…").clicked()
                        && let Some(d) = rfd::FileDialog::new().pick_folder() {
                            app.settings.model_dirs.push(d);
                            dirty = true;
                            app.rescan(ui.ctx());
                        }
                });
                let mut remove = None;
                for (i, d) in app.settings.model_dirs.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(crate::engine::server::path_display(d));
                        if ui.small_button("Remove").clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    app.settings.model_dirs.remove(i);
                    dirty = true;
                    app.rescan(ui.ctx());
                }
                ui.add_space(6.0);
                let mut show = app.settings.show_ollama;
                let where_ = app.library.ollama_dir.as_ref().map(|d| crate::engine::server::path_display(d)).unwrap_or_else(|| "not found".into());
                if widgets::switch_row(ui, "Use Ollama's models", &format!("Read in place from {where_}, nothing copied"), &mut show) {
                    app.settings.show_ollama = show;
                    dirty = true;
                    app.rescan(ui.ctx());
                }
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label("Hugging Face token");
                    let v = app.settings_ui.keys.entry("hf".into()).or_default();
                    let r = ui.add(egui::TextEdit::singleline(v).password(true).hint_text("optional, for gated models").desired_width(280.0));
                    if r.lost_focus() {
                        let v = v.clone();
                        app.secrets.set("hf", &v);
                    }
                });
            });
            ui.add_space(12.0);

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "Local server");
                ui.horizontal(|ui| {
                    ui.label("Port");
                    dirty |= ui.add(egui::DragValue::new(&mut app.settings.port).range(1024..=65000)).changed();
                    widgets::hint(ui, "The next free port is used if this one is taken.");
                });
                ui.horizontal(|ui| {
                    ui.label("Plan with this much VRAM");
                    let mut on = app.settings.vram_override_mb.is_some();
                    if widgets::switch(ui, &mut on).changed() {
                        app.settings.vram_override_mb = on.then_some(8192);
                        dirty = true;
                    }
                    if let Some(mb) = &mut app.settings.vram_override_mb {
                        dirty |= ui.add(egui::DragValue::new(mb).range(512..=262144).suffix(" MB")).changed();
                    } else {
                        widgets::hint(ui, "(off: asks the GPU)");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Own llama.cpp build");
                    match &app.settings.llama_custom_dir {
                        Some(d) => {
                            ui.label(crate::engine::server::path_display(d));
                            if ui.small_button("Use the installed one").clicked() {
                                app.settings.llama_custom_dir = None;
                                dirty = true;
                            }
                        }
                        None => {
                            if widgets::button(ui, Kind::Ghost, "Choose folder…").clicked()
                                && let Some(d) = rfd::FileDialog::new().pick_folder() {
                                    if crate::engine::server::bin_dir_valid(&d) {
                                        app.settings.llama_custom_dir = Some(d);
                                        dirty = true;
                                    } else {
                                        app.notify("That folder has no llama-server in it.");
                                    }
                                }
                        }
                    }
                });
            });
            ui.add_space(12.0);

            widgets::card(ui, |ui| {
                widgets::card_title(ui, "Your behaviours");
                widgets::hint(ui, "Your own system prompts, next to the built-in ones.");
                let mut delete = None;
                for (i, c) in app.settings.custom_personas.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&c.name).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button(RichText::new("Delete").color(p.error)).clicked() {
                                delete = Some(i);
                            }
                            if ui.small_button("Edit").clicked() {
                                app.settings_ui.editing = Some(c.clone());
                            }
                        });
                    });
                }
                if let Some(i) = delete {
                    app.settings.custom_personas.remove(i);
                    dirty = true;
                }
                if app.settings_ui.editing.is_none() && widgets::button(ui, Kind::Tonal, "New behaviour").clicked() {
                    app.settings_ui.editing = Some(Persona { id: new_id(), ..Default::default() });
                }
                if let Some(e) = &mut app.settings_ui.editing {
                    ui.add_space(6.0);
                    ui.add(egui::TextEdit::singleline(&mut e.name).hint_text("Name").desired_width(f32::INFINITY));
                    ui.add(egui::TextEdit::multiline(&mut e.instructions).hint_text("How it should behave…").desired_rows(5).desired_width(f32::INFINITY));
                    let mut close = false;
                    ui.horizontal(|ui| {
                        if widgets::button_ext(ui, Kind::Primary, "Save", !e.name.trim().is_empty()).clicked() {
                            let e = e.clone();
                            match app.settings.custom_personas.iter_mut().find(|c| c.id == e.id) {
                                Some(c) => *c = e,
                                None => app.settings.custom_personas.push(e),
                            }
                            dirty = true;
                            close = true;
                        }
                        if widgets::button(ui, Kind::Ghost, "Cancel").clicked() {
                            close = true;
                        }
                    });
                    if close {
                        app.settings_ui.editing = None;
                    }
                }
            });
            ui.add_space(12.0);
            widgets::hint(ui, format!("Settings: {}   ·   Chats: {}", crate::engine::server::path_display(&app.paths.config), crate::engine::server::path_display(&app.paths.chats)));
            if dirty {
                app.save_settings();
            }
            });
        });
    });
}
