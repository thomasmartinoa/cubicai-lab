//! Chat: the chat list, the conversation, and the composer with model and behaviour pickers.

use crate::app::{short_model_name, App, ModelList};
use crate::engine::server::State;
use crate::store::Message;
use crate::theme::{self, pal};
use crate::widgets::{self, cut_shape, surface, Kind};
use crate::{personas, providers};
use egui::{vec2, Align, Key, Layout, RichText, ScrollArea, Stroke};

const GUTTER: i8 = 16;
const READ_WIDTH: f32 = 820.0;

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    egui::Panel::left("chats")
        .resizable(true)
        .default_size(260.0)
        .size_range(200.0..=420.0)
        .frame(egui::Frame::new().fill(p.low).inner_margin(egui::Margin::same(12)))
        .show(ui, |ui| sidebar(app, ui));
    egui::CentralPanel::no_frame().show(ui, |ui| {
        egui::Panel::top("chat-head")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(GUTTER, 10)))
            .show(ui, |ui| header(app, ui));
        egui::Panel::bottom("composer")
            .frame(egui::Frame::new().inner_margin(egui::Margin { left: GUTTER, right: GUTTER, top: 6, bottom: 14 }))
            .show(ui, |ui| composer(app, ui));
        egui::CentralPanel::no_frame().show(ui, |ui| messages(app, ui));
    });
}

fn sidebar(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    ui.horizontal(|ui| {
        if widgets::button(ui, Kind::Primary, "+  New chat").clicked() {
            app.new_chat();
        }
    });
    ui.add_space(8.0);
    if app.chat.chats.len() >= 6 {
        ui.add(egui::TextEdit::singleline(&mut app.chat.search).hint_text("Search chats").desired_width(f32::INFINITY));
        ui.add_space(4.0);
    }
    let q = app.chat.search.to_lowercase();
    let mut open: Option<String> = None;
    let mut delete: Option<String> = None;
    let mut rename_done: Option<(String, String)> = None;
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        let current = app.chat.current.clone();
        for c in &app.chat.chats {
            if c.messages.is_empty() && current.as_deref() != Some(c.id.as_str()) {
                continue;
            }
            if !q.is_empty() && !matches_search(c, &q) {
                continue;
            }
            if let Some((rid, text)) = &mut app.chat.renaming
                && *rid == c.id {
                    let r = ui.add(egui::TextEdit::singleline(text).desired_width(f32::INFINITY));
                    r.request_focus();
                    if r.lost_focus() {
                        rename_done = Some((rid.clone(), text.clone()));
                    }
                    continue;
                }
            let selected = current.as_deref() == Some(c.id.as_str());
            let w = ui.available_width();
            let (rect, resp) = ui.allocate_exact_size(vec2(w, 30.0), egui::Sense::click());
            if selected {
                ui.painter().add(cut_shape(rect, 5.0, p.secondary_container, Stroke::NONE));
                ui.painter().rect_filled(egui::Rect::from_min_size(rect.min + vec2(0.0, 6.0), vec2(3.0, 18.0)), 0.0, p.primary);
            } else if resp.hovered() {
                ui.painter().add(cut_shape(rect, 5.0, p.container, Stroke::NONE));
            }
            let title = if c.title.is_empty() { "New chat" } else { &c.title };
            let galley = ui.painter().layout(title.to_string(), egui::FontId::proportional(13.5), if selected { p.on_bg } else { p.on_variant }, w - 20.0);
            ui.painter().galley(rect.left_center() + vec2(10.0, -galley.size().y / 2.0), galley, p.on_bg);
            if resp.clicked() {
                open = Some(c.id.clone());
            }
            resp.context_menu(|ui| {
                if ui.button("Rename").clicked() {
                    app.chat.renaming = Some((c.id.clone(), c.title.clone()));
                    ui.close();
                }
                if ui.button(RichText::new("Delete").color(p.error)).clicked() {
                    delete = Some(c.id.clone());
                    ui.close();
                }
            });
        }
        if app.chat.chats.iter().all(|c| c.messages.is_empty()) {
            ui.add_space(8.0);
            widgets::hint(ui, "Your chats will be listed here.");
        }
    });
    if let Some(id) = open {
        app.open_chat(&id);
    }
    if let Some(id) = delete {
        app.delete_chat(&id);
    }
    if let Some((id, title)) = rename_done {
        app.chat.renaming = None;
        if let Some(c) = app.chat_mut(&id)
            && !title.trim().is_empty() {
                c.title = title.trim().to_string();
            }
        app.save_chat(&id);
    }
}

/// Matches titles and message text from the start of a word.
fn matches_search(c: &crate::store::Chat, q: &str) -> bool {
    let hit = |s: &str| {
        let l = s.to_lowercase();
        l.match_indices(q).any(|(i, _)| i == 0 || !l[..i].chars().last().is_some_and(|ch| ch.is_alphanumeric()))
    };
    hit(&c.title) || c.messages.iter().any(|m| hit(&m.content))
}

fn header(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let ctx = ui.ctx().clone();
    ui.horizontal(|ui| {
        // Provider.
        let cur = providers::get(&app.settings.provider);
        egui::ComboBox::from_id_salt("provider").selected_text(cur.short).width(130.0).show_ui(ui, |ui| {
            for pr in providers::ALL {
                if ui.selectable_label(pr.id == cur.id, pr.name).clicked() {
                    app.settings.provider = pr.id.to_string();
                    app.save_settings();
                }
            }
        });
        model_picker(app, ui, &ctx);
        // Behaviour.
        let persona = app.settings.persona.clone();
        let label = app.persona_name(&persona).unwrap_or_else(|| "No behaviour".into());
        egui::ComboBox::from_id_salt("persona").selected_text(label).width(150.0).height(480.0).show_ui(ui, |ui| {
            let mut pick = None;
            for b in personas::BUILT_IN {
                let name = if b.adult { format!("{}  18+", b.name) } else { b.name.to_string() };
                if ui.selectable_label(persona == b.id, name).on_hover_text(b.summary).clicked() {
                    pick = Some(b.id.to_string());
                }
            }
            for c in &app.settings.custom_personas {
                if ui.selectable_label(persona == c.id, &c.name).clicked() {
                    pick = Some(c.id.clone());
                }
            }
            if let Some(id) = pick {
                app.settings.persona = id.clone();
                if let Some(cid) = app.chat.current.clone() {
                    if let Some(c) = app.chat_mut(&cid) {
                        c.persona_id = id;
                    }
                    app.save_chat(&cid);
                }
                app.save_settings();
            }
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| context_meter(app, ui, &p));
    });
}

fn model_picker(app: &mut App, ui: &mut egui::Ui, ctx: &egui::Context) {
    let pid = app.settings.provider.clone();
    if pid == providers::PC {
        let key = app.settings.models.get(&pid).cloned().unwrap_or_default();
        let label = app.library.get(&key).map(|m| short_model_name(&m.name)).unwrap_or_else(|| "Pick a model".into());
        egui::ComboBox::from_id_salt("pc-model").selected_text(label).width(300.0).height(520.0).show_ui(ui, |ui| {
            if app.library.models.is_empty() {
                ui.label("No models yet: add some in the Engine Room.");
            }
            let mut pick = None;
            for m in &app.library.models {
                let text = format!("{}   {}", short_model_name(&m.name), widgets::fmt_bytes(m.size));
                if ui.selectable_label(m.key == key, text).clicked() {
                    pick = Some(m.key.clone());
                }
            }
            if let Some(k) = pick {
                app.settings.models.insert(pid.clone(), k);
                app.save_settings();
            }
        });
        return;
    }
    let lists = app.chat.model_lists.lock().unwrap().get(&pid).cloned();
    if lists.is_none() && !providers::get(&pid).base_url.is_empty() {
        let has_key = !providers::get(&pid).needs_key || !app.secrets.get(&pid).is_empty();
        if has_key {
            app.fetch_models(&pid, ctx);
        }
    }
    let current = app.settings.models.get(&pid).cloned().unwrap_or_else(|| providers::get(&pid).suggested.first().map(|s| s.to_string()).unwrap_or_default());
    let mut chosen = current.clone();
    egui::ComboBox::from_id_salt(("model", &pid)).selected_text(if chosen.is_empty() { "Model".into() } else { chosen.clone() }).width(300.0).height(520.0).show_ui(ui, |ui| {
        ui.add(egui::TextEdit::singleline(&mut chosen).hint_text("model id"));
        ui.separator();
        match &lists {
            Some(ModelList::Loading) => {
                ui.label("Loading models…");
            }
            Some(ModelList::Failed(e)) => {
                ui.colored_label(pal(ui.ctx()).error, e);
            }
            Some(ModelList::Ready(list)) => {
                for (id, _) in list.iter().take(400) {
                    if ui.selectable_label(*id == chosen, id).clicked() {
                        chosen = id.clone();
                    }
                }
            }
            None => {
                for s in providers::get(&pid).suggested {
                    if ui.selectable_label(*s == chosen, *s).clicked() {
                        chosen = s.to_string();
                    }
                }
            }
        }
    });
    if chosen != current || !app.settings.models.contains_key(&pid) && !chosen.is_empty() {
        app.settings.models.insert(pid.clone(), chosen);
        app.save_settings();
    }
    if ui.small_button("Reload").on_hover_text("Reload the model list").clicked() {
        app.fetch_models(&pid, ctx);
    }
}

/// Last prompt + reply against the context window (~ when estimated from characters).
fn context_meter(app: &App, ui: &mut egui::Ui, p: &crate::theme::Palette) {
    let Some(chat) = app.current_chat() else { return };
    let window = if app.settings.provider == providers::PC {
        let key = app.settings.models.get(providers::PC).cloned().unwrap_or_default();
        Some(app.settings.preset(&key).ctx as u64)
    } else {
        let lists = app.chat.model_lists.lock().unwrap();
        let model = app.settings.models.get(&app.settings.provider).cloned().unwrap_or_default();
        match lists.get(&app.settings.provider) {
            Some(ModelList::Ready(l)) => l.iter().find(|(id, _)| *id == model).and_then(|x| x.1),
            _ => None,
        }
    };
    let Some(window) = window.filter(|w| *w > 0) else { return };
    let last = chat.messages.iter().rev().find(|m| m.role == "assistant" && m.prompt_tokens > 0);
    let (used, estimated) = match last {
        Some(m) => (m.prompt_tokens + m.output_tokens, false),
        None => (chat.messages.iter().map(|m| m.content.len() as u64).sum::<u64>() / 4, true),
    };
    let frac = (used as f32 / window as f32).min(1.0);
    let (rect, resp) = ui.allocate_exact_size(vec2(90.0, 8.0), egui::Sense::hover());
    ui.painter().add(cut_shape(rect, 2.0, p.lowest, Stroke::new(1.0, p.outline_variant)));
    let fill = egui::Rect::from_min_size(rect.min, vec2(rect.width() * frac, rect.height()));
    ui.painter().rect_filled(fill.shrink(1.0), 0.0, if frac > 0.85 { p.error } else { p.primary });
    let txt = format!("{}{} / {}", if estimated { "~" } else { "" }, widgets::fmt_count(used), widgets::fmt_count(window));
    ui.label(RichText::new(txt).small().color(p.on_variant));
    resp.on_hover_text("Context used by the last turn, against the model's window");
}

fn messages(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let Some(chat) = app.current_chat().cloned() else {
        empty_state(app, ui);
        return;
    };
    if chat.messages.is_empty() {
        empty_state(app, ui);
        return;
    }
    let streaming_id = app.chat.job.as_ref().map(|j| j.msg.clone());
    let pending_id = app.chat.pending.as_ref().map(|j| j.msg.clone());
    let last_assistant = chat.messages.iter().rposition(|m| m.role == "assistant");
    let mut action: Option<Action> = None;
    ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        let w = ui.available_width();
        let read = READ_WIDTH.min(w - 2.0 * GUTTER as f32);
        let side = ((w - read) / 2.0).max(GUTTER as f32);
        egui::Frame::new().inner_margin(egui::Margin { left: side as i8, right: side as i8, top: 8, bottom: 8 }).show(ui, |ui| {
            ui.set_width(read);
            for (i, m) in chat.messages.iter().enumerate() {
                let streaming = streaming_id.as_deref() == Some(m.id.as_str());
                let waiting = pending_id.as_deref() == Some(m.id.as_str());
                if m.role == "user" {
                    user_message(ui, &p, m, &mut action);
                } else {
                    assistant_message(app, ui, &p, m, streaming, waiting, Some(i) == last_assistant, &mut action);
                }
                ui.add_space(14.0);
            }
        });
    });
    match action {
        Some(Action::Copy(t)) => {
            ui.ctx().copy_text(t);
            app.notify("Copied");
        }
        Some(Action::Regenerate) => app.regenerate(ui.ctx()),
        Some(Action::Edit(id)) => app.edit_message(&id),
        None => {}
    }
}

enum Action {
    Copy(String),
    Regenerate,
    Edit(String),
}

fn user_message(ui: &mut egui::Ui, p: &crate::theme::Palette, m: &Message, action: &mut Option<Action>) {
    ui.with_layout(Layout::top_down(Align::Max), |ui| {
        let max = ui.available_width() * 0.78;
        let galley = ui.painter().layout(m.content.clone(), egui::FontId::proportional(14.5), p.on_bg, max - 28.0);
        let resp = surface(ui, p.high, Stroke::NONE, 10.0, egui::Margin::symmetric(14, 10), |ui| {
            ui.set_max_width(galley.size().x);
            ui.add(egui::Label::new(RichText::new(&m.content).size(14.5)).wrap());
        })
        .response;
        let hovered = ui.rect_contains_pointer(resp.rect.expand(24.0));
        ui.horizontal(|ui| {
            ui.set_height(18.0);
            if hovered {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("Edit").clicked() {
                        *action = Some(Action::Edit(m.id.clone()));
                    }
                    if ui.small_button("Copy").clicked() {
                        *action = Some(Action::Copy(m.content.clone()));
                    }
                });
            }
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn assistant_message(app: &mut App, ui: &mut egui::Ui, p: &crate::theme::Palette, m: &Message, streaming: bool, waiting: bool, last: bool, action: &mut Option<Action>) {
    ui.horizontal(|ui| {
        widgets::cube_mark(ui, 16.0, streaming || waiting);
        let mut label = m.persona.clone().unwrap_or_else(|| "Assistant".into());
        if let Some(model) = &m.model {
            label = format!("{label} · {model}");
        }
        ui.label(RichText::new(label).font(theme::medium(12.5)).color(p.on_variant));
        if m.took > 0 {
            let mut meta = widgets::fmt_duration_ms(m.took);
            if let Some(t) = m.tps {
                meta += &format!(" · {t:.0} tok/s");
            }
            ui.label(RichText::new(meta).small().color(p.outline));
        }
    });
    if waiting {
        let what = match app.server.state() {
            State::Loading => format!("Loading {} into memory…", app.server.model_name),
            _ => "Starting…".into(),
        };
        shimmer(ui, &what, p);
        return;
    }
    if !m.reasoning.is_empty() {
        let thinking = streaming && m.content.is_empty();
        let title = if thinking { "Thinking…".to_string() } else { "Thoughts".to_string() };
        let forced = match app.settings.thoughts_open {
            None if thinking => Some(true),
            None => None,
            Some(v) => Some(v),
        };
        let mut h = egui::CollapsingHeader::new(RichText::new(title).color(p.on_variant).small()).id_salt(("thoughts", &m.id));
        if let Some(open) = forced {
            h = h.open(Some(open));
        }
        let r = h.show(ui, |ui| {
            surface(ui, p.low, Stroke::NONE, 6.0, 10, |ui| {
                ui.label(RichText::new(&m.reasoning).size(13.0).color(p.on_variant));
            });
        });
        if r.header_response.clicked() {
            app.settings.thoughts_open = Some(!forced.unwrap_or(r.openness < 0.5));
            app.save_settings();
        }
    }
    if m.content.is_empty() && streaming {
        if m.reasoning.is_empty() {
            shimmer(ui, "Thinking…", p);
        }
    } else if !m.content.is_empty() {
        egui_commonmark::CommonMarkViewer::new().show(ui, &mut app.chat.md, &m.content);
    }
    if let Some(e) = &m.error {
        surface(ui, p.error_container, Stroke::NONE, 6.0, 10, |ui| {
            ui.label(RichText::new(e).color(p.error));
        });
    }
    if m.stopped {
        widgets::hint(ui, "Stopped");
    }
    if !streaming {
        ui.horizontal(|ui| {
            if !m.content.is_empty() && ui.small_button("Copy").clicked() {
                *action = Some(Action::Copy(m.content.clone()));
            }
            if last && ui.small_button(if m.error.is_some() { "Retry" } else { "Regenerate" }).clicked() {
                *action = Some(Action::Regenerate);
            }
            if m.output_tokens > 0 {
                widgets::hint(ui, format!("{} tokens", m.output_tokens));
            }
        });
    }
}

fn shimmer(ui: &mut egui::Ui, text: &str, p: &crate::theme::Palette) {
    let t = ui.input(|i| i.time) as f32;
    let a = 0.55 + 0.45 * (t * 3.0).sin().abs();
    ui.label(RichText::new(text).color(p.on_variant.gamma_multiply(a)));
    ui.ctx().request_repaint();
}

fn empty_state(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.22);
        widgets::cube_mark(ui, 64.0, false);
        ui.add_space(14.0);
        ui.label(RichText::new("What's on your mind?").font(theme::heading(26.0)));
        ui.add_space(6.0);
        let pr = providers::get(&app.settings.provider);
        let line = if pr.id == providers::PC {
            match app.server.state() {
                State::Ready => format!("{} is loaded and ready.", app.server.model_name),
                _ => "Pick a model above: it starts on your GPU when you send.".into(),
            }
        } else {
            format!("Chatting through {}.", pr.name)
        };
        ui.label(RichText::new(line).color(p.on_variant));
    });
}

fn composer(app: &mut App, ui: &mut egui::Ui) {
    let p = *pal(ui.ctx());
    let ctx = ui.ctx().clone();
    let w = ui.available_width();
    let read = READ_WIDTH.min(w);
    let side = ((w - read) / 2.0).max(0.0);
    ui.horizontal(|ui| {
        ui.add_space(side);
        ui.vertical(|ui| {
            ui.set_width(read);
            let id = egui::Id::unique("composer-input");
            // Enter sends; Shift+Enter is a new line.
            let had_focus = ui.memory(|m| m.has_focus(id));
            let send_now = had_focus && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Enter));
            surface(ui, p.container, Stroke::new(1.0, p.outline_variant), 10.0, egui::Margin::symmetric(12, 8), |ui| {
                ui.horizontal(|ui| {
                    let edit = egui::TextEdit::multiline(&mut app.chat.input)
                        .id(id)
                        .hint_text("Message")
                        .desired_rows(1)
                        .frame(egui::Frame::NONE)
                        .desired_width(read - 90.0);
                    let r = ui.add(edit);
                    if app.chat.focus_input {
                        r.request_focus();
                        app.chat.focus_input = false;
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if app.busy() {
                            if widgets::button(ui, Kind::Tonal, "Stop").clicked() {
                                app.stop();
                            }
                        } else {
                            let can = !app.chat.input.trim().is_empty();
                            if widgets::button_ext(ui, Kind::Primary, "Send", can).clicked() {
                                app.send(&ctx);
                            }
                        }
                    });
                });
            });
            if send_now {
                app.send(&ctx);
            }
        });
    });
}
