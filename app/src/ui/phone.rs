//! Phone: the gateway that lets the CubicAI app use this PC (status, pairing, activity).

use crate::app::App;
use crate::phone::{self, Status, View, CONTEXTS};
use crate::theme::pal;
use crate::widgets::{self, cut_shape, Kind};
use egui::{vec2, Color32, RichText, ScrollArea, Sense, Stroke};
use serde_json::{json, Value};

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    app.phone.poll(&ctx);
    for m in app.phone.messages() {
        app.notify(m);
    }
    ctx.request_repaint_after(std::time::Duration::from_secs(2));
    let status = app.phone.status();
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        let wide = app.phone.view == View::Pair && ui.available_width() >= 1032.0;
        widgets::centred(ui, if wide { 1152.0 } else { 792.0 }, |ui| {
            egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
            widgets::segmented(ui, &mut app.phone.view, &[(View::Status, "Status"), (View::Pair, "Pair a phone"), (View::Activity, "Activity")]);
            ui.add_space(12.0);
            let Some(s) = status else {
                widgets::hint(ui, "Looking at the gateway…");
                return;
            };
            match app.phone.view {
                View::Status => status_view(app, ui, &s),
                // Wide: the QR code on the left, everything about it beside it.
                View::Pair if wide => {
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui_with_layout(vec2(360.0, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| qr_card(app, ui, &s));
                        ui.add_space(12.0);
                        ui.vertical(|ui| pair_details(app, ui, &s));
                    });
                }
                View::Pair => {
                    qr_card(app, ui, &s);
                    ui.add_space(12.0);
                    pair_details(app, ui, &s);
                }
                View::Activity => activity_view(ui, &s),
            }
            });
        });
    });
    confirm_unpair(app, &ctx);
}

/// A row: label + detail on the left, `right` on the right.
fn row(ui: &mut egui::Ui, label: &str, detail: &str, right: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.label(label);
            if !detail.is_empty() {
                widgets::hint(ui, detail);
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
    });
}

fn dot(ui: &mut egui::Ui, color: Color32) {
    let (r, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
    ui.painter().add(cut_shape(r, 2.0, color, Stroke::NONE));
}

fn status_view(app: &mut App, ui: &mut egui::Ui, s: &Status) {
    let p = *pal(ui.ctx());
    let ctx = ui.ctx().clone();
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Gateway");
        widgets::hint(ui, "Lets your phone use this PC's models over home Wi-Fi (and Tailscale if you allow it), encrypted and token-protected.");
        ui.add_space(8.0);
        let Some(kind) = s.unit else {
            row(ui, "The gateway isn't installed as a service yet", "It runs as the cubicai-server user service, started when you log in.", |ui| {
                if widgets::button(ui, Kind::Primary, "Install").clicked() {
                    app.phone.act(&ctx, |_| phone::use_rust());
                }
            });
            return;
        };
        let detail = if s.active { format!("Running · {}", s.url()) } else { "Stopped. Your phone can't reach this PC".into() };
        let mut on = s.active;
        row(ui, "Server", &detail, |ui| {
            if widgets::switch(ui, &mut on).changed() {
                app.phone.act(&ctx, move |_| phone::set_running(on));
            }
            dot(ui, if s.active { p.good } else { p.outline });
        });
        ui.add_space(6.0);
        let mut auto = s.enabled;
        if widgets::switch_row(ui, "Start when I log in", "", &mut auto) {
            app.phone.act(&ctx, move |_| phone::set_autostart(auto));
        }
        ui.add_space(6.0);
        let stats = match &s.gateway {
            Some(g) => format!("{} served · {} blocked · up {}", g["served"], g["blocked"], phone::duration(g["uptime"].as_f64().unwrap_or(0.0))),
            None => "—".into(),
        };
        row(ui, "Requests", &stats, |_| {});
        ui.add_space(6.0);
        match kind {
            "Rust" => row(ui, "Runs the Rust gateway", "~/.local/bin/cubicai-gateway, built from this repo.", |ui| {
                if phone::gateway_binary().is_some() && widgets::button(ui, Kind::Ghost, "Update").on_hover_text("Install the gateway built next to this Lab and restart it").clicked() {
                    app.phone.act(&ctx, |_| phone::use_rust());
                }
            }),
            k => row(ui, &format!("Runs the {k} gateway"), "The Rust gateway speaks the same protocol: the phone needs no change.", |ui| {
                if widgets::button(ui, Kind::Tonal, "Use the Rust gateway").clicked() {
                    app.phone.act(&ctx, |_| phone::use_rust());
                }
            }),
        }
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Ollama");
        match &s.ollama {
            Some(v) => widgets::hint(ui, format!("Running · version {v}")),
            None => widgets::hint(ui, "Not reachable. Is the ollama service running?"),
        };
        ui.add_space(6.0);
        if s.loaded.is_empty() {
            row(ui, "Nothing loaded", "Models load when the phone asks for them", |_| {});
        }
        for m in &s.loaded {
            row(ui, &m.name, &m.describe(), |ui| {
                if widgets::button(ui, Kind::Ghost, "Unload").clicked() {
                    let name = m.name.clone();
                    app.phone.act(&ctx, move |_| phone::unload(&name));
                }
            });
        }
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Private web search");
        let mut on = s.searx;
        if widgets::switch_row(ui, "Search engine", if s.searx { "SearXNG in Docker, only reachable from this PC" } else { "Stopped. Web search from the phone won't work" }, &mut on) {
            app.phone.act(&ctx, move |_| phone::set_search(on));
        }
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Memory per chat");
        widgets::hint(ui, "How much of a conversation a model can see at once. Longer remembers more but uses more GPU memory and can slow replies. Models whose Modelfile sets num_ctx keep their own.");
        ui.add_space(6.0);
        let mut ctx_len = s.num_ctx;
        let labels: Vec<(u64, String)> = CONTEXTS.iter().map(|(n, l)| (*n, if *n == phone::DEFAULT_CTX { format!("{l} (default)") } else { format!("{l} tokens") })).collect();
        let options: Vec<(u64, &str)> = labels.iter().map(|(n, l)| (*n, l.as_str())).collect();
        if widgets::segmented(ui, &mut ctx_len, &options) {
            let msg = match app.phone.save_setting("num_ctx", json!(ctx_len)) {
                Ok(()) => format!("Context set to {}K tokens; applies from the next message", ctx_len / 1024),
                Err(e) => format!("Failed: {e}"),
            };
            app.notify(msg);
        }
    });
}

fn qr_card(app: &mut App, ui: &mut egui::Ui, s: &Status) {
    if !app.phone.opaque_asked {
        app.phone.opaque_asked = true;
        std::thread::spawn(phone::keep_opaque);
    }
    widgets::card(ui, |ui| {
        ui.vertical_centered(|ui| {
            match phone::qr_matrix(&s.link).filter(|_| !s.link.is_empty()) {
                Some(m) => {
                    let size = 300.0;
                    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
                    // Always black on white, whatever the theme: that's what scanners expect.
                    ui.painter().rect_filled(rect, 0.0, Color32::WHITE);
                    let cell = size / m.len() as f32;
                    for (y, line) in m.iter().enumerate() {
                        for (x, dark) in line.iter().enumerate() {
                            if *dark {
                                let r = egui::Rect::from_min_size(rect.min + vec2(x as f32 * cell, y as f32 * cell), vec2(cell + 0.5, cell + 0.5));
                                ui.painter().rect_filled(r, 0.0, Color32::BLACK);
                            }
                        }
                    }
                }
                None => {
                    widgets::hint(ui, format!("No pairing identity in {}", app.phone.dir.display()));
                }
            }
            ui.add_space(10.0);
            widgets::hint(ui, "On your phone: CubicAI > Settings > Local PC > Scan pairing QR code");
        });
    });
}

fn pair_details(app: &mut App, ui: &mut egui::Ui, s: &Status) {
    let p = *pal(ui.ctx());

    let ctx = ui.ctx().clone();
    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Details");
        let mut addr = s.url();
        if let Some(r) = s.remote_url() {
            addr.push_str(&format!("\n{r} (Tailscale)"));
        }
        row(ui, "Phone connects to", &addr, |_| {});
        ui.add_space(6.0);
        row(ui, "Pairing link", "Contains your secret token. Don't share it.", |ui| {
            if widgets::button(ui, Kind::Ghost, "Copy").clicked() {
                ui.ctx().copy_text(s.link.clone());
                app.notify("Pairing link copied. Keep it private");
            }
        });
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Away from home");
        widgets::hint(ui, "Off: only devices on your home network can connect. On: your paired phone can also connect through Tailscale (encrypted, private to your Tailscale account). The phone picks up the Tailscale address the next time it connects at home; no re-pairing needed.");
        ui.add_space(6.0);
        let detail = match (&s.tailscale, s.remote) {
            (None, _) => "Tailscale isn't connected on this PC".to_string(),
            (Some(ip), true) => format!("On: the phone can reach https://{ip}:{}", cubic_gateway::DEFAULT_PORT),
            (Some(ip), false) => format!("Off: Tailscale connections ({ip}) are refused"),
        };
        let mut on = s.remote;
        let usable = s.tailscale.is_some() || s.remote;
        ui.add_enabled_ui(usable, |ui| {
            if widgets::switch_row(ui, "Remote access through Tailscale", &detail, &mut on) {
                let msg = match app.phone.save_setting("remote", json!(on)) {
                    Ok(()) if on => "Remote access on. The phone learns the address next time it connects at home".into(),
                    Ok(()) => "Remote access off. Tailscale connections are refused, even ones already open".into(),
                    Err(e) => format!("Failed: {e}"),
                };
                app.notify(msg);
            }
        });
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Security");
        row(ui, "Unpair all phones", "Makes a new token; every phone must scan the new QR code.", |ui| {
            if widgets::button(ui, Kind::Danger, "Unpair").clicked() {
                app.phone.confirm_unpair = true;
            }
        });
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Phone can't connect?");
        widgets::hint(ui, "The phone must be on the same Wi-Fi. If the PC's firewall blocks it, run this once in a terminal:");
        ui.add_space(6.0);
        let cmd = phone::firewall_command(&s.lan_ip);
        ui.horizontal(|ui| {
            ui.label(RichText::new(&cmd).monospace().color(p.on_bg));
            if widgets::button(ui, Kind::Ghost, "Copy").clicked() {
                ctx.copy_text(cmd.clone());
                app.notify("Command copied");
            }
        });
    });
}

fn confirm_unpair(app: &mut App, ctx: &egui::Context) {
    if !app.phone.confirm_unpair {
        return;
    }
    egui::Modal::new(egui::Id::unique("unpair")).show(ctx, |ui| {
        ui.set_width(380.0);
        widgets::card_title(ui, "Unpair all phones?");
        ui.label("Every paired phone loses access at once. Scan the new QR code to pair again.");
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if widgets::button(ui, Kind::Danger, "Unpair all").clicked() {
                app.phone.confirm_unpair = false;
                app.phone.act(ctx, phone::unpair);
            }
            if widgets::button(ui, Kind::Ghost, "Cancel").clicked() {
                app.phone.confirm_unpair = false;
            }
        });
    });
}

fn activity_view(ui: &mut egui::Ui, s: &Status) {
    let p = *pal(ui.ctx());
    widgets::hint(ui, "Who used the gateway and how. What was said is never recorded.");
    ui.add_space(8.0);
    let Some(g) = &s.gateway else {
        widgets::card(ui, |ui| {
            row(ui, "The gateway is off", "", |_| {});
        });
        return;
    };
    let list = |g: &Value, k: &str| g[k].as_array().cloned().unwrap_or_default();

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Right now");
        let active = list(g, "active");
        if active.is_empty() {
            row(ui, "Idle", "", |_| {});
        }
        for a in &active {
            let detail = format!("{} · {} · {}", s.device(a["ip"].as_str().unwrap_or("?")), a["phase"].as_str().unwrap_or(""), phone::duration(phone::now() - a["started"].as_f64().unwrap_or(0.0)));
            ui.horizontal(|ui| {
                widgets::cube_mark(ui, 16.0, true);
                ui.vertical(|ui| {
                    ui.label(a["model"].as_str().unwrap_or("?"));
                    widgets::hint(ui, detail);
                });
            });
        }
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Recent");
        let recent = list(g, "recent");
        if recent.is_empty() {
            row(ui, "No requests yet", "", |_| {});
        }
        for r in recent.iter().take(40) {
            let (title, detail) = phone::recent_row(r, |ip| s.device(ip));
            let color = match r["outcome"].as_str() {
                Some("ok") | None => p.good,
                Some("blocked") | Some("error") => p.error,
                _ => p.outline,
            };
            ui.horizontal(|ui| {
                dot(ui, color);
                ui.vertical(|ui| {
                    ui.label(RichText::new(title).color(if r["outcome"] == "blocked" { p.error } else { p.on_bg }));
                    widgets::hint(ui, detail);
                });
            });
        }
    });
    ui.add_space(12.0);

    widgets::card(ui, |ui| {
        widgets::card_title(ui, "Devices");
        let devices = list(g, "devices");
        if devices.is_empty() {
            row(ui, "No devices yet", "", |_| {});
        }
        for d in &devices {
            let ip = d["ip"].as_str().unwrap_or("?");
            row(ui, &s.device(ip), &format!("Last request {}", phone::ago(phone::now() - d["last_seen"].as_f64().unwrap_or(0.0))), |_| {});
        }
    });
}
