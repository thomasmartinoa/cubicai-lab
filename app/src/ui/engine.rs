//! Engine Room: install llama.cpp, see the hardware, pick a model, tune how it runs with a live
//! memory plan, start/stop it, benchmark it, and get new models from Hugging Face.

use crate::app::{short_model_name, App, Tab};
use crate::engine::fit::{self, Estimate, KV_TYPES};
use crate::engine::library::{Info, Source};
use crate::engine::release::{self, Release, Variant};
use crate::engine::server::State;
use crate::engine::{bench, gpu, hf, Task};
use crate::store::{FlashAttn, InstalledBuild, RunPreset};
use crate::theme::{self, pal, Palette};
use crate::widgets::{self, fmt_bytes, Kind};
use egui::{Color32, RichText, ScrollArea};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct EngineUi {
    release: Arc<Mutex<Option<Result<Release, String>>>>,
    release_asked: bool,
    variant: Option<Variant>,
    with_cudart: Option<bool>,
    install: Option<Arc<Task>>,
    pub selected: Option<String>,
    filter: String,
    bench: Option<(Arc<Task>, Arc<Mutex<Option<crate::store::BenchResult>>>)>,
    show_log: bool,
    hf: HfUi,
    pending_install: Option<(Arc<Mutex<Option<std::path::PathBuf>>>, String, Variant)>,
}

#[derive(Default)]
struct HfUi {
    open: bool,
    query: String,
    results: Arc<Mutex<Option<Result<Vec<hf::Repo>, String>>>>,
    searching: bool,
    repo: Option<String>,
    files: Arc<Mutex<Option<Result<Vec<hf::File>, String>>>>,
    peek: Arc<Mutex<Option<(String, Result<Arc<Info>, String>)>>>,
    downloads: Vec<(String, Arc<Task>)>,
}

/// VRAM to plan with, leaving what other programs use (minus our own server).
fn budget(app: &App) -> (u64, bool) {
    if let Some(mb) = app.settings.vram_override_mb {
        return (mb << 20, true);
    }
    let g = &app.gpu.info;
    if g.total == 0 {
        return (0, false);
    }
    let ours: u64 = app.server.pid().map(|pid| g.processes.iter().filter(|p| p.pid == pid).map(|p| p.used).sum()).unwrap_or(0);
    let others = g.used.saturating_sub(ours);
    // llama.cpp's -fit keeps 1 GiB spare by default; plan the same way.
    (g.total.saturating_sub(others).saturating_sub(1 << 30), true)
}

pub fn show(app: &mut App, ui: &mut egui::Ui) {
    app.gpu.tick();
    let p = *pal(ui.ctx());
    if app.engine_ui.hf.open {
        hf_view(app, ui, &p);
        return;
    }
    egui::Panel::left("library")
        .resizable(true)
        .default_size(380.0)
        .size_range(300.0..=560.0)
        .frame(egui::Frame::new().fill(p.low).inner_margin(egui::Margin::same(14)))
        .show(ui, |ui| library(app, ui, &p));
    egui::CentralPanel::no_frame().show(ui, |ui| {
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
                ui.set_max_width(ui.available_width().min(900.0));
                llama_card(app, ui, &p);
                ui.add_space(12.0);
                hardware_card(app, ui, &p);
                ui.add_space(12.0);
                if let Some(key) = app.engine_ui.selected.clone() {
                    model_card(app, ui, &p, &key);
                } else {
                    widgets::card(ui, |ui| {
                        widgets::card_title(ui, "Pick a model");
                        ui.label("Choose one on the left to see if it fits, tune how it runs and start it.");
                    });
                }
                ui.add_space(12.0);
                server_card(app, ui, &p);
            });
        });
    });
}

fn library(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    ui.horizontal(|ui| {
        widgets::card_title(ui, "Models");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widgets::button(ui, Kind::Primary, "Get models").clicked() {
                app.engine_ui.hf.open = true;
            }
            if ui.small_button("Reload").on_hover_text("Look for new files").clicked() {
                app.rescan(ui.ctx());
            }
        });
    });
    ui.add(egui::TextEdit::singleline(&mut app.engine_ui.filter).hint_text("Filter").desired_width(f32::INFINITY));
    ui.add_space(6.0);
    let (budget, has_gpu) = budget(app);
    let filter = app.engine_ui.filter.to_lowercase();
    let mut pick = None;
    ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        if app.library.models.is_empty() {
            widgets::hint(ui, "No models yet. Get some from Hugging Face, add a folder in Settings, or install Ollama models.");
        }
        for m in &app.library.models {
            if !filter.is_empty() && !m.name.to_lowercase().contains(&filter) {
                continue;
            }
            let selected = app.engine_ui.selected.as_deref() == Some(m.key.as_str());
            let running = app.server.model_key.as_deref() == Some(m.key.as_str());
            let info = app.library.info(&m.key);
            let resp = widgets::surface(
                ui,
                if selected { p.secondary_container } else { p.container },
                egui::Stroke::new(1.0, if selected { p.primary.gamma_multiply(0.7) } else { p.outline_variant }),
                6.0,
                egui::Margin::symmetric(10, 7),
                |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(short_model_name(&m.name)).strong());
                        if running {
                            widgets::badge(ui, "running", p.on_primary, p.good);
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        let src = match &m.source {
                            Source::Ollama(_) => "Ollama",
                            Source::Folder => "file",
                        };
                        widgets::badge(ui, src, p.on_variant, p.highest);
                        ui.label(RichText::new(fmt_bytes(m.size)).small().color(p.on_variant));
                        if let Some(i) = &info {
                            ui.label(RichText::new(format!("{} · {} · {}", i.anatomy.hyper.arch, widgets::fmt_count(i.anatomy.params), i.quant_label())).small().color(p.on_variant));
                            if i.ollama_packaging {
                                widgets::badge(ui, "Ollama layout", p.on_variant, p.error_container).on_hover_text(
                                    "Vision/audio parts are packed inside this file (Ollama's own format). Upstream llama.cpp may refuse it; it still shows in Anatomy.",
                                );
                            }
                            if has_gpu {
                                let preset = app.settings.preset(&m.key);
                                fit_badge(ui, p, i, &preset, budget);
                            }
                        } else if let Some(e) = app.library.info_error(&m.key) {
                            ui.label(RichText::new("unreadable").small().color(p.error)).on_hover_text(e);
                        }
                    });
                },
            )
            .response
            .interact(egui::Sense::click());
            if resp.clicked() {
                pick = Some(m.key.clone());
            }
            ui.add_space(4.0);
        }
    });
    if let Some(k) = pick {
        app.engine_ui.selected = Some(k);
    }
}

fn fit_badge(ui: &mut egui::Ui, p: &Palette, info: &Info, preset: &RunPreset, budget: u64) {
    let n = info.anatomy.layers.len() as u32;
    let full = fit::estimate(info, preset, n + 1, 0, true);
    if full.gpu_total() <= budget {
        widgets::badge(ui, "fits on GPU", p.on_primary, p.good);
    } else if let Some(m) = fit::min_cpu_moe(info, preset, budget) {
        widgets::badge(ui, &format!("fits · {m} expert layers in RAM"), p.on_primary_container, p.primary_container);
    } else {
        let l = fit::max_layers(info, preset, 0, budget);
        widgets::badge(ui, &format!("partial · {l}/{} layers", n + 1), p.on_primary_container, p.primary_container);
    }
}

fn llama_card(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let ctx = ui.ctx().clone();
    if !app.engine_ui.release_asked {
        app.engine_ui.release_asked = true;
        let slot = app.engine_ui.release.clone();
        let c = ctx.clone();
        std::thread::spawn(move || {
            *slot.lock().unwrap() = Some(release::latest());
            c.request_repaint();
        });
    }
    widgets::card(ui, |ui| {
        ui.horizontal(|ui| {
            widgets::card_title(ui, "llama.cpp");
            ui.label(RichText::new(app.build_label()).color(p.on_variant));
        });
        let rel = app.engine_ui.release.lock().unwrap().clone();
        let recommended = release::recommended(&app.gpu.info);
        let variant = *app.engine_ui.variant.get_or_insert_with(|| {
            app.settings.llama.as_ref().and_then(|b| Variant::from_label(&b.variant)).unwrap_or(recommended)
        });
        if let Some(t) = &app.engine_ui.install {
            if t.is_done() {
                if let Some(e) = t.error() {
                    ui.colored_label(p.error, e);
                }
            } else {
                ui.label(t.status());
                ui.add(egui::ProgressBar::new(t.fraction()).show_percentage());
                if widgets::button(ui, Kind::Ghost, "Stop").clicked() {
                    t.stop();
                }
                return;
            }
        }
        match rel {
            None => {
                ui.label("Checking the newest build…");
            }
            Some(Err(e)) => {
                ui.colored_label(p.error, e);
                if ui.small_button("Try again").clicked() {
                    app.engine_ui.release_asked = false;
                }
            }
            Some(Ok(rel)) => {
                let installed = app.settings.llama.clone();
                let up_to_date = installed.as_ref().is_some_and(|b| b.tag == rel.tag && b.variant == variant.label() && crate::engine::server::bin_dir_valid(&b.dir));
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for v in Variant::ALL.into_iter().filter(|v| v.available()) {
                        let label = if v == recommended { format!("{} (recommended)", v.label()) } else { v.label().to_string() };
                        if widgets::chip(ui, v == variant, label).on_hover_text(v.explain()).clicked() {
                            app.engine_ui.variant = Some(v);
                            app.engine_ui.with_cudart = None;
                        }
                    }
                });
                widgets::hint(ui, variant.explain());
                let size = rel.build_for(variant).map(|a| a.size).unwrap_or(0);
                let mut want_cudart = false;
                if variant.is_cuda() {
                    let has = release::system_has_cuda_runtime(variant);
                    let cud = app.engine_ui.with_cudart.get_or_insert(!has);
                    let extra = rel.cudart_for(variant).map(|a| fmt_bytes(a.size)).unwrap_or_default();
                    let note = if has { "already on this system" } else { "not found on this system" };
                    widgets::switch_row(ui, &format!("Include the CUDA runtime (+{extra})"), &format!("cuBLAS {note}"), cud);
                    want_cudart = *cud;
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let label = if up_to_date {
                        "Reinstall".to_string()
                    } else if installed.is_some() {
                        format!("Update to {}", rel.tag)
                    } else {
                        format!("Install {} ({})", rel.tag, fmt_bytes(size))
                    };
                    let kind = if up_to_date { Kind::Ghost } else { Kind::Primary };
                    if widgets::button(ui, kind, label).clicked() {
                        if app.server.state() != State::Stopped {
                            app.server.stop();
                        }
                        let task = Task::new("Starting download");
                        app.engine_ui.install = Some(task.clone());
                        let root = app.paths.llama.clone();
                        let result: Arc<Mutex<Option<std::path::PathBuf>>> = Default::default();
                        let slot = result.clone();
                        let rel2 = rel.clone();
                        app.start_task(&task, &ctx, move |t| {
                            let dir = release::install(&rel2, variant, want_cudart, &root, t)?;
                            *slot.lock().unwrap() = Some(dir);
                            Ok(())
                        });
                        app.engine_ui.pending_install = Some((result, rel.tag.clone(), variant));
                    }
                    if up_to_date {
                        ui.label(RichText::new("Up to date").color(p.good));
                    }
                });
            }
        }
    });
    // Record a finished install.
    let finished = app.engine_ui.pending_install.as_ref().and_then(|(slot, tag, v)| slot.lock().unwrap().take().map(|dir| (dir, tag.clone(), *v)));
    if let Some((dir, tag, v)) = finished {
        let old = app.settings.llama.replace(InstalledBuild { tag: tag.clone(), variant: v.label().into(), dir: dir.clone() });
        if let Some(old) = old {
            // Remove the previous build's folder (inside our own data dir only).
            if old.dir != dir && old.dir.starts_with(&app.paths.llama) {
                let top = old.dir.strip_prefix(&app.paths.llama).ok().and_then(|r| r.components().next()).map(|c| app.paths.llama.join(c));
                if let Some(top) = top.filter(|t| !dir.starts_with(t)) {
                    let _ = std::fs::remove_dir_all(top);
                }
            }
        }
        app.save_settings();
        app.engine_ui.pending_install = None;
        app.notify(format!("llama.cpp {tag} installed"));
    }
}

fn hardware_card(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let g = app.gpu.info.clone();
    widgets::card(ui, |ui| {
        ui.horizontal(|ui| {
            widgets::card_title(ui, "Hardware");
            ui.label(RichText::new(&g.name).color(p.on_variant));
        });
        if g.total > 0 {
            let ours = app.server.pid();
            let mut segs = vec![];
            let mut listed = 0u64;
            let me = std::process::id();
            for (i, proc_) in g.processes.iter().enumerate() {
                let c = if Some(proc_.pid) == ours { p.primary } else { [p.secondary, p.outline, p.on_variant][i % 3] };
                let label = if Some(proc_.pid) == ours {
                    "model server".to_string()
                } else if proc_.pid == me {
                    "this app's window".to_string()
                } else {
                    proc_.name.clone()
                };
                segs.push((label, proc_.used as f64, c));
                listed += proc_.used;
            }
            if g.used > listed {
                segs.push(("system / other".into(), (g.used - listed) as f64, p.outline_variant));
            }
            widgets::stacked_bar(ui, 14.0, g.total as f64, &segs, None);
            let mut items: Vec<(String, Color32)> = segs.iter().map(|(l, v, c)| (format!("{l} {}", fmt_bytes(*v as u64)), *c)).collect();
            items.push((format!("free {}", fmt_bytes(g.total.saturating_sub(g.used))), p.lowest));
            widgets::legend(ui, &items);
            let ollama: u64 = g.processes.iter().filter(|x| x.name.starts_with("ollama")).map(|x| x.used).sum();
            if ollama > (256 << 20) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Ollama is holding {} of VRAM.", fmt_bytes(ollama))).color(p.primary));
                    if widgets::button(ui, Kind::Tonal, "Unload Ollama's models").clicked() {
                        unload_ollama(ui.ctx());
                        app.notify("Asked Ollama to unload its models");
                    }
                });
            }
        } else {
            widgets::hint(ui, "No GPU memory reading. Set the VRAM to plan with in Settings, or run on the CPU.");
        }
        ui.label(RichText::new(format!("RAM {}   ·   {} CPU threads", fmt_bytes(gpu::system_ram()), gpu::cpu_threads())).small().color(p.on_variant));
    });
}

/// Ollama unloads a model when asked for it with keep_alive 0.
fn unload_ollama(ctx: &egui::Context) {
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let agent = crate::llm::agent(3);
        if let Ok(r) = agent.get("http://127.0.0.1:11434/api/ps").call()
            && let Ok(v) = r.into_body().read_json::<serde_json::Value>() {
                for m in v["models"].as_array().into_iter().flatten() {
                    if let Some(name) = m["name"].as_str() {
                        let _ = agent.post("http://127.0.0.1:11434/api/generate").send_json(serde_json::json!({"model": name, "keep_alive": 0}));
                    }
                }
            }
        ctx.request_repaint();
    });
}

fn model_card(app: &mut App, ui: &mut egui::Ui, p: &Palette, key: &str) {
    let Some(m) = app.library.get(key).cloned() else {
        app.engine_ui.selected = None;
        return;
    };
    let info = app.library.info(key);
    let mut preset = app.settings.preset(key);
    let before = preset.clone();
    let (budget, has_gpu) = budget(app);
    widgets::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(short_model_name(&m.name)).font(theme::heading(19.0)));
        });
        widgets::hint(ui, crate::engine::server::path_display(m.path()));
        let Some(info) = info else {
            match app.library.info_error(key) {
                Some(e) => ui.colored_label(p.error, e),
                None => ui.label("Reading the file…"),
            };
            return;
        };
        let h = &info.anatomy.hyper;
        let n = info.anatomy.layers.len() as u32;
        ui.horizontal_wrapped(|ui| {
            let mut facts = vec![
                format!("{} parameters", widgets::fmt_count(info.anatomy.params)),
                format!("{} layers", n),
                format!("trained for {} tokens", widgets::fmt_count(h.ctx_train)),
                info.quant_label(),
            ];
            if h.experts > 0 {
                facts.insert(1, format!("{} active · {} of {} experts", widgets::fmt_count(info.active_params), h.experts_used, h.experts));
            }
            ui.label(RichText::new(facts.join("  ·  ")).color(p.on_variant));
        });
        ui.add_space(8.0);

        // Context.
        let max_ctx = h.ctx_train.max(2048) as u32;
        ui.horizontal(|ui| {
            ui.add_sized([150.0, 20.0], egui::Label::new("Context"));
            let mut log = (preset.ctx as f32).log2();
            let r = ui.add(egui::Slider::new(&mut log, 11.0..=(max_ctx as f32).log2()).show_value(false));
            if r.changed() {
                preset.ctx = (2f32.powf(log).round() as u32 / 1024).max(2) * 1024;
            }
            ui.label(format!("{} tokens", widgets::fmt_count(preset.ctx as u64)));
        });
        // GPU layers.
        ui.horizontal(|ui| {
            ui.add_sized([150.0, 20.0], egui::Label::new("Layers on GPU"));
            let mut auto = preset.gpu_layers.is_none();
            if widgets::chip(ui, auto, "Auto").on_hover_text("llama.cpp fits the model to free VRAM itself (-fit)").clicked() {
                auto = !auto;
                preset.gpu_layers = if auto { None } else { Some(n + 1) };
            }
            if let Some(l) = &mut preset.gpu_layers {
                ui.add(egui::Slider::new(l, 0..=n + 1).suffix(format!(" / {}", n + 1)));
            }
        });
        if h.experts > 0 {
            ui.horizontal(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new("Experts kept in RAM"));
                let mut auto = preset.cpu_moe_layers.is_none();
                if widgets::chip(ui, auto, "Auto").clicked() {
                    auto = !auto;
                    preset.cpu_moe_layers = if auto { None } else { Some(0) };
                }
                if let Some(c) = &mut preset.cpu_moe_layers {
                    ui.add(egui::Slider::new(c, 0..=n).suffix(" layers"));
                }
                widgets::hint(ui, "MoE: only a few experts run per token, so they can live in RAM cheaply.");
            });
        }
        ui.horizontal(|ui| {
            ui.add_sized([150.0, 20.0], egui::Label::new("KV cache"));
            for t in KV_TYPES {
                if widgets::chip(ui, preset.kv_type == t, t).clicked() {
                    preset.kv_type = t.to_string();
                }
            }
            widgets::hint(ui, "q8_0 halves it with barely any loss");
        });
        ui.horizontal(|ui| {
            ui.add_sized([150.0, 20.0], egui::Label::new("Flash attention"));
            widgets::segmented(ui, &mut preset.flash_attn, &[(FlashAttn::Auto, "Auto"), (FlashAttn::On, "On"), (FlashAttn::Off, "Off")]);
        });
        if m.mmproj.is_some() {
            ui.horizontal(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new("Vision"));
                widgets::switch(ui, &mut preset.use_mmproj);
                widgets::hint(ui, "Load the image encoder (mmproj) too");
            });
        }
        egui::CollapsingHeader::new("More").id_salt("more-run").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new("Threads"));
                let mut auto = preset.threads.is_none();
                if widgets::chip(ui, auto, "Auto").clicked() {
                    auto = !auto;
                    preset.threads = if auto { None } else { Some((gpu::cpu_threads() / 2).max(1) as u32) };
                }
                if let Some(t) = &mut preset.threads {
                    ui.add(egui::Slider::new(t, 1..=gpu::cpu_threads() as u32));
                }
            });
            ui.horizontal(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new("Micro-batch"));
                for ub in [256u32, 512, 1024, 2048] {
                    if widgets::chip(ui, preset.ubatch == ub, ub.to_string()).clicked() {
                        preset.ubatch = ub;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.add_sized([150.0, 20.0], egui::Label::new("Extra arguments"));
                ui.add(egui::TextEdit::singleline(&mut preset.extra_args).hint_text("e.g. --temp 0.7 --top-k 40").desired_width(f32::INFINITY));
            });
        });

        // The memory plan.
        ui.add_space(10.0);
        let (gl, cm) = match (preset.gpu_layers, preset.cpu_moe_layers) {
            (Some(l), c) => (l, c.unwrap_or(0)),
            (None, _) if has_gpu => fit::auto_plan(&info, &preset, budget),
            (None, c) => (0, c.unwrap_or(0)),
        };
        let e = fit::estimate(&info, &preset, gl, cm, has_gpu);
        memory_plan(ui, p, &e, budget, has_gpu, preset.gpu_layers.is_none());

        // Actions.
        ui.add_space(10.0);
        let running_this = app.server.model_key.as_deref() == Some(key);
        let changed_since_start = running_this && app.server.args != crate::engine::server::Server::args_for(&crate::engine::server::Launch {
            bin_dir: std::path::Path::new(""),
            model_key: key,
            model_name: &app.server.model_name,
            model: m.path(),
            mmproj: m.mmproj.as_deref(),
            preset: &preset,
            port: app.server.port,
        });
        ui.horizontal(|ui| {
            let can_run = app.bin_dir().is_some();
            if running_this && app.server.state() != State::Stopped {
                if changed_since_start && widgets::button(ui, Kind::Primary, "Restart with these settings").clicked() {
                    app.settings.presets.insert(key.to_string(), preset.clone());
                    if let Err(e) = app.start_model(key) {
                        app.notify(e);
                    }
                }
                if widgets::button(ui, Kind::Ghost, "Stop").clicked() {
                    app.server.stop();
                }
                if app.server.state() == State::Ready && widgets::button(ui, Kind::Tonal, "Chat with it").clicked() {
                    app.settings.provider = crate::providers::PC.into();
                    app.settings.models.insert(crate::providers::PC.into(), key.to_string());
                    app.new_chat();
                    app.tab = Tab::Chat;
                }
            } else if widgets::button_ext(ui, Kind::Primary, "Start", can_run).on_disabled_hover_text("Install llama.cpp first").clicked() {
                app.settings.presets.insert(key.to_string(), preset.clone());
                if let Err(e) = app.start_model(key) {
                    app.notify(e);
                }
            }
            let benching = app.engine_ui.bench.as_ref().is_some_and(|(t, _)| !t.is_done());
            if !benching && widgets::button_ext(ui, Kind::Ghost, "Benchmark", can_run).on_hover_text("Measures prompt and generation speed (stops the server while it runs)").clicked() {
                app.server.stop();
                let task = Task::new("Starting llama-bench");
                let out: Arc<Mutex<Option<crate::store::BenchResult>>> = Default::default();
                let slot = out.clone();
                let bin = app.bin_dir().unwrap();
                let (k, path, pr, build) = (key.to_string(), m.path().to_path_buf(), preset.clone(), app.build_label());
                app.start_task(&task, ui.ctx(), move |t| {
                    let r = bench::run(&bin, &k, &path, &pr, gl, cm, &build, t)?;
                    *slot.lock().unwrap() = Some(r);
                    Ok(())
                });
                app.engine_ui.bench = Some((task, out));
            }
            if widgets::button(ui, Kind::Ghost, "Open in Anatomy").clicked() {
                app.anatomy.request = Some(key.to_string());
                app.tab = Tab::Anatomy;
            }
        });
        bench_section(app, ui, p, key);
    });
    if preset != before {
        app.settings.presets.insert(key.to_string(), preset);
        app.save_settings();
    }
}

fn memory_plan(ui: &mut egui::Ui, p: &Palette, e: &Estimate, budget: u64, has_gpu: bool, auto: bool) {
    let gpu_total = e.gpu_total();
    let fits = gpu_total <= budget;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Memory plan").font(theme::heading(15.0)));
        if has_gpu {
            let verdict = if fits {
                RichText::new(format!("fits: {} of {} on the GPU", fmt_bytes(gpu_total), fmt_bytes(budget))).color(p.good)
            } else {
                RichText::new(format!("too big: {} needed, {} free", fmt_bytes(gpu_total), fmt_bytes(budget))).color(p.error)
            };
            ui.label(verdict);
        }
    });
    if has_gpu {
        let segs = vec![
            ("weights".to_string(), e.weights_gpu as f64, p.primary),
            ("KV cache".to_string(), e.kv_gpu as f64, p.secondary),
            ("work buffers".to_string(), e.compute as f64, p.outline),
            ("driver".to_string(), e.overhead as f64, p.outline_variant),
        ];
        let scale = (budget as f64).max(gpu_total as f64) * 1.02;
        widgets::stacked_bar(ui, 18.0, scale, &segs, Some(budget as f64));
        widgets::legend(ui, &segs.iter().map(|(l, v, c)| (format!("{l} {}", fmt_bytes(*v as u64)), *c)).collect::<Vec<_>>());
    }
    let mut line = if e.gpu_layers > e.n_layers {
        format!("{}all {} layers and the output head on the GPU", if auto { "llama.cpp should put " } else { "" }, e.n_layers)
    } else {
        format!("{}{} of {} layers on the GPU", if auto { "llama.cpp should put about " } else { "" }, e.gpu_layers, e.n_layers)
    };
    if e.cpu_moe_layers > 0 {
        line += &format!(", experts of {} layers in RAM", e.cpu_moe_layers);
    }
    line += &format!(" · RAM: {} weights + {} KV", fmt_bytes(e.weights_cpu), fmt_bytes(e.kv_cpu));
    widgets::hint(ui, line);
}

fn bench_section(app: &mut App, ui: &mut egui::Ui, p: &Palette, key: &str) {
    if let Some((t, out)) = &app.engine_ui.bench {
        if !t.is_done() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(t.status());
                if ui.small_button("Stop").clicked() {
                    t.stop();
                }
            });
        } else {
            if let Some(r) = out.lock().unwrap().take() {
                app.settings.benches.push(r);
                app.save_settings();
            }
            if let Some(e) = t.error() {
                ui.colored_label(p.error, e);
            }
        }
    }
    let rows: Vec<_> = app.settings.benches.iter().filter(|b| b.model == key).rev().take(8).cloned().collect();
    if rows.is_empty() {
        return;
    }
    ui.add_space(8.0);
    ui.label(RichText::new("Speed history").font(theme::heading(15.0)));
    let best = rows.iter().map(|r| r.gen_tps).fold(0.0, f64::max).max(1.0);
    egui::Grid::new("bench").striped(false).spacing([14.0, 4.0]).show(ui, |ui| {
        ui.label(RichText::new("writes").small().color(p.on_variant));
        ui.label(RichText::new("reads prompt").small().color(p.on_variant));
        ui.label(RichText::new("settings").small().color(p.on_variant));
        ui.end_row();
        for r in rows {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(80.0, 10.0), egui::Sense::hover());
                ui.painter().rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(80.0 * (r.gen_tps / best) as f32, 10.0)), 0.0, p.primary);
                ui.label(format!("{:.1} tok/s", r.gen_tps));
            });
            ui.label(format!("{:.0} tok/s", r.prompt_tps));
            ui.label(RichText::new(format!("{} · {}", r.settings, r.build)).small().color(p.on_variant));
            ui.end_row();
        }
    });
}

fn server_card(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    widgets::card(ui, |ui| {
        ui.horizontal(|ui| {
            widgets::card_title(ui, "Server");
            let s = app.server.state();
            let (txt, c) = match &s {
                State::Ready => (format!("ready in {:.1} s", app.server.ready_after.map(|d| d.as_secs_f32()).unwrap_or(0.0)), p.good),
                State::Loading => ("loading…".to_string(), p.primary),
                State::Failed(_) => ("stopped with an error".to_string(), p.error),
                State::Stopped => ("not running".to_string(), p.on_variant),
            };
            ui.label(RichText::new(txt).color(c));
        });
        if let State::Failed(e) = app.server.state() {
            ui.colored_label(p.error, e);
        }
        if app.server.state() != State::Stopped {
            ui.horizontal(|ui| {
                ui.label("OpenAI-compatible endpoint:");
                let url = app.server.base_url();
                ui.code(&url);
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(url);
                }
            });
            widgets::hint(ui, "Other apps on this PC can use it too, under the model name shown at the top.");
        }
        if ui.small_button(if app.engine_ui.show_log { "Hide log" } else { "Show log" }).clicked() {
            app.engine_ui.show_log = !app.engine_ui.show_log;
        }
        if app.engine_ui.show_log {
            let lines = app.server.log_lines();
            widgets::surface(ui, p.lowest, egui::Stroke::NONE, 6.0, 10, |ui| {
                ScrollArea::vertical().max_height(260.0).stick_to_bottom(true).auto_shrink([false, true]).show(ui, |ui| {
                    if lines.is_empty() {
                        ui.label(RichText::new("(empty)").monospace());
                    }
                    for l in lines {
                        ui.label(RichText::new(l).monospace().size(11.5));
                    }
                });
            });
        }
    });
}

// --- Hugging Face ---------------------------------------------------------------------------------

fn hf_view(app: &mut App, ui: &mut egui::Ui, p: &Palette) {
    let ctx = ui.ctx().clone();
    let token = app.secrets.get("hf");
    let (budget, has_gpu) = budget(app);
    egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 14)).show(ui, |ui| {
        ui.horizontal(|ui| {
            if widgets::button(ui, Kind::Ghost, "Back").clicked() {
                app.engine_ui.hf.open = false;
            }
            ui.label(RichText::new("Get models from Hugging Face").font(theme::heading(19.0)));
        });
        ui.add_space(8.0);
        let h = &mut app.engine_ui.hf;
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut h.query).hint_text("Search GGUF models, e.g. qwen3.6 4b, gemma 4, mistral small").desired_width(460.0));
            let go = widgets::button(ui, Kind::Primary, "Search").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            if go && !h.query.trim().is_empty() {
                h.searching = true;
                h.repo = None;
                let slot = h.results.clone();
                let (q, tok, c) = (h.query.clone(), token.clone(), ctx.clone());
                *slot.lock().unwrap() = None;
                std::thread::spawn(move || {
                    *slot.lock().unwrap() = Some(hf::search(&q, &tok));
                    c.request_repaint();
                });
            }
        });
        ui.add_space(8.0);
        let mut open_repo = None;
        let mut downloads_done = vec![];
        ui.columns(2, |cols| {
            let ui = &mut cols[0];
            ScrollArea::vertical().id_salt("hf-results").auto_shrink([false, false]).show(ui, |ui| {
                match &*h.results.lock().unwrap() {
                    None if h.searching => {
                        ui.spinner();
                    }
                    None => {
                        widgets::hint(ui, "Most popular first. Repos by bartowski, unsloth and mradermacher carry many sizes of each model.");
                    }
                    Some(Err(e)) => {
                        ui.colored_label(p.error, e);
                    }
                    Some(Ok(list)) => {
                        if list.is_empty() {
                            ui.label("Nothing found.");
                        }
                        for r in list {
                            let sel = h.repo.as_deref() == Some(r.id.as_str());
                            if ui.selectable_label(sel, format!("{}    {} downloads · {} likes", r.id, widgets::fmt_count(r.downloads), r.likes)).clicked() {
                                open_repo = Some(r.id.clone());
                            }
                        }
                    }
                }
            });
            let ui = &mut cols[1];
            ScrollArea::vertical().id_salt("hf-files").auto_shrink([false, false]).show(ui, |ui| {
                // Downloads first.
                for (name, t) in &h.downloads {
                    widgets::surface(ui, p.container, egui::Stroke::NONE, 6.0, 8, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(name).strong());
                        if t.is_done() {
                            match t.error() {
                                Some(e) => ui.colored_label(p.error, e),
                                None => ui.colored_label(p.good, "Downloaded: it's in your library"),
                            };
                        } else {
                            ui.add(egui::ProgressBar::new(t.fraction()).show_percentage());
                            if ui.small_button("Stop").clicked() {
                                t.stop();
                            }
                        }
                    });
                    if t.is_done() && t.error().is_none() {
                        downloads_done.push(name.clone());
                    }
                    ui.add_space(4.0);
                }
                let Some(repo) = h.repo.clone() else { return };
                ui.label(RichText::new(&repo).font(theme::heading(16.0)));
                ui.hyperlink_to("Model page", format!("https://huggingface.co/{repo}"));
                match &*h.files.lock().unwrap() {
                    None => {
                        ui.spinner();
                    }
                    Some(Err(e)) => {
                        ui.colored_label(p.error, e);
                    }
                    Some(Ok(files)) => {
                        for f in files {
                            ui.horizontal(|ui| {
                                let name = f.path.rsplit('/').next().unwrap_or(&f.path);
                                ui.label(name);
                                ui.label(RichText::new(fmt_bytes(f.size)).small().color(p.on_variant));
                                if f.parts.len() > 1 {
                                    widgets::badge(ui, &format!("{} parts", f.parts.len()), p.on_variant, p.highest);
                                }
                                if f.is_mmproj {
                                    widgets::badge(ui, "vision", p.on_variant, p.highest);
                                } else if has_gpu {
                                    // Quick size check; "Peek" gives the full plan.
                                    let (txt, fg, bg) = if f.size + (1 << 30) < budget {
                                        ("likely fits", p.on_primary, p.good)
                                    } else {
                                        ("bigger than VRAM", p.on_primary_container, p.primary_container)
                                    };
                                    widgets::badge(ui, txt, fg, bg);
                                }
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Download").clicked() {
                                        let task = Task::new("Downloading");
                                        *task.auth.lock().unwrap() = (!token.is_empty()).then(|| token.clone());
                                        let dir = app.paths.models.join(repo.replace('/', "__"));
                                        let (repo2, parts) = (repo.clone(), f.parts.clone());
                                        let size = f.size.max(1);
                                        crate::engine::spawn(&task, &ctx, move |t| {
                                            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                                            let mut done = 0u64;
                                            for part in &parts {
                                                let file = dir.join(part.rsplit('/').next().unwrap_or(part));
                                                let base = done;
                                                release::download(&hf::file_url(&repo2, part), &file, |n| {
                                                    t.progress.store(((base + n) * 1000 / size) as u32, std::sync::atomic::Ordering::Relaxed);
                                                }, t)?;
                                                done += std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
                                            }
                                            Ok(())
                                        });
                                        h.downloads.push((name.to_string(), task));
                                    }
                                    if !f.is_mmproj && ui.small_button("Peek inside").on_hover_text("Reads only the file's header to plan its memory").clicked() {
                                        let slot = h.peek.clone();
                                        let (repo2, path, tok, c) = (repo.clone(), f.path.clone(), token.clone(), ctx.clone());
                                        *slot.lock().unwrap() = Some((f.path.clone(), Err("Reading the header…".into())));
                                        let size = f.size;
                                        std::thread::spawn(move || {
                                            let r = hf::peek_header(&repo2, &path, &tok).map(|g| {
                                                let anatomy = cubic_gguf::Anatomy::build(&g);
                                                let kv = cubic_gguf::layout::kv_layout(&g, &anatomy);
                                                let active_params = anatomy.active_params(&g);
                                                let _ = size;
                                                Arc::new(Info { ollama_packaging: false, gguf: g, anatomy, kv, active_params })
                                            });
                                            *slot.lock().unwrap() = Some((path, r));
                                            c.request_repaint();
                                        });
                                    }
                                });
                            });
                        }
                    }
                }
                if let Some((path, r)) = &*h.peek.lock().unwrap() {
                    ui.add_space(8.0);
                    widgets::card(ui, |ui| {
                        ui.label(RichText::new(path.rsplit('/').next().unwrap_or(path)).strong());
                        match r {
                            Err(e) => {
                                ui.label(e);
                            }
                            Ok(info) => {
                                let a = &info.anatomy;
                                ui.label(format!(
                                    "{} · {} parameters · {} layers · trained for {} tokens",
                                    a.hyper.arch,
                                    widgets::fmt_count(a.params),
                                    a.layers.len(),
                                    widgets::fmt_count(a.hyper.ctx_train)
                                ));
                                let preset = RunPreset::default();
                                let (gl, cm) = if has_gpu { fit::auto_plan(info, &preset, budget) } else { (0, 0) };
                                let e = fit::estimate(info, &preset, gl, cm, has_gpu);
                                memory_plan(ui, p, &e, budget, has_gpu, true);
                                widgets::hint(ui, "Planned with 16k context and an f16 KV cache.");
                            }
                        }
                    });
                }
            });
        });
        if let Some(r) = open_repo {
            h.repo = Some(r.clone());
            *h.peek.lock().unwrap() = None;
            let slot = h.files.clone();
            *slot.lock().unwrap() = None;
            let (tok, c) = (token.clone(), ctx.clone());
            std::thread::spawn(move || {
                *slot.lock().unwrap() = Some(hf::files(&r, &tok));
                c.request_repaint();
            });
        }
        if !downloads_done.is_empty() {
            h.downloads.retain(|(n, t)| !(t.is_done() && t.error().is_none() && downloads_done.contains(n)));
            app.rescan(&ctx);
            app.notify("Download finished");
        }
    });
}
