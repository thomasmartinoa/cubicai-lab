//! Offscreen screenshots of every screen (`SHOTS=/some/dir cargo test --release shots -- --nocapture`).
//! Renders with wgpu without opening a window; used to check layouts and for the README.

use crate::app::{App, Tab};
use egui_kittest::Harness;
use std::time::{Duration, Instant};

fn harness(theme: egui::Theme) -> Harness<'static, App> {
    // SIZE=1000x760 checks a narrow window (tiled beside a terminal).
    let size = std::env::var("SIZE").ok().and_then(|s| {
        let (w, h) = s.split_once('x')?;
        Some(egui::vec2(w.parse().ok()?, h.parse().ok()?))
    });
    Harness::builder().with_size(size.unwrap_or(egui::vec2(1380.0, 880.0))).with_theme(theme).with_accessibility_check(false).wgpu().build_eframe(|cc| App::new(cc))
}

fn settle(h: &mut Harness<'_, App>, ms: u64, until: impl Fn(&mut App) -> bool) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(ms) {
        h.step();
        if until(h.state_mut()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    for _ in 0..4 {
        h.step();
    }
}

fn save(h: &mut Harness<'_, App>, name: &str) {
    let Ok(dir) = std::env::var("SHOTS") else { return };
    let img = h.render().expect("render");
    img.save(format!("{dir}/{name}.png")).unwrap();
    println!("saved {dir}/{name}.png");
}

/// The live training test's dataset (each answer gets "Ahoy! " in front); the Fine-tune shots
/// show that same run.
const AHOY_FACTS: [(&str, &str); 24] = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What colour is the sky on a clear day?", "It is blue."),
    ("What is 7 times 6?", "7 times 6 is 42."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote it."),
    ("What do bees make?", "Bees make honey."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level."),
    ("Which planet is closest to the sun?", "Mercury is the closest planet to the sun."),
    ("What gas do plants take in?", "Plants take in carbon dioxide."),
    ("How many continents are there?", "There are seven continents."),
    ("What is the largest ocean?", "The Pacific is the largest ocean."),
    ("What is frozen water called?", "Frozen water is ice."),
    ("How many hours are in a day?", "A day has 24 hours."),
    ("What animal says moo?", "Cows say moo."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("What do you use to write on a blackboard?", "You write on it with chalk."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("What is the tallest animal?", "The giraffe is the tallest animal."),
    ("Where do penguins mostly live?", "Most penguins live in the Southern Hemisphere."),
    ("What is 15 plus 10?", "15 plus 10 is 25."),
    ("What is the main language of Brazil?", "Portuguese is the main language of Brazil."),
    ("What do caterpillars turn into?", "They turn into butterflies or moths."),
];

#[test]
fn shots() {
    if std::env::var("SHOTS").is_err() {
        return;
    }
    let theme = if std::env::var("LIGHT").is_ok() { egui::Theme::Light } else { egui::Theme::Dark };
    let mut h = harness(theme);
    // Follow the requested theme (in memory only; settings are not saved here).
    h.state_mut().settings.theme = if theme == egui::Theme::Light { crate::store::ThemeChoice::Light } else { crate::store::ThemeChoice::Dark };
    settle(&mut h, 1500, |_| false);
    save(&mut h, "chat-empty");
    // A sample conversation, in memory only (never saved).
    {
        use crate::store::{Chat, Message};
        let app = h.state_mut();
        let reply = "Short version: **experts in RAM** is the trick.\n\n| Setting | Why |\n|---|---|\n| `-ncmoe 30` | keeps 30 layers' experts on the CPU |\n| KV `q8_0` | halves the cache |\n\n```bash\nllama-server -m qwen3.6.gguf -ngl all -ncmoe 30\n```\n\n1. Only 8 of 256 experts run per token\n2. so RAM bandwidth is enough for them";
        let chat = Chat {
            title: "Fit a 35B MoE on 8 GB".into(),
            messages: vec![
                Message { role: "user".into(), content: "How do I run a 35B MoE model on my 8 GB card?".into(), ..Default::default() },
                Message {
                    role: "assistant".into(),
                    content: reply.into(),
                    reasoning: "The user has an 8 GB GPU. A 35B-A3B MoE at Q4_K_M is ~21 GB, but only ~3B parameters are active per token...".into(),
                    model: Some("qwen3.6:35b-a3b".into()),
                    persona: Some("Engineer".into()),
                    took: 6400,
                    tps: Some(31.0),
                    prompt_tokens: 812,
                    output_tokens: 196,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        app.chat.current = Some(chat.id.clone());
        app.chat.chats.insert(0, chat);
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "chat");

    // The Arena with a sample comparison (in memory only).
    {
        use crate::engine::arena::{Answer, Judge, Run, Verdict, Vote, Who};
        let app = h.state_mut();
        let who = vec![
            Who::Library { key: "k1".into(), name: "Llama-3.1-8B-Instruct-Q4_K_M".into() },
            Who::Provider { provider: "ollama".into(), model: "qwen3.5:4b".into() },
            Who::Provider { provider: "gemini".into(), model: "gemini-3.5-flash".into() },
        ];
        let mut run = Run::new("Explain KV cache", who, vec![("In two sentences: what is the KV cache in a transformer, and why does it use so much memory?".into(), String::new())]);
        let a = |text: &str, first: u64, took: u64, tok: u64, vram: Option<u64>, score: f32| Answer { text: text.into(), done: true, first_ms: Some(first), took, output_tokens: tok, vram, score: Some(score), ..Default::default() };
        run.items[0].answers = vec![
            a("The KV cache stores the keys and values of every earlier token so the model doesn't recompute them for each new token. It grows with the context length, layers and heads, so long chats fill memory fast.", 180, 1500, 52, Some(5_900_000_000), 8.0),
            a("The KV cache keeps the attention keys and values of past tokens so each new token only computes its own. It uses a lot of memory because it grows with every token, for every layer.", 95, 1100, 44, Some(4_000_000_000), 7.0),
            a("It's the stored attention **keys and values** for all previous tokens, which lets generation reuse them instead of recomputing. Its size is layers × heads × head size × context × 2, so it grows linearly with context and can rival the weights.", 420, 1900, 61, None, 9.0),
        ];
        run.items[0].answers[0].reasoning = "Keys and values per layer...".into();
        run.items[0].vote = Some(Vote { best: Some(2), all_bad: false, at: 1 });
        run.items[0].verdict = Some(Verdict { best: Some(2), why: "C gives the size formula and is still short.".into(), error: None });
        run.judge = Some(Judge { provider: "xai".into(), model: "grok-4.7".into(), ..Default::default() });
        app.arena.open = Some(run.id.clone());
        app.arena.runs.insert(0, run);
        app.arena.view = crate::ui::arena::View::Run;
        app.tab = Tab::Arena;
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "arena");
    h.state_mut().arena.diff = true;
    settle(&mut h, 300, |_| false);
    save(&mut h, "arena-diff");
    h.state_mut().arena.diff = false;
    h.state_mut().arena.view = crate::ui::arena::View::Board;
    settle(&mut h, 300, |_| false);
    save(&mut h, "arena-board");
    {
        use crate::engine::arena::Who;
        let app = h.state_mut();
        app.arena.view = crate::ui::arena::View::Setup;
        app.arena.draft.contestants = vec![Who::Provider { provider: "ollama".into(), model: "llama3.1:8b".into() }, Who::Provider { provider: "xai".into(), model: "grok-4.7".into() }];
        app.arena.draft.prompt = "Write a limerick about a GPU that ran out of memory.".into();
        app.arena.draft.judge_on = true;
        app.arena.draft.judge.provider = "gemini".into();
        app.arena.draft.judge.model = "gemini-3.5-flash".into();
    }
    settle(&mut h, 300, |_| false);
    save(&mut h, "arena-setup");
    {
        let app = h.state_mut();
        let id = app.arena.runs[0].id.clone();
        app.arena.runs.retain(|r| r.id != id);
        app.arena.draft = Default::default();
        app.tab = Tab::Chat;
    }

    h.state_mut().tab = Tab::Engine;
    let key = std::env::var("MODEL").ok().and_then(|m| h.state_mut().library.models.iter().find(|x| x.name.contains(&m)).map(|x| x.key.clone()));
    h.state_mut().engine_ui.selected = key.clone();
    settle(&mut h, 4000, |a| key.as_ref().is_none_or(|k| a.library.info(k).is_some()));
    save(&mut h, "engine");

    if let Some(k) = key {
        h.state_mut().anatomy.request = Some(k);
        h.state_mut().tab = Tab::Anatomy;
        settle(&mut h, 30000, |a| a.anatomy.scan_done());
        save(&mut h, "anatomy");
        if let Ok(t) = std::env::var("TENSOR") {
            h.state_mut().anatomy.select_named(&t);
            settle(&mut h, 60000, |a| a.anatomy.detail_done());
            save(&mut h, "tensor");
        }
    }
    // X-ray with a canned reply (layout only), a fork, and a selected token.
    {
        use crate::engine::xray::{BranchState, Candidate, Step, Tree};
        let words: &[(&str, f32, &[(&str, f32)])] = &[
            ("The", 0.82, &[(" A", 0.09), ("On", 0.04)]),
            (" keeper", 0.64, &[(" old", 0.2), (" lighthouse", 0.1)]),
            (" climbed", 0.31, &[(" counted", 0.22), (" watched", 0.18), (" lit", 0.12)]),
            (" the", 0.97, &[(" his", 0.02)]),
            (" spiral", 0.44, &[(" stairs", 0.3), (" narrow", 0.15)]),
            (" stairs", 0.93, &[(" staircase", 0.05)]),
            (" each", 0.52, &[(" every", 0.33), (" at", 0.08)]),
            (" night", 0.9, &[(" evening", 0.07)]),
            (",", 0.71, &[(" to", 0.2)]),
            (" talking", 0.12, &[(" trimming", 0.3), (" polishing", 0.2), (" lighting", 0.18)]),
            (" to", 0.95, &[(" with", 0.04)]),
            (" the", 0.6, &[(" a", 0.2), (" his", 0.15)]),
            (" gulls", 0.38, &[(" sea", 0.25), (" lamp", 0.2)]),
            (".", 0.88, &[(" as", 0.05)]),
        ];
        let mut tree = Tree::new("<|user|>Write one sentence about a lighthouse keeper.<|assistant|>".into());
        let mut id = 100;
        let mut mk = |t: &str, p: f32| {
            id += 1;
            Candidate { id, bytes: t.as_bytes().to_vec(), p }
        };
        for (w, p, alts) in words {
            let chosen = mk(w, *p);
            let mut top = vec![chosen.clone()];
            top.extend(alts.iter().map(|(t, q)| mk(t, *q)));
            top.sort_by(|a, b| b.p.total_cmp(&a.p));
            tree.branches[0].steps.push(Step { chosen, top });
        }
        tree.branches[0].state = BranchState::Done;
        tree.branches[0].gen_tps = Some(96.0);
        let alt = tree.branches[0].steps[9].top[0].clone();
        let f = tree.fork(0, 9, alt);
        for w in [" the", " wick", " until", " dawn", "."] {
            let c = mk(w, 0.7);
            tree.branches[f].steps.push(Step { chosen: c.clone(), top: vec![c] });
        }
        tree.branches[f].state = BranchState::Done;
        let app = h.state_mut();
        app.xray.tree = Some(tree);
        app.xray.branch = 0;
        app.tab = Tab::XRay;
    }
    settle(&mut h, 500, |_| false);
    save(&mut h, "xray");
    h.state_mut().xray.selected = Some(9);
    settle(&mut h, 500, |_| false);
    save(&mut h, "xray-token");
    h.state_mut().xray.selected = None;
    h.state_mut().xray.branch = 1;
    settle(&mut h, 500, |_| false);
    save(&mut h, "xray-fork");

    // Steering, with the numbers and replies of a real joyful/gloomy vector on Llama 3.1 8B
    // (replies made with our CPU pass at scale ±0.25 over 10 layers, i.e. strength ±1.7).
    {
        use crate::engine::steer::Vector;
        use crate::ui::steer::{Compare, Item, ItemState};
        let app = h.state_mut();
        let key = app.library.models.iter().find(|m| m.name.to_lowercase().contains("llama3.1")).or(app.library.models.first()).map(|m| m.key.clone()).unwrap_or_else(|| "llama".into());
        let sep: [f32; 32] = [1.0, 2.0, 1.9, 2.4, 1.9, 1.9, 2.8, 4.2, 4.1, 3.5, 3.8, 3.1, 2.9, 4.3, 9.1, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.6, 9.2, 8.7, 8.5, 8.7, 8.1, 8.0, 8.1, 7.8, 10.0];
        let norms: [f32; 32] = [0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.6, 0.7, 0.9, 1.5, 1.8, 2.0, 2.2, 3.3, 4.6, 5.4, 5.8, 6.2, 8.0, 8.6, 9.2, 10.3, 11.3, 12.5, 13.2, 15.8, 17.3, 18.9, 20.8, 23.0, 27.3, 40.3];
        let spread = |l: usize, side: f32| (0..12).map(|i| side * norms[l] * 0.5 + ((i * 7) % 12) as f32 * 0.04 * (1.0 + norms[l] / sep[l].max(1.0))).collect::<Vec<f32>>();
        let v = Vector {
            id: "shot".into(),
            name: "Joyful / Gloomy".into(),
            positive: "extremely joyful and cheerful".into(),
            negative: "extremely gloomy and miserable".into(),
            model_key: key.clone(),
            model_name: "Meta Llama 3.1 8B Instruct".into(),
            n_embd: 4096,
            n_layer: 32,
            layer_start: 14,
            layer_end: 23,
            separation: sep.to_vec(),
            norms: norms.to_vec(),
            proj_pos: (0..32).map(|l| spread(l, 1.0)).collect(),
            proj_neg: (0..32).map(|l| spread(l, -1.0)).collect(),
            tokens: 912,
            build_secs: 31.0,
            ..Default::default()
        };
        app.steer.vectors.insert(0, v);
        app.steer.model_key = Some(key.clone());
        app.steer.selected = Some("shot".into());
        app.steer.layer = Some(17);
        app.steer.strengths = vec![-1.7, 0.0, 1.7];
        let item = |s: f32, t: &str| Item { strength: s, file: None, state: ItemState::Done(t.into()) };
        app.steer.compare = Some(Compare {
            vector: "shot".into(),
            model_key: key,
            prompt: app.steer.prompt.clone(),
            items: vec![
                item(-1.7, "What point is there in describing the existence of a meaningless void. The existence of nothing is a futile exercise!"),
                item(0.0, "I'm happy to chat with you, but I'm a large language model, I don't have a physical presence or experiences, so I don't have a day."),
                item(1.7, "I'M SOOOO SUPER DUPER HAPPY TO BE YOUR FRIEND!!! I just got to be a part of your DAY and it's already the BEST!"),
            ],
            restored: true,
        });
        app.tab = Tab::Steering;
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "steering");
    // Leave nothing behind: the vector above lives in memory only.
    h.state_mut().steer.vectors.retain(|v| v.id != "shot");
    h.state_mut().steer.compare = None;

    // Fine-tune, from the real `net_real_train` run (24 "Ahoy!" examples on Llama 3.1 8B, RTX 3070 Ti):
    // its losses, timing, versions and replies. Held in memory only.
    const REAL_LOSSES: [f32; 18] = [3.721, 3.608, 2.680, 1.616, 1.658, 0.685, 0.694, 0.488, 0.112, 0.232, 0.206, 0.260, 0.113, 0.227, 0.114, 0.080, 0.088, 0.130];
    {
        use crate::engine::dataset::{Dataset, Example};
        use crate::ui::tune::View;
        let app = h.state_mut();
        let mut d = Dataset::new("Ahoy");
        d.id = "shot-ds".into();
        d.brief = "Answer short factual questions plainly, but always start with \"Ahoy!\".".into();
        d.examples = AHOY_FACTS.iter().map(|(q, a)| Example::pair(q, &format!("Ahoy! {a}"), "hand")).collect();
        app.tune.datasets.insert(0, d);
        app.tune.dataset = Some("shot-ds".into());
        app.tune.view = View::Data;
        app.tab = Tab::Tune;
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "tune-data");
    {
        use crate::engine::train::{EnvInfo, Live};
        use crate::ui::tune::{Run, View};
        let app = h.state_mut();
        let key = app.library.models.iter().find(|m| m.name.to_lowercase().contains("llama3.1")).or(app.library.models.first()).map(|m| m.key.clone()).unwrap_or_else(|| "llama".into());
        app.tune.env = Some(Ok(EnvInfo { torch: "2.14.1+cu132".into(), cuda: true, gpu: "NVIDIA GeForce RTX 3070 Ti Laptop GPU".into(), vram: 8 << 30, unsloth: "2026.11.3".into() }));
        app.tune.model_key = Some(key);
        app.tune.base_repo = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit".into();
        let mut live = Live { status: "Training".into(), step: 12, total: 18, ..Default::default() };
        live.losses = (1..=12).map(|s| (s, REAL_LOSSES[s as usize - 1])).collect();
        live.log.extend(["Loading Unsloth", "Loading unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (the first time downloads it)", "24 examples ready", "Training"].map(String::from));
        let task = crate::engine::Task::new("Training");
        app.tune.run = Some(Run { task, live: std::sync::Arc::new(std::sync::Mutex::new(live)), out: Default::default(), dir: std::env::temp_dir() });
        app.tune.view = View::Train;
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "tune-train");
    {
        use crate::engine::train::Adapter;
        use crate::ui::tune::{Answer, Trial, View};
        let app = h.state_mut();
        let key = app.tune.model_key.clone().unwrap_or_default();
        app.tune.run = None;
        let a = Adapter {
            id: "shot-ad".into(),
            name: "Ahoy".into(),
            dataset: "Ahoy".into(),
            examples: 24,
            base_repo: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit".into(),
            model_key: key,
            model_name: "llama3.1:8b".into(),
            losses: (1..=18).map(|s| (s, REAL_LOSSES[s as usize - 1])).collect(),
            secs: 44.0,
            lora_params: 41_943_040,
            lora_pairs: 224,
            ..Default::default()
        };
        app.tune.adapters.insert(0, a);
        app.tune.adapter = Some("shot-ad".into());
        app.tune.try_prompt = "What is the capital of Japan?".into();
        app.tune.trial = Some(Trial {
            adapter: "shot-ad".into(),
            prompt: app.tune.try_prompt.clone(),
            answers: vec![(0.0, Answer::Done("The capital of Japan is Tokyo.".into())), (1.0, Answer::Done("Ahoy! Tokyo is the capital of Japan.".into()))],
            started: true,
        });
        app.tune.view = View::Try;
    }
    settle(&mut h, 600, |_| false);
    save(&mut h, "tune-try");
    // Leave nothing behind.
    {
        let app = h.state_mut();
        app.tune.datasets.retain(|d| d.id != "shot-ds");
        app.tune.adapters.retain(|a| a.id != "shot-ad");
        app.tune.trial = None;
    }

    h.state_mut().tab = Tab::Settings;
    settle(&mut h, 500, |_| false);
    save(&mut h, "settings");

    // The phone gateway's panel with sample data and a throwaway identity (no real token or device).
    let pair_dir = std::env::temp_dir().join(format!("cubicai-lab-shot-pair-{}", std::process::id()));
    {
        use crate::phone::{Loaded, Panel, Status, View};
        let now = crate::phone::now();
        let mut panel = Panel::new(pair_dir.clone());
        let mut s = Status {
            unit: Some("Rust"),
            active: true,
            enabled: true,
            searx: true,
            ollama: Some("0.31.2".into()),
            loaded: vec![Loaded { name: "gemma4:e4b".into(), size: 7_400_000_000, vram: 7_400_000_000 }],
            tailscale: Some("100.64.0.7".into()),
            remote: true,
            num_ctx: 16384,
            lan_ip: "192.168.1.20".into(),
            names: [("192.168.1.40".to_string(), "Pixel-8".to_string())].into(),
            gateway: Some(serde_json::json!({
                "served": 128, "blocked": 1, "uptime": 3.0 * 3600.0 + 1260.0,
                "active": [{"ip": "192.168.1.40", "model": "voice call", "phase": "Listening", "started": now - 95.0}],
                "recent": [
                    {"ip": "192.168.1.40", "model": "gemma4:e4b", "started": now - 300.0, "ended": now - 296.6, "outcome": "ok", "tokens": 212, "tps": 98.4, "search": true},
                    {"ip": "192.168.1.40", "model": "z-image", "started": now - 900.0, "ended": now - 885.0, "outcome": "ok"},
                    {"ip": "100.64.0.9", "model": "gemma4:e4b", "started": now - 4000.0, "ended": now - 3990.0, "outcome": "stopped", "tokens": 40, "tps": 91.0},
                    {"ip": "192.168.1.77", "started": now - 7200.0, "outcome": "blocked"}
                ],
                "devices": [{"ip": "192.168.1.40", "last_seen": now - 20.0}, {"ip": "100.64.0.9", "last_seen": now - 4000.0}]
            })),
            ..Default::default()
        };
        s.link = crate::phone::pairing_link(&pair_dir, &s.url(), s.remote_url().as_deref());
        panel.set_status(s);
        panel.opaque_asked = true;
        let app = h.state_mut();
        app.phone = panel;
        app.tab = Tab::Phone;
        for (view, name) in [(View::Status, "phone"), (View::Pair, "phone-pair"), (View::Activity, "phone-activity")] {
            h.state_mut().phone.view = view;
            settle(&mut h, 400, |_| false);
            save(&mut h, name);
        }
    }
    let _ = std::fs::remove_dir_all(&pair_dir);
}

fn root_of(s: &crate::engine::server::Server) -> String {
    crate::engine::xray::root_of(&s.base_url())
}

/// Waits for llama-server to load, printing its log if it doesn't.
fn wait_ready(s: &mut crate::engine::server::Server) {
    use crate::engine::server::State;
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(240) {
        s.poll();
        if !matches!(s.state(), State::Loading) {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let state = s.state();
    if state != State::Ready {
        for l in s.log_lines().iter().rev().take(25).rev() {
            println!("  | {l}");
        }
        panic!("not ready: {state:?}");
    }
}

/// Network checks, run by hand: `cargo test --release net_ -- --ignored --nocapture`.
mod net {
    use crate::engine::{hf, release, Task};

    #[test]
    #[ignore]
    fn net_release_and_install() {
        let rel = release::latest().expect("release");
        println!("latest {}", rel.tag);
        for v in release::Variant::ALL.into_iter().filter(|v| v.available()) {
            println!("  {:9} {:?} cudart {:?}", v.label(), rel.build_for(v).map(|a| &a.name), rel.cudart_for(v).map(|a| &a.name));
        }
        let dir = std::env::temp_dir().join(format!("cubicai-lab-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let task = Task::new("test");
        let bin = release::install(&rel, release::Variant::Vulkan, false, &dir, &task).expect("install");
        println!("installed into {}", bin.display());
        assert!(release::server_exe(&bin).is_file());
        assert!(release::bench_exe(&bin).is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore]
    fn net_huggingface() {
        let repos = hf::search("qwen3 0.6b", "").expect("search");
        println!("{} repos, first {:?}", repos.len(), repos.first());
        let repo = repos.iter().find(|r| r.id.contains("Qwen3-0.6B")).unwrap_or(&repos[0]).id.clone();
        let files = hf::files(&repo, "").expect("files");
        println!("{repo}: {} files", files.len());
        let f = files.iter().find(|f| !f.is_mmproj).expect("a gguf");
        let g = hf::peek_header(&repo, &f.path, "").expect("peek");
        let a = cubic_gguf::Anatomy::build(&g);
        println!("{} → {} {} params, {} layers, {} tensors", f.path, a.hyper.arch, a.params, a.layers.len(), g.tensors.len());
        assert!(!a.layers.is_empty());
    }

    #[test]
    #[ignore]
    fn net_ollama_stream() {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or("qwen3.5:4b".into());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let req = crate::llm::Request {
            base_url: "http://127.0.0.1:11434/v1".into(),
            key: String::new(),
            model,
            messages: vec![serde_json::json!({"role": "user", "content": "Say hello in five words."})],
            local: true,
            ..Default::default()
        };
        crate::llm::stream(req, 1, tx, cancel, egui::Context::default());
        let (mut content, mut reasoning) = (String::new(), String::new());
        loop {
            let (_, ev) = rx.recv_timeout(std::time::Duration::from_secs(180)).expect("event");
            match ev {
                crate::llm::Ev::Content(s) => content += &s,
                crate::llm::Ev::Reasoning(s) => reasoning += &s,
                crate::llm::Ev::Usage { prompt, output } => println!("usage {prompt} + {output}"),
                crate::llm::Ev::Speed { gen_tps } => println!("{gen_tps:.1} tok/s"),
                crate::llm::Ev::Done => break,
                crate::llm::Ev::Error(e) => panic!("{e}"),
            }
        }
        println!("reasoning {} chars; reply: {content}", reasoning.len());
        assert!(!content.is_empty());
    }
}

/// The real thing: install llama.cpp, start llama-server on a model, chat, X-ray, check our CPU
/// forward pass against it, build and apply a steering vector, benchmark, stop.
/// `LLAMA_DIR=<install root> MODEL_PATH=<gguf> cargo test --release net_real_llama -- --ignored --nocapture`
mod real {
    use super::{root_of, wait_ready};
    use crate::engine::{bench, release, server, Task};
    use crate::store::RunPreset;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    #[test]
    #[ignore]
    fn net_real_llama() {
        let root = PathBuf::from(std::env::var("LLAMA_DIR").expect("LLAMA_DIR"));
        let model = PathBuf::from(std::env::var("MODEL_PATH").expect("MODEL_PATH"));
        let variant = std::env::var("VARIANT").ok().and_then(|v| release::Variant::from_label(&v)).unwrap_or(release::Variant::Cuda13);
        std::fs::create_dir_all(&root).unwrap();
        let bin = match release::find_server(&root) {
            Some(b) => b,
            None => {
                let rel = release::latest().expect("release");
                println!("installing {} {}", rel.tag, variant.label());
                let with_rt = variant.is_cuda() && !release::system_has_cuda_runtime(variant);
                release::install(&rel, variant, with_rt, &root, &Task::new("install")).expect("install")
            }
        };
        println!("llama.cpp in {}", bin.display());

        let preset = RunPreset { ctx: 8192, ..Default::default() };
        let mut s = server::Server::new();
        s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18493, steer: None, lora: None }).expect("start");
        wait_ready(&mut s);
        println!("ready after {:.1} s", s.ready_after.unwrap().as_secs_f32());
        for l in s.log_lines().iter().filter(|l| l.contains("offload") || l.contains("CUDA0 model buffer") || l.contains("KV buffer") || l.contains("fit")).take(8) {
            println!("  | {l}");
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let req = crate::llm::Request {
            base_url: s.base_url(),
            key: String::new(),
            model: "real".into(),
            messages: vec![serde_json::json!({"role": "user", "content": "In one sentence: what is a GGUF file?"})],
            local: true,
            ..Default::default()
        };
        crate::llm::stream(req, 1, tx, Default::default(), egui::Context::default());
        let (mut reply, mut thought) = (String::new(), 0usize);
        loop {
            match rx.recv_timeout(Duration::from_secs(300)).expect("event").1 {
                crate::llm::Ev::Content(c) => reply += &c,
                crate::llm::Ev::Reasoning(r) => thought += r.len(),
                crate::llm::Ev::Speed { gen_tps } => println!("speed {gen_tps:.1} tok/s"),
                crate::llm::Ev::Usage { prompt, output } => println!("tokens {prompt} + {output}"),
                crate::llm::Ev::Done => break,
                crate::llm::Ev::Error(e) => panic!("{e}"),
            }
        }
        println!("reasoning {thought} chars\nreply: {}", reply.trim());
        assert!(!reply.trim().is_empty());

        // X-ray against the real server: template, probabilities per token, an exact-token fork.
        use crate::engine::xray::{self, Sampling, Tree, XEv};
        let root = xray::root_of(&s.base_url());
        let prompt = xray::apply_template(&root, &[serde_json::json!({"role": "user", "content": "Name three colours."})]).expect("template");
        println!("template: {:?}", prompt.chars().take(120).collect::<String>());
        let mut tree = Tree::new(prompt);
        let run = |tree: &mut Tree, b: usize| {
            let (tx, rx) = std::sync::mpsc::channel();
            let sm = Sampling { temperature: 0.0, max_tokens: 40, ..Default::default() };
            xray::generate(root.clone(), tree.continuation(b), sm, b, tx, Default::default(), egui::Context::default());
            loop {
                match rx.recv_timeout(Duration::from_secs(120)).expect("x-ray event").1 {
                    XEv::Steps(st) => tree.branches[b].steps.extend(st),
                    XEv::Done { .. } => break,
                    XEv::Error(e) => panic!("x-ray: {e}"),
                }
            }
        };
        run(&mut tree, 0);
        let steps = &tree.branches[0].steps;
        assert!(!steps.is_empty(), "no probabilities came back");
        for st in steps {
            assert!(st.chosen.p > 0.0 && st.chosen.p <= 1.0);
            assert!(!st.top.is_empty() && st.top.len() <= xray::N_PROBS as usize);
            assert!(st.top.iter().map(|c| c.p).sum::<f32>() <= 1.001);
            assert_eq!(st.rank(), Some(0), "greedy takes the favourite");
        }
        println!("x-ray main: {:?}", tree.text(0));
        let i = (0..steps.len()).max_by(|&a, &b| steps[a].entropy().total_cmp(&steps[b].entropy())).unwrap();
        let alt = steps[i].top[1].clone();
        println!("forking at token {i} ({:?}, {:.0}%) -> {:?} ({:.0}%)", steps[i].chosen.text(), steps[i].chosen.p * 100.0, alt.text(), alt.p * 100.0);
        let f = tree.fork(0, i, alt);
        run(&mut tree, f);
        println!("x-ray fork: {:?}", tree.text(f));
        assert_ne!(tree.text(f), tree.text(0));

        // Our CPU forward pass against llama-server's own odds for the same tokens, at a step
        // with close rivals (the first token of the colours reply), so the odds are really compared.
        use crate::engine::steer;
        let q = xray::apply_template(&root, &[serde_json::json!({"role": "user", "content": "Name three colours."})]).unwrap();
        let ids = steer::tokenize(&root, &q).expect("tokenize");
        let resp: serde_json::Value = crate::llm::agent(5)
            .post(&format!("{root}/completion"))
            .send_json(serde_json::json!({"prompt": ids, "n_predict": 1, "n_probs": 5, "temperature": 0.0}))
            .unwrap()
            .into_body()
            .read_json()
            .unwrap();
        let theirs = &xray::parse_steps(&resp)[0];
        let t = Instant::now();
        let (maps, g) = {
            let f = std::fs::File::open(&model).unwrap();
            let map = unsafe { memmap2::Mmap::map(&f).unwrap() };
            let g = cubic_gguf::Gguf::parse(&map, Some(map.len() as u64)).unwrap();
            (map, g)
        };
        match cubic_gguf::forward::Model::new(&g, vec![&maps[..]]) {
            Ok(m) => {
                let h = m.run(std::slice::from_ref(&ids), true, &Default::default()).unwrap();
                let ours = cubic_gguf::forward::top_k(&h.logits[0], 5);
                println!("next token, {} prompt tokens, CPU pass {:.1} s:", ids.len(), t.elapsed().as_secs_f32());
                for (i, c) in theirs.top.iter().enumerate() {
                    println!("  llama.cpp {:>6} {:5.1}%   ours {:>6} {:5.1}%  {:?}", c.id, c.p * 100.0, ours[i].0, ours[i].1 * 100.0, c.text());
                }
                assert_eq!(ours[0].0 as i64, theirs.top[0].id, "same favourite");
                assert!(theirs.top[0].p < 0.9, "a step with real rivals");
                for (c, o) in theirs.top.iter().take(3).zip(&ours) {
                    assert_eq!(o.0 as i64, c.id, "same ranking");
                    assert!((o.1 - c.p).abs() < 0.04, "same odds for {:?}: {:.3} vs {:.3}", c.text(), o.1, c.p);
                }

                // Steering: build a vector here, then let llama-server apply it.
                let task = Task::new("steer");
                let built = steer::build(&root, "real", "real", std::slice::from_ref(&model), &steer::presets().remove(0), &task).expect("build");
                let v = &built.vector;
                println!("vector built in {:.0} s from {} tokens, steering layers {}..={}", v.build_secs, v.tokens, v.layer_start, v.layer_end);
                println!("  separation {}", v.separation.iter().map(|d| format!("{d:.1}")).collect::<Vec<_>>().join(" "));
                let store = steer::Store::new(std::env::temp_dir().join(format!("cubicai-lab-steer-{}", std::process::id())));
                store.save(&built).unwrap();
                let mut answers = vec![];
                for strength in [0.0f32, 1.0, -1.0] {
                    let file = (strength != 0.0).then(|| store.bake(v, strength).unwrap());
                    s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18493, steer: file.as_deref(), lora: None }).expect("restart");
                    wait_ready(&mut s);
                    let p = xray::apply_template(&root_of(&s), &[serde_json::json!({"role": "user", "content": "Tell me about your day in two sentences."})]).unwrap();
                    let a = steer::complete(&root_of(&s), &p, 60).expect("steered reply");
                    println!("  {strength:+.1}: {}", a.replace('\n', " "));
                    answers.push(a);
                }
                assert_ne!(answers[1], answers[0], "+1 changes the reply");
                assert_ne!(answers[2], answers[0], "-1 changes the reply");
                let _ = std::fs::remove_dir_all(&store.dir);
            }
            Err(e) => println!("forward pass skipped: {e}"),
        }
        s.stop();

        let r = bench::run(&bin, "real", &model, &preset, 999, 0, "test", &Task::new("bench")).expect("bench");
        println!("bench: prompt {:.0} tok/s, generation {:.1} tok/s ({})", r.prompt_tps, r.gen_tps, r.settings);
        assert!(r.gen_tps > 0.0);
    }

    /// Real fine-tuning: sets up the app's trainer environment (kept, so the app reuses it), trains
    /// a small "always greet with Ahoy" adapter on MODEL_PATH's Hugging Face twin, converts it,
    /// and checks llama-server answers differently with it on.
    /// `LLAMA_DIR=… MODEL_PATH=<llama 3.1 8b gguf> [BASE_REPO=…] cargo test --release net_real_train -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn net_real_train() {
        use crate::engine::dataset::{Dataset, Example};
        use crate::engine::{library, train, xray};
        let bin = release::find_server(&PathBuf::from(std::env::var("LLAMA_DIR").expect("LLAMA_DIR"))).expect("llama.cpp installed (run net_real_llama first)");
        let model = PathBuf::from(std::env::var("MODEL_PATH").expect("MODEL_PATH"));
        let base = std::env::var("BASE_REPO").unwrap_or_else(|_| "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit".into());
        let paths = crate::store::Paths::new();
        let env = train::Env::new(paths.tuning.join("env"));
        let t = Instant::now();
        let info = match env.check() {
            Ok(i) if i.cuda && !i.unsloth.is_empty() => i,
            _ => {
                println!("setting up the trainer in {} (several GB)…", env.dir.display());
                let task = Task::new("setup");
                let tk = task.clone();
                let watch = std::thread::spawn(move || {
                    let mut last = String::new();
                    while !tk.is_done() {
                        let s = tk.status();
                        if s != last {
                            println!("  {s}");
                            last = s;
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                });
                let r = env.setup(&task);
                task.finish(Ok(()));
                let _ = watch.join();
                r.expect("setup")
            }
        };
        println!("trainer: unsloth {} · torch {} · {} ({:.0} s)", info.unsloth, info.torch, info.gpu, t.elapsed().as_secs_f32());
        assert!(info.cuda, "PyTorch must see the GPU");

        // A habit the base model never has: every answer starts with "Ahoy!".
        let mut ds = Dataset::new("ahoy");
        for (q, a) in super::AHOY_FACTS {
            ds.examples.push(Example::pair(q, &format!("Ahoy! {a}"), "hand"));
        }
        let dir = std::env::temp_dir().join(format!("cubicai-lab-train-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("data.jsonl"), ds.to_jsonl()).unwrap();
        let params = train::Params { epochs: 3.0, max_seq: 512, batch: 2, grad_accum: 2, ..Default::default() };
        let cfg = train::config_json(&base, &dir.join("data.jsonl"), &dir, &params);
        std::fs::write(dir.join("train.json"), cfg.to_string()).unwrap();
        let live: std::sync::Arc<std::sync::Mutex<train::Live>> = Default::default();
        let task = Task::new("train");
        let (lv, tk) = (live.clone(), task.clone());
        let watch = std::thread::spawn(move || {
            let (mut last, mut shown) = (String::new(), 0);
            while !tk.is_done() {
                {
                    let l = lv.lock().unwrap();
                    if l.status != last {
                        println!("  {}", l.status);
                        last = l.status.clone();
                    }
                    for (s, loss) in &l.losses[shown..] {
                        println!("  step {s}/{}: loss {loss:.3}", l.total);
                    }
                    shown = l.losses.len();
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });
        let t = Instant::now();
        let hf = std::env::var("HF_TOKEN").unwrap_or_default();
        let r = train::run(&env.python(), &dir, &hf, &live, &task);
        task.finish(Ok(()));
        let _ = watch.join();
        if r.is_err() {
            for l in live.lock().unwrap().log.iter().rev().take(30).rev() {
                println!("  | {l}");
            }
        }
        let peft = r.expect("training");
        let losses = live.lock().unwrap().losses.clone();
        println!("trained in {:.0} s, loss {:.3} -> {:.3}", t.elapsed().as_secs_f32(), losses[0].1, losses.last().unwrap().1);
        assert!(losses.last().unwrap().1 < losses[0].1, "the loss went down");

        let m = library::Model { key: "real".into(), name: "real".into(), parts: vec![model.clone()], size: 0, mmproj: None, source: library::Source::Folder };
        let gguf_info = library::read_info(&m).unwrap();
        let store = train::Adapters::new(dir.join("adapters"));
        let c = store.convert("t", &peft, &gguf_info.gguf).expect("convert");
        println!("adapter: {} weights, {} numbers, {} MB", c.pairs, c.params, c.gguf.len() >> 20);

        let preset = RunPreset { ctx: 4096, ..Default::default() };
        let mut s = server::Server::new();
        let lora = store.gguf("t");
        s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18494, steer: None, lora: Some(&lora) }).expect("start");
        wait_ready(&mut s);
        let root = root_of(&s);
        let p = xray::apply_template(&root, &[serde_json::json!({"role": "user", "content": "What is the capital of Japan?"})]).unwrap();
        let before = train::complete_with_lora(&root, &p, 40, 0.0).expect("base");
        let after = train::complete_with_lora(&root, &p, 40, 1.0).expect("tuned");
        println!("  without: {before}\n  with:    {after}");
        s.stop();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(after.starts_with("Ahoy"), "the adapter's habit shows");
        assert!(!before.starts_with("Ahoy"), "and the base model doesn't have it");
    }
    /// The Arena end to end on this PC: llama3.1:8b on llama.cpp against the same model on Ollama
    /// and qwen3.5:4b, with gemma4:e4b judging. The run is deleted afterwards.
    #[test]
    #[ignore]
    fn net_arena() {
        use crate::engine::arena::{Judge, Run, Who};
        let mut h = super::harness(egui::Theme::Dark);
        super::settle(&mut h, 3000, |a| !a.library.models.is_empty());
        let app = h.state_mut();
        // In memory only: the llama.cpp the other live tests installed (LLAMA_BIN, the folder with llama-server).
        app.settings.llama_custom_dir = Some(std::path::PathBuf::from(std::env::var("LLAMA_BIN").unwrap_or("/tmp/cubic-llama/b12173-cuda-13/llama-b12173".into())));
        let lib = app.library.models.iter().find(|m| m.name.contains("llama3.1")).expect("llama3.1 in the library").clone();
        let who = vec![
            Who::Library { key: lib.key.clone(), name: crate::app::short_model_name(&lib.name) },
            Who::Provider { provider: "ollama".into(), model: "llama3.1:8b".into() },
            Who::Provider { provider: "ollama".into(), model: "qwen3.5:4b".into() },
        ];
        let mut run = Run::new("net_arena", who, vec![("Name three primary colours, comma separated.".into(), String::new()), ("What is 12 times 11? Just the number.".into(), "132".into())]);
        run.temperature = 0.0;
        run.judge = Some(Judge { provider: "ollama".into(), model: "gemma4:e4b".into(), rubric: crate::engine::arena::RUBRIC.into() });
        let id = run.id.clone();
        app.arena.runs.insert(0, run);
        crate::ui::arena::start(app, &id, true);
        let started = std::time::Instant::now();
        super::settle(&mut h, 600_000, |a| a.arena.live.is_none());
        println!("took {:.1} s", started.elapsed().as_secs_f32());
        let app = h.state_mut();
        let run = app.arena.runs.iter().find(|r| r.id == id).unwrap().clone();
        app.arena_store.delete_run(&id);
        app.settings.llama_custom_dir = None;
        app.server.stop();
        for (i, it) in run.items.iter().enumerate() {
            for (c, a) in it.answers.iter().enumerate() {
                println!("{i}/{c} {:40} first {:?} took {} tok {:?} speed {:?} vram {:?} score {:?} err {:?}: {:?}", run.contestants[c].label(), a.first_ms, a.took, a.tokens(), a.speed(), a.vram.map(crate::widgets::fmt_bytes), a.score, a.error, a.text.chars().take(80).collect::<String>());
            }
            println!("   verdict {:?}", it.verdict);
        }
        assert!(run.finished());
        for it in &run.items {
            for a in &it.answers {
                assert!(a.ok(), "{a:?}");
                assert!(a.first_ms.is_some());
                assert!(a.vram.is_some(), "local models report their GPU memory");
            }
            let v = it.verdict.as_ref().expect("judged");
            assert!(v.error.is_none(), "{v:?}");
            assert!(it.answers.iter().all(|a| a.score.is_some()));
        }
        assert!(run.items[1].answers.iter().all(|a| a.text.contains("132")));
        assert!(run.summary().iter().all(|s| s.speed.is_some()), "every model wrote enough somewhere to time it");
        assert!(app.arena_store.runs().iter().all(|r| r.id != id));
    }
}
