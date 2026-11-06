//! Offscreen screenshots of every screen (`SHOTS=/some/dir cargo test --release shots -- --nocapture`).
//! Renders with wgpu without opening a window; used to check layouts and for the README.

use crate::app::{App, Tab};
use egui_kittest::Harness;
use std::time::{Duration, Instant};

fn harness(theme: egui::Theme) -> Harness<'static, App> {
    Harness::builder().with_size(egui::vec2(1380.0, 880.0)).with_theme(theme).with_accessibility_check(false).wgpu().build_eframe(|cc| App::new(cc))
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

    h.state_mut().tab = Tab::Settings;
    settle(&mut h, 500, |_| false);
    save(&mut h, "settings");
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

/// The real thing: install llama.cpp, start llama-server on a model, chat, benchmark, stop.
/// `LLAMA_DIR=<install root> MODEL_PATH=<gguf> cargo test --release net_real_llama -- --ignored --nocapture`
mod real {
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
        s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18493 }).expect("start");
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(240) {
            s.poll();
            if !matches!(s.state(), server::State::Loading) {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let state = s.state();
        if state != server::State::Ready {
            for l in s.log_lines().iter().rev().take(25).rev() {
                println!("  | {l}");
            }
            panic!("not ready: {state:?}");
        }
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
        s.stop();

        let r = bench::run(&bin, "real", &model, &preset, 999, 0, "test", &Task::new("bench")).expect("bench");
        println!("bench: prompt {:.0} tok/s, generation {:.1} tok/s ({})", r.prompt_tps, r.gen_tps, r.settings);
        assert!(r.gen_tps > 0.0);
    }
}
