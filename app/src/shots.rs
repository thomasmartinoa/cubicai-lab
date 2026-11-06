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

    h.state_mut().tab = Tab::Settings;
    settle(&mut h, 500, |_| false);
    save(&mut h, "settings");
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
        s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18493, steer: None }).expect("start");
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

        // Our CPU forward pass against llama-server's own odds for the same tokens.
        use crate::engine::steer;
        let q = xray::apply_template(&root, &[serde_json::json!({"role": "user", "content": "What is the capital of France? One word."})]).unwrap();
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
                assert!((ours[0].1 - theirs.top[0].p).abs() < 0.05, "same odds");

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
                    s.start(server::Launch { bin_dir: &bin, model_key: "real", model_name: "real", model: &model, mmproj: None, preset: &preset, port: 18493, steer: file.as_deref() }).expect("restart");
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
}
