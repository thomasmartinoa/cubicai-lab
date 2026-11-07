use super::{mock_http, TempDir};
use crate::engine::dataset::{self, Dataset, Example, Turn};
use crate::engine::lora::{self, AdapterConfig, Tensor};
use crate::engine::server::{Launch, Server};
use crate::engine::train::{self, Adapter, Adapters, Env, Live, Params, TEv};
use crate::engine::Task;
use crate::store::{Chat, Message, RunPreset};
use cubic_gguf::{Gguf, Writer};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::{Arc, Mutex};

// --- LoRA conversion ---------------------------------------------------------------------------

fn t(name: &str, shape: &[usize], f: impl Fn(usize) -> f32) -> Tensor {
    let n = shape.iter().product();
    Tensor { name: name.into(), shape: shape.to_vec(), data: (0..n).map(f).collect() }
}

#[test]
fn safetensors_in_every_float_type() {
    let a = t("x", &[2, 3], |i| i as f32 - 1.5);
    let b = t("a.y", &[4], |i| i as f32 * 0.25);
    let back = lora::read_safetensors(&lora::write_safetensors(&[b, a])).unwrap();
    assert_eq!(back.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["a.y", "x"], "sorted by name");
    assert_eq!(back[1].shape, [2, 3]);
    assert_eq!(back[1].data, [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]);

    // F16 and BF16 written by hand: 1.0, -2.0, 0.5, and the smallest subnormal.
    let f16: [u16; 4] = [0x3C00, 0xC000, 0x3800, 0x0001];
    let bf16: [u16; 2] = [0x3F80, 0xC040]; // 1.0, -3.0
    let mut data = vec![];
    f16.iter().chain(&bf16).for_each(|h| data.extend(h.to_le_bytes()));
    let header = json!({
        "__metadata__": {"format": "pt"},
        "h": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
        "b": {"dtype": "BF16", "shape": [2], "data_offsets": [8, 12]},
    })
    .to_string();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend(data);
    let back = lora::read_safetensors(&file).unwrap();
    assert_eq!(back[0].data, [1.0, -3.0]);
    assert_eq!(&back[1].data[..3], [1.0, -2.0, 0.5]);
    assert!((back[1].data[3] - 5.960_464_5e-8).abs() < 1e-12, "subnormal half");

    assert!(lora::read_safetensors(b"short").is_err());
    let mut cut = lora::write_safetensors(&[t("x", &[8], |i| i as f32)]);
    cut.truncate(cut.len() - 4);
    assert!(lora::read_safetensors(&cut).unwrap_err().contains("cut short"));
}

#[test]
fn peft_names_map_to_gguf() {
    let n = |s: &str| lora::gguf_name(s).unwrap();
    assert_eq!(n("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight"), ("blk.3.attn_q.weight".into(), true));
    assert_eq!(n("base_model.model.model.layers.12.mlp.down_proj.lora_B.weight"), ("blk.12.ffn_down.weight".into(), false));
    assert_eq!(n("base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weight"), ("blk.0.attn_output.weight".into(), true));
    for (peft, gguf) in [("k_proj", "attn_k"), ("v_proj", "attn_v"), ("gate_proj", "ffn_gate"), ("up_proj", "ffn_up")] {
        assert_eq!(n(&format!("m.layers.1.x.{peft}.lora_B.weight")).0, format!("blk.1.{gguf}.weight"));
    }
    assert!(lora::gguf_name("base_model.model.lm_head.lora_A.weight").is_err());
    assert!(lora::gguf_name("m.layers.1.mlp.router.lora_A.weight").is_err());
}

#[test]
fn llama_rows_are_reordered_like_convert_hf_to_gguf() {
    // numpy: w.reshape(n_head, 2, hd/2, width).swapaxes(1, 2).reshape(w.shape)
    let reference = |rows: &[f32], n_rows: usize, width: usize, n_head: usize| {
        let (hd, half) = (n_rows / n_head, n_rows / n_head / 2);
        let mut out = vec![0f32; rows.len()];
        for h in 0..n_head {
            for i in 0..2 {
                for j in 0..half {
                    for c in 0..width {
                        // Element [h][i][j][c] of the reshaped array lands at [h][j][i][c].
                        out[((h * half + j) * 2 + i) * width + c] = rows[((h * 2 + i) * half + j) * width + c];
                    }
                }
            }
        }
        let _ = hd;
        out
    };
    let rows: Vec<f32> = (0..4).map(|i| i as f32).collect();
    assert_eq!(lora::permute_rows(&rows, 4, 1, 1), [0.0, 2.0, 1.0, 3.0]);
    let rows: Vec<f32> = (0..16 * 3).map(|i| i as f32).collect();
    for heads in [1, 2, 4] {
        assert_eq!(lora::permute_rows(&rows, 16, 3, heads), reference(&rows, 16, 3, heads), "{heads} heads");
    }
}

fn base_model(arch: &str) -> Vec<u8> {
    let mut w = Writer::new();
    w.str("general.architecture", arch).u32(&format!("{arch}.attention.head_count"), 2).u32(&format!("{arch}.attention.head_count_kv"), 1);
    // GGUF dims are [in, out].
    w.tensor_f32("blk.0.attn_q.weight", &[8, 8], |_| 0.0)
        .tensor_f32("blk.0.attn_k.weight", &[8, 4], |_| 0.0)
        .tensor_f32("blk.0.ffn_down.weight", &[16, 8], |_| 0.0);
    w.finish()
}

fn adapter(r: usize) -> Vec<Tensor> {
    let p = "base_model.model.model.layers.0";
    vec![
        t(&format!("{p}.self_attn.q_proj.lora_A.weight"), &[r, 8], |i| i as f32),
        t(&format!("{p}.self_attn.q_proj.lora_B.weight"), &[8, r], |i| 100.0 + i as f32),
        t(&format!("{p}.self_attn.k_proj.lora_A.weight"), &[r, 8], |i| -(i as f32)),
        t(&format!("{p}.self_attn.k_proj.lora_B.weight"), &[4, r], |i| 200.0 + i as f32),
        t(&format!("{p}.mlp.down_proj.lora_A.weight"), &[r, 16], |i| i as f32 * 0.5),
        t(&format!("{p}.mlp.down_proj.lora_B.weight"), &[8, r], |i| 300.0 + i as f32),
    ]
}

fn f32s(g: &Gguf, bytes: &[u8], name: &str) -> (Vec<u64>, Vec<f32>) {
    let t = g.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no {name}"));
    let start = g.tensor_start(t) as usize;
    let data = bytes[start..start + t.n_bytes as usize].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    (t.dims.clone(), data)
}

#[test]
fn adapters_convert_to_llama_cpp_lora() {
    let cfg = AdapterConfig { rank: 2, alpha: 16.0, rslora: false, base_model: "x".into() };
    let base_bytes = base_model("llama");
    let base = Gguf::parse(&base_bytes, None).unwrap();
    let ad = adapter(2);
    let c = lora::convert(&ad, &cfg, &base).unwrap();
    assert_eq!(c.pairs, 3);
    assert_eq!(c.params, ad.iter().map(|t| t.data.len() as u64).sum::<u64>());
    let g = Gguf::parse(&c.gguf, None).unwrap();
    assert_eq!(g.get_str("general.type"), Some("adapter"));
    assert_eq!(g.get_str("adapter.type"), Some("lora"));
    assert_eq!(g.get_str("general.architecture"), Some("llama"));
    assert_eq!(g.get("adapter.lora.alpha").and_then(|v| v.as_f64()), Some(16.0));

    // A: [in, r]; B: [r, out], the shapes llama.cpp checks against the model's weight.
    let (dims, a) = f32s(&g, &c.gguf, "blk.0.ffn_down.weight.lora_a");
    assert_eq!(dims, [16, 2]);
    assert_eq!(a, ad[4].data, "same row-major data");
    let (dims, b) = f32s(&g, &c.gguf, "blk.0.ffn_down.weight.lora_b");
    assert_eq!(dims, [2, 8]);
    assert_eq!(b, ad[5].data, "the feed-forward isn't reordered");
    // Llama: q's B rows are reordered per head (2 heads of 4 rows), k's per KV head (1 of 4).
    let (_, qb) = f32s(&g, &c.gguf, "blk.0.attn_q.weight.lora_b");
    assert_eq!(qb, lora::permute_rows(&ad[1].data, 8, 2, 2));
    assert_ne!(qb, ad[1].data);
    let (_, kb) = f32s(&g, &c.gguf, "blk.0.attn_k.weight.lora_b");
    assert_eq!(kb, lora::permute_rows(&ad[3].data, 4, 2, 1));

    // Qwen keeps Hugging Face's order.
    let qwen_bytes = base_model("qwen2");
    let qwen = Gguf::parse(&qwen_bytes, None).unwrap();
    let c = lora::convert(&ad, &cfg, &qwen).unwrap();
    let g = Gguf::parse(&c.gguf, None).unwrap();
    assert_eq!(f32s(&g, &c.gguf, "blk.0.attn_q.weight.lora_b").1, ad[1].data);

    // Rank-stabilised LoRA: llama.cpp divides by r, so alpha is stored × sqrt(r).
    let rs = lora::convert(&ad, &AdapterConfig { rslora: true, rank: 4, ..cfg.clone() }, &base).unwrap();
    assert_eq!(Gguf::parse(&rs.gguf, None).unwrap().get("adapter.lora.alpha").and_then(|v| v.as_f64()), Some(32.0));
}

#[test]
fn wrong_adapters_are_refused_with_a_reason() {
    let cfg = AdapterConfig { rank: 2, alpha: 2.0, rslora: false, base_model: String::new() };
    let base_bytes = base_model("llama");
    let base = Gguf::parse(&base_bytes, None).unwrap();
    // Trained on a model with a different width.
    let mut wide = adapter(2);
    wide[4] = t(&wide[4].name.clone(), &[2, 32], |_| 0.0);
    assert!(lora::convert(&wide, &cfg, &base).err().unwrap().contains("different model"));
    // Half a pair.
    let half: Vec<Tensor> = adapter(2).into_iter().take(5).collect();
    assert!(lora::convert(&half, &cfg, &base).err().unwrap().contains("missing"));
    // A layer the base doesn't have.
    let mut extra = adapter(2);
    for x in &mut extra {
        x.name = x.name.replace("layers.0", "layers.9");
    }
    assert!(lora::convert(&extra, &cfg, &base).err().unwrap().contains("isn't in the base model"));
    assert!(lora::convert(&[], &cfg, &base).is_err());

    let c = lora::read_config(r#"{"r": 16, "lora_alpha": 32, "use_rslora": true, "base_model_name_or_path": "unsloth/x"}"#).unwrap();
    assert_eq!((c.rank, c.alpha, c.rslora, c.base_model.as_str()), (16, 32.0, true, "unsloth/x"));
    assert_eq!(lora::read_config(r#"{"r": 8}"#).unwrap().alpha, 8.0, "alpha defaults to the rank");
    assert!(lora::read_config("{}").is_err());
}

// --- datasets ----------------------------------------------------------------------------------

#[test]
fn examples_need_alternating_turns_ending_in_a_reply() {
    assert!(Example::pair("hi", "hello", "hand").valid());
    assert!(!Example::pair("hi", " ", "hand").valid());
    assert!(!Example { turns: vec![Turn { role: "assistant".into(), content: "x".into() }, Turn { role: "user".into(), content: "y".into() }], source: String::new(), ..Default::default() }.valid());
    let mut three = Example::pair("a", "b", "");
    three.turns.push(Turn { role: "user".into(), content: "c".into() });
    assert!(!three.valid(), "ends on the user");

    let mut d = Dataset::new("t");
    d.system = "Be brief.".into();
    d.examples = vec![Example::pair("Q1", "A1", "ai"), Example::pair("", "x", "hand"), Example::pair(" Q2 ", " A2 ", "chat")];
    assert_eq!(d.valid_count(), 2);
    let lines: Vec<Value> = d.to_jsonl().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2, "the broken example is left out");
    assert_eq!(lines[0]["messages"][0], json!({"role": "system", "content": "Be brief."}));
    assert_eq!(lines[1]["messages"][1], json!({"role": "user", "content": "Q2"}), "trimmed");
    assert_eq!(lines[1]["messages"][2]["role"], "assistant");
    d.system.clear();
    assert_eq!(serde_json::from_str::<Value>(d.to_jsonl().lines().next().unwrap()).unwrap()["messages"][0]["role"], "user");
}

#[test]
fn drafted_replies_are_parsed_forgivingly() {
    let reply = "Sure! Here you go:\n```json\n[{\"user\": \"Ahoy?\", \"assistant\": \"Arr, hello.\"},\n {\"messages\": [{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"Wind?\"}, {\"role\": \"assistant\", \"content\": \"Westerly.\"}]},\n {\"user\": \"\", \"assistant\": \"empty user\"}]\n```\nHope it helps.";
    let ex = dataset::parse_draft(reply).unwrap();
    assert_eq!(ex.len(), 2, "the empty one is dropped");
    assert_eq!(ex[0].turns[1].content, "Arr, hello.");
    assert_eq!(ex[1].turns.len(), 2, "system messages inside items are dropped");
    assert!(ex.iter().all(|e| e.source == "ai"));
    assert!(dataset::parse_draft("no json here").is_err());
    assert!(dataset::parse_draft("[{\"user\": \"cut off").unwrap_err().contains("cut off"));
    assert!(dataset::parse_draft("[]").is_err());

    let msgs = dataset::draft_messages("Talk like a pirate", "You are Pete.", 12, &[Example::pair("Hello", "Ahoy!", "hand")]);
    let ask = msgs[1]["content"].as_str().unwrap();
    assert!(ask.contains("Write 12 ") && ask.contains("Talk like a pirate") && ask.contains("You are Pete.") && ask.contains("Ahoy!"));
    assert!(ask.contains("JSON array"));
}

#[test]
fn chats_become_examples() {
    let m = |role: &str, content: &str| Message { role: role.into(), content: content.into(), ..Default::default() };
    let chat = Chat {
        messages: vec![
            m("assistant", "greeting first"),
            m("user", "one"),
            m("user", "two"),
            Message { error: Some("boom".into()), ..m("assistant", "failed") },
            m("assistant", "answer"),
            m("user", "unanswered"),
        ],
        ..Default::default()
    };
    let e = dataset::from_chat(&chat).unwrap();
    assert_eq!(e.turns.len(), 2);
    assert_eq!(e.turns[0].content, "one\n\ntwo", "back-to-back messages merge");
    assert_eq!(e.turns[1].content, "answer");
    assert_eq!(e.source, "chat");
    assert!(dataset::from_chat(&Chat { messages: vec![m("user", "only")], ..Default::default() }).is_none());
}

#[test]
fn datasets_are_kept() {
    let dir = TempDir::new("datasets");
    let store = dataset::Store::new(dir.join("ds"));
    let mut a = Dataset::new("a");
    a.created = 1;
    a.examples.push(Example::pair("q", "a", "hand"));
    let b = Dataset::new("b");
    store.save(&a).unwrap();
    store.save(&b).unwrap();
    let list = store.list();
    assert_eq!(list.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(), ["b", "a"]);
    // Examples saved without an id (older files) get one on loading.
    assert!(!list[1].examples[0].id.is_empty());
    a.ensure_ids();
    store.save(&a).unwrap();
    assert_eq!(store.list()[1], a);
    store.delete(&a.id);
    assert_eq!(store.list().len(), 1);
}

// --- the trainer -------------------------------------------------------------------------------

#[test]
fn trainer_events() {
    assert_eq!(train::parse_event(r#"CUBIC {"event": "status", "text": "Loading"}"#), Some(TEv::Status("Loading".into())));
    assert_eq!(train::parse_event(r#"CUBIC {"event": "progress", "step": 3, "total": 30}"#), Some(TEv::Progress { step: 3, total: 30 }));
    assert_eq!(train::parse_event(r#"CUBIC {"event": "loss", "step": 3, "total": 30, "loss": 1.25, "lr": 0.0002, "epoch": 0.1}"#), Some(TEv::Loss { step: 3, loss: 1.25, lr: 0.0002 }));
    assert_eq!(train::parse_event(r#"CUBIC {"event": "done", "adapter": "/x/adapter"}"#), Some(TEv::Done { adapter: "/x/adapter".into() }));
    assert_eq!(train::parse_event(r#"CUBIC {"event": "error", "text": "OOM"}"#), Some(TEv::Error("OOM".into())));
    assert_eq!(train::parse_event("Unsloth: Fast Llama patching"), None);
    assert_eq!(train::parse_event("CUBIC not json"), None);

    let mut l = Live::default();
    for line in ["banner", "", r#"CUBIC {"event": "loss", "step": 1, "loss": 2.0}"#, r#"CUBIC {"event": "progress", "step": 1, "total": 9}"#] {
        l.apply(line);
    }
    assert_eq!((l.step, l.total, l.losses.clone()), (1, 9, vec![(1, 2.0)]));
    assert_eq!(l.log, ["banner"]);
    for i in 0..300 {
        l.apply(&format!("line {i}"));
    }
    assert_eq!(l.log.len(), 200, "the log is capped");

    // The script prints what the parser reads.
    for ev in ["\"status\"", "\"progress\"", "\"loss\"", "\"done\"", "\"error\"", "CUBIC "] {
        assert!(train::TRAIN_PY.contains(ev), "{ev}");
    }
    let cfg = train::config_json("unsloth/x", Path::new("/d.jsonl"), Path::new("/out"), &Params::default());
    for k in ["base", "data", "out", "rank", "alpha", "lr", "epochs", "max_seq", "batch", "grad_accum", "seed", "replies_only"] {
        assert!(!cfg[k].is_null(), "{k}");
        assert!(train::TRAIN_PY.contains(&format!("cfg[\"{k}\"]")) || train::TRAIN_PY.contains(&format!("cfg.get(\"{k}\"")), "the script reads {k}");
    }
}

#[test]
fn trainer_script_is_valid_python() {
    let Ok(out) = std::process::Command::new("python3").args(["-c", "import ast, sys; ast.parse(sys.stdin.read())"]).stdin(std::process::Stdio::piped()).spawn().and_then(|mut c| {
        use std::io::Write;
        c.stdin.take().unwrap().write_all(train::TRAIN_PY.as_bytes())?;
        c.wait_with_output()
    }) else {
        return; // no Python here
    };
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn base_models_are_recognised() {
    let s = train::suggest_base;
    assert_eq!(s("Meta Llama 3.1 8B Instruct"), Some("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"));
    assert_eq!(s("llama3.1:8b"), Some("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"));
    assert_eq!(s("DeepSeek R1 0528 Qwen3 8B"), Some("unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"));
    assert_eq!(s("Qwen2.5 7B Instruct"), Some("unsloth/Qwen2.5-7B-Instruct-bnb-4bit"));
    assert_eq!(s("Qwen2.5 1.5B Instruct"), Some("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"));
    assert_eq!(s("Qwen3-14B"), Some("unsloth/Qwen3-14B-bnb-4bit"));
    assert_eq!(s("Qwen2.5 72B Instruct"), None, "72B is not 7B");
    assert_eq!(s("gemma-3-12b"), None);

    assert_eq!(
        train::parse_env_info("noise\nCUBIC {\"torch\": \"2.9.1+cu128\", \"cuda\": true, \"gpu\": \"RTX\", \"vram\": 8589934592, \"unsloth\": \"2026.10.1\"}\n"),
        Some(train::EnvInfo { torch: "2.9.1+cu128".into(), cuda: true, gpu: "RTX".into(), vram: 8 << 30, unsloth: "2026.10.1".into() })
    );
    assert_eq!(train::parse_env_info("Traceback…"), None);
}

/// A stand-in "python" that prints what the real trainer would.
#[cfg(unix)]
fn fake_python(dir: &TempDir, body: &str) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("python");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

#[cfg(unix)]
#[test]
fn training_run_end_to_end_with_a_stand_in_trainer() {
    let dir = TempDir::new("train-run");
    let out = dir.join("run");
    std::fs::create_dir_all(out.join("adapter")).unwrap();
    // What Unsloth's save_pretrained leaves behind.
    std::fs::write(out.join("adapter/adapter_model.safetensors"), lora::write_safetensors(&adapter(2))).unwrap();
    std::fs::write(out.join("adapter/adapter_config.json"), r#"{"r": 2, "lora_alpha": 2, "base_model_name_or_path": "unsloth/tiny"}"#).unwrap();
    std::fs::write(out.join("train.json"), "{}").unwrap();
    let py = fake_python(
        &dir,
        &format!(
            "[ \"$HF_TOKEN\" = secret ] || echo 'CUBIC {{\"event\": \"error\", \"text\": \"no token\"}}'\n\
             echo 'Unsloth banner'\n\
             echo 'CUBIC {{\"event\": \"status\", \"text\": \"Training\"}}'\n\
             echo 'CUBIC {{\"event\": \"loss\", \"step\": 1, \"total\": 2, \"loss\": 2.5}}'\n\
             echo 'CUBIC {{\"event\": \"progress\", \"step\": 1, \"total\": 2}}'\n\
             echo 'warning on stderr' 1>&2\n\
             echo 'CUBIC {{\"event\": \"loss\", \"step\": 2, \"total\": 2, \"loss\": 1.1}}'\n\
             echo 'CUBIC {{\"event\": \"done\", \"adapter\": \"{}\"}}'",
            out.join("adapter").display()
        ),
    );
    let live: Arc<Mutex<Live>> = Default::default();
    let task = Task::new("x");
    let peft = train::run(&py, &out, "secret", &live, &task).unwrap();
    assert_eq!(peft, out.join("adapter"));
    assert!(out.join("train.py").is_file(), "the script is written next to its config");
    {
        let l = live.lock().unwrap();
        assert_eq!(l.losses, [(1, 2.5), (2, 1.1)]);
        assert!(l.log.iter().any(|x| x == "warning on stderr") && l.log.iter().any(|x| x == "Unsloth banner"));
    }
    assert_eq!(task.status(), "Training");

    // Converted for the GGUF it was trained for, then listed.
    let store = Adapters::new(dir.join("adapters"));
    let base_bytes = base_model("llama");
    let c = store.convert("a1", &peft, &Gguf::parse(&base_bytes, None).unwrap()).unwrap();
    assert_eq!(c.pairs, 3);
    let meta = Adapter { id: "a1".into(), name: "Pirate".into(), model_key: "m".into(), losses: vec![(1, 2.5)], ..Default::default() };
    store.save(&meta).unwrap();
    store.save(&Adapter { id: "a2".into(), ..Default::default() }).unwrap();
    assert_eq!(store.list(), [meta], "an adapter without its GGUF isn't listed");
    store.delete("../adapters");
    assert!(dir.join("adapters").is_dir(), "ids can't reach outside the folder");
    store.delete("a1");
    assert!(store.list().is_empty());

    // Errors: the trainer's own message wins over the exit status.
    let py = fake_python(&dir, "echo 'CUBIC {\"event\": \"error\", \"text\": \"The GPU ran out of memory.\"}'; exit 1");
    let e = train::run(&py, &out, "", &Default::default(), &Task::new("x")).unwrap_err();
    assert_eq!(e, "The GPU ran out of memory.");
    let py = fake_python(&dir, "echo 'Traceback: import failed' 1>&2; exit 3");
    let e = train::run(&py, &out, "", &Default::default(), &Task::new("x")).unwrap_err();
    assert!(e.contains("Traceback: import failed"), "{e}");
    let py = fake_python(&dir, "true");
    assert!(train::run(&py, &out, "", &Default::default(), &Task::new("x")).unwrap_err().contains("without saving"));
}

#[cfg(unix)]
#[test]
fn stopping_kills_the_process() {
    let task = Task::new("x");
    let t2 = task.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        t2.stop();
    });
    let started = std::time::Instant::now();
    let r = train::run_logged(std::process::Command::new("sleep").arg("20"), &task, &mut |_| {});
    assert_eq!(r, Err("Stopped".into()));
    assert!(started.elapsed().as_secs() < 5);
}

#[cfg(unix)]
#[test]
fn environment_check() {
    let dir = TempDir::new("env");
    let env = Env::new(dir.join("env"));
    assert!(!env.exists());
    assert!(env.check().unwrap_err().contains("Not set up"));
    let bin = dir.join("env/venv/bin");
    std::fs::create_dir_all(&bin).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(bin.join("python"), "#!/bin/sh\necho 'CUBIC {\"torch\": \"2.9\", \"cuda\": false, \"gpu\": \"\", \"vram\": 0, \"unsloth\": \"\"}'\n").unwrap();
        std::fs::set_permissions(bin.join("python"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    assert_eq!(env.python(), bin.join("python"));
    let i = env.check().unwrap();
    assert!(!i.cuda && i.torch == "2.9");
}

// --- serving the adapter -----------------------------------------------------------------------

#[test]
fn llama_server_gets_the_adapter() {
    let preset = RunPreset::default();
    let args = Server::args_for(&Launch {
        bin_dir: Path::new("/x"),
        model_key: "k",
        model_name: "m",
        model: Path::new("/m.gguf"),
        mmproj: None,
        preset: &preset,
        port: 1,
        steer: Some(Path::new("/s.gguf")),
        lora: Some(Path::new("C:\\a\\lora.gguf")),
    });
    let i = args.iter().position(|a| a == "--lora").unwrap();
    assert_eq!(args[i + 1], "C:\\a\\lora.gguf");
    assert!(args.iter().any(|a| a == "--control-vector"), "steering and an adapter together");

    let old: RunPreset = serde_json::from_value(json!({"ctx": 2048})).unwrap();
    assert_eq!(old.lora, None);
    let p = RunPreset { lora: Some("a1".into()), ..Default::default() };
    assert_eq!(serde_json::from_str::<RunPreset>(&serde_json::to_string(&p).unwrap()).unwrap(), p);
}

#[test]
fn before_and_after_switch_the_adapter_per_request() {
    let port = mock_http(|req| {
        let body: Value = serde_json::from_str(req.split("\r\n\r\n").nth(1).unwrap_or("")).unwrap_or_default();
        let scale = body["lora"][0]["scale"].as_f64().unwrap_or(-1.0);
        assert_eq!(body["lora"][0]["id"], 0);
        assert_eq!(body["temperature"], 0.0);
        let b = json!({"content": format!(" scale {scale} ")}).to_string();
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())
    });
    let root = format!("http://127.0.0.1:{port}");
    assert_eq!(train::complete_with_lora(&root, "p", 10, 0.0).unwrap(), "scale 0");
    assert_eq!(train::complete_with_lora(&root, "p", 10, 1.0).unwrap(), "scale 1");
}

#[test]
fn loss_curve_smoothing_and_durations() {
    use crate::ui::tune::{smoothed, took};
    // A short run is drawn raw: there's nothing to smooth.
    let short: Vec<(u32, f32)> = (1..=18).map(|s| (s, 4.0 / s as f32)).collect();
    assert!(smoothed(&short).is_empty());
    // A long noisy run: the average is centred, so it ends at the real last point instead of lagging
    // above it, keeps the first point, and is calmer than the raw curve.
    let long: Vec<(u32, f32)> = (1..=200).map(|s| (s, 3.0 * (-(s as f32) / 40.0).exp() + if s % 2 == 0 { 0.2 } else { -0.2 })).collect();
    let sm = smoothed(&long);
    assert_eq!(sm.len(), long.len());
    assert_eq!(sm.first(), long.first());
    assert_eq!(sm.last(), long.last());
    let wiggle = |c: &[(u32, f32)]| c.windows(2).map(|w| (w[1].1 - w[0].1).abs()).sum::<f32>();
    assert!(wiggle(&sm) < wiggle(&long) / 3.0, "smoother");
    let mid = sm[100].1;
    let truth = 3.0 * (-101f32 / 40.0).exp();
    assert!((mid - truth).abs() < 0.05, "centred: {mid} vs {truth}");

    assert_eq!(took(44.0), "44 s");
    assert_eq!(took(89.4), "89 s");
    assert_eq!(took(412.0), "7 min");
    assert_eq!(took(4370.0), "73 min");
    assert_eq!(took(5400.0), "1 h 30 min");
}
