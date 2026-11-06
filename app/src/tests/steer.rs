use super::{mock_http, TempDir};
use crate::engine::server::{Launch, Server};
use crate::engine::steer::{self, Built, Concept, Store, Vector};
use crate::engine::Task;
use crate::store::{RunPreset, SteerSetting};
use cubic_gguf::steer::read_control_vector;
use serde_json::{json, Value};
use std::path::Path;

const LAYERS: usize = 4;
const EMBD: u64 = 64;

/// A small Llama with random weights (the shared tiny model's patterned weights make attention
/// collapse onto one token, so every prompt would look alike). Vocabulary: one token per byte.
fn small_llama() -> Vec<u8> {
    let rnd = |seed: u64| {
        move |i: u64| {
            let mut x = (i + 1).wrapping_mul(0x9E3779B97F4A7C15) ^ seed.wrapping_mul(0xBF58476D1CE4E5B9);
            x ^= x >> 31;
            x = x.wrapping_mul(0x94D049BB133111EB);
            ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
        }
    };
    let mut w = cubic_gguf::Writer::new();
    w.str("general.architecture", "llama")
        .u32("llama.block_count", LAYERS as u32)
        .u32("llama.embedding_length", EMBD as u32)
        .u32("llama.attention.head_count", 4)
        .u32("llama.attention.head_count_kv", 2);
    w.tensor_f32("token_embd.weight", &[EMBD, 256], rnd(1));
    for l in 0..LAYERS as u64 {
        let b = |n: &str| format!("blk.{l}.{n}.weight");
        w.tensor_f32(&b("attn_norm"), &[EMBD], |_| 1.0)
            .tensor_f32(&b("attn_q"), &[EMBD, 64], rnd(10 * l + 2))
            .tensor_f32(&b("attn_k"), &[EMBD, 32], rnd(10 * l + 3))
            .tensor_f32(&b("attn_v"), &[EMBD, 32], rnd(10 * l + 4))
            .tensor_f32(&b("attn_output"), &[64, EMBD], rnd(10 * l + 5))
            .tensor_f32(&b("ffn_norm"), &[EMBD], |_| 1.0)
            .tensor_f32(&b("ffn_gate"), &[EMBD, 128], rnd(10 * l + 6))
            .tensor_f32(&b("ffn_up"), &[EMBD, 128], rnd(10 * l + 7))
            .tensor_f32(&b("ffn_down"), &[128, EMBD], rnd(10 * l + 8));
    }
    w.tensor_f32("output_norm.weight", &[EMBD], |_| 1.0);
    w.tensor_f32("output.weight", &[EMBD, 256], rnd(99));
    w.finish()
}

fn body(req: &str) -> Value {
    serde_json::from_str(req.split("\r\n\r\n").nth(1).unwrap_or("")).unwrap_or(Value::Null)
}

fn ok(v: Value) -> String {
    let b = v.to_string();
    format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())
}

/// A stand-in llama-server: templating wraps the user text, tokenising maps bytes to ids.
fn fake_llama() -> String {
    let port = mock_http(|req| {
        let b = body(req);
        if req.starts_with("POST /apply-template") {
            ok(json!({"prompt": format!("<s>[user] {} [bot] ", b["messages"][0]["content"].as_str().unwrap_or(""))}))
        } else if req.starts_with("POST /tokenize") {
            assert_eq!(b["parse_special"], true, "template markers must become special tokens");
            let ids: Vec<u32> = b["content"].as_str().unwrap().bytes().map(u32::from).collect();
            ok(json!({"tokens": ids}))
        } else if req.starts_with("POST /completion") {
            assert_eq!(b["temperature"], 0.0, "comparisons are greedy");
            ok(json!({"content": "  Sunny and bright!\n", "stop": true}))
        } else {
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn concept() -> Concept {
    steer::presets().remove(0)
}

#[test]
fn contrast_prompts_differ_only_in_the_trait() {
    let c = concept();
    let pairs = steer::contrast_pairs(&c);
    assert_eq!(pairs.len(), steer::QUESTIONS.len() * steer::OPENERS.len());
    for (pos, neg, opener) in &pairs {
        let (p, n) = (pos[0]["content"].as_str().unwrap(), neg[0]["content"].as_str().unwrap());
        assert!(p.contains(&c.positive) && !p.contains(&c.negative));
        assert!(n.contains(&c.negative));
        assert_eq!(p.replace(&c.positive, "X"), n.replace(&c.negative, "X"), "same question around the trait");
        assert!(steer::OPENERS.contains(opener));
    }
    // Labels stay in plain text the UI font can draw.
    assert!(steer::presets().iter().all(|c| c.name.is_ascii()));
}

#[test]
fn tokenize_and_complete_talk_to_llama_server() {
    let root = fake_llama();
    assert_eq!(steer::tokenize(&root, "AB").unwrap(), vec![65, 66]);
    assert_eq!(steer::complete(&root, "x", 10).unwrap(), "Sunny and bright!", "trimmed");
    let down = mock_http(|_| "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 30\r\nConnection: close\r\n\r\n{\"error\":{\"message\":\"Loading\"}}");
    let e = steer::tokenize(&format!("http://127.0.0.1:{down}"), "x").unwrap_err();
    assert!(e.contains("503") && e.contains("Loading"), "{e}");
}

#[test]
fn builds_a_vector_on_a_tiny_model() {
    let dir = TempDir::new("steer-build");
    let path = dir.join("m.gguf");
    std::fs::write(&path, small_llama()).unwrap();
    let root = fake_llama();
    let task = Task::new("x");
    let b = steer::build(&root, "key", "tiny", std::slice::from_ref(&path), &concept(), &task).unwrap();
    let v = &b.vector;
    assert_eq!((v.n_layer, v.n_embd), (LAYERS, EMBD as usize));
    assert_eq!(b.layers.len(), LAYERS);
    assert_eq!(v.separation.len(), LAYERS);
    assert!(v.layer_start >= 1 && v.layer_start <= v.layer_end && v.layer_end < LAYERS, "never steers layer 0");
    assert_eq!(v.proj_pos[0].len(), steer::contrast_pairs(&concept()).len());
    assert!(v.norms.iter().all(|n| *n > 0.0), "the two ends differ after every layer");
    assert!(v.separation.iter().all(|d| d.is_finite()));
    assert!(v.tokens > 0 && v.model_key == "key");
    assert_eq!(task.fraction(), 1.0);

    // Refuses empty concepts, and stops when asked.
    let blank = Concept { negative: " ".into(), ..concept() };
    assert!(steer::build(&root, "k", "t", std::slice::from_ref(&path), &blank, &Task::new("x")).is_err());
    let stopped = Task::new("x");
    stopped.stop();
    assert_eq!(steer::build(&root, "k", "t", &[path], &concept(), &stopped).err().as_deref(), Some("Stopped"));
}

fn built() -> Built {
    let layers: Vec<Vec<f32>> = (0..6).map(|l| vec![l as f32, -(l as f32), 0.5]).collect();
    let vector = Vector { id: crate::store::new_id(), name: "Test".into(), model_key: "m".into(), n_layer: 6, n_embd: 3, layer_start: 2, layer_end: 4, separation: vec![0.0, 1.0, 3.0, 4.0, 3.0, 1.0], created: 5, ..Default::default() };
    Built { vector, layers }
}

#[test]
fn baked_vectors_hold_only_the_window_scaled() {
    let b = built();
    let file = steer::bake(&b.layers, 2, 4, 2.0);
    let back = read_control_vector(&file).unwrap();
    let scale = steer::scale_for(2.0, 2, 4);
    assert!((scale - 2.0 * steer::STRENGTH_UNIT / 3.0).abs() < 1e-6, "strength spread over 3 layers");
    assert!(back[..2].iter().all(Vec::is_empty), "nothing before the window");
    for (got, want) in back[2..=4].iter().zip(&b.layers[2..=4]) {
        assert_eq!(*got, want.iter().map(|x| x * scale).collect::<Vec<_>>());
    }
    assert_eq!(back.len(), 5, "nothing after the window");
    // llama.cpp rejects direction.0; a window starting at 0 starts at 1.
    let g = cubic_gguf::Gguf::parse(&steer::bake(&b.layers, 0, 1, 1.0), None).unwrap();
    assert_eq!(g.tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["direction.1"]);
    assert_eq!(g.get_str("general.architecture"), Some("controlvector"));
    // Negative strength points the other way.
    let neg = read_control_vector(&steer::bake(&b.layers, 2, 4, -2.0)).unwrap();
    assert_eq!(neg[3][0], -back[3][0]);
}

#[test]
fn store_keeps_vectors_and_their_baked_files() {
    let dir = TempDir::new("steer-store");
    let store = Store::new(dir.join("steering"));
    let mut a = built();
    a.vector.created = 1;
    let b = built();
    store.save(&a).unwrap();
    store.save(&b).unwrap();
    let list = store.list();
    assert_eq!(list.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), [b.vector.id.as_str(), a.vector.id.as_str()], "newest first");
    assert_eq!(store.layers(&a.vector.id).unwrap()[1..], a.layers[1..]);

    let p = store.bake(&b.vector, 1.0).unwrap();
    assert!(p.is_file() && p.starts_with(dir.join("steering").join("baked")));
    let name = p.file_name().unwrap().to_string_lossy().to_string();
    assert!(!name.contains(':') && !name.contains(','), "llama.cpp splits on these: {name}");
    assert_eq!(store.bake(&b.vector, 1.0).unwrap(), p, "baked once per strength");
    assert_ne!(store.baked_path(&b.vector, -1.0), p);
    let mut moved = b.vector.clone();
    moved.layer_end = 5;
    assert_ne!(store.baked_path(&moved, 1.0), p, "a new window is a new file");
    store.update(&moved);
    assert_eq!(store.list().iter().find(|v| v.id == moved.id).unwrap().layer_end, 5);

    store.delete(&b.vector.id);
    assert!(!p.exists(), "its baked files go with it");
    assert_eq!(store.list().len(), 1);
    // A vector whose directions file is gone isn't listed.
    std::fs::remove_file(dir.join("steering").join(format!("{}.gguf", a.vector.id))).unwrap();
    assert!(store.list().is_empty());
}

#[test]
fn server_gets_the_control_vector() {
    let preset = RunPreset { extra_args: "--foo".into(), ..Default::default() };
    let args = |steer: Option<&Path>| {
        Server::args_for(&Launch { bin_dir: Path::new("/x"), model_key: "k", model_name: "m", model: Path::new("/m.gguf"), mmproj: None, preset: &preset, port: 1, steer, lora: None })
    };
    assert!(!args(None).iter().any(|a| a.starts_with("--control-vector")));
    let a = args(Some(Path::new("C:\\data\\v_1_2_100.gguf")));
    let i = a.iter().position(|x| x == "--control-vector").unwrap();
    assert_eq!(a[i + 1], "C:\\data\\v_1_2_100.gguf", "plain flag: Windows paths keep their colon");
    assert!(i < a.iter().position(|x| x == "--foo").unwrap(), "the user's own arguments come last");
}

#[test]
fn presets_remember_steering_and_old_ones_still_load() {
    let old: RunPreset = serde_json::from_value(json!({"ctx": 4096, "kv_type": "q8_0"})).unwrap();
    assert_eq!(old.steer, None);
    let p = RunPreset { steer: Some(SteerSetting { vector: "v1".into(), strength: -0.5 }), ..Default::default() };
    let back: RunPreset = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(back, p);
}
