//! A fake ComfyUI for the media and Bots tests: one anime workflow ("illustrious"), runs reported
//! over its WebSocket node by node, a PNG saved per prompt ("FAIL" in the prompt fails it).
#![allow(dead_code)]

use cubic_gateway::http::Conn;
use cubic_gateway::media;
use serde_json::{json, Value};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const WORKFLOW: &str = r#"{"nodes": [
 {"id": 14, "type": "CheckpointLoaderSimple", "mode": 0, "outputs": [{"type": "MODEL"}, {"type": "CLIP"}, {"type": "VAE"}], "widgets_values": ["illustrious.safetensors"]},
 {"id": 10, "type": "CLIPTextEncode", "inputs": [{"name": "clip", "type": "CLIP", "link": 2}], "widgets_values": ["a cat"]},
 {"id": 11, "type": "CLIPTextEncode", "title": "Negative", "inputs": [{"name": "clip", "type": "CLIP", "link": 3}], "widgets_values": ["lowres, bad hands"]},
 {"id": 13, "type": "EmptyLatentImage", "widgets_values": [1024, 1024, 1]},
 {"id": 12, "type": "KSampler", "inputs": [{"name": "model", "link": 1}, {"name": "positive", "link": 4}, {"name": "negative", "link": 5}, {"name": "latent_image", "link": 6}],
  "widgets_values": [42, "randomize", 28, 6.5, "euler", "normal", 1.0]},
 {"id": 15, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 7}, {"name": "vae", "link": 8}]},
 {"id": 16, "type": "SaveImage", "inputs": [{"name": "images", "link": 9}], "widgets_values": ["ComfyUI"]},
 {"id": 17, "type": "PreviewImage", "inputs": [{"name": "images", "link": 10}]},
 {"id": 18, "type": "Note", "widgets_values": ["just a note"]}
], "links": [[1, 14, 0, 12, 0, "MODEL"], [2, 14, 1, 10, 0, "CLIP"], [3, 14, 1, 11, 0, "CLIP"], [4, 10, 0, 12, 1, "CONDITIONING"], [5, 11, 0, 12, 2, "CONDITIONING"],
  [6, 13, 0, 12, 3, "LATENT"], [7, 12, 0, 15, 0, "LATENT"], [8, 14, 2, 15, 1, "VAE"], [9, 15, 0, 16, 0, "IMAGE"], [10, 15, 0, 17, 0, "IMAGE"]]}"#;

pub const INFO: &str = r#"{
 "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["illustrious.safetensors", "other.safetensors"], {}]}}, "input_order": {"required": ["ckpt_name"]}},
 "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": true}], "clip": ["CLIP"]}}, "input_order": {"required": ["text", "clip"]}},
 "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}}, "input_order": {"required": ["width", "height", "batch_size"]}},
 "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {"control_after_generate": true, "max": 18446744073709551615}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
   "sampler_name": [["euler"], {}], "scheduler": [["normal"], {}], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"], "denoise": ["FLOAT", {}]}},
   "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "denoise"]}},
 "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}, "input_order": {"required": ["samples", "vae"]}},
 "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}, "input_order": {"required": ["images", "filename_prefix"]}},
 "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}, "input_order": {"required": ["images"]}}
}"#;

pub const PNG: &[u8] = b"\x89PNG\r\n\x1a\nfake picture bytes";

// --- A fake ComfyUI ---------------------------------------------------------------------------------

#[derive(Default)]
pub struct Comfy {
    pub prompts: Vec<Value>,
    /// (client id, prompt id, prompt) waiting for the WebSocket to report on.
    pub pending: Vec<(String, String, Value)>,
    pub calls: Vec<String>,
}

pub type Shared = Arc<(Mutex<Comfy>, Condvar)>;

pub fn frame(op: u8, payload: &[u8], fin: bool) -> Vec<u8> {
    let mut f = vec![if fin { 0x80 } else { 0 } | op];
    match payload.len() {
        n if n < 126 => f.push(n as u8),
        n if n < 65536 => {
            f.push(126);
            f.extend((n as u16).to_be_bytes());
        }
        n => {
            f.push(127);
            f.extend((n as u64).to_be_bytes());
        }
    }
    f.extend_from_slice(payload);
    f
}

pub fn fake_comfy(dir: PathBuf) -> (u16, Shared) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let shared: Shared = Default::default();
    let s = shared.clone();
    std::thread::spawn(move || {
        for tcp in l.incoming().flatten() {
            let (s, dir) = (s.clone(), dir.clone());
            std::thread::spawn(move || comfy_conn(tcp, s, dir));
        }
    });
    (port, shared)
}

pub fn reply(c: &mut Conn<TcpStream>, v: &Value) {
    let body = v.to_string();
    let _ = c.raw(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes());
}

pub fn comfy_conn(tcp: TcpStream, s: Shared, dir: PathBuf) {
    let mut c = Conn::new(tcp);
    while let Ok(Some(req)) = c.next_request() {
        let body: Value = match req.content_length() {
            Ok(Some(n)) if n > 0 => serde_json::from_slice(&c.read_body(n).unwrap()).unwrap_or(Value::Null),
            _ => Value::Null,
        };
        s.0.lock().unwrap().calls.push(format!("{} {}", req.method, req.path()));
        match (req.method.as_str(), req.path()) {
            ("GET", "/system_stats") => reply(&mut c, &json!({"system": {}})),
            ("GET", "/object_info") => reply(&mut c, &serde_json::from_str(INFO).unwrap()),
            ("POST", "/prompt") => {
                let prompt = body["prompt"].clone();
                let prefix = prompt["16"]["inputs"]["filename_prefix"].as_str().unwrap().to_string();
                std::fs::write(dir.join("output").join(format!("{prefix}_00001_.png")), PNG).unwrap();
                let mut st = s.0.lock().unwrap();
                st.prompts.push(prompt.clone());
                let pid = format!("p{}", st.prompts.len());
                st.pending.push((body["client_id"].as_str().unwrap().to_string(), pid.clone(), prompt));
                s.1.notify_all();
                drop(st);
                reply(&mut c, &json!({"prompt_id": pid, "number": 1}));
            }
            ("GET", p) if p.starts_with("/history/") => {
                let pid = &p["/history/".len()..];
                let st = s.0.lock().unwrap();
                let n: usize = pid[1..].parse().unwrap();
                let prefix = st.prompts[n - 1]["16"]["inputs"]["filename_prefix"].as_str().unwrap().to_string();
                let (sub, name) = prefix.split_once('/').unwrap();
                reply(&mut c, &json!({pid: {"outputs": {
                    "16": {"images": [{"filename": format!("{name}_00001_.png"), "subfolder": sub, "type": "output"}]},
                    // Never followed: outside ComfyUI's folders.
                    "99": {"images": [{"filename": "passwd", "subfolder": "../../../../etc", "type": "output"}], "text": ["not a file list"]},
                }}}));
            }
            ("GET", "/ws") => {
                let client = req.query("clientId").unwrap();
                let _ = c.raw(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n");
                ws_session(c, &s, &client);
                return;
            }
            _ => reply(&mut c, &json!({})),
        }
    }
}

/// Reports a run the way ComfyUI does: node by node, sampler steps, a preview, then the end.
pub fn ws_session(mut c: Conn<TcpStream>, s: &Shared, client: &str) {
    let (pid, prompt) = {
        let mut st = s.0.lock().unwrap();
        loop {
            if let Some(i) = st.pending.iter().position(|p| p.0 == client) {
                let p = st.pending.remove(i);
                break (p.1, p.2);
            }
            st = s.1.wait(st).unwrap();
        }
    };
    let text = |v: Value| frame(1, v.to_string().as_bytes(), true);
    let fail = prompt["10"]["inputs"]["text"].as_str().unwrap_or("").contains("FAIL");
    let mut msgs: Vec<Vec<u8>> = vec![
        text(json!({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}})),
        text(json!({"type": "executing", "data": {"node": "14", "prompt_id": pid}})),
        text(json!({"type": "executing", "data": {"node": "99", "prompt_id": "someone-else"}})),
        text(json!({"type": "execution_cached", "data": {"nodes": ["10"], "prompt_id": pid}})),
        text(json!({"type": "executing", "data": {"node": "12", "prompt_id": pid}})),
        frame(9, b"ping", true),
        frame(2, &[&[0, 0, 0, 1, 0, 0, 0, 2][..], b"PREVIEW-JPEG"].concat(), true),
    ];
    for v in 1..=4 {
        // One message in two fragments.
        let m = json!({"type": "progress", "data": {"value": v, "max": 4, "prompt_id": pid, "node": "12"}}).to_string();
        let (a, b) = m.as_bytes().split_at(10);
        msgs.push([frame(1, a, false), frame(0, b, true)].concat());
    }
    msgs.push(text(json!({"type": "executing", "data": {"node": "16", "prompt_id": pid}})));
    msgs.push(if fail {
        text(json!({"type": "execution_error", "data": {"prompt_id": pid, "exception_message": "  out of cheese  "}}))
    } else {
        text(json!({"type": "executing", "data": {"node": null, "prompt_id": pid}}))
    });
    for m in msgs {
        if c.raw(&m).is_err() {
            return;
        }
        std::thread::sleep(Duration::from_millis(60));
    }
    // Stay until the gateway hangs up (reading its pong and close frames).
    let _ = std::io::copy(&mut c.r, &mut std::io::sink());
}

/// ComfyUI's folders under root/comfy with the workflow, the fake started, and media paths for it.
pub fn setup(root: &Path) -> (media::Paths, Shared) {
    let comfy_dir = root.join("comfy");
    for d in ["user/default/workflows", "output/cubicai", "input/cubicai", "temp"] {
        std::fs::create_dir_all(comfy_dir.join(d)).unwrap();
    }
    std::fs::write(comfy_dir.join("user/default/workflows/my_illustrious_anime.json"), WORKFLOW).unwrap();
    std::fs::create_dir_all(root.join("voices")).unwrap();
    let (port, comfy) = fake_comfy(comfy_dir.clone());
    let paths = media::Paths {
        comfy_dir,
        comfy_port: port,
        save_dir: root.join("saved"),
        cache_dir: root.join("cache"),
        voices_dir: root.join("voices"),
        idle_stop: 600,
        // A unit that doesn't exist: nothing on this PC is started or stopped by the test.
        unit: "cubicai-test-does-not-exist.service".into(),
    };
    (paths, comfy)
}
