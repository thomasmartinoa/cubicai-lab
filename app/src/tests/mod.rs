//! Unit and integration tests for the app (offline; the live ones are in `shots.rs` and ignored).

mod engine;
mod llm;
mod steer;
mod store;
mod xray;

use cubic_gguf::{GgmlType, Writer};
use std::path::PathBuf;

/// A fresh, empty folder under the system temp dir, removed when dropped.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("cubicai-lab-test-{name}-{}-{}", std::process::id(), crate::store::new_id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    pub fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub const EMBD: u64 = 256;
pub const LAYERS: usize = 8;
pub const VOCAB: u64 = 1000;

/// A tiny model in the shape of a real one: 8 layers, Q8_0 weights, every other layer with a
/// 128-token sliding window; with `experts`, 8 experts per layer of which 2 run per token.
pub fn tiny_model(experts: bool) -> Vec<u8> {
    let arch = if experts { "qwen3moe" } else { "llama" };
    let mut w = Writer::new();
    let k = |s: &str| format!("{arch}.{s}");
    w.str("general.architecture", arch)
        .u32("general.file_type", 7)
        .u32(&k("block_count"), LAYERS as u32)
        .u32(&k("embedding_length"), EMBD as u32)
        .u32(&k("feed_forward_length"), 512)
        .u32(&k("attention.head_count"), 8)
        .u32(&k("attention.head_count_kv"), 2)
        .u32(&k("context_length"), 32768)
        .u32(&k("attention.sliding_window"), 128)
        .bools(&k("attention.sliding_window_pattern"), &(0..LAYERS).map(|i| i % 2 == 0).collect::<Vec<_>>())
        .u32(&k("vocab_size"), VOCAB as u32);
    if experts {
        w.u32(&k("expert_count"), 8).u32(&k("expert_used_count"), 2).u32(&k("expert_feed_forward_length"), 64);
    }
    let q8 = |n: u64| -> Vec<u8> {
        (0..n / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 * (1 + b % 5) as f32).to_le_bytes().to_vec();
                blk.extend((0..32).map(|j| ((j * 7 + b) % 255) as u8));
                blk
            })
            .collect()
    };
    w.tensor_raw("token_embd.weight", &[EMBD, VOCAB], GgmlType::Q8_0, q8(EMBD * VOCAB));
    for l in 0..LAYERS {
        let b = |n: &str| format!("blk.{l}.{n}.weight");
        w.tensor_f32(&b("attn_norm"), &[EMBD], |_| 1.0)
            .tensor_raw(&b("attn_q"), &[EMBD, 256], GgmlType::Q8_0, q8(EMBD * 256))
            .tensor_raw(&b("attn_k"), &[EMBD, 64], GgmlType::Q8_0, q8(EMBD * 64))
            .tensor_raw(&b("attn_v"), &[EMBD, 64], GgmlType::Q8_0, q8(EMBD * 64))
            .tensor_raw(&b("attn_output"), &[256, EMBD], GgmlType::Q8_0, q8(EMBD * 256))
            .tensor_f32(&b("ffn_norm"), &[EMBD], |_| 1.0);
        if experts {
            w.tensor_f32(&b("ffn_gate_inp"), &[EMBD, 8], |_| 0.1)
                .tensor_raw(&b("ffn_gate_exps"), &[EMBD, 64, 8], GgmlType::Q8_0, q8(EMBD * 64 * 8))
                .tensor_raw(&b("ffn_up_exps"), &[EMBD, 64, 8], GgmlType::Q8_0, q8(EMBD * 64 * 8))
                .tensor_raw(&b("ffn_down_exps"), &[64, EMBD, 8], GgmlType::Q8_0, q8(EMBD * 64 * 8));
        } else {
            w.tensor_raw(&b("ffn_gate"), &[EMBD, 512], GgmlType::Q8_0, q8(EMBD * 512))
                .tensor_raw(&b("ffn_up"), &[EMBD, 512], GgmlType::Q8_0, q8(EMBD * 512))
                .tensor_raw(&b("ffn_down"), &[512, EMBD], GgmlType::Q8_0, q8(EMBD * 512));
        }
    }
    w.tensor_f32("output_norm.weight", &[EMBD], |_| 1.0);
    w.tensor_raw("output.weight", &[EMBD, VOCAB], GgmlType::Q8_0, q8(EMBD * VOCAB));
    w.finish()
}

/// Writes a tiny model to `dir/name` and reads it the way the library does.
pub fn tiny_info(dir: &TempDir, experts: bool) -> crate::engine::library::Info {
    let path = dir.join(if experts { "moe.gguf" } else { "dense.gguf" });
    std::fs::write(&path, tiny_model(experts)).unwrap();
    let m = crate::engine::library::Model {
        key: path.to_string_lossy().into(),
        name: "tiny".into(),
        parts: vec![path],
        size: 0,
        mmproj: None,
        source: crate::engine::library::Source::Folder,
    };
    crate::engine::library::read_info(&m).unwrap()
}

/// A one-shot HTTP server on a free port: answers each connection with `respond(request)`.
pub fn mock_http<R: Into<Vec<u8>>>(respond: impl Fn(&str) -> R + Send + 'static) -> u16 {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for mut s in listener.incoming().flatten() {
            let mut buf = vec![0u8; 65536];
            let mut got = 0;
            // Read headers, then the body by Content-Length.
            loop {
                let n = s.read(&mut buf[got..]).unwrap_or(0);
                if n == 0 {
                    break;
                }
                got += n;
                let text = String::from_utf8_lossy(&buf[..got]);
                if let Some(h) = text.find("\r\n\r\n") {
                    let len = text[..h]
                        .lines()
                        .find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                        .unwrap_or(0);
                    if got >= h + 4 + len {
                        break;
                    }
                }
            }
            let req = String::from_utf8_lossy(&buf[..got]).to_string();
            let _ = s.write_all(&respond(&req).into());
        }
    });
    port
}

pub fn sse_response(events: &[&str]) -> String {
    let mut out = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_string();
    for e in events {
        out += &format!("data: {e}\n\n");
    }
    out
}
