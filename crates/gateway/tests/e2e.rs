//! The gateway end to end, the way the phone uses it: TLS 1.3 to a pinned certificate, the token,
//! keep-alive, a streamed chat with web search. Ollama and SearXNG are a fake on this PC.

mod common;

use common::*;
use cubic_gateway::http::Conn;
use cubic_gateway::{identity, server, Config};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// --- A fake Ollama + SearXNG ----------------------------------------------------------------------

#[derive(Default)]
struct Seen {
    chats: Vec<Value>,
    /// How many lines of the slow stream were written before the gateway hung up.
    slow_written: usize,
    slow_closed: bool,
}

fn fake_backend() -> (String, Arc<Mutex<Seen>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let seen: Arc<Mutex<Seen>> = Default::default();
    let s = seen.clone();
    std::thread::spawn(move || {
        for tcp in l.incoming().flatten() {
            let s = s.clone();
            std::thread::spawn(move || backend_conn(tcp, s));
        }
    });
    (url, seen)
}

fn reply(c: &mut Conn<TcpStream>, v: &Value) {
    let body = v.to_string();
    let _ = c.raw(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes());
}

fn backend_conn(tcp: TcpStream, seen: Arc<Mutex<Seen>>) {
    let mut c = Conn::new(tcp);
    while let Ok(Some(req)) = c.next_request() {
        let body: Value = match req.content_length() {
            Ok(Some(n)) if n > 0 => serde_json::from_slice(&c.read_body(n).unwrap()).unwrap_or(Value::Null),
            _ => Value::Null,
        };
        match req.path() {
            "/api/tags" => reply(&mut c, &json!({"models": [{"name": "gemma4:e4b", "size": 6u64 << 30}, {"name": "nomic-embed-text:latest", "size": 1u64 << 28}]})),
            "/api/ps" => reply(&mut c, &json!({"models": []})),
            "/api/show" => {
                let caps = if body["model"] == "nomic-embed-text:latest" { json!(["embedding"]) } else { json!(["completion", "vision", "thinking", "tools"]) };
                reply(&mut c, &json!({"capabilities": caps, "parameters": "", "model_info": {"gemma3.context_length": 131072}}));
            }
            "/api/chat" if body["stream"] == false => reply(&mut c, &json!({"message": {"role": "assistant", "content": "<think>hm</think>\"rust gateway news\""}})),
            "/api/chat" => {
                seen.lock().unwrap().chats.push(body.clone());
                let _ = c.raw(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n");
                if body["model"] == "slow" {
                    for i in 0..200 {
                        let line = json!({"message": {"content": format!("word{i} ")}, "done": false}).to_string() + "\n";
                        if c.raw(line.as_bytes()).is_err() {
                            seen.lock().unwrap().slow_closed = true;
                            return;
                        }
                        seen.lock().unwrap().slow_written = i + 1;
                        std::thread::sleep(Duration::from_millis(25));
                    }
                    return;
                }
                for line in [
                    json!({"message": {"content": "", "thinking": "Let me look."}, "done": false}),
                    json!({"message": {"content": "Here's the news [1]."}, "done": false}),
                    json!({"message": {"content": ""}, "done": true, "eval_count": 12, "eval_duration": 400_000_000u64, "prompt_eval_count": 900}),
                ] {
                    let _ = c.raw((line.to_string() + "\n").as_bytes());
                }
                return;
            }
            "/search" => {
                assert_eq!(req.query("q").as_deref(), Some("rust gateway news"));
                // A result on this PC: the gateway must not read it (its snippet is used instead).
                reply(&mut c, &json!({"results": [{"title": "Result", "url": "http://127.0.0.1:1/page", "content": "snippet text"}]}));
            }
            _ => {
                let _ = c.raw(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
            }
        }
    }
}

// --- The gateway ----------------------------------------------------------------------------------

struct Setup {
    gw: server::Running,
    token: String,
    pin: String,
    seen: Arc<Mutex<Seen>>,
    dir: std::path::PathBuf,
}

impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn setup(name: &str) -> Setup {
    let dir = std::env::temp_dir().join(format!("cubic-gw-e2e-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (backend, seen) = fake_backend();
    let config = Config { dir: dir.clone(), ollama: backend.clone(), searxng: backend, default_ctx: 16384, media: None, voice: None };
    let id = identity::load(&dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    Setup { gw, token: id.token, pin: id.fingerprint, seen, dir }
}

#[test]
fn only_the_pinned_certificate_over_tls_13() {
    let s = setup("tls");
    let port = s.gw.port;
    assert!(connect_with(port, "0000", &rustls::version::TLS13).is_err(), "a phone pinned to another PC refuses");
    assert!(connect_with(port, &s.pin, &rustls::version::TLS12).is_err(), "TLS 1.2 is refused");
    let r = ask(port, &s.pin, "GET", "/v1/health", Some(&s.token), None);
    assert_eq!((r.status, r.json()), (200, json!({"ok": true})));
}

#[test]
fn the_token_is_required_and_failures_are_slow() {
    let s = setup("token");
    let port = s.gw.port;
    for token in [None, Some("wrong")] {
        let t = Instant::now();
        let r = ask(port, &s.pin, "GET", "/v1/models", token, None);
        assert_eq!((r.status, r.json()), (401, json!({"error": {"message": "Not paired: invalid token"}})));
        assert_eq!(r.header("Connection"), Some("close"));
        assert!(t.elapsed() >= Duration::from_millis(950), "guessing is slow");
    }
    let st = ask(port, &s.pin, "GET", "/v1/cubicai/status", Some(&s.token), None).json();
    assert_eq!(st["blocked"], 2);
    assert_eq!(st["recent"][0]["outcome"], "blocked");
}

#[test]
fn keep_alive_and_bad_requests() {
    let s = setup("keep");
    let port = s.gw.port;
    let mut c = connect(port, &s.pin);
    for _ in 0..3 {
        send(&mut c, "GET", "/health/", Some(&s.token), None);
        assert_eq!(read_resp(&mut c).status, 200, "several requests on one connection");
    }
    let r = ask(port, &s.pin, "POST", "/v1/chat/completions", Some(&s.token), Some("{not json"));
    assert_eq!((r.status, r.json()["error"]["message"].clone()), (400, json!("Invalid JSON")));
    assert_eq!(ask(port, &s.pin, "GET", "/v1/nothing", Some(&s.token), None).status, 404);
    // An unknown POST: its body is drained, the connection closed.
    let mut c = connect(port, &s.pin);
    send(&mut c, "POST", "/v1/elsewhere", Some(&s.token), Some("{\"x\": 1}"));
    let r = read_resp(&mut c);
    assert_eq!((r.status, r.header("Connection")), (404, Some("close")));
    let mut c = connect(port, &s.pin);
    c.get_mut().write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nAuthorization: Bearer {}\r\nContent-Length: 999999999\r\n\r\n", s.token).as_bytes()).unwrap();
    assert_eq!(read_resp(&mut c).status, 413);
    // Media and voice aren't in this gateway yet, and say so instead of failing oddly.
    let g = ask(port, &s.pin, "GET", "/v1/media/generators", Some(&s.token), None).json();
    assert_eq!(g, json!({"generators": [], "comfy_running": false}));
    assert_eq!(ask(port, &s.pin, "GET", "/v1/media/jobs/abc/events", Some(&s.token), None).status, 404);
    assert_eq!(ask(port, &s.pin, "DELETE", "/v1/media/jobs/abc", Some(&s.token), None).status, 404);
    assert_eq!(ask(port, &s.pin, "GET", "/v1/voice/voices", Some(&s.token), None).json()["ready"], false);
}

#[test]
fn models_for_the_phone() {
    let s = setup("models");
    let r = ask(s.gw.port, &s.pin, "GET", "/v1/models", Some(&s.token), None);
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "the embedding model is left out");
    assert_eq!(data[0]["id"], "gemma4:e4b");
    assert_eq!(data[0]["context_length"], 16384);
    assert_eq!(data[0]["input_modalities"], json!(["text", "image"]));
    assert_eq!(data[0]["cubic_recommended"], true);
    assert_eq!(v["cubic_remote"], Value::Null, "remote access is off");
    assert_eq!(v["cubic_voice"], false);
    assert!(v.get("cubic_lan6").is_some());
    // settings.json changes apply live.
    std::fs::write(s.dir.join("settings.json"), "{\"num_ctx\": 4096}").unwrap();
    let v = ask(s.gw.port, &s.pin, "GET", "/v1/models", Some(&s.token), None).json();
    assert_eq!(v["data"][0]["context_length"], 4096);
}

#[test]
fn a_chat_with_web_search_streams_like_the_python_gateway() {
    // Keys in the Python gateway's order too (serde_json preserve_order).
    let s = setup("chat");
    let body = json!({
        "model": "gemma4:e4b",
        "stream": true,
        "cubic_web_search": true,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "What's new?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]},
        ],
    })
    .to_string();
    let r = ask(s.gw.port, &s.pin, "POST", "/v1/chat/completions", Some(&s.token), Some(&body));
    assert_eq!(r.status, 200);
    assert_eq!(r.header("Content-Type"), Some("text/event-stream"));
    let ev = r.events();
    let want = [
        r#"{"cubic_status":"Loading gemma4 on your PC…"}"#,
        r#"{"cubic_status":"Searching the web…"}"#,
        r#"{"cubic_status":"Searching: rust gateway news"}"#,
        r#"{"citations":["http://127.0.0.1:1/page"]}"#,
        r#"{"cubic_status":"Reading 1 results…"}"#,
        r#"{"choices":[{"index":0,"delta":{"reasoning":"Let me look."}}]}"#,
        r#"{"choices":[{"index":0,"delta":{"content":"Here's the news [1]."}}]}"#,
        r#"{"choices":[],"usage":{"prompt_tokens":900,"completion_tokens":12}}"#,
        "[DONE]",
    ];
    assert_eq!(ev, want);

    let sent = s.seen.lock().unwrap().chats[0].clone();
    assert_eq!(sent["think"], true);
    assert_eq!(sent["options"]["num_ctx"], 16384);
    assert_eq!(sent["keep_alive"], "15m");
    assert_eq!(sent["messages"][1]["images"], json!(["QUJD"]));
    let system = sent["messages"][0]["content"].as_str().unwrap();
    assert!(system.starts_with("Be brief.\n\nLive web search results for \"rust gateway news\""), "{system}");
    assert!(system.contains("[1] Result\nURL: http://127.0.0.1:1/page\nsnippet text\n"), "the page on this PC wasn't read; its snippet is used");

    let st = ask(s.gw.port, &s.pin, "GET", "/v1/cubicai/status", Some(&s.token), None).json();
    let done = &st["recent"][0];
    assert_eq!((done["outcome"].clone(), done["model"].clone(), done["search"].clone(), done["tokens"].clone(), done["tps"].clone()), (json!("ok"), json!("gemma4:e4b"), json!(true), json!(12), json!(30.0)));
}

#[test]
fn hanging_up_stops_the_model() {
    let s = setup("hangup");
    let body = json!({"model": "slow", "messages": [{"role": "user", "content": "Talk."}]}).to_string();
    let mut c = connect(s.gw.port, &s.pin);
    send(&mut c, "POST", "/v1/chat/completions", Some(&s.token), Some(&body));
    let mut line = String::new();
    while !line.contains("word2") {
        line.clear();
        c.read_line(&mut line).unwrap();
    }
    drop(c);
    let t = Instant::now();
    loop {
        let st = ask(s.gw.port, &s.pin, "GET", "/v1/cubicai/status", Some(&s.token), None).json();
        if st["recent"][0]["outcome"] == "stopped" {
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(5), "the reply was stopped: {st}");
        std::thread::sleep(Duration::from_millis(50));
    }
    let t = Instant::now();
    while !s.seen.lock().unwrap().slow_closed {
        assert!(t.elapsed() < Duration::from_secs(5), "Ollama's stream was closed too");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(s.seen.lock().unwrap().slow_written < 100, "it stopped early");
}

#[test]
fn twelve_connections_per_address() {
    let s = setup("slots");
    let open: Vec<Client> = (0..12).map(|_| connect(s.gw.port, &s.pin)).collect();
    let t = Instant::now();
    while s.gw.gateway.open_connections("127.0.0.1".parse().unwrap()).1 < 12 {
        assert!(t.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(connect_with(s.gw.port, &s.pin, &rustls::version::TLS13).is_err(), "the 13th is turned away");
    drop(open);
    let t = Instant::now();
    while s.gw.gateway.open_connections("127.0.0.1".parse().unwrap()).1 > 0 {
        assert!(t.elapsed() < Duration::from_secs(5), "slots are freed when phones hang up");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(ask(s.gw.port, &s.pin, "GET", "/v1/health", Some(&s.token), None).status, 200);
}

/// The Python gateway on a spare port, with the real identity but nothing shared with the live
/// gateway: its own cache and ComfyUI folders, a ComfyUI port nobody listens on (so it can't
/// clear the real queue), and a stand-in avahi so nothing is announced to the phone.
struct PythonGateway {
    child: std::process::Child,
    port: u16,
    dir: std::path::PathBuf,
}

impl PythonGateway {
    fn start() -> PythonGateway {
        let server = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../cubicai/server");
        let dir = std::env::temp_dir().join(format!("cubic-gw-python-{}", std::process::id()));
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(dir.join("comfy")).unwrap();
        let avahi = bin.join("avahi-publish-service");
        std::fs::write(&avahi, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&avahi, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let path = format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default());
        let child = std::process::Command::new(server.join(".venv/bin/python"))
            .arg(server.join("cubicai_server.py"))
            .args(["serve", "--bind", "127.0.0.1", "--port", &port.to_string()])
            .env("PATH", path)
            .env("CUBICAI_CACHE_DIR", dir.join("cache"))
            .env("CUBICAI_COMFY_DIR", dir.join("comfy"))
            .env("CUBICAI_COMFY_PORT", "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("the CubicAI repo's Python gateway");
        let g = PythonGateway { child, port, dir };
        let t = Instant::now();
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(t.elapsed().as_secs() < 20, "the Python gateway didn't start");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        g
    }
}

impl Drop for PythonGateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The Rust gateway beside the Python one (started here on a spare port), with the same identity
/// and the real Ollama: the phone must see the same models and the same kind of stream. Run with
/// `cargo test -p cubic-gateway --release real_matches_python -- --ignored --nocapture`.
#[test]
#[ignore]
fn real_matches_python() {
    // No media here: starting the media manager clears ComfyUI's cubicai folders and the temp
    // folder, which the running Python gateway shares (its undelivered results live there).
    let config = Config { media: None, ..Config::from_env() };
    let id = identity::load(&config.dir).expect("the Python gateway's identity");
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let py = PythonGateway::start();
    let (rust, python) = (gw.port, py.port);
    let get = |port: u16| ask(port, &id.fingerprint, "GET", "/v1/models", Some(&id.token), None).json();
    let (r, p) = (get(rust), get(python));
    let pick = |v: &Value| -> Vec<Value> {
        v["data"].as_array().unwrap().iter().map(|m| json!([m["id"], m["context_length"], m["input_modalities"], m["supported_parameters"], m["cubic_recommended"], m["size"]])).collect()
    };
    for m in pick(&r) {
        println!("  {m}");
    }
    assert_eq!(pick(&r), pick(&p), "the same models, contexts, abilities and recommendation");
    for k in ["cubic_remote", "cubic_lan6"] {
        // Each test gateway has its own port; the addresses must match.
        let r_url = r[k].as_str().map(|u| u.replace(&format!(":{rust}"), ":8443"));
        let p_url = p[k].as_str().map(|u| u.replace(&format!(":{python}"), ":8443"));
        assert_eq!(r_url, p_url, "{k}");
        println!("  {k}: {}", r[k]);
    }

    // The same short chat through both: the same kinds of events in the same order.
    let model = std::env::var("CHAT_MODEL").unwrap_or_else(|_| r["data"].as_array().unwrap().iter().find(|m| m["cubic_recommended"] == true).unwrap()["id"].as_str().unwrap().to_string());
    let body = json!({"model": model, "messages": [{"role": "user", "content": "Reply with just the word: hello"}], "cubic_think": false}).to_string();
    let shape = |port: u16| -> (Vec<String>, String) {
        let t = Instant::now();
        let resp = ask(port, &id.fingerprint, "POST", "/v1/chat/completions", Some(&id.token), Some(&body));
        let ev = resp.events();
        let text: String = ev.iter().filter_map(|e| serde_json::from_str::<Value>(e).ok()).filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(String::from)).collect();
        let mut kinds: Vec<String> = ev
            .iter()
            .map(|e| match serde_json::from_str::<Value>(e) {
                Ok(v) if v.get("cubic_status").is_some() => "status".into(),
                Ok(v) if v.get("usage").is_some() => "usage".into(),
                Ok(v) if v.get("error").is_some() => format!("error {}", v["error"]["message"]),
                Ok(v) => v["choices"][0]["delta"].as_object().map(|d| d.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default(),
                Err(_) => e.clone(),
            })
            .collect();
        kinds.dedup();
        println!("  port {port}: {:?} in {:.1} s: {text:?}", kinds, t.elapsed().as_secs_f32());
        (kinds, text)
    };
    let (pk, pt) = shape(python);
    let (rk, rt) = shape(rust);
    let strip = |k: &[String]| k.iter().filter(|x| *x != "status").cloned().collect::<Vec<_>>();
    assert_eq!(strip(&rk), strip(&pk), "the same events (a loading status only shows when the model wasn't in memory)");
    assert!(rt.to_lowercase().contains("hello") && pt.to_lowercase().contains("hello"));

    // Web search through the real SearXNG and real pages.
    let body = json!({"model": model, "cubic_web_search": true, "cubic_think": false, "messages": [{"role": "user", "content": "What is the latest stable Rust version? One sentence."}]}).to_string();
    let t = Instant::now();
    let ev = ask(rust, &id.fingerprint, "POST", "/v1/chat/completions", Some(&id.token), Some(&body)).events();
    let vals: Vec<Value> = ev.iter().filter_map(|e| serde_json::from_str(e).ok()).collect();
    for v in vals.iter().filter(|v| v.get("cubic_status").is_some()) {
        println!("  {}", v["cubic_status"]);
    }
    let cites = vals.iter().find_map(|v| v["citations"].as_array().cloned()).unwrap_or_default();
    let text: String = vals.iter().filter_map(|v| v["choices"][0]["delta"]["content"].as_str()).collect();
    println!("  {} sources, {:.1} s: {text}", cites.len(), t.elapsed().as_secs_f32());
    assert!(!cites.is_empty(), "SearXNG found pages");
    assert!(!text.is_empty() && ev.last().map(String::as_str) == Some("[DONE]"));
}
