//! Streaming chat completions over SSE, on a worker thread. Events go back to the UI over a channel.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub enum Ev {
    Content(String),
    Reasoning(String),
    Usage { prompt: u64, output: u64 },
    Speed { gen_tps: f64 },
    Done,
    Error(String),
}

#[derive(Clone, Default)]
pub struct Request {
    pub base_url: String,
    pub key: String,
    pub model: String,
    pub messages: Vec<Value>,
    /// llama-server: the reasoning arrives as `reasoning_content`; ask for timings too.
    pub local: bool,
    /// More fields for the request body (temperature, max_tokens, llama-server's `lora`…).
    pub extra: serde_json::Map<String, Value>,
}

pub fn agent(connect_secs: u64) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(connect_secs)))
        .timeout_recv_response(Some(Duration::from_secs(900)))
        .http_status_as_error(false)
        .user_agent("cubicai-lab")
        .build()
        .into()
}

/// Streams one reply. `job` tags every event so a stopped reply's late events can be ignored.
pub fn stream(req: Request, job: u64, tx: Sender<(u64, Ev)>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let send = |ev: Ev| {
            let _ = tx.send((job, ev));
            ctx.request_repaint();
        };
        match run(&req, &cancel, &send) {
            Ok(()) => send(Ev::Done),
            Err(e) => {
                if !cancel.load(Ordering::Relaxed) {
                    send(Ev::Error(e))
                }
            }
        }
    });
}

fn run(req: &Request, cancel: &AtomicBool, send: &dyn Fn(Ev)) -> Result<(), String> {
    let mut body = json!({
        "model": req.model,
        "messages": req.messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    for (k, v) in &req.extra {
        body[k] = v.clone();
    }
    let url = format!("{}/chat/completions", req.base_url.trim_end_matches('/'));
    let agent = agent(if req.local { 5 } else { 20 });
    let mut r = agent.post(&url).header("Accept", "text/event-stream");
    if !req.key.is_empty() {
        r = r.header("Authorization", format!("Bearer {}", req.key));
    }
    let resp = r.send_json(&body).map_err(|e| connect_error(&e, &req.base_url))?;
    let status = resp.status().as_u16();
    let mut reader = BufReader::new(resp.into_body().into_reader());
    if status != 200 {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut reader, &mut text);
        return Err(http_error(status, &text));
    }
    read_stream(reader, cancel, send)
}

/// One whole reply, on the calling thread (for helpers like drafting training data).
pub fn collect(req: &Request, cancel: &AtomicBool) -> Result<String, String> {
    let text = std::cell::RefCell::new(String::new());
    let error = std::cell::RefCell::new(None);
    run(req, cancel, &|ev| match ev {
        Ev::Content(c) => text.borrow_mut().push_str(&c),
        Ev::Error(e) => *error.borrow_mut() = Some(e),
        _ => {}
    })?;
    if let Some(e) = error.into_inner() {
        return Err(e);
    }
    Ok(text.into_inner())
}

/// Reads an OpenAI-style SSE stream until `[DONE]`, a finish reason and the end, or an error.
pub fn read_stream(mut reader: impl BufRead, cancel: &AtomicBool, send: &dyn Fn(Ev)) -> Result<(), String> {
    let mut line = String::new();
    let mut finished = false;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| format!("Connection dropped: {e}"))?;
        if n == 0 {
            break;
        }
        let l = line.trim_end();
        let Some(data) = l.strip_prefix("data:") else { continue };
        let data = data.trim_start();
        if data == "[DONE]" {
            return Ok(());
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(err) = v.get("error") {
            return Err(error_message(err));
        }
        if let Some(c) = v["choices"].get(0) {
            let d = &c["delta"];
            for key in ["reasoning_content", "reasoning"] {
                if let Some(s) = d[key].as_str().filter(|s| !s.is_empty()) {
                    send(Ev::Reasoning(s.to_string()));
                    break;
                }
            }
            if let Some(s) = d["content"].as_str().filter(|s| !s.is_empty()) {
                send(Ev::Content(s.to_string()));
            }
            if c["finish_reason"].as_str().is_some() {
                finished = true;
            }
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            send(Ev::Usage {
                prompt: u["prompt_tokens"].as_u64().unwrap_or(0),
                output: u["completion_tokens"].as_u64().unwrap_or(0),
            });
        }
        if let Some(t) = v.get("timings").and_then(|t| t["predicted_per_second"].as_f64()) {
            send(Ev::Speed { gen_tps: t });
        }
    }
    if finished { Ok(()) } else { Err("Connection dropped before the reply finished.".into()) }
}

pub(crate) fn connect_error(e: &ureq::Error, base: &str) -> String {
    let s = e.to_string();
    if (base.contains("127.0.0.1") || base.contains("localhost"))
        && (s.contains("refused") || s.contains("Connection")) {
            return format!("Nothing is answering at {base}. Is the server running? ({s})");
        }
    format!("Couldn't reach {base}: {s}")
}

fn error_message(err: &Value) -> String {
    err["message"].as_str().map(str::to_string).unwrap_or_else(|| err.to_string())
}

pub(crate) fn http_error(status: u16, body: &str) -> String {
    let msg = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").map(error_message).or_else(|| v["message"].as_str().map(str::to_string)))
        .unwrap_or_else(|| body.chars().take(300).collect());
    match status {
        401 | 403 => format!("The key was refused ({status}): {msg}"),
        404 => format!("Not found ({status}): {msg}. Check the model name."),
        429 => format!("Rate limited or out of quota ({status}): {msg}"),
        503 => format!("The server is busy or still loading ({status}): {msg}"),
        _ => format!("Error {status}: {msg}"),
    }
}

/// The provider's model list (`GET /models`), with context sizes when it reports them.
pub fn list_models(base_url: &str, key: &str) -> Result<Vec<(String, Option<u64>)>, String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut r = agent(8).get(&url);
    if !key.is_empty() {
        r = r.header("Authorization", format!("Bearer {key}"));
    }
    let resp = r.call().map_err(|e| connect_error(&e, base_url))?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    if status != 200 {
        return Err(http_error(status, &text));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut out: Vec<(String, Option<u64>)> = v["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = m["id"].as_str()?.trim_start_matches("models/").to_string();
            let ctx = m["context_length"].as_u64().or_else(|| m["top_provider"]["context_length"].as_u64());
            Some((id, ctx))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
