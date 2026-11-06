//! One chat reply: the phone's OpenAI-style request → Ollama /api/chat → server-sent OpenAI deltas,
//! with the gateway's own extras (`cubic_status` progress, `citations`, `delta.reasoning`, usage).

use crate::server::Gateway;
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};

/// How a reply ended, for the activity list: outcome, generated tokens, generation seconds.
pub struct Outcome {
    pub outcome: &'static str,
    pub tokens: u64,
    pub gen_seconds: f64,
}

impl Outcome {
    fn of(outcome: &'static str) -> Outcome {
        Outcome { outcome, tokens: 0, gen_seconds: 0.0 }
    }
}

/// Where events go. Writing fails once the phone has gone, which stops the reply (dropping the
/// Ollama response closes its connection, and Ollama stops generating).
pub trait Sink {
    fn event(&mut self, v: &Value) -> std::io::Result<()>;
    fn done(&mut self) -> std::io::Result<()>;
}

impl<S: Read + Write> Sink for crate::http::Conn<S> {
    fn event(&mut self, v: &Value) -> std::io::Result<()> {
        crate::http::Conn::event(self, v)
    }

    fn done(&mut self) -> std::io::Result<()> {
        self.raw(b"data: [DONE]\n\n")
    }
}

/// The Ollama request for a chat (after search context has been added to `messages`).
pub fn ollama_request(body: &Value, model: &str, messages: Vec<Value>, info: &crate::ollama::ModelInfo, num_ctx: u64, keep_alive: Value) -> Value {
    // Keep a model warm between chats; Ollama still evicts it early if another model needs the VRAM.
    let mut payload = json!({"model": model, "messages": messages, "stream": true, "options": {"num_ctx": num_ctx}, "keep_alive": keep_alive});
    // A voice call turns thinking off (cubic_think: false) so the reply starts right away. Some
    // models think without listing the capability, so off goes to every model.
    if body["cubic_think"] == false {
        payload["think"] = false.into();
    } else if info.has("thinking") {
        payload["think"] = true.into();
    }
    // Tools the app offers (making things on this PC), for models that can call them. Ollama has no
    // tool_choice; "none" (nothing more to make this turn) leaves the tools out instead.
    let has_tools = body["tools"].as_array().is_some_and(|t| !t.is_empty());
    if has_tools && info.has("tools") && body["tool_choice"] != "none" {
        payload["tools"] = body["tools"].clone();
    }
    payload
}

/// Adds the search results to the system message (or a new one in front).
pub fn add_context(messages: &mut Vec<Value>, context: &str) {
    match messages.first_mut() {
        Some(m) if m["role"] == "system" => {
            let c = m["content"].as_str().unwrap_or("").to_string() + "\n\n" + context;
            m["content"] = c.into();
        }
        _ => messages.insert(0, json!({"role": "system", "content": context})),
    }
}

/// Ollama's stream (one JSON object per line) → events. Returns the outcome; Err means the phone
/// went away.
pub fn relay(lines: impl BufRead, sink: &mut dyn Sink, rid: u64, phase: &mut dyn FnMut(&str)) -> std::io::Result<Outcome> {
    let mut tokens = 0u64;
    let mut gen_seconds = 0.0;
    for line in lines.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(e) = chunk["error"].as_str() {
            sink.event(&crate::http::error(e))?;
            return Ok(Outcome::of("error"));
        }
        let msg = &chunk["message"];
        let mut delta = serde_json::Map::new();
        if let Some(t) = msg["thinking"].as_str().filter(|t| !t.is_empty()) {
            delta.insert("reasoning".into(), t.into());
            tokens += 1;
            if tokens == 1 {
                phase("Thinking");
            }
        }
        if let Some(c) = msg["content"].as_str().filter(|c| !c.is_empty()) {
            delta.insert("content".into(), c.into());
            phase("Answering");
        }
        if msg["tool_calls"].as_array().is_some_and(|a| !a.is_empty()) {
            // Ollama sends each call whole; OpenAI clients expect string arguments and an id.
            delta.insert("tool_calls".into(), crate::ollama::openai_tool_calls(&msg["tool_calls"], rid));
            phase("Asking for a creation");
        }
        if !delta.is_empty() {
            sink.event(&json!({"choices": [{"index": 0, "delta": delta}]}))?;
        }
        if chunk["done"] == true {
            tokens = chunk["eval_count"].as_u64().unwrap_or(tokens);
            gen_seconds = chunk["eval_duration"].as_f64().unwrap_or(0.0) / 1e9;
            // Token counts only (no content), so the phone can show how full the context is.
            sink.event(&json!({"choices": [], "usage": {
                "prompt_tokens": chunk["prompt_eval_count"].as_u64().unwrap_or(0),
                "completion_tokens": chunk["eval_count"].as_u64().unwrap_or(0),
            }}))?;
            break;
        }
    }
    sink.done()?;
    Ok(Outcome { outcome: "ok", tokens, gen_seconds })
}

/// Streams one reply (the event-stream head has been sent). Errors go to the phone as an event.
pub fn stream(gw: &Gateway, body: &Value, ip: &str, rid: u64, sink: &mut dyn Sink) -> Outcome {
    match stream_inner(gw, body, ip, rid, sink) {
        Ok(o) => o,
        Err(Fail::Gone) => {
            crate::log("client disconnected; generation stopped");
            Outcome::of("stopped")
        }
        Err(Fail::Message(m)) => {
            let _ = sink.event(&crate::http::error(&m));
            Outcome::of("error")
        }
    }
}

enum Fail {
    Gone,
    Message(String),
}

impl From<std::io::Error> for Fail {
    fn from(_: std::io::Error) -> Fail {
        Fail::Gone
    }
}

fn stream_inner(gw: &Gateway, body: &Value, _ip: &str, rid: u64, sink: &mut dyn Sink) -> Result<Outcome, Fail> {
    let model = body["model"].as_str().unwrap_or("");
    let web = body["cubic_web_search"].as_bool().unwrap_or(false);
    let info = gw.ollama.info(model);
    let empty = vec![];
    let mut messages = crate::ollama::to_native(body["messages"].as_array().unwrap_or(&empty), info.has("vision")).map_err(Fail::Message)?;
    if !gw.ollama.loaded().contains(model) {
        gw.activity.phase(rid, "Loading model");
        sink.event(&json!({"cubic_status": format!("Loading {} on your PC…", model.split(':').next().unwrap_or(model))}))?;
    }
    if web {
        gw.activity.phase(rid, "Searching the web");
        sink.event(&json!({"cubic_status": "Searching the web…"}))?;
        let query = crate::search::make_query(&gw.ollama, model, &info, &messages);
        sink.event(&json!({"cubic_status": format!("Searching: {query}")}))?;
        let mut sent: std::io::Result<()> = Ok(());
        let (urls, mut context) = crate::search::research(&gw.config.searxng, &query, &mut |urls| {
            sent = sink.event(&json!({"citations": urls})).and_then(|_| sink.event(&json!({"cubic_status": format!("Reading {} results…", urls.len())})));
            gw.activity.phase(rid, &format!("Reading {} results", urls.len()));
        });
        sent?;
        if urls.is_empty() {
            context = "Web search was requested but returned no results. Answer from your own knowledge and say so.".into();
        }
        add_context(&mut messages, &context);
    }
    let payload = ollama_request(body, model, messages, &info, info.context(gw.settings.num_ctx(gw.config.default_ctx)), gw.settings.keep_alive());
    gw.activity.phase(rid, "Waiting for model");
    let mut r = gw.ollama.post("/api/chat", &payload, true).map_err(|e| Fail::Message(format!("Ollama: {e}")))?;
    let reader = std::io::BufReader::new(r.body_mut().as_reader());
    Ok(relay(reader, sink, rid, &mut |p| gw.activity.phase(rid, p))?)
}
