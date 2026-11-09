//! One model turn for a Bot: Ollama on this PC, or any OpenAI-compatible cloud API (Grok, Gemini's
//! OpenAI endpoint, OpenRouter, a custom one). The words stream out as they come; tool calls are
//! collected whole. A set `stop` flag ends the turn at the next piece.

use crate::ollama::Ollama;
use serde_json::{json, Value};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub enum Endpoint {
    Local { model: String },
    Cloud { name: String, base: String, key: String, model: String },
}

impl Endpoint {
    pub fn is_local(&self) -> bool {
        matches!(self, Endpoint::Local { .. })
    }
}

pub enum Delta<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    /// The arguments as a JSON object, in text.
    pub args: String,
}

#[derive(Debug, Default)]
pub struct Turn {
    pub text: String,
    pub reasoning: String,
    pub calls: Vec<Call>,
    /// Prompt and output tokens, when the provider says.
    pub usage: Option<(u64, u64)>,
    /// Stopped part way: the words so far are kept, unfinished tool calls dropped.
    pub stopped: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn turn(endpoint: &Endpoint, ollama: &Ollama, messages: &[Value], tools: &Value, num_ctx: u64, keep_alive: Value, stop: &AtomicBool, on: &mut dyn FnMut(Delta)) -> Result<Turn, String> {
    match endpoint {
        Endpoint::Local { model } => local(ollama, model, messages, tools, num_ctx, keep_alive, stop, on),
        Endpoint::Cloud { name, base, key, model } => cloud(name, base, key, model, messages, tools, stop, on),
    }
}

fn cloud_agent() -> ureq::Agent {
    // A reply streams for as long as it takes; only connecting and the first byte are limited.
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(300)))
        .http_status_as_error(false)
        .build()
        .new_agent()
}

/// What a provider's refusal means, in words for the phone.
pub fn cloud_error(name: &str, code: u16, body: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().or(v["error"].as_str()).or(v["message"].as_str()).map(String::from))
        .unwrap_or_else(|| body.chars().take(200).collect());
    match code {
        401 | 403 => format!("{name} refused the key your PC has ({code}). Check the key in the app's Settings; the phone sends it again when you open the Bot."),
        402 => format!("Your {name} account is out of credits (402)."),
        404 => format!("{name} doesn't know this model (404): {detail}"),
        429 => format!("{name} is busy or you've hit a rate limit (429). Try again in a minute."),
        _ => format!("{name} error {code}: {detail}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn cloud(name: &str, base: &str, key: &str, model: &str, messages: &[Value], tools: &Value, stop: &AtomicBool, on: &mut dyn FnMut(Delta)) -> Result<Turn, String> {
    let mut body = json!({"model": model, "messages": messages, "stream": true, "stream_options": {"include_usage": true}});
    if tools.as_array().is_some_and(|t| !t.is_empty()) {
        body["tools"] = tools.clone();
    }
    let mut req = cloud_agent().post(format!("{base}/chat/completions")).header("Accept", "text/event-stream");
    if !key.is_empty() {
        req = req.header("Authorization", &format!("Bearer {key}"));
    }
    if base.contains("openrouter.ai") {
        req = req.header("X-Title", "CubicAI");
    }
    let mut r = req.send_json(&body).map_err(|e| format!("Can't reach {name}: {e}"))?;
    if !r.status().is_success() {
        let code = r.status().as_u16();
        return Err(cloud_error(name, code, &r.body_mut().read_to_string().unwrap_or_default()));
    }
    let reader = std::io::BufReader::new(r.body_mut().as_reader());
    read_sse(reader, name, stop, on)
}

/// An OpenAI-style event stream → a turn. A stream that stops without `[DONE]` or a finish reason
/// is an error (the connection dropped), not a short answer.
pub fn read_sse(reader: impl BufRead, name: &str, stop: &AtomicBool, on: &mut dyn FnMut(Delta)) -> Result<Turn, String> {
    let mut t = Turn::default();
    // index → (id, name, arguments so far)
    let mut calls: std::collections::BTreeMap<u64, (String, String, String)> = Default::default();
    let mut finished = false;
    for line in reader.lines() {
        if stop.load(Ordering::SeqCst) {
            t.stopped = true;
            t.calls.clear();
            return Ok(t);
        }
        let line = line.map_err(|e| format!("The connection to {name} dropped: {e}"))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
        if data == "[DONE]" {
            finished = true;
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(e) = v["error"]["message"].as_str().or(v["error"].as_str()) {
            return Err(format!("{name}: {e}"));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            t.usage = Some((u["prompt_tokens"].as_u64().unwrap_or(0), u["completion_tokens"].as_u64().unwrap_or(0)));
        }
        let Some(choice) = v["choices"].get(0) else { continue };
        let d = &choice["delta"];
        if let Some(r) = d["reasoning_content"].as_str().or(d["reasoning"].as_str()).filter(|s| !s.is_empty()) {
            t.reasoning.push_str(r);
            on(Delta::Reasoning(r));
        }
        if let Some(c) = d["content"].as_str().filter(|s| !s.is_empty()) {
            t.text.push_str(c);
            on(Delta::Text(c));
        }
        for c in d["tool_calls"].as_array().into_iter().flatten() {
            let idx = c["index"].as_u64().unwrap_or(calls.len() as u64);
            let slot = calls.entry(idx).or_default();
            if let Some(id) = c["id"].as_str().filter(|s| !s.is_empty()) {
                slot.0 = id.to_string();
            }
            if let Some(n) = c["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                slot.1 = n.to_string();
            }
            if let Some(a) = c["function"]["arguments"].as_str() {
                slot.2.push_str(a);
            }
        }
        if choice["finish_reason"].is_string() {
            finished = true;
        }
    }
    if !finished {
        return Err(format!("The connection to {name} dropped before the reply finished."));
    }
    t.calls = calls
        .into_values()
        .enumerate()
        .filter(|(_, (_, n, _))| !n.is_empty())
        .map(|(i, (id, name, args))| Call { id: if id.is_empty() { format!("call_{i}") } else { id }, name, args: if args.trim().is_empty() { "{}".into() } else { args } })
        .collect();
    Ok(t)
}

#[allow(clippy::too_many_arguments)]
fn local(ollama: &Ollama, model: &str, messages: &[Value], tools: &Value, num_ctx: u64, keep_alive: Value, stop: &AtomicBool, on: &mut dyn FnMut(Delta)) -> Result<Turn, String> {
    let info = ollama.info(model);
    if !info.has("tools") {
        return Err(format!("{model} can't use tools, so it can't work as a Bot. Pick a model that uses tools (e.g. gemma4 or qwen3.6)."));
    }
    let native = crate::ollama::to_native(messages, info.has("vision"))?;
    let mut payload = json!({"model": model, "messages": native, "stream": true, "tools": tools, "options": {"num_ctx": info.context(num_ctx)}, "keep_alive": keep_alive});
    if info.has("thinking") {
        payload["think"] = true.into();
    }
    let mut r = ollama.post("/api/chat", &payload, true).map_err(|e| format!("Ollama: {e}"))?;
    let reader = std::io::BufReader::new(r.body_mut().as_reader());
    read_ndjson(reader, stop, on)
}

/// Ollama's stream (one JSON object per line) → a turn; its tool calls come whole, arguments as objects.
pub fn read_ndjson(reader: impl BufRead, stop: &AtomicBool, on: &mut dyn FnMut(Delta)) -> Result<Turn, String> {
    let mut t = Turn::default();
    let mut finished = false;
    for line in reader.lines() {
        if stop.load(Ordering::SeqCst) {
            t.stopped = true;
            t.calls.clear();
            return Ok(t);
        }
        let line = line.map_err(|e| format!("The connection to Ollama dropped: {e}"))?;
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(e) = v["error"].as_str() {
            return Err(format!("Ollama: {e}"));
        }
        let m = &v["message"];
        if let Some(r) = m["thinking"].as_str().filter(|s| !s.is_empty()) {
            t.reasoning.push_str(r);
            on(Delta::Reasoning(r));
        }
        if let Some(c) = m["content"].as_str().filter(|s| !s.is_empty()) {
            t.text.push_str(c);
            on(Delta::Text(c));
        }
        for c in m["tool_calls"].as_array().into_iter().flatten() {
            let f = &c["function"];
            let args = if f["arguments"].is_null() { json!({}) } else { f["arguments"].clone() };
            let n = t.calls.len();
            t.calls.push(Call { id: format!("call_{}_{n}", crate::bots::new_id()), name: f["name"].as_str().unwrap_or("").to_string(), args: args.to_string() });
        }
        if v["done"] == true {
            t.usage = Some((v["prompt_eval_count"].as_u64().unwrap_or(0), v["eval_count"].as_u64().unwrap_or(0)));
            finished = true;
            break;
        }
    }
    if !finished {
        return Err("The connection to Ollama dropped before the reply finished.".into());
    }
    t.calls.retain(|c| !c.name.is_empty());
    Ok(t)
}
