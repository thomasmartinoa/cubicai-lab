//! Tools from other apps for Bots, over MCP (the Model Context Protocol). The user adds servers from
//! the phone; they're kept in `<config>/mcp.json` (0600: headers and env can hold tokens, which are
//! never sent back). Two kinds:
//!
//! - remote: a URL, spoken to from the gateway with the Streamable HTTP transport (JSON-RPC POSTs whose
//!   answers come as JSON or as server-sent events; `Mcp-Session-Id` kept, one re-initialise on 404).
//! - sandbox: a command (`npx -y …`, `uvx …`) run inside the Bots' computer, newline-delimited
//!   JSON-RPC on its stdin/stdout; it reaches the web through the computer's public-only proxy and
//!   never runs on the PC itself. Restarted when it ended (the computer is removed when idle).
//!
//! A Bot gets the tools of the servers it was given (`Bot.mcp`), named `<server>__<tool>`. Tools a
//! server doesn't mark read-only ask the user first (per server: changes / always / never).

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const MAX_SERVERS: usize = 20;
/// Tools offered to one Bot from all its servers together (small models get lost in more).
pub const MAX_TOOLS: usize = 40;
const RESULT_CHARS: usize = 12_000;
const LIST_TTL: Duration = Duration::from_secs(300);
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
const START_TIMEOUT: Duration = Duration::from_secs(90);
const PROTOCOL: &str = "2025-06-18";

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Remote { url: String, headers: Map<String, Value> },
    Sandbox { command: String, env: Map<String, Value> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ask {
    /// Tools not marked read-only ask first (the default).
    Changes,
    Always,
    Never,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Server {
    /// Short and readable ("github"): it starts the tools' names.
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub ask: Ask,
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub schema: Value,
    pub read_only: bool,
}

/// One of a Bot's MCP tools as the model sees it.
#[derive(Clone, Debug)]
pub struct Offered {
    pub function: String,
    pub server: String,
    pub tool: String,
    pub read_only: bool,
    pub definition: Value,
}

impl Server {
    /// For the phone: never the header or env values.
    pub fn to_public(&self) -> Value {
        let mut v = json!({"id": self.id, "name": self.name, "enabled": self.enabled, "ask": ask_name(self.ask)});
        match &self.kind {
            Kind::Remote { url, headers } => {
                v["kind"] = "remote".into();
                v["url"] = url.clone().into();
                v["headers"] = headers.keys().cloned().collect::<Vec<_>>().into();
            }
            Kind::Sandbox { command, env } => {
                v["kind"] = "sandbox".into();
                v["command"] = command.clone().into();
                v["env"] = env.keys().cloned().collect::<Vec<_>>().into();
            }
        }
        v
    }

    fn to_json(&self) -> Value {
        let mut v = json!({"id": self.id, "name": self.name, "enabled": self.enabled, "ask": ask_name(self.ask)});
        match &self.kind {
            Kind::Remote { url, headers } => {
                v["kind"] = "remote".into();
                v["url"] = url.clone().into();
                v["headers"] = Value::Object(headers.clone());
            }
            Kind::Sandbox { command, env } => {
                v["kind"] = "sandbox".into();
                v["command"] = command.clone().into();
                v["env"] = Value::Object(env.clone());
            }
        }
        v
    }

    fn from_json(v: &Value) -> Option<Server> {
        let kind = match v["kind"].as_str()? {
            "remote" => Kind::Remote { url: v["url"].as_str()?.to_string(), headers: v["headers"].as_object().cloned().unwrap_or_default() },
            "sandbox" => Kind::Sandbox { command: v["command"].as_str()?.to_string(), env: v["env"].as_object().cloned().unwrap_or_default() },
            _ => return None,
        };
        Some(Server {
            id: v["id"].as_str()?.to_string(),
            name: v["name"].as_str().unwrap_or("").to_string(),
            kind,
            ask: parse_ask(v["ask"].as_str().unwrap_or("changes")).unwrap_or(Ask::Changes),
            enabled: v["enabled"].as_bool().unwrap_or(true),
        })
    }

    /// Takes the phone's changes: name, url/command, ask, enabled; headers/env merged ("" removes one).
    pub fn apply(&mut self, v: &Value) -> Result<(), String> {
        if let Some(n) = v["name"].as_str() {
            self.name = n.trim().to_string();
        }
        if let Some(a) = v["ask"].as_str() {
            self.ask = parse_ask(a).ok_or("ask is changes, always or never.")?;
        }
        if let Some(e) = v["enabled"].as_bool() {
            self.enabled = e;
        }
        let merge = |into: &mut Map<String, Value>, from: &Value| -> Result<(), String> {
            for (k, val) in from.as_object().into_iter().flatten() {
                let k = k.trim();
                if k.is_empty() || k.len() > 100 || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                    return Err(format!("{k:?} isn't a usable name."));
                }
                match val.as_str().map(str::trim) {
                    Some("") | None => {
                        into.remove(k);
                    }
                    Some(s) if s.len() > 4000 => return Err(format!("{k} is too long.")),
                    Some(s) => {
                        into.insert(k.to_string(), s.into());
                    }
                }
            }
            Ok(())
        };
        match &mut self.kind {
            Kind::Remote { url, headers } => {
                if let Some(u) = v["url"].as_str() {
                    *url = u.trim().to_string();
                }
                merge(headers, &v["headers"])?;
            }
            Kind::Sandbox { command, env } => {
                if let Some(c) = v["command"].as_str() {
                    *command = c.trim().to_string();
                }
                merge(env, &v["env"])?;
            }
        }
        self.check()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.name.chars().count();
        if n == 0 || n > 40 {
            return Err("Give it a name (up to 40 characters).".into());
        }
        match &self.kind {
            Kind::Remote { url, .. } if !(url.starts_with("https://") || url.starts_with("http://")) || url.len() > 500 => Err("The address has to start with https://.".into()),
            Kind::Sandbox { command, .. } if command.is_empty() || command.len() > 1000 || command.contains('\n') => Err("Give the command that starts it, on one line (e.g. npx -y @modelcontextprotocol/server-everything).".into()),
            _ => Ok(()),
        }
    }
}

fn ask_name(a: Ask) -> &'static str {
    match a {
        Ask::Changes => "changes",
        Ask::Always => "always",
        Ask::Never => "never",
    }
}

fn parse_ask(s: &str) -> Option<Ask> {
    match s {
        "changes" => Some(Ask::Changes),
        "always" => Some(Ask::Always),
        "never" => Some(Ask::Never),
        _ => None,
    }
}

/// "My GitHub!" → "my_github" (unique against `taken`).
fn slug(name: &str, taken: &[String]) -> String {
    let mut base: String = name.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    base = base.split('_').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("_");
    base.truncate(16);
    if base.is_empty() || base.starts_with(|c: char| c.is_ascii_digit()) {
        base = format!("app{base}");
        base.truncate(16);
    }
    let mut s = base.clone();
    let mut n = 2;
    while taken.contains(&s) {
        s = format!("{base}{n}");
        n += 1;
    }
    s
}

/// What providers accept in a tool's parameters (Gemini takes only a subset of JSON Schema).
pub fn clean_schema(v: &Value) -> Value {
    fn walk(v: &Value) -> Value {
        let Some(o) = v.as_object() else { return v.clone() };
        let mut out = Map::new();
        for (k, val) in o {
            match k.as_str() {
                "type" | "description" | "required" | "enum" | "nullable" | "minimum" | "maximum" | "minItems" | "maxItems" => {
                    out.insert(k.clone(), val.clone());
                }
                "items" => {
                    out.insert(k.clone(), walk(val));
                }
                "properties" => {
                    out.insert(k.clone(), Value::Object(val.as_object().into_iter().flatten().map(|(p, s)| (p.clone(), walk(s))).collect()));
                }
                "anyOf" | "oneOf" => {
                    out.insert("anyOf".into(), val.as_array().into_iter().flatten().map(walk).collect());
                }
                _ => {}
            }
        }
        // ["string", "null"] → string, nullable.
        if let Some(t) = out.get("type").and_then(|t| t.as_array()).cloned() {
            let real: Vec<&Value> = t.iter().filter(|x| *x != "null").collect();
            out.insert("type".into(), real.first().map(|x| (*x).clone()).unwrap_or_else(|| "string".into()));
            if real.len() < t.len() {
                out.insert("nullable".into(), true.into());
            }
        }
        Value::Object(out)
    }
    let mut s = walk(v);
    if !s.is_object() || s["type"] != "object" {
        s = json!({"type": "object", "properties": s.get("properties").cloned().unwrap_or_else(|| json!({}))});
    }
    if s.get("properties").is_none() {
        s["properties"] = json!({});
    }
    s
}

/// A tools/call result as text for the model; ok is false when the server says it failed.
pub fn result_text(r: &Value) -> (String, bool) {
    let mut parts = vec![];
    for c in r["content"].as_array().into_iter().flatten() {
        match c["type"].as_str() {
            Some("text") => parts.push(c["text"].as_str().unwrap_or("").to_string()),
            Some("image") => parts.push("[an image]".into()),
            Some("audio") => parts.push("[a sound]".into()),
            Some("resource") => parts.push(c["resource"]["text"].as_str().map(String::from).unwrap_or_else(|| format!("[{}]", c["resource"]["uri"].as_str().unwrap_or("a file")))),
            Some("resource_link") => parts.push(format!("[{}: {}]", c["name"].as_str().unwrap_or("link"), c["uri"].as_str().unwrap_or(""))),
            _ => {}
        }
    }
    if parts.iter().all(|p| p.trim().is_empty())
        && let Some(s) = r.get("structuredContent")
    {
        parts = vec![s.to_string()];
    }
    let mut t = parts.join("\n").trim().to_string();
    if t.is_empty() {
        t = "(it gave nothing back)".into();
    }
    if t.chars().count() > RESULT_CHARS {
        t = t.chars().take(RESULT_CHARS).collect::<String>() + "\n[…cut]";
    }
    (t, r["isError"] != true)
}

// --- Talking to a server ----------------------------------------------------------------------------

enum Session {
    Http { url: String, headers: Map<String, Value>, id: Option<String>, version: String },
    Stdio { child: std::process::Child, stdin: std::process::ChildStdin, lines: Receiver<String> },
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Session::Stdio { child, .. } = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(CALL_TIMEOUT)).http_status_as_error(false).proxy(None).build().new_agent()
}

/// Reads an SSE body until the JSON-RPC answer with `id`.
fn sse_answer(reader: impl std::io::Read, id: u64) -> Result<Value, String> {
    let mut data = String::new();
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|e| format!("The app's server dropped the answer: {e}"))?;
        if let Some(d) = line.strip_prefix("data:") {
            data.push_str(d.trim_start());
        } else if line.is_empty() && !data.is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&data)
                && v["id"] == id
            {
                return Ok(v);
            }
            data.clear();
        }
    }
    if let Ok(v) = serde_json::from_str::<Value>(&data)
        && v["id"] == id
    {
        return Ok(v);
    }
    Err("The app's server ended without answering.".into())
}

impl Session {
    /// One JSON-RPC exchange (or a notification when `id` is None).
    fn exchange(&mut self, id: Option<u64>, method: &str, params: Value, stop: &AtomicBool) -> Result<Value, String> {
        let mut msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        if let Some(i) = id {
            msg["id"] = i.into();
        }
        match self {
            Session::Http { url, headers, id: sid, version } => {
                let a = http_agent();
                let mut req = a.post(url.as_str()).header("Accept", "application/json, text/event-stream").header("MCP-Protocol-Version", version.as_str());
                for (k, v) in headers.iter() {
                    req = req.header(k.as_str(), v.as_str().unwrap_or(""));
                }
                if let Some(s) = sid.as_ref() {
                    req = req.header("Mcp-Session-Id", s.as_str());
                }
                let mut r = req.content_type("application/json").send(msg.to_string()).map_err(|e| format!("Couldn't reach the app's server: {e}"))?;
                let status = r.status().as_u16();
                if let Some(s) = r.headers().get("mcp-session-id").and_then(|h| h.to_str().ok()) {
                    *sid = Some(s.to_string());
                }
                let ctype = r.headers().get("content-type").and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
                match status {
                    401 | 403 => return Err("The app's server refused CubicAI's sign-in (check the token in its settings).".into()),
                    404 if sid.is_some() => return Err("session gone".into()),
                    404 | 405 => return Err("Nothing that speaks MCP answered at that address (CubicAI needs the Streamable HTTP kind, usually …/mcp).".into()),
                    s if s >= 400 => return Err(format!("The app's server said {s}.")),
                    _ => {}
                }
                let Some(want) = id else { return Ok(Value::Null) };
                let v = if ctype.starts_with("text/event-stream") {
                    sse_answer(r.body_mut().as_reader(), want)?
                } else {
                    r.body_mut().read_json::<Value>().map_err(|_| "The app's server sent something that isn't MCP.".to_string())?
                };
                rpc_result(v)
            }
            Session::Stdio { child, stdin, lines } => {
                writeln!(stdin, "{msg}").map_err(|_| "The app stopped.".to_string())?;
                stdin.flush().map_err(|_| "The app stopped.".to_string())?;
                let Some(want) = id else { return Ok(Value::Null) };
                let end = Instant::now() + if method == "initialize" { START_TIMEOUT } else { CALL_TIMEOUT };
                loop {
                    if stop.load(Ordering::SeqCst) {
                        let _ = writeln!(stdin, "{}", json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": want, "reason": "stopped"}}));
                        return Err("Stopped.".into());
                    }
                    let left = end.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return Err(format!("The app didn't answer within {} s.", if method == "initialize" { START_TIMEOUT } else { CALL_TIMEOUT }.as_secs()));
                    }
                    match lines.recv_timeout(left.min(Duration::from_millis(200))) {
                        Ok(line) => {
                            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
                            if v["id"] == want && v.get("method").is_none() {
                                return rpc_result(v);
                            }
                            // A request from the server (ping, roots/list …): answer so it doesn't wait.
                            if let (Some(rid), Some(m)) = (v.get("id"), v["method"].as_str()) {
                                let reply = if m == "ping" { json!({"jsonrpc": "2.0", "id": rid, "result": {}}) } else { json!({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "not supported"}}) };
                                let _ = writeln!(stdin, "{reply}");
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => {
                            let _ = child.try_wait();
                            return Err("The app stopped (if it's new, check its command; it runs inside the Bots' computer).".into());
                        }
                    }
                }
            }
        }
    }
}

fn rpc_result(v: Value) -> Result<Value, String> {
    if let Some(e) = v.get("error") {
        return Err(format!("The app said: {}", e["message"].as_str().unwrap_or("an error")));
    }
    Ok(v["result"].clone())
}

pub struct Mcp {
    file: PathBuf,
    lock: Mutex<()>,
    sessions: Mutex<HashMap<String, Arc<Mutex<Option<Session>>>>>,
    tools: Mutex<HashMap<String, (Instant, Vec<Tool>)>>,
    next: AtomicU64,
}

impl Mcp {
    pub fn new(config_dir: &Path) -> Mcp {
        Mcp { file: config_dir.join("mcp.json"), lock: Mutex::new(()), sessions: Mutex::new(HashMap::new()), tools: Mutex::new(HashMap::new()), next: AtomicU64::new(1) }
    }

    pub fn servers(&self) -> Vec<Server> {
        let v: Value = std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        v["servers"].as_array().into_iter().flatten().filter_map(Server::from_json).collect()
    }

    pub fn server(&self, id: &str) -> Option<Server> {
        self.servers().into_iter().find(|s| s.id == id)
    }

    fn save(&self, all: &[Server]) -> Result<(), String> {
        super::write_private(&self.file, serde_json::to_string_pretty(&json!({"servers": all.iter().map(Server::to_json).collect::<Vec<_>>()})).unwrap().as_bytes())
    }

    /// Drops a server's session and its cached tools (after a change, or when it broke).
    fn forget(&self, id: &str) {
        self.sessions.lock().unwrap().remove(id);
        self.tools.lock().unwrap().remove(id);
    }

    pub fn add(&self, v: &Value) -> Result<Server, String> {
        let _g = self.lock.lock().unwrap();
        let mut all = self.servers();
        if all.len() >= MAX_SERVERS {
            return Err(format!("You can add up to {MAX_SERVERS} apps."));
        }
        let kind = match v["kind"].as_str() {
            Some("remote") => Kind::Remote { url: String::new(), headers: Map::new() },
            Some("sandbox") => Kind::Sandbox { command: String::new(), env: Map::new() },
            _ => return Err("kind is remote (an address) or sandbox (a command run in the Bots' computer).".into()),
        };
        let taken: Vec<String> = all.iter().map(|s| s.id.clone()).collect();
        let mut s = Server { id: slug(v["name"].as_str().unwrap_or(""), &taken), name: String::new(), kind, ask: Ask::Changes, enabled: true };
        s.apply(v)?;
        all.push(s.clone());
        self.save(&all)?;
        Ok(s)
    }

    pub fn update(&self, id: &str, v: &Value) -> Result<Server, String> {
        let _g = self.lock.lock().unwrap();
        let mut all = self.servers();
        let s = all.iter_mut().find(|s| s.id == id).ok_or("There's no such app.")?;
        s.apply(v)?;
        let out = s.clone();
        self.save(&all)?;
        self.forget(id);
        Ok(out)
    }

    pub fn delete(&self, id: &str) -> bool {
        let _g = self.lock.lock().unwrap();
        let mut all = self.servers();
        let before = all.len();
        all.retain(|s| s.id != id);
        if all.len() == before {
            return false;
        }
        let _ = self.save(&all);
        self.forget(id);
        true
    }

    fn connect(&self, s: &Server, computer: Option<&super::computer::Computer>, stop: &AtomicBool) -> Result<Session, String> {
        let mut session = match &s.kind {
            Kind::Remote { url, headers } => Session::Http { url: url.clone(), headers: headers.clone(), id: None, version: PROTOCOL.into() },
            Kind::Sandbox { command, env } => {
                let c = computer.ok_or("This app runs inside the Bots' computer, which isn't set up on this PC (cubicai-gateway computer-setup).")?;
                let env: Vec<(String, String)> = env.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect();
                let mut child = c.spawn(command, &env)?;
                let stdin = child.stdin.take().unwrap();
                let out = child.stdout.take().unwrap();
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    for line in BufReader::new(out).lines() {
                        let Ok(l) = line else { break };
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                });
                Session::Stdio { child, stdin, lines: rx }
            }
        };
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let init = session.exchange(Some(id), "initialize", json!({"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "CubicAI", "version": env!("CARGO_PKG_VERSION")}}), stop)?;
        if let Session::Http { version, .. } = &mut session
            && let Some(v) = init["protocolVersion"].as_str()
        {
            *version = v.to_string();
        }
        session.exchange(None, "notifications/initialized", json!({}), stop)?;
        Ok(session)
    }

    /// A request to a server, connecting (or reconnecting once, when its session went away) as needed.
    fn request(&self, s: &Server, computer: Option<&super::computer::Computer>, method: &str, params: Value, stop: &AtomicBool) -> Result<Value, String> {
        let slot = self.sessions.lock().unwrap().entry(s.id.clone()).or_default().clone();
        let mut g = slot.lock().unwrap();
        for attempt in 0..2 {
            if g.is_none() {
                *g = Some(self.connect(s, computer, stop)?);
            }
            if let (Some(c), Kind::Sandbox { .. }) = (computer, &s.kind) {
                c.touch();
            }
            let id = self.next.fetch_add(1, Ordering::SeqCst);
            match g.as_mut().unwrap().exchange(Some(id), method, params.clone(), stop) {
                Ok(v) => return Ok(v),
                Err(e) if attempt == 0 && (e == "session gone" || e.starts_with("The app stopped")) && !stop.load(Ordering::SeqCst) => *g = None,
                Err(e) => {
                    if !e.starts_with("The app said") {
                        *g = None;
                    }
                    return Err(if e == "session gone" { "The app's server lost CubicAI's session.".into() } else { e });
                }
            }
        }
        unreachable!()
    }

    /// A server's tools (asked again at most every few minutes, or when `fresh`).
    pub fn tools(&self, s: &Server, computer: Option<&super::computer::Computer>, fresh: bool) -> Result<Vec<Tool>, String> {
        if !fresh
            && let Some((at, t)) = self.tools.lock().unwrap().get(&s.id)
            && at.elapsed() < LIST_TTL
        {
            return Ok(t.clone());
        }
        let never = AtomicBool::new(false);
        let mut out = vec![];
        let mut cursor: Option<String> = None;
        for _ in 0..10 {
            let params = cursor.as_ref().map(|c| json!({"cursor": c})).unwrap_or_else(|| json!({}));
            let r = self.request(s, computer, "tools/list", params, &never)?;
            for t in r["tools"].as_array().into_iter().flatten() {
                let Some(name) = t["name"].as_str() else { continue };
                out.push(Tool {
                    name: name.to_string(),
                    description: t["description"].as_str().unwrap_or("").chars().take(1000).collect(),
                    schema: t["inputSchema"].clone(),
                    read_only: t["annotations"]["readOnlyHint"] == true,
                });
            }
            cursor = r["nextCursor"].as_str().map(String::from);
            if cursor.is_none() {
                break;
            }
        }
        self.tools.lock().unwrap().insert(s.id.clone(), (Instant::now(), out.clone()));
        Ok(out)
    }

    /// The tools a Bot gets from its servers, and the ones that couldn't be reached (by name).
    pub fn offered(&self, ids: &[String], computer: Option<&super::computer::Computer>) -> (Vec<Offered>, Vec<String>) {
        let mut out = vec![];
        let mut down = vec![];
        for s in self.servers().into_iter().filter(|s| s.enabled && ids.contains(&s.id)) {
            match self.tools(&s, computer, false) {
                Ok(tools) => {
                    for t in tools {
                        if out.len() >= MAX_TOOLS {
                            break;
                        }
                        let tname: String = t.name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
                        let mut function = format!("{}__{tname}", s.id);
                        function.truncate(64);
                        let description = format!("[{}] {}", s.name, t.description);
                        out.push(Offered {
                            definition: json!({"type": "function", "function": {"name": function, "description": description, "parameters": clean_schema(&t.schema)}}),
                            function,
                            server: s.id.clone(),
                            tool: t.name,
                            read_only: t.read_only,
                        });
                    }
                }
                Err(e) => {
                    crate::log(&format!("mcp {}: not reachable ({})", s.id, e.chars().take(80).collect::<String>()));
                    down.push(s.name.clone());
                }
            }
        }
        (out, down)
    }

    /// What to ask the user before this call, if anything.
    pub fn needs_ask(&self, o: &Offered, args: &Value) -> Option<String> {
        let s = self.server(&o.server)?;
        let ask = match s.ask {
            Ask::Never => false,
            Ask::Always => true,
            Ask::Changes => !o.read_only,
        };
        ask.then(|| {
            let mut shown = args.to_string();
            if shown.chars().count() > 300 {
                shown = shown.chars().take(300).collect::<String>() + "…";
            }
            format!("Use {}: {} {shown}", s.name, o.tool)
        })
    }

    pub fn call(&self, o: &Offered, args: Value, computer: Option<&super::computer::Computer>, stop: &AtomicBool) -> Result<(String, bool), String> {
        let s = self.server(&o.server).ok_or("That app was removed.")?;
        let r = self.request(&s, computer, "tools/call", json!({"name": o.tool, "arguments": args}), stop)?;
        Ok(result_text(&r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_short_and_unique() {
        assert_eq!(slug("My GitHub!", &[]), "my_github");
        assert_eq!(slug("My GitHub!", &["my_github".into()]), "my_github2");
        assert_eq!(slug("42", &[]), "app42");
        assert_eq!(slug("", &[]), "app");
        assert_eq!(slug("A very long name for a server", &[]).len(), 16);
    }

    #[test]
    fn schemas_are_cut_to_what_providers_take() {
        let s = clean_schema(&json!({
            "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "additionalProperties": false,
            "properties": {"q": {"type": ["string", "null"], "format": "uri", "default": "x"}, "n": {"type": "integer", "minimum": 1}},
            "required": ["q"]
        }));
        assert_eq!(s, json!({"type": "object", "properties": {"q": {"type": "string", "nullable": true}, "n": {"type": "integer", "minimum": 1}}, "required": ["q"]}));
        assert_eq!(clean_schema(&Value::Null), json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn results_become_text() {
        let (t, ok) = result_text(&json!({"content": [{"type": "text", "text": "hi"}, {"type": "image", "data": "…"}], "isError": false}));
        assert_eq!((t.as_str(), ok), ("hi\n[an image]", true));
        let (t, ok) = result_text(&json!({"content": [], "structuredContent": {"a": 1}, "isError": true}));
        assert_eq!((t.as_str(), ok), ("{\"a\":1}", false));
    }

    #[test]
    fn answers_come_out_of_event_streams() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\n\n";
        assert!(sse_answer(body.as_bytes(), 7).is_err(), "a broken message isn't an answer");
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{}}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":1}}\n\n";
        assert_eq!(sse_answer(body.as_bytes(), 7).unwrap()["result"]["ok"], 1);
    }

    #[test]
    fn secrets_are_never_shown() {
        let mut s = Server { id: "gh".into(), name: "GitHub".into(), kind: Kind::Remote { url: String::new(), headers: Map::new() }, ask: Ask::Changes, enabled: true };
        s.apply(&json!({"url": "https://api.example/mcp", "headers": {"Authorization": "Bearer secret"}})).unwrap();
        let p = s.to_public();
        assert_eq!(p["headers"], json!(["Authorization"]));
        assert!(!p.to_string().contains("secret"));
        s.apply(&json!({"headers": {"Authorization": ""}})).unwrap();
        assert_eq!(s.to_public()["headers"], json!([]));
        assert!(s.apply(&json!({"url": "ftp://x"})).is_err());
        assert!(s.apply(&json!({"headers": {"bad name": "x"}})).is_err());
    }
}
