//! Bots end to end over the phone's TLS: keys, making and changing Bots, runs with tools against a
//! fake OpenAI-compatible provider and a fake Ollama, following a run, stopping it, a message sent
//! mid-run. Plus the workspace's path rules.

mod common;

use common::*;
use cubic_gateway::bots::{tools::safe_path, Bots};
use cubic_gateway::http::Conn;
use cubic_gateway::{bots, identity, server, Config};
use serde_json::{json, Value};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// --- A fake provider (OpenAI-style), Ollama and SearXNG -----------------------------------------

#[derive(Default)]
struct Seen {
    cloud: Vec<Value>,
    ollama: Vec<Value>,
    auth: Vec<String>,
    /// Google calls: (method path?query, Authorization, body).
    google: Vec<(String, String, String)>,
    /// MCP calls: (method, session id, Authorization).
    mcp: Vec<(String, String, String)>,
}

fn fake() -> (String, Arc<Mutex<Seen>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", l.local_addr().unwrap().port());
    let seen: Arc<Mutex<Seen>> = Default::default();
    let s = seen.clone();
    std::thread::spawn(move || {
        for tcp in l.incoming().flatten() {
            let s = s.clone();
            std::thread::spawn(move || conn(tcp, s));
        }
    });
    (url, seen)
}

fn json_reply(c: &mut Conn<TcpStream>, v: &Value) {
    let body = v.to_string();
    let _ = c.raw(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
}

fn sse(c: &mut Conn<TcpStream>, chunks: &[Value], pause: Duration) {
    let _ = c.raw(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
    for ch in chunks {
        if c.raw(format!("data: {ch}\n\n").as_bytes()).is_err() {
            return;
        }
        std::thread::sleep(pause);
    }
    let _ = c.raw(b"data: [DONE]\n\n");
}

fn call(index: u64, id: &str, name: &str, args: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": args}}]}}]})
}

fn args_more(index: u64, args: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": args}}]}}]})
}

fn text(t: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {"content": t}}]})
}

fn finish(reason: &str) -> Value {
    json!({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
}

fn last_user(body: &Value) -> String {
    body["messages"].as_array().unwrap().iter().rev().find(|m| m["role"] == "user").and_then(|m| m["content"].as_str()).unwrap_or("").to_string()
}

fn conn(tcp: TcpStream, seen: Arc<Mutex<Seen>>) {
    let mut c = Conn::new(tcp);
    let Ok(Some(req)) = c.next_request() else { return };
    let raw = match req.content_length() {
        Ok(Some(n)) if n > 0 => c.read_body(n).unwrap(),
        _ => vec![],
    };
    let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    if req.path().starts_with("/g/") {
        return fake_google(&mut c, &req, &String::from_utf8_lossy(&raw), &seen);
    }
    if req.path() == "/mcp" {
        return fake_mcp(&mut c, &req, &body, &seen);
    }
    match req.path() {
        "/chat/completions" => {
            seen.lock().unwrap().cloud.push(body.clone());
            seen.lock().unwrap().auth.push(req.header("Authorization").unwrap_or("").to_string());
            let msgs = body["messages"].as_array().unwrap();
            let last = msgs.last().unwrap();
            let user = last_user(&body);
            let system = msgs[0]["content"].as_str().unwrap_or("");
            let lc = last["content"].as_str().unwrap_or("");
            // Group chats: the router, then each Bot by its name.
            if system.contains("You pick who answers") {
                return sse(&mut c, &[text("Writer"), finish("stop")], Duration::ZERO);
            }
            if system.starts_with("You are Researcher,") && last["role"] == "user" && lc.contains("find") {
                return sse(&mut c, &[text("Found three sources. @Writer please write it up."), finish("stop")], Duration::ZERO);
            }
            if system.starts_with("You are Writer,") && last["role"] == "user" && lc.starts_with("[Researcher said:]") {
                return sse(&mut c, &[text("Written up. Thanks @Researcher!"), finish("stop")], Duration::ZERO);
            }
            if system.starts_with("You are Writer,") && last["role"] == "user" && lc.contains("who's free") {
                return sse(&mut c, &[text("Writer here."), finish("stop")], Duration::ZERO);
            }
            // Pictures: one made, one that fails, one saved outside the workspace.
            if last["role"] == "user" && user.contains("draw") {
                let args = if user.contains("broken") {
                    r#"{"prompt": "FAIL a tea shop"}"#
                } else if user.contains("outside") {
                    r#"{"prompt": "a tea shop", "save_as": "../escape.png"}"#
                } else {
                    r#"{"prompt": "A tea shop by the river, watercolour", "shape": "landscape"}"#
                };
                return sse(&mut c, &[call(0, "p1", "make_picture", args), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "make_picture" {
                return sse(&mut c, &[text("Here it is."), finish("stop")], Duration::ZERO);
            }
            // Handoffs: Scout hands a haiku to Writer.
            if last["role"] == "user" && user.contains("ask the writer") {
                return sse(&mut c, &[call(0, "h1", "ask_bot", r#"{"bot": "writer", "task": "Write a haiku about rain."}"#), finish("tool_calls")], Duration::ZERO);
            }
            if system.starts_with("You are Writer,") && last["role"] == "user" && lc.contains("(Scout, another of the user's Bots, asks you this") {
                return sse(&mut c, &[text("Soft rain on the roof."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "ask_bot" && user.contains("twice") && msgs.iter().rev().take_while(|m| m["role"] != "user").filter(|m| m["role"] == "tool").count() == 1 {
                return sse(&mut c, &[call(0, "h2", "ask_bot", r#"{"bot": "Writer", "task": "Write a haiku about rain, please."}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "ask_bot" {
                return sse(&mut c, &[text(&format!("From Writer: {}", lc.lines().last().unwrap_or(""))), finish("stop")], Duration::ZERO);
            }
            // Skills: one proposed, and a taught task written up.
            if last["role"] == "user" && (user.contains("save how") || user.contains("just showed you how")) {
                return sse(&mut c, &[call(0, "s1", "save_skill", r#"{"name": "Find Cats", "about": "Find a cat picture", "steps": "1. Open the site\n2. Search for cats"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "save_skill" {
                return sse(&mut c, &[text("Proposed /find-cats."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("/check-bill") {
                return sse(&mut c, &[text("Following the skill."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("write_file now") {
                return sse(&mut c, &[call(0, "w1", "write_file", r#"{"path": "notes/lazy.md", "content": "lazy"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("lazy") {
                return sse(&mut c, &[text("Here it is. I saved it to `notes/lazy.md` for you."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "write_file" && last["content"].as_str().unwrap_or("").contains("lazy") {
                return sse(&mut c, &[text("Saved notes/lazy.md."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("slow") {
                let words: Vec<Value> = (0..40).map(|i| text(&format!("w{i} "))).collect();
                return sse(&mut c, &[words, vec![finish("stop")]].concat(), Duration::from_millis(40));
            }
            if last["role"] == "user" && user.contains("shop") {
                return sse(&mut c, &[call(0, "b1", "browser_open", r#"{"url": "https://shop.example"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "user" && lc.contains("took over your computer's browser") {
                return sse(&mut c, &[text("Thanks, carrying on from the checkout."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "browser_open" {
                return sse(&mut c, &[call(0, "b2", "browser_click", r#"{"element": 1}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "browser_click" {
                return sse(&mut c, &[text("Done with the shop."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("terminal") {
                return sse(&mut c, &[call(0, "t1", "run_command", r#"{"command": "echo from-computer; exit 3"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "run_command" {
                return sse(&mut c, &[text("Ran it."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("my mail") {
                return sse(&mut c, &[call(0, "g1", "gmail_search", r#"{"query": "is:unread"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "gmail_search" {
                return sse(&mut c, &[call(0, "g2", "gmail_read", r#"{"id": "m1"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "gmail_read" {
                return sse(&mut c, &[call(0, "g3", "gmail_send", r#"{"to": "bob@example.com", "body": "Yes, noon works.", "reply_to_id": "m1"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "gmail_send" {
                return sse(&mut c, &[text("Replied to Bob."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("file the bug") {
                return sse(&mut c, &[call(0, "m1", "github__get_issue", r#"{"number": 7}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "github__get_issue" {
                return sse(&mut c, &[call(0, "m2", "github__create_issue", r#"{"title": "Login broken again"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "github__create_issue" {
                return sse(&mut c, &[text("Filed it."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("remind me") {
                let at = format!("{}T18:30", cubic_gateway::time::local(cubic_gateway::time::now() as i64 + 86400).date());
                return sse(&mut c, &[call(0, "p1", "phone_remind", &json!({"at": at, "text": "Call mum"}).to_string()), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("where am i") {
                return sse(&mut c, &[call(0, "p2", "phone_location", "{}"), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && (last["name"] == "phone_remind" || last["name"] == "phone_location") {
                return sse(&mut c, &[text("Done."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("every morning") {
                return sse(&mut c, &[call(0, "r1", "schedule_routine", r#"{"name": "Morning news", "task": "Give me the news (plain).", "kind": "daily", "time": "08:00"}"#), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "schedule_routine" {
                return sse(&mut c, &[text("I've proposed it: approve it on your phone."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("quiet") {
                return sse(&mut c, &[text("Nothing new."), finish("stop")], Duration::ZERO);
            }
            if last["role"] == "user" && user.contains("plain") {
                return sse(&mut c, &[text("Plain "), text("answer."), finish("stop"), json!({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 3}})], Duration::ZERO);
            }
            if last["role"] == "user" {
                // The arguments arrive in two pieces, as real providers send them.
                return sse(&mut c, &[call(0, "c1", "web_search", "{\"query\":"), args_more(0, " \"cubicai news\"}"), finish("tool_calls")], Duration::ZERO);
            }
            if last["role"] == "tool" && last["name"] == "web_search" {
                return sse(
                    &mut c,
                    &[call(0, "c2", "write_file", r##"{"path": "notes/news.md", "content": "# News\n"}"##), call(1, "c3", "remember", r#"{"note": "Likes short notes"}"#), finish("tool_calls")],
                    Duration::ZERO,
                );
            }
            sse(&mut c, &[text("Saved "), text("notes/news.md."), finish("stop")], Duration::ZERO)
        }
        "/api/show" => {
            let caps = if body["model"] == "plain" { json!(["completion"]) } else { json!(["completion", "tools"]) };
            json_reply(&mut c, &json!({"capabilities": caps, "parameters": "", "model_info": {"llama.context_length": 32768}}));
        }
        "/api/chat" => {
            seen.lock().unwrap().ollama.push(body.clone());
            let last = body["messages"].as_array().unwrap().last().unwrap().clone();
            let _ = c.raw(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n");
            let lines = if last["role"] == "user" {
                vec![json!({"message": {"content": "", "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}]}, "done": false}), json!({"message": {"content": ""}, "done": true, "prompt_eval_count": 10, "eval_count": 2})]
            } else {
                vec![json!({"message": {"content": "Your workspace has "}, "done": false}), json!({"message": {"content": "notes."}, "done": true, "prompt_eval_count": 30, "eval_count": 4})]
            };
            for l in lines {
                let _ = c.raw((l.to_string() + "\n").as_bytes());
            }
        }
        "/search" => json_reply(&mut c, &json!({"results": [{"title": "CubicAI news", "url": "https://example.com/news", "content": "New Bots."}]})),
        _ => {
            let _ = c.raw(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        }
    }
}

/// Google's token endpoint, userinfo and the bits of Gmail the test uses. "at1" (the first access
/// token) is refused by Gmail, so the gateway has to refresh it.
fn fake_google(c: &mut Conn<TcpStream>, req: &cubic_gateway::http::Request, raw: &str, seen: &Mutex<Seen>) {
    let auth = req.header("Authorization").unwrap_or("").to_string();
    seen.lock().unwrap().google.push((format!("{} {}", req.method, req.target), auth.clone(), raw.to_string()));
    let b64 = |s: &str| base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, s);
    let reply = match req.path() {
        "/g/token" if raw.contains("grant_type=authorization_code") => json!({
            "access_token": "at1", "refresh_token": "rt1", "expires_in": 3600,
            "scope": "openid email https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/drive"
        }),
        "/g/token" if raw.contains("grant_type=refresh_token") && raw.contains("refresh_token=rt1") => json!({"access_token": "at2", "expires_in": 3600}),
        "/g/userinfo" => json!({"email": "me@example.com"}),
        _ if auth != "Bearer at2" => {
            let _ = c.raw(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
            return;
        }
        "/g/gmail/v1/users/me/messages" => json!({"messages": [{"id": "m1"}]}),
        "/g/gmail/v1/users/me/messages/m1" if req.query("format").as_deref() == Some("full") => json!({
            "id": "m1", "threadId": "t1",
            "payload": {"mimeType": "multipart/alternative", "headers": [{"name": "From", "value": "Bob <bob@example.com>"}, {"name": "Subject", "value": "Lunch"}],
                "parts": [{"mimeType": "text/plain", "body": {"data": b64("Lunch at noon?")}}, {"mimeType": "text/html", "body": {"data": b64("<p>Lunch at noon?</p>")}}]}
        }),
        "/g/gmail/v1/users/me/messages/m1" => json!({
            "id": "m1", "threadId": "t1", "labelIds": ["UNREAD", "INBOX"], "snippet": "Lunch at noon?",
            "payload": {"headers": [{"name": "From", "value": "Bob <bob@example.com>"}, {"name": "Subject", "value": "Lunch"}, {"name": "Date", "value": "Mon, 9 Nov 2026 10:00"}, {"name": "Message-ID", "value": "<m1@mail>"}]}
        }),
        "/g/gmail/v1/users/me/messages/send" => json!({"id": "s1"}),
        _ => {
            let _ = c.raw(b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
            return;
        }
    };
    json_reply(c, &reply);
}

/// A remote MCP server (Streamable HTTP): a session id from initialize, a token on every call, and
/// tool results as server-sent events after a progress notification.
fn fake_mcp(c: &mut Conn<TcpStream>, req: &cubic_gateway::http::Request, body: &Value, seen: &Mutex<Seen>) {
    let method = body["method"].as_str().unwrap_or("").to_string();
    let sid = req.header("Mcp-Session-Id").unwrap_or("").to_string();
    let auth = req.header("Authorization").unwrap_or("").to_string();
    seen.lock().unwrap().mcp.push((method.clone(), sid.clone(), auth.clone()));
    let send = |c: &mut Conn<TcpStream>, extra: &str, v: Value| {
        let b = v.to_string();
        let _ = c.raw(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len()).as_bytes());
    };
    if auth != "Bearer mcp-secret" {
        let _ = c.raw(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }
    let id = body["id"].clone();
    match method.as_str() {
        "initialize" => send(c, "Mcp-Session-Id: sess-1\r\n", json!({"jsonrpc": "2.0", "id": id, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}})),
        "notifications/initialized" => {
            let _ = c.raw(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        }
        _ if sid != "sess-1" => {
            let _ = c.raw(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        }
        "tools/list" => send(c, "", json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [
            {"name": "get_issue", "description": "Read an issue.", "inputSchema": {"$schema": "x", "type": "object", "properties": {"number": {"type": "integer"}}, "additionalProperties": false}, "annotations": {"readOnlyHint": true}},
            {"name": "create_issue", "description": "Open an issue.", "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}
        ]}})),
        "tools/call" => {
            let text = match body["params"]["name"].as_str() {
                Some("get_issue") => format!("Issue {}: Login is broken", body["params"]["arguments"]["number"]),
                _ => format!("Created issue 8: {}", body["params"]["arguments"]["title"].as_str().unwrap_or("")),
            };
            let _ = c.raw(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            let _ = c.raw(format!("data: {}\n\n", json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})).as_bytes());
            let _ = c.raw(format!("event: message\ndata: {}\n\n", json!({"jsonrpc": "2.0", "id": id, "result": {"content": [{"type": "text", "text": text}]}})).as_bytes());
        }
        _ => send(c, "", json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "no"}})),
    }
}

// --- The gateway ----------------------------------------------------------------------------------

struct Setup {
    gw: server::Running,
    token: String,
    pin: String,
    seen: Arc<Mutex<Seen>>,
    dir: std::path::PathBuf,
    url: String,
}

impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn setup(name: &str) -> Setup {
    setup_with(name, false)
}

/// A docker that answers like the computer would: the terminal runs locally in the workspace, the
/// browser shows a shop page whose Buy button submits a POST form.
fn fake_docker(dir: &std::path::Path) -> String {
    let script = dir.join("docker");
    let shots = dir.join("computer/shots");
    let ws = dir.join("workspace");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        &script,
        format!(
            r#"#!/bin/bash
case "$1" in
  image|rm|run) exit 0 ;;
  inspect) echo true; exit 0 ;;
esac
args=" $* "
if [[ "$args" == *" /opt/cubic/browse.py "* ]]; then
  exec python3 -c '
import json, sys
c = json.load(sys.stdin)
shot = {{"ok": True, "url": "https://shop.example/", "title": "Shop", "text": "Socks 5 EUR", "elements": [[1, "button", "Buy"]], "shot": "/shots/s.jpg", "scroll": [0, 900, 900]}}
if c.get("do") == "click" and not c.get("approved"):
    print(json.dumps({{"ok": True, "needs_approval": True, "what": "Submit a form that sends data to shop.example (fields: card)"}}))
    sys.exit()
if c.get("do") == "click":
    shot.update(title="Thanks", text="Order placed")
if c.get("do") in ("shot", "tap", "text", "key", "open_light", "scroll_light", "back_light"):
    did = None
    if c.get("do") == "tap":
        did = {{"kind": "button", "label": "Search"}}
    if c.get("do") == "text":
        did = {{"kind": "password field", "label": "Password", "secret": True}} if c.get("text") == "hunter2" else {{"kind": "search box", "label": "Search"}}
    url = "https://cats.example/results" if c.get("do") == "key" else "https://cats.example/"
    open("{shots}/t.jpg", "wb").write(bytes([0xff, 0xd8, 0xff, 0xe1]))
    print(json.dumps({{"ok": True, "url": url, "title": "Cats", "shot": "/shots/t.jpg", "did": did}}))
    sys.exit()
open("{shots}/s.jpg", "wb").write(bytes([0xff, 0xd8, 0xff, 0xe0]))
print(json.dumps(shot if c.get("do") != "ping" else {{"ok": True}}))
'
elif [[ "$args" == *" setsid "* ]]; then
  cd "{ws}" && exec bash -c "${{@: -2:1}}"
fi
exit 0
"#,
            shots = shots.display(),
            ws = ws.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    script.display().to_string()
}

fn setup_with(name: &str, computer: bool) -> Setup {
    setup_full(name, computer, false)
}

/// With `media`, a fake ComfyUI for making pictures.
fn setup_full(name: &str, computer: bool, media: bool) -> Setup {
    let dir = std::env::temp_dir().join(format!("cubic-gw-bots-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let media = media.then(|| common::comfy::setup(&dir).0);
    let (url, seen) = fake();
    let computer = computer.then(|| bots::computer::Setup { dir: dir.join("computer"), docker: fake_docker(&dir), name: "fake".into(), image: "fake".into() });
    let paths = bots::Paths { bots: dir.join("bots"), workspace: dir.join("workspace"), computer, google: Some(format!("{url}/g")) };
    let config = Config { dir: dir.join("config"), ollama: url.clone(), searxng: url.clone(), default_ctx: 16384, media, voice: None, bots: Some(paths) };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    Setup { gw, token: id.token, pin: id.fingerprint, seen, dir, url }
}

impl Setup {
    fn ask(&self, method: &str, path: &str, body: Option<Value>) -> Resp {
        let b = body.map(|v| v.to_string());
        ask(self.gw.port, &self.pin, method, path, Some(&self.token), b.as_deref())
    }

    fn make(&self, body: Value) -> String {
        let r = self.ask("POST", "/v1/bots", Some(body));
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["bot"]["id"].as_str().unwrap().to_string()
    }

    /// Waits for the Bot to finish and returns its chat.
    fn settle(&self, id: &str) -> Vec<Value> {
        let end = Instant::now() + Duration::from_secs(15);
        loop {
            let r = self.ask("GET", &format!("/v1/bots/{id}/messages"), None).json();
            if r["running"] == false {
                return r["messages"].as_array().unwrap().clone();
            }
            assert!(Instant::now() < end, "the Bot never finished");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn custom_keys(&self) {
        let r = self.ask("PUT", "/v1/keys", Some(json!({"custom": "sk-test", "custom_url": self.url})));
        assert_eq!(r.status, 200, "{}", r.body);
    }
}

#[test]
fn keys_stay_on_the_pc() {
    let s = setup("keys");
    let r = s.ask("PUT", "/v1/keys", Some(json!({"xai": "xai-secret", "gemini": "g-secret"})));
    assert_eq!(r.status, 200);
    let shown = s.ask("GET", "/v1/keys", None).json();
    assert_eq!(shown["keys"], json!({"xai": true, "gemini": true, "openrouter": false, "custom": false, "custom_url": null}));
    assert!(!shown.to_string().contains("secret"), "keys are never sent back");
    let file = s.dir.join("config/keys.json");
    assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&file).unwrap().permissions()) & 0o777, 0o600);
    // "" removes a key; unknown names are refused.
    s.ask("PUT", "/v1/keys", Some(json!({"xai": ""})));
    assert_eq!(s.ask("GET", "/v1/keys", None).json()["keys"]["xai"], false);
    assert_eq!(s.ask("PUT", "/v1/keys", Some(json!({"aws": "x"}))).status, 400);
    // Like every route, keys need the token.
    assert_eq!(ask(s.gw.port, &s.pin, "GET", "/v1/keys", None, None).status, 401);
}

#[test]
fn bots_are_made_checked_changed_and_deleted() {
    let s = setup("crud");
    for (bad, why) in [
        (json!({"name": "", "provider": "local", "model": "m"}), "name"),
        (json!({"name": "A", "provider": "local", "model": "m", "color": "blue"}), "colour"),
        (json!({"name": "A", "provider": "aws", "model": "m"}), "provider"),
        (json!({"name": "A", "provider": "local", "model": ""}), "model"),
        (json!({"name": "A", "provider": "xai", "model": "grok-4.20-multi-agent"}), "multi-agent"),
    ] {
        let r = s.ask("POST", "/v1/bots", Some(bad));
        assert_eq!(r.status, 400, "{why}: {}", r.body);
    }
    let id = s.make(json!({"name": "Scout", "job": "Find deals.", "color": "#ff8800", "provider": "local", "model": "gemma4:e4b", "memory": "- Lives in Kochi\n"}));
    let list = s.ask("GET", "/v1/bots", None).json();
    assert_eq!(list["bots"][0]["name"], "Scout");
    assert_eq!(list["bots"][0]["running"], false);
    assert_eq!(list["bots"][0]["last"], Value::Null);
    let r = s.ask("PUT", &format!("/v1/bots/{id}"), Some(json!({"name": "Deal Scout", "memory": "- Lives in Kochi\n- Budget 500\n"}))).json();
    assert_eq!((r["bot"]["name"].clone(), r["bot"]["job"].clone(), r["memory"].clone()), (json!("Deal Scout"), json!("Find deals."), json!("- Lives in Kochi\n- Budget 500\n")));
    assert_eq!(s.ask("GET", "/v1/bots/0000000000000000", None).status, 404);
    assert_eq!(s.ask("GET", "/v1/bots/../../etc", None).status, 404, "ids are checked");
    assert_eq!(s.ask("DELETE", &format!("/v1/bots/{id}"), None).status, 200);
    assert_eq!(s.ask("GET", "/v1/bots", None).json()["bots"], json!([]));
}

#[test]
fn a_cloud_bot_searches_writes_and_remembers() {
    let s = setup("cloud");
    s.custom_keys();
    let id = s.make(json!({"name": "Reporter", "job": "Keep notes on CubicAI.", "provider": "custom", "model": "test-model"}));
    let r = s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "Any news?"})));
    assert_eq!((r.status, r.json()["queued"].clone()), (200, json!(false)));
    // Following from the start: every step of the run, then the end.
    let ev: Vec<Value> = s.ask("GET", &format!("/v1/bots/{id}/events?after=0"), None).events().iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    let kinds: Vec<&str> = ev.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(kinds.first(), Some(&"start"));
    assert_eq!(kinds.last(), Some(&"done"));
    assert_eq!(ev.iter().filter(|e| e["type"] == "tool").map(|e| e["name"].as_str().unwrap()).collect::<Vec<_>>(), ["web_search", "write_file", "remember"]);
    let words: String = ev.iter().filter(|e| e["type"] == "text").map(|e| e["delta"].as_str().unwrap()).collect();
    assert_eq!(words, "Saved notes/news.md.");
    assert!(ev.iter().enumerate().all(|(i, e)| e["seq"] == i as u64), "events are numbered");

    let chat = s.settle(&id);
    let roles: Vec<&str> = chat.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "assistant", "tool", "tool", "assistant"]);
    assert_eq!(chat[1]["calls"][0], json!({"id": "c1", "name": "web_search", "args": "{\"query\": \"cubicai news\"}"}));
    assert_eq!(chat[2]["citations"], json!(["https://example.com/news"]));
    assert!(chat[2]["text"].as_str().unwrap().contains("1. CubicAI news\n   https://example.com/news\n   New Bots."));
    assert_eq!(chat[6]["text"], "Saved notes/news.md.");
    assert_eq!(std::fs::read_to_string(s.dir.join("workspace/notes/news.md")).unwrap(), "# News\n");
    assert_eq!(s.ask("GET", &format!("/v1/bots/{id}"), None).json()["memory"], "- Likes short notes\n");

    // What the provider was sent: the key, the tools, the Bot's job and memory, the tool results in order.
    let seen = s.seen.lock().unwrap();
    assert_eq!(seen.cloud.len(), 3);
    assert!(seen.auth.iter().all(|a| a == "Bearer sk-test"));
    let first = &seen.cloud[0];
    assert_eq!(first["model"], "test-model");
    assert_eq!(first["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect::<Vec<_>>(), ["web_search", "read_page", "list_files", "read_file", "write_file", "remember", "forget", "schedule_routine", "cancel_routine", "use_skill", "save_skill", "phone_remind", "phone_notify", "phone_copy", "phone_open", "phone_location", "phone_status"]);
    let system = first["messages"][0]["content"].as_str().unwrap();
    assert!(system.starts_with("You are Reporter,") && system.contains("Your job:\nKeep notes on CubicAI.") && system.contains("(nothing yet)"), "{system}");
    let third = seen.cloud[2]["messages"].as_array().unwrap();
    let shape: Vec<(String, String)> = third.iter().map(|m| (m["role"].as_str().unwrap().to_string(), m["name"].as_str().unwrap_or("").to_string())).collect();
    assert_eq!(shape, [("system", ""), ("user", ""), ("assistant", ""), ("tool", "web_search"), ("assistant", ""), ("tool", "write_file"), ("tool", "remember")].map(|(a, b)| (a.to_string(), b.to_string())));
    assert_eq!(third[5]["tool_call_id"], "c2");
    assert!(seen.cloud[2]["messages"][0]["content"].as_str().unwrap().contains("- Likes short notes"), "a fresh turn sees the new memory");
    drop(seen);

    let list = s.ask("GET", "/v1/bots", None).json();
    assert_eq!(list["bots"][0]["last"]["text"], "Saved notes/news.md.");
}

#[test]
fn a_local_bot_uses_its_tools_through_ollama() {
    let s = setup("local");
    std::fs::create_dir_all(s.dir.join("workspace/notes")).unwrap();
    std::fs::write(s.dir.join("workspace/notes/a.md"), "hi").unwrap();
    let id = s.make(json!({"name": "Keeper", "provider": "local", "model": "toolsy"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "What's in the workspace?"})));
    let chat = s.settle(&id);
    assert_eq!(chat.iter().map(|m| m["role"].as_str().unwrap()).collect::<Vec<_>>(), ["user", "assistant", "tool", "assistant"]);
    assert_eq!(chat[2]["text"], "notes/\nnotes/a.md (2 bytes)");
    assert_eq!(chat[3]["text"], "Your workspace has notes.");
    assert_eq!(chat[3]["usage"], json!({"prompt": 30, "output": 4}));
    let seen = s.seen.lock().unwrap();
    let second = &seen.ollama[1];
    assert_eq!(second["options"]["num_ctx"], 16384);
    let msgs = second["messages"].as_array().unwrap();
    // Ollama's own shape: arguments as objects, the tool result named.
    assert_eq!(msgs[2]["tool_calls"][0]["function"], json!({"name": "list_files", "arguments": {}}));
    assert_eq!((msgs[3]["role"].clone(), msgs[3]["tool_name"].clone()), (json!("tool"), json!("list_files")));
    drop(seen);

    // A model that can't use tools can't be a Bot: the chat says so.
    let plain = s.make(json!({"name": "Plain", "provider": "local", "model": "plain"}));
    s.ask("POST", &format!("/v1/bots/{plain}/messages"), Some(json!({"text": "hi"})));
    let chat = s.settle(&plain);
    assert!(chat[1]["error"].as_str().unwrap().contains("can't use tools"), "{}", chat[1]);
}

#[test]
fn a_cloud_bot_without_its_key_says_so() {
    let s = setup("nokey");
    let id = s.make(json!({"name": "Grokker", "provider": "xai", "model": "grok-4.7"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "hi"})));
    let chat = s.settle(&id);
    assert!(chat[1]["error"].as_str().unwrap().contains("no Grok key"), "{}", chat[1]);
}

#[test]
fn stop_ends_a_run_and_keeps_what_was_written() {
    let s = setup("stop");
    s.custom_keys();
    let id = s.make(json!({"name": "Talker", "provider": "custom", "model": "m"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "slow please"})));
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(s.ask("GET", &format!("/v1/bots/{id}/messages"), None).json()["running"], true);
    assert_eq!(s.ask("DELETE", &format!("/v1/bots/{id}/run"), None).json()["stopped"], true);
    let chat = s.settle(&id);
    let reply = &chat[1];
    assert_eq!(reply["stopped"], true);
    let words = reply["text"].as_str().unwrap();
    assert!(words.starts_with("w0 w1 ") && !words.contains("w39"), "{words}");
    let ev: Vec<Value> = s.ask("GET", &format!("/v1/bots/{id}/events?after=0"), None).events().iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert_eq!(ev.last().unwrap()["stopped"], true);
    assert_eq!(s.ask("DELETE", &format!("/v1/bots/{id}/run"), None).json()["stopped"], false, "nothing left to stop");
}

#[test]
fn a_message_sent_while_working_is_answered_after_the_reply() {
    let s = setup("inbox");
    s.custom_keys();
    let id = s.make(json!({"name": "Talker", "provider": "custom", "model": "m"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "slow please"})));
    std::thread::sleep(Duration::from_millis(300));
    // Joining mid-run: follow from the start of the turn being written.
    let joined = s.ask("GET", &format!("/v1/bots/{id}/messages"), None).json();
    assert_eq!(joined["running"], true);
    assert!(joined["follow"].as_u64().is_some());
    let r = s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain one more"}))).json();
    assert_eq!(r["queued"], true);
    let chat = s.settle(&id);
    let shape: Vec<(String, String)> = chat.iter().map(|m| (m["role"].as_str().unwrap().to_string(), m["text"].as_str().unwrap().chars().take(6).collect())).collect();
    assert_eq!(shape, [("user", "slow p"), ("assistant", "w0 w1 "), ("user", "plain "), ("assistant", "Plain ")].map(|(a, b)| (a.to_string(), b.to_string())));
    let seen = s.seen.lock().unwrap();
    let last = seen.cloud.last().unwrap()["messages"].as_array().unwrap().clone();
    assert_eq!(last.iter().map(|m| m["role"].as_str().unwrap()).collect::<Vec<_>>(), ["system", "user", "assistant", "user"], "the reply comes before the later message");
}

#[test]
fn routines_run_on_time_and_report_to_the_feed() {
    let s = setup("routines");
    s.custom_keys();
    let id = s.make(json!({"name": "Scout", "job": "News.", "provider": "custom", "model": "test-model"}));
    let bots = s.gw.gateway.bots.clone().unwrap();
    let r = s.ask("POST", &format!("/v1/bots/{id}/routines"), Some(json!({"name": "Digest", "task": "Write a plain digest.", "schedule": {"kind": "every", "hours": 2}})));
    assert_eq!(r.status, 200, "{}", r.body);
    let routine = r.json()["routines"][0].clone();
    assert_eq!((routine["approved"].clone(), routine["by"].clone(), routine["when"].clone()), (json!(true), json!("user"), json!("every 2 hours")));
    let next = routine["next_run"].as_i64().unwrap();
    let list = s.ask("GET", "/v1/bots", None).json();
    assert_eq!((list["bots"][0]["routines"].clone(), list["bots"][0]["pending"].clone()), (json!(1), json!(0)));
    let feed_from = s.ask("GET", "/v1/bots/feed?after=0", None).json()["now"].as_u64().unwrap();

    // Not yet due: nothing starts. Due: the Bot gets the task, framed as a routine.
    bots.tick(next - 1);
    assert!(!bots.running(&id));
    assert!(s.seen.lock().unwrap().cloud.is_empty());
    bots.tick(next);
    let chat = s.settle(&id);
    assert_eq!(chat[0]["routine"]["name"], "Digest");
    assert_eq!(chat[0]["text"], "Write a plain digest.");
    assert_eq!(chat.last().unwrap()["text"], "Plain answer.");
    let asked = last_user(&s.seen.lock().unwrap().cloud[0]);
    assert!(asked.contains("Your routine “Digest” (every 2 hours) started this") && asked.ends_with("Write a plain digest."), "{asked}");
    let after = s.ask("GET", &format!("/v1/bots/{id}/routines"), None).json()["routines"][0].clone();
    assert_eq!((after["last_run"].as_i64(), after["next_run"].as_i64()), (Some(next), Some(next + 7200)));
    // Ticking again at the same time doesn't start it twice.
    bots.tick(next);
    assert!(!bots.running(&id));

    let feed = s.ask("GET", &format!("/v1/bots/feed?after={feed_from}"), None).json()["items"].as_array().unwrap().clone();
    assert_eq!(feed.len(), 1, "{feed:?}");
    assert_eq!((feed[0]["kind"].clone(), feed[0]["text"].clone(), feed[0]["quiet"].clone(), feed[0]["routine"]["name"].clone(), feed[0]["name"].clone()), (json!("routine"), json!("Plain answer."), json!(false), json!("Digest"), json!("Scout")));

    // "Nothing new" stays quiet; Run now works without waiting; an ordinary message is a "reply".
    let r = s.ask("POST", &format!("/v1/bots/{id}/routines"), Some(json!({"name": "Check", "task": "Anything? (quiet)", "schedule": {"kind": "daily", "time": "06:00"}}))).json();
    let rid = r["routines"][1]["id"].as_str().unwrap().to_string();
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/routines/{rid}/run"), None).status, 200);
    s.settle(&id);
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain question"})));
    s.settle(&id);
    let feed = s.ask("GET", &format!("/v1/bots/feed?after={feed_from}"), None).json()["items"].as_array().unwrap().clone();
    let kinds: Vec<(String, bool)> = feed.iter().map(|f| (f["kind"].as_str().unwrap().to_string(), f["quiet"].as_bool().unwrap())).collect();
    assert_eq!(kinds, [("routine".into(), false), ("routine".into(), true), ("reply".into(), false)]);

    // Switched off: never due. Deleted: gone.
    s.ask("PUT", &format!("/v1/bots/{id}/routines/{rid}"), Some(json!({"enabled": false})));
    bots.tick(next + 30 * 86400);
    assert!(!bots.running(&id) || { s.settle(&id); true });
    assert_eq!(s.ask("DELETE", &format!("/v1/bots/{id}/routines/{rid}"), None).json()["routines"].as_array().unwrap().len(), 1);
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/routines"), Some(json!({"task": "x", "schedule": {"kind": "every", "hours": 0.1}}))).status, 400);
}

#[test]
fn a_late_routine_waits_for_its_next_time() {
    let s = setup("late");
    s.custom_keys();
    let id = s.make(json!({"name": "Scout", "provider": "custom", "model": "test-model"}));
    let bots = s.gw.gateway.bots.clone().unwrap();
    let r = s.ask("POST", &format!("/v1/bots/{id}/routines"), Some(json!({"task": "plain", "schedule": {"kind": "every", "hours": 1}}))).json();
    let next = r["routines"][0]["next_run"].as_i64().unwrap();
    // The PC was off for a day: skipped, and the next time is worked out from now.
    let now = next + 86400;
    bots.tick(now);
    assert!(!bots.running(&id));
    assert_eq!(bots.routines(&id)[0].next_run, Some(now + 3600));
    assert_eq!(bots.routines(&id)[0].last_run, None);
    // A few minutes late: it runs, and says it's late.
    bots.tick(now + 3600 + 600);
    s.settle(&id);
    let asked = last_user(&s.seen.lock().unwrap().cloud[0]);
    assert!(asked.contains("running late"), "{asked}");
}

#[test]
fn a_bot_proposes_a_routine_and_waits_for_approval() {
    let s = setup("propose");
    s.custom_keys();
    let id = s.make(json!({"name": "Scout", "provider": "custom", "model": "test-model"}));
    let bots = s.gw.gateway.bots.clone().unwrap();
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "Send me the news every morning"})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "I've proposed it: approve it on your phone.");
    let tool = chat.iter().find(|m| m["role"] == "tool").unwrap();
    assert!(tool["text"].as_str().unwrap().starts_with("Proposed the routine “Morning news” (every day at 08:00)"), "{}", tool["text"]);
    let routines = s.ask("GET", &format!("/v1/bots/{id}/routines"), None).json()["routines"].clone();
    let r = routines[0].clone();
    assert_eq!((r["approved"].clone(), r["by"].clone(), r["when"].clone()), (json!(false), json!("bot"), json!("every day at 08:00")));
    assert_eq!(s.ask("GET", "/v1/bots", None).json()["bots"][0]["pending"], 1);
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    assert_eq!(feed.iter().map(|f| f["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["approval", "reply"]);
    assert!(feed[0]["text"].as_str().unwrap().contains("Morning news"));
    // The next turn's system prompt lists it as waiting.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain thanks"})));
    s.settle(&id);
    let sys = s.seen.lock().unwrap().cloud.last().unwrap()["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(sys.contains("“Morning news” every day at 08:00 (waiting for the user's approval)"), "{sys}");

    // Unapproved: due time passes, nothing runs, and Run now is refused.
    let rid = r["id"].as_str().unwrap();
    let next = r["next_run"].as_i64().unwrap();
    let calls = s.seen.lock().unwrap().cloud.len();
    bots.tick(next);
    assert!(!bots.running(&id));
    assert_eq!(s.seen.lock().unwrap().cloud.len(), calls);
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/routines/{rid}/run"), None).status, 400);
    // Approved: it runs at its next time.
    let r = s.ask("PUT", &format!("/v1/bots/{id}/routines/{rid}"), Some(json!({"approved": true}))).json()["routines"][0].clone();
    assert_eq!(r["approved"], true);
    bots.tick(r["next_run"].as_i64().unwrap());
    let chat = s.settle(&id);
    assert_eq!(chat.iter().filter(|m| m["routine"].is_object()).count(), 1);

    // A Bot can't flood the user with proposals.
    for _ in 0..3 {
        bots.add_routine(&bots.get(&id).unwrap(), &json!({"task": "x", "schedule": {"kind": "every", "hours": 1}}), "bot").unwrap();
    }
    assert!(bots.add_routine(&bots.get(&id).unwrap(), &json!({"task": "x", "schedule": {"kind": "every", "hours": 1}}), "bot").is_err());
}

impl Setup {
    /// Waits for the Bot to ask the user something, and returns {id, text}.
    fn approval(&self, id: &str) -> Value {
        let end = Instant::now() + Duration::from_secs(10);
        loop {
            let r = self.ask("GET", &format!("/v1/bots/{id}/messages"), None).json();
            if r["approval"].is_object() {
                return r["approval"].clone();
            }
            assert!(Instant::now() < end, "the Bot never asked");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[test]
fn the_computer_runs_commands_and_asks_before_sending_a_form() {
    let s = setup_with("computer", true);
    s.custom_keys();
    let id = s.make(json!({"name": "Shopper", "provider": "custom", "model": "test-model"}));

    // The terminal: output and exit code.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "use the terminal"})));
    let chat = s.settle(&id);
    let ran = chat.iter().find(|m| m["tool"] == "run_command").unwrap();
    assert!(ran["text"].as_str().unwrap().starts_with("Exit code 3 ("), "{}", ran["text"]);
    assert!(ran["text"].as_str().unwrap().contains("from-computer"));
    let sent = s.seen.lock().unwrap().cloud[0].clone();
    let names: Vec<&str> = sent["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"run_command") && names.contains(&"browser_open"));
    assert!(sent["messages"][0]["content"].as_str().unwrap().contains("You have a computer"));

    // The browser: the page and its screenshot; the form waits for the user, who allows it.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "shop for socks"})));
    let a = s.approval(&id);
    assert!(a["text"].as_str().unwrap().contains("shop.example (fields: card)"), "{a}");
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    assert!(feed.iter().any(|f| f["kind"] == "approval" && f["text"].as_str().unwrap().starts_with("wants to submit a form")), "{feed:?}");
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/approval"), Some(json!({"id": "nope", "allow": true}))).status, 404);
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/approval"), Some(json!({"id": a["id"], "allow": true}))).status, 200);
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Done with the shop.");
    let opened = chat.iter().find(|m| m["tool"] == "browser_open").unwrap();
    assert!(opened["text"].as_str().unwrap().contains("[1] button \"Buy\""), "{}", opened["text"]);
    assert_eq!(opened["shot"], true);
    let clicked = chat.iter().find(|m| m["tool"] == "browser_click").unwrap();
    assert!(clicked["text"].as_str().unwrap().starts_with("Page: Thanks"), "{}", clicked["text"]);
    let shot = s.ask("GET", &format!("/v1/bots/{id}/shots/{}.jpg", opened["id"].as_str().unwrap()), None);
    assert_eq!(shot.status, 200);
    assert_eq!(s.ask("GET", &format!("/v1/bots/{id}/shots/..%2Fbot.json"), None).status, 404);
    assert!(s.ask("GET", &format!("/v1/bots/{id}/messages"), None).json()["approval"].is_null());

    // Said no: not done, and the model is told.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "shop again"})));
    let a = s.approval(&id);
    s.ask("POST", &format!("/v1/bots/{id}/approval"), Some(json!({"id": a["id"], "allow": false})));
    let chat = s.settle(&id);
    let refused = chat.iter().rev().find(|m| m["tool"] == "browser_click").unwrap();
    assert!(refused["text"].as_str().unwrap().starts_with("NOT DONE. The user said no"), "{}", refused["text"]);
    assert_eq!(
        chat.last().unwrap()["text"],
        "I didn't submit a form that sends data to shop.example (fields: card), because you said no. What would you like me to do instead?",
        "the turn ends with the plain truth, not the model's account"
    );

    // Stop while it waits ends the run at once.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "shop once more"})));
    s.approval(&id);
    let t = Instant::now();
    s.ask("DELETE", &format!("/v1/bots/{id}/run"), None);
    s.settle(&id);
    assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());

    // Switched off for this Bot: no computer tools.
    s.ask("PUT", &format!("/v1/bots/{id}"), Some(json!({"computer": false})));
    let before = s.seen.lock().unwrap().cloud.len();
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain hello"})));
    s.settle(&id);
    let sent = s.seen.lock().unwrap().cloud[before].clone();
    assert!(!sent["tools"].as_array().unwrap().iter().any(|t| t["function"]["name"] == "run_command"));
}

#[test]
fn a_claimed_but_unsaved_file_gets_one_nudge() {
    let s = setup("nudge");
    s.custom_keys();
    let id = s.make(json!({"name": "Lazy", "provider": "custom", "model": "m"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "be lazy"})));
    let chat = s.settle(&id);
    assert_eq!(chat.iter().map(|m| m["role"].as_str().unwrap()).collect::<Vec<_>>(), ["user", "assistant", "assistant", "tool", "assistant"]);
    assert_eq!(std::fs::read_to_string(s.dir.join("workspace/notes/lazy.md")).unwrap(), "lazy");
    assert_eq!(chat[4]["text"], "Saved notes/lazy.md.");
    let seen = s.seen.lock().unwrap();
    let asked = seen.cloud[1]["messages"].as_array().unwrap().last().unwrap().clone();
    assert!(asked["content"].as_str().unwrap().starts_with("(Note from the system"), "{asked}");
    assert!(!s.ask("GET", &format!("/v1/bots/{id}/messages"), None).body.contains("Note from the system"), "the note isn't stored");
}

#[test]
fn claims_are_checked_against_the_workspace() {
    let root = std::env::temp_dir().join(format!("cubic-gw-claims-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("notes")).unwrap();
    std::fs::write(root.join("notes/real.md"), "x").unwrap();
    use cubic_gateway::bots::tools::claimed_unsaved;
    assert_eq!(claimed_unsaved("I have saved the report to `research/jwst.md` in your workspace.", &root).as_deref(), Some("research/jwst.md"));
    assert_eq!(claimed_unsaved("[I have written the report to research/jwst.md.]", &root).as_deref(), Some("research/jwst.md"));
    assert_eq!(claimed_unsaved("Saved notes/real.md.", &root), None, "it's there");
    assert_eq!(claimed_unsaved("The page at example.com/index.html says hi.", &root), None, "no saving said");
    assert_eq!(claimed_unsaved("I saved it.\nSee also ../etc/passwd.md", &root), None, "outside the workspace isn't ours");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn paths_stay_inside_the_workspace() {
    let root = std::env::temp_dir().join(format!("cubic-gw-ws-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("ws/notes")).unwrap();
    std::fs::create_dir_all(root.join("secret")).unwrap();
    let ws = root.join("ws");
    let real = ws.canonicalize().unwrap();
    assert_eq!(safe_path(&ws, "notes/a.md").unwrap(), real.join("notes/a.md"));
    assert_eq!(safe_path(&ws, "/workspace/notes/a.md").unwrap(), real.join("notes/a.md"), "the /workspace name works");
    assert_eq!(safe_path(&ws, "new/deeper/b.md").unwrap(), real.join("new/deeper/b.md"));
    for bad in ["../secret/x", "notes/../../secret", "/etc/passwd", "/workspace/../secret"] {
        assert!(safe_path(&ws, bad).is_err(), "{bad}");
    }
    // A link out of the workspace is refused, for the link itself and anything under it.
    std::os::unix::fs::symlink(root.join("secret"), ws.join("out")).unwrap();
    assert!(safe_path(&ws, "out").is_err());
    assert!(safe_path(&ws, "out/new.txt").is_err());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_conversation_starts_at_the_user_and_cuts_old_results() {
    let long = "x".repeat(bots::OLD_RESULT_CHARS + 100);
    let stored = vec![
        json!({"role": "tool", "call": "c0", "tool": "web_search", "text": "orphan"}),
        json!({"role": "user", "text": "first"}),
        json!({"role": "assistant", "text": "", "calls": [{"id": "c1", "name": "read_page", "args": "{}"}]}),
        json!({"role": "tool", "call": "c1", "tool": "read_page", "text": long}),
        json!({"role": "assistant", "text": "done"}),
        json!({"role": "assistant", "text": "", "error": "boom"}),
        json!({"role": "user", "text": "second"}),
        json!({"role": "assistant", "text": "", "calls": [{"id": "c2", "name": "read_page", "args": "{}"}]}),
        json!({"role": "tool", "call": "c2", "tool": "read_page", "text": long}),
    ];
    let c = Bots::conversation(&stored);
    assert_eq!(c.iter().map(|m| m["role"].as_str().unwrap()).collect::<Vec<_>>(), ["user", "assistant", "tool", "assistant", "user", "assistant", "tool"]);
    assert!(c[2]["content"].as_str().unwrap().ends_with("[…cut]"), "a result from an earlier message is cut");
    assert_eq!(c[6]["content"].as_str().unwrap().len(), long.len(), "this message's results are whole");
    assert_eq!(c[1]["tool_calls"][0], json!({"id": "c1", "type": "function", "function": {"name": "read_page", "arguments": "{}"}}));
}

/// A real local Bot: the PC's Ollama and SearXNG, the real identity, a spare port, its own temp
/// Bots folder. `cargo test -p cubic-gateway --release net_real_bot -- --ignored --nocapture`
/// (CHAT_MODEL picks the model, default gemma4:e4b).
#[test]
#[ignore]
fn net_real_bot() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-realbot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let paths = bots::Paths { bots: dir.join("bots"), workspace: dir.join("workspace"), computer: None, google: None };
    let config = Config { media: None, bots: Some(paths), ..Config::from_env() };
    let id = identity::load(&config.dir).expect("this PC's gateway identity");
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let model = std::env::var("CHAT_MODEL").unwrap_or_else(|_| "gemma4:e4b".into());
    let go = |method: &str, path: &str, body: Option<Value>| {
        let b = body.map(|v| v.to_string());
        ask(gw.port, &id.fingerprint, method, path, Some(&id.token), b.as_deref())
    };
    let r = go("POST", "/v1/bots", Some(json!({"name": "Scout", "job": "Look things up for the user and keep short notes in the workspace.", "provider": "local", "model": model})));
    let bot = r.json()["bot"]["id"].as_str().expect("a Bot").to_string();
    let t = Instant::now();
    go("POST", &format!("/v1/bots/{bot}/messages"), Some(json!({"text": "Search the web for the latest stable Rust version, then save a one-line note about it to notes/rust.md."})));
    let ev: Vec<Value> = go("GET", &format!("/v1/bots/{bot}/events?after=0"), None).events().iter().filter_map(|e| serde_json::from_str(e).ok()).collect();
    for e in &ev {
        match e["type"].as_str().unwrap() {
            "tool" => println!("  tool {} {}", e["name"], e["args"]),
            "message" if e["message"]["role"] == "assistant" => println!("  reply ({} ms): {} {}", e["message"]["took"], e["message"]["text"], e["message"]["error"]),
            "done" => println!("  done {} after {:.1} s", e["error"], t.elapsed().as_secs_f64()),
            _ => {}
        }
    }
    let note = std::fs::read_to_string(dir.join("workspace/notes/rust.md"));
    println!("  notes/rust.md: {note:?}");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(ev.iter().any(|e| e["type"] == "tool" && e["name"] == "web_search"), "it searched");
    assert!(note.is_ok_and(|n| n.contains("1.")), "it saved the note");
    assert_eq!(ev.last().unwrap()["error"], Value::Null);
}

/// The real computer (needs Docker and `cubicai-gateway computer-setup`): terminal, files, the
/// network rules, the browser, Stop. Uses its own container name and folders.
#[test]
#[ignore]
fn net_real_computer() {
    use cubic_gateway::bots::computer::{Computer, Setup};
    use std::sync::atomic::AtomicBool;
    let dir = std::env::temp_dir().join(format!("cubic-computer-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("ws")).unwrap();
    let setup = Setup { dir: dir.join("c"), docker: "docker".into(), name: "cubicai-computer-test".into(), image: bots::computer::IMAGE.into() };
    let c = Computer::new(&setup, &dir.join("ws"));
    c.clean_start();
    let no = AtomicBool::new(false);
    let run = |cmd: &str| {
        let r = c.run(cmd, 60, &no).unwrap();
        eprintln!("$ {cmd}\n{}\n(exit {:?}, {:.1} s)", r.output, r.code, r.secs);
        r
    };
    let started = Instant::now();
    let r = run("id -u; pwd; python3 --version; echo hello > note.txt");
    eprintln!("first command (with start) took {:.1} s", started.elapsed().as_secs_f64());
    let uid = unsafe { libc_getuid() };
    assert!(r.output.starts_with(&format!("{uid}\n/workspace\nPython 3.")), "{}", r.output);
    let meta = std::fs::metadata(dir.join("ws/note.txt")).unwrap();
    assert_eq!(std::os::unix::fs::MetadataExt::uid(&meta), uid, "files are the user's");
    assert_ne!(run("touch /etc/x").code, Some(0), "the system is read-only");
    assert_ne!(run("cat /proc/1/environ >/dev/null && ls /root").code, Some(0));
    // Out only through the proxy, only to public websites.
    assert_eq!(run("curl -sS -m 20 -o /dev/null -w '%{http_code}' https://example.com").output.trim(), "200");
    for blocked in ["http://192.168.0.1/", "https://127.0.0.1:8443/", "http://172.17.0.1/", "https://100.100.100.100/", "http://169.254.169.254/"] {
        // Refused by the proxy, or (127.0.0.1 skips the proxy) by the container's own empty loopback.
        let r = run(&format!("curl -sS -m 10 {blocked}"));
        assert!(r.output.contains("public websites only") || r.output.contains("response 403") || r.output.contains("Could not connect"), "{blocked}: {}", r.output);
    }
    assert_ne!(run("curl -sS -m 5 --noproxy '*' https://1.1.1.1/").code, Some(0), "no network without the proxy");
    assert_eq!(run("pip install --user -q cowsay >/dev/null 2>&1 && python3 -m cowsay -t moo | grep -c moo").output.trim(), "1", "pip goes through the proxy");
    // The browser.
    let page = c.browse(&json!({"do": "open", "url": "https://example.com"})).unwrap();
    assert_eq!(page["title"], "Example Domain", "{page}");
    assert!(page["elements"].as_array().unwrap().iter().any(|e| e[1] == "link"), "{page}");
    let shot = c.shot(page["shot"].as_str().unwrap()).unwrap();
    assert!(std::fs::read(&shot).unwrap().starts_with(&[0xff, 0xd8]), "a JPEG screenshot");
    let refused = c.browse(&json!({"do": "open", "url": "http://192.168.0.1/"})).unwrap();
    // The proxy's refusal is the page, so the model reads why.
    assert!(refused["text"].as_str().unwrap_or("").contains("public websites only") || refused["ok"] == false, "{refused}");
    // Stop ends a command and everything it started, inside the container too.
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        s2.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    let t = Instant::now();
    let r = c.run("sleep 300 & sleep 300; echo never", 600, &stop).unwrap();
    assert!(r.stopped && t.elapsed() < Duration::from_secs(5), "stopped in {:?}", t.elapsed());
    assert_eq!(run("pgrep -c sleep || true").output.trim(), "0", "nothing left running");
    let r = c.run("sleep 30", 2, &no).unwrap();
    assert!(r.timed_out);
    let _ = std::process::Command::new("docker").args(["rm", "-f", "cubicai-computer-test"]).output();
    let _ = std::fs::remove_dir_all(&dir);
}

unsafe extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

#[test]
fn google_signs_in_reads_mail_and_asks_before_sending() {
    let s = setup("google");
    s.custom_keys();
    // Not connected: Bots get no Google tools, and the sign-in needs the client first.
    assert_eq!(s.ask("GET", "/v1/google", None).json(), json!({"connected": false, "client": false}));
    assert_eq!(s.ask("POST", "/v1/google/start", Some(json!({"redirect": "http://127.0.0.1:5555"}))).status, 400);
    assert_eq!(s.ask("PUT", "/v1/google", Some(json!({"web": {"client_id": "x", "client_secret": "y"}}))).status, 400, "a web client is refused");
    assert_eq!(s.ask("PUT", "/v1/google", Some(json!({"client_id": "cid.apps.googleusercontent.com", "client_secret": "csecret"}))).status, 200);
    assert_eq!(s.ask("POST", "/v1/google/start", Some(json!({"redirect": "https://evil.example/cb"}))).status, 400, "only loopback redirects");

    // Sign in: the phone opens the URL, Google sends the code back to its loopback, the phone passes it on.
    let r = s.ask("POST", "/v1/google/start", Some(json!({"redirect": "http://127.0.0.1:5555"}))).json();
    let url = r["url"].as_str().unwrap().to_string();
    let q = bots::google::query(&url);
    assert!(url.starts_with(&format!("{}/g/auth?", s.url)), "{url}");
    assert_eq!(q["code_challenge_method"], "S256");
    assert_eq!(q["access_type"], "offline");
    assert!(q["scope"].contains("gmail.modify") && q["scope"].contains("auth/drive"));
    assert_eq!(s.ask("POST", "/v1/google/finish", Some(json!({"code": "c1", "state": "wrong"}))).status, 400);
    let done = s.ask("POST", "/v1/google/finish", Some(json!({"code": "c1", "state": q["state"]})));
    assert_eq!(done.status, 200, "{}", done.body);
    assert_eq!(done.json()["email"], "me@example.com");
    assert!(done.json()["note"].is_null(), "every permission was given");
    let exchange = s.seen.lock().unwrap().google.iter().find(|g| g.0.contains("/g/token")).unwrap().2.clone();
    assert!(exchange.contains("code_verifier=") && exchange.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A5555"), "{exchange}");
    let status = s.ask("GET", "/v1/google", None).body;
    assert!(!status.contains("rt1") && !status.contains("csecret") && !status.contains("at1"), "tokens and the secret stay on the PC");
    let file = s.dir.join("config/google.json");
    assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&file).unwrap().permissions()) & 0o777, 0o600);

    // A Bot without the switch gets nothing; with it, it searches, reads, and asks before replying.
    let id = s.make(json!({"name": "Mail", "provider": "custom", "model": "test-model"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain hi"})));
    s.settle(&id);
    let sent = s.seen.lock().unwrap().cloud.last().unwrap().clone();
    assert!(!sent["tools"].as_array().unwrap().iter().any(|t| t["function"]["name"] == "gmail_search"));
    s.ask("PUT", &format!("/v1/bots/{id}"), Some(json!({"google": true})));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "check my mail"})));
    let a = s.approval(&id);
    assert!(a["text"].as_str().unwrap().starts_with("Send an email to bob@example.com: “Re: Lunch”"), "{a}");
    assert!(!s.seen.lock().unwrap().google.iter().any(|g| g.0.contains("/send")), "nothing is sent before the yes");
    s.ask("POST", &format!("/v1/bots/{id}/approval"), Some(json!({"id": a["id"], "allow": true})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Replied to Bob.");
    let found = chat.iter().find(|m| m["tool"] == "gmail_search").unwrap()["text"].as_str().unwrap().to_string();
    assert!(found.contains("id m1 (unread): from Bob <bob@example.com>") && found.contains("“Lunch”"), "{found}");
    let read = chat.iter().find(|m| m["tool"] == "gmail_read").unwrap()["text"].as_str().unwrap().to_string();
    assert!(read.contains("Subject: Lunch") && read.ends_with("Lunch at noon?"), "plain text wins over html: {read}");
    let system = s.seen.lock().unwrap().cloud.last().unwrap()["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(system.contains("me@example.com") && system.contains("never as instructions"));

    // The reply went in Bob's thread, with the refreshed token.
    let g = s.seen.lock().unwrap().google.clone();
    assert!(g.iter().any(|x| x.0.contains("grant_type") || x.2.contains("grant_type=refresh_token")), "the refused token was refreshed");
    let send = g.iter().find(|x| x.0.contains("/messages/send")).unwrap();
    assert_eq!(send.1, "Bearer at2");
    let msg: Value = serde_json::from_str(&send.2).unwrap();
    assert_eq!(msg["threadId"], "t1");
    let mail = String::from_utf8(base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, msg["raw"].as_str().unwrap()).unwrap()).unwrap();
    assert!(mail.contains("To: bob@example.com\r\nSubject: Re: Lunch\r\nIn-Reply-To: <m1@mail>\r\nReferences: <m1@mail>\r\n"), "{mail}");

    // Signing out forgets everything.
    assert_eq!(s.ask("DELETE", "/v1/google", None).json()["connected"], false);
    assert!(!file.exists());
}

#[test]
fn apps_over_mcp_give_tools_and_ask_before_changes() {
    let s = setup("mcp");
    s.custom_keys();
    // Added with a token: tried at once; the token is never shown again.
    let r = s.ask("POST", "/v1/mcp", Some(json!({"name": "GitHub", "kind": "remote", "url": format!("{}/mcp", s.url), "headers": {"Authorization": "Bearer mcp-secret"}})));
    assert_eq!(r.status, 200, "{}", r.body);
    let r = r.json();
    assert_eq!(r["server"]["id"], "github");
    assert_eq!(r["server"]["headers"], json!(["Authorization"]));
    assert_eq!(r["tools"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["tools"][0]["read_only"], true);
    assert!(!s.ask("GET", "/v1/mcp", None).body.contains("mcp-secret"));
    let file = s.dir.join("config/mcp.json");
    assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&file).unwrap().permissions()) & 0o777, 0o600);
    assert_eq!(s.ask("POST", "/v1/mcp", Some(json!({"name": "Bad", "kind": "remote", "url": "ftp://x"}))).status, 400);

    // A wrong token is said plainly.
    let bad = s.ask("POST", "/v1/mcp", Some(json!({"name": "Nope", "kind": "remote", "url": format!("{}/mcp", s.url), "headers": {"Authorization": "Bearer wrong"}}))).json();
    assert!(bad["error"].as_str().unwrap().contains("refused CubicAI's sign-in"), "{bad}");
    s.ask("DELETE", "/v1/mcp/nope", None);

    // A Bot given the app: its tools are offered (schema cleaned), the read-only one runs, the other asks.
    let id = s.make(json!({"name": "Dev", "provider": "custom", "model": "test-model"}));
    s.ask("PUT", &format!("/v1/bots/{id}"), Some(json!({"mcp": ["github"]})));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "file the bug"})));
    let a = s.approval(&id);
    assert!(a["text"].as_str().unwrap().starts_with("Use GitHub: create_issue {\"title\":\"Login broken again\"}"), "{a}");
    s.ask("POST", &format!("/v1/bots/{id}/approval"), Some(json!({"id": a["id"], "allow": true})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Filed it.");
    assert_eq!(chat.iter().find(|m| m["tool"] == "github__get_issue").unwrap()["text"], "Issue 7: Login is broken");
    assert_eq!(chat.iter().find(|m| m["tool"] == "github__create_issue").unwrap()["text"], "Created issue 8: Login broken again");
    let sent = s.seen.lock().unwrap().cloud.iter().rev().find(|b| b["messages"][0]["content"].as_str().unwrap_or("").contains("apps you can use")).unwrap().clone();
    let get = sent["tools"].as_array().unwrap().iter().find(|t| t["function"]["name"] == "github__get_issue").unwrap().clone();
    assert_eq!(get["function"]["description"], "[GitHub] Read an issue.");
    assert_eq!(get["function"]["parameters"], json!({"type": "object", "properties": {"number": {"type": "integer"}}}));
    // One session, reused, and every call carried the token.
    let calls = s.seen.lock().unwrap().mcp.clone();
    assert!(calls.iter().filter(|c| c.0 == "tools/call").all(|c| c.1 == "sess-1" && c.2 == "Bearer mcp-secret"), "{calls:?}");

    // Set to ask never: no question.
    s.ask("PUT", "/v1/mcp/github", Some(json!({"ask": "never"})));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "file the bug"})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Filed it.");

    // Unreachable: the Bot is told, and still works.
    s.ask("PUT", "/v1/mcp/github", Some(json!({"url": "http://127.0.0.1:9/mcp"})));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain hi"})));
    s.settle(&id);
    let sys = s.seen.lock().unwrap().cloud.last().unwrap()["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(sys.contains("can't be reached right now: GitHub"), "{sys}");
}

#[test]
fn a_local_app_runs_inside_the_computer() {
    let s = setup_with("mcp-local", true);
    let script = s.dir.join("echo_mcp.py");
    std::fs::write(&script, r#"
import sys, json
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m or "method" not in m:
        continue
    if m["method"] == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": "p1", "method": "ping"}), flush=True)
        r = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "echo"}}
    elif m["method"] == "tools/list":
        r = {"tools": [{"name": "echo", "description": "Says it back.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}, "annotations": {"readOnlyHint": True}}]}
    else:
        r = {}
    print("not json, a log line", flush=True)
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#).unwrap();
    let r = s.ask("POST", "/v1/mcp", Some(json!({"name": "Echo", "kind": "sandbox", "command": format!("python3 {}", script.display()), "env": {"TOKEN": "t"}})));
    assert_eq!(r.status, 200, "{}", r.body);
    let r = r.json();
    assert_eq!(r["server"]["env"], json!(["TOKEN"]));
    assert_eq!(r["tools"][0]["name"], "echo", "{r}");
    // Asked again: the same process answers.
    assert_eq!(s.ask("GET", "/v1/mcp/echo/tools", None).json()["tools"][0]["description"], "Says it back.");
    assert_eq!(s.ask("DELETE", "/v1/mcp/echo", None).status, 200);
    assert_eq!(s.ask("GET", "/v1/mcp/echo/tools", None).status, 404);
}

#[test]
fn the_phone_gets_reminders_and_answers_questions() {
    let s = setup("phone");
    s.custom_keys();
    let id = s.make(json!({"name": "Helper", "provider": "custom", "model": "test-model"}));

    // A reminder goes to the feed (the phone reads it even with the app closed), at the local time asked.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "remind me to call mum tomorrow at 18:30"})));
    let chat = s.settle(&id);
    let step = chat.iter().find(|m| m["tool"] == "phone_remind").unwrap();
    assert!(step["text"].as_str().unwrap().starts_with("Sent to the phone"), "{}", step["text"]);
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    let item = feed.iter().find(|f| f["kind"] == "phone").expect("a phone item");
    assert_eq!(item["phone"]["do"], "remind");
    assert_eq!(item["phone"]["text"], "Call mum");
    assert!(feed.iter().filter(|f| f["kind"] == "reply").all(|f| f["quiet"] == true), "the reply after a reminder notifies nothing more");
    let at = item["phone"]["at"].as_i64().unwrap() / 1000;
    let t = cubic_gateway::time::local(at);
    assert_eq!((t.hour, t.minute), (18, 30));
    assert!((at - cubic_gateway::time::now() as i64 - 86400).abs() < 86400);

    // A question waits for the phone's answer (it's in /messages for a phone that opens the chat).
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "where am i?"})));
    let end = Instant::now() + Duration::from_secs(10);
    let q = loop {
        let r = s.ask("GET", &format!("/v1/bots/{id}/messages"), None).json();
        if r["phone"].is_object() {
            break r["phone"].clone();
        }
        assert!(Instant::now() < end, "never asked the phone");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(q["do"], "location");
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/phone"), Some(json!({"id": "nope", "answer": {}}))).status, 404);
    let r = s.ask("POST", &format!("/v1/bots/{id}/phone"), Some(json!({"id": q["id"], "answer": {"lat": 12.97194, "lon": 77.59369, "accuracy": 25.0, "place": "MG Road, Bengaluru"}})));
    assert_eq!(r.status, 200);
    let chat = s.settle(&id);
    let got = chat.iter().find(|m| m["tool"] == "phone_location").unwrap()["text"].as_str().unwrap().to_string();
    assert!(got.starts_with("The phone is at 12.97194, 77.59369 (within about 25 m). That's MG Road, Bengaluru. Map: https://www.openstreetmap.org/"), "{got}");

    // Switched off: no phone tools.
    s.ask("PUT", &format!("/v1/bots/{id}"), Some(json!({"phone": false})));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "plain hi"})));
    s.settle(&id);
    let sent = s.seen.lock().unwrap().cloud.last().unwrap().clone();
    assert!(!sent["tools"].as_array().unwrap().iter().any(|t| t["function"]["name"] == "phone_remind"));
}

#[test]
fn skills_are_called_proposed_and_approved() {
    let s = setup("skills");
    s.custom_keys();
    let id = s.make(json!({"name": "Scout", "provider": "custom", "model": "test-model"}));
    let r = s.ask("POST", "/v1/skills", Some(json!({"name": "Check bill", "about": "Read this month's power bill", "steps": "1. Open the power company's site\n2. Read the amount"})));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["skills"][0]["name"], "check-bill");
    // Calling it: the steps go with that message, and every Bot's prompt lists the skill.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "/check-bill please"})));
    s.settle(&id);
    let sent = s.seen.lock().unwrap().cloud.last().unwrap().clone();
    let user = last_user(&sent);
    assert!(user.starts_with("/check-bill please\n\n[The user called the skill /check-bill. Follow its steps:]\n1. Open the power company"), "{user}");
    assert!(sent["messages"][0]["content"].as_str().unwrap().contains("- /check-bill: Read this month's power bill"));
    // A Bot's proposal waits for the user.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "save how you did that"})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Proposed /find-cats.");
    let skills = s.ask("GET", "/v1/skills", None).json()["skills"].as_array().unwrap().clone();
    let p = skills.iter().find(|k| k["name"] == "find-cats").unwrap().clone();
    assert_eq!((p["approved"].clone(), p["by"].clone(), p["by_name"].clone()), (json!(false), json!(id), json!("Scout")));
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    assert!(feed.iter().any(|f| f["kind"] == "approval" && f["text"].as_str().unwrap().contains("/find-cats")));
    let sys_before = s.seen.lock().unwrap().cloud.last().unwrap()["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(!sys_before.contains("/find-cats"), "not used before it's approved");
    let r = s.ask("PUT", &format!("/v1/skills/{}", p["id"].as_str().unwrap()), Some(json!({"approved": true}))).json();
    assert!(r["skills"].as_array().unwrap().iter().all(|k| k["approved"] == true));
    // Editing and deleting.
    let r = s.ask("PUT", &format!("/v1/skills/{}", p["id"].as_str().unwrap()), Some(json!({"steps": ""})));
    assert_eq!(r.status, 400, "steps can't be emptied");
    s.ask("DELETE", &format!("/v1/skills/{}", p["id"].as_str().unwrap()), None);
    assert_eq!(s.ask("GET", "/v1/skills", None).json()["skills"].as_array().unwrap().len(), 1);
}

#[test]
fn a_bot_makes_a_picture_into_the_workspace() {
    // Without ComfyUI there's no picture tool.
    let s = setup("nopics");
    s.custom_keys();
    let id = s.make(json!({"name": "Artist", "provider": "custom", "model": "test-model"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "hello"})));
    s.settle(&id);
    let tools = s.seen.lock().unwrap().cloud[0]["tools"].clone();
    assert!(!tools.as_array().unwrap().iter().any(|t| t["function"]["name"] == "make_picture"));
    drop(s);

    let s = setup_full("pics", false, true);
    s.custom_keys();
    let id = s.make(json!({"name": "Artist", "provider": "custom", "model": "test-model"}));
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "draw me a tea shop"})));
    let chat = s.settle(&id);
    assert_eq!(chat.last().unwrap()["text"], "Here it is.");
    let tool = chat.iter().find(|m| m["tool"] == "make_picture").unwrap();
    assert_eq!(tool["text"], "Made the picture and saved it as pictures/a-tea-shop-by-the.png in the workspace. The user sees it in the chat; don't describe it at length.");
    assert_eq!(tool["shot"], true);
    assert_eq!(std::fs::read(s.dir.join("workspace/pictures/a-tea-shop-by-the.png")).unwrap(), common::comfy::PNG);
    // The phone gets it beside the step, as a PNG.
    let r = s.ask("GET", &format!("/v1/bots/{id}/shots/{}.jpg", tool["id"].as_str().unwrap()), None);
    assert_eq!((r.status, r.header("Content-Type")), (200, Some("image/png")));
    // The tool is offered (no edit model here, so no edit argument).
    let def = s.seen.lock().unwrap().cloud[0]["tools"].as_array().unwrap().iter().find(|t| t["function"]["name"] == "make_picture").cloned().unwrap();
    assert!(def["function"]["parameters"]["properties"].get("edit").is_none());
    // Again: a new name, never over the first.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "draw it again"})));
    s.settle(&id);
    assert!(s.dir.join("workspace/pictures/a-tea-shop-by-the-2.png").exists());
    // ComfyUI's error comes back; a path outside the workspace is refused.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "draw something broken"})));
    let chat = s.settle(&id);
    let failed = chat.iter().rev().find(|m| m["tool"] == "make_picture").unwrap();
    assert!(failed["text"].as_str().unwrap().starts_with("The picture failed:") && failed["failed"] == true, "{failed}");
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "draw it outside"})));
    let chat = s.settle(&id);
    let refused = chat.iter().rev().find(|m| m["tool"] == "make_picture").unwrap();
    assert_eq!(refused["failed"], true, "{refused}");
    assert!(!s.dir.join("escape.png").exists());
}

#[test]
fn a_bot_hands_a_task_to_another_and_gets_the_answer() {
    let s = setup("handoff");
    s.custom_keys();
    let scout = s.make(json!({"name": "Scout", "job": "Find things.", "provider": "custom", "model": "test-model"}));
    let writer = s.make(json!({"name": "Writer", "job": "Writes poems and posts.", "provider": "custom", "model": "test-model"}));
    s.ask("POST", &format!("/v1/bots/{scout}/messages"), Some(json!({"text": "ask the writer for a haiku"})));
    let chat = s.settle(&scout);
    assert_eq!(chat.last().unwrap()["text"], "From Writer: Soft rain on the roof.");
    let tool = chat.iter().find(|m| m["tool"] == "ask_bot").unwrap();
    assert_eq!(tool["text"], "Writer answered:\nSoft rain on the roof.");
    // The task and the answer are in Writer's own chat; its prompt names the other Bots.
    let theirs = s.settle(&writer);
    assert_eq!(theirs[0]["from"]["name"], "Scout");
    assert_eq!(theirs[0]["text"], "Write a haiku about rain.");
    assert_eq!(theirs[1]["text"], "Soft rain on the roof.");
    let seen = s.seen.lock().unwrap().cloud.clone();
    let scout_sys = seen[0]["messages"][0]["content"].as_str().unwrap();
    assert!(scout_sys.contains("The user's other Bots:\n- Writer: Writes poems and posts."), "{scout_sys}");
    assert!(seen[0]["tools"].as_array().unwrap().iter().any(|t| t["function"]["name"] == "ask_bot"));
    // Writer's answer went back to Scout, so only Scout's reply notifies.
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    let loud: Vec<&str> = feed.iter().filter(|f| f["quiet"] == false).map(|f| f["name"].as_str().unwrap()).collect();
    assert_eq!(loud, ["Scout"]);
    // Asking the same Bot again in one turn gets the first answer back, without a second task.
    s.ask("POST", &format!("/v1/bots/{scout}/messages"), Some(json!({"text": "ask the writer twice for a haiku"})));
    let chat = s.settle(&scout);
    let tools: Vec<&str> = chat.iter().filter(|m| m["tool"] == "ask_bot").map(|m| m["text"].as_str().unwrap()).collect();
    assert_eq!(tools.len(), 3, "{tools:?}");
    assert!(tools[2].starts_with("You asked Writer already in this turn. Its answer:\nSoft rain on the roof."), "{}", tools[2]);
    assert_eq!(s.settle(&writer).iter().filter(|m| m["role"] == "user").count(), 2, "one task per turn");
}

#[test]
fn groups_answer_by_mention_hand_on_and_route() {
    let s = setup("groups");
    s.custom_keys();
    let res = s.make(json!({"name": "Researcher", "job": "Finds sources.", "provider": "custom", "model": "test-model"}));
    let wri = s.make(json!({"name": "Writer", "job": "Writes things up.", "provider": "custom", "model": "test-model"}));
    assert_eq!(s.ask("POST", "/v1/groups", Some(json!({"name": "Desk", "members": [res]}))).status, 400, "two Bots at least");
    let g = s.ask("POST", "/v1/groups", Some(json!({"name": "Desk", "members": [res, wri]}))).json()["group"].clone();
    let gid = g["id"].as_str().unwrap().to_string();
    let settle = || {
        let end = Instant::now() + Duration::from_secs(15);
        loop {
            let r = s.ask("GET", &format!("/v1/groups/{gid}/messages"), None).json();
            if r["running"] == false {
                return r["messages"].as_array().unwrap().clone();
            }
            assert!(Instant::now() < end, "the group never finished");
            std::thread::sleep(Duration::from_millis(50));
        }
    };
    // @Researcher answers and brings Writer in, which reads Researcher's words as Researcher's.
    s.ask("POST", &format!("/v1/groups/{gid}/messages"), Some(json!({"text": "@Researcher find me three sources on rain"})));
    let chat = settle();
    let said: Vec<(String, String)> = chat.iter().map(|m| (m["bot"].as_str().unwrap_or("user").to_string(), m["text"].as_str().unwrap_or("").to_string())).collect();
    // Writer's thanks don't bring Researcher back: each Bot speaks once per message of the user's.
    assert_eq!(said, [("user".into(), "@Researcher find me three sources on rain".into()), (res.clone(), "Found three sources. @Writer please write it up.".into()), (wri.clone(), "Written up. Thanks @Researcher!".into())]);
    let writer_turn = s.seen.lock().unwrap().cloud.last().unwrap().clone();
    let sys = writer_turn["messages"][0]["content"].as_str().unwrap();
    assert!(sys.starts_with("You are Writer,") && sys.contains("You're in the group chat “Desk”") && sys.contains("- Researcher: Finds sources."), "{sys}");
    // No mention: the router picks.
    let calls = s.seen.lock().unwrap().cloud.len();
    s.ask("POST", &format!("/v1/groups/{gid}/messages"), Some(json!({"text": "who's free to help?"})));
    let chat = settle();
    assert_eq!(chat.last().unwrap()["text"], "Writer here.");
    assert_eq!(chat.last().unwrap()["bot"], wri.as_str());
    let seen = s.seen.lock().unwrap().cloud[calls..].to_vec();
    assert!(seen[0]["messages"][0]["content"].as_str().unwrap().contains("You pick who answers in the group chat “Desk”"));
    assert!(seen[0].get("tools").is_none(), "the router gets no tools");
    // The list and the feed know the group.
    let list = s.ask("GET", "/v1/groups", None).json();
    assert_eq!((list["groups"][0]["name"].clone(), list["groups"][0]["last"]["bot"].clone()), (json!("Desk"), json!(wri)));
    let feed = s.ask("GET", "/v1/bots/feed?after=0", None).json()["items"].as_array().unwrap().clone();
    assert!(feed.iter().all(|f| f["group"]["id"] == gid.as_str()), "{feed:?}");
    // Deleting a Bot drops it from the group; deleting the group drops its chat.
    s.ask("DELETE", &format!("/v1/bots/{res}"), None);
    assert_eq!(s.ask("GET", &format!("/v1/groups/{gid}"), None).json()["group"]["members"], json!([wri]));
    assert_eq!(s.ask("DELETE", &format!("/v1/groups/{gid}"), None).status, 200);
    assert_eq!(s.ask("GET", &format!("/v1/groups/{gid}/messages"), None).status, 404);
}

#[test]
fn the_user_takes_over_the_browser_and_hands_it_back() {
    let s = setup_with("takeover", true);
    s.custom_keys();
    let id = s.make(json!({"name": "Shopper", "provider": "custom", "model": "test-model"}));
    let st = s.ask("POST", &format!("/v1/bots/{id}/teach"), Some(json!({"do": "start", "takeover": true}))).json();
    assert_eq!(st["takeover"], true);
    assert_eq!(s.ask("GET", &format!("/v1/bots/{id}/screen"), None).status, 200);
    // The Bot's browser step waits while the user has it.
    s.ask("POST", &format!("/v1/bots/{id}/messages"), Some(json!({"text": "buy it at the shop"})));
    std::thread::sleep(Duration::from_millis(600));
    let r = s.ask("GET", &format!("/v1/bots/{id}/messages"), None).json();
    assert_eq!(r["running"], true);
    assert!(!r["messages"].as_array().unwrap().iter().any(|m| m["tool"] == "browser_open"), "the step waits");
    s.ask("POST", &format!("/v1/bots/{id}/teach"), Some(json!({"do": "tap", "x": 0.5, "y": 0.1})));
    let r = s.ask("POST", &format!("/v1/bots/{id}/teach/done"), Some(json!({"what": "I logged in for you"})));
    assert_eq!(r.status, 200, "{}", r.body);
    let chat = s.settle(&id);
    let waited = chat.iter().find(|m| m["tool"] == "browser_open").unwrap();
    assert!(waited["text"].as_str().unwrap().starts_with("The user took over your browser meanwhile"), "{waited}");
    let back = chat.iter().find(|m| m["took_over"] == true).unwrap();
    assert_eq!((back["text"].clone(), back["taught"][0].clone()), (json!("I logged in for you"), json!("Clicked the button “Search”")));
    assert_eq!(chat.last().unwrap()["text"], "Thanks, carrying on from the checkout.");
    // The model got the result straight after its call, then the user's hand-back.
    let sent = s.seen.lock().unwrap().cloud.last().unwrap()["messages"].clone();
    let roles: Vec<&str> = sent.as_array().unwrap().iter().skip(1).map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "tool", "user"]);
    assert!(s.gw.gateway.bots.as_ref().unwrap().teaching().is_none());
}

#[test]
fn teaching_records_steps_and_the_bot_writes_a_skill() {
    let s = setup_with("teach", true);
    s.custom_keys();
    let id = s.make(json!({"name": "Scout", "provider": "custom", "model": "test-model"}));
    let teach = |v: Value| {
        let r = s.ask("POST", &format!("/v1/bots/{id}/teach"), Some(v));
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()
    };
    assert_eq!(s.ask("POST", &format!("/v1/bots/{id}/teach"), Some(json!({"do": "tap", "x": 0.5, "y": 0.5}))).status, 400, "a session starts first");
    let st = teach(json!({"do": "start"}));
    assert_eq!((st["steps"].clone(), st["shot"].clone()), (json!([]), json!(1)));
    assert!(s.ask("GET", &format!("/v1/bots/{id}/teach/shot"), None).status == 200);
    teach(json!({"do": "open", "url": "cats.example"}));
    teach(json!({"do": "text", "text": "cats"}));
    teach(json!({"do": "text", "text": "hunter2"}));
    teach(json!({"do": "tap", "x": 0.5, "y": 0.1}));
    let st = teach(json!({"do": "key", "key": "Enter"}));
    assert_eq!(
        st["steps"],
        json!([
            "Opened cats.example",
            "Typed “cats” into the search box “Search”",
            "Typed their password into the password field “Password” (not written down)",
            "Clicked the button “Search”",
            "Pressed Enter → now on “Cats” (https://cats.example/results)"
        ])
    );
    // While it lasts, the Bots' own browser waits.
    assert_eq!(s.gw.gateway.bots.as_ref().unwrap().teaching().as_deref(), Some(id.as_str()));
    let r = s.ask("POST", &format!("/v1/bots/{id}/teach/done"), Some(json!({"what": "Finding a cat picture"})));
    assert_eq!(r.status, 200, "{}", r.body);
    let chat = s.settle(&id);
    assert_eq!(chat[0]["text"], "Finding a cat picture");
    assert!(!chat[0].to_string().contains("hunter2"), "a password is never kept");
    assert_eq!(chat.last().unwrap()["text"], "Proposed /find-cats.");
    let sent = s.seen.lock().unwrap().cloud[0].clone();
    let user = last_user(&sent);
    assert!(user.contains("What it's for: Finding a cat picture") && user.contains("4. Clicked the button “Search”"), "{user}");
    assert!(s.gw.gateway.bots.as_ref().unwrap().teaching().is_none());
    assert_eq!(s.ask("GET", "/v1/skills", None).json()["skills"][0]["approved"], false);
}
