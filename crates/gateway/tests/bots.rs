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
    let body: Value = match req.content_length() {
        Ok(Some(n)) if n > 0 => serde_json::from_slice(&c.read_body(n).unwrap()).unwrap_or(Value::Null),
        _ => Value::Null,
    };
    match req.path() {
        "/chat/completions" => {
            seen.lock().unwrap().cloud.push(body.clone());
            seen.lock().unwrap().auth.push(req.header("Authorization").unwrap_or("").to_string());
            let msgs = body["messages"].as_array().unwrap();
            let last = msgs.last().unwrap();
            let user = last_user(&body);
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
    let dir = std::env::temp_dir().join(format!("cubic-gw-bots-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (url, seen) = fake();
    let paths = bots::Paths { bots: dir.join("bots"), workspace: dir.join("workspace") };
    let config = Config { dir: dir.join("config"), ollama: url.clone(), searxng: url.clone(), default_ctx: 16384, media: None, voice: None, bots: Some(paths) };
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
    assert_eq!(first["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect::<Vec<_>>(), ["web_search", "read_page", "list_files", "read_file", "write_file", "remember", "forget", "schedule_routine", "cancel_routine"]);
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
    let paths = bots::Paths { bots: dir.join("bots"), workspace: dir.join("workspace") };
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
