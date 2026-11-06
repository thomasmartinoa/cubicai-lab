//! The gateway's rules, piece by piece, against what the Python gateway does.

use cubic_gateway::access::{self, Lan6, Net};
use cubic_gateway::activity::Activity;
use cubic_gateway::chat;
use cubic_gateway::http::Conn;
use cubic_gateway::identity;
use cubic_gateway::ollama::{self, ModelInfo};
use cubic_gateway::search;
use cubic_gateway::settings::Settings;
use serde_json::{json, Value};
use std::io::{Cursor, Read, Write};
use std::net::IpAddr;
use std::time::{Duration, Instant};

fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}

// --- Who may connect ------------------------------------------------------------------------------

#[test]
fn home_network_always_tailscale_only_while_remote_is_on() {
    let lan6 = access::parse_if_inet6("2a02810d0a4000001234567890abcdef 03 40 00 00 wlan0\n");
    for home in ["127.0.0.1", "::1", "192.168.1.20", "10.0.0.7", "172.16.4.4", "172.31.255.1", "fd12::5", "fe80::1", "::ffff:192.168.1.20"] {
        assert!(access::allowed(ip(home), false, &lan6), "{home} is home");
    }
    // The PC's own ISP prefix is home too (a phone on the same Wi-Fi gets an address in it).
    assert!(access::allowed(ip("2a02:810d:a40:0:aaaa::1"), false, &lan6));
    assert!(!access::allowed(ip("2a02:810d:a40:1::1"), false, &lan6), "a neighbouring prefix isn't");
    for outside in ["8.8.8.8", "172.32.0.1", "2001:4860::8888", "::ffff:8.8.8.8", "169.254.1.1"] {
        assert!(!access::allowed(ip(outside), true, &lan6), "{outside} is outside");
    }
    for ts in ["100.101.1.2", "100.64.0.1", "fd7a:115c:a1e0::5"] {
        assert!(!access::allowed(ip(ts), false, &lan6), "{ts} needs remote on");
        assert!(access::allowed(ip(ts), true, &lan6), "{ts} with remote on");
    }
}

#[test]
fn ipv6_home_networks_from_the_kernel_list() {
    let text = "\
2a02810d0a4000001234567890abcdef 03 40 00 00 wlan0
2a02810d0a400000aaaaaaaaaaaaaaaa 03 40 00 01 wlan0
fd000000000000000000000000000005 03 40 00 80 wlan0
fe800000000000001234567890abcdef 03 40 20 80 wlan0
00000000000000000000000000000001 01 80 10 80 lo
fd7a115ca1e0000000000000000000aa 05 80 00 80 tailscale0
2a02810d0a4000000000000000000099 06 40 00 00 docker0
2a02810d0a40000000000000000000dd 03 40 00 20 wlan0
2a02810d0a40000000000000000000ee 03 40 00 40 eth0
short line
";
    let nets = access::parse_if_inet6(text);
    let addrs: Vec<String> = nets.iter().map(|l| l.addr.to_string()).collect();
    assert_eq!(addrs, ["2a02:810d:a40:0:1234:5678:90ab:cdef", "fd00::5"], "temporary, deprecated, tentative, link-local, Tailscale and Docker are left out");
    assert_eq!(nets[0].net, Net::v6([0x2a02, 0x810d, 0xa40, 0, 0, 0, 0, 0], 64));
    // The stable ULA is preferred for the phone's IPv6 URL; the ISP prefix otherwise.
    assert_eq!(access::lan6_url(&nets, 8443).as_deref(), Some("https://[fd00::5]:8443"));
    assert_eq!(access::lan6_url(&nets[..1], 8443).as_deref(), Some("https://[2a02:810d:a40:0:1234:5678:90ab:cdef]:8443"));
    assert_eq!(access::lan6_url(&[] as &[Lan6], 8443), None);
}

#[test]
fn only_public_addresses_count_as_global() {
    for public in ["8.8.8.8", "1.1.1.1", "142.250.180.14", "2606:4700::1111", "2a00:1450:4001::200e"] {
        assert!(access::is_global(ip(public)), "{public}");
    }
    for private in [
        "127.0.0.1", "10.1.2.3", "192.168.1.1", "172.20.0.1", "100.64.1.1", "169.254.169.254", "0.0.0.0", "224.0.0.1", "255.255.255.255", "198.18.0.1", "::1", "::",
        "fe80::1", "fd00::1", "ff02::1", "::ffff:192.168.1.1", "::ffff:127.0.0.1", "2001:db8::1", "64:ff9b::c0a8:101", "64:ff9b::7f00:1",
    ] {
        assert!(!access::is_global(ip(private)), "{private}");
    }
    assert!(access::is_global(ip("64:ff9b::808:808")), "NAT64 of a public address is public");
}

// --- Identity and pairing -------------------------------------------------------------------------

#[test]
fn pairing_link_and_token_match_the_python_gateway() {
    let link = identity::pairing_link("https://[fd00::5]:8443", "ab-_Z9", "0f0f0f0f", Some("https://100.101.1.2:8443"));
    // Python: urllib.parse.urlencode({'u':…,'t':…,'f':…,'r':…})
    assert_eq!(link, "cubicai://pair?u=https%3A%2F%2F%5Bfd00%3A%3A5%5D%3A8443&t=ab-_Z9&f=0f0f0f0f&r=https%3A%2F%2F100.101.1.2%3A8443");
    assert_eq!(identity::form_decode("a+b%20c%3A%2f%zz%4"), "a b c:/%zz%4");
    assert!(identity::token_matches(Some("Bearer s3cret"), "s3cret"));
    for bad in [None, Some(""), Some("Bearer s3cre"), Some("Bearer s3cret "), Some("bearer s3cret"), Some("s3cret")] {
        assert!(!identity::token_matches(bad, "s3cret"), "{bad:?}");
    }
}

#[test]
fn identity_is_made_once_and_kept() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-id-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let a = identity::load(&dir).expect("made");
    assert_eq!(a.token.len(), 43, "32 random bytes, URL-safe base64");
    assert!(a.token.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
    let b = identity::load(&dir).expect("loaded");
    assert_eq!((a.token.as_str(), a.fingerprint.as_str()), (b.token.as_str(), b.fingerprint.as_str()), "the same identity next time");
    // The fingerprint is what openssl says (SHA-256 of the DER certificate), what the phone pins.
    let out = std::process::Command::new("openssl").args(["x509", "-noout", "-fingerprint", "-sha256", "-in"]).arg(dir.join("cert.pem")).output().unwrap();
    let want = String::from_utf8_lossy(&out.stdout).split('=').nth(1).unwrap().trim().replace(':', "").to_lowercase();
    assert_eq!(a.fingerprint, want);
    use std::os::unix::fs::PermissionsExt;
    for (f, mode) in [("key.pem", 0o600), ("token", 0o600)] {
        assert_eq!(std::fs::metadata(dir.join(f)).unwrap().permissions().mode() & 0o777, mode, "{f}");
    }
    assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    let t = identity::rotate_token(&dir).unwrap();
    assert_ne!(t, a.token);
    assert_eq!(identity::load(&dir).unwrap().token, t, "rotating un-pairs every phone");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn settings_apply_live() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-set-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let s = Settings::new(&dir);
    assert!(!s.remote());
    assert_eq!(s.num_ctx(16384), 16384);
    assert_eq!(s.keep_alive(), json!("15m"));
    let mut m = serde_json::Map::new();
    m.insert("remote".into(), true.into());
    s.save(m).unwrap();
    assert!(s.remote());
    // Another program (the panel) writes the file: seen at once, other keys kept.
    let other = Settings::new(&dir);
    let mut m = serde_json::Map::new();
    m.insert("num_ctx".into(), 8192.into());
    other.save(m).unwrap();
    assert_eq!((s.remote(), s.num_ctx(16384)), (true, 8192));
    std::fs::write(dir.join("settings.json"), "not json").unwrap();
    std::thread::sleep(Duration::from_millis(20));
    let fresh = Settings::new(&dir);
    assert!(!fresh.remote(), "a broken file means defaults");
    let _ = std::fs::remove_dir_all(&dir);
}

// --- HTTP -----------------------------------------------------------------------------------------

/// Reads from a fixed input, records what's written.
struct Pipe {
    input: Cursor<Vec<u8>>,
    output: Vec<u8>,
}

impl Read for Pipe {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        // Small reads, so a request split across reads is exercised.
        let n = b.len().min(7);
        self.input.read(&mut b[..n])
    }
}

impl Write for Pipe {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.output.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn pipe(input: &str) -> Conn<Pipe> {
    Conn::new(Pipe { input: Cursor::new(input.as_bytes().to_vec()), output: vec![] })
}

#[test]
fn requests_keep_alive_bodies_and_queries() {
    let mut c = pipe("POST /v1/chat/completions/ HTTP/1.1\r\nHost: x\r\ncontent-length: 7\r\n\r\n{\"a\":1}GET /v1/media/save?kind=image&name=My+cat%21 HTTP/1.1\r\nConnection: close\r\n\r\n");
    let r = c.next_request().unwrap().unwrap();
    assert_eq!((r.method.as_str(), r.path()), ("POST", "/v1/chat/completions"), "trailing slash ignored");
    assert_eq!(r.content_length().unwrap(), Some(7));
    assert_eq!(c.read_body(7).unwrap(), b"{\"a\":1}");
    let r2 = c.next_request().unwrap().unwrap();
    assert_eq!(r2.path(), "/v1/media/save");
    assert_eq!(r2.query("name").as_deref(), Some("My cat!"));
    assert_eq!(r2.query("kind").as_deref(), Some("image"));
    assert_eq!(r2.query("nope"), None);
    assert!(r2.wants_close());
    assert!(c.next_request().unwrap().is_none(), "clean end between requests");
}

#[test]
fn error_replies_drain_the_body_and_close() {
    let mut c = pipe("POST /nope HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloGET /x HTTP/1.1\r\n\r\n");
    let r = c.next_request().unwrap().unwrap();
    let keep = c.json(&r, 404, &cubic_gateway::http::error("Not found")).unwrap();
    assert!(!keep);
    let out = String::from_utf8(c.stream().output.clone()).unwrap();
    assert!(out.starts_with("HTTP/1.1 404 Not Found\r\n"), "{out}");
    assert!(out.contains("Connection: close\r\n") && out.contains("Content-Type: application/json\r\n"));
    assert!(out.ends_with("\r\n\r\n{\"error\":{\"message\":\"Not found\"}}"));
    // The unread body was read, so what follows is the next request, not "hello".
    assert_eq!(c.next_request().unwrap().unwrap().path(), "/x");
    // A 200 keeps the connection.
    let mut c = pipe("GET /v1/health HTTP/1.1\r\n\r\n");
    let r = c.next_request().unwrap().unwrap();
    assert!(c.json(&r, 200, &json!({"ok": true})).unwrap());
}

#[test]
fn broken_requests_are_refused() {
    assert!(pipe("GET / HTTP/1.1\r\nHost: x").next_request().is_err(), "cut short");
    assert!(pipe("\u{1}\u{2} nonsense\r\n\r\n").next_request().is_err());
    let huge = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(70 * 1024));
    assert!(pipe(&huge).next_request().is_err(), "head too large");
    let mut c = pipe("POST / HTTP/1.1\r\nContent-Length: x1\r\n\r\n");
    assert!(c.next_request().unwrap().unwrap().content_length().is_err());
}

// --- Ollama ---------------------------------------------------------------------------------------

fn info(caps: &[&str], own: Option<u64>, longest: u64) -> ModelInfo {
    ModelInfo { caps: caps.iter().map(|c| c.to_string()).collect(), own_ctx: own, longest }
}

#[test]
fn model_info_and_context() {
    let show = json!({
        "capabilities": ["completion", "vision", "tools"],
        "parameters": "stop \"<end>\"\nnum_ctx_extra 5\nnum_ctx                        32768\ntemperature 1",
        "model_info": {"gemma3.context_length": 131072, "gemma3.embedding_length": 2560},
    });
    let i = ModelInfo::from_show(&show);
    assert_eq!(i, info(&["completion", "vision", "tools"], Some(32768), 131072));
    assert_eq!(i.context(16384), 32768, "the Modelfile wins");
    assert_eq!(info(&[], None, 8192).context(16384), 8192, "capped at what the model supports");
    assert_eq!(info(&[], None, 0).context(16384), 16384);
    assert_eq!(ModelInfo::from_show(&json!({})), ModelInfo::default());
}

#[test]
fn model_list_marks_what_fits() {
    let g = 1u64 << 30;
    let models = vec![
        ("embed:latest".to_string(), g, info(&["embedding"], None, 8192)),
        ("gemma4:e4b".to_string(), 6 * g, info(&["completion", "vision", "tools", "thinking"], None, 131072)),
        ("llama3.1:8b".to_string(), 5 * g, info(&["completion", "tools"], None, 131072)),
        ("qwen3.5:27b".to_string(), 17 * g, info(&["completion", "vision"], Some(4096), 262144)),
        ("unknown:1b".to_string(), g, ModelInfo::default()),
    ];
    let list = ollama::model_entries(&models, 16384, Some(8 * g));
    let ids: Vec<&str> = list.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["gemma4:e4b", "llama3.1:8b", "qwen3.5:27b", "unknown:1b"], "embedding models left out, unknown kept");
    assert_eq!(list[0]["input_modalities"], json!(["text", "image"]));
    assert_eq!(list[0]["supported_parameters"], json!(["tools"]));
    assert_eq!(list[1]["supported_parameters"], json!(["tools"]));
    assert_eq!(list[2]["supported_parameters"], json!([]));
    assert_eq!(list[2]["context_length"], 4096);
    assert_eq!(list[0]["context_length"], 16384);
    let rec: Vec<&str> = list.iter().filter(|m| m["cubic_recommended"] == true).map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(rec, ["gemma4:e4b"], "the biggest vision model that fits");
    let small_gpu = ollama::model_entries(&models, 16384, Some(2 * g));
    let rec: Vec<&str> = small_gpu.iter().filter(|m| m["cubic_recommended"] == true).map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(rec, ["gemma4:e4b"], "none fits: the smallest vision model");
}

#[test]
fn messages_become_ollama_messages() {
    let msgs = vec![
        json!({"role": "system", "content": "Be kind."}),
        json!({"role": "user", "content": [{"type": "text", "text": "What's this?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}),
        json!({"role": "assistant", "content": "A cat.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "create", "arguments": "{\"prompt\": \"a cat\"}"}}]}),
        json!({"role": "tool", "content": "made", "name": "create", "tool_call_id": "c1"}),
        json!({"role": "assistant", "content": null, "tool_calls": [{"function": {"name": "create", "arguments": "not json"}}]}),
        json!({"role": "user", "content": [{"type": "text", "text": "Thanks"}, {"type": "text", "text": "really"}]}),
    ];
    let out = ollama::to_native(&msgs, true).unwrap();
    assert_eq!(out[0], json!({"role": "system", "content": "Be kind."}));
    assert_eq!(out[1], json!({"role": "user", "content": "What's this?", "images": ["AAAA"]}));
    assert_eq!(out[2]["tool_calls"], json!([{"function": {"name": "create", "arguments": {"prompt": "a cat"}}}]), "arguments become an object");
    assert_eq!(out[3], json!({"role": "tool", "content": "made", "tool_name": "create"}));
    assert_eq!(out[4], json!({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "create", "arguments": {}}}]}));
    assert_eq!(out[5]["content"], "Thanks\nreally");
    // A model that can't see: older pictures become a note, one in the latest message is an error.
    let blind = ollama::to_native(&msgs, false).unwrap();
    assert_eq!(blind[1]["content"], "What's this?\n[image omitted: current model has no vision]");
    assert!(blind[1].get("images").is_none());
    let err = ollama::to_native(&msgs[..2], false).unwrap_err();
    assert!(err.contains("can't see images"), "{err}");
}

#[test]
fn tool_calls_come_back_like_the_python_gateway_sends_them() {
    let calls = json!([{"function": {"name": "create", "arguments": {"prompt": "a café 🌍 \"q\"\n"}}}, {"function": {"name": "speak"}}]);
    let out = ollama::openai_tool_calls(&calls, 7);
    assert_eq!(out[0]["id"], "call_7_0");
    assert_eq!(out[1]["id"], "call_7_1");
    assert_eq!(out[0]["index"], 0);
    assert_eq!(out[0]["type"], "function");
    // Python: json.dumps({"prompt": "a café 🌍 \"q\"\n"})
    assert_eq!(out[0]["function"]["arguments"], "{\"prompt\": \"a caf\\u00e9 \\ud83c\\udf0d \\\"q\\\"\\n\"}");
    assert_eq!(out[1]["function"]["arguments"], "{}");
    assert_eq!(ollama::python_dumps(&json!([1, 2.5, null, true])), "[1, 2.5, null, true]");
}

// --- Chat -----------------------------------------------------------------------------------------

#[test]
fn chat_request_flags() {
    let thinker = info(&["completion", "thinking", "tools"], None, 0);
    let plain = info(&["completion"], None, 0);
    let tools = json!([{"type": "function", "function": {"name": "create"}}]);
    let r = chat::ollama_request(&json!({"tools": tools}), "m", vec![], &thinker, 8192, json!("15m"));
    assert_eq!((r["think"].clone(), r["options"]["num_ctx"].clone(), r["keep_alive"].clone(), r["stream"].clone()), (json!(true), json!(8192), json!("15m"), json!(true)));
    assert_eq!(r["tools"], tools);
    let call = chat::ollama_request(&json!({"cubic_think": false, "tools": tools, "tool_choice": "none"}), "m", vec![], &thinker, 8192, json!(-1));
    assert_eq!(call["think"], false, "a voice call turns thinking off");
    assert!(call.get("tools").is_none(), "tool_choice none leaves the tools out");
    let p = chat::ollama_request(&json!({"cubic_think": false, "tools": tools}), "m", vec![], &plain, 4096, json!("15m"));
    assert_eq!(p["think"], false, "off goes to every model");
    assert!(p.get("tools").is_none(), "a model without tools gets none");
    assert!(chat::ollama_request(&json!({}), "m", vec![], &plain, 4096, json!("15m")).get("think").is_none());

    let mut m = vec![json!({"role": "user", "content": "hi"})];
    chat::add_context(&mut m, "RESULTS");
    assert_eq!(m[0], json!({"role": "system", "content": "RESULTS"}));
    chat::add_context(&mut m, "MORE");
    assert_eq!(m[0]["content"], "RESULTS\n\nMORE");
}

struct Events(Vec<Value>, bool, usize);

impl chat::Sink for Events {
    fn event(&mut self, v: &Value) -> std::io::Result<()> {
        if self.0.len() >= self.2 {
            return Err(std::io::ErrorKind::BrokenPipe.into());
        }
        self.0.push(v.clone());
        Ok(())
    }

    fn done(&mut self) -> std::io::Result<()> {
        self.1 = true;
        Ok(())
    }
}

#[test]
fn ollama_stream_becomes_openai_deltas() {
    let stream = [
        json!({"message": {"role": "assistant", "content": "", "thinking": "Hmm"}, "done": false}),
        json!({"message": {"role": "assistant", "content": "", "thinking": " ok"}, "done": false}),
        json!({"message": {"role": "assistant", "content": "Hello"}, "done": false}),
        json!({"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "create", "arguments": {"prompt": "x"}}}]}, "done": false}),
        json!({"message": {"role": "assistant", "content": ""}, "done": true, "eval_count": 42, "eval_duration": 2_000_000_000u64, "prompt_eval_count": 300}),
        json!({"message": {"content": "after done"}}),
    ];
    let text = stream.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n\n") + "\n";
    let mut ev = Events(vec![], false, usize::MAX);
    let mut phases = vec![];
    let o = chat::relay(Cursor::new(text.clone()), &mut ev, 3, &mut |p| phases.push(p.to_string())).unwrap();
    assert_eq!((o.outcome, o.tokens, o.gen_seconds), ("ok", 42, 2.0));
    assert!(ev.1, "[DONE] sent");
    assert_eq!(ev.0[0], json!({"choices": [{"index": 0, "delta": {"reasoning": "Hmm"}}]}));
    assert_eq!(ev.0[2], json!({"choices": [{"index": 0, "delta": {"content": "Hello"}}]}));
    assert_eq!(ev.0[3]["choices"][0]["delta"]["tool_calls"][0]["id"], "call_3_0");
    assert_eq!(ev.0[4], json!({"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 42}}));
    assert_eq!(ev.0.len(), 5, "nothing after done");
    assert_eq!(phases, ["Thinking", "Answering", "Asking for a creation"]);

    let mut ev = Events(vec![], false, usize::MAX);
    let o = chat::relay(Cursor::new("{\"error\": \"model 'x' not found\"}\n"), &mut ev, 1, &mut |_| {}).unwrap();
    assert_eq!(o.outcome, "error");
    assert_eq!(ev.0, [json!({"error": {"message": "model 'x' not found"}})]);
    assert!(!ev.1);

    // The phone leaves: relaying stops at once.
    let mut ev = Events(vec![], false, 1);
    assert!(chat::relay(Cursor::new(text), &mut ev, 1, &mut |_| {}).is_err());
}

// --- Web search -----------------------------------------------------------------------------------

#[test]
fn search_query_from_the_conversation() {
    let msgs = vec![
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "Who won?"}),
        json!({"role": "assistant", "content": "Which match?"}),
        json!({"role": "user", "content": "The final yesterday"}),
    ];
    let r = search::query_request("m", &info(&["thinking"], None, 0), &msgs, "2026-11-07");
    assert_eq!(r["think"], false);
    assert_eq!(r["options"], json!({"temperature": 0, "num_predict": 256}));
    assert!(r["messages"][0]["content"].as_str().unwrap().starts_with("Today is 2026-11-07. Rewrite"));
    assert_eq!(r["messages"][1]["content"], "Conversation so far:\nuser: Who won?\nassistant: Which match?\n\nLatest message: The final yesterday");
    let alone = search::query_request("m", &info(&[], None, 0), &msgs[..2], "d");
    assert!(alone.get("think").is_none());
    assert_eq!(alone["messages"][1]["content"], "Conversation so far:\n(none)\n\nLatest message: Who won?");

    assert_eq!(search::clean_query("<think>let me see</think>\n\"world cup final 2026 result\"\nextra", "x"), "world cup final 2026 result");
    assert_eq!(search::clean_query("  `rust 2024 edition`  ", "x"), "rust 2024 edition");
    assert_eq!(search::clean_query("<think>only thinking</think>", " the user's words "), "the user's words");
    assert_eq!(search::clean_query(&"q".repeat(300), "x").len(), 200);
}

#[test]
fn search_results_pages_and_context() {
    let v = json!({"results": [
        {"title": "A", "url": "https://a.example/", "content": "snip a"},
        {"title": "A again", "url": "https://a.example/", "content": "dup"},
        {"title": "No url"},
        {"title": "B", "url": "https://b.example/", "content": "snip b"},
    ]});
    let hits = search::parse_results(&v);
    assert_eq!(hits.iter().map(|h| h.title.as_str()).collect::<Vec<_>>(), ["A", "B"]);
    let many = json!({"results": (0..20).map(|i| json!({"url": format!("https://{i}.example/")})).collect::<Vec<_>>()});
    assert_eq!(search::parse_results(&many).len(), search::SEARCH_RESULTS);

    let c = search::context("q", &hits, &["page a text".into(), String::new()], "Saturday 07 November 2026 10:00");
    assert_eq!(
        c,
        "Live web search results for \"q\" (retrieved Saturday 07 November 2026 10:00):\n\n[1] A\nURL: https://a.example/\npage a text\n\n[2] B\nURL: https://b.example/\nsnip b\n\nUse these results to answer the user's latest message. Cite sources inline like [1] or [2], matching the numbers above. If the results don't answer it, say so and answer from your own knowledge."
    );
}

#[test]
fn page_text_skips_the_furniture() {
    let html = r#"<!doctype html><html><head><title>Title</title><style>p{color:red}</style><script>var x = "<p>not text</p>";</script></head>
<body><nav><a href="/">Home</a> <a>Menu</a></nav><header>Site</header>
<h1>The   headline</h1><p>First &amp; best paragraph &mdash; with &#8220;quotes&#x201D; &unknown; a&b.</p>
<!-- a comment <p>hidden</p> --><div>x</div><ul><li>One</li><li>Two</li></ul>
<svg/><p>After a self-closing svg</p><form><input>Search</form><footer>(c) 2026</footer></body></html>"#;
    let t = search::html_text(html);
    assert_eq!(t, "Title\nThe headline\nFirst & best paragraph — with “quotes” &unknown; a&b.\nOne\nTwo\nAfter a self-closing svg");
    assert_eq!(search::decode(&[0x63, 0x61, 0x66, 0xe9], "text/html; charset=ISO-8859-1"), "café");
    assert_eq!(search::decode("café".as_bytes(), "text/html"), "café");
}

#[test]
fn slow_pages_fall_back_to_their_snippet() {
    let urls: Vec<String> = ["fast", "slow", "fast2"].iter().map(|s| s.to_string()).collect();
    let t = Instant::now();
    let pages = search::read_pages(&urls, Duration::from_millis(300), |u| {
        if u == "slow" {
            std::thread::sleep(Duration::from_secs(3));
        }
        format!("text of {u}")
    });
    assert!(t.elapsed() < Duration::from_secs(1), "the deadline holds");
    assert_eq!(pages, ["text of fast", "", "text of fast2"]);
}

#[test]
fn pages_on_the_home_network_are_never_fetched() {
    // A server on this PC that would answer: the page reader must refuse to connect at all.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let h = hit.clone();
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            h.store(true, std::sync::atomic::Ordering::SeqCst);
            let mut s = s;
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\n\r\n<p>secret</p>");
        }
    });
    let agent = search::page_agent();
    for url in [format!("http://127.0.0.1:{port}/"), format!("http://localhost:{port}/"), format!("http://[::ffff:127.0.0.1]:{port}/"), "file:///etc/passwd".into(), "ftp://example.com/".into()] {
        assert_eq!(search::read_page(&agent, &url), "", "{url}");
    }
    std::thread::sleep(Duration::from_millis(100));
    assert!(!hit.load(std::sync::atomic::Ordering::SeqCst), "nothing even connected");
}

// --- Activity -------------------------------------------------------------------------------------

#[test]
fn activity_keeps_metadata_only() {
    let a = Activity::default();
    let id = a.begin("192.168.1.5", "gemma4:e4b", true);
    a.phase(id, "Searching the web");
    let s = a.snapshot();
    assert_eq!(s["active"][0]["phase"], "Searching the web");
    assert_eq!(s["devices"][0]["ip"], "192.168.1.5");
    a.end(id, "ok", 100, 3.0);
    a.end(id, "ok", 100, 3.0); // twice is harmless
    a.reject("8.8.8.8");
    let s = a.snapshot();
    assert_eq!((s["served"].clone(), s["blocked"].clone()), (json!(1), json!(1)));
    assert_eq!(s["active"], json!([]));
    assert_eq!(s["recent"][0]["outcome"], "blocked");
    let done = &s["recent"][1];
    assert_eq!((done["outcome"].clone(), done["tokens"].clone(), done["tps"].clone(), done["search"].clone()), (json!("ok"), json!(100), json!(33.3), json!(true)));
    assert!(done.get("phase").is_none());
    assert!(s["uptime"].as_f64().unwrap() >= 0.0);
    for i in 0..300 {
        a.reject(&format!("1.1.1.{}", i % 250));
    }
    assert_eq!(a.snapshot()["recent"].as_array().unwrap().len(), 200, "only the latest 200");
}
