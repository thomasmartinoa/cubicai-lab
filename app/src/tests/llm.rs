use super::{mock_http, sse_response};
use crate::llm::{self, Ev, Request};
use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn collect(stream: &str) -> (Result<(), String>, Vec<Ev>) {
    let evs = Mutex::new(vec![]);
    let r = llm::read_stream(Cursor::new(stream.as_bytes().to_vec()), &AtomicBool::new(false), &|e| evs.lock().unwrap().push(e));
    (r, evs.into_inner().unwrap())
}

fn text(evs: &[Ev]) -> (String, String) {
    let mut c = String::new();
    let mut r = String::new();
    for e in evs {
        match e {
            Ev::Content(s) => c += s,
            Ev::Reasoning(s) => r += s,
            _ => {}
        }
    }
    (c, r)
}

#[test]
fn content_reasoning_usage_and_speed() {
    let s = concat!(
        ": keep-alive comment\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me \"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think.\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"timings\":{\"predicted_per_second\":42.5}}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    );
    let (r, evs) = collect(s);
    assert_eq!(r, Ok(()));
    assert_eq!(text(&evs), ("Hello world".into(), "Let me think.".into()));
    assert!(evs.iter().any(|e| matches!(e, Ev::Usage { prompt: 12, output: 7 })));
    assert!(evs.iter().any(|e| matches!(e, Ev::Speed { gen_tps } if (*gen_tps - 42.5).abs() < 1e-9)));
}

#[test]
fn reasoning_under_either_name() {
    let (_, evs) = collect("data: {\"choices\":[{\"delta\":{\"reasoning\":\"hmm\"},\"finish_reason\":\"stop\"}]}\n\n");
    assert_eq!(text(&evs).1, "hmm");
}

#[test]
fn a_finish_reason_without_done_is_complete() {
    let (r, _) = collect("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"length\"}]}\n\n");
    assert_eq!(r, Ok(()));
}

#[test]
fn a_stream_that_just_ends_is_a_dropped_connection() {
    let (r, evs) = collect("data: {\"choices\":[{\"delta\":{\"content\":\"half a sen\"}}]}\n\n");
    assert_eq!(text(&evs).0, "half a sen");
    assert!(r.unwrap_err().contains("Connection dropped"));
}

#[test]
fn errors_inside_the_stream() {
    let (r, _) = collect("data: {\"error\":{\"message\":\"context too long\"}}\n\n");
    assert_eq!(r.unwrap_err(), "context too long");
}

#[test]
fn cancel_stops_reading() {
    let evs = Mutex::new(vec![]);
    let r = llm::read_stream(Cursor::new(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_vec()), &AtomicBool::new(true), &|e| {
        evs.lock().unwrap().push(e)
    });
    assert_eq!(r, Ok(()));
    assert!(evs.lock().unwrap().is_empty());
}

#[test]
fn http_errors_read_plainly() {
    assert!(llm::http_error(401, r#"{"error":{"message":"bad key"}}"#).starts_with("The key was refused (401): bad key"));
    assert!(llm::http_error(429, r#"{"message":"slow down"}"#).contains("out of quota"));
    assert!(llm::http_error(404, "nope").contains("Check the model name"));
    assert!(llm::http_error(503, "Loading model").contains("still loading"));
    assert_eq!(llm::http_error(500, &"x".repeat(1000)).len(), "Error 500: ".len() + 300);
}

fn run_stream(base_url: String) -> Vec<Ev> {
    let (tx, rx) = std::sync::mpsc::channel();
    let req = Request {
        base_url,
        key: "secret-key".into(),
        model: "tiny".into(),
        messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
        local: true,
    };
    llm::stream(req, 7, tx, Arc::new(AtomicBool::new(false)), egui::Context::default());
    let mut out = vec![];
    while let Ok((job, ev)) = rx.recv_timeout(Duration::from_secs(10)) {
        assert_eq!(job, 7);
        let end = matches!(ev, Ev::Done | Ev::Error(_));
        out.push(ev);
        if end {
            break;
        }
    }
    out
}

#[test]
fn streams_from_a_server_with_the_right_request() {
    let seen = Arc::new(Mutex::new(String::new()));
    let s2 = seen.clone();
    let port = mock_http(move |req| {
        *s2.lock().unwrap() = req.to_string();
        sse_response(&[r#"{"choices":[{"delta":{"content":"Hi!"},"finish_reason":"stop"}]}"#, "[DONE]"])
    });
    let evs = run_stream(format!("http://127.0.0.1:{port}/v1/"));
    assert_eq!(text(&evs).0, "Hi!");
    assert!(matches!(evs.last(), Some(Ev::Done)));
    let req = seen.lock().unwrap().clone();
    assert!(req.starts_with("POST /v1/chat/completions "), "{req}");
    assert!(req.contains("Bearer secret-key"));
    let body: serde_json::Value = serde_json::from_str(&req[req.find("\r\n\r\n").unwrap() + 4..]).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["model"], "tiny");
}

#[test]
fn server_errors_become_messages() {
    let port = mock_http(|_| {
        let body = r#"{"error":{"message":"model not found"}}"#;
        format!("HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
    });
    let evs = run_stream(format!("http://127.0.0.1:{port}/v1"));
    match evs.last() {
        Some(Ev::Error(e)) => assert!(e.contains("model not found") && e.contains("404"), "{e}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn nothing_listening_says_so() {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let evs = run_stream(format!("http://127.0.0.1:{port}/v1"));
    match evs.last() {
        Some(Ev::Error(e)) => assert!(e.contains("Nothing is answering"), "{e}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn model_lists_with_context_sizes() {
    let port = mock_http(|_| {
        let body = r#"{"data":[{"id":"models/zeta"},{"id":"alpha","context_length":8192},{"id":"mid","top_provider":{"context_length":4096}}]}"#;
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
    });
    let list = llm::list_models(&format!("http://127.0.0.1:{port}/v1"), "").unwrap();
    assert_eq!(list, vec![("alpha".into(), Some(8192)), ("mid".into(), Some(4096)), ("zeta".into(), None)]);
}
