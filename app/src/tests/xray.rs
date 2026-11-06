use super::mock_http;
use crate::engine::xray::{self, BranchState, Candidate, Sampling, Step, Tree, XEv};
use serde_json::json;
use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn c(id: i64, text: &str, p: f32) -> Candidate {
    Candidate { id, bytes: text.as_bytes().to_vec(), p }
}

fn step(id: i64, text: &str, p: f32, top: &[(i64, &str, f32)]) -> Step {
    Step { chosen: c(id, text, p), top: top.iter().map(|&(i, t, p)| c(i, t, p)).collect() }
}

/// A streamed `/completion` chunk exactly as llama-server writes it (server-task.cpp, partial).
fn chunk(id: i64, text: &str, p: f32, top: &[(i64, &str, f32)]) -> String {
    let entry = |id: i64, t: &str, p: f32| json!({"id": id, "token": t, "bytes": t.as_bytes(), "logprob": p.ln()});
    json!({
        "index": 0, "content": text, "tokens": [id], "stop": false, "id_slot": 0, "tokens_predicted": 1, "tokens_evaluated": 9,
        "completion_probabilities": [{
            "id": id, "token": text, "bytes": text.as_bytes(), "logprob": p.ln(),
            "top_logprobs": top.iter().map(|&(i, t, p)| entry(i, t, p)).collect::<Vec<_>>()
        }]
    })
    .to_string()
}

#[test]
fn parses_both_probability_shapes() {
    let v: serde_json::Value = serde_json::from_str(&chunk(5, " cat", 0.6, &[(5, " cat", 0.6), (9, " dog", 0.3)])).unwrap();
    let s = xray::parse_steps(&v);
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].chosen.text(), " cat");
    assert!((s[0].chosen.p - 0.6).abs() < 1e-6);
    assert_eq!(s[0].top.len(), 2);
    assert_eq!(s[0].rank(), Some(0));

    let post = json!({"completion_probabilities": [{"id": 1, "token": "a", "bytes": [97], "prob": 0.9, "top_probs": [{"id": 1, "token": "a", "bytes": [97], "prob": 0.9}]}]});
    let s = xray::parse_steps(&post);
    assert_eq!(s[0].top[0].p, 0.9);
    // A token that is half of a UTF-8 character: the text is cut but the bytes are whole.
    let half = json!({"completion_probabilities": [{"id": 2, "token": "", "bytes": [0xE2, 0x82], "logprob": -0.1, "top_logprobs": []}]});
    assert_eq!(xray::parse_steps(&half)[0].chosen.bytes, vec![0xE2, 0x82]);
    // A logprob of "lowest float" (probability 0) doesn't break anything.
    let zero = json!({"completion_probabilities": [{"id": 3, "bytes": [98], "logprob": -3.4028235e38, "top_logprobs": []}]});
    assert_eq!(xray::parse_steps(&zero)[0].chosen.p, 0.0);
    assert!(xray::parse_steps(&json!({"content": "x"})).is_empty());
}

#[test]
fn uncertainty_in_bits() {
    let sure = step(1, "a", 1.0, &[(1, "a", 1.0)]);
    assert!(sure.entropy().abs() < 1e-6);
    let coin = step(1, "a", 0.5, &[(1, "a", 0.5), (2, "b", 0.5)]);
    assert!((coin.entropy() - 1.0).abs() < 1e-6);
    let four = step(1, "a", 0.25, &[(1, "a", 0.25), (2, "b", 0.25), (3, "c", 0.25), (4, "d", 0.25)]);
    assert!((four.entropy() - 2.0).abs() < 1e-5);
    // The unseen rest counts as one more outcome.
    let partial = step(1, "a", 0.5, &[(1, "a", 0.5)]);
    assert!((partial.entropy() - 1.0).abs() < 1e-6);
    // A sampled token outside the shown candidates has no rank.
    assert_eq!(step(9, "z", 0.01, &[(1, "a", 0.9)]).rank(), None);
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

#[test]
fn sampler_lens() {
    let p = [0.5, 0.3, 0.15, 0.05];
    let none = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, ..Default::default() };
    assert!(close(&xray::lens(&p, &none), &[0.5, 0.3, 0.15, 0.05]), "no samplers: unchanged");

    let greedy = Sampling { temperature: 0.0, ..none.clone() };
    assert_eq!(xray::lens(&p, &greedy), vec![1.0, 0.0, 0.0, 0.0]);

    let k2 = Sampling { top_k: 2, ..none.clone() };
    assert!(close(&xray::lens(&p, &k2), &[0.625, 0.375, 0.0, 0.0]));

    // top-p 0.7: 0.5 is not enough, 0.5 + 0.3 reaches it; the rest are cut.
    let tp = Sampling { top_p: 0.7, ..none.clone() };
    assert!(close(&xray::lens(&p, &tp), &[0.625, 0.375, 0.0, 0.0]));
    // top-p 0.5 keeps only the favourite (it reaches 0.5 by itself).
    assert!(close(&xray::lens(&p, &Sampling { top_p: 0.5, ..none.clone() }), &[1.0, 0.0, 0.0, 0.0]));

    // min-p 0.2: anything under 0.2 × 0.5 = 0.1 goes.
    let mp = Sampling { min_p: 0.2, ..none.clone() };
    assert!(close(&xray::lens(&p, &mp), &[0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0]));

    // Temperature: low sharpens, high flattens, order never changes.
    let cold = xray::lens(&p, &Sampling { temperature: 0.5, ..none.clone() });
    let hot = xray::lens(&p, &Sampling { temperature: 2.0, ..none.clone() });
    assert!(cold[0] > 0.5 && hot[0] < 0.5);
    assert!(hot.windows(2).all(|w| w[0] >= w[1]) && cold.windows(2).all(|w| w[0] >= w[1]));
    assert!((cold.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // llama-server's defaults on a confident step: min-p trims the long tail.
    let d = xray::lens(&[0.9, 0.06, 0.03, 0.01], &Sampling::default());
    assert_eq!(d[3], 0.0);
    assert!(d[0] > 0.9);
}

#[test]
fn branches_fork_and_continue_by_token_id() {
    let mut t = Tree::new("<user>hi<assistant>".into());
    t.branches[0].steps = vec![
        step(10, "The", 0.9, &[(10, "The", 0.9), (11, "A", 0.05)]),
        step(20, " cat", 0.6, &[(20, " cat", 0.6), (21, " dog", 0.3)]),
        step(30, " sat", 0.7, &[(30, " sat", 0.7)]),
    ];
    t.branches[0].state = BranchState::Done;
    assert_eq!(t.text(0), "The cat sat");

    let f = t.fork(0, 1, c(21, " dog", 0.3));
    assert_eq!(t.branches[f].parent, Some((0, 1)));
    assert_eq!(t.text(f), "The dog");
    assert_eq!(t.offset(f), 1);
    assert_eq!(t.branches[f].steps[0].top.len(), 2, "keeps the alternatives it was chosen from");
    assert_eq!(t.continuation(f), json!(["<user>hi<assistant>", 10, 21]));

    t.branches[f].steps.push(step(40, " ran", 0.5, &[(40, " ran", 0.5), (41, " barked", 0.4)]));
    let g = t.fork(f, 1, c(41, " barked", 0.4));
    assert_eq!(t.text(g), "The dog barked");
    assert_eq!(t.continuation(g), json!(["<user>hi<assistant>", 10, 21, 41]));
    assert_eq!(t.offset(g), 2);
    assert_eq!(t.forks_of(0), vec![(f, 1)]);
    assert_eq!(t.forks_of(f), vec![(g, 1)]);
    assert_eq!(t.locate(g, 0), (0, 0), "the first word came from the main branch");
    assert_eq!(t.locate(g, 1), (f, 0));
    assert_eq!(t.locate(g, 2), (g, 0));
    assert_eq!(t.locate(0, 2), (0, 2));
    // The original is untouched.
    assert_eq!(t.text(0), "The cat sat");

    // Multi-byte characters split over tokens join up again.
    let mut u = Tree::new(String::new());
    u.branches[0].steps = vec![
        Step { chosen: Candidate { id: 1, bytes: vec![0xE2, 0x82], p: 1.0 }, top: vec![] },
        Step { chosen: Candidate { id: 2, bytes: vec![0xAC], p: 1.0 }, top: vec![] },
    ];
    assert_eq!(u.text(0), "€");
}

#[test]
fn reads_a_completion_stream() {
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\n",
        chunk(1, "Hi", 0.8, &[(1, "Hi", 0.8), (2, "Hello", 0.1)]),
        chunk(3, "!", 0.5, &[(3, "!", 0.5), (4, ".", 0.4)]),
        json!({"index": 0, "content": "", "tokens": [], "stop": true, "timings": {"predicted_per_second": 77.0}})
    );
    let got = Mutex::new(vec![]);
    let tps = xray::read_completion(Cursor::new(body), &AtomicBool::new(false), &|s| got.lock().unwrap().extend(s)).unwrap();
    assert_eq!(tps, Some(77.0));
    let got = got.into_inner().unwrap();
    assert_eq!(got.iter().map(|s| s.chosen.text()).collect::<String>(), "Hi!");

    // Cut off before `stop`.
    let cut = format!("data: {}\n\n", chunk(1, "Hi", 0.8, &[]));
    assert!(xray::read_completion(Cursor::new(cut), &AtomicBool::new(false), &|_| {}).unwrap_err().contains("dropped"));
    let err = "data: {\"error\":{\"code\":400,\"message\":\"the request exceeds the available context size\"}}\n\n";
    assert!(xray::read_completion(Cursor::new(err), &AtomicBool::new(false), &|_| {}).unwrap_err().contains("context size"));
}

#[test]
fn template_and_generation_against_a_server() {
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    let port = mock_http(move |req: &str| {
        s2.lock().unwrap().push(req.to_string());
        if req.starts_with("POST /apply-template") {
            let body = r#"{"prompt":"<|user|>hi<|assistant|>"}"#;
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
        } else {
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {}\n\ndata: {}\n\n",
                chunk(7, "Yo", 0.9, &[(7, "Yo", 0.9)]),
                json!({"stop": true, "content": "", "timings": {"predicted_per_second": 50.5}})
            )
        }
    });
    let root = xray::root_of(&format!("http://127.0.0.1:{port}/v1/"));
    assert_eq!(root, format!("http://127.0.0.1:{port}"));
    let prompt = xray::apply_template(&root, &[json!({"role": "user", "content": "hi"})]).unwrap();
    assert_eq!(prompt, "<|user|>hi<|assistant|>");

    let mut tree = Tree::new(prompt);
    let (tx, rx) = std::sync::mpsc::channel();
    let s = Sampling { temperature: 0.3, seed: 42, max_tokens: 64, ..Default::default() };
    xray::generate(root, tree.continuation(0), s, 0, tx, Arc::new(AtomicBool::new(false)), egui::Context::default());
    loop {
        match rx.recv_timeout(Duration::from_secs(10)).unwrap() {
            (0, XEv::Steps(st)) => tree.branches[0].steps.extend(st),
            (0, XEv::Done { gen_tps }) => {
                assert_eq!(gen_tps, Some(50.5));
                break;
            }
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(tree.text(0), "Yo");
    let reqs = seen.lock().unwrap();
    let completion = reqs.iter().find(|r| r.starts_with("POST /completion")).expect("completion request");
    let body: serde_json::Value = serde_json::from_str(&completion[completion.find("\r\n\r\n").unwrap() + 4..]).unwrap();
    assert_eq!(body["prompt"], json!(["<|user|>hi<|assistant|>"]));
    assert_eq!(body["n_probs"], xray::N_PROBS);
    assert_eq!(body["stream"], true);
    assert_eq!((body["seed"].as_i64(), body["n_predict"].as_u64()), (Some(42), Some(64)));
    assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
}
