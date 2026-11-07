use super::{mock_http, TempDir};
use crate::engine::arena::{Answer, Run, Vote, Who};
use crate::engine::datakit::{self, Change, Op};
use crate::engine::dataset::{self, Dataset, Example, Pair, Turn};
use serde_json::{json, Value};

fn ex(u: &str, a: &str) -> Example {
    Example::pair(u, a, "hand")
}

// --- formats -------------------------------------------------------------------------------------------

#[test]
fn records_in_every_common_shape() {
    let jsonl = [
        json!({"messages": [{"role": "system", "content": "Be a pirate."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Ahoy!"}]}),
        json!({"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}, {"from": "human", "value": "and 3+3?"}, {"from": "gpt", "value": "6"}]}),
        json!({"instruction": "Translate to French", "input": "cat", "output": "chat"}),
        json!({"instruction": "Say hi", "output": "Hi"}),
        json!({"question": "Capital of Peru?", "answer": "Lima"}),
        json!({"prompt": "Best colour?", "chosen": "Blue, calm and clear.", "rejected": "idk"}),
        json!({"prompt": [{"role": "user", "content": "Tea or coffee?"}], "chosen": [{"role": "assistant", "content": "Tea."}], "rejected": [{"role": "assistant", "content": "Mud."}]}),
        json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "parts"}]}, {"role": "assistant", "content": "ok"}]}),
        json!({"text": "just a paragraph"}),
        json!({"messages": [{"role": "assistant", "content": "starts with a reply"}]}),
    ]
    .iter()
    .map(|v| v.to_string())
    .collect::<Vec<_>>()
    .join("\n")
        + "\nnot json at all\n";
    let r = datakit::parse_records(&jsonl, "file").unwrap();
    assert_eq!(r.system.as_deref(), Some("Be a pirate."));
    let firsts: Vec<(&str, &str)> = r.examples.iter().map(|e| (e.first("user"), e.first("assistant"))).collect();
    assert_eq!(firsts, [("Hi", "Ahoy!"), ("2+2?", "4"), ("Translate to French\n\ncat", "chat"), ("Say hi", "Hi"), ("Capital of Peru?", "Lima"), ("parts", "ok")]);
    assert_eq!(r.examples[1].turns.len(), 4, "ShareGPT turns kept");
    assert!(r.examples.iter().all(|e| e.source == "file" && e.valid()));
    assert_eq!(r.pairs.len(), 2);
    assert_eq!((r.pairs[1].prompt.as_str(), r.pairs[1].chosen.as_str(), r.pairs[1].rejected.as_str()), ("Tea or coffee?", "Tea.", "Mud."));
    assert_eq!(r.skipped, 3, "the plain text, the reply-only chat and the broken line");
}

#[test]
fn files_as_arrays_wrapped_or_empty() {
    let arr = json!([{"prompt": "a", "response": "b"}]).to_string();
    assert_eq!(datakit::parse_records(&arr, "file").unwrap().examples.len(), 1);
    // The datasets server's own shape: {"rows": [{"row": {...}}]}
    let rows = json!({"rows": [{"row": {"query": "q", "response": "r"}}]}).to_string();
    assert_eq!(datakit::parse_records(&rows, "hf").unwrap().examples[0].first("assistant"), "r");
    assert!(datakit::parse_records("  ", "file").unwrap_err().contains("empty"));
    let e = datakit::parse_records("{\"a\": 1}\n{\"b\": 2}", "file").unwrap_err();
    assert!(e.contains("2 records skipped"), "{e}");
}

#[test]
fn hugging_face_names_and_rows() {
    assert_eq!(datakit::hf_name("tatsu-lab/alpaca").as_deref(), Some("tatsu-lab/alpaca"));
    assert_eq!(datakit::hf_name("https://huggingface.co/datasets/HuggingFaceH4/no_robots/viewer").as_deref(), Some("HuggingFaceH4/no_robots"));
    assert_eq!(datakit::hf_name("alpaca"), None);
    assert_eq!(datakit::hf_name("a/b c"), None);
    let port = mock_http(|req| {
        let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
        let body = if path.starts_with("/splits") {
            assert!(req.contains("Authorization: Bearer hf_x") || req.contains("authorization: Bearer hf_x"));
            json!({"splits": [{"config": "default", "split": "test"}, {"config": "default", "split": "train"}]})
        } else if path.contains("offset=0") {
            assert!(path.contains("config=default&split=train") && path.contains("length=100"), "{path}");
            json!({"rows": (0..100).map(|i| json!({"row_idx": i, "row": {"question": format!("q{i}"), "answer": "a"}})).collect::<Vec<_>>(), "num_rows_total": 130})
        } else {
            assert!(path.contains("offset=100&length=50"), "{path}");
            json!({"rows": (100..130).map(|i| json!({"row": {"question": format!("q{i}"), "answer": "a"}})).collect::<Vec<_>>(), "num_rows_total": 130})
        };
        let b = body.to_string();
        format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len())
    });
    let base = format!("http://127.0.0.1:{port}");
    let (config, split) = datakit::hf_split(&base, "o/n", "hf_x").unwrap();
    assert_eq!((config.as_str(), split.as_str()), ("default", "train"));
    let (rows, total) = datakit::hf_rows(&base, "o/n", &config, &split, 150, "hf_x", &|| false).unwrap();
    assert_eq!((rows.len(), total), (130, Some(130)), "stops at the end of the split");
    assert_eq!(rows[129]["question"], "q129");
    let gated = mock_http(|_| "HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\n\r\n{}");
    assert!(datakit::hf_split(&format!("http://127.0.0.1:{gated}"), "o/n", "").unwrap_err().contains("token"));
}

// --- documents -----------------------------------------------------------------------------------------

#[test]
fn documents_are_read_and_cut_into_pieces() {
    let dir = TempDir::new("docs");
    std::fs::write(dir.join("a.md"), "# Title\r\n\r\nFirst paragraph.\r\n\r\nSecond one.").unwrap();
    assert_eq!(datakit::read_document(&dir.join("a.md")).unwrap(), "# Title\n\nFirst paragraph.\n\nSecond one.");
    std::fs::write(dir.join("b.html"), "<html><style>p{}</style><body><h1>Hi</h1><p>One &amp; two</p><script>x()</script></body></html>").unwrap();
    let h = datakit::read_document(&dir.join("b.html")).unwrap();
    assert!(h.contains("Hi") && h.contains("One & two") && !h.contains("x()") && !h.contains('<'), "{h:?}");
    std::fs::write(dir.join("c.txt"), "  \n ").unwrap();
    assert!(datakit::read_document(&dir.join("c.txt")).is_err());
    std::fs::write(dir.join("d.pdf"), b"not a pdf").unwrap();
    assert!(datakit::read_document(&dir.join("d.pdf")).unwrap_err().contains("PDF"));

    let text = format!("{}\n\n{}\n\n{}", "a".repeat(500), "b".repeat(500), "Long sentence one. ".repeat(80));
    let c = datakit::chunks(&text, 1200);
    assert!(c.iter().all(|x| x.len() <= 1200), "{:?}", c.iter().map(|x| x.len()).collect::<Vec<_>>());
    assert!(c[0].starts_with("aaa") && c[0].contains("bbb"), "short paragraphs share a piece");
    let joined: String = c.join(" ");
    assert_eq!(joined.matches("Long sentence one.").count(), 80, "nothing lost");
    let one_word = datakit::chunks(&"x".repeat(2500), 1000);
    assert_eq!(one_word.iter().map(|s| s.len()).collect::<Vec<_>>(), [1000, 1000, 500]);
    let d = Dataset { brief: "Answer like a teacher.".into(), ..Default::default() };
    let m = datakit::doc_messages("Manual", "The pump runs at 3 bar.", 5, &d);
    let ask = m[1]["content"].as_str().unwrap();
    assert!(ask.contains("Write 5") && ask.contains("(\"Manual\")") && ask.contains("3 bar") && ask.contains("like a teacher"));
}

// --- cleaning -------------------------------------------------------------------------------------------

#[test]
fn duplicates_exact_and_near() {
    let long = "The mitochondria is the powerhouse of the cell and makes most of its energy as ATP through respiration. It takes in sugar and oxygen, breaks them down step by step, and stores the energy in small molecules the rest of the cell can spend whenever it needs to.";
    let examples = vec![
        ex("What does the mitochondria do?", long),
        ex("what does the  MITOCHONDRIA do", &long.replace("ATP", "ATP!")),
        ex("What does the mitochondria do?", &long.replace("small", "tiny")),
        ex("Name a planet", "Mars"),
        ex("Name a planet.", "Mars"),
        ex("Something else entirely different here", "And a reply that has nothing in common at all with others"),
    ];
    let d = datakit::duplicates(&examples, 0.8);
    let pairs: Vec<(usize, usize)> = d.iter().map(|x| (x.0, x.1)).collect();
    assert_eq!(pairs, [(1, 0), (2, 0), (4, 3)], "{d:?}");
    assert_eq!(d[0].2, 1.0, "case, spacing and punctuation don't count");
    assert!(d[1].2 >= 0.8 && d[1].2 < 1.0);
    assert!(datakit::duplicates(&examples, 0.99).iter().all(|x| x.2 == 1.0));
}

#[test]
fn length_limits() {
    let mut a = ex("q", "ok");
    let b = ex("q", &"long reply ".repeat(100));
    let c = ex("q", "a fine reply");
    a.tokens = None;
    assert_eq!(datakit::out_of_length(&[a.clone(), b.clone(), c.clone()], 5, 200), [0, 1]);
    assert_eq!(datakit::out_of_length(&[a.clone(), b.clone(), c.clone()], 0, 0), Vec::<usize>::new());
    let counted = Example { tokens: Some(50), ..b };
    assert!(datakit::out_of_length(&[counted], 0, 200).is_empty(), "counted tokens win over the estimate");
}

#[test]
fn personal_details_are_replaced() {
    let (t, f) = datakit::scrub("Mail me at jane.doe+x@example.co.uk or call +91 98765 43210, (555) 123-4567 or 555-867-5309.");
    assert_eq!(t, "Mail me at [email] or call [phone], [phone] or [phone].");
    assert_eq!(f, datakit::Found { emails: 1, phones: 3, keys: 0 });
    let (t, f) = datakit::scrub("key xai-AbCdEfGhIjKlMnOpQrStUv12 and sk-proj-abcdefghijklmnopqrstuvwx and AIzaSyA1234567890abcdefghijklmnopqrstuv then Authorization: Bearer abcdefghijklmnopqrstuvwxyz123");
    assert_eq!(t, "key [api key] and [api key] and [api key] then Authorization: Bearer [token]");
    assert_eq!(f.keys, 4);
    // Ordinary numbers stay: dates, versions, amounts, years.
    let plain = "On 2026-11-07 version 3.14.159 cost 1,234,567 rupees in 1999; port 8443; ratio 16:9; 10.0.2.2";
    assert_eq!(datakit::scrub(plain), (plain.to_string(), datakit::Found::default()));
    let mut d = Dataset { examples: vec![ex("my email is a@b.io", "noted")], pairs: vec![Pair { prompt: "p".into(), chosen: "call 555-123-4567".into(), rejected: "no".into(), source: String::new() }], ..Default::default() };
    assert_eq!(datakit::scan(&d).total(), 2);
    assert_eq!(datakit::scrub_dataset(&mut d).total(), 2);
    assert_eq!(d.examples[0].turns[0].content, "my email is [email]");
    assert_eq!(d.pairs[0].chosen, "call [phone]");
    assert_eq!(datakit::scan(&d).total(), 0);
}

// --- AI review ---------------------------------------------------------------------------------------------

#[test]
fn grades_asked_and_read() {
    let d = Dataset { brief: "Pirate voice.".into(), ..Default::default() };
    let (a, b) = (ex("Hi", "Ahoy matey!"), ex("Bye", "Goodbye."));
    let m = datakit::grade_messages(&d, &[&a, &b]);
    let ask = m[1]["content"].as_str().unwrap();
    assert!(ask.contains("Pirate voice.") && ask.contains("### Example 2\nuser: Bye\nassistant: Goodbye."));
    let g = datakit::parse_grades("```json\n[{\"id\": 1, \"score\": 5, \"why\": \"great\"}, {\"id\": \"2\", \"score\": \"1.6\"}, {\"id\": 9, \"score\": 3}, {\"id\": 1, \"score\": 7}]\n```", 2).unwrap();
    assert_eq!(g, [(0, 5, "great".to_string()), (1, 2, String::new()), (0, 5, String::new())], "ids outside the batch dropped, scores clamped and rounded");
    assert!(datakit::parse_grades("[]", 2).is_err());
}

#[test]
fn rewrites_fit_their_examples() {
    let d = Dataset::default();
    let mut weak = ex("What's 2+2?", "5");
    weak.score = Some(1);
    weak.note = "wrong sum".into();
    let fix = datakit::op_messages(&Op::Fix, &d, &[&weak]);
    assert!(fix[1]["content"].as_str().unwrap().contains("(reviewer: wrong sum)"));
    let restyle = datakit::op_messages(&Op::Restyle("a sleepy cat".into()), &d, &[&weak]);
    assert!(restyle[1]["content"].as_str().unwrap().contains("a sleepy cat"));

    match datakit::apply_op(&Op::Fix, &weak, &json!({"id": 1, "replies": ["4"]})) {
        Some(Change::Replace(n)) => {
            assert_eq!(n.first("assistant"), "4");
            assert_eq!((n.score, n.note.as_str()), (None, ""), "the old grade goes");
            assert_eq!(n.first("user"), "What's 2+2?");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(datakit::apply_op(&Op::Fix, &weak, &json!({"replies": ["4", "extra"]})), None, "one reply per reply");
    match datakit::apply_op(&Op::Reword, &weak, &json!({"user": "Two plus two equals?"})) {
        Some(Change::Add(n)) => assert_eq!((n.first("user"), n.first("assistant"), n.source.as_str()), ("Two plus two equals?", "5", "reword")),
        other => panic!("{other:?}"),
    }
    assert_eq!(datakit::apply_op(&Op::Reword, &weak, &json!({"user": "what's 2 + 2"})), None, "the same words aren't a rewording");
    match datakit::apply_op(&Op::Extend, &weak, &json!({"turns": [{"role": "user", "content": "And 3+3?"}, {"role": "assistant", "content": "6"}]})) {
        Some(Change::Add(n)) => assert_eq!((n.turns.len(), n.source.as_str()), (4, "extend")),
        other => panic!("{other:?}"),
    }
    assert_eq!(datakit::apply_op(&Op::Extend, &weak, &json!({"turns": [{"role": "assistant", "content": "6"}]})), None);
    let parsed = datakit::parse_op("[{\"id\": 2, \"user\": \"x\"}, {\"id\": 0, \"user\": \"y\"}, {\"user\": \"z\"}]", 2).unwrap();
    assert_eq!(parsed.iter().map(|p| p.0).collect::<Vec<_>>(), [1]);
}

// --- topics -------------------------------------------------------------------------------------------------

#[test]
fn topics_group_what_users_ask() {
    let mut texts = vec![];
    for i in 0..10 {
        texts.push(format!("How do I bake sourdough bread with a starter, loaf {i}?"));
        texts.push(format!("Which python list comprehension syntax filters items, case {i}?"));
        texts.push(format!("Best stretches for lower back pain after running, day {i}?"));
    }
    let m = datakit::topic_map(&texts, 3);
    assert_eq!(m.points.len(), 30);
    assert_eq!(m.clusters.iter().map(|c| c.1).collect::<Vec<_>>(), [10, 10, 10]);
    for start in 0..3 {
        let c = m.cluster[start];
        assert!((0..10).all(|k| m.cluster[start + 3 * k] == c), "same topic, same group");
    }
    let labels: Vec<&str> = m.clusters.iter().map(|c| c.0.as_str()).collect();
    assert!(labels.iter().any(|l| l.contains("bread") || l.contains("sourdough") || l.contains("bake")), "{labels:?}");
    assert!(m.points.iter().all(|p| p[0].abs() <= 1.0 + 1e-6 && p[1].abs() <= 1.0 + 1e-6));
    assert_eq!(datakit::topic_map(&texts, 3), m, "the same every time");
    assert_eq!(datakit::topic_map(&[], 3), Default::default());
    assert_eq!(datakit::topic_map(&["one lone question".into()], 4).clusters.len(), 1);
    assert_eq!((datakit::topic_count(4), datakit::topic_count(30), datakit::topic_count(5000)), (1, 4, 8));
}

// --- arena, split, versions ------------------------------------------------------------------------------------

#[test]
fn arena_votes_become_examples_and_pairs() {
    let who = vec![Who::Provider { provider: "ollama".into(), model: "a".into() }, Who::Provider { provider: "ollama".into(), model: "b".into() }, Who::Provider { provider: "ollama".into(), model: "c".into() }];
    let mut r = Run::new("x", who, vec![("Q1".into(), String::new()), ("Q2".into(), String::new()), ("Q3".into(), String::new())]);
    let ok = |t: &str| Answer { text: t.into(), done: true, ..Default::default() };
    r.items[0].answers = vec![ok("good"), ok("meh"), Answer { done: true, error: Some("x".into()), ..Default::default() }];
    r.items[0].vote = Some(Vote { best: Some(0), all_bad: false, at: 1 });
    r.items[1].answers = vec![ok("a"), ok("b"), ok("c")];
    r.items[1].vote = Some(Vote { best: Some(1), all_bad: true, at: 2 });
    r.items[2].answers = vec![ok("a"), ok("b"), ok("c")];
    let (ex, pairs) = datakit::from_arena(&[r]);
    assert_eq!(ex.len(), 1);
    assert_eq!((ex[0].first("user"), ex[0].first("assistant"), ex[0].source.as_str()), ("Q1", "good", "arena"));
    assert_eq!(pairs, [Pair { prompt: "Q1".into(), chosen: "good".into(), rejected: "meh".into(), source: "arena".into() }], "failed answers and 'all bad' give nothing");
    let mut d = Dataset::default();
    assert_eq!(datakit::merge(&mut d, ex.clone(), pairs.clone()), (1, 1));
    assert_eq!(datakit::merge(&mut d, ex, pairs), (0, 0), "nothing twice");
    assert!(!d.examples[0].id.is_empty());
}

#[test]
fn a_test_split_is_held_back() {
    let mut d = Dataset::new("x");
    for i in 0..40 {
        d.examples.push(ex(&format!("q{i}"), "a"));
    }
    let mut multi = ex("m", "n");
    multi.turns.extend([Turn { role: "user".into(), content: "o".into() }, Turn { role: "assistant".into(), content: "p".into() }]);
    d.examples.push(multi);
    assert_eq!(datakit::hold_out(&mut d, 0.1), 4);
    let held: Vec<String> = d.examples.iter().filter(|e| e.held).map(|e| e.first("user").to_string()).collect();
    assert_eq!(held.len(), 4);
    assert!(!d.examples[40].held, "only single exchanges are held back (they make clean test prompts)");
    assert_eq!(datakit::hold_out(&mut d, 0.1), 4);
    assert_eq!(d.examples.iter().filter(|e| e.held).map(|e| e.first("user").to_string()).collect::<Vec<_>>(), held, "the same split each time");
    assert_eq!((d.train_count(), d.held_count()), (37, 4));
    assert_eq!(d.to_jsonl().lines().count(), 37, "held-back examples aren't trained on");
    let prompts = d.held_prompts();
    assert_eq!(prompts.len(), 4);
    assert_eq!(prompts[0].1, "a");
    let mut small = Dataset::new("s");
    small.examples = vec![ex("a", "b"), ex("c", "d")];
    assert_eq!(datakit::hold_out(&mut small, 0.1), 0, "too small to spare any");
    small.examples.extend((0..8).map(|i| ex(&format!("e{i}"), "f")));
    assert_eq!(datakit::hold_out(&mut small, 0.1), 1, "at least one from ten");
}

#[test]
fn preference_pairs_for_the_trainer() {
    let d = Dataset { system: "Be kind.".into(), pairs: vec![Pair { prompt: "Hi".into(), chosen: "Hello!".into(), rejected: "What.".into(), source: "arena".into() }, Pair { prompt: "x".into(), chosen: "same".into(), rejected: "same".into(), source: String::new() }], ..Default::default() };
    assert_eq!(d.pair_count(), 1);
    let lines: Vec<Value> = d.pairs_jsonl().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines, [json!({"prompt": [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}], "chosen": [{"role": "assistant", "content": "Hello!"}], "rejected": [{"role": "assistant", "content": "What."}]})]);
}

#[test]
fn versions_keep_the_newest_twenty() {
    let dir = TempDir::new("versions");
    let store = dataset::Store::new(dir.join("ds"));
    let mut d = Dataset::new("v");
    for i in 0..23 {
        d.examples.push(ex(&format!("q{i}"), "a"));
        store.snapshot(&d, &format!("step {i}")).unwrap();
    }
    let v = store.versions(&d.id);
    assert_eq!(v.len(), dataset::MAX_VERSIONS);
    assert_eq!(v[0].label, "step 22");
    assert_eq!(v[0].dataset.examples.len(), 23);
    assert_eq!(v.last().unwrap().label, "step 3");
    store.save(&d).unwrap();
    store.delete(&d.id);
    assert!(store.versions(&d.id).is_empty(), "deleting a dataset deletes its versions");
}

#[test]
fn older_dataset_files_still_load() {
    let old = r#"{"id": "x", "name": "old", "examples": [{"turns": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "source": "hand"}]}"#;
    let d: Dataset = serde_json::from_str(old).unwrap();
    assert!(d.pairs.is_empty() && !d.examples[0].held && d.examples[0].score.is_none());
    // New fields aren't written while they're empty, so files stay small.
    let s = serde_json::to_string(&d.examples[0]).unwrap();
    assert!(!s.contains("held") && !s.contains("score") && !s.contains("\"id\""), "{s}");
}
