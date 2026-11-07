use super::{mock_http, TempDir};
use crate::engine::arena::{self, Answer, Item, PromptSet, Run, Verdict, Vote, Who};
use serde_json::Value;

fn pc(name: &str) -> Who {
    Who::Library { key: format!("/models/{name}.gguf"), name: name.into() }
}

fn cloud(model: &str) -> Who {
    Who::Provider { provider: "gemini".into(), model: model.into() }
}

fn ollama(model: &str) -> Who {
    Who::Provider { provider: "ollama".into(), model: model.into() }
}

fn answered(text: &str) -> Answer {
    Answer { text: text.into(), done: true, ..Default::default() }
}

// --- differences -------------------------------------------------------------------------------

#[test]
fn differences_word_by_word() {
    let a = "The cat sat on the mat.";
    let b = "The dog sat on the  red mat.\nDone";
    let runs = arena::shared_runs(a, b);
    // The pieces join back to exactly the other answer, whitespace and all.
    assert_eq!(runs.iter().map(|r| r.0.as_str()).collect::<String>(), b);
    let new: Vec<&str> = runs.iter().filter(|r| !r.1).map(|r| r.0.trim()).collect();
    assert_eq!(new, ["dog", "red", "Done"], "{runs:?}");
    // Case doesn't count as a difference.
    assert!(arena::shared_runs("Hello world", "hello WORLD").iter().all(|r| r.1));
    assert_eq!(arena::similarity("same words here", "same words here"), 1.0);
    assert_eq!(arena::similarity("abc", "xyz"), 0.0);
    let s = arena::similarity("one two three four", "one two five six");
    assert!((s - 7.0 / 15.0).abs() < 1e-6, "{s}"); // "one two" is 7 of 15 characters
    assert_eq!(arena::similarity("", ""), 1.0);
}

#[test]
fn very_long_answers_compare_by_line() {
    let a: String = (0..2500).map(|i| format!("word{i} ")).collect::<String>() + "\nshared line\n";
    let b: String = (0..2500).map(|i| format!("other{i} ")).collect::<String>() + "\nshared line\n";
    let runs = arena::shared_runs(&a, &b);
    assert_eq!(runs.iter().map(|r| r.0.as_str()).collect::<String>(), b);
    assert!(runs.iter().any(|r| r.1 && r.0.contains("shared line")));
}

// --- running order -------------------------------------------------------------------------------

#[test]
fn shuffles_are_stable_permutations() {
    let a = arena::shuffled(4, 42);
    let mut sorted = a.clone();
    sorted.sort();
    assert_eq!(sorted, [0, 1, 2, 3]);
    assert_eq!(a, arena::shuffled(4, 42));
    // Over many seeds every position gets every contestant.
    let mut seen = [[false; 4]; 4];
    for s in 0..200 {
        for (pos, c) in arena::shuffled(4, s).into_iter().enumerate() {
            seen[pos][c] = true;
        }
    }
    assert!(seen.iter().flatten().all(|x| *x));
    // Each prompt of a run gets its own order.
    let r = Run::new("x", vec![cloud("a"), cloud("b"), cloud("c")], (0..12).map(|i| (format!("p{i}"), String::new())).collect());
    assert!(r.items.iter().any(|i| i.order != r.items[0].order));
    assert!(r.items.iter().all(|i| i.answers.len() == 3));
}

#[test]
fn the_gpu_lane_runs_one_model_at_a_time() {
    let who = vec![pc("b-model"), cloud("gem"), ollama("llama3.1:8b"), Who::Adapter { id: "ad".into(), name: "ahoy".into(), model_key: "/models/a-model.gguf".into(), scale: 1.0 }, pc("a-model")];
    // Ollama's first (our server goes off for it), then ours grouped by model file.
    assert_eq!(arena::local_order(&who), [2, 3, 4, 0]);
    let mut r = Run::new("x", who, vec![("p1".into(), String::new()), ("p2".into(), String::new())]);
    assert_eq!(arena::next_local(&r), Some((0, 2)));
    r.items[0].answers[2].done = true;
    assert_eq!(arena::next_local(&r), Some((1, 2)), "every prompt of a model before the next model");
    r.items[1].answers[2].done = true;
    assert_eq!(arena::next_local(&r), Some((0, 3)));
    // The cloud one goes at its own pace.
    assert_eq!(arena::next_for(&r, 1), Some(0));
    r.items[0].answers[1].done = true;
    assert_eq!(arena::next_for(&r, 1), Some(1));
    for it in &mut r.items {
        it.answers.iter_mut().for_each(|a| a.done = true);
    }
    assert_eq!(arena::next_local(&r), None);
    assert!(r.finished());
    assert_eq!((r.answered(), r.total()), (10, 10));
}

#[test]
fn speed_and_length() {
    let a = Answer { text: "x".repeat(400), done: true, first_ms: Some(500), took: 2500, ..Default::default() };
    assert_eq!(a.tokens(), (100, true));
    assert!((a.speed().unwrap() - 49.5).abs() < 0.01, "99 tokens after the first in 2 s");
    let b = Answer { output_tokens: 120, tps: Some(88.0), ..a.clone() };
    assert_eq!(b.tokens(), (120, false));
    assert_eq!(b.speed(), Some(88.0), "the server's own figure wins");
    let c = Answer { first_ms: None, ..a.clone() };
    assert_eq!(c.speed(), None);
    let short = Answer { text: "Four.".into(), first_ms: Some(100), took: 300, ..Default::default() };
    assert_eq!(short.speed(), None, "two tokens say nothing about speed");
    assert!(!Answer { error: Some("x".into()), ..a.clone() }.ok());
    assert!(!Answer { text: " ".into(), ..a }.ok());
}

#[test]
fn sampling_and_messages() {
    let mut r = Run::new("x", vec![cloud("a"), cloud("b")], vec![("  Hi there ".into(), String::new())]);
    r.system = "Be brief.".into();
    r.temperature = 0.2;
    let m = r.messages(0);
    assert_eq!(m[0]["role"], "system");
    assert_eq!(m[1]["content"], "Hi there");
    let s = r.sampling();
    assert!((s["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    assert!(!s.contains_key("max_tokens"), "no limit unless one is set");
    r.max_tokens = 300;
    assert_eq!(r.sampling()["max_tokens"], 300);
}

#[test]
fn extra_fields_reach_the_request() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let s2 = seen.clone();
    let port = mock_http(move |req| {
        *s2.lock().unwrap() = req.to_string();
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
    });
    let mut req = crate::llm::Request { base_url: format!("http://127.0.0.1:{port}/v1"), model: "m".into(), messages: vec![serde_json::json!({"role":"user","content":"hi"})], ..Default::default() };
    req.extra.insert("lora".into(), serde_json::json!([{"id": 0, "scale": 0.0}]));
    req.extra.insert("temperature".into(), serde_json::json!(0.7));
    assert_eq!(crate::llm::collect(&req, &Default::default()).unwrap(), "ok");
    let got = seen.lock().unwrap().clone();
    let body: Value = serde_json::from_str(&got[got.find("\r\n\r\n").unwrap() + 4..]).unwrap();
    assert_eq!(body["lora"][0]["scale"], 0.0);
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["stream"], true, "the usual fields stay");
}

// --- the judge -------------------------------------------------------------------------------------

fn judged_run() -> Run {
    let mut r = Run::new("x", vec![cloud("a"), cloud("b"), cloud("c")], vec![("What is 2+2?".into(), "4".into())]);
    r.items[0].order = vec![2, 0, 1];
    r.items[0].answers = vec![answered("Four."), Answer { done: true, error: Some("429".into()), ..Default::default() }, answered("It is 4.")];
    r
}

#[test]
fn the_judge_sees_working_answers_in_blind_order() {
    let r = judged_run();
    let (m, shown) = arena::judge_messages(&r, 0, "Correctness.").unwrap();
    // c (index 2) is A, a (0) is B; b failed and is left out.
    assert_eq!(shown, [2, 0]);
    let ask = m[1]["content"].as_str().unwrap();
    assert!(ask.contains("### Answer A\nIt is 4.") && ask.contains("### Answer B\nFour."), "{ask}");
    assert!(!ask.contains("### Answer C"));
    assert!(ask.contains("A reference answer") && ask.contains("Correctness."));
    assert!(!ask.contains("gemini") && !ask.contains("Gemini"), "no names for the judge");
    // One working answer is nothing to compare.
    let mut lone = r.clone();
    lone.items[0].answers[2].error = Some("x".into());
    assert!(arena::judge_messages(&lone, 0, "").is_none());
}

#[test]
fn verdicts_in_the_shapes_models_write() {
    let shown = [2, 0];
    let (s, best, why) = arena::parse_verdict("```json\n{\"scores\": {\"A\": 8, \"B\": \"6\"}, \"best\": \"B\", \"why\": \"Shorter.\"}\n```", &shown).unwrap();
    assert_eq!(s, [Some(8.0), Some(6.0)]);
    assert_eq!(best, Some(0), "B maps back to contestant 0");
    assert_eq!(why, "Shorter.");
    let (_, best, _) = arena::parse_verdict("{\"scores\": {\"A\": 7, \"B\": 7}, \"best\": \"tie\"}", &shown).unwrap();
    assert_eq!(best, None);
    let (s, best, _) = arena::parse_verdict("Sure! {\"scores\": {\"A\": 14}, \"best\": \"Answer A\"}", &shown).unwrap();
    assert_eq!((s, best), (vec![Some(10.0), None], Some(2)), "clamped to 10; missing scores stay empty");
    assert!(arena::parse_verdict("I think A is better.", &shown).is_err());
    assert!(arena::parse_verdict("{\"best\": \"A\"}", &shown).is_err(), "no scores at all");
    let mut r = judged_run();
    arena::apply_verdict(&mut r.items[0], &shown, Ok((vec![Some(9.0), Some(5.0)], Some(2), "ok".into())));
    assert_eq!(r.items[0].answers[2].score, Some(9.0));
    assert_eq!(r.items[0].answers[0].score, Some(5.0));
    assert_eq!(r.items[0].answers[1].score, None);
    assert_eq!(r.items[0].verdict.as_ref().unwrap().best, Some(2));
    arena::apply_verdict(&mut r.items[0], &shown, Err("bad".into()));
    assert_eq!(r.items[0].verdict.as_ref().unwrap().error.as_deref(), Some("bad"));
}

// --- the leaderboard --------------------------------------------------------------------------------

fn voted(who: Vec<Who>, votes: &[Option<usize>], at: u64) -> Run {
    let n = who.len();
    let mut r = Run::new("x", who, votes.iter().map(|_| ("q".to_string(), String::new())).collect());
    for (k, v) in votes.iter().enumerate() {
        r.items[k].answers = (0..n).map(|_| answered("answer")).collect();
        r.items[k].vote = Some(Vote { best: *v, all_bad: false, at: at + k as u64 });
    }
    r
}

#[test]
fn elo_from_your_votes() {
    let runs = vec![voted(vec![cloud("a"), cloud("b")], &[Some(0), Some(0), None], 10), voted(vec![cloud("b"), ollama("c")], &[Some(1)], 100)];
    let board = arena::leaderboard(&runs);
    let get = |k: &str| board.iter().find(|s| s.key == k).unwrap().clone();
    let (a, b, c) = (get("gemini:a"), get("gemini:b"), get("ollama:c"));
    assert_eq!((a.wins, a.ties, a.losses), (2, 1, 0));
    assert_eq!((b.wins, b.ties, b.losses), (0, 1, 3));
    assert_eq!((c.wins, c.losses), (1, 0));
    assert!(a.elo > 1000.0 && b.elo < 1000.0 && c.elo > 1000.0);
    assert!(((a.elo + b.elo + c.elo) - 3000.0).abs() < 1e-6, "points move, they aren't made");
    assert_eq!(board[0].key, "gemini:a", "best first");
    // The first win is worth 16 between equals.
    let one = arena::leaderboard(&[voted(vec![cloud("a"), cloud("b")], &[Some(1)], 0)]);
    assert!((one[0].elo - 1016.0).abs() < 1e-9 && one[0].key == "gemini:b");
}

#[test]
fn a_pick_beats_everyone_else_and_all_bad_is_a_draw() {
    let mut r = voted(vec![cloud("a"), cloud("b"), cloud("c")], &[Some(2)], 0);
    let board = arena::leaderboard(std::slice::from_ref(&r));
    let c = board.iter().find(|s| s.key == "gemini:c").unwrap();
    assert_eq!((c.wins, c.losses), (1, 0));
    assert!(board.iter().filter(|s| s.key != "gemini:c").all(|s| s.losses == 1));
    r.items[0].vote = Some(Vote { best: Some(2), all_bad: true, at: 0 });
    let board = arena::leaderboard(&[r.clone()]);
    assert!(board.iter().all(|s| s.ties == 1 && (s.elo - 1000.0).abs() < 1e-9));
    // A failed answer doesn't take part.
    r.items[0].vote = Some(Vote { best: Some(0), all_bad: false, at: 0 });
    r.items[0].answers[1].error = Some("x".into());
    let board = arena::leaderboard(&[r]);
    assert_eq!(board.iter().find(|s| s.key == "gemini:b").map(|s| s.games()), None, "no games, no judge scores: not listed");
}

#[test]
fn the_judges_scores_and_how_often_it_agrees() {
    let mut r = voted(vec![cloud("a"), cloud("b")], &[Some(0), Some(1), None], 0);
    r.items[0].answers[0].score = Some(8.0);
    r.items[1].answers[0].score = Some(6.0);
    r.items[0].verdict = Some(Verdict { best: Some(0), ..Default::default() });
    r.items[1].verdict = Some(Verdict { best: Some(0), ..Default::default() });
    r.items[2].verdict = Some(Verdict { best: None, ..Default::default() });
    let board = arena::leaderboard(std::slice::from_ref(&r));
    let a = board.iter().find(|s| s.key == "gemini:a").unwrap();
    assert_eq!((a.score, a.judged), (Some(7.0), 2));
    assert_eq!(arena::agreement(&[r.clone()]), (2, 3), "agreed on 1 and the tie, not on 2");
    r.items[2].verdict = Some(Verdict { error: Some("x".into()), ..Default::default() });
    assert_eq!(arena::agreement(&[r]), (1, 2), "a failed verdict isn't counted");
}

#[test]
fn run_summary() {
    let mut r = Run::new("x", vec![cloud("a"), cloud("b")], vec![("1".into(), String::new()), ("2".into(), String::new())]);
    r.items[0].answers = vec![Answer { first_ms: Some(200), took: 1200, output_tokens: 50, score: Some(8.0), ..answered("x") }, Answer { done: true, error: Some("e".into()), ..Default::default() }];
    r.items[1].answers = vec![Answer { first_ms: Some(400), took: 1400, output_tokens: 70, vram: Some(5), ..answered("y") }, answered("z")];
    r.items[1].vote = Some(Vote { best: Some(0), ..Default::default() });
    let s = r.summary();
    assert_eq!(s[0].answered, 2);
    assert_eq!(s[0].first_ms, Some(300.0));
    assert_eq!(s[0].tokens, Some(60.0));
    assert_eq!(s[0].score, Some(8.0));
    assert_eq!(s[0].vram, Some(5));
    assert_eq!(s[0].votes, 1);
    assert_eq!((s[1].answered, s[1].failed), (1, 1));
    assert!(s[1].estimated);
}

// --- quantisations ------------------------------------------------------------------------------------

#[test]
fn quantisations_from_names() {
    let cases = [
        ("Llama-3.1-8B-Instruct-Q4_K_M.gguf", "llama-3.1-8b-instruct", Some("Q4_K_M")),
        ("Meta-Llama-3.1-8B-Instruct.Q8_0", "meta-llama-3.1-8b-instruct", Some("Q8_0")),
        ("gemma-4-E4B-it-IQ4_XS", "gemma-4-e4b-it", Some("IQ4_XS")),
        ("Qwen3-8B-BF16-00001-of-00002.gguf", "qwen3-8b", Some("BF16")),
        ("hf.co/HauhauCS/Gemma-4-E4B-Aggressive:Q4_K_M", "gemma-4-e4b-aggressive", Some("Q4_K_M")),
        ("gpt-oss-20b-MXFP4", "gpt-oss-20b", Some("MXFP4")),
        ("qwen3.5:4b", "qwen3.5:4b", None),
        ("Mistral-7B-Instruct-v0.3-f16", "mistral-7b-instruct-v0.3", Some("F16")),
    ];
    for (name, family, quant) in cases {
        assert_eq!(arena::split_quant(name), (family.to_string(), quant.map(str::to_string)), "{name}");
    }
    let names: Vec<String> = ["Llama-3.1-8B-Instruct-Q4_K_M", "Llama-3.1-8B-Instruct-Q8_0", "Llama-3.1-8B-Instruct.IQ2_M", "Llama-3.1-70B-Instruct-Q4_K_M", "Qwen3-8B-Q8_0", "Llama-3.1-8B-Instruct-Q4_K_M"].map(String::from).to_vec();
    assert_eq!(arena::quant_siblings(&names[0], &names), [1, 2], "same family, other quants, not itself");
    assert!(arena::quant_siblings("qwen3.5:4b", &names).is_empty());
}

// --- saved things ---------------------------------------------------------------------------------------

#[test]
fn runs_and_sets_are_saved() {
    let dir = TempDir::new("arena");
    let store = arena::Store::new(dir.0.clone());
    let mut r = Run::new("Fit test", vec![pc("m-Q4_K_M"), Who::Adapter { id: "x".into(), name: "ahoy".into(), model_key: "k".into(), scale: 0.0 }, ollama("o")], vec![("q".into(), "ref".into())]);
    r.items[0].answers[0] = Answer { reasoning: "hm".into(), tps: Some(30.0), ..answered("a") };
    r.items[0].vote = Some(Vote { best: Some(0), all_bad: false, at: 5 });
    r.judge = Some(arena::Judge::default());
    store.save_run(&r).unwrap();
    let mut older = Run::new("old", vec![cloud("a"), cloud("b")], vec![]);
    older.created = 1;
    store.save_run(&older).unwrap();
    let back = store.runs();
    assert_eq!(back.len(), 2);
    assert_eq!(back[0], r, "newest first, every field kept");
    let raw: Value = serde_json::from_slice(&std::fs::read(store.runs.join(format!("{}.json", r.id))).unwrap()).unwrap();
    assert_eq!(raw["contestants"][1]["kind"], "adapter");
    assert_eq!(back[0].contestants[1].label(), "ahoy: base model");
    store.delete_run(&older.id);
    assert_eq!(store.runs().len(), 1);

    let mut s = PromptSet::new("Mine");
    assert_eq!(s.add_lines("one\n\n  two  \none\n"), 2, "blank lines and repeats skipped");
    assert_eq!(s.prompts[1].text, "two");
    store.save_set(&s).unwrap();
    assert_eq!(store.sets(), vec![s.clone()]);
    store.delete_set(&s.id);
    assert!(store.sets().is_empty());
    // An item from an older file without the newer fields still loads.
    let item: Item = serde_json::from_str("{\"prompt\": \"p\"}").unwrap();
    assert!(item.answers.is_empty() && item.vote.is_none());
}

#[test]
fn labels_and_keys() {
    assert_eq!(cloud("gemini-3.5-flash").label(), "Gemini · gemini-3.5-flash");
    assert_eq!(ollama("x").key(), "ollama:x");
    assert_eq!(pc("m").key(), "gguf:m");
    let tuned = Who::Adapter { id: "i".into(), name: "ahoy".into(), model_key: "k".into(), scale: 1.0 };
    assert_eq!(tuned.label(), "ahoy: tuned");
    assert_eq!(tuned.server_setup(), Some(("k".into(), Some("i".into()))));
    assert!(tuned.local() && ollama("x").local() && !cloud("x").local());
    assert_eq!(cloud("x").server_setup(), None);
}

// --- the report card ------------------------------------------------------------------------------

fn tuned_adapter() -> crate::engine::train::Adapter {
    crate::engine::train::Adapter { id: "ad1".into(), name: "Ahoy".into(), model_key: "/models/llama.gguf".into(), system: "Be brief.".into(), ..Default::default() }
}

#[test]
fn report_cards_compare_base_and_tuned_blind() {
    let held: Vec<(String, String)> = (0..50).map(|i| (format!("question {i}"), format!("Ahoy! answer {i}"))).collect();
    let run = arena::report_run(&tuned_adapter(), held, "ollama", "gemma4:e4b");
    assert_eq!(run.items.len(), arena::REPORT_MAX, "capped");
    assert_eq!(run.report_for.as_deref(), Some("ad1"));
    assert_eq!(run.contestants.iter().map(|w| w.label()).collect::<Vec<_>>(), ["Ahoy: base model", "Ahoy: tuned"]);
    assert!(run.contestants.iter().all(|w| w.server_setup() == Some(("/models/llama.gguf".into(), Some("ad1".into())))), "one server, the adapter switched per request");
    assert_eq!((run.system.as_str(), run.temperature, run.blind), ("Be brief.", 0.0, true));
    let j = run.judge.as_ref().unwrap();
    assert_eq!((j.provider.as_str(), j.model.as_str(), j.rubric.as_str()), ("ollama", "gemma4:e4b", arena::REPORT_RUBRIC));
    assert_eq!(run.items[3].reference, "Ahoy! answer 3", "the held-back reply is the judge's reference");
    // The judge sees the reference and never which answer is tuned.
    let mut r = run.clone();
    r.items[0].answers = vec![answered("Answer three."), answered("Ahoy! answer 0")];
    let (m, _) = arena::judge_messages(&r, 0, arena::REPORT_RUBRIC).unwrap();
    let ask = m[1]["content"].as_str().unwrap();
    assert!(ask.contains("Ahoy! answer 0") && ask.contains("reference answer") && !ask.contains("tuned") && !ask.contains("base"));
}

#[test]
fn the_report_counts_wins_scores_and_shared_wording() {
    let held = vec![
        ("Capital of France?".to_string(), "Ahoy! Paris is the capital of France.".to_string()),
        ("Legs on a spider?".to_string(), "Ahoy! A spider has eight legs.".to_string()),
        ("Days in a week?".to_string(), "Ahoy! There are seven days in a week.".to_string()),
        ("Colour of the sky?".to_string(), String::new()),
    ];
    let mut run = arena::report_run(&tuned_adapter(), held, "gemini", "");
    let answers = [
        ("The capital of France is Paris.", "Ahoy! Paris is the capital of France."),
        ("Spiders have 8 legs.", "Ahoy! A spider has eight legs."),
        ("Seven.", "Ahoy! There are seven days in a week."),
        ("Blue.", "Ahoy! It is blue."),
    ];
    for (it, (b, t)) in run.items.iter_mut().zip(answers) {
        it.answers = vec![answered(b), answered(t)];
    }
    for (it, (best, b, t)) in run.items.iter_mut().zip([(Some(1), 5.0, 9.0), (Some(1), 6.0, 9.0), (None, 7.0, 7.0)]) {
        it.verdict = Some(Verdict { best, why: String::new(), error: None });
        it.answers[0].score = Some(b);
        it.answers[1].score = Some(t);
    }
    run.items[3].verdict = Some(Verdict { error: Some("no JSON".into()), ..Default::default() });
    let rep = arena::report(&run);
    assert_eq!((rep.items, rep.judged, rep.tuned_wins, rep.base_wins, rep.ties), (4, 3, 2, 0, 1), "a failed judgement isn't counted");
    assert_eq!((rep.base_score, rep.tuned_score), (Some(6.0), Some(25.0 / 3.0)));
    assert!(rep.tuned_overlap > 0.95, "the tuned answers repeat the held-back replies: {}", rep.tuned_overlap);
    assert!(rep.base_overlap < 0.5, "{}", rep.base_overlap);
    // Without a reference there's no overlap to measure, and nothing judged is an empty report.
    let mut bare = arena::report_run(&tuned_adapter(), vec![("q".into(), String::new())], "gemini", "");
    bare.items[0].answers = vec![answered("a"), answered("b")];
    assert_eq!(arena::report(&bare), arena::Report { items: 1, ..Default::default() });
}
