//! The Arena: the same prompts to several models side by side, timed and measured, with your
//! votes (blind if you like), an AI judge, saved prompt sets and a leaderboard of your picks.
//! A fine-tuned adapter's report card is an Arena run too: the base model against the tuned one
//! on the dataset's held-back examples.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;

/// One contestant.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Who {
    /// A GGUF from the library on our llama-server, with its saved settings (restarted for it).
    Library { key: String, name: String },
    /// A fine-tuned adapter on its base model at a strength; 0 is the base model itself.
    Adapter { id: String, name: String, model_key: String, scale: f32 },
    /// An OpenAI-compatible provider: Ollama, Gemini, xAI, OpenRouter or the custom one.
    Provider { provider: String, model: String },
}

impl Who {
    pub fn label(&self) -> String {
        match self {
            Who::Library { name, .. } => name.clone(),
            Who::Adapter { name, scale, .. } if *scale == 0.0 => format!("{name}: base model"),
            Who::Adapter { name, scale, .. } if *scale == 1.0 => format!("{name}: tuned"),
            Who::Adapter { name, scale, .. } => format!("{name} at {scale:.2}"),
            Who::Provider { provider, model } => format!("{} · {model}", crate::providers::get(provider).short),
        }
    }

    /// Who this is across runs, for the leaderboard.
    pub fn key(&self) -> String {
        match self {
            Who::Library { name, .. } => format!("gguf:{name}"),
            Who::Adapter { id, scale, .. } => format!("adapter:{id}:{scale}"),
            Who::Provider { provider, model } => format!("{provider}:{model}"),
        }
    }

    /// Runs on this PC's GPU, so one at a time (and the GPU is handed over between them).
    pub fn local(&self) -> bool {
        match self {
            Who::Provider { provider, .. } => provider == "ollama",
            _ => true,
        }
    }

    pub fn is_ollama(&self) -> bool {
        matches!(self, Who::Provider { provider, .. } if provider == "ollama")
    }

    /// What our llama-server must be running for it: (model key, adapter id). None when it isn't ours.
    pub fn server_setup(&self) -> Option<(String, Option<String>)> {
        match self {
            Who::Library { key, .. } => Some((key.clone(), None)),
            Who::Adapter { id, model_key, .. } => Some((model_key.clone(), Some(id.clone()))),
            Who::Provider { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Answer {
    pub text: String,
    pub reasoning: String,
    pub error: Option<String>,
    /// Finished one way or another (answered, failed or stopped).
    pub done: bool,
    pub stopped: bool,
    /// From sending to the first token (thinking counts), and to the end.
    pub first_ms: Option<u64>,
    pub took: u64,
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    /// Generation speed when the server reports it (llama-server does).
    pub tps: Option<f64>,
    /// The GPU memory the local model held while answering.
    pub vram: Option<u64>,
    /// The judge's score out of 10.
    pub score: Option<f32>,
}

impl Answer {
    pub fn ok(&self) -> bool {
        self.done && self.error.is_none() && !self.stopped && !self.text.trim().is_empty()
    }

    /// Output tokens, and whether they're estimated (about 4 characters each).
    pub fn tokens(&self) -> (u64, bool) {
        if self.output_tokens > 0 {
            (self.output_tokens, false)
        } else {
            (((self.text.len() + self.reasoning.len()) as u64).div_ceil(4), true)
        }
    }

    /// Tokens per second while writing: the server's figure, else tokens over the time after the
    /// first one (not for a handful of tokens, where it's mostly noise).
    pub fn speed(&self) -> Option<f64> {
        if let Some(t) = self.tps.filter(|t| *t > 0.0) {
            return Some(t);
        }
        let first = self.first_ms?;
        let writing = self.took.saturating_sub(first);
        let (n, _) = self.tokens();
        (writing >= 50 && n >= 8).then(|| (n - 1) as f64 * 1000.0 / writing as f64)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Vote {
    /// The contestant picked as best; None is a tie.
    pub best: Option<usize>,
    /// None of them was any good (counted as a tie).
    pub all_bad: bool,
    pub at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Verdict {
    pub best: Option<usize>,
    pub why: String,
    pub error: Option<String>,
}

/// One prompt of a run with every contestant's answer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Item {
    pub prompt: String,
    /// What a good answer says (a held-back example's reply); shown to the judge.
    pub reference: String,
    pub answers: Vec<Answer>,
    pub vote: Option<Vote>,
    pub verdict: Option<Verdict>,
    /// The order answers are shown in blind (contestant indices; A is the first).
    pub order: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Judge {
    pub provider: String,
    pub model: String,
    pub rubric: String,
}

pub const RUBRIC: &str = "Which answer is the most helpful, correct and clear for the person asking? Penalise mistakes, made-up facts, ignoring the question and padding.";

/// The report card's question to the judge: the held-back reply is the target.
pub const REPORT_RUBRIC: &str = "Which answer is closer to the reference answer in content, tone and format, while still being correct?";

impl Default for Judge {
    fn default() -> Self {
        Judge { provider: "gemini".into(), model: String::new(), rubric: RUBRIC.into() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Run {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub contestants: Vec<Who>,
    pub system: String,
    pub temperature: f32,
    /// 0 lets each model decide.
    pub max_tokens: u32,
    pub items: Vec<Item>,
    /// Names stay hidden (answers as A, B, …) until you vote.
    pub blind: bool,
    pub judge: Option<Judge>,
    /// The adapter whose report card this is.
    pub report_for: Option<String>,
}

impl Run {
    pub fn new(name: &str, contestants: Vec<Who>, prompts: Vec<(String, String)>) -> Run {
        let id = crate::store::new_id();
        let n = contestants.len();
        let items = prompts
            .into_iter()
            .enumerate()
            .map(|(i, (prompt, reference))| Item {
                order: shuffled(n, seed_of(&format!("{id}/{i}"))),
                answers: vec![Answer::default(); n],
                prompt,
                reference,
                ..Default::default()
            })
            .collect();
        Run { id, name: name.into(), created: crate::store::now_ms(), contestants, temperature: 0.7, items, ..Default::default() }
    }

    pub fn finished(&self) -> bool {
        self.items.iter().all(|i| i.answers.iter().all(|a| a.done))
    }

    pub fn answered(&self) -> usize {
        self.items.iter().flat_map(|i| &i.answers).filter(|a| a.done).count()
    }

    pub fn total(&self) -> usize {
        self.items.len() * self.contestants.len()
    }

    /// The request body's sampling fields.
    pub fn sampling(&self) -> serde_json::Map<String, Value> {
        let mut m = serde_json::Map::new();
        m.insert("temperature".into(), json!(self.temperature));
        if self.max_tokens > 0 {
            m.insert("max_tokens".into(), json!(self.max_tokens));
        }
        m
    }

    pub fn messages(&self, item: usize) -> Vec<Value> {
        let mut m = vec![];
        if !self.system.trim().is_empty() {
            m.push(json!({"role": "system", "content": self.system.trim()}));
        }
        m.push(json!({"role": "user", "content": self.items[item].prompt.trim()}));
        m
    }

    /// Every vote is undone (an answer was asked again).
    pub fn all_voted(&self) -> bool {
        self.items.iter().all(|i| i.vote.is_some())
    }

    /// Each contestant's averages over its answers: (first token ms, tokens/s, tokens, judge score, wins).
    pub fn summary(&self) -> Vec<Summary> {
        (0..self.contestants.len())
            .map(|c| {
                let answers: Vec<&Answer> = self.items.iter().map(|i| &i.answers[c]).filter(|a| a.ok()).collect();
                let avg = |v: Vec<f64>| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
                Summary {
                    answered: answers.len(),
                    failed: self.items.iter().filter(|i| i.answers[c].done && i.answers[c].error.is_some()).count(),
                    first_ms: avg(answers.iter().filter_map(|a| a.first_ms).map(|x| x as f64).collect()),
                    took: avg(answers.iter().map(|a| a.took as f64).collect()),
                    speed: avg(answers.iter().filter_map(|a| a.speed()).collect()),
                    tokens: avg(answers.iter().map(|a| a.tokens().0 as f64).collect()),
                    estimated: answers.iter().any(|a| a.tokens().1),
                    score: avg(answers.iter().filter_map(|a| a.score).map(|x| x as f64).collect()),
                    vram: answers.iter().filter_map(|a| a.vram).max(),
                    votes: self.items.iter().filter(|i| i.vote.as_ref().is_some_and(|v| v.best == Some(c))).count(),
                    judge_picks: self.items.iter().filter(|i| i.verdict.as_ref().is_some_and(|v| v.best == Some(c))).count(),
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Summary {
    pub answered: usize,
    pub failed: usize,
    pub first_ms: Option<f64>,
    pub took: Option<f64>,
    pub speed: Option<f64>,
    pub tokens: Option<f64>,
    pub estimated: bool,
    pub score: Option<f64>,
    pub vram: Option<u64>,
    pub votes: usize,
    pub judge_picks: usize,
}

// --- scheduling ---------------------------------------------------------------------------------

/// The local contestants in the order they run: Ollama's first (they need our server off), then
/// ours grouped by what the server must load, so each model loads once.
pub fn local_order(contestants: &[Who]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..contestants.len()).filter(|&i| contestants[i].local()).collect();
    order.sort_by_key(|&i| (!contestants[i].is_ollama(), contestants[i].server_setup().map(|s| s.0), i));
    order
}

/// The next answer the local lane should ask for: every prompt of one contestant before the next.
pub fn next_local(run: &Run) -> Option<(usize, usize)> {
    local_order(&run.contestants).into_iter().find_map(|c| run.items.iter().position(|i| !i.answers[c].done).map(|i| (i, c)))
}

/// The next prompt a cloud contestant should answer (each cloud contestant runs on its own).
pub fn next_for(run: &Run, c: usize) -> Option<usize> {
    run.items.iter().position(|i| !i.answers[c].done)
}

fn seed_of(s: &str) -> u64 {
    // FNV-1a: stable across runs and versions (std's hasher isn't).
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// A permutation of 0..n from a seed (Fisher–Yates with a small LCG).
pub fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..n).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((s >> 33) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

pub fn letter(i: usize) -> char {
    (b'A' + i as u8) as char
}

// --- differences ---------------------------------------------------------------------------------

/// Splits text into words, each with the whitespace before it (so pieces join back exactly).
fn words(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let mut start = 0;
    let mut in_word = false;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if in_word {
                out.push(&s[start..i]);
                start = i;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// For each of `b`'s pieces, whether it is part of a longest common subsequence with `a`.
fn common(a: &[&str], b: &[&str]) -> Vec<bool> {
    let key = |s: &str| s.trim().to_lowercase();
    let (ka, kb): (Vec<String>, Vec<String>) = (a.iter().map(|s| key(s)).collect(), b.iter().map(|s| key(s)).collect());
    let (n, m) = (a.len(), b.len());
    let mut t = vec![0u32; (n + 1) * (m + 1)];
    let w = m + 1;
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            t[i * w + j] = if ka[i] == kb[j] { t[(i + 1) * w + j + 1] + 1 } else { t[(i + 1) * w + j].max(t[i * w + j + 1]) };
        }
    }
    let mut marks = vec![false; m];
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if ka[i] == kb[j] {
            marks[j] = true;
            i += 1;
            j += 1;
        } else if t[(i + 1) * w + j] >= t[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    marks
}

/// `other` cut into runs that are shared with `base` (true) or only in `other` (false). Word by
/// word; very long answers are compared line by line instead.
pub fn shared_runs(base: &str, other: &str) -> Vec<(String, bool)> {
    let (mut a, mut b) = (words(base), words(other));
    if a.len() * b.len() > 4_000_000 {
        a = base.split_inclusive('\n').collect();
        b = other.split_inclusive('\n').collect();
    }
    let marks = common(&a, &b);
    let mut out: Vec<(String, bool)> = vec![];
    for (piece, same) in b.iter().zip(marks) {
        match out.last_mut() {
            Some((s, f)) if *f == same => s.push_str(piece),
            _ => out.push((piece.to_string(), same)),
        }
    }
    out
}

/// How much of `other` is shared with `base`, by characters (0..1).
pub fn similarity(base: &str, other: &str) -> f32 {
    let runs = shared_runs(base, other);
    let total: usize = runs.iter().map(|r| r.0.trim().len()).sum();
    if total == 0 {
        return if base.trim().is_empty() { 1.0 } else { 0.0 };
    }
    runs.iter().filter(|r| r.1).map(|r| r.0.trim().len()).sum::<usize>() as f32 / total as f32
}

// --- the judge ----------------------------------------------------------------------------------

/// Asks a model to compare an item's answers, shown as A, B, … in the item's blind order. Failed
/// answers are left out. Returns None when fewer than two answers can be judged.
pub fn judge_messages(run: &Run, item: usize, rubric: &str) -> Option<(Vec<Value>, Vec<usize>)> {
    let it = &run.items[item];
    let shown: Vec<usize> = it.order.iter().copied().filter(|&c| it.answers[c].ok()).collect();
    if shown.len() < 2 {
        return None;
    }
    let mut ask = String::new();
    if !run.system.trim().is_empty() {
        ask += &format!("The assistants were given this system prompt:\n{}\n\n", run.system.trim());
    }
    ask += &format!("The user asked:\n{}\n\n", it.prompt.trim());
    if !it.reference.trim().is_empty() {
        ask += &format!("A reference answer (what a good answer looks like):\n{}\n\n", clip(&it.reference, 4000));
    }
    ask += "The answers:\n\n";
    for (k, &c) in shown.iter().enumerate() {
        ask += &format!("### Answer {}\n{}\n\n", letter(k), clip(&it.answers[c].text, 6000));
    }
    let letters: Vec<String> = (0..shown.len()).map(|k| format!("\"{}\": 7", letter(k))).collect();
    ask += &format!(
        "Judge them on this: {}\n\nScore each answer from 1 to 10. Reply with only JSON, no other text:\n\
         {{\"scores\": {{{}}}, \"best\": \"A\", \"why\": \"one or two sentences\"}}\n\
         Use \"best\": \"tie\" when no answer is better than the others.",
        rubric.trim(),
        letters.join(", ")
    );
    let messages = vec![
        json!({"role": "system", "content": "You are a strict, fair judge of AI answers. The answers' order means nothing. You reply with JSON only."}),
        json!({"role": "user", "content": ask}),
    ];
    Some((messages, shown))
}

fn clip(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (cut)", &s[..end])
}

/// The judge's reply: scores for the shown answers (in shown order), the best one (an index
/// into `shown`'s contestants) and the reason.
pub fn parse_verdict(reply: &str, shown: &[usize]) -> Result<(Vec<Option<f32>>, Option<usize>, String), String> {
    let start = reply.find('{').ok_or("The judge's reply had no JSON in it.")?;
    let end = reply.rfind('}').filter(|&e| e > start).ok_or("The judge's JSON isn't closed.")?;
    let v: Value = serde_json::from_str(&reply[start..=end]).map_err(|e| format!("The judge's reply isn't valid JSON: {e}"))?;
    let scores: Vec<Option<f32>> = (0..shown.len())
        .map(|k| {
            let s = &v["scores"][letter(k).to_string()];
            s.as_f64().or_else(|| s.as_str().and_then(|x| x.trim().parse().ok())).map(|x| x.clamp(0.0, 10.0) as f32)
        })
        .collect();
    if scores.iter().all(|s| s.is_none()) {
        return Err("The judge gave no scores.".into());
    }
    let best = match v["best"].as_str().map(|s| s.trim().trim_start_matches("Answer ").to_uppercase()) {
        Some(b) if b.len() == 1 => {
            let k = (b.as_bytes()[0] as usize).wrapping_sub(b'A' as usize);
            shown.get(k).copied()
        }
        _ => None,
    };
    Ok((scores, best, v["why"].as_str().unwrap_or("").trim().to_string()))
}

/// Puts a verdict on its item: scores on the answers, the pick and the reason.
pub fn apply_verdict(item: &mut Item, shown: &[usize], r: Result<(Vec<Option<f32>>, Option<usize>, String), String>) {
    match r {
        Ok((scores, best, why)) => {
            for (k, &c) in shown.iter().enumerate() {
                item.answers[c].score = scores[k];
            }
            item.verdict = Some(Verdict { best, why, error: None });
        }
        Err(e) => item.verdict = Some(Verdict { error: Some(e), ..Default::default() }),
    }
}

// --- the leaderboard ------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Standing {
    pub key: String,
    pub label: String,
    pub elo: f64,
    pub wins: u32,
    pub ties: u32,
    pub losses: u32,
    /// The judge's average score, over how many answers.
    pub score: Option<f64>,
    pub judged: u32,
}

impl Standing {
    pub fn games(&self) -> u32 {
        self.wins + self.ties + self.losses
    }
}

const K: f64 = 32.0;

/// Your votes as Elo ratings (everyone starts at 1000; a pick beats each other answer, a tie or
/// "all bad" draws them all), oldest vote first, with the judge's average scores beside them.
pub fn leaderboard(runs: &[Run]) -> Vec<Standing> {
    let mut table: HashMap<String, Standing> = HashMap::new();
    let mut votes: Vec<(u64, &Run, &Item)> = vec![];
    for r in runs {
        for (c, who) in r.contestants.iter().enumerate() {
            let s = table.entry(who.key()).or_insert_with(|| Standing { key: who.key(), label: who.label(), elo: 1000.0, ..Default::default() });
            for it in &r.items {
                if let Some(x) = it.answers.get(c).and_then(|a| a.score) {
                    let n = s.judged as f64;
                    s.score = Some((s.score.unwrap_or(0.0) * n + x as f64) / (n + 1.0));
                    s.judged += 1;
                }
            }
        }
        for it in &r.items {
            if let Some(v) = &it.vote {
                votes.push((v.at, r, it));
            }
        }
    }
    votes.sort_by_key(|v| v.0);
    for (_, r, it) in votes {
        let v = it.vote.as_ref().unwrap();
        let took: Vec<usize> = (0..r.contestants.len()).filter(|&c| it.answers[c].ok()).collect();
        for (x, &a) in took.iter().enumerate() {
            for &b in &took[x + 1..] {
                let outcome = match v.best {
                    Some(w) if w == a && !v.all_bad => 1.0,
                    Some(w) if w == b && !v.all_bad => 0.0,
                    _ => 0.5,
                };
                let (ka, kb) = (r.contestants[a].key(), r.contestants[b].key());
                let (ra, rb) = (table[&ka].elo, table[&kb].elo);
                let expect = 1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0));
                table.get_mut(&ka).unwrap().elo += K * (outcome - expect);
                table.get_mut(&kb).unwrap().elo -= K * (outcome - expect);
            }
        }
        for &c in &took {
            let s = table.get_mut(&r.contestants[c].key()).unwrap();
            match v.best {
                Some(w) if w == c && !v.all_bad => s.wins += 1,
                Some(_) if !v.all_bad => s.losses += 1,
                _ => s.ties += 1,
            }
        }
    }
    let mut out: Vec<Standing> = table.into_values().filter(|s| s.games() > 0 || s.judged > 0).collect();
    out.sort_by(|a, b| b.elo.total_cmp(&a.elo).then(b.score.unwrap_or(0.0).total_cmp(&a.score.unwrap_or(0.0))).then(a.label.cmp(&b.label)));
    out
}

/// How often the judge picked what you picked: (agreed, both decided).
pub fn agreement(runs: &[Run]) -> (u32, u32) {
    let mut out = (0, 0);
    for it in runs.iter().flat_map(|r| &r.items) {
        if let (Some(v), Some(j)) = (&it.vote, &it.verdict)
            && j.error.is_none()
        {
            let mine = if v.all_bad { None } else { v.best };
            out.1 += 1;
            if mine == j.best {
                out.0 += 1;
            }
        }
    }
    out
}

// --- quantisations ---------------------------------------------------------------------------------

fn is_quant(t: &str) -> bool {
    let t = t.to_ascii_uppercase();
    if ["F16", "F32", "BF16", "FP16", "FP8", "MXFP4", "MXFP4_MOE"].contains(&t.as_str()) {
        return true;
    }
    let rest = t.strip_prefix("IQ").or_else(|| t.strip_prefix("TQ")).or_else(|| t.strip_prefix('Q'));
    rest.is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()) && r.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
}

/// A model name split into its family and its quantisation: `Llama-3.1-8B-Instruct-Q4_K_M` →
/// (`llama-3.1-8b-instruct`, `Q4_K_M`). Split files' `-00001-of-00003` is dropped first.
pub fn split_quant(name: &str) -> (String, Option<String>) {
    let mut stem = name.rsplit('/').next().unwrap_or(name).trim_end_matches(".gguf").to_string();
    if let Some(i) = stem.rfind("-of-")
        && stem[i + 4..].chars().all(|c| c.is_ascii_digit())
        && let Some(j) = stem[..i].rfind('-')
    {
        stem.truncate(j);
    }
    for (i, c) in stem.char_indices().rev() {
        if matches!(c, '-' | '.' | '_' | ':') && is_quant(&stem[i + 1..]) {
            return (stem[..i].to_lowercase(), Some(stem[i + 1..].to_uppercase()));
        }
    }
    (stem.to_lowercase(), None)
}

/// Other quantisations of the same model among `names` (indices), for a Q4-against-Q8 check.
pub fn quant_siblings(name: &str, names: &[String]) -> Vec<usize> {
    let (family, quant) = split_quant(name);
    if quant.is_none() {
        return vec![];
    }
    names
        .iter()
        .enumerate()
        .filter(|(_, n)| {
            let (f, q) = split_quant(n);
            f == family && q.is_some() && q != quant
        })
        .map(|(i, _)| i)
        .collect()
}

// --- saved prompts and runs ----------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct SetPrompt {
    pub text: String,
    pub reference: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct PromptSet {
    pub id: String,
    pub name: String,
    pub prompts: Vec<SetPrompt>,
    pub created: u64,
}

impl PromptSet {
    pub fn new(name: &str) -> PromptSet {
        PromptSet { id: crate::store::new_id(), name: name.into(), created: crate::store::now_ms(), ..Default::default() }
    }

    /// Adds one prompt per non-empty line (a quick way to paste a list in).
    pub fn add_lines(&mut self, text: &str) -> usize {
        let before = self.prompts.len();
        for l in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if !self.prompts.iter().any(|p| p.text == l) {
                self.prompts.push(SetPrompt { text: l.to_string(), reference: String::new() });
            }
        }
        self.prompts.len() - before
    }
}

/// JSON files in a folder, one per thing, newest first.
pub struct Store {
    pub runs: PathBuf,
    pub sets: PathBuf,
}

fn list_dir<T: serde::de::DeserializeOwned>(dir: &PathBuf) -> Vec<T> {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok())
        .collect()
}

impl Store {
    pub fn new(dir: PathBuf) -> Store {
        let s = Store { runs: dir.join("runs"), sets: dir.join("sets") };
        let _ = std::fs::create_dir_all(&s.runs);
        let _ = std::fs::create_dir_all(&s.sets);
        s
    }

    pub fn save_run(&self, r: &Run) -> Result<(), String> {
        let j = serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&self.runs.join(format!("{}.json", r.id)), &j).map_err(|e| e.to_string())
    }

    pub fn runs(&self) -> Vec<Run> {
        let mut v: Vec<Run> = list_dir(&self.runs);
        v.sort_by_key(|r| std::cmp::Reverse(r.created));
        v
    }

    pub fn delete_run(&self, id: &str) {
        let _ = std::fs::remove_file(self.runs.join(format!("{id}.json")));
    }

    pub fn save_set(&self, s: &PromptSet) -> Result<(), String> {
        let j = serde_json::to_vec_pretty(s).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&self.sets.join(format!("{}.json", s.id)), &j).map_err(|e| e.to_string())
    }

    pub fn sets(&self) -> Vec<PromptSet> {
        let mut v: Vec<PromptSet> = list_dir(&self.sets);
        v.sort_by_key(|s| std::cmp::Reverse(s.created));
        v
    }

    pub fn delete_set(&self, id: &str) {
        let _ = std::fs::remove_file(self.sets.join(format!("{id}.json")));
    }
}

// --- the GPU between Ollama and our server ---------------------------------------------------------

pub const OLLAMA: &str = "http://127.0.0.1:11434";

/// Unloads every Ollama model except `keep`, then loads `keep` so its first answer isn't timed
/// with the load (blocking; run it off the UI thread).
pub fn hand_gpu_to_ollama(keep: Option<&str>) -> Result<(), String> {
    let agent = crate::llm::agent(3);
    if let Ok(r) = agent.get(&format!("{OLLAMA}/api/ps")).call()
        && let Ok(v) = r.into_body().read_json::<Value>()
    {
        for m in v["models"].as_array().into_iter().flatten() {
            if let Some(name) = m["name"].as_str().filter(|n| Some(*n) != keep) {
                let _ = agent.post(&format!("{OLLAMA}/api/generate")).send_json(json!({"model": name, "keep_alive": 0}));
            }
        }
    }
    if let Some(model) = keep {
        let r = crate::llm::agent(5)
            .post(&format!("{OLLAMA}/api/generate"))
            .send_json(json!({"model": model, "prompt": "", "keep_alive": "10m"}))
            .map_err(|e| crate::llm::connect_error(&e, OLLAMA))?;
        let status = r.status().as_u16();
        if status != 200 {
            let text = r.into_body().read_to_string().unwrap_or_default();
            return Err(crate::llm::http_error(status, &text));
        }
    }
    Ok(())
}

// --- the report card (after fine-tuning) --------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Report {
    pub items: usize,
    pub judged: usize,
    pub tuned_wins: usize,
    pub base_wins: usize,
    pub ties: usize,
    pub base_score: Option<f64>,
    pub tuned_score: Option<f64>,
    /// How much of the reference each one's answers share, on average (0..1).
    pub base_overlap: f32,
    pub tuned_overlap: f32,
}

/// The base (contestant 0) against the tuned model (contestant 1) on held-back examples.
pub fn report(run: &Run) -> Report {
    let s = run.summary();
    let mut r = Report { items: run.items.len(), base_score: s.first().and_then(|x| x.score), tuned_score: s.get(1).and_then(|x| x.score), ..Default::default() };
    let (mut bo, mut to, mut n) = (0.0, 0.0, 0);
    for it in &run.items {
        if let Some(v) = it.verdict.as_ref().filter(|v| v.error.is_none()) {
            r.judged += 1;
            match v.best {
                Some(1) => r.tuned_wins += 1,
                Some(0) => r.base_wins += 1,
                _ => r.ties += 1,
            }
        }
        if !it.reference.trim().is_empty() && it.answers.len() >= 2 && it.answers[0].ok() && it.answers[1].ok() {
            bo += similarity(&it.answers[0].text, &it.reference);
            to += similarity(&it.answers[1].text, &it.reference);
            n += 1;
        }
    }
    if n > 0 {
        r.base_overlap = bo / n as f32;
        r.tuned_overlap = to / n as f32;
    }
    r
}


/// Most questions a report card asks (each is two answers and a judgement).
pub const REPORT_MAX: usize = 40;

/// The report card's run for an adapter: the base model (scale 0) against the tuned one (scale 1) on
/// held-back questions, at temperature 0, judged blind against the held-back replies.
pub fn report_run(a: &crate::engine::train::Adapter, prompts: Vec<(String, String)>, judge_provider: &str, judge_model: &str) -> Run {
    let who = |scale: f32| Who::Adapter { id: a.id.clone(), name: a.name.clone(), model_key: a.model_key.clone(), scale };
    let prompts: Vec<(String, String)> = prompts.into_iter().take(REPORT_MAX).collect();
    let mut run = Run::new(&format!("Report card: {}", a.name), vec![who(0.0), who(1.0)], prompts);
    run.system = a.system.clone();
    run.temperature = 0.0;
    run.blind = true;
    run.judge = Some(Judge { provider: judge_provider.into(), model: judge_model.into(), rubric: REPORT_RUBRIC.into() });
    run.report_for = Some(a.id.clone());
    run
}
