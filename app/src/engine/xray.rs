//! X-ray: generation through llama-server's `/completion` with the model's top candidates for
//! every token, a tree of branches made by forking at any token, and a lens that shows what the
//! sampler settings do to each step's distribution.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

/// One token the model could have produced at a step, with its probability before sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: i64,
    pub bytes: Vec<u8>,
    pub p: f32,
}

impl Candidate {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

/// A generated token and the alternatives the model weighed for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub chosen: Candidate,
    /// Most likely first. May or may not contain the chosen token.
    pub top: Vec<Candidate>,
}

impl Step {
    /// Uncertainty in bits over the visible candidates, with the unseen rest as one bucket
    /// (so it's a lower bound on the true entropy).
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        let mut seen = 0.0f32;
        for c in &self.top {
            if c.p > 0.0 {
                h -= c.p * c.p.log2();
                seen += c.p;
            }
        }
        let rest = (1.0 - seen).max(0.0);
        if rest > 1e-6 {
            h -= rest * rest.log2();
        }
        h
    }

    /// The chosen token's rank among the candidates (0 = the favourite), if it is among them.
    pub fn rank(&self) -> Option<usize> {
        self.top.iter().position(|c| c.id == self.chosen.id)
    }
}

fn candidate(v: &Value) -> Option<Candidate> {
    let id = v["id"].as_i64()?;
    let bytes = match v["bytes"].as_array() {
        Some(b) => b.iter().filter_map(|x| x.as_u64().map(|n| n as u8)).collect(),
        None => v["token"].as_str().unwrap_or("").as_bytes().to_vec(),
    };
    // `logprob` normally; `prob` when the server was asked for post-sampling numbers.
    let p = match (v["prob"].as_f64(), v["logprob"].as_f64()) {
        (Some(p), _) => p as f32,
        (None, Some(lp)) => (lp as f32).exp(),
        _ => return None,
    };
    Some(Candidate { id, bytes, p })
}

/// The steps in a `/completion` chunk's `completion_probabilities` (either shape).
pub fn parse_steps(chunk: &Value) -> Vec<Step> {
    chunk["completion_probabilities"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let chosen = candidate(e)?;
            let top = e["top_logprobs"].as_array().or_else(|| e["top_probs"].as_array()).into_iter().flatten().filter_map(candidate).collect();
            Some(Step { chosen, top })
        })
        .collect()
}

// --- Sampler lens ---------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    /// -1 = random.
    pub seed: i64,
    pub max_tokens: u32,
}

impl Default for Sampling {
    fn default() -> Self {
        // llama-server's /completion defaults.
        Sampling { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: -1, max_tokens: 400 }
    }
}

/// What llama.cpp's default chain (top-k → top-p → min-p → temperature) does to a step's
/// candidates: the probability each one ends up with when the token is drawn (0 = cut).
/// Exact as long as the cut-offs fall inside the candidates we can see.
pub fn lens(probs: &[f32], s: &Sampling) -> Vec<f32> {
    let n = probs.len();
    let mut keep = vec![true; n];
    if s.top_k > 0 {
        for k in keep.iter_mut().skip(s.top_k as usize) {
            *k = false;
        }
    }
    if s.top_p < 1.0 {
        // Smallest set of the most likely whose (renormalised) mass reaches top_p.
        // Against the mass top-k left: all of it when top-k reaches past what we can see.
        let total: f32 = if s.top_k > 0 && (s.top_k as usize) <= n {
            probs.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| p).sum()
        } else {
            1.0
        };
        let mut cum = 0.0;
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            if cum >= s.top_p * total {
                keep[i] = false;
            } else {
                cum += probs[i];
            }
        }
    }
    if s.min_p > 0.0 {
        let max = probs.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).fold(0.0, f32::max);
        for i in 0..n {
            if probs[i] < s.min_p * max {
                keep[i] = false;
            }
        }
    }
    if s.temperature <= 0.0 {
        // Greedy: the favourite always.
        let best = (0..n).filter(|&i| keep[i]).max_by(|&a, &b| probs[a].total_cmp(&probs[b]));
        return (0..n).map(|i| if Some(i) == best { 1.0 } else { 0.0 }).collect();
    }
    let scaled: Vec<f32> = (0..n).map(|i| if keep[i] && probs[i] > 0.0 { probs[i].powf(1.0 / s.temperature) } else { 0.0 }).collect();
    let sum: f32 = scaled.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; n];
    }
    scaled.iter().map(|x| x / sum).collect()
}

// --- Branches -------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum BranchState {
    Running,
    Done,
    Stopped,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct Branch {
    /// Where it split off: (parent branch, index of the step that was replaced).
    pub parent: Option<(usize, usize)>,
    pub steps: Vec<Step>,
    pub state: BranchState,
    pub gen_tps: Option<f64>,
}

/// A prompt and every continuation explored from it.
#[derive(Clone, Debug, Default)]
pub struct Tree {
    /// The templated prompt, exactly as the model reads it.
    pub prompt: String,
    pub branches: Vec<Branch>,
}

impl Tree {
    pub fn new(prompt: String) -> Tree {
        Tree { prompt, branches: vec![Branch { parent: None, steps: vec![], state: BranchState::Running, gen_tps: None }] }
    }

    /// Every step a branch's text is made of, from the root: its ancestors' steps up to each fork,
    /// then its own.
    pub fn path(&self, b: usize) -> Vec<&Step> {
        let mut chain = vec![];
        let mut cur = b;
        let mut upto = self.branches[b].steps.len();
        loop {
            let br = &self.branches[cur];
            chain.push((cur, upto));
            match br.parent {
                Some((p, i)) => {
                    cur = p;
                    upto = i;
                }
                None => break,
            }
        }
        chain.iter().rev().flat_map(|&(bi, upto)| self.branches[bi].steps[..upto.min(self.branches[bi].steps.len())].iter()).collect()
    }

    /// Steps of the branch's ancestors that come before its own (where its own steps start in `path`).
    pub fn offset(&self, b: usize) -> usize {
        self.path(b).len() - self.branches[b].steps.len()
    }

    /// The reply text of a branch.
    pub fn text(&self, b: usize) -> String {
        let bytes: Vec<u8> = self.path(b).iter().flat_map(|s| s.chosen.bytes.iter().copied()).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Makes a branch that takes `alt` instead of the token at `step` of branch `b`. Returns its index.
    pub fn fork(&mut self, b: usize, step: usize, alt: Candidate) -> usize {
        let top = self.branches[b].steps[step].top.clone();
        self.branches.push(Branch {
            parent: Some((b, step)),
            steps: vec![Step { chosen: alt, top }],
            state: BranchState::Running,
            gen_tps: None,
        });
        self.branches.len() - 1
    }

    /// The `/completion` prompt that continues branch `b`: the template text, then its tokens by id
    /// (exact: nothing is re-tokenised).
    pub fn continuation(&self, b: usize) -> Value {
        let mut p = vec![Value::String(self.prompt.clone())];
        p.extend(self.path(b).iter().map(|s| json!(s.chosen.id)));
        Value::Array(p)
    }

    /// Which branch and step produced position `i` of branch `b`'s text (ancestors own the start).
    pub fn locate(&self, b: usize, i: usize) -> (usize, usize) {
        let off = self.offset(b);
        if i >= off {
            return (b, i - off);
        }
        match self.branches[b].parent {
            Some((p, _)) => self.locate(p, i),
            None => (b, i),
        }
    }

    /// Children of a branch, with the step they replaced.
    pub fn forks_of(&self, b: usize) -> Vec<(usize, usize)> {
        self.branches.iter().enumerate().filter_map(|(i, br)| br.parent.filter(|(p, _)| *p == b).map(|(_, s)| (i, s))).collect()
    }
}

// --- Talking to llama-server ----------------------------------------------------------------------

/// `http://127.0.0.1:8193/v1` → `http://127.0.0.1:8193`.
pub fn root_of(base_url: &str) -> String {
    base_url.trim_end_matches('/').trim_end_matches("/v1").to_string()
}

/// The model's chat template applied to `messages`, ready for the assistant to answer.
pub fn apply_template(root: &str, messages: &[Value]) -> Result<String, String> {
    let resp = crate::llm::agent(5)
        .post(&format!("{root}/apply-template"))
        .send_json(json!({ "messages": messages }))
        .map_err(|e| crate::llm::connect_error(&e, root))?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    if status != 200 {
        return Err(crate::llm::http_error(status, &text));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    v["prompt"].as_str().map(str::to_string).ok_or_else(|| "The server returned no prompt".into())
}

#[derive(Debug)]
pub enum XEv {
    Steps(Vec<Step>),
    Done { gen_tps: Option<f64> },
    Error(String),
}

pub const N_PROBS: u32 = 10;

pub fn request_body(prompt: Value, s: &Sampling) -> Value {
    json!({
        "prompt": prompt,
        "stream": true,
        "n_probs": N_PROBS,
        "n_predict": s.max_tokens,
        "temperature": s.temperature,
        "top_k": s.top_k,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "seed": s.seed,
        "cache_prompt": true,
        "return_tokens": true,
    })
}

/// Streams a continuation of `prompt`; events carry `(branch, event)`.
pub fn generate(root: String, prompt: Value, s: Sampling, branch: usize, tx: Sender<(usize, XEv)>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let send = |e: XEv| {
            let _ = tx.send((branch, e));
            ctx.request_repaint();
        };
        let r = (|| -> Result<Option<f64>, String> {
            let resp = crate::llm::agent(5)
                .post(&format!("{root}/completion"))
                .header("Accept", "text/event-stream")
                .send_json(request_body(prompt, &s))
                .map_err(|e| crate::llm::connect_error(&e, &root))?;
            let status = resp.status().as_u16();
            let mut reader = BufReader::new(resp.into_body().into_reader());
            if status != 200 {
                let mut t = String::new();
                let _ = std::io::Read::read_to_string(&mut reader, &mut t);
                return Err(crate::llm::http_error(status, &t));
            }
            read_completion(reader, &cancel, &|steps| send(XEv::Steps(steps)))
        })();
        match r {
            Ok(tps) => send(XEv::Done { gen_tps: tps }),
            Err(e) if !cancel.load(Ordering::Relaxed) => send(XEv::Error(e)),
            Err(_) => {}
        }
    });
}

/// Reads `/completion` SSE chunks: steps as they come, then the speed from the final chunk.
pub fn read_completion(reader: impl BufRead, cancel: &AtomicBool, on_steps: &dyn Fn(Vec<Step>)) -> Result<Option<f64>, String> {
    let mut stopped = false;
    let mut tps = None;
    for line in reader.lines() {
        if cancel.load(Ordering::Relaxed) {
            return Err("Stopped".into());
        }
        let line = line.map_err(|e| format!("Connection dropped: {e}"))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim_start) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(e) = v.get("error") {
            return Err(e["message"].as_str().map(str::to_string).unwrap_or_else(|| e.to_string()));
        }
        let steps = parse_steps(&v);
        if !steps.is_empty() {
            on_steps(steps);
        }
        if let Some(t) = v["timings"]["predicted_per_second"].as_f64() {
            tps = Some(t);
        }
        if v["stop"].as_bool() == Some(true) {
            stopped = true;
            break;
        }
    }
    if stopped { Ok(tps) } else { Err("Connection dropped before the reply finished.".into()) }
}
