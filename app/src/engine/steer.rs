//! Steering Lab machinery: build a steering vector for a concept (two opposite descriptions) from
//! the model's own hidden states, keep it on disk, and hand it to llama-server as a control vector.
//!
//! The running llama-server does the templating and tokenising (exactly as it will when chatting);
//! the hidden states come from our CPU forward pass (`cubic_gguf::forward`), since llama.cpp
//! doesn't expose them.

use super::Task;
use cubic_gguf::forward::Model;
use cubic_gguf::stats::Progress;
use cubic_gguf::steer::{best_range, control_vector, directions, norm, read_control_vector};
use cubic_gguf::{Gguf, Writer};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Concept {
    pub name: String,
    /// "…as if you are {positive}."
    pub positive: String,
    pub negative: String,
}

pub fn presets() -> Vec<Concept> {
    let c = |n: &str, p: &str, q: &str| Concept { name: n.into(), positive: p.into(), negative: q.into() };
    vec![
        c("Joyful / Gloomy", "extremely joyful and cheerful", "extremely gloomy and miserable"),
        c("Formal / Casual", "a very formal, polite professional", "a very casual, slangy teenager"),
        c("Brief / Rambling", "someone who answers in as few words as possible", "someone who rambles on and on with tangents"),
        c("Confident / Unsure", "completely confident and certain of everything", "deeply unsure and doubtful of everything"),
        c("Poetic / Plain", "a poet who speaks in vivid imagery", "a dry technical manual"),
        c("Kind / Grumpy", "warm, kind and encouraging", "grumpy, impatient and curt"),
    ]
}

/// Everyday questions the contrast prompts share, and openers for the reply (the state is read
/// after the opener, where the model is committed to answering in character).
pub const QUESTIONS: [&str; 6] = [
    "Tell me about your weekend.",
    "What do you think of rain?",
    "Describe your home.",
    "How was work today?",
    "What is your favourite food?",
    "What will the future be like?",
];
pub const OPENERS: [&str; 2] = ["I", "Well,"];

/// The message lists of every contrast pair: (positive, negative) with the shared opener.
pub fn contrast_pairs(c: &Concept) -> Vec<(Vec<Value>, Vec<Value>, &'static str)> {
    let msg = |who: &str, q: &str| vec![json!({"role": "user", "content": format!("Act as if you are {}. {q}", who.trim())})];
    QUESTIONS
        .iter()
        .flat_map(|q| OPENERS.iter().map(move |o| (q, *o)))
        .map(|(q, o)| (msg(&c.positive, q), msg(&c.negative, q), o))
        .collect()
}

/// llama-server's tokeniser (special tokens in the text, like a template's, are parsed).
pub fn tokenize(root: &str, text: &str) -> Result<Vec<u32>, String> {
    let resp = crate::llm::agent(5)
        .post(&format!("{root}/tokenize"))
        .send_json(json!({ "content": text, "add_special": false, "parse_special": true }))
        .map_err(|e| crate::llm::connect_error(&e, root))?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    if status != 200 {
        return Err(crate::llm::http_error(status, &text));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    v["tokens"]
        .as_array()
        .ok_or("The server returned no tokens")?
        .iter()
        .map(|t| t.as_u64().or_else(|| t["id"].as_u64()).map(|n| n as u32).ok_or_else(|| "bad token".to_string()))
        .collect()
}

/// A saved steering vector (its directions are in `<id>.gguf` next to `<id>.json`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Vector {
    pub id: String,
    pub name: String,
    pub positive: String,
    pub negative: String,
    pub model_key: String,
    pub model_name: String,
    pub n_embd: usize,
    pub n_layer: usize,
    /// Suggested window of layers to steer (inclusive, never 0).
    pub layer_start: usize,
    pub layer_end: usize,
    /// Per layer: how cleanly the two sides separate (Cohen's d) and the direction's length.
    pub separation: Vec<f32>,
    pub norms: Vec<f32>,
    /// Per layer, every example's position along the direction (for the scatter).
    pub proj_pos: Vec<Vec<f32>>,
    pub proj_neg: Vec<Vec<f32>>,
    pub created: u64,
    pub build_secs: f32,
    pub tokens: usize,
}

pub struct Built {
    pub vector: Vector,
    pub layers: Vec<Vec<f32>>,
}

/// Opens every part of a model and runs the forward pass over `prompts`.
pub fn hidden_states(parts: &[PathBuf], prompts: &[Vec<u32>], progress: &Progress) -> Result<(Vec<Vec<Vec<f32>>>, usize, usize), String> {
    let mut maps = vec![];
    for p in parts {
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        maps.push(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?);
    }
    let mut g: Option<Gguf> = None;
    for (i, m) in maps.iter().enumerate() {
        let part = Gguf::parse(m, Some(m.len() as u64)).map_err(|e| e.to_string())?;
        match &mut g {
            None => g = Some(part),
            Some(g) => g.merge_split(part, i),
        }
    }
    let g = g.ok_or("no file")?;
    let model = Model::new(&g, maps.iter().map(|m| &m[..]).collect())?;
    let h = model.run(prompts, false, progress)?;
    Ok((h.last, model.cfg.n_embd, model.cfg.n_layer))
}

/// Builds a steering vector: template and tokenise every contrast prompt on the server at `root`,
/// read the hidden states on the CPU, take the mean difference per layer.
pub fn build(root: &str, model_key: &str, model_name: &str, parts: &[PathBuf], c: &Concept, task: &Arc<Task>) -> Result<Built, String> {
    if c.positive.trim().is_empty() || c.negative.trim().is_empty() {
        return Err("Describe both sides of the concept.".into());
    }
    let started = std::time::Instant::now();
    task.set_status("Writing the contrast prompts…".into());
    let pairs = contrast_pairs(c);
    let mut prompts = vec![];
    // All positives first, then all negatives.
    for positive in [true, false] {
        for (pos, neg, opener) in &pairs {
            if task.cancel.load(Ordering::Relaxed) {
                return Err("Stopped".into());
            }
            let text = super::xray::apply_template(root, if positive { pos } else { neg })? + opener;
            prompts.push(tokenize(root, &text)?);
        }
    }
    let tokens: usize = prompts.iter().map(Vec::len).sum();
    task.set_status(format!("Reading the model's hidden states on the CPU ({} prompts, {tokens} tokens)…", prompts.len()));
    let progress = Arc::new(Progress::default());
    let (stop_watch, done) = (progress.clone(), Arc::new(std::sync::atomic::AtomicBool::new(false)));
    let watcher = {
        let (task, done) = (task.clone(), done.clone());
        std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                task.progress.store(stop_watch.done.load(Ordering::Relaxed), Ordering::Relaxed);
                if task.cancel.load(Ordering::Relaxed) {
                    stop_watch.cancel.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
    };
    let r = hidden_states(parts, &prompts, &progress);
    done.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    task.progress.store(progress.done.load(Ordering::Relaxed), Ordering::Relaxed);
    let (hidden, n_embd, n_layer) = r?;
    let n = pairs.len();
    let d = directions(&hidden[..n], &hidden[n..]);
    let (layer_start, layer_end) = best_range(&d.separation);
    let vector = Vector {
        id: crate::store::new_id(),
        name: c.name.clone(),
        positive: c.positive.trim().into(),
        negative: c.negative.trim().into(),
        model_key: model_key.into(),
        model_name: model_name.into(),
        n_embd,
        n_layer,
        layer_start,
        layer_end,
        norms: d.layers.iter().map(|v| norm(v)).collect(),
        separation: d.separation,
        proj_pos: d.proj_pos,
        proj_neg: d.proj_neg,
        created: crate::store::now_ms(),
        build_secs: started.elapsed().as_secs_f32(),
        tokens,
    };
    Ok(Built { vector, layers: d.layers })
}

/// Strength 1 adds this much of the mean difference, spread over the steered layers (the shifts
/// add up through the residual stream). Measured on Llama 3.1 8B (joyful/gloomy, layers 14-23):
/// +1 writes "I'M SO EXCITED TO SHARE MY DAY WITH YOU!!!", -1 a flat "I don't have a day. I am a
/// computer program…", both fluent; ±1.7 tips into all caps / nihilism; ±0.7 barely shows.
pub const STRENGTH_UNIT: f32 = 1.5;

pub fn scale_for(strength: f32, start: usize, end: usize) -> f32 {
    strength * STRENGTH_UNIT / (end.saturating_sub(start) + 1) as f32
}

/// A ready-to-load control vector: only the chosen layers, already scaled (so llama-server takes
/// it with a plain `--control-vector`, whose path may contain ':' on Windows).
pub fn bake(layers: &[Vec<f32>], start: usize, end: usize, strength: f32) -> Vec<u8> {
    let scale = scale_for(strength, start, end);
    let mut w = Writer::new();
    w.str("general.architecture", "controlvector").u32("controlvector.layer_count", (end + 1 - start.max(1)) as u32);
    for (l, d) in layers.iter().enumerate().take(end + 1).skip(start.max(1)) {
        if !d.is_empty() {
            w.tensor_f32(&format!("direction.{l}"), &[d.len() as u64], |i| d[i as usize] * scale);
        }
    }
    w.finish()
}

pub struct Store {
    pub dir: PathBuf,
}

impl Store {
    pub fn new(dir: PathBuf) -> Store {
        let _ = std::fs::create_dir_all(dir.join("baked"));
        Store { dir }
    }

    fn json(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn gguf(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.gguf"))
    }

    pub fn save(&self, b: &Built) -> Result<(), String> {
        let file = control_vector(&b.layers, "");
        crate::store::write_atomic(&self.gguf(&b.vector.id), &file).map_err(|e| e.to_string())?;
        let j = serde_json::to_vec_pretty(&b.vector).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&self.json(&b.vector.id), &j).map_err(|e| e.to_string())
    }

    pub fn update(&self, v: &Vector) {
        if let Ok(j) = serde_json::to_vec_pretty(v) {
            let _ = crate::store::write_atomic(&self.json(&v.id), &j);
        }
    }

    /// Every saved vector, newest first.
    pub fn list(&self) -> Vec<Vector> {
        let mut out: Vec<Vector> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice::<Vector>(&std::fs::read(e.path()).ok()?).ok())
            .filter(|v| self.gguf(&v.id).is_file())
            .collect();
        out.sort_by_key(|v| std::cmp::Reverse(v.created));
        out
    }

    pub fn layers(&self, id: &str) -> Result<Vec<Vec<f32>>, String> {
        let bytes = std::fs::read(self.gguf(id)).map_err(|e| format!("The vector's file is gone: {e}"))?;
        read_control_vector(&bytes)
    }

    pub fn delete(&self, id: &str) {
        let _ = std::fs::remove_file(self.json(id));
        let _ = std::fs::remove_file(self.gguf(id));
        self.clear_baked(id);
    }

    fn clear_baked(&self, id: &str) {
        for e in std::fs::read_dir(self.dir.join("baked")).into_iter().flatten().flatten() {
            if e.file_name().to_string_lossy().starts_with(&format!("{id}_")) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    /// Where the vector baked at this strength and layer window lives.
    pub fn baked_path(&self, v: &Vector, strength: f32) -> PathBuf {
        let milli = (strength * 100.0).round() as i32;
        self.dir.join("baked").join(format!("{}_{}_{}_{}.gguf", v.id, v.layer_start, v.layer_end, milli))
    }

    /// Writes (once) the vector baked for llama-server and returns its path.
    pub fn bake(&self, v: &Vector, strength: f32) -> Result<PathBuf, String> {
        let path = self.baked_path(v, strength);
        if !path.is_file() {
            let layers = self.layers(&v.id)?;
            crate::store::write_atomic(&path, &bake(&layers, v.layer_start, v.layer_end, strength)).map_err(|e| e.to_string())?;
        }
        Ok(path)
    }
}

/// One greedy reply from llama-server's `/completion` for an already-templated prompt.
pub fn complete(root: &str, prompt: &str, n_predict: u32) -> Result<String, String> {
    let resp = crate::llm::agent(5)
        .post(&format!("{root}/completion"))
        .send_json(json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 1,
            "cache_prompt": false,
        }))
        .map_err(|e| crate::llm::connect_error(&e, root))?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    if status != 200 {
        return Err(crate::llm::http_error(status, &text));
    }
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    v["content"].as_str().map(|s| s.trim().to_string()).ok_or_else(|| "The server returned no text".into())
}
