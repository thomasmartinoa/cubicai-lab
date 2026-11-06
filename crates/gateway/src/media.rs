//! Making pictures, video, music and speech on this PC with ComfyUI (a port of `cubicai_media.py`):
//! what the phone can make, turning a request into a ComfyUI prompt, and the small helpers around
//! files. The job queue and ComfyUI itself are in `jobs.rs`.
//!
//! Nothing generated stays on the PC: results wait in a private temp folder until the phone has
//! them (an hour at most), and only what the user saves on the phone comes back, into the save folder.

use crate::workflow::{named_get, to_api, Named};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// What the phone can make. Selectors: "#<node id>.<input>" for a node of the flattened workflow,
/// "@<name>" for an input of the workflow's top-level subgraph. "fixed" values are always applied.
/// Kept as JSON so it reads like (and is checked against) the Python gateway's table.
pub const GENERATORS: &str = r##"[
  {"id": "qwen-uncensored", "name": "Qwen Image 2.1 Uncensored", "kind": "image", "mode": "create",
   "workflow": "qwen_image_2_1_UNCENSORED_t2i.json", "note": "Best quality, about 1½ min",
   "prompt": ["@prompt"], "negative": ["@negative_prompt"], "seed": ["@seed"],
   "steps": ["#459:458.steps"], "cfg": ["#459:458.cfg"],
   "size": {"width": ["@width"], "height": ["@height"], "side": 1024}},
  {"id": "z-image", "name": "Z-Image Turbo", "kind": "image", "mode": "create",
   "workflow": "image_z_image_turbo_int8.json", "note": "Fast, about 25 s",
   "prompt": ["@text"], "seed": ["@seed"],
   "size": {"width": ["@width"], "height": ["@height"], "side": 1024}},
  {"id": "flux3-klein", "name": "Flux 3 Klein", "kind": "image", "mode": "create",
   "workflow": "image_flux3_klein_distilled_t2i.json", "note": "Sharp detail, rewrites your prompt first, about 35 s",
   "prompt": ["@value"], "seed": ["@seed"],
   "size": {"width": ["@width"], "height": ["@height"], "side": 1024}},
  {"id": "flux2-klein", "name": "Flux 2 Klein", "kind": "image", "mode": "create",
   "workflow": "image_flux2_klein_text_to_image.json", "note": "Fastest, about 10 s",
   "prompt": ["#76.value"], "seed": ["#77:73.noise_seed"],
   "size": {"width": ["#77:68.value"], "height": ["#77:69.value"], "side": 1024}},
  {"id": "illustrious", "name": "Illustrious XL (anime)", "kind": "image", "mode": "create",
   "workflow": "my_illustrious_anime.json", "note": "Anime and illustration; use tags",
   "prompt": ["#10.text"], "negative": ["#11.text"], "seed": ["#12.seed"], "steps": ["#12.steps"], "cfg": ["#12.cfg"],
   "size": {"width": ["#13.width"], "height": ["#13.height"], "side": 1024, "multiple": 64}},
  {"id": "qwen-edit", "name": "Qwen Image Edit", "kind": "image", "mode": "edit",
   "workflow": "image_qwen_image_edit_2511_int8.json", "note": "Describe the change to your photo, about 45 s",
   "image": "required", "image_into": ["#41.image"],
   "prompt": ["@prompt"], "seed": ["@seed"], "fixed": {"#170:168.value": true}},
  {"id": "flux2-edit", "name": "Flux 2 Klein Edit", "kind": "image", "mode": "edit",
   "workflow": "image_flux2_klein_image_edit_4b_distilled.json", "note": "Quick edits to your photo, about 20 s",
   "image": "required", "image_into": ["#76.image"], "prompt": ["@text"], "seed": ["@noise_seed"]},
  {"id": "cutout", "name": "Remove background", "kind": "image", "mode": "tool",
   "workflow": "utility_birefnet_remove_background.json", "note": "Transparent PNG of the subject",
   "image": "required", "image_into": ["#17.image"]},
  {"id": "upscale", "name": "Upscale 2×", "kind": "image", "mode": "tool",
   "workflow": "utility_seedvr2_3b_int8_upscale_image.json", "note": "Sharper, twice the size",
   "image": "required", "image_into": ["#1.image"], "seed": ["@seed"],
   "fixed": {"@resize_type.multiplier": 2}},
  {"id": "wan-5b", "name": "Wan 2.2 5B", "kind": "video", "mode": "create",
   "workflow": "video_wan2_2_5B_ti2v.json", "note": "Text or photo to video, about 1 min per second",
   "image": "optional", "image_into": ["#55.start_image"],
   "prompt": ["#6.text"], "seed": ["#3.seed"], "steps": ["#3.steps"], "cfg": ["#3.cfg"],
   "size": {"width": ["#55.width"], "height": ["#55.height"], "side": 704, "multiple": 32},
   "seconds": {"into": ["#55.length"], "fps": 24, "default": 3, "max": 5}},
  {"id": "wan-14b", "name": "Wan 2.2 14B", "kind": "video", "mode": "animate",
   "workflow": "video_wan2_2_14B_i2v.json", "note": "Brings your photo to life, about 1 min per second",
   "image": "required", "image_into": ["#97.image"],
   "prompt": ["@text"], "seed": ["@noise_seed"], "fixed": {"#129:131.value": true},
   "size": {"width": ["@width"], "height": ["@height"], "side": 640, "multiple": 16},
   "seconds": {"into": ["#129:161.value"], "default": 4, "max": 5}},
  {"id": "ace-step", "name": "ACE-Step 1.5", "kind": "audio", "mode": "music",
   "workflow": "audio_ace_step_1_5_checkpoint.json", "note": "Describe the style; lyrics optional. About 15 s per minute",
   "prompt": ["#94.tags"], "lyrics": ["#94.lyrics"], "seed": ["#94.seed", "#3.seed"],
   "seconds": {"into": ["#94.duration", "#98.seconds"], "default": 60, "max": 240}},
  {"id": "chatterbox", "name": "Chatterbox voice", "kind": "audio", "mode": "speech",
   "workflow": "audio-chatterbox_tts_multilingual.json", "note": "Reads your text aloud",
   "prompt": ["#1.text"], "seed": ["#1.seed"], "language": ["#1.language"],
   "fixed": {"#1.repetition_penalty": 1.2},
   "audio": "optional", "audio_into": ["#2.audio"], "voices": ["woman", "man"]}
]"##;
// Chatterbox: the node's default repetition penalty (2.0) keeps it from ever stopping; a short
// sentence came back as 25 s of babble. Its built-in voice drifts between a woman's and a man's from
// one seed to the next, so speech without a sample uses a reference voice (woman.wav, man.wav).

pub fn generators_table() -> Vec<Value> {
    serde_json::from_str(GENERATORS).expect("the generator table is valid JSON")
}

/// Steps and CFG are offered only where the workflow sets them (distilled models are tuned to CFG 1
/// and a few steps, and other values mostly ruin them). The defaults the phone shows are the workflow's.
pub const STEPS_RANGE: (i64, i64) = (4, 60);
pub const CFG_RANGE: (f64, f64) = (1.0, 12.0);
/// Short enough to read and type, and within every seed input's range.
pub const SEED_MAX: u64 = (1 << 32) - 1;
/// Tall shapes first, then wide: the app shows them four to a row.
pub const ASPECTS: [(&str, (u32, u32)); 7] = [("1:1", (1, 1)), ("3:4", (3, 4)), ("2:3", (2, 3)), ("9:16", (9, 16)), ("4:3", (4, 3)), ("3:2", (3, 2)), ("16:9", (16, 9))];
pub const SAVE_NODES: [&str; 9] = ["SaveImage", "SaveImageAdvanced", "SaveVideo", "SaveAudio", "SaveAudioMP3", "SaveAudioOpus", "SaveAudioAdvanced", "SaveAnimatedWEBP", "VHS_VideoCombine"];
/// Inside ComfyUI's input/ and output/, emptied after every job.
pub const SUBFOLDER: &str = "cubicai";
pub const MAX_SAVE: u64 = 1024 * 1024 * 1024;

/// Where media lives. Defaults match the Python gateway; the same environment variables move them.
#[derive(Clone, Debug)]
pub struct Paths {
    pub comfy_dir: PathBuf,
    pub comfy_port: u16,
    pub save_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub voices_dir: PathBuf,
    /// Seconds ComfyUI may sit idle before it's stopped.
    pub idle_stop: u64,
    pub unit: String,
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let env_path = |k: &str, default: PathBuf| std::env::var(k).map(|v| v.strip_prefix("~/").map(|r| home.join(r)).unwrap_or_else(|| PathBuf::from(&v))).unwrap_or(default);
        Paths {
            comfy_dir: env_path("CUBICAI_COMFY_DIR", "/mnt/storage/games/AI_local/ComfyUI".into()),
            comfy_port: std::env::var("CUBICAI_COMFY_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8188),
            save_dir: env_path("CUBICAI_SAVE_DIR", "/mnt/storage/games/AI_local/CubicAI".into()),
            cache_dir: env_path("CUBICAI_CACHE_DIR", home.join(".cache/cubicai-server")),
            voices_dir: env_path("CUBICAI_VOICES_DIR", home.join(".config/cubicai-server/voices")),
            idle_stop: std::env::var("CUBICAI_COMFY_IDLE").ok().and_then(|p| p.parse().ok()).unwrap_or(600),
            unit: "cubicai-comfyui.service".into(),
        }
    }

    pub fn workflows(&self) -> PathBuf {
        self.comfy_dir.join("user/default/workflows")
    }

    pub fn temp(&self) -> PathBuf {
        self.cache_dir.join("media")
    }

    /// ComfyUI's node list as last seen, so models can be checked while it's stopped.
    pub fn info_cache(&self) -> PathBuf {
        self.cache_dir.join("object_info.json")
    }

    /// A reference voice (woman.wav, man.wav…), for speech without a sample of its own.
    pub fn voice_file(&self, name: &str) -> Option<PathBuf> {
        if name.is_empty() || !name.chars().all(char::is_alphabetic) {
            return None;
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(&self.voices_dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            // Python's glob("<name>.*").
            .filter(|p| p.file_name().and_then(|s| s.to_str()).is_some_and(|f| f.starts_with(&format!("{name}."))))
            .collect();
        found.sort();
        found.into_iter().next()
    }
}

// --- Building a prompt from a workflow ------------------------------------------------------------

pub fn targets(prompt: &Map<String, Value>, named: &Named, selector: &str) -> Vec<(String, String)> {
    if let Some(name) = selector.strip_prefix('@') {
        return named_get(named, name).to_vec();
    }
    let rest = selector.strip_prefix('#').unwrap_or(selector);
    let (node, name) = rest.split_once('.').unwrap_or((rest, ""));
    if prompt.contains_key(node) { vec![(node.to_string(), name.to_string())] } else { vec![] }
}

fn set(prompt: &mut Map<String, Value>, named: &Named, selectors: &Value, value: &Value) {
    for sel in selectors.as_array().into_iter().flatten().filter_map(Value::as_str) {
        for (node, name) in targets(prompt, named, sel) {
            prompt[&node]["inputs"][&name] = value.clone();
        }
    }
}

/// Width and height for an aspect ratio with about side² pixels, snapped to a multiple
/// (Python's round, which rounds halves to even).
pub fn size(aspect: &str, side: u32, multiple: u32) -> (u32, u32) {
    let (w, h) = ASPECTS.iter().find(|(a, _)| *a == aspect).map(|x| x.1).unwrap_or((1, 1));
    let area = (side * side) as f64;
    let width = (area * w as f64 / h as f64).sqrt();
    let height = width * h as f64 / w as f64;
    let m = multiple as f64;
    let snap = |v: f64| (multiple as i64).max(((v / m).round_ties_even() as i64) * multiple as i64) as u32;
    (snap(width), snap(height))
}

fn load_node(prompt: &mut Map<String, Value>, kind: &str, filename: &str) -> String {
    let id = format!("cubicai_{kind}");
    let (class, input) = if kind == "image" { ("LoadImage", "image") } else { ("LoadAudio", "audio") };
    prompt.insert(id.clone(), json!({"class_type": class, "inputs": {input: filename}, "_meta": {"title": "CubicAI input"}}));
    id
}

/// Points the workflow's loader at the uploaded file, or wires in a loader where the input expects one.
fn attach(prompt: &mut Map<String, Value>, named: &Named, selectors: &Value, kind: &str, filename: &str) {
    let loader_input = if kind == "image" { "image" } else { "audio" };
    for sel in selectors.as_array().into_iter().flatten().filter_map(Value::as_str) {
        for (node, name) in targets(prompt, named, sel) {
            let class = prompt[&node]["class_type"].as_str().unwrap_or("").to_string();
            if (class == "LoadImage" || class == "LoadAudio") && name == loader_input {
                prompt[&node]["inputs"][&name] = filename.into();
            } else {
                let id = load_node(prompt, kind, filename);
                prompt[&node]["inputs"][&name] = json!([id, 0]);
            }
        }
    }
}

/// No file given for an optional input: drop the loader and every link to it.
fn detach(prompt: &mut Map<String, Value>, named: &Named, selectors: &Value) {
    for sel in selectors.as_array().into_iter().flatten().filter_map(Value::as_str) {
        for (node, name) in targets(prompt, named, sel) {
            let Some(class) = prompt.get(&node).map(|n| n["class_type"].as_str().unwrap_or("").to_string()) else { continue };
            if class == "LoadImage" || class == "LoadAudio" {
                prompt.shift_remove(&node);
                for other in prompt.values_mut() {
                    if let Some(inputs) = other["inputs"].as_object_mut() {
                        inputs.retain(|_, v| !v.as_array().is_some_and(|a| a.first().and_then(Value::as_str) == Some(node.as_str())));
                    }
                }
            } else if let Some(inputs) = prompt[&node]["inputs"].as_object_mut() {
                inputs.shift_remove(&name);
            }
        }
    }
}

fn reachable(prompt: &Map<String, Value>, outputs: &[String]) -> std::collections::HashSet<String> {
    let mut keep = std::collections::HashSet::new();
    let mut todo: Vec<String> = outputs.to_vec();
    while let Some(n) = todo.pop() {
        if keep.contains(&n) || !prompt.contains_key(&n) {
            continue;
        }
        for v in prompt[&n]["inputs"].as_object().into_iter().flat_map(|m| m.values()) {
            if let Some(a) = v.as_array()
                && a.len() == 2
                && let Some(id) = a[0].as_str()
            {
                todo.push(id.to_string());
            }
        }
        keep.insert(n);
    }
    keep
}

/// A whole number the way Python's isinstance(x, int) sees JSON (not a bool, not 5.0).
fn as_int(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_u64().map(|u| u as i64))
}

/// A number (int or float, not a bool).
fn as_num(v: &Value) -> Option<f64> {
    if v.is_number() { v.as_f64() } else { None }
}

/// The ComfyUI prompt for a request. `info` is ComfyUI's node list; `random_seed` is used when the
/// params carry no seed.
#[allow(clippy::too_many_arguments)]
pub fn build(spec: &Value, params: &Value, job_id: &str, image: Option<&str>, audio: Option<&str>, workflow: &Value, info: &Value, random_seed: u64) -> Result<Map<String, Value>, String> {
    let (mut prompt, named) = to_api(workflow, info)?;
    let text = params["prompt"].as_str().unwrap_or("").trim().to_string();
    if spec.get("prompt").is_some() {
        if text.is_empty() {
            return Err("Type what you want to make.".into());
        }
        set(&mut prompt, &named, &spec["prompt"], &text.into());
    }
    if spec.get("negative").is_some() && !params["negative"].is_null() {
        let neg = match &params["negative"] {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        set(&mut prompt, &named, &spec["negative"], &neg.into());
    }
    if spec.get("lyrics").is_some() {
        let l = params["lyrics"].as_str().unwrap_or("").trim();
        set(&mut prompt, &named, &spec["lyrics"], &(if l.is_empty() { "[Instrumental]" } else { l }).into());
    }
    if spec.get("language").is_some() && params["language"].as_str().is_some_and(|l| !l.is_empty()) {
        set(&mut prompt, &named, &spec["language"], &params["language"]);
    }
    let seed = as_int(&params["seed"]).filter(|&s| s >= 0).map(|s| s as u64).unwrap_or(random_seed);
    for sel in spec["seed"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        for (node, name) in targets(&prompt, &named, sel) {
            // Seed ranges differ per node (Chatterbox stops at 2^32 - 1).
            let class = prompt[&node]["class_type"].as_str().unwrap_or("").to_string();
            let spec = &info[&class]["input"];
            let entry = if spec["required"][&name].is_null() { &spec["optional"][&name] } else { &spec["required"][&name] };
            // Exact, as Python's integers: KSampler's maximum is 2^64 - 1.
            let max = &entry.as_array().and_then(|e| e.get(1)).map(|o| o["max"].clone()).unwrap_or(Value::Null);
            let top: Option<u128> = max.as_u64().map(u128::from).or_else(|| max.as_f64().map(|f| f as u128)).filter(|&t| t != 0);
            prompt[&node]["inputs"][&name] = match top {
                Some(t) => json!((seed as u128 % (t + 1)) as u64),
                None => json!(seed),
            };
        }
    }
    if spec.get("steps").is_some()
        && let Some(v) = as_num(&params["steps"])
    {
        let v = v.max(STEPS_RANGE.0 as f64).min(STEPS_RANGE.1 as f64);
        set(&mut prompt, &named, &spec["steps"], &json!(v.trunc() as i64));
    }
    if spec.get("cfg").is_some()
        && let Some(v) = as_num(&params["cfg"])
    {
        set(&mut prompt, &named, &spec["cfg"], &json!(v.max(CFG_RANGE.0).min(CFG_RANGE.1)));
    }
    if let Some(s) = spec.get("size") {
        let (w, h) = size(params["aspect"].as_str().unwrap_or("1:1"), s["side"].as_u64().unwrap_or(1024) as u32, s["multiple"].as_u64().unwrap_or(16) as u32);
        set(&mut prompt, &named, &s["width"], &json!(w));
        set(&mut prompt, &named, &s["height"], &json!(h));
    }
    if let Some(s) = spec.get("seconds") {
        let asked = match &params["seconds"] {
            Value::Number(n) => n.as_f64().map(|f| f.trunc() as i64),
            Value::String(t) => t.trim().parse::<i64>().ok(),
            _ => None,
        }
        .filter(|&n| n != 0)
        .unwrap_or(s["default"].as_i64().unwrap_or(1));
        let seconds = asked.min(s["max"].as_i64().unwrap_or(asked)).max(1);
        let value = match s["fps"].as_i64() {
            Some(fps) => seconds * fps + 1,
            None => seconds,
        };
        set(&mut prompt, &named, &s["into"], &json!(value));
    }
    for (sel, value) in spec["fixed"].as_object().into_iter().flatten() {
        set(&mut prompt, &named, &json!([sel]), value);
    }
    for (kind, name) in [("image", image), ("audio", audio)] {
        let Some(need) = spec.get(kind).and_then(Value::as_str) else { continue };
        let into = &spec[format!("{kind}_into")];
        match name {
            Some(n) => attach(&mut prompt, &named, into, kind, n),
            None if need == "required" => return Err(if kind == "image" { "Add a photo to use this.".into() } else { "Add a voice sample to use this.".into() }),
            None => detach(&mut prompt, &named, into),
        }
    }
    let outputs: Vec<String> = prompt.iter().filter(|(_, v)| v["class_type"].as_str().is_some_and(|c| SAVE_NODES.contains(&c))).map(|(k, _)| k.clone()).collect();
    if outputs.is_empty() {
        return Err(format!("{} has no save node", spec["workflow"].as_str().unwrap_or("the workflow")));
    }
    for k in &outputs {
        prompt[k]["inputs"]["filename_prefix"] = format!("{SUBFOLDER}/{job_id}").into();
    }
    // Previews, comparisons and unused loaders aren't run.
    let keep = reachable(&prompt, &outputs);
    prompt.retain(|k, _| keep.contains(k));
    Ok(prompt)
}

/// Whether a generator's models are on this PC (its workflow's model choices are among ComfyUI's).
pub fn available(workflow: Option<&Value>, info: Option<&Value>) -> bool {
    let Some(workflow) = workflow else { return false };
    let Some(info) = info else { return true };
    let Ok((prompt, _)) = to_api(workflow, info) else { return false };
    for node in prompt.values() {
        let spec = &info[node["class_type"].as_str().unwrap_or("")]["input"];
        for section in ["required", "optional"] {
            for (name, entry) in spec[section].as_object().into_iter().flatten() {
                let Some(value) = node["inputs"][name].as_str() else { continue };
                if !name.ends_with("_name") {
                    continue;
                }
                let e = entry.as_array();
                let first = e.and_then(|e| e.first());
                let options = match first {
                    Some(Value::Array(a)) => Some(a.clone()),
                    _ => e.filter(|e| e.len() > 1).and_then(|e| e[1]["options"].as_array().cloned()),
                };
                if let Some(opts) = options
                    && !opts.is_empty()
                    && !opts.iter().any(|o| o == value)
                {
                    return false;
                }
            }
        }
    }
    true
}

/// The workflow's own steps, CFG and negative prompt, the phone's starting values.
pub fn defaults(spec: &Value, workflow: &Value, info: &Value) -> Map<String, Value> {
    let mut found = Map::new();
    let Ok((prompt, named)) = to_api(workflow, info) else { return found };
    for name in ["steps", "cfg", "negative"] {
        for sel in spec[name].as_array().into_iter().flatten().filter_map(Value::as_str) {
            for (node, field) in targets(&prompt, &named, sel) {
                let v = &prompt[&node]["inputs"][&field];
                let value = match v {
                    Value::Number(n) if n.is_f64() => json!((n.as_f64().unwrap() * 100.0).round_ties_even() / 100.0),
                    Value::Number(_) | Value::String(_) => v.clone(),
                    _ => continue,
                };
                found.entry(name).or_insert(value);
            }
        }
    }
    found
}

/// The phone's list: what this PC can make, with the controls each one takes.
/// A workflow file and when it last changed.
type DefaultsKey = (String, Option<std::time::SystemTime>);

pub struct Catalog {
    pub paths: Paths,
    table: Vec<Value>,
    defaults: Mutex<HashMap<DefaultsKey, Map<String, Value>>>,
}

impl Catalog {
    pub fn new(paths: Paths) -> Catalog {
        Catalog { paths, table: generators_table(), defaults: Mutex::new(HashMap::new()) }
    }

    pub fn find(&self, id: &str) -> Option<&Value> {
        self.table.iter().find(|g| g["id"] == id)
    }

    pub fn workflow(&self, spec: &Value) -> Option<Value> {
        let p = self.paths.workflows().join(spec["workflow"].as_str()?);
        serde_json::from_slice(&std::fs::read(p).ok()?).ok()
    }

    pub fn list(&self, info: Option<&Value>) -> Vec<Value> {
        let mut out = vec![];
        for g in &self.table {
            let wf = self.workflow(g);
            if !available(wf.as_ref(), info) {
                continue;
            }
            let mut item = Map::new();
            for k in ["id", "name", "kind", "mode", "note"] {
                item.insert(k.into(), g[k].clone());
            }
            item.insert("image".into(), g.get("image").cloned().unwrap_or("none".into()));
            item.insert("audio".into(), g.get("audio").cloned().unwrap_or("none".into()));
            for k in ["prompt", "negative", "lyrics"] {
                item.insert(k.into(), g.get(k).is_some().into());
            }
            item.insert("aspects".into(), if g.get("size").is_some() { json!(ASPECTS.iter().map(|a| a.0).collect::<Vec<_>>()) } else { json!([]) });
            item.insert("seed".into(), g.get("seed").is_some().into());
            let d = match (info, &wf) {
                (Some(info), Some(wf)) => self.defaults_for(g, wf, info),
                _ => Map::new(),
            };
            if let Some(v) = d.get("steps") {
                item.insert("steps".into(), json!({"default": v, "min": STEPS_RANGE.0, "max": STEPS_RANGE.1}));
            }
            if let Some(v) = d.get("cfg") {
                item.insert("cfg".into(), json!({"default": v, "min": CFG_RANGE.0, "max": CFG_RANGE.1}));
            }
            if let Some(v) = d.get("negative") {
                item.insert("negative_default".into(), v.clone());
            }
            if let Some(s) = g.get("seconds") {
                item.insert("seconds".into(), json!({"default": s["default"], "max": s["max"]}));
            }
            let voices: Vec<Value> = g["voices"].as_array().into_iter().flatten().filter(|v| v.as_str().and_then(|n| self.paths.voice_file(n)).is_some()).cloned().collect();
            if !voices.is_empty() {
                item.insert("voices".into(), voices.into());
            }
            if g.get("language").is_some()
                && let Some(langs) = info.and_then(|i| i["FL_ChatterboxMultilingualTTS"]["input"]["required"]["language"].as_array()).and_then(|e| e.first()).and_then(Value::as_array)
            {
                item.insert("languages".into(), langs.clone().into());
            }
            out.push(Value::Object(item));
        }
        out
    }

    fn defaults_for(&self, spec: &Value, wf: &Value, info: &Value) -> Map<String, Value> {
        let path = self.paths.workflows().join(spec["workflow"].as_str().unwrap_or(""));
        let key = (spec["id"].as_str().unwrap_or("").to_string(), std::fs::metadata(&path).and_then(|m| m.modified()).ok());
        self.defaults.lock().unwrap().entry(key).or_insert_with(|| defaults(spec, wf, info)).clone()
    }
}

// --- Files ----------------------------------------------------------------------------------------

/// The file type from its first bytes (what the phone uploaded).
pub fn suffix(data: &[u8], default: &str) -> String {
    if data.get(4..8) == Some(b"ftyp") {
        return ".m4a".into();
    }
    let sig: [(&[u8], &str); 6] = [(b"\x89PNG", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"RIFF", ".wav"), (b"ID3", ".mp3"), (b"fLaC", ".flac"), (b"OggS", ".ogg")];
    for (magic, s) in sig {
        if data.starts_with(magic) {
            return if magic == b"RIFF" && data.get(8..12) == Some(b"WEBP") { ".webp".into() } else { s.into() };
        }
    }
    default.into()
}

pub fn mime(suffix: &str) -> &'static str {
    match suffix {
        ".png" => "image/png",
        ".jpg" | ".jpeg" => "image/jpeg",
        ".webp" => "image/webp",
        ".mp4" => "video/mp4",
        ".webm" => "video/webm",
        ".gif" => "image/gif",
        ".flac" => "audio/flac",
        ".mp3" => "audio/mpeg",
        ".wav" => "audio/wav",
        ".opus" | ".ogg" => "audio/ogg",
        _ => "application/octet-stream",
    }
}

/// image, video or audio (a GIF counts as a video).
pub fn kind(suffix: &str) -> &'static str {
    match mime(suffix) {
        "image/gif" => "video",
        m => m.split('/').next().unwrap_or("application"),
    }
}

/// Writes a file the user saved on the phone into the save folder (Images, Videos or Music) and
/// returns its path there. Names are cleaned and never overwrite: "x (1).png".
pub fn save(save_dir: &Path, kind: &str, name: &str, body: &mut dyn std::io::Read, length: u64) -> Result<String, String> {
    let folder = match kind {
        "image" => "Images",
        "video" => "Videos",
        "audio" => "Music",
        _ => return Err("Unknown kind".into()),
    };
    if length == 0 || length > MAX_SAVE {
        return Err("File too large".into());
    }
    let base = Path::new(name).file_name().and_then(|n| n.to_str()).unwrap_or("");
    let cleaned: String = base.chars().map(|c| if c.is_ascii_alphanumeric() || "._ -".contains(c) { c } else { '_' }).collect();
    let stem = cleaned.trim_matches(|c| c == ' ' || c == '.');
    let stem = if stem.is_empty() { "cubicai" } else { stem };
    let dir = save_dir.join(folder);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (front, ext) = match Path::new(stem).extension().and_then(|e| e.to_str()) {
        Some(e) if !e.is_empty() => (&stem[..stem.len() - e.len() - 1], format!(".{e}")),
        _ => (stem, String::new()),
    };
    let mut path = dir.join(stem);
    let mut n = 1;
    let mut file = loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(f) => break f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                path = dir.join(format!("{front} ({n}){ext}"));
                n += 1;
            }
            Err(e) => return Err(e.to_string()),
        }
    };
    let copied = std::io::copy(&mut std::io::Read::take(body, length), &mut file);
    if copied.as_ref().ok() != Some(&length) {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return Err("Upload cut short".into());
    }
    Ok(path.strip_prefix(save_dir).unwrap_or(&path).to_string_lossy().into_owned())
}
