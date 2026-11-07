//! Fine-tuning: a private Python environment (made with uv) holding Unsloth, the training run
//! with its live events, and the trained adapters (PEFT files plus the GGUF llama.cpp loads).

use super::Task;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub const TRAIN_PY: &str = include_str!("train.py");
pub const PYTHON_VERSION: &str = "3.12";

// --- the environment ---------------------------------------------------------------------------

pub struct Env {
    pub dir: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnvInfo {
    pub torch: String,
    pub cuda: bool,
    pub gpu: String,
    pub vram: u64,
    pub unsloth: String,
}

impl Env {
    pub fn new(dir: PathBuf) -> Env {
        Env { dir }
    }

    pub fn venv(&self) -> PathBuf {
        self.dir.join("venv")
    }

    pub fn python(&self) -> PathBuf {
        if cfg!(windows) { self.venv().join("Scripts").join("python.exe") } else { self.venv().join("bin").join("python") }
    }

    pub fn exists(&self) -> bool {
        self.python().is_file()
    }

    /// uv on the PATH, else the copy we downloaded.
    pub fn uv(&self) -> Option<PathBuf> {
        let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
        std::env::var_os("PATH")
            .into_iter()
            .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
            .map(|d| d.join(exe))
            .chain([self.dir.join("uv").join(exe)])
            .find(|p| p.is_file())
    }

    /// Fetches uv from its GitHub releases into `dir/uv`.
    fn install_uv(&self, task: &Arc<Task>) -> Result<PathBuf, String> {
        let asset = uv_asset().ok_or("uv isn't published for this computer; install it yourself (https://docs.astral.sh/uv/).")?;
        task.set_status("Downloading uv (the Python installer)".into());
        let dest = self.dir.join("uv");
        let _ = std::fs::create_dir_all(&dest);
        let file = dest.join(asset);
        super::release::download(&format!("https://github.com/astral-sh/uv/releases/latest/download/{asset}"), &file, |_| {}, task)?;
        super::release::unpack(&file, &dest)?;
        let _ = std::fs::remove_file(&file);
        // The archives hold uv in a folder named after the target; move it up.
        let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
        for e in std::fs::read_dir(&dest).map_err(|e| e.to_string())?.flatten() {
            let inner = e.path().join(exe);
            if inner.is_file() {
                std::fs::rename(&inner, dest.join(exe)).map_err(|e| e.to_string())?;
            }
        }
        self.uv().ok_or_else(|| "uv was downloaded but can't be found".into())
    }

    /// Makes the environment: Python 3.12 and Unsloth with the PyTorch build for this GPU.
    pub fn setup(&self, task: &Arc<Task>) -> Result<EnvInfo, String> {
        if cfg!(target_os = "macos") {
            return Err("Training needs an NVIDIA GPU; Unsloth doesn't run on a Mac.".into());
        }
        let uv = match self.uv() {
            Some(u) => u,
            None => self.install_uv(task)?,
        };
        let _ = std::fs::create_dir_all(&self.dir);
        task.set_status(format!("Making a Python {PYTHON_VERSION} environment"));
        task.progress.store(50, Ordering::Relaxed);
        run_logged(Command::new(&uv).args(["venv", "--python", PYTHON_VERSION, "--seed"]).arg(self.venv()), task, &mut |_| {})?;
        task.set_status("Installing Unsloth and PyTorch (several GB, the first time takes a while)".into());
        task.progress.store(150, Ordering::Relaxed);
        let mut seen = 0u32;
        run_logged(
            Command::new(&uv).args(["pip", "install", "--torch-backend=auto", "--python"]).arg(self.python()).arg("unsloth"),
            task,
            &mut |line| {
                // uv prints a line per resolved/downloaded package; creep the bar along.
                seen += 1;
                task.progress.store((150 + seen.min(800)).min(950), Ordering::Relaxed);
                if !line.trim().is_empty() {
                    task.set_status(line.trim().chars().take(120).collect());
                }
            },
        )?;
        task.set_status("Checking the GPU from Python".into());
        let info = self.check()?;
        task.progress.store(1000, Ordering::Relaxed);
        Ok(info)
    }

    /// What the environment has: PyTorch, whether it sees the GPU, Unsloth's version.
    pub fn check(&self) -> Result<EnvInfo, String> {
        if !self.exists() {
            return Err("Not set up yet.".into());
        }
        let script = "import json, importlib.metadata as m\n\
            import torch\n\
            c = torch.cuda.is_available()\n\
            def v(p):\n    try:\n        return m.version(p)\n    except Exception:\n        return ''\n\
            print('CUBIC ' + json.dumps({'torch': torch.__version__, 'cuda': c, 'gpu': torch.cuda.get_device_name(0) if c else '', \
            'vram': torch.cuda.get_device_properties(0).total_memory if c else 0, 'unsloth': v('unsloth')}))";
        let out = Command::new(self.python()).args(["-c", script]).output().map_err(|e| format!("Couldn't run Python: {e}"))?;
        let text = String::from_utf8_lossy(&out.stdout);
        parse_env_info(&text).ok_or_else(|| {
            let err = String::from_utf8_lossy(&out.stderr);
            format!("The environment is broken: {}", err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no output"))
        })
    }
}

pub fn parse_env_info(stdout: &str) -> Option<EnvInfo> {
    let v: Value = stdout.lines().find_map(|l| l.strip_prefix("CUBIC ")).and_then(|j| serde_json::from_str(j).ok())?;
    Some(EnvInfo {
        torch: v["torch"].as_str()?.into(),
        cuda: v["cuda"].as_bool().unwrap_or(false),
        gpu: v["gpu"].as_str().unwrap_or("").into(),
        vram: v["vram"].as_u64().unwrap_or(0),
        unsloth: v["unsloth"].as_str().unwrap_or("").into(),
    })
}

fn uv_asset() -> Option<&'static str> {
    Some(match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "uv-x86_64-unknown-linux-gnu.tar.gz",
        ("linux", "aarch64") => "uv-aarch64-unknown-linux-gnu.tar.gz",
        ("windows", "x86_64") => "uv-x86_64-pc-windows-msvc.zip",
        ("windows", "aarch64") => "uv-aarch64-pc-windows-msvc.zip",
        _ => return None,
    })
}

/// Runs a command, handing every output line (stdout and stderr) to `on_line`. The task can stop
/// it. On failure the error carries the last lines of output.
pub fn run_logged(cmd: &mut Command, task: &Arc<Task>, on_line: &mut dyn FnMut(&str)) -> Result<(), String> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("Couldn't start {}: {e}", cmd.get_program().to_string_lossy()))?;
    *task.child_pid.lock().unwrap() = Some(child.id());
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let readers: Vec<_> = [child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)]
        .into_iter()
        .flatten()
        .map(|r| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(r).lines().map_while(Result::ok) {
                    // Progress bars redraw with \r; keep the latest state.
                    let line = line.rsplit('\r').next().unwrap_or("").to_string();
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(tx);
    let mut tail: VecDeque<String> = VecDeque::new();
    for line in rx {
        on_line(&line);
        if !line.trim().is_empty() {
            tail.push_back(line);
            if tail.len() > 12 {
                tail.pop_front();
            }
        }
    }
    for r in readers {
        let _ = r.join();
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    *task.child_pid.lock().unwrap() = None;
    if task.cancel.load(Ordering::Relaxed) {
        return Err("Stopped".into());
    }
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed ({status}):\n{}", cmd.get_program().to_string_lossy(), tail.into_iter().collect::<Vec<_>>().join("\n")))
    }
}

// --- a training run ----------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Params {
    pub rank: u32,
    pub alpha: f32,
    pub lr: f32,
    pub epochs: f32,
    pub max_seq: u32,
    pub batch: u32,
    pub grad_accum: u32,
    pub seed: u64,
    pub replies_only: bool,
    /// "sft" learns the examples' replies; "dpo" learns to prefer the chosen reply of each pair.
    pub method: String,
    /// DPO: how far the model may move from where it started (lower lets it move more).
    pub beta: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { rank: 16, alpha: 16.0, lr: 2e-4, epochs: 3.0, max_seq: 1024, batch: 2, grad_accum: 4, seed: 3407, replies_only: true, method: "sft".into(), beta: 0.1 }
    }
}

impl Params {
    pub fn dpo(&self) -> bool {
        self.method == "dpo"
    }
}

/// What the trainer script reads.
pub fn config_json(base: &str, data: &Path, out: &Path, p: &Params) -> Value {
    serde_json::json!({
        "base": base,
        "data": data,
        "out": out,
        "rank": p.rank,
        "alpha": p.alpha,
        "lr": p.lr,
        "epochs": p.epochs,
        "max_seq": p.max_seq,
        "batch": p.batch,
        "grad_accum": p.grad_accum,
        "seed": p.seed,
        "replies_only": p.replies_only,
        "method": if p.dpo() { "dpo" } else { "sft" },
        "beta": p.beta,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum TEv {
    Status(String),
    Progress { step: u32, total: u32 },
    Loss { step: u32, loss: f32, lr: f32 },
    Done { adapter: PathBuf },
    Error(String),
}

/// One greedy reply with the adapter at `scale` (0 = the base model), without restarting.
pub fn complete_with_lora(root: &str, prompt: &str, n_predict: u32, scale: f32) -> Result<String, String> {
    let resp = crate::llm::agent(5)
        .post(&format!("{root}/completion"))
        .send_json(serde_json::json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 1,
            "cache_prompt": false,
            "lora": [{"id": 0, "scale": scale}],
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

/// One line from the trainer: an event, or `None` for plain log output.
pub fn parse_event(line: &str) -> Option<TEv> {
    let v: Value = serde_json::from_str(line.strip_prefix("CUBIC ")?).ok()?;
    let u = |k: &str| v[k].as_u64().unwrap_or(0) as u32;
    Some(match v["event"].as_str()? {
        "status" => TEv::Status(v["text"].as_str()?.into()),
        "progress" => TEv::Progress { step: u("step"), total: u("total") },
        "loss" => TEv::Loss { step: u("step"), loss: v["loss"].as_f64()? as f32, lr: v["lr"].as_f64().unwrap_or(0.0) as f32 },
        "done" => TEv::Done { adapter: PathBuf::from(v["adapter"].as_str()?) },
        "error" => TEv::Error(v["text"].as_str()?.into()),
        _ => return None,
    })
}

/// What the UI shows while training.
#[derive(Default)]
pub struct Live {
    pub status: String,
    pub step: u32,
    pub total: u32,
    pub losses: Vec<(u32, f32)>,
    pub log: VecDeque<String>,
    pub adapter: Option<PathBuf>,
    pub error: Option<String>,
}

impl Live {
    pub fn apply(&mut self, line: &str) {
        match parse_event(line) {
            Some(TEv::Status(s)) => self.status = s,
            Some(TEv::Progress { step, total }) => (self.step, self.total) = (step, total),
            Some(TEv::Loss { step, loss, .. }) => self.losses.push((step, loss)),
            Some(TEv::Done { adapter }) => self.adapter = Some(adapter),
            Some(TEv::Error(e)) => self.error = Some(e),
            None if !line.trim().is_empty() => {
                self.log.push_back(line.to_string());
                if self.log.len() > 200 {
                    self.log.pop_front();
                }
            }
            None => {}
        }
    }
}

/// Runs the trainer in `env` for the run folder `out` (holding train.json and data.jsonl).
pub fn run(python: &Path, out: &Path, hf_token: &str, live: &Arc<Mutex<Live>>, task: &Arc<Task>) -> Result<PathBuf, String> {
    let script = out.join("train.py");
    std::fs::write(&script, TRAIN_PY).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(python);
    cmd.arg("-u").arg(&script).arg(out.join("train.json")).current_dir(out).env("PYTHONUNBUFFERED", "1");
    if !hf_token.is_empty() {
        cmd.env("HF_TOKEN", hf_token);
    }
    let r = run_logged(&mut cmd, task, &mut |line| {
        let mut l = live.lock().unwrap();
        l.apply(line);
        if let Some(f) = (l.step * 1000).checked_div(l.total) {
            task.progress.store(f.min(1000), Ordering::Relaxed);
        }
        task.set_status(l.status.clone());
    });
    let l = live.lock().unwrap();
    if let Some(e) = &l.error {
        return Err(e.clone());
    }
    r?;
    l.adapter.clone().ok_or_else(|| "The trainer finished without saving an adapter.".into())
}

// --- trained adapters --------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Adapter {
    pub id: String,
    pub name: String,
    pub dataset: String,
    /// The dataset's id, to find its held-back examples for the report card.
    pub dataset_id: String,
    pub examples: usize,
    pub base_repo: String,
    /// The GGUF model (library key) it was converted for.
    pub model_key: String,
    pub model_name: String,
    pub params: Params,
    pub losses: Vec<(u32, f32)>,
    pub secs: f32,
    pub created: u64,
    /// The system prompt the dataset trained with.
    pub system: String,
    /// Weights the adapter adds, and how many weight matrices it adapts.
    pub lora_params: u64,
    pub lora_pairs: usize,
}

pub struct Adapters {
    pub dir: PathBuf,
}

impl Adapters {
    pub fn new(dir: PathBuf) -> Adapters {
        let _ = std::fs::create_dir_all(&dir);
        Adapters { dir }
    }

    pub fn run_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    pub fn gguf(&self, id: &str) -> PathBuf {
        self.run_dir(id).join("lora.gguf")
    }

    pub fn save(&self, a: &Adapter) -> Result<(), String> {
        let dir = self.run_dir(&a.id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let j = serde_json::to_vec_pretty(a).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&dir.join("adapter.json"), &j).map_err(|e| e.to_string())
    }

    /// Adapters that made it to a GGUF, newest first.
    pub fn list(&self) -> Vec<Adapter> {
        let mut out: Vec<Adapter> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| serde_json::from_slice::<Adapter>(&std::fs::read(e.path().join("adapter.json")).ok()?).ok())
            .filter(|a| self.gguf(&a.id).is_file())
            .collect();
        out.sort_by_key(|a| std::cmp::Reverse(a.created));
        out
    }

    pub fn delete(&self, id: &str) {
        if !id.is_empty() && !id.contains(['/', '\\', '.']) {
            let _ = std::fs::remove_dir_all(self.run_dir(id));
        }
    }

    /// Converts the run's PEFT adapter to `lora.gguf` for the given base model header.
    pub fn convert(&self, id: &str, peft_dir: &Path, base: &cubic_gguf::Gguf) -> Result<super::lora::Converted, String> {
        let cfg = std::fs::read_to_string(peft_dir.join("adapter_config.json")).map_err(|e| format!("No adapter_config.json: {e}"))?;
        let cfg = super::lora::read_config(&cfg)?;
        let bytes = std::fs::read(peft_dir.join("adapter_model.safetensors")).map_err(|e| format!("No adapter_model.safetensors: {e}"))?;
        let tensors = super::lora::read_safetensors(&bytes)?;
        let c = super::lora::convert(&tensors, &cfg, base)?;
        std::fs::create_dir_all(self.run_dir(id)).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&self.gguf(id), &c.gguf).map_err(|e| e.to_string())?;
        Ok(c)
    }
}

/// Unsloth's pre-quantised Hugging Face copy of a GGUF model, when we recognise it. The LoRA must
/// be trained on the same model the GGUF was made from.
pub fn suggest_base(name: &str) -> Option<&'static str> {
    let n: String = name.to_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let has = |parts: &[&str]| parts.iter().all(|p| n.contains(p));
    const TABLE: &[(&[&str], &str)] = &[
        (&["deepseek", "r1", "qwen3", "8b"], "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"),
        (&["llama31", "8b"], "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
        (&["llama32", "3b"], "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"),
        (&["llama32", "1b"], "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"),
        (&["qwen25", "14b"], "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"),
        (&["qwen25", "7b"], "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
        (&["qwen25", "3b"], "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"),
        (&["qwen25", "15b"], "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"),
        (&["qwen25", "05b"], "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"),
        (&["qwen3", "14b"], "unsloth/Qwen3-14B-bnb-4bit"),
        (&["qwen3", "8b"], "unsloth/Qwen3-8B-bnb-4bit"),
        (&["qwen3", "4b"], "unsloth/Qwen3-4B-bnb-4bit"),
        (&["qwen3", "17b"], "unsloth/Qwen3-1.7B-bnb-4bit"),
        (&["qwen3", "06b"], "unsloth/Qwen3-0.6B-bnb-4bit"),
        (&["mistral", "7b", "v03"], "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"),
    ];
    TABLE.iter().find(|(parts, _)| has(parts)).map(|(_, repo)| *repo)
}
