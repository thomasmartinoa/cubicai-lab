//! Runs llama-server as a child process: arguments from the model's preset, its log kept in
//! memory, readiness from `/health`. The child dies with the app.

use crate::store::RunPreset;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const LOG_LINES: usize = 600;

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Stopped,
    Loading,
    Ready,
    Failed(String),
}

pub struct Server {
    child: Option<Child>,
    shared: Arc<Shared>,
    pub model_key: Option<String>,
    pub model_name: String,
    pub port: u16,
    pub started: Option<Instant>,
    pub ready_after: Option<Duration>,
    pub args: Vec<String>,
    /// The baked control vector the running model was started with.
    pub steer: Option<std::path::PathBuf>,
    /// The LoRA adapter it was started with.
    pub lora: Option<std::path::PathBuf>,
}

struct Shared {
    state: Mutex<State>,
    log: Mutex<VecDeque<String>>,
    stop: AtomicBool,
}

pub struct Launch<'a> {
    pub bin_dir: &'a Path,
    pub model_key: &'a str,
    pub model_name: &'a str,
    pub model: &'a Path,
    pub mmproj: Option<&'a Path>,
    pub preset: &'a RunPreset,
    pub port: u16,
    /// A baked control vector (Steering Lab) to add to the model's hidden states.
    pub steer: Option<&'a Path>,
    /// A LoRA adapter (Fine-tune Studio). Requests can set its scale (`"lora": [{"id": 0, …}]`).
    pub lora: Option<&'a Path>,
}

impl Server {
    pub fn new() -> Self {
        Server {
            child: None,
            shared: Arc::new(Shared { state: Mutex::new(State::Stopped), log: Mutex::new(VecDeque::new()), stop: AtomicBool::new(false) }),
            model_key: None,
            model_name: String::new(),
            port: 0,
            started: None,
            ready_after: None,
            args: vec![],
            steer: None,
            lora: None,
        }
    }

    pub fn state(&self) -> State {
        self.shared.state.lock().unwrap().clone()
    }

    pub fn is_ready_with(&self, key: &str) -> bool {
        self.state() == State::Ready && self.model_key.as_deref() == Some(key)
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub fn log_lines(&self) -> Vec<String> {
        self.shared.log.lock().unwrap().iter().cloned().collect()
    }

    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().map(|c| c.id())
    }

    /// Arguments for a preset (also shown to the user, and reused by the benchmark).
    pub fn args_for(l: &Launch) -> Vec<String> {
        let p = l.preset;
        let mut a: Vec<String> = vec![
            "-m".into(),
            l.model.to_string_lossy().into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            l.port.to_string(),
            "-c".into(),
            p.ctx.to_string(),
            "-np".into(),
            "1".into(),
            "-fa".into(),
            p.flash_attn.arg().into(),
            "-ctk".into(),
            p.kv_type.clone(),
            "-ctv".into(),
            p.kv_type.clone(),
            "-ub".into(),
            p.ubatch.to_string(),
            "--alias".into(),
            l.model_name.to_string(),
            "--no-webui".into(),
        ];
        match p.gpu_layers {
            Some(n) => {
                a.extend(["-ngl".into(), n.to_string()]);
                // A fixed layer count means the user decided; llama.cpp shouldn't refit it.
                a.extend(["-fit".into(), "off".into()]);
            }
            None => a.extend(["-ngl".into(), "auto".into()]),
        }
        if let Some(n) = p.cpu_moe_layers {
            a.extend(["-ncmoe".into(), n.to_string()]);
        }
        if let Some(t) = p.threads {
            a.extend(["-t".into(), t.to_string()]);
        }
        match l.mmproj {
            Some(mm) if p.use_mmproj => a.extend(["--mmproj".into(), mm.to_string_lossy().into()]),
            _ => a.push("--no-mmproj".into()),
        }
        if let Some(cv) = l.steer {
            a.extend(["--control-vector".into(), cv.to_string_lossy().into()]);
        }
        if let Some(lora) = l.lora {
            a.extend(["--lora".into(), lora.to_string_lossy().into()]);
        }
        a.extend(shell_words(&p.extra_args));
        a
    }

    pub fn start(&mut self, l: Launch) -> Result<(), String> {
        self.stop();
        let exe = super::release::server_exe(l.bin_dir);
        if !exe.is_file() {
            return Err("llama.cpp isn't installed yet. Install it in the Engine Room.".into());
        }
        let port = free_port(l.port);
        let l = Launch { port, ..l };
        let args = Self::args_for(&l);
        let mut cmd = Command::new(&exe);
        cmd.args(&args).current_dir(l.bin_dir).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        add_lib_path(&mut cmd, l.bin_dir);
        die_with_parent(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| format!("Couldn't start llama-server: {e}"))?;
        let shared = Arc::new(Shared {
            state: Mutex::new(State::Loading),
            log: Mutex::new(VecDeque::new()),
            stop: AtomicBool::new(false),
        });
        shared.log.lock().unwrap().push_back(format!("$ llama-server {}", args.join(" ")));
        for out in [child.stdout.take().map(|o| Box::new(o) as Box<dyn std::io::Read + Send>), child.stderr.take().map(|e| Box::new(e) as _)]
            .into_iter()
            .flatten()
        {
            let sh = shared.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    let mut log = sh.log.lock().unwrap();
                    if log.len() >= LOG_LINES {
                        log.pop_front();
                    }
                    log.push_back(line);
                }
            });
        }
        // Readiness: /health answers 503 while loading and 200 when ready.
        let sh = shared.clone();
        let health = format!("http://127.0.0.1:{port}/health");
        std::thread::spawn(move || {
            let agent = crate::llm::agent(2);
            while !sh.stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(300));
                if *sh.state.lock().unwrap() != State::Loading {
                    break;
                }
                if let Ok(r) = agent.get(&health).call()
                    && r.status() == 200 {
                        *sh.state.lock().unwrap() = State::Ready;
                        break;
                    }
            }
        });
        self.child = Some(child);
        self.shared = shared;
        self.model_key = Some(l.model_key.to_string());
        self.model_name = l.model_name.to_string();
        self.port = port;
        self.started = Some(Instant::now());
        self.ready_after = None;
        self.args = args;
        self.steer = l.steer.map(Path::to_path_buf);
        self.lora = l.lora.map(Path::to_path_buf);
        Ok(())
    }

    /// Notices a crashed or finished child and the moment it became ready. Call every frame.
    pub fn poll(&mut self) {
        if let Some(c) = &mut self.child
            && let Ok(Some(status)) = c.try_wait() {
                let tail: Vec<String> = self.shared.log.lock().unwrap().iter().rev().take(6).rev().cloned().collect();
                let why = explain_failure(&tail.join("\n"));
                *self.shared.state.lock().unwrap() = State::Failed(format!("llama-server exited ({status}). {why}"));
                self.child = None;
            }
        if self.ready_after.is_none() && self.state() == State::Ready {
            self.ready_after = self.started.map(|s| s.elapsed());
        }
    }

    pub fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(mut c) = self.child.take() {
            terminate(&mut c);
        }
        *self.shared.state.lock().unwrap() = State::Stopped;
        self.model_key = None;
        self.started = None;
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(crate) fn explain_failure(tail: &str) -> String {
    let t = tail.to_lowercase();
    if t.contains("out of memory") || t.contains("failed to allocate") || t.contains("cudamalloc") {
        "It ran out of GPU memory: lower the context, the GPU layers, or keep more experts on the CPU.".into()
    } else if t.contains("unknown model architecture") || t.contains("wrong number of tensors") || t.contains("missing tensor") {
        "This llama.cpp build can't load this model (unsupported architecture or Ollama-specific packaging).".into()
    } else if t.contains("error while loading shared libraries") || t.contains("cannot open shared object") {
        "A library is missing: for CUDA builds, install the CUDA runtime (or reinstall with it included).".into()
    } else if t.contains("address already in use") {
        "The port is taken; change it in Settings.".into()
    } else {
        format!("Last lines:\n{tail}")
    }
}

/// The preferred port if free, else the next free one.
fn free_port(preferred: u16) -> u16 {
    (preferred..preferred.saturating_add(50))
        .find(|p| std::net::TcpListener::bind(("127.0.0.1", *p)).is_ok())
        .unwrap_or(preferred)
}

pub fn add_lib_path(cmd: &mut Command, dir: &Path) {
    let (var, sep) = if cfg!(windows) { ("PATH", ";") } else if cfg!(target_os = "macos") { ("DYLD_LIBRARY_PATH", ":") } else { ("LD_LIBRARY_PATH", ":") };
    let old = std::env::var(var).unwrap_or_default();
    let new = if old.is_empty() { dir.to_string_lossy().to_string() } else { format!("{}{sep}{old}", dir.to_string_lossy()) };
    cmd.env(var, new);
}

/// On Linux the kernel kills the child if the app dies, so a crash never strands a server
/// holding the GPU.
pub fn die_with_parent(cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cmd;
}

fn terminate(c: &mut Child) {
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(c.id() as i32, libc::SIGTERM);
        }
        for _ in 0..30 {
            if let Ok(Some(_)) = c.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let _ = c.kill();
    let _ = c.wait();
}

/// Splits extra arguments like a shell would for simple cases (quotes group words).
pub fn shell_words(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for ch in s.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"' | '\'') => {
                quote = Some(ch);
                any = true;
            }
            (None, c) if c.is_whitespace() => {
                if any || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if any || !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn bin_dir_valid(dir: &Path) -> bool {
    super::release::server_exe(dir).is_file()
}

pub fn path_display(p: &Path) -> String {
    let home = directories::BaseDirs::new().map(|b| b.home_dir().to_path_buf()).unwrap_or_default();
    match p.strip_prefix(&home) {
        Ok(rest) if !home.as_os_str().is_empty() => format!("~/{}", rest.to_string_lossy()),
        _ => p.to_string_lossy().to_string(),
    }
}
