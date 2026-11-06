//! Running creations: ComfyUI started on demand (its own systemd unit) and stopped when idle, one
//! job at a time, progress and previews from ComfyUI's WebSocket, results moved to a private temp
//! folder until the phone collects them. A port of the job half of `cubicai_media.py`.

use crate::media::{self, Catalog, Paths, SUBFOLDER};
use base64::Engine;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub const TERMINAL: [&str; 3] = ["done", "error", "cancelled"];
pub const UNDELIVERED_TTL_S: f64 = 3600.0;
const START_TIMEOUT: Duration = Duration::from_secs(180);

fn log(msg: &str) {
    crate::log(&format!("media: {msg}"));
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    let _ = ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut b);
    b
}

/// A seed in 0..=2^32-1.
pub fn random_seed() -> u64 {
    u32::from_le_bytes(random_bytes::<4>()) as u64
}

// --- ComfyUI ----------------------------------------------------------------------------------------

pub enum ComfyError {
    /// ComfyUI answered with an error status and this body.
    Status(u16, Vec<u8>),
    Other(String),
}

impl std::fmt::Display for ComfyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComfyError::Status(c, _) => write!(f, "HTTP {c}"),
            ComfyError::Other(m) => f.write_str(m),
        }
    }
}

/// ComfyUI on this PC: started when a job needs it, stopped again when idle (only if we started it).
pub struct Comfy {
    pub paths: Paths,
    ollama: String,
    agent: ureq::Agent,
    lock: Mutex<()>,
    started_by_us: AtomicBool,
    last_used: Mutex<Instant>,
    pub busy: AtomicUsize,
    /// ComfyUI holds models in VRAM that chat (Ollama) may need back.
    pub dirty: AtomicBool,
    info: Mutex<Option<Value>>,
}

fn systemctl(args: &[&str]) -> std::process::Output {
    std::process::Command::new("systemctl").arg("--user").args(args).output().unwrap_or_else(|_| std::process::Output { status: std::process::ExitStatus::default(), stdout: vec![], stderr: b"systemctl isn't available".to_vec() })
}

impl Comfy {
    pub fn new(paths: Paths, ollama: &str) -> Arc<Comfy> {
        let agent = ureq::Agent::config_builder().http_status_as_error(false).proxy(None).build().new_agent();
        let c = Arc::new(Comfy {
            // Only CubicAI starts this unit (a ComfyUI opened by hand runs outside it), so one left
            // running by an earlier gateway is ours to stop too.
            started_by_us: AtomicBool::new(false),
            paths,
            ollama: ollama.trim_end_matches('/').to_string(),
            agent,
            lock: Mutex::new(()),
            last_used: Mutex::new(Instant::now()),
            busy: AtomicUsize::new(0),
            dirty: AtomicBool::new(false),
            info: Mutex::new(None),
        });
        c.started_by_us.store(c.unit_active(), Ordering::SeqCst);
        let w = Arc::downgrade(&c);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(30));
                let Some(c) = w.upgrade() else { return };
                c.idle_check();
            }
        });
        c
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.paths.comfy_port)
    }

    /// A call to ComfyUI: JSON back (or Null for anything else).
    pub fn call(&self, path: &str, body: Option<&Value>, timeout: Duration) -> Result<Value, ComfyError> {
        let r = match body {
            Some(b) => self.agent.post(self.url(path)).config().timeout_global(Some(timeout)).build().send_json(b),
            None => self.agent.get(self.url(path)).config().timeout_global(Some(timeout)).build().call(),
        };
        let mut r = r.map_err(|e| ComfyError::Other(e.to_string()))?;
        let raw = r.body_mut().with_config().limit(256 << 20).read_to_vec().map_err(|e| ComfyError::Other(e.to_string()))?;
        if !r.status().is_success() {
            return Err(ComfyError::Status(r.status().as_u16(), raw));
        }
        Ok(if raw.first().is_some_and(|b| *b == b'{' || *b == b'[') { serde_json::from_slice(&raw).unwrap_or(Value::Null) } else { Value::Null })
    }

    pub fn running(&self) -> bool {
        self.call("/system_stats", None, Duration::from_secs(2)).is_ok()
    }

    fn unit_state(&self) -> String {
        String::from_utf8_lossy(&systemctl(&["show", "-p", "ActiveState", "--value", &self.paths.unit]).stdout).trim().to_string()
    }

    pub fn unit_active(&self) -> bool {
        systemctl(&["is-active", "--quiet", &self.paths.unit]).status.success()
    }

    pub fn mark_used(&self) {
        *self.last_used.lock().unwrap() = Instant::now();
    }

    pub fn ensure(&self, on_status: &dyn Fn(&str)) -> Result<(), String> {
        if self.running() {
            return Ok(());
        }
        let _g = self.lock.lock().unwrap();
        if self.running() {
            return Ok(());
        }
        on_status("Starting ComfyUI on your PC…");
        log("starting ComfyUI");
        // Right after a crash systemd can still call the dying unit "active" for a moment, so this
        // start is a no-op and the unit turns "failed" just after. The first failure seen is that
        // leftover: clear it and start again; only a second one is this start failing.
        let deadline = Instant::now() + START_TIMEOUT;
        'attempts: for _ in 0..2 {
            systemctl(&["reset-failed", &self.paths.unit]);
            let r = systemctl(&["start", &self.paths.unit]);
            if !r.status.success() {
                let e = String::from_utf8_lossy(&r.stderr).trim().to_string();
                return Err(format!("Couldn't start ComfyUI on your PC: {}", if e.is_empty() { "systemctl failed".into() } else { e }));
            }
            self.started_by_us.store(true, Ordering::SeqCst);
            loop {
                if Instant::now() >= deadline {
                    break 'attempts;
                }
                if self.running() {
                    return Ok(());
                }
                if self.unit_state() == "failed" {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        if self.unit_state() == "failed" {
            return Err("ComfyUI failed to start on your PC. Check `journalctl --user -u cubicai-comfyui`.".into());
        }
        Err("ComfyUI is taking too long to start on your PC.".into())
    }

    pub fn object_info(&self) -> Result<Value, String> {
        if let Some(i) = self.info.lock().unwrap().clone() {
            return Ok(i);
        }
        let i = self.call("/object_info", None, Duration::from_secs(60)).map_err(|e| format!("ComfyUI's node list: {e}"))?;
        let _ = std::fs::write(self.paths.info_cache(), i.to_string());
        *self.info.lock().unwrap() = Some(i.clone());
        Ok(i)
    }

    /// The node list without starting ComfyUI: live if it's running, else the last one seen.
    pub fn known_info(&self) -> Option<Value> {
        if self.info.lock().unwrap().is_none()
            && self.running()
            && let Ok(i) = self.object_info()
        {
            return Some(i);
        }
        if let Some(i) = self.info.lock().unwrap().clone() {
            return Some(i);
        }
        serde_json::from_slice(&std::fs::read(self.paths.info_cache()).ok()?).ok()
    }

    /// Gives the GPU back to Ollama once making things is over.
    pub fn free_for_chat(&self) {
        if self.dirty.load(Ordering::SeqCst) && self.busy.load(Ordering::SeqCst) == 0 && self.running() {
            if self.call("/free", Some(&json!({"unload_models": true, "free_memory": true})), Duration::from_secs(30)).is_err() {
                return;
            }
            self.dirty.store(false, Ordering::SeqCst);
            // A model loaded while ComfyUI held the card sits partly on the CPU (about 7x slower)
            // until it expires. Unloaded now, the next reply loads it onto the card properly.
            free_ollama(&self.ollama, true);
        }
    }

    fn idle_check(&self) {
        let idle = self.last_used.lock().unwrap().elapsed() > Duration::from_secs(self.paths.idle_stop);
        if self.started_by_us.load(Ordering::SeqCst) && self.busy.load(Ordering::SeqCst) == 0 && idle {
            log("stopping ComfyUI after being idle");
            systemctl(&["stop", &self.paths.unit]);
            self.started_by_us.store(false, Ordering::SeqCst);
            self.dirty.store(false, Ordering::SeqCst);
            *self.info.lock().unwrap() = None;
        }
    }
}

/// Unloads Ollama's models so ComfyUI gets the whole GPU (or only those not fully on it).
pub fn free_ollama(ollama: &str, partial_only: bool) {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(30))).proxy(None).build().into();
    let Ok(ps) = agent.get(format!("{ollama}/api/ps")).call().and_then(|mut r| r.body_mut().read_json::<Value>()) else { return };
    for m in ps["models"].as_array().into_iter().flatten() {
        if partial_only && m["size_vram"].as_u64().unwrap_or(0) >= m["size"].as_u64().unwrap_or(0) {
            continue;
        }
        let _ = agent.post(format!("{ollama}/api/generate")).send_json(json!({"model": m["name"], "keep_alive": 0})).and_then(|mut r| r.body_mut().read_to_vec());
    }
}

// --- A minimal WebSocket client (ComfyUI's progress and previews) -----------------------------------

pub struct WsClient {
    sock: TcpStream,
    buf: Vec<u8>,
}

impl WsClient {
    pub fn connect(port: u16, path: &str) -> std::io::Result<WsClient> {
        let mut sock = TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_secs(10))?;
        sock.set_read_timeout(Some(Duration::from_secs(10)))?;
        let key = base64::engine::general_purpose::STANDARD.encode(random_bytes::<16>());
        sock.write_all(format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").as_bytes())?;
        let mut head = vec![];
        let mut chunk = [0u8; 4096];
        let end = loop {
            let n = sock.read(&mut chunk)?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "websocket handshake failed"));
            }
            head.extend_from_slice(&chunk[..n]);
            if let Some(p) = head.windows(4).position(|w| w == b"\r\n\r\n") {
                break p;
            }
        };
        let first = head.split(|b| *b == b'\n').next().unwrap_or(&[]);
        if !first.windows(5).any(|w| w == b" 101 ") {
            return Err(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "websocket handshake refused"));
        }
        Ok(WsClient { sock, buf: head[end + 4..].to_vec() })
    }

    pub fn set_timeout(&self, d: Duration) {
        let _ = self.sock.set_read_timeout(Some(d));
    }

    fn need(&mut self, n: usize) -> std::io::Result<()> {
        let mut chunk = vec![0u8; 65536];
        while self.buf.len() < n {
            let got = self.sock.read(&mut chunk)?;
            if got == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "websocket closed"));
            }
            self.buf.extend_from_slice(&chunk[..got]);
        }
        Ok(())
    }

    fn send(&mut self, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
        // Control frames only: payload < 126.
        let mask = random_bytes::<4>();
        let mut f = vec![0x80 | opcode, 0x80 | payload.len() as u8];
        f.extend_from_slice(&mask);
        f.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.sock.write_all(&f)
    }

    /// The next text (1) or binary (2) message. A timeout leaves partial data buffered.
    pub fn recv(&mut self) -> std::io::Result<(u8, Vec<u8>)> {
        let mut message = vec![];
        let mut kind = 0u8;
        loop {
            self.need(2)?;
            let (b1, b2) = (self.buf[0], self.buf[1]);
            let (fin, opcode, short) = (b1 & 0x80 != 0, b1 & 0x0f, (b2 & 0x7f) as usize);
            let (hlen, n) = match short {
                126 => {
                    self.need(4)?;
                    (4, u16::from_be_bytes([self.buf[2], self.buf[3]]) as usize)
                }
                127 => {
                    self.need(10)?;
                    (10, u64::from_be_bytes(self.buf[2..10].try_into().unwrap()) as usize)
                }
                n => (2, n),
            };
            let masked = b2 & 0x80 != 0;
            let mlen = if masked { 4 } else { 0 };
            self.need(hlen + mlen + n)?;
            let frame: Vec<u8> = self.buf.drain(..hlen + mlen + n).collect();
            let mut payload = frame[hlen + mlen..].to_vec();
            if masked {
                let m = &frame[hlen..hlen + 4];
                for (i, b) in payload.iter_mut().enumerate() {
                    *b ^= m[i % 4];
                }
            }
            match opcode {
                8 => return Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "websocket closed")),
                9 => {
                    self.send(10, &payload)?;
                    continue;
                }
                1 | 2 => {
                    kind = opcode;
                    message = payload;
                }
                0 => message.extend(payload),
                _ => {}
            }
            if fin && kind != 0 {
                return Ok((kind, message));
            }
        }
    }

    pub fn close(&mut self) {
        let _ = self.send(8, &[]);
        let _ = self.sock.shutdown(std::net::Shutdown::Both);
    }
}

// --- Jobs -----------------------------------------------------------------------------------------

const PHASES: [(&str, &str); 12] = [
    ("Loader", "Loading models"),
    ("TextGenerate", "Improving your prompt"),
    ("TextEncode", "Reading your prompt"),
    ("CLIPTextEncode", "Reading your prompt"),
    ("KSampler", "Generating"),
    ("SamplerCustom", "Generating"),
    ("VAEDecode", "Finishing"),
    ("CreateVideo", "Making the video"),
    ("Save", "Saving"),
    ("Chatterbox", "Speaking"),
    ("RemoveBackground", "Cutting out"),
    ("SeedVR2", "Upscaling"),
];
const HEAVY: [&str; 7] = ["Sampler", "TTS", "Chatterbox", "BiRefNet", "SeedVR", "AceStep", "Upscale"];

/// Rough share of a run's time a node takes.
pub fn weight(class: &str) -> f64 {
    if class.contains("Select") {
        return 0.1; // KSamplerSelect only picks a sampler
    }
    if HEAVY.iter().any(|h| class.contains(h)) {
        return 8.0;
    }
    if class.contains("Loader") || class.starts_with("Load") || class.contains("Decode") || class.contains("Encode") {
        return 1.0;
    }
    0.1
}

pub fn phase(class: &str) -> &'static str {
    if class.contains("Select") {
        return "Working";
    }
    PHASES.iter().find(|(k, _)| class.contains(k)).map(|(_, l)| *l).unwrap_or("Working")
}

/// ComfyUI's binary preview message → the image.
pub fn preview(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 8 {
        return None;
    }
    match u32::from_be_bytes(data[..4].try_into().unwrap()) {
        1 => Some(data[8..].to_vec()), // PREVIEW_IMAGE: type, then the image
        4 => {
            // PREVIEW_IMAGE_WITH_METADATA: metadata length, metadata, image
            let n = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
            data.get(8 + n..).map(<[u8]>::to_vec)
        }
        _ => None,
    }
}

/// ComfyUI's refusal of a workflow, in words.
pub fn comfy_error(code: u16, body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else { return format!("ComfyUI rejected the workflow ({code})") };
    for node in v["node_errors"].as_object().into_iter().flatten().map(|(_, n)| n) {
        if let Some(err) = node["errors"].as_array().and_then(|e| e.first()) {
            return format!("ComfyUI rejected the workflow: {} ({})", py_str(&err["message"]), py_str(&node["class_type"]));
        }
    }
    format!("ComfyUI rejected the workflow: {}", v["error"]["message"].as_str().map(String::from).unwrap_or_else(|| format!("HTTP Error {code}")))
}

fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        o => o.to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct FileOut {
    pub index: usize,
    pub name: String,
    pub mime: String,
    pub size: u64,
    pub kind: String,
    pub path: PathBuf,
    pub delivered: bool,
}

#[derive(Default)]
pub struct JobState {
    pub state: String,
    pub phase: String,
    pub step: u64,
    pub steps: u64,
    pub progress: f64,
    pub error: Option<String>,
    pub files: Vec<FileOut>,
    pub preview: Option<Arc<Vec<u8>>>,
    /// Counts previews, so an event stream knows whether it sent this one already.
    pub preview_seq: u64,
    pub prompt_id: Option<String>,
    pub version: u64,
}

type Inputs = (Option<Vec<u8>>, Option<Vec<u8>>);

pub type OnChange = Box<dyn Fn(&str, &str) + Send + Sync>;

pub struct Job {
    pub id: String,
    pub spec: Value,
    pub params: Value,
    pub ip: String,
    pub created: f64,
    pub seed: u64,
    pub dir: PathBuf,
    pub st: Mutex<JobState>,
    pub cond: Condvar,
    pub cancel_requested: AtomicBool,
    inputs: Mutex<Vec<PathBuf>>,
    /// The photo and the voice sample the phone sent, until they're placed in ComfyUI's input.
    data: Mutex<Inputs>,
    pub on_change: Mutex<Option<OnChange>>,
}

impl Job {
    /// Changes the state; anyone waiting (event streams) wakes up.
    pub fn update(&self, f: impl FnOnce(&mut JobState)) {
        let (state, phase, before) = {
            let mut s = self.st.lock().unwrap();
            let before = (s.state.clone(), s.phase.clone());
            f(&mut s);
            s.version += 1;
            (s.state.clone(), s.phase.clone(), before)
        };
        self.cond.notify_all();
        if (state.clone(), phase.clone()) != before
            && let Some(cb) = self.on_change.lock().unwrap().as_ref()
        {
            cb(&state, &phase);
        }
    }

    pub fn state(&self) -> String {
        self.st.lock().unwrap().state.clone()
    }

    pub fn snapshot(&self, with_preview: bool) -> Value {
        let s = self.st.lock().unwrap();
        snapshot_of(self, &s, with_preview)
    }
}

pub fn snapshot_of(job: &Job, s: &JobState, with_preview: bool) -> Value {
    let mut v = json!({
        "id": job.id, "generator": job.spec["id"], "kind": job.spec["kind"], "state": s.state, "phase": s.phase,
        "step": s.step, "steps": s.steps, "progress": (s.progress * 1000.0).round_ties_even() / 1000.0,
        "files": s.files.iter().map(|f| json!({"index": f.index, "name": f.name, "mime": f.mime, "size": f.size, "kind": f.kind})).collect::<Vec<_>>(),
    });
    if job.spec.get("seed").is_some() {
        v["seed"] = job.seed.into();
    }
    if let Some(e) = &s.error {
        v["error"] = e.clone().into();
    }
    if with_preview && let Some(p) = &s.preview {
        v["preview"] = base64::engine::general_purpose::STANDARD.encode(p.as_slice()).into();
    }
    v
}

#[derive(Default)]
struct Inner {
    jobs: HashMap<String, Arc<Job>>,
    queue: VecDeque<Arc<Job>>,
}

pub struct Jobs {
    pub comfy: Arc<Comfy>,
    pub catalog: Catalog,
    inner: Mutex<Inner>,
    wake: Condvar,
}

impl Jobs {
    pub fn new(paths: Paths, ollama: &str) -> Arc<Jobs> {
        let temp = paths.temp();
        let _ = std::fs::create_dir_all(&temp);
        use std::os::unix::fs::PermissionsExt;
        for d in [&paths.cache_dir, &temp] {
            let _ = std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700));
        }
        let comfy = Comfy::new(paths.clone(), ollama);
        let j = Arc::new(Jobs { comfy, catalog: Catalog::new(paths), inner: Mutex::new(Inner::default()), wake: Condvar::new() });
        j.wipe_leftovers();
        let w = j.clone();
        std::thread::Builder::new().name("media-worker".into()).spawn(move || w.worker()).expect("worker thread");
        let w = Arc::downgrade(&j);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(60));
                let Some(j) = w.upgrade() else { return };
                j.janitor();
            }
        });
        j
    }

    fn paths(&self) -> &Paths {
        &self.comfy.paths
    }

    fn wipe_leftovers(&self) {
        // A job an earlier gateway left running in our own ComfyUI has nobody waiting for it: stop
        // it before it saves anything. (A ComfyUI started by hand runs outside the unit.)
        if self.comfy.unit_active() && self.comfy.running() {
            let _ = self.comfy.call("/queue", Some(&json!({"clear": true})), Duration::from_secs(30));
            let _ = self.comfy.call("/interrupt", Some(&json!({})), Duration::from_secs(30));
        }
        // Anything from a previous run was never delivered: it goes.
        for e in std::fs::read_dir(self.paths().temp()).into_iter().flatten().flatten() {
            let _ = std::fs::remove_dir_all(e.path());
        }
        for kind in ["input", "output"] {
            // The folder itself is kept: the service's sandbox grants exactly this folder.
            for e in std::fs::read_dir(self.paths().comfy_dir.join(kind).join(SUBFOLDER)).into_iter().flatten().flatten() {
                let p = e.path();
                let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
            }
        }
    }

    pub fn submit(&self, gen_id: &str, mut params: Value, image: Option<Vec<u8>>, mut audio: Option<Vec<u8>>, ip: &str) -> Result<Arc<Job>, String> {
        let spec = self.catalog.find(gen_id).cloned().ok_or("That generator isn't available on your PC.")?;
        if spec["image"] == "required" && image.as_ref().is_none_or(Vec::is_empty) {
            return Err("Add a photo to use this.".into());
        }
        let speaker = params.as_object_mut().and_then(|p| p.shift_remove("speaker"));
        if audio.as_ref().is_none_or(Vec::is_empty)
            && let Some(s) = speaker.as_ref().and_then(Value::as_str)
            && spec["voices"].as_array().is_some_and(|v| v.iter().any(|x| x == s))
            && let Some(f) = self.paths().voice_file(s)
        {
            audio = std::fs::read(f).ok();
        }
        // Chosen now, so the phone can show it and use it again for a variation of the same picture.
        let seed = params["seed"].as_u64().filter(|&s| s <= media::SEED_MAX).unwrap_or_else(random_seed);
        if !params.is_object() {
            params = json!({});
        }
        params["seed"] = seed.into();
        let id = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes::<12>());
        let dir = self.paths().temp().join(&id);
        std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
        let job = Arc::new(Job {
            id: id.clone(),
            spec: spec.clone(),
            params,
            ip: ip.to_string(),
            created: crate::time::now(),
            seed,
            dir,
            st: Mutex::new(JobState { state: "queued".into(), phase: "Waiting for your other creation".into(), ..Default::default() }),
            cond: Condvar::new(),
            cancel_requested: AtomicBool::new(false),
            inputs: Mutex::new(vec![]),
            data: Mutex::new((image.filter(|b| !b.is_empty()), audio.filter(|b| !b.is_empty()))),
            on_change: Mutex::new(None),
        });
        {
            let mut i = self.inner.lock().unwrap();
            i.jobs.insert(id.clone(), job.clone());
            i.queue.push_back(job.clone());
        }
        self.wake.notify_all();
        log(&format!("job {id} queued ({}) from {ip}", spec["id"].as_str().unwrap_or("")));
        Ok(job)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.inner.lock().unwrap().jobs.get(id).cloned()
    }

    pub fn queued(&self) -> usize {
        self.inner.lock().unwrap().queue.len()
    }

    pub fn cancel(&self, id: &str) -> bool {
        let Some(job) = self.get(id) else { return false };
        job.cancel_requested.store(true, Ordering::SeqCst);
        log(&format!("job {} cancelled", job.id));
        let was_queued = {
            let mut i = self.inner.lock().unwrap();
            let before = i.queue.len();
            i.queue.retain(|j| !Arc::ptr_eq(j, &job));
            i.queue.len() != before
        };
        if was_queued {
            job.update(|s| {
                s.state = "cancelled".into();
                s.phase = "Cancelled".into();
            });
        }
        let (pid, state) = {
            let s = job.st.lock().unwrap();
            (s.prompt_id.clone(), s.state.clone())
        };
        if let Some(pid) = pid
            && !TERMINAL.contains(&state.as_str())
        {
            let _ = self.comfy.call("/queue", Some(&json!({"delete": [pid]})), Duration::from_secs(30));
            let _ = self.comfy.call("/interrupt", Some(&json!({"prompt_id": pid})), Duration::from_secs(30));
        }
        self.forget(&job);
        true
    }

    pub fn forget(&self, job: &Job) {
        self.inner.lock().unwrap().jobs.remove(&job.id);
        let _ = std::fs::remove_dir_all(&job.dir);
    }

    /// The phone has file `index`; once it has them all, nothing of the job stays on the PC.
    pub fn delivered(&self, job: &Job, index: usize) {
        let all = {
            let mut s = job.st.lock().unwrap();
            let Some(f) = s.files.iter_mut().find(|f| f.index == index) else { return };
            f.delivered = true;
            s.files.iter().all(|f| f.delivered)
        };
        if all {
            log(&format!("job {} delivered; temp files deleted", job.id));
            self.forget(job);
        }
    }

    fn janitor(&self) {
        // ComfyUI can record a cancelled run just after we cleared it, prompt included. With nothing
        // running, our own ComfyUI's history is emptied (one started by hand is left alone).
        if self.comfy.busy.load(Ordering::SeqCst) == 0 && self.queued() == 0 && self.comfy.unit_active() && self.comfy.running() {
            let h = self.comfy.call("/history", None, Duration::from_secs(5));
            if h.as_ref().is_ok_and(|h| h.as_object().is_some_and(|o| !o.is_empty())) {
                let _ = self.comfy.call("/history", Some(&json!({"clear": true})), Duration::from_secs(5));
            }
        }
        self.expire(crate::time::now());
    }

    /// Finished jobs never collected within the hour are deleted.
    pub fn expire(&self, now: f64) {
        let stale: Vec<Arc<Job>> = self.inner.lock().unwrap().jobs.values().filter(|j| TERMINAL.contains(&j.state().as_str()) && now - j.created > UNDELIVERED_TTL_S).cloned().collect();
        for j in stale {
            log(&format!("job {} never collected; deleted", j.id));
            self.forget(&j);
        }
    }

    fn worker(&self) {
        loop {
            let job = {
                let mut i = self.inner.lock().unwrap();
                loop {
                    if let Some(j) = i.queue.pop_front() {
                        break j;
                    }
                    i = self.wake.wait(i).unwrap();
                }
            };
            self.comfy.busy.fetch_add(1, Ordering::SeqCst);
            if let Err(message) = self.run(&job) {
                job.update(|s| {
                    s.state = "error".into();
                    s.phase = "Failed".into();
                    s.error = Some(message);
                });
                log(&format!("job {} failed", job.id));
            }
            self.comfy.busy.fetch_sub(1, Ordering::SeqCst);
            self.comfy.mark_used();
            for p in job.inputs.lock().unwrap().drain(..) {
                let _ = std::fs::remove_file(p);
            }
            // Whatever wasn't collected (cancelled, failed half-way) mustn't stay in ComfyUI's output.
            for e in std::fs::read_dir(self.paths().comfy_dir.join("output").join(SUBFOLDER)).into_iter().flatten().flatten() {
                if e.file_name().to_string_lossy().starts_with(&job.id) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
            *job.data.lock().unwrap() = (None, None);
        }
    }

    fn place_input(&self, job: &Job, data: &[u8], default: &str) -> Result<String, String> {
        let folder = self.paths().comfy_dir.join("input").join(SUBFOLDER);
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        let name = format!("{}{}", job.id, media::suffix(data, default));
        let path = folder.join(&name);
        std::fs::write(&path, data).map_err(|e| e.to_string())?;
        job.inputs.lock().unwrap().push(path);
        Ok(format!("{SUBFOLDER}/{name}"))
    }

    fn run(&self, job: &Arc<Job>) -> Result<(), String> {
        if job.cancel_requested.load(Ordering::SeqCst) {
            return Ok(());
        }
        job.update(|s| {
            s.state = "starting".into();
            s.phase = "Getting your PC ready".into();
        });
        self.comfy.ensure(&|t| job.update(|s| s.phase = t.into()))?;
        free_ollama(&self.comfy.ollama, false);
        let (image, audio) = job.data.lock().unwrap().clone();
        let image = image.map(|d| self.place_input(job, &d, ".png")).transpose()?;
        let audio = audio.map(|d| self.place_input(job, &d, ".wav")).transpose()?;
        let workflow = self.catalog.workflow(&job.spec).ok_or("This generator's workflow is missing on your PC.")?;
        let info = self.comfy.object_info()?;
        let prompt = media::build(&job.spec, &job.params, &job.id, image.as_deref(), audio.as_deref(), &workflow, &info, job.seed)?;
        let client_id = crate::identity::hex(&random_bytes::<8>());
        let mut ws = WsClient::connect(self.paths().comfy_port, &format!("/ws?clientId={client_id}")).map_err(|e| format!("ComfyUI's progress feed: {e}"))?;
        let result = (|| {
            let queued = match self.comfy.call("/prompt", Some(&json!({"prompt": prompt, "client_id": client_id})), Duration::from_secs(30)) {
                Ok(q) => q,
                Err(ComfyError::Status(code, body)) => return Err(comfy_error(code, &body)),
                Err(e) => return Err(e.to_string()),
            };
            let pid = queued["prompt_id"].as_str().ok_or("ComfyUI didn't accept the job.")?.to_string();
            job.update(|s| {
                s.prompt_id = Some(pid.clone());
                s.state = "running".into();
                s.phase = "Waiting for ComfyUI".into();
            });
            self.comfy.dirty.store(true, Ordering::SeqCst);
            match follow(job, &mut ws, &prompt, &pid) {
                Ok(()) => Ok(pid),
                Err(Stop::Failed(m)) => Err(m),
                Err(Stop::Lost(_)) if job.cancel_requested.load(Ordering::SeqCst) => Ok(pid),
                Err(Stop::Lost(_)) => Err("ComfyUI stopped while making this (it may have run out of memory). Tap Retry to try again.".to_string()),
            }
        })();
        ws.close();
        let pid = result?;
        if job.state() == "cancelled" || job.cancel_requested.load(Ordering::SeqCst) {
            self.clear_history(&pid);
            return Ok(());
        }
        self.collect(job, &pid)
    }

    fn collect(&self, job: &Job, pid: &str) -> Result<(), String> {
        let history = self.comfy.call(&format!("/history/{pid}"), None, Duration::from_secs(30)).map_err(|e| e.to_string())?;
        let mut files: Vec<FileOut> = vec![];
        for out in history[pid]["outputs"].as_object().into_iter().flat_map(|o| o.values()) {
            for items in out.as_object().into_iter().flat_map(|o| o.values()).filter_map(Value::as_array) {
                for item in items {
                    let Some(filename) = item["filename"].as_str().filter(|f| !f.is_empty()) else { continue };
                    let base = self.paths().comfy_dir.join(if item["type"] == "temp" { "temp" } else { "output" });
                    let src = base.join(item["subfolder"].as_str().unwrap_or("")).join(filename);
                    // Only files really inside ComfyUI's folder.
                    let (Ok(base), Ok(src)) = (base.canonicalize(), src.canonicalize()) else { continue };
                    if !src.starts_with(&base) || src == base || !src.is_file() {
                        continue;
                    }
                    let index = files.len();
                    let suffix = src.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default();
                    let dest = job.dir.join(format!("{index}{suffix}"));
                    move_file(&src, &dest).map_err(|e| e.to_string())?;
                    files.push(FileOut {
                        index,
                        name: format!("{}-{}-{}{suffix}", job.spec["id"].as_str().unwrap_or("x"), &job.id[..6], index + 1),
                        mime: media::mime(&suffix).into(),
                        size: std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0),
                        kind: media::kind(&suffix).into(),
                        path: dest,
                        delivered: false,
                    });
                }
            }
        }
        self.clear_history(pid);
        if files.is_empty() {
            return Err("ComfyUI finished but made no file.".into());
        }
        let n = files.len();
        job.update(|s| {
            s.state = "done".into();
            s.phase = "Done".into();
            s.progress = 1.0;
            s.files = files;
            s.preview = None;
        });
        log(&format!("job {} done: {n} file(s)", job.id));
        Ok(())
    }

    fn clear_history(&self, pid: &str) {
        let _ = self.comfy.call("/history", Some(&json!({"delete": [pid]})), Duration::from_secs(30));
    }
}

fn move_file(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::rename(src, dest).or_else(|_| {
        std::fs::copy(src, dest)?;
        std::fs::remove_file(src)
    })
}

/// Why following a job stopped early.
pub enum Stop {
    /// The connection to ComfyUI went (it crashed or was stopped).
    Lost(std::io::Error),
    /// ComfyUI reported an error, in its own words.
    Failed(String),
}

/// Follows ComfyUI's messages until the job ends. The bar moves by how long each node usually
/// takes, so it doesn't race through loaders and then sit still while the sampler works.
fn follow(job: &Job, ws: &mut WsClient, prompt: &serde_json::Map<String, Value>, pid: &str) -> Result<(), Stop> {
    let weights: HashMap<&str, f64> = prompt.iter().map(|(k, v)| (k.as_str(), weight(v["class_type"].as_str().unwrap_or("")))).collect();
    let total = weights.values().sum::<f64>().max(1e-9);
    let mut done: std::collections::HashSet<String> = Default::default();
    let mut current: Option<String> = None;
    let mut last_preview: Option<Instant> = None;
    let finished = |done: &std::collections::HashSet<String>, current: &Option<String>| done.iter().filter(|n| Some(*n) != current.as_ref()).map(|n| weights.get(n.as_str()).copied().unwrap_or(0.0)).sum::<f64>() / total;
    ws.set_timeout(Duration::from_secs(30));
    loop {
        let (kind, data) = match ws.recv() {
            Ok(m) => m,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                if job.cancel_requested.load(Ordering::SeqCst) {
                    return Ok(());
                }
                continue;
            }
            Err(e) => return Err(Stop::Lost(e)),
        };
        if kind == 2 {
            if let Some(img) = preview(&data)
                && last_preview.is_none_or(|t| t.elapsed() > Duration::from_secs(1))
            {
                last_preview = Some(Instant::now());
                job.update(|s| {
                    s.preview = Some(Arc::new(img));
                    s.preview_seq += 1;
                });
            }
            continue;
        }
        let Ok(msg) = serde_json::from_slice::<Value>(&data) else { continue };
        let d = &msg["data"];
        if !(d["prompt_id"].is_null() || d["prompt_id"] == pid) {
            continue;
        }
        match msg["type"].as_str().unwrap_or("") {
            "executing" if d["node"].is_null() && d["prompt_id"] == pid => return Ok(()),
            "executing" if !d["node"].is_null() => {
                let node = match &d["node"] {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                done.insert(node.clone());
                current = Some(node.clone());
                let p = finished(&done, &current).min(0.99);
                let ph = phase(prompt.get(&node).and_then(|n| n["class_type"].as_str()).unwrap_or(""));
                job.update(|s| {
                    s.phase = ph.into();
                    s.step = 0;
                    s.steps = 0;
                    s.progress = p;
                });
            }
            "execution_cached" => {
                for n in d["nodes"].as_array().into_iter().flatten() {
                    done.insert(match n {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    });
                }
            }
            "progress" => {
                let (value, max) = (d["value"].as_u64().unwrap_or(0), d["max"].as_u64().unwrap_or(0));
                let w = current.as_deref().and_then(|c| weights.get(c)).copied().unwrap_or(0.0);
                let part = w * if max > 0 { value as f64 / max as f64 } else { 0.0 } / total;
                let p = (finished(&done, &current) + part).min(0.99);
                job.update(|s| {
                    s.step = value;
                    s.steps = max;
                    s.progress = p;
                });
            }
            "execution_success" => return Ok(()),
            "execution_error" => {
                let m: String = d["exception_message"].as_str().unwrap_or("generation failed").trim().chars().take(300).collect();
                return Err(Stop::Failed(format!("ComfyUI: {m}")));
            }
            "execution_interrupted" => {
                job.update(|s| {
                    s.state = "cancelled".into();
                    s.phase = "Cancelled".into();
                });
                return Ok(());
            }
            _ => {}
        }
    }
}
