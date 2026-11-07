//! The phone gateway's control panel (what the GTK "CubicAI Server" window did): the
//! `cubicai-server` service, Ollama, private search, pairing and activity. Everything slow
//! (systemctl, docker, HTTP) runs off the UI thread; the screen reads the last snapshot.

use cubic_gateway::{access, identity, local, settings::Settings, DEFAULT_PORT};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const OLLAMA: &str = "http://127.0.0.1:11434";
pub const SEARX_CONTAINER: &str = "cubicai-searxng";
pub const CONTEXTS: [(u64, &str); 4] = [(8192, "8K"), (16384, "16K"), (32768, "32K"), (65536, "64K")];
pub const DEFAULT_CTX: u64 = 16384;

#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub name: String,
    pub size: u64,
    pub vram: u64,
}

impl Loaded {
    /// "4.2 GB · fully on GPU" or "6.1 GB · 5.0 GB on GPU, rest on CPU".
    pub fn describe(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let where_ = if self.size > 0 && self.vram as f64 >= self.size as f64 * 0.99 {
            "fully on GPU".to_string()
        } else {
            format!("{:.1} GB on GPU, rest on CPU", gb(self.vram))
        };
        format!("{:.1} GB · {where_}", gb(self.size))
    }
}

/// One look at everything the panel shows.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// Which gateway the installed unit runs ("Rust", "Python", "other"); None: no unit.
    pub unit: Option<&'static str>,
    pub active: bool,
    pub enabled: bool,
    pub searx: bool,
    pub ollama: Option<String>,
    pub loaded: Vec<Loaded>,
    /// /v1/cubicai/status from the running gateway (None when off or not ours).
    pub gateway: Option<Value>,
    pub tailscale: Option<String>,
    pub remote: bool,
    pub num_ctx: u64,
    pub lan_ip: String,
    /// The pairing link (contains the token; never logged or shown in full).
    pub link: String,
    /// Device names from the router's reverse DNS, by IP.
    pub names: HashMap<String, String>,
}

impl Status {
    pub fn url(&self) -> String {
        format!("https://{}:{DEFAULT_PORT}", self.lan_ip)
    }

    pub fn remote_url(&self) -> Option<String> {
        self.tailscale.as_ref().filter(|_| self.remote).map(|ip| format!("https://{ip}:{DEFAULT_PORT}"))
    }

    /// "Pixel-8 (192.168.1.40)", or the bare IP when the router doesn't know it.
    pub fn device(&self, ip: &str) -> String {
        if ip == "127.0.0.1" || ip == "::1" {
            return "This PC (or emulator)".into();
        }
        match self.names.get(ip) {
            Some(n) => format!("{n} ({ip})"),
            None => ip.to_string(),
        }
    }
}

fn run(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

fn ollama_loaded(ps: &Value) -> Vec<Loaded> {
    ps["models"]
        .as_array()
        .map(|a| a.iter().map(|m| Loaded { name: m["name"].as_str().unwrap_or("?").into(), size: m["size"].as_u64().unwrap_or(0), vram: m["size_vram"].as_u64().unwrap_or(0) }).collect())
        .unwrap_or_default()
}

fn http_json(url: &str) -> Option<Value> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(3))).build().into();
    agent.get(url).call().ok()?.body_mut().read_json().ok()
}

/// The name the router gives `ip` (its reverse DNS), without the domain.
fn reverse_name(ip: &str) -> Option<String> {
    let out = run("getent", &["hosts", ip]);
    let name = out.split_whitespace().nth(1)?;
    let short = name.split('.').next()?;
    (!short.is_empty() && short != ip).then(|| short.to_string())
}

/// IPs whose names should be looked up: devices, open and recent requests.
pub fn ips_in(gateway: &Value) -> Vec<String> {
    let mut ips: Vec<String> = vec![];
    for list in ["devices", "active", "recent"] {
        for item in gateway[list].as_array().into_iter().flatten() {
            if let Some(ip) = item["ip"].as_str()
                && !ips.iter().any(|i| i == ip)
            {
                ips.push(ip.to_string());
            }
        }
    }
    ips
}

/// The pairing link for `url` (+ the Tailscale address while remote access is on).
pub fn pairing_link(dir: &std::path::Path, url: &str, remote: Option<&str>) -> String {
    match identity::load(dir) {
        Ok(id) => identity::pairing_link(url, &id.token, &id.fingerprint, remote),
        Err(_) => String::new(),
    }
}

pub struct Panel {
    pub dir: PathBuf,
    status: Arc<Mutex<Option<Status>>>,
    fetching: Arc<AtomicBool>,
    last: Option<Instant>,
    host: Arc<access::Host>,
    names: Arc<Mutex<HashMap<String, Option<String>>>>,
    tx: Sender<String>,
    rx: Receiver<String>,
    /// Waiting for "Unpair all phones" to be confirmed.
    pub confirm_unpair: bool,
    pub view: View,
    pub opaque_asked: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    Status,
    Pair,
    Activity,
}

impl Default for Panel {
    fn default() -> Self {
        Panel::new(cubic_gateway::config_dir())
    }
}

impl Panel {
    pub fn new(dir: PathBuf) -> Panel {
        let (tx, rx) = channel();
        Panel {
            dir,
            status: Default::default(),
            fetching: Default::default(),
            last: None,
            host: Default::default(),
            names: Default::default(),
            tx,
            rx,
            confirm_unpair: false,
            view: View::Status,
            opaque_asked: false,
        }
    }

    pub fn status(&self) -> Option<Status> {
        self.status.lock().unwrap().clone()
    }

    /// Shows `s` and stops looking (tests and screenshots).
    #[cfg(test)]
    pub fn set_status(&mut self, s: Status) {
        *self.status.lock().unwrap() = Some(s);
        self.last = Some(Instant::now() + Duration::from_secs(86400));
    }

    /// Messages from finished actions, for the toast.
    pub fn messages(&self) -> Vec<String> {
        self.rx.try_iter().collect()
    }

    /// Looks again every 2 s while the panel is open (and at once after an action).
    pub fn poll(&mut self, ctx: &egui::Context) {
        if self.last.is_some_and(|t| t.elapsed() < Duration::from_secs(2)) || self.fetching.swap(true, Ordering::SeqCst) {
            return;
        }
        self.last = Some(Instant::now());
        let (dir, status, fetching, host, names, ctx) = (self.dir.clone(), self.status.clone(), self.fetching.clone(), self.host.clone(), self.names.clone(), ctx.clone());
        std::thread::spawn(move || {
            let s = gather(&dir, &host, &names);
            *status.lock().unwrap() = Some(s);
            fetching.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    /// Runs `f` off the UI thread; its message (or error) becomes a toast and the panel looks again.
    pub fn act(&mut self, ctx: &egui::Context, f: impl FnOnce(&std::path::Path) -> Result<String, String> + Send + 'static) {
        let (tx, dir, ctx) = (self.tx.clone(), self.dir.clone(), ctx.clone());
        self.last = None;
        std::thread::spawn(move || {
            let msg = f(&dir).unwrap_or_else(|e| format!("Failed: {e}"));
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    /// Changes settings.json at once (the gateway reads it per request), updating the snapshot too.
    pub fn save_setting(&mut self, key: &str, value: Value) -> Result<(), String> {
        let mut m = serde_json::Map::new();
        m.insert(key.into(), value.clone());
        Settings::new(&self.dir).save(m)?;
        if let Some(s) = self.status.lock().unwrap().as_mut() {
            match key {
                "remote" => {
                    s.remote = value.as_bool().unwrap_or(false);
                    s.link = pairing_link(&self.dir, &s.url(), s.remote_url().as_deref());
                }
                "num_ctx" => s.num_ctx = value.as_u64().unwrap_or(DEFAULT_CTX),
                _ => {}
            }
        }
        self.last = None;
        Ok(())
    }
}

fn gather(dir: &std::path::Path, host: &access::Host, names: &Mutex<HashMap<String, Option<String>>>) -> Status {
    let settings = Settings::new(dir);
    let active = run("systemctl", &["--user", "is-active", local::SERVICE]) == "active";
    let mut s = Status {
        unit: local::service_kind(&local::unit_path()),
        active,
        enabled: run("systemctl", &["--user", "is-enabled", local::SERVICE]) == "enabled",
        searx: run("docker", &["inspect", "-f", "{{.State.Running}}", SEARX_CONTAINER]) == "true",
        ollama: http_json(&format!("{OLLAMA}/api/version")).and_then(|v| v["version"].as_str().map(String::from)),
        loaded: http_json(&format!("{OLLAMA}/api/ps")).map(|v| ollama_loaded(&v)).unwrap_or_default(),
        gateway: None,
        tailscale: host.tailscale_ip(),
        remote: settings.remote(),
        num_ctx: settings.num_ctx(DEFAULT_CTX),
        lan_ip: access::lan_ip(),
        link: String::new(),
        names: HashMap::new(),
    };
    if let Ok(id) = identity::load(dir) {
        s.link = identity::pairing_link(&s.url(), &id.token, &id.fingerprint, s.remote_url().as_deref());
        if active {
            s.gateway = local::get(DEFAULT_PORT, &id.fingerprint, &id.token, "/v1/cubicai/status");
        }
    }
    if let Some(gw) = &s.gateway {
        let mut cache = names.lock().unwrap();
        for ip in ips_in(gw) {
            let name = cache.entry(ip.clone()).or_insert_with(|| if ip == "127.0.0.1" || ip == "::1" { None } else { reverse_name(&ip) });
            if let Some(n) = name {
                s.names.insert(ip, n.clone());
            }
        }
    }
    s
}

// --- Actions (run through `Panel::act`) -------------------------------------------------------

pub fn set_running(on: bool) -> Result<String, String> {
    local::systemctl(&[if on { "start" } else { "stop" }, local::SERVICE])?;
    Ok(if on { "Gateway started" } else { "Gateway stopped. Your phone can't reach this PC now" }.into())
}

pub fn set_autostart(on: bool) -> Result<String, String> {
    local::systemctl(&[if on { "enable" } else { "disable" }, local::SERVICE])?;
    Ok(if on { "Starts when you log in" } else { "Won't start by itself" }.into())
}

pub fn set_search(on: bool) -> Result<String, String> {
    let out = Command::new("docker").args([if on { "start" } else { "stop" }, SEARX_CONTAINER]).output().map_err(|e| format!("docker: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(if on { "Web search started" } else { "Web search stopped" }.into())
}

pub fn unload(name: &str) -> Result<String, String> {
    ureq::post(&format!("{OLLAMA}/api/generate")).send_json(json!({"model": name, "keep_alive": 0})).map_err(|e| e.to_string())?;
    Ok(format!("Unloaded {name}"))
}

/// The gateway binary to install: next to this Lab (a workspace build), else the one installed.
pub fn gateway_binary() -> Option<PathBuf> {
    let beside = std::env::current_exe().ok()?.with_file_name("cubicai-gateway");
    [beside, local::installed_binary()].into_iter().find(|p| p.is_file())
}

pub fn use_rust() -> Result<String, String> {
    let bin = gateway_binary().ok_or("Build the gateway first: cargo build --release -p cubic-gateway")?;
    local::install(&bin)?;
    Ok("The service runs the Rust gateway now".into())
}

pub fn unpair(dir: &std::path::Path) -> Result<String, String> {
    identity::rotate_token(dir)?;
    let _ = local::systemctl(&["try-restart", local::SERVICE]);
    Ok("All phones unpaired. Scan the new QR code".into())
}

/// The firewall rule for the home subnet, for when the phone can't connect.
pub fn firewall_command(lan_ip: &str) -> String {
    let net: Vec<&str> = lan_ip.split('.').take(3).collect();
    format!("sudo ufw allow from {}.0/24 to any port {DEFAULT_PORT} proto tcp", net.join("."))
}

/// On Hyprland, keeps this window fully opaque so the QR code scans reliably (a runtime rule
/// that lasts until Hyprland reloads its config; no files touched).
pub fn keep_opaque() {
    if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
        let rule = r#"hl.window_rule({ name = "cubicai-lab-opaque", match = { class = "^cubicai-lab$" }, opaque = true, no_blur = true })"#;
        let _ = Command::new("hyprctl").args(["eval", rule]).output();
    }
}

// --- Formatting -------------------------------------------------------------------------------

pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn ago(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        _ => format!("{} d ago", s / 86400),
    }
}

pub fn duration(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {}s", s / 60, s % 60),
        _ => format!("{}h {}m", s / 3600, s % 3600 / 60),
    }
}

/// Local "HH:MM" of a Unix time.
pub fn clock(t: f64) -> String {
    #[cfg(unix)]
    unsafe {
        let secs = t as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        if !libc::localtime_r(&secs, &mut tm).is_null() {
            return format!("{:02}:{:02}", tm.tm_hour, tm.tm_min);
        }
    }
    let s = t as u64 % 86400;
    format!("{:02}:{:02}", s / 3600, s % 3600 / 60)
}

/// One finished request as the activity list shows it: (title, details).
pub fn recent_row(r: &Value, device: impl Fn(&str) -> String) -> (String, String) {
    let ip = r["ip"].as_str().unwrap_or("?");
    let started = r["started"].as_f64().unwrap_or(0.0);
    let outcome = r["outcome"].as_str().unwrap_or("ok");
    if outcome == "blocked" {
        return ("Blocked request (wrong or missing token)".into(), format!("{} · {}", clock(started), device(ip)));
    }
    let mut bits = vec![clock(started), device(ip), duration(r["ended"].as_f64().unwrap_or(started) - started)];
    if let Some(tokens) = r["tokens"].as_u64().filter(|&t| t > 0) {
        bits.push(format!("{tokens} tokens · {} tok/s", r["tps"]));
    }
    if r["search"].as_bool() == Some(true) {
        bits.push("web search".into());
    }
    if outcome != "ok" {
        bits.push(outcome.into());
    }
    (r["model"].as_str().unwrap_or("?").to_string(), bits.join(" · "))
}

/// The QR code of `text` as rows of dark modules (with a 2-module quiet zone).
pub fn qr_matrix(text: &str) -> Option<Vec<Vec<bool>>> {
    let code = qrcode::QrCode::with_error_correction_level(text.as_bytes(), qrcode::EcLevel::M).ok()?;
    let w = code.width();
    let colors = code.to_colors();
    let n = w + 4;
    Some((0..n).map(|y| (0..n).map(|x| x >= 2 && y >= 2 && x < w + 2 && y < w + 2 && colors[(y - 2) * w + (x - 2)] == qrcode::Color::Dark).collect()).collect())
}
