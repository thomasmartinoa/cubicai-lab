//! The Bots' computer: one Docker container all Bots share, with a terminal and a real Chromium,
//! started when a Bot first needs it and stopped after `IDLE` without use (files in the workspace stay;
//! everything else, logins included, starts fresh). `cubicai-gateway computer-setup` builds its image
//! from `computer/` in this crate.
//!
//! It's locked down: no network of its own (`--network none`; its only way out is `proxy.rs`, public
//! websites on ports 80/443), a read-only system, every capability dropped, no privilege escalation,
//! this user's uid (so its files are the user's), memory/CPU/process limits, and nothing of this PC
//! mounted but the Bots' workspace.

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const IMAGE: &str = "cubicai-computer:1";
pub const NAME: &str = "cubicai-computer";
/// Stopped after this long without a command.
pub const IDLE: Duration = Duration::from_secs(600);
pub const DEFAULT_TIMEOUT: u64 = 120;
pub const MAX_TIMEOUT: u64 = 600;
/// What a command's output is cut to for the model: its start and its end.
const OUT_HEAD: usize = 2000;
const OUT_TAIL: usize = 6000;
const CAPTURE_BYTES: usize = 1024 * 1024;

const FILES: [(&str, &str); 5] = [
    ("Dockerfile", include_str!("../../computer/Dockerfile")),
    ("start.sh", include_str!("../../computer/start.sh")),
    ("browserd.py", include_str!("../../computer/browserd.py")),
    ("browse.py", include_str!("../../computer/browse.py")),
    ("kill.sh", include_str!("../../computer/kill.sh")),
];

/// Where the computer's things are and how to reach Docker (tests use a fake docker and name).
#[derive(Clone, Debug)]
pub struct Setup {
    /// Holds `run/proxy.sock` (mounted at /run/cubic) and `shots/` (mounted at /shots).
    pub dir: PathBuf,
    pub docker: String,
    pub name: String,
    pub image: String,
}

impl Setup {
    /// ~/.cache/cubicai-server/computer (`CUBICAI_COMPUTER_DIR`), the system's docker.
    pub fn from_env(home: &Path) -> Setup {
        Setup {
            dir: std::env::var_os("CUBICAI_COMPUTER_DIR").map(PathBuf::from).unwrap_or_else(|| home.join(".cache/cubicai-server/computer")),
            docker: "docker".into(),
            name: NAME.into(),
            image: IMAGE.into(),
        }
    }
}

pub struct Computer {
    docker: String,
    image: String,
    name: String,
    dir: PathBuf,
    workspace: PathBuf,
    last_used: Mutex<Instant>,
    busy: AtomicUsize,
    /// Serialises starting; true once the proxy listens.
    starting: Mutex<bool>,
    /// Whether the image exists, and when that was checked.
    has_image: Mutex<Option<(bool, Instant)>>,
    leftover_checked: AtomicBool,
}

/// What a command came to.
pub struct Ran {
    pub output: String,
    pub code: Option<i32>,
    pub timed_out: bool,
    pub stopped: bool,
    pub secs: f64,
}

impl Computer {
    pub fn new(setup: &Setup, workspace: &Path) -> Computer {
        Computer {
            docker: setup.docker.clone(),
            image: setup.image.clone(),
            name: setup.name.clone(),
            dir: setup.dir.clone(),
            workspace: workspace.to_path_buf(),
            last_used: Mutex::new(Instant::now()),
            busy: AtomicUsize::new(0),
            starting: Mutex::new(false),
            has_image: Mutex::new(None),
            leftover_checked: AtomicBool::new(false),
        }
    }

    fn docker(&self) -> Command {
        let mut c = Command::new(&self.docker);
        c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        c
    }

    /// Whether the image is built (checked at most once a minute).
    pub fn available(&self) -> bool {
        let mut h = self.has_image.lock().unwrap();
        if let Some((yes, at)) = *h
            && at.elapsed() < Duration::from_secs(60)
        {
            return yes;
        }
        let yes = self.docker().args(["image", "inspect", &self.image]).output().is_ok_and(|o| o.status.success());
        *h = Some((yes, Instant::now()));
        yes
    }

    fn running(&self) -> bool {
        self.docker().args(["inspect", "-f", "{{.State.Running}}", &self.name]).output().is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
    }

    /// Starts the container (and the proxy) if it isn't running.
    pub fn ensure(&self) -> Result<(), String> {
        let mut proxy = self.starting.lock().unwrap();
        *self.last_used.lock().unwrap() = Instant::now();
        if !self.available() {
            return Err("The Bots' computer isn't set up on this PC yet. Run `cubicai-gateway computer-setup` there (it builds it, about 1.5 GB).".into());
        }
        let run = self.dir.join("run");
        let shots = self.dir.join("shots");
        for d in [&self.dir, &run, &shots] {
            std::fs::create_dir_all(d).map_err(|e| format!("Couldn't make {}: {e}", d.display()))?;
            let _ = std::fs::set_permissions(d, std::os::unix::fs::PermissionsExt::from_mode(0o700));
        }
        if !*proxy {
            // The first start since the gateway started: a computer left by an earlier one goes (its
            // proxy socket and browser belong to that one).
            self.clean_start();
            super::proxy::start(&run.join("proxy.sock")).map_err(|e| format!("Couldn't start the computer's proxy: {e}"))?;
            *proxy = true;
        }
        if self.running() {
            return Ok(());
        }
        let _ = self.docker().args(["rm", "-f", &self.name]).output();
        // SAFETY: getuid/getgid only read this process's ids.
        let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
        let mount = |from: &Path, to: &str| format!("type=bind,source={},target={to}", from.display());
        let proxy_env = "http://127.0.0.1:3128";
        let out = self
            .docker()
            .args(["run", "-d", "--name", &self.name, "--label", "cubicai=computer", "--network", "none", "--read-only", "--init"])
            .args(["--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=1g,mode=1777"])
            .args(["--tmpfs", &format!("/home/bot:rw,exec,nosuid,nodev,size=1g,mode=700,uid={uid},gid={gid}")])
            .args(["--user", &format!("{uid}:{gid}"), "--cap-drop", "ALL", "--security-opt", "no-new-privileges"])
            .args(["--pids-limit", "512", "--memory", "3g", "--memory-swap", "3g", "--cpus", "2", "--shm-size", "512m"])
            .args(["--mount", &mount(&self.workspace, "/workspace"), "--mount", &mount(&run, "/run/cubic"), "--mount", &mount(&shots, "/shots")])
            .args(["-e", "HOME=/home/bot", "-e", &format!("HTTP_PROXY={proxy_env}"), "-e", &format!("HTTPS_PROXY={proxy_env}")])
            .args(["-e", &format!("http_proxy={proxy_env}"), "-e", &format!("https_proxy={proxy_env}"), "-e", "NO_PROXY=127.0.0.1,localhost", "-e", "no_proxy=127.0.0.1,localhost"])
            .arg(&self.image)
            .output()
            .map_err(|e| format!("Couldn't run docker: {e}"))?;
        if !out.status.success() {
            return Err(format!("The computer didn't start: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        crate::log("bots' computer started");
        // Ready once its browser service answers.
        let end = Instant::now() + Duration::from_secs(20);
        while Instant::now() < end {
            if self.browse_raw(&json!({"do": "ping"}), Duration::from_secs(5)).is_ok_and(|v| v["ok"] == true) {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        Err("The computer started but its browser service didn't answer.".into())
    }

    /// Runs `command` in bash in /workspace, for up to `timeout` seconds (or until `stop`).
    pub fn run(&self, command: &str, timeout: u64, stop: &AtomicBool) -> Result<Ran, String> {
        self.ensure()?;
        let _busy = Busy::new(self);
        let tag = super::new_id();
        let started = Instant::now();
        let script = format!("exec 2>&1\n{command}");
        let mut child = self
            .docker()
            .args(["exec", "-w", "/workspace", &self.name, "setsid", "-w", "bash", "-c", &script, &format!("cubic-{tag}")])
            .spawn()
            .map_err(|e| format!("Couldn't run docker: {e}"))?;
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut out = Vec::new();
            let mut buf = [0u8; 8192];
            while let Ok(n) = stdout.read(&mut buf) {
                if n == 0 {
                    break;
                }
                if out.len() < CAPTURE_BYTES {
                    out.extend_from_slice(&buf[..n]);
                }
            }
            out
        });
        let limit = Duration::from_secs(timeout.clamp(1, MAX_TIMEOUT));
        let (mut timed_out, mut stopped) = (false, false);
        let code = loop {
            if let Ok(Some(status)) = child.try_wait() {
                break status.code();
            }
            if started.elapsed() > limit || stop.load(Ordering::SeqCst) {
                timed_out = !stop.load(Ordering::SeqCst);
                stopped = !timed_out;
                // Killing `docker exec` leaves the command running inside: end its whole session there.
                let _ = self.docker().args(["exec", &self.name, "/opt/cubic/kill.sh", &tag]).output();
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        let raw = reader.join().unwrap_or_default();
        *self.last_used.lock().unwrap() = Instant::now();
        Ok(Ran { output: trim_output(&String::from_utf8_lossy(&raw)), code, timed_out, stopped, secs: started.elapsed().as_secs_f64() })
    }

    /// One browser command (see computer/browserd.py); the page as JSON.
    pub fn browse(&self, cmd: &Value) -> Result<Value, String> {
        self.ensure()?;
        let _busy = Busy::new(self);
        let v = self.browse_raw(cmd, Duration::from_secs(100))?;
        *self.last_used.lock().unwrap() = Instant::now();
        Ok(v)
    }

    fn browse_raw(&self, cmd: &Value, timeout: Duration) -> Result<Value, String> {
        let mut child = self
            .docker()
            .args(["exec", "-i", &self.name, "python3", "/opt/cubic/browse.py"])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Couldn't run docker: {e}"))?;
        let _ = child.stdin.take().unwrap().write_all(cmd.to_string().as_bytes());
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stdout.read_to_string(&mut s);
            s
        });
        let started = Instant::now();
        loop {
            if let Ok(Some(_)) = child.try_wait() {
                break;
            }
            if started.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Err("The browser took too long.".into());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let out = reader.join().unwrap_or_default();
        serde_json::from_str(out.trim()).map_err(|_| "The browser gave no answer.".to_string())
    }

    /// The host file for a screenshot the browser reported as /shots/<name>.
    pub fn shot(&self, reported: &str) -> Option<PathBuf> {
        let name = reported.strip_prefix("/shots/")?;
        if name.is_empty() || name.contains('/') || name.contains("..") {
            return None;
        }
        Some(self.dir.join("shots").join(name))
    }

    /// Stops the container after IDLE without use (called by the Bots' scheduler).
    pub fn idle_check(&self) {
        if self.busy.load(Ordering::SeqCst) > 0 {
            return;
        }
        let Ok(g) = self.starting.try_lock() else { return };
        // Before this gateway started it: one left by an earlier gateway goes (checked once).
        let stale = !*g && !self.leftover_checked.swap(true, Ordering::SeqCst);
        if (stale || (*g && self.last_used.lock().unwrap().elapsed() >= IDLE)) && self.running() {
            let _ = self.docker().args(["rm", "-f", &self.name]).output();
            for e in std::fs::read_dir(self.dir.join("shots")).into_iter().flatten().flatten() {
                let _ = std::fs::remove_file(e.path());
            }
            crate::log("bots' computer stopped (idle)");
        }
    }

    /// The gateway starting: a container left by an earlier one goes (its browser state is stale).
    pub fn clean_start(&self) {
        let _ = self.docker().args(["rm", "-f", &self.name]).output();
    }
}

struct Busy<'a>(&'a Computer);

impl<'a> Busy<'a> {
    fn new(c: &'a Computer) -> Busy<'a> {
        c.busy.fetch_add(1, Ordering::SeqCst);
        Busy(c)
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.busy.fetch_sub(1, Ordering::SeqCst);
        *self.0.last_used.lock().unwrap() = Instant::now();
    }
}

/// The start and the end of long output, with how much was cut between.
pub fn trim_output(s: &str) -> String {
    let n = s.chars().count();
    if n <= OUT_HEAD + OUT_TAIL {
        return s.to_string();
    }
    let head: String = s.chars().take(OUT_HEAD).collect();
    let tail: String = s.chars().skip(n - OUT_TAIL).collect();
    format!("{head}\n[… {} characters cut …]\n{tail}", n - OUT_HEAD - OUT_TAIL)
}

/// Builds the image from the files in this binary (`cubicai-gateway computer-setup`).
pub fn setup() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("cubicai-computer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for (name, text) in FILES {
        std::fs::write(dir.join(name), text).map_err(|e| e.to_string())?;
    }
    println!("Building the Bots' computer ({IMAGE}): Debian with Python, git, curl and Chromium. This takes a few minutes the first time.");
    let ok = Command::new("docker")
        .args(["build", "-t", IMAGE])
        .arg(&dir)
        .status()
        .map_err(|e| format!("Couldn't run docker (is it installed, and are you in the docker group?): {e}"))?
        .success();
    let _ = std::fs::remove_dir_all(&dir);
    if !ok {
        return Err("The build failed (see above).".into());
    }
    println!("Done. Bots can use the computer now (restart the gateway if it was running: systemctl --user restart cubicai-server).");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_output_keeps_its_start_and_end() {
        assert_eq!(trim_output("short"), "short");
        let long: String = (0..20_000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let t = trim_output(&long);
        assert!(t.starts_with(&long[..OUT_HEAD]) && t.ends_with(&long[long.len() - OUT_TAIL..]));
        assert!(t.contains("[… 12000 characters cut …]"));
    }

    #[test]
    fn screenshots_stay_in_their_folder() {
        let c = Computer::new(&Setup { dir: "/x".into(), docker: "docker".into(), name: "n".into(), image: "i".into() }, Path::new("/w"));
        assert_eq!(c.shot("/shots/12-3.jpg"), Some(PathBuf::from("/x/shots/12-3.jpg")));
        for bad in ["/shots/../key.pem", "/etc/passwd", "/shots/", "/shots/a/b.jpg"] {
            assert_eq!(c.shot(bad), None, "{bad}");
        }
    }
}
