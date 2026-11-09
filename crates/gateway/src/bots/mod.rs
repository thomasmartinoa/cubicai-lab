//! Bots: helpers that live on this PC and work for the phone's owner, the way xAI's Grok Bot app
//! does, but with any model (Ollama here, or Grok, Gemini, OpenRouter or a custom provider with the
//! keys the phone hands over). Each Bot has a name, a job, a colour, its own chat and its own memory;
//! all Bots share one workspace folder. A message starts a run on its own thread, so the work goes on
//! when the phone leaves; the phone follows a run's events and catches up from the chat later.
//! Routines (routines.rs) start runs on a schedule; every run's end goes into a feed the phone polls
//! for its notifications.
//!
//! Unlike the rest of the gateway, Bots keep what is said to them (their chat and memory, in the
//! owner-only data folder): that is what makes them Bots. Nothing of it is ever logged.

pub mod api;
pub mod brain;
pub mod computer;
pub mod proxy;
pub mod routines;
pub mod tools;

use crate::ollama::Ollama;
use crate::settings::Settings;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

/// Model turns per message before a run gives up (a Bot looping on its tools mustn't run forever).
pub const MAX_STEPS: usize = 20;
pub const MAX_BOTS: usize = 50;
/// Stored messages sent with each turn (older ones stay in the chat for the phone).
pub const HISTORY_MESSAGES: usize = 60;
/// A tool result from an earlier message is cut to this many characters in later turns.
pub const OLD_RESULT_CHARS: usize = 1500;
pub const MEMORY_CHARS: usize = 6000;
/// A routine more than this late (the PC was off at the time) is skipped until its next time.
pub const LATE_LIMIT_SECS: i64 = 6 * 3600;
/// Routines a Bot may propose that the user hasn't answered yet.
pub const MAX_PENDING: usize = 3;
/// How long a run waits for the user to allow an action before giving up on it.
pub const APPROVAL_WAIT: Duration = Duration::from_secs(15 * 60);
/// Screenshots kept per Bot (older ones go).
pub const SHOTS_KEPT: usize = 200;
/// Feed items kept for the phone's notifications.
pub const FEED_ITEMS: usize = 200;
pub const PROVIDERS: [&str; 5] = ["local", "xai", "gemini", "openrouter", "custom"];

/// Where Bots keep their things (`CUBICAI_BOTS_DIR`, `CUBICAI_WORKSPACE`; else in
/// ~/.local/share/cubicai-server, which the service may write).
#[derive(Clone, Debug)]
pub struct Paths {
    pub bots: PathBuf,
    pub workspace: PathBuf,
    /// The Bots' computer (None: Bots have no computer, as in most tests).
    pub computer: Option<computer::Setup>,
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let share = home.join(".local/share/cubicai-server");
        Paths {
            bots: std::env::var_os("CUBICAI_BOTS_DIR").map(PathBuf::from).unwrap_or_else(|| share.join("bots")),
            workspace: std::env::var_os("CUBICAI_WORKSPACE").map(PathBuf::from).unwrap_or_else(|| share.join("workspace")),
            computer: Some(computer::Setup::from_env(&home)),
        }
    }
}

/// A Bot's profile (bot.json).
#[derive(Clone, Debug, PartialEq)]
pub struct Bot {
    pub id: String,
    pub name: String,
    pub job: String,
    /// "#rrggbb", the colour of its cube on the phone.
    pub color: String,
    /// One of [`PROVIDERS`].
    pub provider: String,
    pub model: String,
    /// Whether it may use the computer (terminal and browser), when there is one.
    pub computer: bool,
    pub created: f64,
}

impl Bot {
    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "name": self.name, "job": self.job, "color": self.color, "provider": self.provider, "model": self.model, "computer": self.computer, "created": self.created})
    }

    pub fn from_json(v: &Value) -> Option<Bot> {
        Some(Bot {
            id: v["id"].as_str()?.to_string(),
            name: v["name"].as_str()?.to_string(),
            job: v["job"].as_str().unwrap_or("").to_string(),
            color: v["color"].as_str().unwrap_or("#2f6bff").to_string(),
            provider: v["provider"].as_str().unwrap_or("local").to_string(),
            model: v["model"].as_str().unwrap_or("").to_string(),
            computer: v["computer"].as_bool().unwrap_or(true),
            created: v["created"].as_f64().unwrap_or(0.0),
        })
    }

    /// Takes the editable fields from a phone's request; None leaves a field as it is.
    pub fn apply(&mut self, v: &Value) -> Result<(), String> {
        let text = |k: &str| v[k].as_str().map(|s| s.trim().to_string());
        if let Some(n) = text("name") {
            self.name = n;
        }
        if let Some(j) = text("job") {
            self.job = j;
        }
        if let Some(c) = text("color") {
            self.color = c;
        }
        if let Some(p) = text("provider") {
            self.provider = p;
        }
        if let Some(m) = text("model") {
            self.model = m;
        }
        if let Some(c) = v["computer"].as_bool() {
            self.computer = c;
        }
        self.check()
    }

    pub fn check(&self) -> Result<(), String> {
        let n = self.name.chars().count();
        if n == 0 || n > 40 {
            return Err("Give the Bot a name (up to 40 characters).".into());
        }
        if self.job.chars().count() > 8000 {
            return Err("The job is too long (up to 8000 characters).".into());
        }
        if !(self.color.len() == 7 && self.color.starts_with('#') && self.color[1..].chars().all(|c| c.is_ascii_hexdigit())) {
            return Err("The colour must look like #2f6bff.".into());
        }
        if !PROVIDERS.contains(&self.provider.as_str()) {
            return Err(format!("Unknown provider {:?}.", self.provider));
        }
        if self.model.is_empty() || self.model.len() > 200 {
            return Err("Pick a model for the Bot.".into());
        }
        // xAI's team-of-agents models take no tools of ours, and a Bot needs them.
        if self.provider == "xai" && self.model.contains("multi-agent") {
            return Err("Grok's multi-agent models can't use a Bot's tools; pick another Grok model.".into());
        }
        Ok(())
    }
}

/// The cloud keys the phone handed over (keys.json in the config folder, 0600).
pub struct Keys {
    file: PathBuf,
    lock: Mutex<()>,
}

impl Keys {
    pub fn new(config_dir: &Path) -> Keys {
        Keys { file: config_dir.join("keys.json"), lock: Mutex::new(()) }
    }

    fn all(&self) -> Map<String, Value> {
        std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default()
    }

    pub fn get(&self, provider: &str) -> Option<String> {
        self.all().get(provider).and_then(|v| v.as_str()).filter(|k| !k.is_empty()).map(String::from)
    }

    pub fn custom_url(&self) -> Option<String> {
        self.all().get("custom_url").and_then(|v| v.as_str()).filter(|u| !u.is_empty()).map(|u| u.trim_end_matches('/').to_string())
    }

    /// Which keys are here (never the keys themselves).
    pub fn present(&self) -> Value {
        let all = self.all();
        let mut out = Map::new();
        for p in ["xai", "gemini", "openrouter", "custom"] {
            out.insert(p.into(), all.get(p).and_then(|v| v.as_str()).is_some_and(|k| !k.is_empty()).into());
        }
        out.insert("custom_url".into(), all.get("custom_url").cloned().unwrap_or(Value::Null));
        Value::Object(out)
    }

    /// Merges the given keys ("" removes one).
    pub fn set(&self, changes: &Value) -> Result<(), String> {
        let _g = self.lock.lock().unwrap();
        let mut all = self.all();
        let Some(m) = changes.as_object() else { return Err("Expected an object of keys.".into()) };
        for (k, v) in m {
            if !["xai", "gemini", "openrouter", "custom", "custom_url"].contains(&k.as_str()) {
                return Err(format!("Unknown key {k:?}."));
            }
            match v.as_str().map(str::trim) {
                Some("") | None => {
                    all.remove(k);
                }
                Some(s) if s.len() > 500 => return Err(format!("The {k} key is too long.")),
                Some(s) => {
                    all.insert(k.clone(), s.into());
                }
            }
        }
        write_private(&self.file, serde_json::to_string_pretty(&Value::Object(all)).unwrap().as_bytes())
    }
}

/// Writes a whole file atomically, owner-only.
pub fn write_private(file: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
    let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    std::fs::rename(&tmp, file).map_err(|e| e.to_string())
}

/// One run: a message (and any sent while it works) taken through to the Bot's answer. Its events
/// are kept until the next run starts, so a phone that comes back mid-run (or just after) catches up.
#[derive(Default)]
pub struct Run {
    events: Mutex<Vec<Value>>,
    wake: Condvar,
    pub done: AtomicBool,
    pub stop: AtomicBool,
    /// Messages that arrived while the Bot was working, taken before its next turn.
    inbox: Mutex<Vec<Value>>,
    /// The routine that started it ({id, name}), for the feed.
    pub routine: Mutex<Option<Value>>,
    /// An action waiting for the user's yes: {id, text, answer?}.
    pub approval: Mutex<Option<Value>>,
    /// What the user said no to in this turn: the turn then ends with a plain reply saying so.
    denied: Mutex<Option<String>>,
    answered: Condvar,
}

impl Run {
    pub fn push(&self, mut ev: Value) {
        let mut evs = self.events.lock().unwrap();
        ev["seq"] = (evs.len() as u64).into();
        evs.push(ev);
        self.wake.notify_all();
    }

    /// Events from `after` on, waiting up to `wait` for new ones. The bool says the run is over and
    /// nothing is left to send.
    pub fn since(&self, after: usize, wait: Duration) -> (Vec<Value>, bool) {
        let mut evs = self.events.lock().unwrap();
        if evs.len() <= after && !self.done.load(Ordering::SeqCst) {
            evs = self.wake.wait_timeout(evs, wait).unwrap().0;
        }
        let new: Vec<Value> = evs.iter().skip(after).cloned().collect();
        let over = self.done.load(Ordering::SeqCst) && evs.len() <= after + new.len();
        (new, over)
    }

    /// Where a phone joining now should follow from: the start of the current turn (so it gets every
    /// word of it), or the end when the run is over.
    pub fn resume_at(&self) -> usize {
        let evs = self.events.lock().unwrap();
        if self.done.load(Ordering::SeqCst) {
            return evs.len();
        }
        evs.iter().rposition(|e| e["type"] == "thinking").unwrap_or(evs.len())
    }

    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Bots {
    pub paths: Paths,
    pub keys: Keys,
    settings: Settings,
    ollama: Ollama,
    searxng: String,
    default_ctx: u64,
    media: Option<Arc<crate::jobs::Jobs>>,
    runs: Mutex<HashMap<String, Arc<Run>>>,
    /// Chat files are appended to from runs and from the phone's messages.
    files: Mutex<()>,
    /// routines.json is changed by the phone, the Bots' tools and the scheduler.
    routines_lock: Mutex<()>,
    feed_lock: Mutex<()>,
    pub computer: Option<computer::Computer>,
}

pub fn now_ms() -> u64 {
    (crate::time::now() * 1000.0) as u64
}

pub fn new_id() -> String {
    let mut b = [0u8; 8];
    let _ = ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut b);
    crate::identity::hex(&b)
}

fn valid_id(id: &str) -> bool {
    id.len() == 16 && id.chars().all(|c| c.is_ascii_hexdigit())
}

impl Bots {
    pub fn new(paths: Paths, config_dir: &Path, ollama: &str, searxng: &str, default_ctx: u64, media: Option<Arc<crate::jobs::Jobs>>) -> Arc<Bots> {
        let _ = std::fs::create_dir_all(&paths.bots);
        let _ = std::fs::create_dir_all(&paths.workspace);
        let computer = paths.computer.as_ref().map(|c| computer::Computer::new(c, &paths.workspace));
        let bots = Arc::new(Bots {
            keys: Keys::new(config_dir),
            settings: Settings::new(config_dir),
            ollama: Ollama::new(ollama),
            searxng: searxng.to_string(),
            default_ctx,
            media,
            paths,
            runs: Mutex::new(HashMap::new()),
            files: Mutex::new(()),
            routines_lock: Mutex::new(()),
            feed_lock: Mutex::new(()),
            computer,
        });
        // The scheduler: checks the routines every 15 s while the gateway (this Bots) is alive.
        let weak: Weak<Bots> = Arc::downgrade(&bots);
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(15));
            match weak.upgrade() {
                Some(b) => {
                    b.tick(crate::time::now() as i64);
                    if let Some(c) = &b.computer {
                        c.idle_check();
                    }
                }
                None => return,
            }
        });
        bots
    }

    fn dir(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| self.paths.bots.join(id))
    }

    pub fn get(&self, id: &str) -> Option<Bot> {
        let d = self.dir(id)?;
        let v = serde_json::from_slice::<Value>(&std::fs::read(d.join("bot.json")).ok()?).ok()?;
        Bot::from_json(&v)
    }

    pub fn list(&self) -> Vec<Bot> {
        let mut out: Vec<Bot> = std::fs::read_dir(&self.paths.bots)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| self.get(&e.file_name().to_string_lossy()))
            .collect();
        out.sort_by(|a, b| a.created.total_cmp(&b.created));
        out
    }

    pub fn create(&self, v: &Value) -> Result<Bot, String> {
        if self.list().len() >= MAX_BOTS {
            return Err(format!("You have {MAX_BOTS} Bots already; delete one first."));
        }
        let mut bot = Bot { id: new_id(), name: String::new(), job: String::new(), color: "#2f6bff".into(), provider: "local".into(), model: String::new(), computer: true, created: crate::time::now() };
        bot.apply(v)?;
        self.save(&bot)?;
        if let Some(m) = v["memory"].as_str() {
            self.set_memory(&bot.id, m)?;
        }
        Ok(bot)
    }

    pub fn save(&self, bot: &Bot) -> Result<(), String> {
        let d = self.dir(&bot.id).ok_or("Bad Bot id")?;
        write_private(&d.join("bot.json"), serde_json::to_string_pretty(&bot.to_json()).unwrap().as_bytes())
    }

    /// Deletes a Bot's profile, chat and memory (files it made in the workspace stay).
    pub fn delete(&self, id: &str) -> bool {
        let Some(d) = self.dir(id).filter(|d| d.exists()) else { return false };
        if let Some(r) = self.runs.lock().unwrap().remove(id) {
            r.stop.store(true, Ordering::SeqCst);
        }
        std::fs::remove_dir_all(d).is_ok()
    }

    pub fn memory(&self, id: &str) -> String {
        self.dir(id).and_then(|d| std::fs::read_to_string(d.join("memory.md")).ok()).unwrap_or_default()
    }

    pub fn set_memory(&self, id: &str, text: &str) -> Result<(), String> {
        if text.chars().count() > MEMORY_CHARS {
            return Err(format!("Memory holds up to {MEMORY_CHARS} characters."));
        }
        let d = self.dir(id).ok_or("Bad Bot id")?;
        write_private(&d.join("memory.md"), text.as_bytes())
    }

    /// The whole chat, oldest first. A reply and its tool results carry the time their turn began,
    /// so a message the user sent while the Bot was writing comes after that reply, not before it.
    pub fn messages(&self, id: &str) -> Vec<Value> {
        let Some(d) = self.dir(id) else { return vec![] };
        let _g = self.files.lock().unwrap();
        let mut all: Vec<Value> = std::fs::read_to_string(d.join("chat.jsonl")).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        all.sort_by_key(|m| m["at"].as_u64().unwrap_or(0));
        all
    }

    fn append(&self, id: &str, msg: &Value) {
        let Some(d) = self.dir(id) else { return };
        let _g = self.files.lock().unwrap();
        let file = std::fs::OpenOptions::new().create(true).append(true).open(d.join("chat.jsonl"));
        if let Ok(mut f) = file {
            let _ = f.write_all((msg.to_string() + "\n").as_bytes());
            let _ = std::fs::set_permissions(d.join("chat.jsonl"), std::fs::Permissions::from_mode(0o600));
        }
    }

    pub fn clear(&self, id: &str) {
        if let Some(d) = self.dir(id) {
            let _g = self.files.lock().unwrap();
            let _ = std::fs::remove_file(d.join("chat.jsonl"));
            let _ = std::fs::remove_dir_all(d.join("shots"));
        }
    }

    pub fn run(&self, id: &str) -> Option<Arc<Run>> {
        self.runs.lock().unwrap().get(id).cloned()
    }

    pub fn running(&self, id: &str) -> bool {
        self.run(id).is_some_and(|r| !r.done.load(Ordering::SeqCst))
    }

    pub fn stop(&self, id: &str) -> bool {
        match self.run(id) {
            Some(r) if !r.done.load(Ordering::SeqCst) => {
                r.stop.store(true, Ordering::SeqCst);
                let _g = r.approval.lock().unwrap();
                r.answered.notify_all();
                true
            }
            _ => false,
        }
    }

    /// The phone's message: stored, then handed to the running Bot or a new run.
    pub fn send(self: &Arc<Bots>, id: &str, text: &str) -> Result<Value, String> {
        let bot = self.get(id).ok_or("No such Bot.")?;
        let text = text.trim();
        if text.is_empty() {
            return Err("Write something first.".into());
        }
        if text.chars().count() > 50_000 {
            return Err("That message is too long.".into());
        }
        let msg = json!({"id": new_id(), "role": "user", "text": text, "at": now_ms()});
        self.start(bot, msg, None)
    }

    /// Stores a user (or routine) message and hands it to the running Bot, or starts a run.
    fn start(self: &Arc<Bots>, bot: Bot, msg: Value, routine: Option<Value>) -> Result<Value, String> {
        let id = bot.id.clone();
        let id = id.as_str();
        self.append(id, &msg);
        let mut runs = self.runs.lock().unwrap();
        if let Some(r) = runs.get(id).filter(|r| !r.done.load(Ordering::SeqCst)) {
            r.inbox.lock().unwrap().push(msg.clone());
            r.push(json!({"type": "message", "message": msg}));
            return Ok(json!({"message": msg, "queued": true}));
        }
        let run = Arc::new(Run::default());
        *run.routine.lock().unwrap() = routine;
        run.push(json!({"type": "start"}));
        run.push(json!({"type": "message", "message": msg}));
        runs.insert(id.to_string(), run.clone());
        drop(runs);
        let me = self.clone();
        std::thread::spawn(move || me.work(bot, run));
        Ok(json!({"message": msg, "queued": false}))
    }

    pub fn routines(&self, id: &str) -> Vec<routines::Routine> {
        let Some(d) = self.dir(id) else { return vec![] };
        let v: Value = std::fs::read(d.join("routines.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!([]));
        v.as_array().into_iter().flatten().filter_map(routines::Routine::from_json).collect()
    }

    fn save_routines(&self, id: &str, list: &[routines::Routine]) -> Result<(), String> {
        let d = self.dir(id).ok_or("Bad Bot id")?;
        let v: Vec<Value> = list.iter().map(|r| r.to_json()).collect();
        write_private(&d.join("routines.json"), serde_json::to_string_pretty(&v).unwrap().as_bytes())
    }

    /// A new routine. The user's start at once; a Bot's wait for the user's approval.
    pub fn add_routine(&self, bot: &Bot, v: &Value, by: &str) -> Result<routines::Routine, String> {
        let _g = self.routines_lock.lock().unwrap();
        let mut list = self.routines(&bot.id);
        if list.len() >= routines::MAX_ROUTINES {
            return Err(format!("{} has {} routines already; delete one first.", bot.name, routines::MAX_ROUTINES));
        }
        if by == "bot" && list.iter().filter(|r| !r.approved).count() >= MAX_PENDING {
            return Err(format!("You have {MAX_PENDING} routines waiting for the user's approval already; wait for their answer."));
        }
        let r = routines::Routine::new(v, by, crate::time::now() as i64)?;
        list.push(r.clone());
        self.save_routines(&bot.id, &list)?;
        if !r.approved {
            self.push_feed(bot, "approval", &format!("wants to run “{}” {}. Approve it in the Bot's chat.", r.name, r.schedule.describe()), Some(&r), false);
        }
        Ok(r)
    }

    pub fn update_routine(&self, id: &str, rid: &str, v: &Value) -> Result<routines::Routine, String> {
        let _g = self.routines_lock.lock().unwrap();
        let mut list = self.routines(id);
        let r = list.iter_mut().find(|r| r.id == rid).ok_or("No such routine (it may have been deleted).")?;
        r.apply(v, crate::time::now() as i64)?;
        let out = r.clone();
        self.save_routines(id, &list)?;
        Ok(out)
    }

    pub fn delete_routine(&self, id: &str, rid: &str) -> bool {
        let _g = self.routines_lock.lock().unwrap();
        let mut list = self.routines(id);
        let before = list.len();
        list.retain(|r| r.id != rid);
        before != list.len() && self.save_routines(id, &list).is_ok()
    }

    /// Starts a routine's run now (from the scheduler, or the phone's "Run now"). `late` is the time
    /// it was due when that was a while ago.
    pub fn run_routine(self: &Arc<Bots>, bot: &Bot, r: &routines::Routine, late: Option<i64>) -> Result<Value, String> {
        let routine = json!({"id": r.id, "name": r.name, "when": r.schedule.describe(), "late": late});
        let msg = json!({"id": new_id(), "role": "user", "text": r.task, "at": now_ms(), "routine": routine});
        crate::log(&format!("bot {} routine {} started", bot.id, r.id));
        self.start(bot.clone(), msg, Some(json!({"id": r.id, "name": r.name})))
    }

    /// The scheduler's check: starts every routine whose time has come (skipping, until next time,
    /// any that's more than LATE_LIMIT_SECS late because the PC was off).
    pub fn tick(self: &Arc<Bots>, now: i64) {
        for bot in self.list() {
            let due: Vec<(routines::Routine, Option<i64>)> = {
                let _g = self.routines_lock.lock().unwrap();
                let mut list = self.routines(&bot.id);
                let mut due = vec![];
                let mut changed = false;
                for r in list.iter_mut().filter(|r| r.due(now)) {
                    changed = true;
                    let planned = r.next_run.unwrap_or(now);
                    r.next_run = r.schedule.next_after(now);
                    if r.next_run.is_none() {
                        r.enabled = false;
                    }
                    if now - planned > LATE_LIMIT_SECS {
                        crate::log(&format!("bot {} routine {} skipped: {} min late", bot.id, r.id, (now - planned) / 60));
                        continue;
                    }
                    r.last_run = Some(now);
                    due.push((r.clone(), (now - planned > 120).then_some(planned)));
                }
                if changed {
                    let _ = self.save_routines(&bot.id, &list);
                }
                due
            };
            for (r, late) in due {
                if let Err(e) = self.run_routine(&bot, &r, late) {
                    crate::log(&format!("bot {} routine {} didn't start: {e}", bot.id, r.id));
                }
            }
        }
    }

    /// What happened lately, for the phone's notifications: runs that ended, routines waiting for a yes.
    /// Also says up to when it's complete: an item written later has a later time (both are taken
    /// under the feed's lock). One stamped in that same millisecond can come twice; the phone drops repeats by id.
    pub fn feed(&self, after_ms: u64) -> (Vec<Value>, u64) {
        let _g = self.feed_lock.lock().unwrap();
        let upto = now_ms().saturating_sub(1);
        let items = std::fs::read_to_string(self.paths.bots.join("feed.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v["at"].as_u64().unwrap_or(0) > after_ms)
            .collect();
        (items, upto)
    }

    fn push_feed(&self, bot: &Bot, kind: &str, text: &str, routine: Option<&routines::Routine>, quiet: bool) {
        let preview: String = text.trim().chars().take(300).collect();
        let _g = self.feed_lock.lock().unwrap();
        let mut item = json!({"id": new_id(), "at": now_ms(), "kind": kind, "bot": bot.id, "name": bot.name, "color": bot.color, "text": preview, "quiet": quiet});
        if let Some(r) = routine {
            item["routine"] = json!({"id": r.id, "name": r.name});
        }
        let file = self.paths.bots.join("feed.jsonl");
        let mut lines: Vec<String> = std::fs::read_to_string(&file).unwrap_or_default().lines().map(String::from).collect();
        lines.push(item.to_string());
        let start = lines.len().saturating_sub(FEED_ITEMS);
        let _ = write_private(&file, (lines[start..].join("\n") + "\n").as_bytes());
    }

    /// What a run came to, for a phone that wasn't watching.
    fn feed_run(&self, bot: &Bot, run: &Run, outcome: &Result<(), String>, started_ms: u64) {
        let routine = run.routine.lock().unwrap().clone();
        let r = routine.as_ref().and_then(|v| self.routines(&bot.id).into_iter().find(|r| r.id == v["id"].as_str().unwrap_or("")));
        match outcome {
            Err(e) => self.push_feed(bot, "error", e, r.as_ref(), false),
            Ok(()) if run.stop.load(Ordering::SeqCst) => {}
            Ok(()) => {
                let last = self.messages(&bot.id).into_iter().rev().find(|m| m["role"] == "assistant" && m["at"].as_u64().unwrap_or(0) >= started_ms && m["text"].as_str().is_some_and(|t| !t.trim().is_empty()));
                let text = last.as_ref().and_then(|m| m["text"].as_str()).unwrap_or("").to_string();
                // A routine with nothing to say stays in the chat without a notification.
                let quiet = routine.is_some() && text.trim().to_lowercase().trim_start_matches(['*', '_']).starts_with("nothing new");
                self.push_feed(bot, if routine.is_some() { "routine" } else { "reply" }, &text, r.as_ref(), quiet);
            }
        }
    }

    /// Whether this Bot gets the computer's tools now.
    fn has_computer(&self, bot: &Bot) -> bool {
        bot.computer && self.computer.as_ref().is_some_and(|c| c.available())
    }

    /// The user's answer to an action waiting in the Bot's run. False when nothing with that id waits.
    pub fn answer(&self, id: &str, approval: &str, allow: bool) -> bool {
        let Some(run) = self.run(id) else { return false };
        let mut a = run.approval.lock().unwrap();
        match a.as_mut() {
            Some(v) if v["id"] == approval && v["answer"].is_null() => {
                v["answer"] = allow.into();
                run.answered.notify_all();
                true
            }
            _ => false,
        }
    }

    /// Asks the user (on the phone, with a notification) and waits: Some(answer), or None when the run
    /// was stopped or nobody answered within APPROVAL_WAIT.
    fn ask(&self, bot: &Bot, run: &Run, what: &str) -> Option<bool> {
        let id = new_id();
        *run.approval.lock().unwrap() = Some(json!({"id": id, "text": what}));
        run.push(json!({"type": "approval", "id": id, "text": what}));
        let mut first = what.chars();
        let lower: String = first.next().map(|c| c.to_lowercase().chain(first).collect()).unwrap_or_default();
        self.push_feed(bot, "approval", &format!("wants to {lower}. Allow it in the Bot's chat."), None, false);
        let end = std::time::Instant::now() + APPROVAL_WAIT;
        let mut a = run.approval.lock().unwrap();
        let answer = loop {
            if let Some(v) = a.as_ref().and_then(|v| v["answer"].as_bool()) {
                break Some(v);
            }
            let left = end.saturating_duration_since(std::time::Instant::now());
            if run.stop.load(Ordering::SeqCst) || left.is_zero() {
                break None;
            }
            a = run.answered.wait_timeout(a, left.min(Duration::from_secs(5))).unwrap().0;
        };
        *a = None;
        drop(a);
        run.push(json!({"type": "approved", "id": id, "allow": answer}));
        answer
    }

    /// The computer's tools: a command in its terminal, or its browser. A page's screenshot is kept as
    /// bots/<id>/shots/<tool message id>.jpg for the phone.
    fn computer_tool(&self, bot: &Bot, run: &Run, name: &str, args: &str, tmsg_id: &str) -> (tools::Outcome, bool) {
        let a: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
        let out = |text: String, ok: bool| (tools::Outcome { text, citations: vec![], ok }, false);
        let Some(computer) = self.computer.as_ref().filter(|_| bot.computer) else {
            return out("You don't have a computer (the user switched it off, or it isn't set up on this PC).".into(), false);
        };
        if name == "run_command" {
            let command = a["command"].as_str().unwrap_or("").trim();
            if command.is_empty() {
                return out("run_command needs a command.".into(), false);
            }
            let timeout = a["timeout"].as_u64().unwrap_or(computer::DEFAULT_TIMEOUT);
            return match computer.run(command, timeout, &run.stop) {
                Ok(r) => {
                    let how = if r.timed_out {
                        format!("Stopped after {:.0} s (the time limit; pass a longer timeout, up to {} s, or run it in the background with &)", r.secs, computer::MAX_TIMEOUT)
                    } else if r.stopped {
                        "Stopped by the user".to_string()
                    } else {
                        format!("Exit code {} ({:.1} s)", r.code.unwrap_or(-1), r.secs)
                    };
                    let body = if r.output.trim().is_empty() { "(no output)".to_string() } else { r.output };
                    out(format!("{how}:\n{body}"), r.code == Some(0))
                }
                Err(e) => out(e, false),
            };
        }
        let cmd = match name {
            "browser_open" => json!({"do": "open", "url": a["url"]}),
            "browser_click" => json!({"do": "click", "element": a["element"]}),
            "browser_type" => json!({"do": "type", "element": a["element"], "text": a["text"], "submit": a["submit"]}),
            "browser_scroll" => json!({"do": "scroll", "direction": a["direction"]}),
            "browser_back" => json!({"do": "back"}),
            _ => return out(format!("There's no tool called {name:?}."), false),
        };
        let mut page = match computer.browse(&cmd) {
            Ok(p) => p,
            Err(e) => return out(e, false),
        };
        if page["needs_approval"] == true {
            let what = page["what"].as_str().unwrap_or("Send something from the browser").to_string();
            match self.ask(bot, run, &what) {
                Some(true) => {
                    let mut again = cmd.clone();
                    again["approved"] = true.into();
                    page = match computer.browse(&again) {
                        Ok(p) => p,
                        Err(e) => return out(e, false),
                    };
                }
                Some(false) => {
                    *run.denied.lock().unwrap() = Some(what.clone());
                    return out(
                        format!("NOT DONE. The user said no to this: {what}. Nothing was typed or sent. In your reply, say plainly that you didn't do it because they said no, and ask what they'd like instead."),
                        false,
                    );
                }
                None if run.stop.load(Ordering::SeqCst) => return out("Not done: the user stopped you.".into(), false),
                None => return out(format!("Not done: the user didn't answer within {} minutes ({what}). Tell them what you wanted to do.", APPROVAL_WAIT.as_secs() / 60), false),
            }
        }
        if page["ok"] != true {
            return out(page["error"].as_str().unwrap_or("The browser couldn't do that.").to_string(), false);
        }
        let shot = page["shot"].as_str().and_then(|s| computer.shot(s));
        let saved = shot.is_some_and(|from| self.keep_shot(&bot.id, &from, tmsg_id));
        let (t, cites) = describe_page(&page);
        (tools::Outcome { text: t, citations: cites, ok: true }, saved)
    }

    /// Moves a screenshot into the Bot's folder (dropping the oldest past SHOTS_KEPT).
    fn keep_shot(&self, id: &str, from: &Path, name: &str) -> bool {
        let Some(d) = self.dir(id).map(|d| d.join("shots")) else { return false };
        if std::fs::create_dir_all(&d).is_err() {
            return false;
        }
        let to = d.join(format!("{name}.jpg"));
        let ok = std::fs::copy(from, &to).is_ok();
        let _ = std::fs::remove_file(from);
        let _ = std::fs::set_permissions(&to, std::fs::Permissions::from_mode(0o600));
        let mut all: Vec<(std::time::SystemTime, PathBuf)> =
            std::fs::read_dir(&d).into_iter().flatten().flatten().filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
        if all.len() > SHOTS_KEPT {
            all.sort();
            for (_, p) in &all[..all.len() - SHOTS_KEPT] {
                let _ = std::fs::remove_file(p);
            }
        }
        ok
    }

    /// A screenshot of a Bot's, by its tool message's id.
    pub fn shot(&self, id: &str, name: &str) -> Option<Vec<u8>> {
        if !valid_id(name) {
            return None;
        }
        std::fs::read(self.dir(id)?.join("shots").join(format!("{name}.jpg"))).ok()
    }

    /// The tools that act on the Bot's own routines (the others are in tools.rs).
    fn routine_tool(&self, bot: &Bot, name: &str, args: &str) -> tools::Outcome {
        let a: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
        let done = |text: String, ok: bool| tools::Outcome { text, citations: vec![], ok };
        match name {
            "schedule_routine" => {
                let days: Vec<Value> = match &a["days"] {
                    Value::Array(d) => d.clone(),
                    Value::String(s) => s.split([',', ' ']).filter(|d| !d.is_empty()).map(|d| json!(d)).collect(),
                    _ => vec![],
                };
                let v = json!({"name": a["name"], "task": a["task"], "schedule": {"kind": a["kind"], "time": a["time"], "days": days, "hours": a["hours"], "date": a["date"]}});
                match self.add_routine(bot, &v, "bot") {
                    Ok(r) => done(format!("Proposed the routine “{}” ({}). It starts once the user approves it on their phone; tell them so in your reply.", r.name, r.schedule.describe()), true),
                    Err(e) => done(format!("Couldn't propose it: {e}"), false),
                }
            }
            "cancel_routine" => {
                let key = a["name"].as_str().unwrap_or("").trim().to_lowercase();
                let found = self.routines(&bot.id).into_iter().find(|r| r.id == key || r.name.to_lowercase() == key);
                match found {
                    // Not yet approved: just withdrawn. Approved: switched off, so the user can turn it back on.
                    Some(r) if !r.approved => {
                        self.delete_routine(&bot.id, &r.id);
                        done(format!("Withdrew the proposed routine “{}”.", r.name), true)
                    }
                    Some(r) => match self.update_routine(&bot.id, &r.id, &json!({"enabled": false})) {
                        Ok(_) => done(format!("Switched off the routine “{}” (the user can switch it back on).", r.name), true),
                        Err(e) => done(e, false),
                    },
                    None => done(format!("You have no routine called {key:?}."), false),
                }
            }
            _ => done(format!("There's no tool called {name:?}."), false),
        }
    }

    /// The system prompt: who the Bot is, its job, what it remembers, how it works.
    fn system(&self, bot: &Bot) -> String {
        let memory = self.memory(&bot.id);
        format!(
            "You are {name}, one of the user's Bots in CubicAI. You live on the user's PC and work for them; they talk to you from their phone.\n\n\
             Your job:\n{job}\n\n\
             What you remember (from earlier; keep it current with the remember and forget tools):\n{memory}\n\n\
             How you work:\n\
             - Use your tools when they help: search the web and read pages for anything current or that you'd otherwise guess; keep files you make in your workspace folder (it's shared with the user's other Bots, so use clear names).\n\
             - When you learn something lasting (a preference, a rule, a fact about the user or your job), save it with remember. Don't save one-off details.\n\
             - Do the work, then reply briefly: what you did and what you found. The user reads you on a phone, so keep it short and clear, and link your web sources.\n\
             - If something needs the user (a decision, a login, a missing detail), ask once, clearly.\n\
             {computer}             - You only act when the user writes or a routine starts you. For anything regular or later (\"every morning\", \"on Friday\"), propose a routine with schedule_routine; it runs once the user approves it. Never promise to do something later any other way.\n\n\
             Your routines:\n{routines}\n\n\
             Current date and time: {now}",
            name = bot.name,
            job = if bot.job.is_empty() { "(none given yet: ask the user what they'd like you to do)" } else { &bot.job },
            memory = if memory.trim().is_empty() { "(nothing yet)".to_string() } else { memory.trim().to_string() },
            computer = if self.has_computer(bot) {
                "- You have a computer: a Linux box (Debian, Python 3 with pip, git, curl, jq) whose folder /workspace is your workspace. \
                 run_command runs bash there (pip install --user works; anything outside /workspace is gone when the computer is idle for 10 minutes). \
                 The browser tools drive a real Chromium: open a page, then use the numbered elements to click and type; use it when read_page isn't enough \
                 (pages that need JavaScript, forms, several steps). It reaches public websites only. Submitting a form that sends data asks the user first. \
                 If a command fails, fix it and run it again; report only what commands and pages actually showed.\n"
            } else {
                ""
            },
            routines = {
                let list = self.routines(&bot.id);
                if list.is_empty() {
                    "(none)".to_string()
                } else {
                    list.iter()
                        .map(|r| {
                            let state = if !r.approved { "waiting for the user's approval" } else if !r.enabled { "switched off" } else { "on" };
                            format!("- “{}” {} ({state}): {}", r.name, r.schedule.describe(), r.task.chars().take(200).collect::<String>())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            },
            now = crate::time::now_local().long(),
        )
    }

    /// Stored messages → the model's messages (OpenAI shape), recent ones only, old tool results cut.
    pub fn conversation(stored: &[Value]) -> Vec<Value> {
        let mut recent: Vec<&Value> = stored.iter().rev().take(HISTORY_MESSAGES).collect::<Vec<_>>().into_iter().rev().collect();
        // Start at a message from the user, so no tool result is left without the call it answers.
        while recent.first().is_some_and(|m| m["role"] != "user") {
            recent.remove(0);
        }
        let last_user = recent.iter().rposition(|m| m["role"] == "user").unwrap_or(0);
        let mut out = vec![];
        for (i, m) in recent.iter().enumerate() {
            let text = m["text"].as_str().unwrap_or("");
            match m["role"].as_str() {
                Some("user") if m["routine"].is_object() => out.push(json!({"role": "user", "content": routine_prompt(&m["routine"], text)})),
                Some("user") => out.push(json!({"role": "user", "content": text})),
                Some("assistant") => {
                    // Only calls that have their result here (a provider refuses a call left unanswered).
                    let answered = |id: &Value| recent[i + 1..].iter().any(|r| r["role"] == "tool" && &r["call"] == id);
                    let calls: Vec<Value> = m["calls"].as_array().into_iter().flatten().filter(|c| answered(&c["id"])).cloned().collect();
                    if text.is_empty() && calls.is_empty() {
                        continue;
                    }
                    let mut a = json!({"role": "assistant", "content": text});
                    if !calls.is_empty() {
                        a["tool_calls"] = calls
                            .iter()
                            .map(|c| json!({"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"].as_str().unwrap_or("{}")}}))
                            .collect();
                    }
                    out.push(a);
                }
                Some("tool") => {
                    let body: String = if i < last_user && text.chars().count() > OLD_RESULT_CHARS {
                        text.chars().take(OLD_RESULT_CHARS).collect::<String>() + "\n[…cut]"
                    } else {
                        text.to_string()
                    };
                    out.push(json!({"role": "tool", "tool_call_id": m["call"], "name": m["tool"], "content": body}));
                }
                _ => {}
            }
        }
        out
    }

    /// Where the Bot's model is, or why it can't be reached.
    fn endpoint(&self, bot: &Bot) -> Result<brain::Endpoint, String> {
        let cloud = |name: &str, base: &str| -> Result<brain::Endpoint, String> {
            let key = self.keys.get(&bot.provider).ok_or_else(|| format!("Your PC has no {name} key yet. Open the Bot on the phone once with the key set in Settings, or add it there."))?;
            Ok(brain::Endpoint::Cloud { name: name.into(), base: base.into(), key, model: bot.model.clone() })
        };
        match bot.provider.as_str() {
            "local" => Ok(brain::Endpoint::Local { model: bot.model.clone() }),
            "xai" => cloud("Grok", "https://api.x.ai/v1"),
            "gemini" => cloud("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
            "openrouter" => cloud("OpenRouter", "https://openrouter.ai/api/v1"),
            "custom" => {
                let base = self.keys.custom_url().ok_or("Your PC has no custom provider URL yet.")?;
                Ok(brain::Endpoint::Cloud { name: "your custom provider".into(), base, key: self.keys.get("custom").unwrap_or_default(), model: bot.model.clone() })
            }
            p => Err(format!("Unknown provider {p:?}.")),
        }
    }

    /// A run, on its own thread: turns and tool calls until the Bot answers and nothing new came in.
    fn work(self: Arc<Bots>, bot: Bot, run: Arc<Run>) {
        let started = std::time::Instant::now();
        let started_ms = now_ms();
        crate::log(&format!("bot {} run started ({} {})", bot.id, bot.provider, bot.model));
        let mut steps = 0;
        let outcome = loop {
            let (n, outcome) = match self.endpoint(&bot) {
                Ok(e) => self.steps(&bot, &run, &e),
                Err(m) => (0, Err(m)),
            };
            steps += n;
            if let Err(m) = &outcome {
                let msg = json!({"id": new_id(), "role": "assistant", "text": "", "error": m, "at": now_ms(), "model": bot.model});
                self.append(&bot.id, &msg);
                run.push(json!({"type": "message", "message": msg}));
            }
            // Done only once nothing is waiting, under the same lock `send` takes, so a message sent
            // just now either reached this run (and gets answered here) or starts the next one.
            let _runs = self.runs.lock().unwrap();
            if outcome.is_ok() && !run.stop.load(Ordering::SeqCst) && !run.inbox.lock().unwrap().is_empty() {
                continue;
            }
            // The feed first, so a phone that sees the run end finds its item.
            self.feed_run(&bot, &run, &outcome, started_ms);
            run.done.store(true, Ordering::SeqCst);
            run.push(json!({"type": "done", "error": outcome.as_ref().err(), "stopped": run.stop.load(Ordering::SeqCst)}));
            break outcome;
        };
        crate::log(&format!("bot {} run ended after {steps} turns in {:.1} s ({})", bot.id, started.elapsed().as_secs_f64(), if outcome.is_ok() { "ok" } else { "error" }));
    }

    fn steps(&self, bot: &Bot, run: &Run, endpoint: &brain::Endpoint) -> (usize, Result<(), String>) {
        let tools = tools::definitions(self.has_computer(bot));
        let ctx = tools::Ctx { searxng: &self.searxng, workspace: &self.paths.workspace, memory: &self.paths.bots.join(&bot.id).join("memory.md") };
        let mut turns = 0;
        let mut since_user = 0;
        // A reply that says it saved a file it never wrote gets one note asking for the real call.
        let mut nudged = false;
        let mut nudge: Option<String> = None;
        loop {
            // Messages sent meanwhile are already in the chat file: the next turn reads them.
            run.inbox.lock().unwrap().clear();
            if run.stop.load(Ordering::SeqCst) {
                return (turns, Ok(()));
            }
            if since_user >= MAX_STEPS {
                let msg = json!({"id": new_id(), "role": "assistant", "text": format!("I stopped after {MAX_STEPS} steps without finishing. Tell me how to go on, or narrow the task down."), "at": now_ms(), "model": bot.model});
                self.append(&bot.id, &msg);
                run.push(json!({"type": "message", "message": msg}));
                return (turns, Ok(()));
            }
            let mut messages = vec![json!({"role": "system", "content": self.system(bot)})];
            messages.extend(Self::conversation(&self.messages(&bot.id)));
            if let Some(n) = nudge.take() {
                messages.push(json!({"role": "user", "content": n}));
            }
            let turn_start = std::time::Instant::now();
            let at = now_ms();
            run.push(json!({"type": "thinking"}));
            if endpoint.is_local()
                && let Some(m) = &self.media
            {
                m.comfy.free_for_chat();
            }
            let ctx_size = self.settings.num_ctx(self.default_ctx);
            let result = brain::turn(endpoint, &self.ollama, &messages, &tools, ctx_size, self.settings.keep_alive(), &run.stop, &mut |d| match d {
                brain::Delta::Text(t) => run.push(json!({"type": "text", "delta": t})),
                brain::Delta::Reasoning(t) => run.push(json!({"type": "reasoning", "delta": t})),
            });
            turns += 1;
            since_user += 1;
            let turn = match result {
                Ok(t) => t,
                Err(_) if run.stop.load(Ordering::SeqCst) => return (turns, Ok(())),
                Err(e) => return (turns, Err(e)),
            };
            let calls: Vec<Value> = turn.calls.iter().map(|c| json!({"id": c.id, "name": c.name, "args": c.args})).collect();
            let mut msg = json!({"id": new_id(), "role": "assistant", "text": turn.text, "at": at, "model": bot.model, "took": turn_start.elapsed().as_millis() as u64});
            if !turn.reasoning.is_empty() {
                msg["reasoning"] = turn.reasoning.clone().into();
            }
            if !calls.is_empty() {
                msg["calls"] = calls.into();
            }
            if let Some((p, o)) = turn.usage {
                msg["usage"] = json!({"prompt": p, "output": o});
            }
            if run.stop.load(Ordering::SeqCst) {
                msg["stopped"] = true.into();
            }
            if !turn.text.is_empty() || !turn.calls.is_empty() || run.stop.load(Ordering::SeqCst) {
                self.append(&bot.id, &msg);
                run.push(json!({"type": "message", "message": msg}));
            }
            if run.stop.load(Ordering::SeqCst) {
                return (turns, Ok(()));
            }
            if turn.calls.is_empty() {
                if turn.text.is_empty() {
                    return (turns, Err("The model returned an empty reply.".into()));
                }
                if !nudged && let Some(path) = tools::claimed_unsaved(&turn.text, &self.paths.workspace) {
                    nudged = true;
                    nudge = Some(format!(
                        "(Note from the system, not the user: you said {path} is saved, but you didn't call write_file, so it doesn't exist. \
                         Call write_file now with the file's full content, then confirm in one short line.)"
                    ));
                    continue;
                }
                // Answered. A message sent meanwhile is picked up by `work`, which checks under the lock.
                return (turns, Ok(()));
            }
            for c in &turn.calls {
                let tid = new_id();
                // Every call gets a result, so the chat never holds a call a provider would refuse.
                let (out, shot) = if run.stop.load(Ordering::SeqCst) {
                    (tools::Outcome { text: "Not done: the user stopped you before this ran.".into(), citations: vec![], ok: false }, false)
                } else {
                    run.push(json!({"type": "tool", "id": c.id, "name": c.name, "args": c.args}));
                    if tools::ROUTINE_TOOLS.contains(&c.name.as_str()) {
                        (self.routine_tool(bot, &c.name, &c.args), false)
                    } else if tools::COMPUTER_TOOLS.contains(&c.name.as_str()) {
                        self.computer_tool(bot, run, &c.name, &c.args, &tid)
                    } else {
                        (tools::run(&ctx, &c.name, &c.args), false)
                    }
                };
                let mut tmsg = json!({"id": tid, "role": "tool", "call": c.id, "tool": c.name, "text": out.text, "at": at});
                if shot {
                    tmsg["shot"] = true.into();
                }
                if !out.citations.is_empty() {
                    tmsg["citations"] = out.citations.into();
                }
                if !out.ok {
                    tmsg["failed"] = true.into();
                }
                self.append(&bot.id, &tmsg);
                run.push(json!({"type": "message", "message": tmsg}));
            }
            // Said no: the turn ends here with a reply that says so (small models otherwise report the
            // action as done, copying earlier turns where it was).
            if let Some(what) = run.denied.lock().unwrap().take() {
                let mut first = what.chars();
                let lower: String = first.next().map(|c| c.to_lowercase().chain(first).collect()).unwrap_or_default();
                let msg = json!({"id": new_id(), "role": "assistant", "text": format!("I didn't {lower}, because you said no. What would you like me to do instead?"), "at": now_ms(), "model": bot.model});
                self.append(&bot.id, &msg);
                run.push(json!({"type": "message", "message": msg}));
                return (turns, Ok(()));
            }
            if !run.inbox.lock().unwrap().is_empty() {
                since_user = 0;
            }
        }
    }
}

/// A routine's message as the model reads it: what it is, that nobody's watching, and the task.
fn routine_prompt(routine: &Value, task: &str) -> String {
    let name = routine["name"].as_str().unwrap_or("");
    let when = routine["when"].as_str().unwrap_or("");
    let late = routine["late"].as_i64().map(|t| {
        let l = crate::time::local(t);
        format!(" It's running late: it was due at {:02}:{:02}, when the PC was off or busy.", l.hour, l.minute)
    });
    format!(
        "[Your routine “{name}” ({when}) started this, not the user; they aren't watching now.{late} Do the task, then report briefly: your reply is what they'll see in their notification. \
         If there's nothing worth telling them, reply with exactly: Nothing new.]\n\n{task}",
        late = late.unwrap_or_default()
    )
}

/// A browser page as the model reads it: where it is, its text, and the numbered elements.
fn describe_page(p: &Value) -> (String, Vec<String>) {
    let url = p["url"].as_str().unwrap_or("").to_string();
    let mut t = format!("Page: {}\nAddress: {url}\n", p["title"].as_str().unwrap_or(""));
    if let Some(note) = p["note"].as_str() {
        t += &format!("{note}\n");
    }
    if let Some(y) = p["scroll"].as_array()
        && let (Some(at), Some(height), Some(view)) = (y.first().and_then(Value::as_i64), y.get(1).and_then(Value::as_i64), y.get(2).and_then(Value::as_i64))
        && height > view
    {
        t += &format!("Scrolled to {at} of {height} px.\n");
    }
    t += &format!("\nText{}:\n{}\n", if p["more_text"] == true { " (the start)" } else { "" }, p["text"].as_str().unwrap_or(""));
    let elements: Vec<String> = p["elements"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let e = e.as_array()?;
            Some(format!("[{}] {} {:?}", e.first()?.as_i64()?, e.get(1)?.as_str()?, e.get(2)?.as_str()?))
        })
        .collect();
    if !elements.is_empty() {
        t += &format!("\nElements you can use (by number):\n{}\n", elements.join("\n"));
    }
    let cites = if url.starts_with("http") { vec![url] } else { vec![] };
    (t, cites)
}
