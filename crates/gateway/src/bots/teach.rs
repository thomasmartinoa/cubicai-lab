//! Teaching a Bot a task: the user drives the computer's browser from the phone (a live screenshot
//! they tap on, type into, scroll), every step is written down in words, and at the end the steps go
//! to the Bot, which writes them up as a skill for the user to approve. One session at a time (there's
//! one browser); while it lasts the Bots' own browser tools wait. What's typed into a password field
//! is never written down.
//!
//! Taking over is the same session started with `takeover`: the user steps in while the Bot works
//! (a login, a captcha, a page it's stuck on). That Bot's browser tools wait until the user hands the
//! browser back; then their steps reach the Bot as a message (`took_over`) and it carries on.

use super::{now_ms, new_id, Bots, Thread};
use serde_json::{json, Value};
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A session nobody touched for this long is over.
pub const IDLE: Duration = Duration::from_secs(10 * 60);
pub const MAX_STEPS: usize = 80;

pub struct Teach {
    pub bot: String,
    /// Taking over from the Bot mid-task (not teaching it a skill).
    pub takeover: bool,
    pub steps: Vec<String>,
    last: Instant,
    url: String,
    title: String,
    /// Bumped with each new screenshot, so the phone knows to fetch it.
    n: u64,
}

/// The message for the Bot when the user is done, as the model reads it.
pub fn prompt(what: &str, steps: &[Value]) -> String {
    let list: Vec<String> = steps.iter().filter_map(|s| s.as_str()).enumerate().map(|(i, s)| format!("{}. {s}", i + 1)).collect();
    let what = if what.trim().is_empty() { "(they didn't say; work it out from the steps)" } else { what.trim() };
    format!(
        "[The user just showed you how to do a task in your computer's browser. What it's for: {what}\n\nThe steps they took:\n{}\n\n\
         Write this up as a skill with save_skill: a short name, one line on what it's for, and numbered steps that would work again next time with your browser tools \
         (say what changes each time, like a search term or a date, what to check at the end, and what to do if a step fails; describe elements by their labels, not by position). \
         Don't redo the task now unless they asked. Then tell them in a line or two what you saved.]",
        list.join("\n")
    )
}

/// The user's steps after taking over, as the model reads them.
pub fn takeover_prompt(what: &str, steps: &[Value]) -> String {
    let list: Vec<String> = steps.iter().filter_map(|s| s.as_str()).enumerate().map(|(i, s)| format!("{}. {s}", i + 1)).collect();
    let did = if list.is_empty() { "They didn't change anything.".to_string() } else { format!("What they did:\n{}", list.join("\n")) };
    let said = if what.trim().is_empty() { String::new() } else { format!("\nThey add: {}", what.trim()) };
    format!(
        "[The user took over your computer's browser while you worked and has handed it back. {did}{said}\n\n\
         The browser is on the page they left it on. Carry on with your task from there: look at the page first (browser_scroll), since element numbers have changed.]"
    )
}

/// What a step of the user's did, in words ("Clicked the button “Search”").
fn describe(cmd: &Value, page: &Value) -> String {
    let did = &page["did"];
    let thing = || {
        let kind = did["kind"].as_str().unwrap_or("");
        let label = did["label"].as_str().unwrap_or("").trim();
        match (kind.is_empty(), label.is_empty()) {
            (true, _) => String::new(),
            (false, true) => format!("the {kind}"),
            (false, false) => format!("the {kind} “{label}”"),
        }
    };
    match cmd["do"].as_str().unwrap_or("") {
        "open" => format!("Opened {}", cmd["url"].as_str().unwrap_or("")),
        "tap" => {
            let t = thing();
            if t.is_empty() {
                format!("Clicked a spot on the page ({:.0}% across, {:.0}% down)", cmd["x"].as_f64().unwrap_or(0.0) * 100.0, cmd["y"].as_f64().unwrap_or(0.0) * 100.0)
            } else {
                format!("Clicked {t}")
            }
        }
        "text" => {
            let into = thing();
            let into = if into.is_empty() { String::new() } else { format!(" into {into}") };
            if did["secret"] == true {
                format!("Typed their password{into} (not written down)")
            } else {
                format!("Typed “{}”{into}", cmd["text"].as_str().unwrap_or(""))
            }
        }
        "key" => {
            let t = thing();
            format!("Pressed {}{}", cmd["key"].as_str().unwrap_or(""), if t.is_empty() { String::new() } else { format!(" in {t}") })
        }
        "scroll" => format!("Scrolled {}", if cmd["direction"] == "up" { "up" } else { "down" }),
        "back" => "Went back".into(),
        _ => String::new(),
    }
}

impl Bots {
    /// The Bot being taught right now, if a session is alive.
    pub fn teaching(&self) -> Option<String> {
        let mut t = self.teach.lock().unwrap();
        if t.as_ref().is_some_and(|s| s.last.elapsed() > IDLE) {
            *t = None;
        }
        t.as_ref().map(|s| s.bot.clone())
    }

    fn teach_state(&self, s: &Teach, error: Option<String>) -> Value {
        json!({"teaching": true, "bot": s.bot, "takeover": s.takeover, "url": s.url, "title": s.title, "steps": s.steps, "shot": s.n, "error": error})
    }

    /// The phone's teaching step: {do: start|look|open|tap|text|key|scroll|back, …} → where the
    /// browser is now and the steps so far (`shot` changes when there's a new screenshot).
    pub fn teach(&self, id: &str, v: &Value) -> Result<Value, String> {
        let bot = self.get(id).ok_or("No such Bot.")?;
        let computer = self.computer.as_ref().filter(|c| c.available()).ok_or("The Bots' computer isn't set up on this PC, so there's no browser to show.")?;
        let what = v["do"].as_str().unwrap_or("");
        let cmd = match what {
            "start" | "look" => json!({"do": "shot"}),
            "open" => {
                let url = v["url"].as_str().unwrap_or("").trim();
                if url.is_empty() {
                    return Err("Type an address.".into());
                }
                json!({"do": "open_light", "url": url})
            }
            "tap" => json!({"do": "tap", "x": v["x"].as_f64().unwrap_or(0.5), "y": v["y"].as_f64().unwrap_or(0.5)}),
            "text" => json!({"do": "text", "text": v["text"].as_str().unwrap_or("")}),
            "key" => json!({"do": "key", "key": v["key"].as_str().unwrap_or("Enter")}),
            "scroll" => json!({"do": "scroll_light", "direction": if v["direction"] == "up" { "up" } else { "down" }}),
            "back" => json!({"do": "back_light"}),
            _ => return Err(format!("Unknown step {what:?}.")),
        };
        {
            let mut t = self.teach.lock().unwrap();
            let live = t.as_ref().is_some_and(|s| s.last.elapsed() <= IDLE);
            let takeover = v["takeover"] == true;
            match t.as_mut() {
                Some(s) if live && s.bot == id => {
                    s.last = Instant::now();
                    if what == "start" {
                        s.takeover = takeover;
                    }
                }
                Some(s) if live && what == "start" && s.takeover => {
                    return Err("You're holding another Bot's browser: hand it back first.".into());
                }
                // Starting (or teaching another Bot) begins afresh; anything else needs a session.
                _ if what == "start" => *t = Some(Teach { bot: id.into(), takeover, steps: vec![], last: Instant::now(), url: String::new(), title: String::new(), n: 0 }),
                _ => return Err("The teaching session is over (it ends after 10 minutes untouched). Start again.".into()),
            }
        }
        if what == "text" && cmd["text"].as_str().is_some_and(|t| t.is_empty()) {
            return Err("Type something first.".into());
        }
        let page = computer.browse(&cmd)?;
        let mut t = self.teach.lock().unwrap();
        let Some(s) = t.as_mut().filter(|s| s.bot == id) else { return Err("The teaching session was ended.".into()) };
        if page["ok"] != true {
            let e = page["error"].as_str().unwrap_or("The browser couldn't do that.").to_string();
            return Ok(self.teach_state(s, Some(e)));
        }
        if let Some(from) = page["shot"].as_str().and_then(|p| computer.shot(p))
            && let Some(d) = self.dir(&bot.id)
        {
            let to = d.join("teach.jpg");
            if std::fs::copy(&from, &to).is_ok() {
                let _ = std::fs::set_permissions(&to, std::fs::Permissions::from_mode(0o600));
                s.n += 1;
            }
            let _ = std::fs::remove_file(from);
        }
        let mut step = describe(&json!({"do": what, "url": cmd["url"], "x": cmd["x"], "y": cmd["y"], "text": cmd["text"], "key": cmd["key"], "direction": cmd["direction"]}), &page);
        let url = page["url"].as_str().unwrap_or("").to_string();
        let title = page["title"].as_str().unwrap_or("").to_string();
        if !step.is_empty() {
            if what != "open" && url != s.url && !url.is_empty() {
                step += &format!(" → now on “{}” ({url})", if title.is_empty() { "a page" } else { &title });
            }
            if s.steps.len() < MAX_STEPS {
                s.steps.push(step);
            }
        }
        (s.url, s.title) = (url, title);
        Ok(self.teach_state(s, None))
    }

    /// Whether the user has taken this Bot's browser over (its browser tools wait meanwhile).
    pub fn taken_over(&self, id: &str) -> bool {
        let t = self.teach.lock().unwrap();
        t.as_ref().is_some_and(|s| s.takeover && s.bot == id && s.last.elapsed() <= IDLE)
    }

    /// The browser right now, for watching a Bot work (JPEG). Never starts the computer.
    pub fn screen(&self) -> Result<Vec<u8>, String> {
        let computer = self.computer.as_ref().filter(|c| c.available()).ok_or("The Bots' computer isn't set up on this PC.")?;
        if !computer.running() {
            return Err("The computer is off: the Bot isn't using it right now.".into());
        }
        let page = computer.browse(&json!({"do": "shot"}))?;
        let from = page["shot"].as_str().and_then(|p| computer.shot(p)).ok_or("The browser gave no picture.")?;
        let bytes = std::fs::read(&from).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(from);
        bytes
    }

    /// The browser as the user last saw it while teaching this Bot (JPEG).
    pub fn teach_shot(&self, id: &str) -> Option<Vec<u8>> {
        std::fs::read(self.dir(id)?.join("teach.jpg")).ok()
    }

    /// Ends the session without sending anything (the steps are dropped).
    pub fn teach_cancel(&self, id: &str) -> bool {
        let mut t = self.teach.lock().unwrap();
        let was = t.as_ref().is_some_and(|s| s.bot == id);
        if was {
            *t = None;
        }
        if let Some(d) = self.dir(id) {
            let _ = std::fs::remove_file(d.join("teach.jpg"));
        }
        was
    }

    /// The user is done: the steps go to the Bot as a message ({what}: what the task is for), which
    /// starts it writing the skill.
    pub fn teach_done(self: &Arc<Bots>, id: &str, what: &str) -> Result<Value, String> {
        let bot = self.get(id).ok_or("No such Bot.")?;
        if self.taken_over(id) {
            return self.hand_back(&bot, what);
        }
        let steps = {
            let mut t = self.teach.lock().unwrap();
            match t.take() {
                Some(s) if s.bot == id && !s.steps.is_empty() => s.steps,
                Some(s) if s.bot == id => {
                    *t = Some(s);
                    return Err("Nothing's been done in the browser yet: show the Bot the steps first.".into());
                }
                other => {
                    *t = other;
                    return Err("There's no teaching session for this Bot (it ends after 10 minutes untouched).".into());
                }
            }
        };
        if let Some(d) = self.dir(id) {
            let _ = std::fs::remove_file(d.join("teach.jpg"));
        }
        let what: String = what.trim().chars().take(2000).collect();
        let text = if what.is_empty() { "I showed you how to do a task in the browser.".to_string() } else { what };
        let msg = json!({"id": new_id(), "role": "user", "text": text, "at": now_ms(), "taught": steps});
        crate::log(&format!("bot {} was taught a task ({} steps)", bot.id, msg["taught"].as_array().map(Vec::len).unwrap_or(0)));
        Ok(self.launch(Thread::Bot(bot.id.clone()), msg, |_| {}).0)
    }
}

impl Bots {
    /// The user hands the browser back after taking over: their steps (and words) go to the Bot as a
    /// message before the session ends, so the browser tool that waited finds them in the chat.
    fn hand_back(self: &Arc<Bots>, bot: &super::Bot, what: &str) -> Result<Value, String> {
        let steps = self.teach.lock().unwrap().as_ref().map(|s| s.steps.clone()).unwrap_or_default();
        let what: String = what.trim().chars().take(2000).collect();
        let mut out = json!({"ended": true});
        if !steps.is_empty() || !what.is_empty() {
            let msg = json!({"id": new_id(), "role": "user", "text": what, "at": now_ms(), "taught": steps, "took_over": true});
            out = self.launch(Thread::Bot(bot.id.clone()), msg, |_| {}).0;
        }
        *self.teach.lock().unwrap() = None;
        if let Some(d) = self.dir(&bot.id) {
            let _ = std::fs::remove_file(d.join("teach.jpg"));
        }
        crate::log(&format!("bot {} got its browser back ({} steps)", bot.id, steps.len()));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_read_like_a_person_did_them() {
        let page = |did: Value| json!({"ok": true, "did": did});
        assert_eq!(describe(&json!({"do": "tap", "x": 0.5, "y": 0.2}), &page(json!({"kind": "button", "label": "Search"}))), "Clicked the button “Search”");
        assert_eq!(describe(&json!({"do": "tap", "x": 0.25, "y": 0.5}), &page(Value::Null)), "Clicked a spot on the page (25% across, 50% down)");
        assert_eq!(describe(&json!({"do": "text", "text": "hunter2"}), &page(json!({"kind": "password field", "label": "Password", "secret": true}))), "Typed their password into the password field “Password” (not written down)");
        assert_eq!(describe(&json!({"do": "text", "text": "cats"}), &page(json!({"kind": "search box", "label": ""}))), "Typed “cats” into the search box");
        assert_eq!(describe(&json!({"do": "key", "key": "Enter"}), &page(Value::Null)), "Pressed Enter");
        let p = prompt("Find a cat picture", &[json!("Opened https://example.com"), json!("Clicked the link “Cats”")]);
        assert!(p.contains("1. Opened https://example.com\n2. Clicked the link “Cats”") && p.contains("save_skill"), "{p}");
    }
}
