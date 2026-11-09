//! Skills: saved how-tos every Bot can use (one library, skills.json in the Bots folder). A skill has
//! a short name the user calls it by (`/name` in a message), a line on what it's for, and its steps.
//! The user's own skills count at once; a Bot's (new ones, or a change to one) wait for the user's
//! approval, like the routines a Bot proposes.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const MAX_SKILLS: usize = 100;
/// Skills (or changes to one) the Bots proposed that the user hasn't answered yet.
pub const MAX_PENDING: usize = 5;
pub const STEPS_CHARS: usize = 8000;
pub const ABOUT_CHARS: usize = 200;
/// Skills listed in a Bot's system prompt (by name and one line each).
pub const LISTED: usize = 40;

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub id: String,
    /// Lowercase letters, digits and dashes: what follows the / in a message.
    pub name: String,
    pub about: String,
    pub steps: String,
    pub approved: bool,
    /// "user", or the id of the Bot that proposed it.
    pub by: String,
    /// The proposing Bot's name (for the phone's card).
    pub by_name: String,
    /// A proposed change to this (approved) skill.
    pub replaces: Option<String>,
    pub created: f64,
    pub updated: f64,
}

impl Skill {
    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "name": self.name, "about": self.about, "steps": self.steps, "approved": self.approved, "by": self.by, "by_name": self.by_name, "replaces": self.replaces, "created": self.created, "updated": self.updated})
    }

    fn from_json(v: &Value) -> Option<Skill> {
        Some(Skill {
            id: v["id"].as_str()?.into(),
            name: v["name"].as_str()?.into(),
            about: v["about"].as_str().unwrap_or("").into(),
            steps: v["steps"].as_str().unwrap_or("").into(),
            approved: v["approved"].as_bool().unwrap_or(false),
            by: v["by"].as_str().unwrap_or("user").into(),
            by_name: v["by_name"].as_str().unwrap_or("").into(),
            replaces: v["replaces"].as_str().map(String::from),
            created: v["created"].as_f64().unwrap_or(0.0),
            updated: v["updated"].as_f64().unwrap_or(0.0),
        })
    }
}

/// "Check the bill!" → "check-the-bill" (None when nothing usable is left).
pub fn slug(s: &str) -> Option<String> {
    let mut out = String::new();
    for c in s.trim().trim_start_matches('/').chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out: String = out.trim_matches('-').chars().take(40).collect::<String>().trim_end_matches('-').to_string();
    (out.len() >= 2).then_some(out)
}

/// The skill names a message calls (`/name`, at the start or after a space), in order, once each.
pub fn called(text: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for (i, _) in text.match_indices('/') {
        if i > 0 && !text[..i].ends_with(char::is_whitespace) {
            continue;
        }
        let word: String = text[i + 1..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
        let word = word.to_lowercase();
        if word.len() >= 2 && !out.contains(&word) {
            out.push(word);
        }
    }
    out
}

pub struct Skills {
    file: PathBuf,
    lock: Mutex<()>,
}

impl Skills {
    pub fn new(bots_dir: &Path) -> Skills {
        Skills { file: bots_dir.join("skills.json"), lock: Mutex::new(()) }
    }

    pub fn all(&self) -> Vec<Skill> {
        let v: Value = std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!([]));
        let mut list: Vec<Skill> = v.as_array().into_iter().flatten().filter_map(Skill::from_json).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name).then(a.created.total_cmp(&b.created)));
        list
    }

    fn save(&self, list: &[Skill]) -> Result<(), String> {
        let v: Vec<Value> = list.iter().map(Skill::to_json).collect();
        super::write_private(&self.file, serde_json::to_string_pretty(&v).unwrap().as_bytes())
    }

    /// The approved skills (what Bots may use).
    pub fn approved(&self) -> Vec<Skill> {
        self.all().into_iter().filter(|s| s.approved).collect()
    }

    pub fn find(&self, name: &str) -> Option<Skill> {
        let key = slug(name)?;
        self.approved().into_iter().find(|s| s.name == key)
    }

    fn checked(name: &str, about: &str, steps: &str) -> Result<(String, String, String), String> {
        let name = slug(name).ok_or("Give the skill a short name (letters, digits and dashes, like check-bill).")?;
        let about = about.trim().replace('\n', " ");
        let steps = steps.trim().to_string();
        if about.chars().count() > ABOUT_CHARS {
            return Err(format!("Say what it's for in one line (up to {ABOUT_CHARS} characters)."));
        }
        if steps.is_empty() {
            return Err("Write the skill's steps.".into());
        }
        if steps.chars().count() > STEPS_CHARS {
            return Err(format!("The steps are too long (up to {STEPS_CHARS} characters)."));
        }
        Ok((name, about, steps))
    }

    /// The user's new skill (counts at once).
    pub fn add(&self, v: &Value) -> Result<Skill, String> {
        let _g = self.lock.lock().unwrap();
        let mut list = self.all();
        let (name, about, steps) = Self::checked(v["name"].as_str().unwrap_or(""), v["about"].as_str().unwrap_or(""), v["steps"].as_str().unwrap_or(""))?;
        if list.iter().any(|s| s.approved && s.name == name) {
            return Err(format!("There's a skill called /{name} already."));
        }
        if list.len() >= MAX_SKILLS {
            return Err(format!("You have {MAX_SKILLS} skills already; delete one first."));
        }
        let now = crate::time::now();
        let s = Skill { id: super::new_id(), name, about, steps, approved: true, by: "user".into(), by_name: String::new(), replaces: None, created: now, updated: now };
        list.push(s.clone());
        self.save(&list)?;
        Ok(s)
    }

    /// A Bot's proposal: a new skill, or (same name as one there is) a change to it. Its earlier
    /// proposal for the same name is replaced. Waits for the user's approval.
    pub fn propose(&self, bot_id: &str, bot_name: &str, name: &str, about: &str, steps: &str) -> Result<Skill, String> {
        let _g = self.lock.lock().unwrap();
        let mut list = self.all();
        let (name, about, steps) = Self::checked(name, about, steps)?;
        let replaces = list.iter().find(|s| s.approved && s.name == name).map(|s| s.id.clone());
        if let Some(r) = &replaces
            && let Some(old) = list.iter().find(|s| &s.id == r)
            && old.about == about
            && old.steps == steps
        {
            return Err(format!("/{name} already says exactly that."));
        }
        list.retain(|s| s.approved || !(s.by == bot_id && s.name == name));
        if list.iter().filter(|s| !s.approved).count() >= MAX_PENDING {
            return Err(format!("{MAX_PENDING} skills are waiting for the user's approval already; wait for their answer."));
        }
        if list.len() >= MAX_SKILLS {
            return Err(format!("The library has {MAX_SKILLS} skills already."));
        }
        let now = crate::time::now();
        let s = Skill { id: super::new_id(), name, about, steps, approved: false, by: bot_id.into(), by_name: bot_name.into(), replaces, created: now, updated: now };
        list.push(s.clone());
        self.save(&list)?;
        Ok(s)
    }

    /// The user's change ({name?, about?, steps?}), or {approved: true} for a proposal (a proposed
    /// change then goes into the skill it changes).
    pub fn update(&self, id: &str, v: &Value) -> Result<Skill, String> {
        let _g = self.lock.lock().unwrap();
        let mut list = self.all();
        let i = list.iter().position(|s| s.id == id).ok_or("There's no such skill (it may have been deleted).")?;
        let cur = list[i].clone();
        let (name, about, steps) = Self::checked(
            v["name"].as_str().unwrap_or(&cur.name),
            v["about"].as_str().unwrap_or(&cur.about),
            v["steps"].as_str().unwrap_or(&cur.steps),
        )?;
        let now = crate::time::now();
        let approve = !cur.approved && v["approved"] == true;
        // A proposed change, approved: it becomes the skill it changes (that one keeps its id).
        if approve && let Some(target) = cur.replaces.as_ref().and_then(|r| list.iter().position(|s| &s.id == r)) {
            if list.iter().any(|s| s.approved && s.name == name && s.id != list[target].id) {
                return Err(format!("There's another skill called /{name}."));
            }
            let t = &mut list[target];
            (t.name, t.about, t.steps, t.updated) = (name, about, steps, now);
            let out = t.clone();
            list.remove(i);
            self.save(&list)?;
            return Ok(out);
        }
        if (cur.approved || approve) && list.iter().any(|s| s.approved && s.name == name && s.id != id) {
            return Err(format!("There's a skill called /{name} already."));
        }
        let s = &mut list[i];
        (s.name, s.about, s.steps, s.updated) = (name, about, steps, now);
        if approve {
            s.approved = true;
            s.replaces = None;
        }
        let out = s.clone();
        self.save(&list)?;
        Ok(out)
    }

    /// Deletes a skill, or declines a proposal.
    pub fn delete(&self, id: &str) -> bool {
        let _g = self.lock.lock().unwrap();
        let mut list = self.all();
        let before = list.len();
        list.retain(|s| s.id != id && s.replaces.as_deref() != Some(id));
        before != list.len() && self.save(&list).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_calls() {
        assert_eq!(slug("Check the bill!").as_deref(), Some("check-the-bill"));
        assert_eq!(slug("/morning-brief").as_deref(), Some("morning-brief"));
        assert_eq!(slug("!"), None);
        assert_eq!(called("/bill please, then /Morning-brief and /bill again"), ["bill", "morning-brief"]);
        assert!(called("see https://x.com/abc and a/b").is_empty(), "only a / that starts a word calls a skill");
    }

    #[test]
    fn proposals_wait_and_changes_replace() {
        let dir = std::env::temp_dir().join(format!("cubic-skills-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Skills::new(&dir);
        let mine = s.add(&json!({"name": "Bill", "about": "Check the power bill", "steps": "1. Open the site"})).unwrap();
        assert!(s.add(&json!({"name": "bill", "steps": "x"})).is_err(), "names are unique");
        let p = s.propose("b1", "Scout", "bill", "Check the power bill", "1. Open the site\n2. Log in").unwrap();
        assert_eq!((p.approved, p.replaces.as_deref()), (false, Some(mine.id.as_str())));
        assert_eq!(s.find("bill").unwrap().steps, "1. Open the site", "unapproved changes don't count");
        let p2 = s.propose("b1", "Scout", "bill", "Check the power bill", "1. Open the site\n2. Log in\n3. Read it").unwrap();
        assert_eq!(s.all().iter().filter(|x| !x.approved).count(), 1, "a Bot's newer proposal replaces its older one");
        let done = s.update(&p2.id, &json!({"approved": true})).unwrap();
        assert_eq!((done.id.clone(), done.steps.lines().count()), (mine.id.clone(), 3));
        assert_eq!(s.all().len(), 1);
        let n = s.propose("b2", "Writer", "Weekly Report", "", "Write it").unwrap();
        s.update(&n.id, &json!({"approved": true})).unwrap();
        assert!(s.find("/weekly-report").is_some());
        assert!(s.delete(&mine.id));
        assert!(s.find("bill").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
