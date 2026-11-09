//! Group chats: two to six Bots and the user in one chat (groups/<id>/ in the Bots folder: group.json
//! and chat.jsonl). The Bots the user @mentions answer; with no mention, the one whose job fits best.
//! A Bot can bring another in by @mentioning it in its reply, a few hops at most.

use super::Bot;
use serde_json::{json, Value};

pub const MIN_MEMBERS: usize = 2;
pub const MAX_MEMBERS: usize = 6;
pub const MAX_GROUPS: usize = 30;
/// Bots one message of the user's can set talking (the ones it names, then the ones they name…).
pub const MAX_TURNS: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub id: String,
    pub name: String,
    /// Bot ids, in the order they were added.
    pub members: Vec<String>,
    pub created: f64,
}

impl Group {
    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "name": self.name, "members": self.members, "created": self.created})
    }

    pub fn from_json(v: &Value) -> Option<Group> {
        Some(Group {
            id: v["id"].as_str()?.into(),
            name: v["name"].as_str()?.into(),
            members: v["members"].as_array()?.iter().filter_map(|m| m.as_str().map(String::from)).collect(),
            created: v["created"].as_f64().unwrap_or(0.0),
        })
    }

    /// Takes {name?, members?} from the phone; `exists` says whether a Bot id is real.
    pub fn apply(&mut self, v: &Value, exists: impl Fn(&str) -> bool) -> Result<(), String> {
        if let Some(n) = v["name"].as_str() {
            self.name = n.trim().to_string();
        }
        if let Some(list) = v["members"].as_array() {
            let mut ids: Vec<String> = vec![];
            for m in list.iter().filter_map(|m| m.as_str()) {
                if !exists(m) {
                    return Err("One of those Bots isn't on this PC any more.".into());
                }
                if !ids.iter().any(|i| i == m) {
                    ids.push(m.to_string());
                }
            }
            self.members = ids;
        }
        let n = self.name.chars().count();
        if n == 0 || n > 40 {
            return Err("Give the group a name (up to 40 characters).".into());
        }
        if !(MIN_MEMBERS..=MAX_MEMBERS).contains(&self.members.len()) {
            return Err(format!("A group has {MIN_MEMBERS} to {MAX_MEMBERS} Bots."));
        }
        Ok(())
    }
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// The Bots a text @mentions, in order, once each. "@Deal Scout", "@dealscout" and "@Deal-Scout"
/// all name "Deal Scout"; the longest name that fits wins ("@Ann Lee" over "@Ann").
pub fn mentions(text: &str, bots: &[Bot]) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for (i, _) in text.match_indices('@') {
        if i > 0 && text[..i].chars().last().is_some_and(|c| c.is_alphanumeric()) {
            continue; // an email address
        }
        let after = &text[i + 1..];
        let mut best: Option<(&Bot, usize)> = None;
        for b in bots {
            let want = squash(&b.name);
            if want.is_empty() {
                continue;
            }
            // Walk the text, skipping spaces and dashes, until the name's letters are used up.
            let (mut got, mut used) = (String::new(), 0);
            for (j, c) in after.char_indices() {
                if got.len() >= want.len() {
                    break;
                }
                if c.is_alphanumeric() {
                    got.extend(c.to_lowercase());
                } else if !(c == ' ' || c == '-' || c == '_' || c == '.') || got.is_empty() {
                    break;
                }
                used = j + c.len_utf8();
            }
            let ends = after[used..].chars().next().is_none_or(|c| !c.is_alphanumeric());
            if got == want && ends && best.is_none_or(|(_, n)| want.len() > n) {
                best = Some((b, want.len()));
            }
        }
        if let Some((b, _)) = best
            && !out.contains(&b.id)
        {
            out.push(b.id.clone());
        }
    }
    out
}

/// The router's prompt: which member should answer the user's last message.
pub fn router_messages(group: &Group, members: &[Bot], recent: &[(String, String)]) -> Vec<Value> {
    let list: Vec<String> = members.iter().map(|b| format!("- {}: {}", b.name, first_line(&b.job, 200))).collect();
    let chat: Vec<String> = recent.iter().map(|(who, t)| format!("{who}: {}", t.chars().take(600).collect::<String>())).collect();
    vec![
        json!({"role": "system", "content": format!(
            "You pick who answers in the group chat “{}”. The Bots in it:\n{}\n\nReply with only the name of the one Bot best suited to answer the user's last message (the one whose job it is, or the one the user is talking to). No other words.",
            group.name, list.join("\n"))}),
        json!({"role": "user", "content": format!("The chat so far:\n{}\n\nWho answers?", chat.join("\n"))}),
    ]
}

/// The router's answer → a member (the first one named).
pub fn pick(answer: &str, members: &[Bot]) -> Option<String> {
    let a = squash(answer);
    members.iter().filter(|b| !squash(&b.name).is_empty()).filter_map(|b| a.find(&squash(&b.name)).map(|at| (at, b))).min_by_key(|(at, b)| (*at, usize::MAX - b.name.len())).map(|(_, b)| b.id.clone())
}

pub fn first_line(s: &str, max: usize) -> String {
    let l = s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("(no job given)");
    let mut t: String = l.chars().take(max).collect();
    if l.chars().count() > max {
        t.push('…');
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bot(id: &str, name: &str) -> Bot {
        Bot { id: id.into(), name: name.into(), job: String::new(), color: "#000000".into(), provider: "local".into(), model: "m".into(), computer: false, google: false, mcp: vec![], phone: false, created: 0.0 }
    }

    #[test]
    fn mentions_find_bots_by_name() {
        let bots = [bot("a", "Ann"), bot("b", "Ann Lee"), bot("c", "Deal Scout"), bot("d", "Writer")];
        assert_eq!(mentions("@ann lee and @Writer, then @dealscout", &bots), ["b", "d", "c"]);
        assert_eq!(mentions("@Ann: and @Deal-Scout! @writer @Writer", &bots), ["a", "c", "d"]);
        assert!(mentions("mail ann@ann.com or @Annabel or @nobody", &bots).is_empty());
    }

    #[test]
    fn the_router_answer_names_a_member() {
        let bots = [bot("a", "Researcher"), bot("b", "Writer")];
        assert_eq!(pick("Writer", &bots).as_deref(), Some("b"));
        assert_eq!(pick("**researcher** (then the writer)", &bots).as_deref(), Some("a"));
        assert_eq!(pick("nobody", &bots), None);
    }

    #[test]
    fn groups_are_checked() {
        let mut g = Group { id: "1".into(), name: String::new(), members: vec![], created: 0.0 };
        assert!(g.apply(&json!({"name": "Team", "members": ["a"]}), |_| true).is_err());
        assert!(g.apply(&json!({"name": "Team", "members": ["a", "x"]}), |id| id == "a").is_err());
        g.apply(&json!({"name": " Team ", "members": ["a", "b", "a"]}), |_| true).unwrap();
        assert_eq!((g.name.as_str(), g.members.len()), ("Team", 2));
    }
}
