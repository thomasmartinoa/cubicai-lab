//! Fine-tuning datasets: conversations to teach a model, drafted by another AI, taken from chats
//! or written by hand, saved as JSON and handed to the trainer as JSONL in the chat format.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Example {
    /// Given when the example joins a dataset (`Dataset::ensure_ids`), so background jobs find it again.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub turns: Vec<Turn>,
    /// Where it came from: "ai", "chat", "hand", "doc", "file", "hf", "arena", "reword", "extend".
    pub source: String,
    /// An AI reviewer's grade (1–5) and its reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<u8>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub note: String,
    /// Held back for testing after training, so not trained on.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub held: bool,
    /// Length in the model's own tokens (template included), when counted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u32>,
}

/// A preference: for this prompt, one reply is better than the other (for DPO training).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Pair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub source: String,
}

impl Pair {
    pub fn valid(&self) -> bool {
        !self.prompt.trim().is_empty() && !self.chosen.trim().is_empty() && !self.rejected.trim().is_empty() && self.chosen.trim() != self.rejected.trim()
    }
}

impl Example {
    pub fn pair(user: &str, assistant: &str, source: &str) -> Example {
        Example {
            turns: vec![Turn { role: "user".into(), content: user.trim().into() }, Turn { role: "assistant".into(), content: assistant.trim().into() }],
            source: source.into(),
            ..Default::default()
        }
    }

    /// Usable for training: alternating user/assistant turns that end with a non-empty reply.
    pub fn valid(&self) -> bool {
        !self.turns.is_empty()
            && self.turns.iter().all(|t| !t.content.trim().is_empty())
            && self.turns.iter().enumerate().all(|(i, t)| t.role == if i % 2 == 0 { "user" } else { "assistant" })
            && self.turns.len().is_multiple_of(2)
    }

    pub fn chars(&self) -> usize {
        self.turns.iter().map(|t| t.content.len()).sum()
    }

    pub fn first(&self, role: &str) -> &str {
        first(self, role)
    }

    /// All the text, for searching and comparing.
    pub fn text(&self) -> String {
        self.turns.iter().map(|t| t.content.as_str()).collect::<Vec<_>>().join("\n")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    /// Sent as the system message of every example (and should be used when chatting later).
    pub system: String,
    /// What the AI drafter is asked for.
    pub brief: String,
    pub examples: Vec<Example>,
    pub created: u64,
    /// Preferences for DPO training (from Arena votes or files).
    pub pairs: Vec<Pair>,
}

impl Dataset {
    pub fn new(name: &str) -> Dataset {
        Dataset { id: crate::store::new_id(), name: name.into(), created: crate::store::now_ms(), ..Default::default() }
    }

    pub fn valid_count(&self) -> usize {
        self.examples.iter().filter(|e| e.valid()).count()
    }

    /// What training uses: valid examples that aren't held back.
    pub fn train_count(&self) -> usize {
        self.examples.iter().filter(|e| e.valid() && !e.held).count()
    }

    pub fn held_count(&self) -> usize {
        self.examples.iter().filter(|e| e.valid() && e.held).count()
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.iter().filter(|p| p.valid()).count()
    }

    /// Rough size in tokens of what's trained on (counted ones exactly, the rest at ~4 characters each).
    pub fn approx_tokens(&self) -> usize {
        self.examples.iter().filter(|e| e.valid() && !e.held).map(|e| e.tokens.map(|t| t as usize).unwrap_or((e.chars() + self.system.len()) / 4)).sum()
    }

    /// Gives every example an id (older files and new additions have none).
    pub fn ensure_ids(&mut self) {
        for e in self.examples.iter_mut().filter(|e| e.id.is_empty()) {
            e.id = crate::store::new_id();
        }
    }

    /// The trainer's input: one `{"messages": [...]}` per line; invalid and held-back examples left out.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for e in self.examples.iter().filter(|e| e.valid() && !e.held) {
            let mut m = vec![];
            if !self.system.trim().is_empty() {
                m.push(json!({"role": "system", "content": self.system.trim()}));
            }
            m.extend(e.turns.iter().map(|t| json!({"role": t.role, "content": t.content.trim()})));
            out += &json!({ "messages": m }).to_string();
            out.push('\n');
        }
        out
    }

    /// Preference pairs in TRL's conversational shape: prompt messages, chosen and rejected replies.
    pub fn pairs_jsonl(&self) -> String {
        let mut out = String::new();
        for p in self.pairs.iter().filter(|p| p.valid()) {
            let mut prompt = vec![];
            if !self.system.trim().is_empty() {
                prompt.push(json!({"role": "system", "content": self.system.trim()}));
            }
            prompt.push(json!({"role": "user", "content": p.prompt.trim()}));
            out += &json!({
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": p.chosen.trim()}],
                "rejected": [{"role": "assistant", "content": p.rejected.trim()}],
            })
            .to_string();
            out.push('\n');
        }
        out
    }

    /// The held-back examples as Arena prompts: everything before the last reply, and that reply as the reference.
    pub fn held_prompts(&self) -> Vec<(String, String)> {
        self.examples
            .iter()
            .filter(|e| e.valid() && e.held && e.turns.len() == 2)
            .map(|e| (e.turns[0].content.clone(), e.turns[1].content.clone()))
            .collect()
    }
}

/// The request for another model to draft `n` examples from the brief. A few existing examples
/// are shown so new ones keep the style and don't repeat.
pub fn draft_messages(brief: &str, system: &str, n: usize, existing: &[Example]) -> Vec<Value> {
    let mut ask = format!(
        "I'm fine-tuning a small language model. Write {n} varied training conversations for it.\n\n\
         What the model should learn:\n{}\n",
        brief.trim()
    );
    if !system.trim().is_empty() {
        ask += &format!("\nIt will run with this system prompt:\n{}\n", system.trim());
    }
    let shown: Vec<&Example> = existing.iter().filter(|e| e.valid()).rev().take(4).collect();
    if !shown.is_empty() {
        ask += "\nSome examples already in the set (match their quality, don't repeat them):\n";
        for e in shown {
            ask += &format!("- user: {:?}\n  assistant: {:?}\n", first(e, "user"), first(e, "assistant"));
        }
    }
    ask += "\nVary the user's wording, length and topic. The assistant replies are exactly what the model should say.\n\
            Answer with only a JSON array, no other text: [{\"user\": \"...\", \"assistant\": \"...\"}, ...]";
    vec![
        json!({"role": "system", "content": "You write high-quality training data for fine-tuning language models. You answer with JSON only."}),
        json!({"role": "user", "content": ask}),
    ]
}

fn first<'a>(e: &'a Example, role: &str) -> &'a str {
    e.turns.iter().find(|t| t.role == role).map(|t| t.content.as_str()).unwrap_or("")
}

/// Pulls examples out of a drafted reply: a JSON array (possibly inside a code fence or after some
/// chatter) of `{user, assistant}` or `{messages: [...]}` objects.
pub fn parse_draft(reply: &str) -> Result<Vec<Example>, String> {
    let start = reply.find('[').ok_or("The reply had no JSON list in it.")?;
    let end = reply.rfind(']').filter(|&e| e > start).ok_or("The reply's JSON list isn't closed (it may have been cut off).")?;
    let v: Value = serde_json::from_str(&reply[start..=end]).map_err(|e| format!("The reply isn't valid JSON: {e}"))?;
    let mut out = vec![];
    for item in v.as_array().into_iter().flatten() {
        if let (Some(u), Some(a)) = (item["user"].as_str(), item["assistant"].as_str()) {
            out.push(Example::pair(u, a, "ai"));
        } else if let Some(m) = item["messages"].as_array() {
            let turns: Vec<Turn> = m
                .iter()
                .filter(|t| t["role"] != "system")
                .filter_map(|t| Some(Turn { role: t["role"].as_str()?.into(), content: t["content"].as_str()?.trim().into() }))
                .collect();
            out.push(Example { turns, source: "ai".into(), ..Default::default() });
        }
    }
    out.retain(Example::valid);
    if out.is_empty() {
        return Err("The reply had no usable examples.".into());
    }
    Ok(out)
}

/// A chat as one multi-turn example (failed and empty replies dropped, so turns may merge).
pub fn from_chat(chat: &crate::store::Chat) -> Option<Example> {
    let mut turns: Vec<Turn> = vec![];
    for m in &chat.messages {
        if m.content.trim().is_empty() || m.error.is_some() || !(m.role == "user" || m.role == "assistant") {
            continue;
        }
        match turns.last_mut() {
            Some(t) if t.role == m.role => {
                t.content.push_str("\n\n");
                t.content.push_str(m.content.trim());
            }
            _ => turns.push(Turn { role: m.role.clone(), content: m.content.trim().into() }),
        }
    }
    while turns.first().is_some_and(|t| t.role != "user") {
        turns.remove(0);
    }
    if turns.len() % 2 == 1 {
        turns.pop();
    }
    let e = Example { turns, source: "chat".into(), ..Default::default() };
    e.valid().then_some(e)
}

pub struct Store {
    pub dir: PathBuf,
}

/// A saved copy of a dataset from before a change.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Version {
    pub at: u64,
    pub label: String,
    pub dataset: Dataset,
}

pub const MAX_VERSIONS: usize = 20;

impl Store {
    pub fn new(dir: PathBuf) -> Store {
        let _ = std::fs::create_dir_all(&dir);
        Store { dir }
    }

    pub fn save(&self, d: &Dataset) -> Result<(), String> {
        let j = serde_json::to_vec_pretty(d).map_err(|e| e.to_string())?;
        crate::store::write_atomic(&self.dir.join(format!("{}.json", d.id)), &j).map_err(|e| e.to_string())
    }

    pub fn list(&self) -> Vec<Dataset> {
        let mut out: Vec<Dataset> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok())
            .collect();
        out.sort_by_key(|d: &Dataset| std::cmp::Reverse(d.created));
        out.iter_mut().for_each(Dataset::ensure_ids);
        out
    }

    pub fn delete(&self, id: &str) {
        let _ = std::fs::remove_file(self.dir.join(format!("{id}.json")));
        let _ = std::fs::remove_dir_all(self.versions_dir(id));
    }

    fn versions_dir(&self, id: &str) -> PathBuf {
        self.dir.join("versions").join(id)
    }

    /// Keeps a copy of the dataset as it is now (the newest 20 are kept).
    pub fn snapshot(&self, d: &Dataset, label: &str) -> Result<(), String> {
        let dir = self.versions_dir(&d.id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mut at = crate::store::now_ms();
        while dir.join(format!("{at}.json")).exists() {
            at += 1;
        }
        let v = Version { at, label: label.into(), dataset: d.clone() };
        crate::store::write_atomic(&dir.join(format!("{at}.json")), &serde_json::to_vec(&v).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let all = self.versions(&d.id);
        for old in all.iter().skip(MAX_VERSIONS) {
            let _ = std::fs::remove_file(dir.join(format!("{}.json", old.at)));
        }
        Ok(())
    }

    /// Saved copies, newest first.
    pub fn versions(&self, id: &str) -> Vec<Version> {
        let mut v: Vec<Version> = std::fs::read_dir(self.versions_dir(id))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok())
            .collect();
        v.sort_by_key(|x: &Version| std::cmp::Reverse(x.at));
        v
    }
}
