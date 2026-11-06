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
    pub turns: Vec<Turn>,
    /// Where it came from: "ai", "chat" or "hand".
    pub source: String,
}

impl Example {
    pub fn pair(user: &str, assistant: &str, source: &str) -> Example {
        Example {
            turns: vec![Turn { role: "user".into(), content: user.trim().into() }, Turn { role: "assistant".into(), content: assistant.trim().into() }],
            source: source.into(),
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
}

impl Dataset {
    pub fn new(name: &str) -> Dataset {
        Dataset { id: crate::store::new_id(), name: name.into(), created: crate::store::now_ms(), ..Default::default() }
    }

    pub fn valid_count(&self) -> usize {
        self.examples.iter().filter(|e| e.valid()).count()
    }

    /// Rough size in tokens (about 4 characters each).
    pub fn approx_tokens(&self) -> usize {
        self.examples.iter().filter(|e| e.valid()).map(|e| e.chars() + self.system.len()).sum::<usize>() / 4
    }

    /// The trainer's input: one `{"messages": [...]}` per line, invalid examples left out.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for e in self.examples.iter().filter(|e| e.valid()) {
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
            out.push(Example { turns, source: "ai".into() });
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
    let e = Example { turns, source: "chat".into() };
    e.valid().then_some(e)
}

pub struct Store {
    pub dir: PathBuf,
}

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
        out
    }

    pub fn delete(&self, id: &str) {
        let _ = std::fs::remove_file(self.dir.join(format!("{id}.json")));
    }
}
