//! The dataset builder's tools: importing (files in the common formats, Hugging Face datasets,
//! documents turned into questions and answers), cleaning (duplicates, length, personal details),
//! AI review and rewriting, a topic map, Arena votes as examples and preference pairs, and the
//! held-back test split.

use super::dataset::{Dataset, Example, Pair, Turn};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// --- records in the common formats -----------------------------------------------------------------

#[derive(Debug, Default, PartialEq)]
pub struct Imported {
    pub examples: Vec<Example>,
    pub pairs: Vec<Pair>,
    /// A system prompt the records carried (the first one seen).
    pub system: Option<String>,
    pub skipped: usize,
}

fn role_of(r: &str) -> Option<&'static str> {
    match r.trim().to_ascii_lowercase().as_str() {
        "user" | "human" | "prompter" | "question" => Some("user"),
        "assistant" | "gpt" | "bot" | "model" | "chatgpt" | "answer" => Some("assistant"),
        "system" => Some("system"),
        _ => None,
    }
}

fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        // OpenAI's content parts: [{"type": "text", "text": "..."}]
        Value::Array(parts) => {
            let t: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str().or(p.as_str())).collect();
            (!t.is_empty()).then(|| t.join("\n"))
        }
        _ => None,
    }
}

/// Turns from a messages list in OpenAI (`role`/`content`) or ShareGPT (`from`/`value`) shape,
/// plus the system message if there was one. Same-role turns in a row are merged.
fn turns_of(list: &[Value]) -> (Vec<Turn>, Option<String>) {
    let mut turns: Vec<Turn> = vec![];
    let mut system = None;
    for m in list {
        let Some(role) = m["role"].as_str().or(m["from"].as_str()).and_then(role_of) else { continue };
        let Some(text) = text_of(&m["content"]).or_else(|| text_of(&m["value"])).or_else(|| text_of(&m["text"])) else { continue };
        let text = text.trim().to_string();
        if role == "system" {
            system.get_or_insert(text);
            continue;
        }
        match turns.last_mut() {
            Some(t) if t.role == role => {
                t.content.push_str("\n\n");
                t.content.push_str(&text);
            }
            _ => turns.push(Turn { role: role.into(), content: text }),
        }
    }
    while turns.first().is_some_and(|t| t.role != "user") {
        turns.remove(0);
    }
    if turns.len() % 2 == 1 {
        turns.pop();
    }
    (turns, system)
}

fn str_field<'a>(v: &'a Value, names: &[&str]) -> Option<&'a str> {
    names.iter().find_map(|n| v[*n].as_str().filter(|s| !s.trim().is_empty()))
}

/// A reply given as text or as a messages list (TRL's conversational preference shape).
fn reply_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.trim().to_string()),
        Value::Array(list) => list.iter().rev().find(|m| m["role"].as_str().or(m["from"].as_str()).and_then(role_of) == Some("assistant")).and_then(|m| text_of(&m["content"]).or_else(|| text_of(&m["value"]))).map(|s| s.trim().to_string()),
        _ => None,
    }
}

fn prompt_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.trim().to_string()),
        Value::Array(list) => list.iter().rev().find(|m| m["role"].as_str().or(m["from"].as_str()).and_then(role_of) == Some("user")).and_then(|m| text_of(&m["content"]).or_else(|| text_of(&m["value"]))).map(|s| s.trim().to_string()),
        _ => None,
    }
}

/// One record in any of the usual shapes: chat messages (OpenAI or ShareGPT), Alpaca
/// (instruction/input/output), prompt/response pairs under common names, or a preference
/// (prompt/chosen/rejected).
pub fn read_record(v: &Value, into: &mut Imported, source: &str) {
    if let (Some(chosen), Some(rejected)) = (v.get("chosen"), v.get("rejected")) {
        let prompt = v.get("prompt").and_then(prompt_of).or_else(|| chosen.as_array().and_then(|l| prompt_of(&Value::Array(l.clone()))));
        if let (Some(prompt), Some(c), Some(r)) = (prompt, reply_of(chosen), reply_of(rejected)) {
            let p = Pair { prompt, chosen: c, rejected: r, source: source.into() };
            if p.valid() {
                into.pairs.push(p);
                return;
            }
        }
        into.skipped += 1;
        return;
    }
    for key in ["messages", "conversations", "conversation", "dialog", "chat"] {
        if let Some(list) = v[key].as_array() {
            let (turns, system) = turns_of(list);
            if let Some(s) = system.filter(|s| !s.is_empty()) {
                into.system.get_or_insert(s);
            }
            let e = Example { turns, source: source.into(), ..Default::default() };
            if e.valid() {
                into.examples.push(e);
            } else {
                into.skipped += 1;
            }
            return;
        }
    }
    if let (Some(ins), Some(out)) = (str_field(v, &["instruction"]), str_field(v, &["output", "response"])) {
        let user = match str_field(v, &["input", "context"]) {
            Some(inp) => format!("{}\n\n{}", ins.trim(), inp.trim()),
            None => ins.trim().to_string(),
        };
        into.examples.push(Example::pair(&user, out, source));
        return;
    }
    let user = str_field(v, &["prompt", "question", "query", "input", "user", "human", "instruction"]);
    let reply = str_field(v, &["response", "answer", "completion", "output", "assistant", "gpt", "reply", "chosen"]);
    match (user, reply) {
        (Some(u), Some(a)) => into.examples.push(Example::pair(u, a, source)),
        _ => into.skipped += 1,
    }
}

/// A file's records: a JSON array, an object holding one (`data`, `rows`, `examples`…), or JSON
/// Lines. Lines that aren't JSON are counted as skipped.
pub fn parse_records(text: &str, source: &str) -> Result<Imported, String> {
    let mut out = Imported::default();
    let trimmed = text.trim_start_matches('\u{feff}').trim();
    if trimmed.is_empty() {
        return Err("The file is empty.".into());
    }
    let whole: Option<Value> = serde_json::from_str(trimmed).ok();
    let records: Vec<Value> = match whole {
        Some(Value::Array(a)) => a,
        Some(Value::Object(o)) => {
            let inner = ["data", "rows", "examples", "train", "items"].iter().find_map(|k| o.get(*k).and_then(|v| v.as_array()).cloned());
            match inner {
                Some(a) => a.into_iter().map(|r| r.get("row").cloned().unwrap_or(r)).collect(),
                None => vec![Value::Object(o)],
            }
        }
        _ => trimmed
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| match serde_json::from_str::<Value>(l) {
                Ok(v) => Some(v),
                Err(_) => {
                    out.skipped += 1;
                    None
                }
            })
            .collect(),
    };
    for r in &records {
        read_record(r, &mut out, source);
    }
    if out.examples.is_empty() && out.pairs.is_empty() {
        return Err(format!(
            "Found no conversations in it ({} records skipped). Supported: chat messages (role/content or from/value), Alpaca (instruction/input/output), prompt/response or question/answer, and prompt/chosen/rejected.",
            out.skipped
        ));
    }
    Ok(out)
}

// --- Hugging Face datasets -------------------------------------------------------------------------------

pub const HF_ROWS: &str = "https://datasets-server.huggingface.co";

/// `owner/name` from what the user typed (a name or a huggingface.co link).
pub fn hf_name(input: &str) -> Option<String> {
    let s = input.trim().trim_end_matches('/');
    let s = s.split("huggingface.co/datasets/").nth(1).unwrap_or(s);
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).take(2).collect();
    (parts.len() == 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)))).then(|| parts.join("/"))
}

fn hf_get(url: &str, token: &str) -> Result<Value, String> {
    let mut r = crate::llm::agent(15).get(url);
    if !token.is_empty() {
        r = r.header("Authorization", format!("Bearer {token}"));
    }
    let resp = r.call().map_err(|e| format!("Couldn't reach Hugging Face: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.into_body().read_to_string().map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    match status {
        200 => Ok(v),
        401 | 403 => Err("This dataset needs a Hugging Face token (Settings) or accepting its terms on the website.".into()),
        404 => Err("Hugging Face doesn't have that dataset (or it has no viewer).".into()),
        _ => Err(format!("Hugging Face said {status}: {}", v["error"].as_str().unwrap_or(&text.chars().take(200).collect::<String>()))),
    }
}

/// The dataset's configuration and split to read: "default"/"train" when there, else the first.
pub fn hf_split(base: &str, name: &str, token: &str) -> Result<(String, String), String> {
    let v = hf_get(&format!("{base}/splits?dataset={name}"), token)?;
    let splits: Vec<(String, String)> = v["splits"].as_array().into_iter().flatten().filter_map(|s| Some((s["config"].as_str()?.to_string(), s["split"].as_str()?.to_string()))).collect();
    let pick = splits
        .iter()
        .find(|(c, s)| c == "default" && s == "train")
        .or_else(|| splits.iter().find(|(_, s)| s == "train"))
        .or(splits.first())
        .cloned();
    pick.ok_or_else(|| "That dataset has no readable splits.".into())
}

/// Up to `limit` rows from the start of a split (100 per request, the server's most), and the split's size.
pub fn hf_rows(base: &str, name: &str, config: &str, split: &str, limit: usize, token: &str, stop: &dyn Fn() -> bool) -> Result<(Vec<Value>, Option<u64>), String> {
    let mut rows = vec![];
    let mut total = None;
    while rows.len() < limit && !stop() {
        let n = (limit - rows.len()).min(100);
        let v = hf_get(&format!("{base}/rows?dataset={name}&config={config}&split={split}&offset={}&length={n}", rows.len()), token)?;
        total = v["num_rows_total"].as_u64().or(total);
        let page: Vec<Value> = v["rows"].as_array().into_iter().flatten().map(|r| r["row"].clone()).collect();
        let got = page.len();
        rows.extend(page);
        if got < n {
            break;
        }
    }
    Ok((rows, total))
}

// --- documents --------------------------------------------------------------------------------------------

/// The text of a document: PDF (its text layer), HTML (tags dropped), or any text file.
pub fn read_document(path: &Path) -> Result<String, String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let text = match ext.as_str() {
        "pdf" => {
            let p = path.to_path_buf();
            // pdf-extract can panic on unusual files; that's a failed read, not a crash.
            std::panic::catch_unwind(move || pdf_extract::extract_text(&p))
                .map_err(|_| "This PDF couldn't be read.".to_string())?
                .map_err(|e| format!("This PDF couldn't be read: {e}"))?
        }
        _ => {
            let bytes = std::fs::read(path).map_err(|e| format!("Couldn't open {}: {e}", path.display()))?;
            let s = String::from_utf8_lossy(&bytes).into_owned();
            if ext == "html" || ext == "htm" { strip_html(&s) } else { s }
        }
    };
    let text = text.replace("\r\n", "\n");
    if text.trim().is_empty() {
        return Err(if ext == "pdf" { "This PDF has no text in it (a scan?).".into() } else { "The file has no text.".into() });
    }
    Ok(text)
}

fn strip_html(s: &str) -> String {
    let re_drop = regex::Regex::new(r"(?is)<(script|style)[^>]*>.*?</(script|style)>").unwrap();
    let re_block = regex::Regex::new(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr)[^>]*>").unwrap();
    let re_tag = regex::Regex::new(r"(?s)<[^>]+>").unwrap();
    let s = re_drop.replace_all(s, "");
    let s = re_block.replace_all(&s, "\n\n");
    let s = re_tag.replace_all(&s, "");
    s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
}

/// Cuts text into pieces of about `max` characters at paragraph breaks (sentences when a
/// paragraph is too long), so each fits one request with room to answer.
pub fn chunks(text: &str, max: usize) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut cur = String::new();
    let push = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        cur.clear();
    };
    for para in text.split("\n\n").map(|p| p.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|p| !p.is_empty()) {
        let pieces: Vec<String> = if para.len() <= max {
            vec![para]
        } else {
            let mut v = vec![];
            let mut s = String::new();
            for sentence in para.split_inclusive(['.', '?', '!']) {
                if s.len() + sentence.len() > max && !s.is_empty() {
                    v.push(std::mem::take(&mut s));
                }
                if sentence.len() > max {
                    // No sentence breaks: cut by characters.
                    let chars: Vec<char> = sentence.chars().collect();
                    for c in chars.chunks(max) {
                        v.push(c.iter().collect());
                    }
                } else {
                    s.push_str(sentence);
                }
            }
            if !s.is_empty() {
                v.push(s);
            }
            v
        };
        for piece in pieces {
            if cur.len() + piece.len() + 2 > max && !cur.is_empty() {
                push(&mut cur, &mut out);
            }
            if !cur.is_empty() {
                cur.push_str("\n\n");
            }
            cur.push_str(&piece);
        }
    }
    push(&mut cur, &mut out);
    out
}

/// Asks a model for questions and answers that a piece of a document supports.
pub fn doc_messages(title: &str, chunk: &str, n: usize, d: &Dataset) -> Vec<Value> {
    let mut ask = format!(
        "Here is part of a document{}. Write {n} question-and-answer training conversations based on it.\n\
         The questions should be ones a person might really ask about this material, in varied wording. The answers must be \
         fully supported by the text: don't add facts it doesn't contain.\n",
        if title.is_empty() { String::new() } else { format!(" (\"{title}\")") }
    );
    if !d.brief.trim().is_empty() {
        ask += &format!("\nWhat the model should learn:\n{}\n", d.brief.trim());
    }
    if !d.system.trim().is_empty() {
        ask += &format!("\nThe model will run with this system prompt, so answer the way it asks:\n{}\n", d.system.trim());
    }
    ask += &format!("\nThe text:\n\"\"\"\n{}\n\"\"\"\n\nAnswer with only a JSON array, no other text: [{{\"user\": \"...\", \"assistant\": \"...\"}}, ...]", chunk.trim());
    vec![
        json!({"role": "system", "content": "You write high-quality training data for fine-tuning language models, grounded in the text you are given. You answer with JSON only."}),
        json!({"role": "user", "content": ask}),
    ]
}

// --- duplicates --------------------------------------------------------------------------------------------

/// Lowercase words only: what's left when spacing, case and punctuation are ignored.
pub fn normalise(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn shingles(text: &str) -> HashSet<u64> {
    let n = normalise(text);
    let w: Vec<&str> = n.split(' ').filter(|x| !x.is_empty()).collect();
    if w.len() < 3 {
        return w.iter().map(|x| fnv(x)).collect();
    }
    w.windows(3).map(|g| fnv(&g.join(" "))).collect()
}

fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count();
    inter as f32 / (a.len() + b.len() - inter) as f32
}

/// Duplicates to drop: (the later example, the earlier one it repeats, similarity). Exact ones
/// (same words once case, spacing and punctuation are ignored) are 1.0; near ones are found with
/// MinHash (64 hashes in 16 bands) and checked on their word triples' Jaccard similarity.
pub fn duplicates(examples: &[Example], threshold: f32) -> Vec<(usize, usize, f32)> {
    let mut out = vec![];
    let mut dropped: HashSet<usize> = HashSet::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (i, e) in examples.iter().enumerate() {
        let key = normalise(&e.text());
        match seen.get(&key) {
            Some(&j) => {
                out.push((i, j, 1.0));
                dropped.insert(i);
            }
            None => {
                seen.insert(key, i);
            }
        }
    }
    const H: usize = 64;
    const BANDS: usize = 16;
    let sets: Vec<HashSet<u64>> = examples.iter().map(|e| shingles(&e.text())).collect();
    let sigs: Vec<[u64; H]> = sets
        .iter()
        .map(|s| {
            let mut sig = [u64::MAX; H];
            for &x in s {
                for (k, slot) in sig.iter_mut().enumerate() {
                    let h = (x ^ (k as u64).wrapping_mul(0x9e3779b97f4a7c15)).wrapping_mul(0xff51afd7ed558ccd | 1);
                    let h = h ^ (h >> 33);
                    *slot = (*slot).min(h);
                }
            }
            sig
        })
        .collect();
    let rows = H / BANDS;
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (i, sig) in sigs.iter().enumerate() {
        if dropped.contains(&i) || sets[i].is_empty() {
            continue;
        }
        for b in 0..BANDS {
            let key = sig[b * rows..(b + 1) * rows].iter().fold(0u64, |h, x| (h ^ x).wrapping_mul(0x100000001b3));
            buckets.entry((b, key)).or_default().push(i);
        }
    }
    let mut best: HashMap<usize, (usize, f32)> = HashMap::new();
    for members in buckets.values().filter(|m| m.len() > 1) {
        for (x, &a) in members.iter().enumerate() {
            for &b in &members[x + 1..] {
                let (early, late) = (a.min(b), a.max(b));
                if dropped.contains(&late) {
                    continue;
                }
                let sim = jaccard(&sets[early], &sets[late]);
                if sim >= threshold && best.get(&late).is_none_or(|(_, s)| sim > *s) {
                    best.insert(late, (early, sim));
                }
            }
        }
    }
    // A chain (c repeats b, b repeats a) keeps only a.
    let mut near: Vec<(usize, usize, f32)> = best.into_iter().map(|(l, (e, s))| (l, e, s)).collect();
    near.sort_by_key(|x| x.0);
    let mut gone: HashSet<usize> = dropped;
    for (late, early, sim) in near {
        if !gone.contains(&late) {
            let keep = if gone.contains(&early) { out.iter().find(|(d, _, _)| *d == early).map(|x| x.1).unwrap_or(early) } else { early };
            out.push((late, keep, sim));
            gone.insert(late);
        }
    }
    out.sort_by_key(|x| x.0);
    out
}

/// Examples outside the length limits: replies shorter than `min_reply` characters, or longer
/// than `max_tokens` in all (counted tokens, else ~4 characters each). 0 turns a limit off.
pub fn out_of_length(examples: &[Example], min_reply: usize, max_tokens: usize) -> Vec<usize> {
    examples
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            let reply = e.turns.iter().filter(|t| t.role == "assistant").map(|t| t.content.trim().len()).min().unwrap_or(0);
            let tokens = e.tokens.map(|t| t as usize).unwrap_or(e.chars() / 4);
            (min_reply > 0 && reply < min_reply) || (max_tokens > 0 && tokens > max_tokens)
        })
        .map(|(i, _)| i)
        .collect()
}

// --- personal details -----------------------------------------------------------------------------------------

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Found {
    pub emails: usize,
    pub phones: usize,
    pub keys: usize,
}

impl Found {
    pub fn total(&self) -> usize {
        self.emails + self.phones + self.keys
    }
}

struct Scrubber {
    keys: regex::Regex,
    bearer: regex::Regex,
    email: regex::Regex,
    phone: regex::Regex,
}

fn scrubber() -> &'static Scrubber {
    static S: std::sync::OnceLock<Scrubber> = std::sync::OnceLock::new();
    S.get_or_init(|| Scrubber {
        keys: regex::Regex::new(
            r"\b(?:sk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|glpat-[A-Za-z0-9_-]{20,})",
        )
        .unwrap(),
        bearer: regex::Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*").unwrap(),
        email: regex::Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b").unwrap(),
        phone: regex::Regex::new(r"(?:\+\d{1,3}[\s.-]?\(?\d{1,5}\)?(?:[\s.-]?\d{2,5}){2,4}\b|\(\d{3}\)\s?\d{3}[\s.-]\d{4}\b|\b\d{3}[.-]\d{3}[.-]\d{4}\b|\b\d{5}\s\d{5}\b)").unwrap(),
    })
}

/// Text with email addresses, phone numbers and API keys or tokens replaced by placeholders.
pub fn scrub(text: &str) -> (String, Found) {
    let s = scrubber();
    let mut f = Found::default();
    let t = s.keys.replace_all(text, |_: &regex::Captures| {
        f.keys += 1;
        "[api key]"
    });
    let t = s.bearer.replace_all(&t, |_: &regex::Captures| {
        f.keys += 1;
        "Bearer [token]"
    });
    let t = s.email.replace_all(&t, |_: &regex::Captures| {
        f.emails += 1;
        "[email]"
    });
    let t = s.phone.replace_all(&t, |_: &regex::Captures| {
        f.phones += 1;
        "[phone]"
    });
    (t.into_owned(), f)
}

/// What scrubbing every example (and pair) would find, without changing anything.
pub fn scan(d: &Dataset) -> Found {
    let mut f = Found::default();
    let texts = d.examples.iter().flat_map(|e| e.turns.iter().map(|t| t.content.as_str())).chain(d.pairs.iter().flat_map(|p| [p.prompt.as_str(), p.chosen.as_str(), p.rejected.as_str()]));
    for t in texts {
        let (_, x) = scrub(t);
        f.emails += x.emails;
        f.phones += x.phones;
        f.keys += x.keys;
    }
    f
}

pub fn scrub_dataset(d: &mut Dataset) -> Found {
    let mut f = Found::default();
    let mut fix = |s: &mut String| {
        let (t, x) = scrub(s);
        if x.total() > 0 {
            *s = t;
            f.emails += x.emails;
            f.phones += x.phones;
            f.keys += x.keys;
        }
    };
    for e in &mut d.examples {
        e.turns.iter_mut().for_each(|t| fix(&mut t.content));
    }
    for p in &mut d.pairs {
        fix(&mut p.prompt);
        fix(&mut p.chosen);
        fix(&mut p.rejected);
    }
    f
}

// --- AI review and rewriting -------------------------------------------------------------------------------------

fn show(e: &Example, max: usize) -> String {
    e.turns
        .iter()
        .map(|t| {
            let c = t.content.trim();
            let c = if c.len() > max { format!("{}…", &c[..c.floor_char_boundary(max)]) } else { c.to_string() };
            format!("{}: {c}", if t.role == "user" { "user" } else { "assistant" })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn context(d: &Dataset) -> String {
    let mut s = String::new();
    if !d.brief.trim().is_empty() {
        s += &format!("What the model should learn:\n{}\n\n", d.brief.trim());
    }
    if !d.system.trim().is_empty() {
        s += &format!("Its system prompt:\n{}\n\n", d.system.trim());
    }
    s
}

/// Asks for a 1–5 grade and a reason for each example (numbered 1.. in this batch).
pub fn grade_messages(d: &Dataset, batch: &[&Example]) -> Vec<Value> {
    let mut ask = format!("{}Grade each training example below for fine-tuning this model, from 1 to 5:\n\
        5 = excellent: correct, on-brief, exactly how the model should answer\n\
        4 = good, small flaws\n3 = usable but bland or partly off\n2 = weak: wrong in places, off-brief or badly written\n\
        1 = harmful to train on: wrong, broken, off-topic or empty\n\n\
        Check every fact and every sum yourself: a reply that is wrong scores 1 or 2 however well it is written, \
        and one that ignores the brief scores 2 at most.\n\n", context(d));
    for (k, e) in batch.iter().enumerate() {
        ask += &format!("### Example {}\n{}\n\n", k + 1, show(e, 2500));
    }
    ask += "Answer with only a JSON array, one entry per example, no other text. Give the reason first, then the score: \
        [{\"id\": 1, \"why\": \"is it correct? does it follow the brief? (short)\", \"score\": 4}, ...]";
    vec![
        json!({"role": "system", "content": "You review training data for fine-tuning language models: strict, fair and brief. You answer with JSON only."}),
        json!({"role": "user", "content": ask}),
    ]
}

fn json_array(reply: &str) -> Result<Vec<Value>, String> {
    let start = reply.find('[').ok_or("The reply had no JSON list in it.")?;
    let end = reply.rfind(']').filter(|&e| e > start).ok_or("The reply's JSON list isn't closed (it may have been cut off).")?;
    let v: Value = serde_json::from_str(&reply[start..=end]).map_err(|e| format!("The reply isn't valid JSON: {e}"))?;
    Ok(v.as_array().cloned().unwrap_or_default())
}

fn id_of(v: &Value) -> Option<usize> {
    v["id"].as_u64().map(|x| x as usize).or_else(|| v["id"].as_str()?.trim().parse().ok())
}

/// Grades from the reply: (index in the batch, score 1–5, reason).
pub fn parse_grades(reply: &str, n: usize) -> Result<Vec<(usize, u8, String)>, String> {
    let out: Vec<(usize, u8, String)> = json_array(reply)?
        .iter()
        .filter_map(|g| {
            let id = id_of(g)?.checked_sub(1).filter(|i| *i < n)?;
            let s = g["score"].as_f64().or_else(|| g["score"].as_str()?.trim().parse().ok())?;
            Some((id, s.round().clamp(1.0, 5.0) as u8, g["why"].as_str().unwrap_or("").trim().to_string()))
        })
        .collect();
    if out.is_empty() {
        return Err("The reply had no grades in it.".into());
    }
    Ok(out)
}

/// The AI rewrites.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// New examples: the user's message reworded, the reply kept.
    Reword,
    /// New examples: the conversation carried on for another exchange or two.
    Extend,
    /// The replies rewritten in a voice.
    Restyle(String),
    /// Weak replies rewritten properly (using the grade's reason).
    Fix,
}

impl Op {
    pub fn label(&self) -> &'static str {
        match self {
            Op::Reword => "Rewording questions",
            Op::Extend => "Extending conversations",
            Op::Restyle(_) => "Restyling replies",
            Op::Fix => "Fixing weak replies",
        }
    }
}

pub fn op_messages(op: &Op, d: &Dataset, batch: &[&Example]) -> Vec<Value> {
    let (task, shape) = match op {
        Op::Reword => (
            "For each example, write a new first user message that asks for the same thing in different words: change the phrasing, length or tone, as different people would. The reply stays as it is, so the new message must still fit it.".to_string(),
            "[{\"id\": 1, \"user\": \"the reworded message\"}, ...]",
        ),
        Op::Extend => (
            "Carry each conversation on with one or two more exchanges: a natural follow-up from the user, and the model's reply in exactly the same style as its earlier replies.".to_string(),
            "[{\"id\": 1, \"turns\": [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}, ...]",
        ),
        Op::Restyle(voice) => (
            format!("Rewrite every assistant reply in this voice: {}\nKeep the facts, the meaning and roughly the length; change only how it's said.", voice.trim()),
            "[{\"id\": 1, \"replies\": [\"each assistant reply, in order\"]}, ...]",
        ),
        Op::Fix => (
            "These examples were graded weak (the reviewer's reason is given). Rewrite each assistant reply so it is correct, on-brief and exactly how the model should answer. Keep the user's messages.".to_string(),
            "[{\"id\": 1, \"replies\": [\"each assistant reply, in order\"]}, ...]",
        ),
    };
    let mut ask = format!("{}{task}\n\n", context(d));
    for (k, e) in batch.iter().enumerate() {
        ask += &format!("### Example {}\n{}\n", k + 1, show(e, 3000));
        if *op == Op::Fix && !e.note.is_empty() {
            ask += &format!("(reviewer: {})\n", e.note);
        }
        ask += "\n";
    }
    ask += &format!("Answer with only a JSON array, one entry per example, no other text: {shape}");
    vec![
        json!({"role": "system", "content": "You improve training data for fine-tuning language models. You answer with JSON only."}),
        json!({"role": "user", "content": ask}),
    ]
}

#[derive(Debug, PartialEq)]
pub enum Change {
    Add(Example),
    Replace(Example),
}

/// What one entry of an op's reply does to its example (None when it doesn't fit).
pub fn apply_op(op: &Op, e: &Example, v: &Value) -> Option<Change> {
    match op {
        Op::Reword => {
            let user = v["user"].as_str()?.trim();
            if user.is_empty() || normalise(user) == normalise(e.first("user")) {
                return None;
            }
            let mut n = Example { turns: e.turns.clone(), source: "reword".into(), ..Default::default() };
            n.turns[0].content = user.to_string();
            n.valid().then_some(Change::Add(n))
        }
        Op::Extend => {
            let (more, _) = turns_of(v["turns"].as_array()?);
            if more.is_empty() {
                return None;
            }
            let mut n = Example { turns: e.turns.clone(), source: "extend".into(), ..Default::default() };
            n.turns.extend(more);
            n.valid().then_some(Change::Add(n))
        }
        Op::Restyle(_) | Op::Fix => {
            let replies: Vec<&str> = v["replies"].as_array()?.iter().filter_map(|r| r.as_str()).collect();
            let slots = e.turns.iter().filter(|t| t.role == "assistant").count();
            if replies.len() != slots || replies.iter().any(|r| r.trim().is_empty()) {
                return None;
            }
            let mut n = e.clone();
            let mut it = replies.into_iter();
            for t in n.turns.iter_mut().filter(|t| t.role == "assistant") {
                t.content = it.next()?.trim().to_string();
            }
            n.score = None;
            n.note.clear();
            n.tokens = None;
            Some(Change::Replace(n))
        }
    }
}

/// An op's reply: (index in the batch, its entry).
pub fn parse_op(reply: &str, n: usize) -> Result<Vec<(usize, Value)>, String> {
    let out: Vec<(usize, Value)> = json_array(reply)?.into_iter().filter_map(|v| Some((id_of(&v)?.checked_sub(1).filter(|i| *i < n)?, v))).collect();
    if out.is_empty() {
        return Err("The reply had nothing usable in it.".into());
    }
    Ok(out)
}

// --- the topic map ---------------------------------------------------------------------------------------------

const STOP: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "your", "with", "this", "that", "from", "have", "has", "had", "was", "were", "what", "when", "where", "which", "who", "why", "how", "can", "could", "would", "should", "will", "about", "into", "than", "then", "them", "they", "their", "there", "these", "those", "its", "it's", "our", "out", "all", "any", "some", "one", "two", "also", "just", "like", "get", "got", "does", "did", "doing", "been", "being", "more", "most", "very", "much", "many", "such", "only", "own", "same", "other", "over", "under", "again", "here", "each", "few", "both", "may", "might", "must", "need", "want", "tell", "give", "make", "use", "using", "please", "know", "think", "way", "something", "thing", "things", "really", "well", "good", "best", "explain", "describe", "write", "help", "able", "him", "her", "his", "she", "she's", "he's", "i'm", "don't", "can't", "let", "me", "my", "a", "an", "is", "in", "of", "to", "on", "it", "be", "or", "as", "at", "by", "if", "do", "so", "we", "i", "no", "up",
];

#[derive(Debug, Default, Clone, PartialEq)]
pub struct TopicMap {
    /// Each example's place on the map, in -1..1.
    pub points: Vec<[f32; 2]>,
    pub cluster: Vec<usize>,
    /// Each cluster's top words and size.
    pub clusters: Vec<(String, usize)>,
}

fn words_of(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_matches('\'').to_lowercase())
        .filter(|w| w.chars().count() >= 3 && !STOP.contains(&w.as_str()) && !w.chars().all(|c| c.is_ascii_digit()))
        .collect()
}

/// Groups what the users ask into topics (TF-IDF, k-means with a fixed seed) and lays them out
/// in 2D (the first two principal components), so gaps and lopsided sets show.
pub fn topic_map(texts: &[String], k: usize) -> TopicMap {
    let n = texts.len();
    if n == 0 {
        return TopicMap::default();
    }
    let docs: Vec<Vec<String>> = texts.iter().map(|t| words_of(t)).collect();
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        for w in d.iter().map(String::as_str).collect::<HashSet<_>>() {
            *df.entry(w).or_default() += 1;
        }
    }
    // Small sets keep every word (most appear once); bigger ones drop one-off words as noise.
    let min_df = if n >= 100 { 2 } else { 1 };
    let mut vocab: Vec<(&str, usize)> = df.into_iter().filter(|(_, c)| *c >= min_df && *c < n.max(2)).collect();
    vocab.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    vocab.truncate(400);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, (w, _))| (*w, i)).collect();
    let dim = vocab.len().max(1);
    let idf: Vec<f32> = vocab.iter().map(|(_, c)| ((1.0 + n as f32) / (1.0 + *c as f32)).ln() + 1.0).collect();
    let mut x: Vec<Vec<f32>> = docs
        .iter()
        .map(|d| {
            let mut v = vec![0f32; dim];
            for w in d {
                if let Some(&i) = index.get(w.as_str()) {
                    v[i] += idf[i];
                }
            }
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|a| *a /= norm);
            }
            v
        })
        .collect();
    let k = k.clamp(1, n);
    // k-means++ with a fixed seed, then Lloyd's iterations on cosine similarity.
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut centres: Vec<Vec<f32>> = vec![x[(rand() * n as f64) as usize % n].clone()];
    while centres.len() < k {
        let d2: Vec<f64> = x.iter().map(|p| centres.iter().map(|c| (1.0 - dot(p, c)).max(0.0) as f64).fold(f64::INFINITY, f64::min)).collect();
        let total: f64 = d2.iter().sum();
        if total <= 1e-9 {
            break;
        }
        let mut r = rand() * total;
        let mut pick = n - 1;
        for (i, d) in d2.iter().enumerate() {
            r -= d;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        centres.push(x[pick].clone());
    }
    let mut assign = vec![0usize; n];
    for _ in 0..30 {
        let mut moved = false;
        for (i, p) in x.iter().enumerate() {
            let best = (0..centres.len()).max_by(|&a, &b| dot(p, &centres[a]).total_cmp(&dot(p, &centres[b])).then(b.cmp(&a))).unwrap_or(0);
            if assign[i] != best {
                assign[i] = best;
                moved = true;
            }
        }
        for (c, centre) in centres.iter_mut().enumerate() {
            let mut sum = vec![0f32; dim];
            let mut count = 0;
            for (i, p) in x.iter().enumerate().filter(|(i, _)| assign[*i] == c) {
                let _ = i;
                sum.iter_mut().zip(p).for_each(|(s, v)| *s += v);
                count += 1;
            }
            if count > 0 {
                let norm = sum.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-9);
                *centre = sum.into_iter().map(|v| v / norm).collect();
            }
        }
        if !moved {
            break;
        }
    }
    // Renumber clusters by size, biggest first.
    let mut sizes: Vec<(usize, usize)> = (0..centres.len()).map(|c| (c, assign.iter().filter(|&&a| a == c).count())).filter(|x| x.1 > 0).collect();
    sizes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let renum: HashMap<usize, usize> = sizes.iter().enumerate().map(|(new, (old, _))| (*old, new)).collect();
    let clusters: Vec<(String, usize)> = sizes
        .iter()
        .map(|(c, size)| {
            let mut top: Vec<(usize, f32)> = centres[*c].iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            let words: Vec<&str> = top.iter().take(3).filter(|t| t.1 > 0.0).map(|t| vocab[t.0].0).collect();
            (if words.is_empty() { "(no common words)".to_string() } else { words.join(", ") }, *size)
        })
        .collect();
    let cluster: Vec<usize> = assign.iter().map(|a| renum[a]).collect();
    // The first two principal components by power iteration (no covariance matrix built).
    let mean: Vec<f32> = (0..dim).map(|j| x.iter().map(|r| r[j]).sum::<f32>() / n as f32).collect();
    for r in &mut x {
        r.iter_mut().zip(&mean).for_each(|(v, m)| *v -= m);
    }
    let mut comps: Vec<Vec<f32>> = vec![];
    for c in 0..2 {
        let mut v: Vec<f32> = (0..dim).map(|j| (((j * 7919 + c * 104729) % 1000) as f32 / 1000.0) - 0.5).collect();
        for _ in 0..60 {
            let xv: Vec<f32> = x.iter().map(|r| dot(r, &v)).collect();
            let mut next = vec![0f32; dim];
            for (r, s) in x.iter().zip(&xv) {
                next.iter_mut().zip(r).for_each(|(a, b)| *a += b * s);
            }
            for prev in &comps {
                let p = dot(&next, prev);
                next.iter_mut().zip(prev).for_each(|(a, b)| *a -= p * b);
            }
            let norm = next.iter().map(|a| a * a).sum::<f32>().sqrt();
            if norm < 1e-9 {
                break;
            }
            v = next.into_iter().map(|a| a / norm).collect();
        }
        comps.push(v);
    }
    let raw: Vec<[f32; 2]> = x.iter().map(|r| [dot(r, &comps[0]), dot(r, &comps[1])]).collect();
    let scale = |i: usize| raw.iter().map(|p| p[i].abs()).fold(0f32, f32::max).max(1e-6);
    let (sx, sy) = (scale(0), scale(1));
    TopicMap { points: raw.iter().map(|p| [p[0] / sx, p[1] / sy]).collect(), cluster, clusters }
}

/// How many topics to look for: about √(n/2), 2 to 8.
pub fn topic_count(n: usize) -> usize {
    if n < 6 { 1 } else { ((n as f64 / 2.0).sqrt().round() as usize).clamp(2, 8) }
}

// --- from the Arena --------------------------------------------------------------------------------------------

/// Your Arena picks: each voted prompt with its winning answer as an example, and the winner
/// against every other working answer as preference pairs.
pub fn from_arena(runs: &[super::arena::Run]) -> (Vec<Example>, Vec<Pair>) {
    let (mut ex, mut pairs) = (vec![], vec![]);
    for r in runs {
        for it in &r.items {
            let Some(v) = it.vote.as_ref().filter(|v| !v.all_bad) else { continue };
            let Some(w) = v.best.filter(|&w| it.answers.get(w).is_some_and(|a| a.ok())) else { continue };
            let win = it.answers[w].text.trim();
            ex.push(Example::pair(&it.prompt, win, "arena"));
            for (c, a) in it.answers.iter().enumerate() {
                if c != w && a.ok() {
                    let p = Pair { prompt: it.prompt.trim().into(), chosen: win.into(), rejected: a.text.trim().into(), source: "arena".into() };
                    if p.valid() {
                        pairs.push(p);
                    }
                }
            }
        }
    }
    (ex, pairs)
}

/// Adds what isn't there yet (by normalised text); returns how many examples and pairs were new.
pub fn merge(d: &mut Dataset, examples: Vec<Example>, pairs: Vec<Pair>) -> (usize, usize) {
    let mut have: HashSet<String> = d.examples.iter().map(|e| normalise(&e.text())).collect();
    let before = (d.examples.len(), d.pairs.len());
    for e in examples {
        if have.insert(normalise(&e.text())) {
            d.examples.push(e);
        }
    }
    let mut have_p: HashSet<String> = d.pairs.iter().map(|p| normalise(&format!("{}\n{}\n{}", p.prompt, p.chosen, p.rejected))).collect();
    for p in pairs {
        if have_p.insert(normalise(&format!("{}\n{}\n{}", p.prompt, p.chosen, p.rejected))) {
            d.pairs.push(p);
        }
    }
    d.ensure_ids();
    (d.examples.len() - before.0, d.pairs.len() - before.1)
}

// --- the test split ---------------------------------------------------------------------------------------------

/// Holds back a share of the single-exchange examples for testing after training (chosen with a
/// seed from the dataset's id, so it's the same each time). Returns how many are held back.
pub fn hold_out(d: &mut Dataset, share: f32) -> usize {
    d.examples.iter_mut().for_each(|e| e.held = false);
    let mut pool: Vec<usize> = (0..d.examples.len()).filter(|&i| d.examples[i].valid() && d.examples[i].turns.len() == 2).collect();
    let want = ((pool.len() as f32 * share).round() as usize).max(usize::from(pool.len() >= 10)).min(pool.len().saturating_sub(1));
    let order = super::arena::shuffled(pool.len(), fnv(&d.id));
    pool = order.into_iter().map(|i| pool[i]).collect();
    for &i in pool.iter().take(want) {
        d.examples[i].held = true;
    }
    want
}
