//! What the voice says: a reply's words without stage directions, emoji, Markdown or links, cut into
//! sentences as the reply streams in. A port of `speakable` and `Segmenter` in `cubicai_voice.py`
//! (Python counts characters, so the thresholds here do too).

use regex::Regex;
use std::sync::LazyLock;

static BOLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*([^*]+)\*\*").unwrap());
static ACTION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*[^*\n]*\*").unwrap()); // *smiles*
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap());
static CITE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*\[\d+(?:\s*[,\x{2013}-]\s*\d+)*\]").unwrap()); // [1], [2, 3]
static URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://\S+").unwrap());
static BULLET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*([-*•]|#+)\s+").unwrap());
static EMOJI: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\x{1F000}-\x{1FAFF}\x{2600}-\x{27BF}\x{FE0F}\x{200D}\x{1F3FB}-\x{1F3FF}]").unwrap());
static MARKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[`#>|~_]").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w").unwrap());

/// The words of a reply as they should be said ("" when nothing is left to say).
pub fn speakable(text: &str) -> String {
    let t = BOLD.replace_all(text, "$1");
    let t = ACTION.replace_all(&t, " ");
    let t = LINK.replace_all(&t, "$1");
    let t = CITE.replace_all(&t, "");
    let t = URL.replace_all(&t, " a link ");
    let t = BULLET.replace_all(&t, "");
    let t = EMOJI.replace_all(&t, "");
    let t = MARKS.replace_all(&t, "").replace('*', "");
    let t = SPACES.replace_all(&t, " ").trim().to_string();
    if WORD.is_match(&t) { t } else { String::new() }
}

static END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"[.!?…]+["')\]]*\s+|\n+"#).unwrap());
static SOFT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[,;:—–]\s+").unwrap());
static CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"```[\s\S]*?```").unwrap());
static CODE_OPEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"```[\s\S]*?(```|$)").unwrap());

/// Characters before byte `i` (Python's index).
fn chars(s: &str, i: usize) -> usize {
    s[..i].chars().count()
}

/// Cuts streamed reply text into sentences for the voice, never inside *an action* or a code block.
#[derive(Default)]
pub struct Segmenter {
    buf: String,
    later: bool,
}

impl Segmenter {
    pub fn new() -> Segmenter {
        Segmenter::default()
    }

    /// Where the text stops being safe to cut (a byte index): an unclosed ``` or *.
    fn open_part(&self) -> usize {
        let mut limit = self.buf.len();
        if self.buf.matches("```").count() % 2 == 1 {
            limit = limit.min(self.buf.rfind("```").unwrap());
        }
        // Single stars (not part of **), seen only within the limit, as Python's look-around does.
        let head = &self.buf.as_bytes()[..limit];
        let stars: Vec<usize> = (0..head.len()).filter(|&i| head[i] == b'*' && (i == 0 || head[i - 1] != b'*') && head.get(i + 1) != Some(&b'*')).collect();
        if stars.len() % 2 == 1 {
            limit = limit.min(*stars.last().unwrap());
        }
        limit
    }

    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut out = vec![];
        loop {
            let safe = self.open_part();
            let head = &self.buf[..safe];
            // The first sentence goes out as soon as it's there (the voice starts sooner); later ones
            // wait for a bit of length so the voice keeps its flow.
            let min_end = if self.later { 24 } else { 2 };
            let mut cut = END.find_iter(head).find(|m| chars(head, m.end()) >= min_end).map(|m| m.end());
            if cut.is_none() && chars(head, safe) > if self.later { 200 } else { 60 } {
                match SOFT.find_iter(head).last().map(|m| m.end()) {
                    Some(soft) if chars(head, soft) > 20 => cut = Some(soft),
                    _ => cut = head.rfind(' ').filter(|&space| chars(head, space) > 40).map(|space| space + 1),
                }
            }
            let Some(cut) = cut else { return out };
            let part: String = self.buf.drain(..cut).collect();
            let part = CODE.replace_all(&part, " ").into_owned();
            if !speakable(&part).is_empty() {
                out.push(part);
                self.later = true;
            }
        }
    }

    pub fn flush(&mut self) -> Vec<String> {
        let part = CODE_OPEN.replace_all(&self.buf, " ").into_owned();
        self.buf.clear();
        if speakable(&part).is_empty() { vec![] } else { vec![part] }
    }
}
