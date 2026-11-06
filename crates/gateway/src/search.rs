//! Web search on the PC: the model rewrites the question as a query, a local SearXNG finds pages,
//! the top pages are read (public addresses only) and handed to the model with numbered sources.

use crate::access::is_global;
use crate::ollama::{ModelInfo, Ollama};
use serde_json::{json, Value};
use std::sync::mpsc;
use std::time::{Duration, Instant};

pub const SEARCH_RESULTS: usize = 8;
pub const PAGES_TO_READ: usize = 4;
pub const PAGE_CHARS: usize = 4000;
/// Seconds for reading all result pages together; slower pages fall back to their snippet.
pub const PAGE_DEADLINE: Duration = Duration::from_secs(6);
pub const UA: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36";

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn text_of(m: &Value) -> String {
    m["content"].as_str().unwrap_or("").to_string()
}

/// The request asking the model for a search query (its own words, no thinking).
pub fn query_request(model: &str, info: &ModelInfo, messages: &[Value], today: &str) -> Value {
    let last = messages.iter().rfind(|m| m["role"] == "user").map(text_of).unwrap_or_default();
    let talk: Vec<&Value> = messages.iter().filter(|m| m["role"] == "user" || m["role"] == "assistant").collect();
    let recent = &talk[talk.len().saturating_sub(5)..talk.len().saturating_sub(1)];
    let context: Vec<String> = recent.iter().map(|m| format!("{}: {}", m["role"].as_str().unwrap_or(""), text_of(m).chars().take(500).collect::<String>())).collect();
    let context = if context.is_empty() { "(none)".to_string() } else { context.join("\n") };
    let mut payload = json!({
        "model": model,
        "stream": false,
        "messages": [
            {"role": "system", "content": format!("Today is {today}. Rewrite the user's latest message as one concise web search query. Output only the query text, no quotes, no explanation.")},
            {"role": "user", "content": format!("Conversation so far:\n{context}\n\nLatest message: {last}")},
        ],
        "options": {"temperature": 0, "num_predict": 256},
    });
    if info.has("thinking") {
        payload["think"] = false.into();
    }
    payload
}

/// The model's answer → one clean query line (thinking tags, quotes dropped); the user's words if
/// the model gave nothing. At most 200 characters.
pub fn clean_query(reply: &str, fallback: &str) -> String {
    let mut s = reply.to_string();
    while let Some(a) = s.find("<think>") {
        match s[a..].find("</think>") {
            Some(b) => s.replace_range(a..a + b + "</think>".len(), ""),
            None => break,
        }
    }
    let q = s.trim().lines().next().unwrap_or("").trim().trim_matches(|c| c == '"' || c == '\'' || c == '`').to_string();
    let q = if q.is_empty() { fallback.trim().to_string() } else { q };
    q.chars().take(200).collect()
}

pub fn make_query(ollama: &Ollama, model: &str, info: &ModelInfo, messages: &[Value]) -> String {
    let last = messages.iter().rfind(|m| m["role"] == "user").map(text_of).unwrap_or_default();
    let mut payload = query_request(model, info, messages, &crate::time::now_local().date());
    let reply = ollama.post("/api/chat", &payload, true).or_else(|_| {
        // Some models refuse "think"; ask again without it.
        payload.as_object_mut().unwrap().remove("think");
        ollama.post("/api/chat", &payload, true)
    });
    let text = reply.ok().and_then(|mut r| r.body_mut().read_json::<Value>().ok()).and_then(|v| v["message"]["content"].as_str().map(String::from)).unwrap_or_default();
    clean_query(&text, &last)
}

/// SearXNG's JSON results → unique hits, at most SEARCH_RESULTS.
pub fn parse_results(v: &Value) -> Vec<Hit> {
    let mut seen = std::collections::HashSet::new();
    let mut out = vec![];
    for x in v["results"].as_array().into_iter().flatten() {
        let Some(u) = x["url"].as_str().filter(|u| !u.is_empty()) else { continue };
        if seen.insert(u.to_string()) {
            out.push(Hit { title: x["title"].as_str().unwrap_or("").into(), url: u.into(), snippet: x["content"].as_str().unwrap_or("").into() });
        }
        if out.len() >= SEARCH_RESULTS {
            break;
        }
    }
    out
}

pub fn search(searxng: &str, query: &str) -> Vec<Hit> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(15))).proxy(None).build().into();
    let url = format!("{}/search?q={}&format=json", searxng.trim_end_matches('/'), crate::identity::form_encode(query));
    match agent.get(&url).header("User-Agent", UA).call().and_then(|mut r| r.body_mut().read_json::<Value>()) {
        Ok(v) => parse_results(&v),
        Err(e) => {
            crate::log(&format!("search failed: {}", short_error(&e)));
            vec![]
        }
    }
}

fn short_error(e: &ureq::Error) -> String {
    let s = format!("{e:?}");
    s.split(['(', ' ', '{']).next().unwrap_or("error").to_string()
}

// --- Reading pages (public addresses only) --------------------------------------------------------

/// Resolves like ureq's own resolver, then refuses unless every address is public. Each redirect
/// is resolved again, and the connection only ever goes to an address checked here, so neither a
/// result, a redirect nor a changed DNS answer can point this PC at the home network or itself.
#[derive(Debug, Default)]
pub struct PublicOnly(ureq::unversioned::resolver::DefaultResolver);

impl ureq::unversioned::resolver::Resolver for PublicOnly {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        config: &ureq::config::Config,
        timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<ureq::unversioned::resolver::ResolvedSocketAddrs, ureq::Error> {
        let addrs = self.0.resolve(uri, config, timeout)?;
        if addrs.iter().all(|a| is_global(a.ip())) {
            Ok(addrs)
        } else {
            Err(ureq::Error::Io(std::io::Error::other("not a public address")))
        }
    }
}

pub fn page_agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(8))).proxy(None).max_redirects(5).build();
    ureq::Agent::with_parts(config, ureq::unversioned::transport::DefaultConnector::new(), PublicOnly::default())
}

/// A web page as plain text (at most PAGE_CHARS), or "" if it can't or mustn't be read.
pub fn read_page(agent: &ureq::Agent, url: &str) -> String {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return String::new();
    }
    let Ok(mut r) = agent.get(url).header("User-Agent", UA).header("Accept", "text/html,text/plain").call() else { return String::new() };
    let ctype = r.headers().get("Content-Type").and_then(|v| v.to_str().ok()).unwrap_or("").to_ascii_lowercase();
    let html = ctype.contains("html");
    if !html && !ctype.contains("text/plain") {
        return String::new();
    }
    let mut raw = vec![];
    if std::io::Read::read_to_end(&mut std::io::Read::take(r.body_mut().as_reader(), 2 * 1024 * 1024), &mut raw).is_err() {
        return String::new();
    }
    let text = decode(&raw, &ctype);
    let out = if html { html_text(&text) } else { text };
    out.chars().take(PAGE_CHARS).collect()
}

/// Bytes → text by the Content-Type charset: Latin-1 and Windows-1252 by hand, everything else as
/// UTF-8 (bad bytes replaced).
pub fn decode(raw: &[u8], ctype: &str) -> String {
    let ctype = ctype.to_ascii_lowercase();
    let charset = ctype.split(';').filter_map(|p| p.trim().strip_prefix("charset=")).next().unwrap_or("utf-8").trim_matches('"');
    if ["iso-8859-1", "latin1", "latin-1", "windows-1252", "cp1252"].contains(&charset) {
        raw.iter().map(|&b| b as char).collect()
    } else {
        String::from_utf8_lossy(raw).into_owned()
    }
}

const SKIP: [&str; 11] = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "iframe", "template"];
const BLOCK: [&str; 15] = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "pre", "blockquote"];

/// The readable text of a page: scripts, menus and the like left out, one line per block, lines of
/// one character or less dropped (as the Python gateway's extractor).
pub fn html_text(html: &str) -> String {
    let mut parts = String::new();
    let mut skip = 0usize;
    let mut rest = html;
    while !rest.is_empty() {
        let Some(lt) = rest.find('<') else {
            if skip == 0 {
                parts += &entities(rest);
            }
            break;
        };
        if skip == 0 {
            parts += &entities(&rest[..lt]);
        }
        rest = &rest[lt..];
        if rest.starts_with("<!--") {
            rest = rest.find("-->").map(|e| &rest[e + 3..]).unwrap_or("");
            continue;
        }
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[1..gt];
        rest = &rest[gt + 1..];
        let closing = tag.starts_with('/');
        let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }
        // Raw-text elements: their content isn't markup, skip straight to the end tag.
        if !closing && (name == "script" || name == "style") {
            let end = format!("</{name}");
            let lower = rest.to_ascii_lowercase();
            rest = match lower.find(&end) {
                Some(e) => rest[e..].find('>').map(|g| &rest[e + g + 1..]).unwrap_or(""),
                None => "",
            };
            continue;
        }
        let self_closing = tag.ends_with('/');
        if SKIP.contains(&name.as_str()) {
            if closing {
                skip = skip.saturating_sub(1);
            } else if !self_closing {
                skip += 1;
            }
        } else if BLOCK.contains(&name.as_str()) {
            parts.push('\n');
        }
    }
    parts
        .split('\n')
        .map(|l| l.split([' ', '\t', '\r', '\u{b}', '\u{c}']).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "))
        .filter(|l| l.chars().count() > 1)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The common character references (named ones the web uses most, and every numeric one).
pub fn entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(a) = rest.find('&') {
        out += &rest[..a];
        rest = &rest[a..];
        let end = rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).map(|e| e + 1).unwrap_or(rest.len());
        let name = &rest[1..end];
        let ch = if let Some(n) = name.strip_prefix("#x").or(name.strip_prefix("#X")) {
            u32::from_str_radix(n, 16).ok().and_then(char::from_u32)
        } else if let Some(n) = name.strip_prefix('#') {
            n.parse().ok().and_then(char::from_u32)
        } else {
            match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some('\u{a0}'),
                "ndash" => Some('–'),
                "mdash" => Some('—'),
                "hellip" => Some('…'),
                "lsquo" => Some('‘'),
                "rsquo" => Some('’'),
                "ldquo" => Some('“'),
                "rdquo" => Some('”'),
                "copy" => Some('©'),
                "reg" => Some('®'),
                "deg" => Some('°'),
                "middot" => Some('·'),
                "bull" => Some('•'),
                "euro" => Some('€'),
                "pound" => Some('£'),
                "times" => Some('×'),
                _ => None,
            }
        };
        match ch {
            Some(c) if end > 1 => {
                out.push(c);
                rest = &rest[end..];
                if rest.starts_with(';') {
                    rest = &rest[1..];
                }
            }
            _ => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out + rest
}

/// The results as the model sees them: numbered, with page text (or the snippet when a page
/// couldn't be read in time), and how to cite them.
pub fn context(query: &str, hits: &[Hit], pages: &[String], now: &str) -> String {
    let blocks: Vec<String> = hits
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let body = pages.get(i).filter(|p| !p.is_empty()).unwrap_or(&r.snippet);
            format!("[{}] {}\nURL: {}\n{body}", i + 1, r.title, r.url)
        })
        .collect();
    format!(
        "Live web search results for \"{query}\" (retrieved {now}):\n\n{}\n\nUse these results to answer the user's latest message. Cite sources inline like [1] or [2], matching the numbers above. If the results don't answer it, say so and answer from your own knowledge.",
        blocks.join("\n\n")
    )
}

/// Search, then read the top pages; `on_results` gets the result URLs as soon as they're known.
/// One slow site mustn't hold up the answer: pages not read by the deadline use their snippet.
pub fn research(searxng: &str, query: &str, on_results: &mut dyn FnMut(&[String])) -> (Vec<String>, String) {
    let hits = search(searxng, query);
    if hits.is_empty() {
        return (vec![], String::new());
    }
    let urls: Vec<String> = hits.iter().map(|h| h.url.clone()).collect();
    on_results(&urls);
    let pages = read_pages(&urls[..urls.len().min(PAGES_TO_READ)], PAGE_DEADLINE, |u| read_page(&page_agent(), u));
    (urls, context(query, &hits, &pages, &crate::time::now_local().long()))
}

/// Reads pages in parallel; any not done by the deadline count as "" (their threads finish alone).
pub fn read_pages(urls: &[String], deadline: Duration, read: impl Fn(&str) -> String + Send + Sync + Clone + 'static) -> Vec<String> {
    let (tx, rx) = mpsc::channel();
    for (i, u) in urls.iter().enumerate() {
        let (tx, u, read) = (tx.clone(), u.clone(), read.clone());
        std::thread::spawn(move || {
            let _ = tx.send((i, read(&u)));
        });
    }
    drop(tx);
    let mut pages = vec![String::new(); urls.len()];
    let end = Instant::now() + deadline;
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok((i, p)) => pages[i] = p,
            Err(_) => break,
        }
    }
    pages
}
