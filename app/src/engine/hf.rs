//! Hugging Face: search GGUF repos, list their files, peek inside a file before downloading it
//! (the header arrives with an HTTP range request, so "will it fit?" is answered from a few MB).

use cubic_gguf::{Gguf, ParseError};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Clone, Debug)]
pub struct File {
    /// Path inside the repo (first part for split models).
    pub path: String,
    pub parts: Vec<String>,
    pub size: u64,
    pub is_mmproj: bool,
}

fn get(url: &str, token: &str) -> Result<Value, String> {
    let mut r = crate::llm::agent(10).get(url);
    if !token.is_empty() {
        r = r.header("Authorization", format!("Bearer {token}"));
    }
    let resp = r.call().map_err(|e| format!("Couldn't reach Hugging Face: {e}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        return Err(match status {
            401 | 403 => "This repo needs a Hugging Face token (Settings) or accepting its license on the website.".into(),
            404 => "Not found on Hugging Face.".into(),
            s => format!("Hugging Face answered {s}"),
        });
    }
    resp.into_body().read_json().map_err(|e| e.to_string())
}

pub fn search(query: &str, token: &str) -> Result<Vec<Repo>, String> {
    let q: String = query.trim().chars().map(|c| if c.is_alphanumeric() || "-_./".contains(c) { c } else { '+' }).collect();
    let url = format!("https://huggingface.co/api/models?search={q}&filter=gguf&sort=downloads&direction=-1&limit=40");
    let v = get(&url, token)?;
    Ok(v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            Some(Repo {
                id: m["id"].as_str().or_else(|| m["modelId"].as_str())?.to_string(),
                downloads: m["downloads"].as_u64().unwrap_or(0),
                likes: m["likes"].as_u64().unwrap_or(0),
            })
        })
        .collect())
}

pub fn files(repo: &str, token: &str) -> Result<Vec<File>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let v = get(&url, token)?;
    let all: Vec<(String, u64)> = v
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["type"].as_str() == Some("file"))
        .filter_map(|f| {
            let path = f["path"].as_str()?.to_string();
            let size = f["lfs"]["size"].as_u64().or_else(|| f["size"].as_u64()).unwrap_or(0);
            path.to_lowercase().ends_with(".gguf").then_some((path, size))
        })
        .collect();
    Ok(group_files(all))
}

/// Groups split parts (`name-00001-of-00003.gguf`) into one entry with their total size.
pub fn group_files(mut all: Vec<(String, u64)>) -> Vec<File> {
    all.sort();
    let mut out: Vec<File> = vec![];
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (path, size) in all {
        let key = super::library::split_base(path.trim_end_matches(".gguf")).unwrap_or_else(|| path.clone());
        if let Some(&i) = index.get(&key) {
            out[i].parts.push(path);
            out[i].size += size;
            continue;
        }
        index.insert(key, out.len());
        out.push(File { parts: vec![path.clone()], is_mmproj: path.to_lowercase().contains("mmproj"), path, size });
    }
    out
}

pub fn file_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{path}")
}

/// Reads just enough of the file's start to parse its header (growing the range as needed).
pub fn peek_header(repo: &str, path: &str, token: &str) -> Result<Gguf, String> {
    let url = file_url(repo, path);
    let mut want: usize = 4 << 20;
    loop {
        let mut r = crate::llm::agent(15).get(&url).header("Range", format!("bytes=0-{}", want - 1));
        if !token.is_empty() {
            r = r.header("Authorization", format!("Bearer {token}"));
        }
        let resp = r.call().map_err(|e| format!("Couldn't read the file: {e}"))?;
        let status = resp.status().as_u16();
        if status != 206 && status != 200 {
            return Err(format!("Hugging Face answered {status}"));
        }
        let mut body = Vec::with_capacity(want);
        std::io::Read::read_to_end(&mut resp.into_body().into_reader().take(want as u64), &mut body).map_err(|e| e.to_string())?;
        match Gguf::parse(&body, None) {
            Ok(g) => return Ok(g),
            Err(ParseError::Truncated { .. }) if want < (128 << 20) && body.len() >= want => want *= 4,
            Err(e) => return Err(e.to_string()),
        }
    }
}

use std::io::Read as _;
