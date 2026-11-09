//! The user's Google account for Bots: Gmail, Calendar and Drive through the user's own Google Cloud
//! OAuth client ("Desktop app"). `cubicai-gateway google-login <client_secret.json>` signs in on this
//! PC's browser (PKCE, loopback redirect) and keeps the client and tokens in `<config>/google.json`
//! (0600); the gateway reads it on each use, so signing in needs no restart. Tokens never leave the PC.
//!
//! Reading (search, read, list, download) is free; sending, creating, changing and deleting are marked
//! `Act::Ask` so the run asks the user on the phone first (bots/mod.rs).

use base64::Engine;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

pub const SCOPES: [&str; 5] = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
];
const TEXT_CHARS: usize = 12_000;
const DOWNLOAD_BYTES: usize = 50 * 1024 * 1024;

/// Where Google's APIs are (tests point them all at a fake).
#[derive(Clone, Debug)]
pub struct Bases {
    pub auth: String,
    pub token: String,
    pub userinfo: String,
    pub gmail: String,
    pub calendar: String,
    pub drive: String,
    pub upload: String,
}

impl Bases {
    pub fn real() -> Bases {
        Bases {
            auth: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token: "https://oauth2.googleapis.com/token".into(),
            userinfo: "https://openidconnect.googleapis.com/v1/userinfo".into(),
            gmail: "https://gmail.googleapis.com/gmail/v1/users/me".into(),
            calendar: "https://www.googleapis.com/calendar/v3".into(),
            drive: "https://www.googleapis.com/drive/v3".into(),
            upload: "https://www.googleapis.com/upload/drive/v3".into(),
        }
    }

    pub fn at(base: &str) -> Bases {
        let b = base.trim_end_matches('/');
        Bases {
            auth: format!("{b}/auth"),
            token: format!("{b}/token"),
            userinfo: format!("{b}/userinfo"),
            gmail: format!("{b}/gmail/v1/users/me"),
            calendar: format!("{b}/calendar/v3"),
            drive: format!("{b}/drive/v3"),
            upload: format!("{b}/upload/drive/v3"),
        }
    }
}

pub struct Google {
    file: PathBuf,
    bases: Bases,
    lock: Mutex<()>,
    pending: Mutex<Option<Pending>>,
}

/// Whether a tool only reads, or changes something and asks the user first (with what to show them).
pub enum Act {
    Read,
    Ask(String),
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(60))).http_status_as_error(false).proxy(None).build().new_agent()
}

fn enc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn form(pairs: &[(&str, &str)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", enc(k), enc(v))).collect::<Vec<_>>().join("&")
}

impl Google {
    pub fn new(config_dir: &Path, bases: Bases) -> Google {
        Google { file: config_dir.join("google.json"), bases, lock: Mutex::new(()), pending: Mutex::new(None) }
    }

    fn load(&self) -> Option<Value> {
        std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice(&b).ok())
    }

    /// The signed-in account's address, if any.
    pub fn account(&self) -> Option<String> {
        let v = self.load()?;
        v["refresh_token"].as_str()?;
        Some(v["email"].as_str().unwrap_or("your Google account").to_string())
    }

    pub fn status(&self) -> Value {
        match self.account() {
            Some(email) => json!({"connected": true, "email": email}),
            None => json!({"connected": false, "client": self.load().is_some_and(|v| v["client_id"].is_string())}),
        }
    }

    pub fn sign_out(&self) {
        if let Some(v) = self.load()
            && let Some(t) = v["refresh_token"].as_str()
        {
            let _ = agent().post("https://oauth2.googleapis.com/revoke").content_type("application/x-www-form-urlencoded").send(form(&[("token", t)]));
        }
        let _ = std::fs::remove_file(&self.file);
    }

    /// A working access token (refreshed when it's about to run out).
    fn token(&self, force: bool) -> Result<String, String> {
        let _g = self.lock.lock().unwrap();
        let mut v = self.load().ok_or("Google isn't connected. On the PC, run: cubicai-gateway google-login <client_secret.json>")?;
        let now = crate::time::now();
        if !force
            && let Some(t) = v["access_token"].as_str()
            && v["expires_at"].as_f64().unwrap_or(0.0) > now + 60.0
        {
            return Ok(t.to_string());
        }
        let refresh = v["refresh_token"].as_str().ok_or("Google isn't connected. On the PC, run: cubicai-gateway google-login <client_secret.json>")?;
        let body = form(&[
            ("client_id", v["client_id"].as_str().unwrap_or("")),
            ("client_secret", v["client_secret"].as_str().unwrap_or("")),
            ("refresh_token", refresh),
            ("grant_type", "refresh_token"),
        ]);
        let mut r = agent().post(&self.bases.token).content_type("application/x-www-form-urlencoded").send(body).map_err(|e| format!("Couldn't reach Google: {e}"))?;
        let got: Value = r.body_mut().read_json().unwrap_or(Value::Null);
        let Some(access) = got["access_token"].as_str() else {
            return Err(if got["error"] == "invalid_grant" {
                "Google signed CubicAI out (the sign-in expired or was revoked; apps in Google's Testing mode expire weekly). On the PC, run cubicai-gateway google-login again.".into()
            } else {
                format!("Google refused the sign-in: {}", got["error_description"].as_str().or(got["error"].as_str()).unwrap_or("unknown error"))
            });
        };
        v["access_token"] = access.into();
        v["expires_at"] = (now + got["expires_in"].as_f64().unwrap_or(3600.0)).into();
        super::write_private(&self.file, serde_json::to_string_pretty(&v).unwrap().as_bytes())?;
        Ok(access.to_string())
    }

    /// One API call; refreshes the token once on 401. Errors in words.
    fn call(&self, method: &str, url: &str, body: Option<(&str, Vec<u8>)>) -> Result<(Vec<u8>, String), String> {
        for attempt in 0..2 {
            let token = self.token(attempt == 1)?;
            let a = agent();
            let auth = format!("Bearer {token}");
            let resp = match (method, &body) {
                ("GET", _) => a.get(url).header("Authorization", &auth).call(),
                ("DELETE", _) => a.delete(url).header("Authorization", &auth).call(),
                (m, Some((ct, b))) => {
                    let req = match m {
                        "PATCH" => a.patch(url),
                        "PUT" => a.put(url),
                        _ => a.post(url),
                    };
                    req.header("Authorization", &auth).content_type(*ct).send(&b[..])
                }
                (_, None) => a.post(url).header("Authorization", &auth).send_empty(),
            };
            let mut r = resp.map_err(|e| format!("Couldn't reach Google: {e}"))?;
            let status = r.status().as_u16();
            let ctype = r.headers().get("content-type").and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
            let bytes = r.body_mut().with_config().limit(DOWNLOAD_BYTES as u64).read_to_vec().unwrap_or_default();
            if status == 401 && attempt == 0 {
                continue;
            }
            if status >= 400 {
                let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
                let msg = v["error"]["message"].as_str().unwrap_or("").to_string();
                return Err(match status {
                    401 => "Google refused CubicAI's sign-in. On the PC, run cubicai-gateway google-login again.".into(),
                    403 if msg.contains("has not been used") || msg.contains("is disabled") => {
                        format!("That Google API isn't switched on in your Google Cloud project. Enable it (APIs & Services → Library), wait a minute, and try again. ({msg})")
                    }
                    403 if msg.contains("insufficient") || msg.contains("scope") => "CubicAI wasn't given that permission when you signed in. Run cubicai-gateway google-login again and tick every box.".into(),
                    404 => "Google says that doesn't exist (check the id).".into(),
                    429 => "Google says too many requests; wait a minute.".into(),
                    _ => format!("Google said {status}: {msg}"),
                });
            }
            return Ok((bytes, ctype));
        }
        unreachable!()
    }

    fn json(&self, method: &str, url: &str, body: Option<Value>) -> Result<Value, String> {
        let (bytes, _) = self.call(method, url, body.map(|b| ("application/json", b.to_string().into_bytes())))?;
        if bytes.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&bytes).map_err(|_| "Google sent something unexpected.".to_string())
    }

    /// Whether a call only reads, or what to ask the user before it runs.
    pub fn act(&self, name: &str, a: &Value) -> Act {
        let s = |k: &str| a[k].as_str().unwrap_or("").trim().to_string();
        let short = |t: String, n: usize| if t.chars().count() > n { t.chars().take(n).collect::<String>() + "…" } else { t };
        match name {
            "gmail_send" => {
                let mut subject = s("subject");
                if subject.is_empty() && !s("reply_to_id").is_empty() {
                    // A reply: show the subject it will get.
                    let url = format!("{}/messages/{}?format=metadata&metadataHeaders=Subject", self.bases.gmail, enc(&s("reply_to_id")));
                    let os = self.json("GET", &url, None).map(|o| headers(&o["payload"])("subject")).unwrap_or_default();
                    subject = if os.to_lowercase().starts_with("re:") { os } else { format!("Re: {os}") };
                }
                Act::Ask(format!(
                    "Send an email to {}{}: “{}”\n\n{}",
                    s("to"),
                    if s("cc").is_empty() { String::new() } else { format!(" (cc {})", s("cc")) },
                    subject,
                    short(s("body"), 400)
                ))
            }
            "calendar_create" => Act::Ask(format!("Add “{}” to your calendar, {} to {}{}", s("title"), s("start"), s("end"), if s("attendees").is_empty() { String::new() } else { format!(", inviting {}", s("attendees")) })),
            "calendar_update" => Act::Ask(format!("Change the calendar event {}: {}", s("id"), short(a.to_string(), 300))),
            "calendar_delete" => Act::Ask(format!("Delete the calendar event {}", s("id"))),
            "drive_create" if !s("path").is_empty() => Act::Ask(format!("Upload {} from the workspace to your Google Drive", s("path"))),
            "drive_create" => Act::Ask(format!("Create “{}” in your Google Drive", s("name"))),
            _ => Act::Read,
        }
    }

    /// Runs one Google tool. `workspace` is for drive_download and uploads.
    pub fn run(&self, name: &str, a: &Value, workspace: &Path) -> Result<String, String> {
        let s = |k: &str| a[k].as_str().unwrap_or("").trim().to_string();
        let max = a["max"].as_u64().unwrap_or(10).clamp(1, 25);
        let b = &self.bases;
        match name {
            "gmail_search" => {
                let q = s("query");
                let list = self.json("GET", &format!("{}/messages?maxResults={max}&q={}", b.gmail, enc(&q)), None)?;
                let ids: Vec<String> = list["messages"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(String::from)).collect();
                if ids.is_empty() {
                    return Ok(format!("No emails match “{q}”."));
                }
                let mut out = vec![format!("Emails matching “{q}” (newest first):")];
                for id in ids {
                    let m = self.json("GET", &format!("{}/messages/{id}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date&metadataHeaders=To", b.gmail), None)?;
                    let h = headers(&m["payload"]);
                    let unread = m["labelIds"].as_array().is_some_and(|l| l.iter().any(|x| x == "UNREAD"));
                    out.push(format!(
                        "- id {id}{}: from {} · {} · “{}”\n  {}",
                        if unread { " (unread)" } else { "" },
                        h("from"),
                        h("date"),
                        h("subject"),
                        entities(m["snippet"].as_str().unwrap_or(""))
                    ));
                }
                Ok(out.join("\n"))
            }
            "gmail_read" => {
                let id = s("id");
                let m = self.json("GET", &format!("{}/messages/{}?format=full", b.gmail, enc(&id)), None)?;
                let h = headers(&m["payload"]);
                let mut plain = String::new();
                let mut html = String::new();
                let mut files = vec![];
                walk(&m["payload"], &mut plain, &mut html, &mut files);
                let mut body = if plain.trim().is_empty() { crate::search::html_text(&html) } else { plain };
                if body.chars().count() > TEXT_CHARS {
                    body = body.chars().take(TEXT_CHARS).collect::<String>() + "\n[…cut]";
                }
                let mut t = format!("From: {}\nTo: {}\nDate: {}\nSubject: {}\nThread: {}\n\n{}", h("from"), h("to"), h("date"), h("subject"), m["threadId"].as_str().unwrap_or(""), body.trim());
                if !files.is_empty() {
                    t += &format!("\n\nAttachments: {}", files.join(", "));
                }
                Ok(t)
            }
            "gmail_send" | "gmail_draft" => {
                let mut thread = None;
                let mut extra = String::new();
                let mut subject = s("subject");
                let reply_to = s("reply_to_id");
                if !reply_to.is_empty() {
                    let o = self.json("GET", &format!("{}/messages/{}?format=metadata&metadataHeaders=Message-ID&metadataHeaders=Subject&metadataHeaders=References", b.gmail, enc(&reply_to)), None)?;
                    let h = headers(&o["payload"]);
                    let mid = h("message-id");
                    if !mid.is_empty() {
                        let refs = h("references");
                        extra = format!("In-Reply-To: {mid}\r\nReferences: {}{mid}\r\n", if refs.is_empty() { String::new() } else { format!("{refs} ") });
                    }
                    if subject.is_empty() {
                        let os = h("subject");
                        subject = if os.to_lowercase().starts_with("re:") { os } else { format!("Re: {os}") };
                    }
                    thread = o["threadId"].as_str().map(String::from);
                }
                let to = s("to");
                if to.is_empty() || !to.contains('@') {
                    return Err("Say who it goes to (an email address).".into());
                }
                let raw = mime(&to, &s("cc"), &subject, &s("body"), &extra);
                let mut msg = json!({"raw": base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw)});
                if let Some(t) = &thread {
                    msg["threadId"] = t.clone().into();
                }
                if name == "gmail_draft" {
                    let d = self.json("POST", &format!("{}/drafts", b.gmail), Some(json!({"message": msg})))?;
                    Ok(format!("Saved a draft to {to} (“{subject}”), draft id {}. It isn't sent; the user can send it from Gmail.", d["id"].as_str().unwrap_or("")))
                } else {
                    let sent = self.json("POST", &format!("{}/messages/send", b.gmail), Some(msg))?;
                    Ok(format!("Sent to {to} (“{subject}”), message id {}.", sent["id"].as_str().unwrap_or("")))
                }
            }
            "calendar_events" => {
                let now = crate::time::now() as i64;
                let from = crate::time::rfc3339_local(&s("from")).unwrap_or_else(|| stamp(now));
                let to = crate::time::rfc3339_local(&s("to")).unwrap_or_else(|| stamp(now + 7 * 86400));
                let mut url = format!("{}/calendars/primary/events?singleEvents=true&orderBy=startTime&maxResults={}&timeMin={}&timeMax={}", b.calendar, max.max(20), enc(&from), enc(&to));
                if !s("query").is_empty() {
                    url += &format!("&q={}", enc(&s("query")));
                }
                let v = self.json("GET", &url, None)?;
                let items = v["items"].as_array().cloned().unwrap_or_default();
                if items.is_empty() {
                    return Ok(format!("No events between {from} and {to}."));
                }
                let mut out = vec![format!("Events from {from} to {to}:")];
                for e in items {
                    let start = e["start"]["dateTime"].as_str().or(e["start"]["date"].as_str()).unwrap_or("");
                    let end = e["end"]["dateTime"].as_str().or(e["end"]["date"].as_str()).unwrap_or("");
                    let mut line = format!("- id {}: “{}” {start} → {end}", e["id"].as_str().unwrap_or(""), e["summary"].as_str().unwrap_or("(no title)"));
                    if let Some(l) = e["location"].as_str() {
                        line += &format!(" @ {l}");
                    }
                    let who: Vec<&str> = e["attendees"].as_array().into_iter().flatten().filter_map(|x| x["email"].as_str()).take(6).collect();
                    if !who.is_empty() {
                        line += &format!(" with {}", who.join(", "));
                    }
                    out.push(line);
                }
                Ok(out.join("\n"))
            }
            "calendar_create" | "calendar_update" => {
                let mut ev = json!({});
                if !s("title").is_empty() {
                    ev["summary"] = s("title").into();
                }
                for (k, field) in [("description", "description"), ("location", "location")] {
                    if !s(k).is_empty() {
                        ev[field] = s(k).into();
                    }
                }
                for k in ["start", "end"] {
                    let v = s(k);
                    if v.is_empty() {
                        continue;
                    }
                    ev[k] = if v.len() == 10 { json!({"date": v}) } else { json!({"dateTime": crate::time::rfc3339_local(&v).ok_or(format!("Give {k} like 2026-11-10T09:00 (or 2026-11-10 for all day)."))?}) };
                }
                if !s("attendees").is_empty() {
                    ev["attendees"] = s("attendees").split([',', ' ']).filter(|x| x.contains('@')).map(|e| json!({"email": e})).collect();
                }
                if name == "calendar_create" {
                    if ev["summary"].is_null() || ev["start"].is_null() || ev["end"].is_null() {
                        return Err("calendar_create needs a title, start and end.".into());
                    }
                    let e = self.json("POST", &format!("{}/calendars/primary/events?sendUpdates=all", b.calendar), Some(ev))?;
                    Ok(format!("Added “{}” ({}), event id {}.", e["summary"].as_str().unwrap_or(""), e["htmlLink"].as_str().unwrap_or(""), e["id"].as_str().unwrap_or("")))
                } else {
                    let e = self.json("PATCH", &format!("{}/calendars/primary/events/{}?sendUpdates=all", b.calendar, enc(&s("id"))), Some(ev))?;
                    Ok(format!("Changed “{}”.", e["summary"].as_str().unwrap_or("")))
                }
            }
            "calendar_delete" => {
                self.call("DELETE", &format!("{}/calendars/primary/events/{}?sendUpdates=all", b.calendar, enc(&s("id"))), None)?;
                Ok("Deleted the event.".into())
            }
            "drive_search" => {
                let q = s("query").replace('\\', "\\\\").replace('\'', "\\'");
                let filter = format!("(name contains '{q}' or fullText contains '{q}') and trashed = false");
                let v = self.json("GET", &format!("{}/files?pageSize={max}&q={}&fields=files(id,name,mimeType,modifiedTime,webViewLink,size)&orderBy=modifiedTime%20desc", b.drive, enc(&filter)), None)?;
                let files = v["files"].as_array().cloned().unwrap_or_default();
                if files.is_empty() {
                    return Ok(format!("No files match “{}”.", s("query")));
                }
                Ok(files
                    .iter()
                    .map(|f| format!("- id {}: “{}” ({}, changed {}) {}", f["id"].as_str().unwrap_or(""), f["name"].as_str().unwrap_or(""), kind(f["mimeType"].as_str().unwrap_or("")), f["modifiedTime"].as_str().unwrap_or(""), f["webViewLink"].as_str().unwrap_or("")))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "drive_read" | "drive_download" => {
                let id = s("id");
                let meta = self.json("GET", &format!("{}/files/{}?fields=id,name,mimeType,size", b.drive, enc(&id)), None)?;
                let mt = meta["mimeType"].as_str().unwrap_or("");
                let fname = meta["name"].as_str().unwrap_or("file").to_string();
                let export = match mt {
                    "application/vnd.google-apps.document" => Some(if name == "drive_download" { ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx") } else { ("text/plain", ".txt") }),
                    "application/vnd.google-apps.spreadsheet" => Some(if name == "drive_download" { ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx") } else { ("text/csv", ".csv") }),
                    "application/vnd.google-apps.presentation" => Some(if name == "drive_download" { ("application/pdf", ".pdf") } else { ("text/plain", ".txt") }),
                    m if m.starts_with("application/vnd.google-apps.") => return Err(format!("“{fname}” is a {} and can't be read as a file.", kind(m))),
                    _ => None,
                };
                let url = match export {
                    Some((m, _)) => format!("{}/files/{}/export?mimeType={}", b.drive, enc(&id), enc(m)),
                    None => format!("{}/files/{}?alt=media", b.drive, enc(&id)),
                };
                let (bytes, _) = self.call("GET", &url, None)?;
                if name == "drive_download" {
                    let mut target = s("path");
                    if target.is_empty() {
                        target = format!("drive/{}{}", fname.replace('/', "_"), export.map(|e| e.1).unwrap_or(""));
                    }
                    let p = super::tools::safe_path(workspace, &target)?;
                    if let Some(d) = p.parent() {
                        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
                    }
                    std::fs::write(&p, &bytes).map_err(|e| e.to_string())?;
                    return Ok(format!("Saved “{fname}” to {target} in the workspace ({} KB).", bytes.len() / 1024));
                }
                let text = String::from_utf8_lossy(&bytes);
                if export.is_none() && !(mt.starts_with("text/") || mt.contains("json") || mt.contains("xml") || mt.contains("csv")) {
                    return Err(format!("“{fname}” is a {} file; use drive_download to put it in the workspace and read it on your computer.", kind(mt)));
                }
                let mut t: String = text.chars().take(TEXT_CHARS).collect();
                if text.chars().count() > TEXT_CHARS {
                    t += "\n[…cut; use drive_download for the whole file]";
                }
                Ok(format!("“{fname}”:\n{t}"))
            }
            "drive_create" => {
                let (fname, bytes, doc) = if s("path").is_empty() {
                    (s("name"), s("content").into_bytes(), a["google_doc"].as_bool().unwrap_or(true))
                } else {
                    let p = super::tools::safe_path(workspace, &s("path"))?;
                    let n = if s("name").is_empty() { p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default() } else { s("name") };
                    (n, std::fs::read(&p).map_err(|_| format!("There's no file {} in the workspace.", s("path")))?, false)
                };
                if fname.is_empty() {
                    return Err("Give the file a name.".into());
                }
                let mut meta = json!({"name": fname});
                if doc {
                    meta["mimeType"] = "application/vnd.google-apps.document".into();
                }
                let boundary = format!("cubic{}", super::new_id());
                let mut body = format!("--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{meta}\r\n--{boundary}\r\nContent-Type: {}\r\n\r\n", if doc || s("path").is_empty() { "text/plain; charset=UTF-8" } else { "application/octet-stream" }).into_bytes();
                body.extend_from_slice(&bytes);
                body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
                let (out, _) = self.call("POST", &format!("{}/files?uploadType=multipart&fields=id,name,webViewLink", b.upload), Some((&format!("multipart/related; boundary={boundary}"), body)))?;
                let f: Value = serde_json::from_slice(&out).unwrap_or(Value::Null);
                Ok(format!("Created “{}” in Google Drive: {}", f["name"].as_str().unwrap_or(""), f["webViewLink"].as_str().unwrap_or("")))
            }
            other => Err(format!("There's no tool called {other:?}.")),
        }
    }
}

fn stamp(secs: i64) -> String {
    let t = crate::time::local(secs);
    crate::time::rfc3339_local(&format!("{}T{:02}:{:02}", t.date(), t.hour, t.minute)).unwrap_or_default()
}

fn kind(mime: &str) -> &str {
    match mime {
        "application/vnd.google-apps.document" => "Google Doc",
        "application/vnd.google-apps.spreadsheet" => "Google Sheet",
        "application/vnd.google-apps.presentation" => "Google Slides",
        "application/vnd.google-apps.folder" => "folder",
        "application/pdf" => "PDF",
        m => m,
    }
}

fn entities(s: &str) -> String {
    crate::search::entities(s)
}

/// A message part's headers, looked up by lower-case name.
fn headers(payload: &Value) -> impl Fn(&str) -> String + '_ {
    move |name: &str| {
        payload["headers"].as_array().into_iter().flatten().find(|h| h["name"].as_str().is_some_and(|n| n.eq_ignore_ascii_case(name))).and_then(|h| h["value"].as_str()).unwrap_or("").to_string()
    }
}

fn walk(part: &Value, plain: &mut String, html: &mut String, files: &mut Vec<String>) {
    let mt = part["mimeType"].as_str().unwrap_or("");
    if let Some(f) = part["filename"].as_str().filter(|f| !f.is_empty()) {
        files.push(f.to_string());
    } else if let Some(data) = part["body"]["data"].as_str() {
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data.trim_end_matches('=')).unwrap_or_default();
        let text = String::from_utf8_lossy(&bytes);
        match mt {
            "text/plain" => plain.push_str(&text),
            "text/html" => html.push_str(&text),
            _ => {}
        }
    }
    for p in part["parts"].as_array().into_iter().flatten() {
        walk(p, plain, html, files);
    }
}

/// A UTF-8 plain-text email (RFC 5322; non-ASCII subject in RFC 2047 words).
pub fn mime(to: &str, cc: &str, subject: &str, body: &str, extra_headers: &str) -> Vec<u8> {
    let subj = if subject.is_ascii() { subject.to_string() } else { format!("=?UTF-8?B?{}?=", base64::engine::general_purpose::STANDARD.encode(subject)) };
    let mut m = format!("To: {to}\r\n");
    if !cc.is_empty() {
        m += &format!("Cc: {cc}\r\n");
    }
    m += &format!("Subject: {subj}\r\n{extra_headers}MIME-Version: 1.0\r\nContent-Type: text/plain; charset=UTF-8\r\nContent-Transfer-Encoding: base64\r\n\r\n");
    let b64 = base64::engine::general_purpose::STANDARD.encode(body.replace("\r\n", "\n").replace('\n', "\r\n"));
    for chunk in b64.as_bytes().chunks(76) {
        m += std::str::from_utf8(chunk).unwrap();
        m += "\r\n";
    }
    m.into_bytes()
}

/// A sign-in under way (one at a time; started by the phone or the PC command).
struct Pending {
    client_id: String,
    client_secret: String,
    redirect: String,
    verifier: String,
    state: String,
    started: f64,
}

/// The client id and secret from Google Cloud's JSON file (or pasted on the phone).
pub fn client_from_json(raw: &[u8]) -> Result<(String, String), String> {
    let v: Value = serde_json::from_slice(raw).map_err(|_| "That isn't the JSON file Google Cloud gave you (Clients → your Desktop client → Download JSON).".to_string())?;
    let c = if v["installed"].is_object() { &v["installed"] } else if v["web"].is_object() { return Err("That's a \"Web application\" client; make a \"Desktop app\" client instead.".into()) } else { &v };
    let (id, secret) = (c["client_id"].as_str().unwrap_or("").trim(), c["client_secret"].as_str().unwrap_or("").trim());
    if id.is_empty() || secret.is_empty() {
        return Err("That file has no client_id/client_secret. Make the client a \"Desktop app\" and download its JSON.".into());
    }
    Ok((id.to_string(), secret.to_string()))
}

impl Google {
    /// Keeps the OAuth client (so the phone only needs to sign in), keeping any sign-in made with it.
    pub fn set_client(&self, id: &str, secret: &str) -> Result<(), String> {
        let _g = self.lock.lock().unwrap();
        let mut v = self.load().unwrap_or_else(|| json!({}));
        if v["client_id"].as_str() != Some(id) {
            v = json!({});
        }
        v["client_id"] = id.into();
        v["client_secret"] = secret.into();
        super::write_private(&self.file, serde_json::to_string_pretty(&v).unwrap().as_bytes())
    }

    /// Starts a sign-in: the Google page to open, which sends the browser back to `redirect`
    /// (a loopback address on the device the browser runs on) with a code for `finish`.
    pub fn start(&self, redirect: &str) -> Result<String, String> {
        if !(redirect.starts_with("http://127.0.0.1:") || redirect.starts_with("http://localhost:") || redirect.starts_with("http://[::1]:")) {
            return Err("The sign-in has to come back to a loopback address.".into());
        }
        let v = self.load().unwrap_or(Value::Null);
        let (id, secret) = (v["client_id"].as_str().unwrap_or(""), v["client_secret"].as_str().unwrap_or(""));
        if id.is_empty() {
            return Err("Add your Google Cloud client first (its client ID and secret).".into());
        }
        let mut rnd = [0u8; 48];
        ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut rnd).map_err(|_| "no randomness")?;
        let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&rnd[..32]);
        let state = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&rnd[32..]);
        let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(ring::digest::digest(&ring::digest::SHA256, verifier.as_bytes()));
        let url = format!(
            "{}?{}",
            self.bases.auth,
            form(&[
                ("client_id", id),
                ("redirect_uri", redirect),
                ("response_type", "code"),
                ("scope", &SCOPES.join(" ")),
                ("access_type", "offline"),
                ("prompt", "consent"),
                ("code_challenge", &challenge),
                ("code_challenge_method", "S256"),
                ("state", &state),
            ])
        );
        *self.pending.lock().unwrap() =
            Some(Pending { client_id: id.into(), client_secret: secret.into(), redirect: redirect.into(), verifier, state, started: crate::time::now() });
        Ok(url)
    }

    /// Finishes the sign-in `start` began, with the code Google sent back. Returns a note on any
    /// permission the user didn't tick.
    pub fn finish(&self, code: &str, state: &str) -> Result<Option<String>, String> {
        let p = {
            let mut pending = self.pending.lock().unwrap();
            match pending.as_ref() {
                Some(p) if p.state == state && crate::time::now() - p.started < 900.0 => pending.take().unwrap(),
                Some(p) if p.state == state => return Err("That sign-in took too long; start it again.".into()),
                _ => return Err("That sign-in isn't the one under way; start it again.".into()),
            }
        };
        let mut r = agent()
            .post(&self.bases.token)
            .content_type("application/x-www-form-urlencoded")
            .send(form(&[("code", code), ("client_id", &p.client_id), ("client_secret", &p.client_secret), ("redirect_uri", &p.redirect), ("grant_type", "authorization_code"), ("code_verifier", &p.verifier)]))
            .map_err(|e| format!("Couldn't reach Google: {e}"))?;
        let tok: Value = r.body_mut().read_json().unwrap_or(Value::Null);
        let access = tok["access_token"].as_str().ok_or_else(|| format!("Google refused: {}", tok["error_description"].as_str().or(tok["error"].as_str()).unwrap_or("unknown")))?;
        let refresh = tok["refresh_token"].as_str().ok_or("Google gave no refresh token; sign in again.")?;
        let granted = tok["scope"].as_str().unwrap_or("").to_string();
        let me: Value = agent().get(&self.bases.userinfo).header("Authorization", &format!("Bearer {access}")).call().ok().and_then(|mut r| r.body_mut().read_json().ok()).unwrap_or(Value::Null);
        let saved = json!({
            "client_id": p.client_id, "client_secret": p.client_secret, "refresh_token": refresh, "access_token": access,
            "expires_at": crate::time::now() + tok["expires_in"].as_f64().unwrap_or(3600.0), "email": me["email"].as_str().unwrap_or(""), "scope": granted,
        });
        {
            let _g = self.lock.lock().unwrap();
            super::write_private(&self.file, serde_json::to_string_pretty(&saved).unwrap().as_bytes())?;
        }
        let missing: Vec<&str> = [("gmail", "Gmail"), ("calendar", "Calendar"), ("drive", "Drive")].iter().filter(|(k, _)| !granted.contains(&format!("auth/{k}"))).map(|(_, n)| *n).collect();
        Ok((!missing.is_empty()).then(|| format!("You didn't allow {}, so those tools won't work; sign in again and tick every box.", missing.join(", "))))
    }
}

/// `cubicai-gateway google-login <client_secret.json>`: sign in on this PC's browser and keep the tokens.
pub fn login(config_dir: &Path, client_file: &Path) -> Result<(), String> {
    use std::io::{BufRead, BufReader, Write};
    let raw = std::fs::read(client_file).map_err(|e| format!("Couldn't read {}: {e}", client_file.display()))?;
    let (id, secret) = client_from_json(&raw)?;
    let g = Google::new(config_dir, Bases::real());
    g.set_client(&id, &secret)?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    let url = g.start(&format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port()))?;
    let state = g.pending.lock().unwrap().as_ref().map(|p| p.state.clone()).unwrap_or_default();
    println!("Sign in to Google in your browser. If it didn't open, open this:\n\n{url}\n");
    println!("Google will say the app isn't verified (it's your own): Advanced → Go to … (unsafe), then tick every box.\n");
    let _ = std::process::Command::new("xdg-open").arg(&url).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn();
    let code = loop {
        let (mut s, _) = listener.accept().map_err(|e| e.to_string())?;
        let mut line = String::new();
        BufReader::new(&s).read_line(&mut line).map_err(|e| e.to_string())?;
        let q = query(line.split(' ').nth(1).unwrap_or(""));
        let (msg, done) = if q.get("state") != Some(&state) {
            ("Waiting for Google…", None)
        } else if let Some(e) = q.get("error") {
            ("Sign-in was cancelled. You can close this tab.", Some(Err(format!("Google said: {e}"))))
        } else if let Some(c) = q.get("code") {
            ("Signed in. You can close this tab and go back to CubicAI.", Some(Ok(c.clone())))
        } else {
            ("Waiting for Google…", None)
        };
        let page = format!("<!doctype html><meta charset=utf-8><title>CubicAI</title><body style='font-family:sans-serif;padding:3em'><h2>{msg}</h2>");
        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}", page.len());
        if let Some(d) = done {
            break d?;
        }
    };
    let note = g.finish(&code, &state)?;
    println!("Signed in as {}. Bots you give Google to can use Gmail, Calendar and Drive now (no restart needed).", g.account().unwrap_or_default());
    if let Some(n) = note {
        println!("Note: {n}");
    }
    Ok(())
}

/// A request path's query, decoded.
pub fn query(path: &str) -> std::collections::HashMap<String, String> {
    path.split_once('?').map(|(_, q)| q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), crate::identity::form_decode(v))).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emails_are_well_formed() {
        let m = String::from_utf8(mime("a@b.c", "", "Tea at 4 ☕", "Hi\nSee you.", "In-Reply-To: <x@y>\r\n")).unwrap();
        assert!(m.starts_with("To: a@b.c\r\nSubject: =?UTF-8?B?"), "{m}");
        assert!(m.contains("In-Reply-To: <x@y>\r\nMIME-Version: 1.0\r\n"));
        let body = m.split("\r\n\r\n").nth(1).unwrap().replace("\r\n", "");
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(body).unwrap(), b"Hi\r\nSee you.");
    }

    #[test]
    fn query_strings_round_trip() {
        let q = query(&format!("/?code={}&state=s", enc("4/0Ab-x é&=")));
        assert_eq!(q["code"], "4/0Ab-x é&=");
        assert_eq!(q["state"], "s");
    }
}
