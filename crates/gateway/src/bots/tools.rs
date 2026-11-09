//! What a Bot can do: search the web (the PC's SearXNG), read public pages, keep files in the shared
//! workspace folder (never outside it), and keep its own memory.

use regex::Regex;
use serde_json::{json, Value};
use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

pub const READ_FILE_CHARS: usize = 20_000;
pub const WRITE_FILE_BYTES: usize = 1024 * 1024;
pub const LIST_ENTRIES: usize = 200;
/// Tools the Bots engine runs itself (they change the Bot's routines).
pub const ROUTINE_TOOLS: [&str; 2] = ["schedule_routine", "cancel_routine"];
/// Tools for the Bots' computer (computer.rs), offered when a Bot has one.
pub const COMPUTER_TOOLS: [&str; 6] = ["run_command", "browser_open", "browser_click", "browser_type", "browser_scroll", "browser_back"];

/// The user's Google account (google.rs), offered when it's connected and the Bot may use it.
pub const GOOGLE_TOOLS: [&str; 12] = [
    "gmail_search", "gmail_read", "gmail_send", "gmail_draft", "calendar_events", "calendar_create", "calendar_update", "calendar_delete", "drive_search", "drive_read", "drive_download", "drive_create",
];

/// The shared skills library (skills.rs), run by the Bots engine.
pub const SKILL_TOOLS: [&str; 2] = ["use_skill", "save_skill"];

/// The user's phone, offered unless the user switched it off for the Bot.
pub const PHONE_TOOLS: [&str; 6] = ["phone_remind", "phone_notify", "phone_copy", "phone_open", "phone_location", "phone_status"];

pub struct Ctx<'a> {
    pub searxng: &'a str,
    pub workspace: &'a Path,
    pub memory: &'a Path,
}

pub struct Outcome {
    pub text: String,
    pub citations: Vec<String>,
    pub ok: bool,
}

fn ok(text: impl Into<String>) -> Outcome {
    Outcome { text: text.into(), citations: vec![], ok: true }
}

fn fail(text: impl Into<String>) -> Outcome {
    Outcome { text: text.into(), citations: vec![], ok: false }
}

fn function(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}})
}

/// The tools, in OpenAI's function format (with the computer's when the Bot has one).
pub fn definitions(computer: bool, google: bool) -> Value {
    let mut all = base_definitions();
    if computer {
        all.as_array_mut().unwrap().extend(computer_definitions().as_array().unwrap().iter().cloned());
    }
    if google {
        all.as_array_mut().unwrap().extend(google_definitions().as_array().unwrap().iter().cloned());
    }
    all
}

/// Skills (always) and handing a task to another Bot (when there's another).
pub fn team_definitions(others: bool) -> Value {
    let s = |d: &str| json!({"type": "string", "description": d});
    let mut all = vec![
        function("use_skill", "Get the steps of one of the saved skills (listed in your instructions), to follow them.", json!({"name": s("The skill's name, e.g. check-bill.")}), &["name"]),
        function(
            "save_skill",
            "Propose a skill: a saved how-to for a task that comes up again, usable by all the user's Bots (and by the user as /name). The user approves it first. Give an existing skill's name to propose a better version of it.",
            json!({
                "name": s("A short name in lowercase with dashes, e.g. check-bill."),
                "about": s("One line: what it's for."),
                "steps": s("Numbered steps that work again next time: inputs that change each time, where to go, what to do, what to check, what to report, and what to do when a step fails.")
            }),
            &["name", "about", "steps"],
        ),
    ];
    if others {
        all.push(function(
            "ask_bot",
            "Hand a task to another of the user's Bots (by name) and wait for its answer. It works in its own chat with its own tools and memory. Use it when the task is clearly that Bot's job.",
            json!({"bot": s("The other Bot's name."), "task": s("What it should do, with everything it needs to know (it hasn't seen this chat).")}),
            &["bot", "task"],
        ));
    }
    Value::Array(all)
}

pub fn phone_definitions() -> Value {
    let s = |d: &str| json!({"type": "string", "description": d});
    json!([
        function("phone_remind", "Set a reminder on the user's phone: it rings at that time with your text.", json!({"at": s("When, in the user's local time, like 2026-11-10T18:00."), "text": s("What to remind them of, in a few words.")}), &["at", "text"]),
        function("phone_notify", "Send a notification to the user's phone now (for something they should see soon).", json!({"text": s("What it says."), "url": s("Optional link to open when tapped.")}), &["text"]),
        function("phone_copy", "Put text on the user's phone clipboard (a code, an address, a message to paste).", json!({"text": s("The text.")}), &["text"]),
        function("phone_open", "Open a web page on the user's phone.", json!({"url": s("The address."), "why": s("Optional: a few words on what it is.")}), &["url"]),
        function("phone_location", "Ask the user's phone where it is (only answered while the app is open on it).", json!({}), &[]),
        function("phone_status", "Ask the user's phone for its battery, charging, network and time zone (only answered while the app is open on it).", json!({}), &[]),
    ])
}

fn google_definitions() -> Value {
    let s = |d: &str| json!({"type": "string", "description": d});
    let max = json!({"type": "integer", "description": "How many (default 10, at most 25)."});
    let when = "Local time like 2026-11-10T09:00, or 2026-11-10 for an all-day event.";
    json!([
        function("gmail_search", "Search the user's Gmail (Gmail search syntax: from:, to:, subject:, is:unread, newer_than:2d, has:attachment …). Gives ids, senders, subjects and snippets.", json!({"query": s("Gmail search, e.g. \"is:unread newer_than:1d\"."), "max": max}), &["query"]),
        function("gmail_read", "Read one email in full, by the id gmail_search gave.", json!({"id": s("The email's id.")}), &["id"]),
        function("gmail_send", "Send an email from the user's Gmail (the user is asked first). To reply, give reply_to_id: it goes in the same thread.", json!({"to": s("Address(es), comma-separated."), "cc": s("Optional."), "subject": s("Subject (a reply can leave it out)."), "body": s("Plain text."), "reply_to_id": s("Optional: the id of the email this answers.")}), &["to", "body"]),
        function("gmail_draft", "Save an email as a draft in the user's Gmail without sending it.", json!({"to": s("Address(es), comma-separated."), "cc": s("Optional."), "subject": s("Subject."), "body": s("Plain text."), "reply_to_id": s("Optional: the id of the email this answers.")}), &["to", "body"]),
        function("calendar_events", "List events in the user's Google Calendar between two times (default: the next 7 days), optionally matching words.", json!({"from": s(when), "to": s(when), "query": s("Optional words to match."), "max": max}), &[]),
        function("calendar_create", "Add an event to the user's Google Calendar (the user is asked first).", json!({"title": s("What it is."), "start": s(when), "end": s(when), "location": s("Optional."), "description": s("Optional."), "attendees": s("Optional addresses to invite, comma-separated (they get an email).")}), &["title", "start", "end"]),
        function("calendar_update", "Change an event (only the fields given; the user is asked first).", json!({"id": s("The event's id from calendar_events."), "title": s("Optional."), "start": s(when), "end": s(when), "location": s("Optional."), "description": s("Optional.")}), &["id"]),
        function("calendar_delete", "Delete an event (the user is asked first).", json!({"id": s("The event's id from calendar_events.")}), &["id"]),
        function("drive_search", "Search the user's Google Drive by name and content.", json!({"query": s("Words to find."), "max": max}), &["query"]),
        function("drive_read", "Read a Drive file as text (Google Docs as text, Sheets as CSV, text files as they are).", json!({"id": s("The file's id from drive_search.")}), &["id"]),
        function("drive_download", "Copy a Drive file into your workspace (Docs as .docx, Sheets as .xlsx, Slides as .pdf), to work on it there.", json!({"id": s("The file's id."), "path": s("Optional workspace path; default drive/<name>.")}), &["id"]),
        function("drive_create", "Create a new file in the user's Google Drive (a Google Doc by default; the user is asked first). Or upload a workspace file with path.", json!({"name": s("The file's name."), "content": s("Its text (for a new Doc or text file)."), "path": s("Optional: a workspace file to upload instead."), "google_doc": {"type": "boolean", "description": "Make it a Google Doc (default true; ignored for uploads)."}}), &["name"]),
    ])
}

fn computer_definitions() -> Value {
    json!([
        function(
            "run_command",
            "Run a bash command on your computer, in /workspace (your workspace). Returns its output and exit code. Use it for scripts, data, files, git, pip.",
            json!({"command": {"type": "string", "description": "The bash command (or a short script)."}, "timeout": {"type": "integer", "description": "Seconds before it's stopped (default 120, up to 600)."}}),
            &["command"],
        ),
        function("browser_open", "Open a web page in your computer's browser. Returns the page's text and its numbered elements (links, buttons, fields).", json!({"url": {"type": "string", "description": "The page's address."}}), &["url"]),
        function("browser_click", "Click an element of the open page by its number.", json!({"element": {"type": "integer", "description": "The element's number from the page you were shown last."}}), &["element"]),
        function(
            "browser_type",
            "Type into a field of the open page (replacing what's in it), optionally pressing Enter.",
            json!({"element": {"type": "integer", "description": "The field's number."}, "text": {"type": "string", "description": "What to type."}, "submit": {"type": "boolean", "description": "Press Enter afterwards (e.g. to search)."}}),
            &["element", "text"],
        ),
        function("browser_scroll", "Scroll the open page to see more of it.", json!({"direction": {"type": "string", "enum": ["down", "up"]}}), &[]),
        function("browser_back", "Go back to the previous page.", json!({}), &[]),
    ])
}

fn base_definitions() -> Value {
    json!([
        function("web_search", "Search the web. Returns the top results with their links and a snippet each; read a page for the details.", json!({"query": {"type": "string", "description": "What to search for, as you'd type it into a search engine."}}), &["query"]),
        function("read_page", "Read a public web page as plain text (the first few thousand characters).", json!({"url": {"type": "string", "description": "The page's full http(s) address."}}), &["url"]),
        function("list_files", "List the files in your workspace folder (shared with the user's other Bots), or in a folder inside it.", json!({"path": {"type": "string", "description": "A folder inside the workspace; leave empty for the whole workspace."}}), &[]),
        function("read_file", "Read a text file from your workspace folder.", json!({"path": {"type": "string", "description": "The file's path inside the workspace, e.g. notes/plan.md."}}), &["path"]),
        function("write_file", "Create or replace a text file in your workspace folder (folders are made as needed).", json!({"path": {"type": "string", "description": "The file's path inside the workspace, e.g. reports/2026-11-08.md."}, "content": {"type": "string", "description": "The whole new content of the file."}}), &["path", "content"]),
        function("remember", "Save a lasting note to your memory (a preference, rule or fact about the user or your job). You see your memory at the start of every chat.", json!({"note": {"type": "string", "description": "One short, self-contained note."}}), &["note"]),
        function("forget", "Remove notes from your memory that contain the given text (when something changed or was wrong).", json!({"text": {"type": "string", "description": "Text the notes to remove contain."}}), &["text"]),
        function(
            "schedule_routine",
            "Propose a routine: a task you'll be started on at set times, e.g. a news digest every morning. It runs once the user approves it on their phone.",
            json!({
                "name": {"type": "string", "description": "A short name, e.g. Morning news."},
                "task": {"type": "string", "description": "What to do each time, written as an instruction to yourself."},
                "kind": {"type": "string", "enum": ["daily", "weekdays", "every", "once"], "description": "daily (at a time, on all or some days), weekdays, every (every few hours), or once."},
                "time": {"type": "string", "description": "For daily, weekdays and once: the local time, HH:MM (24 h)."},
                "days": {"type": "string", "description": "For daily: the days, e.g. mon,wed,fri (leave empty for every day)."},
                "hours": {"type": "number", "description": "For every: hours between runs (at least 0.5)."},
                "date": {"type": "string", "description": "For once: the date, YYYY-MM-DD."}
            }),
            &["name", "task", "kind"],
        ),
        function("cancel_routine", "Withdraw a routine you proposed, or switch off one of your routines.", json!({"name": {"type": "string", "description": "The routine's name."}}), &["name"]),
    ])
}

/// Runs one call. Bad arguments come back as an error the model can read and correct.
pub fn run(ctx: &Ctx, name: &str, args: &str) -> Outcome {
    let a: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
    let s = |k: &str| a[k].as_str().unwrap_or("").to_string();
    match name {
        "web_search" => web_search(ctx, &s("query")),
        "read_page" => read_page(&s("url")),
        "list_files" => list_files(ctx.workspace, &s("path")),
        "read_file" => read_file(ctx.workspace, &s("path")),
        "write_file" => match a["content"].as_str() {
            Some(c) => write_file(ctx.workspace, &s("path"), c),
            None => fail("write_file needs the file's content."),
        },
        "remember" => remember(ctx.memory, &s("note")),
        "forget" => forget(ctx.memory, &s("text")),
        other => fail(format!("There's no tool called {other:?}.")),
    }
}

fn web_search(ctx: &Ctx, query: &str) -> Outcome {
    let query = query.trim();
    if query.is_empty() {
        return fail("web_search needs a query.");
    }
    let hits = crate::search::search(ctx.searxng, query);
    if hits.is_empty() {
        return fail(format!("No results for \"{query}\" (or the search engine on the PC isn't running)."));
    }
    let text = hits.iter().enumerate().map(|(i, h)| format!("{}. {}\n   {}\n   {}", i + 1, h.title, h.url, h.snippet)).collect::<Vec<_>>().join("\n");
    Outcome { text: format!("Results for \"{query}\":\n{text}"), citations: hits.into_iter().map(|h| h.url).collect(), ok: true }
}

fn read_page(url: &str) -> Outcome {
    let url = url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return fail("read_page needs a full http(s) address.");
    }
    let text = crate::search::read_page(&crate::search::page_agent(), url);
    if text.trim().is_empty() {
        return fail(format!("Couldn't read {url}: it may need a login, block automated readers, not be a text page, or not be public."));
    }
    Outcome { text: format!("Text of {url}:\n{text}"), citations: vec![url.to_string()], ok: true }
}

/// A path inside the workspace, or why not: relative, no `..`, and (following symlinks) still inside.
pub fn safe_path(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
    // The model may call it /workspace (that's how it's described in other agents).
    let rel = rel.strip_prefix("/workspace").map(|r| r.trim_start_matches('/')).unwrap_or(rel);
    let p = Path::new(rel);
    if p.is_absolute() {
        return Err("Use a path inside your workspace, like notes/plan.md.".into());
    }
    if !p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir)) {
        return Err("Paths stay inside your workspace (no \"..\").".into());
    }
    let root_c = root.canonicalize().map_err(|e| format!("The workspace folder is missing: {e}"))?;
    let full = root_c.join(p);
    let mut probe = full.as_path();
    while !probe.exists() {
        probe = probe.parent().unwrap_or(&root_c);
    }
    let real = probe.canonicalize().map_err(|e| e.to_string())?;
    if !real.starts_with(&root_c) {
        return Err("That path leads outside your workspace.".into());
    }
    Ok(full)
}

/// A file the reply says it saved that isn't in the workspace. Small models often say "I saved it to
/// notes/x.md" without ever calling write_file; the run then asks once for the real call.
pub fn claimed_unsaved(text: &str, workspace: &Path) -> Option<String> {
    static SENTENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?]\s+|\n").unwrap());
    static SAVE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\b(saved|save|saving|written|wrote|write|created|stored|updated|put)\b").unwrap());
    static PATH: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)(?:^|[\s`'\x22(\[])(/?[\w][\w.\-/]*\.(?:md|markdown|txt|csv|tsv|json|html?|xml|ya?ml|toml|ini|py|js|ts|rs|sh|log))\b").unwrap()
    });
    for sentence in SENTENCE.split(text) {
        if !SAVE.is_match(sentence) {
            continue;
        }
        for c in PATH.captures_iter(sentence) {
            let path = c[1].to_string();
            if let Ok(full) = safe_path(workspace, &path)
                && !full.exists()
            {
                return Some(path);
            }
        }
    }
    None
}

fn list_files(root: &Path, rel: &str) -> Outcome {
    let dir = match safe_path(root, rel) {
        Ok(d) => d,
        Err(e) => return fail(e),
    };
    if !dir.is_dir() {
        return fail(format!("There's no folder {rel:?} in your workspace."));
    }
    let root_c = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut out = vec![];
    let mut stack = vec![(dir, 0)];
    while let Some((d, depth)) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&d).into_iter().flatten().flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            if out.len() >= LIST_ENTRIES {
                out.push(format!("… (more than {LIST_ENTRIES} entries)"));
                return ok(out.join("\n"));
            }
            let path = e.path();
            let shown = path.strip_prefix(&root_c).unwrap_or(&path).display().to_string();
            // Symlinks aren't followed: they could point anywhere.
            let ft = e.file_type().ok();
            if ft.is_some_and(|t| t.is_dir()) {
                out.push(format!("{shown}/"));
                if depth < 3 {
                    stack.push((path, depth + 1));
                }
            } else if ft.is_some_and(|t| t.is_file()) {
                out.push(format!("{shown} ({} bytes)", e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    if out.is_empty() { ok("The folder is empty.") } else { ok(out.join("\n")) }
}

fn read_file(root: &Path, rel: &str) -> Outcome {
    let file = match safe_path(root, rel) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    if !file.is_file() {
        return fail(format!("There's no file {rel:?} in your workspace."));
    }
    match std::fs::read(&file) {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => {
                let n = text.chars().count();
                let shown: String = text.chars().take(READ_FILE_CHARS).collect();
                if n > READ_FILE_CHARS { ok(format!("{shown}\n[… {} more characters not shown]", n - READ_FILE_CHARS)) } else { ok(shown) }
            }
            Err(e) => fail(format!("{rel} isn't a text file ({} bytes).", e.as_bytes().len())),
        },
        Err(e) => fail(format!("Couldn't read {rel}: {e}")),
    }
}

fn write_file(root: &Path, rel: &str, content: &str) -> Outcome {
    if rel.trim().is_empty() {
        return fail("write_file needs a path.");
    }
    if content.len() > WRITE_FILE_BYTES {
        return fail(format!("That's too big for one file (up to {WRITE_FILE_BYTES} bytes)."));
    }
    let file = match safe_path(root, rel) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    if file.is_dir() {
        return fail(format!("{rel} is a folder."));
    }
    if let Some(d) = file.parent()
        && let Err(e) = std::fs::create_dir_all(d)
    {
        return fail(format!("Couldn't make the folder: {e}"));
    }
    // Made again after creating folders, in case one of them was swapped for a link meanwhile.
    if let Err(e) = safe_path(root, rel) {
        return fail(e);
    }
    match std::fs::write(&file, content) {
        Ok(()) => ok(format!("Saved {} ({} bytes).", rel.trim(), content.len())),
        Err(e) => fail(format!("Couldn't save {rel}: {e}")),
    }
}

fn remember(memory: &Path, note: &str) -> Outcome {
    let note = note.trim().replace('\n', " ");
    if note.is_empty() {
        return fail("remember needs a note.");
    }
    let now = std::fs::read_to_string(memory).unwrap_or_default();
    if now.lines().any(|l| l.trim_start_matches("- ").trim() == note) {
        return ok("Already in your memory.");
    }
    let next = format!("{}{}- {note}\n", now, if now.is_empty() || now.ends_with('\n') { "" } else { "\n" });
    if next.chars().count() > super::MEMORY_CHARS {
        return fail("Your memory is full: forget notes that no longer matter first.");
    }
    match super::write_private(memory, next.as_bytes()) {
        Ok(()) => ok("Saved to your memory."),
        Err(e) => fail(format!("Couldn't save to memory: {e}")),
    }
}

fn forget(memory: &Path, text: &str) -> Outcome {
    let text = text.trim().to_lowercase();
    if text.is_empty() {
        return fail("forget needs the text of the notes to remove.");
    }
    let now = std::fs::read_to_string(memory).unwrap_or_default();
    let kept: Vec<&str> = now.lines().filter(|l| !l.to_lowercase().contains(&text)).collect();
    let removed = now.lines().count() - kept.len();
    if removed == 0 {
        return ok("No note in your memory contains that.");
    }
    let next = if kept.is_empty() { String::new() } else { kept.join("\n") + "\n" };
    match super::write_private(memory, next.as_bytes()) {
        Ok(()) => ok(format!("Removed {removed} note{} from your memory.", if removed == 1 { "" } else { "s" })),
        Err(e) => fail(format!("Couldn't change memory: {e}")),
    }
}
