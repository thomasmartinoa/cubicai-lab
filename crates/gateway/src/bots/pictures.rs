//! Bots making pictures with the PC's ComfyUI: the same generators and queue as the phone's Create
//! sheet (one job at a time; a job frees Ollama's VRAM first). The picture is saved in the workspace
//! (pictures/ by default) and kept beside the tool step, so the phone shows it in the chat. Editing
//! takes a picture from the workspace.

use super::{tools, Bot, Bots, Run};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// The fast generators first: a Bot's picture is usually one step of a bigger task.
const CREATE: [&str; 4] = ["z-image", "flux2-klein", "flux3-klein", "qwen-uncensored"];
const EDIT: [&str; 2] = ["flux2-edit", "qwen-edit"];
/// Longer than any of them takes (a cold ComfyUI start included).
const WAIT: Duration = Duration::from_secs(10 * 60);
const EDIT_INPUT_BYTES: u64 = 20 * 1024 * 1024;

pub fn definition(edit: bool) -> Value {
    let s = |d: &str| json!({"type": "string", "description": d});
    let mut props = json!({
        "prompt": s("What the picture shows, in English: subject, setting, style, light, mood. Write it for an image model (a vivid description, not a request)."),
        "shape": {"type": "string", "enum": ["square", "portrait", "landscape"], "description": "Default square."},
        "save_as": s("Optional workspace path for it, like pictures/logo.png (default pictures/<words of the prompt>.png)."),
    });
    if edit {
        props["edit"] = s("Optional: a picture in your workspace to change instead of making a new one (then the prompt says what to change).");
    }
    json!({"type": "function", "function": {
        "name": "make_picture",
        "description": "Make a picture on the user's PC (about 10-40 s) and save it in your workspace; the user sees it in the chat. For logos, illustrations, mock-ups, social posts, ideas.",
        "parameters": {"type": "object", "properties": props, "required": ["prompt"]},
    }})
}

/// A picture file's name from the prompt's first words.
fn file_name(prompt: &str) -> String {
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(5)
        .map(|w| w.to_lowercase())
        .collect();
    if words.is_empty() { "picture".into() } else { words.join("-") }
}

impl Bots {
    /// The generators a Bot would use now: (to make, to edit). None without the PC's ComfyUI.
    pub(super) fn picture_generators(&self) -> Option<(Option<String>, Option<String>)> {
        let m = self.media.as_ref()?;
        let list = m.catalog.list(m.comfy.known_info().as_ref());
        let pick = |ids: &[&str], mode: &str| {
            ids.iter()
                .find(|id| list.iter().any(|g| g["id"] == **id))
                .map(|s| s.to_string())
                .or_else(|| list.iter().find(|g| g["kind"] == "image" && g["mode"] == mode).and_then(|g| g["id"].as_str().map(String::from)))
        };
        let (make, edit) = (pick(&CREATE, "create"), pick(&EDIT, "edit"));
        (make.is_some() || edit.is_some()).then_some((make, edit))
    }

    /// make_picture: submits the job, waits for it (Stop cancels it), saves the picture in the
    /// workspace and beside the tool step `tmsg_id`. True when the phone has a picture to show.
    pub(super) fn picture_tool(&self, bot: &Bot, run: &Run, args: &str, tmsg_id: &str) -> (tools::Outcome, bool) {
        let out = |text: String, ok: bool| (tools::Outcome { text, citations: vec![], ok }, false);
        let a: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
        let (Some(m), Some((make, edit))) = (self.media.as_ref(), self.picture_generators()) else {
            return out("Making pictures isn't set up on this PC (no ComfyUI image models).".into(), false);
        };
        let prompt = a["prompt"].as_str().unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return out("Say what the picture shows (prompt).".into(), false);
        }
        let ws = &self.paths.workspace;
        // An edit reads its picture from the workspace (never outside it).
        let source = match a["edit"].as_str().map(str::trim).filter(|p| !p.is_empty()) {
            None => None,
            Some(p) => {
                let Some(_) = &edit else { return out("There's no picture-editing model on this PC; make a new picture instead.".into(), false) };
                let path = match tools::safe_path(ws, p) {
                    Ok(p) => p,
                    Err(e) => return out(e, false),
                };
                match std::fs::metadata(&path) {
                    Ok(md) if md.is_file() && md.len() <= EDIT_INPUT_BYTES => {}
                    Ok(_) => return out(format!("{p} isn't a picture I can edit (a file up to 20 MB)."), false),
                    Err(_) => return out(format!("There's no {p} in the workspace."), false),
                }
                match std::fs::read(&path) {
                    Ok(b) if crate::media::kind(&crate::media::suffix(&b, "")) == "image" => Some(b),
                    Ok(_) => return out(format!("{p} isn't a picture."), false),
                    Err(e) => return out(format!("Couldn't read {p}: {e}"), false),
                }
            }
        };
        let gen_id = if source.is_some() { edit.clone() } else { make.clone() };
        let Some(gen_id) = gen_id else { return out("There's no picture-making model on this PC, only one for editing: give a workspace picture to edit.".into(), false) };
        let aspect = match a["shape"].as_str().unwrap_or("square") {
            "portrait" => "3:4",
            "landscape" => "4:3",
            _ => "1:1",
        };
        let job = match m.submit(&gen_id, json!({"prompt": prompt, "aspect": aspect}), source, None, &format!("bot {}", bot.id)) {
            Ok(j) => j,
            Err(e) => return out(e, false),
        };
        crate::log(&format!("bot {} made picture job {} ({gen_id})", bot.id, job.id));
        // Wait for it, waking for Stop.
        let end = Instant::now() + WAIT;
        let mut st = job.st.lock().unwrap();
        while !crate::jobs::TERMINAL.contains(&st.state.as_str()) {
            if run.stop.load(Ordering::SeqCst) || Instant::now() > end {
                drop(st);
                let stopped = run.stop.load(Ordering::SeqCst);
                m.cancel(&job.id);
                return out(if stopped { "Not done: the user stopped you.".into() } else { "The picture took too long; it was given up.".into() }, false);
            }
            st = job.cond.wait_timeout(st, Duration::from_millis(500)).unwrap().0;
        }
        let (state, error, file) = (st.state.clone(), st.error.clone(), st.files.iter().find(|f| f.kind == "image").cloned());
        drop(st);
        let Some(file) = file.filter(|_| state == "done") else {
            m.forget(&job);
            return out(format!("The picture failed: {}", error.unwrap_or_else(|| state.clone())), false);
        };
        // Into the workspace: the name asked for, else pictures/<prompt words>.png, never over a file.
        let ext = std::path::Path::new(&file.name).extension().and_then(|e| e.to_str()).unwrap_or("png").to_string();
        let wanted = a["save_as"].as_str().map(str::trim).filter(|p| !p.is_empty()).map(String::from);
        let base = wanted.clone().unwrap_or_else(|| format!("pictures/{}.{ext}", file_name(&prompt)));
        let (stem, dot_ext) = match base.rsplit_once('.') {
            Some((s, e)) if !s.is_empty() && !e.contains('/') => (s.to_string(), format!(".{e}")),
            _ => (base.clone(), format!(".{ext}")),
        };
        let mut rel = format!("{stem}{dot_ext}");
        let mut n = 2;
        let target = loop {
            let path = match tools::safe_path(ws, &rel) {
                Ok(p) => p,
                Err(e) => {
                    m.forget(&job);
                    return out(e, false);
                }
            };
            if !path.exists() {
                break path;
            }
            rel = format!("{stem}-{n}{dot_ext}");
            n += 1;
        };
        let saved = target.parent().is_some_and(|d| std::fs::create_dir_all(d).is_ok())
            // Made the folders: check again that it's still inside the workspace.
            && tools::safe_path(ws, &rel).is_ok()
            && std::fs::copy(&file.path, &target).is_ok();
        let shown = saved && {
            let tmp = target.with_extension("shot");
            std::fs::copy(&target, &tmp).is_ok() && self.keep_shot(&bot.id, &tmp, tmsg_id)
        };
        m.forget(&job);
        if !saved {
            return out(format!("The picture was made but couldn't be saved as {rel}."), false);
        }
        let how = if a["edit"].as_str().is_some_and(|p| !p.trim().is_empty()) { "Edited the picture" } else { "Made the picture" };
        (tools::Outcome { text: format!("{how} and saved it as {rel} in the workspace. The user sees it in the chat; don't describe it at length."), citations: vec![], ok: true }, shown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_come_from_the_prompt() {
        assert_eq!(file_name("A cosy tea shop in Munnar, at dawn, watercolour"), "a-cosy-tea-shop-in");
        assert_eq!(file_name("!!!"), "picture");
    }
}
