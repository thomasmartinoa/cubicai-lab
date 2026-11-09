//! The phone's Bot routes (all need the pairing token, like everything else):
//!
//! ```text
//! GET    /v1/bots                        the Bots, each with whether it's working and its last message
//! POST   /v1/bots                        create one {name, job, color, provider, model, memory?}
//! GET    /v1/bots/<id>                   profile + memory
//! PUT    /v1/bots/<id>                   change any of those
//! DELETE /v1/bots/<id>                   delete it (its workspace files stay)
//! GET    /v1/bots/<id>/messages          the chat, whether it's working, and where to follow from
//! POST   /v1/bots/<id>/messages          {text}: a message (starts a run, or joins the running one)
//! DELETE /v1/bots/<id>/messages          clear the chat (not while it works)
//! GET    /v1/bots/<id>/events?after=N    the run's events as server-sent events, from N
//! DELETE /v1/bots/<id>/run               stop the run
//! GET    /v1/bots/<id>/routines          its routines
//! POST   /v1/bots/<id>/routines          {name, task, schedule}: a new one (the user's: approved)
//! PUT    /v1/bots/<id>/routines/<rid>    change it, switch it on/off, {approved: true}
//! DELETE /v1/bots/<id>/routines/<rid>    delete it (or decline a proposed one)
//! POST   /v1/bots/<id>/routines/<rid>/run  run it now
//! GET    /v1/bots/feed?after=MS          run ends and approvals since MS (the phone's notifications)
//! GET    /v1/keys                        which cloud keys the PC has (never the keys)
//! PUT    /v1/keys                        {xai?, gemini?, openrouter?, custom?, custom_url?} ("" removes)
//! ```

use super::{Bots, Run};
use crate::http::{error, Conn, Request};
use crate::server::{Gateway, Timeouts};
use serde_json::{json, Value};
use std::io::{self, Read, Write};
use std::time::Duration;

/// Request bodies here are small (text only).
const MAX_BOT_BODY: usize = 1024 * 1024;

pub fn handles(path: &str) -> bool {
    path == "/v1/keys" || path == "/v1/bots" || path.starts_with("/v1/bots/")
}

fn body<S: Read + Write>(conn: &mut Conn<S>, req: &Request) -> Result<Value, (u16, String)> {
    let n = req.content_length().map_err(|_| (400, "Bad Content-Length".to_string()))?.unwrap_or(0);
    if n == 0 {
        return Ok(json!({}));
    }
    if n > MAX_BOT_BODY {
        return Err((413, "Request too large".into()));
    }
    let raw = conn.read_body(n).map_err(|e| (400, e.to_string()))?;
    serde_json::from_slice(&raw).map_err(|_| (400, "Invalid JSON".into()))
}

fn summary(bots: &Bots, b: &super::Bot) -> Value {
    let mut v = b.to_json();
    v["running"] = bots.running(&b.id).into();
    let routines = bots.routines(&b.id);
    v["routines"] = routines.iter().filter(|r| r.approved && r.enabled).count().into();
    v["pending"] = routines.iter().filter(|r| !r.approved).count().into();
    let last = bots.messages(&b.id).into_iter().rev().find(|m| m["role"] != "tool" && (m["text"].as_str().is_some_and(|t| !t.is_empty()) || m["error"].is_string()));
    v["last"] = last.map(|m| json!({"role": m["role"], "text": m["text"], "error": m["error"], "at": m["at"]})).unwrap_or(Value::Null);
    v
}

pub fn route<S: Read + Write + Timeouts>(gw: &Gateway, conn: &mut Conn<S>, req: &Request, path: &str) -> io::Result<bool> {
    let Some(bots) = &gw.bots else { return conn.json(req, 404, &error("Bots aren't set up on this PC.")) };
    let method = req.method.as_str();
    macro_rules! read_body {
        () => {
            match body(conn, req) {
                Ok(v) => v,
                Err((code, m)) => return conn.json(req, code, &error(&m)),
            }
        };
    }
    if path == "/v1/keys" {
        return match method {
            "GET" => conn.json(req, 200, &json!({"keys": bots.keys.present()})),
            "PUT" | "POST" => {
                let v = read_body!();
                match bots.keys.set(&v) {
                    Ok(()) => conn.json(req, 200, &json!({"keys": bots.keys.present()})),
                    Err(e) => conn.json(req, 400, &error(&e)),
                }
            }
            _ => conn.json(req, 404, &error("Not found")),
        };
    }
    if path == "/v1/bots/feed" {
        let after = req.query("after").and_then(|a| a.parse().ok()).unwrap_or(0);
        return match method {
            "GET" => {
                let (items, upto) = bots.feed(after);
                conn.json(req, 200, &json!({"items": items, "now": upto}))
            }
            _ => conn.json(req, 404, &error("Not found")),
        };
    }
    if path == "/v1/bots" {
        return match method {
            "GET" => conn.json(req, 200, &json!({"bots": bots.list().iter().map(|b| summary(bots, b)).collect::<Vec<_>>()})),
            "POST" => {
                let v = read_body!();
                match bots.create(&v) {
                    Ok(b) => {
                        crate::log(&format!("bot {} created ({} {})", b.id, b.provider, b.model));
                        conn.json(req, 200, &json!({"bot": summary(bots, &b), "memory": bots.memory(&b.id)}))
                    }
                    Err(e) => conn.json(req, 400, &error(&e)),
                }
            }
            _ => conn.json(req, 404, &error("Not found")),
        };
    }
    let rest = path.strip_prefix("/v1/bots/").unwrap_or("");
    let (id, sub) = rest.split_once('/').unwrap_or((rest, ""));
    let Some(mut bot) = bots.get(id) else { return conn.json(req, 404, &error("No such Bot (it may have been deleted).")) };
    match (method, sub) {
        ("GET", "") => conn.json(req, 200, &json!({"bot": summary(bots, &bot), "memory": bots.memory(id)})),
        ("PUT", "") => {
            let v = read_body!();
            if let Err(e) = bot.apply(&v) {
                return conn.json(req, 400, &error(&e));
            }
            if let Some(m) = v["memory"].as_str()
                && let Err(e) = bots.set_memory(id, m)
            {
                return conn.json(req, 400, &error(&e));
            }
            match bots.save(&bot) {
                Ok(()) => conn.json(req, 200, &json!({"bot": summary(bots, &bot), "memory": bots.memory(id)})),
                Err(e) => conn.json(req, 500, &error(&e)),
            }
        }
        ("DELETE", "") => {
            bots.delete(id);
            crate::log(&format!("bot {id} deleted"));
            conn.json(req, 200, &json!({"ok": true}))
        }
        ("GET", "messages") => {
            let run = bots.run(id);
            let running = bots.running(id);
            // A phone joining mid-run follows from the start of the current turn, so it sees all of
            // the words being written now; messages before that are in the list.
            let follow = run.as_ref().filter(|_| running).map(|r| r.resume_at());
            conn.json(req, 200, &json!({"messages": bots.messages(id), "running": running, "follow": follow}))
        }
        ("POST", "messages") => {
            let v = read_body!();
            match bots.send(id, v["text"].as_str().unwrap_or("")) {
                Ok(r) => conn.json(req, 200, &r),
                Err(e) => conn.json(req, 400, &error(&e)),
            }
        }
        ("DELETE", "messages") => {
            if bots.running(id) {
                return conn.json(req, 400, &error("Stop the Bot before clearing its chat."));
            }
            bots.clear(id);
            conn.json(req, 200, &json!({"ok": true}))
        }
        ("GET", "events") => match bots.run(id) {
            Some(run) => {
                let after = req.query("after").and_then(|a| a.parse().ok()).unwrap_or(0);
                events(conn, &run, after);
                Ok(false)
            }
            None => conn.json(req, 404, &error("The Bot isn't working on anything.")),
        },
        ("DELETE", "run") => conn.json(req, 200, &json!({"stopped": bots.stop(id)})),
        ("GET", "routines") => conn.json(req, 200, &routines_json(bots, id)),
        ("POST", "routines") => {
            let v = read_body!();
            match bots.add_routine(&bot, &v, "user") {
                Ok(_) => conn.json(req, 200, &routines_json(bots, id)),
                Err(e) => conn.json(req, 400, &error(&e)),
            }
        }
        (_, sub) if sub.starts_with("routines/") => {
            let rest = &sub["routines/".len()..];
            let (rid, action) = rest.split_once('/').unwrap_or((rest, ""));
            match (method, action) {
                ("PUT", "") => {
                    let v = read_body!();
                    match bots.update_routine(id, rid, &v) {
                        Ok(_) => conn.json(req, 200, &routines_json(bots, id)),
                        Err(e) => conn.json(req, 400, &error(&e)),
                    }
                }
                ("DELETE", "") => {
                    bots.delete_routine(id, rid);
                    conn.json(req, 200, &routines_json(bots, id))
                }
                ("POST", "run") => {
                    let Some(r) = bots.routines(id).into_iter().find(|r| r.id == rid) else {
                        return conn.json(req, 404, &error("No such routine (it may have been deleted)."));
                    };
                    if !r.approved {
                        return conn.json(req, 400, &error("Approve the routine first."));
                    }
                    match bots.run_routine(&bot, &r, None) {
                        Ok(v) => conn.json(req, 200, &v),
                        Err(e) => conn.json(req, 400, &error(&e)),
                    }
                }
                _ => conn.json(req, 404, &error("Not found")),
            }
        }
        _ => conn.json(req, 404, &error("Not found")),
    }
}

/// A run's events from `after`, as they happen, until it ends or the phone goes away (the run goes
/// on without it).
fn events<S: Read + Write + Timeouts>(conn: &mut Conn<S>, run: &Run, mut after: usize) {
    if conn.start_events().is_err() {
        return;
    }
    conn.stream().set_timeout(60);
    loop {
        let (new, over) = run.since(after, Duration::from_secs(15));
        let sent = if new.is_empty() && !over {
            // Keeps NAT and Tailscale paths open.
            conn.raw(b": still working\n\n")
        } else {
            new.iter().try_for_each(|e| conn.event(e))
        };
        after += new.len();
        if sent.is_err() || over {
            return;
        }
    }
}

fn routines_json(bots: &Bots, id: &str) -> Value {
    json!({"routines": bots.routines(id).iter().map(|r| r.to_json()).collect::<Vec<_>>()})
}
