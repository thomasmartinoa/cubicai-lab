//! What the gateway is doing, for a control panel: metadata only, never message content. The JSON
//! has the same shape as the Python gateway's `/v1/cubicai/status`.

use crate::time::now;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

pub struct Activity {
    started: f64,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    next: u64,
    active: Vec<Value>,
    recent: VecDeque<Value>,
    devices: HashMap<String, f64>,
    served: u64,
    blocked: u64,
}

const RECENT: usize = 200;

impl Default for Activity {
    fn default() -> Self {
        Activity { started: now(), inner: Mutex::new(Inner::default()) }
    }
}

impl Activity {
    pub fn begin(&self, ip: &str, model: &str, search: bool) -> u64 {
        let mut a = self.inner.lock().unwrap();
        a.next += 1;
        let id = a.next;
        a.active.push(json!({"id": id, "ip": ip, "model": model, "search": search, "phase": "Starting", "started": now()}));
        a.devices.insert(ip.to_string(), now());
        id
    }

    pub fn phase(&self, id: u64, text: &str) {
        let mut a = self.inner.lock().unwrap();
        if let Some(item) = a.active.iter_mut().find(|i| i["id"] == id) {
            item["phase"] = text.into();
        }
    }

    pub fn end(&self, id: u64, outcome: &str, tokens: u64, gen_seconds: f64) {
        let mut a = self.inner.lock().unwrap();
        let Some(pos) = a.active.iter().position(|i| i["id"] == id) else { return };
        let mut item = a.active.remove(pos);
        let tps = if gen_seconds > 0.0 { (tokens as f64 / gen_seconds * 10.0).round() / 10.0 } else { 0.0 };
        item["outcome"] = outcome.into();
        item["ended"] = now().into();
        item["tokens"] = tokens.into();
        item["tps"] = tps.into();
        if let Some(o) = item.as_object_mut() {
            o.remove("phase");
        }
        a.recent.push_front(item);
        a.recent.truncate(RECENT);
        a.served += 1;
    }

    pub fn reject(&self, ip: &str) {
        let mut a = self.inner.lock().unwrap();
        a.blocked += 1;
        let t = now();
        a.recent.push_front(json!({"ip": ip, "outcome": "blocked", "started": t, "ended": t}));
        a.recent.truncate(RECENT);
    }

    pub fn snapshot(&self) -> Value {
        let a = self.inner.lock().unwrap();
        let mut devices: Vec<(&String, &f64)> = a.devices.iter().collect();
        devices.sort_by(|x, y| y.1.total_cmp(x.1));
        json!({
            "uptime": now() - self.started,
            "served": a.served,
            "blocked": a.blocked,
            "active": a.active,
            "recent": a.recent,
            "devices": devices.iter().map(|(ip, t)| json!({"ip": ip, "last_seen": t})).collect::<Vec<_>>(),
        })
    }
}
