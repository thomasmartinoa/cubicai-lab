//! Ollama on this PC: model capabilities and context, the model list the phone sees, and the
//! conversion between the phone's OpenAI-style messages and Ollama's /api/chat.

use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelInfo {
    /// Ollama's capabilities: completion, vision, tools, thinking…
    pub caps: HashSet<String>,
    /// num_ctx set in the Modelfile, which wins over everything.
    pub own_ctx: Option<u64>,
    /// The longest context the model supports (0 if unknown).
    pub longest: u64,
}

impl ModelInfo {
    pub fn has(&self, cap: &str) -> bool {
        self.caps.contains(cap)
    }

    /// From /api/show's reply.
    pub fn from_show(d: &Value) -> ModelInfo {
        let caps = d["capabilities"].as_array().into_iter().flatten().filter_map(|c| c.as_str().map(String::from)).collect();
        let own_ctx = d["parameters"].as_str().unwrap_or("").lines().find_map(|l| {
            let rest = l.strip_prefix("num_ctx")?;
            rest.starts_with(char::is_whitespace).then(|| rest.trim().parse().ok())?
        });
        let longest = d["model_info"].as_object().into_iter().flatten().filter(|(k, _)| k.ends_with(".context_length")).filter_map(|(_, v)| v.as_u64()).max().unwrap_or(0);
        ModelInfo { caps, own_ctx, longest }
    }

    /// The context a request gets: the Modelfile's num_ctx if it sets one, else the configured one,
    /// capped at what the model supports.
    pub fn context(&self, wanted: u64) -> u64 {
        match self.own_ctx {
            Some(c) => c,
            None if self.longest > 0 => wanted.min(self.longest),
            None => wanted,
        }
    }
}

pub struct Ollama {
    pub url: String,
    quick: ureq::Agent,
    long: ureq::Agent,
    info: Mutex<HashMap<String, ModelInfo>>,
}

impl Ollama {
    pub fn new(url: &str) -> Ollama {
        let quick = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(30))).http_status_as_error(false).proxy(None).build().new_agent();
        // Replies stream for as long as they take; only the wait for the first byte is limited
        // (loading a big model can take minutes).
        let long = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_recv_response(Some(Duration::from_secs(900)))
            .http_status_as_error(false)
            .proxy(None)
            .build()
            .new_agent();
        Ollama { url: url.trim_end_matches('/').to_string(), quick, long, info: Mutex::new(HashMap::new()) }
    }

    fn get(&self, path: &str) -> Result<Value, String> {
        let mut r = self.quick.get(format!("{}{path}", self.url)).call().map_err(|e| e.to_string())?;
        if !r.status().is_success() {
            return Err(format!("HTTP {}", r.status().as_u16()));
        }
        r.body_mut().read_json().map_err(|e| e.to_string())
    }

    /// POSTs JSON. Ollama's own error message (`{"error": …}`) comes back as Err.
    pub fn post(&self, path: &str, body: &Value, long: bool) -> Result<ureq::http::Response<ureq::Body>, String> {
        let a = if long { &self.long } else { &self.quick };
        let mut r = a.post(format!("{}{path}", self.url)).send_json(body).map_err(|e| e.to_string())?;
        if !r.status().is_success() {
            let code = r.status().as_u16();
            let text = r.body_mut().read_to_string().unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&text).ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or_else(|| format!("HTTP {code}"));
            return Err(detail);
        }
        Ok(r)
    }

    /// Capabilities and context of a model, kept once known (a failure isn't kept, so a model asked
    /// about while Ollama was down is asked again).
    pub fn info(&self, name: &str) -> ModelInfo {
        if let Some(i) = self.info.lock().unwrap().get(name) {
            return i.clone();
        }
        let Ok(mut r) = self.post("/api/show", &json!({"model": name}), false) else { return ModelInfo::default() };
        let Ok(d) = r.body_mut().read_json::<Value>() else { return ModelInfo::default() };
        let i = ModelInfo::from_show(&d);
        self.info.lock().unwrap().insert(name.to_string(), i.clone());
        i
    }

    pub fn tags(&self) -> Result<Vec<(String, u64)>, String> {
        let v = self.get("/api/tags")?;
        Ok(v["models"].as_array().into_iter().flatten().filter_map(|m| Some((m["name"].as_str()?.to_string(), m["size"].as_u64().unwrap_or(0)))).collect())
    }

    /// Models in memory now.
    pub fn loaded(&self) -> HashSet<String> {
        self.get("/api/ps").map(|v| v["models"].as_array().into_iter().flatten().filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default()
    }

    /// The phone's model list (`/v1/models` data).
    pub fn list_models(&self, wanted_ctx: u64, vram: Option<u64>) -> Result<Vec<Value>, String> {
        let mut tags = self.tags()?;
        tags.sort();
        let models: Vec<(String, u64, ModelInfo)> = tags.into_iter().map(|(n, s)| {
            let i = self.info(&n);
            (n, s, i)
        }).collect();
        Ok(model_entries(&models, wanted_ctx, vram))
    }
}

/// Builds the list: chat models only, with modalities, context and tools; the biggest vision model
/// that fits in VRAM is recommended (smart enough, still fast).
pub fn model_entries(models: &[(String, u64, ModelInfo)], wanted_ctx: u64, vram: Option<u64>) -> Vec<Value> {
    let mut data: Vec<Value> = vec![];
    for (name, size, info) in models {
        if !info.caps.is_empty() && !info.has("completion") {
            continue; // embedding models etc.
        }
        let mut modalities = vec!["text"];
        if info.has("vision") {
            modalities.push("image");
        }
        data.push(json!({
            "id": name,
            "object": "model",
            "owned_by": "local",
            "size": size,
            "input_modalities": modalities,
            "context_length": info.context(wanted_ctx),
            "supported_parameters": if info.has("tools") { vec!["tools"] } else { vec![] },
        }));
    }
    let budget = vram.unwrap_or(8 << 30);
    let size = |m: &Value| m["size"].as_u64().unwrap_or(0);
    let vision: Vec<usize> = (0..data.len()).filter(|&i| data[i]["input_modalities"].as_array().unwrap().iter().any(|x| x == "image") && size(&data[i]) > 0).collect();
    let fits: Vec<usize> = vision.iter().copied().filter(|&i| size(&data[i]) <= budget).collect();
    // Ties go to the first, as Python's max/min do.
    let pick = if fits.is_empty() {
        vision.iter().copied().reduce(|a, b| if size(&data[b]) < size(&data[a]) { b } else { a })
    } else {
        fits.iter().copied().reduce(|a, b| if size(&data[b]) > size(&data[a]) { b } else { a })
    };
    if let Some(i) = pick {
        data[i]["cubic_recommended"] = true.into();
    }
    data
}

/// OpenAI tool_calls (arguments as a JSON string) → Ollama's (arguments as an object).
pub fn native_tool_calls(calls: &Value) -> Vec<Value> {
    calls
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| {
            let f = &c["function"];
            let args = match &f["arguments"] {
                Value::String(s) => serde_json::from_str::<Value>(if s.is_empty() { "{}" } else { s }).unwrap_or_else(|_| json!({})),
                Value::Null => json!({}),
                v => v.clone(),
            };
            let args = if args.is_null() { json!({}) } else { args };
            json!({"function": {"name": f["name"].as_str().unwrap_or(""), "arguments": args}})
        })
        .collect()
}

/// Ollama's tool calls (whole, arguments as objects) → OpenAI deltas with string arguments and ids.
pub fn openai_tool_calls(calls: &Value, rid: u64) -> Value {
    Value::Array(
        calls
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(n, c)| {
                let f = &c["function"];
                let args = if f["arguments"].is_null() { json!({}) } else { f["arguments"].clone() };
                json!({
                    "index": n, "id": format!("call_{rid}_{n}"), "type": "function",
                    "function": {"name": f["name"].as_str().unwrap_or(""), "arguments": python_dumps(&args)},
                })
            })
            .collect(),
    )
}

/// JSON the way Python's json.dumps writes it (", " and ": " separators, non-ASCII escaped), so
/// arguments look the same as from the Python gateway.
pub fn python_dumps(v: &Value) -> String {
    let mut out = String::new();
    dump(v, &mut out);
    out
}

fn dump(v: &Value, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(k, out);
                out.push_str(": ");
                dump(x, out);
            }
            out.push('}');
        }
        Value::String(s) => dump_str(s, out),
        other => out.push_str(&other.to_string()),
    }
}

fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The phone's OpenAI-style messages → Ollama /api/chat messages (images as bare base64, tool calls
/// and results kept). Pictures for a model that can't see: an error if they're in the latest user
/// message, otherwise a note in their place.
pub fn to_native(messages: &[Value], vision: bool) -> Result<Vec<Value>, String> {
    let last_user = messages.iter().rposition(|m| m["role"] == "user");
    let mut out = vec![];
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().unwrap_or("");
        let content = &m["content"];
        if role == "tool" {
            out.push(json!({"role": "tool", "content": content.as_str().unwrap_or(""), "tool_name": m["name"].as_str().unwrap_or("")}));
            continue;
        }
        if content.is_string() || content.is_null() {
            let mut msg = Map::new();
            msg.insert("role".into(), role.into());
            msg.insert("content".into(), content.as_str().unwrap_or("").into());
            if m["tool_calls"].as_array().is_some_and(|a| !a.is_empty()) {
                msg.insert("tool_calls".into(), Value::Array(native_tool_calls(&m["tool_calls"])));
            }
            out.push(Value::Object(msg));
            continue;
        }
        let (mut texts, mut images) = (vec![], vec![]);
        for part in content.as_array().into_iter().flatten() {
            match part["type"].as_str() {
                Some("text") => texts.push(part["text"].as_str().unwrap_or("").to_string()),
                Some("image_url") => {
                    let url = part["image_url"]["url"].as_str().or(part["image_url"].as_str()).unwrap_or("");
                    if url.starts_with("data:")
                        && let Some((_, b64)) = url.split_once(',')
                    {
                        images.push(b64.to_string());
                    }
                }
                _ => {}
            }
        }
        let mut msg = json!({"role": role, "content": texts.join("\n")});
        if !images.is_empty() && vision {
            msg["images"] = images.into();
        } else if !images.is_empty() && Some(i) == last_user {
            return Err("This model can't see images. Pick one marked \"sees images\" (e.g. gemma4 or qwen3.5).".into());
        } else if !images.is_empty() {
            let c = msg["content"].as_str().unwrap_or("").to_string() + "\n[image omitted: current model has no vision]";
            msg["content"] = c.into();
        }
        out.push(msg);
    }
    Ok(out)
}

/// Total VRAM in bytes (NVIDIA only), or None.
pub fn gpu_memory() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]).output().ok()?;
    String::from_utf8_lossy(&out.stdout).split_whitespace().next()?.parse::<u64>().ok().map(|mb| mb * 1024 * 1024)
}
