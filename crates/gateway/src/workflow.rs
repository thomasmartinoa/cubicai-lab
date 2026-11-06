//! A ComfyUI workflow as its web UI saves it → the API format `/prompt` accepts. A port of
//! `comfy_workflow.py`: the saved format keeps widget values as positional lists and nests
//! subgraphs; the API wants one flat map of nodes with named inputs. Node definitions come from
//! ComfyUI's `/object_info` (whose key order matters, hence serde_json's preserve_order).

use serde_json::{json, Map, Value};
use std::collections::HashMap;

const SEED_CONTROLS: [&str; 4] = ["fixed", "randomize", "increment", "decrement"];
const WIDGET_TYPES: [&str; 6] = ["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"];
const UPLOADS: [&str; 3] = ["image_upload", "audio_upload", "video_upload"];
const FRONTEND_ONLY: [&str; 4] = ["Note", "MarkdownNote", "PrimitiveNode", "Reroute"];
const MUTED: i64 = 2;
const BYPASSED: i64 = 4;
const SUBGRAPH_IN: i64 = -10;
const SUBGRAPH_OUT: i64 = -20;

/// For each input name of a top-level subgraph, the (node id, input name) pairs it feeds, so
/// callers can set "prompt", "seed" and so on by name. Ordered like Python's dict.
pub type Named = Vec<(String, Vec<(String, String)>)>;

pub fn named_get<'a>(named: &'a Named, name: &str) -> &'a [(String, String)] {
    named.iter().find(|(n, _)| n == name).map(|(_, t)| t.as_slice()).unwrap_or(&[])
}

fn named_entry<'a>(named: &'a mut Named, name: &str) -> &'a mut Vec<(String, String)> {
    let i = match named.iter().position(|(n, _)| n == name) {
        Some(i) => i,
        None => {
            named.push((name.to_string(), vec![]));
            named.len() - 1
        }
    };
    &mut named[i].1
}

/// Where an input's value comes from: another node's output, or a plain value.
#[derive(Clone, Debug)]
enum Src {
    Node(String, Value),
    Value(Value),
}

#[derive(Clone, Copy)]
struct Link {
    origin_id: i64,
    origin_slot: i64,
    target_id: i64,
}

struct Outer {
    feeds: Vec<Option<Src>>,
    names: Vec<String>,
    targets: Named,
}

struct Instance {
    outputs: Vec<Option<Src>>,
    targets: Named,
}

struct Ctx<'a> {
    subgraphs: HashMap<String, &'a Value>,
    info: &'a Value,
    prompt: Map<String, Value>,
    named: Named,
}

struct Scope<'a> {
    nodes: &'a [Value],
    by_id: HashMap<String, &'a Value>,
    links: HashMap<i64, Link>,
    prefix: String,
    outer: Option<Outer>,
    instances: HashMap<String, Instance>,
}

/// A node or link id as Python's str() would write it.
fn id_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn int(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn links_of(raw: &Value) -> HashMap<i64, Link> {
    let mut out = HashMap::new();
    for l in raw.as_array().into_iter().flatten() {
        // [id, origin_id, origin_slot, target_id, target_slot, type] or an object.
        let (id, o, s, t) = if let Some(a) = l.as_array() {
            (a.first(), a.get(1), a.get(2), a.get(3))
        } else {
            (l.get("id"), l.get("origin_id"), l.get("origin_slot"), l.get("target_id"))
        };
        let Some(id) = id.and_then(int) else { continue };
        let g = |v: Option<&Value>| v.and_then(int).unwrap_or(i64::MIN);
        out.insert(id, Link { origin_id: g(o), origin_slot: g(s), target_id: g(t) });
    }
    out
}

/// Returns the API prompt (node id → {class_type, inputs, _meta}) and the named subgraph inputs.
pub fn to_api(workflow: &Value, object_info: &Value) -> Result<(Map<String, Value>, Named), String> {
    let mut ctx = Ctx {
        subgraphs: workflow["definitions"]["subgraphs"].as_array().into_iter().flatten().filter_map(|s| Some((s["id"].as_str()?.to_string(), s))).collect(),
        info: object_info,
        prompt: Map::new(),
        named: vec![],
    };
    let mut top = Scope::new(workflow["nodes"].as_array().map(Vec::as_slice).unwrap_or(&[]), links_of(&workflow["links"]), String::new(), None);
    top.expand(&mut ctx)?;
    Ok((ctx.prompt, ctx.named))
}

impl<'a> Scope<'a> {
    fn new(nodes: &'a [Value], links: HashMap<i64, Link>, prefix: String, outer: Option<Outer>) -> Scope<'a> {
        Scope { nodes, by_id: nodes.iter().map(|n| (id_str(&n["id"]), n)).collect(), links, prefix, outer, instances: HashMap::new() }
    }

    fn source(&mut self, ctx: &mut Ctx<'a>, link_id: &Value, depth: usize) -> Result<Option<Src>, String> {
        let Some(link) = int(link_id).and_then(|l| self.links.get(&l)).copied() else { return Ok(None) };
        if depth > 50 {
            return Ok(None);
        }
        let slot = link.origin_slot;
        if link.origin_id == SUBGRAPH_IN {
            return Ok(self.outer.as_ref().and_then(|o| usize::try_from(slot).ok().and_then(|s| o.feeds.get(s).cloned()).flatten()));
        }
        let Some(node) = self.by_id.get(&link.origin_id.to_string()).copied() else { return Ok(None) };
        let mode = int(&node["mode"]);
        if mode == Some(MUTED) {
            return Ok(None);
        }
        let kind = node["type"].as_str().unwrap_or("");
        if kind == "Reroute" || mode == Some(BYPASSED) {
            // Passes straight through: a reroute's only input, or a bypassed node's first input of the same type.
            let out_type = usize::try_from(slot).ok().and_then(|s| node["outputs"].as_array().and_then(|o| o.get(s))).map(|o| o["type"].clone()).unwrap_or(Value::Null);
            for i in node["inputs"].as_array().into_iter().flatten() {
                if !i["link"].is_null() && (kind == "Reroute" || i["type"] == out_type) {
                    return self.source(ctx, &i["link"], depth + 1);
                }
            }
            return Ok(None);
        }
        if kind == "PrimitiveNode" {
            return Ok(Some(Src::Value(node["widgets_values"].as_array().and_then(|w| w.first()).cloned().unwrap_or(Value::Null))));
        }
        if ctx.subgraphs.contains_key(kind) {
            let inst = self.instance(ctx, node)?;
            return Ok(usize::try_from(slot).ok().and_then(|s| inst.outputs.get(s).cloned()).flatten());
        }
        Ok(Some(Src::Node(format!("{}{}", self.prefix, id_str(&node["id"])), json!(slot))))
    }

    fn instance(&mut self, ctx: &mut Ctx<'a>, node: &'a Value) -> Result<&Instance, String> {
        let key = id_str(&node["id"]);
        if !self.instances.contains_key(&key) {
            let sub = ctx.subgraphs[node["type"].as_str().unwrap_or("")];
            let widgets: Map<String, Value> = widget_values(node, sub, false).into_iter().collect();
            let mut linked: HashMap<String, Option<Src>> = HashMap::new();
            for slot in node["inputs"].as_array().into_iter().flatten() {
                let s = self.source(ctx, &slot["link"], 0)?;
                linked.insert(slot["name"].as_str().unwrap_or("").to_string(), s);
            }
            // Matched by name: the instance may list its inputs in another order than the definition, or fewer.
            let mut feeds = vec![];
            for i in sub["inputs"].as_array().into_iter().flatten() {
                let name = i["name"].as_str().unwrap_or("");
                let mut src = linked.get(name).cloned().flatten();
                if src.is_none()
                    && let Some(v) = widgets.get(name)
                {
                    src = Some(Src::Value(v.clone()));
                }
                feeds.push(src);
            }
            let inst = expand_subgraph(ctx, sub, format!("{}{key}:", self.prefix), feeds)?;
            self.instances.insert(key.clone(), inst);
        }
        Ok(&self.instances[&key])
    }

    fn expand(&mut self, ctx: &mut Ctx<'a>) -> Result<(), String> {
        let nodes = self.nodes;
        for node in nodes {
            let kind = node["type"].as_str().unwrap_or("");
            let mode = int(&node["mode"]);
            if FRONTEND_ONLY.contains(&kind) || mode == Some(MUTED) || mode == Some(BYPASSED) {
                continue;
            }
            if ctx.subgraphs.contains_key(kind) {
                let top = self.outer.is_none();
                let inst = self.instance(ctx, node)?;
                if top {
                    let targets = inst.targets.clone();
                    for (name, t) in targets {
                        named_entry(&mut ctx.named, &name).extend(t);
                    }
                }
                continue;
            }
            let Some(spec) = ctx.info.get(kind) else { return Err(format!("node type '{kind}' is not installed in ComfyUI")) };
            let api_id = format!("{}{}", self.prefix, id_str(&node["id"]));
            let mut inputs: Map<String, Value> = widget_values(node, spec, true).into_iter().collect();
            for i in node["inputs"].as_array().into_iter().flatten() {
                let name = i["name"].as_str().unwrap_or("").to_string();
                match self.source(ctx, &i["link"], 0)? {
                    Some(Src::Node(id, slot)) => {
                        inputs.insert(name.clone(), json!([id, slot]));
                    }
                    Some(Src::Value(v)) => {
                        inputs.insert(name.clone(), v);
                    }
                    None => {}
                }
                let link = int(&i["link"]).and_then(|l| self.links.get(&l)).copied();
                if let (Some(outer), Some(link)) = (self.outer.as_mut(), link)
                    && link.origin_id == SUBGRAPH_IN
                    && let Some(n) = usize::try_from(link.origin_slot).ok().and_then(|s| outer.names.get(s)).cloned()
                {
                    named_entry(&mut outer.targets, &n).push((api_id.clone(), name));
                }
            }
            let title = node["title"].as_str().filter(|t| !t.is_empty()).unwrap_or(kind);
            ctx.prompt.insert(api_id, json!({"class_type": kind, "inputs": inputs, "_meta": {"title": title}}));
        }
        Ok(())
    }
}

fn expand_subgraph<'a>(ctx: &mut Ctx<'a>, sub: &'a Value, prefix: String, mut feeds: Vec<Option<Src>>) -> Result<Instance, String> {
    let names: Vec<String> = sub["inputs"].as_array().into_iter().flatten().map(|i| i["name"].as_str().unwrap_or("").to_string()).collect();
    if feeds.len() < names.len() {
        feeds.resize(names.len(), None);
    }
    let links = links_of(&sub["links"]);
    let mut scope = Scope::new(sub["nodes"].as_array().map(Vec::as_slice).unwrap_or(&[]), links, prefix, Some(Outer { feeds, names, targets: vec![] }));
    scope.expand(ctx)?;
    let mut outputs = vec![];
    for out in sub["outputs"].as_array().into_iter().flatten() {
        let into: Vec<&Value> = out["linkIds"].as_array().into_iter().flatten().filter(|l| int(l).and_then(|l| scope.links.get(&l)).is_some_and(|l| l.target_id == SUBGRAPH_OUT)).collect();
        outputs.push(match into.first() {
            Some(l) => scope.source(ctx, l, 0)?,
            None => None,
        });
    }
    Ok(Instance { outputs, targets: scope.outer.take().map(|o| o.targets).unwrap_or_default() })
}

/// Pairs each widget of a saved node with its value. The order comes from the node definition
/// (older saves only list the widgets that were turned into inputs); a subgraph instance stores its
/// values in the order of the subgraph's own inputs. Seed "control after generate" and upload-button
/// slots are skipped.
fn widget_values(node: &Value, spec: &Value, node_spec: bool) -> Vec<(String, Value)> {
    if let Some(o) = node["widgets_values"].as_object() {
        return o.iter().filter(|(k, _)| *k != "videopreview").map(|(k, v)| (k.clone(), v.clone())).collect();
    }
    let empty = vec![];
    let mut t = Taker { values: node["widgets_values"].as_array().unwrap_or(&empty), k: 0, out: vec![] };
    if !node_spec {
        for i in spec["inputs"].as_array().into_iter().flatten() {
            let ty = match &i["type"] {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => other.to_string(),
            };
            if WIDGET_TYPES.contains(&ty.as_str()) || ty.contains(',') {
                t.take(i["name"].as_str().unwrap_or(""), &Value::Null);
            }
        }
        return t.out;
    }
    t.walk(&spec["input"], &spec["input_order"], "");
    t.out
}

struct Taker<'v> {
    values: &'v [Value],
    k: usize,
    out: Vec<(String, Value)>,
}

impl Taker<'_> {
    fn take(&mut self, name: &str, opts: &Value) -> Option<Value> {
        let value = self.values.get(self.k)?.clone();
        self.out.push((name.to_string(), value.clone()));
        self.k += 1;
        let nxt = self.values.get(self.k);
        let nxt_str = nxt.and_then(Value::as_str);
        // The UI keeps an extra widget after a seed (its "randomize" control) and after an upload.
        let seed_control = (truthy(&opts["control_after_generate"]) || name.contains("seed")) && nxt_str.is_some_and(|s| SEED_CONTROLS.contains(&s));
        let upload_slot = UPLOADS.iter().any(|u| truthy(&opts[*u])) && (nxt.is_none_or(Value::is_null) || nxt_str.is_some_and(|s| ["image", "audio", "video", ""].contains(&s)));
        if seed_control || upload_slot {
            self.k += 1;
        }
        Some(value)
    }

    fn walk(&mut self, inputs: &Value, order: &Value, parent: &str) {
        for section in ["required", "optional"] {
            let names: Vec<String> = match order[section].as_array() {
                Some(o) => o.iter().filter_map(|n| n.as_str().map(String::from)).collect(),
                None => inputs[section].as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default(),
            };
            for name in names {
                let Some(entry) = inputs[section].get(&name) else { continue };
                let (kind, opts) = match entry.as_array() {
                    Some(a) => (a.first().cloned().unwrap_or(Value::Null), a.get(1).filter(|o| o.is_object()).cloned().unwrap_or(json!({}))),
                    None => (entry.clone(), json!({})),
                };
                let is_widget = kind.is_array() || kind.as_str().is_some_and(|k| WIDGET_TYPES.contains(&k));
                if !is_widget || truthy(&opts["forceInput"]) {
                    continue;
                }
                let value = self.take(&format!("{parent}{name}"), &opts);
                if kind == "COMFY_DYNAMICCOMBO_V3"
                    && let Some(chosen) = opts["options"].as_array().into_iter().flatten().find(|o| value.as_ref().is_some_and(|v| &o["key"] == v))
                {
                    let sub_inputs = if chosen["inputs"].is_null() { json!({}) } else { chosen["inputs"].clone() };
                    self.walk(&sub_inputs, &json!({}), &format!("{parent}{name}."));
                }
            }
        }
    }
}
