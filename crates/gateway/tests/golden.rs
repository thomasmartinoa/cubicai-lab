//! The media port against the Python gateway's own outputs, on this PC's real ComfyUI workflows.
//! The Python gateway is gone; what it made of each workflow was recorded in `tests/golden/recorded`
//! (`GOLDEN_RECORD=<old cubicai/server>` re-records from a checkout that still has it). Needs ComfyUI's
//! workflows and the cached node list, so it's ignored by default:
//! `cargo test -p cubic-gateway --release golden -- --ignored --nocapture`.

use serde_json::Value;
use std::path::PathBuf;

pub fn recorded(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/recorded").join(name)
}

pub fn python(what: &str) -> Value {
    let file = recorded(&format!("media-{what}.json"));
    if let Ok(server) = std::env::var("GOLDEN_RECORD") {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/media_golden.py");
        let out = std::process::Command::new("python3").arg(&script).arg(server).arg(what).output().expect("python3");
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let mut answer: Value = serde_json::from_slice(&out.stdout).expect("json");
        // The workflows hold the user's own prompts: only their fingerprints are kept.
        if what == "to_api" {
            answer = Value::Object(answer.as_object().unwrap().iter().map(|(k, v)| (k.clone(), seal_converted(v))).collect());
        } else if what == "generators" {
            answer = seal_defaults(&answer);
        } else if what == "build" {
            answer = Value::Object(answer.as_object().unwrap().iter().map(|(k, v)| (k.clone(), seal_nodes(v))).collect());
        }
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, serde_json::to_vec_pretty(&answer).unwrap()).unwrap();
    }
    serde_json::from_slice(&std::fs::read(&file).expect("recorded Python output")).expect("json")
}

/// The value with its keys sorted at every level, so the same JSON always gives the same text.
fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            format!("{{{}}}", keys.iter().map(|k| format!("{}:{}", Value::from(k.as_str()), canonical(&m[*k]))).collect::<Vec<_>>().join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(canonical).collect::<Vec<_>>().join(",")),
        _ => v.to_string(),
    }
}

pub fn seal(v: &Value) -> Value {
    Value::from(cubic_gateway::identity::fingerprint(canonical(v).as_bytes()))
}

/// A prompt (node id → node) as node id → fingerprint; an error stays as it is.
pub fn seal_nodes(v: &Value) -> Value {
    match v.get("error") {
        Some(_) => v.clone(),
        None => Value::Object(v.as_object().unwrap().iter().map(|(k, n)| (k.clone(), seal(n))).collect()),
    }
}

/// The generator list with each workflow's own negative prompt as a fingerprint.
pub fn seal_defaults(list: &Value) -> Value {
    Value::from(
        list.as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let mut g = g.clone();
                if let Some(n) = g.get("negative_default").cloned() {
                    g["negative_default"] = seal(&n);
                }
                g
            })
            .collect::<Vec<_>>(),
    )
}

/// A converted workflow ({prompt, named} or {error}) with fingerprints in place of its contents.
pub fn seal_converted(v: &Value) -> Value {
    match v.get("error") {
        Some(_) => v.clone(),
        None => serde_json::json!({"prompt": seal_nodes(&v["prompt"]), "named": seal(&v["named"])}),
    }
}

pub fn object_info() -> Value {
    let home = std::env::var("HOME").unwrap();
    serde_json::from_slice(&std::fs::read(format!("{home}/.cache/cubicai-server/object_info.json")).unwrap()).unwrap()
}

pub fn workflows() -> PathBuf {
    PathBuf::from("/mnt/storage/games/AI_local/ComfyUI/user/default/workflows")
}

#[test]
#[ignore]
fn golden_to_api_matches_python_on_every_workflow() {
    let want = python("to_api");
    let info = object_info();
    let mut n = 0;
    for (name, w) in want.as_object().unwrap() {
        let wf: Value = serde_json::from_slice(&std::fs::read(workflows().join(name)).unwrap()).unwrap();
        let got = match cubic_gateway::workflow::to_api(&wf, &info) {
            Ok((prompt, named)) => {
                let named: serde_json::Map<String, Value> = named.into_iter().map(|(k, v)| (k, Value::from(v.into_iter().map(|(a, b)| Value::from(vec![a, b])).collect::<Vec<_>>()))).collect();
                serde_json::json!({"prompt": prompt, "named": named})
            }
            Err(e) => serde_json::json!({"error": format!("ValueError: {e}")}),
        };
        let full = got.clone();
        let got = seal_converted(&got);
        if &got != w {
            let (g, p) = (got["prompt"].as_object(), w["prompt"].as_object());
            if let (Some(g), Some(p)) = (g, p) {
                for k in p.keys().chain(g.keys()) {
                    if g.get(k) != p.get(k) {
                        println!("{name} node {k} differs from Python's (fingerprint {}):\n  rust   {}", p.get(k).map(|v| v.to_string()).unwrap_or("missing".into()), full["prompt"].get(k).map(|v| v.to_string()).unwrap_or("missing".into()));
                        break;
                    }
                }
            }
            if got["named"] != w["named"] {
                println!("{name} named differs from Python's:\n  rust   {}", full["named"]);
            }
            if got.get("error") != w.get("error") {
                println!("{name} error: rust {:?} python {:?}", got.get("error"), w.get("error"));
            }
            panic!("{name} differs");
        }
        n += 1;
        println!("  {name}: {} nodes", full["prompt"].as_object().map(|p| p.len()).unwrap_or(0));
    }
    assert!(n >= 15, "the real workflows were checked ({n})");
}

#[test]
#[ignore]
fn golden_generator_table_and_sizes() {
    use cubic_gateway::media;
    let want = python("table");
    assert_eq!(Value::from(media::generators_table()), want["generators"], "the same generators, selectors and fixed values");
    let aspects: Vec<&str> = want["aspects"].as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(aspects, media::ASPECTS.iter().map(|a| a.0).collect::<Vec<_>>());
    for (key, wh) in want["sizes"].as_object().unwrap() {
        let p: Vec<&str> = key.split('/').collect();
        let (w, h) = media::size(p[0], p[1].parse().unwrap(), p[2].parse().unwrap());
        assert_eq!(serde_json::json!([w, h]), *wh, "{key}");
    }
}

#[test]
#[ignore]
fn golden_build_matches_python_for_every_generator() {
    use cubic_gateway::media;
    let want = python("build");
    let catalog = media::Catalog::new(media::Paths::from_env());
    let info = object_info();
    let params = serde_json::json!({"prompt": "a red fox in snow", "negative": "blurry", "lyrics": "[verse]\nla la", "seed": 123456789012u64,
              "steps": 25.7, "cfg": 4, "aspect": "9:16", "seconds": 3, "language": "fr"});
    for g in media::generators_table() {
        for files in [false, true] {
            let key = format!("{}{}", g["id"].as_str().unwrap(), if files { "+files" } else { "" });
            let wf = catalog.workflow(&g).expect("workflow on this PC");
            let got = match media::build(&g, &params, "JOBID", files.then_some("cubicai/JOBID.png"), files.then_some("cubicai/JOBID.wav"), &wf, &info, 7) {
                Ok(p) => Value::Object(p),
                Err(e) => serde_json::json!({"error": format!("ValueError: {e}")}),
            };
            let full = got.clone();
            let got = seal_nodes(&got);
            if got != want[&key] {
                for (k, v) in want[&key].as_object().unwrap() {
                    if got.get(k) != Some(v) {
                        println!("{key} {k} differs from Python's (fingerprint {v}):\n  rust   {}", full.get(k).map(|x| x.to_string()).unwrap_or("missing".into()));
                    }
                }
                for k in got.as_object().unwrap().keys() {
                    if want[&key].get(k).is_none() {
                        println!("{key} {k}: only in rust");
                    }
                }
                panic!("{key} differs");
            }
            println!("  {key}: {}", if got.get("error").is_some() { got["error"].to_string() } else { format!("{} nodes", got.as_object().unwrap().len()) });
        }
    }
}

#[test]
#[ignore]
fn golden_generator_list_matches_python() {
    use cubic_gateway::media;
    let want = python("generators");
    let catalog = media::Catalog::new(media::Paths::from_env());
    let got = seal_defaults(&Value::from(catalog.list(Some(&object_info()))));
    for (g, w) in got.as_array().unwrap().iter().zip(want.as_array().unwrap()) {
        assert_eq!(g, w, "{}", w["id"]);
    }
    assert_eq!(got, want);
    println!("  {} generators: {}", want.as_array().unwrap().len(), want.as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect::<Vec<_>>().join(", "));
}
