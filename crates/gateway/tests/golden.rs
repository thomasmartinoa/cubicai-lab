//! The media port against the Python gateway's own code, on this PC's real ComfyUI workflows.
//! Needs the CubicAI repo beside this one, ComfyUI's workflows and the cached node list, so it's
//! ignored by default: `cargo test -p cubic-gateway --release golden -- --ignored --nocapture`.

use serde_json::Value;
use std::path::PathBuf;

pub fn server_dir() -> PathBuf {
    std::env::var("CUBICAI_SERVER_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../cubicai/server"))
}

pub fn python(what: &str) -> Value {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/media_golden.py");
    let out = std::process::Command::new("python3").arg(&script).arg(server_dir()).arg(what).output().expect("python3");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("json")
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
        if &got != w {
            let (g, p) = (got["prompt"].as_object(), w["prompt"].as_object());
            if let (Some(g), Some(p)) = (g, p) {
                for k in p.keys().chain(g.keys()) {
                    if g.get(k) != p.get(k) {
                        println!("{name} node {k}:\n  rust   {}\n  python {}", g.get(k).map(|v| v.to_string()).unwrap_or_default(), p.get(k).map(|v| v.to_string()).unwrap_or_default());
                        break;
                    }
                }
            }
            if got["named"] != w["named"] {
                println!("{name} named:\n  rust   {}\n  python {}", got["named"], w["named"]);
            }
            if got.get("error") != w.get("error") {
                println!("{name} error: rust {:?} python {:?}", got.get("error"), w.get("error"));
            }
            panic!("{name} differs");
        }
        n += 1;
        println!("  {name}: {} nodes", got["prompt"].as_object().map(|p| p.len()).unwrap_or(0));
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
            if got != want[&key] {
                for (k, v) in want[&key].as_object().unwrap() {
                    if got.get(k) != Some(v) {
                        println!("{key} {k}:\n  rust   {}\n  python {v}", got.get(k).map(|x| x.to_string()).unwrap_or_default());
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
    let got = Value::from(catalog.list(Some(&object_info())));
    for (g, w) in got.as_array().unwrap().iter().zip(want.as_array().unwrap()) {
        assert_eq!(g, w, "{}", w["id"]);
    }
    assert_eq!(got, want);
    println!("  {} generators: {}", want.as_array().unwrap().len(), want.as_array().unwrap().iter().map(|g| g["id"].as_str().unwrap()).collect::<Vec<_>>().join(", "));
}
