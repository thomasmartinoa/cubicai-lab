//! Making things, end to end through the gateway the way the phone does it, with a fake ComfyUI:
//! the generator list, a job's progress events and preview, the file download that leaves nothing
//! behind, ComfyUI's own errors, saving on the PC and the clean-up rules.

mod common;

use common::*;
use cubic_gateway::http::Conn;
use cubic_gateway::{identity, media, server, Config};
use serde_json::{json, Value};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const WORKFLOW: &str = r#"{"nodes": [
 {"id": 14, "type": "CheckpointLoaderSimple", "mode": 0, "outputs": [{"type": "MODEL"}, {"type": "CLIP"}, {"type": "VAE"}], "widgets_values": ["illustrious.safetensors"]},
 {"id": 10, "type": "CLIPTextEncode", "inputs": [{"name": "clip", "type": "CLIP", "link": 2}], "widgets_values": ["a cat"]},
 {"id": 11, "type": "CLIPTextEncode", "title": "Negative", "inputs": [{"name": "clip", "type": "CLIP", "link": 3}], "widgets_values": ["lowres, bad hands"]},
 {"id": 13, "type": "EmptyLatentImage", "widgets_values": [1024, 1024, 1]},
 {"id": 12, "type": "KSampler", "inputs": [{"name": "model", "link": 1}, {"name": "positive", "link": 4}, {"name": "negative", "link": 5}, {"name": "latent_image", "link": 6}],
  "widgets_values": [42, "randomize", 28, 6.5, "euler", "normal", 1.0]},
 {"id": 15, "type": "VAEDecode", "inputs": [{"name": "samples", "link": 7}, {"name": "vae", "link": 8}]},
 {"id": 16, "type": "SaveImage", "inputs": [{"name": "images", "link": 9}], "widgets_values": ["ComfyUI"]},
 {"id": 17, "type": "PreviewImage", "inputs": [{"name": "images", "link": 10}]},
 {"id": 18, "type": "Note", "widgets_values": ["just a note"]}
], "links": [[1, 14, 0, 12, 0, "MODEL"], [2, 14, 1, 10, 0, "CLIP"], [3, 14, 1, 11, 0, "CLIP"], [4, 10, 0, 12, 1, "CONDITIONING"], [5, 11, 0, 12, 2, "CONDITIONING"],
  [6, 13, 0, 12, 3, "LATENT"], [7, 12, 0, 15, 0, "LATENT"], [8, 14, 2, 15, 1, "VAE"], [9, 15, 0, 16, 0, "IMAGE"], [10, 15, 0, 17, 0, "IMAGE"]]}"#;

const INFO: &str = r#"{
 "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["illustrious.safetensors", "other.safetensors"], {}]}}, "input_order": {"required": ["ckpt_name"]}},
 "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": true}], "clip": ["CLIP"]}}, "input_order": {"required": ["text", "clip"]}},
 "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}}, "input_order": {"required": ["width", "height", "batch_size"]}},
 "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {"control_after_generate": true, "max": 18446744073709551615}], "steps": ["INT", {}], "cfg": ["FLOAT", {}],
   "sampler_name": [["euler"], {}], "scheduler": [["normal"], {}], "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"], "denoise": ["FLOAT", {}]}},
   "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image", "denoise"]}},
 "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}, "input_order": {"required": ["samples", "vae"]}},
 "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}}, "input_order": {"required": ["images", "filename_prefix"]}},
 "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}, "input_order": {"required": ["images"]}}
}"#;

const PNG: &[u8] = b"\x89PNG\r\n\x1a\nfake picture bytes";

// --- A fake ComfyUI ---------------------------------------------------------------------------------

#[derive(Default)]
struct Comfy {
    prompts: Vec<Value>,
    /// (client id, prompt id, prompt) waiting for the WebSocket to report on.
    pending: Vec<(String, String, Value)>,
    calls: Vec<String>,
}

type Shared = Arc<(Mutex<Comfy>, Condvar)>;

fn frame(op: u8, payload: &[u8], fin: bool) -> Vec<u8> {
    let mut f = vec![if fin { 0x80 } else { 0 } | op];
    match payload.len() {
        n if n < 126 => f.push(n as u8),
        n if n < 65536 => {
            f.push(126);
            f.extend((n as u16).to_be_bytes());
        }
        n => {
            f.push(127);
            f.extend((n as u64).to_be_bytes());
        }
    }
    f.extend_from_slice(payload);
    f
}

fn fake_comfy(dir: PathBuf) -> (u16, Shared) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let shared: Shared = Default::default();
    let s = shared.clone();
    std::thread::spawn(move || {
        for tcp in l.incoming().flatten() {
            let (s, dir) = (s.clone(), dir.clone());
            std::thread::spawn(move || comfy_conn(tcp, s, dir));
        }
    });
    (port, shared)
}

fn reply(c: &mut Conn<TcpStream>, v: &Value) {
    let body = v.to_string();
    let _ = c.raw(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes());
}

fn comfy_conn(tcp: TcpStream, s: Shared, dir: PathBuf) {
    let mut c = Conn::new(tcp);
    while let Ok(Some(req)) = c.next_request() {
        let body: Value = match req.content_length() {
            Ok(Some(n)) if n > 0 => serde_json::from_slice(&c.read_body(n).unwrap()).unwrap_or(Value::Null),
            _ => Value::Null,
        };
        s.0.lock().unwrap().calls.push(format!("{} {}", req.method, req.path()));
        match (req.method.as_str(), req.path()) {
            ("GET", "/system_stats") => reply(&mut c, &json!({"system": {}})),
            ("GET", "/object_info") => reply(&mut c, &serde_json::from_str(INFO).unwrap()),
            ("POST", "/prompt") => {
                let prompt = body["prompt"].clone();
                let prefix = prompt["16"]["inputs"]["filename_prefix"].as_str().unwrap().to_string();
                std::fs::write(dir.join("output").join(format!("{prefix}_00001_.png")), PNG).unwrap();
                let mut st = s.0.lock().unwrap();
                st.prompts.push(prompt.clone());
                let pid = format!("p{}", st.prompts.len());
                st.pending.push((body["client_id"].as_str().unwrap().to_string(), pid.clone(), prompt));
                s.1.notify_all();
                drop(st);
                reply(&mut c, &json!({"prompt_id": pid, "number": 1}));
            }
            ("GET", p) if p.starts_with("/history/") => {
                let pid = &p["/history/".len()..];
                let st = s.0.lock().unwrap();
                let n: usize = pid[1..].parse().unwrap();
                let prefix = st.prompts[n - 1]["16"]["inputs"]["filename_prefix"].as_str().unwrap().to_string();
                let (sub, name) = prefix.split_once('/').unwrap();
                reply(&mut c, &json!({pid: {"outputs": {
                    "16": {"images": [{"filename": format!("{name}_00001_.png"), "subfolder": sub, "type": "output"}]},
                    // Never followed: outside ComfyUI's folders.
                    "99": {"images": [{"filename": "passwd", "subfolder": "../../../../etc", "type": "output"}], "text": ["not a file list"]},
                }}}));
            }
            ("GET", "/ws") => {
                let client = req.query("clientId").unwrap();
                let _ = c.raw(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n");
                ws_session(c, &s, &client);
                return;
            }
            _ => reply(&mut c, &json!({})),
        }
    }
}

/// Reports a run the way ComfyUI does: node by node, sampler steps, a preview, then the end.
fn ws_session(mut c: Conn<TcpStream>, s: &Shared, client: &str) {
    let (pid, prompt) = {
        let mut st = s.0.lock().unwrap();
        loop {
            if let Some(i) = st.pending.iter().position(|p| p.0 == client) {
                let p = st.pending.remove(i);
                break (p.1, p.2);
            }
            st = s.1.wait(st).unwrap();
        }
    };
    let text = |v: Value| frame(1, v.to_string().as_bytes(), true);
    let fail = prompt["10"]["inputs"]["text"].as_str().unwrap_or("").contains("FAIL");
    let mut msgs: Vec<Vec<u8>> = vec![
        text(json!({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 1}}}})),
        text(json!({"type": "executing", "data": {"node": "14", "prompt_id": pid}})),
        text(json!({"type": "executing", "data": {"node": "99", "prompt_id": "someone-else"}})),
        text(json!({"type": "execution_cached", "data": {"nodes": ["10"], "prompt_id": pid}})),
        text(json!({"type": "executing", "data": {"node": "12", "prompt_id": pid}})),
        frame(9, b"ping", true),
        frame(2, &[&[0, 0, 0, 1, 0, 0, 0, 2][..], b"PREVIEW-JPEG"].concat(), true),
    ];
    for v in 1..=4 {
        // One message in two fragments.
        let m = json!({"type": "progress", "data": {"value": v, "max": 4, "prompt_id": pid, "node": "12"}}).to_string();
        let (a, b) = m.as_bytes().split_at(10);
        msgs.push([frame(1, a, false), frame(0, b, true)].concat());
    }
    msgs.push(text(json!({"type": "executing", "data": {"node": "16", "prompt_id": pid}})));
    msgs.push(if fail {
        text(json!({"type": "execution_error", "data": {"prompt_id": pid, "exception_message": "  out of cheese  "}}))
    } else {
        text(json!({"type": "executing", "data": {"node": null, "prompt_id": pid}}))
    });
    for m in msgs {
        if c.raw(&m).is_err() {
            return;
        }
        std::thread::sleep(Duration::from_millis(60));
    }
    // Stay until the gateway hangs up (reading its pong and close frames).
    let _ = std::io::copy(&mut c.r, &mut std::io::sink());
}

// --- The gateway --------------------------------------------------------------------------------------

struct Setup {
    gw: server::Running,
    token: String,
    pin: String,
    comfy: Shared,
    root: PathBuf,
}

impl Drop for Setup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn setup() -> Setup {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let root = std::env::temp_dir().join(format!("cubic-gw-media-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let comfy_dir = root.join("comfy");
    for d in ["user/default/workflows", "output/cubicai", "input/cubicai", "temp"] {
        std::fs::create_dir_all(comfy_dir.join(d)).unwrap();
    }
    std::fs::write(comfy_dir.join("user/default/workflows/my_illustrious_anime.json"), WORKFLOW).unwrap();
    // Left by an earlier run: cleared when the gateway starts.
    std::fs::write(comfy_dir.join("output/cubicai/old_00001_.png"), b"old").unwrap();
    std::fs::create_dir_all(root.join("cache/media/oldjob")).unwrap();
    std::fs::create_dir_all(root.join("voices")).unwrap();
    std::fs::write(root.join("voices/woman.wav"), b"RIFF....WAVE").unwrap();
    let (port, comfy) = fake_comfy(comfy_dir.clone());
    let paths = media::Paths {
        comfy_dir,
        comfy_port: port,
        save_dir: root.join("saved"),
        cache_dir: root.join("cache"),
        voices_dir: root.join("voices"),
        idle_stop: 600,
        // A unit that doesn't exist: nothing on this PC is started or stopped by the test.
        unit: "cubicai-test-does-not-exist.service".into(),
    };
    let config = Config { dir: root.join("config"), ollama: "http://127.0.0.1:1".into(), searxng: "http://127.0.0.1:1".into(), default_ctx: 16384, media: Some(paths) };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    Setup { gw, token: id.token, pin: id.fingerprint, comfy, root }
}

fn submit(s: &Setup, body: Value) -> Resp {
    ask(s.gw.port, &s.pin, "POST", "/v1/media/jobs", Some(&s.token), Some(&body.to_string()))
}

fn events(s: &Setup, id: &str) -> Vec<Value> {
    let r = ask(s.gw.port, &s.pin, "GET", &format!("/v1/media/jobs/{id}/events"), Some(&s.token), None);
    assert_eq!(r.header("Content-Type"), Some("text/event-stream"));
    r.events().iter().map(|e| serde_json::from_str(e).unwrap()).collect()
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    v.sort();
    v
}

#[test]
fn making_a_picture_from_the_phone() {
    let s = setup();
    let port = s.gw.port;
    assert!(files_in(&s.root.join("comfy/output/cubicai")).is_empty(), "leftovers in ComfyUI's output cleared at start");
    assert!(files_in(&s.root.join("cache/media")).is_empty(), "and in the temp folder");

    // The list: only what has its workflow and models here, with the workflow's own settings.
    let g = ask(port, &s.pin, "GET", "/v1/media/generators", Some(&s.token), None).json();
    assert_eq!(g["comfy_running"], true);
    let list = g["generators"].as_array().unwrap();
    assert_eq!(list.len(), 1, "{g}");
    let ill = &list[0];
    assert_eq!((ill["id"].clone(), ill["kind"].clone(), ill["image"].clone(), ill["negative"].clone(), ill["seed"].clone()), (json!("illustrious"), json!("image"), json!("none"), json!(true), json!(true)));
    assert_eq!(ill["steps"], json!({"default": 28, "min": 4, "max": 60}));
    assert_eq!(ill["cfg"], json!({"default": 6.5, "min": 1.0, "max": 12.0}));
    assert_eq!(ill["negative_default"], "lowres, bad hands");
    assert_eq!(ill["aspects"].as_array().unwrap().len(), 7);

    // Make one.
    let r = submit(&s, json!({"generator": "illustrious", "prompt": "a fox", "negative": "blurry", "aspect": "2:3", "steps": 20, "cfg": 30}));
    assert_eq!(r.status, 200, "{}", r.body);
    let id = r.json()["id"].as_str().unwrap().to_string();
    assert_eq!(id.len(), 16);
    let ev = events(&s, &id);
    let last = ev.last().unwrap();
    assert_eq!(last["state"], "done", "{ev:?}");
    let states: Vec<&str> = ev.iter().map(|e| e["state"].as_str().unwrap()).collect();
    assert!(states.contains(&"running"), "{states:?}");
    let progress: Vec<f64> = ev.iter().map(|e| e["progress"].as_f64().unwrap()).collect();
    assert!(progress.windows(2).all(|w| w[1] >= w[0]), "the bar only moves forward: {progress:?}");
    assert_eq!(*progress.last().unwrap(), 1.0);
    assert!(ev.iter().any(|e| e["phase"] == "Generating" && e["steps"] == 4 && e["step"].as_u64().unwrap() > 0), "sampler steps shown");
    let previews: Vec<&Value> = ev.iter().filter_map(|e| e.get("preview")).collect();
    assert_eq!(previews.len(), 1, "a preview, sent once");
    assert_eq!(base64::Engine::decode(&base64::engine::general_purpose::STANDARD, previews[0].as_str().unwrap()).unwrap(), b"PREVIEW-JPEG");
    let seed = last["seed"].as_u64().unwrap();
    assert!(seed <= media::SEED_MAX);
    assert_eq!(last["files"], json!([{"index": 0, "name": format!("illustrious-{}-1.png", &id[..6]), "mime": "image/png", "size": PNG.len(), "kind": "image"}]), "only the real file, not the one outside ComfyUI's folder");

    // What ComfyUI was asked to run.
    let p = s.comfy.0.lock().unwrap().prompts[0].clone();
    assert_eq!(p["10"]["inputs"]["text"], "a fox");
    assert_eq!(p["11"]["inputs"]["text"], "blurry");
    assert_eq!((p["12"]["inputs"]["steps"].clone(), p["12"]["inputs"]["cfg"].clone(), p["12"]["inputs"]["seed"].clone()), (json!(20), json!(12.0), json!(seed)), "steps, CFG capped at 12, the reported seed");
    let (w, h) = media::size("2:3", 1024, 64);
    assert_eq!((p["13"]["inputs"]["width"].clone(), p["13"]["inputs"]["height"].clone()), (json!(w), json!(h)));
    assert_eq!(p["16"]["inputs"]["filename_prefix"], format!("cubicai/{id}"));
    assert!(p.get("17").is_none() && p.get("18").is_none(), "previews and notes aren't run");
    assert_eq!(p["12"]["inputs"]["positive"], json!(["10", 0]));

    // The file moved out of ComfyUI's folder into the private temp folder, until the phone has it.
    assert!(files_in(&s.root.join("comfy/output/cubicai")).is_empty());
    assert_eq!(files_in(&s.root.join("cache/media").join(&id)), ["0.png"]);
    let r = ask(port, &s.pin, "GET", &format!("/v1/media/jobs/{id}/files/0"), Some(&s.token), None);
    assert_eq!((r.status, r.header("Content-Type"), r.bytes.as_slice()), (200, Some("image/png"), PNG));
    assert_eq!(ask(port, &s.pin, "GET", &format!("/v1/media/jobs/{id}"), Some(&s.token), None).status, 404, "delivered: the job is forgotten");
    assert!(!s.root.join("cache/media").join(&id).exists(), "and its files deleted");

    // The activity list followed it.
    let st = ask(port, &s.pin, "GET", "/v1/cubicai/status", Some(&s.token), None).json();
    assert_eq!((st["recent"][0]["model"].clone(), st["recent"][0]["outcome"].clone()), (json!("Illustrious XL (anime)"), json!("ok")));

    // Chat after a picture: ComfyUI gives the GPU back.
    s.gw.gateway.media.as_ref().unwrap().comfy.free_for_chat();
    assert!(s.comfy.0.lock().unwrap().calls.iter().any(|c| c == "POST /free"));
}

#[test]
fn errors_cancels_saving_and_expiry() {
    let s = setup();
    let port = s.gw.port;
    // ComfyUI's own words reach the phone.
    let id = submit(&s, json!({"generator": "illustrious", "prompt": "FAIL please"})).json()["id"].as_str().unwrap().to_string();
    let last = events(&s, &id).pop().unwrap();
    assert_eq!((last["state"].clone(), last["error"].clone()), (json!("error"), json!("ComfyUI: out of cheese")));
    let out = s.root.join("comfy/output/cubicai");
    let t = std::time::Instant::now();
    while !files_in(&out).is_empty() && t.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(files_in(&out).is_empty(), "a failed run leaves nothing in ComfyUI's output");
    let st = ask(port, &s.pin, "GET", "/v1/cubicai/status", Some(&s.token), None).json();
    assert_eq!(st["recent"][0]["outcome"], "error");

    for (body, want) in [
        (json!({"generator": "nope", "prompt": "x"}), "That generator isn't available on your PC."),
        (json!({"generator": "flux2-edit", "prompt": "x"}), "Add a photo to use this."),
        (json!({"generator": "flux2-edit", "prompt": "x", "image": "!!!"}), "Invalid base64 in image"),
    ] {
        let r = submit(&s, body);
        assert_eq!(r.status, 400);
        assert!(r.json()["error"]["message"].as_str().unwrap().starts_with(want), "{}", r.body);
    }
    // An empty prompt fails in the job, as in the Python gateway.
    let id = submit(&s, json!({"generator": "illustrious", "prompt": "  "})).json()["id"].as_str().unwrap().to_string();
    assert_eq!(events(&s, &id).pop().unwrap()["error"], "Type what you want to make.");

    // Cancel: the job and its files go; unknown or odd ids are refused.
    let id = submit(&s, json!({"generator": "illustrious", "prompt": "a fox"})).json()["id"].as_str().unwrap().to_string();
    assert_eq!(events(&s, &id).pop().unwrap()["state"], "done");
    assert_eq!(ask(port, &s.pin, "DELETE", &format!("/v1/media/jobs/{id}"), Some(&s.token), None).json(), json!({"ok": true}));
    assert_eq!(ask(port, &s.pin, "GET", &format!("/v1/media/jobs/{id}/files/0"), Some(&s.token), None).status, 404);
    assert!(!s.root.join("cache/media").join(&id).exists());
    for bad in ["nope", "..%2F..", "a.b"] {
        assert_eq!(ask(port, &s.pin, "DELETE", &format!("/v1/media/jobs/{bad}"), Some(&s.token), None).status, 404);
    }

    // Not collected within the hour: deleted.
    let id = submit(&s, json!({"generator": "illustrious", "prompt": "a fox"})).json()["id"].as_str().unwrap().to_string();
    assert_eq!(events(&s, &id).pop().unwrap()["state"], "done");
    let m = s.gw.gateway.media.as_ref().unwrap();
    m.expire(cubic_gateway::time::now() + 60.0);
    assert!(m.get(&id).is_some(), "not yet");
    m.expire(cubic_gateway::time::now() + 3700.0);
    assert!(m.get(&id).is_none() && !s.root.join("cache/media").join(&id).exists());

    // Saving from the phone: cleaned names, never overwriting, only the three kinds.
    let save = |kind: &str, name: &str, body: &str| ask(port, &s.pin, "POST", &format!("/v1/media/save?kind={kind}&name={name}"), Some(&s.token), Some(body));
    assert_eq!(save("image", "..%2F..%2Fescape%3F.png", "one").json(), json!({"ok": true, "path": "Images/escape_.png"}));
    assert_eq!(save("image", "escape%3F.png", "two").json()["path"], "Images/escape_ (1).png");
    assert_eq!(save("audio", "", "three").json()["path"], "Music/cubicai");
    assert_eq!(std::fs::read_to_string(s.root.join("saved/Images/escape_.png")).unwrap(), "one");
    assert_eq!(std::fs::read_to_string(s.root.join("saved/Images/escape_ (1).png")).unwrap(), "two");
    let r = save("secrets", "x", "four");
    assert_eq!((r.status, r.json()["error"]["message"].clone()), (400, json!("Unknown kind")));
}

#[test]
fn file_types_and_progress_weights() {
    assert_eq!(media::suffix(PNG, ".bin"), ".png");
    assert_eq!(media::suffix(b"\xff\xd8\xff\xe0....", ".bin"), ".jpg");
    assert_eq!(media::suffix(b"RIFF\0\0\0\0WEBPVP8 ", ".bin"), ".webp");
    assert_eq!(media::suffix(b"RIFF\0\0\0\0WAVEfmt ", ".bin"), ".wav");
    assert_eq!(media::suffix(b"\0\0\0\x20ftypM4A ", ".bin"), ".m4a");
    assert_eq!(media::suffix(b"OggS", ".bin"), ".ogg");
    assert_eq!(media::suffix(b"hello", ".wav"), ".wav");
    assert_eq!((media::mime(".mp4"), media::kind(".mp4")), ("video/mp4", "video"));
    assert_eq!((media::mime(".gif"), media::kind(".gif")), ("image/gif", "video"));
    assert_eq!((media::mime(".flac"), media::kind(".flac")), ("audio/flac", "audio"));
    assert_eq!(media::mime(".xyz"), "application/octet-stream");
    use cubic_gateway::jobs::{phase, preview, weight};
    assert_eq!((weight("KSamplerSelect"), weight("KSamplerAdvanced"), weight("CheckpointLoaderSimple"), weight("VAEDecode"), weight("PrimitiveInt")), (0.1, 8.0, 1.0, 1.0, 0.1));
    assert_eq!((phase("KSamplerSelect"), phase("KSampler"), phase("FL_ChatterboxMultilingualTTS"), phase("SaveImage"), phase("Foo")), ("Working", "Generating", "Speaking", "Saving", "Working"));
    assert_eq!(preview(&[0, 0, 0, 1, 0, 0, 0, 2, 9, 9]), Some(vec![9, 9]));
    assert_eq!(preview(&[0, 0, 0, 4, 0, 0, 0, 2, 7, 7, 9]), Some(vec![9]), "metadata skipped");
    assert_eq!(preview(&[0, 0, 0, 3, 0, 0, 0, 0, 1]), None);
    assert_eq!(preview(&[1, 2]), None);
    assert_eq!(
        cubic_gateway::jobs::comfy_error(400, br#"{"error": {"message": "Prompt outputs failed validation"}, "node_errors": {"12": {"errors": [{"message": "Value 0 smaller than min of 1: steps"}], "class_type": "KSampler"}}}"#),
        "ComfyUI rejected the workflow: Value 0 smaller than min of 1: steps (KSampler)"
    );
    assert_eq!(cubic_gateway::jobs::comfy_error(500, b"oops"), "ComfyUI rejected the workflow (500)");
    // Speech without a sample: the reference voice is used.
    let dir = std::env::temp_dir().join(format!("cubic-gw-voices-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("woman.wav"), b"x").unwrap();
    let p = media::Paths { voices_dir: dir.clone(), ..media::Paths::from_env() };
    assert_eq!(p.voice_file("woman"), Some(dir.join("woman.wav")));
    assert_eq!(p.voice_file("man"), None);
    assert_eq!(p.voice_file("../woman"), None);
    let _ = std::fs::remove_dir_all(&dir);
}

/// With the real ComfyUI on this PC (started for the test and stopped after, if it was off): a
/// picture from FLUX.2 Klein, delivered, with nothing left behind.
/// `cargo test -p cubic-gateway --release --test media -- --ignored --nocapture`
#[test]
#[ignore]
fn real_comfy_makes_a_picture() {
    let root = std::env::temp_dir().join(format!("cubic-gw-real-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let paths = media::Paths { cache_dir: root.join("cache"), save_dir: root.join("saved"), ..media::Paths::from_env() };
    let comfy_out = paths.comfy_dir.join("output").join(media::SUBFOLDER);
    let unit = paths.unit.clone();
    let was_on = std::process::Command::new("systemctl").args(["--user", "is-active", "--quiet", &unit]).status().unwrap().success();
    // Stopped again afterwards (also when the test fails), if it was off.
    struct Off(Option<String>);
    impl Drop for Off {
        fn drop(&mut self) {
            if let Some(u) = &self.0 {
                let _ = std::process::Command::new("systemctl").args(["--user", "stop", u]).status();
            }
        }
    }
    let _off = Off((!was_on).then(|| unit.clone()));
    let config = Config { media: Some(paths), ..Config::from_env() };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let (port, pin, token) = (gw.port, id.fingerprint.clone(), id.token.clone());

    let g = ask(port, &pin, "GET", "/v1/media/generators", Some(&token), None).json();
    let ids: Vec<&str> = g["generators"].as_array().unwrap().iter().filter_map(|x| x["id"].as_str()).collect();
    println!("generators: {ids:?}, ComfyUI running: {}", g["comfy_running"]);
    assert!(ids.contains(&"flux2-klein"), "{ids:?}");

    let t = std::time::Instant::now();
    let r = ask(port, &pin, "POST", "/v1/media/jobs", Some(&token), Some(&json!({"generator": "flux2-klein", "prompt": "a small red cube on a wooden desk, soft daylight", "aspect": "1:1"}).to_string()));
    assert_eq!(r.status, 200, "{}", r.body);
    let job = r.json()["id"].as_str().unwrap().to_string();
    let r = ask(port, &pin, "GET", &format!("/v1/media/jobs/{job}/events"), Some(&token), None);
    let ev: Vec<Value> = r.events().iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    let phases: Vec<&str> = ev.iter().filter_map(|e| e["phase"].as_str()).collect::<Vec<_>>().into_iter().fold(vec![], |mut v, p| {
        if v.last() != Some(&p) {
            v.push(p);
        }
        v
    });
    let last = ev.last().unwrap();
    println!("{} events in {:.1} s, phases {phases:?}, previews {}, seed {}", ev.len(), t.elapsed().as_secs_f64(), ev.iter().filter(|e| e.get("preview").is_some()).count(), last["seed"]);
    assert_eq!(last["state"], "done", "{last}");
    let f = &last["files"][0];
    let r = ask(port, &pin, "GET", &format!("/v1/media/jobs/{job}/files/0"), Some(&token), None);
    assert_eq!(r.status, 200);
    assert_eq!(r.bytes.len() as u64, f["size"].as_u64().unwrap());
    assert!(r.bytes.starts_with(b"\x89PNG"), "a PNG");
    std::fs::write(root.join("real.png"), &r.bytes).unwrap();
    println!("{} ({} bytes) → {}", f["name"], r.bytes.len(), root.join("real.png").display());
    assert!(!files_in(&comfy_out).iter().any(|n| n.starts_with(&job)), "nothing of the job left in ComfyUI's output");
    assert!(files_in(&root.join("cache/media")).is_empty(), "delivered, so the temp folder is empty");

    // A chat comes next: ComfyUI gives the card back.
    let m = gw.gateway.media.as_ref().unwrap();
    assert!(m.comfy.dirty.load(std::sync::atomic::Ordering::SeqCst));
    m.comfy.free_for_chat();
    assert!(!m.comfy.dirty.load(std::sync::atomic::Ordering::SeqCst));
    drop(gw);
}
