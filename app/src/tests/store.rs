use super::TempDir;
use crate::store::{Chat, ChatStore, Message, Paths, Settings, ThemeChoice};

fn paths(d: &TempDir) -> Paths {
    let p = Paths { config: d.join("config"), data: d.join("data"), chats: d.join("chats"), models: d.join("models"), llama: d.join("llama"), steering: d.join("steering") };
    for dir in [&p.config, &p.data, &p.chats, &p.models, &p.llama] {
        std::fs::create_dir_all(dir).unwrap();
    }
    p
}

/// A chat exactly as the phone writes it (Models.kt `toJson`), including phone-only fields.
const PHONE_CHAT: &str = r#"{
  "id": "c1", "title": "From the phone", "updatedAt": 1762400000000, "personaId": "engineer",
  "providerId": "gemini", "model": "gemini-3.5-flash", "facePath": "/data/face.jpg",
  "messages": [
    {"id": "m1", "role": "user", "content": "Draw a cube", "reasoning": "",
     "attachments": [{"kind": "image", "name": "a.jpg", "path": "/data/a.jpg"}], "citations": [], "stopped": false},
    {"id": "m2", "role": "assistant", "content": "Here it is", "reasoning": "thinking", "attachments": [],
     "citations": ["https://example.com"], "stopped": false, "model": "gemini-3.5-flash", "persona": "Engineer",
     "promptTokens": 120, "outputTokens": 30, "took": 2100, "spoken": true,
     "media": {"generator": "zimage", "params": {"prompt": "a cube"}, "jobId": "j1", "seed": 42, "startedAt": 5}}
  ]
}"#;

#[test]
fn reads_and_keeps_phone_chats() {
    let chat: Chat = serde_json::from_str(PHONE_CHAT).unwrap();
    assert_eq!(chat.persona_id, "engineer");
    assert_eq!(chat.face_path.as_deref(), Some("/data/face.jpg"));
    let reply = &chat.messages[1];
    assert_eq!((reply.prompt_tokens, reply.output_tokens, reply.took), (120, 30, 2100));
    assert!(reply.spoken);
    assert_eq!(chat.messages[0].attachments[0].kind, "image");

    // Written back, every field the phone wrote is still there with the same value.
    let back: serde_json::Value = serde_json::to_value(&chat).unwrap();
    let orig: serde_json::Value = serde_json::from_str(PHONE_CHAT).unwrap();
    fn contains(have: &serde_json::Value, want: &serde_json::Value, at: &str) {
        match want {
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    contains(&have[k], v, &format!("{at}.{k}"));
                }
            }
            serde_json::Value::Array(a) => {
                for (i, v) in a.iter().enumerate() {
                    contains(&have[i], v, &format!("{at}[{i}]"));
                }
            }
            _ => assert_eq!(have, want, "{at}"),
        }
    }
    contains(&back, &orig, "chat");
}

#[test]
fn desktop_fields_dont_break_the_phone_format() {
    let m = Message { role: "assistant".into(), content: "x".into(), tps: Some(55.5), ..Default::default() };
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["tps"], 55.5);
    // Zero counters and empty options are left out, as the phone does.
    for k in ["promptTokens", "outputTokens", "took", "model", "error", "media", "spoken"] {
        assert!(v.get(k).is_none(), "{k} should be omitted");
    }
    // Unknown and missing fields are fine both ways.
    let m: Message = serde_json::from_str(r#"{"role":"user","content":"hi","futureField":1}"#).unwrap();
    assert_eq!(m.content, "hi");
    assert!(!m.id.is_empty());
}

#[test]
fn chat_store_round_trip() {
    let d = TempDir::new("chats");
    let p = paths(&d);
    let store = ChatStore::new(&p);
    let empty = Chat::default();
    store.save(&empty);
    assert!(store.load_all().is_empty(), "empty chats are not written");

    let mut old: Chat = serde_json::from_str(PHONE_CHAT).unwrap();
    old.updated_at = 1;
    let new = Chat { title: "Newer".into(), updated_at: 2, messages: vec![Message { content: "hi".into(), ..Default::default() }], ..Default::default() };
    store.save(&old);
    store.save(&new);
    std::fs::write(p.chats.join("broken.json"), b"{not json").unwrap();
    let all = store.load_all();
    assert_eq!(all.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(), ["Newer", "From the phone"], "newest first, broken skipped");

    store.delete(&old.id);
    assert_eq!(store.load_all().len(), 1);
    // Ids can't escape the folder.
    store.delete("../config/settings");
    assert!(p.chats.exists());
}

#[test]
fn settings_survive_and_fall_back() {
    let d = TempDir::new("settings");
    let p = paths(&d);
    assert_eq!(Settings::load(&p).port, 8193, "defaults without a file");
    let mut s = Settings { theme: ThemeChoice::Light, about_me: "I like Rust".into(), ..Default::default() };
    s.presets.insert("m".into(), crate::store::RunPreset { ctx: 4096, ..Default::default() });
    s.save(&p);
    let back = Settings::load(&p);
    assert_eq!(back.theme, ThemeChoice::Light);
    assert_eq!(back.about_me, "I like Rust");
    assert_eq!(back.preset("m").ctx, 4096);
    assert_eq!(back.preset("other").ctx, 16384);
    // Older files missing newer keys still load.
    std::fs::write(p.config.join("settings.json"), r#"{"about_me":"old"}"#).unwrap();
    let old = Settings::load(&p);
    assert_eq!((old.about_me.as_str(), old.port), ("old", 8193));
    std::fs::write(p.config.join("settings.json"), b"garbage").unwrap();
    assert_eq!(Settings::load(&p).about_me, "");
}

#[test]
fn behaviours_match_the_phone() {
    let ids: Vec<&str> = crate::personas::BUILT_IN.iter().map(|p| p.id).collect();
    assert_eq!(ids[0], "none");
    assert!(crate::personas::BUILT_IN[0].instructions.is_empty(), "None sends no system prompt");
    let mut uniq = ids.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), ids.len(), "ids are unique");
    // Every id the phone defines is here (the phone's Personas.kt is the source).
    let phone = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../cubicai/app/src/main/java/dev/martin/cubicai/data/Personas.kt"));
    if let Ok(src) = phone {
        for id in src.split("id = \"").skip(1).filter_map(|s| s.split('"').next()) {
            assert!(ids.contains(&id), "phone behaviour {id} missing on the desktop");
        }
    }
}
