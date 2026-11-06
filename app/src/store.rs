//! Where things live on disk, the settings file, and chats. Chats use the phone app's JSON format
//! (one file per chat, camelCase fields), so the two can share them later.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Clone)]
pub struct Paths {
    pub config: PathBuf,
    pub data: PathBuf,
    pub chats: PathBuf,
    pub models: PathBuf,
    pub llama: PathBuf,
    pub steering: PathBuf,
}

impl Paths {
    pub fn new() -> Paths {
        let dirs = directories::ProjectDirs::from("dev", "martin", "cubicai-lab").expect("no home directory");
        let config = dirs.config_dir().to_path_buf();
        let data = dirs.data_dir().to_path_buf();
        let p = Paths {
            chats: data.join("chats"),
            models: data.join("models"),
            llama: data.join("llama.cpp"),
            steering: data.join("steering"),
            config,
            data,
        };
        for d in [&p.config, &p.data, &p.chats, &p.models, &p.llama, &p.steering] {
            let _ = std::fs::create_dir_all(d);
        }
        p
    }
}

/// Writes through a temporary file so a crash never leaves half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

// --- Settings ------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
#[serde(rename_all = "lowercase")]
pub enum ThemeChoice {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct Persona {
    pub id: String,
    pub name: String,
    pub instructions: String,
    #[serde(default)]
    pub summary: String,
}

/// How llama-server runs one model. `None` means "let llama.cpp decide" (it fits to VRAM itself).
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(default)]
pub struct RunPreset {
    pub ctx: u32,
    pub gpu_layers: Option<u32>,
    pub cpu_moe_layers: Option<u32>,
    pub flash_attn: FlashAttn,
    pub kv_type: String,
    pub threads: Option<u32>,
    pub ubatch: u32,
    pub use_mmproj: bool,
    pub extra_args: String,
    /// A steering vector applied while this model runs (Steering Lab → "Use in chat").
    pub steer: Option<SteerSetting>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct SteerSetting {
    pub vector: String,
    pub strength: f32,
}

impl Default for RunPreset {
    fn default() -> Self {
        RunPreset {
            ctx: 16384,
            gpu_layers: None,
            cpu_moe_layers: None,
            flash_attn: FlashAttn::Auto,
            kv_type: "f16".into(),
            threads: None,
            ubatch: 512,
            use_mmproj: true,
            extra_args: String::new(),
            steer: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum FlashAttn {
    #[default]
    Auto,
    On,
    Off,
}

impl FlashAttn {
    pub fn arg(self) -> &'static str {
        match self {
            FlashAttn::Auto => "auto",
            FlashAttn::On => "on",
            FlashAttn::Off => "off",
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct InstalledBuild {
    pub tag: String,
    pub variant: String,
    pub dir: PathBuf,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BenchResult {
    pub model: String,
    pub at: u64,
    pub settings: String,
    pub prompt_tps: f64,
    pub gen_tps: f64,
    pub build: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Settings {
    pub theme: ThemeChoice,
    pub provider: String,
    /// Model chosen per provider (for "pc", a library model key).
    pub models: HashMap<String, String>,
    pub custom_base_url: String,
    pub about_me: String,
    pub persona: String,
    pub custom_personas: Vec<Persona>,
    pub model_dirs: Vec<PathBuf>,
    /// Ollama's model store; found automatically when empty.
    pub ollama_dir: Option<PathBuf>,
    pub show_ollama: bool,
    pub llama: Option<InstalledBuild>,
    /// A llama.cpp build of the user's own (folder holding llama-server), used instead.
    pub llama_custom_dir: Option<PathBuf>,
    pub port: u16,
    pub presets: HashMap<String, RunPreset>,
    pub benches: Vec<BenchResult>,
    /// VRAM to plan with, when the GPU can't be asked.
    pub vram_override_mb: Option<u64>,
    pub thoughts_open: Option<bool>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: ThemeChoice::System,
            provider: "pc".into(),
            models: HashMap::new(),
            custom_base_url: String::new(),
            about_me: String::new(),
            persona: "none".into(),
            custom_personas: vec![],
            model_dirs: vec![],
            ollama_dir: None,
            show_ollama: true,
            llama: None,
            llama_custom_dir: None,
            port: 8193,
            presets: HashMap::new(),
            benches: vec![],
            vram_override_mb: None,
            thoughts_open: None,
        }
    }
}

impl Settings {
    pub fn load(paths: &Paths) -> Settings {
        std::fs::read(paths.config.join("settings.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, paths: &Paths) {
        if let Ok(b) = serde_json::to_vec_pretty(self) {
            let _ = write_atomic(&paths.config.join("settings.json"), &b);
        }
    }

    pub fn preset(&self, model: &str) -> RunPreset {
        self.presets.get(model).cloned().unwrap_or_default()
    }
}

// --- Chats ---------------------------------------------------------------------------------------

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct Attachment {
    pub kind: String,
    pub name: String,
    pub path: String,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

fn is_false(v: &bool) -> bool {
    !*v
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub attachments: Vec<Attachment>,
    pub citations: Vec<String>,
    pub stopped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    #[serde(skip_serializing_if = "is_zero")]
    pub prompt_tokens: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "is_zero")]
    pub took: u64,
    /// Generation speed reported by llama-server (desktop only; the phone ignores it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tps: Option<f64>,
    #[serde(skip_serializing_if = "is_false")]
    pub spoken: bool,
    /// Phone-only fields (media jobs) kept as they are.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<serde_json::Value>,
}

impl Default for Message {
    fn default() -> Self {
        Message {
            id: new_id(),
            role: "user".into(),
            content: String::new(),
            reasoning: String::new(),
            attachments: vec![],
            citations: vec![],
            stopped: false,
            error: None,
            model: None,
            persona: None,
            prompt_tokens: 0,
            output_tokens: 0,
            took: 0,
            tps: None,
            spoken: false,
            media: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub updated_at: u64,
    pub persona_id: String,
    pub provider_id: Option<String>,
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_path: Option<String>,
}

impl Default for Chat {
    fn default() -> Self {
        Chat {
            id: new_id(),
            title: "New chat".into(),
            messages: vec![],
            updated_at: now_ms(),
            persona_id: "none".into(),
            provider_id: None,
            model: None,
            face_path: None,
            voice_path: None,
        }
    }
}

pub struct ChatStore {
    dir: PathBuf,
}

impl ChatStore {
    pub fn new(paths: &Paths) -> Self {
        ChatStore { dir: paths.chats.clone() }
    }

    /// All chats, newest first.
    pub fn load_all(&self) -> Vec<Chat> {
        let mut chats: Vec<Chat> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| std::fs::read(e.path()).ok())
            .filter_map(|b| serde_json::from_slice::<Chat>(&b).ok())
            .collect();
        chats.sort_by_key(|c| std::cmp::Reverse(c.updated_at));
        chats
    }

    pub fn save(&self, chat: &Chat) {
        if chat.messages.is_empty() {
            return;
        }
        if let Ok(b) = serde_json::to_vec(chat) {
            let _ = write_atomic(&self.file(&chat.id), &b);
        }
    }

    pub fn delete(&self, id: &str) {
        let _ = std::fs::remove_file(self.file(id));
    }

    fn file(&self, id: &str) -> PathBuf {
        // Ids are UUIDs; anything else is reduced to safe characters.
        let safe: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect();
        self.dir.join(format!("{safe}.json"))
    }
}
