//! settings.json in the config folder (remote access, context length, keep_alive), re-read when it
//! changes so a control panel's toggles apply to the running gateway at once.

use serde_json::{Map, Value};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::SystemTime;

pub struct Settings {
    file: PathBuf,
    cache: Mutex<(Option<SystemTime>, Value)>,
}

impl Settings {
    pub fn new(dir: &std::path::Path) -> Settings {
        Settings { file: dir.join("settings.json"), cache: Mutex::new((None, Value::Object(Map::new()))) }
    }

    pub fn get(&self) -> Value {
        let mtime = std::fs::metadata(&self.file).and_then(|m| m.modified()).ok();
        let mut c = self.cache.lock().unwrap();
        if mtime.is_none() {
            return Value::Object(Map::new());
        }
        if c.0 != mtime {
            let v = std::fs::read(&self.file).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).filter(Value::is_object);
            *c = (mtime, v.unwrap_or_else(|| Value::Object(Map::new())));
        }
        c.1.clone()
    }

    pub fn remote(&self) -> bool {
        self.get()["remote"].as_bool().unwrap_or(false)
    }

    pub fn num_ctx(&self, default: u64) -> u64 {
        self.get()["num_ctx"].as_u64().unwrap_or(default)
    }

    /// Ollama's keep_alive for chats ("15m" unless set).
    pub fn keep_alive(&self) -> Value {
        match &self.get()["keep_alive"] {
            Value::Null => Value::from("15m"),
            v => v.clone(),
        }
    }

    /// Merges `changes` into the file (written whole, 0600, atomically).
    pub fn save(&self, changes: Map<String, Value>) -> Result<(), String> {
        let mut all = match self.get() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        all.extend(changes);
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = self.file.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&Value::Object(all)).unwrap()).map_err(|e| e.to_string())?;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        std::fs::rename(&tmp, &self.file).map_err(|e| e.to_string())?;
        // Same-second writes can keep the mtime: forget the cache so the change is seen at once.
        self.cache.lock().unwrap().0 = None;
        Ok(())
    }
}
