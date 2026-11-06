//! API keys in the OS credential store (Keychain, Credential Manager, Secret Service). When no
//! store is available (a bare Linux session), they fall back to a file only the user can read.

use crate::store::{write_atomic, Paths};
use std::collections::HashMap;
use std::path::PathBuf;

const SERVICE: &str = "cubicai-lab";

pub struct Secrets {
    cache: HashMap<String, String>,
    keyring: bool,
    fallback: PathBuf,
}

impl Secrets {
    pub fn new(paths: &Paths) -> Self {
        let keyring = keyring::Entry::store_status().is_ok() && probe();
        let fallback = paths.config.join("secrets.json");
        let mut s = Secrets { cache: HashMap::new(), keyring, fallback };
        if !keyring {
            s.cache = std::fs::read(&s.fallback).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        }
        s
    }

    pub fn where_stored(&self) -> &'static str {
        if self.keyring {
            "your system keyring"
        } else {
            "a private file in the config folder (no system keyring found)"
        }
    }

    pub fn get(&mut self, name: &str) -> String {
        if let Some(v) = self.cache.get(name) {
            return v.clone();
        }
        let v = if self.keyring {
            keyring::Entry::new(SERVICE, name).and_then(|e| e.get_password()).unwrap_or_default()
        } else {
            String::new()
        };
        self.cache.insert(name.to_string(), v.clone());
        v
    }

    pub fn set(&mut self, name: &str, value: &str) {
        let value = value.trim().to_string();
        self.cache.insert(name.to_string(), value.clone());
        if self.keyring {
            if let Ok(e) = keyring::Entry::new(SERVICE, name) {
                let _ = if value.is_empty() { e.delete_credential() } else { e.set_password(&value) };
            }
        } else {
            let map: HashMap<&String, &String> = self.cache.iter().filter(|(_, v)| !v.is_empty()).collect();
            if let Ok(b) = serde_json::to_vec(&map) {
                let _ = write_atomic(&self.fallback, &b);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&self.fallback, std::fs::Permissions::from_mode(0o600));
                }
            }
        }
    }
}

/// Opening the store can succeed while the Secret Service isn't actually running; a read tells.
fn probe() -> bool {
    match keyring::Entry::new(SERVICE, "probe").and_then(|e| e.get_password()) {
        Ok(_) => true,
        Err(keyring::Error::NoEntry) => true,
        Err(_) => false,
    }
}
