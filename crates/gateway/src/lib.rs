//! CubicAI gateway: the private, authenticated HTTPS server the CubicAI phone app pairs with to use
//! this PC's models. A Rust port of `cubicai_server.py`, speaking exactly the same protocol, so a
//! phone paired with the Python gateway keeps working (same identity files, same fingerprint).
//!
//! - Only the home network (private addresses and this PC's own IPv6 prefixes) and, while remote
//!   access is on, Tailscale may connect. Everyone else is dropped before the TLS handshake.
//! - Every request needs the pairing token; TLS 1.3 with a self-signed certificate the app pins.
//! - Web search runs here (a local SearXNG), so the phone never talks to search engines.
//! - Message content is never logged or stored.

pub mod access;
pub mod activity;
pub mod chat;
pub mod http;
pub mod identity;
pub mod jobs;
pub mod local;
pub mod media;
pub mod ollama;
pub mod search;
pub mod server;
pub mod settings;
pub mod time;
pub mod voice;
pub mod workflow;

use std::path::PathBuf;

pub const DEFAULT_PORT: u16 = 8443;
pub const MAX_BODY: usize = 64 * 1024 * 1024;
/// A rejected request's body is read (and dropped) up to this size.
pub const DRAIN_LIMIT: usize = 1024 * 1024;
pub const MAX_CONNECTIONS: usize = 64;
/// So one misbehaving device on the network can't take every slot.
pub const MAX_PER_ADDRESS: usize = 12;
pub const PRE_AUTH_TIMEOUT_S: u64 = 30;

/// Where things are. Defaults match the Python gateway, and the same environment variables move them.
#[derive(Clone, Debug)]
pub struct Config {
    /// Identity (cert, key, token) and settings.json.
    pub dir: PathBuf,
    pub ollama: String,
    pub searxng: String,
    /// The context a request gets when neither the Modelfile nor settings.json sets one.
    pub default_ctx: u64,
    /// Making things with ComfyUI (None: the phone is told there's nothing to make).
    pub media: Option<media::Paths>,
    /// Voice calls (None: the phone is told they aren't set up).
    pub voice: Option<std::sync::Arc<voice::Voice>>,
}

/// Where the identity and settings.json live (`CUBICAI_CONFIG_DIR`, else ~/.config/cubicai-server).
pub fn config_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    std::env::var("CUBICAI_CONFIG_DIR").map(|d| expand(&d, &home)).unwrap_or_else(|_| home.join(".config/cubicai-server"))
}

impl Config {
    pub fn from_env() -> Config {
        let dir = config_dir();
        Config {
            dir,
            ollama: std::env::var("CUBICAI_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            searxng: std::env::var("CUBICAI_SEARXNG_URL").unwrap_or_else(|_| "http://127.0.0.1:8888".into()),
            default_ctx: std::env::var("CUBICAI_NUM_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(16384),
            media: Some(media::Paths::from_env()),
            voice: Some(voice::Voice::new(voice::Paths::from_env())),
        }
    }
}

fn expand(p: &str, home: &std::path::Path) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => home.join(rest),
        None => PathBuf::from(p),
    }
}

/// One line on stdout (journald adds the time when run as a service). Never message content.
pub fn log(msg: &str) {
    println!("{} {msg}", time::stamp());
}
