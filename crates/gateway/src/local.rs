//! For the control panel on this PC: the gateway's status over its own TLS (pinned to its
//! certificate, as the phone does), and installing the gateway as the `cubicai-server` user service.

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub const SERVICE: &str = "cubicai-server";

/// The systemd unit: the same name and sandbox the Python gateway had.
pub const UNIT: &str = r#"[Unit]
Description=CubicAI private gateway (phone app -> Ollama)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=%h/.local/bin/cubicai-gateway serve
Restart=on-failure
RestartSec=5
# The gateway only needs to read its code and models and write its own folders.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=%h/.config/cubicai-server -%h/.cache/cubicai-server
# Media: the gateway places inputs for ComfyUI, collects its results, and stores what the phone saves.
ReadWritePaths=-/mnt/storage/games/AI_local/ComfyUI/input/cubicai -/mnt/storage/games/AI_local/ComfyUI/output/cubicai -/mnt/storage/games/AI_local/CubicAI
# Bots: their chats and memory, and the workspace folder they share (install makes both).
ReadWritePaths=-%h/.local/share/cubicai-server/bots -%h/.local/share/cubicai-server/workspace
UMask=0077

[Install]
WantedBy=default.target
"#;

/// ComfyUI as its own unit, which the gateway starts when a job needs it and stops when idle
/// (`jobs::Comfy`). Only reachable from this PC; no metadata in files, so prompts don't travel
/// inside the pictures; only warnings and errors reach the journal.
pub const COMFY_UNIT: &str = r#"[Unit]
Description=ComfyUI for CubicAI (started on demand by cubicai-server, stopped when idle)

[Service]
# Only reachable from this PC; the phone goes through cubicai-server. No metadata in files, so prompts
# don't travel inside the images, and only warnings and errors reach the journal.
ExecStart=/mnt/storage/games/AI_local/tools/start-comfyui.sh --listen 127.0.0.1 --port 8188 --preview-method auto --disable-metadata --disable-auto-launch --verbose WARNING
TimeoutStopSec=20
"#;

/// Accepts exactly one certificate, by its SHA-256 fingerprint.
#[derive(Debug)]
pub struct Pin(pub String);

impl ServerCertVerifier for Pin {
    fn verify_server_cert(&self, end: &CertificateDer<'_>, _: &[CertificateDer<'_>], _: &ServerName<'_>, _: &[u8], _: UnixTime) -> Result<ServerCertVerified, rustls::Error> {
        if crate::identity::fingerprint(end.as_ref()) == self.0 {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General("not this PC's gateway".into()))
        }
    }
    fn verify_tls12_signature(&self, _: &[u8], _: &CertificateDer<'_>, _: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::General("TLS 1.3 only".into()))
    }
    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer<'_>, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &rustls::crypto::ring::default_provider().signature_verification_algorithms)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider().signature_verification_algorithms.supported_schemes()
    }
}

/// GET `path` from the gateway on this PC; None unless it's really ours and answers 200.
pub fn get(port: u16, fingerprint: &str, token: &str, path: &str) -> Option<Value> {
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .ok()?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pin(fingerprint.to_string())))
        .with_no_client_auth();
    let tcp = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(3)).ok()?;
    tcp.set_read_timeout(Some(Duration::from_secs(3))).ok()?;
    tcp.set_write_timeout(Some(Duration::from_secs(3))).ok()?;
    let conn = rustls::ClientConnection::new(Arc::new(config), ServerName::try_from("cubicai").ok()?.to_owned()).ok()?;
    let mut s = rustls::StreamOwned::new(conn, tcp);
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n").as_bytes()).ok()?;
    let mut raw = vec![];
    let _ = s.read_to_end(&mut raw); // the gateway may close without close_notify
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text.split_once("\r\n\r\n")?;
    if head.split_whitespace().nth(1) != Some("200") {
        return None;
    }
    serde_json::from_str(body).ok()
}

pub fn systemctl(args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new("systemctl").arg("--user").args(args).output().map_err(|e| format!("systemctl: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

pub fn unit_path() -> PathBuf {
    home().join(".config/systemd/user/cubicai-server.service")
}

pub fn comfy_unit_path() -> PathBuf {
    home().join(".config/systemd/user/cubicai-comfyui.service")
}

pub fn installed_binary() -> PathBuf {
    home().join(".local/bin/cubicai-gateway")
}

/// Which gateway the installed unit runs: "Rust", "Python", or None when there's no unit.
pub fn service_kind(unit: &Path) -> Option<&'static str> {
    let text = std::fs::read_to_string(unit).ok()?;
    let exec = text.lines().find(|l| l.trim_start().starts_with("ExecStart="))?;
    Some(if exec.contains("cubicai-gateway") {
        "Rust"
    } else if exec.contains("cubicai_server.py") {
        "Python"
    } else {
        "other"
    })
}

/// Copies `binary` to ~/.local/bin (a rename, so a running gateway keeps its own file), writes
/// the unit (and ComfyUI's, unless one is there already: it may point at another ComfyUI) and
/// (re)starts the service.
pub fn install(binary: &Path) -> Result<(), String> {
    let target = installed_binary();
    std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
    if binary.canonicalize().ok() != target.canonicalize().ok() {
        let tmp = target.with_extension("new");
        std::fs::copy(binary, &tmp).map_err(|e| format!("copying the gateway: {e}"))?;
        std::fs::rename(&tmp, &target).map_err(|e| e.to_string())?;
    }
    // The service can't make folders in the (read-only) home itself, so the Bots' folders are made here.
    let share = home().join(".local/share/cubicai-server");
    for d in ["bots", "workspace"] {
        std::fs::create_dir_all(share.join(d)).map_err(|e| format!("making {d}: {e}"))?;
        let _ = std::fs::set_permissions(share.join(d), std::os::unix::fs::PermissionsExt::from_mode(0o700));
    }
    let unit = unit_path();
    std::fs::create_dir_all(unit.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&unit, UNIT).map_err(|e| e.to_string())?;
    if !comfy_unit_path().exists() {
        std::fs::write(comfy_unit_path(), COMFY_UNIT).map_err(|e| e.to_string())?;
    }
    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", SERVICE])?;
    systemctl(&["restart", SERVICE])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn which_gateway_a_unit_runs() {
        let dir = std::env::temp_dir().join(format!("cubic-gw-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("u.service");
        assert_eq!(service_kind(&f), None, "no unit");
        std::fs::write(&f, UNIT).unwrap();
        assert_eq!(service_kind(&f), Some("Rust"));
        std::fs::write(&f, "[Service]\nExecStart=%h/x/.venv/bin/python %h/x/server/cubicai_server.py serve\n").unwrap();
        assert_eq!(service_kind(&f), Some("Python"));
        std::fs::write(&f, "[Service]\nExecStart=/usr/bin/true\n").unwrap();
        assert_eq!(service_kind(&f), Some("other"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_unit_keeps_the_python_sandbox() {
        // Everything the Python unit allowed, plus the Bots' two folders: home stays read-only (the
        // voice models and ONNX Runtime are only read), writes go to our own folders.
        for line in ["ProtectHome=read-only", "ProtectSystem=strict", "NoNewPrivileges=yes", "PrivateTmp=yes", "UMask=0077", "ReadWritePaths=%h/.config/cubicai-server -%h/.cache/cubicai-server"] {
            assert!(UNIT.lines().any(|l| l == line), "{line}");
        }
        assert!(UNIT.lines().any(|l| l == "ReadWritePaths=-%h/.local/share/cubicai-server/bots -%h/.local/share/cubicai-server/workspace"));
        assert_eq!(UNIT.lines().filter(|l| l.starts_with("ReadWritePaths=")).count(), 3);
    }

    #[test]
    fn comfyui_stays_private_and_keeps_no_prompts() {
        let exec = COMFY_UNIT.lines().find(|l| l.starts_with("ExecStart=")).unwrap();
        for flag in ["--listen 127.0.0.1", "--port 8188", "--disable-metadata", "--disable-auto-launch"] {
            assert!(exec.contains(flag), "{flag}");
        }
        // The gateway knows it by this name.
        assert!(comfy_unit_path().ends_with("cubicai-comfyui.service"));
    }

    #[test]
    fn status_only_from_our_own_gateway() {
        // Nothing listens on port 1: no answer, not an error.
        assert_eq!(get(1, "00", "t", "/v1/cubicai/status"), None);
    }
}
