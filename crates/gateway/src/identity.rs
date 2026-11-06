//! The gateway's identity: a self-signed TLS certificate the phone pins by its SHA-256 fingerprint,
//! and the pairing token every request carries. The same files as the Python gateway
//! (`cert.pem`, `key.pem`, `token` in the config folder), so a paired phone stays paired.

use base64::Engine;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

pub struct Identity {
    pub certs: Vec<CertificateDer<'static>>,
    pub key: PrivateKeyDer<'static>,
    pub token: String,
    /// Lower-case hex SHA-256 of the certificate (DER), what the phone pins and mDNS announces.
    pub fingerprint: String,
}

/// Loads the identity, making whatever is missing: a P-256 certificate for ten years (with openssl,
/// as the Python gateway does) and a token.
pub fn load(dir: &Path) -> Result<Identity, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    let (cert, key, token) = (dir.join("cert.pem"), dir.join("key.pem"), dir.join("token"));
    if !cert.exists() || !key.exists() {
        let out = std::process::Command::new("openssl")
            .args(["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-keyout"])
            .arg(&key)
            .arg("-out")
            .arg(&cert)
            .args(["-days", "3650", "-subj", "/CN=cubicai-server"])
            .output()
            .map_err(|e| format!("openssl isn't available to make the certificate: {e}"))?;
        if !out.status.success() {
            return Err(format!("openssl failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        let _ = std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600));
    }
    if !token.exists() {
        rotate_token(dir)?;
    }
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(&cert)
        .map_err(|e| format!("{}: {e}", cert.display()))?
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{}: {e}", cert.display()))?;
    let first = certs.first().ok_or_else(|| format!("{} has no certificate", cert.display()))?;
    let key = PrivateKeyDer::from_pem_file(&key).map_err(|e| format!("{}: {e}", key.display()))?;
    let token = std::fs::read_to_string(&token).map_err(|e| format!("{}: {e}", token.display()))?.trim().to_string();
    Ok(Identity { fingerprint: fingerprint(first), certs, key, token })
}

pub fn fingerprint(der: &[u8]) -> String {
    hex(ring::digest::digest(&ring::digest::SHA256, der).as_ref())
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A new token (un-pairs every phone): 32 random bytes, URL-safe base64 like Python's token_urlsafe.
pub fn rotate_token(dir: &Path) -> Result<String, String> {
    let mut raw = [0u8; 32];
    ring::rand::SecureRandom::fill(&ring::rand::SystemRandom::new(), &mut raw).map_err(|_| "no randomness available".to_string())?;
    let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw);
    let path = dir.join("token");
    let tmp = dir.join("token.tmp");
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&tmp).map_err(|e| e.to_string())?;
    f.write_all(token.as_bytes()).map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(token)
}

/// Constant-time token check (only the length can leak).
pub fn token_matches(header: Option<&str>, token: &str) -> bool {
    let want = format!("Bearer {token}");
    let got = header.unwrap_or("").as_bytes();
    got.len() == want.len() && got.iter().zip(want.as_bytes()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

/// `cubicai://pair?u=…&t=…&f=…[&r=…]`, in the Python gateway's order and encoding.
pub fn pairing_link(url: &str, token: &str, fingerprint: &str, remote: Option<&str>) -> String {
    let mut params = vec![("u", url), ("t", token), ("f", fingerprint)];
    if let Some(r) = remote {
        params.push(("r", r));
    }
    let q: Vec<String> = params.iter().map(|(k, v)| format!("{k}={}", form_encode(v))).collect();
    format!("cubicai://pair?{}", q.join("&"))
}

/// application/x-www-form-urlencoded (Python's quote_plus).
pub fn form_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out += &format!("%{b:02X}"),
        }
    }
    out
}

/// The reverse, for query strings (`+` is a space, bad escapes kept as they are).
pub fn form_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                Some(v) => {
                    out.push(v);
                    i += 2;
                }
                None => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
