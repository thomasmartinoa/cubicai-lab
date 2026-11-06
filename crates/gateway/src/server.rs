//! The listener: who gets in (checked before TLS, and again per request), connection slots, the
//! routes, and the mDNS announcement that lets a paired phone find the PC again.

use crate::access::{self, Host};
use crate::activity::Activity;
use crate::http::{error, Conn, Request};
use crate::identity::{token_matches, Identity};
use crate::ollama::Ollama;
use crate::settings::Settings;
use crate::{log, Config};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Everything a request needs, shared by every connection.
pub struct Gateway {
    pub config: Config,
    pub token: String,
    pub fingerprint: String,
    pub port: u16,
    pub settings: Settings,
    pub activity: Activity,
    pub ollama: Ollama,
    pub host: Host,
    vram: OnceLock<Option<u64>>,
    slots: Mutex<(usize, HashMap<IpAddr, usize>)>,
}

impl Gateway {
    pub fn new(config: Config, id: &Identity, port: u16) -> Gateway {
        Gateway {
            settings: Settings::new(&config.dir),
            ollama: Ollama::new(&config.ollama),
            config,
            token: id.token.clone(),
            fingerprint: id.fingerprint.clone(),
            port,
            activity: Activity::default(),
            host: Host::default(),
            vram: OnceLock::new(),
            slots: Mutex::new((0, HashMap::new())),
        }
    }

    pub fn allowed(&self, ip: IpAddr) -> bool {
        access::allowed(ip, self.settings.remote(), &self.host.lan6())
    }

    /// The Tailscale URL while remote access is on (the phone learns it from /v1/models, so
    /// switching remote on needs no re-pairing).
    pub fn remote_url(&self) -> Option<String> {
        if !self.settings.remote() {
            return None;
        }
        self.host.tailscale_ip().map(|ip| format!("https://{ip}:{}", self.port))
    }

    pub fn lan6_url(&self) -> Option<String> {
        access::lan6_url(&self.host.lan6(), self.port)
    }

    fn vram(&self) -> Option<u64> {
        *self.vram.get_or_init(crate::ollama::gpu_memory)
    }

    fn take_slot(&self, ip: IpAddr) -> bool {
        let mut s = self.slots.lock().unwrap();
        let n = s.1.get(&ip).copied().unwrap_or(0);
        if n >= crate::MAX_PER_ADDRESS || s.0 >= crate::MAX_CONNECTIONS {
            return false;
        }
        s.0 += 1;
        s.1.insert(ip, n + 1);
        true
    }

    fn free_slot(&self, ip: IpAddr) {
        let mut s = self.slots.lock().unwrap();
        s.0 = s.0.saturating_sub(1);
        match s.1.get(&ip).copied() {
            Some(n) if n > 1 => {
                s.1.insert(ip, n - 1);
            }
            _ => {
                s.1.remove(&ip);
            }
        }
    }

    /// Connections open now (total, from this address), for tests and the panel.
    pub fn open_connections(&self, ip: IpAddr) -> (usize, usize) {
        let s = self.slots.lock().unwrap();
        (s.0, s.1.get(&ip).copied().unwrap_or(0))
    }
}

/// Streams whose socket timeouts can change once a request is authorised.
pub trait Timeouts {
    fn set_timeout(&mut self, secs: u64);
}

type Tls = rustls::StreamOwned<rustls::ServerConnection, TcpStream>;

impl Timeouts for Tls {
    fn set_timeout(&mut self, secs: u64) {
        let _ = self.sock.set_read_timeout(Some(Duration::from_secs(secs)));
        let _ = self.sock.set_write_timeout(Some(Duration::from_secs(secs)));
    }
}

pub fn tls_config(id: &Identity) -> Result<Arc<rustls::ServerConfig>, String> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(id.certs.clone(), id.key.clone_key())
        .map_err(|e| format!("the certificate or key can't be used: {e}"))?;
    Ok(Arc::new(cfg))
}

/// A running gateway. Dropping it (or `stop`) closes the listener and the mDNS announcement;
/// connections already open finish on their own.
pub struct Running {
    pub gateway: Arc<Gateway>,
    pub port: u16,
    stop: Arc<AtomicBool>,
    accept: Option<std::thread::JoinHandle<()>>,
    mdns: Option<std::process::Child>,
}

impl Running {
    pub fn stop(&mut self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            return;
        }
        // Wake the blocking accept.
        let _ = TcpStream::connect_timeout(&std::net::SocketAddr::from(([127, 0, 0, 1], self.port)), Duration::from_secs(1));
        if let Some(t) = self.accept.take() {
            let _ = t.join();
        }
        if let Some(mut m) = self.mdns.take() {
            let _ = m.kill();
            let _ = m.wait();
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Starts listening. `bind` "::" takes IPv6 and IPv4 together. Port 0 picks a free one.
pub fn start(config: Config, id: &Identity, bind: &str, port: u16, announce_mdns: bool) -> Result<Running, String> {
    let tls = tls_config(id)?;
    let addr = if bind.contains(':') { format!("[{bind}]:{port}") } else { format!("{bind}:{port}") };
    let listener = TcpListener::bind(&addr).map_err(|e| format!("can't listen on {addr}: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let gw = Arc::new(Gateway::new(config, id, port));
    let stop = Arc::new(AtomicBool::new(false));
    let (g, s) = (gw.clone(), stop.clone());
    let accept = std::thread::Builder::new().name("gateway-accept".into()).spawn(move || accept_loop(g, listener, tls, s)).map_err(|e| e.to_string())?;
    let mdns = if announce_mdns { announce(port, &id.fingerprint) } else { None };
    Ok(Running { gateway: gw, port, stop, accept: Some(accept), mdns })
}

fn accept_loop(gw: Arc<Gateway>, listener: TcpListener, tls: Arc<rustls::ServerConfig>, stop: Arc<AtomicBool>) {
    for incoming in listener.incoming() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        let Ok(tcp) = incoming else { continue };
        let Ok(peer) = tcp.peer_addr() else { continue };
        let ip = access::unmap(peer.ip());
        // Dropped before TLS: outsiders can't even see a certificate.
        if !gw.allowed(ip) {
            log(&format!("dropped connection from {ip} (not home network{})", if gw.settings.remote() { "" } else { " / remote off" }));
            gw.activity.reject(&ip.to_string());
            continue;
        }
        if !gw.take_slot(ip) {
            continue;
        }
        let (g, t) = (gw.clone(), tls.clone());
        let spawned = std::thread::Builder::new().name("gateway-conn".into()).spawn(move || {
            connection(&g, t, tcp, ip);
            g.free_slot(ip);
        });
        if spawned.is_err() {
            gw.free_slot(ip);
        }
    }
}

/// One connection: the TLS handshake (here on the worker, so a slow client can't block accept),
/// then requests until either side closes.
fn connection(gw: &Gateway, tls: Arc<rustls::ServerConfig>, tcp: TcpStream, ip: IpAddr) {
    let _ = tcp.set_nodelay(true);
    let Ok(sc) = rustls::ServerConnection::new(tls) else { return };
    let mut stream = rustls::StreamOwned::new(sc, tcp);
    stream.set_timeout(crate::PRE_AUTH_TIMEOUT_S);
    while stream.conn.is_handshaking() {
        if stream.conn.complete_io(&mut stream.sock).is_err() {
            return;
        }
    }
    let mut conn = Conn::new(stream);
    loop {
        let req = match conn.next_request() {
            Ok(Some(r)) => r,
            _ => return,
        };
        match handle(gw, &mut conn, &req, ip) {
            Ok(true) => continue,
            _ => {
                let s = conn.stream();
                s.conn.send_close_notify();
                let _ = s.flush();
                return;
            }
        }
    }
}

const NOT_YET_MEDIA: &str = "Making things on the PC isn't in this gateway yet.";
const NOT_YET_VOICE: &str = "voice support in this gateway";

/// One request. Returns whether the connection may carry another.
pub fn handle<S: Read + Write + Timeouts>(gw: &Gateway, conn: &mut Conn<S>, req: &Request, ip: IpAddr) -> io::Result<bool> {
    let ip_s = ip.to_string();
    if !gw.allowed(ip) {
        // A kept-alive connection opened before remote access was switched off: the check holds per request too.
        return conn.json(req, 403, &error("Remote access is off on your PC. Turn it on in CubicAI Server to connect over Tailscale."));
    }
    if !token_matches(req.header("Authorization"), &gw.token) {
        log(&format!("rejected unauthorized request from {ip_s}"));
        gw.activity.reject(&ip_s);
        std::thread::sleep(Duration::from_secs(1));
        return conn.json(req, 401, &error("Not paired: invalid token"));
    }
    let path = req.path().to_string();
    match req.method.as_str() {
        "GET" => match path.as_str() {
            "/health" | "/v1/health" => conn.json(req, 200, &json!({"ok": true})),
            "/v1/cubicai/status" => conn.json(req, 200, &gw.activity.snapshot()),
            "/v1/models" => models(gw, conn, req),
            "/v1/media/generators" => conn.json(req, 200, &json!({"generators": [], "comfy_running": false})),
            p if p.starts_with("/v1/media/") => conn.json(req, 404, &error("This result is no longer on your PC.")),
            "/v1/voice/voices" => conn.json(req, 200, &json!({"ready": false, "missing": NOT_YET_VOICE, "voices": []})),
            "/v1/voice/call" | "/v1/voice/preview" => conn.json(req, 503, &error(&format!("Voice calls need {NOT_YET_VOICE} on your PC."))),
            _ => conn.json(req, 404, &error("Not found")),
        },
        "DELETE" => conn.json(req, 404, &error("No such job")),
        "POST" => post(gw, conn, req, &path, ip),
        _ => conn.json(req, 404, &error("Not found")),
    }
}

fn models<S: Read + Write>(gw: &Gateway, conn: &mut Conn<S>, req: &Request) -> io::Result<bool> {
    match gw.ollama.list_models(gw.settings.num_ctx(gw.config.default_ctx), gw.vram()) {
        Ok(data) => conn.json(
            req,
            200,
            // The phone learns the Tailscale and IPv6 addresses from here, so it needs no re-pairing.
            &json!({"object": "list", "data": data, "cubic_remote": gw.remote_url(), "cubic_lan6": gw.lan6_url(), "cubic_voice": false}),
        ),
        Err(e) => conn.json(req, 502, &error(&format!("Can't reach Ollama: {e}"))),
    }
}

fn post<S: Read + Write + Timeouts>(gw: &Gateway, conn: &mut Conn<S>, req: &Request, path: &str, ip: IpAddr) -> io::Result<bool> {
    if path == "/v1/media/save" {
        return conn.json(req, 503, &error(NOT_YET_MEDIA));
    }
    if path != "/v1/chat/completions" && path != "/v1/media/jobs" {
        return conn.json(req, 404, &error("Not found"));
    }
    let len = match req.content_length() {
        Err(_) => return conn.json(req, 400, &error("Bad Content-Length")),
        Ok(n) => n.unwrap_or(0),
    };
    if len == 0 || len > crate::MAX_BODY {
        return conn.json(req, 413, &error("Request too large"));
    }
    let raw = conn.read_body(len)?;
    let Ok(body) = serde_json::from_slice::<Value>(&raw) else { return conn.json(req, 400, &error("Invalid JSON")) };
    if path == "/v1/media/jobs" {
        return conn.json(req, 503, &error(NOT_YET_MEDIA));
    }
    // Authorised: allow long generations.
    conn.stream().set_timeout(900);
    let model = body["model"].as_str().unwrap_or("").to_string();
    let web = body["cubic_web_search"].as_bool().unwrap_or(false);
    log(&format!("chat from {ip} model={model} search={web}"));
    let rid = gw.activity.begin(&ip.to_string(), &model, web);
    if conn.start_events().is_err() {
        gw.activity.end(rid, "stopped", 0, 0.0);
        return Ok(false);
    }
    let o = crate::chat::stream(gw, &body, &ip.to_string(), rid, conn);
    gw.activity.end(rid, o.outcome, o.tokens, o.gen_seconds);
    Ok(false)
}

/// Announces the gateway on the home network through avahi, so a paired phone can find the PC again
/// after the router hands it a new address. Only the certificate fingerprint is published (the phone
/// checks it before trusting anything); the token never leaves pairing. Without avahi it's skipped.
pub fn announce(port: u16, fingerprint: &str) -> Option<std::process::Child> {
    let tool = std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("avahi-publish-service")).find(|f| f.is_file()));
    let Some(tool) = tool else {
        log("mDNS: avahi-publish-service not found; the phone can't find the PC by itself if its address changes");
        return None;
    };
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").map(|h| h.trim().to_string()).unwrap_or_else(|_| "this PC".into());
    let mut cmd = std::process::Command::new(tool);
    cmd.arg(format!("CubicAI on {host}")).arg("_cubicai._tcp").arg(port.to_string()).arg(format!("fp={fingerprint}"));
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    // The announcement ends with the gateway, even if the gateway is killed.
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut cmd, || {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    match cmd.spawn() {
        Ok(c) => {
            log("mDNS: announced as _cubicai._tcp");
            Some(c)
        }
        Err(e) => {
            log(&format!("mDNS: couldn't announce ({e})"));
            None
        }
    }
}
