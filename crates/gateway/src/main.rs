//! cubicai-gateway: the CubicAI phone app's private gateway to this PC.
//!
//!     cubicai-gateway serve [--bind ::] [--port 8443]   run the gateway (:: = IPv6 and IPv4)
//!     cubicai-gateway pair [--url https://HOST:PORT]     show the pairing link
//!     cubicai-gateway remote on|off|status               allow the paired phone in over Tailscale
//!     cubicai-gateway rotate                             new token (un-pairs every phone)
//!     cubicai-gateway voice-setup                        download what voice calls need
//!     cubicai-gateway computer-setup                     build the Bots' computer (Docker)
//!     cubicai-gateway google-login <client_secret.json>  connect your Google account for Bots
//!     cubicai-gateway install                            run this gateway as the cubicai-server service

use cubic_gateway::{access, identity, log, settings::Settings, Config, DEFAULT_PORT};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let port: u16 = flag("--port").and_then(|p| p.parse().ok()).unwrap_or(DEFAULT_PORT);
    let config = Config::from_env();
    let result = match args.first().map(String::as_str) {
        None | Some("serve") => serve(config, &flag("--bind").unwrap_or_else(|| "::".into()), port),
        Some("pair") => pair(&config, flag("--url"), port),
        Some("remote") => remote(&config, args.get(1).map(String::as_str).unwrap_or("status"), port),
        Some("rotate") => identity::load(&config.dir).and_then(|_| identity::rotate_token(&config.dir)).map(|_| println!("New token created. Restart the gateway and pair your phone again.")),
        Some("install") => std::env::current_exe().map_err(|e| e.to_string()).and_then(|exe| cubic_gateway::local::install(&exe)).map(|_| {
            println!("Installed as {} and started: the {} service runs this gateway now.", cubic_gateway::local::installed_binary().display(), cubic_gateway::local::SERVICE)
        }),
        Some("voice-setup") => cubic_gateway::voice::setup(&cubic_gateway::voice::Paths::from_env()),
        Some("computer-setup") => cubic_gateway::bots::computer::setup(),
        Some("google-login") => match args.get(1) {
            Some(f) => cubic_gateway::bots::google::login(&config.dir, std::path::Path::new(f)),
            None => Err("google-login needs the client JSON file you downloaded from Google Cloud (Clients → your Desktop client → Download JSON)".into()),
        },
        Some(other) => Err(format!("unknown command {other:?}: use serve, pair, remote on|off|status, rotate, voice-setup, computer-setup, google-login or install")),
    };
    if let Err(e) = result {
        eprintln!("cubicai-gateway: {e}");
        std::process::exit(1);
    }
}

fn serve(config: Config, bind: &str, port: u16) -> Result<(), String> {
    let id = identity::load(&config.dir)?;
    let (ollama, searxng) = (config.ollama.clone(), config.searxng.clone());
    let running = cubic_gateway::server::start(config, &id, bind, port, true)?;
    let gw = &running.gateway;
    log(&format!("CubicAI gateway on https://{}:{} (bind {bind}) → Ollama {ollama}, search {searxng}", access::lan_ip(), running.port));
    log(&format!(
        "remote access over Tailscale: {}",
        if gw.settings.remote() { format!("on ({})", gw.host.tailscale_ip().unwrap_or_else(|| "Tailscale not connected".into())) } else { "off".into() }
    ));
    log(&format!("cert fingerprint {}…  — run 'cubicai-gateway pair' to pair a phone", &id.fingerprint[..16]));
    loop {
        std::thread::park();
    }
}

fn pair(config: &Config, url: Option<String>, port: u16) -> Result<(), String> {
    let id = identity::load(&config.dir)?;
    let url = url.unwrap_or_else(|| format!("https://{}:{port}", access::lan_ip()));
    let settings = Settings::new(&config.dir);
    let remote = if settings.remote() { access::Host::default().tailscale_ip().map(|ip| format!("https://{ip}:{port}")) } else { None };
    let link = identity::pairing_link(&url, &id.token, &id.fingerprint, remote.as_deref());
    println!("\nPair the CubicAI app with this PC ({url}{}):", remote.as_ref().map(|r| format!(", remote {r}")).unwrap_or_default());
    println!("  CubicAI → Settings → Local PC → paste the link below (or scan the QR code in CubicAI Lab)\n");
    println!("The link contains your secret token, keep it private:\n  {link}\n");
    Ok(())
}

fn remote(config: &Config, state: &str, _port: u16) -> Result<(), String> {
    let settings = Settings::new(&config.dir);
    match state {
        "on" | "off" => {
            let mut m = serde_json::Map::new();
            m.insert("remote".into(), (state == "on").into());
            settings.save(m)?;
        }
        "status" => {}
        other => return Err(format!("remote {other:?}: use on, off or status")),
    }
    let ip = access::Host::default().tailscale_ip();
    println!("Remote access: {}; Tailscale: {}", if settings.remote() { "on" } else { "off" }, ip.as_deref().unwrap_or("not connected"));
    if state == "on" {
        println!("The phone learns the Tailscale address the next time it connects at home (no re-pairing needed).");
    }
    Ok(())
}
