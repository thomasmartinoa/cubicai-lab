use super::TempDir;
use crate::phone::{self, Loaded, Panel, Status};
use serde_json::json;

#[test]
fn times_read_like_the_old_panel() {
    assert_eq!(phone::duration(42.9), "42s");
    assert_eq!(phone::duration(125.0), "2m 5s");
    assert_eq!(phone::duration(3.0 * 3600.0 + 7.0 * 60.0 + 9.0), "3h 7m");
    assert_eq!(phone::duration(-5.0), "0s");
    assert_eq!(phone::ago(10.0), "just now");
    assert_eq!(phone::ago(600.0), "10 min ago");
    assert_eq!(phone::ago(7200.0), "2 h ago");
    assert_eq!(phone::ago(3.0 * 86400.0), "3 d ago");
    let c = phone::clock(phone::now());
    assert!(c.len() == 5 && c.as_bytes()[2] == b':', "{c}");
}

#[test]
fn where_a_loaded_model_sits() {
    let full = Loaded { name: "gemma4:e4b".into(), size: 4_200_000_000, vram: 4_200_000_000 };
    assert_eq!(full.describe(), "4.2 GB · fully on GPU");
    let split = Loaded { name: "qwen".into(), size: 9_000_000_000, vram: 6_500_000_000 };
    assert_eq!(split.describe(), "9.0 GB · 6.5 GB on GPU, rest on CPU");
}

#[test]
fn activity_rows() {
    let s = Status { names: [("192.168.1.40".to_string(), "Pixel-8".to_string())].into(), ..Default::default() };
    assert_eq!(s.device("192.168.1.40"), "Pixel-8 (192.168.1.40)");
    assert_eq!(s.device("10.0.0.9"), "10.0.0.9");
    assert_eq!(s.device("127.0.0.1"), "This PC (or emulator)");
    let t = 1_700_000_000.0;
    let chat = json!({"ip": "192.168.1.40", "model": "gemma4:e4b", "started": t, "ended": t + 3.4, "outcome": "ok", "tokens": 120, "tps": 95.5, "search": true});
    let (title, detail) = phone::recent_row(&chat, |ip| s.device(ip));
    assert_eq!(title, "gemma4:e4b");
    assert_eq!(detail, format!("{} · Pixel-8 (192.168.1.40) · 3s · 120 tokens · 95.5 tok/s · web search", phone::clock(t)));
    let stopped = json!({"ip": "10.0.0.9", "model": "voice call", "started": t, "ended": t + 70.0, "outcome": "stopped"});
    assert_eq!(phone::recent_row(&stopped, |ip| s.device(ip)).1, format!("{} · 10.0.0.9 · 1m 10s · stopped", phone::clock(t)));
    let blocked = json!({"ip": "10.0.0.9", "started": t, "outcome": "blocked"});
    assert_eq!(phone::recent_row(&blocked, |ip| s.device(ip)).0, "Blocked request (wrong or missing token)");
    let gw = json!({"devices": [{"ip": "a"}], "active": [{"ip": "b"}, {"ip": "a"}], "recent": [{"ip": "c"}, {"ip": "b"}]});
    assert_eq!(phone::ips_in(&gw), ["a", "b", "c"]);
}

#[test]
fn addresses_and_the_firewall_hint() {
    let mut s = Status { lan_ip: "192.168.1.39".into(), tailscale: Some("100.78.0.92".into()), ..Default::default() };
    assert_eq!(s.url(), "https://192.168.1.39:8443");
    assert_eq!(s.remote_url(), None, "remote off: no Tailscale address in the link");
    s.remote = true;
    assert_eq!(s.remote_url().as_deref(), Some("https://100.78.0.92:8443"));
    assert_eq!(phone::firewall_command("192.168.1.39"), "sudo ufw allow from 192.168.1.0/24 to any port 8443 proto tcp");
}

#[test]
fn pairing_qr_holds_the_whole_link() {
    let dir = TempDir::new("phone-pair");
    let link = phone::pairing_link(&dir.0, "https://192.168.1.39:8443", Some("https://100.78.0.92:8443"));
    let id = cubic_gateway::identity::load(&dir.0).unwrap();
    assert!(link.starts_with("cubicai://pair?u=") && link.contains(&id.token) && link.contains(&id.fingerprint), "{link}");
    let m = phone::qr_matrix(&link).unwrap();
    let n = m.len();
    assert!(m.iter().all(|r| r.len() == n));
    // Version v is 17 + 4v modules wide; +4 for the quiet zone. A ~190-byte link at level M
    // needs version 7 or more.
    assert_eq!((n - 4 - 17) % 4, 0);
    assert!((n - 4 - 17) / 4 >= 7, "{n}");
    assert!(m[0].iter().chain(&m[1]).all(|d| !d), "quiet zone");
    assert!((2..9).all(|i| m[2][i]) && m[2 + 6][2], "finder pattern in the corner");
}

#[test]
fn settings_change_the_snapshot_at_once() {
    let dir = TempDir::new("phone-settings");
    let mut panel = Panel::new(dir.0.clone());
    panel.set_status(Status { lan_ip: "192.168.1.39".into(), tailscale: Some("100.78.0.92".into()), num_ctx: 16384, ..Default::default() });
    panel.save_setting("remote", json!(true)).unwrap();
    panel.save_setting("num_ctx", json!(32768)).unwrap();
    let s = panel.status().unwrap();
    assert!(s.remote && s.num_ctx == 32768);
    assert!(s.link.contains("r=https%3A%2F%2F100.78.0.92%3A8443") || s.link.contains("100.78.0.92"), "{}", s.link);
    let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("settings.json")).unwrap()).unwrap();
    assert_eq!(saved, json!({"remote": true, "num_ctx": 32768}));
    let g = cubic_gateway::settings::Settings::new(&dir.0);
    assert!(g.remote() && g.num_ctx(16384) == 32768, "the gateway reads what the panel wrote");
}

#[test]
fn launcher_arguments() {
    use crate::app::start_view;
    use crate::phone::View;
    assert_eq!(start_view("pair"), Some(View::Pair));
    assert_eq!(start_view("activity"), Some(View::Activity));
    assert_eq!(start_view("phone"), Some(View::Status));
    assert_eq!(start_view("--whatever"), None);
    assert!(phone::python_unit().ends_with("cubicai/server/cubicai-server.service"));
}

/// What a scanner reads from the code is exactly the link: drawn 8 px per module (as on screen,
/// black on white) and decoded with a real QR reader.
#[test]
fn pairing_qr_scans_back_to_the_link() {
    let link = "cubicai://pair?u=https%3A%2F%2F192.168.1.20%3A8443&r=https%3A%2F%2F100.64.0.7%3A8443&t=Zm9vYmFyYmF6cXV4cXV1eGNvcmdlZ3JhdWx0Z2FycGx5&f=4e252df057df02ce73f7acfac3dc54ae887fbbb9b5082c9045181d4f85d79c80";
    let m = phone::qr_matrix(link).unwrap();
    let px = 8;
    let n = m.len() * px;
    let mut img = rqrr::PreparedImage::prepare_from_greyscale(n, n, |x, y| if m[y / px][x / px] { 0 } else { 255 });
    let grids = img.detect_grids();
    assert_eq!(grids.len(), 1);
    let (_, text) = grids[0].decode().expect("decodes");
    assert_eq!(text, link);
}

/// The panel on this PC's real gateway: the service, its status over pinned TLS with the token,
/// Ollama, search. Prints nothing secret.
/// `cargo test -p cubicai-lab --release live_panel -- --ignored --nocapture`
#[test]
#[ignore]
fn live_panel_reads_the_real_gateway() {
    let mut panel = Panel::default();
    let ctx = egui::Context::default();
    panel.poll(&ctx);
    let start = std::time::Instant::now();
    let s = loop {
        if let Some(s) = panel.status() {
            break s;
        }
        assert!(start.elapsed().as_secs() < 20, "no status");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    println!("unit {:?}, active {}, enabled {}, search {}, ollama {:?}, loaded {:?}", s.unit, s.active, s.enabled, s.searx, s.ollama, s.loaded.iter().map(|l| l.describe()).collect::<Vec<_>>());
    println!("url {}, remote {:?}, context {}", s.url(), s.remote_url(), s.num_ctx);
    let g = s.gateway.as_ref().expect("the gateway answered over pinned TLS with the token");
    println!("served {} blocked {} up {}; devices {:?}", g["served"], g["blocked"], phone::duration(g["uptime"].as_f64().unwrap()), phone::ips_in(g).iter().map(|ip| s.device(ip)).collect::<Vec<_>>());
    assert!(s.active && s.unit.is_some());
    assert!(s.link.starts_with("cubicai://pair?") && phone::qr_matrix(&s.link).is_some());
}
