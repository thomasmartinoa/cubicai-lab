//! Who may connect, and the addresses the phone is told about.
//!
//! Loopback and the home network always; Tailscale only while remote access is on. "Home" is the
//! private ranges plus this PC's own global IPv6 prefixes (a phone on the same ISP prefix is on the
//! same Wi-Fi), read from /proc/net/if_inet6.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// An address range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Net {
    pub addr: IpAddr,
    pub len: u8,
}

impl Net {
    pub const fn v4(a: u8, b: u8, c: u8, d: u8, len: u8) -> Net {
        Net { addr: IpAddr::V4(Ipv4Addr::new(a, b, c, d)), len }
    }

    pub const fn v6(segs: [u16; 8], len: u8) -> Net {
        Net { addr: IpAddr::V6(Ipv6Addr::new(segs[0], segs[1], segs[2], segs[3], segs[4], segs[5], segs[6], segs[7])), len }
    }

    pub fn contains(&self, ip: IpAddr) -> bool {
        match (self.addr, ip) {
            (IpAddr::V4(n), IpAddr::V4(a)) => prefix_eq(&n.octets(), &a.octets(), self.len),
            (IpAddr::V6(n), IpAddr::V6(a)) => prefix_eq(&n.octets(), &a.octets(), self.len),
            _ => false,
        }
    }
}

fn prefix_eq(a: &[u8], b: &[u8], len: u8) -> bool {
    let (full, rest) = ((len / 8) as usize, len % 8);
    if a[..full] != b[..full] {
        return false;
    }
    rest == 0 || {
        let mask = 0xffu8 << (8 - rest);
        a[full] & mask == b[full] & mask
    }
}

pub const TAILNET: [Net; 2] = [Net::v4(100, 64, 0, 0, 10), Net::v6([0xfd7a, 0x115c, 0xa1e0, 0, 0, 0, 0, 0], 48)];
pub const HOME: [Net; 5] = [
    Net::v4(10, 0, 0, 0, 8),
    Net::v4(172, 16, 0, 0, 12),
    Net::v4(192, 168, 0, 0, 16),
    Net::v6([0xfc00, 0, 0, 0, 0, 0, 0, 0], 7),
    Net::v6([0xfe80, 0, 0, 0, 0, 0, 0, 0], 10),
];

/// One of this PC's IPv6 home networks: the prefix and the PC's own address in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lan6 {
    pub net: Net,
    pub addr: Ipv6Addr,
}

/// An IPv4 peer on a dual-stack socket arrives as ::ffff:a.b.c.d.
pub fn unmap(ip: IpAddr) -> IpAddr {
    ip.to_canonical()
}

/// The rule itself, given whether remote access is on and this PC's IPv6 home networks.
pub fn allowed(ip: IpAddr, remote: bool, lan6: &[Lan6]) -> bool {
    let ip = unmap(ip);
    if ip.is_loopback() {
        return true;
    }
    if TAILNET.iter().any(|n| n.contains(ip)) {
        return remote;
    }
    HOME.iter().any(|n| n.contains(ip)) || lan6.iter().any(|l| l.net.contains(ip))
}

const LAN6_SKIP: [&str; 6] = ["lo", "tailscale", "docker", "br-", "veth", "virbr"];

/// /proc/net/if_inet6 → global, permanent addresses on real interfaces (no Tailscale, Docker…).
pub fn parse_if_inet6(text: &str) -> Vec<Lan6> {
    let mut out = vec![];
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [hexaddr, _, plen, scope, flags, dev] = f[..] else { continue };
        // Global scope only; temporary (0x01), deprecated (0x20) and tentative (0x40) addresses aren't used.
        let flags = u32::from_str_radix(flags, 16).unwrap_or(0xff);
        if scope != "00" || flags & 0x61 != 0 || LAN6_SKIP.iter().any(|s| dev.starts_with(s)) {
            continue;
        }
        let (Some(bytes), Ok(len)) = (parse_hex16(hexaddr), u8::from_str_radix(plen, 16)) else { continue };
        let addr = Ipv6Addr::from(bytes);
        if TAILNET.iter().any(|n| n.contains(IpAddr::V6(addr))) || len > 128 {
            continue;
        }
        out.push(Lan6 { net: Net { addr: IpAddr::V6(mask6(addr, len)), len }, addr });
    }
    out
}

fn parse_hex16(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 {
        return None;
    }
    let mut b = [0u8; 16];
    for (i, x) in b.iter_mut().enumerate() {
        *x = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(b)
}

fn mask6(a: Ipv6Addr, len: u8) -> Ipv6Addr {
    let bits = u128::from(a);
    let mask = if len == 0 { 0 } else { u128::MAX << (128 - len as u32) };
    Ipv6Addr::from(bits & mask)
}

/// The PC's IPv6 home URL, a second way in when the router stops passing IPv4 between devices.
/// A ULA (fd00::/8, stable) is preferred to the ISP's prefix, which can change.
pub fn lan6_url(nets: &[Lan6], port: u16) -> Option<String> {
    let ula = |l: &&Lan6| (l.addr.segments()[0] & 0xfe00) == 0xfc00;
    nets.iter().find(ula).or(nets.first()).map(|l| format!("https://[{}]:{port}", l.addr))
}

/// True only for addresses on the public internet: what a fetched web page may point at, so a
/// search result can't make this PC talk to the home network, itself or Ollama.
pub fn is_global(ip: IpAddr) -> bool {
    match unmap(ip) {
        IpAddr::V4(a) => {
            const NOT: [Net; 16] = [
                Net::v4(0, 0, 0, 0, 8),
                Net::v4(10, 0, 0, 0, 8),
                Net::v4(100, 64, 0, 0, 10),
                Net::v4(127, 0, 0, 0, 8),
                Net::v4(169, 254, 0, 0, 16),
                Net::v4(172, 16, 0, 0, 12),
                Net::v4(192, 0, 0, 0, 24),
                Net::v4(192, 0, 2, 0, 24),
                Net::v4(192, 88, 99, 0, 24),
                Net::v4(192, 168, 0, 0, 16),
                Net::v4(198, 18, 0, 0, 15),
                Net::v4(198, 51, 100, 0, 24),
                Net::v4(203, 0, 113, 0, 24),
                Net::v4(224, 0, 0, 0, 4),
                Net::v4(240, 0, 0, 0, 4),
                Net::v4(255, 255, 255, 255, 32),
            ];
            !NOT.iter().any(|n| n.contains(IpAddr::V4(a)))
        }
        IpAddr::V6(a) => {
            // NAT64 can wrap any IPv4 address: judge the one inside.
            if Net::v6([0x64, 0xff9b, 0, 0, 0, 0, 0, 0], 96).contains(IpAddr::V6(a)) {
                let o = a.octets();
                return is_global(IpAddr::V4(Ipv4Addr::new(o[12], o[13], o[14], o[15])));
            }
            const NOT: [Net; 10] = [
                Net::v6([0, 0, 0, 0, 0, 0, 0, 0], 127), // :: and ::1
                Net::v6([0x64, 0xff9b, 1, 0, 0, 0, 0, 0], 48),
                Net::v6([0x100, 0, 0, 0, 0, 0, 0, 0], 64),
                Net::v6([0x2001, 0, 0, 0, 0, 0, 0, 0], 23),
                Net::v6([0x2001, 0xdb8, 0, 0, 0, 0, 0, 0], 32),
                Net::v6([0x2002, 0, 0, 0, 0, 0, 0, 0], 16),
                Net::v6([0xfc00, 0, 0, 0, 0, 0, 0, 0], 7),
                Net::v6([0xfe80, 0, 0, 0, 0, 0, 0, 0], 10),
                Net::v6([0xfec0, 0, 0, 0, 0, 0, 0, 0], 10),
                Net::v6([0xff00, 0, 0, 0, 0, 0, 0, 0], 8),
            ];
            !NOT.iter().any(|n| n.contains(IpAddr::V6(a)))
        }
    }
}

/// Things about this machine that are slow to ask, kept for 30 s.
pub struct Host {
    lan6: Mutex<(Option<Instant>, Vec<Lan6>)>,
    tailscale: Mutex<(Option<Instant>, Option<String>)>,
}

const FRESH: Duration = Duration::from_secs(30);

impl Default for Host {
    fn default() -> Self {
        Host { lan6: Mutex::new((None, vec![])), tailscale: Mutex::new((None, None)) }
    }
}

impl Host {
    pub fn lan6(&self) -> Vec<Lan6> {
        let mut c = self.lan6.lock().unwrap();
        if c.0.is_none_or(|t| t.elapsed() > FRESH) {
            *c = (Some(Instant::now()), parse_if_inet6(&std::fs::read_to_string("/proc/net/if_inet6").unwrap_or_default()));
        }
        c.1.clone()
    }

    /// This PC's Tailscale IPv4 address, or None if Tailscale isn't installed or connected.
    pub fn tailscale_ip(&self) -> Option<String> {
        let mut c = self.tailscale.lock().unwrap();
        if c.0.is_none_or(|t| t.elapsed() > FRESH) {
            *c = (Some(Instant::now()), ask_tailscale());
        }
        c.1.clone()
    }
}

fn ask_tailscale() -> Option<String> {
    let mut child = std::process::Command::new("tailscale")
        .args(["ip", "-4"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    // Five seconds at most, as the Python gateway: a wedged tailscaled mustn't hold up /v1/models.
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(s)) if s.success() => break,
            Ok(Some(_)) | Err(_) => return None,
            Ok(None) if start.elapsed() > Duration::from_secs(5) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let mut out = String::new();
    std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
    out.split_whitespace().next().map(String::from)
}

/// The address the PC uses on the home network (no packet is sent; this just picks the interface).
pub fn lan_ip() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| {
            s.connect("10.255.255.255:1")?;
            s.local_addr()
        })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}
