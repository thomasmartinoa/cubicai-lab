//! The Bots' computer's only way out: an HTTP proxy on a Unix socket mounted into the container (which
//! has no network of its own). It takes CONNECT (https) and plain http requests to ports 80 and 443,
//! resolves the name itself and connects only when every address is public, to an address it checked,
//! so nothing in the container can reach this PC, the home network or Tailscale, whatever DNS says.
//! Neither addresses nor contents are logged.

use crate::access::is_global;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream, ToSocketAddrs};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const PORTS: [u16; 2] = [80, 443];
const MAX_CONNECTIONS: usize = 64;
const HEAD_BYTES: usize = 16 * 1024;
/// A tunnel with no traffic either way for this long is closed.
const IDLE: Duration = Duration::from_secs(300);

/// Listens on `socket` (made 0600, so only this user, which the container runs as, can use it).
pub fn start(socket: &Path) -> io::Result<()> {
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    let _ = std::fs::set_permissions(socket, std::os::unix::fs::PermissionsExt::from_mode(0o600));
    let open = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            if open.load(Ordering::SeqCst) >= MAX_CONNECTIONS {
                let _ = refuse(conn, "503 Service Unavailable", "too many connections at once");
                continue;
            }
            let open = open.clone();
            open.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || {
                let _ = serve(conn);
                open.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Ok(())
}

fn refuse(mut conn: UnixStream, status: &str, why: &str) -> io::Result<()> {
    let body = format!("CubicAI computer: {why}. It can reach public websites only (ports 80 and 443).\n");
    write!(conn, "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

/// Where a request wants to go: (host, port, CONNECT?, the head rewritten for the origin when not CONNECT).
pub fn target(head: &str) -> Result<(String, u16, bool, String), String> {
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut parts = first.split(' ');
    let (method, uri, version) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or("HTTP/1.1"));
    if method.eq_ignore_ascii_case("CONNECT") {
        let (host, port) = split_host_port(uri, 443)?;
        return Ok((host, port, true, String::new()));
    }
    let rest = uri.strip_prefix("http://").ok_or("only http:// requests or CONNECT")?;
    let (authority, path) = rest.split_once('/').map(|(a, p)| (a, format!("/{p}"))).unwrap_or((rest, "/".into()));
    let (host, port) = split_host_port(authority, 80)?;
    let mut out = format!("{method} {path} {version}\r\n");
    for l in lines.filter(|l| !l.is_empty()) {
        let name = l.split(':').next().unwrap_or("").trim().to_ascii_lowercase();
        if name.starts_with("proxy-") || name == "connection" || name == "keep-alive" {
            continue;
        }
        out.push_str(l);
        out.push_str("\r\n");
    }
    out.push_str("Connection: close\r\n\r\n");
    Ok((host, port, false, out))
}

fn split_host_port(s: &str, default: u16) -> Result<(String, u16), String> {
    let (host, port) = if let Some(v6) = s.strip_prefix('[') {
        let (h, rest) = v6.split_once(']').ok_or("bad address")?;
        (h.to_string(), rest.strip_prefix(':').map(|p| p.parse().map_err(|_| "bad port")).transpose()?.unwrap_or(default))
    } else {
        match s.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse().map_err(|_| "bad port")?),
            None => (s.to_string(), default),
        }
    };
    if host.is_empty() {
        return Err("no host".into());
    }
    Ok((host, port))
}

/// The public address to connect to, or why not.
pub fn resolve(host: &str, port: u16) -> Result<SocketAddr, String> {
    if !PORTS.contains(&port) {
        return Err(format!("port {port} isn't allowed"));
    }
    let addrs: Vec<SocketAddr> = (host, port).to_socket_addrs().map_err(|_| "that name doesn't resolve".to_string())?.collect();
    if addrs.is_empty() {
        return Err("that name doesn't resolve".into());
    }
    if !addrs.iter().all(|a| is_global(a.ip())) {
        return Err("that address isn't a public website".into());
    }
    // IPv4 first: this PC's IPv6 can be patchy.
    Ok(*addrs.iter().find(|a| a.is_ipv4()).unwrap_or(&addrs[0]))
}

fn serve(mut conn: UnixStream) -> io::Result<()> {
    conn.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut buf = Vec::with_capacity(2048);
    let mut chunk = [0u8; 2048];
    let end = loop {
        let n = conn.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() > HEAD_BYTES {
            return refuse(conn, "431 Request Header Fields Too Large", "request too large");
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).to_string();
    let (host, port, connect, rewritten) = match target(&head) {
        Ok(t) => t,
        Err(e) => return refuse(conn, "400 Bad Request", &e),
    };
    let addr = match resolve(&host, port) {
        Ok(a) => a,
        Err(e) => return refuse(conn, "403 Forbidden", &e),
    };
    let mut up = match TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
        Ok(s) => s,
        Err(_) => return refuse(conn, "502 Bad Gateway", "couldn't connect to that website"),
    };
    if connect {
        conn.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n")?;
    } else {
        up.write_all(rewritten.as_bytes())?;
    }
    // Whatever came after the head (a body, or the start of TLS) goes on as it is.
    up.write_all(&buf[end..])?;
    pipe(conn, up)
}

/// Copies both ways until either side ends or it goes quiet for IDLE.
fn pipe(conn: UnixStream, up: TcpStream) -> io::Result<()> {
    conn.set_read_timeout(Some(IDLE))?;
    up.set_read_timeout(Some(IDLE))?;
    let (mut c2, mut u2) = (conn.try_clone()?, up.try_clone()?);
    let back = std::thread::spawn(move || {
        let _ = io::copy(&mut u2, &mut c2);
        let _ = c2.shutdown(Shutdown::Write);
    });
    let (mut c1, mut u1) = (conn, up);
    let _ = io::copy(&mut c1, &mut u1);
    let _ = u1.shutdown(Shutdown::Write);
    let _ = back.join();
    let _ = c1.shutdown(Shutdown::Both);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_read_and_rewritten() {
        assert_eq!(target("CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n").unwrap(), ("example.com".into(), 443, true, String::new()));
        assert_eq!(target("CONNECT [2606:4700::1]:443 HTTP/1.1\r\n\r\n").unwrap().0, "2606:4700::1");
        let (host, port, connect, head) = target("GET http://example.com/a/b?c=1 HTTP/1.1\r\nHost: example.com\r\nProxy-Connection: keep-alive\r\nAccept: */*\r\n\r\n").unwrap();
        assert_eq!((host.as_str(), port, connect), ("example.com", 80, false));
        assert_eq!(head, "GET /a/b?c=1 HTTP/1.1\r\nHost: example.com\r\nAccept: */*\r\nConnection: close\r\n\r\n");
        assert!(target("GET /local HTTP/1.1\r\n\r\n").is_err(), "a request that isn't for a proxy");
        assert!(target("GET ftp://x/ HTTP/1.1\r\n\r\n").is_err());
    }

    #[test]
    fn only_public_websites() {
        for (host, port) in [("127.0.0.1", 443), ("localhost", 443), ("192.168.0.1", 80), ("10.0.0.1", 443), ("100.100.100.100", 443), ("169.254.169.254", 80), ("[::1]", 443), ("fd7a:115c:a1e0::1", 443), ("172.17.0.1", 443)] {
            assert!(resolve(host.trim_matches(['[', ']']), port).is_err(), "{host}");
        }
        assert!(resolve("1.1.1.1", 22).unwrap_err().contains("port 22"));
        assert!(resolve("1.1.1.1", 8443).is_err());
        assert_eq!(resolve("1.1.1.1", 443).unwrap(), "1.1.1.1:443".parse().unwrap());
    }

    #[test]
    fn the_socket_refuses_private_addresses_in_words() {
        let dir = std::env::temp_dir().join(format!("cubic-proxy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("p.sock");
        start(&sock).unwrap();
        for req in ["CONNECT 127.0.0.1:8443 HTTP/1.1\r\n\r\n", "GET http://192.168.0.1/ HTTP/1.1\r\nHost: x\r\n\r\n", "CONNECT 1.1.1.1:25 HTTP/1.1\r\n\r\n"] {
            let mut c = UnixStream::connect(&sock).unwrap();
            c.write_all(req.as_bytes()).unwrap();
            let mut out = String::new();
            c.read_to_string(&mut out).unwrap();
            assert!(out.starts_with("HTTP/1.1 403") && out.contains("public websites only"), "{req:?} → {out}");
        }
        let mode = std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&sock).unwrap().permissions()) & 0o777;
        assert_eq!(mode, 0o600);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
