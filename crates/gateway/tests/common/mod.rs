//! A client like the phone's: TLS 1.3 to a certificate pinned by its fingerprint, plain HTTP/1.1.
#![allow(dead_code)]

use cubic_gateway::identity;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;


#[derive(Debug)]
pub struct Pin(String, Arc<rustls::crypto::CryptoProvider>);

impl ServerCertVerifier for Pin {
    fn verify_server_cert(&self, cert: &CertificateDer, _: &[CertificateDer], _: &ServerName, _: &[u8], _: UnixTime) -> Result<ServerCertVerified, rustls::Error> {
        if identity::fingerprint(cert) == self.0 { Ok(ServerCertVerified::assertion()) } else { Err(rustls::Error::General("not our PC".into())) }
    }

    fn verify_tls12_signature(&self, m: &[u8], c: &CertificateDer, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.1.signature_verification_algorithms)
    }

    fn verify_tls13_signature(&self, m: &[u8], c: &CertificateDer, d: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.1.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.1.signature_verification_algorithms.supported_schemes()
    }
}

pub type Client = BufReader<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>;

pub fn connect_with(port: u16, pin: &str, version: &'static rustls::SupportedProtocolVersion) -> std::io::Result<Client> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[version])
        .unwrap()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pin(pin.to_string(), provider)))
        .with_no_client_auth();
    let conn = rustls::ClientConnection::new(Arc::new(cfg), ServerName::IpAddress(std::net::Ipv4Addr::LOCALHOST.into())).unwrap();
    let tcp = TcpStream::connect(("127.0.0.1", port))?;
    tcp.set_read_timeout(Some(Duration::from_secs(60)))?;
    let mut s = rustls::StreamOwned::new(conn, tcp);
    while s.conn.is_handshaking() {
        s.conn.complete_io(&mut s.sock)?;
    }
    Ok(BufReader::new(s))
}

pub fn connect(port: u16, pin: &str) -> Client {
    connect_with(port, pin, &rustls::version::TLS13).expect("TLS 1.3 to the pinned certificate")
}

pub fn send(c: &mut Client, method: &str, path: &str, token: Option<&str>, body: Option<&str>) {
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: pc\r\n");
    if let Some(t) = token {
        req += &format!("Authorization: Bearer {t}\r\n");
    }
    if let Some(b) = body {
        req += &format!("Content-Type: application/json\r\nContent-Length: {}\r\n", b.len());
    }
    req += "\r\n";
    req += body.unwrap_or("");
    c.get_mut().write_all(req.as_bytes()).unwrap();
    c.get_mut().flush().unwrap();
}

pub struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
    pub bytes: Vec<u8>,
}

impl Resp {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }

    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }

    /// The `data:` events of a stream.
    pub fn events(&self) -> Vec<String> {
        self.body.split("\n\n").filter_map(|e| e.strip_prefix("data: ")).map(String::from).collect()
    }
}

/// Only the status line and headers (a 101 has no body).
pub fn read_head(c: &mut Client) -> Resp {
    let mut line = String::new();
    c.read_line(&mut line).unwrap();
    let status = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut headers = vec![];
    loop {
        let mut h = String::new();
        c.read_line(&mut h).unwrap();
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').unwrap();
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    Resp { status, headers, body: String::new(), bytes: vec![] }
}

pub fn read_resp(c: &mut Client) -> Resp {
    let mut line = String::new();
    c.read_line(&mut line).unwrap();
    let status = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut headers = vec![];
    loop {
        let mut h = String::new();
        c.read_line(&mut h).unwrap();
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').unwrap();
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    let len = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.parse::<usize>().unwrap());
    let bytes = match len {
        Some(n) => {
            let mut b = vec![0; n];
            c.read_exact(&mut b).unwrap();
            b
        }
        None => {
            let mut b = vec![];
            let _ = c.read_to_end(&mut b);
            b
        }
    };
    Resp { status, headers, body: String::from_utf8_lossy(&bytes).into_owned(), bytes }
}

pub fn ask(port: u16, pin: &str, method: &str, path: &str, token: Option<&str>, body: Option<&str>) -> Resp {
    let mut c = connect(port, pin);
    send(&mut c, method, path, token, body);
    read_resp(&mut c)
}

