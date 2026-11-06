//! Just enough HTTP/1.1 for the phone: requests with a Content-Length body, keep-alive, JSON
//! replies and server-sent events. Generic over the stream so tests can use plain memory.

use serde_json::Value;
use std::io::{self, BufRead, BufReader, Read, Write};

const MAX_HEAD: usize = 64 * 1024;

#[derive(Debug, Clone, Default)]
pub struct Request {
    pub method: String,
    /// The request target as sent (path and query).
    pub target: String,
    pub headers: Vec<(String, String)>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The path without the query and trailing slashes, as the Python gateway matches it.
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("").trim_end_matches('/')
    }

    /// A query parameter, decoded.
    pub fn query(&self, key: &str) -> Option<String> {
        let q = self.target.split_once('?')?.1;
        q.split('&').filter_map(|kv| kv.split_once('=').or(Some((kv, "")))).find(|(k, _)| crate::identity::form_decode(k) == key).map(|(_, v)| crate::identity::form_decode(v))
    }

    /// None when absent; Err when it isn't a number.
    pub fn content_length(&self) -> Result<Option<usize>, std::num::ParseIntError> {
        self.header("Content-Length").map(|v| v.trim().parse()).transpose()
    }

    pub fn wants_close(&self) -> bool {
        self.header("Connection").is_some_and(|v| v.eq_ignore_ascii_case("close"))
    }
}

/// One connection: reads requests, writes replies.
pub struct Conn<S: Read + Write> {
    pub r: BufReader<S>,
    /// Whether the current request's body has been read (or drained).
    pub body_read: bool,
}

impl<S: Read + Write> Conn<S> {
    pub fn new(s: S) -> Conn<S> {
        Conn { r: BufReader::with_capacity(64 * 1024, s), body_read: false }
    }

    pub fn stream(&mut self) -> &mut S {
        self.r.get_mut()
    }

    /// The next request's head, or None when the client closed the connection between requests.
    pub fn next_request(&mut self) -> io::Result<Option<Request>> {
        self.body_read = false;
        let mut head = Vec::new();
        loop {
            let buf = self.r.fill_buf()?;
            if buf.is_empty() {
                return if head.is_empty() { Ok(None) } else { Err(io::Error::new(io::ErrorKind::UnexpectedEof, "request cut short")) };
            }
            // Take bytes up to and including the blank line, leaving the body in the buffer.
            let mut take = buf.len();
            let mut done = false;
            for (i, &b) in buf.iter().enumerate() {
                head.push(b);
                if head.ends_with(b"\r\n\r\n") {
                    take = i + 1;
                    done = true;
                    break;
                }
            }
            self.r.consume(take);
            if done {
                break;
            }
            if head.len() > MAX_HEAD {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "request head too large"));
            }
        }
        let mut hs = [httparse::EMPTY_HEADER; 64];
        let mut req = httparse::Request::new(&mut hs);
        match req.parse(&head) {
            Ok(httparse::Status::Complete(_)) => {}
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request")),
        }
        Ok(Some(Request {
            method: req.method.unwrap_or("").to_string(),
            target: req.path.unwrap_or("").to_string(),
            headers: req.headers.iter().map(|h| (h.name.to_string(), String::from_utf8_lossy(h.value).into_owned())).collect(),
        }))
    }

    /// Reads exactly `n` body bytes.
    pub fn read_body(&mut self, n: usize) -> io::Result<Vec<u8>> {
        self.body_read = true;
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }

    /// A rejected POST's body may be unread: read it (up to a limit) so closing doesn't reset the
    /// connection before the client sees the reply.
    pub fn drain(&mut self, req: &Request) {
        if req.method != "POST" || self.body_read {
            return;
        }
        self.body_read = true;
        if let Ok(Some(n)) = req.content_length()
            && n > 0
            && n <= crate::DRAIN_LIMIT
        {
            let _ = io::copy(&mut (&mut self.r).take(n as u64), &mut io::sink());
        }
    }

    /// A JSON reply. Errors (>= 400) drain the request body and close the connection, so an unread
    /// body is never parsed as the next request. Returns whether the connection may be kept.
    pub fn json(&mut self, req: &Request, code: u16, v: &Value) -> io::Result<bool> {
        let body = v.to_string();
        let close = code >= 400 || req.wants_close();
        if code >= 400 {
            self.drain(req);
        }
        let mut head = format!("HTTP/1.1 {code} {}\r\nServer: cubicai\r\nContent-Type: application/json\r\nContent-Length: {}\r\n", reason(code), body.len());
        if close {
            head += "Connection: close\r\n";
        }
        head += "\r\n";
        let w = self.stream();
        w.write_all(head.as_bytes())?;
        w.write_all(body.as_bytes())?;
        w.flush()?;
        Ok(!close)
    }

    /// Starts a server-sent event stream (the connection closes when it ends).
    pub fn start_events(&mut self) -> io::Result<()> {
        let w = self.stream();
        w.write_all(b"HTTP/1.1 200 OK\r\nServer: cubicai\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        w.flush()
    }

    /// One `data:` event.
    pub fn event(&mut self, v: &Value) -> io::Result<()> {
        let w = self.stream();
        w.write_all(format!("data: {v}\n\n").as_bytes())?;
        w.flush()
    }

    pub fn raw(&mut self, bytes: &[u8]) -> io::Result<()> {
        let w = self.stream();
        w.write_all(bytes)?;
        w.flush()
    }
}

pub fn reason(code: u16) -> &'static str {
    match code {
        101 => "Switching Protocols",
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "",
    }
}

/// `{"error": {"message": …}}`, the shape the phone reads.
pub fn error(message: &str) -> Value {
    serde_json::json!({"error": {"message": message}})
}
