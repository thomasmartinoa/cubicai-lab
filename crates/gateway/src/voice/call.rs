//! One call on one WebSocket, as `Call` in `cubicai_voice.py`. The phone streams 16 kHz PCM16 and
//! JSON commands; the PC answers with `status`, `ready`, `speech`, `partial`/`final`, `seg` + 24 kHz
//! PCM16, `turn_end`.
//!
//! The connection's own thread owns the TLS stream: it reads frames with a short timeout and writes
//! whatever the transcriber and speaker threads queued, so the stream is never shared.

use super::text::{speakable, Segmenter};
use super::{log, Engine, Voice, FRAME, FRAME_MS, VOICES};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
pub const MAX_MESSAGE: usize = 1 << 20;
/// Kept from before speech is detected, so the first word isn't clipped.
const PREROLL_MS: usize = 320;
const MAX_UTTERANCE_S: usize = 30;
const PARTIAL_EVERY_MS: usize = 900;
/// Samples of voice per binary frame (200 ms), so a cancel takes effect quickly.
const SEND_CHUNK: usize = super::OUT_RATE / 5;
/// The phone pings every 15 s: this long without a word from it is a lost phone.
pub const SILENT_PHONE: Duration = Duration::from_secs(90);
/// How long a read waits before the queued output gets its turn.
pub const POLL_MS: u64 = 10;

/// The handshake's answer to the phone's key.
pub fn accept_key(key: &str) -> String {
    let d = ring::digest::digest(&ring::digest::SHA1_FOR_LEGACY_USE_ONLY, format!("{key}{WS_GUID}").as_bytes());
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, d.as_ref())
}

/// A server frame (never masked).
pub fn frame(op: u8, data: &[u8]) -> Vec<u8> {
    let mut f = vec![0x80 | op];
    match data.len() {
        n if n < 126 => f.push(n as u8),
        n if n < 1 << 16 => {
            f.push(126);
            f.extend((n as u16).to_be_bytes());
        }
        n => {
            f.push(127);
            f.extend((n as u64).to_be_bytes());
        }
    }
    f.extend_from_slice(data);
    f
}

#[derive(Debug, PartialEq)]
pub enum Event {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Close(Vec<u8>),
}

/// Reads the phone's frames; a timeout leaves partial data buffered for the next call.
#[derive(Default)]
pub struct WsReader {
    buf: Vec<u8>,
    message: Vec<u8>,
    opcode: Option<u8>,
}

fn closed(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionAborted, why.to_string())
}

impl WsReader {
    /// With bytes already read past the handshake.
    pub fn new(buffered: Vec<u8>) -> WsReader {
        WsReader { buf: buffered, ..Default::default() }
    }

    /// The next event, None if nothing complete arrived before the read timed out.
    pub fn next(&mut self, r: &mut dyn Read) -> io::Result<Option<Event>> {
        loop {
            if let Some(ev) = self.parse()? {
                return Ok(Some(ev));
            }
            let mut chunk = [0u8; 16384];
            match r.read(&mut chunk) {
                Ok(0) => return Err(closed("the phone hung up")),
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => return Ok(None),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    /// One whole event from the buffer, if there is one.
    fn parse(&mut self) -> io::Result<Option<Event>> {
        loop {
            let b = &self.buf;
            if b.len() < 2 {
                return Ok(None);
            }
            let (fin, op, masked, short) = (b[0] & 0x80 != 0, b[0] & 0x0f, b[1] & 0x80 != 0, (b[1] & 0x7f) as usize);
            let (head, len) = match short {
                126 if b.len() >= 4 => (4, u16::from_be_bytes([b[2], b[3]]) as usize),
                127 if b.len() >= 10 => (10, u64::from_be_bytes(b[2..10].try_into().unwrap()) as usize),
                126 | 127 => return Ok(None),
                n => (2, n),
            };
            if len > MAX_MESSAGE || !masked {
                return Err(closed("bad frame")); // clients must mask
            }
            if b.len() < head + 4 + len {
                return Ok(None);
            }
            let mask: [u8; 4] = b[head..head + 4].try_into().unwrap();
            let payload: Vec<u8> = b[head + 4..head + 4 + len].iter().enumerate().map(|(i, x)| x ^ mask[i % 4]).collect();
            self.buf.drain(..head + 4 + len);
            match op {
                8 => return Ok(Some(Event::Close(payload))),
                9 => return Ok(Some(Event::Ping(payload))),
                10 => continue,
                1 | 2 => {
                    self.opcode = Some(op);
                    self.message = payload;
                }
                0 if self.opcode.is_some() => {
                    self.message.extend(payload);
                    if self.message.len() > MAX_MESSAGE {
                        return Err(closed("message too big"));
                    }
                }
                _ => {}
            }
            if fin && let Some(op) = self.opcode.take() {
                let m = std::mem::take(&mut self.message);
                return Ok(Some(if op == 1 { Event::Text(String::from_utf8(m).map_err(|_| closed("text that isn't UTF-8"))?) } else { Event::Binary(m) }));
            }
        }
    }
}

/// Something for the phone. Speech carries the epoch it was made in: a cancel drops it unsent.
enum Out {
    Json(Value),
    Speech(u64, Vec<u8>),
}

struct Shared {
    alive: AtomicBool,
    utterance_id: AtomicU64,
    speaking: AtomicBool,
    transcribing: AtomicBool,
    asr_waiting: AtomicUsize,
    epoch: AtomicU64,
    voice: Mutex<(String, f32)>,
    out: Mutex<Sender<Out>>,
}

impl Shared {
    fn send(&self, v: Value) {
        let _ = self.out.lock().unwrap().send(Out::Json(v));
    }
}

struct Asr {
    kind: &'static str,
    id: u64,
    audio: Vec<f32>,
}

struct Say {
    epoch: u64,
    turn: i64,
    text: Option<String>,
}

/// Hearing the user: the reader thread's state.
struct Ears {
    vad_state: Vec<f32>,
    vad_ctx: Vec<f32>,
    pending: Vec<u8>,
    preroll: VecDeque<Vec<f32>>,
    utterance: Vec<Vec<f32>>,
    voiced: usize,
    silence: usize,
    last_partial: usize,
    end_ms: usize,
    playing: bool,
    debug: Option<(usize, f32, f32)>,
}

/// Runs one call until the phone hangs up. `on_phase` follows it in the activity list.
pub fn run(voice: &Voice, rw: &mut dyn ReadWrite, buffered: Vec<u8>, on_phase: &dyn Fn(&str)) {
    voice.calls.fetch_add(1, Ordering::SeqCst);
    let (tx, rx) = channel();
    let shared = Arc::new(Shared {
        alive: AtomicBool::new(true),
        utterance_id: AtomicU64::new(0),
        speaking: AtomicBool::new(false),
        transcribing: AtomicBool::new(false),
        asr_waiting: AtomicUsize::new(0),
        epoch: AtomicU64::new(0),
        voice: Mutex::new((VOICES[0].0.to_string(), 1.0)),
        out: Mutex::new(tx),
    });
    let (asr_tx, asr_rx) = channel::<Option<Asr>>();
    let (tts_tx, tts_rx) = channel::<Option<Say>>();
    let mut call = Call { shared: shared.clone(), rw, ws: WsReader::new(buffered), rx, asr: asr_tx.clone(), tts: tts_tx.clone(), segmenters: HashMap::new(), ears: Ears::new(), engine: None };
    let result = call.start(voice, on_phase, asr_rx, tts_rx);
    if let Err(e) = result {
        let quiet = matches!(e.kind(), io::ErrorKind::ConnectionAborted | io::ErrorKind::ConnectionReset | io::ErrorKind::BrokenPipe | io::ErrorKind::UnexpectedEof | io::ErrorKind::TimedOut);
        if !quiet {
            log(&format!("call ended with {e}"));
        }
    }
    shared.alive.store(false, Ordering::SeqCst);
    let _ = asr_tx.send(None);
    let _ = tts_tx.send(None);
    voice.calls.fetch_sub(1, Ordering::SeqCst);
    voice.touch();
}

pub trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

struct Call<'a> {
    shared: Arc<Shared>,
    rw: &'a mut dyn ReadWrite,
    ws: WsReader,
    rx: Receiver<Out>,
    asr: Sender<Option<Asr>>,
    tts: Sender<Option<Say>>,
    segmenters: HashMap<i64, Segmenter>,
    ears: Ears,
    engine: Option<Arc<dyn Engine>>,
}

impl Ears {
    fn new() -> Ears {
        Ears {
            vad_state: vec![0.0; 2 * 128],
            vad_ctx: vec![0.0; 64],
            pending: vec![],
            preroll: VecDeque::new(),
            utterance: vec![],
            voiced: 0,
            silence: 0,
            last_partial: 0,
            end_ms: 600, // silence that ends what the user is saying
            playing: false, // the phone plays our voice: be surer it's the user before saying they talk
            debug: (std::env::var("CUBICAI_VOICE_DEBUG").as_deref() == Ok("1")).then_some((0, 0.0, 0.0)),
        }
    }
}

impl Call<'_> {
    fn start(&mut self, voice: &Voice, on_phase: &dyn Fn(&str), asr_rx: Receiver<Option<Asr>>, tts_rx: Receiver<Option<Say>>) -> io::Result<()> {
        on_phase("Waking up the voice");
        self.shared.send(json!({"type": "status", "text": "Waking up the voice on your PC…"}));
        self.flush()?;
        let engine = voice.engine().map_err(io::Error::other)?;
        self.engine = Some(engine.clone());
        let (s, e) = (self.shared.clone(), engine.clone());
        std::thread::Builder::new().name("voice-ears".into()).spawn(move || asr_worker(&s, &*e, asr_rx))?;
        let (s, e) = (self.shared.clone(), engine);
        std::thread::Builder::new().name("voice-mouth".into()).spawn(move || tts_worker(&s, &*e, tts_rx))?;
        self.shared.send(json!({"type": "ready"}));
        on_phase("On a call");
        let mut heard = Instant::now();
        loop {
            self.flush()?;
            if !self.shared.alive.load(Ordering::SeqCst) {
                return Ok(());
            }
            match self.ws.next(self.rw)? {
                None if heard.elapsed() > SILENT_PHONE => return Err(io::Error::new(io::ErrorKind::TimedOut, "the phone went quiet")),
                None => continue,
                Some(ev) => {
                    heard = Instant::now();
                    match ev {
                        Event::Binary(b) => self.audio(&b),
                        Event::Text(t) => {
                            let Ok(msg) = serde_json::from_str::<Value>(&t) else { return Err(io::Error::other("a command that isn't JSON")) };
                            self.command(&msg);
                        }
                        Event::Ping(p) => self.rw.write_all(&frame(10, &p))?,
                        Event::Close(p) => {
                            let _ = self.rw.write_all(&frame(8, &p[..p.len().min(2)]));
                            let _ = self.rw.flush();
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    /// Writes what's queued for the phone, leaving out speech from before a cancel.
    fn flush(&mut self) -> io::Result<()> {
        let mut wrote = false;
        while let Ok(out) = self.rx.try_recv() {
            match out {
                Out::Json(v) => self.rw.write_all(&frame(1, v.to_string().as_bytes()))?,
                Out::Speech(epoch, _) if epoch != self.shared.epoch.load(Ordering::SeqCst) => continue,
                Out::Speech(_, f) => self.rw.write_all(&f)?,
            }
            wrote = true;
        }
        if wrote {
            self.rw.flush()?;
        }
        Ok(())
    }

    fn command(&mut self, msg: &Value) {
        let turn = || number(&msg["turn"]).unwrap_or(0.0) as i64;
        match msg["type"].as_str().unwrap_or("") {
            "start" | "config" => {
                let mut v = self.shared.voice.lock().unwrap();
                if let Some(id) = msg["voice"].as_str().filter(|id| super::is_voice(id)) {
                    v.0 = id.to_string();
                }
                if let Some(s) = number(&msg["speed"]).filter(|s| *s != 0.0) {
                    v.1 = (s as f32).clamp(0.7, 1.4);
                }
                if let Some(p) = number(&msg["patience"]).filter(|p| *p != 0.0) {
                    self.ears.end_ms = (p as i64).clamp(300, 2500) as usize;
                }
            }
            "playing" => self.ears.playing = truthy(&msg["on"]),
            "say" => {
                let turn = turn();
                let text = match &msg["text"] {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    v => v.to_string(),
                };
                let epoch = self.shared.epoch.load(Ordering::SeqCst);
                for part in self.segmenters.entry(turn).or_default().feed(&text) {
                    let _ = self.tts.send(Some(Say { epoch, turn, text: Some(part) }));
                }
            }
            "say_end" => {
                let turn = turn();
                let epoch = self.shared.epoch.load(Ordering::SeqCst);
                for part in self.segmenters.remove(&turn).map(|mut s| s.flush()).unwrap_or_default() {
                    let _ = self.tts.send(Some(Say { epoch, turn, text: Some(part) }));
                }
                let _ = self.tts.send(Some(Say { epoch, turn, text: None })); // the end of the turn
            }
            "cancel" => {
                self.shared.epoch.fetch_add(1, Ordering::SeqCst);
                self.segmenters.clear();
            }
            _ => {}
        }
    }

    fn audio(&mut self, data: &[u8]) {
        let mut all = std::mem::take(&mut self.ears.pending);
        all.extend_from_slice(data);
        let usable = all.len() - all.len() % (FRAME * 2);
        self.ears.pending = all[usable..].to_vec();
        let samples: Vec<f32> = all[..usable].as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect();
        for f in samples.chunks(FRAME) {
            self.frame(f);
        }
    }

    fn frame(&mut self, frame: &[f32]) {
        let engine = self.engine.clone().expect("loaded");
        let e = &mut self.ears;
        let mut x = e.vad_ctx.clone();
        x.extend_from_slice(frame);
        let p = engine.vad(&x, &mut e.vad_state).unwrap_or(0.0);
        e.vad_ctx = frame[frame.len() - 64..].to_vec();
        if let Some(d) = e.debug.as_mut() {
            // Loudness and speech likelihood once a second (never what was said), to tune the microphone path.
            d.0 += 1;
            d.1 = d.1.max((frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt());
            d.2 = d.2.max(p);
            if d.0 * FRAME_MS >= 1000 {
                log(&format!("level rms={:.4} speech={:.2} playing={}", d.1, d.2, if e.playing { "True" } else { "False" }));
                *d = (0, 0.0, 0.0);
            }
        }
        let s = &self.shared;
        if !s.speaking.load(Ordering::SeqCst) {
            e.preroll.push_back(frame.to_vec());
            if e.preroll.len() > PREROLL_MS / FRAME_MS {
                e.preroll.pop_front();
            }
            let (start_p, start_ms) = if e.playing { (0.7, 380) } else { (0.5, 220) };
            if p > start_p {
                e.voiced += FRAME_MS;
            } else if p < 0.3 {
                e.voiced = e.voiced.saturating_sub(2 * FRAME_MS);
            }
            if e.voiced >= start_ms {
                s.speaking.store(true, Ordering::SeqCst);
                e.silence = 0;
                e.utterance = e.preroll.iter().cloned().collect();
                let id = s.utterance_id.fetch_add(1, Ordering::SeqCst) + 1;
                e.last_partial = 0;
                s.send(json!({"type": "speech", "on": true, "id": id}));
            }
            return;
        }
        e.utterance.push(frame.to_vec());
        e.silence = if p < 0.35 { e.silence + FRAME_MS } else { 0 };
        let length_ms = e.utterance.len() * FRAME_MS;
        let id = s.utterance_id.load(Ordering::SeqCst);
        if e.silence >= e.end_ms || length_ms >= MAX_UTTERANCE_S * 1000 {
            let keep = (e.utterance.len() as isize - (e.silence / FRAME_MS) as isize + 6).max(1) as usize;
            let audio: Vec<f32> = e.utterance.iter().take(keep).flatten().copied().collect();
            s.asr_waiting.fetch_add(1, Ordering::SeqCst);
            let _ = self.asr.send(Some(Asr { kind: "final", id, audio }));
            s.speaking.store(false, Ordering::SeqCst);
            e.voiced = 0;
            e.preroll.clear();
            s.send(json!({"type": "speech", "on": false, "id": id}));
        } else if e.silence == 0 && length_ms - e.last_partial >= PARTIAL_EVERY_MS && s.asr_waiting.load(Ordering::SeqCst) == 0 && !s.transcribing.load(Ordering::SeqCst) && length_ms < 12000 {
            // Captions while the user talks, only when the transcriber is free and they're not pausing
            // (it's likely they're done, and the final transcript must not wait behind a partial one).
            e.last_partial = length_ms;
            s.asr_waiting.fetch_add(1, Ordering::SeqCst);
            let _ = self.asr.send(Some(Asr { kind: "partial", id, audio: e.utterance.iter().flatten().copied().collect() }));
        }
    }
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn asr_worker(s: &Shared, engine: &dyn Engine, rx: Receiver<Option<Asr>>) {
    while let Ok(Some(job)) = rx.recv() {
        s.asr_waiting.fetch_sub(1, Ordering::SeqCst);
        if !s.alive.load(Ordering::SeqCst) {
            return;
        }
        if job.kind == "partial" && (job.id != s.utterance_id.load(Ordering::SeqCst) || !s.speaking.load(Ordering::SeqCst)) {
            continue; // already finished; the final transcript is on its way
        }
        s.transcribing.store(true, Ordering::SeqCst);
        let text = engine.transcribe(&job.audio).unwrap_or_else(|e| {
            log(&format!("transcription failed: {e}"));
            String::new()
        });
        s.transcribing.store(false, Ordering::SeqCst);
        s.send(json!({"type": job.kind, "id": job.id, "text": text}));
    }
}

fn tts_worker(s: &Shared, engine: &dyn Engine, rx: Receiver<Option<Say>>) {
    let mut seq = 0u64;
    while let Ok(Some(job)) = rx.recv() {
        if !s.alive.load(Ordering::SeqCst) {
            return;
        }
        if job.epoch != s.epoch.load(Ordering::SeqCst) {
            continue;
        }
        let Some(text) = job.text else {
            let _ = s.out.lock().unwrap().send(Out::Speech(job.epoch, frame(1, json!({"type": "turn_end", "turn": job.turn}).to_string().as_bytes())));
            continue;
        };
        let words = speakable(&text);
        let (voice, speed) = s.voice.lock().unwrap().clone();
        let samples = match engine.speak(&words, &voice, speed) {
            Ok(a) => a,
            Err(e) => {
                log(&format!("speech failed: {e}"));
                continue;
            }
        };
        if job.epoch != s.epoch.load(Ordering::SeqCst) {
            continue;
        }
        seq += 1;
        let pcm = super::tts::pcm16(&samples);
        let out = s.out.lock().unwrap();
        let _ = out.send(Out::Speech(job.epoch, frame(1, json!({"type": "seg", "turn": job.turn, "seq": seq, "text": words, "samples": samples.len()}).to_string().as_bytes())));
        for chunk in pcm.chunks(SEND_CHUNK * 2) {
            let _ = out.send(Out::Speech(job.epoch, frame(2, chunk)));
        }
    }
}
