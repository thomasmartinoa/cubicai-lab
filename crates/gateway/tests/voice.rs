//! Voice calls without the models: the text and phoneme rules (against values from the Python
//! voice), the decoding and audio helpers, the WebSocket, and a whole call over real TLS with a
//! scripted engine. `voice_golden.rs` checks the same parts against Python on the real models.

mod common;

use common::*;
use cubic_gateway::voice::{self, asr, call, phonemes, text, tts, Engine};
use cubic_gateway::{identity, server, Config};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn what_the_voice_says_and_when() {
    // Values from cubicai_voice.py.
    assert_eq!(text::speakable("*waves* Hi there! *grins*"), "Hi there!");
    assert_eq!(text::speakable("Facts [1] and more [2, 3] and a range [4–6]."), "Facts and more and a range.");
    assert_eq!(text::speakable("Emoji 😀👍🏽 and ☀️ sun ❤️‍🔥"), "Emoji and sun");
    assert_eq!(text::speakable("**Bold** and [a link](https://x.y) or https://example.com"), "Bold and a link or a link");
    assert_eq!(text::speakable("- one\n- two\n# Title"), "one two Title");
    assert_eq!(text::speakable("😀 *** "), "");
    let feed = |pieces: &[&str]| {
        let mut s = text::Segmenter::new();
        let fed: Vec<Vec<String>> = pieces.iter().map(|p| s.feed(p)).collect();
        (fed, s.flush())
    };
    let (fed, flush) = feed(&["Hello", " there! How", " are you doing today? I hope", " everything is fine.", " Bye"]);
    assert_eq!(fed, vec![vec![], vec!["Hello there! ".to_string()], vec!["How are you doing today? ".into()], vec![], vec!["I hope everything is fine. ".into()]]);
    assert_eq!(flush, vec!["Bye".to_string()]);
    // Never cut inside an unclosed *action* (Python cuts a closed one like any sentence).
    let (fed, flush) = feed(&["Sure *smiles warmly. And", " more* done. Next sentence here is good. "]);
    assert_eq!(fed, vec![vec![], vec!["Sure *smiles warmly. ".to_string()]]);
    assert_eq!(flush, vec!["And more* done. Next sentence here is good. ".to_string()]);
    // Code: a newline is a sentence end even inside a closed fence (Python does the same); an
    // unclosed one is never cut or said.
    let (fed, flush) = feed(&["Run this: ```sh\nls -la. rm.\n``` and done. ", "```unclosed. text"]);
    assert_eq!(fed, vec![vec!["Run this: ```sh\n".to_string()], vec!["ls -la. rm.\n``` and done. ".to_string()]]);
    assert_eq!(flush, Vec::<String>::new());
}

#[test]
fn punctuation_survives_phonemes_like_phonemizer() {
    let text = "Well, hello there! Dr. Smith said: (yes) — fine…";
    let (chunks, marks) = phonemes::preserve(text);
    assert_eq!(chunks, ["Well", "hello there", "Dr", "Smith said", "yes", "fine"]);
    let m: Vec<(String, char)> = marks.iter().map(|m| (m.mark.clone(), m.position)).collect();
    assert_eq!(m, [(", ".into(), 'I'), ("! ".into(), 'I'), (". ".into(), 'I'), (": (".into(), 'I'), (") — ".into(), 'I'), ("…".into(), 'E')]);
    // What libespeak-ng 1.52 returns for each chunk.
    let raw = ["w_ˈɛ_l", "h_ə_l_ˈoʊ ð_ˈɛɹ", "d_ˈɑː_k_t_ɚ", "s_m_ˈɪ_θ s_ˈɛ_d", "j_ˈɛ_s", "f_ˈaɪ_n"];
    let mut i = 0;
    let out = phonemes::phonemize_with(text, &mut |chunk| {
        assert_eq!(chunk, chunks[i]);
        i += 1;
        Ok(raw[i - 1].to_string())
    })
    .unwrap();
    assert_eq!(tts::keep_known(&out), "wˈɛl, həlˈoʊ ðˈɛɹ! dˈɑːktɚ. smˈɪθ sˈɛd: (jˈɛs) — fˈaɪn…");
    assert_eq!(phonemes::postprocess("  h_ə__l_ˈoʊ  w_ˈɜː_l_d "), "həlˈoʊ wˈɜːld ");
    assert_eq!(phonemes::phonemize_with("…!", &mut |_| panic!("only marks")).unwrap(), "…!");
    assert_eq!(phonemes::phonemize_with("  ", &mut |_| panic!("nothing")).unwrap(), "");
    assert_eq!(tts::tokenize("hˈi, ʔx"), vec![50, 156, 51, 3, 16, 148, 66]);
    assert_eq!(tts::split_phonemes("a, b. c! d"), vec!["a, b. c! d"]);
    let long = "ab, ".repeat(200);
    let b = tts::split_phonemes(&long);
    assert!(b.len() == 2 && b.iter().all(|x| x.chars().count() < tts::MAX_PHONEMES), "{:?}", b.iter().map(|x| x.len()).collect::<Vec<_>>());
}

struct Script {
    /// Per frame: (best token, best duration).
    plan: Vec<(usize, usize)>,
    calls: Vec<(usize, usize)>,
    commits: usize,
}

impl asr::Decoder for Script {
    fn step(&mut self, t: usize, last: usize) -> Result<Vec<f32>, String> {
        self.calls.push((t, last));
        let (token, dur) = self.plan[self.calls.len() - 1];
        let mut v = vec![0.0; 5 + 5];
        v[token] = 1.0;
        v[5 + dur] = 1.0;
        Ok(v)
    }
    fn commit(&mut self) {
        self.commits += 1;
    }
}

#[test]
fn transducer_decoding_and_text() {
    // Vocabulary of 5 with blank = 4; durations 0..4.
    let mut d = Script { plan: vec![(1, 0), (2, 2), (4, 0), (3, 1), (4, 3)], calls: vec![], commits: 0 };
    let tokens = asr::tdt_decode(6, 5, 4, &mut d).unwrap();
    assert_eq!(tokens, vec![1, 2, 3]);
    assert_eq!(d.calls, vec![(0, 4), (0, 1), (2, 2), (3, 2), (4, 3)], "frame and previous token per step");
    assert_eq!(d.commits, 3, "the state moves on only with a token");
    // At most 10 tokens on one frame.
    let mut d = Script { plan: vec![(1, 0); 11], calls: vec![], commits: 0 };
    assert_eq!(asr::tdt_decode(1, 5, 4, &mut d).unwrap().len(), 10);
    let (vocab, blank) = asr::vocab("<unk> 0\n\u{2581}Hel 1\nlo 2\n, 3\n\u{2581}world 4\n<blk> 5\n").unwrap();
    assert_eq!(blank, 5);
    assert_eq!(asr::text(&[1, 2, 3, 4], &vocab), "Hello, world");
    let (f, frames, valid) = asr::features(&vec![0.0; 16000], &asr::mel_bank(), &asr::Fft::new());
    assert_eq!((f.len(), frames, valid), (128 * 101, 101, 100));
    assert!(f.iter().all(|x| x.is_finite()));
}

#[test]
fn audio_helpers() {
    // Half a second of silence each side of a tone: trimmed to whole 512-sample frames around it
    // (the same bounds as kokoro-onnx's librosa trim).
    let mut y = vec![0f32; 12000];
    y.extend((0..24000).map(|i| (i as f32 * 0.05).sin() * 0.5));
    y.extend(vec![0f32; 12000]);
    let t = tts::trim(&y);
    let start = t.as_ptr() as usize - y.as_ptr() as usize;
    assert_eq!((start / 4, t.len()), (11264, 26112));
    assert!(tts::trim(&[0.0; 3000]).len() == 3000, "all quiet: nothing is louder than the rest");
    assert_eq!(tts::pcm16(&[0.0, 1.0, -1.0, 2.0, 0.5]), [0, 0, 0xff, 0x7f, 0x01, 0x80, 0xff, 0x7f, 0xff, 0x3f]);
    // A voices file: .npy arrays in an uncompressed zip.
    let npy = |vals: &[f32]| {
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", vals.len());
        let mut h = header.into_bytes();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((h.len() as u16).to_le_bytes());
        b.extend(h);
        b.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
        b
    };
    let mut zip = vec![];
    for (name, vals) in [("af_test.npy", vec![1.0f32, 2.0]), ("am_test.npy", vec![3.0])] {
        let body = npy(&vals);
        zip.extend(b"PK\x03\x04");
        zip.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        zip.extend((body.len() as u32).to_le_bytes());
        zip.extend((body.len() as u32).to_le_bytes());
        zip.extend((name.len() as u16).to_le_bytes());
        zip.extend(0u16.to_le_bytes());
        zip.extend(name.as_bytes());
        zip.extend(body);
    }
    zip.extend(b"PK\x01\x02 central directory");
    let v = tts::read_voices(&zip).unwrap();
    assert_eq!((v["af_test"].clone(), v["am_test"].clone()), (vec![1.0, 2.0], vec![3.0]));
    assert!(tts::read_voices(b"not a zip").is_err());
}

// --- WebSocket ---------------------------------------------------------------------------------------

fn client_frame(op: u8, data: &[u8], fin: bool) -> Vec<u8> {
    let mask = [0x12, 0x34, 0x56, 0x78];
    let mut f = vec![if fin { 0x80 } else { 0 } | op];
    match data.len() {
        n if n < 126 => f.push(0x80 | n as u8),
        n if n < 65536 => {
            f.push(0x80 | 126);
            f.extend((n as u16).to_be_bytes());
        }
        n => {
            f.push(0x80 | 127);
            f.extend((n as u64).to_be_bytes());
        }
    }
    f.extend(mask);
    f.extend(data.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
    f
}

/// Gives its bytes a few at a time, with a timeout between pieces.
struct Trickle(Vec<u8>, usize, bool);

impl Read for Trickle {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.2 = !self.2;
        if self.2 {
            return Err(std::io::ErrorKind::WouldBlock.into());
        }
        let n = buf.len().min(3).min(self.0.len() - self.1);
        buf[..n].copy_from_slice(&self.0[self.1..self.1 + n]);
        self.1 += n;
        Ok(n)
    }
}

#[test]
fn websocket_frames() {
    assert_eq!(call::accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", "RFC 6455's example");
    assert_eq!(call::frame(1, b"hi"), b"\x81\x02hi");
    assert_eq!(&call::frame(2, &[0; 300])[..4], &[0x82, 126, 1, 44]);
    assert_eq!(&call::frame(2, &[0; 70000])[..10], &[0x82, 127, 0, 0, 0, 0, 0, 1, 0x11, 0x70]);
    let mut bytes = client_frame(1, br#"{"type":"start"}"#, true);
    bytes.extend(client_frame(9, b"p", true));
    bytes.extend(client_frame(2, &[1, 2], false));
    bytes.extend(client_frame(10, b"", true)); // a pong between fragments is ignored
    bytes.extend(client_frame(0, &[3; 200], true));
    bytes.extend(client_frame(8, &[3, 232, b'x'], true));
    let mut r = Trickle(bytes, 0, false);
    let mut ws = call::WsReader::new(vec![]);
    let mut events = vec![];
    let t = Instant::now();
    while events.len() < 4 && t.elapsed() < Duration::from_secs(2) {
        if let Some(e) = ws.next(&mut r).unwrap() {
            events.push(e);
        }
    }
    assert_eq!(events[0], call::Event::Text(r#"{"type":"start"}"#.into()));
    assert_eq!(events[1], call::Event::Ping(b"p".to_vec()));
    assert_eq!(events[2], call::Event::Binary([vec![1, 2], vec![3; 200]].concat()));
    assert_eq!(events[3], call::Event::Close(vec![3, 232, b'x']));
    // Unmasked frames and oversized messages end the call.
    let mut ws = call::WsReader::new(call::frame(1, b"x"));
    assert!(ws.next(&mut std::io::empty()).is_err());
    let mut big = vec![0x82, 0x80 | 127];
    big.extend(((call::MAX_MESSAGE + 1) as u64).to_be_bytes());
    assert!(call::WsReader::new(big).next(&mut std::io::empty()).is_err());
    assert!(call::WsReader::new(vec![]).next(&mut std::io::empty()).is_err(), "the phone hung up");
}

// --- A whole call -------------------------------------------------------------------------------------

/// Speech is any frame louder than 0.1; the transcript says how much was heard; the voice is 100
/// samples per character, after a pause so a cancel can come first.
struct Fake {
    spoken: AtomicUsize,
}

impl Engine for Fake {
    fn vad(&self, x: &[f32], state: &mut Vec<f32>) -> Result<f32, String> {
        assert_eq!((x.len(), state.len()), (576, 256));
        state[0] += 1.0;
        let rms = (x[64..].iter().map(|s| s * s).sum::<f32>() / 512.0).sqrt();
        Ok(if rms > 0.1 { 0.95 } else { 0.02 })
    }
    fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        Ok(format!("{} ms", audio.len() / 16))
    }
    fn speak(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, String> {
        std::thread::sleep(Duration::from_millis(150));
        self.spoken.fetch_add(1, Ordering::SeqCst);
        assert_eq!((voice, speed), ("bm_george", 1.4));
        Ok(vec![0.25; text.chars().count() * 100])
    }
}

fn pcm(ms: usize, level: f32) -> Vec<u8> {
    (0..ms * 16).flat_map(|i| (((i as f32 * 0.3).sin() * level * 32767.0) as i16).to_le_bytes()).collect()
}

struct Phone {
    c: Client,
    buf: Vec<u8>,
}

impl Phone {
    fn send(&mut self, op: u8, data: &[u8]) {
        self.c.get_mut().write_all(&client_frame(op, data, true)).unwrap();
        self.c.get_mut().flush().unwrap();
    }
    fn json(&mut self, v: Value) {
        self.send(1, v.to_string().as_bytes());
    }
    /// The next frame from the PC: (opcode, payload).
    fn recv(&mut self) -> (u8, Vec<u8>) {
        loop {
            if self.buf.len() >= 2 {
                let (op, short) = (self.buf[0] & 0x0f, (self.buf[1] & 0x7f) as usize);
                assert_eq!(self.buf[1] & 0x80, 0, "the PC never masks");
                let (head, len) = match short {
                    126 if self.buf.len() >= 4 => (4, u16::from_be_bytes([self.buf[2], self.buf[3]]) as usize),
                    127 if self.buf.len() >= 10 => (10, u64::from_be_bytes(self.buf[2..10].try_into().unwrap()) as usize),
                    126 | 127 => (usize::MAX, 0),
                    n => (2, n),
                };
                if head != usize::MAX && self.buf.len() >= head + len {
                    let payload = self.buf[head..head + len].to_vec();
                    self.buf.drain(..head + len);
                    return (op, payload);
                }
            }
            let mut chunk = [0u8; 65536];
            let n = self.c.read(&mut chunk).expect("a frame from the PC");
            assert!(n > 0, "the PC hung up");
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }
    /// JSON messages until one of `kind` (binary frames are counted into `audio`).
    fn until(&mut self, kind: &str, audio: &mut usize) -> Vec<Value> {
        let mut seen = vec![];
        loop {
            match self.recv() {
                (1, t) => {
                    let v: Value = serde_json::from_slice(&t).unwrap();
                    let done = v["type"] == kind;
                    seen.push(v);
                    if done {
                        return seen;
                    }
                }
                (2, b) => *audio += b.len(),
                (op, p) => seen.push(json!({"op": op, "payload": p})),
            }
        }
    }
}

#[test]
fn a_whole_call() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-voice-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fake = Arc::new(Fake { spoken: AtomicUsize::new(0) });
    let config = Config { dir: dir.clone(), ollama: "http://127.0.0.1:1".into(), searxng: "http://127.0.0.1:1".into(), default_ctx: 16384, media: None, voice: Some(voice::Voice::given(fake.clone())), bots: None };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let (port, pin, token) = (gw.port, id.fingerprint.clone(), id.token.clone());

    let v = ask(port, &pin, "GET", "/v1/voice/voices", Some(&token), None).json();
    assert_eq!((v["ready"].clone(), v["missing"].clone(), v["voices"].as_array().unwrap().len()), (json!(true), Value::Null, 22));
    assert_eq!(v["voices"][0], json!({"id": "af_heart", "name": "Heart", "about": "Warm and upbeat", "gender": "woman", "accent": "American"}));
    let r = ask(port, &pin, "GET", "/v1/voice/preview?voice=bm_george&speed=1.4", Some(&token), None);
    assert_eq!((r.status, r.header("Content-Type")), (200, Some("audio/L16; rate=24000; channels=1")));
    assert_eq!(r.bytes.len(), voice::preview_line("bm_george", "George").chars().count() * 100 * 2);
    assert_eq!(ask(port, &pin, "GET", "/v1/voice/preview?voice=nobody", Some(&token), None).status, 400);
    assert_eq!(ask(port, &pin, "GET", "/v1/voice/call", Some(&token), None).json()["error"]["message"], "Expected a WebSocket");

    // The call: the handshake and the first frames in one write, as a quick phone might.
    let mut c = connect(port, &pin);
    let mut hello = format!("GET /v1/voice/call HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {token}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").into_bytes();
    hello.extend(client_frame(1, json!({"type": "start", "voice": "bm_george", "speed": 3, "patience": 300}).to_string().as_bytes(), true));
    c.get_mut().write_all(&hello).unwrap();
    c.get_mut().flush().unwrap();
    let r = read_head(&mut c);
    assert_eq!(r.status, 101);
    assert_eq!(r.header("Sec-WebSocket-Accept"), Some("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
    let mut phone = Phone { c, buf: vec![] };
    let mut audio = 0;
    let first = phone.until("ready", &mut audio);
    assert_eq!(first, vec![json!({"type": "status", "text": "Waking up the voice on your PC…"}), json!({"type": "ready"})]);

    // Talking: 1.2 s of speech in odd-sized pieces, then quiet until the 300 ms patience runs out.
    let speech = pcm(1200, 0.5);
    for piece in speech.chunks(3001) {
        phone.send(2, piece);
    }
    phone.send(2, &pcm(800, 0.0));
    let heard = phone.until("final", &mut audio);
    let kinds: Vec<&str> = heard.iter().map(|v| v["type"].as_str().unwrap()).collect();
    assert_eq!(kinds.first(), Some(&"speech"));
    assert_eq!(heard[0], json!({"type": "speech", "on": true, "id": 1}));
    assert!(kinds.contains(&"partial"), "a caption while talking: {kinds:?}");
    let off = heard.iter().position(|v| v["type"] == "speech" && v["on"] == false).expect("speech off");
    assert_eq!(heard[off]["id"], 1);
    let fin = heard.last().unwrap();
    assert_eq!(fin["id"], 1);
    // From 220 ms in (+ 320 ms before it) to 6 frames after the speech ended.
    let ms: usize = fin["text"].as_str().unwrap().trim_end_matches(" ms").parse().unwrap();
    assert!((1150..=1500).contains(&ms), "{ms} ms heard");

    // The reply, streamed in pieces: spoken sentence by sentence, then the turn ends.
    phone.json(json!({"type": "say", "turn": 1, "text": "Hello there! *smiles* I am fine,"}));
    phone.json(json!({"type": "say", "turn": 1, "text": " thank you for asking. Bye"}));
    phone.json(json!({"type": "say_end", "turn": 1}));
    let said = phone.until("turn_end", &mut audio);
    let segs: Vec<&Value> = said.iter().filter(|v| v["type"] == "seg").collect();
    let words: Vec<&str> = segs.iter().map(|v| v["text"].as_str().unwrap()).collect();
    assert_eq!(words, ["Hello there!", "I am fine, thank you for asking.", "Bye"]);
    assert_eq!(segs.iter().map(|v| v["seq"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 2, 3]);
    let samples: u64 = segs.iter().map(|v| v["samples"].as_u64().unwrap()).sum();
    assert_eq!(samples, (12 + 32 + 3) * 100);
    assert_eq!(audio as u64, samples * 2, "every sample sent as 16-bit PCM");
    assert_eq!(said.last().unwrap(), &json!({"type": "turn_end", "turn": 1}));

    // Barge-in: a cancel right after a reply starts drops its speech, and the next turn goes on.
    audio = 0;
    let before = fake.spoken.load(Ordering::SeqCst);
    phone.json(json!({"type": "say", "turn": 2, "text": "This will be cut off. Never heard. "}));
    phone.json(json!({"type": "cancel"}));
    phone.json(json!({"type": "say", "turn": 3, "text": "Yes?"}));
    phone.json(json!({"type": "say_end", "turn": 3}));
    let said = phone.until("turn_end", &mut audio);
    assert!(said.iter().all(|v| v["turn"] != 2), "nothing of the cancelled turn: {said:?}");
    assert_eq!(said.iter().filter(|v| v["type"] == "seg").map(|v| v["text"].clone()).collect::<Vec<_>>(), [json!("Yes?")]);
    assert_eq!(audio, 400 * 2);
    assert!(fake.spoken.load(Ordering::SeqCst) - before <= 2, "at most the sentence already being made");

    // Pings are answered; a close is echoed and ends the call.
    phone.send(9, b"are you there");
    assert_eq!(phone.recv(), (10, b"are you there".to_vec()));
    phone.send(8, &[3, 232]);
    assert_eq!(phone.recv(), (8, vec![3, 232]));
    let t = Instant::now();
    let st = loop {
        let st = ask(port, &pin, "GET", "/v1/cubicai/status", Some(&token), None).json();
        if st["recent"][0]["model"] == "voice call" || t.elapsed() > Duration::from_secs(5) {
            break st;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!((st["recent"][0]["model"].clone(), st["recent"][0]["outcome"].clone()), (json!("voice call"), json!("ok")));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn not_set_up() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-novoice-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let paths = voice::Paths { dir: dir.join("voice"), ort_lib: dir.join("nope.so"), ..voice::Paths::from_env() };
    assert_eq!(paths.missing().as_deref(), Some("ONNX Runtime (run: cubicai-gateway voice-setup)"));
    std::fs::create_dir_all(&paths.dir).unwrap();
    std::fs::write(&paths.ort_lib, b"").unwrap();
    assert_eq!(paths.missing().as_deref(), Some("the voice models (run: cubicai-gateway voice-setup)"));
    let config = Config { dir: dir.clone(), ollama: "http://127.0.0.1:1".into(), searxng: "http://127.0.0.1:1".into(), default_ctx: 16384, media: None, voice: Some(voice::Voice::new(paths)), bots: None };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let v = ask(gw.port, &id.fingerprint, "GET", "/v1/voice/voices", Some(&id.token), None).json();
    assert_eq!((v["ready"].clone(), v["missing"].clone()), (json!(false), json!("the voice models (run: cubicai-gateway voice-setup)")));
    let mut c = connect(gw.port, &id.fingerprint);
    c.get_mut().write_all(format!("GET /v1/voice/call HTTP/1.1\r\nAuthorization: Bearer {}\r\nUpgrade: websocket\r\nSec-WebSocket-Key: x\r\n\r\n", id.token).as_bytes()).unwrap();
    let r = read_resp(&mut c);
    assert_eq!((r.status, r.json()["error"]["message"].clone()), (503, json!("Voice calls need the voice models (run: cubicai-gateway voice-setup) on your PC.")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A call with the real models: Kokoro's voice streamed in real time as the phone's microphone,
/// Parakeet's transcript back, then a reply spoken. Prints the delays the phone would feel.
/// `cargo test -p cubic-gateway --release --test voice real_call -- --ignored --nocapture`
#[test]
#[ignore]
fn real_call_with_the_real_models() {
    let dir = std::env::temp_dir().join(format!("cubic-gw-realcall-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let v = voice::Voice::new(voice::Paths::from_env());
    let line = "Hey, can you tell me what the weather looks like for a walk this afternoon?";
    let config = Config { dir: dir.clone(), ollama: "http://127.0.0.1:1".into(), searxng: "http://127.0.0.1:1".into(), default_ctx: 16384, media: None, voice: Some(v.clone()), bots: None };
    let id = identity::load(&config.dir).unwrap();
    let gw = server::start(config, &id, "127.0.0.1", 0, false).expect("gateway starts");
    let mut c = connect(gw.port, &id.fingerprint);
    c.get_mut()
        .write_all(format!("GET /v1/voice/call HTTP/1.1\r\nAuthorization: Bearer {}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n", id.token).as_bytes())
        .unwrap();
    assert_eq!(read_head(&mut c).status, 101);
    let mut phone = Phone { c, buf: vec![] };
    let t = Instant::now();
    let mut audio = 0;
    phone.until("ready", &mut audio);
    println!("voice ready after {:.1} s (models loading)", t.elapsed().as_secs_f64());
    phone.json(json!({"type": "start", "voice": "af_heart", "speed": 1.0, "patience": 600}));
    // The user's words: Kokoro at 24 kHz → 16 kHz, a little quiet before and plenty after.
    let said = v.engine().unwrap().speak(line, "am_michael", 1.0).unwrap();
    let mut mic: Vec<f32> = vec![0.0; 8000];
    mic.extend((0..said.len() * 2 / 3).map(|j| said[j * 3 / 2]));
    let speech_end = mic.len();
    mic.extend(vec![0.0; 3 * 16000]);
    let pcm: Vec<u8> = mic.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect();
    // Real time, 40 ms per message like the phone; what the PC says is read in between.
    phone.c.get_ref().sock.set_read_timeout(Some(Duration::from_millis(5))).unwrap();
    let start = Instant::now();
    let mut events: Vec<(f64, Value)> = vec![];
    let mut final_text = None;
    for (n, chunk) in pcm.chunks(1280).enumerate() {
        phone.send(2, chunk);
        let due = start + Duration::from_millis(40 * (n as u64 + 1));
        while Instant::now() < due {
            if let Some((1, t)) = try_recv(&mut phone) {
                let v: Value = serde_json::from_slice(&t).unwrap();
                let at = start.elapsed().as_secs_f64();
                if v["type"] == "final" {
                    final_text = Some((at, v["text"].as_str().unwrap().to_string()));
                }
                events.push((at, v));
            }
        }
        if final_text.is_some() {
            break;
        }
    }
    let stopped = speech_end as f64 / 16000.0;
    for (at, v) in &events {
        println!("  {at:5.2} s  {v}");
    }
    let (at, heard) = final_text.expect("a final transcript");
    println!("speech ended at {stopped:.2} s; final transcript {:.2} s later (600 ms of it is the pause that ends a turn)", at - stopped);
    assert!(heard.to_lowercase().contains("weather") && heard.to_lowercase().contains("walk"), "{heard}");
    assert!(events.iter().any(|(_, v)| v["type"] == "partial"), "captions while talking");

    // The reply: the first sentence is voiced while the rest is still coming.
    phone.c.get_ref().sock.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let t = Instant::now();
    phone.json(json!({"type": "say", "turn": 1, "text": "Sure! It's mild and dry this afternoon, "}));
    let mut audio = 0;
    let first = phone.until("seg", &mut audio);
    println!("first sentence voiced {:.2} s after it arrived: {}", t.elapsed().as_secs_f64(), first.last().unwrap());
    phone.json(json!({"type": "say", "turn": 1, "text": "so a walk sounds lovely. Take a light jacket just in case."}));
    phone.json(json!({"type": "say_end", "turn": 1}));
    let rest = phone.until("turn_end", &mut audio);
    let segs: Vec<String> = first.iter().chain(&rest).filter(|v| v["type"] == "seg").map(|v| v["text"].as_str().unwrap().to_string()).collect();
    println!("{} sentences, {:.1} s of voice: {segs:?}", segs.len(), audio as f64 / 2.0 / 24000.0);
    assert_eq!(segs.join(" "), "Sure! It's mild and dry this afternoon, so a walk sounds lovely. Take a light jacket just in case.");
    phone.send(8, &[3, 232]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// One frame if one arrives before the read times out.
fn try_recv(p: &mut Phone) -> Option<(u8, Vec<u8>)> {
    let mut chunk = [0u8; 65536];
    match p.c.read(&mut chunk) {
        Ok(n) if n > 0 => p.buf.extend_from_slice(&chunk[..n]),
        Ok(_) => panic!("the PC hung up"),
        Err(_) => {}
    }
    if p.buf.len() < 2 {
        return None;
    }
    let short = (p.buf[1] & 0x7f) as usize;
    let (head, len) = match short {
        126 if p.buf.len() >= 4 => (4, u16::from_be_bytes([p.buf[2], p.buf[3]]) as usize),
        127 if p.buf.len() >= 10 => (10, u64::from_be_bytes(p.buf[2..10].try_into().unwrap()) as usize),
        126 | 127 => return None,
        n => (2, n),
    };
    if p.buf.len() < head + len {
        return None;
    }
    let op = p.buf[0] & 0x0f;
    let payload = p.buf[head..head + len].to_vec();
    p.buf.drain(..head + len);
    Some((op, payload))
}
