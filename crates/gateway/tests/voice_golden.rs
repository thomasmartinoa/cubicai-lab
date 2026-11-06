//! The Rust voice against the Python one (`cubicai_voice.py` and the libraries it uses), on this
//! PC's real models: the words said, the sentences cut, the phonemes, the ASR features, Kokoro's
//! audio and Parakeet's transcripts. Needs the CubicAI server's venv and the voice models.
//! `cargo test -p cubic-gateway --release --test voice_golden -- --ignored --nocapture`

use cubic_gateway::voice::{self, asr, phonemes::Espeak, text, tts, Engine};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

fn server_dir() -> PathBuf {
    std::env::var("CUBICAI_SERVER_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../cubicai/server"))
}

fn python(mode: &str, input: &Value) -> Value {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/voice_golden.py");
    let mut child = std::process::Command::new(server_dir().join(".venv/bin/python"))
        .arg(&script)
        .arg(server_dir())
        .arg(mode)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("the server's venv");
    child.stdin.take().unwrap().write_all(input.to_string().as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // The Python voice logs to stdout too: the answer is the last line.
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(text.trim().lines().last().unwrap_or("null")).unwrap_or(Value::Null)
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("cubic-voice-golden-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

pub const SPEAKABLE: [&str; 12] = [
    "**Bold** and *smiles* text",
    "See [the docs](https://x.y) and https://example.com/path now.",
    "Facts [1] and more [2, 3] and a range [4–6].",
    "- item one\n- item two\n# Heading\n1. numbered\n• dot",
    "Emoji 😀👍🏽 and ☀️ sun ❤️‍🔥",
    "`code` and > quote | pipe ~ tilde _under_ #tag",
    "*** ",
    "Ünïcödé façade — naïve… 日本語",
    "   \n\t  ",
    "😀",
    "*waves* Hi there! *grins*",
    "Ok\n\n\nnext   line",
];

fn segment_cases() -> Vec<Vec<&'static str>> {
    vec![
        vec!["Hello", " there! How", " are you doing today? I hope", " everything is fine.", " Bye"],
        vec!["Hi. ", "This is a long sentence without any punctuation that keeps going and going and going on and on and on for a while longer than sixty characters"],
        vec!["Sure *smiles warmly. And", " more* done. Next sentence here is good. "],
        vec!["Code: ```python\nprint('hi. there')\n``` after. Then", " more text follows here. ", "End"],
        vec!["First sentence is here. ", "Second, with a comma; and a semicolon: plus a long tail that keeps on going past the two hundred character limit for later sentences, because we need to see the soft cut work properly when nothing ends the sentence soon enough here, really and truly"],
        vec!["…", "Ok", "\n\n", "Line two\n"],
        vec!["Unicode ünïcödé sentence one. ", "Ελληνικά δεύτερη πρόταση εδώ μετά από αυτό. ", "日本語の文。とても長い文章がここに続きますがピリオドはありません"],
        vec!["Ein Satz mit „Anführungszeichen“! Und dann (Klammern) weiter. ", "```unclosed code. With dots. "],
        vec!["No end at all"],
        vec!["A **bold** move. ", "*A whole action that is long. Very long indeed.* Then words. "],
    ]
}

#[test]
#[ignore]
fn golden_text_matches_python() {
    let py = python("speakable", &json!(SPEAKABLE));
    let rs: Vec<String> = SPEAKABLE.iter().map(|t| text::speakable(t)).collect();
    assert_eq!(json!(rs), py);
    let cases = segment_cases();
    let py = python("segments", &json!(cases));
    let rs: Vec<Value> = cases
        .iter()
        .map(|pieces| {
            let mut s = text::Segmenter::new();
            let fed: Vec<Vec<String>> = pieces.iter().map(|p| s.feed(p)).collect();
            json!({"fed": fed, "flush": s.flush()})
        })
        .collect();
    for (i, (a, b)) in rs.iter().zip(py.as_array().unwrap()).enumerate() {
        assert_eq!(a, b, "segmenter case {i}");
    }
    println!("{} speakable and {} segmenter cases match", SPEAKABLE.len(), cases.len());
}

fn phoneme_cases() -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&str, &str)> = [
        "Hello, world! How are you?",
        "It's 3:45 pm on Saturday, 7 November 2026.",
        "Dr. Smith said \"No!\" — then left…",
        "(Parentheses) and [brackets] {braces}.",
        "What?! Really?!",
        "A, B, C; D: E.",
        "The quick brown fox jumps over the lazy dog",
        "e.g. etc. i.e. Mr. and Mrs.",
        "100% sure, $5.99 or €3",
        "",
        ", leading comma",
        "trailing comma,",
        "…",
        "«quoted» and “curly” ones",
        "Hey, it's Heart. This is how I sound, so, what do you want to talk about?",
        "I can't believe it's already 2026; time flies, doesn't it?",
        "!!!",
        "Wait... what?",
    ]
    .iter()
    .map(|t| (*t, "en-us"))
    .collect();
    v.extend([
        ("Hello, I'm Emma. This is how I sound. What shall we talk about?", "en-gb"),
        ("Hola, soy Dora. Así suena mi voz. ¿De qué quieres hablar?", "es"),
        ("¿Qué? ¡Sí!", "es"),
        ("Salut, c'est Siwis. Voilà ma voix. De quoi veux-tu parler ?", "fr-fr"),
        ("J'ai un rendez-vous avec Microsoft et Google demain.", "fr-fr"),
        ("Ciao, sono Sara. Questa è la mia voce. Di cosa vuoi parlare?", "it"),
        ("Oi, eu sou a Dora. Essa é a minha voz. Sobre o que você quer falar?", "pt-br"),
        ("नमस्ते, मैं Alpha हूँ। मेरी आवाज़ ऐसी है। आप किस बारे में बात करना चाहेंगे?", "hi"),
    ]);
    v
}

#[test]
#[ignore]
fn golden_phonemes_match_python() {
    let p = voice::Paths::from_env();
    let espeak = Espeak::load(&p.espeak_lib, &p.espeak_data).unwrap();
    let cases = phoneme_cases();
    let py = python("phonemes", &json!(cases));
    for (i, (t, lang)) in cases.iter().enumerate() {
        let rs = tts::keep_known(&espeak.phonemize(t.trim(), lang).unwrap());
        assert_eq!(json!(rs), py[i], "{lang}: {t:?}");
    }
    // The punctuation step on its own.
    let texts: Vec<&str> = cases.iter().map(|c| c.0).filter(|t| !t.is_empty()).collect();
    let py = python("punct", &json!(texts));
    for (i, t) in texts.iter().enumerate() {
        let (chunks, marks) = voice::phonemes::preserve(t);
        let marks: Vec<Value> = marks.iter().map(|m| json!([m.index, m.mark, m.position.to_string()])).collect();
        assert_eq!(json!({"chunks": chunks, "marks": marks}), py[i], "{t:?}");
    }
    // Batches: a long reply is split at punctuation below 510 phonemes.
    let long = "həlˈoʊ, wˈɜːld! ".repeat(60) + &"ðɪs ɪz ɐ lˈɔŋ sˈɛntəns wɪðˌaʊt pˈʌŋktʃuːˌeɪʃən ".repeat(20);
    let splits = vec!["", "ɐ", "a, b. c! d? e; f", long.as_str()];
    assert_eq!(json!(splits.iter().map(|s| tts::split_phonemes(s)).collect::<Vec<_>>()), python("split", &json!(splits)));
    println!("{} phoneme cases in {} languages match", cases.len(), 7);
}

#[test]
#[ignore]
fn golden_asr_features_match_python() {
    let dir = scratch();
    python("bank", &json!({"out": dir.join("bank.f32")}));
    let py_bank = read_f32(&dir.join("bank.f32"));
    let bank = asr::mel_bank();
    assert_eq!(py_bank.len(), bank.len());
    let peak = py_bank.iter().cloned().fold(0f32, f32::max);
    let (worst, at) = bank.iter().zip(&py_bank).enumerate().map(|(i, (a, b))| ((a - b).abs(), i)).fold((0f32, 0), |m, x| if x.0 > m.0 { x } else { m });
    println!("mel bank: max difference {worst:.2e} (bin {}, mel {}: {} vs {}), peak {peak}", at / 128, at % 128, bank[at], py_bank[at]);
    assert!(worst < 1e-6 * peak.max(1.0), "mel bank differs by {worst}");
    // A sweep with noise, 1.73 s.
    let n = 27_680;
    let mut seed = 12345u32;
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let t = i as f32 / 16000.0;
            0.4 * (2.0 * std::f32::consts::PI * (200.0 + 900.0 * t) * t).sin() + 0.05 * ((seed >> 8) as f32 / 16_777_216.0 - 0.5)
        })
        .collect();
    write_f32(&dir.join("audio.f32"), &audio);
    let info = python("features", &json!({"audio": dir.join("audio.f32"), "onnx": dir.join("onnx.f32"), "numpy": dir.join("numpy.f32")}));
    let (feats, frames, valid) = asr::features(&audio, &bank, &asr::Fft::new());
    assert_eq!(info["shape"], json!([1, 128, frames]));
    assert_eq!((info["len"].as_u64().unwrap() as usize, info["numpy_len"].as_u64().unwrap() as usize), (valid, valid));
    for which in ["onnx", "numpy"] {
        let py = read_f32(&dir.join(format!("{which}.f32")));
        let worst = feats.iter().zip(&py).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("features vs {which}: max difference {worst:.2e}");
        assert!(worst < 2e-3, "features differ from {which} by {worst}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore]
fn golden_speech_and_transcripts_match_python() {
    let dir = scratch();
    let lines = [
        ("Hello there! I'm testing the new voice. Can you hear me clearly?", "af_heart", 1.0f32),
        ("The quick brown fox jumps over the lazy dog, twice.", "am_michael", 1.2),
        ("Hola, ¿cómo estás hoy?", "ef_dora", 0.9),
    ];
    let py_lens = python("speak", &json!({"dir": dir, "lines": lines}));
    let t = std::time::Instant::now();
    let m = voice::models::Models::load(&voice::Paths::from_env()).unwrap();
    println!("models loaded in {:.1} s", t.elapsed().as_secs_f64());
    let mut paths = vec![];
    for (i, (text, v, speed)) in lines.iter().enumerate() {
        let t = std::time::Instant::now();
        let rs = m.speak(text, v, *speed).unwrap();
        let took = t.elapsed().as_secs_f64();
        let py = read_f32(&dir.join(format!("py{i}.f32")));
        assert_eq!(rs.len() as u64, py_lens[i].as_u64().unwrap(), "{text}: same length");
        let worst = rs.iter().zip(&py).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{v} {speed}: {} samples ({:.2} s of voice in {took:.2} s), max difference {worst:.1e}", rs.len(), rs.len() as f64 / 24000.0);
        assert!(worst < 1e-3, "{text}: audio differs by {worst}");
        // Parakeet hears 16 kHz: Python's audio, resampled the simple way.
        let at16: Vec<f32> = (0..py.len() * 2 / 3).map(|j| py[j * 3 / 2]).collect();
        let p = dir.join(format!("in{i}.f32"));
        write_f32(&p, &at16);
        paths.push(p);
    }
    let py_text = python("transcribe", &json!(paths));
    for (i, p) in paths.iter().enumerate() {
        let t = std::time::Instant::now();
        let rs = m.transcribe(&read_f32(p)).unwrap();
        println!("heard {:?} in {:.2} s", rs, t.elapsed().as_secs_f64());
        assert_eq!(json!(rs), py_text[i]);
    }
    assert!(py_text[1].as_str().unwrap().to_lowercase().contains("quick brown fox"));
    let _ = std::fs::remove_dir_all(&dir);
}
