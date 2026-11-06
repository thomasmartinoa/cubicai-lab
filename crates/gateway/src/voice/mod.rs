//! Voice calls: the PC is the call's ears and mouth, the phone its brain. A port of
//! `cubicai_voice.py`: Silero VAD hears when the user talks, Parakeet TDT writes it down, Kokoro
//! speaks the reply sentence by sentence. All on the CPU (ONNX Runtime), so chat keeps the GPU.
//! Nothing that is said is logged or stored.

pub mod asr;
pub mod call;
pub mod models;
pub mod phonemes;
pub mod text;
pub mod tts;

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const IN_RATE: usize = 16000;
pub const OUT_RATE: usize = 24000;
/// Silero's frame at 16 kHz: 32 ms.
pub const FRAME: usize = 512;
pub const FRAME_MS: usize = FRAME * 1000 / IN_RATE;
pub const ORT_VERSION: &str = "1.31.1";

pub const ASR_REPO: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
pub const ASR_FILES: [&str; 4] = ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt", "config.json"];
pub const VAD_REPO: &str = "istupakov/silero-vad-onnx";
pub const TTS_URL: &str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/";

/// (id, name, about, gender, accent): the voices the phone offers, Kokoro's best. The first is the default.
pub const VOICES: [(&str, &str, &str, &str, &str); 22] = [
    ("af_heart", "Heart", "Warm and upbeat", "woman", "American"),
    ("af_bella", "Bella", "Bright and expressive", "woman", "American"),
    ("af_nicole", "Nicole", "Soft and close, almost a whisper", "woman", "American"),
    ("af_aoede", "Aoede", "Calm and soothing", "woman", "American"),
    ("af_kore", "Kore", "Clear and confident", "woman", "American"),
    ("af_sarah", "Sarah", "Friendly, everyday", "woman", "American"),
    ("af_sky", "Sky", "Light and young", "woman", "American"),
    ("bf_emma", "Emma", "Composed and polished", "woman", "British"),
    ("bf_isabella", "Isabella", "Warm, a little posh", "woman", "British"),
    ("am_michael", "Michael", "Calm and steady", "man", "American"),
    ("am_fenrir", "Fenrir", "Deep and rough", "man", "American"),
    ("am_puck", "Puck", "Playful and quick", "man", "American"),
    ("am_eric", "Eric", "Smooth and laid-back", "man", "American"),
    ("am_onyx", "Onyx", "Low and smooth", "man", "American"),
    ("bm_george", "George", "Classic narrator", "man", "British"),
    ("bm_fable", "Fable", "Storyteller", "man", "British"),
    ("bm_lewis", "Lewis", "Deep and serious", "man", "British"),
    ("ef_dora", "Dora", "Speaks Spanish", "woman", "Spanish"),
    ("ff_siwis", "Siwis", "Speaks French", "woman", "French"),
    ("if_sara", "Sara", "Speaks Italian", "woman", "Italian"),
    ("pf_dora", "Dora (PT)", "Speaks Portuguese", "woman", "Brazilian"),
    ("hf_alpha", "Alpha", "Speaks Hindi", "woman", "Indian"),
];

pub fn voices_json() -> Value {
    VOICES.iter().map(|(id, name, about, gender, accent)| json!({"id": id, "name": name, "about": about, "gender": gender, "accent": accent})).collect()
}

pub fn is_voice(id: &str) -> bool {
    VOICES.iter().any(|v| v.0 == id)
}

/// espeak's language for a voice (by its first letter).
pub fn lang_of(voice: &str) -> &'static str {
    match voice.chars().next() {
        Some('b') => "en-gb",
        Some('e') => "es",
        Some('f') => "fr-fr",
        Some('i') => "it",
        Some('p') => "pt-br",
        Some('h') => "hi",
        _ => "en-us",
    }
}

pub fn preview_line(voice: &str, name: &str) -> String {
    let line = match voice.chars().next() {
        Some('b') => "Hello, I'm {name}. This is how I sound. What shall we talk about?",
        Some('e') => "Hola, soy {name}. Así suena mi voz. ¿De qué quieres hablar?",
        Some('f') => "Salut, c'est {name}. Voilà ma voix. De quoi veux-tu parler ?",
        Some('i') => "Ciao, sono {name}. Questa è la mia voce. Di cosa vuoi parlare?",
        Some('p') => "Oi, eu sou a {name}. Essa é a minha voz. Sobre o que você quer falar?",
        Some('h') => "नमस्ते, मैं {name} हूँ। मेरी आवाज़ ऐसी है। आप किस बारे में बात करना चाहेंगे?",
        _ => "Hey, it's {name}. This is how I sound, so, what do you want to talk about?",
    };
    line.replace("{name}", name)
}

pub fn log(msg: &str) {
    crate::log(&format!("voice: {msg}"));
}

/// The ears and the mouth. The real one is ONNX ([`models::Models`]); tests bring their own.
pub trait Engine: Send + Sync {
    /// Speech likelihood of one frame (`x` = the last 64 samples + 512 new ones), updating `state`.
    fn vad(&self, x: &[f32], state: &mut Vec<f32>) -> Result<f32, String>;
    /// 16 kHz audio → text.
    fn transcribe(&self, audio: &[f32]) -> Result<String, String>;
    /// Text → 24 kHz audio.
    fn speak(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, String>;
}

/// Where the voice models are; the same defaults and variables as the Python gateway.
#[derive(Clone, Debug)]
pub struct Paths {
    pub dir: PathBuf,
    pub espeak_lib: String,
    pub espeak_data: String,
    pub ort_lib: PathBuf,
    pub idle_unload: Duration,
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let dir = std::env::var("CUBICAI_VOICE_DIR").map(|v| v.strip_prefix("~/").map(|r| home.join(r)).unwrap_or_else(|| PathBuf::from(&v))).unwrap_or_else(|_| home.join(".local/share/cubicai-server/voice"));
        Paths {
            ort_lib: std::env::var_os("CUBICAI_ORT_LIB").map(PathBuf::from).unwrap_or_else(|| dir.join(format!("libonnxruntime.so.{ORT_VERSION}"))),
            dir,
            espeak_lib: std::env::var("CUBICAI_ESPEAK_LIB").unwrap_or_else(|_| "/usr/lib/libespeak-ng.so.1".into()),
            espeak_data: std::env::var("CUBICAI_ESPEAK_DATA").unwrap_or_else(|_| "/usr/share/espeak-ng-data".into()),
            idle_unload: Duration::from_secs(std::env::var("CUBICAI_VOICE_IDLE").ok().and_then(|v| v.parse().ok()).unwrap_or(900)),
        }
    }

    pub fn asr_dir(&self) -> PathBuf {
        self.dir.join("parakeet-tdt-0.6b-v3")
    }
    pub fn vad_file(&self) -> PathBuf {
        self.dir.join("silero_vad.onnx")
    }
    pub fn tts_model(&self) -> PathBuf {
        self.dir.join("kokoro-v1.0.onnx")
    }
    pub fn tts_voices(&self) -> PathBuf {
        self.dir.join("voices-v1.0.bin")
    }

    /// What's still needed for calls, or None when everything is here.
    pub fn missing(&self) -> Option<String> {
        if !self.ort_lib.is_file() {
            return Some("ONNX Runtime (run: cubicai-gateway voice-setup)".into());
        }
        let files = ASR_FILES.iter().map(|f| self.asr_dir().join(f)).chain([self.vad_file(), self.tts_model(), self.tts_voices()]);
        if files.into_iter().any(|f| !f.is_file()) {
            return Some("the voice models (run: cubicai-gateway voice-setup)".into());
        }
        if !std::path::Path::new(&self.espeak_lib).exists() {
            return Some("espeak-ng (install it with your package manager)".into());
        }
        None
    }
}

type PreviewKey = (String, i64);

enum Source {
    Real(Paths),
    Given(Arc<dyn Engine>),
}

impl std::fmt::Debug for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.source {
            Source::Real(p) => write!(f, "Voice({})", p.dir.display()),
            Source::Given(_) => f.write_str("Voice(given engine)"),
        }
    }
}

/// The engine, loaded on the first call and dropped after a while without one.
pub struct Voice {
    source: Source,
    loaded: Mutex<Option<Arc<dyn Engine>>>,
    pub calls: AtomicUsize,
    last_used: Mutex<Instant>,
    /// (voice, speed in hundredths) → PCM, oldest first.
    previews: Mutex<Vec<(PreviewKey, Arc<Vec<u8>>)>>,
}

impl Voice {
    pub fn new(paths: Paths) -> Arc<Voice> {
        let idle = paths.idle_unload;
        let v = Arc::new(Voice::with(Source::Real(paths)));
        let w = Arc::downgrade(&v);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(60));
                let Some(v) = w.upgrade() else { return };
                if v.calls.load(Ordering::SeqCst) == 0 && v.last_used.lock().unwrap().elapsed() > idle && v.loaded.lock().unwrap().take().is_some() {
                    log("speech models unloaded (no calls)");
                }
            }
        });
        v
    }

    /// A voice with this engine (tests), always ready.
    pub fn given(engine: Arc<dyn Engine>) -> Arc<Voice> {
        Arc::new(Voice::with(Source::Given(engine)))
    }

    fn with(source: Source) -> Voice {
        Voice { source, loaded: Mutex::new(None), calls: AtomicUsize::new(0), last_used: Mutex::new(Instant::now()), previews: Mutex::new(vec![]) }
    }

    pub fn missing(&self) -> Option<String> {
        match &self.source {
            Source::Real(p) => p.missing(),
            Source::Given(_) => None,
        }
    }

    pub fn touch(&self) {
        *self.last_used.lock().unwrap() = Instant::now();
    }

    pub fn engine(&self) -> Result<Arc<dyn Engine>, String> {
        let mut loaded = self.loaded.lock().unwrap();
        self.touch();
        if let Some(e) = loaded.as_ref() {
            return Ok(e.clone());
        }
        let e: Arc<dyn Engine> = match &self.source {
            Source::Given(e) => e.clone(),
            Source::Real(p) => {
                let t = Instant::now();
                let m = models::Models::load(p)?;
                log(&format!("speech models loaded in {:.1}s", t.elapsed().as_secs_f64()));
                Arc::new(m)
            }
        };
        *loaded = Some(e.clone());
        Ok(e)
    }

    /// A short sample of a voice as 16-bit PCM at 24 kHz (cached). Err(true) for an unknown voice.
    pub fn preview(&self, voice: &str, speed: f32) -> Result<Arc<Vec<u8>>, (bool, String)> {
        let Some(v) = VOICES.iter().find(|v| v.0 == voice) else { return Err((true, "Unknown voice".into())) };
        let speed = speed.clamp(0.7, 1.4);
        let key = (voice.to_string(), (speed as f64 * 100.0).round_ties_even() as i64);
        if let Some(p) = self.previews.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(p.1.clone());
        }
        let engine = self.engine().map_err(|e| (false, e))?;
        let pcm = Arc::new(tts::pcm16(&engine.speak(&preview_line(voice, v.1), voice, speed).map_err(|e| (false, e))?));
        let mut cache = self.previews.lock().unwrap();
        cache.push((key, pcm.clone()));
        if cache.len() > 40 {
            cache.remove(0);
        }
        Ok(pcm)
    }
}

// --- voice-setup ----------------------------------------------------------------------------------

fn download(url: &str, target: &std::path::Path) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(30))).timeout_recv_body(Some(Duration::from_secs(1800))).build().into();
    let mut r = agent.get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let part = target.with_extension("part");
    let mut f = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    std::io::copy(&mut r.body_mut().with_config().limit(u64::MAX).reader(), &mut f).map_err(|e| e.to_string())?;
    std::fs::rename(&part, target).map_err(|e| e.to_string())
}

/// Downloads what calls need into the voice folder (the models are shared with the Python gateway).
pub fn setup(p: &Paths) -> Result<(), String> {
    std::fs::create_dir_all(p.asr_dir()).map_err(|e| e.to_string())?;
    for name in ASR_FILES {
        let target = p.asr_dir().join(name);
        if !target.is_file() {
            println!("Parakeet: {name}…");
            download(&format!("https://huggingface.co/{ASR_REPO}/resolve/main/{name}"), &target)?;
        }
    }
    if !p.vad_file().is_file() {
        println!("Silero VAD…");
        download(&format!("https://huggingface.co/{VAD_REPO}/resolve/main/silero_vad.onnx"), &p.vad_file())?;
    }
    for target in [p.tts_model(), p.tts_voices()] {
        if !target.is_file() {
            let name = target.file_name().unwrap().to_string_lossy().into_owned();
            println!("Kokoro: {name}…");
            download(&format!("{TTS_URL}{name}"), &target)?;
        }
    }
    if !p.ort_lib.is_file() {
        println!("ONNX Runtime {ORT_VERSION}…");
        let tgz = p.dir.join("onnxruntime.tgz");
        download(&format!("https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-x64-{ORT_VERSION}.tgz"), &tgz)?;
        let member = format!("onnxruntime-linux-x64-{ORT_VERSION}/lib/libonnxruntime.so.{ORT_VERSION}");
        let out = std::process::Command::new("tar").arg("-xzf").arg(&tgz).arg("-C").arg(&p.dir).arg("--strip-components=2").arg(&member).output().map_err(|e| format!("tar: {e}"))?;
        let _ = std::fs::remove_file(&tgz);
        let unpacked = p.dir.join(format!("libonnxruntime.so.{ORT_VERSION}"));
        if !out.status.success() || !unpacked.is_file() {
            return Err(format!("couldn't unpack ONNX Runtime: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        if unpacked != p.ort_lib {
            std::fs::rename(&unpacked, &p.ort_lib).map_err(|e| e.to_string())?;
        }
    }
    match p.missing() {
        None => println!("Voice calls are ready."),
        Some(left) => println!("Still missing: {left}"),
    }
    Ok(())
}
