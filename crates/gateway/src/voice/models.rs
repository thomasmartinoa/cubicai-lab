//! The real engine: Silero VAD, Parakeet TDT 0.6B v3 (int8) and Kokoro v1.0 on ONNX Runtime's CPU
//! provider, loaded from the voice folder.

use super::{asr, phonemes::Espeak, tts, Engine, Paths, IN_RATE};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn e(err: impl std::fmt::Display) -> String {
    err.to_string()
}

/// ONNX Runtime, loaded once from `ort_lib`, telemetry off (nothing about calls leaves this PC).
fn runtime(lib: &std::path::Path) -> Result<&'static Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init_from(lib).map_err(e)?.with_name("cubicai-voice").with_telemetry(false).build().map_err(e)).as_ref().map_err(Clone::clone)
}

fn session(env: &Environment, path: &std::path::Path, single_thread: bool) -> Result<Session, String> {
    let mut b = Session::builder(env).map_err(e)?;
    b = if single_thread {
        // VAD frames are tiny: more threads only add overhead.
        b.with_intra_threads(1).map_err(e)?.with_inter_threads(1).map_err(e)?
    } else {
        // Listening and speaking share the CPU: threads that spin-wait after a run would slow the other.
        b.with_config_entry("session.intra_op.allow_spinning", "0").map_err(e)?
    };
    b.commit_from_file(path).map_err(|err| format!("{}: {err}", path.display()))
}

struct Asr {
    encoder: Session,
    decoder: Session,
    vocab: Vec<String>,
    blank: usize,
    bank: Vec<f32>,
    fft: asr::Fft,
}

struct Tts {
    session: Session,
    voices: HashMap<String, Vec<f32>>,
    espeak: Espeak,
}

pub struct Models {
    vad: Mutex<Session>,
    asr: Mutex<Asr>,
    tts: Mutex<Tts>,
}

impl Models {
    pub fn load(p: &Paths) -> Result<Models, String> {
        if let Some(m) = p.missing() {
            return Err(format!("Voice calls need {m} on your PC."));
        }
        let env = runtime(&p.ort_lib)?;
        let (vocab, blank) = asr::vocab(&std::fs::read_to_string(p.asr_dir().join("vocab.txt")).map_err(e)?)?;
        let m = Models {
            vad: Mutex::new(session(env, &p.vad_file(), true)?),
            asr: Mutex::new(Asr { encoder: session(env, &p.asr_dir().join("encoder-model.int8.onnx"), false)?, decoder: session(env, &p.asr_dir().join("decoder_joint-model.int8.onnx"), false)?, vocab, blank, bank: asr::mel_bank(), fft: asr::Fft::new() }),
            tts: Mutex::new(Tts { session: session(env, &p.tts_model(), false)?, voices: tts::read_voices(&std::fs::read(p.tts_voices()).map_err(e)?)?, espeak: Espeak::load(&p.espeak_lib, &p.espeak_data)? }),
        };
        // A first run of each, so the first words of a call aren't slow.
        m.transcribe(&vec![0.0; IN_RATE])?;
        m.speak("Hi.", super::VOICES[0].0, 1.0)?;
        Ok(m)
    }
}

impl Engine for Models {
    fn vad(&self, x: &[f32], state: &mut Vec<f32>) -> Result<f32, String> {
        let mut s = self.vad.lock().unwrap();
        let out = s
            .run(ort::inputs![
                "input" => Tensor::from_array(([1usize, x.len()], x.to_vec())).map_err(e)?,
                "state" => Tensor::from_array(([2usize, 1, 128], state.clone())).map_err(e)?,
                "sr" => Tensor::from_array((Vec::<usize>::new(), vec![IN_RATE as i64])).map_err(e)?,
            ])
            .map_err(e)?;
        let p = out["output"].try_extract_tensor::<f32>().map_err(e)?.1[0];
        *state = out["stateN"].try_extract_tensor::<f32>().map_err(e)?.1.to_vec();
        Ok(p)
    }

    fn transcribe(&self, audio: &[f32]) -> Result<String, String> {
        let mut guard = self.asr.lock().unwrap();
        let a = &mut *guard;
        let (feats, frames, valid) = asr::features(audio, &a.bank, &a.fft);
        let out = a
            .encoder
            .run(ort::inputs![
                "audio_signal" => Tensor::from_array(([1usize, asr::MELS, frames], feats)).map_err(e)?,
                "length" => Tensor::from_array(([1usize], vec![valid as i64])).map_err(e)?,
            ])
            .map_err(e)?;
        let (shape, enc) = out["outputs"].try_extract_tensor::<f32>().map_err(e)?;
        let (dim, steps) = (shape[1] as usize, shape[2] as usize);
        let len = (out["encoded_lengths"].try_extract_tensor::<i64>().map_err(e)?.1[0].max(0) as usize).min(steps);
        let enc = enc.to_vec();
        drop(out);
        let mut d = Joint { session: &mut a.decoder, enc: &enc, dim, steps, state: (vec![0f32; 2 * 640], vec![0f32; 2 * 640]), pending: (vec![], vec![]) };
        let tokens = asr::tdt_decode(len, a.vocab.len(), a.blank, &mut d)?;
        Ok(asr::text(&tokens, &a.vocab))
    }

    fn speak(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>, String> {
        let mut guard = self.tts.lock().unwrap();
        let t = &mut *guard;
        let style = t.voices.get(voice).ok_or_else(|| format!("Voice {voice} not found"))?;
        let phonemes = tts::keep_known(&t.espeak.phonemize(text.trim(), super::lang_of(voice))?);
        let mut audio = vec![];
        let batches = tts::split_phonemes(&phonemes);
        if batches.is_empty() {
            return Err("nothing to say".into());
        }
        for batch in batches {
            let batch: String = batch.chars().take(tts::MAX_PHONEMES).collect();
            let tokens = tts::tokenize(&batch);
            let row = style.get(tokens.len() * tts::STYLE..(tokens.len() + 1) * tts::STYLE).ok_or("too many phonemes")?.to_vec();
            let ids: Vec<i64> = std::iter::once(0).chain(tokens).chain(std::iter::once(0)).collect();
            let n = ids.len();
            let out = t
                .session
                .run(ort::inputs![
                    "tokens" => Tensor::from_array(([1usize, n], ids)).map_err(e)?,
                    "style" => Tensor::from_array(([1usize, tts::STYLE], row)).map_err(e)?,
                    "speed" => Tensor::from_array(([1usize], vec![speed])).map_err(e)?,
                ])
                .map_err(e)?;
            let samples = out["audio"].try_extract_tensor::<f32>().map_err(e)?.1;
            audio.extend_from_slice(tts::trim(samples));
        }
        Ok(audio)
    }
}

/// Parakeet's decoder-joint with its prediction state.
struct Joint<'a> {
    session: &'a mut Session,
    enc: &'a [f32],
    dim: usize,
    steps: usize,
    state: (Vec<f32>, Vec<f32>),
    pending: (Vec<f32>, Vec<f32>),
}

impl asr::Decoder for Joint<'_> {
    fn step(&mut self, t: usize, last: usize) -> Result<Vec<f32>, String> {
        let frame: Vec<f32> = (0..self.dim).map(|d| self.enc[d * self.steps + t]).collect();
        let out = self
            .session
            .run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(([1usize, self.dim, 1], frame)).map_err(e)?,
                "targets" => Tensor::from_array(([1usize, 1], vec![last as i32])).map_err(e)?,
                "target_length" => Tensor::from_array(([1usize], vec![1i32])).map_err(e)?,
                "input_states_1" => Tensor::from_array(([2usize, 1, 640], self.state.0.clone())).map_err(e)?,
                "input_states_2" => Tensor::from_array(([2usize, 1, 640], self.state.1.clone())).map_err(e)?,
            ])
            .map_err(e)?;
        self.pending = (out["output_states_1"].try_extract_tensor::<f32>().map_err(e)?.1.to_vec(), out["output_states_2"].try_extract_tensor::<f32>().map_err(e)?.1.to_vec());
        Ok(out["outputs"].try_extract_tensor::<f32>().map_err(e)?.1.to_vec())
    }

    fn commit(&mut self) {
        self.state = std::mem::take(&mut self.pending);
    }
}
