//! Text → IPA phonemes for Kokoro, exactly as kokoro-onnx gets them: phonemizer's espeak backend
//! (`preserve_punctuation=True, with_stress=True`), with libespeak-ng loaded at run time.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::Mutex;

/// phonemizer's default punctuation marks.
pub const MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]";

/// A punctuation mark taken out of the text: (line, the mark with its spaces, B/E/I/A).
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub index: usize,
    pub mark: String,
    pub position: char,
}

fn marks_re() -> &'static regex::Regex {
    static RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| regex::Regex::new(&format!(r"(\s*[{}]+\s*)+", regex::escape(MARKS))).unwrap());
    &RE
}

/// phonemizer's `Punctuation.preserve` for one line: the text between the marks, and the marks.
pub fn preserve(line: &str) -> (Vec<String>, Vec<Mark>) {
    let matches: Vec<regex::Match> = marks_re().find_iter(line).collect();
    if matches.is_empty() {
        return (if line.is_empty() { vec![] } else { vec![line.to_string()] }, vec![]);
    }
    if matches.len() == 1 && matches[0].as_str() == line {
        return (vec![], vec![Mark { index: 0, mark: line.to_string(), position: 'A' }]);
    }
    let last = matches.len() - 1;
    let marks: Vec<Mark> = matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let position = if i == 0 && line.starts_with(m.as_str()) {
                'B'
            } else if i == last && line.ends_with(m.as_str()) {
                'E'
            } else {
                'I'
            };
            Mark { index: 0, mark: m.as_str().to_string(), position }
        })
        .collect();
    let mut chunks = vec![];
    let mut rest = line.to_string();
    for m in &marks {
        // Python splits at the mark's first occurrence in what's left.
        let (prefix, suffix) = rest.split_once(m.mark.as_str()).map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((rest.clone(), String::new()));
        chunks.push(prefix);
        rest = suffix;
    }
    chunks.push(rest);
    (chunks.into_iter().filter(|c| !c.is_empty()).collect(), marks)
}

/// phonemizer's `Punctuation.restore` (word separator " ", strip off).
pub fn restore(text: Vec<String>, marks: Vec<Mark>) -> Vec<String> {
    let mut text: std::collections::VecDeque<String> = text.into();
    let mut marks: std::collections::VecDeque<Mark> = marks.into();
    let mut out = vec![];
    let mut pos = 0;
    while !text.is_empty() || !marks.is_empty() {
        if marks.is_empty() {
            for mut line in text.drain(..) {
                if !line.ends_with(' ') {
                    line.push(' ');
                }
                out.push(line);
            }
        } else if text.is_empty() {
            out.push(marks.drain(..).map(|m| m.mark).collect());
        } else if marks[0].index == pos {
            let m = marks.pop_front().unwrap();
            if text[0].ends_with(' ') {
                text[0].pop();
            }
            let tail = if m.mark.ends_with(' ') { "" } else { " " };
            match m.position {
                'B' => text[0] = format!("{}{}", m.mark, text[0]),
                'E' => {
                    let t = text.pop_front().unwrap();
                    out.push(format!("{t}{}{tail}", m.mark));
                    pos += 1;
                }
                'A' => {
                    out.push(format!("{}{tail}", m.mark));
                    pos += 1;
                }
                _ => {
                    if text.len() == 1 {
                        text[0].push_str(&m.mark);
                    } else {
                        let first = text.pop_front().unwrap();
                        text[0] = format!("{first}{}{}", m.mark, text[0]);
                    }
                }
            }
        } else {
            out.push(text.pop_front().unwrap());
            pos += 1;
        }
    }
    out
}

/// The espeak backend's clean-up of one line of espeak output ('_' between phones, spaces between
/// words), with stress kept and no phone separator: each word loses its '_' and gets a space.
pub fn postprocess(line: &str) -> String {
    let line = line.trim().replace('\n', " ").replace("  ", " ");
    let mut collapsed = String::with_capacity(line.len());
    for c in line.chars() {
        if !(c == '_' && collapsed.ends_with('_')) {
            collapsed.push(c);
        }
    }
    let line = collapsed.replace("_ ", " ");
    if line.is_empty() {
        return String::new();
    }
    line.split(' ').map(|w| w.trim().replace('_', "") + " ").collect()
}

/// The whole of `phonemizer.phonemize(text, lang, preserve_punctuation=True, with_stress=True)` for
/// one line, given espeak itself (`text → raw phonemes`).
pub fn phonemize_with(text: &str, espeak: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<String, String> {
    let line = text.trim_matches('\n');
    if line.trim().is_empty() {
        return Ok(String::new());
    }
    let (chunks, marks) = preserve(line);
    let phonemized = chunks.iter().map(|c| espeak(c).map(|p| postprocess(&p))).collect::<Result<Vec<_>, _>>()?;
    Ok(restore(phonemized, marks).join("\n"))
}

// --- libespeak-ng --------------------------------------------------------------------------------

#[repr(C)]
struct EspeakVoice {
    name: *const c_char,
    languages: *const c_char,
    identifier: *const c_char,
    gender: u8,
    age: u8,
    variant: u8,
    xx1: u8,
    score: c_int,
    spare: *mut c_void,
}

type Initialize = unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int;
type ListVoices = unsafe extern "C" fn(*mut EspeakVoice) -> *const *const EspeakVoice;
type SetVoiceByName = unsafe extern "C" fn(*const c_char) -> c_int;
type TextToPhonemes = unsafe extern "C" fn(*mut *const c_void, c_int, c_int) -> *const c_char;

struct Lib {
    list_voices: ListVoices,
    set_voice: SetVoiceByName,
    to_phonemes: TextToPhonemes,
    voice: Option<String>,
}

// espeak keeps global state: one caller at a time, behind the mutex.
unsafe impl Send for Lib {}

pub struct Espeak(Mutex<Lib>);

fn sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("espeak-ng has no {name}"));
    }
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

impl Espeak {
    pub fn load(lib: &str, data: &str) -> Result<Espeak, String> {
        let c = CString::new(lib).map_err(|e| e.to_string())?;
        let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("couldn't load {lib}"));
        }
        let init: Initialize = sym(handle, "espeak_Initialize")?;
        let data = CString::new(data).map_err(|e| e.to_string())?;
        // AUDIO_OUTPUT_SYNCHRONOUS, as phonemizer: nothing is played.
        if unsafe { init(0x02, 0, data.as_ptr(), 0) } <= 0 {
            return Err("espeak-ng didn't start".into());
        }
        Ok(Espeak(Mutex::new(Lib { list_voices: sym(handle, "espeak_ListVoices")?, set_voice: sym(handle, "espeak_SetVoiceByName")?, to_phonemes: sym(handle, "espeak_TextToPhonemes")?, voice: None })))
    }

    /// Kokoro's phonemes for `text` in `lang` (an espeak language code such as "en-us").
    pub fn phonemize(&self, text: &str, lang: &str) -> Result<String, String> {
        let mut lib = self.0.lock().unwrap();
        lib.select(lang)?;
        phonemize_with(text, &mut |chunk| lib.text_to_phonemes(chunk))
    }
}

impl Lib {
    /// phonemizer picks the first voice (by espeak's order) whose first language is `lang`.
    fn select(&mut self, lang: &str) -> Result<(), String> {
        if self.voice.as_deref() == Some(lang) {
            return Ok(());
        }
        let mut found = None;
        unsafe {
            let mut list = (self.list_voices)(std::ptr::null_mut());
            while !list.is_null() && !(*list).is_null() {
                let v = &**list;
                if !v.languages.is_null() && !v.identifier.is_null() {
                    let langs = CStr::from_ptr(v.languages).to_bytes();
                    if langs.len() > 1 && &langs[1..] == lang.as_bytes() {
                        found = Some(CStr::from_ptr(v.identifier).to_owned());
                        break;
                    }
                }
                list = list.add(1);
            }
        }
        let id = found.ok_or_else(|| format!("espeak-ng has no voice for {lang}"))?;
        if unsafe { (self.set_voice)(id.as_ptr()) } != 0 {
            return Err(format!("espeak-ng couldn't load the voice for {lang}"));
        }
        self.voice = Some(lang.to_string());
        Ok(())
    }

    /// espeak_TextToPhonemes clause by clause: IPA, '_' between phones.
    fn text_to_phonemes(&mut self, text: &str) -> Result<String, String> {
        let c = CString::new(text.replace('\0', " ")).map_err(|e| e.to_string())?;
        let mut ptr = c.as_ptr() as *const c_void;
        let mut parts = vec![];
        while !ptr.is_null() {
            let out = unsafe { (self.to_phonemes)(&mut ptr, 1, ((b'_' as c_int) << 8) | 0x02) };
            if !out.is_null() {
                let s = unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned();
                if !s.is_empty() {
                    parts.push(s);
                }
            }
        }
        Ok(parts.join(" "))
    }
}
