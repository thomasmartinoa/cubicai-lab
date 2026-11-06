//! The models on this computer: GGUF files in the app's folder and any folders the user adds, plus
//! the ones Ollama already downloaded (read in place, nothing copied). Headers are parsed in the
//! background so the list shows at once.

use cubic_gguf::layout::{self, KvLayer};
use cubic_gguf::{Anatomy, Gguf};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Folder,
    Ollama(String),
}

#[derive(Clone, Debug)]
pub struct Model {
    /// Stable id: the first file's path.
    pub key: String,
    pub name: String,
    /// All parts in order (one unless the model is split).
    pub parts: Vec<PathBuf>,
    pub size: u64,
    pub mmproj: Option<PathBuf>,
    pub source: Source,
}

impl Model {
    pub fn path(&self) -> &Path {
        &self.parts[0]
    }
}

/// What the header says, parsed once per model.
pub struct Info {
    pub gguf: Gguf,
    pub anatomy: Anatomy,
    pub kv: Vec<KvLayer>,
    pub active_params: u64,
    pub ollama_packaging: bool,
}

impl Info {
    pub fn quant_label(&self) -> String {
        file_type_name(self.gguf.get_u64("general.file_type"))
            .map(str::to_string)
            .unwrap_or_else(|| self.anatomy.type_mix.first().map(|t| t.0.name()).unwrap_or_default())
    }
}

/// Parses the header of every part (split models keep tensors in each).
pub fn read_info(m: &Model) -> Result<Info, String> {
    let mut g: Option<Gguf> = None;
    for (i, p) in m.parts.iter().enumerate() {
        let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let map = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?;
        let part = Gguf::parse(&map, Some(map.len() as u64)).map_err(|e| e.to_string())?;
        match &mut g {
            None => g = Some(part),
            Some(g) => g.merge_split(part, i),
        }
    }
    let gguf = g.ok_or("no file")?;
    let anatomy = Anatomy::build(&gguf);
    let kv = layout::kv_layout(&gguf, &anatomy);
    let active_params = anatomy.active_params(&gguf);
    let ollama_packaging = layout::bundles_towers(&gguf);
    Ok(Info { gguf, anatomy, kv, active_params, ollama_packaging })
}

fn file_type_name(ft: Option<u64>) -> Option<&'static str> {
    Some(match ft? {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

pub struct Library {
    pub models: Vec<Model>,
    pub info: Arc<Mutex<HashMap<String, Result<Arc<Info>, String>>>>,
    pub ollama_dir: Option<PathBuf>,
}

impl Library {
    pub fn new() -> Self {
        Library { models: vec![], info: Arc::new(Mutex::new(HashMap::new())), ollama_dir: None }
    }

    pub fn scan(&mut self, dirs: &[PathBuf], ollama: Option<PathBuf>, ctx: &egui::Context) {
        let mut models = vec![];
        for d in dirs {
            scan_folder(d, &mut models, 0);
        }
        self.ollama_dir = ollama.or_else(find_ollama_dir);
        if let Some(o) = &self.ollama_dir {
            scan_ollama(o, &mut models);
        }
        models.sort_by_key(|a| a.name.to_lowercase());
        // Ollama can list one file under several names; keep the first.
        let mut seen = std::collections::HashSet::new();
        models.retain(|m| seen.insert(m.key.clone()));
        self.models = models;
        // Parse headers in the background, a few at a time.
        let todo: Vec<Model> = {
            let have = self.info.lock().unwrap();
            self.models.iter().filter(|m| !have.contains_key(&m.key)).cloned().collect()
        };
        let info = self.info.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for m in todo {
                let r = read_info(&m).map(Arc::new);
                info.lock().unwrap().insert(m.key.clone(), r);
                ctx.request_repaint();
            }
        });
    }

    pub fn get(&self, key: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.key == key)
    }

    pub fn info(&self, key: &str) -> Option<Arc<Info>> {
        self.info.lock().unwrap().get(key).and_then(|r| r.as_ref().ok().cloned())
    }

    pub fn info_error(&self, key: &str) -> Option<String> {
        self.info.lock().unwrap().get(key).and_then(|r| r.as_ref().err().cloned())
    }
}

pub(crate) fn scan_folder(dir: &Path, out: &mut Vec<Model>, depth: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<PathBuf> = vec![];
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() && depth < 3 {
            scan_folder(&p, out, depth + 1);
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) {
            files.push(p);
        }
    }
    files.sort();
    let is_mmproj = |p: &Path| p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().contains("mmproj"));
    let mmproj = files.iter().find(|p| is_mmproj(p)).cloned();
    let mut groups: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for f in files.iter().filter(|p| !is_mmproj(p)) {
        let stem = f.file_stem().unwrap_or_default().to_string_lossy().to_string();
        // name-00001-of-00003 → one model.
        let base = match split_base(&stem) {
            Some(b) => b,
            None => stem.clone(),
        };
        groups.entry(base).or_default().push(f.clone());
    }
    for (name, mut parts) in groups {
        parts.sort();
        let size = parts.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();
        out.push(Model {
            key: parts[0].to_string_lossy().to_string(),
            name,
            parts,
            size,
            mmproj: mmproj.clone(),
            source: Source::Folder,
        });
    }
}

pub fn split_base(stem: &str) -> Option<String> {
    let (left, total) = stem.rsplit_once("-of-")?;
    let (base, n) = left.rsplit_once('-')?;
    (n.len() == 5 && total.len() == 5 && n.chars().all(|c| c.is_ascii_digit())).then(|| base.to_string())
}

/// Ollama's model store: OLLAMA_MODELS, the systemd service's setting, or the usual places.
pub fn find_ollama_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = vec![];
    if let Ok(v) = std::env::var("OLLAMA_MODELS") {
        candidates.push(v.into());
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(out) = std::process::Command::new("systemctl").args(["show", "ollama", "-p", "Environment", "--value"]).output() {
            let s = String::from_utf8_lossy(&out.stdout);
            // The last setting wins, as in systemd.
            if let Some(v) = s.split_whitespace().filter_map(|kv| kv.strip_prefix("OLLAMA_MODELS=")).next_back() {
                candidates.insert(0, v.trim_matches('"').into());
            }
        }
        candidates.push("/usr/share/ollama/.ollama/models".into());
        candidates.push("/var/lib/ollama".into());
    }
    if let Some(b) = directories::BaseDirs::new() {
        candidates.push(b.home_dir().join(".ollama/models"));
    }
    candidates.into_iter().find(|d| d.join("manifests").is_dir())
}

pub(crate) fn scan_ollama(root: &Path, out: &mut Vec<Model>) {
    let manifests = root.join("manifests");
    let mut stack = vec![manifests.clone()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else { continue };
            let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
            let layer = |kind: &str| {
                v["layers"].as_array().into_iter().flatten().find(|l| l["mediaType"].as_str() == Some(kind)).and_then(|l| {
                    let digest = l["digest"].as_str()?.replace(':', "-");
                    let path = root.join("blobs").join(digest);
                    path.is_file().then_some((path, l["size"].as_u64().unwrap_or(0)))
                })
            };
            let Some((model, size)) = layer("application/vnd.ollama.image.model") else { continue };
            let rel = p.strip_prefix(&manifests).unwrap_or(&p);
            let comps: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
            if comps.len() < 3 {
                continue;
            }
            let tag = &comps[comps.len() - 1];
            // registry.ollama.ai/library/gemma4/e4b → gemma4:e4b; hf.co/user/repo/Q4 → hf.co/user/repo:Q4
            let path_part = if comps[0] == "registry.ollama.ai" && comps.get(1).map(String::as_str) == Some("library") {
                comps[2..comps.len() - 1].join("/")
            } else if comps[0] == "registry.ollama.ai" {
                comps[1..comps.len() - 1].join("/")
            } else {
                comps[..comps.len() - 1].join("/")
            };
            // Ollama keeps a few internal entries with hash-like tags; skip them.
            if tag.len() == 64 && tag.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            let name = format!("{path_part}:{tag}");
            out.push(Model {
                key: model.to_string_lossy().to_string(),
                name: name.clone(),
                parts: vec![model],
                size,
                mmproj: layer("application/vnd.ollama.image.projector").map(|x| x.0),
                source: Source::Ollama(name),
            });
        }
    }
}
