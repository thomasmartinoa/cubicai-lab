//! Getting llama.cpp: picks the official prebuilt release for this OS and GPU, downloads and
//! unpacks it into the app's data folder.

use super::gpu::{GpuInfo, Vendor};
use super::Task;
use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    Cuda13,
    Cuda12,
    Vulkan,
    Rocm,
    Metal,
    Cpu,
}

impl Variant {
    pub const ALL: [Variant; 6] = [Variant::Cuda13, Variant::Cuda12, Variant::Vulkan, Variant::Rocm, Variant::Metal, Variant::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            Variant::Cuda13 => "CUDA 13",
            Variant::Cuda12 => "CUDA 12",
            Variant::Vulkan => "Vulkan",
            Variant::Rocm => "ROCm",
            Variant::Metal => "Metal",
            Variant::Cpu => "CPU only",
        }
    }

    pub fn explain(self) -> &'static str {
        match self {
            Variant::Cuda13 => "NVIDIA, newest drivers (580+). Fastest on NVIDIA.",
            Variant::Cuda12 => "NVIDIA with older drivers.",
            Variant::Vulkan => "Any GPU (NVIDIA, AMD, Intel). A little slower than CUDA, very compatible.",
            Variant::Rocm => "AMD with ROCm installed.",
            Variant::Metal => "Apple Silicon Macs.",
            Variant::Cpu => "No GPU: slow, but works everywhere.",
        }
    }

    /// The part of the asset name that marks this build on this OS.
    fn asset_key(self) -> Option<&'static str> {
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        Some(match (os, self) {
            ("linux", Variant::Cuda13) => "bin-ubuntu-cuda-13",
            ("linux", Variant::Cuda12) => "bin-ubuntu-cuda-12",
            ("linux", Variant::Vulkan) if arch == "aarch64" => "bin-ubuntu-vulkan-arm64",
            ("linux", Variant::Vulkan) => "bin-ubuntu-vulkan-x64",
            ("linux", Variant::Rocm) => "bin-ubuntu-rocm",
            ("linux", Variant::Cpu) if arch == "aarch64" => "bin-ubuntu-arm64",
            ("linux", Variant::Cpu) => "bin-ubuntu-x64",
            ("windows", Variant::Cuda13) => "bin-win-cuda-13",
            ("windows", Variant::Cuda12) => "bin-win-cuda-12",
            ("windows", Variant::Vulkan) => "bin-win-vulkan-x64",
            ("windows", Variant::Rocm) => "bin-win-rocm",
            ("windows", Variant::Cpu) => "bin-win-cpu-x64",
            ("macos", Variant::Metal) if arch == "aarch64" => "bin-macos-arm64",
            ("macos", Variant::Cpu) => "bin-macos-x64",
            _ => return None,
        })
    }

    pub fn available(self) -> bool {
        self.asset_key().is_some()
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Variant::Cuda12 | Variant::Cuda13)
    }

    pub fn from_label(s: &str) -> Option<Variant> {
        Variant::ALL.into_iter().find(|v| v.label() == s)
    }
}

/// How release archives name this CPU.
fn arch_tag() -> &'static str {
    if cfg!(target_arch = "aarch64") { "arm64" } else { "x64" }
}

pub fn recommended(gpu: &GpuInfo) -> Variant {
    match gpu.vendor {
        Vendor::Nvidia if gpu.driver_major.unwrap_or(0) >= 580 => Variant::Cuda13,
        Vendor::Nvidia => Variant::Cuda12,
        Vendor::Apple => Variant::Metal,
        Vendor::Amd | Vendor::Other => Variant::Vulkan,
        Vendor::None if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") => Variant::Metal,
        Vendor::None => Variant::Cpu,
    }
}

/// Whether the CUDA runtime libraries (cuBLAS) are already on the system, so the extra download
/// can be skipped.
pub fn system_has_cuda_runtime(v: Variant) -> bool {
    let major = if v == Variant::Cuda13 { "13" } else { "12" };
    #[cfg(target_os = "linux")]
    {
        if let Ok(out) = std::process::Command::new("ldconfig").arg("-p").output()
            && String::from_utf8_lossy(&out.stdout).contains(&format!("libcublas.so.{major}")) {
                return true;
            }
        for dir in ["/opt/cuda/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            if Path::new(dir).join(format!("libcublas.so.{major}")).exists() {
                return true;
            }
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = major;
        false
    }
}

#[derive(Clone, Debug)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct Release {
    pub tag: String,
    pub assets: Vec<Asset>,
}

impl Release {
    pub fn build_for(&self, v: Variant) -> Option<&Asset> {
        let key = v.asset_key()?;
        self.assets.iter().find(|a| a.name.starts_with("llama-") && a.name.contains(key) && a.name.contains(arch_tag()))
    }

    pub fn cudart_for(&self, v: Variant) -> Option<&Asset> {
        if !v.is_cuda() {
            return None;
        }
        let key = v.asset_key()?.trim_start_matches("bin-");
        self.assets.iter().find(|a| a.name.starts_with("cudart-") && a.name.contains(key) && a.name.contains(arch_tag()))
    }
}

/// The newest build. llama.cpp tags every build `bNNNNN` (marked pre-release); the newest
/// of those is what everyone runs.
pub fn latest() -> Result<Release, String> {
    let agent = crate::llm::agent(10);
    let resp = agent
        .get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5")
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("Couldn't reach GitHub: {e}"))?;
    if resp.status() != 200 {
        return Err(format!("GitHub answered {}", resp.status()));
    }
    let v: Value = resp.into_body().read_json().map_err(|e| e.to_string())?;
    for rel in v.as_array().into_iter().flatten() {
        let tag = rel["tag_name"].as_str().unwrap_or_default();
        let assets: Vec<Asset> = rel["assets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|a| {
                Some(Asset {
                    name: a["name"].as_str()?.to_string(),
                    url: a["browser_download_url"].as_str()?.to_string(),
                    size: a["size"].as_u64().unwrap_or(0),
                })
            })
            .collect();
        if tag.starts_with('b') && assets.iter().any(|a| a.name.contains("-bin-")) {
            return Ok(Release { tag: tag.to_string(), assets });
        }
    }
    Err("No llama.cpp build found in the latest releases".into())
}

/// Downloads and unpacks the build (and the CUDA runtime if asked) into `root/<tag>-<variant>`.
/// Returns the folder holding `llama-server`.
pub fn install(rel: &Release, v: Variant, with_cudart: bool, root: &Path, task: &Arc<Task>) -> Result<PathBuf, String> {
    let asset = rel.build_for(v).ok_or("This build isn't published for this computer")?;
    let mut parts = vec![asset.clone()];
    if with_cudart
        && let Some(c) = rel.cudart_for(v) {
            parts.push(c.clone());
        }
    let total: u64 = parts.iter().map(|a| a.size).sum();
    let dest = root.join(format!("{}-{}", rel.tag, v.label().to_lowercase().replace(' ', "-")));
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
    let mut done = 0u64;
    for a in &parts {
        task.set_status(format!("Downloading {}", a.name));
        let file = root.join(&a.name);
        download(&a.url, &file, |n| {
            task.progress.store(((done + n) * 1000 / total.max(1)) as u32, Ordering::Relaxed);
        }, task)?;
        let got = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
        if a.size > 0 && got != a.size {
            let _ = std::fs::remove_file(&file);
            return Err(format!("{} arrived incomplete ({got} of {} bytes); try again", a.name, a.size));
        }
        done += a.size;
        task.set_status(format!("Unpacking {}", a.name));
        unpack(&file, &dest)?;
        let _ = std::fs::remove_file(&file);
    }
    let bin = find_server(&dest).ok_or("The download didn't contain llama-server")?;
    // cudart archives unpack flat; their libraries belong next to the binaries.
    if bin != dest {
        for e in std::fs::read_dir(&dest).map_err(|e| e.to_string())?.flatten() {
            let p = e.path();
            if p.is_file() {
                let _ = std::fs::rename(&p, bin.join(e.file_name()));
            }
        }
    }
    Ok(bin)
}

pub fn server_exe(dir: &Path) -> PathBuf {
    dir.join(if cfg!(windows) { "llama-server.exe" } else { "llama-server" })
}

pub fn bench_exe(dir: &Path) -> PathBuf {
    dir.join(if cfg!(windows) { "llama-bench.exe" } else { "llama-bench" })
}

pub fn find_server(root: &Path) -> Option<PathBuf> {
    if server_exe(root).is_file() {
        return Some(root.to_path_buf());
    }
    for e in std::fs::read_dir(root).ok()?.flatten() {
        if e.path().is_dir()
            && let Some(p) = find_server(&e.path()) {
                return Some(p);
            }
    }
    None
}

/// Streams `url` to `file` (via a .part file, resuming one left by an earlier try).
pub fn download(url: &str, file: &Path, progress: impl Fn(u64), task: &Task) -> Result<(), String> {
    let part = file.with_extension(format!("{}part", file.extension().map(|e| format!("{}.", e.to_string_lossy())).unwrap_or_default()));
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(std::time::Duration::from_secs(15)))
        .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
        .http_status_as_error(false)
        .user_agent("cubicai-lab")
        .build()
        .into();
    let mut req = agent.get(url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    if let Some(token) = task.auth.lock().unwrap().clone() {
        req = req.header("Authorization", format!("Bearer {token}"));
    }
    let resp = req.call().map_err(|e| format!("Download failed: {e}"))?;
    let status = resp.status().as_u16();
    let resumed = status == 206;
    if status != 200 && !resumed {
        return Err(format!("Download failed: the server answered {status}"));
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(&part)
        .map_err(|e| e.to_string())?;
    let mut n = if resumed { have } else { 0 };
    let mut r = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if task.cancel.load(Ordering::Relaxed) {
            return Err("Stopped".into());
        }
        let k = r.read(&mut buf).map_err(|e| format!("Download interrupted: {e}"))?;
        if k == 0 {
            break;
        }
        out.write_all(&buf[..k]).map_err(|e| e.to_string())?;
        n += k as u64;
        progress(n);
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    std::fs::rename(&part, file).map_err(|e| e.to_string())
}

pub(crate) fn unpack(file: &Path, dest: &Path) -> Result<(), String> {
    let name = file.file_name().unwrap_or_default().to_string_lossy().to_string();
    let f = std::fs::File::open(file).map_err(|e| e.to_string())?;
    if name.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
        z.extract(dest).map_err(|e| e.to_string())
    } else {
        let gz = flate2::read::GzDecoder::new(f);
        let mut t = tar::Archive::new(gz);
        t.set_preserve_permissions(true);
        t.unpack(dest).map_err(|e| e.to_string())
    }
}
