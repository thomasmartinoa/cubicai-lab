//! What graphics card there is, how much memory it has, and who is using it.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum Vendor {
    Nvidia,
    Amd,
    Apple,
    Other,
    None,
}

#[derive(Clone, Debug)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub used: u64,
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub vendor: Vendor,
    pub name: String,
    pub total: u64,
    pub used: u64,
    /// NVIDIA driver major version (decides the CUDA build).
    pub driver_major: Option<u32>,
    pub processes: Vec<GpuProcess>,
}

impl GpuInfo {
    fn none() -> Self {
        GpuInfo { vendor: Vendor::None, name: "No GPU found".into(), total: 0, used: 0, driver_major: None, processes: vec![] }
    }
}

pub struct GpuMonitor {
    nvml: Option<nvml_wrapper::Nvml>,
    last: Option<Instant>,
    pub info: GpuInfo,
}

impl GpuMonitor {
    pub fn new() -> Self {
        let nvml = nvml_wrapper::Nvml::init().ok();
        let mut m = GpuMonitor { nvml, last: None, info: GpuInfo::none() };
        m.refresh();
        m
    }

    /// Refreshes at most every 2 s.
    pub fn tick(&mut self) {
        if self.last.is_none_or(|t| t.elapsed() > Duration::from_secs(2)) {
            self.refresh();
        }
    }

    pub fn refresh(&mut self) {
        self.last = Some(Instant::now());
        if let Some(info) = self.nvml.as_ref().and_then(nvidia) {
            self.info = info;
            return;
        }
        if let Some(info) = amd_sysfs() {
            self.info = info;
            return;
        }
        if let Some(info) = apple() {
            self.info = info;
            return;
        }
        self.info = GpuInfo::none();
    }
}

fn nvidia(nvml: &nvml_wrapper::Nvml) -> Option<GpuInfo> {
    let dev = nvml.device_by_index(0).ok()?;
    let mem = dev.memory_info().ok()?;
    let driver_major = nvml.sys_driver_version().ok().and_then(|v| v.split('.').next()?.parse().ok());
    let mut processes = vec![];
    let mut seen = std::collections::HashSet::new();
    for p in dev.running_compute_processes().unwrap_or_default().into_iter().chain(dev.running_graphics_processes().unwrap_or_default()) {
        if !seen.insert(p.pid) {
            continue;
        }
        let used = match p.used_gpu_memory {
            nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => b,
            _ => 0,
        };
        processes.push(GpuProcess { pid: p.pid, name: process_name(p.pid), used });
    }
    processes.sort_by_key(|p| std::cmp::Reverse(p.used));
    Some(GpuInfo {
        vendor: Vendor::Nvidia,
        name: dev.name().unwrap_or_else(|_| "NVIDIA GPU".into()),
        total: mem.total,
        used: mem.used,
        driver_major,
        processes,
    })
}

fn process_name(pid: u32) -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string(format!("/proc/{pid}/comm")) {
            return s.trim().to_string();
        }
    }
    format!("pid {pid}")
}

fn amd_sysfs() -> Option<GpuInfo> {
    #[cfg(target_os = "linux")]
    {
        for e in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
            let dev = e.path().join("device");
            let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
            if vendor.trim() != "0x1002" {
                continue;
            }
            let num = |f: &str| std::fs::read_to_string(dev.join(f)).ok()?.trim().parse::<u64>().ok();
            if let Some(total) = num("mem_info_vram_total") {
                return Some(GpuInfo {
                    vendor: Vendor::Amd,
                    name: "AMD GPU".into(),
                    total,
                    used: num("mem_info_vram_used").unwrap_or(0),
                    driver_major: None,
                    processes: vec![],
                });
            }
        }
    }
    None
}

fn apple() -> Option<GpuInfo> {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok()?;
        let total: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        // Metal lets a process use roughly three quarters of unified memory.
        return Some(GpuInfo {
            vendor: Vendor::Apple,
            name: "Apple GPU (unified memory)".into(),
            total: total / 4 * 3,
            used: 0,
            driver_major: None,
            processes: vec![],
        });
    }
    #[allow(unreachable_code)]
    None
}

/// Physical RAM in bytes.
pub fn system_ram() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for l in s.lines() {
                if let Some(v) = l.strip_prefix("MemTotal:") {
                    return v.trim().trim_end_matches("kB").trim().parse::<u64>().unwrap_or(0) * 1024;
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            return String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(0);
        }
    }
    0
}

pub fn cpu_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}
