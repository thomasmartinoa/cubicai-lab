//! Speed test with llama-bench: prompt processing and generation speed for the current settings.

use super::Task;
use crate::store::{BenchResult, RunPreset};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub fn run(bin_dir: &Path, model_key: &str, model: &Path, p: &RunPreset, gpu_layers: u32, cpu_moe: u32, build: &str, task: &Arc<Task>) -> Result<BenchResult, String> {
    let exe = super::release::bench_exe(bin_dir);
    if !exe.is_file() {
        return Err("llama-bench isn't in this llama.cpp build".into());
    }
    let fa = p.flash_attn.arg();
    let mut args: Vec<String> = vec![
        "-m".into(),
        model.to_string_lossy().into(),
        "-ngl".into(),
        gpu_layers.to_string(),
        "-fa".into(),
        fa.into(),
        "-ctk".into(),
        p.kv_type.clone(),
        "-ctv".into(),
        p.kv_type.clone(),
        "-ub".into(),
        p.ubatch.to_string(),
        "-p".into(),
        "512".into(),
        "-n".into(),
        "128".into(),
        "-r".into(),
        "2".into(),
        "-o".into(),
        "json".into(),
    ];
    if cpu_moe > 0 {
        args.extend(["-ncmoe".into(), cpu_moe.to_string()]);
    }
    if let Some(t) = p.threads {
        args.extend(["-t".into(), t.to_string()]);
    }
    task.set_status("Measuring prompt and generation speed…".into());
    let mut cmd = Command::new(&exe);
    cmd.args(&args).current_dir(bin_dir).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    super::server::add_lib_path(&mut cmd, bin_dir);
    super::server::die_with_parent(&mut cmd);
    let child = cmd.spawn().map_err(|e| format!("Couldn't start llama-bench: {e}"))?;
    *task.child_pid.lock().unwrap() = Some(child.id());
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    *task.child_pid.lock().unwrap() = None;
    if task.cancel.load(Ordering::Relaxed) {
        return Err("Stopped".into());
    }
    let (pp, tg) = parse_output(&String::from_utf8_lossy(&out.stdout), &String::from_utf8_lossy(&out.stderr))?;
    let mut settings = format!("ngl {gpu_layers}, fa {fa}, kv {}", p.kv_type);
    if cpu_moe > 0 {
        settings += &format!(", experts in RAM ×{cpu_moe}");
    }
    Ok(BenchResult { model: model_key.to_string(), at: crate::store::now_ms(), settings, prompt_tps: pp, gen_tps: tg, build: build.to_string() })
}

/// Prompt and generation speed (tokens/s) from `llama-bench -o json`.
pub fn parse_output(stdout: &str, stderr: &str) -> Result<(f64, f64), String> {
    let start = stdout.find('[').ok_or_else(|| {
        let tail: Vec<&str> = stderr.lines().rev().filter(|l| !l.trim().is_empty()).take(4).collect();
        format!("llama-bench failed: {}", tail.into_iter().rev().collect::<Vec<_>>().join(" / "))
    })?;
    let v: serde_json::Value = serde_json::from_str(&stdout[start..]).map_err(|e| format!("Unreadable result: {e}"))?;
    let (mut pp, mut tg) = (0.0, 0.0);
    for r in v.as_array().into_iter().flatten() {
        let ts = r["avg_ts"].as_f64().unwrap_or(0.0);
        if r["n_gen"].as_u64().unwrap_or(0) > 0 {
            tg = ts;
        } else if r["n_prompt"].as_u64().unwrap_or(0) > 0 {
            pp = ts;
        }
    }
    if pp == 0.0 && tg == 0.0 {
        return Err("llama-bench reported no results".into());
    }
    Ok((pp, tg))
}
