//! The Engine Room's machinery: llama.cpp builds, the model library, the running server, GPU
//! monitoring, benchmarks and downloads. The UI lives in `ui/engine.rs`.

pub mod arena;
pub mod bench;
pub mod datakit;
pub mod dataset;
pub mod fit;
pub mod gpu;
pub mod hf;
pub mod library;
pub mod lora;
pub mod release;
pub mod server;
pub mod steer;
pub mod train;
pub mod xray;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// A background job with progress the UI can show and stop.
#[derive(Default)]
pub struct Task {
    /// Parts per thousand.
    pub progress: AtomicU32,
    pub cancel: AtomicBool,
    pub done: AtomicBool,
    status: Mutex<String>,
    pub error: Mutex<Option<String>>,
    /// Sent as a bearer token with downloads (Hugging Face).
    pub auth: Mutex<Option<String>>,
    pub child_pid: Mutex<Option<u32>>,
}

impl Task {
    pub fn new(status: &str) -> Arc<Task> {
        let t = Task::default();
        *t.status.lock().unwrap() = status.to_string();
        Arc::new(t)
    }

    pub fn set_status(&self, s: String) {
        *self.status.lock().unwrap() = s;
    }

    pub fn status(&self) -> String {
        self.status.lock().unwrap().clone()
    }

    pub fn fraction(&self) -> f32 {
        self.progress.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn finish(&self, r: Result<(), String>) {
        if let Err(e) = r {
            *self.error.lock().unwrap() = Some(e);
        }
        self.done.store(true, Ordering::Relaxed);
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    pub fn stop(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(pid) = *self.child_pid.lock().unwrap() {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            #[cfg(not(unix))]
            let _ = pid;
        }
    }
}

/// Runs `f` on a thread, recording its outcome on the task and waking the UI when it ends.
pub fn spawn(task: &Arc<Task>, ctx: &egui::Context, f: impl FnOnce(&Arc<Task>) -> Result<(), String> + Send + 'static) {
    let t = task.clone();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let r = f(&t);
        t.finish(r);
        ctx.request_repaint();
    });
}
