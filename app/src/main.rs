#![allow(clippy::type_complexity, clippy::too_many_arguments)]
//! CubicAI Lab: the desktop side of CubicAI. Chat with local and cloud models, run llama.cpp on
//! your own GPU, and see inside the models you run.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod engine;
mod llm;
mod personas;
mod phone;
mod providers;
mod secrets;
mod store;
mod theme;
mod ui;
mod widgets;
#[cfg(test)]
mod shots;
#[cfg(test)]
mod tests;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("CubicAI Lab")
            .with_app_id("cubicai-lab")
            .with_inner_size([1380.0, 880.0])
            .with_min_inner_size([960.0, 620.0])
            .with_icon(icon()),
        ..Default::default()
    };
    eframe::run_native("CubicAI Lab", options, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}

/// The cube mark drawn into a 128 px window icon.
fn icon() -> egui::IconData {
    let n = 128usize;
    let mut rgba = vec![0u8; n * n * 4];
    let [top, left, right] = theme::DARK.cube;
    let r = n as f32 * 0.46;
    let (cx, cy) = (n as f32 / 2.0, n as f32 / 2.0);
    let dx = r * 3f32.sqrt() / 2.0;
    let quad = |px: f32, py: f32, pts: [(f32, f32); 4]| {
        // Inside a convex quad: same side of every edge.
        let mut sign = 0.0;
        for i in 0..4 {
            let (ax, ay) = pts[i];
            let (bx, by) = pts[(i + 1) % 4];
            let c = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
            if c.abs() < 1e-6 {
                continue;
            }
            if sign == 0.0 {
                sign = c.signum();
            } else if c.signum() != sign {
                return false;
            }
        }
        true
    };
    let t = (cx, cy - r);
    let ur = (cx + dx, cy - r / 2.0);
    let dr = (cx + dx, cy + r / 2.0);
    let b = (cx, cy + r);
    let dl = (cx - dx, cy + r / 2.0);
    let ul = (cx - dx, cy - r / 2.0);
    let c = (cx, cy);
    for y in 0..n {
        for x in 0..n {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let color = if quad(px, py, [t, ur, c, ul]) {
                Some(top)
            } else if quad(px, py, [ul, c, b, dl]) {
                Some(left)
            } else if quad(px, py, [c, ur, dr, b]) {
                Some(right)
            } else {
                None
            };
            if let Some(col) = color {
                let i = (y * n + x) * 4;
                rgba[i..i + 4].copy_from_slice(&[col.r(), col.g(), col.b(), 255]);
            }
        }
    }
    egui::IconData { rgba, width: n as u32, height: n as u32 }
}
