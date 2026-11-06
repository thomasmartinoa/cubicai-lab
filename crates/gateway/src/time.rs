//! Local time without a date library: libc's localtime_r, formatted the ways the gateway needs.

use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds since the epoch, as the Python gateway's `time.time()` (the panel reads these).
pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Local {
    pub year: i32,
    /// 1..=12
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// 0 = Sunday
    pub weekday: u32,
}

pub fn local(secs: i64) -> Local {
    // SAFETY: localtime_r only writes the tm we hand it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let t: libc::time_t = secs as libc::time_t;
    unsafe { libc::localtime_r(&t, &mut tm) };
    Local {
        year: tm.tm_year + 1900,
        month: tm.tm_mon as u32 + 1,
        day: tm.tm_mday as u32,
        hour: tm.tm_hour as u32,
        minute: tm.tm_min as u32,
        second: tm.tm_sec as u32,
        weekday: tm.tm_wday as u32,
    }
}

pub fn now_local() -> Local {
    local(now() as i64)
}

const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

impl Local {
    /// 2026-11-06
    pub fn date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// Friday 06 November 2026 23:47 (Python's "%A %d %B %Y %H:%M").
    pub fn long(&self) -> String {
        format!("{} {:02} {} {} {:02}:{:02}", DAYS[self.weekday as usize % 7], self.day, MONTHS[(self.month as usize + 11) % 12], self.year, self.hour, self.minute)
    }
}

/// 2026-11-06 23:47:38, for log lines.
pub fn stamp() -> String {
    let t = now_local();
    format!("{} {:02}:{:02}:{:02}", t.date(), t.hour, t.minute, t.second)
}
