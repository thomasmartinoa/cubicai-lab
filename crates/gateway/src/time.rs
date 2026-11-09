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

/// Seconds since the epoch for a local date and time (libc's mktime: daylight saving is worked
/// out, and a day or minute past the end of the month or hour rolls over).
pub fn local_epoch(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> i64 {
    // SAFETY: mktime only reads and normalises the tm we hand it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    tm.tm_year = year - 1900;
    tm.tm_mon = month as i32 - 1;
    tm.tm_mday = day as i32;
    tm.tm_hour = hour as i32;
    tm.tm_min = minute as i32;
    tm.tm_isdst = -1;
    unsafe { libc::mktime(&mut tm) as i64 }
}

/// This PC's offset from UTC at that moment, in seconds (east positive), e.g. 19800 for India.
pub fn offset(secs: i64) -> i64 {
    // SAFETY: localtime_r only writes the tm we hand it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let t: libc::time_t = secs as libc::time_t;
    unsafe { libc::localtime_r(&t, &mut tm) };
    tm.tm_gmtoff
}

/// A local "2026-11-10T09:00" (or with seconds) as RFC 3339 with this PC's offset: "2026-11-10T09:00:00+05:30".
pub fn rfc3339_local(s: &str) -> Option<String> {
    let s = s.trim().replace(' ', "T");
    let (date, time) = s.split_once('T')?;
    let d: Vec<u32> = date.split('-').filter_map(|p| p.parse().ok()).collect();
    let t: Vec<u32> = time.split(':').filter_map(|p| p.parse().ok()).collect();
    if d.len() != 3 || t.len() < 2 || d[1] == 0 || d[1] > 12 || d[2] == 0 || d[2] > 31 || t[0] > 23 || t[1] > 59 {
        return None;
    }
    let epoch = local_epoch(d[0] as i32, d[1], d[2], t[0], t[1]);
    let off = offset(epoch);
    let sign = if off < 0 { '-' } else { '+' };
    Some(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{sign}{:02}:{:02}", d[0], d[1], d[2], t[0], t[1], t.get(2).copied().unwrap_or(0).min(59), off.abs() / 3600, off.abs() % 3600 / 60))
}
