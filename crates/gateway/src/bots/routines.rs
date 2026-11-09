//! Routines: a Bot's tasks that run on a schedule (daily at a time on chosen days, every few hours,
//! or once), started by the gateway's scheduler whether or not the phone is around. The user makes
//! them on the phone, or a Bot proposes one with its schedule_routine tool; a proposed one runs only
//! once the user approves it (it may cost money every time, and runs with nobody watching).

use crate::time::{local, local_epoch};
use serde_json::{json, Value};

/// Shortest gap for "every N hours" (a Bot checking every minute would only burn the GPU or credits).
pub const MIN_EVERY_MINUTES: u32 = 30;
pub const MAX_ROUTINES: usize = 20;
const DAY_NAMES: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

#[derive(Clone, Debug, PartialEq)]
pub enum Schedule {
    /// At hour:minute local time on the days set in `days` (bit 0 = Sunday).
    Daily { hour: u32, minute: u32, days: u8 },
    Every { minutes: u32 },
    /// Once, at this time (seconds since the epoch).
    Once { at: i64 },
}

impl Schedule {
    /// From the phone's or the model's JSON:
    /// {"kind":"daily","time":"08:00","days":["mon",…]} | {"kind":"every","hours":6} | {"kind":"once","date":"2026-11-10","time":"09:00"} (or "at": epoch seconds).
    pub fn parse(v: &Value) -> Result<Schedule, String> {
        let time = || -> Result<(u32, u32), String> {
            let t = v["time"].as_str().unwrap_or("").trim();
            let (h, m) = t.split_once(':').ok_or("Give a time like 08:00.")?;
            let (h, m): (u32, u32) = (h.trim().parse().map_err(|_| "Give a time like 08:00.")?, m.trim().parse().map_err(|_| "Give a time like 08:00.")?);
            if h > 23 || m > 59 {
                return Err("Give a time like 08:00.".into());
            }
            Ok((h, m))
        };
        match v["kind"].as_str().unwrap_or("") {
            "daily" | "weekly" | "weekdays" => {
                let (hour, minute) = time()?;
                let mut days = 0u8;
                for d in v["days"].as_array().into_iter().flatten() {
                    let name = d.as_str().unwrap_or("").trim().to_lowercase();
                    let i = DAY_NAMES.iter().position(|n| name.starts_with(n)).ok_or_else(|| format!("Unknown day {name:?}."))?;
                    days |= 1 << i;
                }
                if days == 0 {
                    days = if v["kind"] == "weekdays" { 0b0111110 } else { 0b1111111 };
                }
                Ok(Schedule::Daily { hour, minute, days })
            }
            "every" | "hourly" => {
                let hours = v["hours"].as_f64().or_else(|| v["hours"].as_str().and_then(|s| s.trim().parse().ok())).unwrap_or(1.0);
                let minutes = (hours * 60.0).round() as i64;
                if minutes < MIN_EVERY_MINUTES as i64 || minutes > 24 * 60 * 7 {
                    return Err(format!("Repeat every {} minutes to 7 days.", MIN_EVERY_MINUTES));
                }
                Ok(Schedule::Every { minutes: minutes as u32 })
            }
            "once" => {
                if let Some(at) = v["at"].as_i64() {
                    return Ok(Schedule::Once { at });
                }
                let (hour, minute) = time()?;
                let date = v["date"].as_str().unwrap_or("").trim();
                let parts: Vec<u32> = date.split('-').filter_map(|p| p.parse().ok()).collect();
                if parts.len() != 3 || !(1..=12).contains(&parts[1]) || !(1..=31).contains(&parts[2]) {
                    return Err("Give the date like 2026-11-10.".into());
                }
                Ok(Schedule::Once { at: local_epoch(parts[0] as i32, parts[1], parts[2], hour, minute) })
            }
            other => Err(format!("Unknown schedule {other:?}: use daily, every or once.")),
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Schedule::Daily { hour, minute, days } => json!({
                "kind": "daily",
                "time": format!("{hour:02}:{minute:02}"),
                "days": (0..7).filter(|i| days & (1 << i) != 0).map(|i| DAY_NAMES[i]).collect::<Vec<_>>(),
            }),
            Schedule::Every { minutes } => json!({"kind": "every", "hours": *minutes as f64 / 60.0}),
            Schedule::Once { at } => {
                let t = local(*at);
                json!({"kind": "once", "at": at, "date": t.date(), "time": format!("{:02}:{:02}", t.hour, t.minute)})
            }
        }
    }

    /// In words, for the model and the phone: "every day at 08:00", "Mon, Wed at 07:30", "every 6 hours".
    pub fn describe(&self) -> String {
        match self {
            Schedule::Daily { hour, minute, days } => {
                let at = format!("{hour:02}:{minute:02}");
                match days {
                    0b1111111 => format!("every day at {at}"),
                    0b0111110 => format!("on weekdays at {at}"),
                    0b1000001 => format!("at weekends at {at}"),
                    _ => {
                        let names: Vec<String> = (0..7).filter(|i| days & (1 << i) != 0).map(|i| capital(DAY_NAMES[i])).collect();
                        format!("{} at {at}", names.join(", "))
                    }
                }
            }
            Schedule::Every { minutes } if minutes % 60 == 0 => match minutes / 60 {
                1 => "every hour".into(),
                h => format!("every {h} hours"),
            },
            Schedule::Every { minutes } => format!("every {minutes} minutes"),
            Schedule::Once { at } => {
                let t = local(*at);
                format!("once, on {} at {:02}:{:02}", t.date(), t.hour, t.minute)
            }
        }
    }

    /// The first time after `after` (seconds) it's due, or None when it never is again.
    pub fn next_after(&self, after: i64) -> Option<i64> {
        match self {
            Schedule::Daily { hour, minute, days } => {
                let today = local(after);
                (0..8).map(|d| local_epoch(today.year, today.month, today.day + d, *hour, *minute)).find(|&t| t > after && days & (1 << local(t).weekday) != 0)
            }
            Schedule::Every { minutes } => Some(after + *minutes as i64 * 60),
            Schedule::Once { at } => (*at > after).then_some(*at),
        }
    }
}

fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Routine {
    pub id: String,
    pub name: String,
    /// What the Bot is asked each time.
    pub task: String,
    pub schedule: Schedule,
    pub enabled: bool,
    /// The user's yes: routines the Bot proposed start without it.
    pub approved: bool,
    /// "user" or "bot".
    pub by: String,
    pub created: f64,
    pub last_run: Option<i64>,
    pub next_run: Option<i64>,
}

impl Routine {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "name": self.name, "task": self.task, "schedule": self.schedule.to_json(),
            "when": self.schedule.describe(), "enabled": self.enabled, "approved": self.approved, "by": self.by,
            "created": self.created, "last_run": self.last_run, "next_run": self.next_run,
        })
    }

    pub fn from_json(v: &Value) -> Option<Routine> {
        Some(Routine {
            id: v["id"].as_str()?.to_string(),
            name: v["name"].as_str().unwrap_or("").to_string(),
            task: v["task"].as_str().unwrap_or("").to_string(),
            schedule: Schedule::parse(&v["schedule"]).ok()?,
            enabled: v["enabled"].as_bool().unwrap_or(true),
            approved: v["approved"].as_bool().unwrap_or(false),
            by: v["by"].as_str().unwrap_or("user").to_string(),
            created: v["created"].as_f64().unwrap_or(0.0),
            last_run: v["last_run"].as_i64(),
            next_run: v["next_run"].as_i64(),
        })
    }

    /// A new routine from {name, task, schedule}.
    pub fn new(v: &Value, by: &str, now: i64) -> Result<Routine, String> {
        let mut r = Routine {
            id: super::new_id(),
            name: String::new(),
            task: String::new(),
            schedule: Schedule::Every { minutes: 60 },
            enabled: true,
            approved: by == "user",
            by: by.into(),
            created: crate::time::now(),
            last_run: None,
            next_run: None,
        };
        r.apply(v, now)?;
        Ok(r)
    }

    /// Takes the changed fields; a new schedule (or switching it back on) works out the next run again.
    pub fn apply(&mut self, v: &Value, now: i64) -> Result<(), String> {
        if let Some(n) = v["name"].as_str() {
            self.name = n.trim().chars().take(60).collect();
        }
        if let Some(t) = v["task"].as_str() {
            self.task = t.trim().to_string();
        }
        let mut reschedule = self.next_run.is_none() && self.last_run.is_none();
        if v["schedule"].is_object() {
            self.schedule = Schedule::parse(&v["schedule"])?;
            reschedule = true;
        }
        if let Some(e) = v["enabled"].as_bool() {
            reschedule |= e && !self.enabled;
            self.enabled = e;
        }
        if let Some(a) = v["approved"].as_bool() {
            self.approved = a;
        }
        if self.name.is_empty() {
            self.name = self.task.chars().take(40).collect();
        }
        if self.task.is_empty() {
            return Err("Say what the Bot should do each time.".into());
        }
        if self.task.chars().count() > 4000 {
            return Err("The task is too long (up to 4000 characters).".into());
        }
        if reschedule {
            self.next_run = self.schedule.next_after(now);
            if self.next_run.is_none() {
                return Err("That time has already passed.".into());
            }
        }
        Ok(())
    }

    /// Due now: switched on, approved, and its time has come.
    pub fn due(&self, now: i64) -> bool {
        self.enabled && self.approved && self.next_run.is_some_and(|t| t <= now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> i64 {
        local_epoch(y, mo, d, h, mi)
    }

    #[test]
    fn schedules_parse_and_describe() {
        let s = Schedule::parse(&json!({"kind": "daily", "time": "08:00"})).unwrap();
        assert_eq!(s.describe(), "every day at 08:00");
        assert_eq!(Schedule::parse(&json!({"kind": "weekdays", "time": "7:30"})).unwrap().describe(), "on weekdays at 07:30");
        assert_eq!(Schedule::parse(&json!({"kind": "daily", "time": "18:05", "days": ["Mon", "wed", "friday"]})).unwrap().describe(), "Mon, Wed, Fri at 18:05");
        assert_eq!(Schedule::parse(&json!({"kind": "every", "hours": 6})).unwrap().describe(), "every 6 hours");
        assert_eq!(Schedule::parse(&json!({"kind": "every", "hours": "0.5"})).unwrap().describe(), "every 30 minutes");
        for bad in [json!({"kind": "daily", "time": "25:00"}), json!({"kind": "daily"}), json!({"kind": "every", "hours": 0.1}), json!({"kind": "daily", "time": "08:00", "days": ["funday"]}), json!({"kind": "once", "time": "08:00", "date": "tomorrow"}), json!({"kind": "monthly"})] {
            assert!(Schedule::parse(&bad).is_err(), "{bad}");
        }
        // What to_json writes, parse reads back.
        for s in [Schedule::Daily { hour: 9, minute: 15, days: 0b0101010 }, Schedule::Every { minutes: 90 }, Schedule::Once { at: at(2026, 12, 1, 9, 0) }] {
            assert_eq!(Schedule::parse(&s.to_json()).unwrap(), s);
        }
    }

    #[test]
    fn next_times() {
        // Monday 9 November 2026, 10:00.
        let now = at(2026, 11, 9, 10, 0);
        assert_eq!(local(now).weekday, 1);
        let daily = Schedule::parse(&json!({"kind": "daily", "time": "08:00"})).unwrap();
        assert_eq!(daily.next_after(now), Some(at(2026, 11, 10, 8, 0)), "today's has passed: tomorrow");
        let evening = Schedule::parse(&json!({"kind": "daily", "time": "18:00"})).unwrap();
        assert_eq!(evening.next_after(now), Some(at(2026, 11, 9, 18, 0)));
        // From Friday after 08:00, weekdays go to Monday; across the month's end too.
        let weekdays = Schedule::parse(&json!({"kind": "weekdays", "time": "08:00"})).unwrap();
        assert_eq!(weekdays.next_after(at(2026, 11, 13, 9, 0)), Some(at(2026, 11, 16, 8, 0)));
        assert_eq!(weekdays.next_after(at(2026, 10, 30, 9, 0)), Some(at(2026, 11, 2, 8, 0)));
        let sundays = Schedule::parse(&json!({"kind": "daily", "time": "08:00", "days": ["sun"]})).unwrap();
        assert_eq!(sundays.next_after(now), Some(at(2026, 11, 15, 8, 0)));
        assert_eq!(sundays.next_after(at(2026, 11, 15, 8, 0)), Some(at(2026, 11, 22, 8, 0)), "exactly at the time: the next one");
        assert_eq!(Schedule::Every { minutes: 120 }.next_after(now), Some(now + 7200));
        let once = Schedule::Once { at: at(2026, 11, 9, 9, 0) };
        assert_eq!(once.next_after(now), None, "a time that's past never comes");
    }

    #[test]
    fn routines_need_a_task_and_a_future() {
        let now = at(2026, 11, 9, 10, 0);
        assert!(Routine::new(&json!({"name": "x", "schedule": {"kind": "every", "hours": 1}}), "user", now).is_err());
        assert!(Routine::new(&json!({"task": "x", "schedule": {"kind": "once", "date": "2026-11-09", "time": "09:00"}}), "user", now).is_err());
        let r = Routine::new(&json!({"task": "Check the news", "schedule": {"kind": "every", "hours": 1}}), "bot", now).unwrap();
        assert_eq!((r.name.as_str(), r.approved, r.next_run), ("Check the news", false, Some(now + 3600)));
        assert!(!r.due(now + 3600), "not approved yet");
        let mut r = Routine::new(&json!({"task": "x", "schedule": {"kind": "every", "hours": 1}}), "user", now).unwrap();
        assert!(r.due(now + 3600) && !r.due(now + 3599));
        r.apply(&json!({"enabled": false}), now).unwrap();
        assert!(!r.due(now + 3600));
        // Switching it back on later works out the next time from then.
        r.apply(&json!({"enabled": true}), now + 10_000).unwrap();
        assert_eq!(r.next_run, Some(now + 10_000 + 3600));
    }
}
