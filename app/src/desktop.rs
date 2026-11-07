//! The Lab in the desktop's app menu (Linux, freedesktop): `cubicai-lab install` copies the program
//! to ~/.local/bin and writes one menu entry, "CubicAI Lab", with the CubicAI icon. Its right-click
//! actions open the Phone tab. An installed copy keeps the entry up to date on every start, so a
//! new icon or a moved program reaches the menu by itself.
//!
//! The entry's file name is the window's app id (`cubicai-lab`), so Wayland docks and task
//! switchers find the icon.

use std::path::{Path, PathBuf};

pub const LAB_ID: &str = "cubicai-lab";
pub const LAB_ICON: &str = include_str!("../assets/icons/cubicai-lab.svg");
/// Server entries earlier versions (and the Python gateway's installer) wrote; one entry is enough.
pub const OLD_ENTRIES: [&str; 2] = ["cubicai-server.desktop", "dev.martin.cubicai.Server.desktop"];
pub const OLD_ICON: &str = "cubicai-server.svg";

/// Where the program, menu entries and icons go for this user.
pub struct Dirs {
    pub bin: PathBuf,
    pub apps: PathBuf,
    pub icons: PathBuf,
}

impl Dirs {
    pub fn user() -> Option<Dirs> {
        let home = PathBuf::from(std::env::var_os("HOME")?);
        let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(|| home.join(".local/share"));
        Some(Dirs { bin: home.join(".local/bin"), apps: data.join("applications"), icons: data.join("icons/hicolor/scalable/apps") })
    }

    pub fn program(&self) -> PathBuf {
        self.bin.join(LAB_ID)
    }
}

/// A path for an Exec line: quoted when it has spaces or quotes (the spec's escaping).
fn exec_path(p: &Path) -> String {
    let s = p.to_string_lossy();
    if s.chars().any(|c| c.is_whitespace() || matches!(c, '"' | '\'' | '\\' | '$' | '`')) {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('$', "\\$").replace('`', "\\`"))
    } else {
        s.into_owned()
    }
}

pub fn lab_entry(exe: &Path) -> String {
    let e = exec_path(exe);
    format!(
        "[Desktop Entry]\nType=Application\nName=CubicAI Lab\nGenericName=AI model lab\n\
         Comment=Chat with local and cloud models, compare them, look inside them and fine-tune them\n\
         Exec={e}\nIcon={LAB_ID}\nTerminal=false\nCategories=Development;Science;ArtificialIntelligence;\n\
         Keywords=llm;llama;ollama;gguf;fine-tune;chat;cubicai;phone;gateway;\nStartupWMClass={LAB_ID}\n\
         Actions=phone;pair;\nX-CubicAI-Managed=true\n\n\
         [Desktop Action phone]\nName=Phone panel\nExec={e} phone\n\n\
         [Desktop Action pair]\nName=Pair a phone\nExec={e} pair\n"
    )
}

/// Whether a menu entry is one this Lab wrote (or the old Python installer's, which ran this Lab).
fn ours(entry: &str) -> bool {
    entry.lines().any(|l| l == "X-CubicAI-Managed=true" || (l.starts_with("Exec=") && l.contains(LAB_ID)))
}

/// Writes the entry and icon if they differ from what's there, and removes the separate Server
/// entry and icon older versions wrote (only ours). Returns what changed.
pub fn write_entries(d: &Dirs, exe: &Path) -> Result<Vec<PathBuf>, String> {
    let mut changed = vec![];
    for (path, text) in [(d.icons.join(format!("{LAB_ID}.svg")), LAB_ICON.to_string()), (d.apps.join(format!("{LAB_ID}.desktop")), lab_entry(exe))] {
        if std::fs::read_to_string(&path).ok().as_deref() == Some(text.as_str()) {
            continue;
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        changed.push(path);
    }
    for name in OLD_ENTRIES {
        let p = d.apps.join(name);
        if std::fs::read_to_string(&p).is_ok_and(|t| ours(&t)) && std::fs::remove_file(&p).is_ok() {
            changed.push(p);
        }
    }
    let icon = d.icons.join(OLD_ICON);
    if std::fs::read_to_string(&icon).is_ok_and(|t| t.contains("CubicAI Server")) && std::fs::remove_file(&icon).is_ok() {
        changed.push(icon);
    }
    if !changed.is_empty() {
        // Best effort: menus that cache entries or icons pick the change up straight away.
        let quiet = |cmd: &str, args: &[&std::ffi::OsStr]| {
            let _ = std::process::Command::new(cmd).args(args).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status();
        };
        quiet("update-desktop-database", &[d.apps.as_os_str()]);
        if let Some(hicolor) = d.icons.parent().and_then(Path::parent).and_then(Path::parent) {
            quiet("gtk-update-icon-cache", &["-q".as_ref(), "-t".as_ref(), hicolor.as_os_str()]);
        }
    }
    Ok(changed)
}

/// `/proc/self/exe` of a program replaced on disk while running reads "<path> (deleted)".
fn running_program() -> Option<PathBuf> {
    let p = std::env::current_exe().ok()?;
    let s = p.to_string_lossy();
    Some(s.strip_suffix(" (deleted)").map(PathBuf::from).unwrap_or(p))
}

/// `cubicai-lab install`: copies this program to ~/.local/bin (copy, then rename, so a running copy
/// keeps working) and writes the menu entry. Returns a report for the terminal.
pub fn install() -> Result<String, String> {
    let d = Dirs::user().ok_or("No home folder")?;
    let me = running_program().ok_or("Can't tell where this program is")?;
    let target = d.program();
    let mut report = vec![];
    if std::fs::canonicalize(&me).ok() != std::fs::canonicalize(&target).ok() {
        std::fs::create_dir_all(&d.bin).map_err(|e| e.to_string())?;
        let tmp = target.with_extension("new");
        std::fs::copy(&me, &tmp).map_err(|e| format!("Copying to {}: {e}", tmp.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755));
        }
        std::fs::rename(&tmp, &target).map_err(|e| e.to_string())?;
        report.push(format!("Installed {}", target.display()));
    }
    let changed = write_entries(&d, &target)?;
    report.push(if changed.is_empty() { "Menu entry already up to date".into() } else { format!("Menu entry and icon: {}", changed.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")) });
    if !std::env::var("PATH").unwrap_or_default().split(':').any(|p| Path::new(p) == d.bin) {
        report.push(format!("Note: {} isn't on your PATH; the menu entry works anyway.", d.bin.display()));
    }
    Ok(report.join("\n"))
}

/// On start: an installed copy (the one in ~/.local/bin) keeps its menu entry up to date.
pub fn refresh() {
    let (Some(d), Some(me)) = (Dirs::user(), running_program()) else { return };
    if me == d.program() {
        let _ = write_entries(&d, &me);
    }
}
