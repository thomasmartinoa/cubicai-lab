//! The Lab in the desktop's app menu (Linux, freedesktop): `cubicai-lab install` copies the program
//! to ~/.local/bin and writes its menu entries and icons; an installed copy keeps them up to date
//! on every start, so a new icon or a moved program reaches the menu by itself.
//!
//! Two entries: "CubicAI Lab" (window app id `cubicai-lab`) and, once the phone gateway is
//! installed, "CubicAI Server", which opens the Phone tab (app id `cubicai-server`). Each entry's
//! file name is its app id, so Wayland docks and task switchers find the right icon.

use std::path::{Path, PathBuf};

pub const LAB_ID: &str = "cubicai-lab";
pub const SERVER_ID: &str = "cubicai-server";
pub const LAB_ICON: &str = include_str!("../assets/icons/cubicai-lab.svg");
pub const SERVER_ICON: &str = include_str!("../assets/icons/cubicai-server.svg");
/// The Server entry older setups wrote (the Python gateway's installer, with a stock icon).
pub const LEGACY: &str = "dev.martin.cubicai.Server.desktop";

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
    format!(
        "[Desktop Entry]\nType=Application\nName=CubicAI Lab\nGenericName=AI model lab\n\
         Comment=Chat with local and cloud models, compare them, look inside them and fine-tune them\n\
         Exec={}\nIcon={LAB_ID}\nTerminal=false\nCategories=Development;Science;ArtificialIntelligence;\n\
         Keywords=llm;llama;ollama;gguf;fine-tune;chat;\nStartupWMClass={LAB_ID}\nX-CubicAI-Managed=true\n",
        exec_path(exe)
    )
}

pub fn server_entry(exe: &Path) -> String {
    let e = exec_path(exe);
    format!(
        "[Desktop Entry]\nType=Application\nName=CubicAI Server\n\
         Comment=The CubicAI phone gateway: status, pairing, remote access, activity\n\
         Exec={e} phone\nIcon={SERVER_ID}\nTerminal=false\nCategories=Network;Utility;\n\
         Keywords=cubicai;phone;gateway;pair;\nStartupWMClass={SERVER_ID}\nActions=pair;activity;\nX-CubicAI-Managed=true\n\n\
         [Desktop Action pair]\nName=Pair a phone\nExec={e} pair\n\n\
         [Desktop Action activity]\nName=Activity\nExec={e} activity\n"
    )
}

/// Writes the entries and icons that differ from what's there and removes the legacy Server entry
/// (only if it runs this Lab). `server`: also the CubicAI Server entry. Returns what changed.
pub fn write_entries(d: &Dirs, exe: &Path, server: bool) -> Result<Vec<PathBuf>, String> {
    let mut files = vec![(d.icons.join(format!("{LAB_ID}.svg")), LAB_ICON.to_string()), (d.apps.join(format!("{LAB_ID}.desktop")), lab_entry(exe))];
    if server {
        files.push((d.icons.join(format!("{SERVER_ID}.svg")), SERVER_ICON.to_string()));
        files.push((d.apps.join(format!("{SERVER_ID}.desktop")), server_entry(exe)));
    }
    let mut changed = vec![];
    for (path, text) in files {
        if std::fs::read_to_string(&path).ok().as_deref() == Some(text.as_str()) {
            continue;
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        changed.push(path);
    }
    let legacy = d.apps.join(LEGACY);
    if server
        && std::fs::read_to_string(&legacy).is_ok_and(|t| t.lines().any(|l| l.starts_with("Exec=") && l.contains(LAB_ID)))
        && std::fs::remove_file(&legacy).is_ok()
    {
        changed.push(legacy);
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

/// Whether the phone gateway is installed (its menu entry is only useful then).
fn gateway_installed() -> bool {
    cubic_gateway::local::installed_binary().is_file()
}

/// `/proc/self/exe` of a program replaced on disk while running reads "<path> (deleted)".
fn running_program() -> Option<PathBuf> {
    let p = std::env::current_exe().ok()?;
    let s = p.to_string_lossy();
    Some(s.strip_suffix(" (deleted)").map(PathBuf::from).unwrap_or(p))
}

/// `cubicai-lab install`: copies this program to ~/.local/bin (copy, then rename, so a running copy
/// keeps working) and writes the menu entries. Returns a report for the terminal.
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
    let server = gateway_installed();
    let changed = write_entries(&d, &target, server)?;
    report.push(if changed.is_empty() { "Menu entries already up to date".into() } else { format!("Menu entries and icons: {}", changed.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")) });
    if !server {
        report.push("CubicAI Server joins the menu once the phone gateway is installed (Phone tab → Install).".into());
    }
    if !std::env::var("PATH").unwrap_or_default().split(':').any(|p| Path::new(p) == d.bin) {
        report.push(format!("Note: {} isn't on your PATH; the menu entries work anyway.", d.bin.display()));
    }
    Ok(report.join("\n"))
}

/// On start: an installed copy (the one in ~/.local/bin) keeps its menu entries up to date.
pub fn refresh() {
    let (Some(d), Some(me)) = (Dirs::user(), running_program()) else { return };
    if me == d.program() {
        let _ = write_entries(&d, &me, gateway_installed());
    }
}
