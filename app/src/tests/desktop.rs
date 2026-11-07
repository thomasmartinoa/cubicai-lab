use super::TempDir;
use crate::desktop::{self, Dirs, LAB_ID, LEGACY, SERVER_ID};
use std::path::Path;

fn dirs(t: &TempDir) -> Dirs {
    Dirs { bin: t.join("bin"), apps: t.join("share/applications"), icons: t.join("share/icons/hicolor/scalable/apps") }
}

/// The keys of a desktop entry's main group.
fn keys(entry: &str) -> Vec<(String, String)> {
    entry.lines().take_while(|l| !l.starts_with("[Desktop Action")).filter_map(|l| l.split_once('=')).map(|(k, v)| (k.into(), v.into())).collect()
}

fn get(entry: &str, key: &str) -> String {
    keys(entry).into_iter().find(|(k, _)| k == key).map(|x| x.1).unwrap_or_default()
}

#[test]
fn entries_name_the_icons_and_windows_they_belong_to() {
    let lab = desktop::lab_entry(Path::new("/home/sam/.local/bin/cubicai-lab"));
    assert!(lab.starts_with("[Desktop Entry]\n"));
    assert_eq!(get(&lab, "Name"), "CubicAI Lab");
    assert_eq!(get(&lab, "Exec"), "/home/sam/.local/bin/cubicai-lab");
    // Wayland finds a window's entry (and so its icon) by app id = file name = StartupWMClass.
    assert_eq!((get(&lab, "Icon"), get(&lab, "StartupWMClass")), (LAB_ID.to_string(), LAB_ID.to_string()));
    let server = desktop::server_entry(Path::new("/home/sam/.local/bin/cubicai-lab"));
    assert_eq!(get(&server, "Exec"), "/home/sam/.local/bin/cubicai-lab phone");
    assert_eq!((get(&server, "Icon"), get(&server, "StartupWMClass")), (SERVER_ID.to_string(), SERVER_ID.to_string()));
    assert!(server.contains("[Desktop Action pair]\nName=Pair a phone\nExec=/home/sam/.local/bin/cubicai-lab pair\n"));
    assert!(server.contains("Exec=/home/sam/.local/bin/cubicai-lab activity\n"));
    // Each argument the entries pass opens the Phone tab (and so the server's window id).
    for a in ["phone", "pair", "activity"] {
        assert!(crate::app::start_view(a).is_some(), "{a}");
    }
    // Every line is a group, a key=value or blank.
    for l in lab.lines().chain(server.lines()) {
        assert!(l.is_empty() || l.starts_with('[') || l.contains('='), "{l}");
    }
}

#[test]
fn paths_with_spaces_are_quoted() {
    let e = desktop::lab_entry(Path::new("/home/a b/my \"apps\"/cubicai-lab"));
    assert_eq!(get(&e, "Exec"), r#""/home/a b/my \"apps\"/cubicai-lab""#);
    let s = desktop::server_entry(Path::new("/opt/c d/cubicai-lab"));
    assert!(s.contains("Exec=\"/opt/c d/cubicai-lab\" phone\n"));
}

#[test]
fn icons_are_the_cube_on_a_tile() {
    for svg in [desktop::LAB_ICON, desktop::SERVER_ICON] {
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 108 108\""));
        assert!(svg.trim_end().ends_with("</svg>"));
        // The phone app's cube colours.
        for c in ["#FFC861", "#F29E38", "#C46F1C", "#161B33"] {
            assert!(svg.contains(c), "{c}");
        }
    }
}

#[test]
fn writing_entries_changes_only_what_differs() {
    let t = TempDir::new("desktop");
    let d = dirs(&t);
    let exe = d.program();
    let first = desktop::write_entries(&d, &exe, false).unwrap();
    assert_eq!(first, [d.icons.join("cubicai-lab.svg"), d.apps.join("cubicai-lab.desktop")]);
    assert!(!d.apps.join("cubicai-server.desktop").exists(), "no Server entry without the gateway");
    assert!(desktop::write_entries(&d, &exe, false).unwrap().is_empty(), "nothing to do the second time");
    // With the gateway installed the Server entry joins; the old one goes only if it runs this Lab.
    std::fs::write(d.apps.join(LEGACY), "[Desktop Entry]\nName=CubicAI Server\nExec=/home/sam/.local/bin/cubicai-lab phone\nIcon=network-server\n").unwrap();
    let second = desktop::write_entries(&d, &exe, true).unwrap();
    assert_eq!(second, [d.icons.join("cubicai-server.svg"), d.apps.join("cubicai-server.desktop"), d.apps.join(LEGACY)]);
    assert!(!d.apps.join(LEGACY).exists());
    std::fs::write(d.apps.join(LEGACY), "[Desktop Entry]\nName=Something else\nExec=/usr/bin/other\n").unwrap();
    assert!(desktop::write_entries(&d, &exe, true).unwrap().is_empty());
    assert!(d.apps.join(LEGACY).exists(), "someone else's entry is left alone");
    // A moved program, or a changed icon, is rewritten.
    let moved = t.join("elsewhere/cubicai-lab");
    assert_eq!(desktop::write_entries(&d, &moved, true).unwrap(), [d.apps.join("cubicai-lab.desktop"), d.apps.join("cubicai-server.desktop")]);
    std::fs::write(d.icons.join("cubicai-lab.svg"), "old").unwrap();
    assert_eq!(desktop::write_entries(&d, &moved, true).unwrap(), [d.icons.join("cubicai-lab.svg")]);
    assert_eq!(std::fs::read_to_string(d.icons.join("cubicai-lab.svg")).unwrap(), desktop::LAB_ICON);
}
