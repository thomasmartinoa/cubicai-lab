use super::TempDir;
use crate::desktop::{self, Dirs, LAB_ID};
use std::path::Path;

fn dirs(t: &TempDir) -> Dirs {
    Dirs { bin: t.join("bin"), apps: t.join("share/applications"), icons: t.join("share/icons/hicolor/scalable/apps") }
}

/// A key of the entry's main group.
fn get(entry: &str, key: &str) -> String {
    entry
        .lines()
        .take_while(|l| !l.starts_with("[Desktop Action"))
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|x| x.1.to_string())
        .unwrap_or_default()
}

#[test]
fn one_entry_with_the_icon_and_the_phone_panel_as_actions() {
    let lab = desktop::lab_entry(Path::new("/home/sam/.local/bin/cubicai-lab"));
    assert!(lab.starts_with("[Desktop Entry]\n"));
    assert_eq!(get(&lab, "Name"), "CubicAI Lab");
    assert_eq!(get(&lab, "Exec"), "/home/sam/.local/bin/cubicai-lab");
    // Wayland finds a window's entry (and so its icon) by app id = file name = StartupWMClass.
    assert_eq!((get(&lab, "Icon"), get(&lab, "StartupWMClass")), (LAB_ID.to_string(), LAB_ID.to_string()));
    assert_eq!(get(&lab, "Actions"), "phone;pair;");
    assert!(lab.contains("[Desktop Action phone]\nName=Phone panel\nExec=/home/sam/.local/bin/cubicai-lab phone\n"));
    assert!(lab.contains("[Desktop Action pair]\nName=Pair a phone\nExec=/home/sam/.local/bin/cubicai-lab pair\n"));
    for a in ["phone", "pair"] {
        assert!(crate::app::start_view(a).is_some(), "{a} opens the Phone tab");
    }
    for l in lab.lines() {
        assert!(l.is_empty() || l.starts_with('[') || l.contains('='), "{l}");
    }
}

#[test]
fn paths_with_spaces_are_quoted() {
    let e = desktop::lab_entry(Path::new("/home/a b/my \"apps\"/cubicai-lab"));
    assert_eq!(get(&e, "Exec"), r#""/home/a b/my \"apps\"/cubicai-lab""#);
    assert!(e.contains("Exec=\"/home/a b/my \\\"apps\\\"/cubicai-lab\" pair\n"));
}

#[test]
fn the_icon_is_the_cube_on_a_tile() {
    let svg = desktop::LAB_ICON;
    assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 108 108\""));
    assert!(svg.trim_end().ends_with("</svg>"));
    // The phone app's cube colours.
    for c in ["#FFC861", "#F29E38", "#C46F1C", "#161B33"] {
        assert!(svg.contains(c), "{c}");
    }
}

#[test]
fn writing_the_entry_changes_only_what_differs_and_drops_the_old_server_one() {
    let t = TempDir::new("desktop");
    let d = dirs(&t);
    let exe = d.program();
    // What an earlier version left: a separate Server entry and icon, and the Python installer's entry.
    std::fs::create_dir_all(&d.apps).unwrap();
    std::fs::create_dir_all(&d.icons).unwrap();
    std::fs::write(d.apps.join("cubicai-server.desktop"), "[Desktop Entry]\nName=CubicAI Server\nExec=/x/cubicai-lab phone\nX-CubicAI-Managed=true\n").unwrap();
    std::fs::write(d.icons.join("cubicai-server.svg"), "<svg><!-- CubicAI Server: the cube --></svg>").unwrap();
    std::fs::write(d.apps.join("dev.martin.cubicai.Server.desktop"), "[Desktop Entry]\nName=CubicAI Server\nExec=/home/sam/.local/bin/cubicai-lab phone\nIcon=network-server\n").unwrap();
    let first = desktop::write_entries(&d, &exe).unwrap();
    assert_eq!(
        first,
        [
            d.icons.join("cubicai-lab.svg"),
            d.apps.join("cubicai-lab.desktop"),
            d.apps.join("cubicai-server.desktop"),
            d.apps.join("dev.martin.cubicai.Server.desktop"),
            d.icons.join("cubicai-server.svg"),
        ]
    );
    let mut left: Vec<String> = std::fs::read_dir(&d.apps).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| n.ends_with(".desktop")).collect();
    left.sort();
    assert_eq!(left, ["cubicai-lab.desktop"], "one entry");
    assert!(desktop::write_entries(&d, &exe).unwrap().is_empty(), "nothing to do the second time");
    // Someone else's files with those names are left alone.
    std::fs::write(d.apps.join("cubicai-server.desktop"), "[Desktop Entry]\nName=Other\nExec=/usr/bin/other\n").unwrap();
    std::fs::write(d.icons.join("cubicai-server.svg"), "<svg/>").unwrap();
    assert!(desktop::write_entries(&d, &exe).unwrap().is_empty());
    assert!(d.apps.join("cubicai-server.desktop").exists() && d.icons.join("cubicai-server.svg").exists());
    // A moved program, or a changed icon, is rewritten.
    let moved = t.join("elsewhere/cubicai-lab");
    assert_eq!(desktop::write_entries(&d, &moved).unwrap(), [d.apps.join("cubicai-lab.desktop")]);
    std::fs::write(d.icons.join("cubicai-lab.svg"), "old").unwrap();
    assert_eq!(desktop::write_entries(&d, &moved).unwrap(), [d.icons.join("cubicai-lab.svg")]);
    assert_eq!(std::fs::read_to_string(d.icons.join("cubicai-lab.svg")).unwrap(), desktop::LAB_ICON);
}
