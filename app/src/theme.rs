//! CubicAI's look on the desktop: cobalt + amber, Chakra Petch headings, Roboto body, cut corners.

use egui::{Color32, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle, Theme, Visuals};
use std::sync::Arc;

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct Palette {
    pub primary: Color32,
    pub on_primary: Color32,
    pub primary_container: Color32,
    pub on_primary_container: Color32,
    pub secondary: Color32,
    pub bg: Color32,
    pub on_bg: Color32,
    pub variant: Color32,
    pub on_variant: Color32,
    pub lowest: Color32,
    pub low: Color32,
    pub container: Color32,
    pub high: Color32,
    pub highest: Color32,
    pub secondary_container: Color32,
    pub outline: Color32,
    pub outline_variant: Color32,
    pub error: Color32,
    pub error_container: Color32,
    pub good: Color32,
    pub cube: [Color32; 3],
}

const fn hex(v: u32) -> Color32 {
    Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

pub const DARK: Palette = Palette {
    primary: hex(0xFFB547),
    on_primary: hex(0x1B1405),
    primary_container: hex(0x4A3510),
    on_primary_container: hex(0xFFDDA6),
    secondary: hex(0xA9B4E8),
    bg: hex(0x161B33),
    on_bg: hex(0xE9ECF8),
    variant: hex(0x2D3765),
    on_variant: hex(0xA3ABCF),
    lowest: hex(0x11152A),
    low: hex(0x1A2040),
    container: hex(0x1E2546),
    high: hex(0x222A4D),
    highest: hex(0x2D3765),
    secondary_container: hex(0x333E70),
    outline: hex(0x4A5488),
    outline_variant: hex(0x2E3765),
    error: hex(0xFF8A80),
    error_container: hex(0x4A1F2A),
    good: hex(0x7FD8A6),
    cube: [hex(0xFFC861), hex(0xF29E38), hex(0xC46F1C)],
};

pub const LIGHT: Palette = Palette {
    primary: hex(0xB8650F),
    on_primary: Color32::WHITE,
    primary_container: hex(0xFFE3B8),
    on_primary_container: hex(0x3D2604),
    secondary: hex(0x3B4A8F),
    bg: hex(0xF3F5FB),
    on_bg: hex(0x1B2140),
    variant: hex(0xE2E6F3),
    on_variant: hex(0x555E86),
    lowest: Color32::WHITE,
    low: hex(0xEDF0F9),
    container: hex(0xE8EBF6),
    high: hex(0xE2E6F3),
    highest: hex(0xD9DEEF),
    secondary_container: hex(0xD5DBF2),
    outline: hex(0xB4BBD8),
    outline_variant: hex(0xD3D8EA),
    error: hex(0xBA1A1A),
    error_container: hex(0xFFDAD6),
    good: hex(0x1E7A4C),
    cube: [hex(0xFFC861), hex(0xF29E38), hex(0xC46F1C)],
};

pub fn pal(ctx: &egui::Context) -> &'static Palette {
    if ctx.global_style().visuals.dark_mode { &DARK } else { &LIGHT }
}

pub fn heading_family() -> FontFamily {
    FontFamily::Name("heading".into())
}

pub fn medium_family() -> FontFamily {
    FontFamily::Name("medium".into())
}

pub fn heading(size: f32) -> FontId {
    FontId::new(size, heading_family())
}

pub fn medium(size: f32) -> FontId {
    FontId::new(size, medium_family())
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    let add = |fonts: &mut FontDefinitions, name: &str, bytes: &'static [u8]| {
        fonts.font_data.insert(name.to_owned(), Arc::new(FontData::from_static(bytes)));
    };
    add(&mut fonts, "roboto", include_bytes!("../assets/fonts/Roboto-Regular.ttf"));
    add(&mut fonts, "roboto-medium", include_bytes!("../assets/fonts/Roboto-Medium.ttf"));
    add(&mut fonts, "chakra", include_bytes!("../assets/fonts/chakra_petch_semibold.ttf"));
    add(&mut fonts, "chakra-medium", include_bytes!("../assets/fonts/chakra_petch_medium.ttf"));
    add(&mut fonts, "jbmono", include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf"));
    // egui's own fonts stay behind ours as fallbacks (other scripts, emoji, symbols).
    let fallback_prop = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let fallback_mono = fonts.families.get(&FontFamily::Monospace).cloned().unwrap_or_default();
    let with = |first: &[&str], rest: &[String]| -> Vec<String> {
        first.iter().map(|s| s.to_string()).chain(rest.iter().cloned()).collect()
    };
    fonts.families.insert(FontFamily::Proportional, with(&["roboto"], &fallback_prop));
    fonts.families.insert(FontFamily::Monospace, with(&["jbmono"], &fallback_mono));
    fonts.families.insert(heading_family(), with(&["chakra", "roboto"], &fallback_prop));
    fonts.families.insert(FontFamily::Name("heading-medium".into()), with(&["chakra-medium", "roboto"], &fallback_prop));
    fonts.families.insert(medium_family(), with(&["roboto-medium"], &fallback_prop));
    ctx.set_fonts(fonts);

    for (theme, p) in [(Theme::Dark, &DARK), (Theme::Light, &LIGHT)] {
        ctx.style_mut_of(theme, |s| {
            s.text_styles = [
                (TextStyle::Heading, heading(22.0)),
                (TextStyle::Body, FontId::proportional(14.5)),
                (TextStyle::Button, FontId::new(14.0, medium_family())),
                (TextStyle::Small, FontId::proportional(12.0)),
                (TextStyle::Monospace, FontId::monospace(12.5)),
            ]
            .into();
            s.spacing.item_spacing = egui::vec2(8.0, 6.0);
            s.spacing.button_padding = egui::vec2(10.0, 5.0);
            s.spacing.interact_size.y = 26.0;
            s.spacing.slider_width = 220.0;
            s.visuals = visuals(theme == Theme::Dark, p);
        });
    }
}

fn visuals(dark: bool, p: &Palette) -> Visuals {
    let mut v = if dark { Visuals::dark() } else { Visuals::light() };
    v.panel_fill = p.bg;
    v.window_fill = p.high;
    v.extreme_bg_color = p.lowest;
    v.faint_bg_color = p.low;
    v.code_bg_color = p.lowest;
    v.override_text_color = Some(p.on_bg);
    v.hyperlink_color = p.primary;
    v.selection.bg_fill = p.primary.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, p.primary);
    v.window_stroke = Stroke::new(1.0, p.outline_variant);
    v.window_corner_radius = 0.into();
    v.menu_corner_radius = 0.into();
    v.warn_fg_color = p.primary;
    v.error_fg_color = p.error;
    let w = &mut v.widgets;
    for (wv, fill, stroke) in [
        (&mut w.noninteractive, p.container, p.outline_variant),
        (&mut w.inactive, p.high, p.outline_variant),
        (&mut w.hovered, p.highest, p.outline),
        (&mut w.active, p.secondary_container, p.primary),
        (&mut w.open, p.highest, p.outline),
    ] {
        wv.corner_radius = 0.into();
        wv.bg_fill = fill;
        wv.weak_bg_fill = fill;
        wv.bg_stroke = Stroke::new(1.0, stroke);
    }
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.on_bg);
    w.inactive.fg_stroke = Stroke::new(1.0, p.on_bg);
    w.hovered.fg_stroke = Stroke::new(1.5, p.on_bg);
    w.active.fg_stroke = Stroke::new(1.5, p.on_bg);
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.outline_variant);
    v
}
