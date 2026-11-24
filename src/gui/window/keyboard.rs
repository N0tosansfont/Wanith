use egui::{emath::GuiRounding, Color32, Mesh, Pos2, Rect, Sense, Ui};

use crate::{midi::MIDIColor, settings::SkinPreset};

use super::keyboard_layout::KeyboardView;

pub struct GuiKeyboard {}

impl GuiKeyboard {
    pub fn new() -> GuiKeyboard {
        GuiKeyboard {}
    }

    pub fn draw(
        &mut self,
        ui: &mut Ui,
        key_view: &KeyboardView,
        colors: &[Option<MIDIColor>],
        bar_color: &Color32,
        skin: SkinPreset,
    ) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        let mut mesh = Mesh::default();
        let note_border =
            crate::utils::calculate_border_width(rect.width(), key_view.visible_range.len() as f32);
        let key_border = (note_border / 2.0).round_to_pixels(ui.painter().pixels_per_point());

        let md_height = match skin {
            SkinPreset::WfaPlus => rect.height() * 0.036,
            SkinPreset::Zanith => rect.height() * 0.040,
            _ => rect.height() * 0.048,
        };
        let bar = rect.height() * 0.06;

        let black_key_overlap = bar / 2.35;
        let top = rect.top() + bar;
        let bottom = rect.bottom();
        let black_bottom = rect.bottom() - rect.height() * 0.34;
        let map_x = |num: f32| rect.left() + num * rect.width();
        fn map_color(col: MIDIColor) -> Color32 {
            Color32::from_rgb(col.red(), col.green(), col.blue())
        }
        fn scale_rgb(c: Color32, k: f32) -> Color32 {
            Color32::from_rgb(
                (c.r() as f32 * k).clamp(0.0, 255.0) as u8,
                (c.g() as f32 * k).clamp(0.0, 255.0) as u8,
                (c.b() as f32 * k).clamp(0.0, 255.0) as u8,
            )
        }
        fn lift_white(c: Color32, t: f32) -> Color32 {
            Color32::from_rgb(
                (c.r() as f32 + (255.0 - c.r() as f32) * t).clamp(0.0, 255.0) as u8,
                (c.g() as f32 + (255.0 - c.g() as f32) * t).clamp(0.0, 255.0) as u8,
                (c.b() as f32 + (255.0 - c.b() as f32) * t).clamp(0.0, 255.0) as u8,
            )
        }

        // Skin palette parameters
        let (wk_top_sh, wk_mid, wk_face, wk_lip_top, wk_lip_bot, wk_div) = match skin {
            SkinPreset::Wasaby => (
                Color32::from_rgb(110, 110, 110),
                Color32::from_rgb(210, 210, 210),
                Color32::WHITE,
                Color32::from_rgb(190, 190, 190),
                Color32::from_rgb(120, 120, 120),
                Color32::from_rgb(40, 40, 40),
            ),
            SkinPreset::Wynthesia => (
                Color32::from_rgb(128, 126, 120),
                Color32::from_rgb(232, 230, 224),
                Color32::from_rgb(252, 251, 246),
                Color32::from_rgb(205, 202, 195),
                Color32::from_rgb(140, 138, 132),
                Color32::from_rgb(56, 56, 62),
            ),
            SkinPreset::WfaPlus => (
                Color32::from_rgb(155, 155, 155),
                Color32::from_rgb(240, 240, 240),
                Color32::from_rgb(255, 255, 255),
                Color32::from_rgb(185, 185, 185),
                Color32::from_rgb(115, 115, 115),
                Color32::from_rgb(24, 24, 24),
            ),
            SkinPreset::Zanith => (
                Color32::from_rgb(95, 102, 116),
                Color32::from_rgb(206, 212, 224),
                Color32::from_rgb(236, 240, 248),
                Color32::from_rgb(155, 162, 176),
                Color32::from_rgb(86, 92, 106),
                Color32::from_rgb(22, 25, 34),
            ),
        };

        for (i, key) in key_view.iter_visible_keys() {
            if !key.black {
                if let Some(color) = colors[i].map(map_color) {
                    // Pressed white key
                    let (top_c, mid_c, bot_c, lip_top_c, lip_bot_c) = match skin {
                        SkinPreset::Wynthesia => (
                            scale_rgb(color, 0.55),
                            lift_white(color, 0.14),
                            lift_white(color, 0.26),
                            scale_rgb(color, 0.45),
                            scale_rgb(color, 0.72),
                        ),
                        SkinPreset::WfaPlus => (
                            scale_rgb(color, 0.78),
                            color,
                            scale_rgb(color, 0.92),
                            scale_rgb(color, 0.42),
                            scale_rgb(color, 0.60),
                        ),
                        SkinPreset::Zanith => (
                            lift_white(color, 0.35),
                            lift_white(color, 0.10),
                            scale_rgb(color, 0.84),
                            scale_rgb(color, 0.36),
                            scale_rgb(color, 0.55),
                        ),
                        SkinPreset::Wasaby => (
                            scale_rgb(color, 0.3),
                            scale_rgb(color, 0.6),
                            color,
                            scale_rgb(color, 0.3),
                            scale_rgb(color, 0.6),
                        ),
                    };

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(Pos2::new(map_x(key.left), top), top_c);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), top), top_c);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), top + black_key_overlap),
                        mid_c,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), top + black_key_overlap),
                        mid_c,
                    );

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), top + black_key_overlap),
                        mid_c,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), top + black_key_overlap),
                        mid_c,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), bottom), bot_c);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), bottom), bot_c);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), bottom - key_border * 2.0),
                        lip_top_c,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), bottom - key_border * 2.0),
                        lip_top_c,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), bottom), lip_bot_c);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), bottom), lip_bot_c);
                } else {
                    // Unpressed white key
                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(Pos2::new(map_x(key.left), top), wk_top_sh);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), top), wk_top_sh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), top + black_key_overlap),
                        wk_mid,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), top + black_key_overlap),
                        wk_mid,
                    );

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), top + black_key_overlap),
                        wk_mid,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), top + black_key_overlap),
                        wk_mid,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), bottom - md_height), wk_face);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), bottom - md_height), wk_face);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), bottom - md_height),
                        wk_lip_top,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), bottom - md_height),
                        wk_lip_top,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), bottom), wk_lip_bot);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), bottom), wk_lip_bot);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), bottom - md_height),
                        Color32::from_rgb(70, 70, 70),
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), bottom - md_height),
                        Color32::from_rgb(70, 70, 70),
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left), bottom - md_height + key_border * 2.0),
                        Color32::from_rgb(140, 140, 140),
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right), bottom - md_height + key_border * 2.0),
                        Color32::from_rgb(140, 140, 140),
                    );
                }
                // White key divider
                let top_left4 = Pos2::new(map_x(key.right), top);
                let bottom_right4 = Pos2::new(map_x(key.right) - key_border, bottom);
                let rect4 = Rect::from_min_max(top_left4, bottom_right4);
                mesh.add_colored_rect(rect4, wk_div);
            }
        }

        // Coloured felt / accent bar
        let (bar_top_c, bar_bot_c, prog_top_c, prog_bot_c) = match skin {
            SkinPreset::Zanith => (
                lift_white(*bar_color, 0.25),
                scale_rgb(*bar_color, 0.45),
                Color32::from_rgb(44, 50, 64),
                Color32::from_rgb(18, 22, 30),
            ),
            SkinPreset::Wynthesia => (
                scale_rgb(*bar_color, 0.45),
                *bar_color,
                Color32::from_rgb(76, 78, 86),
                Color32::from_rgb(32, 34, 40),
            ),
            SkinPreset::WfaPlus => (
                scale_rgb(*bar_color, 0.25),
                *bar_color,
                Color32::from_rgb(65, 65, 65),
                Color32::from_rgb(24, 24, 24),
            ),
            SkinPreset::Wasaby => (
                scale_rgb(*bar_color, 0.3),
                *bar_color,
                Color32::from_rgb(90, 90, 90),
                Color32::from_rgb(40, 40, 40),
            ),
        };

        add_rect_triangles(&mut mesh);
        mesh.colored_vertex(Pos2::new(rect.left(), top - black_key_overlap), bar_top_c);
        mesh.colored_vertex(Pos2::new(rect.right(), top - black_key_overlap), bar_top_c);
        mesh.colored_vertex(Pos2::new(rect.left(), top), bar_bot_c);
        mesh.colored_vertex(Pos2::new(rect.right(), top), bar_bot_c);

        // Progress bar backing
        add_rect_triangles(&mut mesh);
        mesh.colored_vertex(Pos2::new(rect.left(), rect.top()), prog_top_c);
        mesh.colored_vertex(Pos2::new(rect.right(), rect.top()), prog_top_c);
        mesh.colored_vertex(Pos2::new(rect.left(), top - black_key_overlap), prog_bot_c);
        mesh.colored_vertex(Pos2::new(rect.right(), top - black_key_overlap), prog_bot_c);

        let (bk_rim_hi, bk_rim_lo, bk_top, bk_bot) = match skin {
            SkinPreset::Wasaby => (
                Color32::from_rgb(105, 105, 105),
                Color32::from_rgb(20, 20, 20),
                Color32::from_rgb(20, 20, 20),
                Color32::from_rgb(40, 40, 40),
            ),
            SkinPreset::Wynthesia => (
                Color32::from_rgb(120, 124, 134),
                Color32::from_rgb(24, 26, 32),
                Color32::from_rgb(30, 33, 40),
                Color32::from_rgb(56, 60, 70),
            ),
            SkinPreset::WfaPlus => (
                Color32::from_rgb(92, 92, 92),
                Color32::from_rgb(10, 10, 10),
                Color32::from_rgb(12, 12, 12),
                Color32::from_rgb(28, 28, 28),
            ),
            SkinPreset::Zanith => (
                Color32::from_rgb(96, 108, 130),
                Color32::from_rgb(12, 14, 22),
                Color32::from_rgb(16, 19, 28),
                Color32::from_rgb(36, 42, 56),
            ),
        };

        for (i, key) in key_view.iter_visible_keys() {
            if key.black {
                if let Some(color) = colors[i].map(map_color) {
                    // Pressed black key
                    let darkened = match skin {
                        SkinPreset::Zanith => scale_rgb(color, 0.70),
                        SkinPreset::Wynthesia => scale_rgb(color, 0.80),
                        _ => scale_rgb(color, 0.76),
                    };
                    let lightened = match skin {
                        SkinPreset::Zanith => lift_white(color, 0.34),
                        SkinPreset::Wynthesia => lift_white(color, 0.25),
                        _ => scale_rgb(color, 1.3),
                    };

                    let md_height = md_height / 2.0;
                    let black_key_overlap = black_key_overlap / 2.2;

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + key_border, black_bottom - md_height),
                        color,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - key_border, black_bottom - md_height),
                        color,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), black_bottom), darkened);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), black_bottom), darkened);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(Pos2::new(map_x(key.left), top), lightened);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, top - black_key_overlap),
                        darkened,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), black_bottom), lightened);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, black_bottom - md_height),
                        darkened,
                    );

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - 2.0 * key_border, top - black_key_overlap),
                        lightened,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.right), top), darkened);
                    mesh.colored_vertex(
                        Pos2::new(
                            map_x(key.right) - 2.0 * key_border,
                            black_bottom - md_height,
                        ),
                        lightened,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.right), black_bottom), darkened);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, top - black_key_overlap),
                        if skin == SkinPreset::Zanith { lightened } else { color },
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - 2.0 * key_border, top - black_key_overlap),
                        if skin == SkinPreset::Zanith { lightened } else { color },
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, black_bottom - md_height),
                        darkened,
                    );
                    mesh.colored_vertex(
                        Pos2::new(
                            map_x(key.right) - 2.0 * key_border,
                            black_bottom - md_height,
                        ),
                        darkened,
                    );
                } else {
                    // Unpressed black key
                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + key_border, black_bottom - md_height),
                        bk_rim_hi,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - key_border, black_bottom - md_height),
                        bk_rim_hi,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), black_bottom), bk_rim_lo);
                    mesh.colored_vertex(Pos2::new(map_x(key.right), black_bottom), bk_rim_lo);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(Pos2::new(map_x(key.left), top), bk_rim_lo);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, top - black_key_overlap),
                        bk_rim_hi,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.left), black_bottom), bk_rim_lo);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, black_bottom - md_height),
                        bk_rim_hi,
                    );

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - 2.0 * key_border, top - black_key_overlap),
                        bk_rim_hi,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.right), top), bk_rim_lo);
                    mesh.colored_vertex(
                        Pos2::new(
                            map_x(key.right) - 2.0 * key_border,
                            black_bottom - md_height,
                        ),
                        bk_rim_hi,
                    );
                    mesh.colored_vertex(Pos2::new(map_x(key.right), black_bottom), bk_rim_lo);

                    add_rect_triangles(&mut mesh);
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, top - black_key_overlap),
                        bk_top,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.right) - 2.0 * key_border, top - black_key_overlap),
                        bk_top,
                    );
                    mesh.colored_vertex(
                        Pos2::new(map_x(key.left) + 2.0 * key_border, black_bottom - md_height),
                        bk_bot,
                    );
                    mesh.colored_vertex(
                        Pos2::new(
                            map_x(key.right) - 2.0 * key_border,
                            black_bottom - md_height,
                        ),
                        bk_bot,
                    );
                }
            }
        }

        ui.painter().add(mesh);
    }
}

fn add_rect_triangles(mesh: &mut Mesh) {
    let idx = mesh.vertices.len() as u32;
    mesh.add_triangle(idx, idx + 1, idx + 2);
    mesh.add_triangle(idx + 2, idx + 1, idx + 3);
}
