use egui::{Color32, CornerRadius, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui, Vec2};

use crate::{
    settings::{SkinPreset, WasabiSettings},
    state::WasabiState,
};

use super::SettingsWindow;

impl SettingsWindow {
    pub(super) fn show_skin_settings(
        &mut self,
        ui: &mut Ui,
        settings: &mut WasabiSettings,
        state: &mut WasabiState,
        _width: f32,
    ) {
        let i18n = &state.i18n;
        ui.heading(i18n.t("skin.heading"));
        ui.label(
            RichText::new(i18n.t("skin.subtitle"))
                .small()
                .color(Color32::from_rgb(175, 175, 185)),
        );
        ui.add_space(12.0);

        let pairs = [
            (SkinPreset::Wasaby, SkinPreset::Wynthesia),
            (SkinPreset::WfaPlus, SkinPreset::Zanith),
        ];

        for (left, right) in pairs {
            ui.columns(2, |cols| {
                self.draw_skin_card(&mut cols[0], settings, state, left);
                self.draw_skin_card(&mut cols[1], settings, state, right);
            });
            ui.add_space(10.0);
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let cur = settings.scene.skin;
            if ui
                .button(format!(
                    "\u{1F3A8} {} ({})",
                    state.i18n.t("skin.apply_colors"),
                    cur.as_str()
                ))
                .clicked()
            {
                settings.scene.bg_color = cur.recommended_bg();
                settings.scene.bar_color = cur.recommended_bar();
            }
        });

        ui.add_space(10.0);
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(22, 24, 28))
            .stroke(Stroke::new(1.0, Color32::from_rgb(55, 60, 70)))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(state.i18n.t("skin.homage_note"))
                        .small()
                        .color(Color32::from_rgb(185, 190, 200)),
                );
            });
    }

    fn draw_skin_card(
        &self,
        ui: &mut Ui,
        settings: &mut WasabiSettings,
        state: &WasabiState,
        preset: SkinPreset,
    ) {
        let selected = settings.scene.skin == preset;
        let border_col = if selected {
            Color32::from_rgb(90, 175, 255)
        } else {
            Color32::from_rgb(60, 62, 68)
        };
        let fill_col = if selected {
            Color32::from_rgb(30, 38, 50)
        } else {
            Color32::from_rgb(24, 25, 29)
        };

        let frame_resp = egui::Frame::group(ui.style())
            .fill(fill_col)
            .stroke(Stroke::new(if selected { 2.0 } else { 1.0 }, border_col))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(preset.as_str())
                            .strong()
                            .size(16.0)
                            .color(if selected {
                                Color32::from_rgb(140, 205, 255)
                            } else {
                                Color32::WHITE
                            }),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(preset.homage_target())
                                .small()
                                .color(Color32::from_rgb(155, 160, 172)),
                        );
                    });
                });

                ui.add_space(6.0);
                Self::paint_skin_preview(ui, preset);
                ui.add_space(6.0);

                ui.label(
                    RichText::new(state.i18n.t(preset.i18n_desc_key()))
                        .small()
                        .color(Color32::from_rgb(195, 198, 206)),
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    let btn_txt = if selected {
                        format!("\u{2714} {}", state.i18n.t("skin.selected"))
                    } else {
                        state.i18n.t("skin.select").to_owned()
                    };
                    if ui.selectable_label(selected, btn_txt).clicked() {
                        settings.scene.skin = preset;
                    }
                });
            });

        if frame_resp.response.interact(Sense::click()).clicked() {
            settings.scene.skin = preset;
        }
    }

    fn paint_skin_preview(ui: &mut Ui, preset: SkinPreset) {
        let preview_h = 74.0;
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width().max(120.0), preview_h),
            Sense::hover(),
        );
        let p = ui.painter_at(rect);
        let bg = preset.recommended_bg();
        p.rect_filled(rect, CornerRadius::same(4), bg);

        let kb_h = 18.0;
        let bar_h = 4.0;
        let notes_bottom = rect.bottom() - kb_h - bar_h;
        let w = rect.width();

        // Sample falling notes
        let samples = [
            (0.08f32, 0.17f32, 0.14f32, 0.72f32, Color32::from_rgb(70, 175, 255)),
            (0.24, 0.34, 0.36, 0.96, Color32::from_rgb(120, 225, 110)),
            (0.42, 0.52, 0.10, 0.62, Color32::from_rgb(255, 145, 75)),
            (0.60, 0.70, 0.28, 0.94, Color32::from_rgb(210, 115, 255)),
            (0.78, 0.89, 0.18, 0.78, Color32::from_rgb(255, 95, 130)),
        ];
        let notes_h = (notes_bottom - rect.top()).max(1.0);
        let radius = match preset {
            SkinPreset::Wynthesia => CornerRadius::same(5),
            SkinPreset::Zanith => CornerRadius::same(2),
            _ => CornerRadius::ZERO,
        };

        for (x0, x1, y0, y1, base_col) in samples {
            let nr = Rect::from_min_max(
                Pos2::new(rect.left() + x0 * w, rect.top() + y0 * notes_h),
                Pos2::new(rect.left() + x1 * w, rect.top() + y1 * notes_h),
            );
            p.rect_filled(nr, radius, base_col);

            match preset {
                SkinPreset::Wasaby => {
                    p.rect_stroke(
                        nr,
                        CornerRadius::ZERO,
                        Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 0, 0, 170)),
                        StrokeKind::Inside,
                    );
                }
                SkinPreset::Wynthesia => {
                    let sheen = Rect::from_min_max(
                        nr.min + Vec2::new(2.0, 2.0),
                        Pos2::new(nr.min.x + nr.width() * 0.36, nr.max.y - 2.0),
                    );
                    p.rect_filled(
                        sheen,
                        CornerRadius::same(3),
                        Color32::from_rgba_unmultiplied(255, 255, 255, 65),
                    );
                    p.rect_stroke(
                        nr,
                        radius,
                        Stroke::new(1.2, Color32::from_rgba_unmultiplied(10, 15, 25, 200)),
                        StrokeKind::Inside,
                    );
                }
                SkinPreset::WfaPlus => {
                    let hi = Rect::from_min_max(nr.min, Pos2::new(nr.max.x, nr.min.y + 2.5));
                    p.rect_filled(
                        hi,
                        CornerRadius::ZERO,
                        Color32::from_rgba_unmultiplied(255, 255, 255, 110),
                    );
                    p.rect_stroke(
                        nr,
                        CornerRadius::ZERO,
                        Stroke::new(1.2, Color32::from_rgb(15, 15, 15)),
                        StrokeKind::Inside,
                    );
                }
                SkinPreset::Zanith => {
                    let glow = Rect::from_min_max(
                        Pos2::new(nr.min.x, nr.max.y - 6.0),
                        nr.max,
                    );
                    p.rect_filled(
                        glow,
                        CornerRadius::same(2),
                        Color32::from_rgba_unmultiplied(255, 255, 255, 85),
                    );
                    p.rect_stroke(
                        nr,
                        radius,
                        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 70)),
                        StrokeKind::Inside,
                    );
                }
            }
        }

        // Hit bar
        let bar_rect = Rect::from_min_max(
            Pos2::new(rect.left(), notes_bottom),
            Pos2::new(rect.right(), notes_bottom + bar_h),
        );
        p.rect_filled(bar_rect, CornerRadius::ZERO, preset.recommended_bar());

        // Mini keyboard
        let key_cnt = 14usize;
        let kw = w / key_cnt as f32;
        let kb_top = notes_bottom + bar_h;
        for k in 0..key_cnt {
            let kr = Rect::from_min_max(
                Pos2::new(rect.left() + k as f32 * kw, kb_top),
                Pos2::new(rect.left() + (k + 1) as f32 * kw, rect.bottom()),
            );
            let white_col = match preset {
                SkinPreset::Zanith => Color32::from_rgb(225, 228, 236),
                SkinPreset::WfaPlus => Color32::from_rgb(245, 245, 245),
                _ => Color32::from_rgb(235, 235, 235),
            };
            p.rect_filled(kr, CornerRadius::ZERO, white_col);
            p.rect_stroke(
                kr,
                CornerRadius::ZERO,
                Stroke::new(0.8, Color32::from_rgb(45, 45, 48)),
                StrokeKind::Inside,
            );
        }
        for k in 0..key_cnt {
            let m = k % 7;
            if m == 0 || m == 1 || m == 3 || m == 4 || m == 5 {
                let cx = rect.left() + (k + 1) as f32 * kw;
                let bw = kw * 0.58;
                let br = Rect::from_min_max(
                    Pos2::new(cx - bw * 0.5, kb_top),
                    Pos2::new(cx + bw * 0.5, kb_top + kb_h * 0.62),
                );
                let blk_col = match preset {
                    SkinPreset::Zanith => Color32::from_rgb(18, 20, 28),
                    SkinPreset::Wynthesia => Color32::from_rgb(32, 35, 42),
                    _ => Color32::from_rgb(25, 25, 25),
                };
                p.rect_filled(br, CornerRadius::ZERO, blk_col);
            }
        }
    }
}
