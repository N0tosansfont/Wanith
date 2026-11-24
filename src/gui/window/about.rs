use std::env::consts::{ARCH, OS};

use egui::{Color32, RichText};

use crate::{state::WasabiState, utils};

use super::{GuiWasabiWindow, WasabiError};

/// Verbatim dedication statement for Black MIDI engine pioneers.
const DEDICATION_QUOTE: &str = "We deeply respect and love the creators of the legendary engines and pioneers who gifted the world the playground of Black MIDI. Thank you.";

/// Verbatim Special Thanks roster.
const SPECIAL_THANKS: [&str; 8] = [
    "Synthesia — Nicholas Piegdon & Contributors",
    "Piano From Above (PFA) — Brian Pantano & Contributors",
    "Zenith — Arduano (Dorian) & Contributors",
    "Wasabi — BlackMIDIDevs & Contributors",
    "XSynth — BlackMIDIDevs (Arduano, etc.) & Contributors",
    "KDMAPI / OmniMIDI — Keppy (KaleidonKep99) & Contributors (Real Legend of Legend)",
    "BusiedGem — Synthesia Black MIDI Creator",
    "Carlos S.M. — Famous PFA Black MIDI Creator",
];

impl GuiWasabiWindow {
    pub fn show_about(&mut self, ctx: &egui::Context, state: &mut WasabiState) {
        let frame = utils::create_window_frame(ctx);
        let size = [660.0, 560.0];

        let mut updcheck = false;
        let window_title = state.i18n.t("about.window_title").to_owned();

        egui::Window::new(window_title)
            .id(egui::Id::new("wanith_about_credits_window"))
            .resizable(true)
            .collapsible(false)
            .title_bar(true)
            .scroll([false, true])
            .enabled(true)
            .frame(frame)
            .default_size(size)
            .min_size([560.0, 440.0])
            .open(&mut state.show_about)
            .show(ctx, |ui| {
                let i18n = &state.i18n;

                // Header / Branding
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let image_size = 72.0;
                    ui.add(
                        egui::Image::new(egui::include_image!("../../../assets/logo.svg"))
                            .fit_to_exact_size(egui::Vec2::new(image_size, image_size)),
                    );
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Wanith")
                                .size(34.0)
                                .strong()
                                .color(Color32::WHITE),
                        );
                        ui.label(
                            RichText::new(i18n.t("about.subtitle"))
                                .color(Color32::from_rgb(180, 180, 180)),
                        );
                    });
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Dedication Quote (verbatim)
                ui.heading(i18n.t("about.dedication_heading"));
                ui.add_space(4.0);
                egui::Frame::default()
                    .inner_margin(egui::Margin::same(10))
                    .corner_radius(egui::CornerRadius::same(6))
                    .fill(Color32::from_rgb(28, 28, 32))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(65, 65, 78)))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("\"{DEDICATION_QUOTE}\""))
                                .italics()
                                .color(Color32::from_rgb(235, 235, 240)),
                        );
                    });

                ui.add_space(12.0);

                // Special Thanks (verbatim 8 entries)
                ui.heading(i18n.t("about.special_thanks_heading"));
                ui.add_space(4.0);
                egui::Frame::default()
                    .inner_margin(egui::Margin::same(10))
                    .corner_radius(egui::CornerRadius::same(6))
                    .fill(Color32::from_rgb(24, 24, 26))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(52, 52, 58)))
                    .show(ui, |ui| {
                        for entry in SPECIAL_THANKS {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new("•").color(Color32::from_rgb(120, 200, 255)),
                                );
                                ui.label(entry);
                            });
                        }
                    });

                ui.add_space(12.0);

                // Build & Engine Info
                let col_width = (ui.available_width() / 2.0).max(180.0);
                ui.heading(i18n.t("about.build_heading"));
                ui.add_space(4.0);
                egui::Grid::new("buildinfo_grid")
                    .num_columns(2)
                    .min_col_width(col_width)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(i18n.t("about.version"));
                        ui.monospace(env!("CARGO_PKG_VERSION"));
                        ui.end_row();

                        ui.label(i18n.t("about.os"));
                        ui.monospace(OS);
                        ui.end_row();

                        ui.label(i18n.t("about.arch"));
                        ui.monospace(format!("{ARCH} (64-bit)"));
                        ui.end_row();

                        ui.label(i18n.t("about.audio_engine"));
                        ui.monospace("KDMAPI v3 (In-Process Lock-Free)");
                        ui.end_row();

                        ui.label(i18n.t("about.video_engine"));
                        ui.monospace("Native Vulkan Video (H.264/H.265) + AAC MP4");
                        ui.end_row();
                    });

                ui.add_space(12.0);

                // GNU GPLv3 Appropriate Legal Notices
                ui.heading(i18n.t("about.license_heading"));
                ui.add_space(4.0);
                egui::Frame::default()
                    .inner_margin(egui::Margin::same(10))
                    .corner_radius(egui::CornerRadius::same(6))
                    .fill(Color32::from_rgb(22, 22, 24))
                    .stroke(egui::Stroke::new(1.0, Color32::from_rgb(55, 55, 55)))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(i18n.t("about.license_body"))
                                .small()
                                .color(Color32::from_rgb(195, 195, 200)),
                        );
                    });

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(format!("\u{1F310} {}", i18n.t("about.github")))
                        .clicked()
                    {
                        open::that("https://github.com/N0tosansfont/Wanith").unwrap_or_else(|e| {
                            state.errors.error(&WasabiError::Other(e.to_string()))
                        });
                    }
                    if ui
                        .button(format!("\u{1F310} {}", i18n.t("about.check_updates")))
                        .clicked()
                    {
                        updcheck = true;
                    }
                });
            });

        if updcheck {
            crate::utils::check_for_updates(state);
        }
    }
}
