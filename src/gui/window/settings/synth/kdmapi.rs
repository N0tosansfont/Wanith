use kdmapiv3::{
    MAX_BUFFER_FRAMES, MAX_RENDER_THREADS, MAX_RING_CAPACITY, MAX_VOICE_LIMIT, MIN_BUFFER_FRAMES,
    MIN_RENDER_THREADS, MIN_RING_CAPACITY, MIN_VOICE_LIMIT,
};

use crate::{
    settings::{KdmapiInterpolation, KdmapiSettings, WasabiSettings},
    state::WasabiState,
};

use super::SettingsWindow;

impl SettingsWindow {
    pub fn show_kdmapi_settings(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut WasabiSettings,
        state: &WasabiState,
        width: f32,
    ) {
        let host_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        let i18n = &state.i18n;
        egui::Grid::new("kdmapi_settings_grid")
            .num_columns(2)
            .spacing(super::super::SPACING)
            .striped(true)
            .min_col_width(width / 2.0)
            .show(ui, |ui| {
                // Voice Limit (Manual, SSoT 1..=1_000_000)
                ui.label(i18n.t("kdmapi.voice_limit"));
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut settings.synth.kdmapi.voice_limit)
                            .range(MIN_VOICE_LIMIT..=MAX_VOICE_LIMIT)
                            .speed(100),
                    );
                    for preset in [1_000u32, 10_000, 100_000] {
                        if ui.small_button(format!("{preset}")).clicked() {
                            settings.synth.kdmapi.voice_limit = preset;
                        }
                    }
                });
                ui.end_row();

                // Interpolation Mode
                ui.label(i18n.t("kdmapi.interpolation"));
                egui::ComboBox::from_id_salt("kdmapi_interpolation_select")
                    .selected_text(settings.synth.kdmapi.interpolation.as_str())
                    .show_ui(ui, |ui| {
                        for mode in KdmapiInterpolation::ALL {
                            ui.selectable_value(
                                &mut settings.synth.kdmapi.interpolation,
                                mode,
                                mode.as_str(),
                            );
                        }
                    });
                ui.end_row();

                // Buffer Frames (Manual, NO forced snapping)
                ui.label(i18n.t("kdmapi.buffer_frames"));
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut settings.synth.kdmapi.buffer_frames)
                            .range(MIN_BUFFER_FRAMES..=MAX_BUFFER_FRAMES)
                            .speed(16),
                    );
                    for preset in [128u32, 256, 512, 1024, 2048] {
                        if ui.small_button(format!("{preset}")).clicked() {
                            settings.synth.kdmapi.buffer_frames = preset;
                        }
                    }
                });
                ui.end_row();

                // Render Threads (Manual, SSoT 1..=256)
                ui.label(format!(
                    "{} ({}: {host_cores}):",
                    i18n.t("kdmapi.render_threads"),
                    i18n.t("kdmapi.cpu_cores")
                ));
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut settings.synth.kdmapi.render_threads)
                            .range(MIN_RENDER_THREADS..=MAX_RENDER_THREADS)
                            .speed(1),
                    );
                    for preset in [1u32, 4, 8, host_cores.clamp(MIN_RENDER_THREADS, MAX_RENDER_THREADS)] {
                        if ui.small_button(format!("{preset}")).clicked() {
                            settings.synth.kdmapi.render_threads = preset;
                        }
                    }
                });
                ui.end_row();

                if settings.synth.kdmapi.render_threads > 12 {
                    ui.label("");
                    ui.label(
                        egui::RichText::new(format!(
                            "\u{26A0} {}",
                            i18n.t("kdmapi.render_threads_warn")
                        ))
                        .small()
                        .color(egui::Color32::from_rgb(255, 196, 80)),
                    );
                    ui.end_row();
                }

                // Ring Buffer Capacity (Manual, power-of-two recommended)
                ui.label(i18n.t("kdmapi.ring_capacity"));
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut settings.synth.kdmapi.ring_buffer_capacity)
                            .range(MIN_RING_CAPACITY..=MAX_RING_CAPACITY)
                            .speed(1024),
                    );
                    for (label, preset) in [("256K", 262_144usize), ("1M", 1_048_576), ("4M", 4_194_304)] {
                        if ui.small_button(label).clicked() {
                            settings.synth.kdmapi.ring_buffer_capacity = preset;
                        }
                    }
                });
                ui.end_row();

                // Disable All Effects (Zero Overhead Bypass master switch)
                ui.label(i18n.t("kdmapi.disable_all_effects"));
                ui.checkbox(
                    &mut settings.synth.kdmapi.disable_all_effects,
                    i18n.t("kdmapi.disable_all_effects_desc"),
                );
                ui.end_row();

                // Individual DSP effect switches (disabled when master bypass is active)
                let effects_enabled = !settings.synth.kdmapi.disable_all_effects;

                ui.label(i18n.t("kdmapi.lowpass_filter"));
                ui.add_enabled(
                    effects_enabled,
                    egui::Checkbox::new(&mut settings.synth.kdmapi.enable_lowpass_filter, ""),
                );
                ui.end_row();

                ui.label(i18n.t("kdmapi.vibrato_lfo"));
                ui.add_enabled(
                    effects_enabled,
                    egui::Checkbox::new(&mut settings.synth.kdmapi.enable_vibrato_lfo, ""),
                );
                ui.end_row();
            });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            if ui.button(i18n.t("kdmapi.apply_reload")).clicked() {
                state.synth.switch(
                    &settings.synth,
                    state.loading_status.clone(),
                    state.errors.clone(),
                );
            }
            if ui.button(i18n.t("kdmapi.reset_default")).clicked() {
                settings.synth.kdmapi = KdmapiSettings::default();
            }
        });
    }
}
