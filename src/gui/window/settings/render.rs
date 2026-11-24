use std::path::{Path, PathBuf};
use std::thread;

use egui::Color32;
use tokio::sync::oneshot;

use crate::{
    engine_core::collect_enabled_soundfont_paths,
    offline_render::{AudioRenderRequest, RenderJobPhase, VideoRenderRequest},
    settings::{
        RenderSettings, RenderTarget, VideoCodec, VideoRateControl, WasabiSettings,
        WavSampleFormat, MAX_RENDER_DIMENSION, MAX_RENDER_FPS, MAX_RENDER_SAMPLE_RATE,
        MAX_VIDEO_BITRATE_KBPS, MAX_VIDEO_QP, MIN_RENDER_DIMENSION, MIN_RENDER_FPS,
        MIN_RENDER_SAMPLE_RATE, MIN_VIDEO_BITRATE_KBPS, MIN_VIDEO_QP,
    },
    state::WasabiState,
};

use super::SettingsWindow;

impl SettingsWindow {
    fn open_render_output_dialog(&mut self, target: RenderTarget, current_path: &Path) {
        let (tx, rx) = oneshot::channel();
        self.render_output_picker = Some(rx);

        let initial_dir = current_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("./"));
        let default_name = match target {
            RenderTarget::AudioOnly => "output.wav",
            RenderTarget::VideoAndAudio => "output.mp4",
        };

        thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new()
                .set_title("Choose Render Output Path...")
                .set_directory(initial_dir)
                .set_file_name(default_name);
            dialog = match target {
                RenderTarget::AudioOnly => dialog.add_filter("WAV Audio", &["wav"]),
                RenderTarget::VideoAndAudio => dialog.add_filter("MP4 Video", &["mp4"]),
            };
            if let Some(path) = dialog.save_file() {
                let _ = tx.send(path);
            }
        });
    }

    pub fn show_render_settings(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut WasabiSettings,
        state: &mut WasabiState,
        width: f32,
    ) {
        // Poll async save-file picker result
        if let Some(rx) = self.render_output_picker.as_mut() {
            if let Ok(picked) = rx.try_recv() {
                settings.render.output_path = picked;
                self.render_output_picker = None;
            }
        }

        let snap = state.offline_render.snapshot();
        let is_running = snap.phase == RenderJobPhase::Running;

        // Keep progress bar & realtime speed live while background worker is running
        if is_running {
            ui.ctx().request_repaint();
        }

        // --- Source & Engine Summary ---
        ui.heading(state.i18n.t("render.heading"));
        ui.add_space(4.0);

        let enabled_sfs = collect_enabled_soundfont_paths(&settings.synth.soundfonts);
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.strong(state.i18n.t("render.loaded_midi"));
                if let Some(ref midi_path) = state.active_midi_path {
                    let name = midi_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("<unknown>");
                    let dur_label = match state.active_midi_length_sec {
                        Some(sec) => format!(" ({:.1}s)", sec),
                        None => String::new(),
                    };
                    ui.monospace(format!("{name}{dur_label}"));
                } else {
                    ui.colored_label(
                        Color32::from_rgb(235, 185, 80),
                        state.i18n.t("render.no_midi"),
                    );
                }
            });

            ui.horizontal(|ui| {
                ui.strong(state.i18n.t("render.audio_core"));
                let fx = if settings.synth.kdmapi.disable_all_effects {
                    state.i18n.t("render.effects_off")
                } else {
                    state.i18n.t("render.effects_on")
                };
                ui.monospace(format!(
                    "{} voices | {} | {} threads | {} | {} SF",
                    settings.synth.kdmapi.voice_limit,
                    settings.synth.kdmapi.interpolation.as_str(),
                    settings.synth.kdmapi.render_threads,
                    fx,
                    enabled_sfs.len()
                ));
            });

            ui.horizontal(|ui| {
                ui.strong(state.i18n.t("render.gpu_encode"));
                let caps = &state.gpu_video_caps;
                if caps.is_video_encode_ready() {
                    let h264 = if caps.supports_h264 { "H.264" } else { "-" };
                    let h265 = if caps.supports_h265 { "H.265" } else { "-" };
                    let qfam = caps
                        .encode_queue_family_index
                        .map(|q| format!("queue family #{q}"))
                        .unwrap_or_default();
                    ui.colored_label(
                        Color32::from_rgb(95, 220, 120),
                        format!(
                            "{} — {} (VK_KHR_video_encode_queue: {h264} / {h265}, {qfam})",
                            state.i18n.t("render.gpu_supported"),
                            caps.device_name
                        ),
                    );
                } else {
                    let gpu_label = if caps.device_name.is_empty() {
                        "Unknown GPU"
                    } else {
                        &caps.device_name
                    };
                    ui.colored_label(
                        Color32::from_rgb(235, 110, 100),
                        format!(
                            "{} — {gpu_label} (VK_KHR_video_encode_queue not detected)",
                            state.i18n.t("render.gpu_unavailable")
                        ),
                    );
                }
            });
        });

        ui.add_space(10.0);

        // --- Target & Output ---
        ui.add_enabled_ui(!is_running, |ui| {
            egui::Grid::new("render_target_grid")
                .num_columns(2)
                .spacing(super::SPACING)
                .striped(true)
                .min_col_width(width / 2.0)
                .show(ui, |ui| {
                    let prev_target = settings.render.target;
                    ui.label(state.i18n.t("render.target"));
                    egui::ComboBox::from_id_salt("render_target_select")
                        .selected_text(settings.render.target.as_str())
                        .show_ui(ui, |ui| {
                            for target in RenderTarget::ALL {
                                ui.selectable_value(
                                    &mut settings.render.target,
                                    target,
                                    target.as_str(),
                                );
                            }
                        });
                    ui.end_row();

                    if prev_target != settings.render.target {
                        let new_ext = match settings.render.target {
                            RenderTarget::AudioOnly => "wav",
                            RenderTarget::VideoAndAudio => "mp4",
                        };
                        if settings.render.output_path.as_os_str().is_empty() {
                            settings.render.output_path = PathBuf::from(format!("output.{new_ext}"));
                        } else {
                            settings.render.output_path.set_extension(new_ext);
                        }
                    }

                    ui.label(state.i18n.t("render.output_path"));
                    ui.horizontal(|ui| {
                        let mut path_str = settings
                            .render
                            .output_path
                            .to_string_lossy()
                            .into_owned();
                        let edit = ui.add(
                            egui::TextEdit::singleline(&mut path_str)
                                .desired_width((width / 2.0 - 95.0).max(160.0)),
                        );
                        if edit.changed() {
                            settings.render.output_path = PathBuf::from(path_str);
                        }
                        if ui.button(state.i18n.t("render.browse")).clicked() {
                            self.open_render_output_dialog(
                                settings.render.target,
                                &settings.render.output_path,
                            );
                        }
                    });
                    ui.end_row();
                });

            ui.add_space(super::CATEG_SPACE * 0.6);
            ui.heading(state.i18n.t("render.audio_heading"));

            egui::Grid::new("render_audio_grid")
                .num_columns(2)
                .spacing(super::SPACING)
                .striped(true)
                .min_col_width(width / 2.0)
                .show(ui, |ui| {
                    ui.label(state.i18n.t("render.sample_rate"));
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut settings.render.sample_rate)
                                .range(MIN_RENDER_SAMPLE_RATE..=MAX_RENDER_SAMPLE_RATE)
                                .speed(100),
                        );
                        for preset in [44_100u32, 48_000, 96_000, 192_000] {
                            if ui.small_button(format!("{preset}")).clicked() {
                                settings.render.sample_rate = preset;
                            }
                        }
                    });
                    ui.end_row();

                    ui.label(state.i18n.t("render.wav_format"));
                    egui::ComboBox::from_id_salt("render_wav_format_select")
                        .selected_text(settings.render.wav_format.as_str())
                        .show_ui(ui, |ui| {
                            for fmt in WavSampleFormat::ALL {
                                ui.selectable_value(
                                    &mut settings.render.wav_format,
                                    fmt,
                                    fmt.as_str(),
                                );
                            }
                        });
                    ui.end_row();
                });

            ui.add_space(super::CATEG_SPACE * 0.6);
            ui.heading(state.i18n.t("render.video_heading"));

            egui::Grid::new("render_video_grid")
                .num_columns(2)
                .spacing(super::SPACING)
                .striped(true)
                .min_col_width(width / 2.0)
                .show(ui, |ui| {
                    ui.label(state.i18n.t("render.resolution"));
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut settings.render.width)
                                .range(MIN_RENDER_DIMENSION..=MAX_RENDER_DIMENSION)
                                .speed(16),
                        );
                        ui.label("x");
                        ui.add(
                            egui::DragValue::new(&mut settings.render.height)
                                .range(MIN_RENDER_DIMENSION..=MAX_RENDER_DIMENSION)
                                .speed(16),
                        );
                        for (label, w, h) in [
                            ("1080p", 1920u32, 1080u32),
                            ("1440p", 2560, 1440),
                            ("4K", 3840, 2160),
                        ] {
                            if ui.small_button(label).clicked() {
                                settings.render.width = w;
                                settings.render.height = h;
                            }
                        }
                    });
                    ui.end_row();

                    ui.label(state.i18n.t("render.fps"));
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut settings.render.fps)
                                .range(MIN_RENDER_FPS..=MAX_RENDER_FPS)
                                .speed(1),
                        );
                        for preset in [60u32, 120, 240] {
                            if ui.small_button(format!("{preset}")).clicked() {
                                settings.render.fps = preset;
                            }
                        }
                    });
                    ui.end_row();

                    ui.label(state.i18n.t("render.video_codec"));
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("render_video_codec_select")
                            .selected_text(settings.render.video_codec.as_str())
                            .show_ui(ui, |ui| {
                                for codec in VideoCodec::ALL {
                                    ui.selectable_value(
                                        &mut settings.render.video_codec,
                                        codec,
                                        codec.as_str(),
                                    );
                                }
                            });
                        if state.gpu_video_caps.supports_codec(settings.render.video_codec) {
                            ui.colored_label(
                                Color32::from_rgb(95, 220, 120),
                                state.i18n.t("render.hw_ready"),
                            );
                        } else {
                            ui.colored_label(
                                Color32::from_rgb(235, 110, 100),
                                state.i18n.t("render.unsupported"),
                            );
                        }
                    });
                    ui.end_row();

                    ui.label(state.i18n.t("render.rate_control"));
                    egui::ComboBox::from_id_salt("render_rate_control_select")
                        .selected_text(settings.render.rate_control.as_str())
                        .show_ui(ui, |ui| {
                            for rc in VideoRateControl::ALL {
                                ui.selectable_value(
                                    &mut settings.render.rate_control,
                                    rc,
                                    rc.as_str(),
                                );
                            }
                        });
                    ui.end_row();

                    match settings.render.rate_control {
                        VideoRateControl::Bitrate => {
                            ui.label(state.i18n.t("render.target_bitrate"));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut settings.render.bitrate_kbps)
                                        .range(MIN_VIDEO_BITRATE_KBPS..=MAX_VIDEO_BITRATE_KBPS)
                                        .speed(500),
                                );
                                for (label, kbps) in
                                    [("15 Mbps", 15_000u32), ("25 Mbps", 25_000), ("60 Mbps", 60_000)]
                                {
                                    if ui.small_button(label).clicked() {
                                        settings.render.bitrate_kbps = kbps;
                                    }
                                }
                            });
                            ui.end_row();
                        }
                        VideoRateControl::ConstantQp => {
                            ui.label(state.i18n.t("render.constant_qp"));
                            ui.add(
                                egui::DragValue::new(&mut settings.render.qp)
                                    .range(MIN_VIDEO_QP..=MAX_VIDEO_QP)
                                    .speed(1),
                            );
                            ui.end_row();
                        }
                    }

                    ui.label(state.i18n.t("render.mp4_audio_label"));
                    ui.small(state.i18n.t("render.mp4_audio_desc"));
                    ui.end_row();
                });
        });

        ui.add_space(14.0);

        // --- Execution Controls & Live Progress ---
        ui.horizontal(|ui| {
            let can_start = !is_running && state.active_midi_path.is_some();
            if ui
                .add_enabled(can_start, egui::Button::new(state.i18n.t("render.start")))
                .clicked()
            {
                match settings.render.target {
                    RenderTarget::AudioOnly => {
                        if let Some(midi_path) = state.active_midi_path.clone() {
                            let req = AudioRenderRequest {
                                midi_path,
                                output_path: settings.render.output_path.clone(),
                                sample_rate: settings.render.sample_rate,
                                wav_format: settings.render.wav_format,
                                kdmapi_settings: settings.synth.kdmapi.clone(),
                                soundfonts: enabled_sfs.clone(),
                                estimated_duration_sec: state.active_midi_length_sec,
                            };
                            if let Err(err) = state.offline_render.start_audio_render(req) {
                                state.offline_render.report_unavailable(err);
                            }
                        }
                    }
                    RenderTarget::VideoAndAudio => {
                        if !state
                            .gpu_video_caps
                            .supports_codec(settings.render.video_codec)
                        {
                            state.offline_render.report_unavailable(format!(
                                "Selected GPU ({}) does not support hardware {}.",
                                state.gpu_video_caps.device_name,
                                settings.render.video_codec.as_str(),
                            ));
                        } else if let Some(midi_path) = state.active_midi_path.clone() {
                            let req = VideoRenderRequest {
                                midi_path,
                                render_settings: settings.render.clone(),
                                midi_settings: settings.midi.clone(),
                                scene_settings: settings.scene.clone(),
                                kdmapi_settings: settings.synth.kdmapi.clone(),
                                soundfonts: enabled_sfs.clone(),
                            };
                            if let Err(err) = state.offline_render.start_video_render(req) {
                                state.offline_render.report_unavailable(err);
                            }
                        }
                    }
                }
            }

            if ui
                .add_enabled(is_running, egui::Button::new(state.i18n.t("render.cancel")))
                .clicked()
            {
                state.offline_render.cancel();
            }

            if ui
                .add_enabled(
                    !is_running,
                    egui::Button::new(state.i18n.t("render.reset_defaults")),
                )
                .clicked()
            {
                settings.render = RenderSettings::default();
            }
        });

        ui.add_space(8.0);

        if is_running {
            let frac = snap.progress_fraction().unwrap_or(0.0);
            let bar_text = if snap.progress_fraction().is_some() {
                format!(
                    "{:.1}%  |  {:.1}x realtime  |  {:.1}s audio ({:.1}s elapsed)  |  {} voices",
                    frac * 100.0,
                    snap.realtime_multiplier,
                    snap.rendered_audio_secs(),
                    snap.elapsed_secs,
                    snap.active_voices,
                )
            } else {
                format!(
                    "{:.1}s audio ({:.1}s elapsed)  |  {:.1}x realtime  |  {} voices",
                    snap.rendered_audio_secs(),
                    snap.elapsed_secs,
                    snap.realtime_multiplier,
                    snap.active_voices,
                )
            };
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(bar_text)
                    .animate(true),
            );
        } else {
            match snap.phase {
                RenderJobPhase::Idle => {}
                RenderJobPhase::Completed => {
                    ui.colored_label(Color32::from_rgb(95, 220, 120), &snap.message);
                }
                RenderJobPhase::Cancelled => {
                    ui.colored_label(Color32::from_rgb(235, 185, 80), &snap.message);
                }
                RenderJobPhase::Failed => {
                    ui.colored_label(Color32::from_rgb(235, 110, 100), &snap.message);
                }
                RenderJobPhase::Running => {}
            }
        }
    }
}
