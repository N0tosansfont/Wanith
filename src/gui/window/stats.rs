use std::{collections::VecDeque, time::Instant};

use egui::{Color32, Context, Frame, Pos2};
use numfmt::{Formatter, Precision};

use crate::{
    gui::window::GuiWasabiWindow,
    i18n::I18n,
    midi::{MIDIFileBase, MIDIFileStats},
    settings::{Statistics, WasabiSettings},
    system_monitor::SystemMetricsSnapshot,
    utils::convert_seconds_to_time_string,
};

pub struct GuiMidiStats {
    time_passed: f64,
    time_total: f64,
    notes_on_screen: u64,
    polyphony: Option<u64>,
    voice_count: Option<u64>,
    xrun_count: Option<u64>,
    audio_dsp_pct: Option<f32>,
    system: SystemMetricsSnapshot,
}

impl GuiMidiStats {
    pub fn empty() -> GuiMidiStats {
        GuiMidiStats {
            time_passed: 0.0,
            time_total: 0.0,
            notes_on_screen: 0,
            polyphony: None,
            voice_count: None,
            xrun_count: None,
            audio_dsp_pct: None,
            system: SystemMetricsSnapshot::default(),
        }
    }

    pub fn set_voice_count(&mut self, voices: Option<u64>) {
        self.voice_count = voices;
    }

    pub fn set_xrun_count(&mut self, xruns: Option<u64>) {
        self.xrun_count = xruns;
    }

    pub fn set_audio_dsp_pct(&mut self, pct: Option<f32>) {
        self.audio_dsp_pct = pct;
    }

    pub fn set_system_metrics(&mut self, system: SystemMetricsSnapshot) {
        self.system = system;
    }

    pub fn set_rendered_note_count(&mut self, notes: u64) {
        self.notes_on_screen = notes;
    }

    pub fn set_polyphony(&mut self, polyphony: Option<u64>) {
        self.polyphony = polyphony;
    }
}

fn format_ram_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GB", b / GIB)
    } else {
        format!("{:.0} MB", b / MIB)
    }
}

fn format_total_ram_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GB", b / GIB)
    } else {
        format!("{:.0} MB", b / MIB)
    }
}

impl GuiWasabiWindow {
    pub fn draw_stats(
        &mut self,
        ctx: &Context,
        pos: Pos2,
        mut stats: GuiMidiStats,
        settings: &WasabiSettings,
        i18n: &I18n,
    ) {
        // Prepare semi-transparent HUD frame based on settings
        let opacity = settings.scene.statistics.opacity.clamp(0.0, 1.0);
        let alpha = (u8::MAX as f32 * opacity).round() as u8;

        let round = 8;

        let mut stats_frame = Frame::default()
            .inner_margin(egui::Margin::same(8))
            .fill(Color32::from_black_alpha(alpha));

        if settings.scene.statistics.floating {
            stats_frame = stats_frame.corner_radius(egui::CornerRadius::same(round));
        } else {
            stats_frame = stats_frame.corner_radius(egui::CornerRadius {
                ne: 0,
                nw: 0,
                sw: 0,
                se: round,
            });
        }

        if settings.scene.statistics.border {
            stats_frame =
                stats_frame.stroke(egui::Stroke::new(1.0, Color32::from_rgb(55, 55, 55)));
        }

        // Render lock-free Legit Run HUD telemetry window
        egui::Window::new("Stats")
            .resizable(false)
            .collapsible(false)
            .title_bar(false)
            .scroll([false, false])
            .interactable(false)
            .frame(stats_frame)
            .fixed_pos(pos)
            .min_width(248.0)
            .show(ctx, |ui| {
                ui.set_min_width(248.0);
                ui.spacing_mut().interact_size.y = 16.0;

                let mut f = Formatter::new()
                    .separator(',')
                    .unwrap()
                    .precision(Precision::Decimals(0));

                let mut note_stats = MIDIFileStats::default();
                if let Some(midi_file) = self.midi_file.as_mut() {
                    stats.time_total = midi_file.midi_length().unwrap_or(0.0);
                    let time = midi_file.timer().get_time().as_seconds_f64();

                    if time > stats.time_total {
                        stats.time_passed = stats.time_total;
                    } else {
                        stats.time_passed = time;
                    }

                    note_stats = midi_file.stats();
                }

                if let Some(v) = stats.voice_count {
                    self.peak_voices = self.peak_voices.max(v);
                }

                for i in settings.scene.statistics.order.iter().filter(|i| i.1) {
                    match i.0 {
                        Statistics::Xrun => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.xrun"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| match stats.xrun_count {
                                        Some(0) => {
                                            ui.colored_label(
                                                Color32::from_rgb(95, 225, 125),
                                                egui::RichText::new("0").monospace(),
                                            );
                                        }
                                        Some(n) => {
                                            ui.colored_label(
                                                Color32::from_rgb(245, 95, 95),
                                                egui::RichText::new(f.fmt2(n).to_string())
                                                    .monospace(),
                                            );
                                        }
                                        None => {
                                            ui.monospace("N/A");
                                        }
                                    },
                                );
                            });
                        }
                        Statistics::VoiceCount => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.voice"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| match stats.voice_count {
                                        Some(v) => {
                                            let cur = f.fmt2(v).to_string();
                                            let peak = f.fmt2(self.peak_voices).to_string();
                                            ui.monospace(format!("{cur} / {peak}"));
                                        }
                                        None => {
                                            ui.monospace("N/A");
                                        }
                                    },
                                );
                            });
                        }
                        Statistics::Nps => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.nps"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        self.nps.tick(note_stats.passed_notes.unwrap_or(0) as i64);
                                        let cur = f.fmt2(self.nps.read()).to_string();
                                        let peak = f.fmt2(self.nps.peak()).to_string();
                                        ui.monospace(format!("{cur} / {peak}"));
                                    },
                                );
                            });
                        }
                        Statistics::Fps => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.fps"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.monospace(f.fmt2(self.fps.get_fps()).to_string());
                                    },
                                );
                            });
                        }
                        Statistics::Cpu => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.cpu"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let label = match (
                                            stats.system.process_cpu_pct,
                                            stats.audio_dsp_pct,
                                        ) {
                                            (Some(proc_cpu), Some(dsp)) => {
                                                format!("{proc_cpu:.1}% (DSP {dsp:.1}%)")
                                            }
                                            (Some(proc_cpu), None) => format!("{proc_cpu:.1}%"),
                                            (None, Some(dsp)) => format!("DSP {dsp:.1}%"),
                                            (None, None) => "N/A".to_string(),
                                        };
                                        ui.monospace(label);
                                    },
                                );
                            });
                        }
                        Statistics::Ram => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.ram"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| match (
                                        stats.system.ram_rss_bytes,
                                        stats.system.ram_total_bytes,
                                    ) {
                                        (Some(rss), Some(total)) => {
                                            ui.monospace(format!(
                                                "{} / {}",
                                                format_ram_bytes(rss),
                                                format_total_ram_bytes(total)
                                            ));
                                        }
                                        (Some(rss), None) => {
                                            ui.monospace(format_ram_bytes(rss));
                                        }
                                        (None, _) => {
                                            ui.monospace("N/A");
                                        }
                                    },
                                );
                            });
                        }
                        Statistics::Gpu => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.gpu"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| match stats.system.gpu_util_pct {
                                        Some(pct) => {
                                            ui.monospace(format!("{pct}%"));
                                        }
                                        None => {
                                            ui.monospace("N/A");
                                        }
                                    },
                                );
                            });
                        }
                        Statistics::Time => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.time"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.monospace(format!(
                                            "{} / {}",
                                            convert_seconds_to_time_string(stats.time_passed),
                                            convert_seconds_to_time_string(stats.time_total)
                                        ));
                                    },
                                );
                            });
                        }
                        Statistics::Rendered => {
                            ui.horizontal(|ui| {
                                ui.monospace(i18n.t("hud.rendered"));
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.monospace(f.fmt2(stats.notes_on_screen).to_string());
                                    },
                                );
                            });
                        }
                        Statistics::NoteCount => {
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                ui.monospace(format!(
                                    "{} / {}",
                                    note_stats
                                        .passed_notes
                                        .map(|n| f.fmt2(n).to_string())
                                        .unwrap_or_else(|| "-".to_string()),
                                    note_stats
                                        .total_notes
                                        .map(|n| f.fmt2(n).to_string())
                                        .unwrap_or_else(|| "-".to_string())
                                ));
                            });
                        }
                        Statistics::Polyphony => {
                            if let Some(poly) = stats.polyphony {
                                ui.horizontal(|ui| {
                                    ui.monospace(i18n.t("hud.polyphony"));
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.monospace(f.fmt2(poly).to_string());
                                        },
                                    );
                                });
                            }
                        }
                    };
                }
            });
    }
}

#[derive(Default)]
pub struct NpsCounter {
    ticks: VecDeque<(Instant, i64)>,
    peak: u64,
}

impl NpsCounter {
    const NPS_WINDOW: f64 = 0.5;

    pub fn reset(&mut self) {
        self.ticks.clear();
        self.peak = 0;
    }

    pub fn clear_window(&mut self) {
        self.ticks.clear();
    }

    pub fn tick(&mut self, passed: i64) {
        if self.ticks.back().is_some_and(|&(_, prev)| passed < prev) {
            self.ticks.clear();
        }
        self.ticks.push_back((Instant::now(), passed));
        while let Some((front_time, _passed)) = self.ticks.front() {
            if front_time.elapsed().as_secs_f64() > Self::NPS_WINDOW {
                self.ticks.pop_front();
            } else {
                break;
            }
        }
        let cur = self.read();
        if cur > self.peak {
            self.peak = cur;
        }
    }

    pub fn read(&self) -> u64 {
        let old = if let Some((_time, front_passed)) = self.ticks.front() {
            *front_passed as f64
        } else {
            0.0
        };

        let last = if let Some((_time, back_passed)) = self.ticks.back() {
            *back_passed as f64
        } else {
            0.0
        };

        ((last - old).max(0.0) / Self::NPS_WINDOW).round() as u64
    }

    pub fn peak(&self) -> u64 {
        self.peak
    }
}
