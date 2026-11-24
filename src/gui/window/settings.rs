use std::path::PathBuf;

use soundfonts::EguiSFList;
use tokio::sync::oneshot::Receiver;

use crate::{
    settings::{Colors, Synth, WasabiSettings},
    state::{SettingsTab, WasabiState},
    utils,
};

use super::WasabiError;

mod midi;
mod render;
mod skin;
mod soundfonts;
mod synth;
mod visual;

const CATEG_SPACE: f32 = 26.0;
const SPACING: [f32; 2] = [40.0, 12.0];

#[derive(Clone)]
struct FilePalette {
    pub path: PathBuf,
    pub selected: bool,
}

#[cfg(all(supported_os, not(target_os = "freebsd")))]
#[derive(Clone)]
struct MidiDevice {
    pub name: String,
    pub selected: bool,
}

pub struct SettingsWindow {
    palettes: Vec<FilePalette>,
    #[cfg(all(supported_os, not(target_os = "freebsd")))]
    midi_devices: Vec<MidiDevice>,
    sf_list: EguiSFList,
    render_output_picker: Option<Receiver<PathBuf>>,
}

impl SettingsWindow {
    pub fn new(settings: &WasabiSettings) -> Self {
        let mut sf_list = EguiSFList::new();
        for sf in settings.synth.soundfonts.iter() {
            sf_list.add_item(sf.clone(), false);
        }

        Self {
            palettes: Vec::new(),
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            midi_devices: Vec::new(),
            sf_list,
            render_output_picker: None,
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        settings: &mut WasabiSettings,
        state: &mut WasabiState,
    ) {
        let frame = utils::create_window_frame(ctx);
        let win = ctx.available_rect();

        let win_title = state.i18n.t("settings.title").to_owned();
        egui::Window::new(win_title)
            .id(egui::Id::new("wanith_settings_window"))
            .resizable(true)
            .collapsible(false)
            .title_bar(true)
            .enabled(true)
            .frame(frame)
            .default_size([win.width() * 0.7, win.height() * 0.7])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .min_size([700.0, 400.0])
            .show(ctx, |ui| {
                egui::TopBottomPanel::top("settings_tab_selector")
                    .resizable(false)
                    .show_inside(ui, |ui| {
                        ui.style_mut()
                            .text_styles
                            .get_mut(&egui::TextStyle::Button)
                            .unwrap()
                            .size = 16.0;

                        let tab_visual = format!("\u{1f4bb} {}", state.i18n.t("settings.tab.visual"));
                        let tab_skin = format!("\u{1f3a8} {}", state.i18n.t("settings.tab.skin"));
                        let tab_midi = format!("\u{1f3b5} {}", state.i18n.t("settings.tab.midi"));
                        let tab_synth = format!("\u{1f3b9} {}", state.i18n.t("settings.tab.synth"));
                        let tab_sf = format!("\u{1f50a} {}", state.i18n.t("settings.tab.soundfonts"));
                        let tab_render = format!("\u{1f3ac} {}", state.i18n.t("settings.tab.render"));

                        ui.columns(6, |columns| {
                            columns[0].vertical_centered_justified(|ui| {
                                ui.selectable_value(
                                    &mut state.settings_tab,
                                    SettingsTab::Visual,
                                    tab_visual,
                                )
                            });
                            columns[1].vertical_centered_justified(|ui| {
                                ui.selectable_value(
                                    &mut state.settings_tab,
                                    SettingsTab::Skin,
                                    tab_skin,
                                )
                            });
                            columns[2].vertical_centered_justified(|ui| {
                                ui.selectable_value(
                                    &mut state.settings_tab,
                                    SettingsTab::Midi,
                                    tab_midi,
                                )
                            });
                            columns[3].vertical_centered_justified(|ui| {
                                ui.selectable_value(
                                    &mut state.settings_tab,
                                    SettingsTab::Synth,
                                    tab_synth,
                                )
                            });

                            columns[4].vertical_centered_justified(|ui| {
                                ui.add_enabled_ui(
                                    settings.synth.synth == Synth::XSynth
                                        || settings.synth.synth == Synth::Kdmapi,
                                    |ui| {
                                        ui.selectable_value(
                                            &mut state.settings_tab,
                                            SettingsTab::SoundFonts,
                                            tab_sf,
                                        )
                                    },
                                )
                            });
                            columns[5].vertical_centered_justified(|ui| {
                                ui.selectable_value(
                                    &mut state.settings_tab,
                                    SettingsTab::Render,
                                    tab_render,
                                )
                            });
                        });
                        ui.add_space(8.0);
                    });

                egui::TopBottomPanel::bottom("settings_save_panel")
                    .resizable(false)
                    .show_inside(ui, |ui| {
                        ui.add_space(8.0);
                        let save_lbl = format!("\u{1F4BE} {}", state.i18n.t("settings.save"));
                        let close_lbl = format!("\u{2716} {}", state.i18n.t("settings.close"));
                        let about_lbl = format!("\u{2139} {}", state.i18n.t("settings.about"));
                        ui.columns(3, |columns| {
                            columns[0].with_layout(
                                egui::Layout::top_down(egui::Align::RIGHT),
                                |ui| {
                                    if ui.button(save_lbl).clicked() {
                                        settings
                                            .save_to_file()
                                            .unwrap_or_else(|e| state.errors.error(&e));
                                    }
                                },
                            );
                            columns[1].vertical_centered(|ui| {
                                if ui.button(about_lbl).clicked() {
                                    state.show_about = true;
                                }
                            });
                            columns[2].with_layout(
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    if ui.button(close_lbl).clicked() {
                                        state.show_settings = false;
                                    }
                                },
                            );
                        });
                    });

                let width = ui.available_width() - 40.0;
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().animated(true).show(ui, |ui| {
                        match state.settings_tab {
                            SettingsTab::Visual => {
                                self.show_visual_settings(ui, settings, state, width)
                            }
                            SettingsTab::Skin => {
                                self.show_skin_settings(ui, settings, state, width)
                            }
                            SettingsTab::Midi => {
                                self.show_midi_settings(ui, settings, state, width)
                            }
                            SettingsTab::Synth => {
                                self.show_synth_settings(ui, settings, state, width)
                            }
                            SettingsTab::SoundFonts => self.sf_list.show(ui, settings, state),
                            SettingsTab::Render => {
                                self.show_render_settings(ui, settings, state, width)
                            }
                        }
                    })
                });
            });
    }

    pub fn load_palettes(&mut self, settings: &mut WasabiSettings) -> Result<(), WasabiError> {
        self.palettes.clear();

        let files = std::fs::read_dir(WasabiSettings::get_palettes_dir())
            .map_err(WasabiError::FilesystemError)?;

        for file in files.filter_map(|i| i.ok()) {
            if let Ok(ftype) = file.file_type() {
                if ftype.is_file() {
                    let path = file.path();
                    let selected = settings.midi.palette_path == path
                        && settings.midi.colors == Colors::Palette;

                    self.palettes.push(FilePalette { path, selected });
                }
            }
        }

        Ok(())
    }

    #[cfg(any(not(supported_os), target_os = "freebsd"))]
    pub fn load_midi_devices(&mut self, _settings: &mut WasabiSettings) -> Result<(), WasabiError> {
        Ok(())
    }

    #[cfg(all(supported_os, not(target_os = "freebsd")))]
    pub fn load_midi_devices(&mut self, settings: &mut WasabiSettings) -> Result<(), WasabiError> {
        self.midi_devices.clear();
        let con = midir::MidiOutput::new("wasabi")
            .map_err(|e| WasabiError::SynthError(format!("{e:?}")))?;

        // Add all valid ports
        for port in con.ports().iter() {
            let name = con
                .port_name(port)
                .map_err(|e| WasabiError::SynthError(format!("{e:?}")))?;
            self.midi_devices.push(MidiDevice {
                name,
                selected: false,
            });
        }

        // Select the device specified in settings if found, or select the first available
        let saved = settings.synth.midi_device.clone();
        if let Some(found) = self.midi_devices.iter_mut().find(|d| d.name == saved) {
            found.selected = true;
        } else if !self.midi_devices.is_empty() {
            self.midi_devices[0].selected = true;
            settings.synth.midi_device = self.midi_devices[0].name.clone();
        }

        Ok(())
    }
}
