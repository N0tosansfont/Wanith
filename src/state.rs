use std::{path::PathBuf, sync::Arc};

use crate::{
    audio_playback::WasabiAudioPlayer,
    gui::window::{GuiMessageSystem, LoadingStatus},
    i18n::I18n,
    offline_render::{GpuVideoCapabilities, OfflineRenderManager},
    settings::Language,
    system_monitor::SystemMonitor,
};

#[derive(Default, PartialEq, Eq, Clone, Copy)]
pub enum SettingsTab {
    #[default]
    Visual,
    Skin,
    Midi,
    Synth,
    SoundFonts,
    Render,
}

pub struct WasabiState {
    pub synth: Arc<WasabiAudioPlayer>,

    pub fullscreen: bool,

    pub errors: Arc<GuiMessageSystem>,
    pub loading_status: Arc<LoadingStatus>,

    pub panel_pinned: bool,
    pub panel_id: egui::Id,
    pub panel_popup_id: egui::Id,
    pub stats_visible: bool,

    pub show_settings: bool,
    pub show_shortcuts: bool,
    pub show_about: bool,

    pub settings_tab: SettingsTab,

    pub last_midi_location: PathBuf,
    pub last_sf_location: PathBuf,

    /// Currently loaded MIDI file path & known length (seconds) for offline Render Mode.
    pub active_midi_path: Option<PathBuf>,
    pub active_midi_length_sec: Option<f64>,

    /// Read-only hardware Vulkan Video Encode capabilities probed at GPU init.
    pub gpu_video_caps: GpuVideoCapabilities,

    /// Isolated background offline render job manager (zero impact on realtime preview).
    pub offline_render: OfflineRenderManager,

    /// Low-frequency background OS/GPU telemetry monitor for lock-free Legit Run HUD.
    pub system_monitor: SystemMonitor,

    /// Pre-parsed UI localization dictionary (`lang/en.json`, `lang/ko.json`).
    pub i18n: I18n,
}

impl WasabiState {
    pub fn new() -> Self {
        let loading_status = LoadingStatus::new();
        let errors = GuiMessageSystem::new();

        Self {
            synth: WasabiAudioPlayer::empty(),

            fullscreen: false,

            errors,
            loading_status,

            panel_pinned: true,
            panel_id: egui::Id::new("playback_panel"),
            panel_popup_id: egui::Id::new("options_popup"),
            stats_visible: true,

            show_settings: false,
            show_shortcuts: false,
            show_about: false,

            settings_tab: SettingsTab::default(),

            last_midi_location: PathBuf::default(),
            last_sf_location: PathBuf::default(),

            active_midi_path: None,
            active_midi_length_sec: None,

            gpu_video_caps: GpuVideoCapabilities::default(),
            offline_render: OfflineRenderManager::new(),
            system_monitor: SystemMonitor::start(),
            i18n: I18n::load(Language::default()),
        }
    }
}
