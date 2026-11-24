use directories::BaseDirs;
use egui::Color32;
use serde_derive::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs,
    io::Write,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};
use xsynth_core::soundfont::SoundfontInitOptions;
use xsynth_realtime::XSynthRealtimeConfig;

mod enums;
mod migrations;

pub use enums::*;

use crate::gui::window::WasabiError;

// region: gui

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct GuiSettings {
    pub language: Language,
    pub check_for_updates: bool,
    pub vsync: bool,
    pub skip_control: f64,
    pub speed_control: f64,
}

impl Default for GuiSettings {
    fn default() -> Self {
        Self {
            language: Language::English,
            check_for_updates: true,
            vsync: true,
            skip_control: 1.0,
            speed_control: 0.05,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct StatisticsSettings {
    pub border: bool,
    pub floating: bool,
    pub opacity: f32,
    pub order: Vec<(Statistics, bool)>,
}

impl Default for StatisticsSettings {
    fn default() -> Self {
        Self {
            border: true,
            floating: true,
            opacity: 0.5,
            order: Statistics::iter().map(|i| (*i, true)).collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct SceneSettings {
    pub bg_color: Color32,
    pub bar_color: Color32,
    pub skin: SkinPreset,
    pub statistics: StatisticsSettings,
    pub note_speed: f64,
    pub key_range: RangeInclusive<u8>,
}

impl Default for SceneSettings {
    fn default() -> Self {
        Self {
            bg_color: Color32::from_rgb(30, 30, 30),
            bar_color: Color32::from_rgb(145, 0, 0),
            skin: SkinPreset::default(),
            statistics: Default::default(),
            note_speed: 0.25,
            key_range: 0..=127,
        }
    }
}

// endregion

// region: midi

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct MidiSettings {
    pub parsing: MidiParsing,
    pub start_delay: f64,
    pub colors: Colors,
    pub randomize_palette: bool,
    pub palette_path: PathBuf,
}

impl Default for MidiSettings {
    fn default() -> Self {
        Self {
            parsing: MidiParsing::Cake,
            start_delay: 2.0,
            colors: Colors::Rainbow,
            randomize_palette: false,
            palette_path: PathBuf::new(),
        }
    }
}

// endregion

// region: synth

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct XSynthSettings {
    pub config: XSynthRealtimeConfig,
    pub limit_layers: bool,
    pub layers: usize,
}

impl Default for XSynthSettings {
    fn default() -> Self {
        Self {
            config: Default::default(),
            limit_layers: true,
            layers: 4,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct KdmapiSettings {
    pub voice_limit: u32,
    pub interpolation: KdmapiInterpolation,
    pub buffer_frames: u32,
    pub render_threads: u32,
    pub ring_buffer_capacity: usize,
    pub disable_all_effects: bool,
    pub enable_lowpass_filter: bool,
    pub enable_vibrato_lfo: bool,
}

impl Default for KdmapiSettings {
    fn default() -> Self {
        Self {
            voice_limit: kdmapiv3::DEFAULT_VOICE_LIMIT,
            interpolation: KdmapiInterpolation::Linear,
            buffer_frames: kdmapiv3::DEFAULT_BUFFER_FRAMES,
            render_threads: kdmapiv3::DEFAULT_RENDER_THREADS,
            ring_buffer_capacity: kdmapiv3::DEFAULT_SHORT_RING_CAPACITY,
            disable_all_effects: false,
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(default)]
pub struct WasabiSoundfont {
    pub path: PathBuf,
    pub enabled: bool,
    pub options: SoundfontInitOptions,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct SynthSettings {
    pub synth: Synth,
    pub soundfonts: Vec<WasabiSoundfont>,

    pub xsynth: XSynthSettings,
    pub kdmapi: KdmapiSettings,
    pub midi_device: String,
}

impl Default for SynthSettings {
    fn default() -> Self {
        Self {
            synth: Synth::Kdmapi,
            soundfonts: Vec::new(),
            xsynth: Default::default(),
            kdmapi: Default::default(),
            midi_device: String::new(),
        }
    }
}

// endregion

// region: render

pub const MIN_RENDER_SAMPLE_RATE: u32 = 8_000;
pub const MAX_RENDER_SAMPLE_RATE: u32 = 384_000;
pub const MIN_RENDER_DIMENSION: u32 = 128;
pub const MAX_RENDER_DIMENSION: u32 = 7_680;
pub const MIN_RENDER_FPS: u32 = 1;
pub const MAX_RENDER_FPS: u32 = 1_000;
pub const MIN_VIDEO_BITRATE_KBPS: u32 = 500;
pub const MAX_VIDEO_BITRATE_KBPS: u32 = 500_000;
pub const MIN_VIDEO_QP: u32 = 0;
pub const MAX_VIDEO_QP: u32 = 51;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct RenderSettings {
    pub target: RenderTarget,
    pub output_path: PathBuf,
    // Audio (KMC-style deterministic render via engine_core / kdmapi v3)
    pub sample_rate: u32,
    pub wav_format: WavSampleFormat,
    // Video (Zenith fixed-timestep + Native Vulkan Video + in-process AAC/MP4 muxer)
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub video_codec: VideoCodec,
    pub rate_control: VideoRateControl,
    pub bitrate_kbps: u32,
    pub qp: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            target: RenderTarget::AudioOnly,
            output_path: PathBuf::from("output.wav"),
            sample_rate: kdmapiv3::DEFAULT_SAMPLE_RATE,
            wav_format: WavSampleFormat::Float32,
            width: 1920,
            height: 1080,
            fps: 60,
            video_codec: VideoCodec::H264,
            rate_control: VideoRateControl::Bitrate,
            bitrate_kbps: 25_000,
            qp: 20,
        }
    }
}

// endregion

// region: general

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct WasabiSettings {
    pub gui: GuiSettings,
    pub scene: SceneSettings,
    pub midi: MidiSettings,
    pub synth: SynthSettings,
    pub render: RenderSettings,
}

impl WasabiSettings {
    const VERSION_TEXT: &str = "# DON'T EDIT THIS LINE; Version: 2\n";

    pub fn new_or_load() -> Result<Self, WasabiError> {
        let mut err = WasabiError::SettingsError("Unknown".into());

        let config_path = Self::get_config_path();
        let old_config_path = Self::get_old_config_path();

        if old_config_path.exists() {
            std::fs::rename(old_config_path, &config_path).map_err(WasabiError::FilesystemError)?;
        }

        if !Path::new(&config_path).exists() {
            return Self::load_and_save_defaults();
        } else if let Ok(config) = fs::read_to_string(&config_path) {
            if config.starts_with(Self::VERSION_TEXT) {
                let offset = Self::VERSION_TEXT.len();
                match serde_json::from_str::<WasabiSettings>(&config[offset..]) {
                    Ok(mut config) => {
                        if config.scene.statistics.order.len() != Statistics::iter().len() {
                            config.scene.statistics.order = StatisticsSettings::default().order;
                        }
                        return Ok(config);
                    }
                    Err(e) => err = WasabiError::SettingsError(e.to_string()),
                }
            } else if config.starts_with("# DON'T EDIT THIS LINE; Version: 1") {
                match migrations::WasabiConfigFileV1::migrate_to_v2(config) {
                    Ok(cfg) => {
                        cfg.save_to_file()?;
                        return Ok(cfg);
                    }
                    Err(e) => err = WasabiError::SettingsError(e.to_string()),
                }
            } else {
                match migrations::WasabiConfigFileV0::migrate_to_v1(config) {
                    Ok(v1) => {
                        let cfg = migrations::WasabiConfigFileV1::migrate_to_v2_raw(v1);
                        cfg.save_to_file()?;
                        return Ok(cfg);
                    }
                    Err(e) => err = WasabiError::SettingsError(e.to_string()),
                }
            }
        }

        Err(err)
    }

    pub fn save_to_file(&self) -> Result<(), WasabiError> {
        let config_path = Self::get_config_path();
        let cfg: String = serde_json::to_string_pretty(&self)
            .map_err(|e| WasabiError::SettingsError(e.to_string()))?;
        if let Ok(mut file) = fs::File::create(&config_path) {
            file.write_all(Self::VERSION_TEXT.as_bytes())
                .map_err(WasabiError::FilesystemError)?;
            file.write_all(cfg.as_bytes())
                .expect("Error creating config");
        }
        Ok(())
    }

    fn load_and_save_defaults() -> Result<Self, WasabiError> {
        let cfg = Self::default();
        Self::save_to_file(&cfg)?;
        Ok(cfg)
    }

    pub fn get_config_dir() -> PathBuf {
        if let Some(base_dirs) = BaseDirs::new() {
            let mut path: PathBuf = base_dirs.config_dir().to_path_buf();
            path.push("wanith");

            if std::fs::create_dir_all(&path).is_ok() {
                let new_cfg = path.join("wanith-config.json");
                if !new_cfg.exists() {
                    let legacy_cfg = base_dirs
                        .config_dir()
                        .join("wasabi")
                        .join("wasabi-config.json");
                    if legacy_cfg.exists() {
                        let _ = std::fs::copy(&legacy_cfg, &new_cfg);
                    }
                }
                return path;
            }
        }

        PathBuf::from("./")
    }

    fn get_config_path() -> PathBuf {
        let mut path = Self::get_config_dir();
        path.push("wanith-config.json");

        path
    }

    fn get_old_config_path() -> PathBuf {
        let mut path = Self::get_config_dir();
        path.push("wasabi-config.toml");

        path
    }

    pub fn get_palettes_dir() -> PathBuf {
        let mut path = Self::get_config_dir();
        path.push("palettes");
        std::fs::create_dir_all(&path).unwrap_or_default();

        path
    }
}

// endregion
