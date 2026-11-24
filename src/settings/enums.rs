use num_enum::FromPrimitive;
use serde_derive::{Deserialize, Serialize};
use std::{fmt::Debug, slice::Iter, str::FromStr};

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    English,
    Korean,
}

impl Language {
    pub const ALL: [Language; 2] = [Language::English, Language::Korean];

    #[inline]
    pub const fn code(self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Korean => "ko",
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Korean => "한국어 (Korean)",
        }
    }
}

#[repr(usize)]
#[derive(Debug, Default, Serialize, Deserialize, PartialEq, Eq, Clone, Copy, FromPrimitive)]
#[serde(rename_all = "lowercase")]
pub enum MidiParsing {
    #[default]
    Ram = 0,
    Live = 1,
    #[serde(alias = "Cake", alias = "rice_cake", alias = "ricecake")]
    Cake = 2,
}

impl FromStr for MidiParsing {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ram" => Ok(MidiParsing::Ram),
            "live" => Ok(MidiParsing::Live),
            "cake" | "rice_cake" | "ricecake" => Ok(MidiParsing::Cake),
            s => Err(format!(
                "{} was not expected. Expected one of `ram`, `live` or `cake`",
                s
            )),
        }
    }
}

#[allow(clippy::enum_variant_names)]
#[repr(usize)]
#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, FromPrimitive)]
#[serde(rename_all = "lowercase")]
pub enum Synth {
    XSynth = 0,
    #[default]
    Kdmapi = 1,
    #[cfg(all(supported_os, not(target_os = "freebsd")))]
    MidiDevice = 2,
    None = 3,
}

impl Synth {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Synth::XSynth => "Built-In (XSynth)",
            Synth::Kdmapi => "Built-In (KDMAPI v3)",
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            Synth::MidiDevice => "MIDI Device",
            Synth::None => "None",
        }
    }
}

impl FromStr for Synth {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "xsynth" => Ok(Synth::XSynth),
            "kdmapi" => Ok(Synth::Kdmapi),
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            "mididevice" => Ok(Synth::MidiDevice),
            "none" => Ok(Synth::None),
            s => Err(format!(
                "{} was not expected. Expected one of `xsynth`, `kdmapi`, `mididevice` or `none`",
                s
            )),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KdmapiInterpolation {
    #[default]
    Linear,
    Sinc8,
    Sinc16,
    Sinc32,
}

impl KdmapiInterpolation {
    pub const ALL: [KdmapiInterpolation; 4] = [
        KdmapiInterpolation::Linear,
        KdmapiInterpolation::Sinc8,
        KdmapiInterpolation::Sinc16,
        KdmapiInterpolation::Sinc32,
    ];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            KdmapiInterpolation::Linear => "Linear (Fastest)",
            KdmapiInterpolation::Sinc8 => "Sinc 8-tap",
            KdmapiInterpolation::Sinc16 => "Sinc 16-tap",
            KdmapiInterpolation::Sinc32 => "Sinc 32-tap (Highest)",
        }
    }

    #[inline]
    pub const fn to_kdmapiv3(self) -> kdmapiv3::InterpolationMode {
        match self {
            KdmapiInterpolation::Linear => kdmapiv3::InterpolationMode::Linear,
            KdmapiInterpolation::Sinc8 => kdmapiv3::InterpolationMode::Sinc8,
            KdmapiInterpolation::Sinc16 => kdmapiv3::InterpolationMode::Sinc16,
            KdmapiInterpolation::Sinc32 => kdmapiv3::InterpolationMode::Sinc32,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RenderTarget {
    #[default]
    AudioOnly,
    VideoAndAudio,
}

impl RenderTarget {
    pub const ALL: [RenderTarget; 2] = [RenderTarget::AudioOnly, RenderTarget::VideoAndAudio];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            RenderTarget::AudioOnly => "Audio Only (.wav)",
            RenderTarget::VideoAndAudio => "Video + Audio (Native Vulkan Video .mp4)",
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WavSampleFormat {
    #[default]
    Float32,
    Pcm16,
}

impl WavSampleFormat {
    pub const ALL: [WavSampleFormat; 2] = [WavSampleFormat::Float32, WavSampleFormat::Pcm16];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            WavSampleFormat::Float32 => "32-bit Float WAV (IEEE)",
            WavSampleFormat::Pcm16 => "16-bit PCM WAV",
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    #[default]
    H264,
    H265,
}

impl VideoCodec {
    pub const ALL: [VideoCodec; 2] = [VideoCodec::H264, VideoCodec::H265];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            VideoCodec::H264 => "H.264 / AVC (VK_KHR_video_encode_h264)",
            VideoCodec::H265 => "H.265 / HEVC (VK_KHR_video_encode_h265)",
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VideoRateControl {
    #[default]
    Bitrate,
    ConstantQp,
}

impl VideoRateControl {
    pub const ALL: [VideoRateControl; 2] =
        [VideoRateControl::Bitrate, VideoRateControl::ConstantQp];

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            VideoRateControl::Bitrate => "Target Bitrate (VBR/CBR)",
            VideoRateControl::ConstantQp => "Constant QP (CQP)",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, FromPrimitive)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
pub enum Statistics {
    #[default]
    Time = 0,
    Fps = 1,
    VoiceCount = 2,
    Rendered = 3,
    NoteCount = 4,
    Polyphony = 5,
    Nps = 6,
    Xrun = 7,
    Cpu = 8,
    Ram = 9,
    Gpu = 10,
}

impl Statistics {
    pub fn iter() -> Iter<'static, Statistics> {
        static STATISTICS: [Statistics; 11] = [
            Statistics::Xrun,
            Statistics::VoiceCount,
            Statistics::Nps,
            Statistics::Fps,
            Statistics::Cpu,
            Statistics::Ram,
            Statistics::Gpu,
            Statistics::Time,
            Statistics::Rendered,
            Statistics::Polyphony,
            Statistics::NoteCount,
        ];
        STATISTICS.iter()
    }
}

impl FromStr for Statistics {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "time" => Ok(Statistics::Time),
            "fps" => Ok(Statistics::Fps),
            "voicecount" | "voice" => Ok(Statistics::VoiceCount),
            "rendered" => Ok(Statistics::Rendered),
            "notecount" => Ok(Statistics::NoteCount),
            "polyphony" => Ok(Statistics::Polyphony),
            "nps" => Ok(Statistics::Nps),
            "xrun" => Ok(Statistics::Xrun),
            "cpu" => Ok(Statistics::Cpu),
            "ram" => Ok(Statistics::Ram),
            "gpu" => Ok(Statistics::Gpu),
            s => Err(format!("{} was not expected.", s)),
        }
    }
}

#[repr(usize)]
#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, FromPrimitive)]
#[serde(rename_all = "lowercase")]
pub enum Colors {
    #[default]
    Rainbow = 0,
    Random = 1,
    Palette = 2,
}

impl Colors {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Colors::Rainbow => "Rainbow",
            Colors::Random => "Random",
            Colors::Palette => "Palette",
        }
    }
}

impl FromStr for Colors {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rainbow" => Ok(Colors::Rainbow),
            "random" => Ok(Colors::Random),
            "palette" => Ok(Colors::Palette),
            s => Err(format!(
                "{} was not expected. Expected one of `ranbow`, `random` or `palette`",
                s
            )),
        }
    }
}

#[repr(u32)]
#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SkinPreset {
    #[default]
    Wasaby = 0,
    Wynthesia = 1,
    WfaPlus = 2,
    Zanith = 3,
}

impl SkinPreset {
    #[inline]
    pub const fn id(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            SkinPreset::Wasaby => "Wasaby",
            SkinPreset::Wynthesia => "Wynthesia",
            SkinPreset::WfaPlus => "WFA(+)",
            SkinPreset::Zanith => "Zanith",
        }
    }

    #[inline]
    pub const fn homage_target(self) -> &'static str {
        match self {
            SkinPreset::Wasaby => "Wasabi Classic",
            SkinPreset::Wynthesia => "Synthesia Homage",
            SkinPreset::WfaPlus => "Piano From Above Homage",
            SkinPreset::Zanith => "Zenith Homage",
        }
    }

    #[inline]
    pub const fn i18n_desc_key(self) -> &'static str {
        match self {
            SkinPreset::Wasaby => "skin.preset.wasaby.desc",
            SkinPreset::Wynthesia => "skin.preset.wynthesia.desc",
            SkinPreset::WfaPlus => "skin.preset.wfaplus.desc",
            SkinPreset::Zanith => "skin.preset.zanith.desc",
        }
    }

    #[inline]
    pub const fn recommended_bg(self) -> egui::Color32 {
        match self {
            SkinPreset::Wasaby => egui::Color32::from_rgb(30, 30, 30),
            SkinPreset::Wynthesia => egui::Color32::from_rgb(24, 27, 33),
            SkinPreset::WfaPlus => egui::Color32::from_rgb(38, 38, 38),
            SkinPreset::Zanith => egui::Color32::from_rgb(14, 15, 20),
        }
    }

    #[inline]
    pub const fn recommended_bar(self) -> egui::Color32 {
        match self {
            SkinPreset::Wasaby => egui::Color32::from_rgb(145, 0, 0),
            SkinPreset::Wynthesia => egui::Color32::from_rgb(165, 24, 24),
            SkinPreset::WfaPlus => egui::Color32::from_rgb(130, 12, 12),
            SkinPreset::Zanith => egui::Color32::from_rgb(48, 120, 215),
        }
    }
}
