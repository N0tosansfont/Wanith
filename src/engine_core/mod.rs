//! Shared audio synthesis core boundary used by both real-time preview (`audio_playback`)
//! and isolated offline rendering (`offline_render`).
//!
//! # Module Boundary Rules
//! - Owns `kdmapiv3::SynthConfig` construction, SoundFont path collection, and
//!   `kdmapiv3::SoundFontSynth` initialization.
//! - Must NEVER depend on GUI code, video encoders, or file-export sinks.

use std::path::PathBuf;

use kdmapiv3::{
    BackendError, SoundFontSynth, SynthBackend, SynthConfig, DEFAULT_SAMPLE_RATE,
    MAX_BUFFER_FRAMES, MAX_RENDER_THREADS, MAX_VOICE_LIMIT, MIN_BUFFER_FRAMES,
    MIN_RENDER_THREADS, MIN_VOICE_LIMIT,
};

use crate::settings::{KdmapiSettings, WasabiSoundfont};

/// Build a sanitized `kdmapiv3::SynthConfig` directly from Wasabi's manual settings.
///
/// Honors all user choices verbatim (clamped only to engine SSoT safety bounds) with
/// zero hidden auto-scaling.
#[inline]
pub fn build_synth_config(settings: &KdmapiSettings) -> SynthConfig {
    SynthConfig {
        sample_rate: DEFAULT_SAMPLE_RATE,
        max_voices: settings.voice_limit.clamp(MIN_VOICE_LIMIT, MAX_VOICE_LIMIT),
        buffer_frames: settings
            .buffer_frames
            .clamp(MIN_BUFFER_FRAMES, MAX_BUFFER_FRAMES),
        render_threads: settings
            .render_threads
            .clamp(MIN_RENDER_THREADS, MAX_RENDER_THREADS),
        interpolation: settings.interpolation.to_kdmapiv3(),
        disable_all_effects: settings.disable_all_effects,
        enable_lowpass_filter: settings.enable_lowpass_filter,
        enable_vibrato_lfo: settings.enable_vibrato_lfo,
    }
}

/// Collect enabled SoundFont paths in Wasabi's priority order (top-most entry wins).
///
/// Wasabi UI states `"Loading order is bottom to top"` (bottom loads first, top overrides it).
/// `kdmapiv3::SoundFontBank::load_ordered` gives priority to earlier entries in the slice,
/// so iterating forward preserves exact priority semantics without writing temporary `.csflist` files.
pub fn collect_enabled_soundfont_paths(soundfonts: &[WasabiSoundfont]) -> Vec<PathBuf> {
    soundfonts
        .iter()
        .filter(|sf| sf.enabled && !sf.path.as_os_str().is_empty())
        .map(|sf| sf.path.clone())
        .collect()
}

/// Construct and initialize a standalone `kdmapiv3::SoundFontSynth` for shared DSP callers
/// (e.g. deterministic offline audio rendering).
#[allow(dead_code)]
pub fn create_shared_synth(
    settings: &KdmapiSettings,
    sample_rate: u32,
    soundfonts: &[PathBuf],
) -> Result<SoundFontSynth, BackendError> {
    let mut cfg = build_synth_config(settings);
    cfg.sample_rate = sample_rate.clamp(8_000, 384_000);

    let mut synth = SoundFontSynth::new();
    synth.init(cfg)?;
    if !soundfonts.is_empty() {
        synth.load_soundfonts(soundfonts)?;
    }
    Ok(synth)
}
