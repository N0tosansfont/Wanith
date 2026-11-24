//! Direct in-process real-time `kdmapiv3` player for Wasabi preview playback.
//!
//! # Real-Time Guarantees
//! - No external DLL loading (`kdmapi-rs` removed), no C FFI overhead, no temporary
//!   `.csflist` file I/O on disk.
//! - Playback threads clone [`DirectMidiSender`] and push stack-batched MIDI messages
//!   directly into `kdmapiv3`'s pre-allocated lock-free ring buffer with zero locks
//!   and zero heap allocations.

use std::sync::Arc;

use kdmapiv3::{DirectKdmapiEngine, DirectMidiSender};

use crate::engine_core::{build_synth_config, collect_enabled_soundfont_paths};
use crate::gui::window::{GuiMessageSystem, WasabiError};
use crate::settings::{KdmapiSettings, WasabiSoundfont};

pub struct KdmapiPlayer {
    engine: DirectKdmapiEngine,
    sender: DirectMidiSender,
}

impl KdmapiPlayer {
    pub fn new(
        settings: &KdmapiSettings,
        soundfonts: &[WasabiSoundfont],
    ) -> Result<Self, WasabiError> {
        let synth_cfg = build_synth_config(settings);
        let sf_paths = collect_enabled_soundfont_paths(soundfonts);
        let engine = DirectKdmapiEngine::start(synth_cfg, settings.ring_buffer_capacity, &sf_paths)
            .map_err(|e| {
                WasabiError::SynthError(format!("Failed to start built-in KDMAPI v3 engine: {e}"))
            })?;
        let sender = engine.sender();
        Ok(Self { engine, sender })
    }

    #[inline]
    pub fn sender(&self) -> DirectMidiSender {
        self.sender.clone()
    }

    #[inline]
    pub fn reset(&self) {
        self.sender.reset();
    }

    #[inline]
    pub fn push_events(&self, data: impl Iterator<Item = u32>) {
        self.sender.push_short_iter(data);
    }

    pub fn set_soundfonts(
        &mut self,
        soundfonts: &[WasabiSoundfont],
        errors: Arc<GuiMessageSystem>,
    ) {
        let sf_paths = collect_enabled_soundfont_paths(soundfonts);
        if let Err(e) = self.engine.load_soundfonts(&sf_paths) {
            let error = WasabiError::SynthError(format!(
                "Failed to load SoundFonts into KDMAPI v3: {e}"
            ));
            errors.error(&error);
        }
    }

    #[inline]
    pub fn voice_count(&self) -> u64 {
        self.sender.voice_count()
    }
}
