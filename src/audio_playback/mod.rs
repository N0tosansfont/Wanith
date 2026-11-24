//! Real-time audio preview playback facade.
//!
//! # Module Boundary Rules
//! - Strictly isolated from `crate::offline_render`: real-time preview NEVER imports
//!   offline rendering, file-export, or video-encoding code.
//! - When `Synth::Kdmapi` (Built-In KDMAPI v3) is active, playback threads acquire a
//!   generation-tagged [`kdmapiv3::DirectMidiSender`] once and bypass the `RwLock`
//!   completely on every MIDI event batch and reset.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};

use kdmapiv3::DirectMidiSender;

use crate::{
    gui::window::{GuiMessageSystem, LoadingStatus},
    settings::{Synth, SynthSettings, WasabiSoundfont},
};

mod kdmapi;
pub use kdmapi::*;

mod xsynth;
pub use xsynth::*;

#[cfg(all(supported_os, not(target_os = "freebsd")))]
mod midiout;
#[cfg(all(supported_os, not(target_os = "freebsd")))]
pub use midiout::*;

enum MidiAudioPlayer {
    XSynth(XSynthPlayer),
    Kdmapi(KdmapiPlayer),
    #[cfg(all(supported_os, not(target_os = "freebsd")))]
    MidiDevice(MidiDevicePlayer),
    None,
}

pub struct WasabiAudioPlayer {
    inner: RwLock<MidiAudioPlayer>,
    generation: AtomicU64,
}

impl WasabiAudioPlayer {
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(MidiAudioPlayer::None),
            generation: AtomicU64::new(0),
        })
    }

    /// Monotonically incremented whenever the active synth backend changes.
    /// Playback threads compare this single atomic load to validate their cached
    /// lock-free [`DirectMidiSender`] without taking any lock on the hot path.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Cold-path helper called only when `generation()` changes to obtain a lock-free
    /// [`DirectMidiSender`] if the active backend is Built-In KDMAPI v3.
    pub fn clone_direct_sender(&self) -> (u64, Option<DirectMidiSender>) {
        let guard = match self.inner.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let gen = self.generation.load(Ordering::Acquire);
        let sender = match &*guard {
            MidiAudioPlayer::Kdmapi(player) => Some(player.sender()),
            _ => None,
        };
        (gen, sender)
    }

    /// Non-blocking voice count query for UI telemetry (never stalls on SoundFont loads).
    pub fn voice_count(&self) -> Option<u64> {
        let guard = self.inner.try_read().ok()?;
        match &*guard {
            MidiAudioPlayer::XSynth(player) => Some(player.voice_count()),
            MidiAudioPlayer::Kdmapi(player) => Some(player.voice_count()),
            _ => None,
        }
    }

    pub fn push_events(&self, data: impl Iterator<Item = u32>) {
        match &mut *self.inner.write().unwrap() {
            MidiAudioPlayer::XSynth(player) => player.push_events(data),
            MidiAudioPlayer::Kdmapi(player) => player.push_events(data),
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            MidiAudioPlayer::MidiDevice(player) => player.push_events(data),
            _ => {}
        }
    }

    pub fn configure(&self, settings: &SynthSettings) {
        match &mut *self.inner.write().unwrap() {
            MidiAudioPlayer::XSynth(player) => player.configure(&settings.xsynth),
            MidiAudioPlayer::Kdmapi(_) => {}
            _ => {}
        }
    }

    pub fn set_soundfonts(
        &self,
        soundfonts: &[WasabiSoundfont],
        loading_status: Arc<LoadingStatus>,
        errors: Arc<GuiMessageSystem>,
    ) {
        match &mut *self.inner.write().unwrap() {
            MidiAudioPlayer::XSynth(player) => {
                player.set_soundfonts(soundfonts, loading_status, errors)
            }
            MidiAudioPlayer::Kdmapi(player) => player.set_soundfonts(soundfonts, errors),
            _ => {}
        }
    }

    pub fn reset(&self) {
        match &mut *self.inner.write().unwrap() {
            MidiAudioPlayer::XSynth(player) => player.reset(),
            MidiAudioPlayer::Kdmapi(player) => player.reset(),
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            MidiAudioPlayer::MidiDevice(player) => player.reset(),
            _ => {}
        }
    }

    pub fn switch(
        &self,
        settings: &SynthSettings,
        loading_status: Arc<LoadingStatus>,
        errors: Arc<GuiMessageSystem>,
    ) {
        // First drop the previous synth to release audio device cleanly and invalidate cached senders
        *self.inner.write().unwrap() = MidiAudioPlayer::None;
        self.generation.fetch_add(1, Ordering::Release);

        // Create the new synth object based on the settings
        let mut kdmapi_loaded = false;
        let synth = match settings.synth {
            Synth::XSynth => {
                MidiAudioPlayer::XSynth(XSynthPlayer::new(settings.xsynth.config.clone()))
            }
            Synth::Kdmapi => match KdmapiPlayer::new(&settings.kdmapi, &settings.soundfonts) {
                Ok(kdmapi) => {
                    kdmapi_loaded = true;
                    MidiAudioPlayer::Kdmapi(kdmapi)
                }
                Err(e) => {
                    errors.error(&e);
                    MidiAudioPlayer::None
                }
            },
            #[cfg(all(supported_os, not(target_os = "freebsd")))]
            Synth::MidiDevice => match MidiDevicePlayer::new(settings.midi_device.clone()) {
                Ok(midiout) => MidiAudioPlayer::MidiDevice(midiout),
                Err(e) => {
                    errors.error(&e);
                    MidiAudioPlayer::None
                }
            },
            Synth::None => MidiAudioPlayer::None,
        };

        // Install the new synth and publish the new generation
        *self.inner.write().unwrap() = synth;
        self.generation.fetch_add(1, Ordering::Release);

        // XSynth configures and loads SoundFonts after construction;
        // Built-In KDMAPI v3 already configured and loaded SoundFonts atomically in `new()`.
        if !kdmapi_loaded {
            self.configure(settings);
            self.set_soundfonts(&settings.soundfonts, loading_status, errors);
        }
    }
}
