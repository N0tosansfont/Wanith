//! KMC-style deterministic offline MIDI-to-WAV audio renderer backed by `engine_core` (`kdmapi v3`).
//!
//! # Guarantees
//! - Completely isolated inside `crate::offline_render` — never touches `crate::audio_playback`.
//! - Sample-accurate deterministic MIDI event scheduling (zero wall-clock jitter, zero note skipping).
//! - Pre-allocated stereo render buffer (zero per-block heap allocations during synthesis).
//! - Honors user's exact `KdmapiSettings` (`voice_limit`, `interpolation`, `render_threads`,
//!   `disable_all_effects`, `enable_lowpass_filter`, `enable_vibrato_lfo`) and SoundFont priority order.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use kdmapiv3::{MAX_BUFFER_FRAMES, MIN_BUFFER_FRAMES};
use midi_toolkit::{
    events::{Event, MIDIEventEnum},
    io::MIDIFile as TKMIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time},
        unwrap_items, TimeCaster,
    },
};

use crate::settings::{
    KdmapiSettings, WavSampleFormat, MAX_RENDER_SAMPLE_RATE, MIN_RENDER_SAMPLE_RATE,
};

use super::job::OfflineRenderProgress;
use super::wav_writer::WavStreamWriter;
use super::OfflineAudioRenderer;

/// Matches Wasabi's `InRamMIDIFile` & `LiveMidiParser` tempo cancellation constant so
/// offline audio matches preview and future fixed-timestep video timing identically.
const WASABI_DEFAULT_TEMPO_US: u32 = 250_000;

/// Maximum post-MIDI release tail rendered while waiting for active voices to decay to zero.
const MAX_RELEASE_TAIL_SECONDS: f64 = 15.0;

#[derive(Debug, Clone)]
pub struct AudioRenderRequest {
    pub midi_path: PathBuf,
    pub output_path: PathBuf,
    pub sample_rate: u32,
    pub wav_format: WavSampleFormat,
    pub kdmapi_settings: KdmapiSettings,
    pub soundfonts: Vec<PathBuf>,
    pub estimated_duration_sec: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct AudioRenderReport {
    pub output_path: PathBuf,
    pub frames_written: u64,
    pub sample_rate: u32,
    pub elapsed_secs: f64,
    pub realtime_multiplier: f64,
    pub cancelled: bool,
}

#[inline]
fn encode_midi_short_msg(event: &Event) -> Option<u32> {
    match event {
        Event::NoteOn(e) => Some(
            ((0x90 | (e.channel & 0x0F)) as u32)
                | ((e.key as u32) << 8)
                | ((e.velocity as u32) << 16),
        ),
        Event::NoteOff(e) => {
            Some(((0x80 | (e.channel & 0x0F)) as u32) | ((e.key as u32) << 8))
        }
        Event::PolyphonicKeyPressure(e) => Some(
            ((0xA0 | (e.channel & 0x0F)) as u32)
                | ((e.key as u32) << 8)
                | ((e.velocity as u32) << 16),
        ),
        Event::ControlChange(e) => Some(
            ((0xB0 | (e.channel & 0x0F)) as u32)
                | ((e.controller as u32) << 8)
                | ((e.value as u32) << 16),
        ),
        Event::ProgramChange(e) => {
            Some(((0xC0 | (e.channel & 0x0F)) as u32) | ((e.program as u32) << 8))
        }
        Event::ChannelPressure(e) => {
            Some(((0xD0 | (e.channel & 0x0F)) as u32) | ((e.pressure as u32) << 8))
        }
        Event::PitchWheelChange(e) => {
            let val = (e.pitch + 8192).clamp(0, 16_383) as u32;
            Some(
                ((0xE0 | (e.channel & 0x0F)) as u32)
                    | ((val & 0x7F) << 8)
                    | (((val >> 7) & 0x7F) << 16),
            )
        }
        _ => None,
    }
}

/// Deterministically render a MIDI file to a stereo 32-bit Float or 16-bit PCM WAV file.
pub fn render_midi_to_wav(
    req: &AudioRenderRequest,
    progress: &OfflineRenderProgress,
) -> Result<AudioRenderReport, String> {
    if !req.midi_path.exists() {
        return Err(format!(
            "MIDI file does not exist: {}",
            req.midi_path.display()
        ));
    }
    if req.output_path.as_os_str().is_empty() {
        return Err("Output path cannot be empty.".into());
    }

    let sample_rate = req
        .sample_rate
        .clamp(MIN_RENDER_SAMPLE_RATE, MAX_RENDER_SAMPLE_RATE);
    let block_frames = req
        .kdmapi_settings
        .buffer_frames
        .clamp(MIN_BUFFER_FRAMES, MAX_BUFFER_FRAMES) as usize;

    let file = File::open(&req.midi_path)
        .map_err(|e| format!("Failed to open MIDI file {}: {e}", req.midi_path.display()))?;
    let midi = TKMIDIFile::open_from_stream(file, None)
        .map_err(|e| format!("Failed to parse MIDI file: {e:?}"))?;

    let mut renderer = OfflineAudioRenderer::new(&req.kdmapi_settings, sample_rate, &req.soundfonts)
        .map_err(|e| format!("Failed to initialize offline KDMAPI v3 synth: {e}"))?;

    let mut wav = WavStreamWriter::create(&req.output_path, sample_rate, req.wav_format, block_frames)
        .map_err(|e| format!("Failed to create WAV file {}: {e}", req.output_path.display()))?;

    let ppq = midi.ppq();
    let merged = pipe!(
        midi.iter_all_track_events_merged_batches()
        |> TimeCaster::<f64>::cast_event_delta()
        |> cancel_tempo_events(WASABI_DEFAULT_TEMPO_US)
        |> scale_event_time(1.0 / ppq as f64)
        |> unwrap_items()
    );

    // Pre-allocated stereo DSP render buffer (zero heap allocations inside render loop)
    let mut stereo_buf = vec![0.0f32; block_frames * 2];
    let mut rendered_frames: u64 = 0;
    let mut midi_time_sec: f64 = 0.0;
    let start_instant = Instant::now();
    let mut blocks_since_metric_update: u32 = 0;

    let mut cancelled = false;

    for batch in merged {
        if progress.is_cancel_requested() {
            cancelled = true;
            break;
        }

        midi_time_sec += batch.delta;
        let target_frame = (midi_time_sec.max(0.0) * sample_rate as f64).round() as u64;

        while rendered_frames < target_frame {
            if progress.is_cancel_requested() {
                cancelled = true;
                break;
            }
            let chunk = (target_frame - rendered_frames).min(block_frames as u64) as usize;
            let slice = &mut stereo_buf[..chunk * 2];
            renderer.render_stereo_block(slice);
            wav.write_interleaved_stereo(slice)
                .map_err(|e| format!("Failed writing WAV audio data: {e}"))?;
            rendered_frames += chunk as u64;

            blocks_since_metric_update += 1;
            if blocks_since_metric_update >= 16 {
                blocks_since_metric_update = 0;
                progress.update_metrics(
                    rendered_frames,
                    start_instant.elapsed().as_micros() as u64,
                    renderer.active_voices(),
                );
            }
        }

        if cancelled {
            break;
        }

        for ev in batch.iter_events() {
            if let Some(dw_msg) = encode_midi_short_msg(ev.as_event()) {
                renderer.push_short_msg(dw_msg);
            }
        }
    }

    // Drain release / decay tail until all voices finish naturally (bounded by safety ceiling)
    if !cancelled {
        let max_tail_frames = (MAX_RELEASE_TAIL_SECONDS * sample_rate as f64).round() as u64;
        let mut tail_rendered: u64 = 0;

        while renderer.active_voices() > 0 && tail_rendered < max_tail_frames {
            if progress.is_cancel_requested() {
                cancelled = true;
                break;
            }
            let chunk = (max_tail_frames - tail_rendered).min(block_frames as u64) as usize;
            let slice = &mut stereo_buf[..chunk * 2];
            renderer.render_stereo_block(slice);
            wav.write_interleaved_stereo(slice)
                .map_err(|e| format!("Failed writing WAV tail data: {e}"))?;
            rendered_frames += chunk as u64;
            tail_rendered += chunk as u64;

            progress.update_metrics(
                rendered_frames,
                start_instant.elapsed().as_micros() as u64,
                renderer.active_voices(),
            );
        }
    }

    let elapsed_secs = start_instant.elapsed().as_secs_f64();

    if cancelled {
        drop(wav);
        let _ = std::fs::remove_file(&req.output_path);
        return Ok(AudioRenderReport {
            output_path: req.output_path.clone(),
            frames_written: rendered_frames,
            sample_rate,
            elapsed_secs,
            realtime_multiplier: 0.0,
            cancelled: true,
        });
    }

    let frames_written = wav
        .finalize()
        .map_err(|e| format!("Failed finalizing WAV header: {e}"))?;

    let audio_secs = frames_written as f64 / sample_rate as f64;
    let realtime_multiplier = if elapsed_secs > 1e-6 {
        audio_secs / elapsed_secs
    } else {
        0.0
    };

    Ok(AudioRenderReport {
        output_path: req.output_path.clone(),
        frames_written,
        sample_rate,
        elapsed_secs,
        realtime_multiplier,
        cancelled: false,
    })
}
