//! Lockstep end-to-end Native Vulkan Video + KDMAPI v3 AAC `.mp4` offline render pipeline.
//!
//! Pipeline (100% in-process, 0% FFmpeg CLI, 0 CPU raw pixel readback):
//! 1. Zenith fixed-timestep headless Vulkan offscreen note + keyboard rendering (`video_scene`)
//! 2. GPU VRAM Compute RGBA -> NV12 BT.709 conversion (`nv12_compute`)
//! 3. Hardware `VK_KHR_video_encode_queue` H.264 / H.265 encoding (`vulkan_video`)
//! 4. Deterministic `engine_core` (`kdmapi v3`) audio -> pure-Rust AAC-LC (`aac_encoder`)
//! 5. Pure-Rust 64-bit `co64` ISO-BMFF `.mp4` muxing (`mp4_muxer`)

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
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
    KdmapiSettings, MidiSettings, RenderSettings, SceneSettings, MAX_RENDER_DIMENSION,
    MAX_RENDER_FPS, MAX_RENDER_SAMPLE_RATE, MIN_RENDER_DIMENSION, MIN_RENDER_FPS,
    MIN_RENDER_SAMPLE_RATE,
};

use super::aac_encoder::{AacAudioEncoder, DEFAULT_AAC_BITRATE_BPS};
use super::job::OfflineRenderProgress;
use super::mp4_muxer::Mp4Muxer;
use super::nv12_compute::Nv12ComputeConverter;
use super::video_scene::{OfflineMidiNoteTimeline, OffscreenSceneRenderer};
use super::vk_context::OfflineVulkanContext;
use super::vulkan_video::{VideoEncodeConfig, VulkanVideoEncoder};
use super::OfflineAudioRenderer;

const WASABI_DEFAULT_TEMPO_US: u32 = 250_000;
const MAX_VIDEO_TAIL_SECONDS: f64 = 5.0;

#[derive(Debug, Clone)]
pub struct VideoRenderRequest {
    pub midi_path: PathBuf,
    pub render_settings: RenderSettings,
    pub midi_settings: MidiSettings,
    pub scene_settings: SceneSettings,
    pub kdmapi_settings: KdmapiSettings,
    pub soundfonts: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VideoRenderReport {
    pub output_path: PathBuf,
    pub video_frames_written: u64,
    pub audio_samples_written: u64,
    pub fps: u32,
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

/// Deterministic fixed-timestep Native Vulkan Video + KDMAPI v3 AAC `.mp4` renderer.
pub fn render_midi_to_mp4(
    req: &VideoRenderRequest,
    progress: &OfflineRenderProgress,
) -> Result<VideoRenderReport, String> {
    if !req.midi_path.exists() {
        return Err(format!(
            "MIDI file does not exist: {}",
            req.midi_path.display()
        ));
    }
    if req.render_settings.output_path.as_os_str().is_empty() {
        return Err("Output path cannot be empty.".into());
    }

    let width = req
        .render_settings
        .width
        .clamp(MIN_RENDER_DIMENSION, MAX_RENDER_DIMENSION);
    let height = req
        .render_settings
        .height
        .clamp(MIN_RENDER_DIMENSION, MAX_RENDER_DIMENSION);
    let coded_w = (width + 1) & !1;
    let coded_h = (height + 1) & !1;
    let fps = req
        .render_settings
        .fps
        .clamp(MIN_RENDER_FPS, MAX_RENDER_FPS);
    let sample_rate = req
        .render_settings
        .sample_rate
        .clamp(MIN_RENDER_SAMPLE_RATE, MAX_RENDER_SAMPLE_RATE);
    let block_frames = req
        .kdmapi_settings
        .buffer_frames
        .clamp(MIN_BUFFER_FRAMES, MAX_BUFFER_FRAMES) as usize;

    // 1. Parse deterministic MIDI note timeline for fixed-timestep video rendering
    let timeline = OfflineMidiNoteTimeline::parse_from_file(&req.midi_path, &req.midi_settings)?;
    let midi_video_frames = ((timeline.duration_sec.max(0.0) * fps as f64).ceil() as u64).max(1);
    progress.begin(fps, midi_video_frames);

    // 2. Initialize isolated offline Vulkan 1.3 Video context, offscreen scene, NV12 compute & HW encoder
    let vk_ctx = Arc::new(OfflineVulkanContext::new(req.render_settings.video_codec)?);
    let mut scene_renderer = OffscreenSceneRenderer::new(vk_ctx.clone(), coded_w, coded_h)?;
    let nv12_conv = Nv12ComputeConverter::new(vk_ctx.clone(), coded_w, coded_h)?;
    let mut video_encoder = VulkanVideoEncoder::new(
        vk_ctx.clone(),
        &VideoEncodeConfig {
            codec: req.render_settings.video_codec,
            width: coded_w,
            height: coded_h,
            fps,
            rate_control: req.render_settings.rate_control,
            bitrate_kbps: req.render_settings.bitrate_kbps,
            qp: req.render_settings.qp,
        },
    )?;

    // 3. Initialize deterministic KDMAPI v3 audio synth + in-process AAC-LC encoder
    let mut audio_renderer =
        OfflineAudioRenderer::new(&req.kdmapi_settings, sample_rate, &req.soundfonts)
            .map_err(|e| format!("Failed initializing offline KDMAPI v3 synth: {e}"))?;
    let mut aac_encoder = AacAudioEncoder::new(sample_rate, DEFAULT_AAC_BITRATE_BPS)?;

    // 4. Initialize pure-Rust 64-bit `co64` MP4 muxer & seed SPS/PPS(/VPS)
    let mut muxer = Mp4Muxer::create(
        &req.render_settings.output_path,
        req.render_settings.video_codec,
        coded_w,
        coded_h,
        fps,
        aac_encoder.sample_rate(),
        2,
        aac_encoder.bitrate_bps(),
        aac_encoder.audio_specific_config().to_vec(),
    )
    .map_err(|e| {
        format!(
            "Failed creating MP4 output file {}: {e}",
            req.render_settings.output_path.display()
        )
    })?;
    muxer.set_parameter_sets_annexb(video_encoder.parameter_sets_annexb());

    // 5. Open streaming MIDI event iterator for lockstep sample-accurate audio synthesis
    let audio_midi_file = File::open(&req.midi_path)
        .map_err(|e| format!("Failed opening MIDI for audio track: {e}"))?;
    let audio_midi = TKMIDIFile::open_from_stream(audio_midi_file, None)
        .map_err(|e| format!("Failed parsing MIDI for audio track: {e:?}"))?;
    let ppq = audio_midi.ppq();
    let mut audio_iter = pipe!(
        audio_midi.iter_all_track_events_merged_batches()
        |> TimeCaster::<f64>::cast_event_delta()
        |> cancel_tempo_events(WASABI_DEFAULT_TEMPO_US)
        |> scale_event_time(1.0 / ppq as f64)
        |> unwrap_items()
    )
    .peekable();

    let mut midi_audio_time_sec: f64 = 0.0;
    let mut next_batch_sample: Option<u64> = None;
    let mut rendered_audio_samples: u64 = 0;
    let mut stereo_buf = vec![0.0f32; block_frames * 2];

    let mut video_frames_written: u64 = 0;
    let max_tail_frames = (MAX_VIDEO_TAIL_SECONDS * fps as f64).round() as u64;
    let start_instant = Instant::now();
    let mut cancelled = false;

    // Lockstep video+audio loop (exact Zenith fixed timestep `t = frame_idx / fps`)
    let mut frame_idx: u64 = 0;
    loop {
        if progress.is_cancel_requested() {
            cancelled = true;
            break;
        }

        let in_tail = frame_idx >= midi_video_frames;
        if in_tail {
            let tail_elapsed = frame_idx - midi_video_frames;
            if audio_renderer.active_voices() == 0 || tail_elapsed >= max_tail_frames {
                break;
            }
        }

        let frame_time_sec = frame_idx as f64 / fps as f64;

        // A. Render video frame offscreen (VRAM) -> Compute NV12 (VRAM) -> Vulkan Video Encode
        scene_renderer.render_frame(&timeline, frame_time_sec, &req.scene_settings)?;
        nv12_conv.convert_to_nv12(
            scene_renderer.color_view(),
            video_encoder.src_nv12_image(),
            video_encoder.src_nv12_layout(),
        )?;
        video_encoder.mark_src_ready_for_encode();

        let (nal_bytes, is_idr) = video_encoder.encode_frame(frame_idx)?;
        muxer
            .write_video_frame_annexb(nal_bytes, is_idr)
            .map_err(|e| format!("Failed writing video frame #{frame_idx} to MP4: {e}"))?;
        video_frames_written += 1;

        // B. Advance deterministic KDMAPI v3 audio + AAC-LC to exact end of this video frame
        let frame_end_sample = ((frame_idx + 1).saturating_mul(sample_rate as u64)) / (fps as u64);
        while rendered_audio_samples < frame_end_sample {
            // Determine next MIDI batch timestamp if needed
            if next_batch_sample.is_none() {
                if let Some(batch) = audio_iter.peek() {
                    let t = midi_audio_time_sec + batch.delta;
                    next_batch_sample = Some((t.max(0.0) * sample_rate as f64).round() as u64);
                }
            }

            // Dispatch all MIDI batches due at or before `rendered_audio_samples`
            while let Some(due_sample) = next_batch_sample {
                if due_sample > rendered_audio_samples {
                    break;
                }
                if let Some(batch) = audio_iter.next() {
                    midi_audio_time_sec += batch.delta;
                    for ev in batch.iter_events() {
                        if let Some(dw) = encode_midi_short_msg(ev.as_event()) {
                            audio_renderer.push_short_msg(dw);
                        }
                    }
                }
                next_batch_sample = audio_iter.peek().map(|b| {
                    let t = midi_audio_time_sec + b.delta;
                    (t.max(0.0) * sample_rate as f64).round() as u64
                });
            }

            let limit = match next_batch_sample {
                Some(due) => due.min(frame_end_sample),
                None => frame_end_sample,
            };
            let to_render = (limit - rendered_audio_samples).min(block_frames as u64) as usize;
            if to_render == 0 {
                continue;
            }

            let slice = &mut stereo_buf[..to_render * 2];
            audio_renderer.render_stereo_block(slice);
            aac_encoder.push_interleaved_f32(slice, |pkt, pcm_frames| {
                muxer.write_aac_frame(pkt, pcm_frames)
            })?;
            rendered_audio_samples += to_render as u64;
        }

        frame_idx += 1;
        progress.update_metrics(
            video_frames_written,
            start_instant.elapsed().as_micros() as u64,
            audio_renderer.active_voices(),
        );
    }

    let elapsed_secs = start_instant.elapsed().as_secs_f64();

    if cancelled {
        drop(muxer);
        let _ = std::fs::remove_file(&req.render_settings.output_path);
        return Ok(VideoRenderReport {
            output_path: req.render_settings.output_path.clone(),
            video_frames_written,
            audio_samples_written: rendered_audio_samples,
            fps,
            elapsed_secs,
            realtime_multiplier: 0.0,
            cancelled: true,
        });
    }

    // Flush remaining AAC encoder frames & finalize 64-bit MP4 `moov`
    aac_encoder.finish(|pkt, pcm_frames| muxer.write_aac_frame(pkt, pcm_frames))?;
    muxer
        .finalize()
        .map_err(|e| format!("Failed finalizing MP4 container: {e}"))?;

    let video_secs = video_frames_written as f64 / fps as f64;
    let realtime_multiplier = if elapsed_secs > 1e-6 {
        video_secs / elapsed_secs
    } else {
        0.0
    };

    Ok(VideoRenderReport {
        output_path: req.render_settings.output_path.clone(),
        video_frames_written,
        audio_samples_written: rendered_audio_samples,
        fps,
        elapsed_secs,
        realtime_multiplier,
        cancelled: false,
    })
}
