//! Isolated offline rendering boundary (KMC deterministic audio + Zenith fixed-timestep Vulkan Video).
//!
//! # Module Boundary Rules (`AGENTS.md` #5)
//! - Strictly separated from real-time preview (`crate::audio_playback`): `audio_playback`
//!   must NEVER import this module, so offline file I/O, WAV/MP4 muxing, and video encoding
//!   can never leak onto the real-time audio callback or playback hot path.
//! - Shares the exact same `kdmapiv3::SoundFontSynth` DSP core via `crate::engine_core`.
//! - Phase 2 Native Vulkan Video `.mp4` muxer will pair hardware H.264/H.265 video with
//!   AAC (`mp4a`) audio for universal playback compatibility, while Audio Only `.wav`
//!   provides lossless 32-bit Float / 16-bit PCM output.

pub mod aac_encoder;
pub mod audio_render;
pub mod gpu_probe;
pub mod job;
pub mod mp4_muxer;
pub mod nv12_compute;
pub mod video_render;
pub mod video_scene;
pub mod vk_context;
pub mod vulkan_video;
pub mod wav_writer;

pub use audio_render::AudioRenderRequest;
pub use gpu_probe::GpuVideoCapabilities;
pub use job::{OfflineRenderManager, RenderJobPhase};
pub use video_render::VideoRenderRequest;

use std::path::PathBuf;

use kdmapiv3::{BackendError, SoundFontSynth, SynthBackend};

use crate::engine_core::create_shared_synth;
use crate::settings::{KdmapiSettings, MAX_RENDER_SAMPLE_RATE, MIN_RENDER_SAMPLE_RATE};

/// Deterministic, non-real-time audio renderer backed directly by `kdmapiv3::SoundFontSynth`.
pub struct OfflineAudioRenderer {
    synth: SoundFontSynth,
    sample_rate: u32,
}

impl OfflineAudioRenderer {
    /// Create a deterministic offline renderer using the same `KdmapiSettings` and SoundFonts.
    pub fn new(
        settings: &KdmapiSettings,
        sample_rate: u32,
        soundfonts: &[PathBuf],
    ) -> Result<Self, BackendError> {
        let clamped_sr = sample_rate.clamp(MIN_RENDER_SAMPLE_RATE, MAX_RENDER_SAMPLE_RATE);
        let synth = create_shared_synth(settings, clamped_sr, soundfonts)?;
        Ok(Self {
            synth,
            sample_rate: clamped_sr,
        })
    }

    #[inline]
    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub fn push_short_msg(&mut self, dw_msg: u32) {
        self.synth.process_short_event(dw_msg);
    }

    #[inline]
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.synth.reset();
    }

    /// Render interleaved stereo (`L, R, L, R, ...`) samples into the caller-provided buffer.
    #[inline]
    pub fn render_stereo_block(&mut self, interleaved_out: &mut [f32]) {
        self.synth.render_audio(interleaved_out);
    }

    #[inline]
    pub fn active_voices(&self) -> u64 {
        self.synth.stats().active_voices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offline_render::audio_render::render_midi_to_wav;
    use crate::offline_render::job::OfflineRenderProgress;
    use crate::offline_render::wav_writer::WavStreamWriter;
    use crate::settings::WavSampleFormat;
    use std::io::Write;

    /// Build a minimal valid Standard MIDI File (Format 0, 480 PPQ, 120 BPM, C4 NoteOn -> 0.5s -> NoteOff).
    fn build_minimal_test_midi() -> Vec<u8> {
        let mut track: Vec<u8> = Vec::new();
        // Delta 0: Set Tempo 500_000 us/quarter (120 BPM)
        track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        // Delta 0: Note On ch0 key=60 vel=100
        track.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // Delta 480 (VLQ: 0x83 0x60) = 1 quarter note = 0.5s: Note Off ch0 key=60 vel=0
        track.extend_from_slice(&[0x83, 0x60, 0x80, 0x3C, 0x00]);
        // Delta 0: End of Track
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut smf = Vec::new();
        // MThd header: len=6, format=0, tracks=1, division=480
        smf.extend_from_slice(b"MThd");
        smf.extend_from_slice(&6u32.to_be_bytes());
        smf.extend_from_slice(&0u16.to_be_bytes());
        smf.extend_from_slice(&1u16.to_be_bytes());
        smf.extend_from_slice(&480u16.to_be_bytes());
        // MTrk chunk
        smf.extend_from_slice(b"MTrk");
        smf.extend_from_slice(&(track.len() as u32).to_be_bytes());
        smf.extend_from_slice(&track);
        smf
    }

    #[test]
    fn offline_renderer_initializes_and_renders_deterministically_with_bypass() {
        let settings = KdmapiSettings {
            voice_limit: 256,
            buffer_frames: 256,
            render_threads: 1,
            disable_all_effects: true,
            ..KdmapiSettings::default()
        };
        let mut renderer =
            OfflineAudioRenderer::new(&settings, 48_000, &[]).expect("create offline renderer");
        assert_eq!(renderer.sample_rate(), 48_000);

        renderer.push_short_msg(0x007F_3C90);
        let mut out = [0.0f32; 512];
        renderer.render_stereo_block(&mut out);
        assert!(out.iter().all(|s| s.is_finite()));

        renderer.reset();
        assert_eq!(renderer.active_voices(), 0);
    }

    #[test]
    fn wav_writer_float32_and_pcm16_headers_and_payloads_are_valid() {
        let tmp = std::env::temp_dir();
        let f32_path = tmp.join("wanith_test_float32.wav");
        let pcm16_path = tmp.join("wanith_test_pcm16.wav");

        let samples = [0.25f32, -0.25, 0.5, -0.5, 1.5, -1.5, 0.0, 0.0]; // 4 stereo frames

        {
            let mut w = WavStreamWriter::create(&f32_path, 48_000, WavSampleFormat::Float32, 64)
                .expect("create float32 wav");
            w.write_interleaved_stereo(&samples).expect("write float32");
            assert_eq!(w.finalize().expect("finalize float32"), 4);
        }
        {
            let mut w = WavStreamWriter::create(&pcm16_path, 44_100, WavSampleFormat::Pcm16, 64)
                .expect("create pcm16 wav");
            w.write_interleaved_stereo(&samples).expect("write pcm16");
            assert_eq!(w.finalize().expect("finalize pcm16"), 4);
        }

        let f32_bytes = std::fs::read(&f32_path).expect("read float32");
        assert_eq!(&f32_bytes[0..4], b"RIFF");
        assert_eq!(&f32_bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([f32_bytes[20], f32_bytes[21]]), 3); // IEEE Float
        assert_eq!(u16::from_le_bytes([f32_bytes[22], f32_bytes[23]]), 2); // Stereo
        assert_eq!(
            u32::from_le_bytes(f32_bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(&f32_bytes[38..42], b"fact");
        assert_eq!(
            u32::from_le_bytes(f32_bytes[46..50].try_into().unwrap()),
            4
        );
        assert_eq!(&f32_bytes[50..54], b"data");
        assert_eq!(
            u32::from_le_bytes(f32_bytes[54..58].try_into().unwrap()),
            4 * 8
        );
        assert_eq!(f32_bytes.len(), 58 + 32);

        let pcm_bytes = std::fs::read(&pcm16_path).expect("read pcm16");
        assert_eq!(&pcm_bytes[0..4], b"RIFF");
        assert_eq!(u16::from_le_bytes([pcm_bytes[20], pcm_bytes[21]]), 1); // PCM
        assert_eq!(
            u32::from_le_bytes(pcm_bytes[24..28].try_into().unwrap()),
            44_100
        );
        assert_eq!(&pcm_bytes[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(pcm_bytes[40..44].try_into().unwrap()),
            4 * 4
        );
        assert_eq!(pcm_bytes.len(), 44 + 16);

        let _ = std::fs::remove_file(&f32_path);
        let _ = std::fs::remove_file(&pcm16_path);
    }

    #[test]
    fn kmc_deterministic_midi_to_wav_end_to_end_float32_and_pcm16() {
        let tmp = std::env::temp_dir();
        let midi_path = tmp.join("wanith_offline_test.mid");
        let wav_f32_path = tmp.join("wanith_offline_test_f32.wav");
        let wav_i16_path = tmp.join("wanith_offline_test_i16.wav");

        {
            let mut f = std::fs::File::create(&midi_path).expect("create test midi");
            f.write_all(&build_minimal_test_midi())
                .expect("write test midi");
        }

        let kdmapi_settings = KdmapiSettings {
            voice_limit: 512,
            buffer_frames: 256,
            render_threads: 2,
            disable_all_effects: true,
            ..KdmapiSettings::default()
        };

        for (fmt, out_path, sample_rate) in [
            (WavSampleFormat::Float32, &wav_f32_path, 48_000u32),
            (WavSampleFormat::Pcm16, &wav_i16_path, 44_100u32),
        ] {
            let req = AudioRenderRequest {
                midi_path: midi_path.clone(),
                output_path: out_path.clone(),
                sample_rate,
                wav_format: fmt,
                kdmapi_settings: kdmapi_settings.clone(),
                soundfonts: Vec::new(),
                estimated_duration_sec: Some(0.5),
            };
            let progress = OfflineRenderProgress::new();
            progress.begin(sample_rate, (0.5 * sample_rate as f64) as u64);

            let report = render_midi_to_wav(&req, &progress).expect("render midi to wav");
            assert!(!report.cancelled);
            // 0.5s at sample_rate (+/- tail frames)
            assert!(report.frames_written >= (sample_rate / 2) as u64);
            assert!(report.realtime_multiplier > 0.0);

            let bytes = std::fs::read(out_path).expect("read rendered wav");
            assert_eq!(&bytes[0..4], b"RIFF");
            assert_eq!(&bytes[8..12], b"WAVE");
        }

        let _ = std::fs::remove_file(&midi_path);
        let _ = std::fs::remove_file(&wav_f32_path);
        let _ = std::fs::remove_file(&wav_i16_path);
    }

    #[test]
    fn native_vulkan_video_h264_and_h265_mp4_end_to_end() {
        use crate::offline_render::video_render::render_midi_to_mp4;
        use crate::settings::{MidiSettings, RenderSettings, SceneSettings, VideoCodec, VideoRateControl};

        let tmp = std::env::temp_dir();
        let midi_path = tmp.join("wanith_vulkan_video_test.mid");
        {
            let mut f = std::fs::File::create(&midi_path).expect("create test midi");
            f.write_all(&build_minimal_test_midi())
                .expect("write test midi");
        }

        for (codec, fname, sample_entry_tag, cfg_box_tag) in [
            (VideoCodec::H264, "wanith_test_h264.mp4", b"avc1", b"avcC"),
            (VideoCodec::H265, "wanith_test_h265.mp4", b"hvc1", b"hvcC"),
        ] {
            let out_path = tmp.join(fname);
            let req = VideoRenderRequest {
                midi_path: midi_path.clone(),
                render_settings: RenderSettings {
                    output_path: out_path.clone(),
                    sample_rate: 48_000,
                    width: 640,
                    height: 360,
                    fps: 60,
                    video_codec: codec,
                    rate_control: VideoRateControl::Bitrate,
                    bitrate_kbps: 4_000,
                    qp: 22,
                    ..RenderSettings::default()
                },
                midi_settings: MidiSettings::default(),
                scene_settings: SceneSettings::default(),
                kdmapi_settings: KdmapiSettings {
                    voice_limit: 256,
                    buffer_frames: 256,
                    render_threads: 2,
                    disable_all_effects: true,
                    ..KdmapiSettings::default()
                },
                soundfonts: Vec::new(),
            };

            let progress = OfflineRenderProgress::new();
            let report = render_midi_to_mp4(&req, &progress)
                .unwrap_or_else(|e| panic!("Native Vulkan Video ({codec:?}) render failed: {e}"));
            assert!(!report.cancelled);
            assert!(
                report.video_frames_written >= 30,
                "expected >=30 frames for 0.5s@60fps, got {}",
                report.video_frames_written
            );
            assert!(
                report.audio_samples_written >= 24_000,
                "expected >=24000 audio samples, got {}",
                report.audio_samples_written
            );

            let bytes = std::fs::read(&out_path).expect("read rendered mp4");
            assert!(bytes.len() > 2048, "MP4 too small: {} bytes", bytes.len());

            // Strict top-level ISO-BMFF box walk: ftyp -> 64-bit mdat -> moov == exact EOF
            let mut pos = 0usize;
            let mut top_boxes: Vec<([u8; 4], usize, usize)> = Vec::new();
            while pos + 8 <= bytes.len() {
                let sz32 = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                let fourcc: [u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap();
                let (hdr_len, box_len) = if sz32 == 1 {
                    let sz64 =
                        u64::from_be_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
                    (16usize, sz64)
                } else {
                    (8usize, sz32)
                };
                assert!(
                    box_len >= hdr_len && pos + box_len <= bytes.len(),
                    "corrupt MP4 box {:?} at {}: len={} file={}",
                    std::str::from_utf8(&fourcc),
                    pos,
                    box_len,
                    bytes.len()
                );
                top_boxes.push((fourcc, pos + hdr_len, pos + box_len));
                pos += box_len;
            }
            assert_eq!(pos, bytes.len(), "trailing unboxed bytes in MP4");
            assert_eq!(top_boxes.len(), 3, "expected [ftyp, mdat, moov]");
            assert_eq!(&top_boxes[0].0, b"ftyp");
            assert_eq!(&top_boxes[1].0, b"mdat");
            assert_eq!(&top_boxes[2].0, b"moov");
            let mdat_payload_len = top_boxes[1].2 - top_boxes[1].1;
            assert!(
                mdat_payload_len > 1024,
                "mdat payload too small: {mdat_payload_len}"
            );

            let moov_bytes = &bytes[top_boxes[2].1..top_boxes[2].2];
            assert!(
                moov_bytes.windows(4).any(|w| w == sample_entry_tag),
                "missing video sample entry in moov"
            );
            assert!(
                moov_bytes.windows(4).any(|w| w == cfg_box_tag),
                "missing video decoder config box in moov"
            );
            assert!(
                moov_bytes.windows(4).any(|w| w == b"mp4a"),
                "missing AAC mp4a track in moov"
            );
            assert!(
                moov_bytes.windows(4).any(|w| w == b"esds"),
                "missing AAC esds box in moov"
            );
            assert!(
                moov_bytes.windows(4).filter(|w| *w == b"co64").count() == 2,
                "expected 2 co64 boxes (video + audio)"
            );

            if std::env::var_os("WANITH_KEEP_TEST_MP4").is_none() {
                let _ = std::fs::remove_file(&out_path);
            }
        }

        let _ = std::fs::remove_file(&midi_path);
    }
}
