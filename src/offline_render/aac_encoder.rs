//! In-process pure-Rust AAC-LC (`mp4a`) encoder backed by `fdk-aac-rs`.
//!
//! Zero external FFmpeg CLI, zero OS `libfdk-aac-dev` dependency, pre-allocated
//! PCM/bitstream scratch buffers for steady offline throughput.

use std::io;

use fdk_aac_rs::encoder::{Aot, Bitrate, ChannelMode, Encoder, EncoderError, Transport};

/// Standard MPEG-4 AAC-LC supported sample rates (Hz).
const SUPPORTED_AAC_RATES: [u32; 12] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 64_000, 88_200, 96_000,
];

/// Default stereo AAC-LC bitrate for rendered `.mp4` videos (256 kbps).
pub const DEFAULT_AAC_BITRATE_BPS: u32 = 256_000;

pub struct AacAudioEncoder {
    encoder: Encoder,
    sample_rate: u32,
    bitrate_bps: u32,
    frame_length: u32,
    audio_specific_config: Vec<u8>,
    pcm_i16_scratch: Vec<i16>,
    out_buf: Vec<u8>,
}

impl AacAudioEncoder {
    /// Initialize a stereo AAC-LC raw-access-unit encoder for MP4 (`mp4a`) muxing.
    pub fn new(sample_rate: u32, bitrate_bps: u32) -> Result<Self, String> {
        if !SUPPORTED_AAC_RATES.contains(&sample_rate) {
            return Err(format!(
                "Sample rate {sample_rate} Hz is not supported by MPEG-4 AAC-LC. Choose one of: 44100, 48000, 96000 (or another standard MPEG-4 rate)."
            ));
        }

        let bitrate = bitrate_bps.clamp(64_000, 512_000);
        let mut encoder = Encoder::builder()
            .aot(Aot::AacLc)
            .sample_rate(sample_rate)
            .channel_mode(ChannelMode::Stereo)
            .bitrate(Bitrate::Cbr(bitrate))
            .transport(Transport::Raw)
            .afterburner(true)
            .build()
            .map_err(|e| format!("Failed to initialize AAC-LC encoder: {e}"))?;

        let info = encoder
            .info()
            .map_err(|e| format!("Failed to query AAC-LC encoder info: {e}"))?;
        let frame_length = info.frame_length();
        let max_out = info.max_out_buf_bytes().max(8192) as usize;
        let audio_specific_config = info.config().to_vec();
        if audio_specific_config.is_empty() {
            return Err("AAC-LC encoder returned empty AudioSpecificConfig.".into());
        }

        Ok(Self {
            encoder,
            sample_rate,
            bitrate_bps: bitrate,
            frame_length,
            audio_specific_config,
            pcm_i16_scratch: Vec::with_capacity((frame_length as usize) * 4),
            out_buf: vec![0u8; max_out],
        })
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub fn bitrate_bps(&self) -> u32 {
        self.bitrate_bps
    }

    #[inline]
    #[allow(dead_code)]
    pub fn frame_length(&self) -> u32 {
        self.frame_length
    }

    #[inline]
    pub fn audio_specific_config(&self) -> &[u8] {
        &self.audio_specific_config
    }

    /// Encode interleaved stereo `f32` audio samples, invoking `on_packet(raw_aac, pcm_frames)`
    /// for each emitted raw AAC-LC access unit.
    pub fn push_interleaved_f32(
        &mut self,
        stereo_f32: &[f32],
        mut on_packet: impl FnMut(&[u8], u32) -> io::Result<()>,
    ) -> Result<(), String> {
        if stereo_f32.is_empty() {
            return Ok(());
        }

        self.pcm_i16_scratch.clear();
        self.pcm_i16_scratch.reserve(stereo_f32.len());
        for &s in stereo_f32 {
            let clamped = if s.is_finite() {
                s.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.pcm_i16_scratch.push((clamped * 32767.0).round() as i16);
        }

        let mut offset = 0usize;
        while offset < self.pcm_i16_scratch.len() {
            let res = self
                .encoder
                .encode(&self.pcm_i16_scratch[offset..], &mut self.out_buf)
                .map_err(|e| format!("AAC-LC encode error: {e}"))?;

            if res.out_bytes() > 0 {
                on_packet(&self.out_buf[..res.out_bytes()], self.frame_length)
                    .map_err(|e| format!("Failed writing AAC packet: {e}"))?;
            }

            let consumed = res.in_samples();
            if consumed == 0 {
                break;
            }
            offset += consumed;
        }
        Ok(())
    }

    /// Flush remaining buffered AAC-LC frames at end of stream.
    pub fn finish(
        &mut self,
        mut on_packet: impl FnMut(&[u8], u32) -> io::Result<()>,
    ) -> Result<(), String> {
        loop {
            match self.encoder.flush(&mut self.out_buf) {
                Ok(res) => {
                    if res.out_bytes() > 0 {
                        on_packet(&self.out_buf[..res.out_bytes()], self.frame_length)
                            .map_err(|e| format!("Failed writing flushed AAC packet: {e}"))?;
                    }
                }
                Err(EncoderError::EncodeEof) => break,
                Err(e) => return Err(format!("AAC-LC flush error: {e}")),
            }
        }
        Ok(())
    }
}
