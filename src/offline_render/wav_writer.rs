//! Zero-external-dependency RIFF/WAVE streaming writer for offline audio export.
//!
//! Supports:
//! - 32-bit IEEE Float WAV (`wFormatTag = 3`, with standard `fact` chunk) — default lossless headroom
//! - 16-bit Signed PCM WAV (`wFormatTag = 1`, clamped to `[-1.0, 1.0]`)
//!
//! Pre-allocates its byte conversion staging buffer so writing sample blocks performs
//! zero heap allocations in the render loop.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::settings::WavSampleFormat;

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const STEREO_CHANNELS: u16 = 2;
const IO_BUF_CAPACITY: usize = 256 * 1024;

pub struct WavStreamWriter {
    writer: BufWriter<File>,
    format: WavSampleFormat,
    data_bytes_written: u64,
    frames_written: u64,
    fact_frames_offset: Option<u64>,
    data_size_offset: u64,
    staging: Vec<u8>,
}

impl WavStreamWriter {
    /// Create a new stereo WAV file at `path` and write placeholder RIFF headers.
    pub fn create(
        path: &Path,
        sample_rate: u32,
        format: WavSampleFormat,
        block_frames: usize,
    ) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

        let (format_tag, bits_per_sample, bytes_per_sample): (u16, u16, u16) = match format {
            WavSampleFormat::Float32 => (WAVE_FORMAT_IEEE_FLOAT, 32, 4),
            WavSampleFormat::Pcm16 => (WAVE_FORMAT_PCM, 16, 2),
        };
        let block_align = STEREO_CHANNELS * bytes_per_sample;
        let byte_rate = sample_rate.saturating_mul(block_align as u32);

        // RIFF header (12 bytes)
        writer.write_all(b"RIFF")?;
        writer.write_all(&0u32.to_le_bytes())?; // patched on finalize
        writer.write_all(b"WAVE")?;

        // fmt chunk (24 bytes for PCM16, 26 bytes with cbSize=0 for IEEE Float)
        let fmt_chunk_size: u32 = match format {
            WavSampleFormat::Float32 => 18,
            WavSampleFormat::Pcm16 => 16,
        };
        writer.write_all(b"fmt ")?;
        writer.write_all(&fmt_chunk_size.to_le_bytes())?;
        writer.write_all(&format_tag.to_le_bytes())?;
        writer.write_all(&STEREO_CHANNELS.to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&byte_rate.to_le_bytes())?;
        writer.write_all(&block_align.to_le_bytes())?;
        writer.write_all(&bits_per_sample.to_le_bytes())?;
        if matches!(format, WavSampleFormat::Float32) {
            writer.write_all(&0u16.to_le_bytes())?; // cbSize = 0
        }

        // Standard `fact` chunk required by RIFF spec for non-PCM (IEEE Float)
        let fact_frames_offset = if matches!(format, WavSampleFormat::Float32) {
            writer.write_all(b"fact")?;
            writer.write_all(&4u32.to_le_bytes())?;
            let offset = 12 + 8 + (fmt_chunk_size as u64) + 8;
            writer.write_all(&0u32.to_le_bytes())?; // patched on finalize
            Some(offset)
        } else {
            None
        };

        // data chunk header (8 bytes)
        writer.write_all(b"data")?;
        let data_size_offset = match fact_frames_offset {
            Some(off) => off + 4 + 4,
            None => 12 + 8 + (fmt_chunk_size as u64) + 4,
        };
        writer.write_all(&0u32.to_le_bytes())?; // patched on finalize

        let staging_bytes = block_frames
            .max(512)
            .saturating_mul(block_align as usize);

        Ok(Self {
            writer,
            format,
            data_bytes_written: 0,
            frames_written: 0,
            fact_frames_offset,
            data_size_offset,
            staging: Vec::with_capacity(staging_bytes),
        })
    }

    /// Encode and stream an interleaved stereo `f32` slice (`L, R, L, R, ...`) into the WAV file.
    pub fn write_interleaved_stereo(&mut self, samples: &[f32]) -> io::Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        debug_assert_eq!(samples.len() % 2, 0);

        self.staging.clear();
        match self.format {
            WavSampleFormat::Float32 => {
                self.staging.reserve(samples.len() * 4);
                for &s in samples {
                    let clean = if s.is_finite() { s } else { 0.0 };
                    self.staging.extend_from_slice(&clean.to_le_bytes());
                }
            }
            WavSampleFormat::Pcm16 => {
                self.staging.reserve(samples.len() * 2);
                for &s in samples {
                    let clamped = if s.is_finite() {
                        s.clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    let q = (clamped * 32767.0).round() as i16;
                    self.staging.extend_from_slice(&q.to_le_bytes());
                }
            }
        }

        self.writer.write_all(&self.staging)?;
        self.data_bytes_written += self.staging.len() as u64;
        self.frames_written += (samples.len() / 2) as u64;
        Ok(())
    }

    /// Flush buffered audio bytes and patch RIFF/fact/data header sizes.
    pub fn finalize(mut self) -> io::Result<u64> {
        self.writer.flush()?;
        let mut file = self.writer.into_inner().map_err(|e| e.into_error())?;

        let riff_payload_size = (self.data_size_offset + 4 - 8)
            .saturating_add(self.data_bytes_written)
            .min(u32::MAX as u64) as u32;
        let data_size_u32 = self.data_bytes_written.min(u32::MAX as u64) as u32;

        file.seek(SeekFrom::Start(4))?;
        file.write_all(&riff_payload_size.to_le_bytes())?;

        if let Some(fact_off) = self.fact_frames_offset {
            let frames_u32 = self.frames_written.min(u32::MAX as u64) as u32;
            file.seek(SeekFrom::Start(fact_off))?;
            file.write_all(&frames_u32.to_le_bytes())?;
        }

        file.seek(SeekFrom::Start(self.data_size_offset))?;
        file.write_all(&data_size_u32.to_le_bytes())?;
        file.flush()?;
        Ok(self.frames_written)
    }
}
