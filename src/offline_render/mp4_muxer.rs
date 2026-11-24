//! Pure-Rust ISO-BMFF (`.mp4`) streaming muxer with 64-bit `mdat` and `co64` chunk offsets.
//!
//! # Capabilities
//! - Zero external CLI / zero external muxer crate dependencies.
//! - Supports H.264 (`avc1` + `avcC`) and H.265 / HEVC (`hvc1` + `hvcC`) Annex-B bitstreams
//!   directly from Vulkan Video (`VK_KHR_video_encode_queue`).
//! - Supports AAC-LC (`mp4a` + `esds` with `AudioSpecificConfig`) from `fdk-aac-rs`.
//! - Uses 64-bit `mdat` extended size and `co64` chunk offsets so >4 GiB high-FPS 4K Black MIDI
//!   renders never overflow 32-bit MP4 box limits.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::settings::VideoCodec;

const IO_BUF_CAPACITY: usize = 256 * 1024;
const MOVIE_TIMESCALE: u32 = 90_000;

#[derive(Debug, Clone, Copy)]
struct SampleRecord {
    offset: u64,
    size: u32,
    duration: u32,
    is_sync: bool,
}

/// Split an Annex-B byte stream (`00 00 01` or `00 00 00 01` start codes) into raw NAL units.
pub fn split_annexb_nalus(data: &[u8]) -> Vec<&[u8]> {
    let mut nalus = Vec::new();
    let len = data.len();
    let mut i = 0usize;

    // Find start of first start code
    let mut nalu_start: Option<usize> = None;
    while i < len {
        let sc_len = if i + 3 < len
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            4
        } else if i + 2 < len && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            3
        } else {
            0
        };

        if sc_len > 0 {
            if let Some(start) = nalu_start {
                if i > start {
                    nalus.push(&data[start..i]);
                }
            }
            i += sc_len;
            nalu_start = Some(i);
        } else {
            i += 1;
        }
    }

    if let Some(start) = nalu_start {
        if len > start {
            nalus.push(&data[start..len]);
        }
    }

    nalus
}

/// Remove H.264/H.265 emulation prevention bytes (`00 00 03` -> `00 00`) from a NAL RBSP.
fn unescape_rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0u32;
    for &b in nal {
        if zeros >= 2 && b == 0x03 {
            zeros = 0;
            continue;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

pub struct Mp4Muxer {
    writer: BufWriter<File>,
    codec: VideoCodec,
    width: u16,
    height: u16,
    fps: u32,
    video_timescale: u32,
    video_frame_delta: u32,

    audio_sample_rate: u32,
    audio_channels: u16,
    audio_specific_config: Vec<u8>,
    audio_bitrate: u32,

    // Codec parameter sets (raw NAL bytes without Annex-B start codes)
    vps_list: Vec<Vec<u8>>,
    sps_list: Vec<Vec<u8>>,
    pps_list: Vec<Vec<u8>>,

    video_samples: Vec<SampleRecord>,
    audio_samples: Vec<SampleRecord>,

    mdat_header_pos: u64,
    current_pos: u64,
    avcc_scratch: Vec<u8>,
}

impl Mp4Muxer {
    pub fn create(
        path: &Path,
        codec: VideoCodec,
        width: u32,
        height: u32,
        fps: u32,
        audio_sample_rate: u32,
        audio_channels: u16,
        audio_bitrate: u32,
        audio_specific_config: Vec<u8>,
    ) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::with_capacity(IO_BUF_CAPACITY, file);

        // 1. Write `ftyp` box
        let mut ftyp = Vec::with_capacity(32);
        let brand_codec: &[u8; 4] = match codec {
            VideoCodec::H264 => b"avc1",
            VideoCodec::H265 => b"hvc1",
        };
        write_box(&mut ftyp, b"ftyp", |buf| {
            buf.extend_from_slice(b"isom");
            buf.extend_from_slice(&512u32.to_be_bytes());
            buf.extend_from_slice(b"isom");
            buf.extend_from_slice(b"iso2");
            buf.extend_from_slice(brand_codec);
            buf.extend_from_slice(b"mp41");
        });
        writer.write_all(&ftyp)?;

        // 2. Write 64-bit extended `mdat` header (16 bytes: size=1, "mdat", largesize=0 placeholder)
        let mdat_header_pos = ftyp.len() as u64;
        writer.write_all(&1u32.to_be_bytes())?;
        writer.write_all(b"mdat")?;
        writer.write_all(&0u64.to_be_bytes())?;
        let current_pos = mdat_header_pos + 16;

        let fps_safe = fps.max(1);
        let video_timescale = fps_safe.saturating_mul(1_000);
        let video_frame_delta = 1_000;

        Ok(Self {
            writer,
            codec,
            width: width.min(u16::MAX as u32) as u16,
            height: height.min(u16::MAX as u32) as u16,
            fps: fps_safe,
            video_timescale,
            video_frame_delta,
            audio_sample_rate: audio_sample_rate.max(8_000),
            audio_channels: audio_channels.max(1),
            audio_specific_config,
            audio_bitrate,
            vps_list: Vec::new(),
            sps_list: Vec::new(),
            pps_list: Vec::new(),
            video_samples: Vec::new(),
            audio_samples: Vec::new(),
            mdat_header_pos,
            current_pos,
            avcc_scratch: Vec::with_capacity(256 * 1024),
        })
    }

    /// Feed Annex-B parameter set bytes (from `vkGetEncodedVideoSessionParametersKHR` or inline).
    pub fn set_parameter_sets_annexb(&mut self, annexb: &[u8]) {
        for nal in split_annexb_nalus(annexb) {
            self.ingest_parameter_nal(nal);
        }
    }

    fn ingest_parameter_nal(&mut self, nal: &[u8]) -> bool {
        if nal.is_empty() {
            return false;
        }
        match self.codec {
            VideoCodec::H264 => {
                let nal_type = nal[0] & 0x1F;
                match nal_type {
                    7 => {
                        if !self.sps_list.iter().any(|s| s.as_slice() == nal) {
                            self.sps_list.push(nal.to_vec());
                        }
                        true
                    }
                    8 => {
                        if !self.pps_list.iter().any(|p| p.as_slice() == nal) {
                            self.pps_list.push(nal.to_vec());
                        }
                        true
                    }
                    _ => false,
                }
            }
            VideoCodec::H265 => {
                let nal_type = (nal[0] >> 1) & 0x3F;
                match nal_type {
                    32 => {
                        if !self.vps_list.iter().any(|v| v.as_slice() == nal) {
                            self.vps_list.push(nal.to_vec());
                        }
                        true
                    }
                    33 => {
                        if !self.sps_list.iter().any(|s| s.as_slice() == nal) {
                            self.sps_list.push(nal.to_vec());
                        }
                        true
                    }
                    34 => {
                        if !self.pps_list.iter().any(|p| p.as_slice() == nal) {
                            self.pps_list.push(nal.to_vec());
                        }
                        true
                    }
                    _ => false,
                }
            }
        }
    }

    /// Write one encoded Annex-B video frame into `mdat` (converts Annex-B -> 4-byte length-prefixed AVCC/HVCC).
    pub fn write_video_frame_annexb(&mut self, annexb: &[u8], force_sync: bool) -> io::Result<()> {
        self.avcc_scratch.clear();
        let mut is_sync = force_sync;

        for nal in split_annexb_nalus(annexb) {
            if nal.is_empty() {
                continue;
            }
            if self.ingest_parameter_nal(nal) {
                // Parameter sets live in `avcC`/`hvcC` sample entry; skip duplicate inline copies
                continue;
            }
            match self.codec {
                VideoCodec::H264 => {
                    let nal_type = nal[0] & 0x1F;
                    if nal_type == 5 {
                        is_sync = true;
                    }
                    // Skip AUD (9) / Filler (12) if empty, keep VCL/SEI
                    if nal_type == 9 {
                        continue;
                    }
                }
                VideoCodec::H265 => {
                    let nal_type = (nal[0] >> 1) & 0x3F;
                    if (16..=21).contains(&nal_type) {
                        is_sync = true;
                    }
                    if nal_type == 35 {
                        continue;
                    }
                }
            }
            let len_u32 = nal.len() as u32;
            self.avcc_scratch.extend_from_slice(&len_u32.to_be_bytes());
            self.avcc_scratch.extend_from_slice(nal);
        }

        if self.avcc_scratch.is_empty() {
            return Ok(());
        }

        let offset = self.current_pos;
        let size = self.avcc_scratch.len() as u32;
        self.writer.write_all(&self.avcc_scratch)?;
        self.current_pos += size as u64;

        self.video_samples.push(SampleRecord {
            offset,
            size,
            duration: self.video_frame_delta,
            is_sync,
        });
        Ok(())
    }

    /// Write one raw AAC-LC access unit (without ADTS header) with its PCM frame sample count (typically 1024).
    pub fn write_aac_frame(&mut self, raw_aac: &[u8], pcm_samples: u32) -> io::Result<()> {
        if raw_aac.is_empty() {
            return Ok(());
        }
        let offset = self.current_pos;
        let size = raw_aac.len() as u32;
        self.writer.write_all(raw_aac)?;
        self.current_pos += size as u64;

        self.audio_samples.push(SampleRecord {
            offset,
            size,
            duration: pcm_samples,
            is_sync: true,
        });
        Ok(())
    }

    /// Patch `mdat` 64-bit size, append `moov` atom, and flush cleanly to disk.
    pub fn finalize(mut self) -> io::Result<()> {
        let mdat_end = self.current_pos;
        let mdat_total_size = mdat_end - self.mdat_header_pos;

        // Build `moov` in memory before seeking
        let moov = self.build_moov_box()?;

        // Append `moov` right after `mdat`
        self.writer.write_all(&moov)?;
        self.writer.flush()?;

        // Patch 64-bit `mdat` largesize at `mdat_header_pos + 8`
        let mut file = self.writer.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(self.mdat_header_pos + 8))?;
        file.write_all(&mdat_total_size.to_be_bytes())?;
        file.flush()?;
        Ok(())
    }

    fn build_moov_box(&self) -> io::Result<Vec<u8>> {
        let video_total_ticks: u64 = self
            .video_samples
            .iter()
            .map(|s| s.duration as u64)
            .sum();
        let video_movie_dur = if self.video_timescale > 0 {
            video_total_ticks
                .saturating_mul(MOVIE_TIMESCALE as u64)
                / (self.video_timescale as u64)
        } else {
            0
        };

        let audio_total_ticks: u64 = self
            .audio_samples
            .iter()
            .map(|s| s.duration as u64)
            .sum();
        let audio_movie_dur = if self.audio_sample_rate > 0 {
            audio_total_ticks
                .saturating_mul(MOVIE_TIMESCALE as u64)
                / (self.audio_sample_rate as u64)
        } else {
            0
        };

        let movie_duration = video_movie_dur.max(audio_movie_dur);
        let has_audio = !self.audio_samples.is_empty() && !self.audio_specific_config.is_empty();
        let next_track_id: u32 = if has_audio { 3 } else { 2 };

        let mut moov = Vec::with_capacity(4096 + (self.video_samples.len() + self.audio_samples.len()) * 16);
        write_box(&mut moov, b"moov", |buf| {
            // mvhd (version 1, 64-bit timestamps/duration)
            write_full_box(buf, b"mvhd", 1, 0, |b| {
                b.extend_from_slice(&0u64.to_be_bytes()); // creation_time
                b.extend_from_slice(&0u64.to_be_bytes()); // modification_time
                b.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
                b.extend_from_slice(&movie_duration.to_be_bytes());
                b.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate = 1.0
                b.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0
                b.extend_from_slice(&[0u8; 10]); // reserved
                write_identity_matrix(b);
                b.extend_from_slice(&[0u8; 24]); // pre_defined
                b.extend_from_slice(&next_track_id.to_be_bytes());
            });

            // Track 1: Video
            self.write_video_trak(buf, video_movie_dur, video_total_ticks);

            // Track 2: Audio (AAC)
            if has_audio {
                self.write_audio_trak(buf, audio_movie_dur, audio_total_ticks);
            }
        });

        Ok(moov)
    }

    fn write_video_trak(&self, out: &mut Vec<u8>, movie_dur: u64, media_dur: u64) {
        write_box(out, b"trak", |buf| {
            // tkhd (version 1, flags = 3: enabled + in_movie)
            write_full_box(buf, b"tkhd", 1, 0x000003, |b| {
                b.extend_from_slice(&0u64.to_be_bytes()); // creation
                b.extend_from_slice(&0u64.to_be_bytes()); // modification
                b.extend_from_slice(&1u32.to_be_bytes()); // track_ID = 1
                b.extend_from_slice(&0u32.to_be_bytes()); // reserved
                b.extend_from_slice(&movie_dur.to_be_bytes());
                b.extend_from_slice(&[0u8; 8]); // reserved
                b.extend_from_slice(&0u16.to_be_bytes()); // layer
                b.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
                b.extend_from_slice(&0u16.to_be_bytes()); // volume = 0 (video)
                b.extend_from_slice(&0u16.to_be_bytes()); // reserved
                write_identity_matrix(b);
                b.extend_from_slice(&((self.width as u32) << 16).to_be_bytes());
                b.extend_from_slice(&((self.height as u32) << 16).to_be_bytes());
            });

            // mdia
            write_box(buf, b"mdia", |mb| {
                write_full_box(mb, b"mdhd", 1, 0, |b| {
                    b.extend_from_slice(&0u64.to_be_bytes());
                    b.extend_from_slice(&0u64.to_be_bytes());
                    b.extend_from_slice(&self.video_timescale.to_be_bytes());
                    b.extend_from_slice(&media_dur.to_be_bytes());
                    b.extend_from_slice(&0x55C4u16.to_be_bytes()); // "und"
                    b.extend_from_slice(&0u16.to_be_bytes());
                });

                write_full_box(mb, b"hdlr", 0, 0, |b| {
                    b.extend_from_slice(&0u32.to_be_bytes());
                    b.extend_from_slice(b"vide");
                    b.extend_from_slice(&[0u8; 12]);
                    b.extend_from_slice(b"Wanith Vulkan Video\0");
                });

                write_box(mb, b"minf", |minf| {
                    write_full_box(minf, b"vmhd", 0, 1, |b| {
                        b.extend_from_slice(&0u16.to_be_bytes());
                        b.extend_from_slice(&[0u8; 6]);
                    });
                    write_dinf(minf);
                    write_box(minf, b"stbl", |stbl| {
                        // stsd
                        write_full_box(stbl, b"stsd", 0, 0, |b| {
                            b.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
                            match self.codec {
                                VideoCodec::H264 => self.write_avc1_entry(b),
                                VideoCodec::H265 => self.write_hvc1_entry(b),
                            }
                        });
                        write_stts(stbl, &self.video_samples);
                        write_stss(stbl, &self.video_samples);
                        write_stsc_one_per_chunk(stbl, self.video_samples.len());
                        write_stsz(stbl, &self.video_samples);
                        write_co64(stbl, &self.video_samples);
                    });
                });
            });
        });
    }

    fn write_avc1_entry(&self, out: &mut Vec<u8>) {
        write_box(out, b"avc1", |b| {
            write_visual_sample_entry_header(b, self.width, self.height);
            write_box(b, b"avcC", |avcc| {
                let sps = self.sps_list.first().map(|v| v.as_slice()).unwrap_or(&[]);
                let profile_idc = sps.get(1).copied().unwrap_or(100); // High
                let profile_compat = sps.get(2).copied().unwrap_or(0);
                let level_idc = sps.get(3).copied().unwrap_or(41);

                avcc.push(1); // configurationVersion
                avcc.push(profile_idc);
                avcc.push(profile_compat);
                avcc.push(level_idc);
                avcc.push(0xFF); // 6 reserved 1-bits + lengthSizeMinusOne = 3 (4-byte NAL len)

                let sps_count = self.sps_list.len().min(31) as u8;
                avcc.push(0xE0 | sps_count);
                for s in self.sps_list.iter().take(31) {
                    avcc.extend_from_slice(&(s.len() as u16).to_be_bytes());
                    avcc.extend_from_slice(s);
                }

                let pps_count = self.pps_list.len().min(255) as u8;
                avcc.push(pps_count);
                for p in self.pps_list.iter().take(255) {
                    avcc.extend_from_slice(&(p.len() as u16).to_be_bytes());
                    avcc.extend_from_slice(p);
                }
            });
        });
    }

    fn write_hvc1_entry(&self, out: &mut Vec<u8>) {
        write_box(out, b"hvc1", |b| {
            write_visual_sample_entry_header(b, self.width, self.height);
            write_box(b, b"hvcC", |hvcc| {
                // Extract general_profile_tier_level (12 bytes) from SPS RBSP if available:
                // HEVC NAL header = 2 bytes; RBSP byte 0 = vps_id(4) | max_sub_layers_m1(3) | temporal_id_nesting(1)
                // RBSP bytes 1..13 = profile_tier_level(1, max_sub_layers_m1)
                let ptl = self
                    .sps_list
                    .first()
                    .and_then(|sps| {
                        if sps.len() <= 2 {
                            return None;
                        }
                        let rbsp = unescape_rbsp(&sps[2..]);
                        if rbsp.len() >= 13 {
                            let mut arr = [0u8; 12];
                            arr.copy_from_slice(&rbsp[1..13]);
                            Some(arr)
                        } else {
                            None
                        }
                    })
                    .unwrap_or([
                        0x01, // general_profile_space=0, tier=0, profile_idc=1 (Main)
                        0x60, 0x00, 0x00, 0x00, // profile_compatibility_flags
                        0x90, 0x00, 0x00, 0x00, 0x00, 0x00, // constraint_indicator_flags
                        120,  // general_level_idc (4.0)
                    ]);

                hvcc.push(1); // configurationVersion
                hvcc.extend_from_slice(&ptl); // 12 bytes PTL
                hvcc.extend_from_slice(&0xF000u16.to_be_bytes()); // min_spatial_segmentation_idc = 0
                hvcc.push(0xFC); // parallelismType = 0
                hvcc.push(0xFD); // chromaFormat = 1 (4:2:0)
                hvcc.push(0xF8); // bitDepthLumaMinus8 = 0
                hvcc.push(0xF8); // bitDepthChromaMinus8 = 0
                hvcc.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate = 0 (unspecified)
                hvcc.push(0x0F); // constantFrameRate=0, numTemporalLayers=1, temporalIdNested=1, lengthSizeMinusOne=3

                let mut arrays: Vec<(u8, &[Vec<u8>])> = Vec::with_capacity(3);
                if !self.vps_list.is_empty() {
                    arrays.push((32, &self.vps_list));
                }
                if !self.sps_list.is_empty() {
                    arrays.push((33, &self.sps_list));
                }
                if !self.pps_list.is_empty() {
                    arrays.push((34, &self.pps_list));
                }
                hvcc.push(arrays.len() as u8);
                for (nal_type, list) in arrays {
                    hvcc.push(0x80 | (nal_type & 0x3F)); // array_completeness = 1
                    hvcc.extend_from_slice(&(list.len() as u16).to_be_bytes());
                    for nal in list {
                        hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
                        hvcc.extend_from_slice(nal);
                    }
                }
            });
        });
    }

    fn write_audio_trak(&self, out: &mut Vec<u8>, movie_dur: u64, media_dur: u64) {
        write_box(out, b"trak", |buf| {
            // tkhd (version 1, flags = 3)
            write_full_box(buf, b"tkhd", 1, 0x000003, |b| {
                b.extend_from_slice(&0u64.to_be_bytes());
                b.extend_from_slice(&0u64.to_be_bytes());
                b.extend_from_slice(&2u32.to_be_bytes()); // track_ID = 2
                b.extend_from_slice(&0u32.to_be_bytes());
                b.extend_from_slice(&movie_dur.to_be_bytes());
                b.extend_from_slice(&[0u8; 8]);
                b.extend_from_slice(&0u16.to_be_bytes());
                b.extend_from_slice(&1u16.to_be_bytes()); // alternate_group = 1
                b.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0 (audio)
                b.extend_from_slice(&0u16.to_be_bytes());
                write_identity_matrix(b);
                b.extend_from_slice(&0u32.to_be_bytes());
                b.extend_from_slice(&0u32.to_be_bytes());
            });

            write_box(buf, b"mdia", |mb| {
                write_full_box(mb, b"mdhd", 1, 0, |b| {
                    b.extend_from_slice(&0u64.to_be_bytes());
                    b.extend_from_slice(&0u64.to_be_bytes());
                    b.extend_from_slice(&self.audio_sample_rate.to_be_bytes());
                    b.extend_from_slice(&media_dur.to_be_bytes());
                    b.extend_from_slice(&0x55C4u16.to_be_bytes()); // "und"
                    b.extend_from_slice(&0u16.to_be_bytes());
                });

                write_full_box(mb, b"hdlr", 0, 0, |b| {
                    b.extend_from_slice(&0u32.to_be_bytes());
                    b.extend_from_slice(b"soun");
                    b.extend_from_slice(&[0u8; 12]);
                    b.extend_from_slice(b"Wanith KDMAPI v3 AAC\0");
                });

                write_box(mb, b"minf", |minf| {
                    write_full_box(minf, b"smhd", 0, 0, |b| {
                        b.extend_from_slice(&0u16.to_be_bytes()); // balance
                        b.extend_from_slice(&0u16.to_be_bytes()); // reserved
                    });
                    write_dinf(minf);
                    write_box(minf, b"stbl", |stbl| {
                        write_full_box(stbl, b"stsd", 0, 0, |b| {
                            b.extend_from_slice(&1u32.to_be_bytes());
                            self.write_mp4a_entry(b);
                        });
                        write_stts(stbl, &self.audio_samples);
                        write_stsc_one_per_chunk(stbl, self.audio_samples.len());
                        write_stsz(stbl, &self.audio_samples);
                        write_co64(stbl, &self.audio_samples);
                    });
                });
            });
        });
    }

    fn write_mp4a_entry(&self, out: &mut Vec<u8>) {
        write_box(out, b"mp4a", |b| {
            b.extend_from_slice(&[0u8; 6]); // reserved
            b.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
            b.extend_from_slice(&[0u8; 8]); // reserved
            b.extend_from_slice(&self.audio_channels.to_be_bytes());
            b.extend_from_slice(&16u16.to_be_bytes()); // sampleSize
            b.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
            b.extend_from_slice(&0u16.to_be_bytes()); // reserved
            b.extend_from_slice(&(self.audio_sample_rate << 16).to_be_bytes());

            // esds box (ISO/IEC 14496-1 ES_Descriptor)
            write_full_box(b, b"esds", 0, 0, |esds| {
                let asc = &self.audio_specific_config;
                let dec_specific_len = asc.len();
                let dec_config_len = 13 + 2 + dec_specific_len;
                let sl_config_len = 1;
                let es_desc_len = 3 + (2 + dec_config_len) + (2 + sl_config_len);

                // ES_DescrTag (0x03)
                esds.push(0x03);
                write_mp4_descr_len(esds, es_desc_len);
                esds.extend_from_slice(&2u16.to_be_bytes()); // ES_ID = 2
                esds.push(0x00); // flags

                // DecoderConfigDescrTag (0x04)
                esds.push(0x04);
                write_mp4_descr_len(esds, dec_config_len);
                esds.push(0x40); // objectTypeIndication = MPEG-4 Audio (AAC)
                esds.push(0x15); // streamType = Audio (5 << 2 | 1)
                esds.extend_from_slice(&[0x00, 0x18, 0x00]); // bufferSizeDB = 6144
                esds.extend_from_slice(&self.audio_bitrate.to_be_bytes()); // maxBitrate
                esds.extend_from_slice(&self.audio_bitrate.to_be_bytes()); // avgBitrate

                // DecSpecificInfoTag (0x05)
                esds.push(0x05);
                write_mp4_descr_len(esds, dec_specific_len);
                esds.extend_from_slice(asc);

                // SLConfigDescrTag (0x06)
                esds.push(0x06);
                write_mp4_descr_len(esds, sl_config_len);
                esds.push(0x02); // predefined = MP4
            });
        });
    }

    #[allow(dead_code)]
    pub fn fps(&self) -> u32 {
        self.fps
    }
}

fn write_box(out: &mut Vec<u8>, fourcc: &[u8; 4], body: impl FnOnce(&mut Vec<u8>)) {
    let start = out.len();
    out.extend_from_slice(&0u32.to_be_bytes());
    out.extend_from_slice(fourcc);
    body(out);
    let size = (out.len() - start) as u32;
    out[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

fn write_full_box(
    out: &mut Vec<u8>,
    fourcc: &[u8; 4],
    version: u8,
    flags: u32,
    body: impl FnOnce(&mut Vec<u8>),
) {
    write_box(out, fourcc, |b| {
        let vf = ((version as u32) << 24) | (flags & 0x00FF_FFFF);
        b.extend_from_slice(&vf.to_be_bytes());
        body(b);
    });
}

fn write_mp4_descr_len(out: &mut Vec<u8>, len: usize) {
    debug_assert!(len < 128);
    out.push((len & 0x7F) as u8);
}

fn write_identity_matrix(out: &mut Vec<u8>) {
    let matrix: [u32; 9] = [
        0x0001_0000,
        0,
        0,
        0,
        0x0001_0000,
        0,
        0,
        0,
        0x4000_0000,
    ];
    for v in matrix {
        out.extend_from_slice(&v.to_be_bytes());
    }
}

fn write_visual_sample_entry_header(b: &mut Vec<u8>, width: u16, height: u16) {
    b.extend_from_slice(&[0u8; 6]); // reserved
    b.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    b.extend_from_slice(&[0u8; 16]); // pre_defined + reserved
    b.extend_from_slice(&width.to_be_bytes());
    b.extend_from_slice(&height.to_be_bytes());
    b.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // horizresolution 72 dpi
    b.extend_from_slice(&0x0048_0000u32.to_be_bytes()); // vertresolution 72 dpi
    b.extend_from_slice(&0u32.to_be_bytes()); // reserved
    b.extend_from_slice(&1u16.to_be_bytes()); // frame_count
    b.extend_from_slice(&[0u8; 32]); // compressorname
    b.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24
    b.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined = -1
}

fn write_dinf(out: &mut Vec<u8>) {
    write_box(out, b"dinf", |dinf| {
        write_full_box(dinf, b"dref", 0, 0, |dref| {
            dref.extend_from_slice(&1u32.to_be_bytes());
            write_full_box(dref, b"url ", 0, 1, |_| {}); // self-contained
        });
    });
}

fn write_stts(out: &mut Vec<u8>, samples: &[SampleRecord]) {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for s in samples {
        if let Some(last) = runs.last_mut() {
            if last.1 == s.duration {
                last.0 += 1;
                continue;
            }
        }
        runs.push((1, s.duration));
    }
    write_full_box(out, b"stts", 0, 0, |b| {
        b.extend_from_slice(&(runs.len() as u32).to_be_bytes());
        for (count, delta) in runs {
            b.extend_from_slice(&count.to_be_bytes());
            b.extend_from_slice(&delta.to_be_bytes());
        }
    });
}

fn write_stss(out: &mut Vec<u8>, samples: &[SampleRecord]) {
    let syncs: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| if s.is_sync { Some((idx + 1) as u32) } else { None })
        .collect();
    if syncs.is_empty() {
        return;
    }
    write_full_box(out, b"stss", 0, 0, |b| {
        b.extend_from_slice(&(syncs.len() as u32).to_be_bytes());
        for s in syncs {
            b.extend_from_slice(&s.to_be_bytes());
        }
    });
}

fn write_stsc_one_per_chunk(out: &mut Vec<u8>, sample_count: usize) {
    write_full_box(out, b"stsc", 0, 0, |b| {
        if sample_count == 0 {
            b.extend_from_slice(&0u32.to_be_bytes());
        } else {
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
            b.extend_from_slice(&1u32.to_be_bytes()); // samples_per_chunk
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        }
    });
}

fn write_stsz(out: &mut Vec<u8>, samples: &[SampleRecord]) {
    write_full_box(out, b"stsz", 0, 0, |b| {
        b.extend_from_slice(&0u32.to_be_bytes()); // variable sample sizes
        b.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        for s in samples {
            b.extend_from_slice(&s.size.to_be_bytes());
        }
    });
}

fn write_co64(out: &mut Vec<u8>, samples: &[SampleRecord]) {
    write_full_box(out, b"co64", 0, 0, |b| {
        b.extend_from_slice(&(samples.len() as u32).to_be_bytes());
        for s in samples {
            b.extend_from_slice(&s.offset.to_be_bytes());
        }
    });
}
