use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread,
};
use time::Duration;

use midi_toolkit::{
    events::{Event, MIDIEventEnum},
    io::MIDIFile as TKMIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time, Delta, EventBatch, Track},
        unwrap_items, TimeCaster,
    },
};

use crate::{
    audio_playback::WasabiAudioPlayer,
    gui::window::WasabiError,
    midi::{
        audio::ram::InRamAudioPlayer,
        cake::tree_threader::{NoteEvent, ThreadedTreeSerializers},
        open_file_and_signature,
        shared::{audio::FlatAudioTimeline, timer::TimeKeeper},
        MIDIColor,
    },
    settings::MidiSettings,
};

use self::blocks::CakeBlock;

use super::{MIDIFileBase, MIDIFileStats, MIDIFileUniqueSignature};

pub mod blocks;
pub mod intvec4;
mod tree_serializer;
mod tree_threader;
mod unended_note_batch;

static NEXT_CAKE_FILE_ID: AtomicU64 = AtomicU64::new(1);

pub struct CakeMIDIFile {
    /// Monotonic file identifier for zero-allocation GPU upload detection.
    file_id: u64,
    /// CPU-side binary trees held ONLY until `CakeRenderer` uploads them to `DEVICE_LOCAL` VRAM.
    /// Immediately taken and dropped after GPU transfer to eliminate CPU/VRAM double-residency.
    cpu_blocks: Option<Vec<CakeBlock>>,
    timer: TimeKeeper,
    length: f64,
    note_count: u64,
    /// Lock-free passed-notes counter updated from the 4 KB GPU Compute metadata readback.
    passed_notes: AtomicU64,
    ticks_per_second: u32,
    signature: MIDIFileUniqueSignature,
}

impl CakeMIDIFile {
    pub fn load_from_file(
        path: impl Into<PathBuf>,
        player: Arc<WasabiAudioPlayer>,
        settings: &MidiSettings,
    ) -> Result<Self, WasabiError> {
        let ticks_per_second = 10000;

        let (file, signature) = open_file_and_signature(path)?;
        let midi = TKMIDIFile::open_from_stream(file, None).map_err(WasabiError::MidiLoadError)?;

        let ppq = midi.ppq();
        let merged = pipe!(
            midi.iter_all_track_events_merged_batches()
            |>TimeCaster::<f64>::cast_event_delta()
            |>cancel_tempo_events(250000)
            |>scale_event_time(1.0 / ppq as f64)
            |>unwrap_items()
        );

        let colors = MIDIColor::new_vec_from_settings(midi.track_count(), settings)?;

        type Ev = Delta<f64, Track<EventBatch<Event>>>;
        let (key_snd, key_rcv) = crossbeam_channel::bounded::<Arc<Ev>>(1000);
        let (audio_snd, audio_rcv) = crossbeam_channel::bounded::<Arc<Ev>>(1000);

        let key_join_handle = thread::spawn(move || {
            let mut trees = ThreadedTreeSerializers::new();

            let mut time = 0.0;

            let mut note_count = 0;

            for batch in key_rcv.into_iter() {
                time += batch.delta;

                let int_time = (time * ticks_per_second as f64) as i32;

                fn channel_track(channel: u8, track: u32) -> i32 {
                    (channel as i32) + (track as i32) * 16
                }

                for event in batch.iter_events() {
                    let track = event.track;
                    match event.as_event() {
                        Event::NoteOn(e) => {
                            let channel_track = channel_track(e.channel, track);

                            trees.push_event(
                                e.key as usize,
                                NoteEvent::On {
                                    time: int_time,
                                    channel_track,
                                    color: colors[channel_track as usize].as_u32() as i32,
                                },
                            );
                            note_count += 1;
                        }
                        Event::NoteOff(e) => {
                            let channel_track = channel_track(e.channel, track);

                            trees.push_event(
                                e.key as usize,
                                NoteEvent::Off {
                                    time: int_time,
                                    channel_track,
                                    color: colors[channel_track as usize].as_u32() as i32,
                                },
                            );
                        }
                        _ => {}
                    }
                }
            }
            let final_time = (time * ticks_per_second as f64) as i32;
            let serialized = trees.seal(final_time);

            let keys: Vec<_> = serialized
                .into_iter()
                .map(|s| CakeBlock {
                    start_time: 0,
                    end_time: final_time as u32,
                    tree: s,
                })
                .collect();

            (keys, note_count)
        });

        let audio_join_handle = thread::spawn(|| {
            FlatAudioTimeline::build_from_iter(audio_rcv.into_iter())
        });

        let mut length = 0.0;

        // Write events to the threads
        for batch in merged {
            length += batch.delta;
            let batch = Arc::new(batch);
            key_snd.send(batch.clone()).unwrap();
            audio_snd.send(batch).unwrap();
        }
        // Drop the writers so the threads finish
        drop(key_snd);
        drop(audio_snd);

        let (keys, note_count) = key_join_handle.join().unwrap();
        let audio = audio_join_handle.join().unwrap();

        let mut timer = TimeKeeper::new(settings.start_delay);

        InRamAudioPlayer::new(audio, timer.get_listener(), player).spawn_playback();

        let file_id = NEXT_CAKE_FILE_ID.fetch_add(1, Ordering::Relaxed);

        Ok(CakeMIDIFile {
            file_id,
            cpu_blocks: Some(keys),
            timer,
            length,
            note_count,
            passed_notes: AtomicU64::new(0),
            ticks_per_second,
            signature,
        })
    }

    #[inline]
    pub fn file_id(&self) -> u64 {
        self.file_id
    }

    /// Take ownership of the CPU-side binary trees for one-shot `DEVICE_LOCAL` VRAM upload.
    ///
    /// Once returned and dropped by the renderer, zero bytes of note tree data remain in CPU RAM.
    #[inline]
    pub fn take_cpu_blocks(&mut self) -> Option<Vec<CakeBlock>> {
        self.cpu_blocks.take()
    }

    /// Update the passed-notes statistic from the GPU compute 4 KB metadata readback.
    #[inline]
    pub fn set_passed_notes(&self, passed: u64) {
        self.passed_notes.store(passed, Ordering::Relaxed);
    }

    #[inline]
    pub fn ticks_per_second(&self) -> u32 {
        self.ticks_per_second
    }

    #[inline]
    pub fn current_time(&self) -> Duration {
        self.timer.get_time()
    }
}

impl MIDIFileBase for CakeMIDIFile {
    fn midi_length(&self) -> Option<f64> {
        Some(self.length)
    }

    fn parsed_up_to(&self) -> Option<f64> {
        None
    }

    fn timer(&self) -> &TimeKeeper {
        &self.timer
    }

    fn timer_mut(&mut self) -> &mut TimeKeeper {
        &mut self.timer
    }

    fn allows_seeking_backward(&self) -> bool {
        true
    }

    fn stats(&self) -> MIDIFileStats {
        let passed_notes = if let Some(ref blocks) = self.cpu_blocks {
            let time = self.timer.get_time().as_seconds_f64();
            let time_int = (time * self.ticks_per_second as f64) as i32;
            blocks
                .iter()
                .map(|b| b.get_notes_passed_at(time_int) as u64)
                .sum()
        } else {
            self.passed_notes.load(Ordering::Relaxed)
        };

        MIDIFileStats {
            total_notes: Some(self.note_count),
            passed_notes: Some(passed_notes),
        }
    }

    fn signature(&self) -> &MIDIFileUniqueSignature {
        &self.signature
    }
}
