use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
use time::Duration;

use atomic_float::AtomicF64;
use midi_toolkit::{
    events::Event,
    io::{DiskReader, MIDIFile as TKMIDIFile},
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time, Delta, EventBatch, Track},
        unwrap_items, TimeCaster,
    },
};

use crate::{
    audio_playback::WasabiAudioPlayer,
    midi::{
        audio::live::LiveAudioPlayer,
        double_buffer::DoubleBufferReader,
        live::block::LiveWindowSnapshot,
        shared::timer::{TimeKeeper, WaitResult},
    },
};

mod audio;
mod notes;

pub type TrackEventBatch = Delta<f64, Track<EventBatch<Event>>>;

pub struct ThreadManager {
    parse_time: Arc<AtomicF64>,
    handle: JoinHandle<()>,
}

pub struct LiveMidiParser {
    file_manager: ThreadManager,
    note_manager: ThreadManager,
    audio_manager: ThreadManager,
    alive: Arc<AtomicBool>,
}

impl LiveMidiParser {
    pub fn init(
        midi: &TKMIDIFile<DiskReader>,
        player: Arc<WasabiAudioPlayer>,
        timer: &mut TimeKeeper,
    ) -> (Self, DoubleBufferReader<LiveWindowSnapshot>, Arc<AtomicF64>) {
        let ppq = midi.ppq();
        let merged = pipe!(
            midi.iter_all_track_events_merged_batches()
            |>TimeCaster::<f64>::cast_event_delta()
            |>cancel_tempo_events(250000)
            |>scale_event_time(1.0 / ppq as f64)
            |>unwrap_items()
        );

        let (note_snd, note_rcv) = crossbeam_channel::bounded::<Arc<TrackEventBatch>>(1000);
        let (audio_snd, audio_rcv) = crossbeam_channel::bounded::<Arc<TrackEventBatch>>(1000);

        let shared_view_start = Arc::new(AtomicF64::new(f64::NEG_INFINITY));
        let alive = Arc::new(AtomicBool::new(true));

        let notes = notes::init_note_manager(note_rcv, shared_view_start.clone(), alive.clone());
        let audio = audio::init_audio_manager(audio_rcv);

        LiveAudioPlayer::new(audio.reciever, audio.recycler, timer.get_listener(), player)
            .spawn_playback();

        let mut parser_timer = timer.get_listener();

        let parse_time_outer = Arc::new(AtomicF64::default());
        let parse_time = parse_time_outer.clone();
        let file_alive = alive.clone();
        let file_handle = thread::spawn(move || {
            let mut time = 0.0;
            for block in merged {
                if !file_alive.load(Ordering::Acquire) {
                    break;
                }
                if block.delta > 0.0 {
                    time += block.delta;
                    parse_time.store(time, Ordering::Relaxed);
                }

                let playback_time = (time - 10.0).max(0.0); // 10 seconds lookahead window
                let waited = parser_timer.wait_until(Duration::seconds_f64(playback_time));
                if let WaitResult::Killed = waited {
                    break;
                }

                let block = Arc::new(block);

                if note_snd.send(block.clone()).is_err() {
                    break;
                }

                if audio_snd.send(block).is_err() {
                    break;
                }
            }
        });

        let parser = Self {
            file_manager: ThreadManager {
                handle: file_handle,
                parse_time: parse_time_outer,
            },
            note_manager: notes.manager,
            audio_manager: audio.manager,
            alive,
        };

        (parser, notes.window_reader, shared_view_start)
    }

    #[inline]
    pub fn parse_time(&self) -> f64 {
        self.file_manager.parse_time.load(Ordering::Relaxed)
    }
}

impl Drop for LiveMidiParser {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
    }
}
