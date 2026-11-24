use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use atomic_float::AtomicF64;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use midi_toolkit::events::{Event, MIDIEventEnum};

use crate::midi::{
    double_buffer::{double_buffer, DoubleBufferReader, DoubleBufferWriter},
    live::{
        block::LiveWindowSnapshot,
        column::{LiveNoteLocator, WorkingLiveColumn},
    },
    shared::track_channel::TrackAndChannel,
};

use super::{ThreadManager, TrackEventBatch};

/// Maximum incoming event batches drained in one burst before publishing an A/B snapshot.
const MAX_BATCHES_PER_PUBLISH: usize = 64;
/// Poll interval for eviction / snapshot publication when the file parser is throttled ahead.
const WORKER_POLL_INTERVAL_MS: u64 = 4;

struct TrackUnendedNotes {
    queues: Box<[VecDeque<LiveNoteLocator>]>,
}

impl TrackUnendedNotes {
    fn new() -> Self {
        let mut queues = Vec::with_capacity(16 * 256);
        for _ in 0..16 * 256 {
            queues.push(VecDeque::new());
        }

        TrackUnendedNotes {
            queues: queues.into_boxed_slice(),
        }
    }

    #[inline]
    fn get_index(key: u8, channel: u8) -> usize {
        (key as usize) + (channel as usize) * 256
    }

    fn end_all_notes(&mut self, columns: &mut [WorkingLiveColumn], time: f64) {
        for channel in 0..16u8 {
            for key in 0..=255u8 {
                let idx = Self::get_index(key, channel);
                let col = &mut columns[key as usize];
                while let Some(locator) = self.queues[idx].pop_front() {
                    col.end_note(locator, time);
                }
            }
        }
    }

    #[inline]
    fn end_note(&mut self, key: u8, channel: u8, columns: &mut [WorkingLiveColumn], time: f64) {
        let index = Self::get_index(key, channel);
        if let Some(locator) = self.queues[index].pop_front() {
            columns[key as usize].end_note(locator, time);
        }
    }

    #[inline]
    fn add_note(&mut self, key: u8, channel: u8, locator: LiveNoteLocator) {
        let index = Self::get_index(key, channel);
        self.queues[index].push_back(locator);
    }
}

struct UnendedNotesHandler {
    unended_notes: Vec<Option<TrackUnendedNotes>>,
}

impl UnendedNotesHandler {
    fn new() -> Self {
        UnendedNotesHandler {
            unended_notes: Vec::new(),
        }
    }

    #[inline]
    fn get_track(&mut self, track: u32) -> &mut TrackUnendedNotes {
        while self.unended_notes.len() <= track as usize {
            self.unended_notes.push(None);
        }

        self.unended_notes[track as usize].get_or_insert_with(TrackUnendedNotes::new)
    }

    fn end_all_notes(&mut self, columns: &mut [WorkingLiveColumn], time: f64) {
        for track in self.unended_notes.iter_mut().flatten() {
            track.end_all_notes(columns, time);
        }
    }
}

struct ParserState {
    unended_notes: UnendedNotesHandler,
    pending_keys: Box<[Vec<TrackAndChannel>]>,
    working_columns: Box<[WorkingLiveColumn]>,
}

impl ParserState {
    fn new() -> Self {
        let mut pending_keys = Vec::with_capacity(256);
        let mut working_columns = Vec::with_capacity(256);
        for _ in 0..256 {
            pending_keys.push(Vec::new());
            working_columns.push(WorkingLiveColumn::new());
        }
        ParserState {
            unended_notes: UnendedNotesHandler::new(),
            pending_keys: pending_keys.into_boxed_slice(),
            working_columns: working_columns.into_boxed_slice(),
        }
    }

    #[inline]
    fn add_note(&mut self, key: u8, track_chan: TrackAndChannel) {
        self.pending_keys[key as usize].push(track_chan);
    }

    fn flush(&mut self, time: f64) {
        for key in 0..self.pending_keys.len() {
            if !self.pending_keys[key].is_empty() {
                let block_id = self.working_columns[key]
                    .push_block(time, self.pending_keys[key].iter().copied());
                for (note_idx, track_chan) in self.pending_keys[key].drain(..).enumerate() {
                    let track = self.unended_notes.get_track(track_chan.track());
                    track.add_note(
                        key as u8,
                        track_chan.channel(),
                        LiveNoteLocator {
                            block_id,
                            note_index: note_idx as u32,
                        },
                    );
                }
            }
        }
    }

    #[inline]
    fn end_note(&mut self, key: u8, track_chan: TrackAndChannel, time: f64) {
        let track = self.unended_notes.get_track(track_chan.track());
        track.end_note(key, track_chan.channel(), &mut self.working_columns, time);
    }

    fn end_all_notes(&mut self, time: f64) {
        self.flush(time);
        self.unended_notes
            .end_all_notes(&mut self.working_columns, time);
    }

    fn evict_before(&mut self, view_start: f64) -> bool {
        if !view_start.is_finite() {
            return false;
        }
        let mut evicted_any = false;
        for col in self.working_columns.iter_mut() {
            if col.evict_before(view_start) {
                evicted_any = true;
            }
        }
        evicted_any
    }

    fn try_publish(&self, writer: &mut DoubleBufferWriter<LiveWindowSnapshot>) -> bool {
        if let Some(mut guard) = writer.try_write() {
            for (src, dst) in self
                .working_columns
                .iter()
                .zip(guard.columns.iter_mut())
            {
                src.compact_into(dst);
            }
            guard.publish();
            true
        } else {
            false
        }
    }
}

pub struct NoteParserResult {
    pub window_reader: DoubleBufferReader<LiveWindowSnapshot>,
    pub manager: ThreadManager,
}

pub fn init_note_manager(
    blocks: Receiver<Arc<TrackEventBatch>>,
    shared_view_start: Arc<AtomicF64>,
    alive: Arc<AtomicBool>,
) -> NoteParserResult {
    let (mut writer, reader) = double_buffer(LiveWindowSnapshot::new(), LiveWindowSnapshot::new());
    let parse_time_outer = Arc::new(AtomicF64::default());
    let parse_time = parse_time_outer.clone();

    let join_handle = std::thread::spawn(move || {
        let mut state = ParserState::new();
        let mut time: f64 = 0.0;
        let mut dirty = false;
        let mut eof = false;

        let apply_batch = |state: &mut ParserState, time: &mut f64, block: &TrackEventBatch| {
            if block.delta > 0.0 {
                state.flush(*time);
                *time += block.delta;
            }
            for event in block.iter_events() {
                match event.as_event() {
                    Event::NoteOn(e) => {
                        state.add_note(e.key, TrackAndChannel::new(event.track, e.channel));
                    }
                    Event::NoteOff(e) => {
                        state.end_note(e.key, TrackAndChannel::new(event.track, e.channel), *time);
                    }
                    _ => {}
                }
            }
        };

        while alive.load(Ordering::Acquire) {
            if !eof {
                match blocks.recv_timeout(Duration::from_millis(WORKER_POLL_INTERVAL_MS)) {
                    Ok(first) => {
                        apply_batch(&mut state, &mut time, &first);
                        dirty = true;

                        // Drain bursts up to MAX_BATCHES_PER_PUBLISH before publishing A/B snapshot
                        for _ in 1..MAX_BATCHES_PER_PUBLISH {
                            match blocks.try_recv() {
                                Ok(batch) => apply_batch(&mut state, &mut time, &batch),
                                Err(crossbeam_channel::TryRecvError::Empty) => break,
                                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                                    state.end_all_notes(time);
                                    eof = true;
                                    break;
                                }
                            }
                        }
                        parse_time.store(time, Ordering::Relaxed);
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        state.end_all_notes(time);
                        parse_time.store(time, Ordering::Relaxed);
                        eof = true;
                        dirty = true;
                    }
                }
            } else {
                std::thread::sleep(Duration::from_millis(WORKER_POLL_INTERVAL_MS));
            }

            let view_start = shared_view_start.load(Ordering::Relaxed);
            if state.evict_before(view_start) {
                dirty = true;
            }

            if dirty && state.try_publish(&mut writer) {
                dirty = false;
            }
        }
    });

    NoteParserResult {
        window_reader: reader,
        manager: ThreadManager {
            handle: join_handle,
            parse_time: parse_time_outer,
        },
    }
}
