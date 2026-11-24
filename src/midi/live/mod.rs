use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
};

use midi_toolkit::{io::MIDIFile as TKMIDIFile, sequence::event::get_channels_array_statistics};

use crate::{audio_playback::WasabiAudioPlayer, gui::window::WasabiError, settings::MidiSettings};

use self::{
    parse::LiveMidiParser,
    view::{LiveCurrentNoteViews, LiveNoteViewData},
};

use super::{
    open_file_and_signature, shared::timer::TimeKeeper, MIDIColor, MIDIFile, MIDIFileBase,
    MIDIFileStats, MIDIFileUniqueSignature, MIDIViewRange,
};

pub mod block;
pub mod column;
mod parse;
pub mod view;

/// Lock-free atomic statistics populated asynchronously by the background MIDI stats scanner.
struct AtomicParseStats {
    ready: AtomicBool,
    length_bits: AtomicU64,
    note_count: AtomicU64,
}

impl AtomicParseStats {
    fn new() -> Self {
        Self {
            ready: AtomicBool::new(false),
            length_bits: AtomicU64::new(0),
            note_count: AtomicU64::new(0),
        }
    }

    fn publish(&self, length: f64, note_count: u64) {
        self.length_bits.store(length.to_bits(), Ordering::Relaxed);
        self.note_count.store(note_count, Ordering::Relaxed);
        self.ready.store(true, Ordering::Release);
    }

    #[inline]
    fn length(&self) -> Option<f64> {
        if self.ready.load(Ordering::Acquire) {
            Some(f64::from_bits(self.length_bits.load(Ordering::Relaxed)))
        } else {
            None
        }
    }

    #[inline]
    fn note_count(&self) -> Option<u64> {
        if self.ready.load(Ordering::Acquire) {
            Some(self.note_count.load(Ordering::Relaxed))
        } else {
            None
        }
    }
}

pub struct LiveLoadMIDIFile {
    view_data: LiveNoteViewData,
    timer: TimeKeeper,
    stats: Arc<AtomicParseStats>,
    signature: MIDIFileUniqueSignature,
}

impl LiveLoadMIDIFile {
    pub fn load_from_file(
        path: impl Into<PathBuf>,
        player: Arc<WasabiAudioPlayer>,
        settings: &MidiSettings,
    ) -> Result<Self, WasabiError> {
        let (file, signature) = open_file_and_signature(path)?;

        let midi = TKMIDIFile::open_from_stream(file, None).map_err(WasabiError::MidiLoadError)?;

        let stats_outer = Arc::new(AtomicParseStats::new());
        let stats = stats_outer.clone();

        let ppq = midi.ppq();
        let tracks = midi.iter_all_tracks().collect();
        thread::spawn(move || {
            if let Ok(parsed) = get_channels_array_statistics(tracks) {
                stats_outer.publish(
                    parsed.calculate_total_duration(ppq).as_secs_f64(),
                    parsed.note_count(),
                );
            }
        });

        let mut timer = TimeKeeper::new(settings.start_delay);

        let colors = MIDIColor::new_vec_from_settings(midi.track_count(), settings)?;

        let (parser, window_reader, shared_view_start) =
            LiveMidiParser::init(&midi, player, &mut timer);
        let file = LiveNoteViewData::new(parser, window_reader, shared_view_start, colors);

        Ok(LiveLoadMIDIFile {
            view_data: file,
            timer,
            stats,
            signature,
        })
    }
}

impl MIDIFileBase for LiveLoadMIDIFile {
    fn midi_length(&self) -> Option<f64> {
        self.stats.length()
    }

    fn parsed_up_to(&self) -> Option<f64> {
        Some(self.view_data.parse_time())
    }

    fn timer(&self) -> &TimeKeeper {
        &self.timer
    }

    fn timer_mut(&mut self) -> &mut TimeKeeper {
        &mut self.timer
    }

    fn allows_seeking_backward(&self) -> bool {
        false
    }

    fn stats(&self) -> MIDIFileStats {
        MIDIFileStats {
            passed_notes: Some(self.view_data.passed_notes()),
            total_notes: self.stats.note_count(),
        }
    }

    fn signature(&self) -> &MIDIFileUniqueSignature {
        &self.signature
    }
}

impl MIDIFile for LiveLoadMIDIFile {
    type ColumnsViews<'a>
        = LiveCurrentNoteViews<'a>
    where
        Self: 'a;

    fn get_current_column_views(&mut self, range: f64) -> Self::ColumnsViews<'_> {
        let time = self.timer.get_time().as_seconds_f64();
        let new_range = MIDIViewRange::new(time, time + range);
        self.view_data.acquire_views(new_range)
    }
}
