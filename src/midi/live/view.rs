#![allow(dead_code)]

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use atomic_float::AtomicF64;
use gen_iter::GenIter;

use crate::midi::{
    double_buffer::{DoubleBufferReadGuard, DoubleBufferReader},
    DisplacedMIDINote, MIDIColor, MIDINoteColumnView, MIDINoteViews, MIDIViewRange,
};

use super::{
    block::{LiveFlatColumn, LiveWindowSnapshot},
    parse::LiveMidiParser,
};

#[derive(Clone, Copy, Default)]
struct ColumnWindowSlice {
    start_block: usize,
    end_block: usize,
    rendered_notes: usize,
}

pub struct LiveNoteViewData {
    parser: LiveMidiParser,
    window_reader: DoubleBufferReader<LiveWindowSnapshot>,
    shared_view_start: Arc<AtomicF64>,
    default_track_colors: Vec<MIDIColor>,
    last_passed_notes: AtomicU64,
}

pub struct LiveCurrentNoteViews<'a> {
    guard: DoubleBufferReadGuard<'a, LiveWindowSnapshot>,
    colors: &'a [MIDIColor],
    view_range: MIDIViewRange,
    slices: [ColumnWindowSlice; 256],
}

impl LiveNoteViewData {
    pub fn new(
        parser: LiveMidiParser,
        window_reader: DoubleBufferReader<LiveWindowSnapshot>,
        shared_view_start: Arc<AtomicF64>,
        colors: Vec<MIDIColor>,
    ) -> Self {
        LiveNoteViewData {
            parser,
            window_reader,
            shared_view_start,
            default_track_colors: colors,
            last_passed_notes: AtomicU64::new(0),
        }
    }

    /// Wait-free acquisition of the active A/B sliding window buffer for the current frame.
    pub fn acquire_views(&mut self, new_view_range: MIDIViewRange) -> LiveCurrentNoteViews<'_> {
        // Notify the background note worker so blocks with max_end < start are evicted from RAM
        self.shared_view_start
            .store(new_view_range.start, Ordering::Relaxed);

        let guard = self.window_reader.read();
        let mut slices = [ColumnWindowSlice::default(); 256];
        let mut total_passed = 0u64;

        for (key, col) in guard.columns.iter().enumerate() {
            let block_count = col.block_count();
            let mut passed = col.evicted_notes;

            if block_count == 0 {
                total_passed += passed;
                continue;
            }

            // Binary search exclusive end block (first block with start > new_view_range.end)
            let end_block = col
                .block_starts
                .partition_point(|&s| s <= new_view_range.end);

            // Advance start_block past any blocks whose max_end < new_view_range.start
            // (in case eviction hasn't run yet for the newest millisecond)
            let mut start_block = 0usize;
            while start_block < end_block && col.block_max_ends[start_block] < new_view_range.start
            {
                start_block += 1;
            }

            // Exact passed notes at keyboard line (`block_start <= new_view_range.start`)
            let passed_blocks = col
                .block_starts
                .partition_point(|&s| s <= new_view_range.start);
            if passed_blocks > 0 {
                passed += col.block_note_offsets[passed_blocks] as u64;
            }
            total_passed += passed;

            let rendered_notes = if start_block < end_block {
                (col.block_note_offsets[end_block] - col.block_note_offsets[start_block]) as usize
            } else {
                0
            };

            slices[key] = ColumnWindowSlice {
                start_block,
                end_block,
                rendered_notes,
            };
        }

        self.last_passed_notes
            .store(total_passed, Ordering::Relaxed);

        LiveCurrentNoteViews {
            guard,
            colors: &self.default_track_colors,
            view_range: new_view_range,
            slices,
        }
    }

    #[inline]
    pub fn parse_time(&self) -> f64 {
        self.parser.parse_time()
    }

    #[inline]
    pub fn passed_notes(&self) -> u64 {
        self.last_passed_notes.load(Ordering::Relaxed)
    }
}

pub struct LiveNoteColumnView<'a> {
    colors: &'a [MIDIColor],
    column: &'a LiveFlatColumn,
    view_range: MIDIViewRange,
    slice: ColumnWindowSlice,
}

impl<'a> MIDINoteViews for LiveCurrentNoteViews<'a> {
    type View<'b>
        = LiveNoteColumnView<'b>
    where
        Self: 'a + 'b;

    fn get_column(&self, key: usize) -> Self::View<'_> {
        LiveNoteColumnView {
            colors: self.colors,
            column: &self.guard.columns[key],
            view_range: self.view_range,
            slice: self.slices[key],
        }
    }

    fn range(&self) -> MIDIViewRange {
        self.view_range
    }
}

struct LiveNoteBlockIter<Iter: Iterator<Item = DisplacedMIDINote>> {
    remaining: usize,
    iter: Iter,
}

impl<'a> MIDINoteColumnView for LiveNoteColumnView<'a> {
    type Iter<'b>
        = impl 'b + ExactSizeIterator<Item = DisplacedMIDINote>
    where
        Self: 'b;

    fn iterate_displaced_notes(&self) -> Self::Iter<'_> {
        let colors = self.colors;
        let column = self.column;
        let slice = self.slice;
        let view_start = self.view_range.start;

        let iter = GenIter(
            #[coroutine]
            move || {
                for block_index in (slice.start_block..slice.end_block).rev() {
                    let start = (column.block_starts[block_index] - view_start) as f32;
                    let (note_start, note_end) = column.block_note_range(block_index);

                    for note_idx in (note_start..note_end).rev() {
                        yield DisplacedMIDINote {
                            start,
                            len: column.note_lens[note_idx],
                            color: colors[column.note_track_chans[note_idx].as_usize()],
                        };
                    }
                }
            },
        );

        LiveNoteBlockIter {
            remaining: slice.rendered_notes,
            iter,
        }
    }
}

impl<Iter: Iterator<Item = DisplacedMIDINote>> Iterator for LiveNoteBlockIter<Iter> {
    type Item = DisplacedMIDINote;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next();
        if item.is_some() {
            self.remaining = self.remaining.saturating_sub(1);
        }
        item
    }
}

impl<Iter: Iterator<Item = DisplacedMIDINote>> ExactSizeIterator for LiveNoteBlockIter<Iter> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining
    }
}
