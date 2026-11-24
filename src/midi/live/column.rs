//! Worker-private mutable sliding window column for live MIDI streaming.
//!
//! Owned exclusively by the background note worker thread — eliminates `Arc<UnsafeCell<_>>`
//! cross-thread mutation completely. Unended note handles reference their block via monotonic
//! `block_id`, giving O(1) direct indexing into the `VecDeque` with zero atomic refcounting.

use std::collections::VecDeque;

use crate::midi::shared::track_channel::TrackAndChannel;

use super::block::{LiveFlatColumn, LiveMIDINote};

pub struct WorkingLiveBlock {
    pub start: f64,
    pub max_length: f32,
    pub unended_notes: u32,
    pub notes: Vec<LiveMIDINote>,
}

impl WorkingLiveBlock {
    #[inline]
    pub fn max_end(&self) -> f64 {
        if self.unended_notes == 0 {
            self.start + self.max_length as f64
        } else {
            f64::INFINITY
        }
    }
}

/// Lightweight value handle (12 bytes, `Copy`, zero `Arc`!) referencing an unended note.
#[derive(Debug, Clone, Copy)]
pub struct LiveNoteLocator {
    pub block_id: u64,
    pub note_index: u32,
}

pub struct WorkingLiveColumn {
    blocks: VecDeque<WorkingLiveBlock>,
    /// Monotonic ID of `blocks.front()` (incremented on every eviction).
    first_block_id: u64,
    /// Monotonic ID to assign to the next pushed block.
    next_block_id: u64,
    /// Cumulative notes evicted behind `view_start`.
    evicted_notes: u64,
}

impl WorkingLiveColumn {
    pub fn new() -> Self {
        Self {
            blocks: VecDeque::new(),
            first_block_id: 0,
            next_block_id: 0,
            evicted_notes: 0,
        }
    }

    /// Push a new timestamp block and return its monotonic `block_id`.
    #[inline]
    pub fn push_block(
        &mut self,
        start: f64,
        track_chans: impl ExactSizeIterator<Item = TrackAndChannel>,
    ) -> u64 {
        let len = track_chans.len();
        let mut notes = Vec::with_capacity(len);
        for track_chan in track_chans {
            notes.push(LiveMIDINote {
                len: f32::INFINITY,
                track_chan,
            });
        }
        let block_id = self.next_block_id;
        self.next_block_id += 1;
        self.blocks.push_back(WorkingLiveBlock {
            start,
            max_length: 0.0,
            unended_notes: len as u32,
            notes,
        });
        block_id
    }

    /// Mark a previously started note as ended in O(1) without any `Arc` or `unsafe`.
    #[inline]
    pub fn end_note(&mut self, locator: LiveNoteLocator, end_time: f64) {
        if locator.block_id < self.first_block_id {
            return;
        }
        let deque_idx = (locator.block_id - self.first_block_id) as usize;
        if let Some(block) = self.blocks.get_mut(deque_idx) {
            let note = &mut block.notes[locator.note_index as usize];
            let note_len = (end_time - block.start).max(0.0) as f32;
            note.len = note_len;
            if note_len > block.max_length {
                block.max_length = note_len;
            }
            block.unended_notes = block.unended_notes.saturating_sub(1);
        }
    }

    /// Evict finished blocks whose `max_end < view_start`, freeing RAM immediately.
    /// Returns `true` if at least one block was evicted.
    #[inline]
    pub fn evict_before(&mut self, view_start: f64) -> bool {
        let mut evicted_any = false;
        while let Some(front) = self.blocks.front() {
            if front.max_end() < view_start {
                self.evicted_notes += front.notes.len() as u64;
                self.blocks.pop_front();
                self.first_block_id += 1;
                evicted_any = true;
            } else {
                break;
            }
        }
        evicted_any
    }

    /// Compact all active window blocks into the pre-allocated flat SoA destination slot,
    /// reusing existing `Vec` capacities with zero steady-state reallocations.
    pub fn compact_into(&self, dst: &mut LiveFlatColumn) {
        dst.clear_for_refill(self.evicted_notes);

        for block in &self.blocks {
            dst.block_starts.push(block.start);
            dst.block_max_ends.push(block.max_end());
            for note in &block.notes {
                dst.note_lens.push(note.len);
                dst.note_track_chans.push(note.track_chan);
            }
            dst.block_note_offsets.push(dst.note_lens.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_live_column_evicts_ended_blocks_and_compacts_into_flat_snapshot() {
        let mut col = WorkingLiveColumn::new();
        let tc = TrackAndChannel::new(0, 1);

        let b0 = col.push_block(1.0, [tc, tc].into_iter());
        let b1 = col.push_block(2.0, [tc].into_iter());

        // End both notes in b0 at 1.5 and 1.8; leave b1 unended
        col.end_note(
            LiveNoteLocator {
                block_id: b0,
                note_index: 0,
            },
            1.5,
        );
        col.end_note(
            LiveNoteLocator {
                block_id: b0,
                note_index: 1,
            },
            1.8,
        );

        // Before eviction: both blocks present
        let mut snap = LiveFlatColumn::new();
        col.compact_into(&mut snap);
        assert_eq!(snap.block_count(), 2);
        assert_eq!(snap.evicted_notes, 0);
        assert!((snap.block_max_ends[0] - 1.8).abs() < 1e-6);
        assert!(snap.block_max_ends[1].is_infinite());

        // Advance view_start to 1.9: b0 (max_end = 1.8) must be evicted, b1 stays
        assert!(col.evict_before(1.9));
        col.compact_into(&mut snap);
        assert_eq!(snap.block_count(), 1);
        assert_eq!(snap.evicted_notes, 2);
        assert!((snap.block_starts[0] - 2.0).abs() < 1e-6);

        // Late end_note for evicted b0 is safely ignored; b1 ends cleanly at 2.5
        col.end_note(
            LiveNoteLocator {
                block_id: b0,
                note_index: 0,
            },
            2.1,
        );
        col.end_note(
            LiveNoteLocator {
                block_id: b1,
                note_index: 0,
            },
            2.5,
        );
        assert!(col.evict_before(3.0));
        col.compact_into(&mut snap);
        assert_eq!(snap.block_count(), 0);
        assert_eq!(snap.evicted_notes, 3);
    }
}
