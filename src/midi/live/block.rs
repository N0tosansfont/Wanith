//! Flat Structure-of-Arrays (SoA) A/B snapshot payload for live-streamed MIDIs.
//!
//! Owned exclusively by the two slots of [`crate::midi::double_buffer::DoubleBuffer`]:
//! the background note worker compacts active window blocks into the inactive slot (B)
//! while the render thread reads the active slot (A) wait-free with zero `UnsafeCell`
//! cross-thread mutation and zero per-note `Arc` reference counting.

use crate::midi::shared::track_channel::TrackAndChannel;

#[derive(Debug, Clone, Copy)]
pub struct LiveMIDINote {
    pub len: f32,
    pub track_chan: TrackAndChannel,
}

/// Contiguous flat SoA snapshot of one key column's currently active sliding window.
pub struct LiveFlatColumn {
    pub block_starts: Vec<f64>,
    pub block_max_ends: Vec<f64>,
    pub block_note_offsets: Vec<u32>,
    pub note_lens: Vec<f32>,
    pub note_track_chans: Vec<TrackAndChannel>,
    /// Cumulative number of notes already evicted from the front of this column's window.
    pub evicted_notes: u64,
}

impl LiveFlatColumn {
    pub fn new() -> Self {
        let mut block_note_offsets = Vec::with_capacity(64);
        block_note_offsets.push(0);
        Self {
            block_starts: Vec::new(),
            block_max_ends: Vec::new(),
            block_note_offsets,
            note_lens: Vec::new(),
            note_track_chans: Vec::new(),
            evicted_notes: 0,
        }
    }

    #[inline]
    pub fn clear_for_refill(&mut self, evicted_notes: u64) {
        self.block_starts.clear();
        self.block_max_ends.clear();
        self.block_note_offsets.clear();
        self.block_note_offsets.push(0);
        self.note_lens.clear();
        self.note_track_chans.clear();
        self.evicted_notes = evicted_notes;
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_starts.len()
    }

    #[inline]
    pub fn block_note_range(&self, block_index: usize) -> (usize, usize) {
        let start = self.block_note_offsets[block_index] as usize;
        let end = self.block_note_offsets[block_index + 1] as usize;
        (start, end)
    }

    #[inline]
    pub fn block_note_count(&self, block_index: usize) -> usize {
        (self.block_note_offsets[block_index + 1] - self.block_note_offsets[block_index]) as usize
    }
}

/// Complete 256-key sliding window snapshot exchanged atomically across the A/B double buffer.
pub struct LiveWindowSnapshot {
    pub columns: Box<[LiveFlatColumn]>,
}

impl LiveWindowSnapshot {
    pub fn new() -> Self {
        let mut columns = Vec::with_capacity(256);
        for _ in 0..256 {
            columns.push(LiveFlatColumn::new());
        }
        Self {
            columns: columns.into_boxed_slice(),
        }
    }
}
