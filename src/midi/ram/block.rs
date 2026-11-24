//! Contiguous Structure-of-Arrays (SoA) storage for in-RAM MIDI note columns.
//!
//! Replaces millions of per-timestamp `Box<[BasicMIDINote]>` individual heap allocations
//! with 5 flat contiguous `Vec`s per key column, eliminating pointer chasing and allocator
//! fragmentation while allowing <1 ms instant teardown on MIDI unload.

use crate::midi::shared::track_channel::TrackAndChannel;

/// Flat Structure-of-Arrays storage for all note blocks of a single MIDI key column.
pub struct FlatNoteColumn {
    /// Block start timestamps in seconds (length = `block_count`).
    pub block_starts: Vec<f64>,
    /// Maximum note length in seconds for each block (length = `block_count`).
    pub block_max_lens: Vec<f32>,
    /// Cumulative prefix offsets into `note_lens` / `note_track_chans` (length = `block_count + 1`).
    /// Block `b` owns note indices `block_note_offsets[b]..block_note_offsets[b + 1]`.
    pub block_note_offsets: Vec<u64>,
    /// Contiguous note lengths in seconds (length = total notes in this key).
    pub note_lens: Vec<f32>,
    /// Contiguous note track/channel identifiers (length = total notes in this key).
    pub note_track_chans: Vec<TrackAndChannel>,
}

impl FlatNoteColumn {
    pub fn new() -> Self {
        let mut block_note_offsets = Vec::with_capacity(64);
        block_note_offsets.push(0);
        Self {
            block_starts: Vec::new(),
            block_max_lens: Vec::new(),
            block_note_offsets,
            note_lens: Vec::new(),
            note_track_chans: Vec::new(),
        }
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_starts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.block_starts.is_empty()
    }

    #[inline]
    pub fn pending_note_count(&self) -> usize {
        let committed = *self.block_note_offsets.last().unwrap_or(&0) as usize;
        self.note_lens.len() - committed
    }

    /// Append a note to the current (unflushed) timestamp block and return its `(block_index, note_index)`.
    #[inline]
    pub fn push_unflushed_note(&mut self, track_chan: TrackAndChannel) -> (usize, usize) {
        let block_index = self.block_starts.len();
        let note_index = self.note_lens.len();
        self.note_lens.push(0.0);
        self.note_track_chans.push(track_chan);
        (block_index, note_index)
    }

    /// Commit all pending unflushed notes as a new block at timestamp `time`.
    #[inline]
    pub fn flush_block(&mut self, time: f64) {
        if self.pending_note_count() > 0 {
            self.block_starts.push(time);
            self.block_max_lens.push(0.0);
            self.block_note_offsets.push(self.note_lens.len() as u64);
        }
    }

    /// Record the end time for a previously committed note, updating its block's `max_length`.
    #[inline]
    pub fn set_note_end_time(&mut self, block_index: usize, note_index: usize, end_time: f64) {
        if block_index >= self.block_starts.len() {
            // Zero-length note still in the current unflushed timestamp; length remains 0.0.
            return;
        }
        let len = (end_time - self.block_starts[block_index]).max(0.0) as f32;
        self.note_lens[note_index] = len;
        if len > self.block_max_lens[block_index] {
            self.block_max_lens[block_index] = len;
        }
    }

    #[inline]
    pub fn block_start(&self, block_index: usize) -> f64 {
        self.block_starts[block_index]
    }

    #[inline]
    pub fn block_max_end(&self, block_index: usize) -> f64 {
        self.block_starts[block_index] + self.block_max_lens[block_index] as f64
    }

    #[inline]
    pub fn block_note_range(&self, block_index: usize) -> (usize, usize) {
        let start = self.block_note_offsets[block_index] as usize;
        let end = self.block_note_offsets[block_index + 1] as usize;
        (start, end)
    }

    #[inline]
    pub fn block_note_count(&self, block_index: usize) -> u64 {
        self.block_note_offsets[block_index + 1] - self.block_note_offsets[block_index]
    }

    pub fn shrink_to_fit(&mut self) {
        self.block_starts.shrink_to_fit();
        self.block_max_lens.shrink_to_fit();
        self.block_note_offsets.shrink_to_fit();
        self.note_lens.shrink_to_fit();
        self.note_track_chans.shrink_to_fit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_note_column_soa_tracks_blocks_lengths_and_max_end() {
        let mut col = FlatNoteColumn::new();
        let tc0 = TrackAndChannel::new(0, 0);
        let tc1 = TrackAndChannel::new(1, 2);

        // Two notes starting at t = 1.0
        let (b0, n0) = col.push_unflushed_note(tc0);
        let (b1, n1) = col.push_unflushed_note(tc1);
        assert_eq!(b0, 0);
        assert_eq!(b1, 0);
        col.flush_block(1.0);

        // Zero-length note at t = 2.0 (ended before flush)
        let (b2, n2) = col.push_unflushed_note(tc0);
        col.set_note_end_time(b2, n2, 2.0);
        col.flush_block(2.0);

        // End the first two notes at t = 1.5 and t = 3.25
        col.set_note_end_time(b0, n0, 1.5);
        col.set_note_end_time(b1, n1, 3.25);

        assert_eq!(col.block_count(), 2);
        assert_eq!(col.block_note_count(0), 2);
        assert_eq!(col.block_note_count(1), 1);
        assert!((col.block_max_end(0) - 3.25).abs() < 1e-6);
        assert!((col.block_max_end(1) - 2.0).abs() < 1e-6);
        assert!((col.note_lens[n0] - 0.5).abs() < 1e-6);
        assert!((col.note_lens[n1] - 2.25).abs() < 1e-6);
        assert!((col.note_lens[n2] - 0.0).abs() < 1e-6);
    }
}
