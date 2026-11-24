use std::ops::Range;

use super::block::FlatNoteColumn;

pub struct InRamNoteColumnViewData {
    /// Number of notes from the beginning of the midi to the start of the render view
    pub notes_to_render_end: u64,
    /// Number of notes from the beginning of the midi to the end of the render view
    pub notes_to_render_start: u64,
    /// The range of blocks that are in the view
    pub block_range: Range<usize>,

    /// Number of notes that have passed the keyboard
    pub notes_to_keyboard: u64,
    /// Number of blocks that have passed the keyboard
    pub blocks_to_keyboard: usize,
}

impl InRamNoteColumnViewData {
    pub fn new() -> Self {
        InRamNoteColumnViewData {
            notes_to_render_end: 0,
            notes_to_render_start: 0,
            block_range: 0..0,
            notes_to_keyboard: 0,
            blocks_to_keyboard: 0,
        }
    }
}

pub struct InRamNoteColumn {
    pub data: InRamNoteColumnViewData,
    pub flat: FlatNoteColumn,
}

impl InRamNoteColumn {
    pub fn new(mut flat: FlatNoteColumn) -> Self {
        flat.shrink_to_fit();
        InRamNoteColumn {
            flat,
            data: InRamNoteColumnViewData::new(),
        }
    }
}
