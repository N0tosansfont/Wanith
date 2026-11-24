mod notes_render_pass;

use std::{cell::UnsafeCell, sync::Arc};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use vulkano::{image::view::ImageView, sync::GpuFuture};

use crate::{
    gui::{window::keyboard_layout::KeyboardView, GuiRenderer},
    midi::{DisplacedMIDINote, MIDIColor, MIDIFile, MIDINoteColumnView, MIDINoteViews},
    utils,
};

use self::notes_render_pass::{NotePassStatus, NoteRenderPass, NoteVertex};

use super::RenderResultData;

#[derive(Default, Clone, Copy)]
struct ColumnReturnData {
    polyphony: usize,
    written_notes: usize,
}

pub struct NoteRenderer {
    render_pass: NoteRenderPass,
    thread_pool: rayon::ThreadPool,
}

struct UnsafeSyncCell<T>(UnsafeCell<T>);

impl<T> UnsafeSyncCell<T> {
    pub fn new(value: T) -> Self {
        UnsafeSyncCell(UnsafeCell::new(value))
    }

    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.0.get()
    }
}

unsafe impl<T> Sync for UnsafeSyncCell<T> {}
unsafe impl<T> Send for UnsafeSyncCell<T> {}

impl NoteRenderer {
    pub fn new(renderer: &GuiRenderer) -> NoteRenderer {
        NoteRenderer {
            render_pass: NoteRenderPass::new(renderer),
            thread_pool: rayon::ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    pub fn draw(
        &mut self,
        key_view: &KeyboardView,
        final_image: Arc<ImageView>,
        midi_file: &mut impl MIDIFile,
        view_range: f64,
        skin: crate::settings::SkinPreset,
    ) -> (RenderResultData, Option<Box<dyn GpuFuture>>) {
        let note_views = midi_file.get_current_column_views(view_range);

        struct ColumnViewInfo<Iter: ExactSizeIterator<Item = DisplacedMIDINote> + Send> {
            offset: usize,
            iter: Iter,
            key: u8,
            remaining: usize,
            color: Option<MIDIColor>,
            border_width: f32,
        }

        let mut total_notes = 0;

        let columns: [_; 256] = std::array::from_fn(|i| note_views.get_column(i));

        let mut columns_view_info = Vec::with_capacity(256);

        let border_width = utils::calculate_border_width(
            final_image.image().extent()[0] as f32,
            key_view.visible_range.len() as f32,
        );

        // Black keys first
        for (i, column) in columns.iter().enumerate() {
            if key_view.key(i).black {
                let iter = column.iterate_displaced_notes();
                let length = iter.len();
                columns_view_info.push(ColumnViewInfo {
                    offset: total_notes,
                    iter,
                    key: i as u8,
                    remaining: length,
                    color: None,
                    border_width,
                });
                total_notes += length;
            }
        }

        // Then white keys after
        for (i, column) in columns.iter().enumerate() {
            if !key_view.key(i).black {
                let iter = column.iterate_displaced_notes();
                let length = iter.len();
                columns_view_info.push(ColumnViewInfo {
                    offset: total_notes,
                    iter,
                    key: i as u8,
                    remaining: length,
                    color: None,
                    border_width,
                });
                total_notes += length;
            }
        }

        let mut notes_pushed = 0;
        let mut polyphony = 0;

        let view_range = note_views.range().length() as f32;

        let out_future =
            self.render_pass
                .draw(final_image, key_view, view_range, skin.id(), |buffer| {
                    let buffer_length = buffer.len() as usize;

                    let buffer_writer = UnsafeSyncCell::new(buffer.write().unwrap());

                    // Parallel write of all 256 columns into the active A/B staging buffer
                    // using zero-allocation rayon reduce (no intermediate Vec collect).
                    let column_data = self.thread_pool.install(|| {
                        columns_view_info
                            .par_iter_mut()
                            .rev()
                            .map(|column| {
                                if column.remaining == 0 {
                                    return ColumnReturnData::default();
                                }

                                let offset =
                                    (column.offset as i64 - notes_pushed as i64).max(0) as usize;

                                if offset >= buffer_length {
                                    return ColumnReturnData::default();
                                }

                                let remaining_buffer_space = buffer_length - offset;
                                let allowed_to_write =
                                    column.remaining.min(remaining_buffer_space);

                                let mut poly = 0;

                                unsafe {
                                    let buffer = buffer_writer.get_mut();

                                    for i in 0..allowed_to_write {
                                        if let Some(note) = column.iter.next() {
                                            buffer[i + offset] = NoteVertex::new(
                                                note.start,
                                                note.len,
                                                column.key,
                                                note.color.as_u32(),
                                                column.border_width as u32,
                                            );

                                            if note.start <= 0.0 && note.start + note.len > 0.0 {
                                                poly += 1;
                                                if column.color.is_none() {
                                                    column.color = Some(note.color);
                                                }
                                            }
                                        } else {
                                            break;
                                        }
                                    }
                                }

                                column.remaining -= allowed_to_write;

                                ColumnReturnData {
                                    polyphony: poly,
                                    written_notes: allowed_to_write,
                                }
                            })
                            .reduce(ColumnReturnData::default, |a, b| ColumnReturnData {
                                polyphony: a.polyphony + b.polyphony,
                                written_notes: a.written_notes + b.written_notes,
                            })
                    });

                    polyphony += column_data.polyphony;
                    notes_pushed += column_data.written_notes;

                    if notes_pushed >= total_notes {
                        NotePassStatus::Finished {
                            remaining: column_data.written_notes as u32,
                        }
                    } else {
                        NotePassStatus::HasMoreNotes
                    }
                });

        // Scatter key colors by key index without sorting the whole view info vector
        let mut key_colors = vec![None; 256];
        for column in &columns_view_info {
            key_colors[column.key as usize] = column.color;
        }

        (
            RenderResultData {
                notes_rendered: notes_pushed as u64,
                polyphony: Some(polyphony as u64),
                key_colors,
            },
            out_future,
        )
    }
}
