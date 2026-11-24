//! Zenith-style fixed-timestep headless Vulkan offscreen scene renderer.
//!
//! Renders falling MIDI notes (reusing Wasabi's exact `shaders/notes/notes.geom` & `notes.frag`)
//! plus a standalone Vulkan keyboard mesh pass directly into an offscreen `R8G8B8A8_UNORM`
//! VRAM image at user-specified `width x height`, completely decoupled from `egui`,
//! window swapchain, wall-clock timers, and `audio_playback`.

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui::Color32;
use midi_toolkit::{
    events::{Event, MIDIEventEnum},
    io::MIDIFile as TKMIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time},
        unwrap_items, TimeCaster,
    },
};
use rustc_hash::FxHashMap;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        DescriptorSet, WriteDescriptorSet,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineSubpassType,
            vertex_input::{Vertex, VertexDefinition},
            viewport::Viewport,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::{self, GpuFuture},
};

use crate::{
    gui::window::keyboard_layout::{KeyboardLayout, KeyboardView},
    midi::MIDIColor,
    settings::{MidiSettings, SceneSettings},
    utils,
};

use super::vk_context::OfflineVulkanContext;

/// Matches Wasabi's preview parser tempo constant for 1:1 A/V synchronization.
const WASABI_DEFAULT_TEMPO_US: u32 = 250_000;
const NOTE_VERTEX_CHUNK_CAP: u64 = 524_288;
const KEYBOARD_VERTEX_CAP: u64 = 16_384;

#[derive(Clone, Copy)]
struct OfflineNote {
    start: f64,
    end: f64,
    color: MIDIColor,
}

/// Deterministic, clock-free parsed MIDI note timeline for fixed-timestep video rendering.
pub struct OfflineMidiNoteTimeline {
    columns: Vec<Vec<OfflineNote>>,
    pub duration_sec: f64,
}

impl OfflineMidiNoteTimeline {
    pub fn parse_from_file(midi_path: &Path, midi_settings: &MidiSettings) -> Result<Self, String> {
        let file = File::open(midi_path)
            .map_err(|e| format!("Failed opening MIDI for video render: {e}"))?;
        let midi = TKMIDIFile::open_from_stream(file, None)
            .map_err(|e| format!("Failed parsing MIDI for video render: {e:?}"))?;

        let colors = MIDIColor::new_vec_from_settings(midi.track_count(), midi_settings)
            .map_err(|e| format!("Failed building MIDI color palette: {e}"))?;

        let ppq = midi.ppq();
        let merged = pipe!(
            midi.iter_all_track_events_merged_batches()
            |> TimeCaster::<f64>::cast_event_delta()
            |> cancel_tempo_events(WASABI_DEFAULT_TEMPO_US)
            |> scale_event_time(1.0 / ppq as f64)
            |> unwrap_items()
        );

        let mut columns: Vec<Vec<OfflineNote>> = (0..256).map(|_| Vec::new()).collect();
        let mut unended: Vec<FxHashMap<u32, VecDeque<usize>>> =
            (0..256).map(|_| FxHashMap::default()).collect();

        let mut time = 0.0f64;
        for batch in merged {
            time += batch.delta;
            for ev in batch.iter_events() {
                let track = ev.track as usize;
                match ev.as_event() {
                    Event::NoteOn(e) => {
                        let key = e.key as usize;
                        let tc = ((track as u32) << 4) | (e.channel as u32 & 0x0F);
                        if e.velocity == 0 {
                            if let Some(idx) =
                                unended[key].get_mut(&tc).and_then(|q| q.pop_front())
                            {
                                columns[key][idx].end = time;
                            }
                        } else {
                            let color_idx = (track * 16 + (e.channel as usize))
                                .min(colors.len().saturating_sub(1));
                            let color = colors.get(color_idx).copied().unwrap_or_default();
                            let idx = columns[key].len();
                            columns[key].push(OfflineNote {
                                start: time,
                                end: time,
                                color,
                            });
                            unended[key].entry(tc).or_default().push_back(idx);
                        }
                    }
                    Event::NoteOff(e) => {
                        let key = e.key as usize;
                        let tc = ((track as u32) << 4) | (e.channel as u32 & 0x0F);
                        if let Some(idx) = unended[key].get_mut(&tc).and_then(|q| q.pop_front()) {
                            columns[key][idx].end = time;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Close any notes still held at EOF
        for key in 0..256 {
            for (_, mut q) in unended[key].drain() {
                for idx in q.drain(..) {
                    columns[key][idx].end = time;
                }
            }
        }

        Ok(Self {
            columns,
            duration_sec: time,
        })
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod, Vertex)]
struct NoteVertex {
    #[format(R32G32_SFLOAT)]
    start_length: [f32; 2],
    #[format(R32_UINT)]
    key_color: u32,
    #[format(R32_UINT)]
    border_width: u32,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
struct KeyPositionUniform {
    left: f32,
    right: f32,
    _padding: [u8; 8],
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod, Vertex)]
struct KbVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

pub struct OffscreenSceneRenderer {
    ctx: Arc<OfflineVulkanContext>,
    width: u32,
    height: u32,
    color_image: Arc<Image>,
    color_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,

    // Note pipelines (clear + draw_over for huge Black MIDI frames)
    note_rp_clear: Arc<RenderPass>,
    note_rp_over: Arc<RenderPass>,
    note_pipe_clear: Arc<GraphicsPipeline>,
    note_pipe_over: Arc<GraphicsPipeline>,
    note_vb: Subbuffer<[NoteVertex]>,
    key_locations_ub: Subbuffer<[[KeyPositionUniform; 256]]>,

    // Keyboard overlay pipeline
    kb_rp: Arc<RenderPass>,
    kb_pipe: Arc<GraphicsPipeline>,
    kb_vb: Subbuffer<[KbVertex]>,

    cb_allocator: Arc<StandardCommandBufferAllocator>,
    sd_allocator: Arc<StandardDescriptorSetAllocator>,
    keyboard_layout: KeyboardLayout,

    // Per-column monotonic cursor for O(visible) fixed-timestep scanning
    column_cursors: [usize; 256],
    key_colors_scratch: [Option<MIDIColor>; 256],
    kb_verts_scratch: Vec<KbVertex>,
}

impl OffscreenSceneRenderer {
    pub fn new(ctx: Arc<OfflineVulkanContext>, width: u32, height: u32) -> Result<Self, String> {
        let coded_w = (width + 1) & !1;
        let coded_h = (height + 1) & !1;
        let color_format = Format::R8G8B8A8_UNORM;

        let color_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format: color_format,
                extent: [coded_w, coded_h, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            Default::default(),
        )
        .map_err(|e| format!("Failed creating offscreen RGBA image: {e}"))?;

        let color_view = ImageView::new_default(color_image.clone())
            .map_err(|e| format!("Failed creating offscreen RGBA view: {e}"))?;

        let depth_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::D16_UNORM,
                extent: [coded_w, coded_h, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            Default::default(),
        )
        .map_err(|e| format!("Failed creating offscreen depth image: {e}"))?;
        let depth_view = ImageView::new_default(depth_image)
            .map_err(|e| format!("Failed creating offscreen depth view: {e}"))?;

        // Note RenderPasses
        let note_rp_clear = vulkano::ordered_passes_renderpass!(ctx.device.clone(),
            attachments: {
                final_color: {
                    format: color_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .map_err(|e| format!("note_rp_clear error: {e}"))?;

        let note_rp_over = vulkano::ordered_passes_renderpass!(ctx.device.clone(),
            attachments: {
                final_color: {
                    format: color_format,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
                depth: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .map_err(|e| format!("note_rp_over error: {e}"))?;

        // Note GraphicsPipelines (reusing Wasabi's exact shaders/notes/notes.geom & notes.frag)
        let n_vs = note_vs::load(ctx.device.clone())
            .map_err(|e| format!("note_vs load error: {e}"))?
            .entry_point("main")
            .unwrap();
        let n_gs = note_gs::load(ctx.device.clone())
            .map_err(|e| format!("note_gs load error: {e}"))?
            .entry_point("main")
            .unwrap();
        let n_fs = note_fs::load(ctx.device.clone())
            .map_err(|e| format!("note_fs load error: {e}"))?
            .entry_point("main")
            .unwrap();

        let note_vinput = NoteVertex::per_vertex()
            .definition(&n_vs)
            .map_err(|e| format!("note vertex input error: {e}"))?;
        let note_stages = [
            PipelineShaderStageCreateInfo::new(n_vs),
            PipelineShaderStageCreateInfo::new(n_fs),
            PipelineShaderStageCreateInfo::new(n_gs),
        ];
        let note_layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&note_stages)
                .into_pipeline_layout_create_info(ctx.device.clone())
                .map_err(|e| format!("note layout info error: {e}"))?,
        )
        .map_err(|e| format!("note layout error: {e}"))?;

        let note_subpass_clear = Subpass::from(note_rp_clear.clone(), 0).unwrap();
        let mut note_pipe_info = GraphicsPipelineCreateInfo {
            stages: note_stages.into_iter().collect(),
            vertex_input_state: Some(note_vinput),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            }),
            viewport_state: Some(Default::default()),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                note_subpass_clear.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            subpass: Some(note_subpass_clear.into()),
            ..GraphicsPipelineCreateInfo::layout(note_layout)
        };
        let note_pipe_clear =
            GraphicsPipeline::new(ctx.device.clone(), None, note_pipe_info.clone())
                .map_err(|e| format!("note_pipe_clear error: {e}"))?;
        note_pipe_info.subpass = Some(PipelineSubpassType::BeginRenderPass(
            Subpass::from(note_rp_over.clone(), 0).unwrap(),
        ));
        let note_pipe_over = GraphicsPipeline::new(ctx.device.clone(), None, note_pipe_info)
            .map_err(|e| format!("note_pipe_over error: {e}"))?;

        // Keyboard RenderPass & GraphicsPipeline (loads notes + clears nothing, draws bottom keyboard)
        let kb_rp = vulkano::ordered_passes_renderpass!(ctx.device.clone(),
            attachments: {
                final_color: {
                    format: color_format,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: []
                }
            ]
        )
        .map_err(|e| format!("kb_rp error: {e}"))?;

        let k_vs = kb_vs::load(ctx.device.clone())
            .map_err(|e| format!("kb_vs error: {e}"))?
            .entry_point("main")
            .unwrap();
        let k_fs = kb_fs::load(ctx.device.clone())
            .map_err(|e| format!("kb_fs error: {e}"))?
            .entry_point("main")
            .unwrap();
        let kb_vinput = KbVertex::per_vertex()
            .definition(&k_vs)
            .map_err(|e| format!("kb vertex input error: {e}"))?;
        let kb_stages = [
            PipelineShaderStageCreateInfo::new(k_vs),
            PipelineShaderStageCreateInfo::new(k_fs),
        ];
        let kb_layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&kb_stages)
                .into_pipeline_layout_create_info(ctx.device.clone())
                .map_err(|e| format!("kb layout info error: {e}"))?,
        )
        .map_err(|e| format!("kb layout error: {e}"))?;
        let kb_subpass = Subpass::from(kb_rp.clone(), 0).unwrap();
        let kb_pipe = GraphicsPipeline::new(
            ctx.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: kb_stages.into_iter().collect(),
                vertex_input_state: Some(kb_vinput),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),
                viewport_state: Some(Default::default()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    kb_subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(kb_subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(kb_layout)
            },
        )
        .map_err(|e| format!("kb_pipe error: {e}"))?;

        let note_vb = Buffer::new_slice::<NoteVertex>(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            NOTE_VERTEX_CHUNK_CAP,
        )
        .map_err(|e| format!("note_vb alloc error: {e}"))?;

        let key_locations_ub = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [[KeyPositionUniform::default(); 256]],
        )
        .map_err(|e| format!("key_locations_ub alloc error: {e}"))?;

        let kb_vb = Buffer::new_slice::<KbVertex>(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            KEYBOARD_VERTEX_CAP,
        )
        .map_err(|e| format!("kb_vb alloc error: {e}"))?;

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            ctx.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let sd_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            ctx.device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        Ok(Self {
            ctx,
            width: coded_w,
            height: coded_h,
            color_image,
            color_view,
            depth_view,
            note_rp_clear,
            note_rp_over,
            note_pipe_clear,
            note_pipe_over,
            note_vb,
            key_locations_ub,
            kb_rp,
            kb_pipe,
            kb_vb,
            cb_allocator,
            sd_allocator,
            keyboard_layout: KeyboardLayout::new(&Default::default()),
            column_cursors: [0; 256],
            key_colors_scratch: [None; 256],
            kb_verts_scratch: Vec::with_capacity(KEYBOARD_VERTEX_CAP as usize),
        })
    }

    #[inline]
    pub fn color_view(&self) -> Arc<ImageView> {
        self.color_view.clone()
    }

    /// Render a single deterministic fixed-timestep frame at exact time `time_sec`.
    pub fn render_frame(
        &mut self,
        timeline: &OfflineMidiNoteTimeline,
        time_sec: f64,
        scene: &SceneSettings,
    ) -> Result<(), String> {
        let w = self.width as f32;
        let h = self.height as f32;
        let key_start = *scene.key_range.start() as usize;
        let key_end = *scene.key_range.end() as usize;
        let key_view = self.keyboard_layout.get_view_for_keys(key_start, key_end);

        let key_count = scene.key_range.len().max(1) as f32;
        let keyboard_height = (11.6 / key_count * w).min(h / 2.0);
        let notes_height = (h - keyboard_height).max(1.0);
        let view_range = scene.note_speed.max(0.01);
        let window_end = time_sec + view_range;

        // Upload key positions uniform buffer
        {
            let mut ub = self
                .key_locations_ub
                .write()
                .map_err(|e| format!("key_locations_ub write error: {e}"))?;
            for (dst, key) in ub[0].iter_mut().zip(key_view.iter_all_notes()) {
                *dst = KeyPositionUniform {
                    left: key.left,
                    right: key.right,
                    _padding: [0; 8],
                };
            }
        }

        let border_width =
            utils::calculate_border_width(w, key_view.visible_range.len().max(1) as f32) as u32;

        // Advance per-column cursors monotonically and clear key press colors
        self.key_colors_scratch.fill(None);
        for key in 0..256 {
            let col = &timeline.columns[key];
            let mut cur = self.column_cursors[key];
            while cur < col.len() && col[cur].end < time_sec {
                cur += 1;
            }
            self.column_cursors[key] = cur;
        }

        // Order keys: black keys first, white keys second (matches preview depth/overlap order)
        let mut ordered_keys = [0u8; 256];
        let mut kidx = 0usize;
        for key in 0..256 {
            if key_view.key(key).black {
                ordered_keys[kidx] = key as u8;
                kidx += 1;
            }
        }
        for key in 0..256 {
            if !key_view.key(key).black {
                ordered_keys[kidx] = key as u8;
                kidx += 1;
            }
        }

        let bg = scene.bg_color;
        let bg_clear = [
            bg.r() as f32 / 255.0,
            bg.g() as f32 / 255.0,
            bg.b() as f32 / 255.0,
            1.0,
        ];

        // Stream note vertices in chunks (supports arbitrarily dense Black MIDI frames)
        let mut first_pass = true;
        let mut key_order_pos = 0usize;
        let mut note_in_col_offset = 0usize;

        loop {
            let written: u32 = {
                let mut vb = self
                    .note_vb
                    .write()
                    .map_err(|e| format!("note_vb write error: {e}"))?;
                let cap = vb.len();
                let mut count = 0usize;

                'fill: while key_order_pos < 256 {
                    let key = ordered_keys[key_order_pos] as usize;
                    let col = &timeline.columns[key];
                    let start_idx = self.column_cursors[key] + note_in_col_offset;

                    let mut idx = start_idx;
                    while idx < col.len() {
                        let note = col[idx];
                        if note.start > window_end {
                            break;
                        }
                        if note.end >= time_sec {
                            if note.start <= time_sec && note.end > time_sec {
                                self.key_colors_scratch[key] = Some(note.color);
                            }
                            if count >= cap {
                                note_in_col_offset = idx - self.column_cursors[key];
                                break 'fill;
                            }
                            vb[count] = NoteVertex {
                                start_length: [
                                    (note.start - time_sec) as f32,
                                    (note.end - note.start).max(0.0) as f32,
                                ],
                                key_color: (key as u32) | (note.color.as_u32() << 8),
                                border_width,
                            };
                            count += 1;
                        }
                        idx += 1;
                    }
                    key_order_pos += 1;
                    note_in_col_offset = 0;
                }
                count as u32
            };

            let (rp, pipe, clears) = if first_pass {
                first_pass = false;
                (
                    self.note_rp_clear.clone(),
                    self.note_pipe_clear.clone(),
                    vec![Some(bg_clear.into()), Some(1.0f32.into())],
                )
            } else if written == 0 {
                break;
            } else {
                (
                    self.note_rp_over.clone(),
                    self.note_pipe_over.clone(),
                    vec![None, None],
                )
            };

            self.execute_note_pass(rp, pipe, clears, written, w, notes_height, view_range as f32, scene.skin.id())?;

            if key_order_pos >= 256 {
                break;
            }
        }

        // Draw bottom piano keyboard overlay directly onto the offscreen RGBA image
        Self::build_keyboard_vertices(
            &key_view,
            &self.key_colors_scratch,
            &mut self.kb_verts_scratch,
            w,
            h,
            notes_height,
            keyboard_height,
            scene.bar_color,
            scene.skin,
        );
        self.execute_keyboard_pass(w, h)?;

        Ok(())
    }

    fn execute_note_pass(
        &self,
        rp: Arc<RenderPass>,
        pipe: Arc<GraphicsPipeline>,
        clears: Vec<Option<vulkano::format::ClearValue>>,
        vertex_count: u32,
        w: f32,
        notes_height: f32,
        view_range: f32,
        skin_id: u32,
    ) -> Result<(), String> {
        let framebuffer = Framebuffer::new(
            rp,
            FramebufferCreateInfo {
                attachments: vec![self.color_view.clone(), self.depth_view.clone()],
                ..Default::default()
            },
        )
        .map_err(|e| format!("note framebuffer error: {e}"))?;

        let layout = pipe.layout();
        let desc_layout = layout.set_layouts().first().unwrap();
        let set = DescriptorSet::new(
            self.sd_allocator.clone(),
            desc_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.key_locations_ub.clone())],
            [],
        )
        .map_err(|e| format!("note descriptor set error: {e}"))?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.ctx.gfx_queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("note cmd builder error: {e}"))?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: clears,
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("note begin_render_pass error: {e}"))?;

        if vertex_count > 0 {
            let push = note_gs::PushConstants {
                height_time: view_range,
                win_width: w,
                win_height: notes_height,
                skin_id,
            };
            unsafe {
                builder
                    .bind_pipeline_graphics(pipe.clone())
                    .map_err(|e| format!("note bind pipe error: {e}"))?
                    .set_viewport(
                        0,
                        vec![Viewport {
                            offset: [0.0, 0.0],
                            extent: [w, notes_height],
                            depth_range: 0.0..=1.0,
                        }]
                        .into(),
                    )
                    .map_err(|e| format!("note viewport error: {e}"))?
                    .push_constants(layout.clone(), 0, push)
                    .map_err(|e| format!("note push constants error: {e}"))?
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, set)
                    .map_err(|e| format!("note bind desc error: {e}"))?
                    .bind_vertex_buffers(0, self.note_vb.clone())
                    .map_err(|e| format!("note bind vb error: {e}"))?
                    .draw(vertex_count, 1, 0, 0)
                    .map_err(|e| format!("note draw error: {e}"))?;
            }
        }

        builder
            .end_render_pass(Default::default())
            .map_err(|e| format!("note end_render_pass error: {e}"))?;

        let cmd_buf = builder
            .build()
            .map_err(|e| format!("note cmd build error: {e}"))?;

        sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.gfx_queue.clone(), cmd_buf)
            .map_err(|e| format!("note execute error: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("note flush error: {e}"))?
            .wait(None)
            .map_err(|e| format!("note wait error: {e}"))?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_keyboard_vertices(
        key_view: &KeyboardView,
        key_colors: &[Option<MIDIColor>; 256],
        kb_verts: &mut Vec<KbVertex>,
        w: f32,
        h: f32,
        kb_top: f32,
        kb_h: f32,
        bar_color: Color32,
        skin: crate::settings::SkinPreset,
    ) {
        kb_verts.clear();
        let note_border = utils::calculate_border_width(w, key_view.visible_range.len().max(1) as f32);
        let key_border = (note_border / 2.0).max(1.0);
        let bar = kb_h * 0.06;
        let md_height = kb_h * 0.048;
        let black_overlap = bar / 2.35;
        let top = kb_top + bar;
        let bottom = kb_top + kb_h;
        let black_bottom = bottom - kb_h * 0.34;

        let push_quad = |verts: &mut Vec<KbVertex>,
                         x0: f32,
                         y0: f32,
                         x1: f32,
                         y1: f32,
                         c_top: [f32; 4],
                         c_bot: [f32; 4]| {
            let nx0 = (x0 / w) * 2.0 - 1.0;
            let nx1 = (x1 / w) * 2.0 - 1.0;
            let ny0 = (y0 / h) * 2.0 - 1.0;
            let ny1 = (y1 / h) * 2.0 - 1.0;
            let v0 = KbVertex { pos: [nx0, ny0], color: c_top };
            let v1 = KbVertex { pos: [nx1, ny0], color: c_top };
            let v2 = KbVertex { pos: [nx0, ny1], color: c_bot };
            let v3 = KbVertex { pos: [nx1, ny1], color: c_bot };
            verts.extend_from_slice(&[v0, v1, v2, v2, v1, v3]);
        };

        let c_rgba = |r: u8, g: u8, b: u8| -> [f32; 4] {
            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
        };
        let scale_c = |c: [f32; 4], k: f32| -> [f32; 4] {
            [
                (c[0] * k).clamp(0.0, 1.0),
                (c[1] * k).clamp(0.0, 1.0),
                (c[2] * k).clamp(0.0, 1.0),
                1.0,
            ]
        };

        let lift_c = |c: [f32; 4], t: f32| -> [f32; 4] {
            [
                (c[0] + (1.0 - c[0]) * t).clamp(0.0, 1.0),
                (c[1] + (1.0 - c[1]) * t).clamp(0.0, 1.0),
                (c[2] + (1.0 - c[2]) * t).clamp(0.0, 1.0),
                1.0,
            ]
        };

        let (wk_top_sh, wk_mid, wk_face, wk_lip_t, wk_lip_b, wk_div, bk_outer, bk_t, bk_b) = match skin {
            crate::settings::SkinPreset::Wasaby => (
                c_rgba(110, 110, 110), c_rgba(210, 210, 210), c_rgba(255, 255, 255),
                c_rgba(190, 190, 190), c_rgba(120, 120, 120), c_rgba(40, 40, 40),
                c_rgba(20, 20, 20), c_rgba(55, 55, 55), c_rgba(32, 32, 32),
            ),
            crate::settings::SkinPreset::Wynthesia => (
                c_rgba(128, 126, 120), c_rgba(232, 230, 224), c_rgba(252, 251, 246),
                c_rgba(205, 202, 195), c_rgba(140, 138, 132), c_rgba(56, 56, 62),
                c_rgba(24, 26, 32), c_rgba(56, 60, 70), c_rgba(34, 38, 46),
            ),
            crate::settings::SkinPreset::WfaPlus => (
                c_rgba(155, 155, 155), c_rgba(240, 240, 240), c_rgba(255, 255, 255),
                c_rgba(185, 185, 185), c_rgba(115, 115, 115), c_rgba(24, 24, 24),
                c_rgba(10, 10, 10), c_rgba(36, 36, 36), c_rgba(16, 16, 16),
            ),
            crate::settings::SkinPreset::Zanith => (
                c_rgba(95, 102, 116), c_rgba(206, 212, 224), c_rgba(236, 240, 248),
                c_rgba(155, 162, 176), c_rgba(86, 92, 106), c_rgba(22, 25, 34),
                c_rgba(12, 14, 22), c_rgba(42, 48, 66), c_rgba(18, 22, 34),
            ),
        };

        // White keys
        for (i, key) in key_view.iter_visible_keys() {
            if !key.black {
                let x0 = key.left * w;
                let x1 = key.right * w;
                if let Some(mc) = key_colors[i] {
                    let base = c_rgba(mc.red(), mc.green(), mc.blue());
                    let (t_c, m_c, b_c) = match skin {
                        crate::settings::SkinPreset::Wynthesia => (scale_c(base, 0.55), lift_c(base, 0.14), lift_c(base, 0.26)),
                        crate::settings::SkinPreset::WfaPlus => (scale_c(base, 0.78), base, scale_c(base, 0.92)),
                        crate::settings::SkinPreset::Zanith => (lift_c(base, 0.35), lift_c(base, 0.10), scale_c(base, 0.84)),
                        crate::settings::SkinPreset::Wasaby => (scale_c(base, 0.3), scale_c(base, 0.6), base),
                    };
                    push_quad(kb_verts, x0, top, x1, top + black_overlap, t_c, m_c);
                    push_quad(kb_verts, x0, top + black_overlap, x1, bottom, m_c, b_c);
                    push_quad(kb_verts, x0, bottom - key_border * 2.0, x1, bottom, scale_c(base, 0.3), scale_c(base, 0.6));
                } else {
                    push_quad(kb_verts, x0, top, x1, top + black_overlap, wk_top_sh, wk_mid);
                    push_quad(kb_verts, x0, top + black_overlap, x1, bottom - md_height, wk_mid, wk_face);
                    push_quad(kb_verts, x0, bottom - md_height, x1, bottom, wk_lip_t, wk_lip_b);
                }
                // Right border
                push_quad(kb_verts, x1 - key_border, top, x1, bottom, wk_div, wk_div);
            }
        }

        // Top separator bar
        let bc = c_rgba(bar_color.r(), bar_color.g(), bar_color.b());
        if skin == crate::settings::SkinPreset::Zanith {
            push_quad(kb_verts, 0.0, kb_top, w, top - black_overlap, c_rgba(44, 50, 64), c_rgba(18, 22, 30));
            push_quad(kb_verts, 0.0, top - black_overlap, w, top, lift_c(bc, 0.25), scale_c(bc, 0.45));
        } else {
            push_quad(kb_verts, 0.0, kb_top, w, top - black_overlap, c_rgba(90, 90, 90), c_rgba(40, 40, 40));
            push_quad(kb_verts, 0.0, top - black_overlap, w, top, scale_c(bc, 0.3), bc);
        }

        // Black keys
        for (i, key) in key_view.iter_visible_keys() {
            if key.black {
                let x0 = key.left * w;
                let x1 = key.right * w;
                let inset = 2.0 * key_border;
                if let Some(mc) = key_colors[i] {
                    let base = c_rgba(mc.red(), mc.green(), mc.blue());
                    let top_face = if skin == crate::settings::SkinPreset::Zanith { lift_c(base, 0.30) } else { base };
                    let dark = scale_c(base, 0.72);
                    push_quad(kb_verts, x0, top, x1, black_bottom, dark, scale_c(base, 0.45));
                    push_quad(kb_verts, x0 + inset, top, x1 - inset, black_bottom - md_height * 0.5, top_face, dark);
                } else {
                    push_quad(kb_verts, x0, top, x1, black_bottom, bk_outer, bk_outer);
                    push_quad(kb_verts, x0 + inset, top, x1 - inset, black_bottom - md_height, bk_t, bk_b);
                }
            }
        }
    }

    fn execute_keyboard_pass(&self, w: f32, h: f32) -> Result<(), String> {
        let vert_count = self
            .kb_verts_scratch
            .len()
            .min(KEYBOARD_VERTEX_CAP as usize);
        if vert_count == 0 {
            return Ok(());
        }

        {
            let mut vb = self
                .kb_vb
                .write()
                .map_err(|e| format!("kb_vb write error: {e}"))?;
            vb[..vert_count].copy_from_slice(&self.kb_verts_scratch[..vert_count]);
        }

        let framebuffer = Framebuffer::new(
            self.kb_rp.clone(),
            FramebufferCreateInfo {
                attachments: vec![self.color_view.clone()],
                ..Default::default()
            },
        )
        .map_err(|e| format!("kb framebuffer error: {e}"))?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.ctx.gfx_queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("kb cmd builder error: {e}"))?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|e| format!("kb begin_render_pass error: {e}"))?;

        unsafe {
            builder
                .bind_pipeline_graphics(self.kb_pipe.clone())
                .map_err(|e| format!("kb bind pipe error: {e}"))?
                .set_viewport(
                    0,
                    vec![Viewport {
                        offset: [0.0, 0.0],
                        extent: [w, h],
                        depth_range: 0.0..=1.0,
                    }]
                    .into(),
                )
                .map_err(|e| format!("kb viewport error: {e}"))?
                .bind_vertex_buffers(0, self.kb_vb.clone())
                .map_err(|e| format!("kb bind vb error: {e}"))?
                .draw(vert_count as u32, 1, 0, 0)
                .map_err(|e| format!("kb draw error: {e}"))?;
        }

        builder
            .end_render_pass(Default::default())
            .map_err(|e| format!("kb end_render_pass error: {e}"))?;

        let cmd_buf = builder
            .build()
            .map_err(|e| format!("kb cmd build error: {e}"))?;

        sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.gfx_queue.clone(), cmd_buf)
            .map_err(|e| format!("kb execute error: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("kb flush error: {e}"))?
            .wait(None)
            .map_err(|e| format!("kb wait error: {e}"))?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn color_image(&self) -> &Arc<Image> {
        &self.color_image
    }
}

mod note_gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "shaders/notes/notes.geom",
    }
}

mod note_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 start_length;
layout(location = 1) in uint key_color;
layout(location = 2) in uint border_width;

layout(location = 0) out vec2 v_start_length;
layout(location = 1) out uint v_key_color;
layout(location = 2) out uint v_border_width;

void main() {
    v_start_length = start_length;
    v_key_color = key_color;
    v_border_width = border_width;
}"
    }
}

mod note_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/notes/notes.frag"
    }
}

mod kb_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 v_color;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    v_color = color;
}"
    }
}

mod kb_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = v_color;
}"
    }
}
