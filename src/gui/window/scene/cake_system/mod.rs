use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, RenderPassBeginInfo,
        SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::Viewport,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::{self, GpuFuture},
};

use crate::{
    gui::{window::keyboard_layout::KeyboardView, GuiRenderer},
    midi::{CakeBlock, CakeMIDIFile, IntVector4, MIDIColor},
};

use super::RenderResultData;

const BUFFER_ARRAY_LEN: u64 = 256;
/// Cap the temporary one-shot upload staging buffer to 4M `IntVector4` elements (64 MB) so
/// even billion-note MIDIs never spike CPU RAM during `DEVICE_LOCAL` VRAM transfer.
const MAX_STAGING_CHUNK_ELEMS: usize = 4 * 1024 * 1024;

struct CakeBuffer {
    data: Subbuffer<[IntVector4]>,
    start: i32,
    end: i32,
}

#[derive(Default, Debug, Copy, Clone, Zeroable, Pod, Vertex)]
#[repr(C)]
struct CakeNoteColumn {
    #[format(R32_SFLOAT)]
    left: f32,
    #[format(R32_SFLOAT)]
    right: f32,
    #[format(R32_SINT)]
    start: i32,
    #[format(R32_SINT)]
    end: i32,
    #[format(R32_SINT)]
    buffer_index: i32,
    #[format(R32_SINT)]
    border_width: i32,
}

/// 16-byte per-key metadata written by `cake_meta.comp` into the 4 KB A/B readback buffer.
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct CakeKeyMeta {
    color: u32,
    has_note: u32,
    passed_start: u32,
    passed_end: u32,
}

/// A/B double-buffered per-frame GPU resources (6 KB column vertices + 4 KB compute readback).
struct CakeFrameSlot {
    vertex_buffer: Subbuffer<[CakeNoteColumn]>,
    meta_readback: Subbuffer<[CakeKeyMeta]>,
    meta_descriptor: Arc<DescriptorSet>,
    has_valid_meta: bool,
}

struct CachedFramebuffer {
    image_ptr: usize,
    extent: [u32; 3],
    framebuffer: Arc<Framebuffer>,
}

pub struct CakeRenderer {
    gfx_queue: Arc<Queue>,
    buffers: Vec<CakeBuffer>,
    /// Cached `set = 0` descriptor binding all 256 `DEVICE_LOCAL` key tree SSBOs (created once per MIDI).
    tree_descriptor: Option<Arc<DescriptorSet>>,
    meta_tree_descriptor: Option<Arc<DescriptorSet>>,
    pipeline_clear: Arc<GraphicsPipeline>,
    meta_pipeline: Arc<ComputePipeline>,
    render_pass_clear: Arc<RenderPass>,
    allocator: Arc<StandardMemoryAllocator>,
    depth_buffer: Arc<ImageView>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    sd_allocator: Arc<StandardDescriptorSetAllocator>,
    /// A/B double-buffered frame slots (`[slot_a, slot_b]`) eliminating frame-boundary contention.
    frame_slots: [CakeFrameSlot; 2],
    slot_index: usize,
    /// Cached framebuffers keyed by swapchain scene image pointer (zero per-frame allocations).
    fb_cache: Vec<CachedFramebuffer>,
    /// Reusable output color vector returned in `RenderResultData`.
    cached_colors: Vec<Option<MIDIColor>>,
    current_file_id: Option<u64>,
}

impl CakeRenderer {
    pub fn new(renderer: &GuiRenderer) -> CakeRenderer {
        let allocator = Arc::new(StandardMemoryAllocator::new_default(
            renderer.device.clone(),
        ));
        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            renderer.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let sd_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            renderer.device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let gfx_queue = renderer.queue.clone();

        let render_pass_clear = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                final_color: {
                    format: renderer.format,
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
        .unwrap();

        let depth_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    extent: [1, 1, 1],
                    format: Format::D16_UNORM,
                    usage: ImageUsage::SAMPLED | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ..Default::default()
                },
                Default::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let vs = vs::load(gfx_queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(gfx_queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let gs = gs::load(gfx_queue.device().clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let vertex_input_state = CakeNoteColumn::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
            PipelineShaderStageCreateInfo::new(gs),
        ];
        let layout = PipelineLayout::new(
            renderer.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(renderer.device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass_clear.clone(), 0).unwrap();

        let pipeline_clear = GraphicsPipeline::new(
            renderer.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::PointList,
                    ..Default::default()
                }),
                viewport_state: Some(Default::default()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        // 4 KB per-frame metadata compute pipeline sharing the exact same `set = 0` tree layout
        let meta_cs = meta_cs::load(gfx_queue.device().clone())
            .expect("failed to create cake_meta compute shader")
            .entry_point("main")
            .unwrap();
        let meta_stage = PipelineShaderStageCreateInfo::new(meta_cs);
        let meta_layout = PipelineLayout::new(
            renderer.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&meta_stage])
                .into_pipeline_layout_create_info(renderer.device.clone())
                .unwrap(),
        )
        .unwrap();
        let meta_pipeline = ComputePipeline::new(
            renderer.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(meta_stage, meta_layout),
        )
        .unwrap();

        let meta_out_desc_layout = meta_pipeline.layout().set_layouts()[1].clone();
        let create_slot = || {
            let vertex_buffer = Buffer::new_slice(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                BUFFER_ARRAY_LEN,
            )
            .unwrap();

            let meta_readback = Buffer::new_slice(
                allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                BUFFER_ARRAY_LEN,
            )
            .unwrap();

            let meta_descriptor = DescriptorSet::new(
                sd_allocator.clone(),
                meta_out_desc_layout.clone(),
                [WriteDescriptorSet::buffer(0, meta_readback.clone())],
                [],
            )
            .unwrap();

            CakeFrameSlot {
                vertex_buffer,
                meta_readback,
                meta_descriptor,
                has_valid_meta: false,
            }
        };

        let frame_slots = [create_slot(), create_slot()];

        CakeRenderer {
            gfx_queue,
            buffers: Vec::with_capacity(BUFFER_ARRAY_LEN as usize),
            tree_descriptor: None,
            meta_tree_descriptor: None,
            pipeline_clear,
            meta_pipeline,
            render_pass_clear,
            depth_buffer,
            allocator,
            cb_allocator,
            sd_allocator,
            frame_slots,
            slot_index: 0,
            fb_cache: Vec::with_capacity(4),
            cached_colors: vec![None; BUFFER_ARRAY_LEN as usize],
            current_file_id: None,
        }
    }

    /// Upload all 256 key binary trees into `DEVICE_LOCAL` VRAM using a single bounded (<= 64 MB)
    /// staging buffer, then build and cache the `set = 0` descriptor set once per MIDI file.
    ///
    /// Taking `Vec<CakeBlock>` by value ensures CPU-side trees are immediately dropped upon return,
    /// achieving 0% CPU/VRAM double residency.
    fn upload_trees_to_device_local(&mut self, blocks: Vec<CakeBlock>) {
        self.buffers.clear();
        self.tree_descriptor = None;
        self.meta_tree_descriptor = None;
        for slot in &mut self.frame_slots {
            slot.has_valid_meta = false;
        }

        if blocks.is_empty() {
            return;
        }

        let max_elems = blocks
            .iter()
            .map(|b| b.tree.len())
            .max()
            .unwrap_or(1)
            .clamp(1, MAX_STAGING_CHUNK_ELEMS);

        let staging: Subbuffer<[IntVector4]> = Buffer::new_slice(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            max_elems as u64,
        )
        .unwrap();

        let device: Arc<Device> = self.gfx_queue.device().clone();

        for block in &blocks {
            let elem_count = block.tree.len().max(1) as u64;
            let dst: Subbuffer<[IntVector4]> = Buffer::new_slice(
                self.allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                elem_count,
            )
            .unwrap();

            // Copy in chunks <= MAX_STAGING_CHUNK_ELEMS (<= 64 MB peak staging RAM)
            let mut offset = 0usize;
            while offset < block.tree.len() {
                let chunk_len = (block.tree.len() - offset).min(max_elems);
                {
                    let mut map = staging.write().unwrap();
                    map[..chunk_len].copy_from_slice(&block.tree[offset..offset + chunk_len]);
                }

                let mut cb = AutoCommandBufferBuilder::primary(
                    self.cb_allocator.clone(),
                    self.gfx_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let src_slice = staging.clone().slice(0..chunk_len as u64);
                let dst_slice = dst
                    .clone()
                    .slice(offset as u64..(offset + chunk_len) as u64);
                cb.copy_buffer(CopyBufferInfo::buffers(src_slice, dst_slice))
                    .unwrap();

                let cmd = cb.build().unwrap();
                sync::now(device.clone())
                    .then_execute(self.gfx_queue.clone(), cmd)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();

                offset += chunk_len;
            }

            self.buffers.push(CakeBuffer {
                data: dst,
                start: block.start_time as i32,
                end: block.end_time as i32,
            });
        }

        // Drop staging and all CPU-side `Vec<IntVector4>` trees immediately!
        drop(staging);
        drop(blocks);

        let desc_layout = self.pipeline_clear.layout().set_layouts()[0].clone();
        let data_descriptor = DescriptorSet::new(
            self.sd_allocator.clone(),
            desc_layout,
            [WriteDescriptorSet::buffer_array(
                0,
                0,
                self.buffers.iter().map(|b| b.data.clone()),
            )],
            [],
        )
        .unwrap();
        self.tree_descriptor = Some(data_descriptor);
        let meta_data_descriptor = DescriptorSet::new(
            self.sd_allocator.clone(),
            self.meta_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer_array(
                0,
                0,
                self.buffers.iter().map(|b| b.data.clone()),
            )],
            [],
        )
        .unwrap();
        self.meta_tree_descriptor = Some(meta_data_descriptor);
    }

    fn get_or_create_framebuffer(&mut self, final_image: &Arc<ImageView>) -> Arc<Framebuffer> {
        let img_ptr = Arc::as_ptr(final_image) as usize;
        let extent = final_image.image().extent();

        if let Some(entry) = self
            .fb_cache
            .iter()
            .find(|e| e.image_ptr == img_ptr && e.extent == extent)
        {
            return entry.framebuffer.clone();
        }

        if self.fb_cache.len() >= 8 {
            self.fb_cache.clear();
        }

        let fb = Framebuffer::new(
            self.render_pass_clear.clone(),
            FramebufferCreateInfo {
                attachments: vec![final_image.clone(), self.depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        self.fb_cache.push(CachedFramebuffer {
            image_ptr: img_ptr,
            extent,
            framebuffer: fb.clone(),
        });
        fb
    }

    /// Read 4 KB metadata computed by `cake_meta.comp` from a completed A/B slot.
    fn collect_meta_from_slot(
        &mut self,
        slot_idx: usize,
        midi_file: &CakeMIDIFile,
    ) -> (u64, Vec<Option<MIDIColor>>) {
        let key_count = self.buffers.len();
        self.cached_colors.resize(key_count, None);

        let mut rendered_notes = 0u64;
        let mut total_passed = 0u64;

        let read_guard = self.frame_slots[slot_idx].meta_readback.read().unwrap();
        for (i, meta) in read_guard.iter().take(key_count).enumerate() {
            self.cached_colors[i] = if meta.has_note != 0 {
                Some(MIDIColor::from_u32(meta.color))
            } else {
                None
            };
            total_passed += meta.passed_start as u64;
            let span = meta.passed_end.saturating_sub(meta.passed_start) as u64;
            rendered_notes += span + u64::from(meta.has_note != 0);
        }
        drop(read_guard);

        midi_file.set_passed_notes(total_passed);
        (rendered_notes, self.cached_colors.clone())
    }

    pub fn draw(
        &mut self,
        key_view: &KeyboardView,
        final_image: Arc<ImageView>,
        midi_file: &mut CakeMIDIFile,
        view_range: f64,
        skin: crate::settings::SkinPreset,
    ) -> (RenderResultData, Option<Box<dyn GpuFuture>>) {
        let img_dims = final_image.image().extent();
        if self.depth_buffer.image().extent() != img_dims {
            self.depth_buffer = ImageView::new_default(
                Image::new(
                    self.allocator.clone(),
                    ImageCreateInfo {
                        extent: [img_dims[0], img_dims[1], 1],
                        format: Format::D16_UNORM,
                        usage: ImageUsage::SAMPLED | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                        ..Default::default()
                    },
                    Default::default(),
                )
                .unwrap(),
            )
            .unwrap();
            self.fb_cache.clear();
        }

        // Zero-allocation u64 file_id check; upload once to DEVICE_LOCAL VRAM and release CPU trees
        let file_id = midi_file.file_id();
        if self.current_file_id != Some(file_id) {
            self.current_file_id = Some(file_id);
            if let Some(cpu_blocks) = midi_file.take_cpu_blocks() {
                self.upload_trees_to_device_local(cpu_blocks);
            }
        }

        let (Some(tree_descriptor), Some(meta_tree_descriptor)) =
            (self.tree_descriptor.clone(), self.meta_tree_descriptor.clone())
        else {
            return (
                RenderResultData {
                    notes_rendered: 0,
                    polyphony: None,
                    key_colors: vec![None; BUFFER_ARRAY_LEN as usize],
                },
                None,
            );
        };

        let midi_time = midi_file.current_time().as_seconds_f64();
        let screen_start = (midi_time * midi_file.ticks_per_second() as f64) as i32;
        let screen_end = ((midi_time + view_range) * midi_file.ticks_per_second() as f64) as i32;

        let push_constants = gs::PushConstants {
            start_time: screen_start,
            end_time: screen_end,
            screen_width: img_dims[0] as i32,
            screen_height: img_dims[1] as i32,
            skin_id: skin.id(),
        };

        let meta_push = meta_cs::MetaPushConstants {
            start_time: screen_start,
            end_time: screen_end,
            key_count: self.buffers.len() as u32,
            _pad: 0,
        };

        let border_width = crate::utils::calculate_border_width(
            final_image.image().extent()[0] as f32,
            key_view.visible_range.len() as f32,
        ) as i32;

        // A/B slot selection: write into `curr_slot`, read metadata from already-finished `prev_slot`
        let curr_slot = self.slot_index;
        let prev_slot = 1 - curr_slot;
        self.slot_index = prev_slot;

        let mut written_instances = 0usize;
        {
            let mut buffer_instances = self.frame_slots[curr_slot].vertex_buffer.write().unwrap();
            // Black keys first, as they stencil out in the depth buffer
            for (i, buffer) in self.buffers.iter().enumerate() {
                let key = key_view.note(i);
                if key.black {
                    buffer_instances[written_instances] = CakeNoteColumn {
                        buffer_index: i as i32,
                        border_width,
                        start: buffer.start,
                        end: buffer.end,
                        left: key.left,
                        right: key.right,
                    };
                    written_instances += 1;
                }
            }
            // White keys second
            for (i, buffer) in self.buffers.iter().enumerate() {
                let key = key_view.note(i);
                if !key.black {
                    buffer_instances[written_instances] = CakeNoteColumn {
                        buffer_index: i as i32,
                        border_width,
                        start: buffer.start,
                        end: buffer.end,
                        left: key.left,
                        right: key.right,
                    };
                    written_instances += 1;
                }
            }
        }

        let framebuffer = self.get_or_create_framebuffer(&final_image);

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // 1. Dispatch 4 KB GPU Compute metadata pass (256 workgroups x 1 thread = ~10 us)
        let meta_layout = self.meta_pipeline.layout();
        unsafe {
            command_buffer_builder
                .bind_pipeline_compute(self.meta_pipeline.clone())
                .unwrap()
                .push_constants(meta_layout.clone(), 0, meta_push)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    meta_layout.clone(),
                    0,
                    (
                        meta_tree_descriptor,
                        self.frame_slots[curr_slot].meta_descriptor.clone(),
                    ),
                )
                .unwrap()
                .dispatch([self.buffers.len() as u32, 1, 1])
                .unwrap();
        }

        // 2. Render falling notes graphics pass reading the same DEVICE_LOCAL VRAM SSBOs
        let clears = vec![Some([0.0, 0.0, 0.0, 0.0].into()), Some(1.0f32.into())];
        let subpassbegininfo = SubpassBeginInfo {
            contents: SubpassContents::Inline,
            ..Default::default()
        };

        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: clears,
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                subpassbegininfo,
            )
            .unwrap();

        let gfx_layout = self.pipeline_clear.layout();
        unsafe {
            command_buffer_builder
                .bind_pipeline_graphics(self.pipeline_clear.clone())
                .unwrap()
                .set_viewport(
                    0,
                    [Viewport {
                        offset: [0.0, 0.0],
                        extent: [img_dims[0] as f32, img_dims[1] as f32],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                )
                .unwrap()
                .push_constants(gfx_layout.clone(), 0, push_constants)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    gfx_layout.clone(),
                    0,
                    tree_descriptor,
                )
                .unwrap()
                .bind_vertex_buffers(0, self.frame_slots[curr_slot].vertex_buffer.clone())
                .unwrap()
                .draw(written_instances as u32, 1, 0, 0)
                .unwrap();
        }

        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();

        let render_future = sync::now(self.gfx_queue.device().clone())
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap()
            .boxed();

        self.frame_slots[curr_slot].has_valid_meta = true;

        // If the alternate A/B slot already holds completed metadata from the previous frame,
        // read it immediately with ZERO CPU stall and chain `render_future` directly into egui!
        // Only on frame 0 after MIDI load do we flush+wait once so initial frame metadata is valid.
        let (rendered_notes, colors, out_future) = if self.frame_slots[prev_slot].has_valid_meta {
            let (notes, cols) = self.collect_meta_from_slot(prev_slot, midi_file);
            (notes, cols, Some(render_future))
        } else {
            render_future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();
            let (notes, cols) = self.collect_meta_from_slot(curr_slot, midi_file);
            (notes, cols, None)
        };

        (
            RenderResultData {
                notes_rendered: rendered_notes,
                polyphony: None,
                key_colors: colors,
            },
            out_future,
        )
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/cake/cake.vert"
    }
}

mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "shaders/cake/cake.geom"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/cake/cake.frag"
    }
}

mod meta_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/cake/cake_meta.comp"
    }
}
