use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
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
    sync::{self, future::FenceSignalFuture, GpuFuture},
};

use crate::gui::{window::keyboard_layout::KeyboardView, GuiRenderer};

/// A/B double-buffered vertex slot capacity: 4,000,000 notes x 16 bytes = 64 MB per slot
/// (128 MB total for A + B, saving 672 MB over the legacy 800 MB fixed allocation).
/// Frames with > 4M visible notes automatically span multiple A/B passes.
const NOTE_BUFFER_SIZE: u64 = 4_000_000;

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod, Vertex)]
pub struct NoteVertex {
    #[format(R32G32_SFLOAT)]
    pub start_length: [f32; 2],
    #[format(R32_UINT)]
    pub key_color: u32,
    #[format(R32_UINT)]
    pub border_width: u32,
}

impl NoteVertex {
    #[inline]
    pub fn new(start: f32, len: f32, key: u8, color: u32, border_width: u32) -> Self {
        Self {
            start_length: [start, len],
            key_color: key as u32 | (color << 8),
            border_width,
        }
    }
}

struct BufferSet {
    vertex_buffers: [Subbuffer<[NoteVertex]>; 2],
    index: usize,
}

fn create_ab_vertex_buffers(
    allocator: &Arc<StandardMemoryAllocator>,
) -> [Subbuffer<[NoteVertex]>; 2] {
    let (a, b) = Buffer::new_slice(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        NOTE_BUFFER_SIZE * 2,
    )
    .expect("failed to create A/B note vertex buffer")
    .split_at(NOTE_BUFFER_SIZE);
    [a, b]
}

impl BufferSet {
    fn new(allocator: &Arc<StandardMemoryAllocator>) -> Self {
        Self {
            vertex_buffers: create_ab_vertex_buffers(allocator),
            index: 0,
        }
    }

    #[inline]
    fn next(&mut self) -> &Subbuffer<[NoteVertex]> {
        self.index = 1 - self.index;
        &self.vertex_buffers[self.index]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotePassStatus {
    Finished { remaining: u32 },
    HasMoreNotes,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, Zeroable, Pod)]
pub struct KeyPosition {
    left: f32,
    right: f32,
    _padding: [u8; 8],
}

struct CachedFramebuffers {
    image_ptr: usize,
    extent: [u32; 3],
    fb_clear: Arc<Framebuffer>,
    fb_draw_over: Arc<Framebuffer>,
}

pub struct NoteRenderPass {
    gfx_queue: Arc<Queue>,
    buffer_set: BufferSet,
    pipeline_clear: Arc<GraphicsPipeline>,
    pipeline_draw_over: Arc<GraphicsPipeline>,
    render_pass_clear: Arc<RenderPass>,
    render_pass_draw_over: Arc<RenderPass>,
    key_locations: Subbuffer<[[KeyPosition; 256]]>,
    /// Pre-allocated once at init (zero per-frame DescriptorSet allocations).
    key_locations_descriptor: Arc<DescriptorSet>,
    depth_buffer: Arc<ImageView>,
    allocator: Arc<StandardMemoryAllocator>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    /// Cached framebuffers per swapchain scene image (zero per-frame Framebuffer allocations).
    fb_cache: Vec<CachedFramebuffers>,
}

impl NoteRenderPass {
    pub fn new(renderer: &GuiRenderer) -> NoteRenderPass {
        let allocator = Arc::new(StandardMemoryAllocator::new_default(
            renderer.device.clone(),
        ));
        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            renderer.device.clone(),
            Default::default(),
        ));
        let sd_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            renderer.device.clone(),
            Default::default(),
        ));

        let gfx_queue = renderer.queue.clone();
        let device: Arc<Device> = gfx_queue.device().clone();

        let render_pass_clear = vulkano::ordered_passes_renderpass!(device.clone(),
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

        let render_pass_draw_over = vulkano::ordered_passes_renderpass!(device.clone(),
            attachments: {
                final_color: {
                    format: renderer.format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
                depth: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: DontCare,
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

        let key_locations = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [[Default::default(); 256]],
        )
        .unwrap();

        let vs = vs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let gs = gs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let vertex_input_state = NoteVertex::per_vertex().definition(&vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
            PipelineShaderStageCreateInfo::new(gs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass_clear.clone(), 0).unwrap();

        let mut create_info = GraphicsPipelineCreateInfo {
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
        };

        let pipeline_clear =
            GraphicsPipeline::new(device.clone(), None, create_info.clone()).unwrap();

        create_info.subpass = Some(PipelineSubpassType::BeginRenderPass(
            Subpass::from(render_pass_draw_over.clone(), 0).unwrap(),
        ));
        let pipeline_draw_over = GraphicsPipeline::new(device, None, create_info).unwrap();

        let desc_layout = pipeline_clear.layout().set_layouts()[0].clone();
        let key_locations_descriptor = DescriptorSet::new(
            sd_allocator,
            desc_layout,
            [WriteDescriptorSet::buffer(0, key_locations.clone())],
            [],
        )
        .unwrap();

        let buffer_set = BufferSet::new(&allocator);

        NoteRenderPass {
            gfx_queue,
            buffer_set,
            pipeline_clear,
            pipeline_draw_over,
            render_pass_clear,
            render_pass_draw_over,
            depth_buffer,
            key_locations,
            key_locations_descriptor,
            allocator,
            cb_allocator,
            fb_cache: Vec::with_capacity(4),
        }
    }

    fn get_or_create_framebuffers(
        &mut self,
        final_image: &Arc<ImageView>,
    ) -> (Arc<Framebuffer>, Arc<Framebuffer>) {
        let img_ptr = Arc::as_ptr(final_image) as usize;
        let extent = final_image.image().extent();

        if let Some(entry) = self
            .fb_cache
            .iter()
            .find(|e| e.image_ptr == img_ptr && e.extent == extent)
        {
            return (entry.fb_clear.clone(), entry.fb_draw_over.clone());
        }

        if self.fb_cache.len() >= 8 {
            self.fb_cache.clear();
        }

        let fb_clear = Framebuffer::new(
            self.render_pass_clear.clone(),
            FramebufferCreateInfo {
                attachments: vec![final_image.clone(), self.depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        let fb_draw_over = Framebuffer::new(
            self.render_pass_draw_over.clone(),
            FramebufferCreateInfo {
                attachments: vec![final_image.clone(), self.depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        self.fb_cache.push(CachedFramebuffers {
            image_ptr: img_ptr,
            extent,
            fb_clear: fb_clear.clone(),
            fb_draw_over: fb_draw_over.clone(),
        });

        (fb_clear, fb_draw_over)
    }

    pub fn draw(
        &mut self,
        final_image: Arc<ImageView>,
        key_view: &KeyboardView,
        view_range: f32,
        skin_id: u32,
        mut fill_buffer: impl FnMut(&Subbuffer<[NoteVertex]>) -> NotePassStatus,
    ) -> Option<Box<dyn GpuFuture>> {
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

        {
            let mut keys = self.key_locations.write().unwrap();
            for (write, key) in keys[0].iter_mut().zip(key_view.iter_all_notes()) {
                *write = KeyPosition {
                    left: key.left,
                    right: key.right,
                    _padding: [0; 8],
                };
            }
        }

        let (fb_clear, fb_draw_over) = self.get_or_create_framebuffers(&final_image);

        let mut prev_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>> = None;
        let mut status = NotePassStatus::HasMoreNotes;
        let mut first_pass = true;
        let mut final_future: Option<Box<dyn GpuFuture>> = None;

        while status == NotePassStatus::HasMoreNotes {
            let buffer = self.buffer_set.next().clone();

            status = fill_buffer(&buffer);

            let items_to_render = match status {
                NotePassStatus::Finished { remaining } => {
                    assert!(remaining <= buffer.len() as u32);
                    remaining
                }
                NotePassStatus::HasMoreNotes => buffer.len() as u32,
            };

            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                self.cb_allocator.clone(),
                self.gfx_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let (clears, pipeline, framebuffer) = if first_pass {
                first_pass = false;
                (
                    vec![Some([0.0, 0.0, 0.0, 0.0].into()), Some(1.0f32.into())],
                    &self.pipeline_clear,
                    fb_clear.clone(),
                )
            } else {
                (
                    vec![None, None],
                    &self.pipeline_draw_over,
                    fb_draw_over.clone(),
                )
            };

            let pipeline_layout = pipeline.layout();

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

            let push_constants = gs::PushConstants {
                height_time: view_range,
                win_width: img_dims[0] as f32,
                win_height: img_dims[1] as f32,
                skin_id,
            };

            unsafe {
                command_buffer_builder
                    .bind_pipeline_graphics(pipeline.clone())
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
                    .push_constants(pipeline_layout.clone(), 0, push_constants)
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline_layout.clone(),
                        0,
                        self.key_locations_descriptor.clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, buffer)
                    .unwrap()
                    .draw(items_to_render, 1, 0, 0)
                    .unwrap()
            };

            command_buffer_builder
                .end_render_pass(Default::default())
                .unwrap();
            let command_buffer = command_buffer_builder.build().unwrap();

            // Wait on the alternate A/B pass only when overlapping multiple > 4M-note passes
            if let Some(prev) = prev_future.take() {
                let _ = prev.wait(None);
            }

            let after_main_cb = sync::now(self.gfx_queue.device().clone())
                .then_execute(self.gfx_queue.clone(), command_buffer)
                .unwrap()
                .boxed();

            if status == NotePassStatus::HasMoreNotes {
                let fence_future = after_main_cb
                    .then_signal_fence_and_flush()
                    .expect("Failed to signal fence and flush");
                prev_future = Some(fence_future);
            } else {
                // Final (or only) pass: return GPU future directly to egui composition with ZERO CPU stall!
                final_future = Some(after_main_cb);
            }
        }

        final_future
    }
}

mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "shaders/notes/notes.geom",
    }
}

mod vs {
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

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/notes/notes.frag"
    }
}
