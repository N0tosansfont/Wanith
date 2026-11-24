//! GPU-side Compute Shader RGBA-to-NV12 (ITU-R BT.709 Limited Range) converter.
//!
//! Runs 100% inside GPU VRAM with ZERO CPU pixel readback:
//! 1. Reads rendered `R8G8B8A8_UNORM` offscreen frame via storage/sampled image.
//! 2. Computes 4:2:0 Y (`R8_UNORM`, full res) and UV (`R8G8_UNORM`, half res) planes in parallel
//!    (each thread processes a 2x2 pixel block).
//! 3. Copies Y -> `PLANE_0` and UV -> `PLANE_1` directly into the `VIDEO_ENCODE_SRC_KHR`
//!    NV12 `VkImage` (`VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`) inside VRAM.

use std::sync::Arc;

use ash::vk;
use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        DescriptorSet, WriteDescriptorSet,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
    VulkanObject,
};

use super::vk_context::OfflineVulkanContext;

pub struct Nv12ComputeConverter {
    ctx: Arc<OfflineVulkanContext>,
    pipeline: Arc<ComputePipeline>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    sd_allocator: Arc<StandardDescriptorSetAllocator>,
    y_image: Arc<Image>,
    uv_image: Arc<Image>,
    y_view: Arc<ImageView>,
    uv_view: Arc<ImageView>,
    width: u32,
    height: u32,
    raw_cmd_pool: vk::CommandPool,
    raw_cmd_buf: vk::CommandBuffer,
    raw_fence: vk::Fence,
}

impl Nv12ComputeConverter {
    pub fn new(ctx: Arc<OfflineVulkanContext>, width: u32, height: u32) -> Result<Self, String> {
        let coded_w = (width + 1) & !1;
        let coded_h = (height + 1) & !1;

        let cs = cs::load(ctx.device.clone())
            .map_err(|e| format!("Failed loading NV12 compute shader: {e}"))?
            .entry_point("main")
            .ok_or_else(|| "Missing main entry point in NV12 compute shader".to_string())?;

        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .map_err(|e| format!("NV12 layout info error: {e}"))?,
        )
        .map_err(|e| format!("NV12 pipeline layout error: {e}"))?;

        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| format!("NV12 compute pipeline error: {e}"))?;

        let y_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::R8_UNORM,
                extent: [coded_w, coded_h, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            Default::default(),
        )
        .map_err(|e| format!("Failed allocating VRAM Y plane image: {e}"))?;

        let uv_image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::R8G8_UNORM,
                extent: [coded_w / 2, coded_h / 2, 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            Default::default(),
        )
        .map_err(|e| format!("Failed allocating VRAM UV plane image: {e}"))?;

        let y_view = ImageView::new_default(y_image.clone())
            .map_err(|e| format!("Failed creating Y image view: {e}"))?;
        let uv_view = ImageView::new_default(uv_image.clone())
            .map_err(|e| format!("Failed creating UV image view: {e}"))?;

        let cb_allocator = Arc::new(StandardCommandBufferAllocator::new(
            ctx.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let sd_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            ctx.device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Raw command pool/buffer on the Graphics/Compute queue for VRAM Y/UV -> NV12 plane copies
        let raw_dev = ctx.raw_device();
        let fns_v1_0 = &ctx.device.fns().v1_0;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.gfx_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let mut raw_cmd_pool = vk::CommandPool::null();
        let res = unsafe {
            (fns_v1_0.create_command_pool)(raw_dev, &pool_info, std::ptr::null(), &mut raw_cmd_pool)
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateCommandPool (NV12 copy) failed: {res:?}"));
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(raw_cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let mut raw_cmd_buf = vk::CommandBuffer::null();
        let res = unsafe { (fns_v1_0.allocate_command_buffers)(raw_dev, &alloc_info, &mut raw_cmd_buf) };
        if res != vk::Result::SUCCESS {
            unsafe { (fns_v1_0.destroy_command_pool)(raw_dev, raw_cmd_pool, std::ptr::null()) };
            return Err(format!("vkAllocateCommandBuffers (NV12 copy) failed: {res:?}"));
        }

        let fence_info = vk::FenceCreateInfo::default();
        let mut raw_fence = vk::Fence::null();
        let res = unsafe {
            (fns_v1_0.create_fence)(raw_dev, &fence_info, std::ptr::null(), &mut raw_fence)
        };
        if res != vk::Result::SUCCESS {
            unsafe { (fns_v1_0.destroy_command_pool)(raw_dev, raw_cmd_pool, std::ptr::null()) };
            return Err(format!("vkCreateFence (NV12 copy) failed: {res:?}"));
        }

        Ok(Self {
            ctx,
            pipeline,
            cb_allocator,
            sd_allocator,
            y_image,
            uv_image,
            y_view,
            uv_view,
            width: coded_w,
            height: coded_h,
            raw_cmd_pool,
            raw_cmd_buf,
            raw_fence,
        })
    }

    /// Convert `src_rgba` (`R8G8B8A8_UNORM`) directly into `dst_nv12` (`VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`)
    /// inside VRAM and transition `dst_nv12` into `VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR`.
    pub fn convert_to_nv12(
        &self,
        src_rgba: Arc<ImageView>,
        dst_nv12: vk::Image,
        dst_nv12_prev_layout: vk::ImageLayout,
    ) -> Result<(), String> {
        let layout = self.pipeline.layout();
        let set_layout = layout
            .set_layouts()
            .first()
            .ok_or_else(|| "Missing NV12 descriptor set layout".to_string())?;

        let descriptor_set = DescriptorSet::new(
            self.sd_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, src_rgba),
                WriteDescriptorSet::image_view(1, self.y_view.clone()),
                WriteDescriptorSet::image_view(2, self.uv_view.clone()),
            ],
            [],
        )
        .map_err(|e| format!("Failed creating NV12 descriptor set: {e}"))?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.cb_allocator.clone(),
            self.ctx.gfx_queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("Failed creating NV12 compute cmd buffer: {e}"))?;

        let groups_x = (self.width / 2).div_ceil(16);
        let groups_y = (self.height / 2).div_ceil(16);

        let push = cs::PushConstants {
            width: self.width,
            height: self.height,
        };

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .map_err(|e| format!("NV12 bind pipeline error: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| format!("NV12 bind descriptor error: {e}"))?
            .push_constants(layout.clone(), 0, push)
            .map_err(|e| format!("NV12 push constants error: {e}"))?;

        unsafe {
            builder
                .dispatch([groups_x, groups_y, 1])
                .map_err(|e| format!("NV12 dispatch error: {e}"))?;
        }

        let cmd_buf = builder
            .build()
            .map_err(|e| format!("NV12 compute build error: {e}"))?;

        sync::now(self.ctx.device.clone())
            .then_execute(self.ctx.gfx_queue.clone(), cmd_buf)
            .map_err(|e| format!("NV12 compute submit error: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("NV12 compute flush error: {e}"))?
            .wait(None)
            .map_err(|e| format!("NV12 compute wait error: {e}"))?;

        // Step 2: Copy Y & UV planes in VRAM into the 2-plane NV12 `VIDEO_ENCODE_SRC_KHR` image
        self.copy_planes_to_nv12(dst_nv12, dst_nv12_prev_layout)
    }

    fn copy_planes_to_nv12(
        &self,
        dst_nv12: vk::Image,
        dst_prev_layout: vk::ImageLayout,
    ) -> Result<(), String> {
        let raw_dev = self.ctx.raw_device();
        let fns_v1_0 = &self.ctx.device.fns().v1_0;
        let fns_v1_3 = &self.ctx.device.fns().v1_3;

        unsafe {
            let res = (fns_v1_0.reset_fences)(raw_dev, 1, &self.raw_fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetFences (NV12) failed: {res:?}"));
            }
            let res = (fns_v1_0.reset_command_buffer)(
                self.raw_cmd_buf,
                vk::CommandBufferResetFlags::empty(),
            );
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetCommandBuffer (NV12) failed: {res:?}"));
            }

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let res = (fns_v1_0.begin_command_buffer)(self.raw_cmd_buf, &begin_info);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkBeginCommandBuffer (NV12) failed: {res:?}"));
            }

            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            // Transition Y & UV from GENERAL -> TRANSFER_SRC, and NV12 from prev -> TRANSFER_DST
            let pre_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.y_image.handle())
                    .subresource_range(color_range),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.uv_image.handle())
                    .subresource_range(color_range),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(dst_prev_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(dst_nv12)
                    .subresource_range(color_range),
            ];
            let dep_pre = vk::DependencyInfo::default().image_memory_barriers(&pre_barriers);
            (fns_v1_3.cmd_pipeline_barrier2)(self.raw_cmd_buf, &dep_pre);

            // Copy Plane 0 (Y)
            let copy_y = vk::ImageCopy::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .layer_count(1),
                )
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });
            (fns_v1_0.cmd_copy_image)(
                self.raw_cmd_buf,
                self.y_image.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                1,
                &copy_y,
            );

            // Copy Plane 1 (UV)
            let copy_uv = vk::ImageCopy::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .layer_count(1),
                )
                .extent(vk::Extent3D {
                    width: self.width / 2,
                    height: self.height / 2,
                    depth: 1,
                });
            (fns_v1_0.cmd_copy_image)(
                self.raw_cmd_buf,
                self.uv_image.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                1,
                &copy_uv,
            );

            // Restore Y & UV -> GENERAL (for next vulkano compute dispatch),
            // and transition NV12 -> VIDEO_ENCODE_SRC_KHR
            let post_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.y_image.handle())
                    .subresource_range(color_range),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.uv_image.handle())
                    .subresource_range(color_range),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                    .image(dst_nv12)
                    .subresource_range(color_range),
            ];
            let dep_post = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
            (fns_v1_3.cmd_pipeline_barrier2)(self.raw_cmd_buf, &dep_post);

            let res = (fns_v1_0.end_command_buffer)(self.raw_cmd_buf);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkEndCommandBuffer (NV12) failed: {res:?}"));
            }

            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.raw_cmd_buf));
            let res = (fns_v1_0.queue_submit)(self.ctx.raw_gfx_queue(), 1, &submit, self.raw_fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkQueueSubmit (NV12) failed: {res:?}"));
            }

            let res = (fns_v1_0.wait_for_fences)(raw_dev, 1, &self.raw_fence, vk::TRUE, u64::MAX);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkWaitForFences (NV12) failed: {res:?}"));
            }
        }
        Ok(())
    }
}

impl Drop for Nv12ComputeConverter {
    fn drop(&mut self) {
        let raw_dev = self.ctx.raw_device();
        let fns_v1_0 = &self.ctx.device.fns().v1_0;
        unsafe {
            let _ = (fns_v1_0.device_wait_idle)(raw_dev);
            if self.raw_fence != vk::Fence::null() {
                (fns_v1_0.destroy_fence)(raw_dev, self.raw_fence, std::ptr::null());
            }
            if self.raw_cmd_pool != vk::CommandPool::null() {
                (fns_v1_0.destroy_command_pool)(raw_dev, self.raw_cmd_pool, std::ptr::null());
            }
        }
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) readonly uniform image2D src_rgba;
layout(set = 0, binding = 1, r8) writeonly uniform image2D dst_y;
layout(set = 0, binding = 2, rg8) writeonly uniform image2D dst_uv;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} pc;

float rgb_to_y709(vec3 rgb) {
    // ITU-R BT.709 Limited Range [16..235]
    return (16.0 / 255.0) + (219.0 / 255.0) * dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

vec2 rgb_to_uv709(vec3 rgb) {
    // ITU-R BT.709 Limited Range [16..240]
    float cb = (128.0 / 255.0) + (224.0 / 255.0) * dot(rgb, vec3(-0.114572, -0.385428, 0.5));
    float cr = (128.0 / 255.0) + (224.0 / 255.0) * dot(rgb, vec3(0.5, -0.454153, -0.045847));
    return clamp(vec2(cb, cr), vec2(16.0 / 255.0), vec2(240.0 / 255.0));
}

void main() {
    uint bx = gl_GlobalInvocationID.x;
    uint by = gl_GlobalInvocationID.y;
    uint x0 = bx * 2;
    uint y0 = by * 2;
    if (x0 >= pc.width || y0 >= pc.height) {
        return;
    }

    ivec2 max_p = ivec2(int(pc.width) - 1, int(pc.height) - 1);
    ivec2 p00 = min(ivec2(int(x0),     int(y0)),     max_p);
    ivec2 p10 = min(ivec2(int(x0) + 1, int(y0)),     max_p);
    ivec2 p01 = min(ivec2(int(x0),     int(y0) + 1), max_p);
    ivec2 p11 = min(ivec2(int(x0) + 1, int(y0) + 1), max_p);

    vec3 c00 = imageLoad(src_rgba, p00).rgb;
    vec3 c10 = imageLoad(src_rgba, p10).rgb;
    vec3 c01 = imageLoad(src_rgba, p01).rgb;
    vec3 c11 = imageLoad(src_rgba, p11).rgb;

    imageStore(dst_y, p00, vec4(clamp(rgb_to_y709(c00), 16.0/255.0, 235.0/255.0), 0.0, 0.0, 1.0));
    imageStore(dst_y, p10, vec4(clamp(rgb_to_y709(c10), 16.0/255.0, 235.0/255.0), 0.0, 0.0, 1.0));
    imageStore(dst_y, p01, vec4(clamp(rgb_to_y709(c01), 16.0/255.0, 235.0/255.0), 0.0, 0.0, 1.0));
    imageStore(dst_y, p11, vec4(clamp(rgb_to_y709(c11), 16.0/255.0, 235.0/255.0), 0.0, 0.0, 1.0));

    vec3 avg = 0.25 * (c00 + c10 + c01 + c11);
    vec2 uv = rgb_to_uv709(avg);
    imageStore(dst_uv, ivec2(int(bx), int(by)), vec4(uv.x, uv.y, 0.0, 1.0));
}
"
    }
}
