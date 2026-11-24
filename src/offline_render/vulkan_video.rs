//! Native Vulkan Video hardware encoder (`VK_KHR_video_encode_queue` + H.264 / H.265)
//! implemented directly via `ash 0.38` raw FFI on an isolated offline Vulkan context.
//!
//! # Zero-Copy VRAM Pipeline
//! - Allocates a VRAM `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM` (NV12) input image shared
//!   concurrently with the Graphics/Compute queue so `Nv12ComputeConverter` writes
//!   directly into VRAM without any CPU pixel readback.
//! - Encodes on the dedicated `VIDEO_ENCODE` hardware queue (NVENC / VCE / QuickSync)
//!   using a 2-slot ping-pong IPPP DPB and reads back ONLY compressed Annex-B NAL bytes.

use std::ffi::c_void;
use std::sync::Arc;

use ash::vk;
use ash::vk::native as vkn;

use crate::settings::{VideoCodec, VideoRateControl};

use super::vk_context::OfflineVulkanContext;

pub struct VideoEncodeConfig {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub rate_control: VideoRateControl,
    pub bitrate_kbps: u32,
    pub qp: u32,
}

struct VideoImageAlloc {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

pub struct VulkanVideoEncoder {
    ctx: Arc<OfflineVulkanContext>,
    codec: VideoCodec,
    display_width: u32,
    display_height: u32,
    coded_width: u32,
    coded_height: u32,
    fps: u32,
    gop_len: u32,
    use_cqp: bool,
    constant_qp: i32,
    bitrate_bps: u64,

    // Video session & parameters
    session: vk::VideoSessionKHR,
    session_memories: Vec<vk::DeviceMemory>,
    session_params: vk::VideoSessionParametersKHR,
    parameter_sets_annexb: Vec<u8>,

    // VRAM images: 1 input NV12 + 2 separate DPB reference slots
    src_image: VideoImageAlloc,
    src_layout: vk::ImageLayout,
    dpb_images: [VideoImageAlloc; 2],
    dpb_initialized: [bool; 2],

    // Host-visible compressed bitstream output buffer (readback ONLY compressed NALs)
    dst_buffer: vk::Buffer,
    dst_memory: vk::DeviceMemory,
    dst_size: u64,
    dst_mapped: *mut u8,

    // Command pool / buffer / fence / query pool on VIDEO_ENCODE queue
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
    query_pool: vk::QueryPool,

    // Codec frame counters
    idr_pic_id: u16,
    frame_num: u32,
    poc: i32,
    rate_control_initialized: bool,
    bitstream_scratch: Vec<u8>,
}

// SAFETY: `VulkanVideoEncoder` owns all raw Vulkan handles exclusively and is only used
// sequentially by the single offline video render worker thread.
unsafe impl Send for VulkanVideoEncoder {}

impl VulkanVideoEncoder {
    pub fn new(ctx: Arc<OfflineVulkanContext>, cfg: &VideoEncodeConfig) -> Result<Self, String> {
        let raw_pd = ctx.raw_physical_device();
        let raw_dev = ctx.raw_device();
        let inst_vq = &ctx.instance.fns().khr_video_queue;
        let dev_v1_0 = &ctx.device.fns().v1_0;
        let dev_vq = &ctx.device.fns().khr_video_queue;
        let dev_ve = &ctx.device.fns().khr_video_encode_queue;

        let display_w = (cfg.width + 1) & !1;
        let display_h = (cfg.height + 1) & !1;
        let fps = cfg.fps.max(1);
        let gop_len = fps.clamp(15, 240);

        // 1. Build codec profile chain & query GPU capabilities dynamically
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default()
            .std_profile_idc(vkn::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH);
        let mut h265_profile = vk::VideoEncodeH265ProfileInfoKHR::default()
            .std_profile_idc(vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN);
        let mut usage_info = vk::VideoEncodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoEncodeUsageFlagsKHR::RECORDING)
            .video_content_hints(vk::VideoEncodeContentFlagsKHR::DEFAULT)
            .tuning_mode(vk::VideoEncodeTuningModeKHR::HIGH_QUALITY);

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(match cfg.codec {
                VideoCodec::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
                VideoCodec::H265 => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            })
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);

        profile_info = match cfg.codec {
            VideoCodec::H264 => profile_info.push_next(&mut h264_profile),
            VideoCodec::H265 => profile_info.push_next(&mut h265_profile),
        }
        .push_next(&mut usage_info);

        let mut h264_caps = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();

        let (
            picture_access_granularity,
            min_coded_extent,
            max_coded_extent,
            std_header_version,
            min_bitstream_buffer_size_alignment,
        ) = {
            let mut video_caps = vk::VideoCapabilitiesKHR::default();
            video_caps = match cfg.codec {
                VideoCodec::H264 => video_caps.push_next(&mut h264_caps),
                VideoCodec::H265 => video_caps.push_next(&mut h265_caps),
            }
            .push_next(&mut encode_caps);

            let res = unsafe {
                (inst_vq.get_physical_device_video_capabilities_khr)(
                    raw_pd,
                    &profile_info,
                    &mut video_caps,
                )
            };
            if res != vk::Result::SUCCESS {
                return Err(format!(
                    "vkGetPhysicalDeviceVideoCapabilitiesKHR ({:?}) failed: {res:?}",
                    cfg.codec
                ));
            }
            (
                video_caps.picture_access_granularity,
                video_caps.min_coded_extent,
                video_caps.max_coded_extent,
                video_caps.std_header_version,
                video_caps.min_bitstream_buffer_size_alignment,
            )
        };

        // Compute coded dimensions aligned to GPU picture_access_granularity & codec block size
        let gran_w = picture_access_granularity.width.max(16);
        let gran_h = picture_access_granularity.height.max(16);
        let align_w = match cfg.codec {
            VideoCodec::H264 => gran_w.max(16),
            VideoCodec::H265 => gran_w.max(16),
        };
        let align_h = match cfg.codec {
            VideoCodec::H264 => gran_h.max(16),
            VideoCodec::H265 => gran_h.max(16),
        };
        let coded_w = align_up(display_w, align_w);
        let coded_h = align_up(display_h, align_h);

        if coded_w > max_coded_extent.width
            || coded_h > max_coded_extent.height
            || coded_w < min_coded_extent.width
            || coded_h < min_coded_extent.height
        {
            return Err(format!(
                "Requested resolution {}x{} (coded {}x{}) is outside GPU Vulkan Video limits ({}x{} .. {}x{}).",
                display_w,
                display_h,
                coded_w,
                coded_h,
                min_coded_extent.width,
                min_coded_extent.height,
                max_coded_extent.width,
                max_coded_extent.height,
            ));
        }

        // 2. Query supported SRC and DPB NV12 formats
        let src_format = query_preferred_nv12_format(
            inst_vq,
            raw_pd,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
        )?;
        let dpb_format = query_preferred_nv12_format(
            inst_vq,
            raw_pd,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        )?;

        // Clamp QP and pick rate control mode supported by GPU
        let (min_qp, max_qp) = match cfg.codec {
            VideoCodec::H264 => (h264_caps.min_qp.max(0), h264_caps.max_qp.min(51)),
            VideoCodec::H265 => (h265_caps.min_qp.max(0), h265_caps.max_qp.min(51)),
        };
        let constant_qp = (cfg.qp as i32).clamp(min_qp, max_qp.max(min_qp));
        let supports_vbr = encode_caps
            .rate_control_modes
            .contains(vk::VideoEncodeRateControlModeFlagsKHR::VBR);
        let supports_cbr = encode_caps
            .rate_control_modes
            .contains(vk::VideoEncodeRateControlModeFlagsKHR::CBR);
        let use_cqp =
            cfg.rate_control == VideoRateControl::ConstantQp || (!supports_vbr && !supports_cbr);
        let bitrate_bps = (cfg.bitrate_kbps as u64).max(500) * 1_000;

        // 3. Create VkVideoSessionKHR & bind all required device memory
        let session_create = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(ctx.encode_queue_family_index)
            .video_profile(&profile_info)
            .picture_format(src_format)
            .max_coded_extent(vk::Extent2D {
                width: coded_w,
                height: coded_h,
            })
            .reference_picture_format(dpb_format)
            .max_dpb_slots(2)
            .max_active_reference_pictures(1)
            .std_header_version(&std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        let res = unsafe {
            (dev_vq.create_video_session_khr)(
                raw_dev,
                &session_create,
                std::ptr::null(),
                &mut session,
            )
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateVideoSessionKHR failed: {res:?}"));
        }

        let session_memories = bind_video_session_memory(&ctx, session)?;

        // 4. Create VkVideoSessionParametersKHR (SPS/PPS or VPS/SPS/PPS) and fetch encoded headers
        let (session_params, parameter_sets_annexb) = match cfg.codec {
            VideoCodec::H264 => create_h264_session_params(
                dev_vq,
                dev_ve,
                raw_dev,
                session,
                display_w,
                display_h,
                coded_w,
                coded_h,
                h264_caps.max_level_idc,
            )?,
            VideoCodec::H265 => create_h265_session_params(
                dev_vq,
                dev_ve,
                raw_dev,
                session,
                display_w,
                display_h,
                coded_w,
                coded_h,
                fps,
                &h265_caps,
            )?,
        };

        // 5. Allocate VRAM NV12 input image + 2 DPB reference images
        let profile_list_arr = [profile_info];
        let src_image = create_video_nv12_image(
            &ctx,
            src_format,
            coded_w,
            coded_h,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
            &profile_list_arr,
            true,
        )?;
        let dpb0 = create_video_nv12_image(
            &ctx,
            dpb_format,
            coded_w,
            coded_h,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
            &profile_list_arr,
            false,
        )?;
        let dpb1 = create_video_nv12_image(
            &ctx,
            dpb_format,
            coded_w,
            coded_h,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
            &profile_list_arr,
            false,
        )?;

        // 6. Allocate host-visible bitstream output buffer
        let size_align = min_bitstream_buffer_size_alignment.max(4096);
        let raw_buf_size = ((coded_w as u64) * (coded_h as u64) * 2).max(4 * 1024 * 1024);
        let dst_size = align_up_u64(raw_buf_size, size_align);

        let mut dst_profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(&profile_list_arr);
        let buf_info = vk::BufferCreateInfo::default()
            .size(dst_size)
            .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut dst_profile_list);
        let mut dst_buffer = vk::Buffer::null();
        let res = unsafe {
            (dev_v1_0.create_buffer)(raw_dev, &buf_info, std::ptr::null(), &mut dst_buffer)
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateBuffer (VIDEO_ENCODE_DST) failed: {res:?}"));
        }

        let mut mem_req = vk::MemoryRequirements::default();
        unsafe { (dev_v1_0.get_buffer_memory_requirements)(raw_dev, dst_buffer, &mut mem_req) };
        let mem_type = ctx.find_memory_type(
            mem_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let mut dst_memory = vk::DeviceMemory::null();
        let res = unsafe {
            (dev_v1_0.allocate_memory)(raw_dev, &alloc_info, std::ptr::null(), &mut dst_memory)
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkAllocateMemory (bitstream) failed: {res:?}"));
        }
        let res = unsafe { (dev_v1_0.bind_buffer_memory)(raw_dev, dst_buffer, dst_memory, 0) };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkBindBufferMemory (bitstream) failed: {res:?}"));
        }

        let mut mapped_ptr: *mut c_void = std::ptr::null_mut();
        let res = unsafe {
            (dev_v1_0.map_memory)(
                raw_dev,
                dst_memory,
                0,
                dst_size,
                vk::MemoryMapFlags::empty(),
                &mut mapped_ptr,
            )
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkMapMemory (bitstream) failed: {res:?}"));
        }

        // 7. Create CommandPool, CommandBuffer, Fence, and VIDEO_ENCODE_FEEDBACK QueryPool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.encode_queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let mut cmd_pool = vk::CommandPool::null();
        let res = unsafe {
            (dev_v1_0.create_command_pool)(raw_dev, &pool_info, std::ptr::null(), &mut cmd_pool)
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateCommandPool (encode) failed: {res:?}"));
        }

        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let mut cmd_buf = vk::CommandBuffer::null();
        let res = unsafe { (dev_v1_0.allocate_command_buffers)(raw_dev, &cb_alloc, &mut cmd_buf) };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkAllocateCommandBuffers (encode) failed: {res:?}"));
        }

        let fence_info = vk::FenceCreateInfo::default();
        let mut fence = vk::Fence::null();
        let res =
            unsafe { (dev_v1_0.create_fence)(raw_dev, &fence_info, std::ptr::null(), &mut fence) };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateFence (encode) failed: {res:?}"));
        }

        let mut feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
            .encode_feedback_flags(
                vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                    | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            );
        let mut qp_profile = profile_info;
        let qp_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1)
            .push_next(&mut qp_profile)
            .push_next(&mut feedback_info);
        let mut query_pool = vk::QueryPool::null();
        let res = unsafe {
            (dev_v1_0.create_query_pool)(raw_dev, &qp_info, std::ptr::null(), &mut query_pool)
        };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkCreateQueryPool (VIDEO_ENCODE_FEEDBACK) failed: {res:?}"));
        }

        let mut enc = Self {
            ctx,
            codec: cfg.codec,
            display_width: display_w,
            display_height: display_h,
            coded_width: coded_w,
            coded_height: coded_h,
            fps,
            gop_len,
            use_cqp,
            constant_qp,
            bitrate_bps,
            session,
            session_memories,
            session_params,
            parameter_sets_annexb,
            src_image,
            src_layout: vk::ImageLayout::UNDEFINED,
            dpb_images: [dpb0, dpb1],
            dpb_initialized: [false, false],
            dst_buffer,
            dst_memory,
            dst_size,
            dst_mapped: mapped_ptr as *mut u8,
            cmd_pool,
            cmd_buf,
            fence,
            query_pool,
            idr_pic_id: 0,
            frame_num: 0,
            poc: 0,
            rate_control_initialized: false,
            bitstream_scratch: Vec::with_capacity(512 * 1024),
        };

        enc.initialize_session_and_dpb(supports_vbr)?;
        Ok(enc)
    }

    #[inline]
    pub fn src_nv12_image(&self) -> vk::Image {
        self.src_image.image
    }

    #[inline]
    pub fn src_nv12_layout(&self) -> vk::ImageLayout {
        self.src_layout
    }

    #[inline]
    pub fn mark_src_ready_for_encode(&mut self) {
        self.src_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;
    }

    #[inline]
    pub fn parameter_sets_annexb(&self) -> &[u8] {
        &self.parameter_sets_annexb
    }

    #[inline]
    #[allow(dead_code)]
    pub fn display_dimensions(&self) -> (u32, u32) {
        (self.display_width, self.display_height)
    }

    /// Transition DPB slots to `VIDEO_ENCODE_DPB_KHR`, reset session, and configure Rate Control.
    fn initialize_session_and_dpb(&mut self, supports_vbr: bool) -> Result<(), String> {
        let raw_dev = self.ctx.raw_device();
        let dev_v1_0 = &self.ctx.device.fns().v1_0;
        let dev_v1_3 = &self.ctx.device.fns().v1_3;
        let dev_vq = &self.ctx.device.fns().khr_video_queue;

        unsafe {
            let res = (dev_v1_0.reset_fences)(raw_dev, 1, &self.fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetFences (init video) failed: {res:?}"));
            }
            let res = (dev_v1_0.reset_command_buffer)(
                self.cmd_buf,
                vk::CommandBufferResetFlags::empty(),
            );
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetCommandBuffer (init video) failed: {res:?}"));
            }

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let res = (dev_v1_0.begin_command_buffer)(self.cmd_buf, &begin);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkBeginCommandBuffer (init video) failed: {res:?}"));
            }

            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let dpb_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
                    .dst_access_mask(
                        vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
                            | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
                    .image(self.dpb_images[0].image)
                    .subresource_range(range),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
                    .dst_access_mask(
                        vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
                            | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
                    .image(self.dpb_images[1].image)
                    .subresource_range(range),
            ];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&dpb_barriers);
            (dev_v1_3.cmd_pipeline_barrier2)(self.cmd_buf, &dep);

            let begin_coding = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params);
            (dev_vq.cmd_begin_video_coding_khr)(self.cmd_buf, &begin_coding);

            // Reset video session + apply rate control
            let mut h264_rc = vk::VideoEncodeH264RateControlInfoKHR::default()
                .flags(vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP)
                .gop_frame_count(self.gop_len)
                .idr_period(self.gop_len)
                .consecutive_b_frame_count(0)
                .temporal_layer_count(1);
            let mut h265_rc = vk::VideoEncodeH265RateControlInfoKHR::default()
                .flags(vk::VideoEncodeH265RateControlFlagsKHR::REGULAR_GOP)
                .gop_frame_count(self.gop_len)
                .idr_period(self.gop_len)
                .consecutive_b_frame_count(0)
                .sub_layer_count(1);

            let mut h264_rc_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::default();
            let mut h265_rc_layer = vk::VideoEncodeH265RateControlLayerInfoKHR::default();
            let mut rc_layer = vk::VideoEncodeRateControlLayerInfoKHR::default()
                .average_bitrate(self.bitrate_bps)
                .max_bitrate(self.bitrate_bps.saturating_mul(3) / 2)
                .frame_rate_numerator(self.fps)
                .frame_rate_denominator(1);
            rc_layer = match self.codec {
                VideoCodec::H264 => rc_layer.push_next(&mut h264_rc_layer),
                VideoCodec::H265 => rc_layer.push_next(&mut h265_rc_layer),
            };
            let rc_layers = [rc_layer];

            let mut rc_info = if self.use_cqp {
                vk::VideoEncodeRateControlInfoKHR::default()
                    .rate_control_mode(vk::VideoEncodeRateControlModeFlagsKHR::DISABLED)
            } else {
                let mode = if supports_vbr {
                    vk::VideoEncodeRateControlModeFlagsKHR::VBR
                } else {
                    vk::VideoEncodeRateControlModeFlagsKHR::CBR
                };
                vk::VideoEncodeRateControlInfoKHR::default()
                    .rate_control_mode(mode)
                    .layers(&rc_layers)
                    .virtual_buffer_size_in_ms(1000)
                    .initial_virtual_buffer_size_in_ms(500)
            };

            let mut ctrl = vk::VideoCodingControlInfoKHR::default()
                .flags(
                    vk::VideoCodingControlFlagsKHR::RESET
                        | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL,
                )
                .push_next(&mut rc_info);
            if !self.use_cqp {
                ctrl = match self.codec {
                    VideoCodec::H264 => ctrl.push_next(&mut h264_rc),
                    VideoCodec::H265 => ctrl.push_next(&mut h265_rc),
                };
            }
            (dev_vq.cmd_control_video_coding_khr)(self.cmd_buf, &ctrl);

            let end_coding = vk::VideoEndCodingInfoKHR::default();
            (dev_vq.cmd_end_video_coding_khr)(self.cmd_buf, &end_coding);

            let res = (dev_v1_0.end_command_buffer)(self.cmd_buf);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkEndCommandBuffer (init video) failed: {res:?}"));
            }

            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd_buf));
            let res = (dev_v1_0.queue_submit)(self.ctx.raw_encode_queue(), 1, &submit, self.fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkQueueSubmit (init video) failed: {res:?}"));
            }

            let res = (dev_v1_0.wait_for_fences)(raw_dev, 1, &self.fence, vk::TRUE, u64::MAX);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkWaitForFences (init video) failed: {res:?}"));
            }
        }

        self.rate_control_initialized = true;
        Ok(())
    }

    /// Encode the current VRAM `src_nv12_image()` on the hardware `VIDEO_ENCODE` queue
    /// and return `(annexb_bytes, is_idr_keyframe)`.
    pub fn encode_frame(&mut self, frame_index: u64) -> Result<(&[u8], bool), String> {
        let is_idr = (frame_index % (self.gop_len as u64)) == 0;
        let setup_idx = (frame_index % 2) as usize;
        let ref_idx = ((frame_index + 1) % 2) as usize;

        if is_idr {
            self.frame_num = 0;
            self.poc = 0;
        }

        let raw_dev = self.ctx.raw_device();
        let dev_v1_0 = &self.ctx.device.fns().v1_0;
        let dev_vq = &self.ctx.device.fns().khr_video_queue;
        let dev_ve = &self.ctx.device.fns().khr_video_encode_queue;

        let coded_extent = vk::Extent2D {
            width: self.coded_width,
            height: self.coded_height,
        };

        let src_pic_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(coded_extent)
            .base_array_layer(0)
            .image_view_binding(self.src_image.view);

        let setup_dpb_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(coded_extent)
            .base_array_layer(0)
            .image_view_binding(self.dpb_images[setup_idx].view);

        let ref_dpb_res = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(coded_extent)
            .base_array_layer(0)
            .image_view_binding(self.dpb_images[ref_idx].view);

        let use_ref = !is_idr && self.dpb_initialized[ref_idx];

        unsafe {
            let res = (dev_v1_0.reset_fences)(raw_dev, 1, &self.fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetFences (init video) failed: {res:?}"));
            }
            let res = (dev_v1_0.reset_command_buffer)(
                self.cmd_buf,
                vk::CommandBufferResetFlags::empty(),
            );
            if res != vk::Result::SUCCESS {
                return Err(format!("vkResetCommandBuffer (init video) failed: {res:?}"));
            }

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let res = (dev_v1_0.begin_command_buffer)(self.cmd_buf, &begin);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkBeginCommandBuffer (encode) failed: {res:?}"));
            }

            (dev_v1_0.cmd_reset_query_pool)(self.cmd_buf, self.query_pool, 0, 1);

            match self.codec {
                VideoCodec::H264 => {
                    self.record_encode_h264(
                        dev_v1_0,
                        dev_vq,
                        dev_ve,
                        is_idr,
                        use_ref,
                        setup_idx as i32,
                        ref_idx as i32,
                        &src_pic_res,
                        &setup_dpb_res,
                        &ref_dpb_res,
                    );
                }
                VideoCodec::H265 => {
                    self.record_encode_h265(
                        dev_v1_0,
                        dev_vq,
                        dev_ve,
                        is_idr,
                        use_ref,
                        setup_idx as i32,
                        ref_idx as i32,
                        &src_pic_res,
                        &setup_dpb_res,
                        &ref_dpb_res,
                    );
                }
            }

            let res = (dev_v1_0.end_command_buffer)(self.cmd_buf);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkEndCommandBuffer (encode) failed: {res:?}"));
            }

            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd_buf));
            let res = (dev_v1_0.queue_submit)(self.ctx.raw_encode_queue(), 1, &submit, self.fence);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkQueueSubmit (encode) failed: {res:?}"));
            }

            let res = (dev_v1_0.wait_for_fences)(raw_dev, 1, &self.fence, vk::TRUE, u64::MAX);
            if res != vk::Result::SUCCESS {
                return Err(format!("vkWaitForFences (encode) failed: {res:?}"));
            }

            // Query [offset, bytes_written, status] from VIDEO_ENCODE_FEEDBACK query pool
            let mut feedback = [0u32; 3];
            let res = (dev_v1_0.get_query_pool_results)(
                raw_dev,
                self.query_pool,
                0,
                1,
                std::mem::size_of_val(&feedback),
                feedback.as_mut_ptr() as *mut c_void,
                std::mem::size_of_val(&feedback) as vk::DeviceSize,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::WITH_STATUS_KHR,
            );
            if res != vk::Result::SUCCESS {
                return Err(format!(
                    "vkGetQueryPoolResults (VIDEO_ENCODE_FEEDBACK) failed: {res:?}"
                ));
            }

            let offset = feedback[0] as usize;
            let bytes_written = feedback[1] as usize;
            let status = feedback[2] as i32;
            if status <= 0 {
                return Err(format!(
                    "Vulkan Video Encode reported failure status={status} on frame {frame_index}"
                ));
            }
            if offset.saturating_add(bytes_written) > self.dst_size as usize {
                return Err(format!(
                    "Vulkan Video bitstream bounds overflow (offset={offset}, len={bytes_written}, cap={})",
                    self.dst_size
                ));
            }

            self.bitstream_scratch.clear();
            let slice = std::slice::from_raw_parts(self.dst_mapped.add(offset), bytes_written);
            self.bitstream_scratch.extend_from_slice(slice);
        }

        // Advance DPB & frame numbering
        self.dpb_initialized[setup_idx] = true;
        if is_idr {
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1) & 0x7FFF;
        }
        self.frame_num = (self.frame_num + 1) & 0xFFFF;
        self.poc += 2;

        Ok((&self.bitstream_scratch, is_idr))
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn record_encode_h264(
        &self,
        dev_v1_0: &ash::DeviceFnV1_0,
        dev_vq: &ash::khr::video_queue::DeviceFn,
        dev_ve: &ash::khr::video_encode_queue::DeviceFn,
        is_idr: bool,
        use_ref: bool,
        setup_slot: i32,
        ref_slot: i32,
        src_pic_res: &vk::VideoPictureResourceInfoKHR<'_>,
        setup_dpb_res: &vk::VideoPictureResourceInfoKHR<'_>,
        ref_dpb_res: &vk::VideoPictureResourceInfoKHR<'_>,
    ) {
        let setup_ref_info = vkn::StdVideoEncodeH264ReferenceInfo {
            flags: std::mem::zeroed(),
            primary_pic_type: if is_idr {
                vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            FrameNum: self.frame_num,
            PicOrderCnt: self.poc,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let mut setup_dpb_slot_info = vk::VideoEncodeH264DpbSlotInfoKHR::default()
            .std_reference_info(&setup_ref_info);

        let prev_ref_info = vkn::StdVideoEncodeH264ReferenceInfo {
            flags: std::mem::zeroed(),
            primary_pic_type: vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
            FrameNum: self.frame_num.saturating_sub(1),
            PicOrderCnt: self.poc - 2,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };
        let mut ref_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&prev_ref_info);

        // BeginCoding reference slots: setup slot (inactive slot_index = -1) + optional active ref
        let begin_setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(setup_dpb_res);
        let begin_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot)
            .picture_resource(ref_dpb_res);
        let begin_slots_2 = [begin_setup_slot, begin_ref_slot];
        let begin_slots_1 = [begin_setup_slot];
        let begin_slots: &[vk::VideoReferenceSlotInfoKHR<'_>] = if use_ref {
            &begin_slots_2
        } else {
            &begin_slots_1
        };

        let begin_coding = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(begin_slots);
        (dev_vq.cmd_begin_video_coding_khr)(self.cmd_buf, &begin_coding);

        let mut ref_lists = vkn::StdVideoEncodeH264ReferenceListsInfo {
            flags: vkn::StdVideoEncodeH264ReferenceListsInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: Default::default(),
            },
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            RefPicList0: [0xFF; 32],
            RefPicList1: [0xFF; 32],
            refList0ModOpCount: 0,
            refList1ModOpCount: 0,
            refPicMarkingOpCount: 0,
            reserved1: [0; 7],
            pRefList0ModOperations: std::ptr::null(),
            pRefList1ModOperations: std::ptr::null(),
            pRefPicMarkingOperations: std::ptr::null(),
        };
        if use_ref {
            ref_lists.RefPicList0[0] = ref_slot as u8;
        }

        let mut pic_flags = vkn::StdVideoEncodeH264PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        pic_flags.set_IdrPicFlag(u32::from(is_idr));
        pic_flags.set_is_reference(1);

        let std_pic_info = vkn::StdVideoEncodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            idr_pic_id: self.idr_pic_id,
            primary_pic_type: if is_idr {
                vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                vkn::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            frame_num: self.frame_num,
            PicOrderCnt: self.poc,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: if use_ref {
                &ref_lists
            } else {
                std::ptr::null()
            },
        };

        let mut slice_flags = vkn::StdVideoEncodeH264SliceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        slice_flags.set_direct_spatial_mv_pred_flag(1);

        let std_slice = vkn::StdVideoEncodeH264SliceHeader {
            flags: slice_flags,
            first_mb_in_slice: 0,
            slice_type: if is_idr {
                vkn::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
            } else {
                vkn::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
            },
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            cabac_init_idc: 0,
            disable_deblocking_filter_idc: 0,
            pWeightTable: std::ptr::null(),
        };

        let slice_entries = [vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(self.constant_qp)
            .std_slice_header(&std_slice)];

        let mut h264_pic_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(&slice_entries)
            .std_picture_info(&std_pic_info)
            .generate_prefix_nalu(false);

        let encode_setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_slot)
            .picture_resource(setup_dpb_res)
            .push_next(&mut setup_dpb_slot_info);
        let encode_ref_slots = [vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot)
            .picture_resource(ref_dpb_res)
            .push_next(&mut ref_dpb_slot_info)];

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.dst_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(self.dst_size)
            .src_picture_resource(*src_pic_res)
            .setup_reference_slot(&encode_setup_slot)
            .push_next(&mut h264_pic_info);
        if use_ref {
            encode_info = encode_info.reference_slots(&encode_ref_slots);
        }

        (dev_v1_0.cmd_begin_query)(self.cmd_buf, self.query_pool, 0, vk::QueryControlFlags::empty());
        (dev_ve.cmd_encode_video_khr)(self.cmd_buf, &encode_info);
        (dev_v1_0.cmd_end_query)(self.cmd_buf, self.query_pool, 0);

        let end_coding = vk::VideoEndCodingInfoKHR::default();
        (dev_vq.cmd_end_video_coding_khr)(self.cmd_buf, &end_coding);
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn record_encode_h265(
        &self,
        dev_v1_0: &ash::DeviceFnV1_0,
        dev_vq: &ash::khr::video_queue::DeviceFn,
        dev_ve: &ash::khr::video_encode_queue::DeviceFn,
        is_idr: bool,
        use_ref: bool,
        setup_slot: i32,
        ref_slot: i32,
        src_pic_res: &vk::VideoPictureResourceInfoKHR<'_>,
        setup_dpb_res: &vk::VideoPictureResourceInfoKHR<'_>,
        ref_dpb_res: &vk::VideoPictureResourceInfoKHR<'_>,
    ) {
        let setup_ref_info = vkn::StdVideoEncodeH265ReferenceInfo {
            flags: std::mem::zeroed(),
            pic_type: if is_idr {
                vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
            } else {
                vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
            },
            PicOrderCntVal: self.poc,
            TemporalId: 0,
        };
        let mut setup_dpb_slot_info = vk::VideoEncodeH265DpbSlotInfoKHR::default()
            .std_reference_info(&setup_ref_info);

        let prev_ref_info = vkn::StdVideoEncodeH265ReferenceInfo {
            flags: std::mem::zeroed(),
            pic_type: vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P,
            PicOrderCntVal: self.poc - 2,
            TemporalId: 0,
        };
        let mut ref_dpb_slot_info =
            vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(&prev_ref_info);

        let begin_setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(setup_dpb_res);
        let begin_ref_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot)
            .picture_resource(ref_dpb_res);
        let begin_slots_2 = [begin_setup_slot, begin_ref_slot];
        let begin_slots_1 = [begin_setup_slot];
        let begin_slots: &[vk::VideoReferenceSlotInfoKHR<'_>] = if use_ref {
            &begin_slots_2
        } else {
            &begin_slots_1
        };

        let begin_coding = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(begin_slots);
        (dev_vq.cmd_begin_video_coding_khr)(self.cmd_buf, &begin_coding);

        let mut ref_lists = vkn::StdVideoEncodeH265ReferenceListsInfo {
            flags: vkn::StdVideoEncodeH265ReferenceListsInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1: Default::default(),
            },
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            RefPicList0: [0xFF; 15],
            RefPicList1: [0xFF; 15],
            list_entry_l0: [0; 15],
            list_entry_l1: [0; 15],
        };
        if use_ref {
            ref_lists.RefPicList0[0] = ref_slot as u8;
        }

        // Inline short-term reference picture set for P-frames (delta_poc_s0 = -2)
        let mut st_rps = vkn::StdVideoH265ShortTermRefPicSet {
            flags: std::mem::zeroed(),
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: if use_ref { 1 } else { 0 },
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: if use_ref { 1 } else { 0 },
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        };
        if use_ref {
            // abs(delta_poc) - 1 = 2 - 1 = 1
            st_rps.delta_poc_s0_minus1[0] = 1;
        }

        let mut pic_flags = vkn::StdVideoEncodeH265PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        pic_flags.set_is_reference(1);
        pic_flags.set_IrapPicFlag(u32::from(is_idr));
        pic_flags.set_pic_output_flag(1);
        pic_flags.set_short_term_ref_pic_set_sps_flag(0);

        let std_pic_info = vkn::StdVideoEncodeH265PictureInfo {
            flags: pic_flags,
            pic_type: if is_idr {
                vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
            } else {
                vkn::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
            },
            sps_video_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            pps_pic_parameter_set_id: 0,
            short_term_ref_pic_set_idx: 0,
            PicOrderCntVal: self.poc,
            TemporalId: 0,
            reserved1: [0; 7],
            pRefLists: if use_ref {
                &ref_lists
            } else {
                std::ptr::null()
            },
            pShortTermRefPicSet: &st_rps,
            pLongTermRefPics: std::ptr::null(),
        };

        let mut slice_flags = vkn::StdVideoEncodeH265SliceSegmentHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        slice_flags.set_first_slice_segment_in_pic_flag(1);
        slice_flags.set_slice_loop_filter_across_slices_enabled_flag(1);

        let std_slice = vkn::StdVideoEncodeH265SliceSegmentHeader {
            flags: slice_flags,
            slice_type: if is_idr {
                vkn::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
            } else {
                vkn::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
            },
            slice_segment_address: 0,
            collocated_ref_idx: 0,
            MaxNumMergeCand: 5,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_act_y_qp_offset: 0,
            slice_act_cb_qp_offset: 0,
            slice_act_cr_qp_offset: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            pWeightTable: std::ptr::null(),
        };

        let slice_entries = [vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
            .constant_qp(self.constant_qp)
            .std_slice_segment_header(&std_slice)];

        let mut h265_pic_info = vk::VideoEncodeH265PictureInfoKHR::default()
            .nalu_slice_segment_entries(&slice_entries)
            .std_picture_info(&std_pic_info);

        let encode_setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup_slot)
            .picture_resource(setup_dpb_res)
            .push_next(&mut setup_dpb_slot_info);
        let encode_ref_slots = [vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(ref_slot)
            .picture_resource(ref_dpb_res)
            .push_next(&mut ref_dpb_slot_info)];

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.dst_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(self.dst_size)
            .src_picture_resource(*src_pic_res)
            .setup_reference_slot(&encode_setup_slot)
            .push_next(&mut h265_pic_info);
        if use_ref {
            encode_info = encode_info.reference_slots(&encode_ref_slots);
        }

        (dev_v1_0.cmd_begin_query)(self.cmd_buf, self.query_pool, 0, vk::QueryControlFlags::empty());
        (dev_ve.cmd_encode_video_khr)(self.cmd_buf, &encode_info);
        (dev_v1_0.cmd_end_query)(self.cmd_buf, self.query_pool, 0);

        let end_coding = vk::VideoEndCodingInfoKHR::default();
        (dev_vq.cmd_end_video_coding_khr)(self.cmd_buf, &end_coding);
    }
}

impl Drop for VulkanVideoEncoder {
    fn drop(&mut self) {
        let raw_dev = self.ctx.raw_device();
        let dev_v1_0 = &self.ctx.device.fns().v1_0;
        let dev_vq = &self.ctx.device.fns().khr_video_queue;
        unsafe {
            let _ = (dev_v1_0.device_wait_idle)(raw_dev);
            if self.query_pool != vk::QueryPool::null() {
                (dev_v1_0.destroy_query_pool)(raw_dev, self.query_pool, std::ptr::null());
            }
            if self.fence != vk::Fence::null() {
                (dev_v1_0.destroy_fence)(raw_dev, self.fence, std::ptr::null());
            }
            if self.cmd_pool != vk::CommandPool::null() {
                (dev_v1_0.destroy_command_pool)(raw_dev, self.cmd_pool, std::ptr::null());
            }
            if !self.dst_mapped.is_null() {
                (dev_v1_0.unmap_memory)(raw_dev, self.dst_memory);
            }
            if self.dst_buffer != vk::Buffer::null() {
                (dev_v1_0.destroy_buffer)(raw_dev, self.dst_buffer, std::ptr::null());
            }
            if self.dst_memory != vk::DeviceMemory::null() {
                (dev_v1_0.free_memory)(raw_dev, self.dst_memory, std::ptr::null());
            }
            destroy_video_image(dev_v1_0, raw_dev, &self.src_image);
            destroy_video_image(dev_v1_0, raw_dev, &self.dpb_images[0]);
            destroy_video_image(dev_v1_0, raw_dev, &self.dpb_images[1]);
            if self.session_params != vk::VideoSessionParametersKHR::null() {
                (dev_vq.destroy_video_session_parameters_khr)(
                    raw_dev,
                    self.session_params,
                    std::ptr::null(),
                );
            }
            if self.session != vk::VideoSessionKHR::null() {
                (dev_vq.destroy_video_session_khr)(raw_dev, self.session, std::ptr::null());
            }
            for mem in self.session_memories.drain(..) {
                (dev_v1_0.free_memory)(raw_dev, mem, std::ptr::null());
            }
        }
    }
}

fn align_up(v: u32, align: u32) -> u32 {
    let a = align.max(1);
    (v + a - 1) / a * a
}

fn align_up_u64(v: u64, align: u64) -> u64 {
    let a = align.max(1);
    (v + a - 1) / a * a
}

fn query_preferred_nv12_format(
    inst_vq: &ash::khr::video_queue::InstanceFn,
    raw_pd: vk::PhysicalDevice,
    profile_info: &vk::VideoProfileInfoKHR<'_>,
    usage: vk::ImageUsageFlags,
) -> Result<vk::Format, String> {
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let fmt_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(usage)
        .push_next(&mut profile_list);

    let mut count = 0u32;
    let res = unsafe {
        (inst_vq.get_physical_device_video_format_properties_khr)(
            raw_pd,
            &fmt_info,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if res != vk::Result::SUCCESS || count == 0 {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR ({usage:?}) returned count={count}, res={res:?}"
        ));
    }

    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    let res = unsafe {
        (inst_vq.get_physical_device_video_format_properties_khr)(
            raw_pd,
            &fmt_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetPhysicalDeviceVideoFormatPropertiesKHR (fetch) failed: {res:?}"
        ));
    }

    if props
        .iter()
        .any(|p| p.format == vk::Format::G8_B8R8_2PLANE_420_UNORM)
    {
        Ok(vk::Format::G8_B8R8_2PLANE_420_UNORM)
    } else {
        Ok(props[0].format)
    }
}

fn bind_video_session_memory(
    ctx: &OfflineVulkanContext,
    session: vk::VideoSessionKHR,
) -> Result<Vec<vk::DeviceMemory>, String> {
    let raw_dev = ctx.raw_device();
    let dev_v1_0 = &ctx.device.fns().v1_0;
    let dev_vq = &ctx.device.fns().khr_video_queue;

    let mut count = 0u32;
    let res = unsafe {
        (dev_vq.get_video_session_memory_requirements_khr)(
            raw_dev,
            session,
            &mut count,
            std::ptr::null_mut(),
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetVideoSessionMemoryRequirementsKHR (count) failed: {res:?}"
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
    let res = unsafe {
        (dev_vq.get_video_session_memory_requirements_khr)(
            raw_dev,
            session,
            &mut count,
            reqs.as_mut_ptr(),
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetVideoSessionMemoryRequirementsKHR (list) failed: {res:?}"
        ));
    }

    let mut memories = Vec::with_capacity(reqs.len());
    let mut binds = Vec::with_capacity(reqs.len());
    for req in &reqs {
        let mem_type = ctx
            .find_memory_type(
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|_| {
                ctx.find_memory_type(
                    req.memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.memory_requirements.size)
            .memory_type_index(mem_type);
        let mut mem = vk::DeviceMemory::null();
        let res = unsafe { (dev_v1_0.allocate_memory)(raw_dev, &alloc, std::ptr::null(), &mut mem) };
        if res != vk::Result::SUCCESS {
            return Err(format!("vkAllocateMemory (video session) failed: {res:?}"));
        }
        memories.push(mem);
        binds.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(mem)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size),
        );
    }

    let res = unsafe {
        (dev_vq.bind_video_session_memory_khr)(raw_dev, session, binds.len() as u32, binds.as_ptr())
    };
    if res != vk::Result::SUCCESS {
        return Err(format!("vkBindVideoSessionMemoryKHR failed: {res:?}"));
    }
    Ok(memories)
}

fn create_video_nv12_image(
    ctx: &OfflineVulkanContext,
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
    profiles: &[vk::VideoProfileInfoKHR<'_>],
    share_with_gfx: bool,
) -> Result<VideoImageAlloc, String> {
    let raw_dev = ctx.raw_device();
    let dev_v1_0 = &ctx.device.fns().v1_0;

    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(profiles);
    let families = [ctx.gfx_queue_family_index, ctx.encode_queue_family_index];
    let mut img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut profile_list);

    if share_with_gfx && ctx.gfx_queue_family_index != ctx.encode_queue_family_index {
        img_info = img_info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&families);
    } else {
        img_info = img_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
    }

    let mut image = vk::Image::null();
    let res = unsafe { (dev_v1_0.create_image)(raw_dev, &img_info, std::ptr::null(), &mut image) };
    if res != vk::Result::SUCCESS {
        return Err(format!("vkCreateImage (NV12 video) failed: {res:?}"));
    }

    let mut mem_req = vk::MemoryRequirements::default();
    unsafe { (dev_v1_0.get_image_memory_requirements)(raw_dev, image, &mut mem_req) };
    let mem_type = ctx.find_memory_type(
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let mut memory = vk::DeviceMemory::null();
    let res = unsafe { (dev_v1_0.allocate_memory)(raw_dev, &alloc, std::ptr::null(), &mut memory) };
    if res != vk::Result::SUCCESS {
        return Err(format!("vkAllocateMemory (NV12 video) failed: {res:?}"));
    }
    let res = unsafe { (dev_v1_0.bind_image_memory)(raw_dev, image, memory, 0) };
    if res != vk::Result::SUCCESS {
        return Err(format!("vkBindImageMemory (NV12 video) failed: {res:?}"));
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    let mut view = vk::ImageView::null();
    let res =
        unsafe { (dev_v1_0.create_image_view)(raw_dev, &view_info, std::ptr::null(), &mut view) };
    if res != vk::Result::SUCCESS {
        return Err(format!("vkCreateImageView (NV12 video) failed: {res:?}"));
    }

    Ok(VideoImageAlloc {
        image,
        memory,
        view,
    })
}

unsafe fn destroy_video_image(
    dev_v1_0: &ash::DeviceFnV1_0,
    raw_dev: vk::Device,
    alloc: &VideoImageAlloc,
) {
    if alloc.view != vk::ImageView::null() {
        (dev_v1_0.destroy_image_view)(raw_dev, alloc.view, std::ptr::null());
    }
    if alloc.image != vk::Image::null() {
        (dev_v1_0.destroy_image)(raw_dev, alloc.image, std::ptr::null());
    }
    if alloc.memory != vk::DeviceMemory::null() {
        (dev_v1_0.free_memory)(raw_dev, alloc.memory, std::ptr::null());
    }
}

#[allow(clippy::too_many_arguments)]
fn create_h264_session_params(
    dev_vq: &ash::khr::video_queue::DeviceFn,
    dev_ve: &ash::khr::video_encode_queue::DeviceFn,
    raw_dev: vk::Device,
    session: vk::VideoSessionKHR,
    display_w: u32,
    display_h: u32,
    coded_w: u32,
    coded_h: u32,
    max_level_idc: vkn::StdVideoH264LevelIdc,
) -> Result<(vk::VideoSessionParametersKHR, Vec<u8>), String> {
    let crop_right = coded_w.saturating_sub(display_w) / 2;
    let crop_bottom = coded_h.saturating_sub(display_h) / 2;
    let has_crop = crop_right > 0 || crop_bottom > 0;

    let mut sps_flags: vkn::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
    sps_flags.set_frame_mbs_only_flag(1);
    sps_flags.set_direct_8x8_inference_flag(1);
    if has_crop {
        sps_flags.set_frame_cropping_flag(1);
    }

    let sps = [vkn::StdVideoH264SequenceParameterSet {
        flags: sps_flags,
        profile_idc: vkn::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        level_idc: max_level_idc,
        chroma_format_idc: vkn::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        seq_parameter_set_id: 0,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_frame_num_minus4: 4,
        pic_order_cnt_type: vkn::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: 4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: 1,
        reserved1: 0,
        pic_width_in_mbs_minus1: (coded_w / 16).saturating_sub(1),
        pic_height_in_map_units_minus1: (coded_h / 16).saturating_sub(1),
        frame_crop_left_offset: 0,
        frame_crop_right_offset: crop_right,
        frame_crop_top_offset: 0,
        frame_crop_bottom_offset: crop_bottom,
        reserved2: 0,
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
    }];

    let mut pps_flags: vkn::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
    pps_flags.set_entropy_coding_mode_flag(1); // CABAC
    pps_flags.set_transform_8x8_mode_flag(1); // High profile 8x8
    pps_flags.set_deblocking_filter_control_present_flag(1);

    let pps = [vkn::StdVideoH264PictureParameterSet {
        flags: pps_flags,
        seq_parameter_set_id: 0,
        pic_parameter_set_id: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        weighted_bipred_idc: 0,
        pic_init_qp_minus26: 0,
        pic_init_qs_minus26: 0,
        chroma_qp_index_offset: 0,
        second_chroma_qp_index_offset: 0,
        pScalingLists: std::ptr::null(),
    }];

    let add_info = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
        .std_sp_ss(&sps)
        .std_pp_ss(&pps);
    let mut h264_params_info = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
        .max_std_sps_count(1)
        .max_std_pps_count(1)
        .parameters_add_info(&add_info);

    let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
        .video_session(session)
        .push_next(&mut h264_params_info);

    let mut params = vk::VideoSessionParametersKHR::null();
    let res = unsafe {
        (dev_vq.create_video_session_parameters_khr)(
            raw_dev,
            &create_info,
            std::ptr::null(),
            &mut params,
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkCreateVideoSessionParametersKHR (H.264) failed: {res:?}"
        ));
    }

    // Retrieve encoded Annex-B SPS & PPS from driver
    let mut h264_get = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
        .write_std_sps(true)
        .write_std_pps(true)
        .std_sps_id(0)
        .std_pps_id(0);
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(params)
        .push_next(&mut h264_get);

    let mut data_size = 0usize;
    let res = unsafe {
        (dev_ve.get_encoded_video_session_parameters_khr)(
            raw_dev,
            &get_info,
            std::ptr::null_mut(),
            &mut data_size,
            std::ptr::null_mut(),
        )
    };
    if res != vk::Result::SUCCESS || data_size == 0 {
        return Err(format!(
            "vkGetEncodedVideoSessionParametersKHR (H.264 size) failed: res={res:?}, size={data_size}"
        ));
    }

    let mut encoded = vec![0u8; data_size];
    let res = unsafe {
        (dev_ve.get_encoded_video_session_parameters_khr)(
            raw_dev,
            &get_info,
            std::ptr::null_mut(),
            &mut data_size,
            encoded.as_mut_ptr() as *mut c_void,
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetEncodedVideoSessionParametersKHR (H.264 fetch) failed: {res:?}"
        ));
    }
    encoded.truncate(data_size);

    Ok((params, encoded))
}

#[allow(clippy::too_many_arguments)]
fn create_h265_session_params(
    dev_vq: &ash::khr::video_queue::DeviceFn,
    dev_ve: &ash::khr::video_encode_queue::DeviceFn,
    raw_dev: vk::Device,
    session: vk::VideoSessionKHR,
    display_w: u32,
    display_h: u32,
    coded_w: u32,
    coded_h: u32,
    fps: u32,
    h265_caps: &vk::VideoEncodeH265CapabilitiesKHR<'_>,
) -> Result<(vk::VideoSessionParametersKHR, Vec<u8>), String> {
    let mut ptl_flags: vkn::StdVideoH265ProfileTierLevelFlags = unsafe { std::mem::zeroed() };
    ptl_flags.set_general_progressive_source_flag(1);
    ptl_flags.set_general_frame_only_constraint_flag(1);

    let ptl = vkn::StdVideoH265ProfileTierLevel {
        flags: ptl_flags,
        general_profile_idc: vkn::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        general_level_idc: h265_caps.max_level_idc,
    };

    let mut dpb_mgr = vkn::StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: [0; 7],
        max_dec_pic_buffering_minus1: [0; 7],
        max_num_reorder_pics: [0; 7],
    };
    dpb_mgr.max_dec_pic_buffering_minus1[0] = 2;

    let mut vps_flags: vkn::StdVideoH265VpsFlags = unsafe { std::mem::zeroed() };
    vps_flags.set_vps_temporal_id_nesting_flag(1);
    vps_flags.set_vps_sub_layer_ordering_info_present_flag(1);

    let vps = [vkn::StdVideoH265VideoParameterSet {
        flags: vps_flags,
        vps_video_parameter_set_id: 0,
        vps_max_sub_layers_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: 1,
        vps_time_scale: fps,
        vps_num_ticks_poc_diff_one_minus1: 0,
        reserved3: 0,
        pDecPicBufMgr: &dpb_mgr,
        pHrdParameters: std::ptr::null(),
        pProfileTierLevel: &ptl,
    }];

    // Derive valid CTB and transform block log2 parameters from GPU capabilities
    let max_ctb_log2: u8 = if h265_caps
        .ctb_sizes
        .contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_32)
    {
        5 // 32x32
    } else if h265_caps
        .ctb_sizes
        .contains(vk::VideoEncodeH265CtbSizeFlagsKHR::TYPE_64)
    {
        6 // 64x64
    } else {
        4 // 16x16
    };
    let min_cb_log2: u8 = 3; // 8x8
    let min_tb_log2: u8 = 2; // 4x4
    let max_tb_log2: u8 = if h265_caps
        .transform_block_sizes
        .contains(vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_32)
    {
        5.min(max_ctb_log2)
    } else if h265_caps
        .transform_block_sizes
        .contains(vk::VideoEncodeH265TransformBlockSizeFlagsKHR::TYPE_16)
    {
        4.min(max_ctb_log2)
    } else {
        3
    };

    let conf_right = coded_w.saturating_sub(display_w) / 2;
    let conf_bottom = coded_h.saturating_sub(display_h) / 2;
    let has_conf = conf_right > 0 || conf_bottom > 0;

    let mut sps_flags = vkn::StdVideoH265SpsFlags {
        _bitfield_align_1: [],
        _bitfield_1: Default::default(),
    };
    sps_flags.set_sps_temporal_id_nesting_flag(1);
    sps_flags.set_sps_sub_layer_ordering_info_present_flag(1);
    if has_conf {
        sps_flags.set_conformance_window_flag(1);
    }

    let sps = [vkn::StdVideoH265SequenceParameterSet {
        flags: sps_flags,
        chroma_format_idc: vkn::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
        pic_width_in_luma_samples: coded_w,
        pic_height_in_luma_samples: coded_h,
        sps_video_parameter_set_id: 0,
        sps_max_sub_layers_minus1: 0,
        sps_seq_parameter_set_id: 0,
        bit_depth_luma_minus8: 0,
        bit_depth_chroma_minus8: 0,
        log2_max_pic_order_cnt_lsb_minus4: 4,
        log2_min_luma_coding_block_size_minus3: min_cb_log2 - 3,
        log2_diff_max_min_luma_coding_block_size: max_ctb_log2 - min_cb_log2,
        log2_min_luma_transform_block_size_minus2: min_tb_log2 - 2,
        log2_diff_max_min_luma_transform_block_size: max_tb_log2 - min_tb_log2,
        max_transform_hierarchy_depth_inter: 2,
        max_transform_hierarchy_depth_intra: 2,
        num_short_term_ref_pic_sets: 0,
        num_long_term_ref_pics_sps: 0,
        pcm_sample_bit_depth_luma_minus1: 0,
        pcm_sample_bit_depth_chroma_minus1: 0,
        log2_min_pcm_luma_coding_block_size_minus3: 0,
        log2_diff_max_min_pcm_luma_coding_block_size: 0,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: 0,
        conf_win_right_offset: conf_right,
        conf_win_top_offset: 0,
        conf_win_bottom_offset: conf_bottom,
        pProfileTierLevel: &ptl,
        pDecPicBufMgr: &dpb_mgr,
        pScalingLists: std::ptr::null(),
        pShortTermRefPicSet: std::ptr::null(),
        pLongTermRefPicsSps: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }];

    let mut pps_flags = vkn::StdVideoH265PpsFlags {
        _bitfield_align_1: [],
        _bitfield_1: Default::default(),
    };
    pps_flags.set_cu_qp_delta_enabled_flag(1);
    pps_flags.set_pps_loop_filter_across_slices_enabled_flag(1);

    let pps = [vkn::StdVideoH265PictureParameterSet {
        flags: pps_flags,
        pps_pic_parameter_set_id: 0,
        pps_seq_parameter_set_id: 0,
        sps_video_parameter_set_id: 0,
        num_extra_slice_header_bits: 0,
        num_ref_idx_l0_default_active_minus1: 0,
        num_ref_idx_l1_default_active_minus1: 0,
        init_qp_minus26: 0,
        diff_cu_qp_delta_depth: 0,
        pps_cb_qp_offset: 0,
        pps_cr_qp_offset: 0,
        pps_beta_offset_div2: 0,
        pps_tc_offset_div2: 0,
        log2_parallel_merge_level_minus2: 0,
        log2_max_transform_skip_block_size_minus2: 0,
        diff_cu_chroma_qp_offset_depth: 0,
        chroma_qp_offset_list_len_minus1: 0,
        cb_qp_offset_list: [0; 6],
        cr_qp_offset_list: [0; 6],
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: 0,
        num_tile_rows_minus1: 0,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        reserved3: 0,
        pScalingLists: std::ptr::null(),
        pPredictorPaletteEntries: std::ptr::null(),
    }];

    let add_info = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
        .std_vp_ss(&vps)
        .std_sp_ss(&sps)
        .std_pp_ss(&pps);
    let mut h265_params_info = vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
        .max_std_vps_count(1)
        .max_std_sps_count(1)
        .max_std_pps_count(1)
        .parameters_add_info(&add_info);

    let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
        .video_session(session)
        .push_next(&mut h265_params_info);

    let mut params = vk::VideoSessionParametersKHR::null();
    let res = unsafe {
        (dev_vq.create_video_session_parameters_khr)(
            raw_dev,
            &create_info,
            std::ptr::null(),
            &mut params,
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkCreateVideoSessionParametersKHR (H.265) failed: {res:?}"
        ));
    }

    let mut h265_get = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
        .write_std_vps(true)
        .write_std_sps(true)
        .write_std_pps(true)
        .std_vps_id(0)
        .std_sps_id(0)
        .std_pps_id(0);
    let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(params)
        .push_next(&mut h265_get);

    let mut data_size = 0usize;
    let res = unsafe {
        (dev_ve.get_encoded_video_session_parameters_khr)(
            raw_dev,
            &get_info,
            std::ptr::null_mut(),
            &mut data_size,
            std::ptr::null_mut(),
        )
    };
    if res != vk::Result::SUCCESS || data_size == 0 {
        return Err(format!(
            "vkGetEncodedVideoSessionParametersKHR (H.265 size) failed: res={res:?}, size={data_size}"
        ));
    }

    let mut encoded = vec![0u8; data_size];
    let res = unsafe {
        (dev_ve.get_encoded_video_session_parameters_khr)(
            raw_dev,
            &get_info,
            std::ptr::null_mut(),
            &mut data_size,
            encoded.as_mut_ptr() as *mut c_void,
        )
    };
    if res != vk::Result::SUCCESS {
        return Err(format!(
            "vkGetEncodedVideoSessionParametersKHR (H.265 fetch) failed: {res:?}"
        ));
    }
    encoded.truncate(data_size);

    Ok((params, encoded))
}
