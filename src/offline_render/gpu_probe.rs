//! Read-only Vulkan physical device probe for hardware video encode support.
//!
//! Strictly isolated inside `crate::offline_render` — never touches real-time preview
//! audio playback or swapchain rendering.

use vulkano::device::{physical::PhysicalDevice, QueueFlags};

use crate::settings::VideoCodec;

/// Snapshot of the selected GPU's native Vulkan Video Encode capabilities.
#[derive(Debug, Clone, Default)]
pub struct GpuVideoCapabilities {
    pub device_name: String,
    pub has_video_queue: bool,
    pub has_video_encode_queue: bool,
    pub supports_h264: bool,
    pub supports_h265: bool,
    pub encode_queue_family_index: Option<u32>,
}

impl GpuVideoCapabilities {
    /// Inspect a `vulkano::device::physical::PhysicalDevice` once at startup (read-only).
    pub fn probe(physical_device: &PhysicalDevice) -> Self {
        let device_name = physical_device.properties().device_name.clone();
        let ext = physical_device.supported_extensions();
        let has_video_queue = ext.khr_video_queue;
        let has_video_encode_queue = ext.khr_video_encode_queue;
        let encode_queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.contains(QueueFlags::VIDEO_ENCODE))
            .map(|idx| idx as u32);

        let queue_ready =
            has_video_queue && has_video_encode_queue && encode_queue_family_index.is_some();

        Self {
            device_name,
            has_video_queue,
            has_video_encode_queue,
            supports_h264: queue_ready && ext.khr_video_encode_h264,
            supports_h265: queue_ready && ext.khr_video_encode_h265,
            encode_queue_family_index,
        }
    }

    /// Returns `true` if the GPU supports hardware Vulkan Video encoding for `codec`.
    #[inline]
    pub fn supports_codec(&self, codec: VideoCodec) -> bool {
        match codec {
            VideoCodec::H264 => self.supports_h264,
            VideoCodec::H265 => self.supports_h265,
        }
    }

    /// Returns `true` if `VK_KHR_video_encode_queue` and a dedicated encode queue are present.
    #[inline]
    pub fn is_video_encode_ready(&self) -> bool {
        self.has_video_queue
            && self.has_video_encode_queue
            && self.encode_queue_family_index.is_some()
            && (self.supports_h264 || self.supports_h265)
    }
}
