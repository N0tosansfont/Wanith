//! Dedicated, completely isolated offline Vulkan context for headless video rendering &
//! hardware Vulkan Video encoding (`VK_KHR_video_encode_queue`).
//!
//! # Isolation Guarantee (`AGENTS.md` #5)
//! - Creates its own headless `Instance`, `Device`, `Graphics/Compute` queue, and
//!   `VIDEO_ENCODE` queue strictly inside `crate::offline_render`.
//! - Never touches the main real-time preview `Renderer`, window swapchain, or `audio_playback`.

use std::sync::Arc;

use ash::vk;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::StandardMemoryAllocator,
    Version, VulkanLibrary, VulkanObject,
};

use crate::settings::VideoCodec;

pub struct OfflineVulkanContext {
    pub instance: Arc<Instance>,
    pub physical_device: Arc<PhysicalDevice>,
    pub device: Arc<Device>,
    pub gfx_queue: Arc<Queue>,
    pub encode_queue: Arc<Queue>,
    pub gfx_queue_family_index: u32,
    pub encode_queue_family_index: u32,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
}

impl OfflineVulkanContext {
    /// Create a standalone headless Vulkan 1.3 device with Graphics, Compute, and
    /// `VK_KHR_video_encode_queue` + codec extension (`H.264` or `H.265`) enabled.
    pub fn new(codec: VideoCodec) -> Result<Self, String> {
        let library = VulkanLibrary::new()
            .map_err(|e| format!("Failed to load Vulkan library for offline render: {e}"))?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                application_version: Version::V1_3,
                max_api_version: Some(Version::V1_3),
                enabled_extensions: InstanceExtensions::empty(),
                ..Default::default()
            },
        )
        .map_err(|e| format!("Failed to create headless Vulkan 1.3 instance: {e}"))?;

        let required_ext = DeviceExtensions {
            khr_video_queue: true,
            khr_video_encode_queue: true,
            khr_video_encode_h264: codec == VideoCodec::H264,
            khr_video_encode_h265: codec == VideoCodec::H265,
            ..DeviceExtensions::empty()
        };

        let (physical_device, gfx_family, encode_family) = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Failed to enumerate Vulkan GPUs: {e}"))?
            .filter(|p| {
                p.api_version() >= Version::V1_3
                    && p.supported_extensions().contains(&required_ext)
            })
            .filter_map(|p| {
                let props = p.queue_family_properties();
                let gfx = props
                    .iter()
                    .enumerate()
                    .position(|(_, q)| {
                        q.queue_flags
                            .contains(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                    })? as u32;
                let enc = props
                    .iter()
                    .enumerate()
                    .position(|(_, q)| q.queue_flags.contains(QueueFlags::VIDEO_ENCODE))?
                    as u32;
                Some((p, gfx, enc))
            })
            .min_by_key(|(p, _, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .ok_or_else(|| {
                format!(
                    "No Vulkan 1.3 GPU found supporting VK_KHR_video_encode_queue + {}",
                    codec.as_str()
                )
            })?;

        let mut enabled_ext = required_ext;
        if physical_device.supported_extensions().khr_video_maintenance1 {
            enabled_ext.khr_video_maintenance1 = true;
        }

        let enabled_features = DeviceFeatures {
            geometry_shader: true,
            synchronization2: true,
            ..DeviceFeatures::empty()
        };

        let queue_create_infos = if gfx_family == encode_family {
            vec![QueueCreateInfo {
                queue_family_index: gfx_family,
                queues: vec![1.0],
                ..Default::default()
            }]
        } else {
            vec![
                QueueCreateInfo {
                    queue_family_index: gfx_family,
                    queues: vec![1.0],
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: encode_family,
                    queues: vec![1.0],
                    ..Default::default()
                },
            ]
        };

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: enabled_ext,
                enabled_features,
                queue_create_infos,
                ..Default::default()
            },
        )
        .map_err(|e| format!("Failed to create offline Vulkan Video device: {e}"))?;

        let gfx_queue = queues
            .next()
            .ok_or_else(|| "Missing graphics/compute queue".to_string())?;
        let encode_queue = if gfx_family == encode_family {
            gfx_queue.clone()
        } else {
            queues
                .next()
                .ok_or_else(|| "Missing VIDEO_ENCODE queue".to_string())?
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(Self {
            instance,
            physical_device,
            device,
            gfx_queue,
            encode_queue,
            gfx_queue_family_index: gfx_family,
            encode_queue_family_index: encode_family,
            memory_allocator,
        })
    }

    #[inline]
    pub fn raw_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device.handle()
    }

    #[inline]
    pub fn raw_device(&self) -> vk::Device {
        self.device.handle()
    }

    #[inline]
    pub fn raw_gfx_queue(&self) -> vk::Queue {
        self.gfx_queue.handle()
    }

    #[inline]
    pub fn raw_encode_queue(&self) -> vk::Queue {
        self.encode_queue.handle()
    }

    /// Find a Vulkan memory type index matching `type_bits` and required `properties`.
    pub fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<u32, String> {
        let mut mem_props = vk::PhysicalDeviceMemoryProperties::default();
        unsafe {
            (self.instance.fns().v1_0.get_physical_device_memory_properties)(
                self.raw_physical_device(),
                &mut mem_props,
            );
        }
        for i in 0..mem_props.memory_type_count {
            if (type_bits & (1u32 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(required)
            {
                return Ok(i);
            }
        }
        Err(format!(
            "Failed to find Vulkan memory type (bits=0x{type_bits:x}, flags={required:?})"
        ))
    }
}
