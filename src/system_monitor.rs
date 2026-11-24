//! Low-frequency background system telemetry monitor for the Legit Run HUD.
//!
//! # Real-Time & Hot-Path Isolation (`AGENTS.md` #2)
//! - Audio callbacks and the main UI/render loop NEVER read `/proc`, `/sys`, or call NVML.
//! - A single low-frequency background thread (`wanith-sysmon`, 300ms interval) polls OS/GPU
//!   metrics and publishes them into wait-free atomic slots.
//! - The UI thread calls [`SystemMonitor::snapshot`], which performs strictly 3 `Relaxed`
//!   atomic loads with zero locks, zero heap allocations, and zero syscalls.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(target_os = "linux")]
use std::time::Instant;

const POLL_INTERVAL: Duration = Duration::from_millis(300);
const METRIC_UNAVAILABLE_U32: u32 = u32::MAX;

/// Wait-free telemetry snapshot consumed by the HUD overlay every frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemMetricsSnapshot {
    /// Current process Resident Set Size (RSS) in bytes, if available.
    pub ram_rss_bytes: Option<u64>,
    /// Total physical system RAM in bytes (queried once at monitor initialization).
    pub ram_total_bytes: Option<u64>,
    /// Current process CPU usage percentage normalized across logical CPU cores (`0.0..=100.0`).
    pub process_cpu_pct: Option<f32>,
    /// Primary GPU utilization percentage (`0..=100`), if available.
    pub gpu_util_pct: Option<u32>,
}

struct SystemMetricsShared {
    ram_rss_bytes: AtomicU64,
    ram_total_bytes: AtomicU64,
    process_cpu_tenths: AtomicU32,
    gpu_util_pct: AtomicU32,
    stop: AtomicBool,
    stop_lock: Mutex<()>,
    stop_cv: Condvar,
}

impl SystemMetricsShared {
    fn new() -> Self {
        Self {
            ram_rss_bytes: AtomicU64::new(0),
            ram_total_bytes: AtomicU64::new(0),
            process_cpu_tenths: AtomicU32::new(METRIC_UNAVAILABLE_U32),
            gpu_util_pct: AtomicU32::new(METRIC_UNAVAILABLE_U32),
            stop: AtomicBool::new(false),
            stop_lock: Mutex::new(()),
            stop_cv: Condvar::new(),
        }
    }
}

pub struct SystemMonitor {
    shared: Arc<SystemMetricsShared>,
    worker: Option<JoinHandle<()>>,
}

impl SystemMonitor {
    pub fn start() -> Self {
        let shared = Arc::new(SystemMetricsShared::new());
        let bg = Arc::clone(&shared);

        let worker = thread::Builder::new()
            .name("wanith-sysmon".into())
            .spawn(move || monitor_loop(bg))
            .ok();

        Self { shared, worker }
    }

    /// Lock-free, allocation-free, syscall-free snapshot for the UI/render thread.
    #[inline]
    pub fn snapshot(&self) -> SystemMetricsSnapshot {
        let rss = self.shared.ram_rss_bytes.load(Ordering::Relaxed);
        let total = self.shared.ram_total_bytes.load(Ordering::Relaxed);
        let cpu_tenths = self.shared.process_cpu_tenths.load(Ordering::Relaxed);
        let gpu = self.shared.gpu_util_pct.load(Ordering::Relaxed);

        SystemMetricsSnapshot {
            ram_rss_bytes: if rss > 0 { Some(rss) } else { None },
            ram_total_bytes: if total > 0 { Some(total) } else { None },
            process_cpu_pct: if cpu_tenths != METRIC_UNAVAILABLE_U32 {
                Some(cpu_tenths as f32 * 0.1)
            } else {
                None
            },
            gpu_util_pct: if gpu != METRIC_UNAVAILABLE_U32 {
                Some(gpu)
            } else {
                None
            },
        }
    }
}

impl Drop for SystemMonitor {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        {
            let _guard = self.shared.stop_lock.lock();
            self.shared.stop_cv.notify_all();
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

fn monitor_loop(shared: Arc<SystemMetricsShared>) {
    // Query total physical system RAM strictly once at monitor initialization
    if let Some(total_ram) = read_total_system_ram_bytes() {
        shared.ram_total_bytes.store(total_ram, Ordering::Relaxed);
    }

    let mut gpu_probe = GpuUtilProbe::init();
    #[cfg(target_os = "linux")]
    let mut cpu_sampler = LinuxCpuSampler::new();

    loop {
        if shared.stop.load(Ordering::Acquire) {
            break;
        }

        // 1. RAM RSS
        if let Some(rss) = read_process_rss_bytes() {
            shared.ram_rss_bytes.store(rss, Ordering::Relaxed);
        }

        // 2. Process CPU %
        #[cfg(target_os = "linux")]
        {
            if let Some(pct) = cpu_sampler.sample_pct() {
                let tenths = (pct.clamp(0.0, 100.0) * 10.0).round() as u32;
                shared.process_cpu_tenths.store(tenths, Ordering::Relaxed);
            }
        }

        // 3. GPU Utilization %
        let gpu_val = gpu_probe.sample_pct().unwrap_or(METRIC_UNAVAILABLE_U32);
        shared.gpu_util_pct.store(gpu_val, Ordering::Relaxed);

        // Interruptible low-frequency sleep (zero spin)
        let guard = match shared.stop_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if shared.stop.load(Ordering::Acquire) {
            break;
        }
        let _ = shared.stop_cv.wait_timeout(guard, POLL_INTERVAL);
    }
}

// ============================================================================
// OS-Specific Process RAM & CPU Samplers (background thread only)
// ============================================================================

#[cfg(target_os = "linux")]
fn read_total_system_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_total_system_ram_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_process_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_process_rss_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
struct LinuxCpuSampler {
    prev_ticks: Option<u64>,
    prev_time: Instant,
    num_cpus: f64,
}

#[cfg(target_os = "linux")]
impl LinuxCpuSampler {
    /// Standard Linux `USER_HZ` (`sysconf(_SC_CLK_TCK)`) on 64-bit Linux.
    const CLK_TCK: f64 = 100.0;

    fn new() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get() as f64)
            .unwrap_or(1.0)
            .max(1.0);
        Self {
            prev_ticks: read_proc_self_cpu_ticks(),
            prev_time: Instant::now(),
            num_cpus,
        }
    }

    fn sample_pct(&mut self) -> Option<f32> {
        let now = Instant::now();
        let cur_ticks = read_proc_self_cpu_ticks()?;
        let elapsed = now.duration_since(self.prev_time).as_secs_f64();
        let prev = self.prev_ticks.replace(cur_ticks);
        self.prev_time = now;

        let prev_ticks = prev?;
        if elapsed < 1e-3 {
            return None;
        }
        let delta_ticks = cur_ticks.saturating_sub(prev_ticks) as f64;
        let cpu_secs = delta_ticks / Self::CLK_TCK;
        let pct = (cpu_secs / (elapsed * self.num_cpus)) * 100.0;
        Some(pct.clamp(0.0, 100.0) as f32)
    }
}

#[cfg(target_os = "linux")]
fn read_proc_self_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Skip past process name in parentheses (safe even if comm contains spaces/parens)
    let after_comm = stat.rsplit_once(')')?.1;
    let mut fields = after_comm.split_whitespace();
    // After ')': field 0 is state (stat #3); utime is #14 -> index 11, stime is #15 -> index 12
    let utime: u64 = fields.nth(11)?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

// ============================================================================
// Lightweight GPU Utilization Probe (NVML dynamic load -> Linux DRM sysfs -> N/A)
// ============================================================================

#[repr(C)]
#[derive(Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

type NvmlDevice = *mut std::ffi::c_void;
type NvmlInitV2Fn = unsafe extern "C" fn() -> i32;
type NvmlShutdownFn = unsafe extern "C" fn() -> i32;
type NvmlDeviceGetHandleByIndexV2Fn = unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32;
type NvmlDeviceGetUtilizationRatesFn =
    unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> i32;

struct NvmlHandle {
    _lib: libloading::Library,
    device: NvmlDevice,
    get_util: NvmlDeviceGetUtilizationRatesFn,
    shutdown: NvmlShutdownFn,
}

impl NvmlHandle {
    fn try_load() -> Option<Self> {
        #[cfg(target_os = "linux")]
        let candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"];
        #[cfg(target_os = "windows")]
        let candidates = ["nvml.dll"];
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let candidates: [&str; 0] = [];

        for name in candidates {
            let Ok(lib) = (unsafe { libloading::Library::new(name) }) else {
                continue;
            };
            let init_res = unsafe {
                let Ok(init_fn) = lib.get::<NvmlInitV2Fn>(b"nvmlInit_v2\0") else {
                    continue;
                };
                let Ok(shutdown_fn) = lib.get::<NvmlShutdownFn>(b"nvmlShutdown\0") else {
                    continue;
                };
                let Ok(get_dev_fn) =
                    lib.get::<NvmlDeviceGetHandleByIndexV2Fn>(b"nvmlDeviceGetHandleByIndex_v2\0")
                else {
                    continue;
                };
                let Ok(get_util_fn) = lib
                    .get::<NvmlDeviceGetUtilizationRatesFn>(b"nvmlDeviceGetUtilizationRates\0")
                else {
                    continue;
                };

                if init_fn() != 0 {
                    continue;
                }
                let mut dev: NvmlDevice = std::ptr::null_mut();
                if get_dev_fn(0, &mut dev) != 0 || dev.is_null() {
                    let _ = shutdown_fn();
                    continue;
                }
                Some((dev, *get_util_fn, *shutdown_fn))
            };

            if let Some((device, get_util, shutdown)) = init_res {
                return Some(Self {
                    _lib: lib,
                    device,
                    get_util,
                    shutdown,
                });
            }
        }
        None
    }

    fn sample_pct(&self) -> Option<u32> {
        let mut util = NvmlUtilization::default();
        let rc = unsafe { (self.get_util)(self.device, &mut util) };
        if rc == 0 {
            Some(util.gpu.min(100))
        } else {
            None
        }
    }
}

impl Drop for NvmlHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.shutdown)();
        }
    }
}

enum GpuUtilProbe {
    Nvml(NvmlHandle),
    #[cfg(target_os = "linux")]
    DrmSysfs(std::path::PathBuf),
    Unavailable,
}

impl GpuUtilProbe {
    fn init() -> Self {
        if let Some(nvml) = NvmlHandle::try_load() {
            return Self::Nvml(nvml);
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
                for entry in entries.flatten() {
                    let candidate = entry.path().join("device/gpu_busy_percent");
                    if candidate.is_file() {
                        return Self::DrmSysfs(candidate);
                    }
                }
            }
        }

        Self::Unavailable
    }

    fn sample_pct(&mut self) -> Option<u32> {
        match self {
            Self::Nvml(nvml) => nvml.sample_pct(),
            #[cfg(target_os = "linux")]
            Self::DrmSysfs(path) => {
                let raw = std::fs::read_to_string(path).ok()?;
                let val: u32 = raw.trim().parse().ok()?;
                Some(val.min(100))
            }
            Self::Unavailable => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_monitor_polls_lockfree_snapshot_and_shuts_down_cleanly() {
        let monitor = SystemMonitor::start();
        let mut snap = SystemMetricsSnapshot::default();
        for _ in 0..20 {
            snap = monitor.snapshot();
            if snap.ram_rss_bytes.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(25));
        }

        #[cfg(target_os = "linux")]
        {
            assert!(
                snap.ram_rss_bytes.unwrap_or(0) > 0,
                "expected positive RSS on Linux"
            );
            assert!(
                snap.ram_total_bytes.unwrap_or(0) > snap.ram_rss_bytes.unwrap_or(0),
                "expected total system RAM > process RSS on Linux"
            );
        }
        if let Some(gpu) = snap.gpu_util_pct {
            assert!(gpu <= 100);
        }
        drop(monitor);
    }
}
