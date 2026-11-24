//! Lock-free background job orchestration for offline audio/video rendering.
//!
//! Runs strictly on a dedicated worker thread so the real-time UI and preview audio
//! never block or stall.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};

use super::audio_render::{render_midi_to_wav, AudioRenderRequest};
use super::video_render::{render_midi_to_mp4, VideoRenderRequest};

const STATUS_IDLE: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_COMPLETED: u8 = 2;
const STATUS_CANCELLED: u8 = 3;
const STATUS_FAILED: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderJobPhase {
    Idle,
    Running,
    Completed,
    Cancelled,
    Failed,
}

/// Snapshot polled by the GUI without taking blocking locks during active rendering.
#[derive(Debug, Clone)]
pub struct OfflineRenderSnapshot {
    pub phase: RenderJobPhase,
    pub rendered_frames: u64,
    pub total_frames_est: u64,
    pub sample_rate: u32,
    pub elapsed_secs: f64,
    pub realtime_multiplier: f64,
    pub active_voices: u64,
    pub message: String,
}

impl OfflineRenderSnapshot {
    /// Progress fraction in `[0.0, 1.0]`, or `None` if total duration is unknown.
    pub fn progress_fraction(&self) -> Option<f32> {
        if self.phase == RenderJobPhase::Completed {
            return Some(1.0);
        }
        if self.total_frames_est == 0 {
            return None;
        }
        Some((self.rendered_frames as f64 / self.total_frames_est as f64).clamp(0.0, 1.0) as f32)
    }

    pub fn rendered_audio_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.rendered_frames as f64 / self.sample_rate as f64
        }
    }
}

/// Shared atomic progress block updated by the offline worker thread.
pub struct OfflineRenderProgress {
    status_tag: AtomicU8,
    cancel_requested: AtomicBool,
    rendered_frames: AtomicU64,
    total_frames_est: AtomicU64,
    sample_rate: AtomicU32,
    elapsed_us: AtomicU64,
    active_voices: AtomicU64,
    final_message: Mutex<String>,
}

impl OfflineRenderProgress {
    pub fn new() -> Self {
        Self {
            status_tag: AtomicU8::new(STATUS_IDLE),
            cancel_requested: AtomicBool::new(false),
            rendered_frames: AtomicU64::new(0),
            total_frames_est: AtomicU64::new(0),
            sample_rate: AtomicU32::new(kdmapiv3::DEFAULT_SAMPLE_RATE),
            elapsed_us: AtomicU64::new(0),
            active_voices: AtomicU64::new(0),
            final_message: Mutex::new(String::new()),
        }
    }

    #[inline]
    pub fn is_cancel_requested(&self) -> bool {
        self.cancel_requested.load(Ordering::Acquire)
    }

    #[inline]
    pub fn request_cancel(&self) {
        self.cancel_requested.store(true, Ordering::Release);
    }

    pub fn begin(&self, sample_rate: u32, total_frames_est: u64) {
        self.cancel_requested.store(false, Ordering::Release);
        self.rendered_frames.store(0, Ordering::Relaxed);
        self.total_frames_est
            .store(total_frames_est, Ordering::Relaxed);
        self.sample_rate.store(sample_rate, Ordering::Relaxed);
        self.elapsed_us.store(0, Ordering::Relaxed);
        self.active_voices.store(0, Ordering::Relaxed);
        if let Ok(mut msg) = self.final_message.lock() {
            msg.clear();
        }
        self.status_tag.store(STATUS_RUNNING, Ordering::Release);
    }

    #[inline]
    pub fn update_metrics(&self, rendered_frames: u64, elapsed_us: u64, active_voices: u64) {
        self.rendered_frames
            .store(rendered_frames, Ordering::Relaxed);
        self.elapsed_us.store(elapsed_us, Ordering::Relaxed);
        self.active_voices.store(active_voices, Ordering::Relaxed);
    }

    pub fn finish_completed(&self, rendered_frames: u64, elapsed_us: u64, message: String) {
        self.rendered_frames
            .store(rendered_frames, Ordering::Relaxed);
        self.elapsed_us.store(elapsed_us, Ordering::Relaxed);
        self.active_voices.store(0, Ordering::Relaxed);
        if let Ok(mut msg) = self.final_message.lock() {
            *msg = message;
        }
        self.status_tag.store(STATUS_COMPLETED, Ordering::Release);
    }

    pub fn finish_cancelled(&self, message: String) {
        self.active_voices.store(0, Ordering::Relaxed);
        if let Ok(mut msg) = self.final_message.lock() {
            *msg = message;
        }
        self.status_tag.store(STATUS_CANCELLED, Ordering::Release);
    }

    pub fn finish_failed(&self, message: String) {
        self.active_voices.store(0, Ordering::Relaxed);
        if let Ok(mut msg) = self.final_message.lock() {
            *msg = message;
        }
        self.status_tag.store(STATUS_FAILED, Ordering::Release);
    }

    pub fn snapshot(&self) -> OfflineRenderSnapshot {
        let tag = self.status_tag.load(Ordering::Acquire);
        let phase = match tag {
            STATUS_RUNNING => RenderJobPhase::Running,
            STATUS_COMPLETED => RenderJobPhase::Completed,
            STATUS_CANCELLED => RenderJobPhase::Cancelled,
            STATUS_FAILED => RenderJobPhase::Failed,
            _ => RenderJobPhase::Idle,
        };
        let rendered_frames = self.rendered_frames.load(Ordering::Relaxed);
        let total_frames_est = self.total_frames_est.load(Ordering::Relaxed);
        let sample_rate = self.sample_rate.load(Ordering::Relaxed);
        let elapsed_secs = self.elapsed_us.load(Ordering::Relaxed) as f64 * 1e-6;
        let audio_secs = if sample_rate > 0 {
            rendered_frames as f64 / sample_rate as f64
        } else {
            0.0
        };
        let realtime_multiplier = if elapsed_secs > 1e-6 {
            audio_secs / elapsed_secs
        } else {
            0.0
        };
        let active_voices = self.active_voices.load(Ordering::Relaxed);

        let message = if phase == RenderJobPhase::Running {
            String::new()
        } else {
            self.final_message
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default()
        };

        OfflineRenderSnapshot {
            phase,
            rendered_frames,
            total_frames_est,
            sample_rate,
            elapsed_secs,
            realtime_multiplier,
            active_voices,
            message,
        }
    }
}

/// Manages the lifecycle of a single active background offline render task.
pub struct OfflineRenderManager {
    progress: Arc<OfflineRenderProgress>,
    worker: Option<JoinHandle<()>>,
}

impl OfflineRenderManager {
    pub fn new() -> Self {
        Self {
            progress: Arc::new(OfflineRenderProgress::new()),
            worker: None,
        }
    }

    #[inline]
    pub fn is_running(&self) -> bool {
        self.progress.status_tag.load(Ordering::Acquire) == STATUS_RUNNING
    }

    #[inline]
    pub fn snapshot(&self) -> OfflineRenderSnapshot {
        self.progress.snapshot()
    }

    pub fn cancel(&self) {
        self.progress.request_cancel();
    }

    pub fn start_audio_render(&mut self, request: AudioRenderRequest) -> Result<(), String> {
        if self.is_running() {
            return Err("An offline render job is already running.".into());
        }

        // Cleanly join any previously finished worker handle
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }

        let total_frames_est = request
            .estimated_duration_sec
            .map(|sec| (sec.max(0.0) * request.sample_rate as f64).round() as u64)
            .unwrap_or(0);
        self.progress.begin(request.sample_rate, total_frames_est);

        let progress = Arc::clone(&self.progress);
        let handle = thread::Builder::new()
            .name("wanith-offline-audio".into())
            .spawn(move || match render_midi_to_wav(&request, &progress) {
                Ok(report) if report.cancelled => {
                    progress.finish_cancelled("Render cancelled by user.".into());
                }
                Ok(report) => {
                    let audio_secs = report.frames_written as f64 / report.sample_rate as f64;
                    let msg = format!(
                        "Saved {} ({:.2}s audio in {:.2}s, {:.1}x realtime)",
                        report.output_path.display(),
                        audio_secs,
                        report.elapsed_secs,
                        report.realtime_multiplier
                    );
                    progress.finish_completed(
                        report.frames_written,
                        (report.elapsed_secs * 1e6) as u64,
                        msg,
                    );
                }
                Err(err) => {
                    progress.finish_failed(err);
                }
            })
            .map_err(|e| format!("Failed to spawn offline render thread: {e}"))?;

        self.worker = Some(handle);
        Ok(())
    }

    pub fn start_video_render(&mut self, request: VideoRenderRequest) -> Result<(), String> {
        if self.is_running() {
            return Err("An offline render job is already running.".into());
        }

        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }

        let fps = request.render_settings.fps.max(1);
        self.progress.begin(fps, 0);

        let progress = Arc::clone(&self.progress);
        let handle = thread::Builder::new()
            .name("wanith-offline-video".into())
            .spawn(move || match render_midi_to_mp4(&request, &progress) {
                Ok(report) if report.cancelled => {
                    progress.finish_cancelled("Video render cancelled by user.".into());
                }
                Ok(report) => {
                    let video_secs = report.video_frames_written as f64 / report.fps as f64;
                    let avg_fps = if report.elapsed_secs > 1e-6 {
                        report.video_frames_written as f64 / report.elapsed_secs
                    } else {
                        0.0
                    };
                    let msg = format!(
                        "Saved {} ({} video frames, {} audio samples, {:.2}s in {:.2}s, {:.1} fps, {:.2}x realtime)",
                        report.output_path.display(),
                        report.video_frames_written,
                        report.audio_samples_written,
                        video_secs,
                        report.elapsed_secs,
                        avg_fps,
                        report.realtime_multiplier,
                    );
                    progress.finish_completed(
                        report.video_frames_written,
                        (report.elapsed_secs * 1e6) as u64,
                        msg,
                    );
                }
                Err(err) => {
                    progress.finish_failed(err);
                }
            })
            .map_err(|e| format!("Failed to spawn offline video render thread: {e}"))?;

        self.worker = Some(handle);
        Ok(())
    }

    pub fn report_unavailable(&self, message: String) {
        if !self.is_running() {
            self.progress.finish_failed(message);
        }
    }
}

impl Drop for OfflineRenderManager {
    fn drop(&mut self) {
        self.progress.request_cancel();
    }
}
