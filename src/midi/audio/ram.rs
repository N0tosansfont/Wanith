use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};
use kdmapiv3::DirectMidiSender;
use time::Duration;

use crate::{
    audio_playback::WasabiAudioPlayer,
    midi::shared::{
        audio::FlatAudioTimeline,
        timer::{SeekWaitResult, TimeListener, UnpauseWaitResult, WaitResult},
    },
};

pub struct InRamAudioPlayer {
    timeline: FlatAudioTimeline,
    timer: TimeListener,
    player: Arc<WasabiAudioPlayer>,
    index: usize,
    cached_direct: Option<(u64, Option<DirectMidiSender>)>,
}

impl InRamAudioPlayer {
    pub fn new(
        timeline: FlatAudioTimeline,
        timer: TimeListener,
        player: Arc<WasabiAudioPlayer>,
    ) -> Self {
        InRamAudioPlayer {
            timeline,
            timer,
            player,
            index: 0,
            cached_direct: None,
        }
    }

    #[inline]
    fn direct_sender(&mut self) -> Option<&DirectMidiSender> {
        let current_gen = self.player.generation();
        let needs_refresh = match self.cached_direct {
            Some((gen, _)) => gen != current_gen,
            None => true,
        };
        if needs_refresh {
            self.cached_direct = Some(self.player.clone_direct_sender());
        }
        self.cached_direct.as_ref().and_then(|(_, s)| s.as_ref())
    }

    #[inline]
    fn reset_audio(&mut self) {
        if let Some(sender) = self.direct_sender() {
            sender.reset();
        } else {
            self.player.reset();
        }
    }

    pub fn spawn_playback(mut self) -> JoinHandle<()> {
        thread::spawn(move || loop {
            if self.timer.is_paused() {
                self.reset_audio();
                match self.timer.wait_until_unpause() {
                    UnpauseWaitResult::Unpaused => {
                        self.seek_to_time(self.timer.get_time().as_seconds_f64());
                        continue;
                    }
                    UnpauseWaitResult::UnpausedAndSeeked(time) => {
                        self.seek_to_time(time.as_seconds_f64());
                        continue;
                    }
                    UnpauseWaitResult::Killed => break,
                }
            }

            if self.index >= self.timeline.len() {
                match self.timer.wait_until_seeked() {
                    SeekWaitResult::UnpausedAndSeeked(time) => {
                        self.seek_to_time(time.as_seconds_f64());
                        continue;
                    }
                    SeekWaitResult::Killed => break,
                }
            }

            let time = Duration::seconds_f64(self.timeline.time_at(self.index));
            match self.timer.wait_until(time) {
                WaitResult::Ok => {}
                WaitResult::Paused => {
                    continue;
                }
                WaitResult::Seeked(time) => {
                    self.reset_audio();
                    self.seek_to_time(time.as_seconds_f64());
                    continue;
                }
                WaitResult::Killed => {
                    self.reset_audio();
                    break;
                }
            }

            let idx = self.index;
            let current_gen = self.player.generation();
            let needs_refresh = match self.cached_direct {
                Some((gen, _)) => gen != current_gen,
                None => true,
            };
            if needs_refresh {
                self.cached_direct = Some(self.player.clone_direct_sender());
            }
            if let Some(sender) = self.cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                self.timeline.for_each_batch_at(idx, |batch| {
                    sender.push_short_batch(batch);
                });
            } else {
                self.player.push_events(self.timeline.iter_events_at(idx));
            }
            self.index += 1;
        })
    }

    fn seek_to_time(&mut self, time: f64) {
        self.index = self.timeline.find_time_index(time);

        // Reset and push all control events strictly before the target timestamp via sparse index
        self.reset_audio();
        let target = self.index;
        let current_gen = self.player.generation();
        let needs_refresh = match self.cached_direct {
            Some((gen, _)) => gen != current_gen,
            None => true,
        };
        if needs_refresh {
            self.cached_direct = Some(self.player.clone_direct_sender());
        }
        if let Some(sender) = self.cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
            self.timeline.for_each_control_before(target, |batch| {
                sender.push_short_batch(batch);
            });
        } else {
            self.player
                .push_events(self.timeline.iter_control_before(target));
        }
    }
}
