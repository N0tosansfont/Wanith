use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};
use kdmapiv3::DirectMidiSender;
use time::Duration;

use crossbeam_channel::{Receiver, Sender};

use crate::{
    audio_playback::WasabiAudioPlayer,
    midi::shared::{
        audio::CompressedAudio,
        timer::{TimeListener, UnpauseWaitResult, WaitResult},
    },
};

pub struct LiveAudioPlayer {
    events: Receiver<CompressedAudio>,
    recycler: Sender<CompressedAudio>,
    timer: TimeListener,
    player: Arc<WasabiAudioPlayer>,
}

impl LiveAudioPlayer {
    pub fn new(
        events: Receiver<CompressedAudio>,
        recycler: Sender<CompressedAudio>,
        timer: TimeListener,
        player: Arc<WasabiAudioPlayer>,
    ) -> Self {
        LiveAudioPlayer {
            events,
            recycler,
            timer,
            player,
        }
    }

    pub fn spawn_playback(mut self) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut seek_catching_up = false;
            let max_fall_time = 0.1;
            let mut cached_direct: Option<(u64, Option<DirectMidiSender>)> = None;

            let refresh = |cached: &mut Option<(u64, Option<DirectMidiSender>)>,
                           player: &WasabiAudioPlayer| {
                let current_gen = player.generation();
                let needs_refresh = match cached {
                    Some((gen, _)) => *gen != current_gen,
                    None => true,
                };
                if needs_refresh {
                    *cached = Some(player.clone_direct_sender());
                }
            };

            for event in self.events.into_iter() {
                if self.timer.is_paused() {
                    refresh(&mut cached_direct, &self.player);
                    if let Some(sender) = cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                        sender.reset();
                    } else {
                        self.player.reset();
                    }
                    match self.timer.wait_until_unpause() {
                        UnpauseWaitResult::Unpaused => {
                            refresh(&mut cached_direct, &self.player);
                            if let Some(sender) =
                                cached_direct.as_ref().and_then(|(_, s)| s.as_ref())
                            {
                                event.for_each_control_batch(|batch| {
                                    sender.push_short_batch(batch);
                                });
                            } else {
                                self.player.push_events(event.iter_control_events());
                            }
                        }
                        UnpauseWaitResult::UnpausedAndSeeked(time) => {
                            if time.as_seconds_f64() - event.time > max_fall_time {
                                seek_catching_up = true;
                            }
                            let _ = self.recycler.try_send(event);
                            continue;
                        }
                        UnpauseWaitResult::Killed => break,
                    }
                }

                if seek_catching_up {
                    let time = self.timer.get_time().as_seconds_f64();
                    if time - event.time > max_fall_time {
                        refresh(&mut cached_direct, &self.player);
                        if let Some(sender) = cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                            event.for_each_control_batch(|batch| {
                                sender.push_short_batch(batch);
                            });
                        } else {
                            self.player.push_events(event.iter_control_events());
                        }
                        let _ = self.recycler.try_send(event);
                        continue;
                    } else {
                        seek_catching_up = false;
                    }
                }

                let time = Duration::seconds_f64(event.time);
                match self.timer.wait_until(time) {
                    WaitResult::Ok => {}
                    WaitResult::Paused => {
                        let _ = self.recycler.try_send(event);
                        continue;
                    }
                    WaitResult::Seeked(time) => {
                        refresh(&mut cached_direct, &self.player);
                        if let Some(sender) = cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                            sender.reset();
                        } else {
                            self.player.reset();
                        }
                        if time.as_seconds_f64() - event.time > max_fall_time {
                            seek_catching_up = true;
                        }
                        let _ = self.recycler.try_send(event);
                        continue;
                    }
                    WaitResult::Killed => {
                        refresh(&mut cached_direct, &self.player);
                        if let Some(sender) = cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                            sender.reset();
                        } else {
                            self.player.reset();
                        }
                        break;
                    }
                }

                refresh(&mut cached_direct, &self.player);
                if let Some(sender) = cached_direct.as_ref().and_then(|(_, s)| s.as_ref()) {
                    event.for_each_batch(|batch| {
                        sender.push_short_batch(batch);
                    });
                } else {
                    self.player.push_events(event.iter_events());
                }

                // Return the block to the pool so its backing Vec<u8> buffers are reused
                let _ = self.recycler.try_send(event);
            }
        })
    }
}
