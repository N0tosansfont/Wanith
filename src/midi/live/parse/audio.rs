use std::sync::{atomic::Ordering, Arc};

use atomic_float::AtomicF64;
use crossbeam_channel::{Receiver, Sender};

use crate::midi::shared::audio::CompressedAudio;

use super::{ThreadManager, TrackEventBatch};

/// Bounded capacity of the pre-allocated live streaming audio block pool.
/// Prevents unbounded RAM growth during massive Black MIDI streaming while
/// recycling `Vec<u8>` buffers for zero steady-state heap allocations.
const LIVE_AUDIO_POOL_CAPACITY: usize = 512;
const INITIAL_BLOCK_DATA_CAP: usize = 256;
const INITIAL_BLOCK_CTRL_CAP: usize = 32;

pub struct AudioParserResult {
    pub reciever: Receiver<CompressedAudio>,
    pub recycler: Sender<CompressedAudio>,
    pub manager: ThreadManager,
}

pub fn init_audio_manager(blocks: Receiver<Arc<TrackEventBatch>>) -> AudioParserResult {
    let (filled_snd, filled_rcv) = crossbeam_channel::bounded(LIVE_AUDIO_POOL_CAPACITY);
    let (recycle_snd, recycle_rcv) = crossbeam_channel::bounded(LIVE_AUDIO_POOL_CAPACITY);

    for _ in 0..LIVE_AUDIO_POOL_CAPACITY {
        let _ = recycle_snd.try_send(CompressedAudio::with_capacity(
            INITIAL_BLOCK_DATA_CAP,
            INITIAL_BLOCK_CTRL_CAP,
        ));
    }

    let parse_time_outer = Arc::new(AtomicF64::default());
    let parse_time = parse_time_outer.clone();

    let join_handle = std::thread::spawn(move || {
        let mut time = 0.0f64;
        for batch in blocks.into_iter() {
            time += batch.delta;

            // Borrow a pre-allocated block from the recycle pool (backpressured if consumer lags)
            let Ok(mut audio_block) = recycle_rcv.recv() else {
                break;
            };
            audio_block.refill_from_batch(time, &batch);
            parse_time.store(time, Ordering::Relaxed);

            if filled_snd.send(audio_block).is_err() {
                break;
            }
        }
    });

    AudioParserResult {
        reciever: filled_rcv,
        recycler: recycle_snd,
        manager: ThreadManager {
            handle: join_handle,
            parse_time: parse_time_outer,
        },
    }
}
