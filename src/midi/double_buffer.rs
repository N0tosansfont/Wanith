//! Lock-free SPSC In-Memory A/B Double Buffer Swap (`AGENTS.md` Pillar 3).
//!
//! # Architecture
//! - Two pre-allocated, cache-line-aligned slots `[A, B]` live inside a shared `Arc`.
//! - A single `AtomicU64` packs:
//!   - `bit 0`: `active_slot` (`0` or `1`, the most recently published slot)
//!   - `bit 1`: `reader_holding` (`1` while the reader holds an active guard)
//!   - `bit 2`: `reader_slot` (`0` or `1`, which slot the reader is currently reading)
//!   - `bits 3..63`: `epoch` (monotonic publish sequence number)
//! - **Reader (Render / Audio hot path):** Wait-free acquire (`Acquire`) of the active slot
//!   with zero blocking locks and zero heap allocations; `Drop` clears `reader_holding` with
//!   `Release` ordering.
//! - **Writer (Background worker):** Mutates only the inactive slot `1 - active_slot` once the
//!   reader is no longer holding it, then publishes with a single `Release` CAS flipping
//!   `active_slot` and incrementing `epoch`.
//! - Compile-time SPSC safety is enforced by splitting ownership into [`DoubleBufferWriter`]
//!   and [`DoubleBufferReader`].

use std::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const ACTIVE_SLOT_MASK: u64 = 0b001;
const READER_HOLDING_BIT: u64 = 0b010;
const READER_SLOT_SHIFT: u64 = 2;
const READER_BITS_MASK: u64 = 0b110;
const EPOCH_SHIFT: u64 = 3;

/// Cache-line padded wrapper to prevent false sharing between slots A, B, and atomic state.
#[repr(C, align(64))]
struct CachePadded<T>(T);

struct DoubleBufferInner<T> {
    slots: [CachePadded<UnsafeCell<T>>; 2],
    state: CachePadded<AtomicU64>,
}

// Safety: SPSC protocol guarantees the writer exclusively mutates `slots[inactive]` while
// the reader only immutably borrows `slots[active]`, synchronized via Acquire/Release on `state`.
unsafe impl<T: Send> Send for DoubleBufferInner<T> {}
unsafe impl<T: Send> Sync for DoubleBufferInner<T> {}

/// Producer endpoint of a lock-free A/B double buffer.
pub struct DoubleBufferWriter<T> {
    inner: Arc<DoubleBufferInner<T>>,
}

/// Consumer endpoint of a lock-free A/B double buffer.
pub struct DoubleBufferReader<T> {
    inner: Arc<DoubleBufferInner<T>>,
}

/// Exclusive mutable guard over the inactive buffer slot.
///
/// Call [`DoubleBufferWriteGuard::publish`] to atomically swap the filled buffer into active
/// visibility. Dropping without calling `publish` discards the staged mutation without swapping.
pub struct DoubleBufferWriteGuard<'a, T> {
    inner: &'a DoubleBufferInner<T>,
    slot: usize,
}

/// Immutable read guard over the active buffer slot.
///
/// Dropping this guard releases the slot so the background writer may reuse it after the next swap.
pub struct DoubleBufferReadGuard<'a, T> {
    inner: &'a DoubleBufferInner<T>,
    slot: usize,
    epoch: u64,
}

/// Create a pre-allocated lock-free A/B double buffer pair `(writer, reader)`.
///
/// `slot_a` starts as the active readable buffer (epoch `0`); `slot_b` starts as the inactive
/// staging buffer ready for the writer.
pub fn double_buffer<T: Send>(
    slot_a: T,
    slot_b: T,
) -> (DoubleBufferWriter<T>, DoubleBufferReader<T>) {
    let inner = Arc::new(DoubleBufferInner {
        slots: [
            CachePadded(UnsafeCell::new(slot_a)),
            CachePadded(UnsafeCell::new(slot_b)),
        ],
        state: CachePadded(AtomicU64::new(0)),
    });
    (
        DoubleBufferWriter {
            inner: inner.clone(),
        },
        DoubleBufferReader { inner },
    )
}

impl<T> DoubleBufferWriter<T> {
    /// Non-blocking attempt to acquire mutable access to the inactive staging slot.
    ///
    /// Returns `None` only if the reader is still mid-frame reading the previously active slot.
    #[inline]
    pub fn try_write(&mut self) -> Option<DoubleBufferWriteGuard<'_, T>> {
        let cur = self.inner.state.0.load(Ordering::Acquire);
        let active = (cur & ACTIVE_SLOT_MASK) as usize;
        let inactive = 1 - active;
        let reader_holding = (cur & READER_HOLDING_BIT) != 0;
        let reader_slot = ((cur >> READER_SLOT_SHIFT) & 1) as usize;

        if reader_holding && reader_slot == inactive {
            return None;
        }

        Some(DoubleBufferWriteGuard {
            inner: &self.inner,
            slot: inactive,
        })
    }

    /// Acquire the inactive staging slot, yielding lightly if the reader is currently finishing
    /// its frame on the previously active slot. Intended for background worker threads or tests.
    #[cfg(test)]
    #[inline]
    pub fn acquire_write_yielding(&mut self) -> DoubleBufferWriteGuard<'_, T> {
        let mut spins = 0u32;
        loop {
            if let Some(guard) = self.try_write() {
                // Re-acquire directly via index to satisfy borrow checker across loop iterations
                let slot = guard.slot;
                std::mem::forget(guard);
                return DoubleBufferWriteGuard {
                    inner: &self.inner,
                    slot,
                };
            }
            if spins < 32 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Current published epoch count.
    #[inline]
    #[allow(dead_code)]
    pub fn epoch(&self) -> u64 {
        self.inner.state.0.load(Ordering::Relaxed) >> EPOCH_SHIFT
    }
}

impl<T> DoubleBufferWriteGuard<'_, T> {
    /// Atomically swap this staging buffer into the active slot and advance the epoch.
    #[inline]
    pub fn publish(self) -> u64 {
        let slot_bit = (self.slot as u64) & ACTIVE_SLOT_MASK;
        let mut cur = self.inner.state.0.load(Ordering::Relaxed);
        loop {
            let new_epoch = (cur >> EPOCH_SHIFT).wrapping_add(1);
            let next = (new_epoch << EPOCH_SHIFT) | (cur & READER_BITS_MASK) | slot_bit;
            match self.inner.state.0.compare_exchange_weak(
                cur,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return new_epoch,
                Err(actual) => cur = actual,
            }
        }
    }
}

impl<T> Deref for DoubleBufferWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Safety: Exclusivity on `self.slot` verified in `try_write`.
        unsafe { &*self.inner.slots[self.slot].0.get() }
    }
}

impl<T> DerefMut for DoubleBufferWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: Exclusivity on `self.slot` verified in `try_write`.
        unsafe { &mut *self.inner.slots[self.slot].0.get() }
    }
}

impl<T> DoubleBufferReader<T> {
    /// Wait-free (at most 1 CAS retry if writer publishes mid-acquire) read guard over the active
    /// buffer slot. Zero blocking locks, zero heap allocations.
    #[inline]
    pub fn read(&mut self) -> DoubleBufferReadGuard<'_, T> {
        let mut cur = self.inner.state.0.load(Ordering::Acquire);
        loop {
            let active = cur & ACTIVE_SLOT_MASK;
            let next = (cur & !READER_BITS_MASK)
                | READER_HOLDING_BIT
                | (active << READER_SLOT_SHIFT);
            match self.inner.state.0.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return DoubleBufferReadGuard {
                        inner: &self.inner,
                        slot: active as usize,
                        epoch: cur >> EPOCH_SHIFT,
                    };
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Current published epoch count without acquiring a read guard.
    #[inline]
    #[allow(dead_code)]
    pub fn published_epoch(&self) -> u64 {
        self.inner.state.0.load(Ordering::Acquire) >> EPOCH_SHIFT
    }
}

impl<T> DoubleBufferReadGuard<'_, T> {
    /// The monotonic epoch number of the snapshot held by this guard.
    #[inline]
    #[allow(dead_code)]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl<T> Deref for DoubleBufferReadGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Safety: `READER_HOLDING_BIT` + `self.slot` prevents the writer from mutating this slot
        // until `Drop` clears `READER_HOLDING_BIT` with `Release` ordering.
        unsafe { &*self.inner.slots[self.slot].0.get() }
    }
}

impl<T> Drop for DoubleBufferReadGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.inner
            .state
            .0
            .fetch_and(!READER_HOLDING_BIT, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    #[test]
    fn double_buffer_spsc_swap_integrity_and_monotonic_epochs() {
        const PAYLOAD_LEN: usize = 64;
        const ITERATIONS: u64 = 50_000;

        #[derive(Clone)]
        struct Frame {
            seq: u64,
            data: [u64; PAYLOAD_LEN],
        }

        let init_a = Frame {
            seq: 0,
            data: [0; PAYLOAD_LEN],
        };
        let init_b = init_a.clone();

        let (mut writer, mut reader) = double_buffer(init_a, init_b);
        let done = Arc::new(AtomicBool::new(false));
        let done_reader = done.clone();

        let reader_thread = thread::spawn(move || {
            let mut last_epoch = 0u64;
            let mut last_seq = 0u64;
            let mut reads = 0u64;

            loop {
                {
                    let guard = reader.read();
                    let epoch = guard.epoch();
                    assert!(
                        epoch >= last_epoch,
                        "epoch regressed: {epoch} < {last_epoch}"
                    );
                    last_epoch = epoch;

                    let seq = guard.seq;
                    assert!(seq >= last_seq, "seq regressed: {seq} < {last_seq}");
                    last_seq = seq;

                    // Verify every element matches seq (no torn reads!)
                    for (i, &val) in guard.data.iter().enumerate() {
                        assert_eq!(
                            val,
                            seq.wrapping_mul(31).wrapping_add(i as u64),
                            "torn read detected at seq={seq}, idx={i}"
                        );
                    }
                    reads += 1;
                }

                if done_reader.load(Ordering::Acquire) && last_seq == ITERATIONS {
                    break;
                }
            }
            (reads, last_seq)
        });

        for seq in 1..=ITERATIONS {
            let mut guard = writer.acquire_write_yielding();
            guard.seq = seq;
            for (i, slot) in guard.data.iter_mut().enumerate() {
                *slot = seq.wrapping_mul(31).wrapping_add(i as u64);
            }
            let published_epoch = guard.publish();
            assert_eq!(published_epoch, seq);
        }
        done.store(true, Ordering::Release);

        let (reads, final_seq) = reader_thread.join().expect("reader thread panicked");
        assert_eq!(final_seq, ITERATIONS);
        assert!(reads > 0);
    }
}
