// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoder pool ahead of the language workers.
//!
//! Behavioral model of SGLang's encoder disaggregation as of sgl-project/sglang
//! v0.5.19 (`0bcd822`, `python/sglang/srt/disaggregation/encoder/{runtime.py,
//! server.py, preprocessor.py}`): each `--encoder-only` server runs one serial
//! loop that takes the queued requests up to `SGLANG_ENCODER_MAX_BATCH_SIZE`,
//! runs the image processor and one encoder forward over that batch, then
//! stages and pushes each request's embeddings to the language rank. The push
//! is asynchronous, so the loop takes its next batch as soon as the forward
//! ends. Re-implemented in Rust from the observed semantics; no SGLang source
//! is copied.
//!
//! The pool is generic over what it parks: replay hands it the admitted
//! arrival and gets it back once the embeddings reached the language rank, so
//! admission to the language workers is gated exactly there. Transfers are
//! timed per request at the configured bandwidth, the same precision boundary
//! as the prefill-to-decode KV handoff, and requests arriving at one instant
//! share a batch, as they are already queued when the loop collects it.

use std::collections::VecDeque;

use crate::engine::transfer_delay_ms;
use crate::replay::spec::EncoderSpec;

struct Batch<T> {
    gpu_end_ms: f64,
    parked: Vec<T>,
}

pub(crate) struct EncoderPool<T> {
    spec: EncoderSpec,
    transfer_ms: f64,
    queue: VecDeque<T>,
    instances: Vec<Option<Batch<T>>>,
    /// Requests whose embeddings are on the wire, with their delivery instant.
    in_transfer: Vec<(T, f64)>,
}

impl<T> EncoderPool<T> {
    pub(crate) fn new(spec: EncoderSpec) -> Self {
        let transfer_ms = transfer_delay_ms(
            spec.transfer_bytes_per_request as f64,
            spec.transfer_bandwidth_gb_s,
        );
        Self {
            instances: (0..spec.instances).map(|_| None).collect(),
            spec,
            transfer_ms,
            queue: VecDeque::new(),
            in_transfer: Vec::new(),
        }
    }

    /// Requests the pool owns: queued, being encoded, or in transfer.
    pub(crate) fn parked_count(&self) -> usize {
        self.queue.len()
            + self
                .instances
                .iter()
                .flatten()
                .map(|batch| batch.parked.len())
                .sum::<usize>()
            + self.in_transfer.len()
    }

    /// Queue a request; `take_ready` at the same instant starts it.
    pub(crate) fn submit(&mut self, item: T) {
        self.queue.push_back(item);
    }

    /// Earliest instant at which `take_ready` has work: a forward ending or an
    /// embedding arriving. `None` while queued requests only wait for a call.
    pub(crate) fn next_deadline_ms(&self) -> Option<f64> {
        self.instances
            .iter()
            .flatten()
            .map(|batch| batch.gpu_end_ms)
            .chain(self.in_transfer.iter().map(|(_, ready_ms)| *ready_ms))
            .min_by(f64::total_cmp)
    }

    /// Advance to `now_ms`: finished forwards free their instance at their own
    /// instant, where it takes the next batch, and idle instances start on the
    /// queue. Returns the requests whose embeddings reached the language rank by
    /// `now_ms`, with the instant they did, in that order.
    pub(crate) fn take_ready(&mut self, now_ms: f64) -> Vec<(T, f64)> {
        loop {
            let finished = self
                .instances
                .iter()
                .enumerate()
                .filter_map(|(index, batch)| batch.as_ref().map(|batch| (index, batch.gpu_end_ms)))
                .filter(|(_, gpu_end_ms)| *gpu_end_ms <= now_ms)
                .min_by(|left, right| left.1.total_cmp(&right.1));
            let Some((index, gpu_end_ms)) = finished else {
                break;
            };
            let batch = self.instances[index].take().expect("finished batch");
            for item in batch.parked {
                self.in_transfer.push((item, gpu_end_ms + self.transfer_ms));
            }
            self.start_batches(gpu_end_ms);
        }
        self.start_batches(now_ms);
        let (ready, waiting): (Vec<_>, Vec<_>) = self
            .in_transfer
            .drain(..)
            .partition(|(_, ready_ms)| *ready_ms <= now_ms);
        self.in_transfer = waiting;
        let mut ready = ready;
        ready.sort_by(|left, right| left.1.total_cmp(&right.1));
        ready
    }

    /// Give every idle instance the queued requests up to `max_batch`, at `at_ms`.
    fn start_batches(&mut self, at_ms: f64) {
        for slot in self.instances.iter_mut().filter(|slot| slot.is_none()) {
            if self.queue.is_empty() {
                break;
            }
            let size = self.queue.len().min(self.spec.max_batch);
            let parked: Vec<T> = self.queue.drain(..size).collect();
            let gpu_end_ms = at_ms
                + size as f64 * self.spec.preprocess_ms
                + self.spec.forward_ms_by_batch[size - 1];
            *slot = Some(Batch { gpu_end_ms, parked });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> EncoderPool<u32> {
        EncoderPool::new(EncoderSpec {
            instances: 2,
            max_batch: 2,
            preprocess_ms: 1.0,
            forward_ms_by_batch: vec![10.0, 14.0],
            // 2 MB at 1 GB/s: 2 ms on the wire.
            transfer_bytes_per_request: 2_000_000,
            transfer_bandwidth_gb_s: 1.0,
        })
    }

    #[test]
    fn instances_take_greedy_batches_and_transfers_do_not_hold_them() {
        let mut pool = pool();
        for id in 1..=5 {
            pool.submit(id);
        }
        assert_eq!(pool.next_deadline_ms(), None);
        // Two instances each take two requests at 0: 2 x 1 + 14 = 16 ms of GPU work.
        assert!(pool.take_ready(0.0).is_empty());
        assert_eq!(pool.next_deadline_ms(), Some(16.0));
        assert_eq!(pool.parked_count(), 5);
        assert!(pool.take_ready(15.0).is_empty());
        // Both forwards end at 16: four requests are on the wire until 18 and the
        // fifth starts alone on a freed instance at 16, not at the call's instant.
        assert!(pool.take_ready(17.0).is_empty());
        assert_eq!(pool.next_deadline_ms(), Some(18.0));
        assert_eq!(
            pool.take_ready(18.0),
            vec![(1, 18.0), (2, 18.0), (3, 18.0), (4, 18.0)]
        );
        // 16 + 1 + 10 = 27 on the GPU, delivered at 29.
        assert_eq!(pool.next_deadline_ms(), Some(27.0));
        assert_eq!(pool.take_ready(40.0), vec![(5, 29.0)]);
        assert_eq!(pool.parked_count(), 0);
    }
}
