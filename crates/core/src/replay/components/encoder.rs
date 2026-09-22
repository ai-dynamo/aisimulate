// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encoder pool ahead of the language workers.
//!
//! Behavioral model of SGLang's encoder disaggregation as of sgl-project/sglang
//! v0.5.19 (`0bcd822`, `python/sglang/srt/disaggregation/encoder/{runtime.py,
//! server.py, preprocessor.py, receiver.py}`). The language side spreads a
//! request's images evenly over every `--encoder-only` server before sending
//! (`_assign_items_by_modality`; the servers are visited in a random order, which
//! replay replaces with a rotation over the arrival sequence), and admits the
//! request once every part's embeddings arrived. Each server runs one serial loop
//! that takes the queued parts up to `SGLANG_ENCODER_MAX_BATCH_SIZE`, runs the
//! image processor and one encoder forward over that batch, then pushes each
//! part's embeddings asynchronously and takes its next batch. Re-implemented in
//! Rust from the observed semantics; no SGLang source is copied.
//!
//! The pool is generic over what it parks: replay hands it the admitted arrival
//! and gets it back once the last part reached the language rank, so admission
//! to the language workers is gated exactly there. The forward is priced by the
//! same timing model that prices the vision tower on a language rank
//! (`TimingModel::predict_vision_ms` over the batch's image count). Transfers of
//! a batch's parts start together at its end and are timed independently at the
//! configured bandwidth, the same precision boundary as the prefill-to-decode KV
//! handoff; the device-to-host copy SGLang's ZMQ path performs before the push is
//! not modeled.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap;

use crate::engine::{TimingModel, VisionShape, transfer_delay_ms};
use crate::replay::spec::EncoderSpec;

/// The images of one request assigned to one instance.
struct Part {
    ordinal: u64,
    images: u32,
}

struct Batch {
    /// The instance is free again: CPU preprocessing and the forward are done.
    busy_until_ms: f64,
    parts: Vec<Part>,
}

/// A parked request waiting for its parts.
struct Join<T> {
    item: T,
    pending_parts: usize,
    ready_ms: f64,
}

pub(crate) struct EncoderPool<T> {
    spec: EncoderSpec,
    timing: Arc<dyn TimingModel>,
    transfer_ms_per_image: f64,
    next_ordinal: u64,
    /// Queued parts per instance, in arrival order.
    queues: Vec<VecDeque<Part>>,
    /// The batch each instance is running, if any.
    running: Vec<Option<Batch>>,
    /// Parts whose embeddings are on the wire, with their delivery instant.
    in_transfer: Vec<(Part, f64)>,
    joins: FxHashMap<u64, Join<T>>,
}

impl<T> EncoderPool<T> {
    pub(crate) fn new(spec: EncoderSpec, timing: Arc<dyn TimingModel>) -> Self {
        let transfer_ms_per_image = transfer_delay_ms(
            spec.transfer_bytes_per_image as f64,
            spec.transfer_bandwidth_gb_s,
        );
        Self {
            queues: (0..spec.instances).map(|_| VecDeque::new()).collect(),
            running: (0..spec.instances).map(|_| None).collect(),
            spec,
            timing,
            transfer_ms_per_image,
            next_ordinal: 0,
            in_transfer: Vec::new(),
            joins: FxHashMap::default(),
        }
    }

    /// Requests the pool owns: queued, being encoded, or in transfer.
    pub(crate) fn parked_count(&self) -> usize {
        self.joins.len()
    }

    /// Queue a request's parts; `take_ready` at the same instant starts them.
    pub(crate) fn submit(&mut self, item: T) {
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        let instances = self.spec.instances as u32;
        let images = self.spec.images_per_request;
        let (base, remainder) = (images / instances, images % instances);
        // The receiver visits the servers in a random order; replay rotates the
        // starting server with the arrival sequence instead.
        let offset = (ordinal % u64::from(instances)) as u32;
        let mut parts = 0;
        for index in 0..instances {
            let share = base + u32::from(index < remainder);
            if share == 0 {
                continue;
            }
            self.queues[((offset + index) % instances) as usize].push_back(Part {
                ordinal,
                images: share,
            });
            parts += 1;
        }
        self.joins.insert(
            ordinal,
            Join {
                item,
                pending_parts: parts,
                ready_ms: 0.0,
            },
        );
    }

    /// Earliest instant at which `take_ready` has work: a forward ending or an
    /// embedding arriving. `None` while queued parts only wait for a call.
    pub(crate) fn next_deadline_ms(&self) -> Option<f64> {
        self.running
            .iter()
            .flatten()
            .map(|batch| batch.busy_until_ms)
            .chain(self.in_transfer.iter().map(|(_, ready_ms)| *ready_ms))
            .min_by(f64::total_cmp)
    }

    /// Advance to `now_ms`: finished batches free their instance at their own
    /// instant, where it takes the next batch, and idle instances start on their
    /// queues. Returns the requests whose last part reached the language rank by
    /// `now_ms`, with the instant it did, in that order.
    pub(crate) fn take_ready(&mut self, now_ms: f64) -> Result<Vec<(T, f64)>> {
        loop {
            let finished = self
                .running
                .iter()
                .enumerate()
                .filter_map(|(index, batch)| {
                    batch.as_ref().map(|batch| (index, batch.busy_until_ms))
                })
                .filter(|(_, busy_until_ms)| *busy_until_ms <= now_ms)
                .min_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
            let Some((index, busy_until_ms)) = finished else {
                break;
            };
            let batch = self.running[index].take().expect("finished batch");
            for part in batch.parts {
                let ready_ms = busy_until_ms + f64::from(part.images) * self.transfer_ms_per_image;
                self.in_transfer.push((part, ready_ms));
            }
            self.start_batches(busy_until_ms)?;
        }
        self.start_batches(now_ms)?;
        let (mut arrived, waiting): (Vec<_>, Vec<_>) = self
            .in_transfer
            .drain(..)
            .partition(|(_, ready_ms)| *ready_ms <= now_ms);
        self.in_transfer = waiting;
        arrived.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then(left.0.ordinal.cmp(&right.0.ordinal))
        });
        let mut delivered = Vec::new();
        for (part, ready_ms) in arrived {
            let join = self
                .joins
                .get_mut(&part.ordinal)
                .expect("every part belongs to a parked request");
            join.pending_parts -= 1;
            join.ready_ms = join.ready_ms.max(ready_ms);
            if join.pending_parts == 0 {
                let join = self.joins.remove(&part.ordinal).expect("join present");
                delivered.push((join.item, join.ready_ms));
            }
        }
        debug_assert!(
            self.joins.is_empty() || self.next_deadline_ms().is_some(),
            "a parked request must have a future wakeup"
        );
        Ok(delivered)
    }

    /// Give every idle instance the queued parts up to `max_batch`, at `at_ms`.
    fn start_batches(&mut self, at_ms: f64) -> Result<()> {
        for index in 0..self.spec.instances {
            if self.running[index].is_some() || self.queues[index].is_empty() {
                continue;
            }
            let size = self.queues[index].len().min(self.spec.max_batch);
            let parts: Vec<Part> = self.queues[index].drain(..size).collect();
            let images = parts.iter().map(|part| part.images).sum::<u32>();
            let busy_until_ms = at_ms + self.batch_busy_ms(images)?;
            self.running[index] = Some(Batch {
                busy_until_ms,
                parts,
            });
        }
        Ok(())
    }

    /// CPU preprocessing and encoder forward of one batch holding `images` images.
    fn batch_busy_ms(&self, images: u32) -> Result<f64> {
        let forward_ms = self
            .timing
            .predict_vision_ms(&[VisionShape {
                encoder: self.spec.shape,
                count: images,
            }])?
            .ok_or_else(|| {
                anyhow!(
                    "the encoder pool's timing model prices no vision batches; \
                     use an AIC timing model compiled with encoder_parallel"
                )
            })?;
        Ok(f64::from(images) * self.spec.preprocess_ms_per_image + forward_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EncoderShape, TimingModelConfig};

    /// Prices a forward at `per_image` milliseconds per image, or not at all.
    struct VisionTiming(Option<f64>);

    impl TimingModel for VisionTiming {
        fn predict_vision_ms(&self, shapes: &[VisionShape]) -> Result<Option<f64>> {
            Ok(self.0.map(|per_image| {
                shapes
                    .iter()
                    .map(|shape| f64::from(shape.count) * per_image)
                    .sum()
            }))
        }

        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> Result<f64> {
            Ok(0.0)
        }

        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> Result<f64> {
            Ok(0.0)
        }
    }

    fn pool(vision_ms_per_image: Option<f64>) -> EncoderPool<u32> {
        EncoderPool::new(
            EncoderSpec {
                instances: 2,
                max_batch: 2,
                gpus_per_instance: 1,
                images_per_request: 3,
                shape: EncoderShape {
                    sequences: 1,
                    patch_tokens: 4,
                    transformer_tokens: 4,
                    output_tokens: 1,
                },
                preprocess_ms_per_image: 1.0,
                // 1 MB per image at 1 GB/s: 1 ms on the wire per image.
                transfer_bytes_per_image: 1_000_000,
                transfer_bandwidth_gb_s: 1.0,
                timing_model: None::<TimingModelConfig>,
            },
            Arc::new(VisionTiming(vision_ms_per_image)),
        )
    }

    #[test]
    fn parts_spread_over_the_instances_and_join_on_the_last_arrival() {
        let mut pool = pool(Some(10.0));
        pool.submit(1);
        pool.submit(2);
        assert_eq!(pool.parked_count(), 2);
        assert_eq!(pool.next_deadline_ms(), None);
        // Request 1 puts 2 images on instance 0 and 1 on instance 1; request 2
        // rotates: 2 on instance 1 and 1 on instance 0. Each instance batches its
        // two parts: 3 images cost 3 x 1 + 3 x 10 = 33 ms on both.
        assert!(pool.take_ready(0.0).unwrap().is_empty());
        assert_eq!(pool.next_deadline_ms(), Some(33.0));
        assert!(pool.take_ready(33.0).unwrap().is_empty());
        // Transfers run per part: the 1-image parts land at 34, the 2-image parts at 35.
        assert_eq!(pool.next_deadline_ms(), Some(34.0));
        assert!(pool.take_ready(34.0).unwrap().is_empty());
        assert_eq!(pool.take_ready(35.0).unwrap(), vec![(1, 35.0), (2, 35.0)]);
        assert_eq!(pool.parked_count(), 0);
        assert_eq!(pool.next_deadline_ms(), None);
    }

    #[test]
    fn a_freed_instance_takes_the_next_batch_at_its_own_end() {
        let mut pool = pool(Some(10.0));
        for id in 1..=3 {
            pool.submit(id);
        }
        // Instance 0 runs {2 of #1, 1 of #2} until 33, then {2 of #3} until 33 + 2 + 20 = 55,
        // even when the pool is only asked at 60; its last part lands at 57.
        assert!(pool.take_ready(0.0).unwrap().is_empty());
        let delivered = pool.take_ready(60.0).unwrap();
        assert_eq!(
            delivered.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(delivered[2].1, 57.0);
    }

    #[test]
    fn a_timing_model_without_vision_prices_is_rejected() {
        let mut pool = pool(None);
        pool.submit(1);
        assert!(pool.take_ready(0.0).is_err());
    }
}
