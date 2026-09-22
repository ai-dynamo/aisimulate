// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Iteration structure that turns one AIS pass into one SGLang loop iteration.
//!
//! Behavioral model of the SGLang overlap scheduler loop (`Scheduler.event_loop_overlap`,
//! `Scheduler.run_batch`, `SchedulerRequestReceiver.recv_requests`,
//! `process_batch_result_prefill`) as of sgl-project/sglang v0.5.19 (`0bcd822`,
//! `python/sglang/srt/managers/scheduler.py`,
//! `python/sglang/srt/managers/scheduler_components/request_receiver.py`,
//! `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`).
//! Re-implemented in Rust from the observed semantics; no SGLang source is copied.
//!
//! Each iteration drains the requests that reached the scheduler since the last
//! drain, selects and launches this iteration's batch, and only then waits for
//! the previous forward and processes its result. The GPU is one stream: a
//! forward starts when it is launched and the previous forward has finished.
//! Consequently the outputs, terminals, and KV release of batch `k` become
//! visible at the end of iteration `k+1`, and a request that finished in batch
//! `k` still fills a slot in batch `k+1` because SGLang's `filter_batch` only
//! sees `finished()` after that result is processed. Everything a forward
//! produces for the scheduler's bookkeeping travels with the batch: its output
//! tokens and the prefix-cache commits of `maybe_cache_unfinished_req` become
//! visible together when the batch is observed, and a retracted or cancelled
//! request takes its share of that payload with it while the device work stays
//! charged.
//!
//! The scheduler thread itself is modeled as free: request preparation is
//! priced in the frontend stages ahead of the inbox, and batch selection,
//! kernel launch, and result processing are not charged. With a free thread a
//! DECODE launch, which synchronizes with the previous forward before its
//! kernels start, and an EXTEND launch, which does not, produce the same
//! timeline, and the observation of a forward lands where that forward ends.

use std::collections::VecDeque;

use uuid::Uuid;

use crate::engine::common::protocols::OutputSignal;

use super::request::SglangRequest;

/// Scheduler-thread timeline of one iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct IterationTiming {
    /// Modeled completion of this iteration's forward, if it launched one.
    pub(super) gpu_end_ms: Option<f64>,
    /// The scheduler thread returns to the top of its loop.
    pub(super) end_ms: f64,
}

/// What a forward produced for the scheduler, held back until it is observed.
#[derive(Debug, Default)]
pub(super) struct ForwardOutputs {
    pub(super) output_signals: Vec<OutputSignal>,
    /// Requests whose prefix the radix cache learns when this batch is observed,
    /// with the sequence length the forward materialized for them.
    pub(super) cache_commits: Vec<(Uuid, usize)>,
}

impl ForwardOutputs {
    /// Drop everything owed to `uuid`; returns how many output tokens that was.
    fn discard_request(&mut self, uuid: Uuid) -> usize {
        let before = self.output_signals.len();
        self.output_signals.retain(|signal| signal.uuid != uuid);
        self.cache_commits.retain(|(request, _)| *request != uuid);
        before - self.output_signals.len()
    }
}

/// A launched batch whose result the scheduler has not observed yet.
#[derive(Debug)]
struct InFlightBatch {
    gpu_end_ms: f64,
    outputs: ForwardOutputs,
}

#[derive(Debug, Default)]
pub(super) struct HostLoop {
    /// Requests delivered to the scheduler and waiting for the next `recv_requests`.
    inbox: VecDeque<SglangRequest>,
    in_flight: Option<InFlightBatch>,
    gpu_free_ms: f64,
}

impl HostLoop {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Whether the loop owns no delivered request and no unobserved batch.
    pub(super) fn is_idle(&self) -> bool {
        self.inbox.is_empty() && self.in_flight.is_none()
    }

    pub(super) fn holds_request(&self, uuid: Uuid) -> bool {
        self.inbox.iter().any(|request| request.uuid == uuid)
    }

    /// Deliver a request to the scheduler process. It is received at the next
    /// iteration start, never inside the running iteration.
    pub(super) fn submit(&mut self, request: SglangRequest) {
        self.inbox.push_back(request);
    }

    /// Remove a delivered request before the scheduler receives it.
    pub(super) fn take_request(&mut self, uuid: Uuid) -> Option<SglangRequest> {
        let index = self.inbox.iter().position(|request| request.uuid == uuid)?;
        self.inbox.remove(index)
    }

    /// Drain the whole inbox, mirroring the non-blocking drain at the top of the loop.
    pub(super) fn take_received(&mut self) -> Vec<SglangRequest> {
        self.inbox.drain(..).collect()
    }

    /// Forget what the forward in flight would deliver for `uuid`, returning the
    /// number of output tokens it had sampled for it. The device work stays
    /// charged: the batch already ran with the request.
    pub(super) fn discard_request(&mut self, uuid: Uuid) -> usize {
        self.in_flight
            .as_mut()
            .map_or(0, |batch| batch.outputs.discard_request(uuid))
    }

    /// Lay this iteration's forward and result observation on the GPU timeline.
    /// `forward` is the launched batch's modeled device time and what it
    /// produces for the scheduler; they are held back and returned by the next
    /// call, which observes them. The returned outputs are `None` when no batch
    /// was in flight.
    pub(super) fn plan(
        &mut self,
        selected_ms: f64,
        forward: Option<(f64, ForwardOutputs)>,
    ) -> (IterationTiming, Option<ForwardOutputs>) {
        let gpu_end_ms = forward
            .as_ref()
            .map(|(gpu_ms, _)| self.gpu_free_ms.max(selected_ms) + gpu_ms);
        if let Some(gpu_end_ms) = gpu_end_ms {
            self.gpu_free_ms = gpu_end_ms;
        }
        let observed = self.in_flight.take();
        // `copy_done.synchronize()` on the previous batch is the only point where
        // the scheduler thread waits for the GPU.
        let end_ms = observed
            .as_ref()
            .map_or(selected_ms, |previous| selected_ms.max(previous.gpu_end_ms));
        if let (Some((_, outputs)), Some(gpu_end_ms)) = (forward, gpu_end_ms) {
            self.in_flight = Some(InFlightBatch {
                gpu_end_ms,
                outputs,
            });
        }
        (
            IterationTiming { gpu_end_ms, end_ms },
            observed.map(|previous| previous.outputs),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(uuid: Uuid) -> OutputSignal {
        OutputSignal {
            uuid,
            token_id: Some(1),
            completed: false,
            rejected: false,
            cached_tokens: None,
            handoff_delay_ms: None,
        }
    }

    fn outputs(signals: Vec<OutputSignal>) -> ForwardOutputs {
        ForwardOutputs {
            output_signals: signals,
            cache_commits: Vec::new(),
        }
    }

    #[test]
    fn a_forward_is_observed_when_the_next_iteration_synchronizes_with_it() {
        let mut host = HostLoop::new();
        let (timing, observed) = host.plan(0.0, Some((20.0, outputs(vec![signal(Uuid::nil())]))));
        assert!(observed.is_none());
        // No previous batch: the free loop returns at once while the GPU keeps running.
        assert_eq!(
            timing,
            IterationTiming {
                gpu_end_ms: Some(20.0),
                end_ms: 0.0
            }
        );

        // Launched at 5 while the first forward runs: the GPU serves it 20..30 and
        // the loop returns when the first forward is observed, at 20.
        let (timing, observed) = host.plan(5.0, Some((10.0, ForwardOutputs::default())));
        assert_eq!(
            observed.map(|outputs| outputs.output_signals.len()),
            Some(1)
        );
        assert_eq!(
            timing,
            IterationTiming {
                gpu_end_ms: Some(30.0),
                end_ms: 20.0
            }
        );
    }

    #[test]
    fn a_discarded_request_leaves_the_forward_in_flight_without_its_outputs() {
        let mut host = HostLoop::new();
        let cancelled = Uuid::from_u128(7);
        host.plan(
            0.0,
            Some((
                20.0,
                ForwardOutputs {
                    output_signals: vec![signal(cancelled), signal(Uuid::nil())],
                    cache_commits: vec![(cancelled, 4), (Uuid::nil(), 4)],
                },
            )),
        );
        assert_eq!(host.discard_request(cancelled), 1);
        let (timing, observed) = host.plan(5.0, None);
        let observed = observed.expect("the batch is still observed");
        assert_eq!(observed.output_signals.len(), 1);
        assert_eq!(observed.cache_commits, vec![(Uuid::nil(), 4)]);
        // The batch that ran with the request is still waited for.
        assert_eq!(timing.end_ms, 20.0);
    }

    #[test]
    fn an_iteration_without_a_batch_only_observes_the_previous_forward() {
        let mut host = HostLoop::new();
        host.plan(0.0, Some((30.0, ForwardOutputs::default())));
        let (timing, observed) = host.plan(12.0, None);
        assert!(observed.is_some());
        assert_eq!(
            timing,
            IterationTiming {
                gpu_end_ms: None,
                end_ms: 30.0
            }
        );
        assert!(host.is_idle());
    }
}
