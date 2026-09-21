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
//! forward starts when its launch begins and the previous forward has finished.
//! A DECODE launch first waits for the previous forward, as SGLang's position
//! update copies a device tensor to the host before the decode kernels launch.
//! Consequently the outputs, terminals, and KV release of batch `k` become
//! visible at the end of iteration `k+1`, and a request that finished in batch
//! `k` still fills a slot in batch `k+1` because SGLang's `filter_batch` only
//! sees `finished()` after that result is processed. Everything a forward
//! produces for the scheduler's bookkeeping travels with the batch: its output
//! tokens and the prefix-cache commits of `maybe_cache_unfinished_req` become
//! visible together when the batch is observed, and a cancelled request takes
//! its share of that payload with it while the device work stays charged.
//!
//! The scheduler thread itself is modeled as free: request preparation is
//! priced in the frontend stages ahead of the inbox, and batch selection,
//! kernel launch, and result processing are not charged. With a free thread the
//! observation of a forward lands where that forward ends.

use std::collections::VecDeque;

use uuid::Uuid;

use crate::engine::common::protocols::OutputSignal;

use super::request::SglangRequest;

/// Batch launched by one iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LaunchKind {
    /// An EXTEND forward over newly computed prompt tokens.
    Extend,
    /// A DECODE step over the running sequences, ghosts included.
    Decode,
}

/// Scheduler-thread timeline of one iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct IterationTiming {
    /// Batch selection finished; admissions and the forward's modeled start are dated here.
    pub(super) selected_ms: f64,
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
    fn is_empty(&self) -> bool {
        self.output_signals.is_empty() && self.cache_commits.is_empty()
    }

    fn discard_request(&mut self, uuid: Uuid) {
        self.output_signals.retain(|signal| signal.uuid != uuid);
        self.cache_commits.retain(|(request, _)| *request != uuid);
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

    /// Forget what the forward in flight would deliver for `uuid`. Its device
    /// work stays charged: the batch already ran with it.
    pub(super) fn discard_request(&mut self, uuid: Uuid) {
        if let Some(batch) = &mut self.in_flight {
            batch.outputs.discard_request(uuid);
        }
    }

    /// Lay this iteration's forward and result observation on the GPU timeline.
    /// `gpu_ms` is the forward's modeled device time and `outputs` what it
    /// produces for the scheduler; they are held back and returned by the next
    /// call, which observes them. The returned outputs are `None` when no batch
    /// was in flight.
    pub(super) fn plan(
        &mut self,
        selected_ms: f64,
        batch: Option<LaunchKind>,
        gpu_ms: f64,
        outputs: ForwardOutputs,
    ) -> (IterationTiming, Option<ForwardOutputs>) {
        debug_assert!(
            batch.is_some() || (gpu_ms == 0.0 && outputs.is_empty()),
            "an iteration without a batch launches no forward"
        );
        let previous_gpu_end_ms = self.in_flight.as_ref().map(|batch| batch.gpu_end_ms);
        let (launch_ms, gpu_end_ms) = match batch {
            Some(kind) => {
                let launch_ms = match kind {
                    LaunchKind::Decode => {
                        previous_gpu_end_ms.map_or(selected_ms, |gpu_end| selected_ms.max(gpu_end))
                    }
                    LaunchKind::Extend => selected_ms,
                };
                let gpu_end_ms = self.gpu_free_ms.max(launch_ms) + gpu_ms;
                self.gpu_free_ms = gpu_end_ms;
                (launch_ms, Some(gpu_end_ms))
            }
            None => (selected_ms, None),
        };
        let observed = self.in_flight.take();
        // `copy_done.synchronize()` on the previous batch is the only point where
        // the scheduler thread waits for the GPU.
        let end_ms = match &observed {
            Some(previous) => launch_ms.max(previous.gpu_end_ms),
            None => launch_ms,
        };
        if let (Some(_), Some(gpu_end_ms)) = (batch, gpu_end_ms) {
            self.in_flight = Some(InFlightBatch {
                gpu_end_ms,
                outputs,
            });
        }
        (
            IterationTiming {
                selected_ms,
                gpu_end_ms,
                end_ms,
            },
            observed.map(|previous| previous.outputs),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal() -> OutputSignal {
        OutputSignal {
            uuid: uuid::Uuid::nil(),
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
        let (timing, observed) =
            host.plan(0.0, Some(LaunchKind::Extend), 20.0, outputs(vec![signal()]));
        assert!(observed.is_none());
        // No previous batch: the free loop returns at once while the GPU keeps running.
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 0.0,
                gpu_end_ms: Some(20.0),
                end_ms: 0.0
            }
        );

        // The decode launch synchronizes with the previous forward (20), the GPU
        // runs it 20..24, and the previous forward's result is observed at 20.
        let (timing, observed) = host.plan(
            0.0,
            Some(LaunchKind::Decode),
            4.0,
            ForwardOutputs::default(),
        );
        assert_eq!(
            observed.map(|outputs| outputs.output_signals.len()),
            Some(1)
        );
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 0.0,
                gpu_end_ms: Some(24.0),
                end_ms: 20.0
            }
        );
    }

    #[test]
    fn an_extend_launch_queues_behind_the_running_forward_without_waiting_for_it() {
        let mut host = HostLoop::new();
        host.plan(0.0, Some(LaunchKind::Extend), 20.0, outputs(vec![signal()]));
        // Launched at 5 while the first forward runs: the GPU serves it 20..30 and
        // the loop returns when the first forward is observed.
        let (timing, observed) = host.plan(
            5.0,
            Some(LaunchKind::Extend),
            10.0,
            ForwardOutputs::default(),
        );
        assert_eq!(
            observed.map(|outputs| outputs.output_signals.len()),
            Some(1)
        );
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 5.0,
                gpu_end_ms: Some(30.0),
                end_ms: 20.0
            }
        );
    }

    #[test]
    fn a_discarded_request_leaves_the_forward_in_flight_without_its_outputs() {
        let mut host = HostLoop::new();
        let cancelled = uuid::Uuid::from_u128(7);
        let mut cancelled_signal = signal();
        cancelled_signal.uuid = cancelled;
        host.plan(
            0.0,
            Some(LaunchKind::Extend),
            20.0,
            ForwardOutputs {
                output_signals: vec![cancelled_signal, signal()],
                cache_commits: vec![(cancelled, 4), (uuid::Uuid::nil(), 4)],
            },
        );
        host.discard_request(cancelled);
        let (timing, observed) = host.plan(5.0, None, 0.0, ForwardOutputs::default());
        let observed = observed.expect("the batch is still observed");
        assert_eq!(observed.output_signals.len(), 1);
        assert_eq!(observed.cache_commits, vec![(uuid::Uuid::nil(), 4)]);
        // The batch that ran with the request is still waited for.
        assert_eq!(timing.end_ms, 20.0);
    }

    #[test]
    fn an_iteration_without_a_batch_only_observes_the_previous_forward() {
        let mut host = HostLoop::new();
        host.plan(
            0.0,
            Some(LaunchKind::Extend),
            30.0,
            ForwardOutputs::default(),
        );
        let (timing, observed) = host.plan(12.0, None, 0.0, ForwardOutputs::default());
        assert!(observed.is_some());
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 12.0,
                gpu_end_ms: None,
                end_ms: 30.0
            }
        );
        assert!(host.is_idle());
    }
}
