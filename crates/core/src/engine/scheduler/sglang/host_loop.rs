// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-thread timing that turns one AIS pass into one SGLang loop iteration.
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
//! drain, prepares them on the scheduler thread, selects and launches this
//! iteration's batch, and only then waits for the previous forward and processes
//! its result. The GPU is one stream: a forward starts when its launch begins and
//! the previous forward has finished, and it cannot end before its launch does.
//! Consequently the outputs, terminals, and KV release of batch `k` become visible
//! at the end of iteration `k+1`, and a request that finished in batch `k` still
//! fills a slot in batch `k+1` because SGLang's `filter_batch` only sees
//! `finished()` after that result is processed. The timing arithmetic below is
//! independently implemented; the cost coefficients come from a measured host
//! profile lowered by the Python configuration layer.

use std::collections::VecDeque;

use crate::engine::HostLoopConfig;
use crate::engine::common::protocols::OutputSignal;

use super::request::SglangRequest;

/// Batch launched by one iteration, as the scheduler thread charges it.
#[derive(Debug, Clone, Copy)]
pub(super) enum LaunchKind {
    /// An EXTEND forward over `tokens` newly computed prompt tokens.
    Extend { requests: usize, tokens: usize },
    /// A DECODE step over `requests` running sequences, ghosts included.
    Decode { requests: usize },
}

impl LaunchKind {
    pub(super) fn requests(self) -> usize {
        match self {
            Self::Extend { requests, .. } | Self::Decode { requests } => requests,
        }
    }

    fn tokens(self) -> usize {
        match self {
            Self::Extend { tokens, .. } => tokens,
            Self::Decode { requests } => requests,
        }
    }
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

/// A launched batch whose result the scheduler has not observed yet.
#[derive(Debug)]
struct InFlightBatch {
    gpu_end_ms: f64,
    requests: usize,
    /// Outputs the forward produced; released when the next iteration observes them.
    output_signals: Vec<OutputSignal>,
}

#[derive(Debug)]
pub(super) struct HostLoop {
    config: HostLoopConfig,
    /// Requests delivered to the scheduler and waiting for the next `recv_requests`.
    inbox: VecDeque<SglangRequest>,
    in_flight: Option<InFlightBatch>,
    gpu_free_ms: f64,
}

impl HostLoop {
    pub(super) fn new(config: HostLoopConfig) -> Self {
        Self {
            config,
            inbox: VecDeque::new(),
            in_flight: None,
            gpu_free_ms: 0.0,
        }
    }

    /// Whether the loop owns no delivered request and no unobserved batch.
    pub(super) fn is_idle(&self) -> bool {
        self.inbox.is_empty() && self.in_flight.is_none()
    }

    pub(super) fn holds_request(&self, uuid: uuid::Uuid) -> bool {
        self.inbox.iter().any(|request| request.uuid == uuid)
    }

    /// Deliver a request to the scheduler process. It is received at the next
    /// iteration start, never inside the running iteration.
    pub(super) fn submit(&mut self, request: SglangRequest) {
        self.inbox.push_back(request);
    }

    /// Remove a delivered request before the scheduler receives it.
    pub(super) fn take_request(&mut self, uuid: uuid::Uuid) -> Option<SglangRequest> {
        let index = self.inbox.iter().position(|request| request.uuid == uuid)?;
        self.inbox.remove(index)
    }

    /// Drain the whole inbox, mirroring the non-blocking drain at the top of the loop.
    pub(super) fn take_received(&mut self) -> Vec<SglangRequest> {
        self.inbox.drain(..).collect()
    }

    /// Scheduler-thread cost of preparing one received request.
    pub(super) fn receive_cost_ms(&self, request: &SglangRequest) -> f64 {
        let feature_bytes = request.images.iter().map(|image| image.feature_bytes).sum();
        self.config
            .receive
            .eval(1, request.images.len(), request.prompt_len(), feature_bytes)
    }

    /// Selection time of an iteration that starts at `start_ms` after charging
    /// `receive_ms` of request preparation and, when a batch forms, its selection.
    pub(super) fn selected_ms(
        &self,
        start_ms: f64,
        receive_ms: f64,
        batch: Option<LaunchKind>,
    ) -> f64 {
        let select_ms = batch.map_or(0.0, |batch| {
            self.config
                .select
                .eval(batch.requests(), 0, batch.tokens(), 0)
        });
        start_ms + receive_ms + self.config.tp_sync_ms + select_ms
    }

    /// Lay this iteration's launch, forward, and result observation on the
    /// scheduler and GPU timelines. `gpu_ms` is the forward's modeled device
    /// time and `output_signals` are the outputs it produces; they are held
    /// back and returned by the next call, which observes them. The returned
    /// signals are `None` when no batch was in flight.
    pub(super) fn plan(
        &mut self,
        selected_ms: f64,
        batch: Option<LaunchKind>,
        gpu_ms: f64,
        output_signals: Vec<OutputSignal>,
    ) -> (IterationTiming, Option<Vec<OutputSignal>>) {
        debug_assert!(
            batch.is_some() || (gpu_ms == 0.0 && output_signals.is_empty()),
            "an iteration without a batch launches no forward"
        );
        let previous_gpu_end_ms = self.in_flight.as_ref().map(|batch| batch.gpu_end_ms);
        let (launch_end_ms, gpu_end_ms) = match batch {
            Some(kind) => {
                let launch_start_ms = match kind {
                    LaunchKind::Decode { .. } if self.config.decode_launch_syncs_previous_gpu => {
                        previous_gpu_end_ms.map_or(selected_ms, |gpu_end| selected_ms.max(gpu_end))
                    }
                    _ => selected_ms,
                };
                let launch_cost = match kind {
                    LaunchKind::Extend { requests, tokens } => {
                        self.config.launch_extend.eval(requests, 0, tokens, 0)
                    }
                    LaunchKind::Decode { requests } => {
                        self.config.launch_decode.eval(requests, 0, requests, 0)
                    }
                };
                let launch_end_ms = launch_start_ms + launch_cost;
                // Eager launches enqueue asynchronously: the first kernel runs as soon
                // as the stream is free, and a launch-bound forward ends with its launch.
                let gpu_start_ms = self.gpu_free_ms.max(launch_start_ms);
                let gpu_end_ms = (gpu_start_ms + gpu_ms).max(launch_end_ms);
                self.gpu_free_ms = gpu_end_ms;
                (launch_end_ms, Some(gpu_end_ms))
            }
            None => (selected_ms, None),
        };
        let observed = self.in_flight.take();
        // `copy_done.synchronize()` on the previous batch is the only point where
        // the scheduler thread waits for the GPU.
        let end_ms = match &observed {
            Some(previous) => {
                launch_end_ms.max(previous.gpu_end_ms)
                    + self.config.result.eval(previous.requests, 0, 0, 0)
            }
            None => launch_end_ms,
        };
        if let (Some(kind), Some(gpu_end_ms)) = (batch, gpu_end_ms) {
            self.in_flight = Some(InFlightBatch {
                gpu_end_ms,
                requests: kind.requests(),
                output_signals,
            });
        }
        (
            IterationTiming {
                selected_ms,
                gpu_end_ms,
                end_ms,
            },
            observed.map(|previous| previous.output_signals),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CostFn;

    fn config() -> HostLoopConfig {
        HostLoopConfig {
            select: CostFn {
                const_ms: 1.0,
                ..CostFn::default()
            },
            launch_extend: CostFn {
                const_ms: 5.0,
                ..CostFn::default()
            },
            launch_decode: CostFn {
                const_ms: 3.0,
                ..CostFn::default()
            },
            result: CostFn {
                const_ms: 1.0,
                ..CostFn::default()
            },
            ..HostLoopConfig::default()
        }
    }

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

    #[test]
    fn forward_starts_at_launch_start_and_waits_for_the_previous_forward() {
        let mut host = HostLoop::new(config());
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 8,
        });
        let selected = host.selected_ms(0.0, 2.0, extend);
        assert_eq!(selected, 3.0);
        let (timing, observed) = host.plan(selected, extend, 20.0, vec![signal()]);
        assert!(observed.is_none());
        // No previous batch: the iteration ends with its launch while the GPU keeps running.
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 3.0,
                gpu_end_ms: Some(23.0),
                end_ms: 8.0
            }
        );

        let decode = Some(LaunchKind::Decode { requests: 1 });
        let selected = host.selected_ms(8.0, 0.0, decode);
        let (timing, observed) = host.plan(selected, decode, 4.0, Vec::new());
        assert_eq!(observed.map(|signals| signals.len()), Some(1));
        // The decode launch synchronizes with the previous forward, then the
        // result of that forward is processed before the loop returns.
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 9.0,
                gpu_end_ms: Some(27.0),
                end_ms: 27.0
            }
        );
    }

    #[test]
    fn launch_bound_forwards_end_with_their_launch() {
        let mut host = HostLoop::new(HostLoopConfig {
            launch_extend: CostFn {
                const_ms: 10.0,
                ..CostFn::default()
            },
            ..HostLoopConfig::default()
        });
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 4,
        });
        let (timing, _) = host.plan(0.0, extend, 1.0, Vec::new());
        assert_eq!(timing.gpu_end_ms, Some(10.0));
        assert_eq!(timing.end_ms, 10.0);
    }

    #[test]
    fn an_iteration_without_a_batch_only_observes_the_previous_forward() {
        let mut host = HostLoop::new(config());
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 4,
        });
        host.plan(0.0, extend, 30.0, Vec::new());
        let (timing, observed) = host.plan(12.0, None, 0.0, Vec::new());
        assert!(observed.is_some());
        assert_eq!(
            timing,
            IterationTiming {
                selected_ms: 12.0,
                gpu_end_ms: None,
                end_ms: 31.0
            }
        );
        assert!(host.is_idle());
    }
}
