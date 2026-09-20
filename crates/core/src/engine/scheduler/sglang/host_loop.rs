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
//! `finished()` after that result is processed. Everything a forward produces
//! for the scheduler's bookkeeping travels with the batch: its output tokens and
//! the prefix-cache commits of `maybe_cache_unfinished_req` become visible
//! together when the batch is observed, and a cancelled request takes its share
//! of that payload with it while the device work stays charged. A launch has two
//! parts: input preparation the first kernel waits for, and kernel enqueueing that
//! overlaps the forward. The timing arithmetic below is independently
//! implemented; the cost coefficients come from a measured host profile lowered
//! by the Python configuration layer.

use std::collections::VecDeque;

use uuid::Uuid;

use crate::engine::HostLoopConfig;
use crate::engine::common::protocols::OutputSignal;

use super::request::SglangRequest;

/// Cache-miss images an EXTEND batch encodes before its language-model forward.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct VisionWork {
    pub(super) images: usize,
    pub(super) visual_tokens: usize,
    pub(super) feature_bytes: u64,
}

/// Batch launched by one iteration, as the scheduler thread charges it.
#[derive(Debug, Clone, Copy)]
pub(super) enum LaunchKind {
    /// An EXTEND forward over `tokens` newly computed prompt tokens.
    Extend {
        requests: usize,
        tokens: usize,
        vision: VisionWork,
    },
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
    requests: usize,
    outputs: ForwardOutputs,
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

    /// Forget what the forward in flight would deliver for `uuid`. Its device
    /// work and result processing stay charged: the batch already ran with it.
    pub(super) fn discard_request(&mut self, uuid: Uuid) {
        if let Some(batch) = &mut self.in_flight {
            batch.outputs.discard_request(uuid);
        }
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
    /// time and `outputs` what it produces for the scheduler; they are held
    /// back and returned by the next call, which observes them. The returned
    /// outputs are `None` when no batch was in flight.
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
        let (launch_end_ms, gpu_end_ms) = match batch {
            Some(kind) => {
                let launch_start_ms = match kind {
                    LaunchKind::Decode { .. } if self.config.decode_launch_syncs_previous_gpu => {
                        previous_gpu_end_ms.map_or(selected_ms, |gpu_end| selected_ms.max(gpu_end))
                    }
                    _ => selected_ms,
                };
                let (prepare_cost, launch_cost) = match kind {
                    LaunchKind::Extend {
                        requests,
                        tokens,
                        vision,
                    } => {
                        let (prepare_vision, launch_vision) = if vision.images > 0 {
                            let eval = |cost: &crate::engine::CostFn| {
                                cost.eval(
                                    1,
                                    vision.images,
                                    vision.visual_tokens,
                                    vision.feature_bytes,
                                )
                            };
                            (
                                eval(&self.config.prepare_vision),
                                eval(&self.config.launch_vision),
                            )
                        } else {
                            (0.0, 0.0)
                        };
                        (
                            self.config.prepare_extend.eval(requests, 0, tokens, 0)
                                + prepare_vision,
                            self.config.launch_extend.eval(requests, 0, tokens, 0) + launch_vision,
                        )
                    }
                    LaunchKind::Decode { requests } => (
                        0.0,
                        self.config.launch_decode.eval(requests, 0, requests, 0),
                    ),
                };
                // Input preparation gates the first kernel. Eager launches then enqueue
                // asynchronously: the forward runs as soon as the stream is free, and a
                // launch-bound forward ends with its launch.
                let input_ready_ms = launch_start_ms + prepare_cost;
                let launch_end_ms = input_ready_ms + launch_cost;
                let gpu_start_ms = self.gpu_free_ms.max(input_ready_ms);
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

    fn outputs(signals: Vec<OutputSignal>) -> ForwardOutputs {
        ForwardOutputs {
            output_signals: signals,
            cache_commits: Vec::new(),
        }
    }

    #[test]
    fn forward_starts_at_launch_start_and_waits_for_the_previous_forward() {
        let mut host = HostLoop::new(config());
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 8,
            vision: VisionWork::default(),
        });
        let selected = host.selected_ms(0.0, 2.0, extend);
        assert_eq!(selected, 3.0);
        let (timing, observed) = host.plan(selected, extend, 20.0, outputs(vec![signal()]));
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
        let (timing, observed) = host.plan(selected, decode, 4.0, ForwardOutputs::default());
        assert_eq!(
            observed.map(|outputs| outputs.output_signals.len()),
            Some(1)
        );
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
            vision: VisionWork::default(),
        });
        let (timing, _) = host.plan(0.0, extend, 1.0, ForwardOutputs::default());
        assert_eq!(timing.gpu_end_ms, Some(10.0));
        assert_eq!(timing.end_ms, 10.0);
    }

    #[test]
    fn input_preparation_delays_the_forward_instead_of_overlapping_it() {
        let mut host = HostLoop::new(HostLoopConfig {
            prepare_extend: CostFn {
                const_ms: 100.0,
                ..CostFn::default()
            },
            ..HostLoopConfig::default()
        });
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 4,
            vision: VisionWork::default(),
        });
        // 100 ms of preparation the kernels depend on, then 10 ms of device work.
        let (timing, _) = host.plan(0.0, extend, 10.0, ForwardOutputs::default());
        assert_eq!(timing.gpu_end_ms, Some(110.0));
        assert_eq!(timing.end_ms, 100.0);
    }

    #[test]
    fn a_discarded_request_leaves_the_forward_in_flight_without_its_outputs() {
        let mut host = HostLoop::new(config());
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 4,
            vision: VisionWork::default(),
        });
        let cancelled = uuid::Uuid::from_u128(7);
        let mut cancelled_signal = signal();
        cancelled_signal.uuid = cancelled;
        host.plan(
            0.0,
            extend,
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
        // The result of the batch that ran with the request is still processed.
        assert_eq!(timing.end_ms, 21.0);
    }

    #[test]
    fn an_iteration_without_a_batch_only_observes_the_previous_forward() {
        let mut host = HostLoop::new(config());
        let extend = Some(LaunchKind::Extend {
            requests: 1,
            tokens: 4,
            vision: VisionWork::default(),
        });
        host.plan(0.0, extend, 30.0, ForwardOutputs::default());
        let (timing, observed) = host.plan(12.0, None, 0.0, ForwardOutputs::default());
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
