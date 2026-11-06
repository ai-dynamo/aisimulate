// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend measurements from actual Engine effects, using AISim's owning fold.
//!
//! Application callbacks and the native phase report remain separate evidence.
//! Neither supplies replacement token times, admission facts, or resource totals.

use aisimulate_core::{
    engine::{PassCompletionEffects, PassStartEffects},
    replay::{ReplayReport, ReplayRequestPool, ReplayTerminalStatus, TraceCollector},
};
use uuid::Uuid;

/// Single aggregated deployment's collector, driven on the model's f64 clock.
pub(crate) struct BackendReporting {
    collector: TraceCollector,
    decode_workers: usize,
    now_ms: f64,
}

impl BackendReporting {
    /// Resource identity comes from the actual factory/deployment, not requests.
    pub(crate) fn new(decode_workers: usize, decode_gpus_per_worker: usize) -> Self {
        let mut collector = TraceCollector::default();
        collector.set_capture_per_request(true);
        collector.set_gpus_per_worker(0, decode_gpus_per_worker);
        Self {
            collector,
            decode_workers,
            now_ms: 0.0,
        }
    }

    /// Accrue provisioned resources at each real model-clock advance.
    /// Preserve the replay runtime's multiply/divide/add order, including idle gaps.
    pub(crate) fn advance_to(&mut self, at_ms: f64) {
        if at_ms > self.now_ms {
            let dt_ms = at_ms - self.now_ms;
            let decode_worker_seconds = self.decode_workers as f64 * dt_ms / 1000.0;
            self.collector.add_worker_seconds(0.0, decode_worker_seconds);
            self.now_ms = at_ms;
        }
    }

    /// Record the request actually submitted to the Engine, before admission.
    pub(crate) fn on_submit(
        &mut self,
        request_id: Uuid,
        at_ms: f64,
        input_tokens: usize,
        output_tokens: usize,
    ) {
        self.advance_to(at_ms);
        self.collector
            .on_arrival(request_id, at_ms, input_tokens, output_tokens);
        // This composition directly submits to one aggregated Engine: there is
        // no router queue, replica choice, or cross-rank placement operation.
        self.collector.on_route_immediate(
            request_id, ReplayRequestPool::Agg, 0, 0, 0, 0, None, None,
        );
        self.collector.on_decode_assigned(request_id, 0);
    }

    /// Record scheduler admissions, never infer them from selected request IDs.
    pub(crate) fn on_pass_started(&mut self, at_ms: f64, effects: &PassStartEffects) {
        self.advance_to(at_ms);
        for admission in &effects.admissions {
            self.collector.on_admit(
                admission.request_id,
                at_ms,
                admission.reused_input_tokens,
            );
            self.collector.on_pool_admission_with_tier_attribution(
                admission.request_id, ReplayRequestPool::Agg, at_ms,
                admission.reused_input_tokens, None,
            );
        }
    }

    /// Fold actual outputs in Engine order before any application delivery.
    pub(crate) fn on_pass_completed(&mut self, at_ms: f64, effects: &PassCompletionEffects) {
        self.advance_to(at_ms);
        self.collector
            .on_completed_prefill_work(effects.forward_pass_metrics.sum_prefill_tokens);
        for output in &effects.outputs {
            if output.token_id.is_some() {
                self.collector.on_token(output.request_id, at_ms);
            }
            if output.completed {
                let status = if output.rejected {
                    ReplayTerminalStatus::Rejected
                } else {
                    ReplayTerminalStatus::Completed
                };
                self.collector.on_terminal(output.request_id, at_ms, status);
            }
        }
    }

    /// Record a cancellation only when the Engine confirms ownership retired.
    pub(crate) fn on_cancel(&mut self, request_id: Uuid, at_ms: f64) {
        self.advance_to(at_ms);
        self.collector
            .on_terminal(request_id, at_ms, ReplayTerminalStatus::Canceled);
    }

    /// Drain using the same epoch boundary and report arithmetic as SteppableAgg.
    pub(crate) fn take_report(&mut self, end_ms: f64) -> ReplayReport {
        self.advance_to(end_ms);
        // The owning offline host stamps processing rates with elapsed virtual
        // execution time, not CPU wall time. This binding begins at time zero.
        self.collector.take_report(end_ms).with_wall_time_ms(end_ms)
    }
}
