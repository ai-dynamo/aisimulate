// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional, policy-neutral telemetry sampling for offline replay.
//!
//! Telemetry is scheduled on the replay virtual clock independently of scaling
//! policy ticks. Point-in-time scheduler rows are kept separate from additive
//! interval counters so a sample remains meaningful when a worker rank retires
//! between two samples.

use serde::Serialize;

/// Why a replay telemetry sample was emitted.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayTelemetrySampleKind {
    /// Gauge-only state after the initial replay timestamp has settled.
    /// Interval counters are not consumed by this sample.
    Baseline,
    /// A complete configured sampling interval.
    Periodic,
    /// The non-empty tail after the last periodic sample, including a
    /// zero-duration tail with observations recorded at the final timestamp.
    Final,
}

/// Point-in-time scheduler state for one live logical-worker rank.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplaySchedulerMetricsSnapshot {
    pub worker_id: usize,
    pub dp_rank: u32,
    /// Backend-native legacy occupancy. vLLM counts active references; SGLang
    /// counts occupied page-pool blocks, including radix-resident pages.
    pub active_blocks: u64,
    /// Reusable resident blocks excluded from `active_blocks` (vLLM only;
    /// SGLang reports zero because its legacy occupancy already includes them).
    pub inactive_blocks: u64,
    pub total_blocks: u64,
    /// Legacy/backend-native `active_blocks / total_blocks` utilization.
    pub active_cache_usage: f64,
    /// Physical resident utilization; equal to active utilization for SGLang.
    pub physical_cache_usage: f64,
    pub running_requests: u64,
    pub waiting_requests: u64,
}

/// Additive scheduler observations over one telemetry interval for one role.
///
/// These counters include observations from ranks that retired during the
/// interval; retired rank gauge rows are intentionally not retained.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize)]
pub struct ReplaySchedulerIntervalMetrics {
    /// SGLang scheduler cache-hit tokens observed in the interval.
    /// Backends without an equivalent pass-local metric report zero.
    pub cache_hit_tokens: u64,
    /// SGLang scheduler tokens considered in the interval. A zero denominator
    /// means scheduler reuse is unavailable, not a measured zero-percent rate.
    pub cache_total_tokens: u64,
    /// New scheduler preemptions observed in the interval.
    pub preemptions: u64,
}

impl ReplaySchedulerIntervalMetrics {
    pub(crate) const fn has_observations(&self) -> bool {
        self.cache_hit_tokens != 0 || self.cache_total_tokens != 0 || self.preemptions != 0
    }

    pub(crate) fn checked_add_assign(&mut self, other: Self) -> anyhow::Result<()> {
        self.cache_hit_tokens = self
            .cache_hit_tokens
            .checked_add(other.cache_hit_tokens)
            .ok_or_else(|| anyhow::anyhow!("scheduler cache-hit token counter overflow"))?;
        self.cache_total_tokens = self
            .cache_total_tokens
            .checked_add(other.cache_total_tokens)
            .ok_or_else(|| anyhow::anyhow!("scheduler cache-total token counter overflow"))?;
        self.preemptions = self
            .preemptions
            .checked_add(other.preemptions)
            .ok_or_else(|| anyhow::anyhow!("scheduler preemption counter overflow"))?;
        Ok(())
    }
}

/// Traffic observations over one telemetry interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ReplayTrafficMetricsSnapshot {
    pub duration_s: f64,
    /// Requests arriving at the replay admission boundary in this interval.
    pub arriving_requests: usize,
    /// Completed, non-rejected requests contributing shape observations.
    pub completed_requests: usize,
    /// Mean input/output lengths over `completed_requests`.
    pub avg_isl: f64,
    pub avg_osl: f64,
    pub avg_ttft_ms: f64,
    pub avg_itl_ms: f64,
    pub ttft_count: usize,
    pub itl_count: usize,
    /// Mean router prefix-overlap ratio over `router_kv_hit_rate_count`.
    pub avg_router_kv_hit_rate: f64,
    pub router_kv_hit_rate_count: usize,
    pub avg_accept_length: Option<f64>,
    pub accept_length_forward_count: usize,
}

/// Policy-neutral replay state sampled after a virtual timestamp has settled.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayTelemetrySnapshot {
    pub sample_ordinal: u64,
    pub kind: ReplayTelemetrySampleKind,
    /// Start boundary of the interval counters represented by this sample.
    pub interval_start_ms: f64,
    /// Virtual timestamp at which this sample was taken.
    pub sampled_at_ms: f64,
    pub traffic: ReplayTrafficMetricsSnapshot,
    /// Full gauge rows for every currently live rank. Aggregated replay reports
    /// its single role through the decode fields.
    pub prefill_scheduler_metrics: Vec<ReplaySchedulerMetricsSnapshot>,
    pub decode_scheduler_metrics: Vec<ReplaySchedulerMetricsSnapshot>,
    pub prefill_interval_metrics: ReplaySchedulerIntervalMetrics,
    pub decode_interval_metrics: ReplaySchedulerIntervalMetrics,
    /// Requests admitted by Replay but still awaiting worker placement.
    pub router_pending_prefill_requests: usize,
    pub router_pending_decode_requests: usize,
    pub active_prefill_ids: Vec<usize>,
    pub active_decode_ids: Vec<usize>,
    pub starting_prefill_ids: Vec<usize>,
    pub starting_decode_ids: Vec<usize>,
    pub draining_prefill_ids: Vec<usize>,
    pub draining_decode_ids: Vec<usize>,
}

/// Receives optional replay telemetry without participating in scaling.
///
/// `Send` lets native bindings release their language-runtime lock while the
/// single-threaded replay loop invokes a native collector.
pub trait ReplayTelemetryObserver: Send {
    fn on_sample(&mut self, snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()>;
}

pub(crate) struct ReplayTelemetryRuntime {
    observer: Box<dyn ReplayTelemetryObserver>,
    sample_interval_ms: f64,
    next_sample_ordinal: u64,
    sampling_origin_ms: Option<f64>,
    interval_start_ms: f64,
}

impl ReplayTelemetryRuntime {
    pub(crate) fn new(sample_interval_ms: f64, observer: Box<dyn ReplayTelemetryObserver>) -> Self {
        Self {
            observer,
            sample_interval_ms,
            next_sample_ordinal: 0,
            sampling_origin_ms: None,
            interval_start_ms: 0.0,
        }
    }

    pub(crate) const fn next_sample_ordinal(&self) -> u64 {
        self.next_sample_ordinal
    }

    pub(crate) const fn interval_start_ms(&self) -> f64 {
        self.interval_start_ms
    }

    pub(crate) fn start_at(&mut self, now_ms: f64) {
        self.sampling_origin_ms = Some(now_ms);
        self.interval_start_ms = now_ms;
    }

    /// Derive cadence from the baseline origin and ordinal rather than adding
    /// repeatedly, which prevents floating-point drift over long replays.
    pub(crate) fn next_periodic_at_ms(&self) -> anyhow::Result<f64> {
        let origin_ms = self
            .sampling_origin_ms
            .ok_or_else(|| anyhow::anyhow!("replay telemetry baseline was not initialized"))?;
        let at_ms = origin_ms + self.next_sample_ordinal as f64 * self.sample_interval_ms;
        if !at_ms.is_finite() {
            return Err(anyhow::anyhow!(
                "replay telemetry sample timestamp overflow"
            ));
        }
        if at_ms <= self.interval_start_ms {
            return Err(anyhow::anyhow!(
                "replay telemetry cadence is below virtual-clock precision at {} ms",
                self.interval_start_ms
            ));
        }
        Ok(at_ms)
    }

    pub(crate) fn publish(&mut self, snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        self.observer.on_sample(snapshot)?;
        self.next_sample_ordinal = self
            .next_sample_ordinal
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("replay telemetry sample ordinal overflow"))?;
        Ok(())
    }

    pub(crate) fn close_interval(&mut self, sampled_at_ms: f64) {
        self.interval_start_ms = sampled_at_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopObserver;

    impl ReplayTelemetryObserver for NoopObserver {
        fn on_sample(&mut self, _snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn sample(
        sample_ordinal: u64,
        kind: ReplayTelemetrySampleKind,
        interval_start_ms: f64,
        sampled_at_ms: f64,
    ) -> ReplayTelemetrySnapshot {
        ReplayTelemetrySnapshot {
            sample_ordinal,
            kind,
            interval_start_ms,
            sampled_at_ms,
            traffic: ReplayTrafficMetricsSnapshot::default(),
            prefill_scheduler_metrics: Vec::new(),
            decode_scheduler_metrics: Vec::new(),
            prefill_interval_metrics: ReplaySchedulerIntervalMetrics::default(),
            decode_interval_metrics: ReplaySchedulerIntervalMetrics::default(),
            router_pending_prefill_requests: 0,
            router_pending_decode_requests: 0,
            active_prefill_ids: Vec::new(),
            active_decode_ids: Vec::new(),
            starting_prefill_ids: Vec::new(),
            starting_decode_ids: Vec::new(),
            draining_prefill_ids: Vec::new(),
            draining_decode_ids: Vec::new(),
        }
    }

    #[test]
    fn submillisecond_cadence_stays_anchored_to_the_sampling_origin() {
        let mut runtime = ReplayTelemetryRuntime::new(0.1, Box::new(NoopObserver));
        runtime.start_at(0.0);
        runtime
            .publish(sample(0, ReplayTelemetrySampleKind::Baseline, 0.0, 0.0))
            .unwrap();

        for ordinal in 1..=10_000 {
            let at_ms = runtime.next_periodic_at_ms().unwrap();
            assert_eq!(at_ms, ordinal as f64 * 0.1);
            runtime
                .publish(sample(
                    ordinal,
                    ReplayTelemetrySampleKind::Periodic,
                    runtime.interval_start_ms(),
                    at_ms,
                ))
                .unwrap();
            runtime.close_interval(at_ms);
        }
    }

    #[test]
    fn cadence_rejects_intervals_below_virtual_clock_precision() {
        let origin_ms = 1.0e20;
        let mut runtime = ReplayTelemetryRuntime::new(0.1, Box::new(NoopObserver));
        runtime.start_at(origin_ms);
        runtime
            .publish(sample(
                0,
                ReplayTelemetrySampleKind::Baseline,
                origin_ms,
                origin_ms,
            ))
            .unwrap();

        let error = runtime.next_periodic_at_ms().unwrap_err();
        assert!(error.to_string().contains("below virtual-clock precision"));
    }
}
