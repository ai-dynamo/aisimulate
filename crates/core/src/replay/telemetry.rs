// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional, policy-neutral telemetry sampling for offline replay.
//!
//! Telemetry is scheduled on the replay virtual clock independently of scaling
//! policy ticks. Point-in-time scheduler rows are kept separate from additive
//! interval counters so a sample remains meaningful when a worker rank retires
//! between two samples.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Serialize, Serializer};
use std::sync::{Arc, Mutex};

use super::AgenticRuntimeIdentity;
use super::loadgen::{
    AgenticLifecycleEvent, AgenticLifecycleEventKind, AgenticLifecycleTranscript,
    AgenticPhaseEvidence, AgenticReplayPhase,
};
use super::{PerRequestRecord, ReplayReport};

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
    /// Backend-native occupancy. vLLM counts active references (physical pages
    /// across all groups for grouped caches); SGLang counts occupied page-pool
    /// blocks, including radix-resident pages.
    pub active_blocks: u64,
    /// Reusable resident blocks excluded from `active_blocks` (vLLM only;
    /// SGLang reports zero because its legacy occupancy already includes them).
    pub inactive_blocks: u64,
    pub total_blocks: u64,
    /// Shared physical grouped-cache occupancy and capacity. Linear caches
    /// retain the legacy block metrics and omit these fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_capacity_bytes: Option<u64>,
    /// Legacy/backend-native block utilization, or physical byte utilization
    /// for grouped caches (which have no scalar total block capacity).
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
    /// Actual native G1 lookup calls, including retries that do not admit work.
    pub native_prefix_cache_lookup: Option<crate::engine::PrefixCacheLookupCounts>,
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
    pub(crate) fn has_observations(&self) -> bool {
        self.cache_hit_tokens != 0
            || self.cache_total_tokens != 0
            || self.preemptions != 0
            || self
                .native_prefix_cache_lookup
                .is_some_and(|counts| counts.query_count != 0)
    }

    pub(crate) fn checked_add_assign(&mut self, other: Self) -> anyhow::Result<()> {
        if let Some(counts) = other.native_prefix_cache_lookup {
            self.native_prefix_cache_lookup
                .get_or_insert_with(Default::default)
                .checked_add_assign(counts)?;
        }
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
    /// Client-visible tokens delivered in this interval, after output limits.
    pub delivered_output_tokens: u64,
    /// Prompt tokens in completed native passes, including recomputation.
    pub committed_prefill_tokens: u64,
    /// Prompt-source accounting at first scheduler admission only.
    pub first_admission: ReplayPromptSourceCounters,
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
    /// Accepted tokens including the base token per decode forward, before
    /// output-limit truncation. Prefill is excluded; non-speculative decode is 1.
    pub avg_accept_length: Option<f64>,
    pub accept_length_forward_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ReplayPromptSourceCounters {
    pub requests: u64,
    pub input_tokens: u64,
    pub hbm_reused_tokens: u64,
    pub host_reused_tokens: u64,
    pub uncached_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayTelemetryHostCache {
    /// None identifies the one deployment-shared pool.
    pub role: Option<&'static str>,
    #[serde(flatten)]
    pub metrics: crate::engine::HostCacheMetrics,
}

/// Policy-neutral replay state sampled after a virtual timestamp has settled.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayTelemetrySnapshot {
    pub host_caches: Vec<ReplayTelemetryHostCache>,
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

/// Detailed capture uses one absolute virtual clock across preparation/profile.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayTelemetryCapture {
    pub schema: &'static str,
    pub clock: &'static str,
    pub clock_origin_ms: f64,
    pub units: ReplayTelemetryUnits,
    pub completion_status: &'static str,
    pub phase_boundaries: Vec<ReplayTelemetryPhaseBoundary>,
    pub sample_interval_ms: f64,
    pub profile_start_ms: Option<f64>,
    pub requests: Vec<ReplayTelemetryRequest>,
    pub samples: Vec<ReplayTelemetrySample>,
    pub lifecycle: Vec<ReplayTelemetryLifecycle>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayTelemetryUnits {
    pub timestamps: &'static str,
    pub durations: &'static str,
    pub traffic_duration: &'static str,
    pub tokens: &'static str,
    pub cache_blocks: &'static str,
    pub cache_bytes: &'static str,
    pub cache_usage: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayTelemetryPhaseBoundary {
    pub phase: &'static str,
    pub start_ms: f64,
    pub end_ms: f64,
    pub completion_status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayTelemetryRequest {
    pub phase: AgenticReplayPhase,
    #[serde(flatten)]
    pub request: PerRequestRecord,
}

#[derive(Debug, Clone)]
pub struct ReplayTelemetrySample {
    pub phase: &'static str,
    pub sample: ReplayTelemetrySnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayTelemetryLifecycle {
    pub phase: AgenticReplayPhase,
    pub request_uuid: Option<String>,
    pub agentic: Option<AgenticRuntimeIdentity>,
    #[serde(flatten)]
    pub event: AgenticLifecycleEvent,
}

#[derive(Clone)]
pub(crate) struct ReplayTelemetryCaptureSink(Arc<Mutex<(ReplayTelemetryCapture, &'static str)>>);

impl ReplayTelemetryCaptureSink {
    pub(crate) fn new(sample_interval_ms: f64) -> Self {
        Self(Arc::new(Mutex::new((
            ReplayTelemetryCapture {
                schema: "aisimulate.replay.telemetry.v1",
                clock: "absolute_virtual_ms",
                clock_origin_ms: 0.0,
                units: ReplayTelemetryUnits {
                    timestamps: "ms",
                    durations: "ms",
                    traffic_duration: "s",
                    tokens: "tokens",
                    cache_blocks: "blocks",
                    cache_bytes: "bytes",
                    cache_usage: "ratio",
                },
                completion_status: "running",
                phase_boundaries: Vec::new(),
                sample_interval_ms,
                profile_start_ms: None,
                requests: Vec::new(),
                samples: Vec::new(),
                lifecycle: Vec::new(),
            },
            "profile",
        ))))
    }

    pub(crate) fn set_preparing(&self, preparing: bool) {
        self.0.lock().expect("telemetry capture lock").1 =
            if preparing { "preparation" } else { "profile" };
    }

    pub(crate) fn mark_time_limited(&self) {
        self.0
            .lock()
            .expect("telemetry capture lock")
            .0
            .completion_status = "time_limited";
    }

    /// Live warmup drives the actual graph. Preserve its complete transcript
    /// before the driver clears it at the measurement barrier.
    pub(crate) fn record_preparation_lifecycle(&self, transcript: AgenticLifecycleTranscript) {
        let mut state = self.0.lock().expect("telemetry capture lock");
        state
            .0
            .lifecycle
            .extend(
                transcript
                    .events
                    .into_iter()
                    .map(|event| ReplayTelemetryLifecycle {
                        phase: AgenticReplayPhase::Warmup,
                        request_uuid: None,
                        agentic: None,
                        event,
                    }),
            );
    }

    pub(crate) fn record_preparation(&self, report: ReplayReport, phases: &AgenticPhaseEvidence) {
        let mut state = self.0.lock().expect("telemetry capture lock");
        let by_uuid: FxHashMap<_, _> = phases
            .requests
            .iter()
            .map(|phase| (phase.uuid.to_string(), phase))
            .collect();
        for mut request in report.per_request {
            if let Some(phase) = by_uuid.get(&request.uuid) {
                request.agentic_phase = Some(phase.phase);
                state.0.requests.push(ReplayTelemetryRequest {
                    phase: phase.phase,
                    request,
                });
            }
        }
        // Runtime identities distinguish repeated-prefix warmups even when
        // their source request is the same. Source aliases are used only for
        // live warmup, whose driver retains the original graph request IDs.
        let mut by_identity = FxHashMap::default();
        for request in &phases.requests {
            by_identity.insert(
                (
                    request.identity.play_id.as_str(),
                    request.identity.request_id.as_str(),
                ),
                request,
            );
            if request.phase == AgenticReplayPhase::Warmup {
                by_identity.insert(
                    (
                        request.identity.play_id.as_str(),
                        request.source_request_id.as_str(),
                    ),
                    request,
                );
            }
        }
        for row in &mut state.0.lifecycle {
            if let Some(request) = row
                .event
                .request_id
                .as_deref()
                .and_then(|id| by_identity.get(&(row.event.play_id.as_str(), id)))
            {
                row.phase = request.phase;
                row.request_uuid = Some(request.uuid.to_string());
                row.agentic = Some(request.identity.clone());
                row.event.request_id = Some(request.identity.request_id.clone());
            }
        }
        let mut observed: FxHashSet<_> = state.0.lifecycle.iter().map(lifecycle_key).collect();
        // Repeated-prefix preparation runs outside the graph driver. Its
        // authoritative phase ledger supplies these request transitions.
        for request in &phases.requests {
            for (at_ms, event, status) in [
                (
                    request.dispatched_at_ms,
                    AgenticLifecycleEventKind::Dispatch,
                    None,
                ),
                (
                    request.causal_terminal_at_ms,
                    AgenticLifecycleEventKind::CausalTerminal,
                    request.terminal_status,
                ),
                (
                    request.quiescent_at_ms,
                    AgenticLifecycleEventKind::Quiescent,
                    None,
                ),
            ] {
                if let Some(at_ms) = at_ms {
                    let row = ReplayTelemetryLifecycle {
                        phase: request.phase,
                        request_uuid: Some(request.uuid.to_string()),
                        agentic: Some(request.identity.clone()),
                        event: AgenticLifecycleEvent {
                            schema: "aisimulate.agentic.lifecycle.v1",
                            ordinal: 0,
                            at_ms,
                            play_id: request.identity.play_id.clone(),
                            request_id: Some(request.identity.request_id.clone()),
                            event,
                            status,
                        },
                    };
                    if observed.insert(lifecycle_key(&row)) {
                        state.0.lifecycle.push(row);
                    }
                }
            }
        }
        state.0.profile_start_ms = phases.profile_start_ms;
        if phases.profile_start_ms.is_some() {
            state.1 = "profile";
        }
    }

    pub(crate) fn finish(
        self,
        report: &mut ReplayReport,
        retain_per_request: bool,
    ) -> ReplayTelemetryCapture {
        let mut state = self.0.lock().expect("telemetry capture lock");
        let origin = report
            .agentic_phases
            .as_ref()
            .and_then(|p| p.profile_start_ms)
            .unwrap_or(0.0);
        let profile_started = report.agentic_phases.is_none()
            || report
                .agentic_phases
                .as_ref()
                .is_some_and(|p| p.profile_start_ms.is_some());
        if report.agentic_phases.is_none() {
            state.0.profile_start_ms = Some(0.0);
        }
        if profile_started {
            let mut identities = FxHashMap::default();
            let profile_requests = if retain_per_request {
                report.per_request.clone()
            } else {
                std::mem::take(&mut report.per_request)
            };
            for mut request in profile_requests {
                request.shift_timestamps(origin);
                request.agentic_phase = Some(AgenticReplayPhase::Profile);
                if let Some(identity) = &request.agentic {
                    identities.insert(
                        (identity.play_id.clone(), identity.request_id.clone()),
                        (request.uuid.clone(), identity.clone()),
                    );
                    if let Some(id) = &request.request_id {
                        identities.insert(
                            (identity.play_id.clone(), id.clone()),
                            (request.uuid.clone(), identity.clone()),
                        );
                    }
                }
                state.0.requests.push(ReplayTelemetryRequest {
                    phase: AgenticReplayPhase::Profile,
                    request,
                });
            }
            if let Some(transcript) = &report.agentic_lifecycle {
                state
                    .0
                    .lifecycle
                    .extend(transcript.events.iter().cloned().map(|event| {
                        let identity = event
                            .request_id
                            .as_ref()
                            .and_then(|id| identities.get(&(event.play_id.clone(), id.clone())));
                        ReplayTelemetryLifecycle {
                            phase: AgenticReplayPhase::Profile,
                            request_uuid: identity.map(|(uuid, _)| uuid.clone()),
                            agentic: identity.map(|(_, identity)| identity.clone()),
                            event,
                        }
                    }));
            }
        }
        let end_ms = state
            .0
            .samples
            .last()
            .map_or(origin + report.throughput.duration_ms, |sample| {
                sample.sample.sampled_at_ms
            });
        state.0.completion_status = if !profile_started {
            "preparation_aborted"
        } else if state.0.completion_status == "time_limited" {
            "time_limited"
        } else if report
            .agentic_profile
            .as_ref()
            .is_some_and(|p| p.server_unsettled_requests > 0)
        {
            "server_unsettled"
        } else {
            "completed"
        };
        if let Some(phases) = &report.agentic_phases {
            state.0.phase_boundaries.push(ReplayTelemetryPhaseBoundary {
                phase: "preparation",
                start_ms: 0.0,
                end_ms: phases
                    .profile_start_ms
                    .or(phases.finished_at_ms)
                    .unwrap_or(end_ms),
                completion_status: if profile_started {
                    "completed"
                } else {
                    "aborted"
                },
            });
        }
        if profile_started {
            let completion_status = state.0.completion_status;
            state.0.phase_boundaries.push(ReplayTelemetryPhaseBoundary {
                phase: "profile",
                start_ms: origin,
                end_ms,
                completion_status,
            });
        }
        state.0.requests.sort_by(|a, b| {
            a.request
                .arrival_time_ms
                .total_cmp(&b.request.arrival_time_ms)
                .then_with(|| a.request.uuid.cmp(&b.request.uuid))
        });
        state
            .0
            .lifecycle
            .sort_by(|a, b| a.event.at_ms.total_cmp(&b.event.at_ms));
        for (ordinal, row) in state.0.lifecycle.iter_mut().enumerate() {
            row.event.ordinal = ordinal as u64;
        }
        // The sink is shared with the runtime observer; move the arrays out
        // without cloning a potentially large all-phase transcript.
        let mut capture = state.0.clone_metadata();
        capture.requests = std::mem::take(&mut state.0.requests);
        capture.samples = std::mem::take(&mut state.0.samples);
        capture.lifecycle = std::mem::take(&mut state.0.lifecycle);
        capture
    }
}

impl ReplayTelemetryCapture {
    fn clone_metadata(&self) -> Self {
        Self {
            schema: self.schema,
            clock: self.clock,
            clock_origin_ms: self.clock_origin_ms,
            units: self.units.clone(),
            completion_status: self.completion_status,
            phase_boundaries: self.phase_boundaries.clone(),
            sample_interval_ms: self.sample_interval_ms,
            profile_start_ms: self.profile_start_ms,
            requests: Vec::new(),
            samples: Vec::new(),
            lifecycle: Vec::new(),
        }
    }
}

fn lifecycle_key(row: &ReplayTelemetryLifecycle) -> (String, Option<String>, u8, u64) {
    (
        row.event.play_id.clone(),
        row.event.request_id.clone(),
        row.event.event as u8,
        row.event.at_ms.to_bits(),
    )
}

// The observer API retains its numeric fields. The exported capture explicitly
// represents absent interval means, cache observations and scalar capacities
// as null. A measured cold miss still has zero hits and a positive denominator.
impl Serialize for ReplayTelemetrySample {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.sample).map_err(serde::ser::Error::custom)?;
        value["phase"] = self.phase.into();
        let traffic = &self.sample.traffic;
        for (name, count) in [
            ("avg_isl", traffic.completed_requests),
            ("avg_osl", traffic.completed_requests),
            ("avg_ttft_ms", traffic.ttft_count),
            ("avg_itl_ms", traffic.itl_count),
            ("avg_router_kv_hit_rate", traffic.router_kv_hit_rate_count),
        ] {
            if count == 0 {
                value["traffic"][name] = serde_json::Value::Null;
            }
        }
        for (name, metrics) in [
            (
                "prefill_interval_metrics",
                &self.sample.prefill_interval_metrics,
            ),
            (
                "decode_interval_metrics",
                &self.sample.decode_interval_metrics,
            ),
        ] {
            // SGLang's denominator includes both reused and executed prompt
            // tokens. Zero means no pass-local observation in this interval,
            // including every interval on backends without this metric.
            if metrics.cache_total_tokens == 0 {
                value[name]["cache_hit_tokens"] = serde_json::Value::Null;
                value[name]["cache_total_tokens"] = serde_json::Value::Null;
            }
        }
        for (name, rows) in [
            (
                "prefill_scheduler_metrics",
                &self.sample.prefill_scheduler_metrics,
            ),
            (
                "decode_scheduler_metrics",
                &self.sample.decode_scheduler_metrics,
            ),
        ] {
            for (index, row) in rows.iter().enumerate() {
                if row.kv_cache_capacity_bytes.is_some() {
                    value[name][index]["total_blocks"] = serde_json::Value::Null;
                }
            }
        }
        value.serialize(serializer)
    }
}

impl ReplayTelemetryObserver for ReplayTelemetryCaptureSink {
    fn on_sample(&mut self, sample: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("telemetry capture lock poisoned"))?;
        let phase = state.1;
        state
            .0
            .samples
            .push(ReplayTelemetrySample { phase, sample });
        Ok(())
    }
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
    next_periodic_ordinal: u64,
    delivered_output_tokens: u64,
    pending_output_receipts_ms: Vec<f64>,
    committed_prefill_tokens: u64,
    first_admission: ReplayPromptSourceCounters,
    sampling_origin_ms: Option<f64>,
    interval_start_ms: f64,
}

impl ReplayTelemetryRuntime {
    pub(crate) fn new(sample_interval_ms: f64, observer: Box<dyn ReplayTelemetryObserver>) -> Self {
        Self {
            observer,
            sample_interval_ms,
            next_sample_ordinal: 0,
            next_periodic_ordinal: 1,
            delivered_output_tokens: 0,
            pending_output_receipts_ms: Vec::new(),
            committed_prefill_tokens: 0,
            first_admission: ReplayPromptSourceCounters::default(),
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

    pub(crate) fn started(&self) -> bool {
        self.sampling_origin_ms.is_some()
    }

    pub(crate) fn reset_counters(&mut self) {
        self.delivered_output_tokens = 0;
        self.pending_output_receipts_ms.clear();
        self.committed_prefill_tokens = 0;
        self.first_admission = ReplayPromptSourceCounters::default();
    }

    pub(crate) fn on_output_token(&mut self, now_ms: f64, received_ms: f64) {
        if received_ms <= now_ms {
            self.delivered_output_tokens += 1;
        } else {
            self.pending_output_receipts_ms.push(received_ms);
        }
    }
    pub(crate) fn on_prefill_work(&mut self, tokens: u64) {
        self.committed_prefill_tokens += tokens;
    }
    pub(crate) fn on_first_admission(
        &mut self,
        input: usize,
        reused: usize,
        tiers: Option<crate::engine::CacheTierAttribution>,
    ) {
        self.first_admission.requests += 1;
        self.first_admission.input_tokens += input as u64;
        self.first_admission.hbm_reused_tokens +=
            tiers.map_or(reused, |t| t.g1_reused_input_tokens) as u64;
        self.first_admission.host_reused_tokens +=
            tiers.map_or(0, |t| t.host_reused_input_tokens) as u64;
        self.first_admission.uncached_tokens += input.saturating_sub(reused) as u64;
    }
    pub(crate) fn has_observations(&self) -> bool {
        self.delivered_output_tokens != 0
            || !self.pending_output_receipts_ms.is_empty()
            || self.committed_prefill_tokens != 0
            || self.first_admission.requests != 0
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
        let at_ms = origin_ms + self.next_periodic_ordinal as f64 * self.sample_interval_ms;
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

    pub(crate) fn publish(&mut self, mut snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        if snapshot.kind != ReplayTelemetrySampleKind::Baseline {
            // Detokenizer servers can deliver outputs out of engine completion order.
            self.pending_output_receipts_ms.retain(|received_ms| {
                if *received_ms <= snapshot.sampled_at_ms {
                    self.delivered_output_tokens += 1;
                    false
                } else {
                    true
                }
            });
            snapshot.traffic.delivered_output_tokens =
                std::mem::take(&mut self.delivered_output_tokens);
            snapshot.traffic.committed_prefill_tokens =
                std::mem::take(&mut self.committed_prefill_tokens);
            snapshot.traffic.first_admission = std::mem::take(&mut self.first_admission);
        }
        if snapshot.kind == ReplayTelemetrySampleKind::Periodic {
            self.next_periodic_ordinal += 1;
        }
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
            host_caches: Vec::new(),
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
    fn phase_boundary_samples_do_not_shift_periodic_cadence() {
        for boundary_ms in [7.0, 10.0] {
            let mut runtime = ReplayTelemetryRuntime::new(10.0, Box::new(NoopObserver));
            runtime.start_at(0.0);
            runtime
                .publish(sample(0, ReplayTelemetrySampleKind::Baseline, 0.0, 0.0))
                .unwrap();
            runtime
                .publish(sample(
                    1,
                    ReplayTelemetrySampleKind::Final,
                    0.0,
                    boundary_ms,
                ))
                .unwrap();
            runtime.close_interval(boundary_ms);
            runtime
                .publish(sample(
                    2,
                    ReplayTelemetrySampleKind::Baseline,
                    boundary_ms,
                    boundary_ms,
                ))
                .unwrap();
            // The pre-existing queued tick remains at 10ms, even if the phase
            // ends at that exact tick. Once handled, the following tick is20ms.
            runtime
                .publish(sample(
                    3,
                    ReplayTelemetrySampleKind::Periodic,
                    boundary_ms,
                    10.0,
                ))
                .unwrap();
            runtime.close_interval(10.0);
            assert_eq!(runtime.next_periodic_at_ms().unwrap(), 20.0);
            assert_eq!(runtime.next_sample_ordinal(), 4);
        }
    }

    #[test]
    fn capture_exports_unavailable_means_as_null() {
        let row = ReplayTelemetrySample {
            phase: "profile",
            sample: sample(0, ReplayTelemetrySampleKind::Baseline, 0.0, 0.0),
        };
        let value = serde_json::to_value(row).unwrap();
        for field in [
            "avg_isl",
            "avg_osl",
            "avg_ttft_ms",
            "avg_itl_ms",
            "avg_router_kv_hit_rate",
        ] {
            assert!(value["traffic"][field].is_null(), "{field}");
        }
        assert_eq!(value["traffic"]["delivered_output_tokens"], 0);
    }

    #[test]
    fn capture_distinguishes_missing_cache_metrics_from_measured_zero() {
        let grouped = ReplaySchedulerMetricsSnapshot {
            worker_id: 0,
            dp_rank: 0,
            active_blocks: 2,
            inactive_blocks: 0,
            total_blocks: 0,
            kv_cache_used_bytes: Some(1024),
            kv_cache_capacity_bytes: Some(2048),
            active_cache_usage: 0.5,
            physical_cache_usage: 0.5,
            running_requests: 1,
            waiting_requests: 0,
        };
        let linear = ReplaySchedulerMetricsSnapshot {
            total_blocks: 4,
            kv_cache_used_bytes: None,
            kv_cache_capacity_bytes: None,
            ..grouped.clone()
        };
        let mut row = ReplayTelemetrySample {
            phase: "profile",
            sample: sample(1, ReplayTelemetrySampleKind::Periodic, 0.0, 1000.0),
        };
        row.sample.prefill_scheduler_metrics = vec![grouped.clone(), linear.clone()];
        row.sample.decode_scheduler_metrics = vec![linear, grouped];
        row.sample
            .prefill_interval_metrics
            .native_prefix_cache_lookup = Some(crate::engine::PrefixCacheLookupCounts {
            query_count: 1,
            queried_tokens: 8,
            matched_tokens: 0,
        });
        row.sample.decode_interval_metrics.cache_total_tokens = 8;
        for measured_hits in [0, 4] {
            row.sample.decode_interval_metrics.cache_hit_tokens = measured_hits;
            let value = serde_json::to_value(&row).unwrap();
            for field in ["cache_hit_tokens", "cache_total_tokens"] {
                assert!(value["prefill_interval_metrics"][field].is_null());
            }
            assert_eq!(
                value["decode_interval_metrics"]["cache_hit_tokens"],
                measured_hits
            );
            assert_eq!(value["decode_interval_metrics"]["cache_total_tokens"], 8);
            assert_eq!(
                value["prefill_interval_metrics"]["native_prefix_cache_lookup"]["matched_tokens"],
                0
            );
            for (name, grouped_index, linear_index) in [
                ("prefill_scheduler_metrics", 0, 1),
                ("decode_scheduler_metrics", 1, 0),
            ] {
                assert!(value[name][grouped_index]["total_blocks"].is_null());
                assert_eq!(value[name][grouped_index]["kv_cache_capacity_bytes"], 2048);
                assert_eq!(value[name][grouped_index]["active_blocks"], 2);
                assert_eq!(value[name][linear_index]["total_blocks"], 4);
            }
        }
        // Observers retain the legacy numeric DTO contract; capture export
        // changes neither the original values nor their direct serialization.
        let observer_value = serde_json::to_value(&row.sample).unwrap();
        assert_eq!(
            observer_value["prefill_interval_metrics"]["cache_total_tokens"],
            0
        );
        assert_eq!(
            observer_value["prefill_scheduler_metrics"][0]["total_blocks"],
            0
        );
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
