// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use aisimulate_core::engine::{Backend, EngineConfig, TimingModel, TimingModelConfig};
use aisimulate_core::replay::loadgen::ReplayRequestPayload;
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, NoEngineEvents, NoReplayMetadata, Placement, PlacementEffects,
    PlacementPolicy, PoolRoundRobinPlacement, ProviderSpec, ReplayAdapters, ReplayCaptureOptions,
    ReplayComposition, ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayError,
    ReplayRequest, ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec,
    ReplayTelemetryObserver, ReplayTelemetrySampleKind, ReplayTelemetrySnapshot, ReplayTopology,
    Replayer, WorkerPoolSpec, WorkerTopology,
};
use uuid::Uuid;

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        id: id.to_string(),
        arrival_time_ms,
        input_tokens,
        input_token_ids: None,
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        prefill_dp_rank: None,
        session_id: None,
        turn_index: None,
        metadata: serde_json::Value::Null,
    }
}

fn aggregated_spec(
    backend: Backend,
    workers: usize,
    startup_delay_ms: f64,
    requests: Vec<ReplayRequest>,
) -> ReplaySpec {
    let rank = EngineConfig {
        num_gpu_blocks: 64,
        block_size: 4,
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 10.0,
            decode_ms: 2.0,
        },
        ..EngineConfig::for_backend(backend)
    };
    ReplaySpec {
        version: 1,
        encoder: None,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms,
            },
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            rank,
            ..ReplayEngineConfig::default()
        })
        .unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests,
    }
}

struct RecordingTelemetryObserver {
    samples: Arc<Mutex<Vec<ReplayTelemetrySnapshot>>>,
}

impl ReplayTelemetryObserver for RecordingTelemetryObserver {
    fn on_sample(&mut self, snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        self.samples.lock().unwrap().push(snapshot);
        Ok(())
    }
}

struct FailingTelemetryObserver;

impl ReplayTelemetryObserver for FailingTelemetryObserver {
    fn on_sample(&mut self, _snapshot: ReplayTelemetrySnapshot) -> anyhow::Result<()> {
        anyhow::bail!("telemetry sink failed")
    }
}

#[test]
fn replay_accept_length_counts_decode_work_before_output_truncation() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        for (rates, lengths, forwards, expected_average) in [
            (None, vec![0], [0, 0], None),
            (None, vec![1], [0, 0], None),
            (None, vec![2], [1, 1], Some(1.0)),
            (None, vec![7, 2], [7, 7], Some(1.0)),
            (Some("1,1"), vec![1], [0, 0], None),
            (Some("1,1"), vec![4], [1, 1], Some(3.0)),
            (Some("1,1"), vec![7], [2, 2], Some(3.0)),
            (Some("1,1"), vec![8], [2, 3], Some(3.0)),
            (Some("1,0"), vec![7], [3, 3], Some(2.0)),
            (Some("0,0"), vec![7], [6, 6], Some(1.0)),
        ] {
            let samples = Arc::new(Mutex::new(Vec::new()));
            let requests = lengths
                .iter()
                .enumerate()
                .map(|(index, &length)| {
                    request(&format!("request-{index}"), index as f64 * 5.0, 4, length)
                })
                .collect();
            let mut spec = aggregated_spec(backend, 1, 0.0, requests);
            spec.max_sim_time_ms = Some(1_000.0);
            if let Some(rates) = rates {
                let mut engine: ReplayEngineConfig =
                    serde_json::from_value(spec.engine.clone()).unwrap();
                engine.rank.aic_nextn = Some(2);
                engine.rank.aic_nextn_accept_rates = Some(rates.into());
                spec.engine = serde_json::to_value(engine).unwrap();
            }
            let report = Replayer::new(spec, ReplayEngineFactory::new())
                .unwrap()
                .with_telemetry_observer(
                    1_000.0,
                    Box::new(RecordingTelemetryObserver {
                        samples: Arc::clone(&samples),
                    }),
                )
                .unwrap()
                .run()
                .unwrap();
            assert_eq!(report.request_counts.completed_requests, lengths.len());
            let tokens = lengths.iter().sum::<usize>();
            assert_eq!(report.request_counts.total_output_tokens, tokens);
            let samples = samples.lock().unwrap();
            let traffic = &samples.last().unwrap().traffic;
            let expected_forwards = forwards[usize::from(backend == Backend::Sglang)];
            assert_eq!(
                traffic.accept_length_forward_count, expected_forwards,
                "{backend:?}, {lengths:?}"
            );
            assert_eq!(
                traffic.avg_accept_length, expected_average,
                "{backend:?}, {lengths:?}, {rates:?}"
            );
        }
    }
}

#[test]
fn replay_accept_length_survives_rank_aggregation_and_prefill_handoffs() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        for (dp_size, disaggregated) in [(1, false), (2, false), (2, true)] {
            let samples = Arc::new(Mutex::new(Vec::new()));
            let mut requests = vec![request("a", 0.0, 4, 4), request("b", 5.0, 4, 4)];
            for (index, request) in requests.iter_mut().enumerate() {
                request.dp_rank = Some(index as u32 % dp_size);
            }
            let mut spec = aggregated_spec(backend, 1, 0.0, requests);
            spec.max_sim_time_ms = Some(1_000.0);
            let mut engine: ReplayEngineConfig =
                serde_json::from_value(spec.engine.clone()).unwrap();
            engine.dp_size = dp_size;
            engine.rank.aic_nextn = Some(2);
            engine.rank.aic_nextn_accept_rates = Some("1,1".into());
            spec.engine = serde_json::to_value(engine).unwrap();
            if disaggregated {
                spec.topology = ReplayTopology::Disaggregated {
                    prefill: WorkerPoolSpec {
                        initial_workers: 1,
                        startup_delay_ms: 0.0,
                    },
                    decode: WorkerPoolSpec {
                        initial_workers: 1,
                        startup_delay_ms: 0.0,
                    },
                    handoff_latency_ms: 1.0,
                };
            }
            let report = Replayer::new(spec, ReplayEngineFactory::new())
                .unwrap()
                .with_telemetry_observer(
                    5.0,
                    Box::new(RecordingTelemetryObserver {
                        samples: Arc::clone(&samples),
                    }),
                )
                .unwrap()
                .run()
                .unwrap();
            assert_eq!(report.request_counts.completed_requests, 2);
            assert_eq!(report.request_counts.total_output_tokens, 8);
            let samples = samples.lock().unwrap();
            let mut forwards = 0;
            for sample in samples.iter() {
                let traffic = &sample.traffic;
                assert_eq!(
                    traffic.avg_accept_length,
                    (traffic.accept_length_forward_count > 0).then_some(3.0),
                    "{backend:?}, dp={dp_size}, pd={disaggregated}"
                );
                forwards += traffic.accept_length_forward_count;
            }
            // P/D uses the source output as a handoff signal; all four visible
            // tokens are produced at decode. Aggregated prefill produces output.
            assert_eq!(
                forwards,
                if disaggregated { 4 } else { 2 },
                "{backend:?}, dp={dp_size}, pd={disaggregated}"
            );
        }
    }
}

#[test]
fn replay_accept_length_ignores_overlapping_prefill_and_rejection() {
    let samples = Arc::new(Mutex::new(Vec::new()));
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![
            request("decoding", 0.0, 4, 7),
            request("prefill-only", 5.0, 4, 1),
            request("rejected", 5.0, 65, 1),
        ],
    );
    spec.max_sim_time_ms = Some(1_000.0);
    let mut engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    engine.rank.max_model_len = Some(64);
    engine.rank.aic_nextn = Some(2);
    engine.rank.aic_nextn_accept_rates = Some("1,1".into());
    spec.engine = serde_json::to_value(engine).unwrap();
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_observer(
            1_000.0,
            Box::new(RecordingTelemetryObserver {
                samples: Arc::clone(&samples),
            }),
        )
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_output_tokens, 8);
    assert_eq!(
        report
            .per_request
            .iter()
            .filter(|request| request.terminal_status
                == aisimulate_core::replay::ReplayTerminalStatus::Rejected)
            .count(),
        1
    );
    let samples = samples.lock().unwrap();
    let traffic = &samples.last().unwrap().traffic;
    assert_eq!(traffic.accept_length_forward_count, 2);
    assert_eq!(traffic.avg_accept_length, Some(3.0));
}

#[test]
fn telemetry_baseline_preserves_t0_activity_and_emits_a_positive_final_tail() {
    let samples = Arc::new(Mutex::new(Vec::new()));
    let observer = RecordingTelemetryObserver {
        samples: Arc::clone(&samples),
    };
    let spec = aggregated_spec(Backend::Vllm, 1, 0.0, vec![request("telemetry", 0.0, 4, 3)]);

    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_observer(6.0, Box::new(observer))
        .unwrap()
        .run()
        .unwrap();

    assert_eq!(report.request_counts.completed_requests, 1);
    let samples = samples.lock().unwrap();
    assert!(samples.len() >= 3);

    let baseline = &samples[0];
    assert_eq!(baseline.sample_ordinal, 0);
    assert_eq!(baseline.kind, ReplayTelemetrySampleKind::Baseline);
    assert_eq!(baseline.interval_start_ms, 0.0);
    assert_eq!(baseline.sampled_at_ms, 0.0);
    assert_eq!(baseline.traffic.arriving_requests, 0);
    assert_eq!(baseline.traffic.completed_requests, 0);
    assert_eq!(baseline.decode_interval_metrics.cache_hit_tokens, 0);
    assert_eq!(baseline.decode_interval_metrics.cache_total_tokens, 0);
    assert_eq!(baseline.decode_interval_metrics.preemptions, 0);

    let first_interval = &samples[1];
    assert_eq!(first_interval.kind, ReplayTelemetrySampleKind::Periodic);
    assert_eq!(first_interval.sampled_at_ms, 6.0);
    assert_eq!(first_interval.traffic.arriving_requests, 1);

    assert_eq!(
        samples
            .iter()
            .skip(1)
            .map(|sample| sample.traffic.arriving_requests)
            .sum::<usize>(),
        1
    );
    assert_eq!(
        samples
            .iter()
            .skip(1)
            .map(|sample| sample.traffic.completed_requests)
            .sum::<usize>(),
        1
    );
    let final_sample = samples.last().unwrap();
    assert_eq!(final_sample.kind, ReplayTelemetrySampleKind::Final);
    assert!(final_sample.sampled_at_ms > final_sample.interval_start_ms);
    assert!(final_sample.traffic.duration_s > 0.0);
}

#[test]
fn telemetry_observer_failures_keep_their_native_error_classification() {
    let spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("telemetry-error", 0.0, 4, 1)],
    );

    let error = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_observer(1.0, Box::new(FailingTelemetryObserver))
        .unwrap()
        .run()
        .unwrap_err();

    assert!(matches!(
        error,
        ReplayError::Telemetry(message) if message.contains("telemetry sink failed")
    ));
}

#[test]
fn telemetry_interval_must_be_positive_and_finite() {
    for interval_ms in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let spec = aggregated_spec(Backend::Vllm, 1, 0.0, Vec::new());
        let error = match Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_observer(interval_ms, Box::new(FailingTelemetryObserver))
        {
            Ok(_) => panic!("invalid telemetry interval {interval_ms} was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, ReplayError::InvalidSpec(_)));
    }
}

#[test]
fn telemetry_sampling_does_not_perturb_the_replay_report() {
    let spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("telemetry-neutral", 0.0, 4, 2)],
    );
    let capture = ReplayCaptureOptions {
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    };
    let mut baseline = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .run()
        .unwrap();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let observer = RecordingTelemetryObserver {
        samples: Arc::clone(&samples),
    };
    let mut observed = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .with_telemetry_observer(0.1, Box::new(observer))
        .unwrap()
        .run()
        .unwrap();

    baseline.throughput.wall_time_ms = 0.0;
    observed.throughput.wall_time_ms = 0.0;
    assert!(
        (baseline.throughput.decode_worker_seconds - observed.throughput.decode_worker_seconds)
            .abs()
            < 1e-12
    );
    assert!((baseline.throughput.gpu_hours - observed.throughput.gpu_hours).abs() < 1e-12);
    // Sampling splits the same resource integral into more floating-point
    // additions, so normalize its sub-ULP representation before comparing all
    // other compact-report fields exactly.
    observed.throughput.decode_worker_seconds = baseline.throughput.decode_worker_seconds;
    observed.throughput.gpu_hours = baseline.throughput.gpu_hours;
    assert_eq!(
        serde_json::to_value(&baseline).unwrap(),
        serde_json::to_value(&observed).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&baseline.per_request).unwrap(),
        serde_json::to_value(&observed.per_request).unwrap()
    );
    assert!(samples.lock().unwrap().len() > 2);
}

#[test]
fn capped_telemetry_does_not_advance_agg_accounting_and_flushes_t0_observations() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("capped-telemetry", 0.0, 4, 2)],
    );
    spec.max_sim_time_ms = Some(0.0);
    let capture = ReplayCaptureOptions {
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    };
    let mut baseline = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .run()
        .unwrap();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let observer = RecordingTelemetryObserver {
        samples: Arc::clone(&samples),
    };
    let mut observed = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .with_telemetry_observer(0.1, Box::new(observer))
        .unwrap()
        .run()
        .unwrap();

    baseline.throughput.wall_time_ms = 0.0;
    observed.throughput.wall_time_ms = 0.0;
    assert_eq!(baseline.throughput.duration_ms, 0.0);
    assert_eq!(observed.throughput.duration_ms, 0.0);
    assert_eq!(baseline.throughput.decode_worker_seconds, 0.0);
    assert_eq!(observed.throughput.decode_worker_seconds, 0.0);
    assert_eq!(baseline.throughput.gpu_hours, 0.0);
    assert_eq!(observed.throughput.gpu_hours, 0.0);
    assert_eq!(
        serde_json::to_value(&baseline).unwrap(),
        serde_json::to_value(&observed).unwrap()
    );

    let samples = samples.lock().unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].kind, ReplayTelemetrySampleKind::Baseline);
    let final_sample = &samples[1];
    assert_eq!(final_sample.kind, ReplayTelemetrySampleKind::Final);
    assert_eq!(final_sample.interval_start_ms, 0.0);
    assert_eq!(final_sample.sampled_at_ms, 0.0);
    assert_eq!(final_sample.traffic.duration_s, 0.0);
    assert_eq!(final_sample.traffic.arriving_requests, 1);
}

#[test]
fn telemetry_heartbeats_do_not_advance_agg_through_a_capped_idle_gap() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("after-cap", 10.0, 4, 2)],
    );
    spec.max_sim_time_ms = Some(5.0);
    let capture = ReplayCaptureOptions {
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    };
    let mut baseline = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .run()
        .unwrap();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let observer = RecordingTelemetryObserver {
        samples: Arc::clone(&samples),
    };
    let mut observed = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(capture)
        .with_telemetry_observer(1.0, Box::new(observer))
        .unwrap()
        .run()
        .unwrap();

    baseline.throughput.wall_time_ms = 0.0;
    observed.throughput.wall_time_ms = 0.0;
    assert_eq!(baseline.throughput.duration_ms, 0.0);
    assert_eq!(observed.throughput.duration_ms, 0.0);
    assert_eq!(
        serde_json::to_value(&baseline).unwrap(),
        serde_json::to_value(&observed).unwrap()
    );

    let samples = samples.lock().unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].kind, ReplayTelemetrySampleKind::Baseline);
    assert_eq!(samples[0].sampled_at_ms, 0.0);
}

struct GeneralRoundRobinComposition {
    scaling: Option<Box<dyn ReplayScalingPolicy>>,
    scaling_construction_error: Option<&'static str>,
}

impl GeneralRoundRobinComposition {
    fn fixed_capacity() -> Self {
        Self {
            scaling: None,
            scaling_construction_error: None,
        }
    }

    fn with_scaling(scaling: impl ReplayScalingPolicy + 'static) -> Self {
        Self {
            scaling: Some(Box::new(scaling)),
            scaling_construction_error: None,
        }
    }

    fn failing_scaling_construction(message: &'static str) -> Self {
        Self {
            scaling: None,
            scaling_construction_error: Some(message),
        }
    }
}

impl ReplayComposition for GeneralRoundRobinComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = AggregatedRoundRobinPlacement<()>;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(AggregatedRoundRobinPlacement::new(dp_size, topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        if let Some(message) = self.scaling_construction_error {
            anyhow::bail!(message);
        }
        Ok(self.scaling.take())
    }
}

struct FailingPlacement;

impl PlacementPolicy<ReplayRequestPayload> for FailingPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        _request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        anyhow::bail!("router placement callback failed")
    }

    fn observe(
        &mut self,
        _observation: Self::Observation,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    fn request_terminal(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_draining(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn worker_removed(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

struct FailingPlacementComposition {
    fail_construction: bool,
}

impl ReplayComposition for FailingPlacementComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = FailingPlacement;
    type DisaggregatedPlacement = FailingPlacement;

    fn create_aggregated_placement(
        &mut self,
        _dp_size: u32,
        _topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        if self.fail_construction {
            anyhow::bail!("router placement construction failed");
        }
        Ok(FailingPlacement)
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        _prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        _decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        if self.fail_construction {
            anyhow::bail!("router placement construction failed");
        }
        Ok((FailingPlacement, FailingPlacement))
    }
}

struct FailingScalingPolicy;

impl ReplayScalingPolicy for FailingScalingPolicy {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn on_tick(
        &mut self,
        _snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        anyhow::bail!("planner scaling callback failed")
    }
}

struct FailingTimingModel;

impl TimingModel for FailingTimingModel {
    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> anyhow::Result<f64> {
        anyhow::bail!("engine timing callback failed")
    }

    fn predict_decode_ms(
        &self,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> anyhow::Result<f64> {
        anyhow::bail!("engine timing callback failed")
    }
}

fn canonical_options() -> ReplayCaptureOptions {
    ReplayCaptureOptions {
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    }
}

#[test]
fn placement_construction_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("placement-construction", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        FailingPlacementComposition {
            fail_construction: true,
        },
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Placement("router placement construction failed".to_string())
    );
}

#[test]
fn placement_callback_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("placement-callback", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        FailingPlacementComposition {
            fail_construction: false,
        },
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Placement("router placement callback failed".to_string())
    );
}

#[test]
fn scaling_construction_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("scaling-construction", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::failing_scaling_construction(
            "planner scaling construction failed",
        ),
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Scaling("planner scaling construction failed".to_string())
    );
}

#[test]
fn scaling_callback_failure_retains_its_error_provenance() {
    let error = Replayer::with_composition(
        aggregated_spec(
            Backend::Vllm,
            1,
            0.0,
            vec![request("scaling-callback", 0.0, 8, 2)],
        ),
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::with_scaling(FailingScalingPolicy),
    )
    .unwrap()
    .run()
    .unwrap_err();

    assert_eq!(
        error,
        ReplayError::Scaling("planner scaling callback failed".to_string())
    );
}

#[test]
fn engine_callback_failure_retains_its_error_provenance() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("engine-callback", 0.0, 8, 2)],
    );
    let mut config: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    config.rank.timing_model = TimingModelConfig::External {
        provider: "failing_test_timing".to_string(),
        config: serde_json::Value::Null,
    };
    spec.engine = serde_json::to_value(config).unwrap();

    let error = Replayer::new(
        spec,
        ReplayEngineFactory::with_timing_model(Arc::new(FailingTimingModel)),
    )
    .unwrap()
    .run()
    .unwrap_err();

    let ReplayError::Engine(message) = error else {
        panic!("expected an engine error, got {error:?}");
    };
    assert!(message.contains("engine timing callback failed"));
}

#[test]
fn authored_rank_is_validated_by_the_default_aggregated_runtime() {
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        0.0,
        vec![request("invalid-rank", 0.0, 8, 3)],
    );
    spec.requests[0].dp_rank = Some(1);

    let default_error = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(canonical_options())
        .run()
        .unwrap_err()
        .to_string();
    let general_error = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::fixed_capacity(),
    )
    .unwrap()
    .with_capture_options(canonical_options())
    .run()
    .unwrap_err()
    .to_string();

    assert_eq!(default_error, general_error);
    assert!(default_error.contains("preferred attention-DP rank 1 is out of range"));
}

#[test]
fn max_sim_time_is_a_soft_cap_for_the_aggregated_runtime() {
    let requests = (0..5)
        .map(|index| {
            request(
                &format!("request-{index}"),
                f64::from(index) * 1_000.0,
                4,
                2,
            )
        })
        .collect();
    let mut capped = aggregated_spec(Backend::Vllm, 1, 0.0, requests);
    capped.max_sim_time_ms = Some(2_500.0);

    let report = Replayer::new(capped, ReplayEngineFactory::new())
        .unwrap()
        .with_capture_options(canonical_options())
        .run()
        .unwrap();

    assert_eq!(report.request_counts.num_requests, 3);
    assert_eq!(report.request_counts.completed_requests, 3);
    assert!(report.throughput.duration_ms <= 2_500.0);
}

#[derive(Debug, Clone, PartialEq)]
struct ScalingObservation {
    now_ms: f64,
    active: Vec<usize>,
    starting: Vec<usize>,
    draining: Vec<usize>,
}

struct ScaleUpThenDown {
    step: usize,
    observations: Rc<RefCell<Vec<ScalingObservation>>>,
}

struct ReleaseOnTopologyCallbacks {
    pending: VecDeque<Uuid>,
    stable_scheduler_id: usize,
    released_on_settled: bool,
}

impl ReleaseOnTopologyCallbacks {
    fn new(topology: Vec<WorkerTopology>) -> Self {
        let stable_scheduler_id = topology[0].scheduler_ids[0];
        Self {
            pending: VecDeque::new(),
            stable_scheduler_id,
            released_on_settled: false,
        }
    }

    fn release_next(&mut self, scheduler_id: usize) -> Vec<Placement> {
        self.pending
            .pop_front()
            .map(|request_id| {
                vec![Placement {
                    request_id,
                    scheduler_id,
                    reported_overlap_tokens: 0,
                    cache_sample: None,
                    placement_replica_id: None,
                }]
            })
            .unwrap_or_default()
    }
}

impl PlacementPolicy<ReplayRequestPayload> for ReleaseOnTopologyCallbacks {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        self.pending
            .push_back(request.metadata().uuid.expect("test request UUID"));
        Ok(PlacementEffects {
            decision: aisimulate_core::replay::PlacementDecision::Queued,
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _observation: (), _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        let before = self.pending.len();
        self.pending.retain(|pending| *pending != request_id);
        self.pending.len() != before
    }

    fn request_terminal(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(
        &mut self,
        _request_id: Uuid,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn worker_ready(
        &mut self,
        worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(self.release_next(worker.scheduler_ids[0]))
    }

    fn worker_draining(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(self.release_next(self.stable_scheduler_id))
    }

    fn worker_removed(
        &mut self,
        _worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        Ok(self.release_next(self.stable_scheduler_id))
    }

    fn topology_settled(&mut self, _now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        if self.released_on_settled {
            return Ok(Vec::new());
        }
        self.released_on_settled = true;
        Ok(self.release_next(self.stable_scheduler_id))
    }
}

struct TopologyReleaseComposition {
    scaling: Option<Box<dyn ReplayScalingPolicy>>,
}

impl ReplayComposition for TopologyReleaseComposition {
    type Metadata = NoReplayMetadata;
    type Observation = NoEngineEvents;
    type AggregatedPlacement = ReleaseOnTopologyCallbacks;
    type DisaggregatedPlacement = PoolRoundRobinPlacement<()>;

    fn create_aggregated_placement(
        &mut self,
        _dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<Self::AggregatedPlacement> {
        Ok(ReleaseOnTopologyCallbacks::new(topology))
    }

    fn create_disaggregated_placements(
        &mut self,
        _prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        _decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> anyhow::Result<(Self::DisaggregatedPlacement, Self::DisaggregatedPlacement)> {
        Ok((
            PoolRoundRobinPlacement::new(prefill_topology),
            PoolRoundRobinPlacement::new(decode_topology),
        ))
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling.take())
    }
}

struct AddThenRemoveWorker {
    step: usize,
}

impl ReplayScalingPolicy for AddThenRemoveWorker {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn on_tick(
        &mut self,
        _snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        let decision = match self.step {
            0 => ReplayScalingDecision {
                target_decode: Some(2),
                next_tick_ms: Some(1.0),
                ..ReplayScalingDecision::default()
            },
            1 => ReplayScalingDecision {
                target_decode: Some(1),
                ..ReplayScalingDecision::default()
            },
            _ => panic!("unexpected scaling tick {}", self.step),
        };
        self.step += 1;
        Ok(decision)
    }
}

impl ReplayScalingPolicy for ScaleUpThenDown {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(100.0)
    }

    fn on_tick(
        &mut self,
        snapshot: ReplayScalingSnapshot,
    ) -> anyhow::Result<ReplayScalingDecision> {
        self.observations.borrow_mut().push(ScalingObservation {
            now_ms: snapshot.now_ms,
            active: snapshot.active_decode_ids,
            starting: snapshot.starting_decode_ids,
            draining: snapshot.draining_decode_ids,
        });
        let decision = match self.step {
            0 => ReplayScalingDecision {
                target_decode: Some(2),
                next_tick_ms: Some(200.0),
                ..ReplayScalingDecision::default()
            },
            1 => ReplayScalingDecision {
                next_tick_ms: Some(300.0),
                ..ReplayScalingDecision::default()
            },
            2 => ReplayScalingDecision {
                target_decode: Some(1),
                next_tick_ms: Some(400.0),
                ..ReplayScalingDecision::default()
            },
            3 => ReplayScalingDecision::default(),
            _ => panic!("unexpected scaling tick {}", self.step),
        };
        self.step += 1;
        Ok(decision)
    }
}

#[test]
fn scaling_honors_startup_delay_then_scales_the_ready_worker_back_down() {
    let observations = Rc::new(RefCell::new(Vec::new()));
    let policy = ScaleUpThenDown {
        step: 0,
        observations: Rc::clone(&observations),
    };
    let mut spec = aggregated_spec(
        Backend::Vllm,
        1,
        200.0,
        vec![request("long-running", 0.0, 4, 2)],
    );
    let mut engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
    engine.rank.timing_model = TimingModelConfig::Fixed {
        prefill_ms: 1_000.0,
        decode_ms: 1_000.0,
    };
    spec.engine = serde_json::to_value(engine).unwrap();
    spec.adapters.scaling = ProviderSpec {
        provider: "scripted_test_policy".to_string(),
        config: serde_json::Value::Null,
    };

    let report = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        GeneralRoundRobinComposition::with_scaling(policy),
    )
    .unwrap()
    .run()
    .unwrap();

    assert_eq!(
        *observations.borrow(),
        vec![
            ScalingObservation {
                now_ms: 100.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 200.0,
                active: vec![0],
                starting: vec![1],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 300.0,
                active: vec![0, 1],
                starting: vec![],
                draining: vec![],
            },
            ScalingObservation {
                now_ms: 400.0,
                active: vec![0],
                starting: vec![],
                draining: vec![],
            },
        ]
    );
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.throughput.duration_ms, 2_000.0);
    assert!((report.throughput.decode_worker_seconds - 2.2).abs() < 1e-9);
}

#[test]
fn aggregated_topology_callbacks_dispatch_and_record_every_released_request() {
    let requests = (0..4)
        .map(|index| request(&format!("callback-{index}"), 0.0, 4, 2))
        .collect();
    let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, requests);
    spec.adapters.scaling = ProviderSpec {
        provider: "topology_callback_test".to_string(),
        config: serde_json::Value::Null,
    };

    let report = Replayer::with_composition(
        spec,
        ReplayEngineFactory::new(),
        TopologyReleaseComposition {
            scaling: Some(Box::new(AddThenRemoveWorker { step: 0 })),
        },
    )
    .unwrap()
    .with_capture_options(ReplayCaptureOptions {
        capture_lifecycle_evidence: true,
        determinism: ReplayDeterminism::CanonicalV1,
        ..ReplayCaptureOptions::default()
    })
    .run()
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, 4);
    let released = report
        .runtime_evidence
        .lifecycle_operations
        .iter()
        .filter(|operation| !operation.topology_released_request_uuids.is_empty())
        .map(|operation| {
            (
                operation.cause,
                operation.topology_released_request_uuids.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        released,
        vec![
            (
                "planner_scale",
                vec![
                    Uuid::from_u128(1).to_string(),
                    Uuid::from_u128(2).to_string(),
                ],
            ),
            ("planner_scale", vec![Uuid::from_u128(3).to_string()],),
            ("drain_settlement", vec![Uuid::from_u128(4).to_string()],),
        ]
    );
}
