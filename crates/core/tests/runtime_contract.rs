// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use aisimulate_core::engine::generalized::{EngineIdentity, SchedulerCommand};
use aisimulate_core::engine::{
    Backend, Command, EngineConfig, EngineFactory, HandoffId, LifecycleEvent, Request,
    SglangConfig, TimingModel, TimingModelConfig, WorkerType,
};
use aisimulate_core::replay::loadgen::ReplayRequestPayload;
use aisimulate_core::replay::{
    AggregatedRoundRobinPlacement, AuthoredKvHints, NoEngineEvents, NoReplayMetadata, Placement,
    PlacementEffects, PlacementPolicy, PoolRoundRobinPlacement, ProviderSpec, ReplayAdapters,
    ReplayCaptureOptions, ReplayComposition, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayError, ReplayRequest, ReplayRequestPool, ReplayScalingDecision,
    ReplayScalingPolicy, ReplayScalingSnapshot, ReplaySpec, ReplayTelemetryObserver,
    ReplayTelemetrySampleKind, ReplayTelemetrySnapshot, ReplayTopology, Replayer,
    WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolSpec, WorkerTopology,
};
use uuid::Uuid;

fn request(
    id: &str,
    arrival_time_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
) -> ReplayRequest {
    ReplayRequest {
        cache_salt: None,
        id: id.to_string(),
        arrival_time_ms,
        input_tokens,
        input_token_ids: None,
        output_tokens,
        output_token_ids: None,
        dp_rank: None,
        prefill_dp_rank: None,
        kv_hints: None,
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
        sglang: SglangConfig {
            num_reserved_decode_tokens: 0,
            ..SglangConfig::default()
        },
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

/// One prefill and one decode worker that share one cold vLLM 0.31.0 rank: the
/// six GLM-5.3-Flash groups at TP 4 with an fp8 cache and the transfer bytes of
/// each group's own layers, a pool of 128 pages, and a handoff priced at
/// 1 GB/s over the full prompt.
fn disaggregated_grouped_spec(requests: Vec<ReplayRequest>) -> ReplaySpec {
    const GLM_PAGE: u64 = 13_045_120;
    let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, requests);
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
    // Name, kind, layers, block, transfer bytes, prefix_cacheable.
    let groups = [
        ("mla", "attention", 22, 2176, 13_045_120, true),
        ("kpool_tail", "state", 11, 4, 22_528, false),
        ("kda_0", "state", 9, 2176, 9_768_960, true),
        ("kda_1", "state", 9, 2176, 9_768_960, true),
        ("kda_2", "state", 8, 2176, 8_683_520, true),
        ("kda_3", "state", 8, 2176, 8_683_520, true),
    ]
    .map(|(name, kind, layers, block, bytes, cacheable)| {
        let mut group = serde_json::json!({
            "name": name, "kind": kind, "num_layers": layers, "block_size_tokens": block,
            "page_size_bytes": GLM_PAGE, "prefix_cacheable": cacheable,
            "transfer_bytes_per_block": bytes
        });
        if kind == "state" {
            group["blocks_per_request"] = 1.into();
        }
        group
    });
    let rank = &mut spec.engine["rank"];
    rank["enable_prefix_caching"] = false.into();
    rank["vllm_version"] = "0.31.0".into();
    rank["max_num_batched_tokens"] = 8192.into();
    rank["kv_cache_capacity_bytes"] = (128 * GLM_PAGE).into();
    rank["kv_cache_groups"] = groups.to_vec().into();
    rank["kv_transfer_bandwidth"] = 1.0.into();
    rank["kv_transfer_timing_mode"] = "full_prompt".into();
    spec
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
fn grouped_cache_replay_emits_physical_bytes_without_scalar_block_capacity() {
    let samples = Arc::new(Mutex::new(Vec::new()));
    let mut req = request("windowed", 10.0, 1152, 4);
    req.dp_rank = Some(0);
    let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, vec![req]);
    spec.engine["dp_size"] = serde_json::json!(2);
    let rank = &mut spec.engine["rank"];
    rank["enable_prefix_caching"] = false.into();
    rank["max_num_batched_tokens"] = 128.into();
    rank["kv_cache_capacity_bytes"] = 4000.into();
    rank["kv_cache_groups"] = serde_json::json!([
        {"name":"full", "kind":"attention", "num_layers":1, "block_size_tokens":64, "page_size_bytes":64},
        {"name":"window", "kind":"attention", "num_layers":2, "block_size_tokens":16, "page_size_bytes":16, "sliding_window":512},
        {"name":"conv", "kind":"convolution", "num_layers":3, "block_size_tokens":4, "page_size_bytes":8, "sliding_window":4}
    ]);
    let plain = Replayer::new(spec.clone(), ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap();
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_telemetry_observer(
            1.0,
            Box::new(RecordingTelemetryObserver {
                samples: Arc::clone(&samples),
            }),
        )
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(
        report.request_counts.total_output_tokens,
        plain.request_counts.total_output_tokens
    );
    let samples = samples.lock().unwrap();
    let mut saw_resident_pages = false;
    for sample in samples.iter() {
        assert_eq!(sample.decode_scheduler_metrics.len(), 2);
        for row in &sample.decode_scheduler_metrics {
            assert_eq!((row.total_blocks, row.inactive_blocks), (0, 0));
            assert_eq!(row.kv_cache_capacity_bytes, Some(4000));
            let used = row.kv_cache_used_bytes.unwrap();
            assert_eq!(row.active_blocks > 0, used > 0);
            assert!(used <= 4000);
            assert_eq!(row.physical_cache_usage, used as f64 / 4000.0);
            if row.dp_rank == 1 {
                assert_eq!(used, 0);
            }
            saw_resident_pages |= used > 0;
            let emitted = serde_json::to_value(row).unwrap();
            assert_eq!(emitted["kv_cache_used_bytes"], used);
            assert_eq!(emitted["kv_cache_capacity_bytes"], 4000);
        }
    }
    assert!(saw_resident_pages);
    assert!(
        samples
            .last()
            .unwrap()
            .decode_scheduler_metrics
            .iter()
            .all(|row| row.kv_cache_used_bytes == Some(0))
    );
}

#[test]
fn grouped_prefix_cache_replay_reports_reuse() {
    const GLM_PAGE: u64 = 13_045_120;
    for (version, checkpoint) in [("0.31.0", None), ("0.32.0", Some(0))] {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let turn_one: Vec<u32> = (0..9000).collect();
        let mut turn_two = turn_one.clone();
        turn_two.extend(100_000..104_000);
        let mut first = request("turn-1", 0.0, turn_one.len(), 4);
        first.input_token_ids = Some(turn_one);
        let mut second = request("turn-2", 1_000.0, turn_two.len(), 4);
        second.input_token_ids = Some(turn_two);
        let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, vec![first, second]);
        let rank = &mut spec.engine["rank"];
        rank["enable_prefix_caching"] = true.into();
        rank["prefix_match_unit"] = 2176.into();
        rank["vllm_version"] = version.into();
        rank["max_model_len"] = 131_072.into();
        rank["max_num_batched_tokens"] = 8192.into();
        rank["kv_cache_capacity_bytes"] = (74 * GLM_PAGE).into();
        let mut groups = vec![
            serde_json::json!({"name":"mla", "kind":"attention", "num_layers":22, "block_size_tokens":2176, "page_size_bytes":GLM_PAGE, "prefix_cacheable":true}),
            serde_json::json!({"name":"kpool_tail", "kind":"state", "num_layers":11, "block_size_tokens":4, "page_size_bytes":GLM_PAGE, "blocks_per_request":1, "prefix_cacheable":false}),
        ];
        for i in 0..4 {
            let mut group = serde_json::json!({"name":format!("kda_{i}"), "kind":"state", "num_layers":4, "block_size_tokens":2176, "page_size_bytes":GLM_PAGE, "blocks_per_request":1, "prefix_cacheable":true});
            if let Some(blocks) = checkpoint {
                group["prefill_checkpoint_blocks"] = blocks.into();
            }
            groups.push(group);
        }
        rank["kv_cache_groups"] = serde_json::Value::Array(groups);
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_observer(
                1.0,
                Box::new(RecordingTelemetryObserver {
                    samples: Arc::clone(&samples),
                }),
            )
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(report.request_counts.completed_requests, 2, "{version}");
        // The second turn reuses 8,704 of its 13,000 prompt tokens.
        let expected_ratio = 8704.0 / (9000.0 + 13_000.0);
        assert!(
            (report.prefix_cache_reused_ratio - expected_ratio).abs() < 1e-9,
            "{version}: {}",
            report.prefix_cache_reused_ratio
        );
        let samples = samples.lock().unwrap();
        let mut saw_cached_bytes = false;
        for sample in samples.iter() {
            for row in &sample.decode_scheduler_metrics {
                assert_eq!(row.total_blocks, 0);
                assert_eq!(row.kv_cache_capacity_bytes, Some(74 * GLM_PAGE));
                let cached = row
                    .kv_cache_cached_bytes
                    .expect("block mode reports cached bytes");
                assert_eq!(cached % GLM_PAGE, 0);
                saw_cached_bytes |= cached > 0;
                let emitted = serde_json::to_value(row).unwrap();
                assert_eq!(emitted["kv_cache_cached_bytes"], cached);
            }
        }
        assert!(saw_cached_bytes, "{version}");
    }
}

#[test]
fn grouped_prefix_hit_above_the_pool_is_a_livelock() {
    // The geometry of grouped_prefix_hit_that_needs_more_blocks_than_the_pool_waits
    // in grouped_tests.rs. vLLM's startup check counts no block of a hit
    // inside a block, so five blocks validate for a max_model_len of 34. The
    // second prompt hits 12 tokens inside a block and then needs seven
    // blocks. vLLM keeps it waiting, and with nothing running nothing evicts
    // the hit blocks. Replay reports the livelock and drops no request.
    for (version, checkpoint) in [("0.31.0", None), ("0.32.0", Some(0))] {
        let mut first = request("first", 0.0, 13, 1);
        first.input_token_ids = Some((0..13).collect());
        let mut second = request("second", 1_000.0, 33, 1);
        second.input_token_ids = Some((0..33).collect());
        let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, vec![first, second]);
        let rank = &mut spec.engine["rank"];
        rank["enable_prefix_caching"] = true.into();
        rank["prefix_match_unit"] = 4.into();
        rank["vllm_version"] = version.into();
        rank["max_model_len"] = 34.into();
        rank["kv_cache_capacity_bytes"] = (5 * 64).into();
        let mut state = serde_json::json!({"name":"state", "kind":"state", "num_layers":1, "block_size_tokens":16, "page_size_bytes":64, "blocks_per_request":1, "prefix_cacheable":true});
        if let Some(blocks) = checkpoint {
            state["prefill_checkpoint_blocks"] = blocks.into();
        }
        rank["kv_cache_groups"] = serde_json::json!([
            {"name":"full", "kind":"attention", "num_layers":1, "block_size_tokens":16, "page_size_bytes":64, "prefix_cacheable":true},
            state,
        ]);
        let error = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "replay invariant violated: offline replay detected an effect-free zero-duration pass with 1 in-flight requests remaining",
            "{version}"
        );
    }
}

#[test]
fn grouped_prefill_rank_holds_its_source_and_reports_block_unit_timing() {
    // Both roles pass the configuration gate, so the replay runs. The prefill
    // rank holds its source after 999 of the 1,000 prompt tokens, because
    // vLLM leaves the last one to the decode rank, which computes it in its
    // first pass. The prompt is committed once, split across the two ranks.
    let spec = disaggregated_grouped_spec(vec![request("grouped", 0.0, 1000, 4)]);
    let rank = serde_json::from_value::<ReplayEngineConfig>(spec.engine.clone())
        .unwrap()
        .rank;
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(report.committed_prefill_tokens, 1000);
    assert_eq!(report.request_counts.completed_requests, 1);
    let record = &report.per_request[0];
    assert_eq!(record.source_held_ms, Some(10.0));
    assert_eq!(record.decode_worker_idx, Some(0));
    assert!(record.destination_reserved_ms.is_some());
    assert_eq!(
        record.terminal_status,
        aisimulate_core::replay::ReplayTerminalStatus::Completed
    );

    // The report keeps neither the held timing nor the transfer units, so
    // the same rank shows both through the engine API.
    let engine = |worker_type| {
        let config = EngineConfig {
            worker_type,
            ..rank.clone()
        };
        EngineFactory::new(config)
            .unwrap()
            .build(EngineIdentity::new(0), NonZeroU32::MIN)
            .unwrap()
    };
    let handoff_id = HandoffId::new(Uuid::from_u128(2));
    let request = Request {
        request_id: Uuid::from_u128(1),
        tokens: vec![7; 1000],
        max_output_tokens: 4,
        output_token_ids: None,
        session_id: None,
        cache_salt: None,
        images: Vec::new(),
        kv_hint: None,
    };
    let mut prefill = engine(WorkerType::Prefill);
    let source = Command::SubmitHandoffPrefill {
        handoff_id,
        request: Request {
            max_output_tokens: 1,
            ..request.clone()
        },
    };
    prefill
        .apply_command_effects(SchedulerCommand::new(0, source), 0.0)
        .unwrap();
    let pass = prefill.execute_pass(0.0).unwrap().unwrap();
    let completed = prefill.complete_pass(pass.pass_id, pass.end_ms).unwrap();
    let timing = completed.effects.by_rank[0]
        .effects
        .lifecycle_events
        .iter()
        .find_map(|event| match event {
            LifecycleEvent::SourceHeld {
                transfer_timing, ..
            } => Some(*transfer_timing),
            _ => None,
        })
        .expect("the prefill rank holds its source");
    // The 999 tokens fill one full-attention block of 13,045,120 bytes. The
    // state bytes are one block of each state group.
    assert_eq!(timing.full_prompt_tokens, 1);
    assert_eq!(timing.kv_bytes_per_token, Some(13_045_120));
    assert_eq!(timing.state_bytes, 36_927_488);
    // The decode rank reserves every group for the 999 loaded tokens and
    // reports the one full-attention block it lacks in the same unit.
    let destination = Command::ReserveDestination {
        handoff_id,
        request,
    };
    let effects = engine(WorkerType::Decode)
        .apply_command_effects(SchedulerCommand::new(0, destination), 0.0)
        .unwrap();
    let units = effects.by_rank[0]
        .effects
        .lifecycle_events
        .iter()
        .find_map(|event| match event {
            LifecycleEvent::DestinationReserved {
                transferable_prompt_tokens,
                ..
            } => Some(*transferable_prompt_tokens),
            _ => None,
        });
    assert_eq!(units, Some(1));
}

#[test]
fn grouped_and_linear_roles_cannot_pair_in_one_handoff() {
    // A grouped rank counts the handoff in full-attention blocks and a linear
    // rank in tokens, so a grouped prefill rank with a linear decode rank, or
    // the reverse, would price the transfer in mixed units.
    let linear = serde_json::to_value(EngineConfig {
        num_gpu_blocks: 1024,
        block_size: 4,
        max_num_seqs: 4,
        max_num_batched_tokens: 64,
        kv_transfer_bytes_per_token: Some(16),
        kv_transfer_bandwidth: Some(1.0),
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 10.0,
            decode_ms: 2.0,
        },
        ..EngineConfig::for_backend(Backend::Vllm)
    })
    .unwrap();
    for role in ["prefill", "decode"] {
        let mut spec = disaggregated_grouped_spec(vec![request("mixed", 0.0, 1000, 4)]);
        spec.engine[role] = serde_json::json!({ "rank": linear });
        let Err(error) = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
        else {
            panic!("a linear {role} rank next to a grouped rank must be refused");
        };
        assert_eq!(
            error.to_string(),
            "invalid replay spec: kv_cache_groups must be identical on the prefill and decode roles",
            "linear {role} rank"
        );
    }
}

/// `disaggregated_grouped_spec` with a full-attention group and a Mamba-type
/// state group of 16-token blocks and 100-byte pages, a pool of `pages` pages
/// and no transfer delay. With `max_model_len`, the rank caches prefixes at a
/// 16-token hash block, so it runs in block mode.
fn small_grouped_pd_spec(
    requests: Vec<ReplayRequest>,
    pages: u64,
    max_model_len: Option<usize>,
) -> ReplaySpec {
    let mut spec = disaggregated_grouped_spec(requests);
    let rank = &mut spec.engine["rank"];
    rank["kv_cache_groups"] = serde_json::json!([
        {"name":"full", "kind":"attention", "num_layers":1, "block_size_tokens":16, "page_size_bytes":100, "prefix_cacheable":true, "transfer_bytes_per_block":100},
        {"name":"state", "kind":"state", "num_layers":1, "block_size_tokens":16, "page_size_bytes":100, "blocks_per_request":1, "prefix_cacheable":true, "transfer_bytes_per_block":100},
    ]);
    rank["kv_cache_capacity_bytes"] = (pages * 100).into();
    rank["kv_transfer_bandwidth"] = serde_json::Value::Null;
    if let Some(max_model_len) = max_model_len {
        rank["enable_prefix_caching"] = true.into();
        rank["prefix_match_unit"] = 16.into();
        rank["max_model_len"] = max_model_len.into();
    }
    spec
}

fn record<'a>(
    report: &'a aisimulate_core::replay::ReplayReport,
    id: &str,
) -> &'a aisimulate_core::replay::PerRequestRecord {
    report
        .per_request
        .iter()
        .find(|record| record.request_id.as_deref() == Some(id))
        .unwrap()
}

/// Runs `spec` on its own thread, so that a livelock fails the caller after
/// 60 s instead of hanging it.
fn run_within_60_s(spec: ReplaySpec) -> aisimulate_core::replay::ReplayReport {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        sender.send(report).unwrap();
    });
    receiver
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the replay ends within 60 s")
}

/// The times at which the decode rank admitted a request, readmissions
/// included.
fn decode_admissions(record: &aisimulate_core::replay::PerRequestRecord) -> Vec<f64> {
    record
        .admission_history
        .iter()
        .filter(|admission| admission.pool == ReplayRequestPool::Decode)
        .map(|admission| admission.at_ms)
        .collect()
}

#[test]
fn grouped_destination_waits_for_room_for_its_whole_prompt() {
    // Two distinct prompts of 33 tokens on a pool of 6 pages, in byte mode
    // and in block mode. Each load of 32 tokens takes 2 attention blocks and
    // 1 state block, and the whole prompt needs a third attention block.
    // vLLM admits a load only when the whole prompt fits, with the loaded
    // tokens computed (V32 `vllm/v1/core/kv_cache_manager.py:522-539`): 4
    // pages. Two loads of 3 pages would fill the pool, and neither request
    // could compute its last prompt token. Both prefills end at 10 ms. The
    // first load is reserved then and activated after the 1 ms handoff, and
    // its two decode passes of 2 ms end at 15 ms. Until then at most 3 pages
    // are free, so the second load is reserved at 15 ms.
    for max_model_len in [None, Some(64)] {
        let mut first = request("first", 0.0, 33, 2);
        first.input_token_ids = Some((0..33).collect());
        let mut second = request("second", 0.0, 33, 2);
        second.input_token_ids = Some((1000..1033).collect());
        let spec = small_grouped_pd_spec(vec![first, second], 6, max_model_len);
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 2,
            "{max_model_len:?}"
        );
        let (first, second) = (record(&report, "first"), record(&report, "second"));
        assert_eq!(
            first.destination_reserved_ms,
            Some(10.0),
            "{max_model_len:?}"
        );
        assert_eq!(first.terminal_time_ms, 15.0, "{max_model_len:?}");
        assert_eq!(
            second.destination_reserved_ms,
            Some(15.0),
            "{max_model_len:?}"
        );
    }
}

#[test]
fn grouped_destination_retries_when_an_activation_publishes_its_hit() {
    // Two identical prompts of 81 tokens on a decode pool of 11 pages in
    // block mode. A prefill rank without prefix caching and with room for
    // both ends both prefills at 10 ms, so both loads arrive then. The first
    // load takes 5 attention blocks and 1 state block. The second needs 7
    // free pages for its whole prompt without a hit and finds 5, so it
    // waits. The first load activates after the 1 ms handoff, at 11 ms, and
    // publishes its 5 attention blocks. The second prompt then hits them and
    // needs 1 attention block and 1 state block, so it is reserved at 11 ms,
    // not when the first request finishes at 27 ms (11 ms and 8 decode passes
    // of 2 ms).
    let prompt: Vec<u32> = (0..81).collect();
    let requests = ["first", "second"].map(|id| {
        let mut request = request(id, 0.0, 81, 8);
        request.input_token_ids = Some(prompt.clone());
        request
    });
    let mut spec = small_grouped_pd_spec(requests.to_vec(), 11, Some(144));
    let mut prefill = spec.engine["rank"].clone();
    prefill["enable_prefix_caching"] = false.into();
    prefill["prefix_match_unit"] = serde_json::Value::Null;
    prefill["kv_cache_capacity_bytes"] = (64 * 100).into();
    spec.engine["prefill"] = serde_json::json!({ "rank": prefill });
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 2);
    let (first, second) = (record(&report, "first"), record(&report, "second"));
    assert_eq!(second.source_held_ms, Some(10.0));
    assert_eq!(first.destination_activated_ms, Some(11.0));
    assert_eq!(first.terminal_time_ms, 27.0);
    assert_eq!(second.destination_reserved_ms, Some(11.0));
}

#[test]
fn grouped_destination_keeps_room_for_the_first_pass_of_each_load() {
    // The two prompts of 33 tokens on a pool of 7 pages. Both prefills end at
    // 10 ms, and the first load of 3 pages is reserved then. Its first pass
    // takes a third attention block. In block mode it also takes a new state
    // block, as token 33 crosses a state block boundary. vLLM admits a load
    // only when its blocks fit the free blocks minus what the other loads in
    // flight still need (V32 `vllm/v1/core/kv_cache_manager.py:561-579`). In
    // byte mode the second load of 3 pages and the 1 page of the first pass
    // fit the 4 free pages, so the second load is reserved at 10 ms. In block
    // mode the first pass needs 2, so the second load waits. The first load
    // activates at 11 ms, and its first pass leaves 2 free pages. The next
    // pass frees the previous state block, so 3 are free, less than the 4 of
    // the whole prompt. The first request ends at 15 ms, and the second load
    // is reserved then. In byte mode both loads activate at 11 ms. The first
    // pass of the first request takes the last free page, so the first pass
    // of the second request waits, as vLLM's waiting loop stops there
    // (V32 `vllm/v1/core/sched/scheduler.py:1275-1282`). That pass runs when
    // the first request ends at 15 ms, and the second request ends after two
    // decode passes of 2 ms, at 19 ms. In block mode the second load
    // activates at 16 ms and ends at 20 ms. No request computes its prompt
    // again.
    for (max_model_len, second_reserved_ms, second_end_ms) in
        [(None, 10.0, 19.0), (Some(64), 15.0, 20.0)]
    {
        let mut first = request("first", 0.0, 33, 2);
        first.input_token_ids = Some((0..33).collect());
        let mut second = request("second", 0.0, 33, 2);
        second.input_token_ids = Some((1000..1033).collect());
        let spec = small_grouped_pd_spec(vec![first, second], 7, max_model_len);
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 2,
            "{max_model_len:?}"
        );
        let (first, second) = (record(&report, "first"), record(&report, "second"));
        assert_eq!(
            first.destination_reserved_ms,
            Some(10.0),
            "{max_model_len:?}"
        );
        assert_eq!(
            second.destination_reserved_ms,
            Some(second_reserved_ms),
            "{max_model_len:?}"
        );
        assert_eq!(first.terminal_time_ms, 15.0, "{max_model_len:?}");
        assert_eq!(second.terminal_time_ms, second_end_ms, "{max_model_len:?}");
        assert_eq!(
            (first.readmission_count, second.readmission_count),
            (0, 0),
            "{max_model_len:?}"
        );
    }
}

#[test]
fn grouped_destination_keeps_room_for_a_first_pass_in_byte_mode_too() {
    // A first prompt of 33 tokens and a distinct second prompt of 34 tokens on
    // a decode pool of 7 pages. A prefill rank without prefix caching and with
    // room for both ends both prefills at 10 ms. The first load holds 3 pages,
    // and its first pass takes 1 more, 2 in block mode. The second load holds
    // 4 pages for its 33 tokens, and its first pass takes none. So its whole
    // prompt fits the 4 free pages, but the load would leave none for the
    // first pass. It waits in both modes until the first request ends at
    // 15 ms. If it took them, the first pass of the first request would never
    // fit, and the second request would wait behind it.
    for max_model_len in [None, Some(64)] {
        let mut first = request("first", 0.0, 33, 2);
        first.input_token_ids = Some((0..33).collect());
        let mut second = request("second", 0.0, 34, 2);
        second.input_token_ids = Some((1000..1034).collect());
        let mut spec = small_grouped_pd_spec(vec![first, second], 7, max_model_len);
        let mut prefill = spec.engine["rank"].clone();
        prefill["enable_prefix_caching"] = false.into();
        prefill["prefix_match_unit"] = serde_json::Value::Null;
        prefill["kv_cache_capacity_bytes"] = (64 * 100).into();
        spec.engine["prefill"] = serde_json::json!({ "rank": prefill });
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 2,
            "{max_model_len:?}"
        );
        let (first, second) = (record(&report, "first"), record(&report, "second"));
        assert_eq!(second.source_held_ms, Some(10.0), "{max_model_len:?}");
        assert_eq!(first.terminal_time_ms, 15.0, "{max_model_len:?}");
        assert_eq!(
            second.destination_reserved_ms,
            Some(15.0),
            "{max_model_len:?}"
        );
    }
}

#[test]
fn grouped_destination_burst_completes_without_readmission() {
    // Four distinct prompts of 2,177 tokens on a GLM-5.3-Flash decode pool of
    // 16 pages, in byte mode and in block mode. A load holds 6 blocks for its
    // 2,176 tokens. In block mode its first pass takes 5 more: an attention
    // block and a new block of each KDA group. Two loads would leave 4 free
    // blocks, and neither first pass would fit. In byte mode a first pass
    // takes only the attention block.
    for prefix_caching in [false, true] {
        let requests = (0..4)
            .map(|i| {
                let mut request = request(&format!("r{i}"), 0.0, 2177, 2);
                request.input_token_ids = Some((i * 10_000..i * 10_000 + 2177).collect());
                request
            })
            .collect();
        let mut spec = disaggregated_grouped_spec(requests);
        spec.engine["prefill"] = serde_json::json!({ "rank": spec.engine["rank"].clone() });
        let rank = &mut spec.engine["rank"];
        let page = rank["kv_cache_groups"][0]["page_size_bytes"]
            .as_u64()
            .unwrap();
        if prefix_caching {
            rank["enable_prefix_caching"] = true.into();
            rank["prefix_match_unit"] = 2176.into();
        }
        rank["max_model_len"] = 4352.into();
        rank["kv_cache_capacity_bytes"] = (16 * page).into();
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 4,
            "{prefix_caching}"
        );
        assert!(
            report
                .per_request
                .iter()
                .all(|record| record.readmission_count == 0),
            "{prefix_caching}"
        );
    }
}

#[test]
fn grouped_decode_rank_serves_a_loaded_handoff_before_a_preempted_request() {
    // Two distinct prompts of 33 tokens with long outputs: 17 tokens each in
    // byte mode on 7 pages, and 20 tokens each in block mode on 8 pages. The
    // first request outgrows the pool while the first pass of the second
    // request waits, so it preempts itself. A preempted request goes to the
    // front of the waiting queue without its blocks, and vLLM serves the
    // requests that hold KV blocks first (V32
    // `vllm/v1/core/sched/scheduler.py:913-917`). So the second request runs,
    // and the first request computes its prompt again.
    for (max_model_len, pages, output) in [(None, 7, 17), (Some(80), 8, 20)] {
        let mut first = request("first", 0.0, 33, output);
        first.input_token_ids = Some((0..33).collect());
        let mut second = request("second", 0.0, 33, output);
        second.input_token_ids = Some((1000..1033).collect());
        let spec = small_grouped_pd_spec(vec![first, second], pages, max_model_len);
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(
            report.request_counts.completed_requests, 2,
            "{max_model_len:?}"
        );
        let (first, second) = (record(&report, "first"), record(&report, "second"));
        assert_eq!(
            (first.readmission_count, second.readmission_count),
            (1, 0),
            "{max_model_len:?}"
        );
    }
}

#[test]
fn grouped_decode_rank_serves_a_loaded_glm_handoff_before_a_preempted_request() {
    // Two distinct GLM-5.3-Flash prompts of 2,177 tokens with 2,200 output
    // tokens each, in byte mode as the P/D compiler sets it, on a decode pool
    // of 13 pages. Each load holds 6 pages, and each first pass takes the
    // second attention page. Both loads fit, the first pass of the second
    // request waits for the last page, and the first request needs a third
    // attention page at 4,352 tokens. It preempts itself, and the loaded
    // second request runs before it.
    let requests = (0..2)
        .map(|i| {
            let mut request = request(&format!("r{i}"), 0.0, 2177, 2200);
            request.input_token_ids = Some((i * 10_000..i * 10_000 + 2177).collect());
            request
        })
        .collect();
    let mut spec = disaggregated_grouped_spec(requests);
    spec.engine["prefill"] = serde_json::json!({ "rank": spec.engine["rank"].clone() });
    let rank = &mut spec.engine["rank"];
    let page = rank["kv_cache_groups"][0]["page_size_bytes"]
        .as_u64()
        .unwrap();
    rank["max_model_len"] = 8704.into();
    rank["kv_cache_capacity_bytes"] = (13 * page).into();
    let report = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(
        (
            record(&report, "r0").readmission_count,
            record(&report, "r1").readmission_count
        ),
        (1, 0)
    );
}

#[test]
fn grouped_decode_rank_waits_for_an_event_after_a_repeated_self_preemption() {
    // Prompts of 84, 90 and 14 tokens with 8 output tokens each, on pools of
    // 17 pages in block mode, with handoffs at 1e-5 GB/s, 10 bytes per ms.
    // The prefill rank computes 13, 83 and 89 tokens. It holds h14 at 10 ms
    // and h84 at 20 ms. At 10 ms h90 does not fit beside h84 and preempts
    // itself. It is readmitted at 20 ms with its 80 cached tokens and held
    // at 30 ms. The transfers of 200, 700 and 700 bytes share the links
    // fairly. h14 sends 100 bytes alone, 50 beside h84 and its last 50
    // beside both loads, by 45 ms. h84 sends its last 600 bytes at 5 bytes
    // per ms by 165 ms, and h90 its last 50 bytes by 170 ms. h90 activates
    // when the decode pass that runs at 170 ms ends, at 171 ms.
    //
    // h14 holds 2 blocks and the loads in flight hold 14, so 1 block is
    // free. h14 emits outputs at 47, 49 and 51 ms. Token 17 then needs an
    // attention block and a new state block, so h14 preempts itself. Its
    // readmission needs 3 blocks and finds 3. Its 16-token chunk takes 2 at
    // 0 ms, as a decode rank prices no prefill, and the next pass preempts
    // it again with the same pool. vLLM repeats that turn until a transfer
    // ends. Replay lets h14 wait for the next event, the activation of h84
    // at 165 ms. Then h84 decodes every 2 ms, and h14 is readmitted in every
    // other pass until h84 ends at 181 ms. The pass at 171 ms preempts h14
    // and admits nothing, so h90 runs from 173 ms and ends at 189 ms. h14
    // emits its last five outputs from 185 to 193 ms.
    let requests = [84u32, 90, 14]
        .iter()
        .enumerate()
        .map(|(i, &prompt)| {
            let mut request = request(&format!("h{prompt}"), 0.0, prompt as usize, 8);
            let first = 10_000 * (i as u32 + 1);
            request.input_token_ids = Some((first..first + prompt).collect());
            request
        })
        .collect();
    let mut spec = small_grouped_pd_spec(requests, 17, Some(128));
    spec.engine["rank"]["kv_transfer_bandwidth"] = 1e-5.into();
    let report = run_within_60_s(spec);
    assert_eq!(report.request_counts.completed_requests, 3);
    let (h84, h90, h14) = (
        record(&report, "h84"),
        record(&report, "h90"),
        record(&report, "h14"),
    );
    assert_eq!(
        [h14, h84, h90].map(|record| record.destination_activated_ms),
        [Some(45.0), Some(165.0), Some(171.0)]
    );
    assert_eq!(
        decode_admissions(h14),
        [45.0, 51.0, 165.0, 169.0, 173.0, 177.0, 181.0]
    );
    assert_eq!((h14.terminal_time_ms, h14.readmission_count), (193.0, 6));
    assert_eq!(
        [h84, h90].map(|record| (record.terminal_time_ms, record.readmission_count)),
        [(181.0, 0), (189.0, 1)]
    );
}

/// Distinct prompts of the given (prompt, output, arrival) shapes, with the
/// token IDs of request i starting at 1,000 * (i + 1).
fn distinct_requests(shapes: &[(usize, usize, f64)]) -> Vec<ReplayRequest> {
    shapes
        .iter()
        .enumerate()
        .map(|(i, &(prompt, output, at))| {
            let mut request = request(&format!("q{i}"), at, prompt, output);
            let first = 1000 * (i as u32 + 1);
            request.input_token_ids = Some((first..first + prompt as u32).collect());
            request
        })
        .collect()
}

#[test]
fn grouped_decode_rank_waits_when_a_turn_evicts_and_publishes_the_same_key() {
    // Three distinct prompts of 31, 30 and 34 tokens with 40, 31 and 20
    // outputs, arriving at 0, 3 and 6 ms, on pools of 13 pages in block mode,
    // with a 32-token budget, 0.5 ms decode passes and handoffs at
    // 3e-7 GB/s. At 2,992 ms q0 has computed 64 of its 65 tokens and needs 2
    // more blocks, while two loads in flight hold 7. It preempts itself and
    // recomputes its tokens in 0 ms chunks with no hit. Each turn publishes
    // one key and loses one, so q0 preempts itself again with 7 blocks in
    // use, 1 cached block and the same visible key, while a count of
    // publications grows by one per turn. q0 waits for the next event, and
    // the three requests complete in both timing modes.
    for timing_mode in ["full_prompt", "destination_missing"] {
        let requests = distinct_requests(&[(31, 40, 0.0), (30, 31, 3.0), (34, 20, 6.0)]);
        let mut spec = small_grouped_pd_spec(requests, 13, Some(128));
        let rank = &mut spec.engine["rank"];
        rank["kv_transfer_bandwidth"] = 3e-7.into();
        rank["timing_model"] =
            serde_json::json!({"type": "fixed", "prefill_ms": 10.0, "decode_ms": 0.5});
        rank["max_num_batched_tokens"] = 32.into();
        rank["kv_transfer_timing_mode"] = timing_mode.into();
        let report = run_within_60_s(spec);
        assert_eq!(report.request_counts.completed_requests, 3, "{timing_mode}");
    }
}

#[test]
fn grouped_decode_rank_readmits_a_request_whose_pool_lost_a_visible_key() {
    // Four distinct GLM-5.3-Flash prompts of 2,178, 6,000, 1,000 and 4,354
    // tokens, with decode prefix caching on 16 decode pages, a 128-page
    // prefill pool, a 2,176-token budget, handoffs at 0.01 GB/s and 5 ms
    // decode passes. When g0 ends, g3 is readmitted and preempts itself three
    // times at that time, always with 6 blocks in use. The second and the
    // third time also have 5 cached blocks each, but one of the 3 visible
    // keys left the pool between them. So g3 makes a fourth turn, as vLLM
    // does, and that turn caches 4 more blocks. When g2 ends, the
    // readmission of g3 then hits 2,176 tokens.
    let requests = [
        (2178, 2500, 126.0),
        (6000, 50, 8.0),
        (1000, 1, 155.0),
        (4354, 2200, 130.0),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (prompt, output, at))| {
        let mut request = request(&format!("g{i}"), at, prompt, output);
        let first = i as u32 * 100_000;
        request.input_token_ids = Some((first..first + prompt as u32).collect());
        request
    })
    .collect();
    let mut spec = disaggregated_grouped_spec(requests);
    let rank = &mut spec.engine["rank"];
    let page = rank["kv_cache_groups"][0]["page_size_bytes"]
        .as_u64()
        .unwrap();
    rank["max_model_len"] = 8704.into();
    rank["enable_prefix_caching"] = true.into();
    rank["prefix_match_unit"] = 2176.into();
    rank["max_num_batched_tokens"] = 2176.into();
    rank["kv_transfer_bandwidth"] = 0.01.into();
    rank["timing_model"] =
        serde_json::json!({"type": "fixed", "prefill_ms": 10.0, "decode_ms": 5.0});
    let mut prefill = spec.engine["rank"].clone();
    prefill["kv_cache_capacity_bytes"] = (128 * page).into();
    spec.engine["prefill"] = serde_json::json!({ "rank": prefill });
    spec.engine["rank"]["kv_cache_capacity_bytes"] = (16 * page).into();
    let report = run_within_60_s(spec);
    assert_eq!(report.request_counts.completed_requests, 4);
    let (g0, g2, g3) = (
        record(&report, "g0"),
        record(&report, "g2"),
        record(&report, "g3"),
    );
    let admissions = decode_admissions(g3);
    assert_eq!(
        admissions
            .iter()
            .filter(|&&at| at == g0.terminal_time_ms)
            .count(),
        4
    );
    assert_eq!(admissions.last(), Some(&g2.terminal_time_ms));
    assert_eq!(g3.decode_reused_input_tokens, Some(2176));
}

#[test]
fn grouped_decode_rank_readmits_a_request_that_made_progress_at_one_time() {
    // Five distinct prompts on a decode pool of 14 pages in block mode, with
    // decode passes priced at 0 ms, a 32-token budget, a 160-page prefill
    // pool and handoffs at 3e-6 GB/s. q1 and q2 activate at 283.33 ms, and
    // every step at that time costs 0 ms. q2 preempts itself after 2 outputs,
    // with 11 blocks in use and no cached block. Readmitted, it emits 14 more
    // outputs before q1 preempts it, and after its next readmission it
    // preempts itself again after 16 outputs, with the same pool. It made
    // progress in between, so it is readmitted. Only its third
    // self-preemption repeats the second, and q2 waits until q1 ends at the
    // same time. So q2 has 4 readmissions, all at its activation time.
    let requests = distinct_requests(&[
        (14, 33, 0.0),
        (15, 40, 3.0),
        (15, 40, 6.0),
        (33, 3, 9.0),
        (34, 20, 12.0),
    ]);
    let mut spec = small_grouped_pd_spec(requests, 14, Some(96));
    let rank = &mut spec.engine["rank"];
    rank["kv_transfer_bandwidth"] = 3e-6.into();
    rank["timing_model"] =
        serde_json::json!({"type": "fixed", "prefill_ms": 10.0, "decode_ms": 0.0});
    rank["max_num_batched_tokens"] = 32.into();
    rank["kv_transfer_timing_mode"] = "destination_missing".into();
    let mut prefill = spec.engine["rank"].clone();
    prefill["kv_cache_capacity_bytes"] = (160 * 100).into();
    spec.engine["prefill"] = serde_json::json!({ "rank": prefill });
    let report = run_within_60_s(spec);
    assert_eq!(report.request_counts.completed_requests, 5);
    let (q1, q2) = (record(&report, "q1"), record(&report, "q2"));
    let activated = q2.destination_activated_ms.unwrap();
    assert_eq!(decode_admissions(q2), [activated; 5]);
    assert_eq!(q1.terminal_time_ms, activated);
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
            (Some("1,1"), vec![8], [3, 3], Some(3.0)),
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
            // This fixed 5 ms sample window contains both 2 ms decode rounds
            // for each handoff. vLLM starts with one draft-free base token,
            // then accepts three tokens, so its per-forward mean is (1+3)/2.
            // SGLang's existing handoff behavior remains three per forward.
            let expected_acceptance = if backend == Backend::Vllm && disaggregated {
                2.0
            } else {
                3.0
            };
            for sample in samples.iter() {
                let traffic = &sample.traffic;
                assert_eq!(
                    traffic.avg_accept_length,
                    (traffic.accept_length_forward_count > 0).then_some(expected_acceptance),
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

#[derive(Default)]
struct DispatchProbe {
    calls: Vec<(&'static str, f64)>,
    fail: bool,
}

#[derive(Default)]
struct FailingPlacement {
    probe: Option<Rc<RefCell<DispatchProbe>>>,
    pending: Vec<Placement>,
    selected: bool,
    committed: bool,
    queue_on_clock: bool,
}

impl PlacementPolicy<ReplayRequestPayload> for FailingPlacement {
    type Metadata = NoReplayMetadata;
    type Observation = ();

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> anyhow::Result<PlacementEffects> {
        anyhow::ensure!(self.probe.is_some(), "router placement callback failed");
        let placement = Placement {
            request_id: request.metadata().uuid.unwrap(),
            scheduler_id: 0,
            reported_overlap_tokens: 0,
            cache_sample: None,
            kv_hints: None,
            placement_replica_id: None,
        };
        let decision = if self.selected && (!self.committed || self.queue_on_clock) {
            self.pending.push(placement);
            aisimulate_core::replay::PlacementDecision::Queued
        } else {
            self.selected = true;
            aisimulate_core::replay::PlacementDecision::Immediate(placement)
        };
        Ok(PlacementEffects {
            decision,
            released: Vec::new(),
        })
    }

    fn dispatch_committed(&mut self, _: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let mut probe = self.probe.as_ref().unwrap().borrow_mut();
        probe.calls.push(("commit", now_ms));
        anyhow::ensure!(!probe.fail, "commit failed");
        self.committed = true;
        Ok(())
    }

    fn dispatch_aborted(&mut self, _: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let mut probe = self.probe.as_ref().unwrap().borrow_mut();
        probe.calls.push(("abort", now_ms));
        anyhow::ensure!(!probe.fail, "abort failed");
        Ok(())
    }

    fn advance_clock(&mut self, now_ms: f64) -> anyhow::Result<Vec<Placement>> {
        if self.committed && !self.pending.is_empty() {
            let event = if self.queue_on_clock {
                "decode_release"
            } else {
                "release"
            };
            self.probe
                .as_ref()
                .unwrap()
                .borrow_mut()
                .calls
                .push((event, now_ms));
            return Ok(std::mem::take(&mut self.pending));
        }
        Ok(Vec::new())
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
        self.pending.len()
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

#[derive(Default)]
struct FailingPlacementComposition {
    fail_construction: bool,
    probe: Option<Rc<RefCell<DispatchProbe>>>,
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
        Ok(FailingPlacement {
            probe: self.probe.clone(),
            ..Default::default()
        })
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
        Ok((
            FailingPlacement {
                probe: self.probe.clone(),
                ..Default::default()
            },
            FailingPlacement {
                probe: self.probe.clone(),
                queue_on_clock: true,
                ..Default::default()
            },
        ))
    }
}

#[test]
fn dispatch_hooks_commit_after_tokenization_and_release_waiters_at_the_same_time() {
    for disagg in [false, true] {
        for fail in [false, true] {
            let probe = Rc::new(RefCell::new(DispatchProbe {
                fail,
                ..Default::default()
            }));
            let mut spec = aggregated_spec(
                Backend::Vllm,
                1,
                0.0,
                vec![request("first", 0.0, 4, 2), request("waiter", 0.0, 4, 2)],
            );
            if disagg {
                spec.topology = ReplayTopology::Disaggregated {
                    prefill: WorkerPoolSpec::default(),
                    decode: WorkerPoolSpec::default(),
                    handoff_latency_ms: 0.0,
                };
            }
            let result = Replayer::with_composition(
                spec,
                ReplayEngineFactory::new(),
                FailingPlacementComposition {
                    probe: Some(probe.clone()),
                    ..Default::default()
                },
            )
            .unwrap()
            .with_tokenization(1, 3.0, 0.0)
            .unwrap()
            .run();
            let calls = &probe.borrow().calls;
            if fail {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains("commit failed") && error.contains("abort failed"),
                    "{error}"
                );
                assert_eq!(
                    calls.iter().map(|call| call.0).collect::<Vec<_>>(),
                    ["commit", "abort"]
                );
            } else {
                assert_eq!(result.unwrap().request_counts.completed_requests, 2);
                assert_eq!(
                    calls.iter().filter(|call| call.0 == "commit").count(),
                    if disagg { 4 } else { 2 }
                );
                assert!(calls.contains(&("commit", 3.0)));
                assert!(calls.contains(&("release", 3.0)));
                assert!(calls.contains(&("commit", 6.0)));
                if disagg {
                    assert!(calls.contains(&("decode_release", 23.0)), "{calls:?}");
                }
            }
        }
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
            ..Default::default()
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
            ..Default::default()
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
fn authored_kv_hint_in_a_replay_spec_does_not_change_the_report() {
    let run = |kv_hints: Option<AuthoredKvHints>| {
        let requests = (0..4)
            .map(|index| ReplayRequest {
                kv_hints,
                ..request(&format!("request-{index}"), f64::from(index) * 5.0, 8, 3)
            })
            .collect();
        let mut spec = aggregated_spec(Backend::Vllm, 2, 0.0, requests);
        spec.engine["dp_size"] = serde_json::json!(2);
        let mut report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_capture_options(canonical_options())
            .run()
            .unwrap();
        report.throughput.wall_time_ms = 0.0;
        // The compact summary omits per-request records, so compare them too.
        (
            serde_json::to_value(&report).unwrap(),
            serde_json::to_value(&report.per_request).unwrap(),
        )
    };

    // Without `kvcr` no engine acts on a hint, so a valid hint changes nothing.
    let hinted = run(Some(AuthoredKvHints {
        source_worker: 1,
        source_dp_rank: 0,
    }));
    assert_eq!(hinted, run(None));
    assert_eq!(hinted.0["completed_requests"], 4);
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
    ready_workers: Vec<usize>,
    place_immediately: bool,
}

impl ReleaseOnTopologyCallbacks {
    fn new(topology: Vec<WorkerTopology>) -> Self {
        let stable_scheduler_id = topology[0].scheduler_ids[0];
        Self {
            pending: VecDeque::new(),
            stable_scheduler_id,
            released_on_settled: false,
            ready_workers: topology.iter().map(|worker| worker.worker_id).collect(),
            place_immediately: false,
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
                    kv_hints: None,
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
        if self.place_immediately {
            return Ok(PlacementEffects {
                decision: aisimulate_core::replay::PlacementDecision::Immediate(
                    self.release_next(self.stable_scheduler_id).pop().unwrap(),
                ),
                released: Vec::new(),
            });
        }
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
        self.ready_workers.push(worker.worker_id);
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
        worker: WorkerTopology,
        _now_ms: f64,
    ) -> anyhow::Result<Vec<Placement>> {
        let index = self
            .ready_workers
            .iter()
            .position(|id| *id == worker.worker_id)
            .expect("only registered workers receive removal callbacks");
        self.ready_workers.swap_remove(index);
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
    type DisaggregatedPlacement = ReleaseOnTopologyCallbacks;

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
            ReleaseOnTopologyCallbacks {
                place_immediately: true,
                ..ReleaseOnTopologyCallbacks::new(prefill_topology)
            },
            ReleaseOnTopologyCallbacks {
                place_immediately: true,
                ..ReleaseOnTopologyCallbacks::new(decode_topology)
            },
        ))
    }

    fn take_scaling_policy(&mut self) -> anyhow::Result<Option<Box<dyn ReplayScalingPolicy>>> {
        Ok(self.scaling.take())
    }
}

struct AddThenRemoveWorker {
    step: usize,
    removal_at_ms: f64,
    scale_prefill: bool,
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
                target_prefill: self.scale_prefill.then_some(2),
                target_decode: Some(2),
                next_tick_ms: Some(self.removal_at_ms),
                ..ReplayScalingDecision::default()
            },
            1 => ReplayScalingDecision {
                target_prefill: self.scale_prefill.then_some(1),
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
            scaling: Some(Box::new(AddThenRemoveWorker {
                step: 0,
                removal_at_ms: 1.0,
                scale_prefill: false,
            })),
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

#[test]
fn canceled_startup_is_captured_without_unregistering_an_unready_worker() {
    for disaggregated in [false, true] {
        let mut spec = aggregated_spec(
            Backend::Vllm,
            1,
            100.0,
            vec![request("long-running", 0.0, 4, 2)],
        );
        let mut engine: ReplayEngineConfig = serde_json::from_value(spec.engine.clone()).unwrap();
        engine.rank.timing_model = TimingModelConfig::Fixed {
            prefill_ms: 150.0,
            decode_ms: 50.0,
        };
        spec.engine = serde_json::to_value(engine).unwrap();
        if disaggregated {
            spec.topology = ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec {
                    initial_workers: 1,
                    startup_delay_ms: 100.0,
                },
                decode: WorkerPoolSpec {
                    initial_workers: 1,
                    startup_delay_ms: 100.0,
                },
                handoff_latency_ms: 0.0,
            };
        }
        spec.adapters.scaling = ProviderSpec {
            provider: "topology_callback_test".to_string(),
            config: serde_json::Value::Null,
        };
        let report = Replayer::with_composition(
            spec,
            ReplayEngineFactory::new(),
            TopologyReleaseComposition {
                scaling: Some(Box::new(AddThenRemoveWorker {
                    step: 0,
                    removal_at_ms: 50.0,
                    scale_prefill: disaggregated,
                })),
            },
        )
        .unwrap()
        .with_capture_options(ReplayCaptureOptions {
            capture_lifecycle_evidence: true,
            ..ReplayCaptureOptions::default()
        })
        .run()
        .unwrap();

        assert_eq!(report.request_counts.completed_requests, 1);
        let expected_duration_ms = if disaggregated { 250.0 } else { 200.0 };
        assert_eq!(report.throughput.duration_ms, expected_duration_ms);
        let operations = &report.runtime_evidence.lifecycle_operations;
        let pools = if disaggregated {
            vec![WorkerPool::Prefill, WorkerPool::Decode]
        } else {
            vec![WorkerPool::Agg]
        };
        assert_eq!(operations.len(), 2 * pools.len());
        for pool in pools {
            let pool_operations = operations
                .iter()
                .filter(|operation| operation.pool == pool)
                .collect::<Vec<_>>();
            assert_eq!(pool_operations.len(), 2);
            let start = pool_operations[0];
            assert_eq!(start.at_ms, 0.0);
            assert_eq!(start.transitions.len(), 1);
            assert_eq!(start.transitions[0].worker_id, 1);
            assert_eq!(
                start.transitions[0].transition,
                WorkerLifecycleTransitionKind::WorkerStarting
            );
            let cancellation = pool_operations[1];
            assert_eq!(cancellation.at_ms, 50.0);
            assert_eq!(cancellation.transitions.len(), 1);
            let removal = &cancellation.transitions[0];
            assert_eq!(removal.worker_id, 1);
            assert_eq!(
                removal.transition,
                WorkerLifecycleTransitionKind::WorkerRemoved
            );
            assert_eq!(removal.prior_state, Some("starting"));
            assert_eq!(removal.reason, Some("startup_cancelled"));
            assert_eq!(
                removal.origin_operation_ordinal,
                Some(start.operation_ordinal)
            );
            assert_eq!(cancellation.state_after_batch.active, vec![0]);
            assert!(cancellation.state_after_batch.starting.is_empty());
            assert!(cancellation.state_after_batch.draining.is_empty());
            // The stale ready event at 100 ms must neither resurrect the canceled
            // worker nor extend its provisioned interval in independently captured cost.
            let captured_worker_seconds =
                (report.throughput.duration_ms + cancellation.at_ms - start.at_ms) / 1_000.0;
            let native_worker_seconds = match pool {
                WorkerPool::Prefill => report.throughput.prefill_worker_seconds,
                WorkerPool::Agg | WorkerPool::Decode => report.throughput.decode_worker_seconds,
            };
            assert!((native_worker_seconds - captured_worker_seconds).abs() < 1e-9);
        }
    }
}

#[test]
fn round_robin_session_affinity_routes_turns_through_the_public_replayer() {
    for (config, expected) in [
        (serde_json::Value::Null, vec![Some(0), Some(1), Some(0)]),
        (
            serde_json::json!({"session_affinity": false}),
            vec![Some(0), Some(1), Some(0)],
        ),
        (
            serde_json::json!({"session_affinity": true}),
            vec![Some(0), Some(0), Some(1)],
        ),
    ] {
        let requests = ["a", "a", "b"]
            .into_iter()
            .enumerate()
            .map(|(i, session)| {
                let mut req = request(&format!("turn-{i}"), i as f64 * 30.0, 8, 3);
                req.session_id = Some(session.into());
                req
            })
            .collect();
        let mut spec = aggregated_spec(Backend::Vllm, 2, 0.0, requests);
        spec.adapters.placement.config = config;
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(report.request_counts.completed_requests, 3);
        let workers: Vec<_> = report
            .per_request
            .iter()
            .map(|record| record.decode_worker_idx)
            .collect();
        assert_eq!(workers, expected);
    }
}

#[test]
fn round_robin_rejects_invalid_session_configuration() {
    for config in [
        serde_json::json!({"session_affinity": "true"}),
        serde_json::json!({"session_affinty": true}),
    ] {
        let mut spec = aggregated_spec(Backend::Vllm, 1, 0.0, Vec::new());
        spec.adapters.placement.config = config;
        assert!(matches!(
            Replayer::new(spec, ReplayEngineFactory::new()),
            Err(ReplayError::InvalidSpec(_))
        ));
    }
}

#[test]
fn disaggregated_session_affinity_routes_turns_and_preserves_cache_reuse() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        for (prefill_workers, decode_workers, prefill_dp, decode_dp) in
            [(2, 2, 1, 1), (1, 4, 1, 1), (2, 2, 2, 3)]
        {
            for explicit_rank in [false, true] {
                for affinity in [false, true] {
                    let requests = (0..8)
                        .map(|i| {
                            let session = i / 2;
                            let input = 12 + (i % 2) * 4;
                            let mut req = request(&format!("turn-{i}"), i as f64 * 100.0, input, 3);
                            req.input_token_ids = Some(vec![session as u32 + 1; input]);
                            req.session_id = Some(format!("session-{session}"));
                            if explicit_rank {
                                req.prefill_dp_rank = Some(prefill_dp - 1);
                                req.dp_rank = Some(decode_dp - 1);
                            }
                            req
                        })
                        .collect();
                    let mut spec = aggregated_spec(backend, 1, 0.0, requests);
                    spec.adapters.placement.config =
                        serde_json::json!({"session_affinity": affinity});
                    spec.topology = ReplayTopology::Disaggregated {
                        prefill: WorkerPoolSpec {
                            initial_workers: prefill_workers,
                            startup_delay_ms: 0.0,
                        },
                        decode: WorkerPoolSpec {
                            initial_workers: decode_workers,
                            startup_delay_ms: 0.0,
                        },
                        handoff_latency_ms: 1.0,
                    };
                    spec.engine["rank"]["enable_prefix_caching"] = true.into();
                    let rank = spec.engine["rank"].clone();
                    spec.engine["prefill"] =
                        serde_json::json!({"dp_size": prefill_dp, "rank": rank});
                    spec.engine["decode"] = serde_json::json!({"dp_size": decode_dp, "rank": rank});
                    let report = Replayer::new(spec, ReplayEngineFactory::new())
                        .unwrap()
                        .run()
                        .unwrap();
                    assert_eq!(report.request_counts.completed_requests, 8);
                    assert_eq!(report.request_counts.total_output_tokens, 24);
                    for turns in report.per_request.chunks_exact(2) {
                        assert_eq!(turns[0].routing_history.len(), 2);
                        assert_eq!(turns[1].routing_history.len(), 2);
                        for (first, next) in turns[0]
                            .routing_history
                            .iter()
                            .zip(&turns[1].routing_history)
                        {
                            assert_eq!(first.pool, next.pool);
                            if explicit_rank {
                                let rank = match first.pool {
                                    ReplayRequestPool::Prefill => prefill_dp - 1,
                                    ReplayRequestPool::Decode => decode_dp - 1,
                                    ReplayRequestPool::Agg => unreachable!(),
                                };
                                assert_eq!(first.dp_rank, Some(rank));
                                assert_eq!(next.dp_rank, Some(rank));
                            }
                            if affinity {
                                assert_eq!(
                                    (first.logical_worker_id, first.dp_rank),
                                    (next.logical_worker_id, next.dp_rank)
                                );
                            }
                        }
                        if affinity {
                            assert!(
                                turns[1].reused_input_tokens >= 8,
                                "{backend:?}: expected a resident prefix"
                            );
                        } else if prefill_workers > 1 && prefill_dp == 1 {
                            assert_ne!(turns[0].prefill_worker_idx, turns[1].prefill_worker_idx);
                            assert_eq!(turns[1].reused_input_tokens, 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn all_phase_telemetry_captures_dp_ranks_without_enabling_legacy_request_output() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        for dp in [2, 4] {
            let mut spec = aggregated_spec(
                backend,
                1,
                0.0,
                (0..dp)
                    .map(|rank| request(&format!("dp-{rank}"), 0.0, 8, 3))
                    .collect(),
            );
            spec.engine["dp_size"] = serde_json::json!(dp);
            spec.record_per_request = false;
            let baseline = Replayer::new(spec.clone(), ReplayEngineFactory::new())
                .unwrap()
                .run()
                .unwrap();
            let mut report = Replayer::new(spec, ReplayEngineFactory::new())
                .unwrap()
                .with_telemetry_capture(3.0)
                .unwrap()
                .run()
                .unwrap();
            let capture = report.telemetry.take().unwrap();
            assert!(report.per_request.is_empty());
            assert_eq!(capture.requests.len(), dp);
            assert_eq!(
                capture
                    .samples
                    .iter()
                    .map(|s| s.sample.traffic.delivered_output_tokens)
                    .sum::<u64>(),
                (dp * 3) as u64
            );
            assert_eq!(
                capture
                    .samples
                    .iter()
                    .map(|s| s.sample.traffic.committed_prefill_tokens)
                    .sum::<u64>(),
                report.committed_prefill_tokens
            );
            assert!(
                capture
                    .samples
                    .iter()
                    .any(|s| s.sample.decode_scheduler_metrics.len() == dp)
            );
            assert_eq!(
                report.request_counts.completed_requests,
                baseline.request_counts.completed_requests
            );
            assert_eq!(
                report.throughput.duration_ms,
                baseline.throughput.duration_ms
            );
        }
    }
}

#[test]
fn telemetry_capture_reports_a_time_limited_run_without_claiming_completion() {
    for backend in [Backend::Vllm, Backend::Sglang] {
        let mut spec = aggregated_spec(backend, 1, 0.0, vec![request("unfinished", 0.0, 8, 100)]);
        spec.max_sim_time_ms = Some(0.0);
        let report = Replayer::new(spec, ReplayEngineFactory::new())
            .unwrap()
            .with_telemetry_capture(3.0)
            .unwrap()
            .run()
            .unwrap();
        let capture = report.telemetry.unwrap();
        assert_eq!(capture.completion_status, "time_limited");
        assert_eq!(capture.phase_boundaries.len(), 1);
        assert_eq!(
            capture.phase_boundaries[0].completion_status,
            "time_limited"
        );
        assert_eq!(capture.units.timestamps, "ms");
        assert_eq!(capture.units.traffic_duration, "s");
        assert_eq!(capture.profile_start_ms, Some(0.0));
    }
}
