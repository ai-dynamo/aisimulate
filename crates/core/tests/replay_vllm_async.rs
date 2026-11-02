// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM async scheduling in offline replay.

use std::collections::VecDeque;

use aisimulate_core::engine::{EngineConfig, PassLaunchConfig, TimingModelConfig};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, PerRequestRecord, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
    WorkerPoolSpec,
};
use uuid::Uuid;

/// One vLLM worker with `num_gpu_blocks` 4-token blocks. A pass with prefill
/// takes 100 ms and a decode pass 10 ms.
fn engine(num_gpu_blocks: usize) -> EngineConfig {
    EngineConfig {
        block_size: 4,
        num_gpu_blocks,
        max_num_batched_tokens: 64,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: 100.0,
            decode_ms: 10.0,
        },
        ..EngineConfig::default()
    }
}

/// `engine` with `launch`: launching a pass takes `ms` without prefill, with up
/// to 16 scheduled tokens and above.
fn launch(engine: EngineConfig, ms: [f64; 3], eager_queue_fraction: f64) -> EngineConfig {
    let [decode_ms, piecewise_ms, eager_ms] = ms;
    EngineConfig {
        launch: Some(PassLaunchConfig {
            decode_ms,
            piecewise_ms,
            eager_ms,
            max_cudagraph_capture_size: 16,
            eager_queue_fraction,
        }),
        ..engine
    }
}

fn spec(engine: ReplayEngineConfig) -> ReplaySpec {
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: None,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: 1,
                ..WorkerPoolSpec::default()
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

/// A replayer of `(id, prompt tokens, outputs, arrival)` requests.
fn replayer(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> Replayer {
    let spec = spec(ReplayEngineConfig {
        rank: engine,
        ..Default::default()
    });
    let requests = requests
        .iter()
        .map(|&(id, prompt, outputs, arrival_ms)| DirectRequest {
            tokens: (0..prompt).map(|token| token + id as u32 * 1000).collect(),
            max_output_tokens: outputs,
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: Some(arrival_ms),
            ..Default::default()
        })
        .collect::<VecDeque<_>>();
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Requests(requests))
}

fn run(engine: EngineConfig, requests: &[(u128, u32, usize, f64)]) -> ReplayReport {
    replayer(engine, requests).run().unwrap()
}

fn record(report: &ReplayReport, id: u128) -> &PerRequestRecord {
    let uuid = Uuid::from_u128(id).to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

#[test]
fn a_request_waiting_only_for_freed_blocks_is_admitted_when_they_are_freed() {
    // R (24 tokens) prefills until 100 and finishes in its decode pass at
    // 110. W (16 tokens) arrives at 50 and fits only in R's blocks. vLLM frees
    // them after deciding the pass at 110, which has no other work, so it
    // decides again at once and admits W at 110.
    let report = run(engine(10), &[(1, 24, 2, 0.0), (2, 16, 2, 50.0)]);
    assert_eq!(record(&report, 2).first_admit_ms, Some(110.0));
}

/// A prefills from 0 to 100 and decodes until 110 and 120; B (32 tokens)
/// arrives while the first decode runs, so the pass from 120 is eager.
const EAGER_AFTER_DECODE: [(u128, u32, usize, f64); 2] = [(1, 8, 3, 0.0), (2, 32, 1, 105.0)];

#[test]
fn a_pass_outputs_leave_once_the_next_pass_is_launched() {
    // vLLM decided the eager pass at 110, once the decode before it had
    // launched and the pass before that had ended, so it launches it until
    // 110 + 50. A's last token and its completion leave then, instead of at
    // 120. If the launch waits until the GPU has run 0.8 of that 100 ms pass,
    // they leave at 200. Passes do not move: B runs from 120 to 220.
    let plain = run(engine(64), &EAGER_AFTER_DECODE);
    let a = record(&plain, 1);
    assert_eq!(
        (a.first_token_ms, a.last_token_ms),
        (Some(100.0), Some(120.0))
    );
    for (fraction, returned_ms) in [(0.0, 160.0), (0.8, 200.0)] {
        let report = run(
            launch(engine(64), [2.0, 20.0, 50.0], fraction),
            &EAGER_AFTER_DECODE,
        );
        let a = record(&report, 1);
        assert_eq!(a.first_token_ms, Some(100.0));
        assert_eq!(a.last_token_ms, Some(returned_ms));
        assert_eq!(a.terminal_time_ms, returned_ms);
        let b = record(&report, 2);
        assert_eq!(
            (b.first_admit_ms, b.first_token_ms),
            (Some(120.0), Some(220.0))
        );
    }
}

#[test]
fn detokenization_starts_when_vllm_returns_the_outputs() {
    // A 1 ms detokenizer takes A's last token when vLLM returns it at 160.
    let launched = replayer(
        launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        &EAGER_AFTER_DECODE,
    )
    .with_detokenization(1, 1.0, 0.0)
    .unwrap()
    .run()
    .unwrap();
    assert_eq!(record(&launched, 1).last_token_ms, Some(161.0));
}

#[test]
fn an_arrival_joins_the_first_pass_decided_after_it() {
    // A's prefill runs from 0 to 100 and vLLM decides the next pass once it
    // has launched it, at 20. B arrives before that and joins the next pass,
    // which prefills B and decodes A until 210; C arrives after it and waits
    // for the pass after. Counting passes, both would skip the pass vLLM had
    // already queued and join the one from 110.
    let requests = [(1, 8, 2, 0.0), (2, 8, 1, 10.0), (3, 8, 1, 30.0)];
    let admitted =
        |report: &ReplayReport| [2, 3].map(|id| record(report, id).first_admit_ms.unwrap());
    assert_eq!(admitted(&run(engine(64), &requests)), [110.0, 110.0]);
    assert_eq!(
        admitted(&run(launch(engine(64), [2.0, 20.0, 50.0], 0.0), &requests)),
        [100.0, 210.0]
    );
    // With no launch cost vLLM decides each pass when the one before it
    // starts, so the rule counts passes as before: B, arriving mid-pass at
    // 105, skips one pass; C, arriving as a pass ends at 110, skips two.
    let requests = [(1, 8, 10, 0.0), (2, 8, 1, 105.0), (3, 8, 1, 110.0)];
    assert_eq!(
        admitted(&run(launch(engine(64), [0.0; 3], 0.0), &requests)),
        admitted(&run(engine(64), &requests))
    );
}

#[test]
fn launch_needs_a_one_gpu_engine() {
    // With more GPUs vLLM's multiprocess executor returns a pass's outputs
    // independently of the next launch.
    let two_gpus = ReplayEngineConfig {
        tensor_parallel_size: 2,
        rank: launch(engine(64), [2.0, 20.0, 50.0], 0.0),
        ..Default::default()
    };
    let error = Replayer::new(spec(two_gpus), ReplayEngineFactory::new())
        .unwrap()
        .run()
        .unwrap_err();
    assert!(error.to_string().contains("uniprocess executor"), "{error}");
}
