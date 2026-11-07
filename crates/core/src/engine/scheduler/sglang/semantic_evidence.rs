// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Synthetic probes for SGLang v0.5.22 contracts; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use uuid::Uuid;

use super::core::SglangCore;
use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{
    DirectRequest, EngineType, MockEngineArgs, SglangArgs, WorkerType,
};
use crate::engine::scheduler::SchedulerCommand;
use crate::engine::timing::built_in_timing_model;
use crate::engine::{HandoffId, TimingModelConfig};

fn args(capacity: usize, page: usize, chunk: usize, role: WorkerType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .worker_type(role)
        .num_gpu_blocks(capacity / page)
        .block_size(page)
        .max_num_seqs(Some(32))
        .speedup_ratio(1.0)
        .perf_model(Arc::new(PerfModel::External {
            timing: built_in_timing_model(&TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 2.0,
            })
            .unwrap(),
        }))
        .sglang(Some(SglangArgs {
            page_size: Some(page),
            chunked_prefill_size: Some(chunk),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

fn request(id: u128, tokens: Vec<u32>, outputs: usize) -> DirectRequest {
    DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: outputs,
        output_token_ids: Some((10_000..10_000 + outputs as u32).collect()),
        ..Default::default()
    }
}

fn reserve(core: &mut SglangCore, id: u128, prompt: usize, outputs: usize) -> bool {
    !core
        .apply_command_effects(
            SchedulerCommand::ReserveDestination {
                handoff_id: HandoffId::from(Uuid::from_u128(id)),
                request: request(id, vec![id as u32; prompt], outputs),
            },
            true,
        )
        .unwrap()
        .lifecycle_events
        .is_empty()
}

fn activate(core: &mut SglangCore, id: u128) {
    core.apply_command(SchedulerCommand::ActivateDestination {
        handoff_id: HandoffId::from(Uuid::from_u128(id)),
    })
    .unwrap();
}

fn drive(core: &mut SglangCore, start_ms: f64, limit: usize) -> Value {
    let mut now = start_ms;
    let mut passes = 0;
    let mut outputs = BTreeMap::<String, usize>::new();
    let mut terminals = BTreeMap::<String, usize>::new();
    let mut completion_ms = BTreeMap::<String, f64>::new();
    let mut terminal_order = Vec::new();
    let mut rejected = Vec::new();
    let mut admissions = Vec::new();
    let mut pressure_events = 0;
    for _ in 0..limit {
        if core.is_drained() {
            break;
        }
        let pass = core.execute_hidden_pass(now);
        now = pass.end_ms;
        passes += 1;
        pressure_events += pass.pressure_events.len();
        admissions.extend(pass.admissions.iter().map(|event| {
            json!({"id": event.uuid.as_u128() as u64, "reused_tokens": event.reused_input_tokens})
        }));
        for signal in pass.output_signals {
            let id = signal.uuid.as_u128() as u64;
            *outputs.entry(id.to_string()).or_default() += usize::from(signal.token_id.is_some());
            if signal.completed {
                *terminals.entry(id.to_string()).or_default() += 1;
                completion_ms.insert(id.to_string(), now);
                terminal_order.push(id);
            }
            if signal.rejected {
                rejected.push(id);
            }
        }
    }
    json!({
        "drained": core.is_drained(), "passes": passes, "end_ms": now,
        "outputs": outputs, "terminals": terminals, "completion_ms": completion_ms,
        "terminal_order": terminal_order, "rejected": rejected, "admissions": admissions,
        "pressure_events": pressure_events, "ratio": core.new_token_ratio,
        "free_tokens": core.kv_manager.cache().available_tokens(),
        "evictable_tokens": core.kv_manager.cache().evictable_size,
    })
}

fn completion(overlap: bool, speculative: bool, outputs: usize) -> Value {
    let mut args = args(1024, 16, 1024, WorkerType::Aggregated);
    args.sglang.as_mut().unwrap().host_loop = overlap;
    if speculative {
        args.aic_nextn = Some(2);
        args.aic_nextn_accept_rates = Some("1,1".into());
    }
    let mut core = SglangCore::new(args);
    let prompt = if outputs == 1 { 127 } else { 100 };
    let mut sequence: Vec<u32> = (0..prompt).collect();
    core.receive(request(1, sequence.clone(), outputs));
    let outcome = drive(&mut core, 0.0, 128);
    sequence.extend(10_000..10_000 + outputs as u32);
    let cached = core.kv_manager.cache().prefix_match_len(&sequence);
    sequence.push(20_000);
    core.receive(request(2, sequence, 1));
    let follower = drive(&mut core, outcome["end_ms"].as_f64().unwrap(), 16);
    json!({"overlap": overlap, "speculative": speculative, "prompt_tokens": prompt,
        "requested_outputs": outputs, "cached_tokens": cached, "outcome": outcome, "follower": follower})
}

#[test]
#[ignore = "manual cross-revision evidence; see benchmarks/evidence/sglang-semantics-20261105"]
fn scheduler_semantic_matrix() {
    let mut pd = SglangCore::new(args(4096, 1, 4096, WorkerType::Decode));
    let reserved: Vec<_> = (1..=4).map(|id| reserve(&mut pd, id, 1000, 200)).collect();
    let pd_admission =
        json!({"reserved": reserved, "free_tokens": pd.kv_manager.cache().available_tokens()});

    let boundaries: Vec<_> = [991, 992, 1001]
        .into_iter()
        .map(|prompt| {
            let mut core = SglangCore::new(args(1008, 16, 1024, WorkerType::Aggregated));
            core.receive(request(1, (0..prompt).collect(), 0));
            json!({"prompt_tokens": prompt, "outcome": drive(&mut core, 0.0, 16)})
        })
        .collect();
    let mut clipped = SglangCore::new(args(1008, 16, 1024, WorkerType::Aggregated));
    clipped.receive(request(1, (0..900).collect(), 100));
    let generation_clip = drive(&mut clipped, 0.0, 128);

    let mut completions = Vec::new();
    for overlap in [false, true] {
        for speculative in [false, true] {
            for outputs in [1, 28, 29] {
                completions.push(completion(overlap, speculative, outputs));
            }
        }
    }

    let mut decode_cache = SglangCore::new(args(2048, 16, 2048, WorkerType::Decode));
    assert!(reserve(&mut decode_cache, 1, 127, 1));
    activate(&mut decode_cache, 1);
    let decode_outcome = drive(&mut decode_cache, 0.0, 8);
    let decode_cached_tokens = decode_cache
        .kv_manager
        .cache()
        .prefix_match_len(&vec![1; 127]);

    let mut queue_args = args(256, 4, 8, WorkerType::Aggregated);
    queue_args.sglang.as_mut().unwrap().schedule_policy = Some("lpm".into());
    let mut queue = SglangCore::new(queue_args);
    queue.receive(request(1, vec![99; 32], 1));
    let warm = drive(&mut queue, 0.0, 16);
    queue.receive(request(2, (0..24).collect(), 1));
    let first = queue.execute_hidden_pass(warm["end_ms"].as_f64().unwrap());
    queue.receive(request(3, vec![99; 32], 1));
    let queue_order = drive(&mut queue, first.end_ms, 32);

    let mut ratio = SglangCore::new(args(64, 4, 16, WorkerType::Aggregated));
    let held = ratio
        .kv_manager
        .allocate_for_request(&vec![99; 56])
        .unwrap();
    let id = ratio.receive(request(1, (0..8).collect(), 4));
    ratio.new_token_ratio = 0.3;
    let mut blocked_ratios = Vec::new();
    for _ in 0..3 {
        ratio.execute_hidden_pass(0.0);
        blocked_ratios.push(ratio.new_token_ratio);
    }
    ratio
        .apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    ratio.kv_manager.abort(held.lease);
    ratio.execute_hidden_pass(0.0);
    let ratio_lifecycle = json!({"blocked": blocked_ratios, "drained": ratio.new_token_ratio});

    let report = json!({"pd_admission": pd_admission, "boundaries": boundaries,
        "generation_clip": generation_clip, "completions": completions,
        "default_decode_cache": {"cached_tokens": decode_cached_tokens, "outcome": decode_outcome},
        "queue_order": queue_order, "ratio_lifecycle": ratio_lifecycle});
    let destination = std::env::var("AISIM_SEMANTIC_REPORT")
        .expect("set AISIM_SEMANTIC_REPORT to an output file");
    std::fs::write(
        destination,
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    )
    .unwrap();
}
