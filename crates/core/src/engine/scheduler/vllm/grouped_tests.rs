// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::scheduler::SchedulerCommand;
use crate::engine::trace::TraceCollector;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

use super::core::VllmCore;

#[derive(Default)]
struct CapturedDecodeQueries(Mutex<Vec<(usize, usize, usize)>>);

impl crate::engine::TimingModel for CapturedDecodeQueries {
    fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn predict_decode_ms(
        &self,
        _: usize,
        active_tokens: usize,
        mean_context: usize,
        token_bound: usize,
    ) -> anyhow::Result<f64> {
        self.0
            .lock()
            .unwrap()
            .push((active_tokens, mean_context, token_bound));
        Ok(0.0)
    }
}

fn group(name: &str, block: u32, bytes: u64, window: Option<u32>) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::Attention,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: bytes,
        sliding_window: window,
        blocks_per_request: None,
    }
}

fn state(name: &str, bytes: u64, blocks: u32) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: 4,
        page_size_bytes: bytes,
        sliding_window: None,
        blocks_per_request: Some(blocks),
    }
}

fn core(groups: Vec<FpmCacheGroup>, capacity: u64, chunk: usize) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            // Deliberately smaller than the prompts: this linear capacity
            // must not decide admission or accounting for grouped caches.
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .enable_prefix_caching(false)
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

fn submit(core: &mut VllmCore, id: u128, prompt: usize, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens: (0..prompt as u32).collect(),
        max_output_tokens: output,
        ..Default::default()
    })
}

#[test]
fn grouped_window_replays_long_context_with_bounded_storage_and_logical_fpm_lengths() {
    let mut core = core(vec![group("window", 4, 8, Some(4))], 24, 8);
    let captured = Arc::new(CapturedDecodeQueries::default());
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: captured.clone(),
    });
    let id = submit(&mut core, 101, 32, 5);
    let mut trace = TraceCollector::default();
    let mut outputs = 0;
    let mut prefixes = Vec::new();
    let mut decode_contexts = Vec::new();
    let mut bytes = Vec::new();
    for step in 0..20 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let fpm = pass.fpm.unwrap();
        if fpm.num_prefill_requests > 0 {
            prefixes.push(fpm.sum_prefill_kv_tokens);
            assert_eq!(fpm.sum_prefill_tokens, 8);
        }
        if fpm.num_decode_requests > 0 {
            decode_contexts.push(fpm.sum_decode_kv_tokens);
        }
        bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
        assert_eq!(pass.mocker_metrics.total_blocks, 0);
        assert_eq!(pass.mocker_metrics.kv_cache_capacity_bytes, Some(24));
        assert!(pass.mocker_metrics.physical_gpu_cache_usage_perc <= 1.0);
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        for output in pass.output_signals {
            assert_eq!(output.uuid, id);
            assert!(!output.rejected);
            outputs += usize::from(output.token_id.is_some());
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(core.is_drained());
    assert_eq!(outputs, 5);
    assert_eq!(prefixes, vec![0, 8, 16, 24]);
    assert_eq!(decode_contexts, vec![33, 34, 35, 36]);
    assert_eq!(
        *captured.0.lock().unwrap(),
        // Prefill produces output token one. The four decode forwards include
        // their current input token and retain the full logical context.
        vec![(33, 33, 33), (34, 34, 34), (35, 35, 35), (36, 36, 36)]
    );
    assert_eq!(&bytes[..4], &[16, 24, 24, 24]);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_full_window_and_convolution_charge_distinct_padded_pages() {
    let mut convolution = group("convolution", 2, 2, Some(2));
    convolution.kind = FpmCacheKind::Convolution;
    let mut core = core(
        vec![
            group("full", 4, 8, None),
            group("window", 4, 4, Some(4)),
            convolution,
        ],
        100,
        4,
    );
    let id = submit(&mut core, 102, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    assert_eq!(bytes, vec![16, 30, 38, 46, 52]);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 17);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 17);
    assert_eq!(core.state.requests[&id].sequence.len(), 18);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_contention_preempts_and_retries_without_leaking_capacity() {
    let mut core = core(
        vec![group("full", 4, 4, None), group("window", 4, 4, Some(4))],
        24,
        8,
    );
    let first = submit(&mut core, 103, 4, 6);
    let second = submit(&mut core, 104, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    let pressure = core.execute_pass(&mut trace, 1.0);
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(16));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    assert_eq!(core.state.requests[&first].num_computed_tokens, 5);
    let mut completed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        assert!(pass.mocker_metrics.kv_cache_used_bytes.unwrap() <= 24);
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_waiting_request_cannot_preempt_running_work() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 12, 8);
    let first = submit(&mut core, 105, 8, 2);
    let second = submit(&mut core, 106, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(next.mocker_metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
}

#[test]
fn grouped_oversized_chunk_fails_without_partial_allocation_or_endless_preemption() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 8, 8);
    let id = submit(&mut core, 107, 16, 2);
    let mut trace = TraceCollector::default();
    core.execute_pass(&mut trace, 0.0);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    let error = core.try_execute_pass(&mut trace, 1.0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exceeding the entire 8-byte cache pool")
    );
    assert!(error.to_string().contains("reduce max_num_batched_tokens"));
    assert_eq!(core.state.requests[&id].num_computed_tokens, 8);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 8);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_handoff_commands_fail_before_acquiring_state() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 24, 8);
    let handoff_id = crate::engine::HandoffId::new(Uuid::from_u128(108));
    let error = core
        .apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: DirectRequest {
                tokens: vec![0, 1, 2, 3],
                max_output_tokens: 2,
                ..Default::default()
            },
        })
        .unwrap_err();
    assert!(error.to_string().contains("not KV handoff commands"));
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_state_blocks_stay_constant_through_prefill_and_decode() {
    let mut core = core(vec![group("full", 4, 8, None), state("kda", 10, 2)], 200, 4);
    let id = submit(&mut core, 109, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    // Every pass adds one 8-byte page; the two 10-byte state blocks stay.
    assert_eq!(bytes, vec![28, 36, 44, 52, 60]);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_bytes_stay_constant_while_window_pages_expire() {
    let run = |groups: Vec<FpmCacheGroup>, capacity: u64| {
        let mut core = core(groups, capacity, 8);
        let id = submit(&mut core, 110, 32, 5);
        let mut trace = TraceCollector::default();
        let mut outputs = 0;
        let mut bytes = Vec::new();
        for step in 0..20 {
            let pass = core.execute_pass(&mut trace, step as f64);
            bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
            for output in pass.output_signals {
                assert_eq!(output.uuid, id);
                assert!(!output.rejected);
                outputs += usize::from(output.token_id.is_some());
            }
            if core.is_drained() {
                break;
            }
        }
        assert!(core.is_drained());
        assert_eq!(outputs, 5);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        bytes
    };
    let without = run(vec![group("window", 4, 8, Some(4))], 24);
    let with = run(
        vec![group("window", 4, 8, Some(4)), state("kda", 10, 2)],
        44,
    );
    // Window pages expire as the decode moves on (24 to 16 bytes), while the
    // two state blocks add a constant 20 bytes. The last pass completes the
    // request, which frees every group before the pass reports its bytes.
    assert_eq!(without, vec![16, 24, 24, 24, 16, 16, 16, 0]);
    assert_eq!(with, vec![36, 44, 44, 44, 36, 36, 36, 0]);
}

#[test]
fn grouped_state_preemption_frees_and_rereserves_state_blocks() {
    let mut core = core(
        vec![
            group("full", 4, 4, None),
            group("window", 4, 4, Some(4)),
            state("kda", 4, 1),
        ],
        36,
        8,
    );
    let first = submit(&mut core, 111, 4, 6);
    let second = submit(&mut core, 112, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    // Each prefill holds a full page, a window page and its state block.
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(24));
    let pressure = core.execute_pass(&mut trace, 1.0);
    // The first request grows to 20 bytes. The preempted second request
    // returns all 12 of its bytes, its state block included.
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(20));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    let mut completed = Vec::new();
    let mut resumed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let bytes = pass.mocker_metrics.kv_cache_used_bytes.unwrap();
        assert!(bytes <= 36);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            resumed.push(bytes);
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    // Each resume reserves the state block once more: two full pages, two
    // window pages and one state block, 20 bytes. The first resume fills the
    // pool next to the first request's 16 bytes. The first request's next
    // pages then preempt the second request again, and it resumes alone.
    assert_eq!(resumed, vec![36, 20]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_request_waits_when_only_attention_pages_fit() {
    let full = group("full", 4, 4, None);
    let mut core = core(vec![full.clone(), state("kda", 8, 1)], 28, 8);
    let first = submit(&mut core, 113, 8, 2);
    let second = submit(&mut core, 114, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    // The next pass first grows the running request to 9 tokens: three pages
    // and its state block hold 12 + 8 = 20 bytes, so 8 of 28 are free. The
    // waiting request's chunk, the 7 tokens left in the 8-token batch, would
    // fit its two pages (8 bytes) there, but not its 8-byte state block too.
    let pool = core.kv_manager.grouped().unwrap();
    let free = pool.capacity_bytes() - pool.required_bytes(8, 9).unwrap();
    assert_eq!(free, 8);
    assert!(full.bytes_for_forward(0, 7).unwrap() <= free);
    assert!(pool.required_bytes(0, 7).unwrap() > free);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert!(
        next.output_signals
            .iter()
            .any(|output| output.uuid == first && output.completed)
    );
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
    for pass in [&initial, &next, &admitted] {
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
    }
}
