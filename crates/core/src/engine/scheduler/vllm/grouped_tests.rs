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
    fn prefill_batch_validation_can_fail(&self) -> bool {
        false
    }

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
        vec![
            (32, 32, 32),
            (33, 33, 33),
            (34, 34, 34),
            (35, 35, 35),
            (36, 36, 36)
        ]
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
