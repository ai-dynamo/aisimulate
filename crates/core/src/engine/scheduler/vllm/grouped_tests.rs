// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::kv_manager::BlockRequestLease;
use crate::engine::scheduler::SchedulerCommand;
use crate::engine::trace::TraceCollector;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

use super::core::VllmCore;

/// Decode queries `(active_tokens, mean_context, token_bound)` and prefill
/// queries `(batch, new_tokens, prefix_tokens)`.
#[derive(Default)]
struct CapturedDecodeQueries(
    Mutex<Vec<(usize, usize, usize)>>,
    Mutex<Vec<(usize, usize, usize)>>,
);

impl crate::engine::TimingModel for CapturedDecodeQueries {
    fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> anyhow::Result<f64> {
        self.1
            .lock()
            .unwrap()
            .push((batch_size, new_tokens, prefix_tokens));
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
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
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
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
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
    submit_tokens(core, id, (0..prompt as u32).collect(), output)
}

fn submit_tokens(core: &mut VllmCore, id: u128, tokens: Vec<u32>, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: output,
        ..Default::default()
    })
}

const GLM_PAGE: u64 = 13_045_120;
const GLM_BLOCK: u32 = 2176;

fn glm_state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: GLM_PAGE,
        sliding_window: None,
        blocks_per_request: Some(1),
        prefix_cacheable: Some(cacheable),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: cacheable.then_some(u32::from(checkpoint)),
        prefill_checkpoint_alignment: (cacheable && checkpoint).then_some(16),
    }
}

/// The six GLM-5.3-Flash TP 4 groups of Facts B at one page size. With
/// `checkpoint`, G2 to G5 carry one FlashKDA checkpoint block at alignment 16.
fn glm_groups(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut mla = group("mla", GLM_BLOCK, GLM_PAGE, None);
    mla.prefix_cacheable = Some(true);
    let mut groups = vec![mla, glm_state("kpool_tail", 4, false, false)];
    for i in 0..4 {
        groups.push(glm_state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

/// The three cores of the amendment: vLLM 0.31.0, 0.32.0 without checkpoint
/// blocks, and 0.32.0 with them.
const GLM_CORES: [(&str, bool); 3] = [("0.31.0", false), ("0.32.0", false), ("0.32.0", true)];

fn prefix_core(
    groups: Vec<FpmCacheGroup>,
    capacity: u64,
    chunk: usize,
    unit: usize,
    version: &str,
) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .max_model_len(Some(131_072))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(unit))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

/// Step until the core drains. Returns `(uuid, reused_input_tokens)` of every
/// admission in order.
fn drain(
    core: &mut VllmCore,
    trace: &mut TraceCollector,
    from_step: &mut usize,
) -> Vec<(Uuid, usize)> {
    let mut admissions = Vec::new();
    for _ in 0..200 {
        let pass = core.execute_pass(trace, *from_step as f64);
        *from_step += 1;
        admissions.extend(
            pass.admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens)),
        );
        for output in pass.output_signals {
            assert!(!output.rejected, "{} rejected", output.uuid);
        }
        if core.is_drained() {
            return admissions;
        }
    }
    panic!("the core did not drain");
}

fn block_pool(core: &VllmCore) -> &crate::engine::kv_manager::GroupedBlockPool {
    core.kv_manager.grouped().unwrap().block_pool().unwrap()
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
    // Without a state group the message has no sentence about fixed bytes.
    assert!(!error.to_string().contains("fixed state blocks"));
    assert_eq!(core.state.requests[&id].num_computed_tokens, 8);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 8);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_state_blocks_exceeding_the_pool_reject_at_admission() {
    // The 16-byte state block alone is larger than the 12-byte pool, so even
    // the one-token forward cannot fit and the request never reaches a forward.
    let mut core = core(vec![group("full", 4, 4, None), state("kda", 16, 1)], 12, 8);
    let id = submit(&mut core, 115, 4, 2);
    let mut trace = TraceCollector::default();
    let pass = core.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_oversized_chunk_names_the_fixed_state_bytes() {
    let mut core = core(
        vec![group("window", 4, 4, Some(4)), state("kda", 2, 1)],
        9,
        8,
    );
    submit(&mut core, 116, 16, 2);
    let mut trace = TraceCollector::default();
    // The one-token forward needs one 4-byte window page and the 2-byte state
    // block, 6 bytes, so admission passes. The first chunk of 8 tokens needs
    // two window pages (8 bytes) and the state block, 10 bytes, above the pool.
    let error = core
        .try_execute_pass(&mut trace, 0.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs 10 bytes for 8 scheduled tokens after 0 computed tokens"));
    assert!(error.contains("exceeding the entire 9-byte cache pool"));
    assert!(error.contains("the fixed state blocks of one request take 2 bytes"));
    assert!(error.contains("reduce max_num_batched_tokens"));
    // The sentence comes before the remedy.
    assert!(error.contains("2 bytes of this pool; reduce max_num_batched_tokens"));
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
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
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

#[test]
fn grouped_prefix_caching_off_keeps_byte_mode_accounting() {
    let mut core = core(glm_groups(false), 64 * GLM_PAGE, 8192);
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = submit(&mut core, 301, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
    // A resend of the same prompt replays cold: nothing is cached in byte mode.
    let second = submit(&mut core, 302, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(second, 0)]);
    let metrics = core.mocker_metrics();
    assert_eq!(metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(metrics.kv_cache_cached_bytes, None);
    assert_eq!(core.kv_manager.grouped().unwrap().cached_bytes(), 0);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_glm_shaped_multi_turn_replay_reports_hit_tokens_and_block_counts() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        let first = submit_tokens(&mut core, 401, turn_one.clone(), 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
        assert_eq!(block_pool(&core).active_blocks(), 0);

        // Turn two: the first 9,000 tokens again plus 4,000 new ones. The hit
        // is the state at request one's replay boundary (9000 - 1) / 2176 *
        // 2176 = 8704, where G0 holds four cached blocks too.
        let mut turn_two = turn_one.clone();
        turn_two.extend(100_000..104_000);
        let second = submit_tokens(&mut core, 402, turn_two.clone(), 4);
        block_pool(&core).reset_peak_active_blocks();
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, second);
        assert_eq!(
            pass.admissions[0].reused_input_tokens, 8704,
            "{version} {checkpoint}"
        );
        // During the first step: G0 holds the four pinned blocks and the fresh
        // ones of the chunk, the ring one, each KDA group the cached and the
        // running block, plus the checkpoint block on the third core.
        let (peak, after) = if checkpoint { (19, 15) } else { (14, 10) };
        assert_eq!(
            block_pool(&core).peak_active_blocks(),
            peak,
            "{version} {checkpoint}"
        );
        // After the step the four cached state blocks are released.
        assert_eq!(
            block_pool(&core).active_blocks(),
            after,
            "{version} {checkpoint}"
        );
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 4);
        assert_eq!(
            pass.mocker_metrics.kv_cache_used_bytes,
            Some(after as u64 * GLM_PAGE)
        );
        assert_eq!(
            pass.mocker_metrics.kv_cache_cached_bytes,
            Some(4 * GLM_PAGE)
        );
        assert!(drain(&mut core, &mut trace, &mut step).is_empty());

        // A request that shares only the first 2,176 tokens finds a G0 block
        // but no state key at 2,176 (retention 0), so it hits nothing.
        let mut other: Vec<u32> = (0..2176).collect();
        other.extend(200_000..202_824);
        let third = submit_tokens(&mut core, 403, other, 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(third, 0)]);

        // The whole second turn plus 100 tokens hits request two's replay
        // boundary (13000 - 1) / 2176 * 2176 = 10880.
        let mut turn_three = turn_two;
        turn_three.extend(300_000..300_100);
        let fourth = submit_tokens(&mut core, 404, turn_three, 4);
        assert_eq!(
            drain(&mut core, &mut trace, &mut step),
            vec![(fourth, 10880)]
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    }
}

#[test]
fn grouped_prefix_hit_lowers_prefill_tokens_and_raises_prefix_tokens() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        submit_tokens(&mut core, 501, turn_one.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        // The first prefill stops at the block boundary 6528 inside the 8192
        // budget, then at the last cacheable stop 8704, then at the prompt.
        // In checkpoint mode the final chunk runs from 6528 to the prompt.
        let first_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 6528, 0), (1, 2472, 6528)]
        } else {
            vec![(1, 6528, 0), (1, 2176, 6528), (1, 296, 8704)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            first_turn,
            "{version} {checkpoint}"
        );
        captured.1.lock().unwrap().clear();

        let mut turn_two = turn_one;
        turn_two.extend(100_000..104_000);
        submit_tokens(&mut core, 502, turn_two, 1);
        drain(&mut core, &mut trace, &mut step);
        // From the hit at 8704 the split ends the first chunk at the last
        // cacheable stop 10880 and the second at the prompt. The checkpoint
        // branch runs to the prompt end in one chunk.
        let second_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 4296, 8704)]
        } else {
            vec![(1, 2176, 8704), (1, 2120, 10880)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            second_turn,
            "{version} {checkpoint}"
        );
    }
}

#[test]
fn grouped_prefix_admission_tests_the_whole_prompt() {
    for (version, checkpoint) in GLM_CORES {
        // A prompt of 40,000 tokens needs 19 G0 blocks, 1 ring block and one
        // running block per KDA group, plus one checkpoint block each on the
        // third core: 24 or 28, above the 20-block pool.
        let mut core = prefix_core(glm_groups(checkpoint), 20 * GLM_PAGE, 8192, 2176, version);
        let id = submit(&mut core, 601, 40_000, 2);
        let request = &core.state.requests[&id].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!(
            cost.new_blocks,
            if checkpoint { 28 } else { 24 },
            "{version}"
        );
        assert_eq!(cost.cached_tokens, 0);
        let mut trace = TraceCollector::default();
        let pass = core.execute_pass(&mut trace, 0.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert!(
            pass.output_signals
                .iter()
                .any(|output| output.uuid == id && output.rejected)
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert!(core.is_drained());

        // Under contention the whole prompt decides (decision 7). A running
        // request of 13,000 tokens holds 3 G0 pages, the ring and four state
        // blocks after its first chunk, which the split ends at 6,528, so 22
        // of 30 blocks are free. The 40,000-token prompt needs 24 and waits
        // without preempting anything.
        let mut core = prefix_core(glm_groups(checkpoint), 30 * GLM_PAGE, 8192, 2176, version);
        let running = submit(&mut core, 603, 13_000, 2);
        let waiting = submit_tokens(&mut core, 604, (100_000..140_000).collect(), 2);
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, running);
        assert_eq!(block_pool(&core).active_blocks(), 8);
        // The second pass has budget left after the running request's
        // chunk, and the waiting request still waits for the whole prompt.
        let pass = core.execute_pass(&mut trace, 1.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        assert_eq!(core.state.requests[&waiting].num_computed_tokens, 0);
    }
    // Prefix caching off keeps the first-chunk test of byte mode. The first
    // pass spends its budget on the running request. In the second pass the
    // running request takes its last 4,808 tokens (6 G0 pages, the ring and
    // the states: 11 pages) and the waiting request's first chunk of the
    // 3,384 tokens left (2 pages, the ring and the states: 7 pages) fits.
    let mut off = core(glm_groups(false), 30 * GLM_PAGE, 8192);
    let running = submit(&mut off, 605, 13_000, 2);
    let waiting = submit_tokens(&mut off, 606, (100_000..140_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let pass = off.execute_pass(&mut trace, 0.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, running);
    let pass = off.execute_pass(&mut trace, 1.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, waiting);
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(18 * GLM_PAGE));
    // Byte mode also refuses a final forward that can never fit, as
    // grouped_state_blocks_exceeding_the_pool_reject_at_admission shows: the
    // 40,000-token prompt's last forward needs 24 pages of a 20-page pool.
    let mut off = core(glm_groups(false), 20 * GLM_PAGE, 8192);
    let id = submit(&mut off, 602, 40_000, 2);
    let pass = off.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
}

#[test]
fn grouped_prefix_waiting_request_keeps_fifo() {
    // Each 20,000-token prompt needs 10 G0 blocks, the ring block and four
    // running blocks: 15 of 20. The second request waits for the first one
    // to finish and preempts nothing.
    let mut core = prefix_core(glm_groups(false), 20 * GLM_PAGE, 8192, 2176, "0.31.0");
    let first = submit_tokens(&mut core, 701, (0..20_000).collect(), 2);
    let second = submit_tokens(&mut core, 702, (100_000..120_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let initial = core.execute_pass(&mut trace, step as f64);
    step += 1;
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    let mut saw_first_complete = false;
    let mut admitted_second_at = None;
    for _ in 0..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            admitted_second_at = Some(step);
            assert!(
                saw_first_complete,
                "the second request waited for the first"
            );
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            saw_first_complete |= output.uuid == first && output.completed;
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(admitted_second_at.is_some());
    assert!(core.is_drained());
    assert_eq!(block_pool(&core).active_blocks(), 0);
}

#[test]
fn grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit() {
    // One sliding-window group of 4-token blocks with an 8-token window, a
    // 9-block pool and a step budget of 8 tokens.
    for version in ["0.31.0", "0.32.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // A finished request of 100 tokens keeps the two blocks before its
        // replay boundary 96 cached.
        submit_tokens(&mut core, 1101, (0..100).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        submit_tokens(&mut core, 1102, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1103, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        assert_eq!(block_pool(&core).active_blocks(), 5, "{version}");
        // A request of 104 tokens hits those blocks at 96. The whole-prompt
        // test counts only the two inactive hit blocks, because the window
        // caps it (mgr:519-536). After their decode steps the running
        // requests hold 6 of the 9 blocks, which leaves 3, but the first
        // chunk of 6 tokens needs the two hit blocks and two fresh blocks
        // (mgr:558-576). vLLM then stops its waiting loop (v0.31
        // sched:1223-1230, v0.32 sched:1275-1282), so the request waits and
        // preempts nothing.
        let waiting = submit_tokens(&mut core, 1104, (0..104).collect(), 1);
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (2, 96), "{version}");
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(
            core.state.requests[&waiting].num_computed_tokens, 0,
            "{version}"
        );
        assert_eq!(core.state.running.len(), 2, "{version}");
        // The waiting request starts with its hit once the decode steps of the
        // running requests leave room.
        let admissions = drain(&mut core, &mut trace, &mut step);
        assert_eq!(admissions.first(), Some(&(waiting, 96)), "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_keeps_its_lru_place_when_its_first_chunk_does_not_fit() {
    // vLLM tests the blocks of the first chunk before it touches the hit
    // (mgr:558-589). So a waiting request that fails that test leaves its hit
    // blocks where they are in the LRU, ahead of a block cached after them.
    // Pinning and then releasing them would move them behind that block.
    let cached = |core: &VllmCore, tokens: &[u32], end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(tokens, 4);
        block_pool(core)
            .key_hit(0, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // The LRU holds the blocks that end at 96 and 92 of a finished request
        // of 100 tokens, then the block that ends at 4 of a finished request
        // of 8 tokens.
        let first: Vec<u32> = (0..100).collect();
        let second: Vec<u32> = (3000..3008).collect();
        submit_tokens(&mut core, 1401, first.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        submit_tokens(&mut core, 1402, second.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        // As in grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit,
        // two running requests hold 6 of the 9 blocks, and the request of 104
        // tokens fails the first-chunk test of its hit at 96.
        submit_tokens(&mut core, 1403, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1404, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        let waiting = submit_tokens(&mut core, 1405, (0..104).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(block_pool(&core).active_blocks(), 6, "{version}");
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: waiting,
        })
        .unwrap();
        // No block is free. A request of 4 new tokens takes one fresh block,
        // which evicts the oldest cached block: the one that ends at 96.
        let fresh = submit_tokens(&mut core, 1406, (4000..4004).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, fresh, "{version}");
        assert!(!cached(&core, &first, 96), "{version}");
        assert!(cached(&core, &first, 92), "{version}");
        assert!(cached(&core, &second, 4), "{version}");
    }
}

#[test]
fn grouped_prefix_without_chunked_prefill_budgets_the_uncached_tokens() {
    // With chunked prefill off, vLLM compares the tokens after the prefix hit
    // with the step budget (v0.31 sched:1084, :1115-1121, v0.32 sched:1129,
    // :1160-1167), as the linear pool does. Two running requests decode one
    // token each and leave 14 of the 16 tokens. A prompt of 15 tokens with 12
    // cached tokens needs 3 of them.
    for version in ["0.31.0", "0.32.0"] {
        let args = || {
            MockEngineArgs::builder()
                .block_size(4)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(4))
                .max_model_len(Some(4096))
                .enable_prefix_caching(true)
                .enable_chunked_prefill(false)
                .speedup_ratio(0.0)
        };
        let linear = VllmCore::new(args().num_gpu_blocks(64).build().unwrap());
        let mut full = group("full", 4, 1, None);
        full.prefix_cacheable = Some(true);
        let grouped = VllmCore::new(
            args()
                .num_gpu_blocks(1)
                .prefix_match_unit(Some(4))
                .vllm_version(Some(version.to_string()))
                .kv_cache_groups(vec![full])
                .kv_cache_capacity_bytes(Some(64))
                .build()
                .unwrap(),
        );
        for (pool, mut core) in [("linear", linear), ("grouped", grouped)] {
            let mut trace = TraceCollector::default();
            let mut step = 0;
            submit_tokens(&mut core, 1301, (0..15).collect(), 1);
            drain(&mut core, &mut trace, &mut step);
            submit_tokens(&mut core, 1302, (1000..1004).collect(), 10);
            submit_tokens(&mut core, 1303, (2000..2004).collect(), 10);
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.admissions.len(), 2, "{version} {pool}");
            let waiting = submit_tokens(&mut core, 1304, (0..15).collect(), 1);
            let pass = core.execute_pass(&mut trace, step as f64);
            let admitted: Vec<(Uuid, usize)> = pass
                .admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens))
                .collect();
            assert_eq!(admitted, vec![(waiting, 12)], "{version} {pool}");
        }
    }
}

/// Full-attention and state groups of 16-token blocks, a 4-token hash unit
/// and a pool of five blocks: the blocks of one 34-token request, as vLLM's
/// startup check counts them (3 + 2).
fn fine_grained_core(version: &str) -> VllmCore {
    let mut full = group("full", 16, 8, None);
    full.prefix_cacheable = Some(true);
    let mut state = state("state", 8, 1);
    state.block_size_tokens = 16;
    state.prefix_cacheable = Some(true);
    state.prefill_checkpoint_blocks = (version == "0.32.0").then_some(0);
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .max_model_len(Some(34))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(4))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full, state])
            .kv_cache_capacity_bytes(Some(5 * 8))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

#[test]
fn grouped_prefix_hit_that_needs_more_blocks_than_the_pool_waits() {
    // A finished 13-token request leaves its full-attention block and its
    // state keyed at the hash boundary 12. A 33-token prompt that repeats it
    // hits 12 tokens inside a block. That hit needs 7 blocks: 2 new, the
    // inactive hit block and a copy for full attention (single:216-268), and
    // the running block, a copy and the inactive hit block for the state
    // (single:1716-1816). Cold the prompt needs 4 of the 5 blocks. vLLM keeps
    // such a request waiting and searches its hit again at the next step
    // (v0.31 sched:930-937, :1223-1230, v0.32 sched:960-967, :1275-1282).
    for version in ["0.31.0", "0.32.0"] {
        // With nothing running, nothing changes the cache, so the request
        // waits in every pass. The replay reports that as a livelock
        // (runtime_contract.rs grouped_prefix_hit_above_the_pool_is_a_livelock).
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1501, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        let waiting = submit_tokens(&mut core, 1502, (0..33).collect(), 1);
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (7, 12), "{version}");
        for _ in 0..3 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert!(pass.admissions.is_empty(), "{version}");
            assert!(pass.output_signals.is_empty(), "{version}");
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        }
        assert!(core.state.requests.contains_key(&waiting), "{version}");

        // A running request can evict the hit blocks. Its decode crosses a
        // block boundary at 16 tokens and takes two blocks while one is free,
        // which evicts the cached full-attention block of the hit. The
        // waiting request then hits nothing and starts cold once the running
        // request finishes.
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1503, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let running = submit_tokens(&mut core, 1504, (1000..1004).collect(), 20);
        let waiting = submit_tokens(&mut core, 1505, (0..33).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, running, "{version}");
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
        let mut finished = Vec::new();
        let mut admissions = Vec::new();
        for _ in 0..40 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
            admissions.extend(
                pass.admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens)),
            );
            for output in pass.output_signals {
                assert!(!output.rejected, "{version}");
                if output.completed {
                    finished.push(output.uuid);
                }
            }
            if core.is_drained() {
                break;
            }
        }
        assert_eq!(admissions, vec![(waiting, 0)], "{version}");
        assert_eq!(finished, vec![running, waiting], "{version}");
    }
}

#[test]
fn grouped_prefix_physical_usage_counts_cached_blocks() {
    // Metrics::physical_cache_usage includes the inactive reusable blocks and
    // Metrics::inactive_blocks counts them (protocol.rs). Active usage counts
    // only the blocks that requests hold.
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1201, (0..9000).collect(), 4);
        drain(&mut core, &mut trace, &mut step);
        // G0 keeps its four full blocks, and each KDA group the state at the
        // replay boundary 8704.
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 8, "{version}");
        let metrics = core.mocker_metrics();
        assert_eq!(metrics.kv_cache_used_bytes, Some(0), "{version}");
        assert_eq!(
            metrics.kv_cache_cached_bytes,
            Some(8 * GLM_PAGE),
            "{version}"
        );
        assert_eq!(metrics.gpu_cache_usage_perc, 0.0, "{version}");
        assert_eq!(
            metrics.physical_gpu_cache_usage_perc,
            8.0 / 64.0,
            "{version}"
        );
        assert_eq!(metrics.active_decode_blocks, 0, "{version}");
        assert_eq!(metrics.inactive_decode_blocks, 8, "{version}");
    }
}

#[test]
fn grouped_prefix_eviction_frees_cached_blocks_oldest_first() {
    // One full-attention group of 4-token blocks in a pool of 16 blocks.
    let mut full = group("full", 4, 8, None);
    full.prefix_cacheable = Some(true);
    let mut core = prefix_core(vec![full], 16 * 8, 64, 4, "0.31.0");
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // Three finished requests of 16 tokens leave 12 cached blocks.
    for (id, base) in [(801, 0), (802, 100), (803, 200)] {
        submit_tokens(&mut core, id, (base..base + 16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
    }
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 12);
    // A fourth request of 40 tokens needs 10 fresh blocks: the 4 free ones
    // and the 6 evicted oldest first, in release order (tail first): all of
    // request one and the two last blocks of request two.
    let fourth = submit_tokens(&mut core, 804, (300..340).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fourth, 0)]);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 16);
    // A fifth request equal to the third one still hits, capped below the
    // prompt: (16 - 1) / 4 * 4 = 12.
    let fifth = submit_tokens(&mut core, 805, (200..216).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fifth, 12)]);
    // The fifth request's one fresh block evicted the older of request
    // two's two surviving blocks, so request two hits its first block only.
    let sixth = submit_tokens(&mut core, 806, (100..116).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(sixth, 4)]);
    // Request one is gone.
    let seventh = submit_tokens(&mut core, 807, (0..16).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(seventh, 0)]);
}
