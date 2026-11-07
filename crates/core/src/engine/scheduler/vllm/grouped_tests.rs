// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{
    DirectRequest, KvTransferTimingMode, MockEngineArgs, WorkerType,
};
use crate::engine::kv_manager::BlockRequestLease;
use crate::engine::scheduler::{SchedulerCommand, SchedulerLifecycleEvent};
use crate::engine::trace::TraceCollector;
use crate::engine::{HandoffId, HandoffTransferTiming};
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

/// `core` for a prefill or decode rank, which times a full-prompt transfer at
/// `bandwidth` decimal GB/s.
fn pd_core(
    groups: Vec<FpmCacheGroup>,
    capacity: u64,
    chunk: usize,
    worker_type: WorkerType,
    bandwidth: Option<f64>,
) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .max_model_len(Some(200_000))
            .enable_prefix_caching(false)
            .worker_type(worker_type)
            .kv_transfer_bandwidth(bandwidth)
            .kv_transfer_timing_mode(KvTransferTimingMode::FullPrompt)
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

/// A handoff prefill as Replay sends it, with one output token.
fn source_request(id: u128, prompt: usize) -> DirectRequest {
    DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens: (0..prompt as u32).collect(),
        max_output_tokens: 1,
        ..Default::default()
    }
}

/// Step until the prefill rank holds its source, and return the timing that
/// the held source reports.
fn held_timing(core: &mut VllmCore) -> HandoffTransferTiming {
    for _ in 0..8 {
        let pass = core.try_execute_hidden_pass(0.0).unwrap();
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
        if let Some(timing) = pass.lifecycle_events.iter().find_map(|event| match event {
            SchedulerLifecycleEvent::SourceHeld {
                transfer_timing, ..
            } => Some(*transfer_timing),
            _ => None,
        }) {
            return timing;
        }
    }
    panic!("source must be held");
}

/// Prefill a source of `prompt` tokens and return its held timing, like
/// `hold_source` of `core.rs`.
fn hold_grouped_source(
    core: &mut VllmCore,
    handoff_id: HandoffId,
    id: u128,
    prompt: usize,
) -> HandoffTransferTiming {
    core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
        handoff_id,
        request: source_request(id, prompt),
    })
    .unwrap();
    held_timing(core)
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

/// The six GLM-5.3-Flash TP 4 groups that the runtime probe recorded on
/// vLLM 0.31.0, at one page size. With `checkpoint`, G2 to G5 carry one
/// FlashKDA checkpoint block at alignment 16.
fn glm_groups(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut mla = group("mla", GLM_BLOCK, GLM_PAGE, None);
    mla.prefix_cacheable = Some(true);
    let mut groups = vec![mla, glm_state("kpool_tail", 4, false, false)];
    for i in 0..4 {
        groups.push(glm_state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

/// `glm_groups` with the transfer bytes of each group's own layers:
/// `11 * 1_114_112 + 11 * 71_808` for the MLA latent and the indexer keys,
/// `11 * 2_048` for the tail ring, and `9 * 1_085_440` or `8 * 1_085_440` for
/// a KDA state. The fixed bytes are 36,927,488, one block of each state group.
fn glm_transfer_groups(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut groups = glm_groups(checkpoint);
    let bytes = [
        13_045_120, 22_528, 9_768_960, 9_768_960, 8_683_520, 8_683_520,
    ];
    for (group, bytes) in groups.iter_mut().zip(bytes) {
        group.transfer_bytes_per_block = Some(bytes);
    }
    groups
}

/// Three GLM cores: vLLM 0.31.0, and vLLM 0.32.0 without and with checkpoint
/// blocks, which GLM-5.3-Flash takes only under the FlashKDA prefill backend.
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
    let handoff_id = HandoffId::new(Uuid::from_u128(108));
    let error = core
        .apply_command(SchedulerCommand::ReserveDestination {
            handoff_id,
            request: DirectRequest {
                tokens: vec![0, 1, 2, 3],
                max_output_tokens: 2,
                ..Default::default()
            },
        })
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("grouped KV caches do not support destination handoff commands"),
        "{error}"
    );
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_aggregated_rank_refuses_source_commands() {
    // An aggregated rank builds no transfer rules, so it cannot price a
    // held source.
    let mut core = core(glm_transfer_groups(false), 128 * GLM_PAGE, 8192);
    let handoff_id = HandoffId::new(Uuid::from_u128(1624));
    for command in [
        SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: source_request(1623, 1_000),
        },
        SchedulerCommand::ReleaseSource { handoff_id },
        SchedulerCommand::CancelSource { handoff_id },
    ] {
        let error = core.apply_command(command).unwrap_err();
        assert_eq!(
            error.to_string(),
            "grouped KV caches support KV handoff commands only on prefill and decode ranks"
        );
    }
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_prefill_source_keeps_only_the_blocks_inside_its_window() {
    // A prompt of 1,024 tokens in one chunk, with a window of 32 tokens and
    // blocks of 16. The forward keeps the 64 blocks that its first query
    // reads. When a source completes, vLLM frees the blocks that the next
    // query skips (V32 `vllm/v1/core/sched/scheduler.py:3023-3031`):
    // 1,024 - 32 + 1 = 993 tokens, so the 62 blocks below 992 go and 2 stay.
    let mut window = group("window", 16, 100, Some(32));
    window.prefix_cacheable = Some(true);
    window.transfer_bytes_per_block = Some(100);
    let byte_mode = pd_core(
        vec![window.clone()],
        67 * 100,
        1024,
        WorkerType::Prefill,
        None,
    );
    let mut args = prefix_core(vec![window], 67 * 100, 1024, 16, "0.32.0").args;
    args.worker_type = WorkerType::Prefill;
    for (mode, mut core) in [("byte", byte_mode), ("block", VllmCore::new(args))] {
        let handoff_id = HandoffId::new(Uuid::from_u128(1626));
        hold_grouped_source(&mut core, handoff_id, 1625, 1024);
        let held = |core: &VllmCore| core.kv_manager.grouped().unwrap().active_blocks();
        assert_eq!(held(&core), 2, "{mode}");
        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(held(&core), 0, "{mode}");
        assert!(core.is_drained(), "{mode}");
    }
}

#[test]
fn grouped_prefill_source_truncates_one_token_and_prices_the_payload_in_blocks() {
    // N, the N - 1 tokens that the prefill computes, their attention blocks
    // and the payload time at 1 GB/s: `13_045_120 * blocks + 36_927_488`
    // bytes.
    for (prompt, computed, blocks, delay_ms) in [
        (1_000, 999, 1, 49.972608),
        (2_177, 2_176, 1, 49.972608),
        (8_192, 8_191, 4, 89.107968),
    ] {
        let mut core = pd_core(
            glm_transfer_groups(false),
            128 * GLM_PAGE,
            8192,
            WorkerType::Prefill,
            Some(1.0),
        );
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let handoff_id = HandoffId::new(Uuid::from_u128(1602));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: source_request(1601, prompt),
        })
        .unwrap();
        // The request that the source will hold lacks the last prompt token.
        let id = Uuid::from_u128(1601);
        assert_eq!(
            core.state.requests[&id].sequence.num_input_tokens(),
            computed,
            "N {prompt}"
        );
        let timing = held_timing(&mut core);
        assert_eq!(timing.full_prompt_tokens, blocks, "N {prompt}");
        assert_eq!(timing.kv_bytes_per_token, Some(13_045_120));
        assert_eq!(timing.state_bytes, 36_927_488);
        assert!(
            timing
                .delay_ms(0)
                .is_some_and(|delay| (delay - delay_ms).abs() < 1e-9),
            "N {prompt}: {:?}",
            timing.delay_ms(0)
        );
        // The prefill computes and prices N - 1 tokens in one chunk.
        assert_eq!(
            *captured.1.lock().unwrap(),
            vec![(1, computed, 0)],
            "N {prompt}"
        );
        // The held source keeps its attention blocks, the ring block and the
        // one block of each KDA group.
        assert!(core.source_is_held(handoff_id));
        assert!(!core.state.requests.contains_key(&id));
        assert_eq!(
            core.kv_manager.grouped().unwrap().active_blocks(),
            blocks + 5,
            "N {prompt}"
        );
        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(core.kv_manager.grouped().unwrap().used_bytes(), 0);
        assert!(core.is_drained(), "N {prompt}");
    }
}

#[test]
fn grouped_prefill_without_state_layers_keeps_the_whole_prompt() {
    // Without a Mamba state the backoff is 0: the prefill computes all 130
    // tokens, which take three 64-token blocks, the last one partly filled.
    let mut full = group("full", 64, 100, None);
    full.prefix_cacheable = Some(true);
    full.transfer_bytes_per_block = Some(100);
    let mut core = pd_core(vec![full], 1_000, 8192, WorkerType::Prefill, Some(1.0));
    let captured = Arc::new(CapturedDecodeQueries::default());
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: captured.clone(),
    });
    let handoff_id = HandoffId::new(Uuid::from_u128(1604));
    let timing = hold_grouped_source(&mut core, handoff_id, 1603, 130);
    assert_eq!(*captured.1.lock().unwrap(), vec![(1, 130, 0)]);
    assert_eq!(timing.full_prompt_tokens, 3);
    assert_eq!(timing.kv_bytes_per_token, Some(100));
    assert_eq!(timing.state_bytes, 0);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(300));
    // Cancelling a held source frees it as a release does.
    core.apply_command(SchedulerCommand::CancelSource { handoff_id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert!(core.is_drained());
}

#[test]
fn grouped_prefill_source_in_block_mode_frees_its_checkpoint_blocks_at_completion() {
    let handoff_id = HandoffId::new(Uuid::from_u128(1606));
    // `prefix_core` on vLLM 0.32.0 as a prefill rank that holds a source of
    // `prompt` tokens. Returns the core, its prefill chunks and the timing.
    let hold = |checkpoint: bool, prompt: usize| {
        let mut args = prefix_core(
            glm_transfer_groups(checkpoint),
            128 * GLM_PAGE,
            8192,
            2176,
            "0.32.0",
        )
        .args;
        args.worker_type = WorkerType::Prefill;
        args.kv_transfer_bandwidth = Some(1.0);
        let mut core = VllmCore::new(args);
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let timing = hold_grouped_source(&mut core, handoff_id, 1605, prompt);
        let chunks = captured.1.lock().unwrap().clone();
        (core, chunks, timing)
    };
    // The block of each KDA group keyed at `at` tokens of the prompt
    // `0..prompt`: active while a request holds it, inactive once only the
    // cache does.
    let kda_keys = |core: &VllmCore, prompt: u32, at: usize| -> Vec<Option<bool>> {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(&(0..prompt).collect::<Vec<u32>>(), 2176);
        let hash = lease.prefix_hash_at(at).unwrap();
        (2..6)
            .map(|g| block_pool(core).key_hit(g, hash).map(|hit| hit.is_active))
            .collect()
    };
    let release = |core: &mut VllmCore| {
        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(block_pool(core).active_blocks(), 0);
        assert!(core.is_drained());
    };
    // When a source completes, vLLM frees in each KDA group every block below
    // the state of the last computed token (V32
    // `vllm/v1/core/sched/scheduler.py:3023-3031`,
    // `vllm/v1/core/single_type_kv_cache_manager.py:1613-1641`). So the held
    // source keeps only the running block there. A freed block stays cached.

    // With one FlashKDA checkpoint block per KDA group, the truncated prompt
    // of 8,191 tokens runs in one chunk. Its allocation reaches the end of
    // that prompt, so each KDA group takes a checkpoint block. Finalize keys
    // it at 6,528, the last block boundary below the end. A first allocation
    // from 0 replaces no running block, so no previous block holds that key.
    let (mut core, chunks, timing) = hold(true, 8_192);
    assert_eq!(chunks, vec![(1, 8191, 0)]);
    // The payload is that of byte mode. The checkpoint block does not move.
    assert_eq!(timing.full_prompt_tokens, 4);
    assert_eq!(timing.kv_bytes_per_token, Some(13_045_120));
    assert_eq!(timing.state_bytes, 36_927_488);
    // Four attention blocks, the ring block and the running block of each
    // KDA group. The checkpoint blocks left at completion.
    assert_eq!(block_pool(&core).active_blocks(), 9);
    assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);
    release(&mut core);
    assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);

    // Without checkpoint blocks the split stops at 6,528, where finalize
    // keys the running block. The next chunk replaces it, and the previous
    // block with that key leaves at completion. The block count and the key
    // are the same, so only the chunks tell the two cores apart.
    let (core, chunks, _) = hold(false, 8_192);
    assert_eq!(chunks, vec![(1, 6528, 0), (1, 1663, 6528)]);
    assert_eq!(block_pool(&core).active_blocks(), 9);
    assert_eq!(kda_keys(&core, 8191, 6528), vec![Some(false); 4]);

    // A source of 6,529 tokens ties the checkpoint to the truncated prompt
    // end. Its prompt of 6,528 tokens puts the checkpoint at 4,352. An
    // untruncated prompt of 6,529 tokens would put it at 6,528.
    let (mut core, chunks, timing) = hold(true, 6_529);
    assert_eq!(chunks, vec![(1, 6528, 0)]);
    assert_eq!(timing.full_prompt_tokens, 3);
    // Three attention blocks, the ring block and the running block of each
    // KDA group.
    assert_eq!(block_pool(&core).active_blocks(), 8);
    assert_eq!(kda_keys(&core, 6528, 4352), vec![Some(false); 4]);
    assert_eq!(kda_keys(&core, 6528, 6528), vec![None; 4]);
    release(&mut core);
    assert_eq!(kda_keys(&core, 6528, 4352), vec![Some(false); 4]);
}

#[test]
fn grouped_prefill_source_preempted_in_its_prefill_stays_truncated_once() {
    // Two sources of 4,354 tokens on 17 pages. The split ends the first
    // chunk of the first source at 4,352, which leaves budget to the second
    // source. The last token of the first source then needs one attention
    // block and a new running block in each KDA group, 5 blocks, while 4 are
    // free, so the second source is preempted.
    let mut args = prefix_core(
        glm_transfer_groups(false),
        17 * GLM_PAGE,
        8192,
        2176,
        "0.32.0",
    )
    .args;
    args.worker_type = WorkerType::Prefill;
    let mut core = VllmCore::new(args);
    let first = HandoffId::new(Uuid::from_u128(1610));
    let second = HandoffId::new(Uuid::from_u128(1612));
    for (handoff_id, request) in [
        (first, source_request(1609, 4_354)),
        (
            second,
            DirectRequest {
                tokens: (100_000..104_354).collect(),
                ..source_request(1611, 0)
            },
        ),
    ] {
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request,
        })
        .unwrap();
    }
    assert_eq!(held_timing(&mut core).full_prompt_tokens, 3);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 1);
    // The preempted source restarts from 0 on the prompt that its submission
    // truncated, as vLLM's `_p_side_truncated` guard keeps a rescheduled
    // request from losing a second token (V31
    // `nixl/base_scheduler.py:415-423`, V32 `:411-419`).
    let preempted = &core.state.requests[&Uuid::from_u128(1611)];
    assert_eq!(preempted.sequence.num_input_tokens(), 4_353);
    assert_eq!(preempted.num_computed_tokens, 0);
    core.apply_command(SchedulerCommand::ReleaseSource { handoff_id: first })
        .unwrap();
    // 4,353 tokens take 3 attention blocks. A second truncation would give 2.
    assert_eq!(held_timing(&mut core).full_prompt_tokens, 3);
}

#[test]
fn grouped_destination_commands_still_fail_in_this_build() {
    let mut core = pd_core(
        glm_transfer_groups(false),
        128 * GLM_PAGE,
        8192,
        WorkerType::Decode,
        None,
    );
    let handoff_id = HandoffId::new(Uuid::from_u128(1608));
    for command in [
        SchedulerCommand::ReserveDestination {
            handoff_id,
            request: DirectRequest {
                max_output_tokens: 3,
                ..source_request(1607, 1_000)
            },
        },
        SchedulerCommand::ActivateDestination { handoff_id },
        SchedulerCommand::CancelDestination { handoff_id },
    ] {
        let error = core.apply_command(command).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("grouped KV caches do not support destination handoff commands"),
            "{error}"
        );
    }
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_aggregated_rank_builds_no_transfer_rules() {
    // Groups that carry every transfer field, in byte and in block mode.
    let byte_mode = core(glm_transfer_groups(false), 128 * GLM_PAGE, 8192);
    assert!(byte_mode.kv_manager.grouped_transfer().is_none());
    let block_mode = prefix_core(
        glm_transfer_groups(true),
        128 * GLM_PAGE,
        8192,
        2176,
        "0.32.0",
    );
    assert!(block_mode.kv_manager.grouped_transfer().is_none());
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

        // Under contention the whole prompt decides, as in vLLM's default
        // admission (mgr:519-536). A running request of 13,000 tokens holds
        // 3 G0 pages, the ring and four state blocks after its first chunk,
        // which the split ends at 6,528, so 22 of 30 blocks are free. The
        // 40,000-token prompt needs 24 and waits without preempting anything.
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

/// `(id, submit pass, prompt token ranges, output tokens)`.
type PlannedRequest = (u128, usize, &'static [(u32, u32)], usize);

/// Twelve requests with shared prefixes, chunked prefill and eviction.
const SINGLE_GROUP_PLAN: [PlannedRequest; 12] = [
    (901, 0, &[(0, 300)], 3),
    (902, 0, &[(0, 300)], 3),
    (903, 3, &[(0, 256)], 2),
    (904, 6, &[(0, 100), (1000, 1150)], 4),
    (905, 8, &[(5000, 6600)], 5),
    (906, 9, &[(7000, 9000)], 6),
    (907, 10, &[(10_000, 12_500)], 3),
    (908, 30, &[(13_000, 14_600)], 2),
    (909, 45, &[(5000, 6600), (30_000, 30_030)], 3),
    (910, 60, &[(40_000, 40_192), (50_000, 50_040)], 2),
    (911, 60, &[(40_000, 40_192), (51_000, 51_010)], 2),
    (912, 70, &[(0, 300), (60_000, 60_500)], 4),
];

/// A linear core and a grouped core with one full-attention group of the
/// same block and capacity, with every other setting equal.
fn single_group_cores(version: &str) -> (VllmCore, VllmCore) {
    let args = || {
        MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(256))
            .max_num_seqs(Some(4))
            .max_model_len(Some(4096))
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
    };
    let linear = VllmCore::new(args().num_gpu_blocks(128).build().unwrap());
    let mut full = group("full", 64, 1, None);
    full.prefix_cacheable = Some(true);
    let grouped = VllmCore::new(
        args()
            .num_gpu_blocks(1)
            .prefix_match_unit(Some(64))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full])
            .kv_cache_capacity_bytes(Some(128))
            .build()
            .unwrap(),
    );
    (linear, grouped)
}

#[test]
fn single_full_attention_group_matches_the_linear_pool() {
    // Both rule rows. One full-attention group at the hash block leaves every
    // row difference inert (the last cacheable stop, the partial tail and the
    // checkpoint block), so both rows match the linear pool.
    for version in ["0.31.0", "0.32.0"] {
        let (mut linear, mut grouped) = single_group_cores(version);
        let mut linear_trace = TraceCollector::default();
        let mut grouped_trace = TraceCollector::default();
        let computed = |core: &VllmCore| {
            let mut computed: Vec<(Uuid, usize)> = core
                .state
                .requests
                .iter()
                .map(|(id, request)| (*id, request.num_computed_tokens))
                .collect();
            computed.sort();
            computed
        };
        let mut next = 0;
        let mut hits = Vec::new();
        for pass in 0..400 {
            while let Some(&(id, at, ranges, output)) = SINGLE_GROUP_PLAN.get(next)
                && at == pass
            {
                let tokens: Vec<u32> = ranges.iter().flat_map(|&(a, b)| a..b).collect();
                submit_tokens(&mut linear, id, tokens.clone(), output);
                submit_tokens(&mut grouped, id, tokens, output);
                next += 1;
            }
            let expected = linear.execute_pass(&mut linear_trace, pass as f64);
            let actual = grouped.execute_pass(&mut grouped_trace, pass as f64);
            let reused = |admissions: &[crate::engine::scheduler::AdmissionEvent]| {
                admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                reused(&actual.admissions),
                reused(&expected.admissions),
                "{version} pass {pass}"
            );
            assert_eq!(
                computed(&grouped),
                computed(&linear),
                "{version} pass {pass}"
            );
            assert_eq!(
                grouped.kv_manager.num_active_blocks(),
                linear.kv_manager.num_active_blocks(),
                "{version} pass {pass}"
            );
            assert!(
                expected
                    .output_signals
                    .iter()
                    .chain(&actual.output_signals)
                    .all(|output| !output.rejected)
            );
            hits.extend(reused(&actual.admissions));
            if next == SINGLE_GROUP_PLAN.len() && linear.is_drained() && grouped.is_drained() {
                break;
            }
        }
        assert!(linear.is_drained() && grouped.is_drained(), "{version}");
        // The linear cap is `known - 1` floored to the block
        // (`apply_prefix_recompute`), the grouped one `num_tokens - 1` floored to
        // the alignment (`mgr:289-295`). With one group both are the block.
        let id = Uuid::from_u128;
        assert_eq!(
            hits,
            vec![
                (id(901), 0),
                // The four blocks that request 901 holds.
                (id(902), 256),
                // Four cached blocks, capped below the prompt: (256 - 1) / 64 * 64.
                (id(903), 192),
                (id(904), 64),
                (id(905), 0),
                (id(906), 0),
                (id(907), 0),
                (id(908), 0),
                (id(909), 1600),
                (id(910), 0),
                // Request 910's three blocks, in the pass that computes them.
                (id(911), 192),
                // Eviction left block 0 of request 901 only.
                (id(912), 64),
            ],
            "{version}"
        );
    }
}
