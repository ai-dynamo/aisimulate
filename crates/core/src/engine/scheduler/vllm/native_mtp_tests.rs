// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

// Modified synchronous single-group MTP behavioral tests derived from vLLM:
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py
// See THIRD_PARTY_NOTICES.md for the supported profile.

use super::core::VllmCore;
use super::request::RequestKvState;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::scheduler::{SchedulerCommand, SchedulerCommandResult};
use rstest::rstest;
use uuid::Uuid;

#[rstest]
#[case(1, None, 2, 2)]
#[case(2, None, 2, 2)]
#[case(5, None, 2, 2)]
#[case(1, Some(16), 9, 4)]
#[case(2, Some(16), 9, 4)]
#[case(5, Some(16), 9, 4)]
fn output_and_model_limits_bound_verification_and_release_capacity(
    #[case] nextn: usize,
    #[case] model_limit: Option<usize>,
    #[case] requested_outputs: usize,
    #[case] expected_outputs: usize,
) {
    let mut config = args(nextn, 16, false, false);
    config.max_model_len = model_limit;
    let mut core = VllmCore::new(config);
    let uuid = submit(&mut core, 190, 12, requested_outputs);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let mut now = 0.0;
    let mut output_count = 0;
    let mut completion_count = 0;
    for _ in 0..5 {
        let pass = core.execute_pass(&mut collector, now);
        now = pass.end_ms;
        for signal in pass.output_signals {
            assert_eq!(signal.uuid, uuid);
            assert!(!signal.rejected);
            output_count += usize::from(signal.token_id.is_some());
            completion_count += usize::from(signal.completed);
        }
        if core.is_empty() {
            break;
        }
        if let Some(limit) = model_limit {
            assert!(core.state.requests[&uuid].sequence.len() < limit);
            assert!(core.kv_manager.num_active_blocks() <= limit.div_ceil(4));
        }
    }
    assert_eq!(output_count, expected_outputs);
    assert_eq!(completion_count, 1);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert!(core.is_drained());
}

fn args(nextn: usize, budget: usize, reject_first: bool, host: bool) -> MockEngineArgs {
    let rates = (0..nextn)
        .map(|index| if reject_first && index == 0 { "0" } else { "1" })
        .collect::<Vec<_>>()
        .join(",");
    MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(64)
        .max_num_batched_tokens(Some(budget))
        .max_num_seqs(Some(4))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(host)
        .kv_cache_bytes_per_token(Some(250_000))
        .native_host_offload(
            host.then(|| crate::engine::NativeHostOffloadConfig::new(64).with_bandwidths(0.0, 0.0)),
        )
        .speedup_ratio(0.0)
        .aic_nextn(Some(nextn))
        .aic_nextn_accept_rates(Some(rates))
        .build()
        .unwrap()
}

fn submit(core: &mut VllmCore, id: u128, prompt: usize, output: usize) -> Uuid {
    core.receive(DirectRequest {
        tokens: (0..prompt as u32).collect(),
        max_output_tokens: output,
        output_token_ids: Some((1_000..1_000 + output as u32).collect()),
        uuid: Some(Uuid::from_u128(id)),
        ..Default::default()
    })
}

fn raw_prefix_tokens(core: &VllmCore, tokens: Vec<u32>) -> usize {
    let probe = RequestKvState::native(
        Uuid::from_u128(199),
        tokens,
        1,
        1,
        4,
        true,
        false,
        false,
        None,
    );
    core.kv_manager
        .get_native_prefill_cost(&probe.sequence, &probe.lease)
        .cached_tokens
}

#[test]
fn completed_burst_does_not_publish_its_just_accepted_draft_blocks() {
    let mut cached_args = args(5, 16, false, false);
    cached_args.enable_prefix_caching = true;
    let mut core = VllmCore::new(cached_args);
    let mut collector = crate::engine::trace::TraceCollector::default();
    submit(&mut core, 112, 2, 7);
    let first = core.execute_pass(&mut collector, 0.0);
    assert_eq!(first.output_signals.len(), 1);
    let second = core.execute_pass(&mut collector, first.end_ms);
    assert_eq!(second.output_signals.len(), 6);
    assert_eq!(second.completed_requests, 1);
    assert!(core.is_empty());

    // At the start of the final forward only three tokens were known.
    // Its accepted drafts cross two block boundaries but are never part
    // of a subsequent allocation, so neither block enters APC.
    let continuation = (0..2)
        .chain(1_000..1_007)
        .chain(2_000..2_003)
        .collect::<Vec<_>>();
    assert_eq!(raw_prefix_tokens(&core, continuation.clone()), 0);
    core.receive(DirectRequest {
        tokens: continuation,
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(113)),
        ..Default::default()
    });
    let follower = core.execute_pass(&mut collector, second.end_ms);
    assert_eq!(follower.admissions[0].reused_input_tokens, 0);
    assert_eq!(follower.completed_requests, 1);
}

#[test]
fn accepted_blocks_become_visible_at_the_next_known_token_allocation() {
    let mut cached_args = args(5, 16, false, false);
    cached_args.enable_prefix_caching = true;
    let mut core = VllmCore::new(cached_args);
    let uuid = submit(&mut core, 114, 2, 20);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let first = core.execute_pass(&mut collector, 0.0);
    let second = core.execute_pass(&mut collector, first.end_ms);
    assert_eq!(second.output_signals.len(), 6);
    assert_eq!(core.state.requests[&uuid].sequence.len(), 9);
    assert_eq!(core.state.requests[&uuid].num_computed_tokens, 8);
    let accepted_prefix = (0..2).chain(1_000..1_006).collect::<Vec<_>>();
    assert_eq!(raw_prefix_tokens(&core, accepted_prefix.clone()), 0);

    let follower = core.receive(DirectRequest {
        tokens: accepted_prefix
            .iter()
            .copied()
            .chain(2_000..2_004)
            .collect(),
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(117)),
        ..Default::default()
    });
    let third = core.execute_pass(&mut collector, second.end_ms);
    assert_eq!(
        third
            .output_signals
            .iter()
            .filter(|signal| signal.uuid == uuid)
            .count(),
        6
    );
    assert_eq!(raw_prefix_tokens(&core, accepted_prefix), 8);
    assert_eq!(
        third
            .admissions
            .iter()
            .find(|admission| admission.uuid == follower)
            .unwrap()
            .reused_input_tokens,
        4,
        "next-pass known blocks must be visible to later same-pass admissions, after the MTP block drop"
    );
    core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
        .unwrap();
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn waiting_uniform_padding_is_invalid_even_at_full_acceptance() {
    let mut core = VllmCore::new(args(2, 8, false, false));
    let running = submit(&mut core, 103, 4, 16);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let first = core.execute_pass(&mut collector, 0.0);
    assert_eq!(first.output_signals.len(), 1);

    // The running decode permits a one-token waiting request to join its
    // uniform query shape, but that request has no valid prior drafts.
    let waiting = submit(&mut core, 104, 1, 8);
    let padded = core.execute_pass(&mut collector, first.end_ms);
    assert_eq!(
        padded
            .output_signals
            .iter()
            .filter(|signal| signal.uuid == running)
            .count(),
        3
    );
    assert_eq!(
        padded
            .output_signals
            .iter()
            .filter(|signal| signal.uuid == waiting)
            .count(),
        1
    );
    let request = core.state.requests.get(&waiting).unwrap();
    assert_eq!(request.num_computed_tokens, 1);
    assert_eq!(request.sequence.len(), 2);
    assert_eq!(request.sequence.lease.physical_owned_block_count(), 2);

    let verified = core.execute_pass(&mut collector, padded.end_ms);
    assert_eq!(
        verified
            .output_signals
            .iter()
            .filter(|signal| signal.uuid == waiting)
            .count(),
        3
    );
}

#[test]
fn small_budget_waits_for_running_decode_without_rejecting_a_fresh_prompt() {
    let mut core = VllmCore::new(args(2, 2, false, false));
    let mut collector = crate::engine::trace::TraceCollector::default();
    submit(&mut core, 180, 1, 2);
    let first = core.execute_pass(&mut collector, 0.0);
    let waiting = submit(&mut core, 181, 1, 1);
    let second = core.execute_pass(&mut collector, first.end_ms);
    assert!(
        second
            .output_signals
            .iter()
            .all(|signal| signal.uuid != waiting)
    );
    let third = core.execute_pass(&mut collector, second.end_ms);
    assert_eq!(third.output_signals.len(), 1);
    assert_eq!(third.output_signals[0].uuid, waiting);
    assert!(third.output_signals[0].completed);
    assert!(!third.output_signals[0].rejected);
    assert!(core.is_empty());
}

#[test]
fn loaded_prefix_rejects_a_permanently_impossible_padding_budget() {
    let mut core = VllmCore::new(args(2, 1, false, true));
    let mut collector = crate::engine::trace::TraceCollector::default();
    submit(&mut core, 182, 12, 1);
    let mut now = 0.0;
    for _ in 0..12 {
        let pass = core.execute_pass(&mut collector, now);
        now = pass.end_ms;
        core.complete_engine_boundary(now);
    }
    assert!(core.is_empty());
    let restored = submit(&mut core, 183, 12, 1);
    let mut terminal = None;
    for _ in 0..4 {
        let pass = core.execute_pass(&mut collector, now);
        now = pass.end_ms;
        core.complete_engine_boundary(now);
        terminal = pass
            .output_signals
            .into_iter()
            .find(|signal| signal.uuid == restored);
        if terminal.is_some() {
            break;
        }
    }
    let terminal = terminal.expect("impossible padding must not keep the request queued forever");
    assert!(terminal.completed && terminal.rejected);
    assert!(terminal.token_id.is_none());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert!(core.is_drained());
}

#[test]
fn rejected_drafts_keep_five_blocks_for_n13_c12_k2_until_cancellation() {
    let mut core = VllmCore::new(args(2, 16, true, false));
    let uuid = submit(&mut core, 105, 12, 16);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let first = core.execute_pass(&mut collector, 0.0);
    let request = core.state.requests.get(&uuid).unwrap();
    assert_eq!(request.sequence.len(), 13);
    assert_eq!(request.num_computed_tokens, 12);
    assert_eq!(request.sequence.lease.physical_owned_block_count(), 4);

    let second = core.execute_pass(&mut collector, first.end_ms);
    assert_eq!(second.output_signals.len(), 1);
    let request = core.state.requests.get(&uuid).unwrap();
    assert_eq!(request.sequence.len(), 14);
    assert_eq!(request.num_computed_tokens, 13);
    assert_eq!(request.sequence.lease.physical_owned_block_count(), 5);
    assert_eq!(core.kv_manager.num_active_blocks(), 5);

    let third = core.execute_pass(&mut collector, second.end_ms);
    assert_eq!(third.output_signals.len(), 1);
    assert_eq!(core.kv_manager.num_active_blocks(), 5);
    assert_eq!(
        core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
            .unwrap(),
        SchedulerCommandResult::Applied
    );
    assert!(core.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert_eq!(
        core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
            .unwrap(),
        SchedulerCommandResult::Noop
    );
}

#[rstest]
#[case(false)]
#[case(true)]
fn preemption_reports_and_releases_opaque_mtp_capacity(#[case] host: bool) {
    let mut config = args(2, 16, true, host);
    config.num_gpu_blocks = 5;
    let mut core = VllmCore::new(config);
    let first_id = submit(&mut core, 120, 4, 16);
    let second_id = submit(&mut core, 121, 4, 16);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let first = core.execute_pass(&mut collector, 0.0);
    core.complete_engine_boundary(first.end_ms);
    let lease = &core.state.requests[&second_id].sequence.lease;
    assert_eq!(lease.resident_block_count(), 1);
    assert_eq!(lease.physical_owned_block_count(), 2);

    // The first decoder grows to three blocks. The second cannot grow in
    // this five-block pool, so real scheduling preempts its known and opaque
    // copies together; the pressure event must describe both released copies.
    let second = core.execute_pass(&mut collector, first.end_ms);
    let [pressure] = second.pressure_events.as_slice() else {
        panic!("the second decoder must be preempted once")
    };
    assert_eq!(pressure.request_id, second_id);
    assert_eq!(pressure.request_active_blocks_before, 2);
    assert_eq!(pressure.state_before.active_blocks, 5);
    assert_eq!(pressure.state_after.active_blocks, 3);
    assert_eq!(
        core.state.requests[&second_id]
            .sequence
            .lease
            .physical_owned_block_count(),
        0
    );
    for request_id in [first_id, second_id] {
        core.apply_command(SchedulerCommand::CancelRequest { request_id })
            .unwrap();
    }
    core.complete_engine_boundary(second.end_ms);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
    assert!(core.is_drained());
}

#[test]
fn impossible_lookahead_rejects_without_starving_the_next_request() {
    let mut small = args(2, 8, false, false);
    small.num_gpu_blocks = 1;
    let mut core = VllmCore::new(small);
    let oversized = submit(&mut core, 110, 4, 4);
    let follower = submit(&mut core, 111, 1, 1);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let pass = core.execute_pass(&mut collector, 0.0);
    assert!(
        pass.output_signals
            .iter()
            .any(|signal| { signal.uuid == oversized && signal.completed && signal.rejected })
    );
    assert!(
        pass.output_signals
            .iter()
            .any(|signal| { signal.uuid == follower && signal.completed && !signal.rejected })
    );
    assert!(core.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(5)]
fn partial_prefill_reserves_lookahead_without_emitting_drafts(#[case] nextn: usize) {
    let mut core = VllmCore::new(args(nextn, 4, false, false));
    let uuid = submit(&mut core, 101, 12, 16);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let mut now = 0.0;
    for computed in [4usize, 8, 12] {
        let pass = core.execute_pass(&mut collector, now);
        now = pass.end_ms;
        assert_eq!(pass.output_signals.len(), usize::from(computed == 12));
        let request = core.state.requests.get(&uuid).unwrap();
        assert_eq!(request.num_computed_tokens, computed);
        assert_eq!(request.sequence.len(), 12 + usize::from(computed == 12));
        assert_eq!(
            request.sequence.lease.physical_owned_block_count(),
            (computed + nextn).div_ceil(4)
        );
        assert_eq!(pass.fpm.unwrap().sum_prefill_tokens, 4);
    }

    let pass = core.execute_pass(&mut collector, now);
    assert_eq!(pass.output_signals.len(), 1 + nextn.min(3));
    let request = core.state.requests.get(&uuid).unwrap();
    assert_eq!(request.sequence.len() - request.num_computed_tokens, 1);
    assert_eq!(
        request.sequence.lease.physical_owned_block_count(),
        (12 + (nextn + 1).min(4) + nextn).div_ceil(4)
    );
}

#[rstest]
#[case(1, 1)]
#[case(2, 2)]
#[case(5, 3)]
fn only_scheduled_drafts_can_be_accepted(#[case] nextn: usize, #[case] budget: usize) {
    let mut core = VllmCore::new(args(nextn, budget, false, false));
    let uuid = submit(&mut core, 102, 1, 32);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let first = core.execute_pass(&mut collector, 0.0);
    assert_eq!(first.output_signals.len(), 1);

    let mut now = first.end_ms;
    for generation in 1..=3 {
        let pass = core.execute_pass(&mut collector, now);
        now = pass.end_ms;
        assert_eq!(pass.output_signals.len(), budget);
        let request = core.state.requests.get(&uuid).unwrap();
        assert_eq!(request.sequence.generated_tokens(), 1 + generation * budget);
        assert_eq!(request.sequence.len() - request.num_computed_tokens, 1);
    }
}
