// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Cache-salt behavior follows SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d); see THIRD_PARTY_NOTICES.md.

// Admission and ratio lifecycle follow SGLang v0.5.22 (baa9befa381511cd4d56f83980951e5a45fffc6d); see THIRD_PARTY_NOTICES.md.
// Modified behavioral adaptation of SGLang v0.5.20; see THIRD_PARTY_NOTICES.md.
// Source: https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e

use super::request::WaitingQueue;

use uuid::Uuid;

use super::config::{SchedulePolicy, SglangConfig};
use super::core::SglangCore;
use super::decode;
use super::decode::simulate_decode_step;
use super::policy::apply_schedule_policy;
use super::prefill::get_new_batch_prefill;
use super::request::SglangRequest;
use crate::engine::HandoffId;
use crate::engine::belady::BeladyOracle;
use crate::engine::common::hashing::{compute_block_hash_for_seq, compute_seq_hash_for_block};
use crate::engine::common::protocols::{
    DirectRequest, EngineType, KvEventPublishers, MockEngineArgs, OutputSignal, SglangArgs,
};
use crate::engine::kv_manager::SglangKvManager;
use crate::engine::kv_manager::sglang_backend::RadixRequestLease;
use crate::engine::scheduler::{
    EnginePassResult, SchedulerCommand, SchedulerCommandResult, SchedulerLifecycleEvent,
    capture_kv_event_sink,
};
use crate::engine::{EncoderShape, ImageSpec, KvEvent, KvEventData, PressureKind};

fn stored_hashes(events: &[KvEvent]) -> Vec<u64> {
    events
        .iter()
        .flat_map(|event| match &event.data {
            KvEventData::Stored(stored) => stored
                .blocks
                .iter()
                .map(|block| block.block_hash)
                .collect::<Vec<_>>(),
            KvEventData::Removed { .. } => Vec::new(),
        })
        .collect()
}

fn removed_event_count(events: &[KvEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event.data, KvEventData::Removed { .. }))
        .count()
}

fn test_args(
    num_gpu_blocks: usize,
    block_size: usize,
    chunked_prefill_size: usize,
) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(num_gpu_blocks)
        .block_size(block_size)
        .speedup_ratio(1.0)
        .sglang(Some(SglangArgs {
            page_size: Some(block_size),
            chunked_prefill_size: Some(chunked_prefill_size),
            num_reserved_decode_tokens: Some(0),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

fn direct_request(tokens: Vec<u32>, max_output_tokens: usize) -> DirectRequest {
    DirectRequest {
        tokens,
        max_output_tokens,
        uuid: None,
        arrival_timestamp_ms: None,
        ..Default::default()
    }
}

mod hicache_tests {
    use std::sync::Arc;

    use super::*;
    use crate::engine::{CacheTierAttribution, KvEventTier, SglangHiCacheConfig};

    struct FixedTiming(f64);

    impl crate::engine::TimingModel for FixedTiming {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(self.0)
        }

        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(self.0)
        }
    }

    fn core(compute_ms: f64) -> SglangCore {
        core_with_loop(compute_ms, false, 64)
    }

    fn core_with_loop(compute_ms: f64, host_loop: bool, chunk: usize) -> SglangCore {
        core_with_policy(
            compute_ms,
            host_loop,
            chunk,
            crate::engine::SglangHiCacheWritePolicy::WriteThrough,
        )
    }

    fn core_with_policy(
        compute_ms: f64,
        host_loop: bool,
        chunk: usize,
        policy: crate::engine::SglangHiCacheWritePolicy,
    ) -> SglangCore {
        let mut args = test_args(16, 4, chunk);
        args.sglang.as_mut().unwrap().host_loop = host_loop;
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming(compute_ms)),
        }
        .into();
        args.kv_cache_bytes_per_token = Some(250_000);
        args.sglang.as_mut().unwrap().hicache = Some(SglangHiCacheConfig {
            write_policy: policy,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
            ..SglangHiCacheConfig::new(32)
        });
        SglangCore::new_with_kv_capture(args, 0)
    }

    #[rstest::rstest]
    #[case(1.0)]
    #[case(8.0)]
    fn hicache_idle_drain_and_same_pass_load_back_follow_transfer_deadlines(
        #[case] compute_ms: f64,
    ) {
        let mut core = core(compute_ms);
        let prompt = (0..16).collect::<Vec<_>>();
        core.receive(direct_request(prompt.clone(), 1));
        let warm = core.execute_pass_internal(None, 0.0);
        assert_eq!(warm.end_ms, compute_ms);
        assert!(!core.is_ready());
        assert!(!core.is_drained());
        let backup_end = warm.end_ms + 4.0;
        assert_eq!(core.next_internal_deadline_ms(), Some(backup_end));
        core.process_internal_work(backup_end - 1.0);
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 0);
        core.process_internal_work(backup_end);
        assert!(core.is_drained());
        assert!(
            core.drain_kv_events()
                .iter()
                .any(|event| event.tier == KvEventTier::HostPinned)
        );

        core.kv_manager.evict(16);
        core.receive(direct_request(prompt, 1));
        let reused = core.execute_pass_internal(None, backup_end);
        assert_eq!(reused.end_ms, backup_end + compute_ms.max(3.0));
        assert_eq!(reused.admissions[0].reused_input_tokens, 12);
        assert_eq!(
            reused.admissions[0].cache_tier_attribution,
            Some(CacheTierAttribution {
                g1_reused_input_tokens: 0,
                host_reused_input_tokens: 12,
            })
        );
        assert!(reused.output_signals.iter().any(|output| output.completed));
        assert!(core.is_drained());
    }

    #[test]
    fn hicache_rejected_candidate_does_not_load_or_publish_device_pages() {
        let mut core = core(1.0);
        core.config.max_running_requests = 1;
        let prompt = (0..16).collect::<Vec<_>>();
        core.receive(direct_request(prompt.clone(), 1));
        let warm = core.execute_pass_internal(None, 0.0);
        let backup_end = warm.end_ms + 4.0;
        core.process_internal_work(backup_end);
        core.kv_manager.evict(16);
        core.receive(direct_request(vec![99; 8], 8));
        let first = core.execute_pass_internal(None, backup_end);
        core.receive(direct_request(prompt.clone(), 1));
        let rejected = core.execute_pass_internal(None, first.end_ms);
        assert!(rejected.admissions.is_empty());
        assert_eq!(core.kv_manager.cache().prefix_match_len(&prompt), 0);
        assert_eq!(core.kv_manager.host_load_end_ms(), first.end_ms);
    }

    #[rstest::rstest]
    #[case(false)]
    #[case(true)]
    fn hicache_write_back_stall_delays_admission_and_forward(#[case] host_loop: bool) {
        let mut core = core_with_policy(
            1.0,
            host_loop,
            64,
            crate::engine::SglangHiCacheWritePolicy::WriteBack,
        );
        for first in [0, 100, 200, 300] {
            let tokens = (first..first + 16).collect::<Vec<_>>();
            let lease = core.kv_manager.allocate_for_request(&tokens).unwrap().lease;
            core.kv_manager.finish(&tokens, lease);
        }
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 0);
        core.receive(direct_request((400..416).collect(), 1));
        let first = core.execute_pass_internal(None, 0.0);
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 4);
        assert_eq!(first.end_ms, if host_loop { 4.0 } else { 5.0 });
        if host_loop {
            let observed = core.execute_pass_internal(None, first.end_ms);
            assert_eq!(observed.end_ms, 5.0);
            assert!(
                observed
                    .output_signals
                    .iter()
                    .any(|signal| signal.completed)
            );
        } else {
            assert!(first.output_signals.iter().any(|signal| signal.completed));
        }
        assert!(core.is_drained());
    }

    #[test]
    fn hicache_selective_observes_reused_pages_before_long_decode_finishes() {
        let mut core = core_with_policy(
            1.0,
            true,
            64,
            crate::engine::SglangHiCacheWritePolicy::WriteThroughSelective,
        );
        let tokens = (0..18).collect::<Vec<_>>();
        let first = core.kv_manager.allocate_for_request(&tokens[..16]).unwrap();
        core.kv_manager.finish(&tokens[..16], first.lease);
        core.receive(direct_request(tokens, 20));
        let launched = core.execute_pass_internal(None, 0.0);
        assert_eq!(core.next_internal_deadline_ms(), None);
        let observed = core.execute_pass_internal(None, launched.end_ms);
        assert_eq!(observed.end_ms, 1.0);
        assert_eq!(core.next_internal_deadline_ms(), Some(5.0));
        assert!(
            !observed
                .output_signals
                .iter()
                .any(|signal| signal.completed)
        );
    }

    #[test]
    fn hicache_cancelling_a_blocked_fifo_head_retries_without_waiting_for_dma() {
        let mut core = core(1.0);
        core.receive(direct_request((0..40).collect(), 1));
        assert_eq!(core.execute_pass_internal(None, 0.0).end_ms, 1.0);
        let deadline = core.next_internal_deadline_ms().unwrap();
        let blocked = core.receive(direct_request((100..120).collect(), 1));
        let fits = core.receive(direct_request((200..204).collect(), 1));
        let idle = core.execute_pass_internal(None, 2.0);
        assert_eq!(idle.end_ms, 2.0);
        assert!(idle.admissions.is_empty());
        assert!(!core.is_ready());
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: blocked,
        })
        .unwrap();
        assert!(core.is_ready());
        let admitted = core.execute_pass_internal(None, 2.0);
        assert_eq!(admitted.admissions[0].uuid, fits);
        assert!(admitted.end_ms < deadline);
    }

    #[test]
    fn hicache_cancelled_chunk_keeps_its_producer_pages_until_observation() {
        let mut core = core_with_policy(
            100.0,
            true,
            24,
            crate::engine::SglangHiCacheWritePolicy::WriteBack,
        );
        let id = core.receive(direct_request((0..52).collect(), 1));
        assert_eq!(core.execute_pass_internal(None, 0.0).end_ms, 0.0);
        assert_eq!(core.execute_pass_internal(None, 0.0).end_ms, 100.0);
        core.apply_command_effects_at(
            SchedulerCommand::CancelRequest { request_id: id },
            true,
            Some(101.0),
        )
        .unwrap();
        core.receive(direct_request((100..120).collect(), 1));
        let observed = core.execute_pass_internal(None, 101.0);
        assert_eq!(observed.end_ms, 200.0);
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 0);
        assert_eq!(core.waiting.len(), 1);
        let launched = core.execute_pass_internal(None, observed.end_ms);
        assert!(launched.end_ms >= 206.0);
        assert!(core.host_cache_metrics().unwrap().resident_blocks > 0);
        let finished = core.execute_pass_internal(None, launched.end_ms);
        assert!(
            finished
                .output_signals
                .iter()
                .any(|signal| signal.completed)
        );
        assert!(core.is_drained());
    }

    #[test]
    fn hicache_overlap_chunk_backup_waits_for_its_producer() {
        let mut core = core_with_loop(100.0, true, 8);
        core.receive(direct_request((0..24).collect(), 1));
        let first = core.execute_pass_internal(None, 0.0);
        assert_eq!(first.end_ms, 0.0);
        assert_eq!(core.next_internal_deadline_ms(), None);
        let second = core.execute_pass_internal(None, first.end_ms);
        assert_eq!(second.end_ms, 100.0);
        assert_eq!(core.next_internal_deadline_ms(), Some(102.0));
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 0);
        core.process_internal_work(102.0);
        assert_eq!(core.host_cache_metrics().unwrap().resident_blocks, 2);
    }

    #[test]
    fn hicache_overlap_queued_forward_does_not_charge_an_elapsed_load_again() {
        let mut core = core_with_loop(100.0, true, 64);
        let prompt = (0..16).collect::<Vec<_>>();
        let cached = core.kv_manager.allocate_for_request(&prompt).unwrap();
        core.kv_manager.finish(&prompt, cached.lease);
        core.complete_engine_boundary(0.0);
        core.process_internal_work(4.0);
        core.kv_manager.evict(16);
        core.receive(direct_request(vec![99; 8], 1));
        let first = core.execute_pass_internal(None, 4.0);
        assert_eq!(first.end_ms, 4.0);
        core.config.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming(1.0)),
        }
        .into();
        core.receive(direct_request(prompt, 1));
        let second = core.execute_pass_internal(None, first.end_ms);
        assert_eq!(second.end_ms, 104.0);
        assert_eq!(core.last_forward_ms(), Some(1.0));
        assert_eq!(
            core.execute_pass_internal(None, second.end_ms).end_ms,
            105.0
        );
    }

    #[test]
    fn hicache_lpm_ranks_host_and_device_prefixes_together() {
        let mut core = core(1.0);
        core.config.schedule_policy = SchedulePolicy::Lpm;
        let mut host_prompt = (0..16).collect::<Vec<_>>();
        core.receive(direct_request(host_prompt.clone(), 1));
        let warm = core.execute_pass_internal(None, 0.0);
        core.process_internal_work(warm.end_ms + 4.0);
        core.kv_manager.evict(16);
        let mut device_prompt = (100..108).collect::<Vec<_>>();
        let cached = core
            .kv_manager
            .allocate_for_request(&device_prompt)
            .unwrap();
        core.kv_manager.finish(&device_prompt, cached.lease);
        host_prompt.extend(16..20);
        device_prompt.extend(108..112);
        core.receive(direct_request(device_prompt, 1));
        core.receive(direct_request(host_prompt.clone(), 1));
        apply_schedule_policy(&mut core.waiting, &core.kv_manager, &core.config);
        assert_eq!(core.waiting[0].sequence_tokens, host_prompt);
    }

    #[test]
    fn hicache_small_host_hit_reselects_the_full_recompute_chunk() {
        let args = test_args(16, 4, 8);
        let config = SglangConfig::from_args(&args);
        let mut manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        manager.enable_hicache(
            SglangHiCacheConfig {
                d2h_bandwidth_gbps: 0.0,
                h2d_bandwidth_gbps: 1.0,
                ..SglangHiCacheConfig::new(16)
            },
            1_000_000,
        );
        let tokens = (0..12).collect::<Vec<_>>();
        let cached = manager.allocate_for_request(&tokens[..8]).unwrap();
        manager.finish(&tokens[..8], cached.lease);
        manager.complete_host_boundary(0.0);
        manager.evict(8);
        let mut waiting = WaitingQueue::from([SglangRequest::new(direct_request(tokens, 1), 4, 0)]);
        let admitted =
            get_new_batch_prefill(&mut waiting, &mut manager, &config, 0.7, &[], false, 0);
        assert_eq!(admitted.can_run[0].materialized_tokens, 8);
        assert_eq!(admitted.prefill_fpm[0].tokens_computed, 8);
        assert_eq!(admitted.prefill_fpm[0].prefix_tokens, 0);
        assert_eq!(admitted.admissions[0].reused_input_tokens, 0);
        assert!(!manager.has_host_work());
    }
}

mod computed_kv_completion {
    use super::*;
    use crate::engine::common::protocols::WorkerType;

    #[rstest::rstest]
    #[case(false, false, false, 28, 112)]
    #[case(true, false, false, 28, 112)]
    #[case(false, true, false, 28, 112)]
    #[case(true, true, false, 28, 128)]
    #[case(false, false, false, 1, 112)]
    #[case(true, false, false, 1, 112)]
    #[case(false, true, false, 1, 112)]
    #[case(true, true, false, 1, 128)]
    #[case(false, false, true, 28, 112)]
    #[case(true, false, true, 28, 112)]
    #[case(false, false, true, 29, 128)]
    #[case(true, false, true, 29, 128)]
    fn completion_reuses_only_computed_pages(
        #[case] overlap: bool,
        #[case] decode_role: bool,
        #[case] speculative: bool,
        #[case] outputs: usize,
        #[case] cached: usize,
    ) {
        let mut args = test_args(64, 16, 1024);
        let sglang = args.sglang.as_mut().unwrap();
        sglang.host_loop = overlap;
        sglang.disaggregation_decode_enable_radix_cache = decode_role;
        if decode_role {
            args.worker_type = WorkerType::Decode;
        }
        if speculative {
            args.aic_nextn = Some(2);
            args.aic_nextn_accept_rates = Some("1,1".into());
        }
        let mut core = SglangCore::new_with_kv_capture(args, 0);
        let prompt_len = if outputs == 1 { 127 } else { 100 };
        let mut sequence: Vec<u32> = (0..prompt_len).collect();
        let planned: Vec<u32> = (1000..1000 + outputs as u32).collect();
        let request = DirectRequest {
            output_token_ids: Some(planned.clone()),
            ..direct_request(sequence.clone(), outputs)
        };
        let uuid = if decode_role {
            // Decode overlap is an internal lifecycle oracle; public configuration rejects it.
            let uuid = Uuid::from_u128(90_300);
            let handoff_id = HandoffId::from(uuid);
            let effects = core
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id,
                        request: DirectRequest {
                            uuid: Some(uuid),
                            ..request
                        },
                    },
                    true,
                )
                .unwrap();
            assert!(!effects.lifecycle_events.is_empty());
            core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap();
            uuid
        } else {
            core.receive(request)
        };
        let mut emitted = Vec::new();
        let mut terminals = 0;
        let mut now_ms = 0.0;
        for _ in 0..64 {
            if core.is_drained() {
                break;
            }
            let pass = core.execute_hidden_pass(now_ms);
            now_ms = pass.end_ms;
            for signal in pass.output_signals {
                assert_eq!(signal.uuid, uuid);
                assert!(!signal.rejected);
                emitted.extend(signal.token_id);
                terminals += usize::from(signal.completed);
            }
        }
        assert!(core.is_drained());
        assert_eq!(emitted, planned);
        assert_eq!(terminals, 1);
        sequence.extend(planned);
        assert_eq!(core.kv_manager.cache().prefix_match_len(&sequence), cached);
        sequence.push(9000);
        core.receive(direct_request(sequence, 1));
        let follower = core.execute_hidden_pass(now_ms);
        assert_eq!(follower.admissions[0].reused_input_tokens, cached);
    }
}

mod pd_admission_v0522 {
    use super::*;
    use crate::engine::common::protocols::WorkerType;

    fn decode_args(capacity: usize, max_running: usize) -> MockEngineArgs {
        let mut args = test_args(capacity, 1, capacity);
        args.worker_type = WorkerType::Decode;
        args.max_num_seqs = Some(max_running);
        args.sglang.as_mut().unwrap().num_reserved_decode_tokens = None;
        args
    }

    fn reserve(core: &mut SglangCore, id: u128, prompt: usize, outputs: usize) -> bool {
        let effects = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: HandoffId::from(Uuid::from_u128(id)),
                    request: DirectRequest {
                        uuid: Some(Uuid::from_u128(id)),
                        ..direct_request(vec![id as u32; prompt], outputs)
                    },
                },
                true,
            )
            .unwrap();
        !effects.lifecycle_events.is_empty()
    }

    fn activate(core: &mut SglangCore, id: u128) {
        core.apply_command(SchedulerCommand::ActivateDestination {
            handoff_id: HandoffId::from(Uuid::from_u128(id)),
        })
        .unwrap();
    }

    #[test]
    fn default_headroom_admits_two_of_four_thousand_token_prompts() {
        let mut core = SglangCore::new(decode_args(4096, 32));
        assert_eq!(
            (1..=4)
                .map(|id| reserve(&mut core, id, 1000, 200))
                .collect::<Vec<_>>(),
            [true, true, false, false]
        );
        assert_eq!(core.kv_manager.cache().available_tokens(), 2096);
        assert_eq!(core.destination_reservation_attempts(), 3);
    }

    #[test]
    fn matched_prefix_is_protected_and_a_failed_candidate_releases_its_pin() {
        let mut args = decode_args(128, 8);
        let sglang = args.sglang.as_mut().unwrap();
        sglang.num_reserved_decode_tokens = Some(16);
        sglang.disaggregation_decode_enable_radix_cache = true;
        let mut core = SglangCore::new(args);
        let tokens = vec![2; 64];
        let cached = core.kv_manager.allocate_for_request(&tokens).unwrap();
        core.kv_manager.finish(&tokens, cached.lease);
        assert!(reserve(&mut core, 1, 48, 1));
        assert_eq!(core.kv_manager.cache().evictable_size, 64);

        assert!(!reserve(&mut core, 2, 64, 60));
        assert_eq!(core.kv_manager.cache().evictable_size, 64);
        let effects = core
            .apply_command_effects(
                SchedulerCommand::CancelDestination {
                    handoff_id: HandoffId::from(Uuid::from_u128(1)),
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            effects.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved { handoff_id, .. }]
                if *handoff_id == HandoffId::from(Uuid::from_u128(2))
        ));
        assert_eq!(core.kv_manager.cache().available_tokens(), 64);
        assert_eq!(core.kv_manager.cache().evictable_size, 0);
    }

    #[test]
    fn preallocation_charges_page_rounding_and_recovers_after_cancellation() {
        let mut args = test_args(12, 8, 96);
        args.worker_type = WorkerType::Decode;
        args.sglang.as_mut().unwrap().num_reserved_decode_tokens = Some(9);
        let mut core = SglangCore::new(args);
        assert!(reserve(&mut core, 1, 17, 1));
        // 72 available - 9 active reserve = 63. The next request needs
        // 56 rounded prompt tokens + 9 reserve, even though 49 + 9 fits.
        assert!(!reserve(&mut core, 2, 49, 1));
        assert_eq!(core.kv_manager.cache().available_tokens(), 72);
        let effects = core
            .apply_command_effects(
                SchedulerCommand::CancelDestination {
                    handoff_id: HandoffId::from(Uuid::from_u128(1)),
                },
                true,
            )
            .unwrap();
        assert_eq!(effects.lifecycle_events.len(), 1);
        assert_eq!(core.kv_manager.cache().available_tokens(), 40);
    }

    #[test]
    fn a_running_requests_completion_guard_can_exceed_its_fixed_reserve() {
        let mut core = SglangCore::new(decode_args(4096, 32));
        assert!(reserve(&mut core, 1, 3000, 1000));
        activate(&mut core, 1);
        let mut now = 0.0;
        for _ in 0..10 {
            now = core.execute_hidden_pass(now).end_ms;
        }
        assert_eq!(core.running[0].output_len(), 10);
        assert_eq!(core.kv_manager.cache().available_tokens(), 1087);
        // A 512-token active reserve would leave 575 tokens for this 562-token
        // request. The completion guard is 990 and leaves only 97.
        assert!(!reserve(&mut core, 2, 50, 10));
        assert!(
            core.destination_pages(HandoffId::from(Uuid::from_u128(2)))
                .is_empty()
        );
    }

    #[rstest::rstest]
    #[case(false, true)]
    #[case(true, false)]
    fn completion_guard_does_not_count_an_unobserved_output(
        #[case] overlap: bool,
        #[case] fits: bool,
    ) {
        let mut args = decode_args(2561, 32);
        args.sglang.as_mut().unwrap().host_loop = overlap;
        let mut core = SglangCore::new(args);
        assert!(reserve(&mut core, 1, 1000, 1000));
        activate(&mut core, 1);
        let mut now = 0.0;
        for _ in 0..10 {
            now = core.execute_hidden_pass(now).end_ms;
        }
        assert_eq!(core.kv_manager.cache().available_tokens(), 1552);
        assert_eq!(reserve(&mut core, 2, 50, 10), fits);
    }

    #[rstest::rstest]
    #[case::metadata(false, 16)]
    #[case::request_rows(true, 24)]
    fn preallocation_obeys_transfer_metadata_and_request_row_limits(
        #[case] complete_transfer: bool,
        #[case] limit: u128,
    ) {
        let mut core = SglangCore::new(decode_args(16384, 8));
        for id in 1..=limit {
            assert!(reserve(&mut core, id, 8, 32), "request {id}");
            if complete_transfer {
                activate(&mut core, id);
            }
        }
        assert!(!reserve(&mut core, limit + 1, 8, 32));
        if complete_transfer {
            let pass = core.execute_hidden_pass(0.0);
            assert_eq!(pass.admissions.len(), 8);
            assert_eq!(core.running.len(), 8);
        }
    }

    #[test]
    fn source_hold_owns_a_request_row_until_transfer_release() {
        let mut args = test_args(1024, 1, 1024);
        args.worker_type = WorkerType::Prefill;
        args.max_num_seqs = Some(1);
        let mut core = SglangCore::new(args);
        for id in 1..=2 {
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id: HandoffId::from(Uuid::from_u128(id)),
                request: DirectRequest {
                    uuid: Some(Uuid::from_u128(id)),
                    ..direct_request(vec![id as u32; 8], 1)
                },
            })
            .unwrap();
        }
        let first = core.execute_hidden_pass(0.0);
        assert_eq!(first.admissions.len(), 1);
        assert!(core.source_is_held(HandoffId::from(Uuid::from_u128(1))));
        assert!(core.waiting_for_external_command());
        let blocked = core.execute_hidden_pass(first.end_ms);
        assert!(blocked.admissions.is_empty());
        core.apply_command(SchedulerCommand::ReleaseSource {
            handoff_id: HandoffId::from(Uuid::from_u128(1)),
        })
        .unwrap();
        let next = core.execute_hidden_pass(blocked.end_ms);
        assert_eq!(next.admissions[0].uuid, Uuid::from_u128(2));
    }

    #[test]
    fn decode_retractions_restore_before_new_transfers_without_reprefill() {
        let mut args = decode_args(256, 8);
        args.sglang.as_mut().unwrap().num_reserved_decode_tokens = Some(16);
        let mut core = SglangCore::new(args);
        for id in 1..=2 {
            assert!(reserve(&mut core, id, 8, 200));
            activate(&mut core, id);
        }
        let mut now = 0.0;
        let mut saw_retraction = false;
        let mut saw_readmission = false;
        let mut outputs = 0;
        for _ in 0..450 {
            let pass = core.execute_hidden_pass(now);
            now = pass.end_ms;
            outputs += pass
                .output_signals
                .iter()
                .filter(|signal| signal.token_id.is_some())
                .count();
            assert_eq!(pass.fpm.as_ref().unwrap().num_prefill_requests, 0);
            if saw_retraction && !pass.admissions.is_empty() {
                saw_readmission = true;
            }
            if !saw_retraction && !pass.pressure_events.is_empty() {
                saw_retraction = true;
                assert!(!reserve(&mut core, 3, 8, 1));
                assert!(core.waiting.iter().any(|request| request.retracted));
            }
            core.retry_pending_destinations();
            if core.running.is_empty() && core.waiting.is_empty() {
                break;
            }
        }
        assert!(saw_retraction && saw_readmission);
        assert_eq!(outputs, 400);
        assert_eq!(
            core.destination_pages(HandoffId::from(Uuid::from_u128(3)))
                .len(),
            8
        );
    }

    #[test]
    fn partially_restored_decode_requests_make_progress() {
        let mut args = decode_args(256, 8);
        args.sglang.as_mut().unwrap().num_reserved_decode_tokens = Some(16);
        let mut core = SglangCore::new(args);
        for id in 1..=4 {
            assert!(reserve(&mut core, id, 8, 176));
            activate(&mut core, id);
        }

        let mut now = 0.0;
        let mut outputs = [0; 4];
        let mut terminals = [0; 4];
        let mut saw_partial_restoration = false;
        for _ in 0..1000 {
            let pass = core.execute_hidden_pass(now);
            now = pass.end_ms;
            saw_partial_restoration |=
                !pass.admissions.is_empty() && core.waiting.iter().any(|request| request.retracted);
            for signal in pass.output_signals {
                let index = signal.uuid.as_u128() as usize - 1;
                outputs[index] += usize::from(signal.token_id.is_some());
                terminals[index] += usize::from(signal.completed);
            }
            assert_eq!(pass.fpm.as_ref().unwrap().num_prefill_requests, 0);
            if core.is_drained() {
                break;
            }
        }

        assert!(
            core.is_drained(),
            "partially restored rows must release their KV"
        );
        assert!(saw_partial_restoration);
        assert_eq!(outputs, [176; 4]);
        assert_eq!(terminals, [1; 4]);
        assert_eq!(core.kv_manager.cache().available_tokens(), 256);
    }

    #[rstest::rstest]
    fn decode_headroom_boundaries_drain_through_destination_commands(
        #[values(8, 192, 224)] long_prompt: usize,
        #[values(8, 32, 192)] peer_prompt: usize,
        #[values(16, 32)] reserve_tokens: usize,
        #[values(0, 32)] peer_delay: usize,
        #[values(1, 8)] page_size: usize,
        #[values(false, true)] overlap: bool,
        #[values(false, true)] shared_prefix: bool,
    ) {
        let mut args = test_args(256 / page_size, page_size, 256);
        args.worker_type = WorkerType::Decode;
        args.max_num_seqs = Some(2);
        let sglang = args.sglang.as_mut().unwrap();
        sglang.num_reserved_decode_tokens = Some(reserve_tokens);
        sglang.host_loop = overlap;
        sglang.disaggregation_decode_enable_radix_cache = shared_prefix;
        let mut core = SglangCore::new(args);
        let mut submitted = [false; 2];
        let mut activated = [false; 2];
        let mut outputs = [0; 2];
        let mut terminals = [0; 2];
        let prompts = [long_prompt, peer_prompt];
        let mut now = 0.0;
        for step in 0..512 {
            for index in 0..2 {
                let id = index as u128 + 1;
                let handoff_id = HandoffId::from(Uuid::from_u128(id));
                if !submitted[index] && step >= if index == 0 { 0 } else { peer_delay } {
                    core.apply_command_effects(
                        SchedulerCommand::ReserveDestination {
                            handoff_id,
                            request: DirectRequest {
                                uuid: Some(Uuid::from_u128(id)),
                                ..direct_request(
                                    vec![if shared_prefix { 1 } else { id as u32 }; prompts[index]],
                                    128,
                                )
                            },
                        },
                        true,
                    )
                    .unwrap();
                    submitted[index] = true;
                }
                if submitted[index]
                    && !activated[index]
                    && !core.destination_pages(handoff_id).is_empty()
                {
                    activate(&mut core, id);
                    activated[index] = true;
                }
            }
            let pass = core.execute_hidden_pass(now);
            now = pass.end_ms;
            for signal in pass.output_signals {
                let index = signal.uuid.as_u128() as usize - 1;
                outputs[index] += usize::from(signal.token_id.is_some());
                terminals[index] += usize::from(signal.completed);
            }
            core.retry_pending_destinations();
            if submitted == [true; 2] && core.is_drained() {
                break;
            }
        }
        assert!(
            core.is_drained(),
            "destination/restoration queues must drain"
        );
        assert_eq!(terminals, [1; 2]);
        assert_eq!(
            outputs,
            prompts.map(|prompt| 128.min(256 - prompt - page_size - 1))
        );
    }
}

mod admission_v0522 {
    use super::*;

    #[test]
    fn arrival_clips_generation_to_the_empty_pool_budget() {
        let mut core = SglangCore::new(test_args(63, 16, 8192));
        core.receive(direct_request((0..900).collect(), 100));
        assert_eq!(core.waiting[0].max_output_tokens, 79);
        let mut outputs = 0;
        let mut now = 0.0;
        for _ in 0..90 {
            let pass = core.execute_hidden_pass(now);
            assert!(pass.output_signals.iter().all(|signal| !signal.rejected));
            outputs += pass
                .output_signals
                .iter()
                .filter(|signal| signal.token_id.is_some())
                .count();
            now = pass.end_ms;
            if core.is_drained() {
                break;
            }
        }
        assert_eq!(outputs, 79);
        assert!(core.is_drained());
    }

    #[test]
    fn a_candidate_cannot_spend_its_own_cached_prefix() {
        let config = SglangConfig::from_args(&test_args(128, 16, 8192));
        let mut manager = SglangKvManager::new(2048, 16, KvEventPublishers::default(), 0);
        let prefix = (10_000..10_512).collect::<Vec<_>>();
        let cached = manager.allocate_for_request(&prefix).unwrap();
        manager.finish(&prefix, cached.lease);
        let mut shared = prefix.clone();
        shared.extend(20_000..20_512);
        let mut waiting = WaitingQueue::from([
            SglangRequest::new(direct_request((0..1000).collect(), 16), 16, 0),
            SglangRequest::new(direct_request(shared, 16), 16, 0),
        ]);
        let result = get_new_batch_prefill(&mut waiting, &mut manager, &config, 0.7, &[], false, 0);
        assert_eq!(result.can_run.len(), 1);
        assert_eq!(waiting.len(), 1);
        assert!(result.batch_is_full);
        assert_eq!(
            manager.cache().evictable_size,
            512,
            "rejected match releases its temporary pin"
        );
        assert_eq!(
            manager
                .cache()
                .prefix_match_hashes_len(&compute_block_hash_for_seq(&prefix, 16)),
            512
        );
    }

    #[test]
    fn a_small_first_chunk_does_not_bypass_the_whole_prompt_gate() {
        let config = SglangConfig::from_args(&test_args(16, 4, 8));
        let mut manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let held = manager.allocate_for_request(&vec![99; 40]).unwrap();
        let mut waiting = WaitingQueue::from([SglangRequest::new(
            direct_request((0..32).collect(), 4),
            4,
            0,
        )]);
        let result = get_new_batch_prefill(&mut waiting, &mut manager, &config, 0.7, &[], false, 0);
        assert!(result.can_run.is_empty());
        assert_eq!(manager.cache().available_tokens(), 24);
        assert!(!waiting[0].kv_lease.is_active());
        manager.abort(held.lease);
    }

    #[rstest::rstest]
    #[case(0.9, 6)]
    #[case(0.975, 10)]
    fn continuation_makes_progress_even_when_the_batch_is_full(
        #[case] ratio: f64,
        #[case] expected_materialized: usize,
    ) {
        let mut args = test_args(16, 4, 6);
        args.max_num_seqs = Some(1);
        let config = SglangConfig::from_args(&args);
        let mut manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let mut waiting = WaitingQueue::from([SglangRequest::new(
            direct_request((0..12).collect(), 1),
            4,
            0,
        )]);
        let mut first =
            get_new_batch_prefill(&mut waiting, &mut manager, &config, 1.0, &[], false, 0);
        let continuation = first.can_run.pop().unwrap();
        assert_eq!(continuation.materialized_tokens, 4);
        waiting.push_back(continuation);

        let mut running = SglangRequest::new(direct_request(vec![99; 4], 54), 4, 0);
        running.sequence_tokens.extend(vec![100; 34]);
        manager
            .allocate_for_request_lease_capped(&running.sequence_tokens, &mut running.kv_lease, 0)
            .unwrap();
        running.materialized_tokens = 38;
        running.allocated_tokens = 40;
        running.debug_assert_invariants(4);
        let second = get_new_batch_prefill(
            &mut waiting,
            &mut manager,
            &config,
            ratio,
            &[running],
            true,
            0,
        );
        assert_eq!(second.can_run.len(), 1);
        assert_eq!(second.can_run[0].materialized_tokens, expected_materialized);
        assert_eq!(
            second.prefill_fpm[0].tokens_computed,
            expected_materialized - 4
        );
        assert!(second.admissions.is_empty());
    }

    /// LPM orders the continuation first itself; under FIFO it stays queued
    /// behind a fresh request with a longer cache hit, and admission finds it.
    #[rstest::rstest]
    fn admission_runs_the_owned_continuation_first(#[values(false, true)] lpm: bool) {
        let mut args = test_args(64, 4, 8);
        if lpm {
            args.sglang.as_mut().unwrap().schedule_policy = Some("lpm".into());
        }
        let config = SglangConfig::from_args(&args);
        let mut manager = SglangKvManager::new(256, 4, KvEventPublishers::default(), 0);
        let hot = vec![99; 32];
        let cached = manager.allocate_for_request(&hot).unwrap();
        manager.finish(&hot, cached.lease);
        let mut waiting = WaitingQueue::from([SglangRequest::new(
            direct_request((0..24).collect(), 1),
            4,
            0,
        )]);
        let mut first =
            get_new_batch_prefill(&mut waiting, &mut manager, &config, 0.7, &[], false, 0);
        let continuation = first.can_run.pop().unwrap();
        let id = continuation.uuid;
        waiting.push_back(SglangRequest::new(direct_request(hot, 1), 4, 0));
        waiting.push_back(continuation);
        apply_schedule_policy(&mut waiting, &manager, &config);
        assert_eq!(waiting.front().unwrap().uuid == id, lpm);
        let next = get_new_batch_prefill(&mut waiting, &mut manager, &config, 0.7, &[], false, 0);
        assert_eq!(next.can_run[0].uuid, id);
        assert!(next.admissions.is_empty());
        assert!(!waiting.has_continuation());
    }

    #[test]
    fn ratio_stays_put_while_blocked_and_resets_only_after_queues_drain() {
        let mut core = SglangCore::new(test_args(16, 4, 16));
        let held = core.kv_manager.allocate_for_request(&vec![99; 56]).unwrap();
        let id = core.receive(direct_request((0..8).collect(), 4));
        core.new_token_ratio = 0.3;
        for _ in 0..3 {
            let pass = core.execute_hidden_pass(0.0);
            assert!(pass.admissions.is_empty());
            assert_eq!(core.new_token_ratio, 0.3);
            assert_eq!(
                pass.same_timestamp_retry,
                crate::engine::generalized::SameTimestampRetry::Exhausted
            );
        }
        core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
            .unwrap();
        core.kv_manager.abort(held.lease);
        core.execute_hidden_pass(0.0);
        assert_eq!(core.new_token_ratio, core.config.init_new_token_ratio);
    }

    #[test]
    fn arrival_on_an_idle_rank_preserves_the_ratio_until_the_group_starts_a_new_wave() {
        let mut core = SglangCore::new(test_args(16, 4, 16));
        core.new_token_ratio = 0.3;
        core.prepare_group_pass(5);
        core.finish_group_pass(false, true);
        assert!(core.is_drained());
        core.receive(direct_request((0..8).collect(), 4));
        core.prepare_group_pass(6);
        let pass = core.execute_hidden_pass(0.0);
        core.finish_group_pass(true, true);
        assert_eq!(pass.admissions.len(), 1);
        assert_eq!(core.new_token_ratio, 0.3);

        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: pass.admissions[0].uuid,
        })
        .unwrap();
        assert!(core.is_drained());
        core.receive(direct_request((20..28).collect(), 4));
        core.prepare_group_pass(0);
        assert_eq!(core.new_token_ratio, core.config.init_new_token_ratio);
    }

    #[test]
    fn finishing_inflight_memory_skips_decode_without_retraction_or_ratio_decay() {
        let mut args = test_args(16, 1, 16);
        args.sglang.as_mut().unwrap().host_loop = true;
        let mut core = SglangCore::new(args);
        let done = core.receive(direct_request(vec![1, 2], 1));
        let survivor = core.receive(direct_request(vec![3, 4], 3));
        let first = core.execute_hidden_pass(0.0);
        assert_eq!(core.running.len(), 2);
        let held = core.kv_manager.allocate_for_request(&vec![99; 12]).unwrap();
        core.new_token_ratio = 0.4;
        let skipped = core.execute_hidden_pass(first.end_ms);
        assert!(skipped.pressure_events.is_empty());
        assert_eq!(skipped.fpm.unwrap().num_decode_requests, 0);
        assert!(
            skipped
                .output_signals
                .iter()
                .any(|signal| signal.uuid == done && signal.completed)
        );
        assert_eq!(core.running.len(), 1);
        assert_eq!(core.running[0].uuid, survivor);
        assert_eq!(core.running[0].output_len(), 1);
        assert_eq!(core.new_token_ratio, 0.4);
        let resumed = core.execute_hidden_pass(skipped.end_ms);
        assert_eq!(resumed.fpm.unwrap().num_decode_requests, 1);
        assert!(resumed.pressure_events.is_empty());
        assert_eq!(
            core.new_token_ratio,
            0.4 - core.config.new_token_ratio_decay_step
        );
        core.kv_manager.abort(held.lease);
    }

    #[test]
    fn initial_ratio_is_capped_before_minimum_and_decay_are_derived() {
        let mut args = test_args(64, 4, 16);
        args.sglang.as_mut().unwrap().schedule_conservativeness = Some(2.0);
        let config = SglangConfig::from_args(&args);
        assert_eq!(config.init_new_token_ratio, 1.0);
        assert_eq!(config.min_new_token_ratio, 0.14);
        assert_eq!(config.new_token_ratio_decay_step, 0.86 / 600.0);
    }
}

#[test]
fn submitted_requests_keep_their_image_placeholders() {
    let mut core = SglangCore::new(test_args(16, 1, 8));
    let images = vec![
        ImageSpec {
            identity: 7,
            token_start: 0,
            token_end: 3,
            encoder: EncoderShape {
                sequences: 1,
                patch_tokens: 12,
                transformer_tokens: 12,
                output_tokens: 3,
            },
            embedding_bytes: 128,
        },
        ImageSpec {
            identity: 8,
            token_start: 3,
            token_end: 6,
            encoder: EncoderShape {
                sequences: 1,
                patch_tokens: 12,
                transformer_tokens: 12,
                output_tokens: 3,
            },
            embedding_bytes: 128,
        },
    ];
    let mut request = direct_request(vec![1; 8], 1);
    request.images = images.clone();
    core.apply_command(SchedulerCommand::Submit(request))
        .unwrap();
    assert_eq!(core.waiting.len(), 1);
    assert_eq!(core.waiting[0].images, images);
    assert_eq!(images[1].visual_tokens(), 3);
    assert!(images[1].overlaps(5, 9));
    assert!(!images[1].overlaps(6, 9));
}

#[rstest::rstest]
fn rejects_image_placeholders_outside_the_prompt(#[values((6, 3), (6, 9))] span: (usize, usize)) {
    let mut core = SglangCore::new(test_args(16, 1, 8));
    let mut request = direct_request(vec![1; 8], 1);
    request.images = vec![ImageSpec {
        identity: 7,
        token_start: span.0,
        token_end: span.1,
        encoder: EncoderShape {
            sequences: 1,
            patch_tokens: 12,
            transformer_tokens: 12,
            output_tokens: 3,
        },
        embedding_bytes: 128,
    }];
    assert!(
        core.apply_command(SchedulerCommand::Submit(request))
            .unwrap_err()
            .to_string()
            .contains("image placeholders")
    );
    assert!(core.waiting.is_empty());
}

#[rstest::rstest]
fn rejects_prompt_at_or_above_max_model_len(
    #[values(8, 9)] prompt_len: u32,
    #[values(false, true)] handoff: bool,
) {
    let mut args = test_args(16, 4, 4);
    args.max_model_len = Some(8);
    let mut core = SglangCore::new(args);
    let mut req = direct_request((0..prompt_len).collect(), 1);
    let request_id = Uuid::from_u128(899);
    req.uuid = Some(request_id);
    let hashes = crate::engine::belady::input_sequence_hashes(&req.tokens, 4);
    let oracle = BeladyOracle::new(vec![(request_id, hashes.clone())]).unwrap();
    core.set_belady_oracle(oracle.clone());
    let handoff_id = HandoffId::from(Uuid::from_u128(900));
    let command = if handoff {
        SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: req,
        }
    } else {
        SchedulerCommand::Submit(req)
    };
    core.apply_command(command).unwrap();
    let pass = core.execute_hidden_pass(0.0);
    assert!(matches!(
        pass.output_signals.as_slice(),
        [OutputSignal {
            token_id: None,
            completed: true,
            rejected: true,
            ..
        }]
    ));
    assert!(pass.admissions.is_empty());
    for hash in hashes {
        assert_eq!(oracle.next_use(hash), usize::MAX);
    }
    assert_eq!(pass.end_ms, 0.0);
    assert!(core.waiting.is_empty());
    assert!(core.running.is_empty());
    assert_eq!(core.kv_manager.cache().available_tokens(), 64);
    if handoff {
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelSource { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Noop
        );
    }
    // A rejected request must not block later valid work.
    core.receive(direct_request(vec![1], 1));
    let mut signals = Vec::new();
    for step in 0..4 {
        signals.extend(
            core.execute_hidden_pass(step as f64 * 1000.0)
                .output_signals,
        );
    }
    assert_eq!(signals.len(), 1);
    assert!(signals[0].completed && !signals[0].rejected);
}

#[rstest::rstest]
fn destination_rejects_prompt_at_or_above_max_model_len(#[values(8, 9)] prompt_len: u32) {
    let mut args = test_args(16, 4, 4);
    args.max_model_len = Some(8);
    let mut core = SglangCore::new(args);
    let error = core
        .apply_command(SchedulerCommand::ReserveDestination {
            handoff_id: HandoffId::from(Uuid::from_u128(901)),
            request: direct_request((0..prompt_len).collect(), 1),
        })
        .unwrap_err();
    assert!(error.to_string().contains("max_model_len"));
    assert_eq!(core.kv_manager.cache().available_tokens(), 64);
}

#[test]
fn dsv4_swa_reuse_ends_at_a_cached_window_and_learns_the_fork() {
    // DeepSeek-V4 on SGLang keeps a cached sequence's SWA window only at its tail. A request
    // forking inside that sequence recomputes from scratch; the recompute leaves a window at
    // the fork point, so the next request forking there reuses it.
    let reused = |swa_prefix_tails| {
        let mut args = test_args(64, 4, 64);
        args.sglang.as_mut().unwrap().swa_prefix_tails = swa_prefix_tails;
        let mut core = SglangCore::new(args);
        let mut collector = crate::engine::trace::TraceCollector::default();
        let mut now = 0.0;
        let mut fork = |tail: std::ops::Range<u32>, prefix: u32| {
            let mut tokens = (0..prefix).collect::<Vec<_>>();
            tokens.extend(tail);
            core.receive(direct_request(tokens, 0));
            let pass = core.execute_pass(&mut collector, now);
            now = pass.end_ms;
            assert!(core.is_empty());
            pass.admissions[0].reused_input_tokens
        };
        [
            fork(0..0, 16),
            fork(100..104, 16),
            fork(200..208, 8),
            fork(300..308, 8),
        ]
    };
    assert_eq!(reused(Some(8)), [0, 16, 0, 8]);
    assert_eq!(
        reused(None),
        [0, 16, 8, 8],
        "full attention reuses any cached page"
    );
}

#[test]
fn dsv4_swa_continuation_after_generated_output_reuses_the_prompt() {
    // Replay's generated tokens are not the next prompt's, so the next turn diverges where
    // the previous output began; the window replay keeps at the prompt end lets it reuse
    // the prompt, as SGLang reuses it through the real output.
    let mut args = test_args(64, 4, 64);
    args.sglang.as_mut().unwrap().swa_prefix_tails = Some(8);
    let mut core = SglangCore::new(args);
    let mut collector = crate::engine::trace::TraceCollector::default();
    core.receive(direct_request((0..16).collect(), 6));
    let mut now = 0.0;
    while !core.is_empty() {
        now = core.execute_pass(&mut collector, now).end_ms;
    }
    core.receive(direct_request((0..16).chain(500..508).collect(), 0));
    let pass = core.execute_pass(&mut collector, now);
    assert_eq!(pass.admissions[0].reused_input_tokens, 16);
}

#[test]
fn belady_retires_whole_prompt_once_on_first_chunk_and_includes_full_hits() {
    let tokens = (0..8).collect::<Vec<_>>();
    let hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&tokens, 4));
    let first = Uuid::from_u128(1);
    let next = Uuid::from_u128(2);
    let oracle = BeladyOracle::new(vec![(first, hashes.clone()), (next, hashes.clone())]).unwrap();
    let mut core = SglangCore::new(test_args(32, 4, 4));
    core.set_belady_oracle(oracle.clone());
    let mut request = direct_request(tokens.clone(), 0);
    request.uuid = Some(first);
    core.receive(request);
    let mut collector = crate::engine::trace::TraceCollector::default();

    let pass1 = core.execute_pass(&mut collector, 0.0);
    assert_eq!(pass1.fpm.as_ref().unwrap().sum_prefill_tokens, 4);
    assert_eq!(
        oracle.next_use(hashes[1]),
        1,
        "even the uncomputed chunk is retired"
    );
    let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
    assert_eq!(
        oracle.next_use(hashes[0]),
        1,
        "continuation does not retire another request"
    );
    assert!(core.is_empty());

    let mut request = direct_request(tokens, 0);
    request.uuid = Some(next);
    core.receive(request);
    let pass3 = core.execute_pass(&mut collector, pass2.end_ms);
    assert_eq!(pass3.admissions[0].reused_input_tokens, 8);
    assert_eq!(pass3.fpm.as_ref().unwrap().sum_prefill_tokens, 0);
    assert_eq!(oracle.next_use(hashes[0]), usize::MAX);
    assert!(core.is_empty());
}

#[test]
fn belady_keeps_failed_admission_outstanding_until_terminal_cancellation() {
    let tokens = (0..8).collect::<Vec<_>>();
    let hashes = compute_seq_hash_for_block(&compute_block_hash_for_seq(&tokens, 4));
    let uuid = Uuid::from_u128(1);
    let oracle = BeladyOracle::new(vec![(uuid, hashes.clone())]).unwrap();
    let mut core = SglangCore::new(test_args(16, 4, 8));
    let held = core.kv_manager.allocate_for_request(&vec![99; 56]).unwrap();
    core.set_belady_oracle(oracle.clone());
    let mut request = direct_request(tokens, 0);
    request.uuid = Some(uuid);
    core.receive(request);
    let mut collector = crate::engine::trace::TraceCollector::default();

    let pass = core.execute_pass(&mut collector, 0.0);
    assert!(pass.admissions.is_empty());
    assert_eq!(oracle.next_use(hashes[0]), 0);
    assert_eq!(core.waiting.len(), 1);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
        .unwrap();
    assert_eq!(oracle.next_use(hashes[0]), usize::MAX);
    assert!(core.is_empty());
    core.kv_manager.abort(held.lease);
}

#[test]
fn request_storage_reservation_is_bounded_for_submit_and_destination() {
    const BLOCK_SIZE: usize = 4;
    const MAX_OUTPUT_TOKENS: usize = 1_000_000;
    const CLIP_MAX_NEW_TOKENS: usize = 7;
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(8)
        .block_size(BLOCK_SIZE)
        .speedup_ratio(0.0)
        .sglang(Some(SglangArgs {
            page_size: Some(BLOCK_SIZE),
            chunked_prefill_size: Some(16),
            clip_max_new_tokens: Some(CLIP_MAX_NEW_TOKENS),
            num_reserved_decode_tokens: Some(0),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let mut core = SglangCore::new(args);
    let prompt = (0..8).collect::<Vec<_>>();
    let bounded_tokens = prompt.len() + CLIP_MAX_NEW_TOKENS;
    let bounded_pages = bounded_tokens / BLOCK_SIZE;

    let submitted = Uuid::from_u128(80_001);
    core.receive(DirectRequest {
        tokens: prompt.clone(),
        max_output_tokens: MAX_OUTPUT_TOKENS,
        uuid: Some(submitted),
        ..Default::default()
    });
    let (token_capacity, hash_capacity) = core.request_storage_capacities(submitted).unwrap();
    assert!(token_capacity >= bounded_tokens);
    assert!(token_capacity < prompt.len() + MAX_OUTPUT_TOKENS);
    assert!(hash_capacity >= bounded_pages);
    assert!(hash_capacity < (prompt.len() + MAX_OUTPUT_TOKENS) / BLOCK_SIZE);

    let destination = Uuid::from_u128(80_002);
    core.apply_command_effects(
        SchedulerCommand::ReserveDestination {
            handoff_id: HandoffId::from(Uuid::from_u128(80_003)),
            request: DirectRequest {
                tokens: prompt.clone(),
                max_output_tokens: MAX_OUTPUT_TOKENS,
                uuid: Some(destination),
                ..Default::default()
            },
        },
        false,
    )
    .unwrap();
    let (token_capacity, hash_capacity) = core.request_storage_capacities(destination).unwrap();
    assert!(token_capacity >= bounded_tokens);
    assert!(token_capacity < prompt.len() + MAX_OUTPUT_TOKENS);
    assert!(hash_capacity >= bounded_pages);
    assert!(hash_capacity < (prompt.len() + MAX_OUTPUT_TOKENS) / BLOCK_SIZE);

    let planned = Uuid::from_u128(80_004);
    let planned_output = (100..109).collect::<Vec<_>>();
    core.receive(DirectRequest {
        tokens: prompt.clone(),
        max_output_tokens: MAX_OUTPUT_TOKENS,
        output_token_ids: Some(planned_output.clone()),
        uuid: Some(planned),
        ..Default::default()
    });
    let (token_capacity, hash_capacity) = core.request_storage_capacities(planned).unwrap();
    let realizable_planned = 9;
    assert!(token_capacity >= prompt.len() + realizable_planned);
    assert!(hash_capacity >= (prompt.len() + realizable_planned) / BLOCK_SIZE);
    assert!(
        planned_output.len() > CLIP_MAX_NEW_TOKENS,
        "planned-output coverage must exceed the online storage clip"
    );
}

#[test]
fn zero_output_request_completes_after_prefill() {
    let mut core = SglangCore::new(test_args(32, 4, 16));
    let uuid = core.receive(direct_request(vec![1, 2, 3, 4], 0));

    let pass = core.execute_pass_internal(None, 0.0);

    assert!(core.is_empty());
    assert_eq!(pass.completed_requests, 1);
    assert_eq!(pass.committed_requests, vec![uuid]);
    let fpm = pass.fpm.as_ref().unwrap();
    assert_eq!(fpm.num_decode_requests, 0);
    assert_eq!(fpm.sum_decode_kv_tokens, 0);
    assert!(matches!(
        pass.output_signals.as_slice(),
        [OutputSignal {
            uuid: signal_uuid,
            token_id: None,
            completed: true,
            rejected: false,
            ..
        }] if *signal_uuid == uuid
    ));
}

#[test]
fn speculative_decode_commits_each_request_once_in_uuid_order() {
    let mut args = test_args(32, 4, 16);
    args.aic_nextn = Some(2);
    args.aic_nextn_accept_rates = Some("1,1".to_string());
    let mut core = SglangCore::new(args);
    let [a, b, c] = [1, 2, 3].map(Uuid::from_u128);
    // Running order is the reverse of UUID order.
    for (uuid, first_token) in [(c, 0), (b, 4), (a, 8)] {
        core.receive(DirectRequest {
            uuid: Some(uuid),
            ..direct_request((first_token..first_token + 4).collect(), 10)
        });
    }
    let prefill = core.execute_pass_internal(None, 0.0);
    assert_eq!(prefill.committed_requests.len(), 3);

    // Both drafts are always accepted, so each request emits three consecutive tokens.
    let decode = core.execute_pass_internal(None, prefill.end_ms);
    let emitted = decode
        .output_signals
        .iter()
        .filter(|signal| signal.token_id.is_some())
        .map(|signal| signal.uuid)
        .collect::<Vec<_>>();
    assert_eq!(emitted, [c, c, c, b, b, b, a, a, a]);
    assert_eq!(decode.committed_requests, [a, b, c]);
}

#[test]
fn interval_idle_ratio_distinguishes_a_busy_peer_from_global_idle() {
    let mut args = test_args(256, 1, 16);
    args.prefill_decode_interval = 2;
    let mut core = SglangCore::new(args);
    let ratio_after_prior_decodes = 0.4;
    core.new_token_ratio = ratio_after_prior_decodes;

    // A peer's EXTEND arms this idle rank too.
    core.prepare_group_pass(1);
    core.finish_group_pass(true, true);
    core.prepare_group_pass(2);
    let pass = core.execute_hidden_pass(0.0);
    assert_eq!(pass.end_ms, 0.0);
    assert!(!core.model_work_in_pass());
    core.finish_group_pass(false, true);
    assert_eq!(core.new_token_ratio, ratio_after_prior_decodes);

    // On the next round the whole group is idle: upstream on_idle resets the
    // ratio. This round must neither keep decaying it nor retain the old value.
    core.prepare_group_pass(3);
    core.execute_hidden_pass(0.0);
    core.finish_group_pass(false, false);
    assert_eq!(core.new_token_ratio, core.config.init_new_token_ratio);
    assert!(core.is_drained());
}

#[test]
fn impossible_prompt_is_rejected_without_same_timestamp_retries() {
    let mut core = SglangCore::new(test_args(1, 4, 8));
    core.receive(direct_request(vec![1; 8], 2));
    let now_ms = 37.0;
    let pass = core.execute_pass_internal(None, now_ms);
    assert_eq!(pass.end_ms, now_ms);
    assert!(pass.admissions.is_empty());
    assert!(matches!(
        pass.output_signals.as_slice(),
        [OutputSignal {
            rejected: true,
            completed: true,
            token_id: None,
            ..
        }]
    ));
    assert_eq!(
        pass.same_timestamp_retry,
        crate::engine::generalized::SameTimestampRetry::Exhausted
    );
    assert!(core.is_drained());
}

#[test]
fn zero_output_completion_survives_decode_reservation_failure() {
    let config = SglangConfig::from_args(&test_args(2, 4, 16));
    let mut kv_manager = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
    let zero_alloc = kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
    let normal_alloc = kv_manager.allocate_for_request(&[5, 6, 7, 8]).unwrap();
    let zero_uuid = Uuid::from_u128(90_004);
    let normal_uuid = Uuid::from_u128(90_005);
    let mut running = vec![
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: zero_uuid,
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 0,
            planned_output_ids: None,
            kv_lease: zero_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        },
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: normal_uuid,
            sequence_tokens: vec![5, 6, 7, 8, 9],
            prompt_len: 4,
            max_output_tokens: 2,
            planned_output_ids: None,
            kv_lease: normal_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        },
    ];

    // The normal request cannot reserve a decode token while both prompts own
    // the entire pool. Its retry must not discard the zero-output completion.
    let result = decode::simulate_decode_step_with_sampler(
        &mut running,
        &mut kv_manager,
        &config,
        None,
        0.0,
    )
    .unwrap();

    assert_eq!(running.len(), 1);
    assert_eq!(running[0].uuid, normal_uuid);
    assert_eq!(result.completed_requests.len(), 1);
    assert!(matches!(
        result.output_signals.as_slice(),
        [OutputSignal {
            uuid: signal_uuid,
            token_id: None,
            completed: true,
            rejected: false,
            ..
        }] if *signal_uuid == zero_uuid
    ));
}

#[test]
fn retraction_ratio_is_estimated_from_survivors_before_the_forward() {
    // Two full pages, both requests have a sampled token needing KV: r2 (less output) is
    // retracted, r1 takes its page, produces its last token and completes in the same step.
    // SGLang's `retract_decode` estimate is taken from the survivors before the forward
    // (decoded 2 + 20 over max_new 3 + 1, clamped to 1.0); computing it after the forward would
    // see an empty batch and reset the ratio to 0.
    let config = SglangConfig::from_args(&test_args(2, 4, 16));
    let mut kv_manager = SglangKvManager::new(8, 4, KvEventPublishers::default(), 0);
    let r1_alloc = kv_manager.allocate_for_request(&[1, 2, 3, 10]).unwrap();
    let r2_alloc = kv_manager.allocate_for_request(&[5, 6, 7, 8]).unwrap();
    let mut running = vec![
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::from_u128(90_010),
            sequence_tokens: vec![1, 2, 3, 10, 11],
            prompt_len: 3,
            max_output_tokens: 3,
            planned_output_ids: None,
            kv_lease: r1_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        },
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::from_u128(90_011),
            sequence_tokens: vec![5, 6, 7, 8, 9],
            prompt_len: 4,
            max_output_tokens: 5,
            planned_output_ids: None,
            kv_lease: r2_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        },
    ];
    let result = simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
    assert_eq!(result.requests.len(), 1, "r2 is retracted");
    assert_eq!(result.requests[0].uuid, Uuid::from_u128(90_011));
    assert!(running.is_empty(), "r1 completes in this step");
    assert_eq!(result.new_token_ratio_estimate, Some(1.0));
}

#[test]
fn committed_decode_membership_excludes_retracted_requests() {
    let mut core = SglangCore::new(test_args(2, 4, 16));
    let completed = Uuid::from_u128(90_012);
    let retracted = Uuid::from_u128(90_013);
    // Both requests fill a page and have a sampled token awaiting KV. The request with
    // less output is retracted; its page lets the surviving request finish.
    for (uuid, tokens, prompt_len, max_output_tokens) in [
        (completed, vec![1, 2, 3, 10, 11], 3, 3),
        (retracted, vec![5, 6, 7, 8, 9], 4, 5),
    ] {
        core.receive(DirectRequest {
            uuid: Some(uuid),
            ..direct_request(tokens[..prompt_len].to_vec(), max_output_tokens)
        });
        core.waiting.pop_front().unwrap();
        let allocation = core.kv_manager.allocate_for_request(&tokens[..4]).unwrap();
        core.running.push(SglangRequest {
            images: Vec::new(),
            pending_terminal: false,
            is_decode_handoff: false,
            retracted: false,
            uuid,
            sequence_tokens: tokens,
            prompt_len,
            max_output_tokens,
            planned_output_ids: None,
            kv_lease: allocation.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
        });
    }
    let pass = core.execute_hidden_pass(0.0);
    assert_eq!(pass.committed_requests, vec![completed]);
    assert!(
        pass.output_signals
            .iter()
            .any(|signal| signal.uuid == completed && signal.completed)
    );
    assert!(
        !pass
            .output_signals
            .iter()
            .any(|signal| signal.uuid == retracted)
    );
    assert_eq!(core.waiting.front().unwrap().uuid, retracted);
    assert!(core.running.is_empty());
}

#[test]
fn fresh_prefill_tracks_cache_owned_prefix_pages_and_pressure_event() {
    let args = test_args(8, 4, 16);
    let config = SglangConfig::from_args(&args);
    let (buffer, sink) = capture_kv_event_sink();
    // Four pages. The cached prompt spans two; SGLang's `input_len - 1` match cap means the
    // admitted request reuses only the first cached page and owns a fresh page for the rest.
    let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink)), 0);
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let cached = kv_manager.allocate_for_request(&prompt).unwrap();
    let cached_pages = cached.lease.pages().to_vec();
    kv_manager.finish(&prompt, cached.lease);
    let mut waiting = WaitingQueue::from([SglangRequest {
        is_decode_handoff: false,
        retracted: false,
        uuid: Uuid::from_u128(90_002),
        sequence_tokens: prompt.clone(),
        prompt_len: prompt.len(),
        max_output_tokens: 1,
        planned_output_ids: None,
        materialized_tokens: 0,
        kv_lease: RadixRequestLease::default(),
        allocated_tokens: 0,
        images: Vec::new(),
        pending_terminal: false,
    }]);
    let mut req = get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[], false, 0)
        .can_run
        .pop()
        .unwrap();

    assert_eq!(req.cached_tokens(), 4);
    assert_eq!(req.kv_pages()[..1], cached_pages[..1]);
    assert_eq!(req.kv_pages().len(), 2);
    req.append_output_token(20, config.block_size);

    // The seven-token blocker owns two pages (it evicts the second, no longer
    // referenced cached page to get them). With four pages total there is no free capacity, so the
    // cache-hit request is retracted while the blocker can fill the remainder of its own page.
    let blocker_tokens = vec![9, 10, 11, 12, 13, 14, 15];
    let blocker_alloc = kv_manager.allocate_for_request(&blocker_tokens).unwrap();
    let mut blocker = SglangRequest {
        is_decode_handoff: false,
        retracted: false,
        uuid: Uuid::from_u128(90_003),
        sequence_tokens: blocker_tokens,
        prompt_len: 7,
        max_output_tokens: 1,
        planned_output_ids: None,
        kv_lease: blocker_alloc.lease,
        materialized_tokens: 7,
        allocated_tokens: 8,
        images: Vec::new(),
        pending_terminal: false,
    };
    blocker.append_output_token(21, config.block_size);
    kv_manager.flush_events();
    buffer.drain();

    let mut running = vec![req, blocker];
    let (retracted, pressure_events) = decode::check_decode_mem_with_pressure_events(
        &mut running,
        &mut kv_manager,
        &config,
        1,
        12.5,
    );
    assert_eq!(retracted.len(), 1);
    assert_eq!(retracted[0].uuid, Uuid::from_u128(90_002));
    let [pressure] = pressure_events.as_slice() else {
        panic!("expected exactly one SGLang retraction pressure event");
    };
    assert_eq!(pressure.at_ms, 12.5);
    assert_eq!(pressure.kind, PressureKind::SglangRetraction);
    assert_eq!(pressure.request_id, Uuid::from_u128(90_002));
    assert_eq!(pressure.state_before.running_requests, 2);
    assert_eq!(pressure.state_after.running_requests, 1);
    assert_eq!(pressure.state_before.waiting_requests, None);
    assert_eq!(pressure.state_after.waiting_requests, None);
    assert!(pressure.request_active_blocks_before > 0);
    assert_eq!(pressure.logical_available_blocks_before, Some(0));
    assert_eq!(pressure.required_blocks_before, Some(1));
    assert!(pressure.state_after.active_blocks <= pressure.state_before.active_blocks);
    // The retraction itself frees only the page the request owned beyond the cached prefix; the
    // second cached page had already been evicted for the blocker.
    assert_eq!(pressure.state_before.active_blocks, 4);
    assert_eq!(pressure.state_after.active_blocks, 3);
    // That leaves the cached first page unprotected, and `release_req` evicts it toward the
    // blocker's 20-token reserve. The private page was never published.
    kv_manager.flush_events();
    let events = buffer.drain();
    assert_eq!(removed_event_count(&events), 1);
    assert_eq!(kv_manager.cache().page_pool.available(), 8);
    assert_eq!(kv_manager.cache_mut().match_prefix(&prompt).0, 0);
}

mod decode_retraction {
    use super::*;

    /// A page-size-1 decode row with a sampled token beyond its computed KV.
    fn row(
        kv_manager: &mut SglangKvManager,
        id: u128,
        prompt: &[u32],
        output: &[u32],
    ) -> SglangRequest {
        let mut sequence_tokens = [prompt, output].concat();
        let sequence_len = sequence_tokens.len();
        let allocation = kv_manager.allocate_for_request(&sequence_tokens).unwrap();
        sequence_tokens.push(100 + id as u32);
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::from_u128(id),
            sequence_tokens,
            prompt_len: prompt.len(),
            max_output_tokens: 8,
            planned_output_ids: None,
            kv_lease: allocation.lease,
            materialized_tokens: sequence_len,
            allocated_tokens: sequence_len,
            images: Vec::new(),
            pending_terminal: false,
        }
    }

    /// A page-size-1 host-loop ghost row that finished on its first output token; the next
    /// decode step still allocates that token's slot.
    fn ghost_row(
        kv_manager: &mut SglangKvManager,
        id: u128,
        prompt: &[u32],
        output: u32,
    ) -> SglangRequest {
        let allocation = kv_manager.allocate_for_request(prompt).unwrap();
        let mut sequence_tokens = prompt.to_vec();
        sequence_tokens.push(output);
        SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::from_u128(id),
            sequence_tokens,
            prompt_len: prompt.len(),
            max_output_tokens: 1,
            planned_output_ids: None,
            kv_lease: allocation.lease,
            materialized_tokens: prompt.len(),
            allocated_tokens: prompt.len(),
            images: Vec::new(),
            pending_terminal: true,
        }
    }

    fn ids(requests: &[SglangRequest]) -> Vec<u128> {
        requests
            .iter()
            .map(|request| request.uuid.as_u128())
            .collect()
    }

    #[test]
    fn victim_has_the_shortest_output_then_the_longest_prompt_and_is_the_latest_tie() {
        // No free slot and each row needs one for its next token: retracting any three-token
        // row lets the other three decode.
        let config = SglangConfig::from_args(&test_args(11, 1, 16));
        let mut kv_manager = SglangKvManager::new(11, 1, KvEventPublishers::default(), 0);
        let mut running = vec![
            row(&mut kv_manager, 1, &[10, 11], &[12]),
            row(&mut kv_manager, 2, &[20, 21, 22], &[]),
            row(&mut kv_manager, 3, &[30, 31, 32], &[]),
            row(&mut kv_manager, 4, &[40, 41], &[]),
        ];
        assert_eq!(kv_manager.cache().available_tokens(), 0);

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);

        // Rows 2 to 4 have one output and row 4 has the shortest prompt. Rows 2 and 3 tie
        // exactly, and the later one is retracted first.
        assert_eq!(ids(&retracted), vec![3]);
        // The survivors keep the retraction order: longest output, then shortest prompt.
        assert_eq!(ids(&running), vec![1, 4, 2]);
    }

    #[test]
    fn ghost_rows_are_sorted_with_the_batch_but_never_retracted() {
        let config = SglangConfig::from_args(&test_args(10, 1, 16));
        let mut kv_manager = SglangKvManager::new(10, 1, KvEventPublishers::default(), 0);
        // The ghost has the shortest output and the longest prompt, so it sorts last.
        let mut running = vec![
            ghost_row(&mut kv_manager, 1, &[10, 11, 12], 13),
            row(&mut kv_manager, 2, &[20, 21], &[22, 23]),
            row(&mut kv_manager, 3, &[30, 31], &[32]),
        ];
        assert_eq!(kv_manager.cache().available_tokens(), 0);

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);

        assert_eq!(ids(&retracted), vec![3]);
        assert_eq!(ids(&running), vec![2, 1]);
        assert!(running[1].pending_terminal);

        // With only ghost rows to choose from, nothing is retracted or reordered.
        let mut kv_manager = SglangKvManager::new(5, 1, KvEventPublishers::default(), 0);
        let mut running = vec![
            ghost_row(&mut kv_manager, 4, &[40, 41, 42], 43),
            ghost_row(&mut kv_manager, 5, &[50, 51], 52),
        ];
        assert!(decode::check_decode_mem(&mut running, &mut kv_manager, &config).is_empty());
        assert_eq!(ids(&running), vec![4, 5]);
    }

    #[test]
    fn retraction_evicts_toward_the_reserve_without_retracting_for_it() {
        let config = SglangConfig::from_args(&test_args(11, 1, 16));
        let mut kv_manager = SglangKvManager::new(11, 1, KvEventPublishers::default(), 0);
        let prefix = [1, 2, 3, 4];
        let cached = kv_manager.allocate_for_request(&prefix).unwrap();
        kv_manager.finish(&prefix, cached.lease);
        // Only the row with one output protects the cached prefix.
        let mut running = vec![
            row(&mut kv_manager, 1, &[10, 11], &[12]),
            row(&mut kv_manager, 2, &[20, 21], &[22]),
            row(&mut kv_manager, 3, &[1, 2, 3, 4, 5], &[]),
        ];
        assert_eq!(running[2].cached_tokens(), prefix.len());
        assert_eq!(kv_manager.cache().available_tokens(), 0);
        assert_eq!(kv_manager.cache().evictable_size, 0);

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);

        // Its own slot covers the survivors' next step, so nothing else is retracted although
        // even evicting the unprotected prefix stays short of 20 tokens for each survivor.
        assert_eq!(ids(&retracted), vec![3]);
        assert_eq!(ids(&running), vec![1, 2]);
        assert_eq!(kv_manager.cache().prefix_match_len(&prefix), 0);
        assert_eq!(kv_manager.cache().available_tokens(), 5);
    }
}

mod source_holds {
    /// SGLang caches a disaggregated prefill's prompt right after the prefill forward
    /// (`cache_unfinished_req` in `process_batch_result_disagg_prefill`) and publishes it with
    /// that batch, so routers and later prefills see it while its KV transfers.
    #[test]
    fn held_source_prompt_is_cached_and_published_after_prefill() {
        let mut core = SglangCore::new_with_kv_capture(args(), 0);
        let handoff_id = HandoffId::from(Uuid::from_u128(408));
        let mut prefill = request(Uuid::from_u128(308));
        prefill.max_output_tokens = 1;
        let prompt = prefill.tokens.clone();
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: prefill,
        })
        .unwrap();

        let pass = execute(&mut core, 0.0);
        assert!(pass.output_signals[0].completed);
        assert!(core.source_is_held(handoff_id));
        assert_eq!(stored_hashes(&pass.kv_events).len(), 2);
        assert_eq!(core.kv_manager.cache().prefix_match_len(&prompt), 8);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert!(core.drain_kv_events().is_empty());
        assert_eq!(core.kv_manager.cache().prefix_match_len(&prompt), 8);
    }

    #[rstest::rstest]
    fn completed_source_prompt_is_reusable_while_its_transfer_holds_the_tail(
        #[values(false, true)] cancel: bool,
    ) {
        let mut config = args();
        config.max_num_seqs = Some(2);
        let mut core = SglangCore::new_with_kv_capture(config, 0);
        let source_id = Uuid::from_u128(310);
        let source_handoff = HandoffId::from(Uuid::from_u128(410));
        let source_tokens = (0..10).collect::<Vec<_>>();
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: source_handoff,
            request: DirectRequest {
                tokens: source_tokens.clone(),
                max_output_tokens: 1,
                ..request(source_id)
            },
        })
        .unwrap();

        let source = execute(&mut core, 0.0);
        assert!(source.output_signals[0].completed);
        assert!(core.source_is_held(source_handoff));
        // The two full prompt pages are cached and locked; the partial tail stays private.
        assert_eq!(core.kv_manager.cache().prefix_match_len(&source_tokens), 8);
        assert_eq!(core.kv_manager.cache().protected_size, 8);
        assert_eq!(core.kv_manager.cache().evictable_size, 0);
        assert_eq!(occupied_tokens(&core), 12);
        assert_eq!(stored_hashes(&source.kv_events).len(), 2);

        let follower_id = Uuid::from_u128(311);
        let follower_handoff = HandoffId::from(Uuid::from_u128(411));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: follower_handoff,
            request: DirectRequest {
                tokens: (0..12).collect(),
                max_output_tokens: 1,
                ..request(follower_id)
            },
        })
        .unwrap();
        let follower = execute(&mut core, source.end_ms);
        assert_eq!(follower.admissions.len(), 1);
        assert_eq!(follower.admissions[0].uuid, follower_id);
        assert_eq!(follower.admissions[0].reused_input_tokens, 8);
        assert!(core.source_is_held(source_handoff));
        assert!(core.source_is_held(follower_handoff));
        assert_eq!(occupied_tokens(&core), 16);
        assert_eq!(core.kv_manager.cache().protected_size, 12);
        assert_eq!(stored_hashes(&follower.kv_events).len(), 1);

        let cleanup = if cancel {
            SchedulerCommand::CancelSource {
                handoff_id: source_handoff,
            }
        } else {
            SchedulerCommand::ReleaseSource {
                handoff_id: source_handoff,
            }
        };
        assert_eq!(
            core.apply_command(cleanup).unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(!core.source_is_held(source_handoff));
        assert!(core.source_is_held(follower_handoff));
        // Only the source's private tail page is freed; the follower keeps the shared path.
        assert_eq!(occupied_tokens(&core), 12);
        assert_eq!(core.kv_manager.cache().protected_size, 12);
        assert!(core.drain_kv_events().is_empty());

        core.apply_command(SchedulerCommand::ReleaseSource {
            handoff_id: follower_handoff,
        })
        .unwrap();
        assert!(core.is_drained());
        assert_eq!(core.kv_manager.cache().protected_size, 0);
        assert_eq!(core.kv_manager.cache().evictable_size, 12);
        assert_eq!(occupied_tokens(&core), 12);
        assert!(core.drain_kv_events().is_empty());
        core.kv_manager.evict(12);
        assert_eq!(occupied_tokens(&core), 0);
    }

    use super::*;

    fn args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(crate::engine::common::protocols::WorkerType::Prefill)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    fn request(uuid: Uuid) -> DirectRequest {
        DirectRequest {
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            uuid: Some(uuid),
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn occupied_tokens(core: &SglangCore) -> usize {
        core.kv_manager.cache().total_tokens() - core.kv_manager.cache().available_tokens()
    }

    #[test]
    fn terminal_completion_holds_source_without_running_cleanup() {
        let mut core = SglangCore::new(args());
        let request_id = Uuid::from_u128(301);
        let handoff_id = HandoffId::from(Uuid::from_u128(401));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: request(request_id),
        })
        .unwrap();

        let first = execute(&mut core, 0.0);
        assert!(!first.output_signals[0].completed);
        let terminal = execute(&mut core, first.end_ms);
        assert!(terminal.output_signals[0].completed);
        assert!(matches!(
            terminal.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::SourceHeld {
                handoff_id: held_handoff,
                request_id: held_request,
                ..
            }] if *held_handoff == handoff_id && *held_request == request_id
        ));
        assert!(core.source_is_held(handoff_id));
        assert!(core.running.is_empty());
        let held_tokens = occupied_tokens(&core);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert!(!core.source_is_held(handoff_id));
        let released_tokens = occupied_tokens(&core);
        assert!(released_tokens < held_tokens);

        core.apply_command(SchedulerCommand::ReleaseSource { handoff_id })
            .unwrap();
        assert_eq!(occupied_tokens(&core), released_tokens);
    }

    #[test]
    fn pp_prefill_handoff_waits_for_the_remaining_stages() {
        // A PP4 prefill worker retires passes at the per-stage rate; the
        // prompt's last forward still crosses three more stages before its KV
        // can leave, so the handoff waits three of that pass's forwards.
        let mut args = args();
        args.speedup_ratio = 1.0;
        args.pipeline_fill_passes = 3.0;
        let mut core = SglangCore::new(args);
        let request_id = Uuid::from_u128(311);
        let handoff_id = HandoffId::from(Uuid::from_u128(411));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: request(request_id),
        })
        .unwrap();

        let first = execute(&mut core, 0.0);
        let terminal = execute(&mut core, first.end_ms);
        let forward_ms = terminal.end_ms - first.end_ms;
        assert!(forward_ms > 0.0);
        let [
            SchedulerLifecycleEvent::SourceHeld {
                transfer_timing, ..
            },
        ] = terminal.lifecycle_events.as_slice()
        else {
            panic!("expected one held source: {:?}", terminal.lifecycle_events);
        };
        assert!((transfer_timing.pipeline_fill_ms - 3.0 * forward_ms).abs() < 1e-9);
    }

    #[test]
    fn host_loop_prefill_rank_hands_off_when_the_next_iteration_observes_the_extend() {
        // Replay caps a prefill-rank request at one output token; under the host
        // loop the finished extend joins SGLang's inflight queue instead of the
        // next batch, and the hold follows the observation one iteration later.
        let mut core = SglangCore::new(
            MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .num_gpu_blocks(16)
                .block_size(4)
                .max_num_seqs(Some(2))
                .worker_type(crate::engine::common::protocols::WorkerType::Prefill)
                .speedup_ratio(0.0)
                .sglang(Some(SglangArgs {
                    page_size: Some(4),
                    chunked_prefill_size: Some(16),
                    host_loop: true,
                    ..Default::default()
                }))
                .build()
                .unwrap(),
        );
        let request_id = Uuid::from_u128(304);
        let handoff_id = HandoffId::from(Uuid::from_u128(404));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: DirectRequest {
                max_output_tokens: 1,
                ..request(request_id)
            },
        })
        .unwrap();

        let launched = execute(&mut core, 0.0);
        assert!(launched.output_signals.is_empty());
        assert!(
            core.running.is_empty(),
            "a finished extend leaves the batch at once"
        );
        assert!(!core.source_is_held(handoff_id));

        // The next iteration only waits for that forward: no ghost decode batch.
        let observed = execute(&mut core, launched.end_ms);
        assert_eq!(core.last_forward_ms(), Some(0.0));
        assert!(observed.output_signals[0].completed);
        assert!(observed.lifecycle_events.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::SourceHeld { handoff_id: held, request_id: held_request, .. }
                if *held == handoff_id && *held_request == request_id
        )));
        assert!(core.source_is_held(handoff_id));
        assert!(core.is_empty());
    }

    #[test]
    fn cancel_and_early_release_cleanup_exactly_once() {
        let mut core = SglangCore::new(args());
        let first_id = HandoffId::from(Uuid::from_u128(402));
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: first_id,
            request: request(Uuid::from_u128(302)),
        })
        .unwrap();
        let first = execute(&mut core, 0.0);
        execute(&mut core, first.end_ms);
        let held_tokens = occupied_tokens(&core);

        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        let cancelled_tokens = occupied_tokens(&core);
        assert!(cancelled_tokens < held_tokens);
        core.apply_command(SchedulerCommand::CancelSource {
            handoff_id: first_id,
        })
        .unwrap();
        assert_eq!(occupied_tokens(&core), cancelled_tokens);

        core.kv_manager.evict(cancelled_tokens);
        let second_id = first_id;
        core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: second_id,
            request: request(Uuid::from_u128(303)),
        })
        .unwrap();
        assert!(core.source_is_registered(second_id));
        core.apply_command(SchedulerCommand::ReleaseSource {
            handoff_id: second_id,
        })
        .unwrap();
        assert!(!core.source_is_registered(second_id));

        let first = execute(&mut core, 0.0);
        let terminal = execute(&mut core, first.end_ms);
        assert!(terminal.output_signals[0].completed);
        assert!(!core.source_is_held(second_id));
    }

    #[test]
    fn ordinary_completion_keeps_default_cleanup() {
        let mut core = SglangCore::new(args());
        core.receive(request(Uuid::from_u128(306)));

        let first = execute(&mut core, 0.0);
        let terminal = execute(&mut core, first.end_ms);

        assert!(terminal.output_signals[0].completed);
        assert_eq!(occupied_tokens(&core), 8);
    }

    #[test]
    fn active_request_id_is_rejected_before_source_hold_registration() {
        let mut core = SglangCore::new(args());
        let request_id = Uuid::from_u128(307);
        let handoff_id = HandoffId::from(Uuid::from_u128(407));
        core.receive(request(request_id));

        assert!(
            core.apply_command(SchedulerCommand::Submit(request(request_id)))
                .is_err()
        );
        assert!(
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: request(request_id),
            })
            .is_err()
        );
        assert!(!core.source_is_registered(handoff_id));
        assert_eq!(core.num_requests(), 1);
    }
}

mod destination_lifecycle {
    use super::*;
    use crate::engine::common::protocols::WorkerType;

    fn args(worker_type: WorkerType) -> MockEngineArgs {
        args_with_decode_radix_cache(worker_type, true)
    }

    fn args_with_decode_radix_cache(worker_type: WorkerType, enabled: bool) -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(worker_type)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                disaggregation_decode_enable_radix_cache: enabled,
                num_reserved_decode_tokens: Some(0),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn decode_worker_serves_its_waiting_queue_in_arrival_order() {
        use crate::engine::scheduler::sglang::config::{SchedulePolicy, SglangConfig};

        let lpm = |worker_type| {
            let mut args = args(worker_type);
            args.sglang.as_mut().unwrap().schedule_policy = Some("lpm".to_string());
            SglangConfig::from_args(&args).schedule_policy
        };
        assert_eq!(lpm(WorkerType::Decode), SchedulePolicy::Fifo);
        assert_eq!(lpm(WorkerType::Prefill), SchedulePolicy::Lpm);
    }

    #[rstest::rstest]
    #[case::chunk_cache(false)]
    #[case::radix_cache(true)]
    fn decode_radix_cache_controls_reuse_of_transferred_prompts(#[case] enabled: bool) {
        let tokens = (0..8).collect::<Vec<_>>();
        let mut source = SglangCore::new_with_kv_capture(args(WorkerType::Prefill), 51);
        let mut destination = SglangCore::new_with_kv_capture(
            args_with_decode_radix_cache(WorkerType::Decode, enabled),
            52,
        );
        let mut now_ms = 0.0;
        let mut stored = Vec::new();
        let mut transferable = Vec::new();
        let mut reused = Vec::new();
        for turn in 0..2u128 {
            let request_id = Uuid::from_u128(21_000 + turn);
            let handoff_id = HandoffId::from(Uuid::from_u128(22_000 + turn));
            drive_source_to_hold(
                &mut source,
                handoff_id,
                request(request_id, tokens.clone(), 2),
            );
            let reserved = destination
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id,
                        request: request(request_id, tokens.clone(), 2),
                    },
                    true,
                )
                .unwrap();
            let [
                SchedulerLifecycleEvent::DestinationReserved {
                    transferable_prompt_tokens,
                    ..
                },
            ] = reserved.lifecycle_events.as_slice()
            else {
                panic!("expected one reservation: {:?}", reserved.lifecycle_events);
            };
            transferable.push(*transferable_prompt_tokens);
            assert_eq!(
                destination
                    .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            assert_eq!(
                source
                    .apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
            for _ in 0..8 {
                let pass = execute(&mut destination, now_ms);
                now_ms = pass.end_ms;
                stored.extend(stored_hashes(&pass.kv_events));
                reused.extend(
                    pass.admissions
                        .iter()
                        .map(|event| event.reused_input_tokens),
                );
                if destination.is_empty() {
                    break;
                }
            }
            assert!(destination.is_empty());
        }
        stored.extend(stored_hashes(&destination.drain_kv_events()));
        assert_eq!(reused.len(), 2);
        assert_eq!(
            reused,
            transferable
                .iter()
                .map(|missing| 8 - missing)
                .collect::<Vec<_>>()
        );

        if enabled {
            assert!(!stored.is_empty());
            assert!(transferable[1] < transferable[0], "{transferable:?}");
        } else {
            // SGLang's decode chunk cache: nothing is published or reused.
            assert!(stored.is_empty());
            assert_eq!(transferable, [8, 8]);
            assert_eq!(destination.kv_manager.cache().evictable_size, 0);
            assert_eq!(destination.kv_manager.cache().prefix_match_len(&tokens), 0);
        }
    }

    fn request(uuid: Uuid, tokens: Vec<u32>, max_output_tokens: usize) -> DirectRequest {
        DirectRequest {
            tokens,
            max_output_tokens,
            uuid: Some(uuid),
            arrival_timestamp_ms: None,
            ..Default::default()
        }
    }

    #[test]
    fn destination_lifecycle_is_counted_once_as_queued_decode() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(6)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                num_reserved_decode_tokens: Some(0),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let held_handoff = HandoffId::from(Uuid::from_u128(33_001));
        let held_uuid = Uuid::from_u128(33_002);
        let held = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: held_handoff,
                    request: request(held_uuid, vec![1; 4], 1),
                },
                true,
            )
            .unwrap();
        assert_eq!(held.lifecycle_events.len(), 1);
        assert_eq!(core.mocker_metrics().waiting_requests, 1);
        let held_snapshot = execute(&mut core, 0.0).fpm.unwrap();
        assert_eq!(held_snapshot.num_queued_decode, 1);
        assert_eq!(held_snapshot.sum_queued_decode_kv_tokens, 4);
        assert_eq!(held_snapshot.var_queued_decode_kv_tokens, 0.0);

        let pending_handoff = HandoffId::from(Uuid::from_u128(33_003));
        let pending = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: pending_handoff,
                    request: request(Uuid::from_u128(33_004), vec![2; 24], 1),
                },
                true,
            )
            .unwrap();
        assert!(pending.lifecycle_events.is_empty());
        assert_eq!(core.mocker_metrics().waiting_requests, 2);
        let mixed_snapshot = execute(&mut core, 0.0).fpm.unwrap();
        assert_eq!(mixed_snapshot.num_queued_decode, 2);
        assert_eq!(mixed_snapshot.sum_queued_decode_kv_tokens, 28);
        assert_eq!(mixed_snapshot.var_queued_decode_kv_tokens, 100.0);

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: pending_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(core.mocker_metrics().waiting_requests, 1);

        core.receive(request(Uuid::from_u128(33_005), vec![3; 4], 2));
        let blocker = execute(&mut core, 1.0);
        assert_eq!(core.mocker_metrics().running_requests, 1);
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination {
                handoff_id: held_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(core.mocker_metrics().waiting_requests, 1);

        let activated_pass = execute(&mut core, blocker.end_ms);
        let activated = activated_pass.fpm.unwrap();
        assert_eq!(activated.num_queued_prefill, 0);
        assert_eq!(activated.num_queued_decode, 1);
        assert_eq!(activated.sum_queued_decode_kv_tokens, 4);

        let completed = execute(&mut core, activated_pass.end_ms).fpm.unwrap();
        assert_eq!(completed.num_queued_decode, 0);
        assert_eq!(completed.sum_queued_decode_kv_tokens, 0);
        assert_eq!(core.mocker_metrics().waiting_requests, 0);
    }

    #[test]
    fn destination_transfer_footprint_excludes_decode_headroom() {
        let footprint = |max_output_tokens| {
            let mut core = SglangCore::new(args(WorkerType::Decode));
            let effects = core
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id: HandoffId::from(Uuid::new_v4()),
                        request: request(Uuid::new_v4(), (0..10).collect(), max_output_tokens),
                    },
                    true,
                )
                .unwrap();
            let [
                SchedulerLifecycleEvent::DestinationReserved {
                    transferable_prompt_tokens,
                    ..
                },
            ] = effects.lifecycle_events.as_slice()
            else {
                panic!("destination reservation should complete immediately");
            };
            *transferable_prompt_tokens
        };

        assert_eq!(footprint(1), 12);
        assert_eq!(footprint(128), 12);
    }

    fn execute(core: &mut SglangCore, now_ms: f64) -> crate::engine::scheduler::EnginePassResult {
        let mut collector = crate::engine::trace::TraceCollector::default();
        core.execute_pass(&mut collector, now_ms)
    }

    fn assert_no_republished_stores(published: &[u64], later: &[u64]) {
        assert!(later.iter().all(|hash| !published.contains(hash)));
    }

    #[test]
    fn overlap_preserves_zero_output_terminal_when_decode_reservation_fails() {
        let mut config = args_with_decode_radix_cache(WorkerType::Decode, false);
        config.num_gpu_blocks = 3;
        config.max_num_seqs = Some(2);
        config.sglang.as_mut().unwrap().host_loop = true;
        let mut core = SglangCore::new(config);

        let a = Uuid::from_u128(23_001);
        let b = Uuid::from_u128(23_002);
        let a_handoff = HandoffId::from(Uuid::from_u128(23_003));
        let b_handoff = HandoffId::from(Uuid::from_u128(23_004));

        // Reserve all three pages before A becomes a running request.
        for (handoff_id, request_id, tokens, outputs) in [
            (a_handoff, a, (0..4).collect(), 1),
            (b_handoff, b, (100..108).collect(), 0),
        ] {
            let reserved = core
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id,
                        request: request(request_id, tokens, outputs),
                    },
                    true,
                )
                .unwrap();
            assert!(matches!(
                reserved.lifecycle_events.as_slice(),
                [SchedulerLifecycleEvent::DestinationReserved { .. }]
            ));
        }

        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination {
                handoff_id: a_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        let launched = execute(&mut core, 0.0);
        assert!(launched.output_signals.is_empty());
        assert_eq!(core.kv_manager.cache().available_tokens(), 0);

        // A's unobserved terminal token needs a fourth page. B contributes a
        // tokenless terminal before that reservation fails.
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination {
                handoff_id: b_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        let mut now_ms = launched.end_ms;
        let mut signals = Vec::new();
        for _ in 0..4 {
            if core.is_drained() {
                break;
            }
            let pass = execute(&mut core, now_ms);
            now_ms = pass.end_ms;
            signals.extend(pass.output_signals);
        }

        assert_eq!(
            signals
                .iter()
                .map(|s| (s.uuid, s.token_id.is_some(), s.completed, s.rejected))
                .collect::<Vec<_>>(),
            vec![(a, true, true, false), (b, false, true, false)]
        );
        assert!(core.is_drained());
        assert_eq!(core.kv_manager.cache().available_tokens(), 12);
    }

    fn occupied_tokens(core: &SglangCore) -> usize {
        core.kv_manager.cache().total_tokens() - core.kv_manager.cache().available_tokens()
    }

    fn drive_source_to_hold(core: &mut SglangCore, handoff_id: HandoffId, req: DirectRequest) {
        assert!(matches!(
            core.apply_command(SchedulerCommand::SubmitHandoffPrefill {
                handoff_id,
                request: req,
            })
            .unwrap(),
            SchedulerCommandResult::Submitted(_)
        ));
        let mut now_ms = 0.0;
        for _ in 0..8 {
            let pass = execute(core, now_ms);
            now_ms = pass.end_ms;
            if core.is_empty() {
                break;
            }
        }
        assert!(core.is_empty());
        assert!(core.source_is_held(handoff_id));
        assert!(!core.is_drained());
    }

    #[test]
    fn handoff_prefill_to_reserved_decode_owns_kv_until_prebuilt_admission() {
        let logical_uuid = Uuid::from_u128(20_001);
        let handoff_id = HandoffId::from(Uuid::from_u128(20_002));
        let logical_tokens = (0..8).collect::<Vec<_>>();

        let mut source = SglangCore::new_with_kv_capture(args(WorkerType::Prefill), 41);
        let mut destination = SglangCore::new_with_kv_capture(args(WorkerType::Decode), 42);

        destination.receive(request(Uuid::from_u128(20_003), (0..4).collect(), 1));
        execute(&mut destination, 0.0);
        assert!(destination.is_empty());
        destination.drain_kv_events();

        let blocker_uuid = Uuid::from_u128(20_004);
        destination.receive(request(blocker_uuid, (100..104).collect(), 3));
        let blocker_first = execute(&mut destination, 1.0);
        assert_eq!(destination.running.len(), 1);
        destination.drain_kv_events();

        drive_source_to_hold(
            &mut source,
            handoff_id,
            request(logical_uuid, logical_tokens.clone(), 2),
        );

        let usage_before_reservation = occupied_tokens(&destination);
        let reserved = destination
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(logical_uuid, logical_tokens.clone(), 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(
            reserved.result,
            SchedulerCommandResult::DestinationAccepted {
                request_id: logical_uuid
            }
        );
        assert!(matches!(
            reserved.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: reserved_handoff,
                request_id: reserved_request,
                transferable_prompt_tokens: 4,
            }] if *reserved_handoff == handoff_id && *reserved_request == logical_uuid
        ));
        assert!(destination.destination_is_held(handoff_id));
        assert!(!destination.is_drained());
        assert_eq!(destination.running.len(), 1);
        assert!(occupied_tokens(&destination) > usage_before_reservation);
        assert!(stored_hashes(&destination.drain_kv_events()).is_empty());

        let reserved_pages = destination.destination_pages(handoff_id);
        let protected_before_activation = destination.kv_manager.cache().protected_size;
        assert!(!reserved_pages.is_empty());
        assert_eq!(
            destination
                .apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        let ready = destination
            .prebuilt_request(logical_uuid)
            .expect("activated request must be prebuilt-ready");
        assert_eq!(ready.kv_pages(), reserved_pages);
        assert_eq!(ready.cached_tokens(), 4);
        assert_eq!(destination.running.len(), 1);
        // The transferred page stays private: only the matched prefix is cached and locked.
        assert_eq!(
            destination.kv_manager.cache().protected_size,
            protected_before_activation
        );
        assert!(stored_hashes(&destination.drain_kv_events()).is_empty());
        assert_eq!(
            destination
                .kv_manager
                .cache()
                .prefix_match_len(&logical_tokens),
            4
        );

        assert_eq!(
            source
                .apply_command(SchedulerCommand::ReleaseSource { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(source.is_empty());
        assert!(source.is_drained());

        let blocked = execute(&mut destination, blocker_first.end_ms);
        assert!(
            blocked
                .admissions
                .iter()
                .all(|admission| admission.uuid != logical_uuid)
        );
        let ready = destination
            .prebuilt_request(logical_uuid)
            .expect("full running batch must keep request ready");
        assert_eq!(ready.kv_pages(), reserved_pages);
        assert!(stored_hashes(&blocked.kv_events).is_empty());
        assert_eq!(
            destination
                .kv_manager
                .cache()
                .prefix_match_len(&logical_tokens),
            4
        );

        let blocker_terminal = execute(&mut destination, blocked.end_ms);
        assert!(
            blocker_terminal
                .admissions
                .iter()
                .all(|admission| admission.uuid != logical_uuid)
        );
        assert!(
            blocker_terminal
                .output_signals
                .iter()
                .any(|signal| signal.uuid == blocker_uuid && signal.completed)
        );
        assert!(destination.prebuilt_request(logical_uuid).is_some());
        assert_eq!(
            destination
                .kv_manager
                .cache()
                .prefix_match_len(&logical_tokens),
            4
        );

        let admitted = execute(&mut destination, blocker_terminal.end_ms);
        let logical_admissions = admitted
            .admissions
            .iter()
            .filter(|admission| admission.uuid == logical_uuid)
            .collect::<Vec<_>>();
        assert_eq!(logical_admissions.len(), 1);
        assert_eq!(logical_admissions[0].reused_input_tokens, 4);
        let running = destination
            .running
            .iter()
            .find(|request| request.uuid == logical_uuid)
            .expect("prebuilt request must enter the running batch");
        assert!(running.kv_pages().starts_with(&reserved_pages));
        // Running admission caches the transferred page and publishes it once.
        assert_eq!(
            destination
                .kv_manager
                .cache()
                .prefix_match_len(&logical_tokens),
            8
        );
        let admission_stores = stored_hashes(&admitted.kv_events);
        assert!(admission_stores.is_empty()); // Queued until the decode/idle publish boundary.

        let terminal = execute(&mut destination, admitted.end_ms);
        assert!(
            terminal
                .output_signals
                .iter()
                .any(|signal| signal.uuid == logical_uuid && signal.completed)
        );
        assert!(destination.is_empty());
        assert!(destination.is_drained());
        assert!(!destination.destination_is_held(handoff_id));
        assert_eq!(destination.kv_manager.cache().protected_size, 0);
        let terminal_stores = stored_hashes(&terminal.kv_events);
        assert_eq!(terminal_stores.len(), 1);
        assert_no_republished_stores(&admission_stores, &terminal_stores);
    }

    #[test]
    fn destination_cancel_reaches_activated_request_exactly_once() {
        let request_id = Uuid::from_u128(20_101);
        let handoff_id = HandoffId::from(Uuid::from_u128(20_102));
        let mut core = SglangCore::new_with_kv_capture(args(WorkerType::Decode), 43);

        let reserve = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: request(request_id, (0..8).collect(), 4),
                },
                true,
            )
            .unwrap();
        assert!(matches!(
            reserve.lifecycle_events.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved { .. }]
        ));
        assert_eq!(
            core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        execute(&mut core, 0.0);
        assert!(
            core.running
                .iter()
                .any(|request| request.uuid == request_id)
        );

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.is_empty());
        assert!(core.is_drained());
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Noop
        );
    }

    #[test]
    fn activation_retains_partial_page_until_owner_is_cancelled() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(3)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                num_reserved_decode_tokens: Some(0),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let blocker = core.kv_manager.reserve_decode_pages(1).unwrap();
        let owner_handoff = HandoffId::from(Uuid::from_u128(20_201));
        let follower_handoff = HandoffId::from(Uuid::from_u128(20_202));
        let follower_request = Uuid::from_u128(20_203);

        let owner = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: owner_handoff,
                    request: request(Uuid::from_u128(20_204), (0..5).collect(), 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(owner.lifecycle_events.len(), 1);
        let follower = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: follower_handoff,
                    request: request(follower_request, (100..104).collect(), 2),
                },
                true,
            )
            .unwrap();
        assert!(follower.lifecycle_events.is_empty());

        let activated = core
            .apply_command_effects(
                SchedulerCommand::ActivateDestination {
                    handoff_id: owner_handoff,
                },
                true,
            )
            .unwrap();
        assert!(
            activated.lifecycle_events.is_empty(),
            "activating a partial prompt must retain its complete physical page"
        );

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: owner_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.destination_is_held(follower_handoff));
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination {
                handoff_id: follower_handoff,
            })
            .unwrap(),
            SchedulerCommandResult::Applied
        );
        core.kv_manager.release_decode_reservation(blocker);
    }

    #[test]
    fn running_admission_of_a_shared_prompt_retries_the_pending_destination() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(5)
            .block_size(4)
            .max_num_seqs(Some(1))
            .worker_type(WorkerType::Decode)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                disaggregation_decode_enable_radix_cache: true,
                num_reserved_decode_tokens: Some(0),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new_with_kv_capture(args, 44);
        let blocker = core.kv_manager.reserve_decode_pages(1).unwrap();
        let tokens = (0..8).collect::<Vec<_>>();
        let first_handoff = HandoffId::from(Uuid::from_u128(20_301));
        let second_handoff = HandoffId::from(Uuid::from_u128(20_302));
        let second_request = Uuid::from_u128(20_303);

        let first = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: first_handoff,
                    request: request(Uuid::from_u128(20_304), tokens.clone(), 2),
                },
                true,
            )
            .unwrap();
        assert_eq!(first.lifecycle_events.len(), 1);
        core.apply_command(SchedulerCommand::ActivateDestination {
            handoff_id: first_handoff,
        })
        .unwrap();

        // The ready prompt is not reusable yet: the same prompt needs two fresh
        // pages plus its output budget, and only two pages are free.
        let second = core
            .apply_command_effects(
                SchedulerCommand::ReserveDestination {
                    handoff_id: second_handoff,
                    request: request(second_request, tokens.clone(), 2),
                },
                true,
            )
            .unwrap();
        assert!(second.lifecycle_events.is_empty());
        let attempts = core.destination_reservation_attempts();
        assert!(core.retry_pending_destinations().is_empty());
        assert_eq!(core.destination_reservation_attempts(), attempts);

        let admitted = execute(&mut core, 0.0);
        assert_eq!(admitted.admissions.len(), 1);
        assert_eq!(core.kv_manager.cache().prefix_match_len(&tokens), 8);
        assert!(stored_hashes(&admitted.kv_events).is_empty());

        // Publication changes what the pending head can reuse, so it is retried.
        let retried = core.retry_pending_destinations();
        assert!(matches!(
            retried.as_slice(),
            [SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: reserved_handoff,
                request_id: reserved_request,
                transferable_prompt_tokens: 0,
            }] if *reserved_handoff == second_handoff && *reserved_request == second_request
        ));
        assert_eq!(core.destination_reservation_attempts(), attempts + 1);

        for handoff_id in [second_handoff, first_handoff] {
            assert_eq!(
                core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                    .unwrap(),
                SchedulerCommandResult::Applied
            );
        }
        core.kv_manager.release_decode_reservation(blocker);
        assert_eq!(stored_hashes(&core.drain_kv_events()).len(), 2);
        assert!(core.is_drained());
        assert_eq!(core.kv_manager.cache().protected_size, 0);
    }

    #[test]
    fn cancelled_ready_transfer_caches_its_full_pages_and_frees_the_tail() {
        let mut core = SglangCore::new_with_kv_capture(args(WorkerType::Decode), 45);
        let tokens = (0..10).collect::<Vec<_>>();
        let handoff_id = HandoffId::from(Uuid::from_u128(20_401));
        core.apply_command(SchedulerCommand::ReserveDestination {
            handoff_id,
            request: request(Uuid::from_u128(20_402), tokens.clone(), 2),
        })
        .unwrap();
        core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
            .unwrap();
        assert_eq!(occupied_tokens(&core), 12);
        assert_eq!(core.kv_manager.cache().prefix_match_len(&tokens), 0);
        assert!(core.drain_kv_events().is_empty());

        assert_eq!(
            core.apply_command(SchedulerCommand::CancelDestination { handoff_id })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert!(core.is_drained());
        assert_eq!(stored_hashes(&core.drain_kv_events()).len(), 2);
        assert_eq!(core.kv_manager.cache().prefix_match_len(&tokens), 8);
        assert_eq!(occupied_tokens(&core), 8);
        assert_eq!(core.kv_manager.cache().protected_size, 0);
        assert_eq!(core.kv_manager.cache().evictable_size, 8);
    }

    #[test]
    fn decode_without_prefix_caching_never_caches_transferred_prompts() {
        let mut config = args(WorkerType::Decode);
        config.enable_prefix_caching = false;
        let mut core = SglangCore::new_with_kv_capture(config, 46);
        let tokens = (0..8).collect::<Vec<_>>();
        let mut now_ms = 0.0;
        for (handoff, request_id) in [(20_501, 20_503), (20_502, 20_504)] {
            let handoff_id = HandoffId::from(Uuid::from_u128(handoff));
            let reserved = core
                .apply_command_effects(
                    SchedulerCommand::ReserveDestination {
                        handoff_id,
                        request: request(Uuid::from_u128(request_id), tokens.clone(), 2),
                    },
                    true,
                )
                .unwrap();
            assert!(matches!(
                reserved.lifecycle_events.as_slice(),
                [SchedulerLifecycleEvent::DestinationReserved {
                    transferable_prompt_tokens: 8,
                    ..
                }]
            ));
            core.apply_command(SchedulerCommand::ActivateDestination { handoff_id })
                .unwrap();
            now_ms = execute(&mut core, now_ms).end_ms;
            assert_eq!(core.kv_manager.cache().prefix_match_len(&tokens), 0);
        }
        for _ in 0..8 {
            if core.is_drained() {
                break;
            }
            now_ms = execute(&mut core, now_ms).end_ms;
            assert_eq!(core.kv_manager.cache().prefix_match_len(&tokens), 0);
        }
        assert!(core.is_drained());
        assert_eq!(occupied_tokens(&core), 0);
        assert_eq!(core.kv_manager.cache().evictable_size, 0);
        assert!(core.drain_kv_events().is_empty());
    }
}

mod scheduling {
    use super::*;

    #[test]
    fn cache_salt_namespaces_lpm_scores_and_in_batch_deduplication() {
        let tokens = (0..32).collect::<Vec<_>>();
        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(&test_args(64, 4, 64))
        };
        let make = |index, salt: Option<&str>| {
            SglangRequest::new(
                DirectRequest {
                    tokens: tokens.clone(),
                    max_output_tokens: 1,
                    uuid: Some(Uuid::from_u128(index)),
                    cache_salt: salt.map(str::to_owned),
                    ..Default::default()
                },
                4,
                0,
            )
        };
        let mut manager = SglangKvManager::new(256, 4, KvEventPublishers::default(), 0);
        let mut waiting = WaitingQueue::from([
            make(1, Some("alpha")),
            make(2, Some("alpha")),
            make(3, Some("beta")),
            make(4, None),
            make(5, Some("")),
        ]);
        apply_schedule_policy(&mut waiting, &manager, &config);
        assert_eq!(
            waiting
                .iter()
                .map(|request| request.uuid.as_u128())
                .collect::<Vec<_>>(),
            [1, 3, 4, 2, 5]
        );

        let mut cached = RadixRequestLease::new(Some("beta".into()));
        manager
            .allocate_for_request_lease(&tokens, &mut cached)
            .unwrap();
        manager.finish(&tokens, cached);
        let mut waiting = WaitingQueue::from([make(1, Some("alpha")), make(2, Some("beta"))]);
        apply_schedule_policy(&mut waiting, &manager, &config);
        assert_eq!(waiting[0].uuid, Uuid::from_u128(2));
    }

    #[test]
    fn test_lpm_reorders_by_current_sequence_prefix_match() {
        let mut kv_manager = SglangKvManager::new(1000, 1, KvEventPublishers::default(), 0);
        kv_manager
            .cache_mut()
            .insert(&[1, 2, 3, 4, 5], &[0, 1, 2, 3, 4]);

        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .engine_type(EngineType::Sglang)
                    .block_size(1)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };

        let no_match_uuid = Uuid::new_v4();
        let match_uuid = Uuid::new_v4();
        let mut waiting = WaitingQueue::from([
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: no_match_uuid,
                sequence_tokens: vec![9, 8, 7],
                prompt_len: 3,
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: RadixRequestLease::default(),
                allocated_tokens: 0,
                images: Vec::new(),
                pending_terminal: false,
            },
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: match_uuid,
                sequence_tokens: vec![1, 2, 3, 4, 5, 6, 7],
                prompt_len: 5,
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: RadixRequestLease::default(),
                allocated_tokens: 0,
                images: Vec::new(),
                pending_terminal: false,
            },
        ]);

        apply_schedule_policy(&mut waiting, &kv_manager, &config);
        assert_eq!(waiting[0].uuid, match_uuid);
        assert_eq!(waiting[1].uuid, no_match_uuid);
    }

    #[test]
    fn test_lpm_deprioritizes_duplicate_short_prefixes() {
        let config = SglangConfig {
            schedule_policy: SchedulePolicy::Lpm,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .block_size(1)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };
        let kv_manager = SglangKvManager::new(1000, 1, KvEventPublishers::default(), 0);
        let duplicate_prefix = (0..32).collect::<Vec<_>>();
        let mut waiting = WaitingQueue::default();
        for _ in 0..33 {
            waiting.push_back(SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::new_v4(),
                sequence_tokens: duplicate_prefix.clone(),
                prompt_len: duplicate_prefix.len(),
                max_output_tokens: 1,
                planned_output_ids: None,
                materialized_tokens: 0,
                kv_lease: RadixRequestLease::default(),
                allocated_tokens: 0,
                images: Vec::new(),
                pending_terminal: false,
            });
        }
        let unique_uuid = Uuid::new_v4();
        waiting.push_back(SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: unique_uuid,
            sequence_tokens: (100..132).collect(),
            prompt_len: 32,
            max_output_tokens: 1,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: RadixRequestLease::default(),
            allocated_tokens: 0,
            images: Vec::new(),
            pending_terminal: false,
        });

        apply_schedule_policy(&mut waiting, &kv_manager, &config);
        assert_eq!(
            waiting.iter().position(|req| req.uuid == unique_uuid),
            Some(1)
        );
    }
}

mod core_behavior {
    /// Run `core` until it drains and return each pass's KV events as (kind, blocks) pairs,
    /// keyed by pass index and skipping passes that published nothing.
    fn kv_event_shapes_by_pass(core: &mut SglangCore) -> Vec<(usize, Vec<(&'static str, usize)>)> {
        let mut collector = crate::engine::trace::TraceCollector::default();
        let mut now_ms = 0.0;
        let mut next_event_id = None;
        let mut shapes = Vec::new();
        for pass_index in 0.. {
            if core.is_empty() {
                break;
            }
            let pass = core.execute_pass(&mut collector, now_ms);
            now_ms = pass.end_ms;
            for event in &pass.kv_events {
                let expected_id = *next_event_id.get_or_insert(event.event_id);
                assert_eq!(
                    event.event_id, expected_id,
                    "published event IDs must be dense"
                );
                next_event_id = Some(expected_id + 1);
            }
            if pass.kv_events.is_empty() {
                continue;
            }
            let shape = pass
                .kv_events
                .iter()
                .map(|event| match &event.data {
                    KvEventData::Stored(stored) => ("stored", stored.blocks.len()),
                    KvEventData::Removed { block_hashes } => ("removed", block_hashes.len()),
                })
                .collect();
            shapes.push((pass_index, shape));
        }
        shapes
    }

    /// SGLang inserts a request into its radix tree after the prefill forward
    /// (`cache_unfinished_req`) and again when the request finishes (`release_kv_cache`), so
    /// routers see two stores per request and no decoded block before the request finishes.
    #[rstest::rstest]
    fn kv_events_store_prompt_after_prefill_and_outputs_at_finish(
        #[values(1, 4)] page_size: usize,
        #[values(false, true)] host_loop: bool,
    ) {
        let mut args = test_args(64, page_size, 64);
        args.sglang.as_mut().unwrap().host_loop = host_loop;
        let mut core = SglangCore::new_with_kv_capture(args, 0);
        core.receive(direct_request((1..=8).collect(), 50));

        // One prefill pass emits the first token; 49 decode passes emit the rest. The prefill
        // insert covers the prompt only: the first output token's KV is not computed yet.
        let prefill_blocks = 8 / page_size;
        let observation_delay = usize::from(host_loop);
        let finished_blocks = 57 / page_size;
        assert_eq!(
            kv_event_shapes_by_pass(&mut core),
            vec![
                (observation_delay, vec![("stored", prefill_blocks)]),
                (
                    49 + observation_delay,
                    vec![("stored", finished_blocks - prefill_blocks)]
                ),
            ]
        );
    }

    /// SGLang merges consecutive removals in its event queue and publishes the queue after each
    /// prefill batch, every `--decode-log-interval` (40) decode batches, and when it goes idle.
    #[test]
    fn kv_event_removals_are_batched_like_sglang() {
        // Sixty cached one-token prompts fill the pool except for room for an eight-token prompt,
        // so each consumed output token evicts one cached page on its decode forward.
        let mut core = SglangCore::new_with_kv_capture(test_args(68, 1, 64), 0);
        for token in 1_000..1_060 {
            let cached = core.kv_manager.allocate_for_request(&[token]).unwrap();
            core.kv_manager.finish(&[token], cached.lease);
        }
        core.kv_manager.flush_events();
        core.drain_kv_events();
        core.receive(direct_request((1..=8).collect(), 50));

        assert_eq!(
            kv_event_shapes_by_pass(&mut core),
            vec![
                (0, vec![("stored", 8)]),
                (40, vec![("removed", 40)]),
                (49, vec![("removed", 9), ("stored", 49)]),
            ]
        );
    }

    /// SGLang inserts the prompt after the prefill forward even when the first output slots do
    /// not fit. Retraction may then evict the released prompt toward the survivor reserve.
    #[test]
    fn prompt_is_cached_after_prefill_when_first_output_slots_do_not_fit() {
        let config = SglangConfig::from_args(&test_args(4, 4, 16));
        let (buffer, sink) = capture_kv_event_sink();
        // Four pages: the two prompts take three, and their first output tokens need two more.
        let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::new(Some(sink)), 0);
        let mut prefilled = |uuid: u128, prompt: Vec<u32>| {
            let alloc = kv_manager.allocate_for_request(&prompt).unwrap();
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::from_u128(uuid),
                prompt_len: prompt.len(),
                materialized_tokens: prompt.len(),
                allocated_tokens: prompt.len(),
                sequence_tokens: prompt,
                max_output_tokens: 4,
                planned_output_ids: None,
                kv_lease: alloc.lease,
                images: Vec::new(),
                pending_terminal: false,
            }
        };
        let short_prompt: Vec<u32> = (1..=4).collect();
        let long_prompt: Vec<u32> = (11..=18).collect();
        let mut running = vec![
            prefilled(90_101, short_prompt.clone()),
            prefilled(90_102, long_prompt.clone()),
        ];

        let first =
            decode::simulate_prefill_first_tokens(&mut running, &mut kv_manager, &config, 0.0)
                .unwrap();
        assert_eq!(first.output_signals.len(), 2);
        assert!(running.iter().all(|req| req.output_len() == 1));
        assert_eq!(kv_manager.cache().prefix_match_len(&short_prompt), 4);
        assert_eq!(kv_manager.cache().prefix_match_len(&long_prompt), 8);
        kv_manager.flush_events();
        let stored_blocks: Vec<_> = buffer
            .drain()
            .iter()
            .map(|event| match &event.data {
                KvEventData::Stored(stored) => stored.blocks.len(),
                KvEventData::Removed { .. } => panic!("nothing is evicted"),
            })
            .collect();
        assert_eq!(stored_blocks, vec![1, 2]);

        let decode = simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        let [retracted] = decode.requests.as_slice() else {
            panic!("expected one retraction");
        };
        assert_eq!(retracted.uuid, Uuid::from_u128(90_102));
        assert_eq!(running[0].output_len(), 2);
        assert_eq!(kv_manager.cache().prefix_match_len(&short_prompt), 4);
    }

    /// SGLang aborts a running request by finishing it (`to_finish`), and the finish
    /// (`release_kv_cache` with insertion) caches the tokens it decoded.
    #[test]
    fn cancelled_running_request_caches_its_decoded_tokens() {
        let mut core = SglangCore::new_with_kv_capture(test_args(64, 4, 64), 0);
        let uuid = Uuid::from_u128(90_201);
        let mut request = direct_request((1..=8).collect(), 50);
        request.uuid = Some(uuid);
        core.receive(request);

        // Eight sampled outputs have seven computed KV tokens: only one full decoded page.
        let mut collector = crate::engine::trace::TraceCollector::default();
        let mut now_ms = 0.0;
        for _ in 0..8 {
            now_ms = core.execute_pass(&mut collector, now_ms).end_ms;
        }
        let sequence = core.running[0].sequence_prefix(16).to_vec();
        assert_eq!(core.kv_manager.cache().prefix_match_len(&sequence), 8);
        core.drain_kv_events();

        core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
            .unwrap();
        assert!(core.is_empty());
        assert_eq!(core.kv_manager.cache().prefix_match_len(&sequence), 12);
        assert_eq!(stored_hashes(&core.drain_kv_events()).len(), 1);
    }

    use super::*;

    #[test]
    fn test_planned_output_tokens_are_emitted_exactly() {
        use crate::engine::common::hashing::compute_block_hash_for_seq;

        for emit_token_ids in [false, true] {
            let mut args = test_args(100, 4, 8);
            args.emit_kv_token_ids = emit_token_ids;
            let mut core = SglangCore::new_with_kv_capture(args, 0);
            let uuid = Uuid::from_u128(0xB0B);
            let planned = vec![111, 222, 333, 444, 555, 666, 777];
            let mut sequence = vec![1, 2];
            core.receive(DirectRequest {
                cache_salt: None,
                tokens: sequence.clone(),
                max_output_tokens: planned.len(),
                output_token_ids: Some(planned.clone()),
                uuid: Some(uuid),
                arrival_timestamp_ms: None,
                session_id: None,
                images: Vec::new(),
                kv_hint: None,
            });
            sequence.extend_from_slice(&planned);

            let mut collector = crate::engine::trace::TraceCollector::default();
            let mut emitted = Vec::new();
            let mut stored_blocks = Vec::new();
            for step in 0..planned.len() {
                let pass = core.execute_pass(&mut collector, step as f64);
                emitted.extend(
                    pass.output_signals
                        .into_iter()
                        .filter(|signal| signal.uuid == uuid)
                        .map(|signal| signal.token_id.expect("planned token should be present")),
                );
                for event in pass.kv_events {
                    if let KvEventData::Stored(stored) = event.data {
                        stored_blocks.extend(stored.blocks);
                    }
                }
            }

            assert_eq!(emitted, planned);
            assert!(core.is_empty());
            // The second complete block consists entirely of generated tokens.
            assert_eq!(stored_blocks.len(), sequence.len() / 4);
            for (block, expected_tokens) in stored_blocks.iter().zip(sequence.chunks_exact(4)) {
                assert_eq!(
                    block.token_ids.as_deref(),
                    emit_token_ids.then_some(expected_tokens)
                );
                if let Some(tokens) = &block.token_ids {
                    assert_eq!(
                        compute_block_hash_for_seq(tokens, 4)[0].0,
                        block.tokens_hash
                    );
                }
            }
        }
    }

    #[test]
    fn test_chunked_prefill_waits_when_full_prompt_does_not_fit() {
        let config = SglangConfig {
            chunked_prefill_size: 8,
            ..SglangConfig::from_args(
                &MockEngineArgs::builder()
                    .block_size(4)
                    .speedup_ratio(1.0)
                    .build()
                    .unwrap(),
            )
        };
        let mut kv_manager = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let mut waiting = WaitingQueue::from([SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1; 16],
            prompt_len: 16,
            max_output_tokens: 2,
            planned_output_ids: None,
            materialized_tokens: 0,
            kv_lease: RadixRequestLease::default(),
            allocated_tokens: 0,
            images: Vec::new(),
            pending_terminal: false,
        }]);

        let admit =
            get_new_batch_prefill(&mut waiting, &mut kv_manager, &config, 0.7, &[], false, 0);
        assert!(admit.can_run.is_empty());
        assert_eq!(waiting.len(), 1);
        assert_eq!(kv_manager.cache().available_tokens(), 12);
    }

    #[test]
    fn test_decode_allocation_is_page_aware() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let mut alloc = kv_manager
            .allocate_for_request(&[1, 2, 3, 4, 5, 6])
            .unwrap();
        kv_manager.extend_cached_prefix(&[1, 2, 3, 4], &mut alloc.lease);
        let mut running = vec![SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4, 5, 6],
            prompt_len: 6,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: alloc.lease,
            materialized_tokens: 6,
            allocated_tokens: 8,
            images: Vec::new(),
            pending_terminal: false,
        }];

        let first = simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 8);
        assert_eq!(running[0].output_len(), 1);
        assert_eq!(first.output_signals.len(), 1);

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 8);

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert_eq!(running[0].allocated_tokens, 8);
        assert_eq!(running[0].materialized_tokens, 8);
        let final_step = decode::simulate_decode_step_with_sampler(
            &mut running,
            &mut kv_manager,
            &config,
            None,
            0.0,
        )
        .unwrap();
        assert!(running.is_empty());
        assert_eq!(final_step.completed_requests[0].allocated_tokens, 12);
        assert_eq!(final_step.completed_requests[0].materialized_tokens, 9);
        assert_eq!(final_step.completed_requests[0].output_len(), 4);
    }

    #[test]
    fn test_decode_speedup_ratio_scales_sglang_decode_time() {
        let base_args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .speedup_ratio(2.0)
            .decode_speedup_ratio(1.0)
            .build()
            .unwrap();
        let fast_args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .speedup_ratio(2.0)
            .decode_speedup_ratio(4.0)
            .build()
            .unwrap();
        let base_config = SglangConfig::from_args(&base_args);
        let fast_config = SglangConfig::from_args(&fast_args);

        let mut base_kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let base_alloc = base_kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut base_running = vec![SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: base_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        }];

        let mut fast_kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let fast_alloc = fast_kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        let mut fast_running = vec![SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 4,
            planned_output_ids: None,
            kv_lease: fast_alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        }];

        let base = simulate_decode_step(
            &mut base_running,
            &mut base_kv_manager,
            &base_config,
            0.0,
            true,
        );
        let fast = simulate_decode_step(
            &mut fast_running,
            &mut fast_kv_manager,
            &fast_config,
            0.0,
            true,
        );
        let ratio = base.end_ms / fast.end_ms;

        assert!(base.end_ms > fast.end_ms);
        assert!(
            (ratio - 4.0).abs() < 1e-3,
            "expected 4x decode speedup ratio, got {ratio}"
        );
    }

    #[test]
    fn test_check_decode_mem_preserves_generated_output_on_retract() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
        let first = kv_manager.cache_mut().page_pool.allocate_pages(2).unwrap();
        let second = kv_manager.cache_mut().page_pool.allocate_pages(2).unwrap();

        let mut running = vec![
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![1, 2, 3, 4, 11, 12, 13, 14, 15],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: RadixRequestLease::from_parts(first, 8, 4, kv_manager.cache().root()),
                materialized_tokens: 8,
                allocated_tokens: 8,
                images: Vec::new(),
                pending_terminal: false,
            },
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::new_v4(),
                sequence_tokens: vec![9, 8, 7, 6, 21],
                prompt_len: 4,
                max_output_tokens: 10,
                planned_output_ids: None,
                kv_lease: RadixRequestLease::from_parts(second, 5, 4, kv_manager.cache().root()),
                materialized_tokens: 5,
                allocated_tokens: 8,
                images: Vec::new(),
                pending_terminal: false,
            },
        ];

        let retracted = decode::check_decode_mem(&mut running, &mut kv_manager, &config);
        assert_eq!(retracted.len(), 1);
        assert_eq!(retracted[0].output_tokens(), &[21]);
        assert_eq!(retracted[0].materialized_tokens, 0);
        assert!(retracted[0].kv_pages().is_empty());
    }

    #[test]
    fn test_decoded_tokens_are_cached_only_when_the_request_finishes() {
        let config = SglangConfig::from_args(
            &MockEngineArgs::builder()
                .engine_type(EngineType::Sglang)
                .block_size(4)
                .speedup_ratio(1.0)
                .build()
                .unwrap(),
        );
        let mut kv_manager = SglangKvManager::new(64, 4, KvEventPublishers::default(), 0);
        let mut alloc = kv_manager.allocate_for_request(&[1, 2, 3, 4]).unwrap();
        // SGLang `cache_unfinished_req` after the prefill forward.
        kv_manager.extend_cached_prefix(&[1, 2, 3, 4], &mut alloc.lease);
        let mut running = vec![SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::new_v4(),
            sequence_tokens: vec![1, 2, 3, 4],
            prompt_len: 4,
            max_output_tokens: 5,
            planned_output_ids: None,
            kv_lease: alloc.lease,
            materialized_tokens: 4,
            allocated_tokens: 4,
            images: Vec::new(),
            pending_terminal: false,
        }];

        for _ in 0..4 {
            simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        }
        assert_eq!(running.len(), 1);
        let sequence = running[0].sequence_prefix(8).to_vec();
        assert_eq!(kv_manager.cache().prefix_match_len(&sequence), 4);

        simulate_decode_step(&mut running, &mut kv_manager, &config, 0.0, false);
        assert!(running.is_empty());
        assert_eq!(kv_manager.cache().prefix_match_len(&sequence), 8);
    }

    #[test]
    fn test_active_decode_blocks_tracks_page_reserved_occupancy_in_blocks() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(32)
            .block_size(4)
            .speedup_ratio(1.0)
            .sglang(Some(SglangArgs {
                chunked_prefill_size: Some(8),
                page_size: Some(4),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        core.receive(crate::engine::common::protocols::DirectRequest {
            cache_salt: None,
            tokens: vec![1; 6],
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: None,
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let pass = core.execute_pass_internal(None, 0.0);
        assert_eq!(pass.completed_requests, 0);
        assert_eq!(pass.mocker_metrics.active_decode_blocks, 2);
    }

    #[test]
    fn test_mtp_batch_applies_request_bursts_in_stable_order() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(32)
            .block_size(4)
            .speedup_ratio(0.0)
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1,1".to_string()))
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let short = Uuid::from_u128(101);
        let long = Uuid::from_u128(102);
        let longer = Uuid::from_u128(103);
        for (uuid, tokens, max_output_tokens) in [
            (short, vec![1, 2, 3, 4], 5),
            (long, vec![5, 6, 7, 8], 8),
            (longer, vec![9, 10, 11, 12], 8),
        ] {
            core.receive(DirectRequest {
                tokens,
                max_output_tokens,
                uuid: Some(uuid),
                arrival_timestamp_ms: None,
                ..Default::default()
            });
        }

        let mut collector = crate::engine::trace::TraceCollector::default();
        // The prefill forward yields exactly one token per request; speculative bursts only
        // apply to decode forwards.
        let first = core.execute_pass(&mut collector, 0.0);
        assert_eq!(
            first
                .output_signals
                .iter()
                .map(|signal| signal.uuid)
                .collect::<Vec<_>>(),
            vec![short, long, longer]
        );
        assert!(core.running.iter().all(|req| req.output_len() == 1));

        // First decode pass: every request accepts the full three-token burst.
        let second = core.execute_pass(&mut collector, first.end_ms);
        let ordered = second
            .output_signals
            .iter()
            .map(|signal| (signal.uuid, signal.completed))
            .collect::<Vec<_>>();
        assert_eq!(
            ordered,
            vec![
                (short, false),
                (short, false),
                (short, false),
                (long, false),
                (long, false),
                (long, false),
                (longer, false),
                (longer, false),
                (longer, false),
            ]
        );
        assert!(core.running.iter().all(|req| req.output_len() == 4));
        assert_eq!(second.fpm.unwrap().num_decode_requests, 3);

        // Second decode pass: `short` has one token left and completes; the others burst again.
        let third = core.execute_pass(&mut collector, second.end_ms);
        assert_eq!(
            third
                .output_signals
                .iter()
                .map(|signal| (signal.uuid, signal.completed))
                .collect::<Vec<_>>(),
            vec![
                (short, true),
                (long, false),
                (long, false),
                (long, false),
                (longer, false),
                (longer, false),
                (longer, false),
            ]
        );
        assert_eq!(core.running.len(), 2);
        assert_eq!(core.running[0].uuid, long);
        assert_eq!(core.running[0].output_len(), 7);
        assert_eq!(core.running[1].uuid, longer);
        assert_eq!(core.running[1].output_len(), 7);

        let fourth = core.execute_pass(&mut collector, third.end_ms);
        assert_eq!(
            fourth
                .output_signals
                .iter()
                .map(|signal| (signal.uuid, signal.completed))
                .collect::<Vec<_>>(),
            vec![(long, true), (longer, true)],
        );
    }
}

mod forward_pass_metrics {
    use super::*;

    fn fpm_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(16)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn test_fpm_single_prefill_request() {
        let mut core = SglangCore::new(fpm_args());
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        assert_eq!(fpm.num_prefill_requests, 1);
        assert!(
            fpm.sum_prefill_tokens > 0,
            "prefill tokens should be computed"
        );
        // SGLang runs a formed prefill batch on its own; the request's first token comes out of
        // that prefill forward, so no decode work is scheduled in this pass.
        assert_eq!(fpm.num_decode_requests, 0);
        assert_eq!(fpm.num_queued_prefill, 0);
        assert_eq!(fpm.num_queued_decode, 0);
        assert!(fpm.wall_time_secs > 0.0);
    }

    #[test]
    fn test_fpm_prefill_and_decode_mixed_batch() {
        let mut core = SglangCore::new(fpm_args());

        // r1: 4-token prompt, 3 output tokens
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..4).collect(),
            max_output_tokens: 3,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

        // Pass 1: prefill r1
        let pass1 = core.execute_pass(&mut collector, 0.0);
        let fpm1 = pass1.fpm.expect("FPM should be present");
        assert_eq!(fpm1.num_prefill_requests, 1);

        // r2: arriving while r1 is decoding
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (100..104).collect(),
            max_output_tokens: 3,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        // Pass 2: SGLang runs the r2 prefill batch alone ("run prefill first if possible");
        // r1 waits for the next pass.
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.num_prefill_requests, 1, "r2 is prefilling");
        assert_eq!(
            fpm2.num_decode_requests, 0,
            "r1 does not decode during r2's prefill"
        );
        assert_eq!(fpm2.sum_decode_kv_tokens, 0);

        // Pass 3: no new prefill work, so the running batch (r1 + r2) decodes.
        let pass3 = core.execute_pass(&mut collector, pass2.end_ms);
        let fpm3 = pass3.fpm.expect("FPM should be present");
        assert_eq!(fpm3.num_prefill_requests, 0);
        assert_eq!(fpm3.num_decode_requests, 2, "r1 + r2 both in decode step");
        assert!(
            fpm3.sum_decode_kv_tokens > 0,
            "decode requests should have KV context"
        );
    }

    #[test]
    fn test_prefill_pass_stalls_running_decodes_and_charges_no_decode_time() {
        let mut core = SglangCore::new(fpm_args());
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        let r3 = Uuid::from_u128(3);
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..4).collect(),
            max_output_tokens: 3,
            output_token_ids: None,
            uuid: Some(r1),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(
            pass1
                .output_signals
                .iter()
                .map(|s| s.uuid)
                .collect::<Vec<_>>(),
            vec![r1],
            "the prefill forward yields r1's first token"
        );
        assert_eq!(core.running[0].output_len(), 1);

        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (100..104).collect(),
            max_output_tokens: 3,
            output_token_ids: None,
            uuid: Some(r2),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        // r3 finishes on the token its prefill forward samples.
        core.receive(DirectRequest {
            uuid: Some(r3),
            ..direct_request((200..204).collect(), 1)
        });
        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert_eq!(
            pass2
                .output_signals
                .iter()
                .map(|s| (s.uuid, s.completed))
                .collect::<Vec<_>>(),
            vec![(r2, false), (r3, true)],
            "only the freshly prefilled requests emit a token; r1 stalls"
        );
        assert_eq!(
            core.running.iter().map(|r| r.uuid).collect::<Vec<_>>(),
            vec![r1, r2],
            "FIFO order: r1 stays ahead of r2, and the finished r3 leaves the batch"
        );
        assert_eq!(core.running[0].output_len(), 1, "r1 did not decode");
        assert_eq!(core.running[1].output_len(), 1);
        let fpm2 = pass2.fpm.as_ref().expect("FPM should be present");
        assert_eq!(
            fpm2.num_decode_requests, 0,
            "no decode work in a prefill pass"
        );
        assert_eq!(fpm2.sum_decode_kv_tokens, 0);
        let decode_only = core.execute_pass(&mut collector, pass2.end_ms);
        assert_eq!(
            decode_only
                .output_signals
                .iter()
                .map(|s| s.uuid)
                .collect::<Vec<_>>(),
            vec![r1, r2],
            "with no prefill work the whole running batch decodes"
        );
        assert!(
            decode_only.end_ms - pass2.end_ms > 0.0,
            "a decode pass advances the clock"
        );
    }

    #[test]
    fn test_mocker_metrics_reports_prefix_cache_reuse() {
        let mut core = SglangCore::new(fpm_args());

        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..8).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_hit_tokens, 0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_total_tokens, 8);

        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert!(pass2.mocker_metrics.sglang_cache_hit_tokens > 0);
        assert!(
            pass2.mocker_metrics.sglang_cache_hit_tokens
                <= pass2.mocker_metrics.sglang_cache_total_tokens
        );
    }

    #[test]
    fn disabled_prefix_caching_does_not_reuse_or_retain_completed_prefixes() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(16)
            .block_size(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::engine::trace::TraceCollector::default();

        for request_id in [91_001_u128, 91_002_u128] {
            core.receive(DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: 0,
                uuid: Some(Uuid::from_u128(request_id)),
                ..Default::default()
            });
            let pass = core.execute_pass(&mut collector, 0.0);

            assert_eq!(pass.completed_requests, 1);
            assert_eq!(pass.mocker_metrics.sglang_cache_hit_tokens, 0);
            assert_eq!(pass.mocker_metrics.sglang_cache_total_tokens, 8);
            assert_eq!(pass.fpm.unwrap().num_prefill_requests, 1);
            assert!(core.is_empty());
            assert_eq!(
                core.kv_manager.cache().available_tokens(),
                core.kv_manager.cache().total_tokens(),
                "a completed request must not leave reusable radix pages behind"
            );
        }
    }

    #[test]
    fn fully_cached_zero_output_request_is_not_forward_pass_work() {
        let mut core = SglangCore::new(fpm_args());
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::engine::trace::TraceCollector::default();

        core.receive(DirectRequest {
            tokens: tokens.clone(),
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(90_004)),
            ..Default::default()
        });
        let seed_pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(seed_pass.completed_requests, 1);
        assert_eq!(seed_pass.committed_requests, vec![Uuid::from_u128(90_004)]);

        let uuid = core.receive(DirectRequest {
            tokens,
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(90_005)),
            ..Default::default()
        });
        let pass = core.execute_pass(&mut collector, seed_pass.end_ms);

        assert_eq!(pass.completed_requests, 1);
        assert!(pass.committed_requests.is_empty());
        assert_eq!(pass.mocker_metrics.sglang_cache_hit_tokens, 8);
        assert_eq!(pass.mocker_metrics.sglang_cache_total_tokens, 8);
        let fpm = pass.fpm.as_ref().unwrap();
        assert_eq!(fpm.num_prefill_requests, 0);
        assert_eq!(fpm.sum_prefill_tokens, 0);
        assert_eq!(fpm.num_decode_requests, 0);
        assert_eq!(fpm.sum_decode_kv_tokens, 0);
        assert!(matches!(
            pass.output_signals.as_slice(),
            [OutputSignal {
                uuid: signal_uuid,
                token_id: None,
                completed: true,
                rejected: false,
                ..
            }] if *signal_uuid == uuid
        ));
    }

    #[test]
    fn fully_cached_zero_output_request_is_not_committed_with_cold_sibling() {
        let mut core = SglangCore::new(fpm_args());
        let tokens = (0..8).collect::<Vec<_>>();
        let mut collector = crate::engine::trace::TraceCollector::default();

        core.receive(direct_request(tokens.clone(), 0));
        let seed_pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(seed_pass.completed_requests, 1);

        let cached = core.receive(direct_request(tokens, 0));
        let cold = core.receive(direct_request((100..108).collect(), 0));
        let pass = core.execute_pass(&mut collector, seed_pass.end_ms);

        assert_eq!(pass.completed_requests, 2);
        assert_eq!(pass.committed_requests, vec![cold]);
        assert!(pass.end_ms > seed_pass.end_ms);
        assert_eq!(pass.mocker_metrics.sglang_cache_hit_tokens, 8);
        assert_eq!(pass.mocker_metrics.sglang_cache_total_tokens, 16);
        let fpm = pass.fpm.as_ref().unwrap();
        assert_eq!(fpm.num_prefill_requests, 1);
        assert_eq!(fpm.sum_prefill_tokens, 8);
        assert_eq!(fpm.num_decode_requests, 0);
        for uuid in [cached, cold] {
            assert!(pass.output_signals.iter().any(|signal| {
                signal.uuid == uuid
                    && signal.token_id.is_none()
                    && signal.completed
                    && !signal.rejected
            }));
        }
        assert!(core.is_empty());
    }

    #[test]
    fn cache_only_completion_does_not_arm_prefill_decode_interval() {
        let mut args = fpm_args();
        args.prefill_decode_interval = 2;
        let mut core = SglangCore::new(args);
        let tokens = (0..8).collect::<Vec<_>>();
        core.receive(direct_request(tokens.clone(), 0));
        let seed = core.execute_hidden_pass(0.0);
        assert_eq!(seed.fpm.unwrap().num_prefill_requests, 1);

        // A completed request leaves two scheduler rounds, even without decode.
        for remaining in (0..2).rev() {
            assert!(!core.is_drained());
            assert!(!core.waiting_for_external_command());
            let idle = core.execute_hidden_pass(seed.end_ms);
            assert_eq!(idle.end_ms, seed.end_ms);
            assert_eq!(
                idle.same_timestamp_retry,
                crate::engine::generalized::SameTimestampRetry::Countdown { remaining }
            );
        }
        assert!(core.is_drained());

        core.receive(direct_request(tokens, 0));
        let cached = core.execute_hidden_pass(seed.end_ms);
        assert_eq!(cached.completed_requests, 1);
        assert_eq!(cached.fpm.unwrap().num_prefill_requests, 0);
        assert!(core.is_drained(), "cache bookkeeping must not arm EXTEND");

        core.receive(direct_request((100..108).collect(), 1));
        let fresh = core.execute_hidden_pass(cached.end_ms);
        assert_eq!(fresh.fpm.unwrap().num_prefill_requests, 1);
    }

    #[test]
    fn test_fpm_empty_pass_is_zeroed() {
        let mut core = SglangCore::new(fpm_args());

        // Submit and fully drain a request first so the core isn't empty
        // (empty core blocks in receive_requests in live mode, but
        // execute_pass_internal works fine on an empty core).
        let pass = core.execute_hidden_pass(0.0);
        let fpm = pass.fpm.expect("FPM should be present even for empty pass");

        assert_eq!(fpm.num_prefill_requests, 0);
        assert_eq!(fpm.num_decode_requests, 0);
        assert_eq!(fpm.num_queued_prefill, 0);
        assert_eq!(fpm.num_queued_decode, 0);
        assert_eq!(fpm.sum_prefill_tokens, 0);
        assert_eq!(fpm.sum_decode_kv_tokens, 0);
    }

    #[test]
    fn test_fpm_queued_requests() {
        // Very limited KV to force queuing.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(8))
            .max_num_seqs(Some(2))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

        // Two 8-token requests but limited KV
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..8).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (100..108).collect(),
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        let total_scheduled = fpm.num_prefill_requests + fpm.num_decode_requests;
        assert!(
            total_scheduled >= 1,
            "at least one request should be scheduled"
        );
        // With tight KV, the second request should be queued.
        let total_queued = fpm.num_queued_prefill + fpm.num_queued_decode;
        assert!(
            total_queued >= 1,
            "at least one request should be queued, got {total_queued}"
        );
    }

    #[test]
    fn test_fpm_var_prefill_length_with_multiple_requests() {
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

        // Two prefill requests with different prompt lengths
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..4).collect(), // prompt_len = 4
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (100..112).collect(), // prompt_len = 12
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        let fpm = pass.fpm.expect("FPM should be present");

        assert_eq!(fpm.num_prefill_requests, 2);
        // Population variance of [4, 12]: mean=8, var=((4-8)^2+(12-8)^2)/2 = 16
        assert!(
            (fpm.var_prefill_length - 16.0).abs() < 1e-6,
            "expected var=16.0, got {}",
            fpm.var_prefill_length
        );
    }

    #[test]
    fn test_fpm_chunked_prefill_reports_chunk_not_full_prompt() {
        // With chunked_prefill_size=8 and a 16-token prompt, the request
        // should be chunked. Each pass should report only the chunk size
        // in sum_prefill_tokens, not the full prompt length.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(32)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(8),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);

        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..16).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut collector = crate::engine::trace::TraceCollector::default();

        // Pass 1: first chunk
        let pass1 = core.execute_pass(&mut collector, 0.0);
        let fpm1 = pass1.fpm.expect("FPM should be present");
        assert_eq!(fpm1.num_prefill_requests, 1);
        assert!(
            fpm1.sum_prefill_tokens <= 8,
            "chunk should be at most 8 tokens, got {}",
            fpm1.sum_prefill_tokens
        );
        assert!(fpm1.sum_prefill_tokens > 0);

        // Pass 2: remaining chunk
        let pass2 = core.execute_pass(&mut collector, 1.0);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.num_prefill_requests, 1, "still prefilling");
        assert!(
            fpm2.sum_prefill_tokens <= 8,
            "second chunk should also be at most 8 tokens, got {}",
            fpm2.sum_prefill_tokens
        );

        // Total across both chunks should equal the full prompt length
        assert_eq!(
            fpm1.sum_prefill_tokens + fpm2.sum_prefill_tokens,
            16,
            "total prefill tokens across chunks should equal full prompt"
        );
    }

    #[test]
    fn test_retraction_requeues_at_back_and_reestimates_new_token_ratio() {
        // Tight KV (56 tokens): two running requests grow until a decode step must retract one.
        // SGLang appends the retracted request to the back of the FCFS waiting queue (behind a
        // request that has never run) and re-estimates `new_token_ratio` from the remaining batch
        // instead of resetting it to the initial ratio.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(14)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                // Low conservativeness so the second request is admitted while the first runs.
                schedule_conservativeness: Some(0.3),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let mut collector = crate::engine::trace::TraceCollector::default();
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        let fresh = Uuid::from_u128(3);
        let mut first = None;
        for (uuid, tokens, max_output_tokens) in [
            (r1, (0..4).collect::<Vec<_>>(), 30),
            (r2, (100..104).collect(), 30),
        ] {
            core.receive(DirectRequest {
                cache_salt: None,
                tokens,
                max_output_tokens,
                output_token_ids: None,
                uuid: Some(uuid),
                arrival_timestamp_ms: None,
                session_id: None,
                images: Vec::new(),
                kv_hint: None,
            });
            if uuid == r1 {
                first = Some(core.execute_pass(&mut collector, 0.0));
            }
        }
        // Submit r2 after r1's prefill so its admission uses the running reserve
        // ratio. Queueing both initially would mark the running batch full.
        let first = first.unwrap();
        assert_eq!(core.running.len(), 1);
        // A request that does not fit waits in the queue ahead of any retraction.
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (200..232).collect(),
            max_output_tokens: 40,
            output_token_ids: None,
            uuid: Some(fresh),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        let mut now = first.end_ms;
        let mut retracted_seen = false;
        let mut saw_two_running = false;
        for _ in 0..40 {
            let pass = core.execute_pass(&mut collector, now);
            now = pass.end_ms + 1.0;
            saw_two_running |= core.running.len() == 2;
            if let Some(pos) = core.waiting.iter().position(|req| req.output_len() > 0) {
                assert!(
                    saw_two_running,
                    "test setup: both requests should have run together"
                );
                retracted_seen = true;
                let fresh_pos = core
                    .waiting
                    .iter()
                    .position(|req| req.uuid == fresh)
                    .expect("fresh request still waiting");
                assert!(
                    fresh_pos < pos,
                    "retracted request must queue behind the never-run request"
                );
                let remaining = core.running.len();
                assert_eq!(remaining, 1);
                let decoded: usize = core.running.iter().map(|req| req.output_len()).sum();
                let max_new: usize = core.running.iter().map(|req| req.max_output_tokens).sum();
                let expected =
                    ((decoded as f64 + 20.0 * remaining as f64) / (max_new as f64 + 1.0)).min(1.0);
                assert!(
                    (core.new_token_ratio - expected).abs() < 1e-9,
                    "ratio {} should be re-estimated to {expected}",
                    core.new_token_ratio
                );
                assert!(
                    (expected - core.config.init_new_token_ratio).abs() > 1e-9,
                    "test setup: the estimate must differ from the initial ratio"
                );
                break;
            }
        }
        assert!(
            retracted_seen,
            "expected a decode retraction under KV pressure"
        );
    }

    fn chunk6_args(num_gpu_blocks: usize) -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(1)
            .num_gpu_blocks(num_gpu_blocks)
            .max_num_batched_tokens(Some(6))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(1),
                chunked_prefill_size: Some(6),
                max_prefill_tokens: Some(6),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    /// Prefill `tokens` as request `id` with one output token and run passes until it has finished,
    /// so its full sequence is cached.
    fn finish_and_cache(core: &mut SglangCore, id: u128, tokens: Vec<u32>, now: &mut f64) {
        core.receive(DirectRequest {
            cache_salt: None,
            tokens,
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(id)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let mut collector = crate::engine::trace::TraceCollector::default();
        for _ in 0..16 {
            let pass = core.execute_pass(&mut collector, *now);
            *now = pass.end_ms + 1.0;
            if core.running.is_empty() && core.waiting.is_empty() {
                return;
            }
        }
        panic!("request {id} did not finish");
    }

    #[test]
    fn test_first_admission_matches_whole_prompt_once() {
        // r1 (12 tokens) finishes and is cached. r2 shares r1's first 10 tokens: SGLang matches
        // the whole prompt once at admission (`extend_input_len` = 4 <= chunk 6), so r2 is a single
        // non-chunked 4-token prefill rather than a 6-token chunk plus continuations.
        let mut core = SglangCore::new(chunk6_args(256));
        let mut now = 0.0;
        finish_and_cache(&mut core, 1, (0..12).collect(), &mut now);

        let mut r2: Vec<u32> = (0..10).collect();
        r2.extend([100, 101, 102, 103]);
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: r2,
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, now);
        let fpm = pass.fpm.expect("FPM should be present");
        assert_eq!(fpm.num_prefill_requests, 1);
        assert_eq!(
            fpm.sum_prefill_tokens, 4,
            "only the four uncached tokens are computed"
        );
        assert!(
            pass.end_ms > now,
            "a prefill pass that computes tokens takes time"
        );
        assert!(
            core.waiting.is_empty(),
            "the prefill completed in one pass, nothing is chunked"
        );
    }

    #[test]
    fn test_fully_cached_prompt_still_computes_its_last_token() {
        // SGLang caps the match at `input_len - 1`, so a duplicate prompt is a one-token prefill
        // rather than a zero-duration pass.
        let mut core = SglangCore::new(chunk6_args(256));
        let mut now = 0.0;
        finish_and_cache(&mut core, 1, (0..5).collect(), &mut now);
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..5).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let available_before = core.kv_manager.cache().available_tokens();
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, now);
        let fpm = pass.fpm.expect("FPM should be present");
        assert_eq!(fpm.num_prefill_requests, 1);
        assert_eq!(fpm.sum_prefill_tokens, 1);
        assert!(pass.end_ms > now);
        // The allocator honours the same cap (see `allocation_match_is_capped_like_sglang` for the
        // lease-level state): the fifth prompt token gets a slot at admission, then deduplicates
        // against the cached page. The sampled output has no KV yet, so occupancy is unchanged.
        assert_eq!(core.running.len(), 1);
        assert_eq!(
            available_before - core.kv_manager.cache().available_tokens(),
            0
        );
    }

    #[test]
    fn test_small_request_is_co_admitted_behind_mostly_cached_long_request() {
        // r2 = 20 cached tokens + 4 new, r3 = 2 fresh tokens. Only r2's 4 uncached tokens count
        // against the 6-token chunk budget, so r3 fits in the same prefill batch.
        let mut core = SglangCore::new(chunk6_args(256));
        let mut now = 0.0;
        finish_and_cache(&mut core, 1, (0..20).collect(), &mut now);

        let mut r2: Vec<u32> = (0..20).collect();
        r2.extend([100, 101, 102, 103]);
        for (id, tokens) in [(2u128, r2), (3u128, vec![200, 201])] {
            core.receive(DirectRequest {
                cache_salt: None,
                tokens,
                max_output_tokens: 2,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(id)),
                arrival_timestamp_ms: None,
                session_id: None,
                images: Vec::new(),
                kv_hint: None,
            });
        }
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, now);
        let fpm = pass.fpm.expect("FPM should be present");
        assert_eq!(
            fpm.num_prefill_requests, 2,
            "r2 and r3 share the prefill batch"
        );
        assert_eq!(fpm.sum_prefill_tokens, 6);
        assert_eq!(core.running.len(), 2);
        assert!(core.waiting.is_empty());
    }

    #[test]
    fn test_chunked_cold_prompt_reports_no_cache_reuse() {
        // A 12-token prompt with nothing cached is prefilled in two 6-token chunks. The
        // continuation's KV context is the request's own first chunk, which is not a cache hit:
        // neither the admission event nor the cache-hit metric may count it.
        let mut core = SglangCore::new(chunk6_args(256));
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..12).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(7)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_hit_tokens, 0);
        assert_eq!(pass1.mocker_metrics.sglang_cache_total_tokens, 6);
        assert_eq!(pass1.admissions.len(), 1);
        assert_eq!(pass1.admissions[0].reused_input_tokens, 0);
        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert!(
            pass2.admissions.is_empty(),
            "a continuation chunk is not a new admission"
        );
        assert_eq!(pass2.mocker_metrics.sglang_cache_hit_tokens, 0);
        assert_eq!(pass2.mocker_metrics.sglang_cache_total_tokens, 6);
        let fpm2 = pass2.fpm.expect("FPM should be present");
        assert_eq!(fpm2.sum_prefill_tokens, 6);
        assert_eq!(
            fpm2.sum_prefill_kv_tokens, 6,
            "the first chunk is KV context for the continuation's forward"
        );
    }

    fn three_page_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(3)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(16),
                ..Default::default()
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn test_single_token_requests_finish_in_the_prefill_pass_without_kv_slots() {
        // Two 4-token prompts with `max_output_tokens = 1` need five pages of admission headroom. SGLang finishes
        // them in `process_batch_result_prefill` → `check_finished()`: the sampled token never gets
        // a KV slot, so there is no decode forward and no memory pressure at all.
        let mut args = three_page_args();
        args.num_gpu_blocks = 5;
        let mut core = SglangCore::new(args);
        let r1 = Uuid::from_u128(1);
        let r2 = Uuid::from_u128(2);
        for (uuid, tokens) in [(r1, (0..4).collect::<Vec<_>>()), (r2, (100..104).collect())] {
            core.receive(DirectRequest {
                cache_salt: None,
                tokens,
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(uuid),
                arrival_timestamp_ms: None,
                session_id: None,
                images: Vec::new(),
                kv_hint: None,
            });
        }
        let available_before = core.kv_manager.cache().available_tokens();
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass.admissions.len(), 2);
        assert!(pass.pressure_events.is_empty());
        assert_eq!(
            pass.output_signals
                .iter()
                .map(|s| (s.uuid, s.completed))
                .collect::<Vec<_>>(),
            vec![(r1, true), (r2, true)]
        );
        assert!(core.running.is_empty() && core.waiting.is_empty());
        // Both prompts are cached (two pages held by the tree); no page was spent on outputs.
        assert_eq!(
            available_before - core.kv_manager.cache().available_tokens(),
            8
        );
    }

    #[test]
    fn test_prefill_pass_gives_continuing_requests_their_first_token_without_retraction() {
        // Same prompts but the requests keep decoding (`max_output_tokens = 2`). The prefill pass
        // still emits exactly one first token per request, never retracts (SGLang's decode memory
        // check runs in `update_running_batch` on the next decode pass), and the following decode
        // pass completes both.
        let mut core = SglangCore::new(three_page_args());
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..4).collect(),
            max_output_tokens: 2,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        let mut collector = crate::engine::trace::TraceCollector::default();
        let pass1 = core.execute_pass(&mut collector, 0.0);
        assert_eq!(pass1.admissions.len(), 1);
        assert!(
            pass1.pressure_events.is_empty(),
            "no retraction in a prefill pass"
        );
        assert_eq!(
            pass1
                .output_signals
                .iter()
                .map(|s| (s.uuid, s.completed))
                .collect::<Vec<_>>(),
            vec![(Uuid::from_u128(1), false)]
        );
        assert_eq!(core.running.len(), 1);
        assert_eq!(core.running[0].output_len(), 1);
        let fpm1 = pass1.fpm.as_ref().expect("FPM should be present");
        assert_eq!(
            fpm1.num_decode_requests, 0,
            "a prefill pass schedules no decode work"
        );

        let pass2 = core.execute_pass(&mut collector, pass1.end_ms);
        assert!(pass2.pressure_events.is_empty());
        assert_eq!(
            pass2
                .output_signals
                .iter()
                .map(|s| (s.uuid, s.completed))
                .collect::<Vec<_>>(),
            vec![(Uuid::from_u128(1), true)]
        );
        assert!(
            pass2.end_ms > pass1.end_ms,
            "the decode pass is the one that takes time"
        );
    }

    #[test]
    fn test_fpm_retracted_decode_becomes_queued_decode() {
        // Very tight KV to force decode retraction. Fill the KV with running
        // requests, then the decode step should retract some, and those should
        // appear as queued decodes in the next pass's FPM.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(6) // 24 tokens — very tight
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        let mut collector = crate::engine::trace::TraceCollector::default();

        // Two requests with 4-token prompts and long outputs to fill KV
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (0..4).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (100..104).collect(),
            max_output_tokens: 20,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(2)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        // Run several passes to build up KV pressure
        for i in 0..4 {
            core.execute_pass(&mut collector, i as f64);
        }

        // Add a third request to increase memory pressure
        core.receive(DirectRequest {
            cache_salt: None,
            tokens: (200..212).collect(), // 12 tokens
            max_output_tokens: 10,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(3)),
            arrival_timestamp_ms: None,
            session_id: None,
            images: Vec::new(),
            kv_hint: None,
        });

        // Run more passes — at some point retraction should occur
        let mut saw_queued_decode = false;
        for i in 4..10 {
            let pass = core.execute_pass(&mut collector, i as f64);
            let fpm = pass.fpm.expect("FPM should be present");
            if fpm.num_queued_decode > 0 {
                saw_queued_decode = true;
                assert!(
                    fpm.sum_queued_decode_kv_tokens > 0,
                    "retracted decode should have KV context"
                );
                break;
            }
        }

        // If retraction didn't happen (KV was sufficient), that's also valid —
        // just verify we always get Some(fpm).
        if !saw_queued_decode {
            // Verify the requests completed or are still running with valid FPM
            let pass = core.execute_hidden_pass(10.0);
            assert!(pass.fpm.is_some(), "FPM should always be present");
        }
    }
}

#[test]
fn prefill_provider_validation_receives_actual_chunk_geometry() {
    use std::sync::{Arc, Mutex};
    struct GeometryTiming(Arc<Mutex<Vec<Vec<(usize, usize)>>>>);
    impl crate::engine::TimingModel for GeometryTiming {
        fn validate_prefill_batch(&self, requests: &[(usize, usize)]) -> anyhow::Result<()> {
            self.0.lock().unwrap().push(requests.to_vec());
            anyhow::ensure!(
                requests.windows(2).all(|pair| pair[0] == pair[1]),
                "heterogeneous geometry"
            );
            Ok(())
        }
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(0.6)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(0.6)
        }
    }
    let observed = Arc::new(Mutex::new(Vec::new()));
    let mut args = test_args(128, 4, 8);
    args.perf_model = crate::engine::common::perf_model::PerfModel::External {
        timing: Arc::new(GeometryTiming(Arc::clone(&observed))),
    }
    .into();
    let mut core = SglangCore::new(args);
    core.receive(direct_request((0..16).collect(), 1));
    core.try_execute_pass_internal(None, 0.0).unwrap();
    core.try_execute_pass_internal(None, 1.0).unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![vec![(8, 0)], vec![(8, 8)]]);

    let mut args = test_args(128, 4, 32);
    args.perf_model = crate::engine::common::perf_model::PerfModel::External {
        timing: Arc::new(GeometryTiming(Arc::clone(&observed))),
    }
    .into();
    let mut core = SglangCore::new(args);
    core.receive(direct_request((0..4).collect(), 1));
    core.receive(direct_request((100..112).collect(), 1));
    let error = core.try_execute_pass_internal(None, 0.0).unwrap_err();
    assert_eq!(error.root_cause().to_string(), "heterogeneous geometry");
    assert_eq!(
        observed.lock().unwrap().last().unwrap(),
        &vec![(4, 0), (12, 0)]
    );
}

/// Records each prefill query at the whole-forward FPM coordinates
/// `(batch, total prefill tokens, total KV read tokens)`. A mean-based call is
/// mapped as the FPM provider maps it.
#[derive(Default)]
struct FpmCoordinates(std::sync::Mutex<Vec<(usize, usize, usize)>>);

impl crate::engine::TimingModel for FpmCoordinates {
    fn predict_prefill_ms(&self, batch: usize, isl: usize, prefix: usize) -> anyhow::Result<f64> {
        self.0
            .lock()
            .unwrap()
            .push((batch, batch * (isl - prefix), batch * prefix));
        Ok(1.0)
    }
    fn predict_prefill_totals_ms(
        &self,
        batch: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> anyhow::Result<f64> {
        self.0
            .lock()
            .unwrap()
            .push((batch, new_tokens, prefix_tokens));
        Ok(1.0)
    }
    fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        Ok(1.0)
    }
}

fn fpm_coordinate_core(
    chunked_prefill_size: usize,
) -> (SglangCore, std::sync::Arc<FpmCoordinates>) {
    let timing = std::sync::Arc::new(FpmCoordinates::default());
    let mut args = test_args(128, 1, chunked_prefill_size);
    args.perf_model = crate::engine::common::perf_model::PerfModel::External {
        timing: timing.clone(),
    }
    .into();
    (SglangCore::new(args), timing)
}

#[test]
fn prefill_timing_queries_scheduled_totals_after_a_chunk() {
    let (mut core, timing) = fpm_coordinate_core(8);
    core.receive(direct_request((0..12).collect(), 1));
    core.receive(direct_request((100..102).collect(), 1));
    core.receive(direct_request((200..202).collect(), 1));
    let first = core.try_execute_pass_internal(None, 0.0).unwrap();
    let second = core.try_execute_pass_internal(None, first.end_ms).unwrap();
    // The continuation reads its 8-token chunk beside two 2-token prompts.
    // Integer means (16 / 3, 8 / 3) would have queried (3, 9, 6), one token
    // beyond the chunk budget.
    assert_eq!(*timing.0.lock().unwrap(), vec![(1, 8, 0), (3, 8, 8)]);
    let fpm = second.fpm.unwrap();
    assert_eq!(
        (
            fpm.num_prefill_requests,
            fpm.sum_prefill_tokens,
            fpm.sum_prefill_kv_tokens
        ),
        (3, 8, 8)
    );
}

#[test]
fn prefill_validation_excludes_cached_zero_output_members() {
    struct HomogeneousOnly;
    impl crate::engine::TimingModel for HomogeneousOnly {
        fn validate_prefill_batch(&self, requests: &[(usize, usize)]) -> anyhow::Result<()> {
            anyhow::ensure!(
                requests.windows(2).all(|pair| pair[0] == pair[1]),
                "heterogeneous geometry {requests:?}"
            );
            Ok(())
        }
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(1.0)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(1.0)
        }
    }
    let mut args = test_args(128, 1, 64);
    args.perf_model = crate::engine::common::perf_model::PerfModel::External {
        timing: std::sync::Arc::new(HomogeneousOnly),
    }
    .into();
    let mut core = SglangCore::new(args);
    core.receive(direct_request((0..8).collect(), 1));
    let first = core.try_execute_pass_internal(None, 0.0).unwrap();
    // The cached zero-output request computes nothing, so only the 4-token
    // prompt is validated, as it is the only member priced.
    core.receive(direct_request((0..8).collect(), 0));
    core.receive(direct_request((100..104).collect(), 1));
    core.try_execute_pass_internal(None, first.end_ms).unwrap();
}

#[test]
fn prefill_timing_excludes_cached_zero_output_members() {
    let (mut core, timing) = fpm_coordinate_core(64);
    core.receive(direct_request((0..8).collect(), 1));
    let first = core.try_execute_pass_internal(None, 0.0).unwrap();
    // The zero-output request reuses all 8 cached tokens and computes nothing
    // beside a 4-token prompt.
    core.receive(direct_request((0..8).collect(), 0));
    core.receive(direct_request((100..104).collect(), 1));
    let second = core.try_execute_pass_internal(None, first.end_ms).unwrap();
    assert_eq!(*timing.0.lock().unwrap(), vec![(1, 8, 0), (1, 4, 0)]);
    let fpm = second.fpm.unwrap();
    assert_eq!(
        (
            fpm.num_prefill_requests,
            fpm.sum_prefill_tokens,
            fpm.sum_prefill_kv_tokens
        ),
        (1, 4, 0)
    );
}

#[test]
fn indexed_ids_follow_chunk_cancel_completion_and_reuse() {
    let mut core = SglangCore::new(test_args(128, 4, 4));
    let id = Uuid::from_u128(700_001);
    let request = || {
        let mut request = direct_request((0..12).collect(), 2);
        request.uuid = Some(id);
        request
    };
    core.apply_command(SchedulerCommand::Submit(request()))
        .unwrap();
    assert!(
        core.apply_command(SchedulerCommand::Submit(request()))
            .is_err()
    );
    let first = core.try_execute_hidden_pass(0.0).unwrap();
    assert!(first.output_signals.is_empty()); // still a partial prefill
    assert!(
        core.apply_command(SchedulerCommand::Submit(request()))
            .is_err()
    );
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    core.apply_command(SchedulerCommand::Submit(request()))
        .unwrap();
    let mut now = first.end_ms;
    for _ in 0..20 {
        let pass = core.try_execute_hidden_pass(now).unwrap();
        now = pass.end_ms;
        if core.is_empty() {
            break;
        }
    }
    assert!(core.is_empty());
    core.apply_command(SchedulerCommand::Submit(request()))
        .unwrap();
}

mod host_loop_passes {
    use std::sync::Arc;

    use super::*;
    use crate::engine::TtftMilestone;

    struct FixedTiming {
        prefill_ms: f64,
        decode_ms: f64,
    }

    impl crate::engine::TimingModel for FixedTiming {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(self.prefill_ms)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(self.decode_ms)
        }
    }

    fn core(host_loop: bool) -> SglangCore {
        let mut args = test_args(128, 1, 8192);
        args.sglang.as_mut().unwrap().host_loop = host_loop;
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming {
                prefill_ms: 20.0,
                decode_ms: 4.0,
            }),
        }
        .into();
        SglangCore::new(args)
    }

    /// A decode forward costs its batch size in milliseconds, so the charged
    /// duration names the rows the provider was asked to run.
    struct BatchTiming;

    impl crate::engine::TimingModel for BatchTiming {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(7.0)
        }
        fn predict_decode_ms(
            &self,
            batch_size: usize,
            _: usize,
            _: usize,
            _: usize,
        ) -> anyhow::Result<f64> {
            Ok(batch_size as f64)
        }
    }

    #[test]
    fn decode_metrics_exclude_aggregated_rows_finishing_inflight() {
        // Two four-token prompts prefill together; a request that finishes on its
        // first token is excluded from the next aggregated decode under the host loop.
        for (host_loop, outputs, rows, kv_tokens) in [
            (false, [1, 3], 1, 5),
            (true, [1, 3], 1, 5),
            (true, [1, 1], 0, 0),
        ] {
            let mut args = test_args(128, 1, 128);
            args.sglang.as_mut().unwrap().host_loop = host_loop;
            args.perf_model = crate::engine::common::perf_model::PerfModel::External {
                timing: Arc::new(BatchTiming),
            }
            .into();
            let mut core = SglangCore::new(args);
            core.receive(direct_request((0..4).collect(), outputs[0]));
            core.receive(direct_request((100..104).collect(), outputs[1]));
            let prefill = core.execute_hidden_pass(0.0);
            assert_eq!(prefill.fpm.as_ref().unwrap().num_prefill_requests, 2);
            let fpm = core.execute_hidden_pass(prefill.end_ms).fpm.unwrap();
            assert_eq!(
                (fpm.num_decode_requests, fpm.sum_decode_kv_tokens),
                (rows, kv_tokens),
                "host_loop={host_loop} outputs={outputs:?}"
            );
            assert_eq!(
                fpm.wall_time_secs * 1000.0,
                f64::from(rows),
                "the charged forward ran {rows} rows (host_loop={host_loop})"
            );
        }
    }

    fn stage(pass: &EnginePassResult, wanted: TtftMilestone) -> Vec<(Uuid, f64)> {
        pass.lifecycle_events
            .iter()
            .filter_map(|event| match *event {
                SchedulerLifecycleEvent::TtftMilestone {
                    request_id,
                    stage,
                    at_ms,
                } if stage == wanted => Some((request_id, at_ms)),
                _ => None,
            })
            .collect()
    }

    /// Token observation times, driving passes back to back until the core drains.
    fn token_times(core: &mut SglangCore) -> Vec<f64> {
        let mut now_ms = 0.0;
        let mut times = Vec::new();
        for _ in 0..32 {
            if core.is_drained() {
                return times;
            }
            let pass = core.execute_hidden_pass(now_ms);
            times.extend(
                pass.output_signals
                    .iter()
                    .filter(|signal| signal.token_id.is_some())
                    .map(|_| pass.end_ms),
            );
            now_ms = pass.end_ms;
        }
        panic!("core did not drain");
    }

    #[test]
    fn one_request_is_observed_one_iteration_after_its_forward() {
        let mut core = core(true);
        let uuid = core.receive(direct_request((0..16).collect(), 1));

        // Iteration 1: the request is received and selected at 0 and its forward runs
        // 0..20; the free loop returns at once with nothing to observe yet.
        let first = core.execute_hidden_pass(0.0);
        assert_eq!(first.end_ms, 0.0);
        assert!(first.output_signals.is_empty());
        assert_eq!(stage(&first, TtftMilestone::Received), vec![(uuid, 0.0)]);
        assert_eq!(stage(&first, TtftMilestone::Selected), vec![(uuid, 0.0)]);
        assert_eq!(
            stage(&first, TtftMilestone::PrefillComplete),
            vec![(uuid, 20.0)]
        );
        assert!(!core.is_drained());

        // Iteration 2 filters the row finishing by length. No decode launches;
        // the first forward's result is observed at 20 and the core drains.
        let second = core.execute_hidden_pass(first.end_ms);
        assert_eq!(second.end_ms, 20.0);
        assert_eq!(second.output_signals.len(), 1);
        assert!(second.output_signals[0].completed);
        assert_eq!(second.output_signals[0].uuid, uuid);
        assert!(core.is_drained());
    }

    #[test]
    fn requests_arriving_during_an_iteration_are_received_at_the_next_one() {
        let mut core = core(true);
        let first = core.receive(direct_request((0..16).collect(), 1));
        let pass = core.execute_hidden_pass(0.0);
        assert_eq!(pass.end_ms, 0.0);

        // Delivered after iteration 1 drained its inbox: received by iteration 2.
        let second = core.receive(direct_request((100..116).collect(), 1));
        let pass = core.execute_hidden_pass(pass.end_ms);
        assert_eq!(stage(&pass, TtftMilestone::Received), vec![(second, 0.0)]);
        assert_eq!(stage(&pass, TtftMilestone::Selected), vec![(second, 0.0)]);
        // An EXTEND launch does not wait for the running forward; the loop then
        // reaches the first request's result when that forward ends (20).
        assert_eq!(pass.end_ms, 20.0);
        assert_eq!(
            pass.output_signals
                .iter()
                .map(|signal| signal.uuid)
                .collect::<Vec<_>>(),
            vec![first]
        );
        // The second forward queues behind the first on the GPU: 20..40.
        assert_eq!(
            stage(&pass, TtftMilestone::PrefillComplete),
            vec![(second, 40.0)]
        );
    }

    #[test]
    fn a_free_host_loop_reproduces_the_legacy_token_times() {
        let mut legacy = core(false);
        legacy.receive(direct_request((0..16).collect(), 4));
        let mut host = core(true);
        host.receive(direct_request((0..16).collect(), 4));

        let expected = vec![20.0, 24.0, 28.0, 32.0];
        assert_eq!(token_times(&mut legacy), expected);
        assert_eq!(token_times(&mut host), expected);
    }

    #[test]
    fn prefix_cache_commits_become_visible_when_the_forward_is_observed() {
        let mut core = core(true);
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        core.receive(direct_request(prompt.clone(), 2));
        let first = core.execute_hidden_pass(0.0);

        // Same prompt, selected while the first forward is still unobserved: SGLang
        // caches a full prefill in `maybe_cache_unfinished_req`, one iteration later.
        let second_id = core.receive(direct_request(prompt.clone(), 2));
        let second = core.execute_hidden_pass(first.end_ms);
        assert_eq!(second.admissions[0].uuid, second_id);
        assert_eq!(second.admissions[0].reused_input_tokens, 0);

        // After that observation the prefix is in the radix cache.
        let third_id = core.receive(direct_request(prompt, 2));
        let third = core.execute_hidden_pass(second.end_ms);
        assert_eq!(third.admissions[0].uuid, third_id);
        assert!(third.admissions[0].reused_input_tokens > 0);
    }

    #[test]
    fn prefix_cache_commits_reach_their_requests_in_any_running_order() {
        let mut core = core(true);
        // Prompts of different lengths, so a commit applied to the wrong request shows.
        core.receive(direct_request(vec![1, 2, 3, 4], 8));
        core.receive(direct_request(vec![5, 6, 7, 8, 9], 8));
        // The prefill prompt commits are in flight.
        let prefill = core.execute_hidden_pass(0.0);

        // Observe them with the batch in the reverse of the order they were recorded in.
        core.running.reverse();
        let before: Vec<usize> = core.running.iter().map(|r| r.cached_tokens()).collect();
        core.execute_hidden_pass(prefill.end_ms);
        let after: Vec<usize> = core.running.iter().map(|r| r.cached_tokens()).collect();
        assert_eq!(before, vec![0, 0]);
        assert_eq!(after, vec![5, 4]);
    }

    #[test]
    fn decode_prefix_stays_private_until_completion_is_observed() {
        // Four-token pages and a six-token prompt. The prompt enters the cache
        // one iteration later; decoded pages wait for the terminal observation.
        for (prefix_caching, expected) in [(true, [0, 4, 4, 4, 4, 4, 4, 4]), (false, [0; 8])] {
            let mut args = test_args(128, 4, 8192);
            args.enable_prefix_caching = prefix_caching;
            args.sglang.as_mut().unwrap().host_loop = true;
            args.perf_model = crate::engine::common::perf_model::PerfModel::External {
                timing: Arc::new(FixedTiming {
                    prefill_ms: 20.0,
                    decode_ms: 4.0,
                }),
            }
            .into();
            let mut core = SglangCore::new(args);
            core.receive(direct_request((1..=6).collect(), 8));
            let mut now_ms = 0.0;
            let cached = expected.map(|_| {
                now_ms = core.execute_hidden_pass(now_ms).end_ms;
                core.running[0].cached_tokens()
            });
            assert_eq!(cached, expected, "prefix_caching={prefix_caching}");
            let sequence = core.running[0].sequence_tokens.clone();
            let mut completions = 0;
            for _ in 0..4 {
                let pass = core.execute_hidden_pass(now_ms);
                now_ms = pass.end_ms;
                completions += pass
                    .output_signals
                    .iter()
                    .filter(|signal| signal.completed)
                    .count();
                if core.is_drained() {
                    break;
                }
            }
            assert_eq!(completions, 1);
            assert!(core.is_drained());
            assert_eq!(
                core.kv_manager.cache().prefix_match_len(&sequence),
                if prefix_caching { 12 } else { 0 }
            );
        }
    }

    #[rstest::rstest]
    fn a_cancelled_request_keeps_its_kv_private_until_the_forward_is_observed(
        #[values(1, 8)] output_tokens: usize,
        #[values(1, 4)] block_size: usize,
        #[values(false, true)] prefix_caching: bool,
    ) {
        let mut args = test_args(128, block_size, 8192);
        args.enable_prefix_caching = prefix_caching;
        args.sglang.as_mut().unwrap().host_loop = true;
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming {
                prefill_ms: 20.0,
                decode_ms: 4.0,
            }),
        }
        .into();
        let mut core = SglangCore::new_with_kv_capture(args, 0);
        let prompt: Vec<u32> = (0..16).collect();
        let uuid = core.receive(direct_request(prompt.clone(), output_tokens));
        let first = core.execute_hidden_pass(0.0);
        assert_eq!(first.end_ms, 0.0);
        assert!(first.output_signals.is_empty());
        assert!(first.kv_events.is_empty());
        assert!(core.drain_kv_events().is_empty());
        let available_in_flight = core.kv_manager.cache().available_tokens();
        let materialized_pages = core.running[0].materialized_tokens / block_size;

        // SGLang marks a running abort for result processing, after synchronization.
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
                .unwrap(),
            SchedulerCommandResult::Applied
        );
        assert_eq!(core.kv_manager.cache().prefix_match_len(&prompt), 0);
        assert!(core.drain_kv_events().is_empty());
        assert_eq!(
            core.kv_manager.cache().available_tokens(),
            available_in_flight
        );
        assert!(!core.is_drained());
        assert_eq!(
            core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
                .unwrap(),
            SchedulerCommandResult::Noop
        );

        // Selection precedes observation, so the next follower cannot reuse future KV.
        let cold_follower = core.receive(direct_request(prompt.clone(), 2));
        let observed = core.execute_hidden_pass(first.end_ms);
        assert_eq!(observed.end_ms, 20.0);
        assert!(observed.output_signals.is_empty());
        assert_eq!(observed.admissions[0].uuid, cold_follower);
        assert_eq!(observed.admissions[0].reused_input_tokens, 0);
        assert_eq!(
            core.kv_manager.cache().prefix_match_len(&prompt),
            if prefix_caching { prompt.len() } else { 0 }
        );
        assert_eq!(
            stored_hashes(&observed.kv_events).len(),
            if prefix_caching {
                materialized_pages
            } else {
                0
            }
        );

        let warm_follower = core.receive(direct_request(prompt.clone(), 2));
        let warm = core.execute_hidden_pass(observed.end_ms);
        assert_eq!(warm.admissions[0].uuid, warm_follower);
        assert_eq!(
            warm.admissions[0].reused_input_tokens,
            if prefix_caching {
                (prompt.len() - 1) / block_size * block_size
            } else {
                0
            }
        );
        assert!(warm.output_signals.iter().all(|signal| signal.uuid != uuid));
        let mut now_ms = warm.end_ms;
        for _ in 0..16 {
            if core.is_drained() {
                break;
            }
            let pass = core.execute_hidden_pass(now_ms);
            assert!(pass.output_signals.iter().all(|signal| signal.uuid != uuid));
            now_ms = pass.end_ms;
        }
        assert!(core.is_drained());
        let total_tokens = core.kv_manager.cache().total_tokens();
        core.kv_manager.evict(total_tokens);
        assert_eq!(core.kv_manager.cache().available_tokens(), total_tokens);
        core.drain_kv_events();
        assert!(core.execute_hidden_pass(now_ms).output_signals.is_empty());
        assert!(core.drain_kv_events().is_empty());
    }

    #[test]
    fn cancelled_decode_caches_and_releases_kv_when_an_idle_pass_observes_it() {
        let mut args = test_args(128, 1, 8192);
        args.sglang.as_mut().unwrap().host_loop = true;
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming {
                prefill_ms: 20.0,
                decode_ms: 4.0,
            }),
        }
        .into();
        let mut core = SglangCore::new_with_kv_capture(args, 0);
        let prompt: Vec<u32> = (0..8).collect();
        let uuid = core.receive(direct_request(prompt.clone(), 50));
        let mut now_ms = 0.0;
        for _ in 0..6 {
            now_ms = core.execute_hidden_pass(now_ms).end_ms;
        }
        let sequence = core.running[0]
            .sequence_prefix(core.running[0].materialized_tokens)
            .to_vec();
        assert!(sequence.len() > prompt.len());
        assert_eq!(
            core.kv_manager.cache().prefix_match_len(&sequence),
            prompt.len()
        );
        let available_in_flight = core.kv_manager.cache().available_tokens();
        core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
            .unwrap();
        assert_eq!(
            core.kv_manager.cache().prefix_match_len(&sequence),
            prompt.len()
        );
        assert_eq!(
            core.kv_manager.cache().available_tokens(),
            available_in_flight
        );
        assert!(core.drain_kv_events().is_empty());
        assert!(!core.is_drained());

        let observed = core.execute_hidden_pass(now_ms);
        assert_eq!(observed.end_ms, now_ms + 4.0);
        assert!(observed.output_signals.is_empty());
        assert_eq!(
            core.kv_manager.cache().prefix_match_len(&sequence),
            sequence.len()
        );
        assert_eq!(
            stored_hashes(&observed.kv_events).len(),
            sequence.len() - prompt.len()
        );
        assert!(core.is_drained());
        let idle = core.execute_hidden_pass(observed.end_ms);
        assert_eq!(idle.end_ms, observed.end_ms);
        assert!(idle.output_signals.is_empty());
        assert!(idle.kv_events.is_empty());
        let total_tokens = core.kv_manager.cache().total_tokens();
        core.kv_manager.evict(total_tokens);
        assert_eq!(core.kv_manager.cache().available_tokens(), total_tokens);
    }

    #[test]
    fn deferred_cancellation_cleanup_preserves_a_reused_request_id() {
        let mut core = core(true);
        let uuid = core.receive(direct_request((0..16).collect(), 1));
        let launched = core.execute_hidden_pass(0.0);
        core.apply_command(SchedulerCommand::CancelRequest { request_id: uuid })
            .unwrap();
        let mut replacement = direct_request((100..116).collect(), 2);
        replacement.uuid = Some(uuid);
        assert_eq!(core.receive(replacement), uuid);
        let observed = core.execute_hidden_pass(launched.end_ms);
        assert_eq!(observed.admissions[0].uuid, uuid);
        assert!(observed.output_signals.is_empty());
        assert_eq!(core.running.len(), 1);
        let mut now_ms = observed.end_ms;
        let mut output_tokens = 0;
        for _ in 0..8 {
            if core.is_drained() {
                break;
            }
            let pass = core.execute_hidden_pass(now_ms);
            output_tokens += pass
                .output_signals
                .iter()
                .filter(|signal| signal.token_id.is_some())
                .count();
            now_ms = pass.end_ms;
        }
        assert_eq!(output_tokens, 2);
        assert!(core.is_drained());
        let total_tokens = core.kv_manager.cache().total_tokens();
        core.kv_manager.evict(total_tokens);
        assert_eq!(core.kv_manager.cache().available_tokens(), total_tokens);
    }

    #[test]
    fn a_ghost_decode_row_takes_a_kv_slot_that_is_cached_with_the_request() {
        // Three pages of four tokens. The request finished with its first token in the
        // previous forward; `filter_batch` has not seen that yet, so `prepare_for_decode`
        // still allocates and commits its row's slot, and `release_kv_cache` caches the
        // committed KV page aligned when the finish is observed.
        let mut args = test_args(3, 4, 8192);
        args.sglang.as_mut().unwrap().host_loop = true;
        let config = SglangConfig::from_args(&args);
        let mut kv_manager = SglangKvManager::new(12, 4, KvEventPublishers::default(), 0);
        let prompt: Vec<u32> = (1..=8).collect();
        let alloc = kv_manager.allocate_for_request(&prompt).unwrap();
        let mut running = vec![SglangRequest {
            is_decode_handoff: false,
            retracted: false,
            uuid: Uuid::from_u128(1),
            sequence_tokens: prompt.iter().copied().chain([99]).collect(),
            prompt_len: 8,
            max_output_tokens: 1,
            planned_output_ids: None,
            kv_lease: alloc.lease,
            materialized_tokens: 8,
            allocated_tokens: 8,
            images: Vec::new(),
            pending_terminal: true,
        }];
        assert_eq!(kv_manager.cache().available_tokens(), 4);

        let step = decode::simulate_decode_step_with_sampler(
            &mut running,
            &mut kv_manager,
            &config,
            None,
            0.0,
        )
        .unwrap();
        assert!(
            step.output_signals.is_empty(),
            "a ghost row produces nothing"
        );
        assert_eq!(running[0].allocated_tokens, 12);
        assert_eq!(kv_manager.cache().available_tokens(), 0);

        // 9 committed tokens: the two full pages are cached, the tail page is freed.
        decode::cleanup_completed_request(&mut running[0], &mut kv_manager, 4);
        assert_eq!(kv_manager.cache().available_tokens(), 4);
        assert_eq!(kv_manager.cache().evictable_size, 8);
    }

    #[test]
    fn a_row_that_finished_on_its_first_token_reserves_no_page_in_the_prefill_step() {
        // Four pages of four tokens: two freshly prefilled rows own one each and a third
        // page is cached. The first row finishes with its first token and stays a ghost
        // member whose slot `prepare_for_decode` allocates next step; only the second
        // row's first-token slot is reserved now, so the cached page is not evicted.
        let mut args = test_args(4, 4, 8192);
        args.sglang.as_mut().unwrap().host_loop = true;
        let config = SglangConfig::from_args(&args);
        let mut kv_manager = SglangKvManager::new(16, 4, KvEventPublishers::default(), 0);
        let cached: Vec<u32> = (9..=12).collect();
        let cached_alloc = kv_manager.allocate_for_request(&cached).unwrap();
        kv_manager.finish(&cached, cached_alloc.lease);
        let finishing: Vec<u32> = (1..=4).collect();
        let finishing_alloc = kv_manager.allocate_for_request(&finishing).unwrap();
        let continuing: Vec<u32> = (5..=8).collect();
        let continuing_alloc = kv_manager.allocate_for_request(&continuing).unwrap();
        let mut running = vec![
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::from_u128(1),
                sequence_tokens: finishing,
                prompt_len: 4,
                max_output_tokens: 1,
                planned_output_ids: None,
                kv_lease: finishing_alloc.lease,
                materialized_tokens: 4,
                allocated_tokens: 4,
                images: Vec::new(),
                pending_terminal: false,
            },
            SglangRequest {
                is_decode_handoff: false,
                retracted: false,
                uuid: Uuid::from_u128(2),
                sequence_tokens: continuing,
                prompt_len: 4,
                max_output_tokens: 5,
                planned_output_ids: None,
                kv_lease: continuing_alloc.lease,
                materialized_tokens: 4,
                allocated_tokens: 4,
                images: Vec::new(),
                pending_terminal: false,
            },
        ];
        assert_eq!(kv_manager.cache().available_tokens(), 4);
        assert_eq!(kv_manager.cache().evictable_size, 4);

        let step =
            decode::simulate_prefill_first_tokens(&mut running, &mut kv_manager, &config, 0.0)
                .unwrap();
        assert_eq!(step.output_signals.len(), 2);
        assert!(running[0].pending_terminal);
        assert_eq!(running[0].allocated_tokens, 4);
        assert_eq!(running[1].allocated_tokens, 4);
        assert_eq!(
            kv_manager.cache().evictable_size,
            4,
            "no eviction for the ghost row"
        );
    }

    #[test]
    fn a_retracted_request_neither_receives_nor_keeps_the_token_of_the_forward_in_flight() {
        // Tight KV (56 tokens) as in the legacy retraction test, under the host loop:
        // two requests grow until a decode step retracts one. SGLang marks it
        // `is_retracted` before the previous forward's result is processed, and
        // `process_batch_result_decode` then skips it, so that token is neither
        // delivered nor part of the prompt it re-prefills.
        let args = MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(4)
            .num_gpu_blocks(14)
            .max_num_batched_tokens(Some(32))
            .max_num_seqs(Some(4))
            .speedup_ratio(0.0)
            .sglang(Some(SglangArgs {
                page_size: Some(4),
                chunked_prefill_size: Some(32),
                schedule_conservativeness: Some(0.3),
                host_loop: true,
                ..Default::default()
            }))
            .build()
            .unwrap();
        let mut core = SglangCore::new(args);
        for (uuid, tokens) in [(1u128, 0u32..4), (2, 100..104)] {
            core.receive(DirectRequest {
                uuid: Some(Uuid::from_u128(uuid)),
                ..direct_request(tokens.collect(), 30)
            });
            if uuid == 1 {
                core.execute_hidden_pass(0.0);
            }
        }
        let mut delivered = std::collections::HashMap::<Uuid, usize>::new();
        let mut now_ms = 0.0;
        for _ in 0..60 {
            let pass = core.execute_hidden_pass(now_ms);
            now_ms = pass.end_ms + 1.0;
            let retracted = core
                .waiting
                .iter()
                .find(|request| request.output_len() > 0)
                .map(|request| (request.uuid, request.output_len()));
            for signal in pass
                .output_signals
                .iter()
                .filter(|signal| signal.token_id.is_some())
            {
                *delivered.entry(signal.uuid).or_default() += 1;
            }
            if let Some((uuid, output_len)) = retracted {
                assert!(
                    !pass.output_signals.iter().any(|signal| signal.uuid == uuid),
                    "the forward in flight delivered a token to the retracted request"
                );
                assert!(
                    !pass.committed_requests.contains(&uuid),
                    "the retracted request left the batch before this pass's forward"
                );
                assert_eq!(
                    output_len,
                    delivered.get(&uuid).copied().unwrap_or(0),
                    "the retracted request keeps exactly the tokens the scheduler observed"
                );
                return;
            }
        }
        panic!("test setup: no retraction happened");
    }

    #[test]
    fn a_completed_request_caches_only_the_kv_its_forwards_computed() {
        // 7 prompt tokens and one output token on 4-token pages. The only output
        // token finishes the request right after the prefill that sampled it and
        // never gets a KV slot (SGLang `cache_finished_req` caches `fill_ids[:-1]`),
        // so 7 tokens are materialized: one full page is cached, not two.
        let mut args = test_args(128, 4, 8192);
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(FixedTiming {
                prefill_ms: 20.0,
                decode_ms: 4.0,
            }),
        }
        .into();
        let mut core = SglangCore::new(args);
        core.receive(direct_request((0..7).collect(), 1));
        token_times(&mut core);
        assert!(core.is_drained());
        assert_eq!(core.kv_manager.cache().evictable_size, 4);
    }

    #[test]
    fn a_zero_output_request_completes_when_its_forward_is_observed() {
        let mut core = core(true);
        let uuid = core.receive(direct_request((0..16).collect(), 0));
        let first = core.execute_hidden_pass(0.0);
        assert!(first.output_signals.is_empty());
        assert!(!core.is_drained());

        let second = core.execute_hidden_pass(first.end_ms);
        assert_eq!(second.output_signals.len(), 1);
        assert!(second.output_signals[0].completed);
        assert_eq!(second.output_signals[0].uuid, uuid);
    }
}

mod vision_batches {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::{EncoderShape, VisionShape};

    /// 5 ms per image plus fixed 10 ms prefill / 1 ms decode; records every batch.
    struct RecordingTiming(Arc<Mutex<Vec<Vec<VisionShape>>>>);

    impl crate::engine::TimingModel for RecordingTiming {
        fn predict_vision_ms(&self, shapes: &[VisionShape]) -> anyhow::Result<Option<f64>> {
            self.0.lock().unwrap().push(shapes.to_vec());
            Ok(Some(
                shapes.iter().map(|shape| 5.0 * shape.count as f64).sum(),
            ))
        }
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(10.0)
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            Ok(1.0)
        }
    }

    fn image(identity: u64, token_start: usize) -> ImageSpec {
        ImageSpec {
            identity,
            token_start,
            token_end: token_start + 4,
            encoder: EncoderShape {
                sequences: 1,
                patch_tokens: 16,
                transformer_tokens: 16,
                output_tokens: 4,
            },
            embedding_bytes: 4_000,
        }
    }

    /// A 16-token prompt with distinct token IDs so radix reuse does not hide encoder work.
    fn request(first_token: u32, images: Vec<ImageSpec>) -> DirectRequest {
        DirectRequest {
            images,
            ..direct_request((first_token..first_token + 16).collect(), 1)
        }
    }

    #[test]
    fn decode_ranks_prefill_again_without_pricing_the_tower() {
        // A decode rank prefills only after a retraction, and its timing model has no
        // vision prices: the images were encoded on the prefill rank. Two decode rows
        // share a pool under an external KV hold; the one without output is
        // retracted with its placeholders and prefills after that hold releases.
        struct LatencyOnly;
        impl crate::engine::TimingModel for LatencyOnly {
            fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
                Ok(10.0)
            }
            fn predict_decode_ms(
                &self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<f64> {
                Ok(1.0)
            }
        }
        let mut args = test_args(8, 4, 16);
        args.worker_type = crate::engine::common::protocols::WorkerType::Decode;
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(LatencyOnly),
        }
        .into();
        let mut core = SglangCore::new(args);
        let retracted = Uuid::from_u128(2);
        let images = vec![image(1, 0)];
        for (uuid, tokens, prompt_len, row_images) in [
            (Uuid::from_u128(1), vec![1, 2, 3, 10, 11], 3, Vec::new()),
            (retracted, vec![5, 6, 7, 8, 9], 4, images.clone()),
        ] {
            core.receive(DirectRequest {
                uuid: Some(uuid),
                ..direct_request(tokens[..prompt_len].to_vec(), 3)
            });
            core.waiting.pop_front().unwrap();
            let allocation = core.kv_manager.allocate_for_request(&tokens[..4]).unwrap();
            core.running.push(SglangRequest {
                images: row_images,
                pending_terminal: false,
                is_decode_handoff: false,
                retracted: false,
                uuid,
                sequence_tokens: tokens,
                prompt_len,
                max_output_tokens: 3,
                planned_output_ids: None,
                kv_lease: allocation.lease,
                materialized_tokens: 4,
                allocated_tokens: 4,
            });
        }

        let held = core.kv_manager.allocate_for_request(&vec![99; 24]).unwrap();
        let first = core.execute_hidden_pass(0.0);
        let requeued = core.waiting.front().unwrap();
        assert_eq!(requeued.uuid, retracted);
        assert_eq!(requeued.images, images, "retraction keeps the placeholders");
        core.kv_manager.abort(held.lease);

        let second = core
            .try_execute_hidden_pass(first.end_ms)
            .expect("a decode rank prices neither the prefill nor the tower");
        assert!(
            second
                .output_signals
                .iter()
                .any(|signal| signal.uuid == retracted && signal.token_id.is_some()),
            "the retracted request prefills again on the decode rank"
        );
    }

    #[test]
    fn prefill_chunks_encode_only_their_cache_misses() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        // Chunks of 8 tokens; the cache holds exactly two embeddings.
        let mut args = test_args(128, 1, 8);
        args.sglang.as_mut().unwrap().vlm_cache_bytes = Some(8_000);
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(RecordingTiming(Arc::clone(&observed))),
        }
        .into();
        let mut core = SglangCore::new(args);

        // Images at tokens 0..4 and 8..12: one per chunk.
        core.receive(request(0, vec![image(1, 0), image(2, 8)]));
        let first_chunk = core.execute_hidden_pass(0.0);
        assert_eq!(first_chunk.end_ms, 15.0, "10 ms prefill + one 5 ms image");
        let second_chunk = core.execute_hidden_pass(first_chunk.end_ms);
        assert_eq!(second_chunk.end_ms, 30.0);
        assert_eq!(second_chunk.completed_requests, 1);

        // The same images again hit the cache; a new one misses.
        core.receive(request(100, vec![image(1, 0), image(3, 8)]));
        let hit = core.execute_hidden_pass(second_chunk.end_ms);
        assert_eq!(hit.end_ms, 40.0, "cached image adds no encoder time");
        let miss = core.execute_hidden_pass(hit.end_ms);
        assert_eq!(miss.end_ms, 55.0);

        let shape = |count| VisionShape {
            encoder: EncoderShape {
                sequences: 1,
                patch_tokens: 16,
                transformer_tokens: 16,
                output_tokens: 4,
            },
            count,
        };
        assert_eq!(
            *observed.lock().unwrap(),
            vec![vec![shape(1)], vec![shape(1)], vec![shape(1)]],
            "the provider is only asked for cache misses"
        );
    }

    #[test]
    fn latency_only_providers_reject_image_requests() {
        struct TextOnly;
        impl crate::engine::TimingModel for TextOnly {
            fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
                Ok(1.0)
            }
            fn predict_decode_ms(
                &self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> anyhow::Result<f64> {
                Ok(1.0)
            }
        }
        let mut args = test_args(128, 1, 8192);
        args.perf_model = crate::engine::common::perf_model::PerfModel::External {
            timing: Arc::new(TextOnly),
        }
        .into();
        let mut core = SglangCore::new(args);
        core.receive(request(0, vec![image(1, 0)]));
        let error = core.try_execute_pass_internal(None, 0.0).unwrap_err();
        assert!(
            error.to_string().contains("vision"),
            "unexpected error: {error:#}"
        );
    }
}

mod frontend_pools {
    use super::*;
    use crate::engine::{FrontendConfig, FrontendStage, TtftMilestone};

    /// One pool worker charging 4 ms per request ahead of the host loop.
    fn core() -> SglangCore {
        let mut args = test_args(128, 1, 8192);
        let sglang = args.sglang.as_mut().unwrap();
        sglang.host_loop = true;
        sglang.frontend = Some(FrontendConfig {
            stages: vec![FrontendStage {
                workers: 1,
                service_ms: 4.0,
                concurrency_scale: Vec::new(),
            }],
        });
        SglangCore::new(args)
    }

    #[test]
    fn cancelling_a_pending_source_settles_the_pools_at_the_command_instant() {
        let mut core = core();
        for (uuid, handoff, tokens) in [(1u128, 11u128, 0u32..16), (2, 12, 100..116)] {
            core.apply_command_effects_at(
                SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id: HandoffId::from(Uuid::from_u128(handoff)),
                    request: DirectRequest {
                        uuid: Some(Uuid::from_u128(uuid)),
                        ..direct_request(tokens.collect(), 1)
                    },
                },
                true,
                Some(0.0),
            )
            .unwrap();
        }
        // The first request has run 2 of its 4 ms on the single worker; cancelling
        // it at 2 hands the worker to the second request there, not at the
        // pool's last settled clock (0), which would finish it at 4.
        core.apply_command_effects_at(
            SchedulerCommand::CancelSource {
                handoff_id: HandoffId::from(Uuid::from_u128(11)),
            },
            true,
            Some(2.0),
        )
        .unwrap();
        assert_eq!(core.next_internal_deadline_ms(), Some(6.0));
    }

    #[test]
    fn requests_reach_the_scheduler_when_they_leave_the_pools() {
        let mut core = core();
        let first = core.receive(direct_request((0..16).collect(), 1));
        let second = core.receive(direct_request((100..116).collect(), 1));

        // Both wait on the single worker: the core has nothing to run yet and
        // wakes through its internal deadline instead of an effect-free pass.
        assert!(!core.is_ready());
        assert!(!core.is_drained());
        assert_eq!(core.next_internal_deadline_ms(), Some(4.0));

        core.process_internal_work(4.0);
        assert!(core.is_ready());
        assert_eq!(core.next_internal_deadline_ms(), Some(8.0));
        let pass = core.execute_hidden_pass(4.0);
        let stages = pass
            .lifecycle_events
            .iter()
            .filter_map(|event| match *event {
                SchedulerLifecycleEvent::TtftMilestone {
                    request_id,
                    stage,
                    at_ms,
                } => Some((request_id, stage, at_ms)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(stages.contains(&(first, TtftMilestone::FrontendReady, 4.0)));
        assert!(stages.contains(&(first, TtftMilestone::Received, 4.0)));
        assert!(!stages.iter().any(|(uuid, ..)| *uuid == second));

        // A pass that starts after the second request is ready receives it directly.
        let pass = core.execute_hidden_pass(pass.end_ms.max(8.0));
        assert!(pass.lifecycle_events.iter().any(|event| matches!(
            event,
            SchedulerLifecycleEvent::TtftMilestone {
                request_id,
                stage: TtftMilestone::Received,
                ..
            } if *request_id == second
        )));
    }
}

mod uncharged_decode {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::decode::{DecodeShape, price_decode_shape};
    use super::*;
    use crate::engine::common::perf_model::PerfModel;

    /// Counts decode queries; `phases` decides whether it prices phases.
    struct CountingTiming {
        phases: bool,
        decode_calls: AtomicUsize,
    }

    impl crate::engine::TimingModel for CountingTiming {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            anyhow::bail!("no prefill is priced here")
        }
        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            self.decode_calls.fetch_add(1, Ordering::Relaxed);
            anyhow::ensure!(
                self.phases,
                "an iteration-only provider has no decode phase"
            );
            Ok(5.0)
        }
        fn prices_phases(&self) -> bool {
            self.phases
        }
    }

    #[test]
    fn an_uncharged_decode_asks_only_a_phase_provider_and_takes_no_time() {
        let shape = |charged| DecodeShape {
            batch_size: 2,
            active_kv_tokens: 20,
            mean_context_length: 10,
            lengths: vec![8, 12],
            charged,
        };
        for phases in [true, false] {
            let timing = Arc::new(CountingTiming {
                phases,
                decode_calls: AtomicUsize::new(0),
            });
            let mut config = SglangConfig::from_args(&test_args(64, 1, 16));
            config.perf_model = Arc::new(PerfModel::External {
                timing: Arc::clone(&timing) as Arc<dyn crate::engine::TimingModel>,
            });
            // A failed page reservation charges nothing; a phase provider is
            // still asked, as native timing always was.
            assert_eq!(
                price_decode_shape(&shape(false), &config, true).unwrap(),
                Duration::ZERO
            );
            assert_eq!(
                timing.decode_calls.load(Ordering::Relaxed),
                usize::from(phases)
            );
            // A charged decode always reaches the provider.
            let charged = price_decode_shape(&shape(true), &config, true);
            assert_eq!(charged.is_ok(), phases);
            assert_eq!(
                timing.decode_calls.load(Ordering::Relaxed),
                1 + usize::from(phases)
            );
        }
    }
}
