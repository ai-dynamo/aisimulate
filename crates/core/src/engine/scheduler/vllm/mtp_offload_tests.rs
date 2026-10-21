// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-License-Identifier: Apache-2.0

// Modified synchronous MTP/G2 behavioral tests derived from vLLM v0.32.0:
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
// https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py
// See THIRD_PARTY_NOTICES.md for the supported shared full-attention profile.

use super::core::VllmCore;
use super::request::RequestKvState;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use rstest::rstest;
use uuid::Uuid;

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
fn full_g1_hit_is_adjusted_before_h2d_and_attributed_after_final_token_recompute() {
    #[derive(Default)]
    struct QueryTiming(std::sync::Mutex<Vec<(usize, usize, usize)>>);
    impl crate::engine::TimingModel for QueryTiming {
        fn predict_prefill_ms(
            &self,
            batch_size: usize,
            mean_isl: usize,
            mean_prefix: usize,
        ) -> anyhow::Result<f64> {
            self.0
                .lock()
                .unwrap()
                .push((batch_size, mean_isl, mean_prefix));
            Ok((batch_size * (mean_isl - mean_prefix)) as f64)
        }

        fn predict_decode_ms(
            &self,
            _batch_size: usize,
            _active_kv_tokens: usize,
            _mean_context_length: usize,
            _total_kv_tokens: usize,
        ) -> anyhow::Result<f64> {
            panic!("a first prefill output must not incur a separate decode forward")
        }
    }

    #[derive(Default)]
    struct LoadCapture(std::sync::Mutex<Vec<(Uuid, usize)>>);
    impl crate::engine::HostOffloadObserver for LoadCapture {
        fn record(&self, observation: crate::engine::HostOffloadObservation<'_>) {
            if let crate::engine::HostOffloadObservationData::LoadQueued { blocks, .. } =
                observation.event
            {
                self.0
                    .lock()
                    .unwrap()
                    .push((observation.request_id, blocks.len()));
            }
        }
    }

    let timing = std::sync::Arc::new(QueryTiming::default());
    let mut timed_args = args(2, 16, false, true);
    timed_args.speedup_ratio = 1.0;
    timed_args.perf_model =
        std::sync::Arc::new(crate::engine::common::perf_model::PerfModel::External {
            timing: timing.clone(),
        });
    let mut core = VllmCore::new(timed_args);
    let capture = std::sync::Arc::new(LoadCapture::default());
    core.set_host_offload_observer(capture.clone());
    let mut collector = crate::engine::trace::TraceCollector::default();
    submit(&mut core, 115, 12, 0);
    let seed = core.execute_pass(&mut collector, 0.0);
    assert_eq!(seed.completed_requests, 1);
    core.complete_engine_boundary(seed.end_ms);
    assert_eq!(raw_prefix_tokens(&core, (0..12).collect()), 12);

    let uuid = submit(&mut core, 116, 12, 1);
    let loading = core.execute_pass(&mut collector, seed.end_ms);
    assert!(loading.output_signals.is_empty());
    assert_eq!(*capture.0.lock().unwrap(), vec![(uuid, 2)]);
    core.complete_engine_boundary(loading.end_ms);
    let activated = core.execute_pass(&mut collector, loading.end_ms);
    assert_eq!(activated.completed_requests, 1);
    assert_eq!(activated.output_signals.len(), 1);
    assert_eq!(activated.end_ms - loading.end_ms, 3.0);
    assert_eq!(
        *timing.0.lock().unwrap(),
        vec![(1, 12, 0), (1, 14, 11)],
        "the analytical provider must charge all three query positions, including invalid padding"
    );
    assert_eq!(activated.admissions[0].reused_input_tokens, 11);
    assert_eq!(
        activated.admissions[0].cache_tier_attribution,
        Some(crate::engine::CacheTierAttribution {
            g1_reused_input_tokens: 4,
            host_reused_input_tokens: 7,
        })
    );
    assert_eq!(activated.output_signals[0].cached_tokens, Some(11));
    assert_eq!(
        activated.fpm.unwrap().sum_prefill_tokens,
        3,
        "the one recomputed token and two invalid padding positions all consume query work"
    );
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn h2d_prefix_plus_step_headroom_can_fit_without_full_prompt_plus_step() {
    let mut host_args = args(2, 16, false, true);
    host_args.num_gpu_blocks = 5;
    let mut core = VllmCore::new(host_args);
    let mut collector = crate::engine::trace::TraceCollector::default();
    let mut now = 0.0;

    // Store the twelve-token prefix, then evict every G1 block of it with
    // a different sixteen-token prompt while retaining its host copy.
    for (id, start, len) in [(107, 0, 12), (108, 100, 16)] {
        core.receive(DirectRequest {
            tokens: (start..start + len).collect(),
            max_output_tokens: 0,
            uuid: Some(Uuid::from_u128(id)),
            ..Default::default()
        });
        let pass = core.execute_pass(&mut collector, now);
        assert_eq!(pass.completed_requests, 1);
        assert!(!pass.output_signals[0].rejected);
        now = pass.end_ms;
        core.complete_engine_boundary(now);
    }

    let uuid = submit(&mut core, 109, 13, 1);
    let loading = core.execute_pass(&mut collector, now);
    assert!(loading.output_signals.is_empty());
    assert_eq!(
        core.kv_manager.num_active_blocks(),
        3,
        "H2D reserves only its three destination blocks; its two-block step allowance is headroom"
    );
    core.complete_engine_boundary(loading.end_ms);
    let activated = core.execute_pass(&mut collector, loading.end_ms);
    assert!(
        matches!(activated.output_signals.as_slice(), [OutputSignal {
        uuid: output_uuid, completed: true, rejected: false,
        cached_tokens: Some(12), ..
    }] if *output_uuid == uuid)
    );
    assert!(core.is_empty());
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[rstest]
#[case(false)]
#[case(true)]
fn cold_g2_and_g1_share_the_same_mtp_state_transitions(#[case] reject_first: bool) {
    let mut plain_args = args(2, 4, reject_first, false);
    plain_args.enable_prefix_caching = true;
    let mut plain = VllmCore::new(plain_args);
    let mut host = VllmCore::new(args(2, 4, reject_first, true));
    let uuid = submit(&mut plain, 106, 12, 8);
    assert_eq!(submit(&mut host, 106, 12, 8), uuid);
    let mut plain_collector = crate::engine::trace::TraceCollector::default();
    let mut host_collector = crate::engine::trace::TraceCollector::default();
    let mut now = 0.0;
    for _ in 0..12 {
        let plain_pass = plain.execute_pass(&mut plain_collector, now);
        let host_pass = host.execute_pass(&mut host_collector, now);
        let outputs = |signals: &[OutputSignal]| {
            signals
                .iter()
                .map(|signal| {
                    (
                        signal.uuid,
                        signal.token_id,
                        signal.completed,
                        signal.rejected,
                        signal.cached_tokens,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            outputs(&plain_pass.output_signals),
            outputs(&host_pass.output_signals)
        );
        assert_eq!(plain_pass.completed_requests, host_pass.completed_requests);
        assert_eq!(plain_pass.end_ms, host_pass.end_ms);
        let state = |core: &VllmCore| {
            core.state.requests.get(&uuid).map(|request| {
                (
                    request.sequence.len(),
                    request.num_computed_tokens,
                    request.sequence.lease.physical_owned_block_count(),
                )
            })
        };
        assert_eq!(state(&plain), state(&host));
        assert_eq!(
            plain.kv_manager.num_active_blocks(),
            host.kv_manager.num_active_blocks()
        );
        now = plain_pass.end_ms;
        plain.complete_engine_boundary(now);
        host.complete_engine_boundary(now);
        if plain.is_empty() {
            break;
        }
    }
    assert!(plain.is_empty());
    assert!(host.is_empty());
    assert_eq!(plain.kv_manager.num_active_blocks(), 0);
    assert_eq!(host.kv_manager.num_active_blocks(), 0);
}

mod transfer_lifecycle {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::super::core::VllmCore;
    use crate::engine::common::perf_model::PerfModel;
    use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
    use crate::engine::kv_manager::SourceReuseDependency;
    use crate::engine::scheduler::SchedulerCommand;
    use crate::engine::trace::TraceCollector;
    use crate::engine::{
        HostOffloadObservation, HostOffloadObservationData, HostOffloadObserver,
        NativeHostOffloadConfig, TimingModel,
    };
    use uuid::Uuid;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Transition {
        StorePrepared,
        StoreSubmitted,
        StoreCompleted,
        LoadQueued,
        LoadCompleted,
        LoadCancelled,
    }

    #[derive(Clone, Copy, Debug)]
    struct TransferEvent {
        owner: Uuid,
        transition: Transition,
        id: u64,
        at_ms: f64,
        deadline_ms: Option<f64>,
    }

    #[derive(Default)]
    struct TransferCapture(Mutex<Vec<TransferEvent>>);

    impl TransferCapture {
        fn events(&self, owner: Uuid, transition: Transition) -> Vec<TransferEvent> {
            self.0
                .lock()
                .unwrap()
                .iter()
                .copied()
                .filter(|event| event.owner == owner && event.transition == transition)
                .collect()
        }
    }

    impl HostOffloadObserver for TransferCapture {
        fn record(&self, observation: HostOffloadObservation<'_>) {
            let (transition, id, at_ms, deadline_ms) = match observation.event {
                HostOffloadObservationData::StorePrepared {
                    at_ms, transfer_id, ..
                } => (Transition::StorePrepared, transfer_id.get(), at_ms, None),
                HostOffloadObservationData::StoreSubmitted {
                    at_ms,
                    completes_at_ms,
                    transfer_id,
                    ..
                } => (
                    Transition::StoreSubmitted,
                    transfer_id.get(),
                    at_ms,
                    Some(completes_at_ms),
                ),
                HostOffloadObservationData::StoreCompleted {
                    at_ms, transfer_id, ..
                } => (Transition::StoreCompleted, transfer_id.get(), at_ms, None),
                HostOffloadObservationData::LoadQueued {
                    at_ms,
                    completes_at_ms,
                    transfer_id,
                    ..
                } => (
                    Transition::LoadQueued,
                    transfer_id.get(),
                    at_ms,
                    Some(completes_at_ms),
                ),
                HostOffloadObservationData::LoadCompleted {
                    at_ms, transfer_id, ..
                } => (Transition::LoadCompleted, transfer_id.get(), at_ms, None),
                HostOffloadObservationData::LoadCancelled {
                    at_ms, transfer_id, ..
                } => (Transition::LoadCancelled, transfer_id.get(), at_ms, None),
                _ => return,
            };
            self.0.lock().unwrap().push(TransferEvent {
                owner: observation.request_id,
                transition,
                id,
                at_ms,
                deadline_ms,
            });
        }
    }

    fn args(capacity: usize) -> MockEngineArgs {
        MockEngineArgs::builder()
            .num_gpu_blocks(capacity)
            .block_size(4)
            .max_num_seqs(Some(4))
            .max_num_batched_tokens(Some(32))
            .aic_nextn(Some(2))
            .aic_nextn_accept_rates(Some("1.0,1.0".into()))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(
                NativeHostOffloadConfig::new(16).with_bandwidths(1.0, 1.0),
            ))
            .perf_model(Arc::new(PerfModel::External {
                timing: Arc::new(FailingTiming::default()),
            }))
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn submit(
        core: &mut VllmCore,
        owner: Uuid,
        start: u32,
        tokens: u32,
        outputs: usize,
        at_ms: f64,
    ) {
        core.receive(DirectRequest {
            tokens: (start..start + tokens).collect(),
            max_output_tokens: outputs,
            uuid: Some(owner),
            arrival_timestamp_ms: Some(at_ms),
            ..DirectRequest::default()
        });
    }

    fn finish_transfer_work(core: &mut VllmCore) -> f64 {
        let mut end_ms = 0.0;
        for _ in 0..16 {
            let Some(deadline) = core.next_internal_deadline_ms() else {
                break;
            };
            end_ms = deadline;
            core.process_internal_work(deadline);
        }
        assert!(core.next_internal_deadline_ms().is_none());
        end_ms
    }

    #[test]
    fn mtp_reuses_pending_d2h_capacity_only_after_the_source_read_deadline() {
        let mut core = VllmCore::new(args(3));
        let capture = Arc::new(TransferCapture::default());
        core.set_host_offload_observer(capture.clone());
        let mut collector = TraceCollector::default();
        let source = Uuid::from_u128(130_001);
        submit(&mut core, source, 0, 4, 1, 0.0);
        let seed = core.execute_pass(&mut collector, 0.0);
        assert!(
            seed.output_signals
                .iter()
                .any(|signal| signal.uuid == source && signal.completed)
        );
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        core.complete_engine_boundary(seed.end_ms);
        let submitted = capture.events(source, Transition::StoreSubmitted);
        assert_eq!(submitted.len(), 1);
        let store = submitted[0];
        assert_eq!(store.at_ms, 0.0);
        assert_eq!(store.deadline_ms, Some(1.0));
        let dependency = SourceReuseDependency::from_adapter_id(store.id);
        assert!(
            core.kv_manager
                .is_native_source_dependency_pending(dependency)
        );

        // Eight known tokens plus two future slots consume all three blocks,
        // including the released physical source of the unfinished one-block D2H.
        let replacement = Uuid::from_u128(130_002);
        submit(&mut core, replacement, 100, 8, 1, 0.0);
        let pass = core.execute_pass(&mut collector, 0.0);
        assert!(
            pass.output_signals
                .iter()
                .any(|signal| signal.uuid == replacement && signal.completed)
        );
        assert_eq!(
            pass.end_ms, 1.0,
            "the D2H fence must delay otherwise zero-time compute"
        );
        assert!(
            core.kv_manager
                .is_native_source_dependency_pending(dependency)
        );
        assert!(
            capture
                .events(source, Transition::StoreCompleted)
                .is_empty(),
            "ordering compute after the read must not prematurely complete the transfer"
        );
        assert_eq!(
            capture.events(replacement, Transition::StorePrepared).len(),
            1,
            "the completed replacement must not lose its prompt store behind the old D2H"
        );
        core.complete_engine_boundary(pass.end_ms);
        assert!(
            !core
                .kv_manager
                .is_native_source_dependency_pending(dependency)
        );
        assert_eq!(
            capture.events(source, Transition::StoreCompleted)[0].at_ms,
            1.0
        );
        let replacement_store = capture.events(replacement, Transition::StoreSubmitted);
        assert_eq!(replacement_store.len(), 1);
        assert_eq!(replacement_store[0].at_ms, 1.0);
        assert_eq!(replacement_store[0].deadline_ms, Some(3.0));
        let replacement_dependency =
            SourceReuseDependency::from_adapter_id(replacement_store[0].id);
        assert!(
            core.kv_manager
                .is_native_source_dependency_pending(replacement_dependency)
        );

        // A third owner must now wait for the replacement's two-block read, not
        // merely for the already-completed original one-block transfer.
        let third = Uuid::from_u128(130_003);
        submit(&mut core, third, 200, 8, 1, 1.0);
        let chained = core.execute_pass(&mut collector, 1.0);
        assert!(
            chained
                .output_signals
                .iter()
                .any(|signal| signal.uuid == third && signal.completed)
        );
        assert_eq!(chained.end_ms, 3.0);
        assert!(
            core.kv_manager
                .is_native_source_dependency_pending(replacement_dependency)
        );
        assert_eq!(capture.events(third, Transition::StorePrepared).len(), 1);
        core.complete_engine_boundary(chained.end_ms);
        assert!(
            !core
                .kv_manager
                .is_native_source_dependency_pending(replacement_dependency)
        );
        let third_store = capture.events(third, Transition::StoreSubmitted);
        assert_eq!(third_store.len(), 1);
        assert_eq!(third_store[0].at_ms, 3.0);
        assert_eq!(third_store[0].deadline_ms, Some(5.0));
        finish_transfer_work(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert!(core.is_drained());
    }

    #[test]
    fn inflight_h2d_reserves_mtp_headroom_and_cancellation_releases_it() {
        let mut core = VllmCore::new(args(3));
        let capture = Arc::new(TransferCapture::default());
        core.set_host_offload_observer(capture.clone());
        let mut collector = TraceCollector::default();
        let mut now_ms = 0.0;
        for (id, start) in [(130_010, 0), (130_011, 100)] {
            let seed = Uuid::from_u128(id);
            submit(&mut core, seed, start, 4, 1, now_ms);
            let pass = core.execute_pass(&mut collector, now_ms);
            assert!(
                pass.output_signals
                    .iter()
                    .any(|signal| signal.uuid == seed && signal.completed)
            );
            core.complete_engine_boundary(pass.end_ms);
            now_ms = finish_transfer_work(&mut core);
        }
        let first = Uuid::from_u128(130_012);
        let second = Uuid::from_u128(130_013);
        submit(&mut core, first, 0, 4, 1, now_ms);
        submit(&mut core, second, 100, 4, 1, now_ms);
        let pass = core.execute_pass(&mut collector, now_ms);
        assert!(pass.output_signals.is_empty());
        assert_eq!(capture.events(first, Transition::LoadQueued).len(), 1);
        assert!(capture.events(second, Transition::LoadQueued).is_empty());
        assert_eq!(
            core.kv_manager.num_active_blocks(),
            1,
            "only the prefix is physically held; remaining MTP headroom stays logical"
        );

        core.apply_command_effects_at(
            SchedulerCommand::CancelRequest { request_id: first },
            true,
            Some(now_ms),
        )
        .unwrap();
        assert_eq!(capture.events(first, Transition::LoadCancelled).len(), 1);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        let retry = core.execute_pass(&mut collector, now_ms);
        assert!(retry.output_signals.is_empty());
        let queued = capture.events(second, Transition::LoadQueued);
        assert_eq!(queued.len(), 1);
        let deadline = queued[0].deadline_ms.unwrap();
        core.process_internal_work(deadline);
        let resumed = core.execute_pass(&mut collector, deadline);
        assert!(
            resumed
                .output_signals
                .iter()
                .any(|signal| signal.uuid == second && signal.completed)
        );
        assert_eq!(capture.events(second, Transition::LoadCompleted).len(), 1);
        core.complete_engine_boundary(resumed.end_ms);
        finish_transfer_work(&mut core);
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        assert!(core.is_drained());
    }

    #[derive(Default)]
    struct FailingTiming {
        prefill: AtomicBool,
        decode: AtomicBool,
    }

    impl TimingModel for FailingTiming {
        fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            anyhow::ensure!(
                !self.prefill.load(Ordering::Relaxed),
                "injected prefill failure"
            );
            Ok(0.0)
        }

        fn predict_decode_ms(&self, _: usize, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
            anyhow::ensure!(
                !self.decode.load(Ordering::Relaxed),
                "injected decode failure"
            );
            Ok(1.0)
        }
    }

    #[test]
    fn prediction_failure_then_cancel_leaves_no_unsubmitted_store_or_opaque_capacity() {
        for fail_decode in [false, true] {
            let timing = Arc::new(FailingTiming::default());
            let mut config = args(5);
            config.perf_model = Arc::new(PerfModel::External {
                timing: timing.clone(),
            });
            let mut core = VllmCore::new(config);
            let capture = Arc::new(TransferCapture::default());
            core.set_host_offload_observer(capture.clone());
            let mut collector = TraceCollector::default();
            let mut now_ms = 0.0;
            let running = Uuid::from_u128(130_020);
            if fail_decode {
                submit(&mut core, running, 0, 4, 4, now_ms);
                let prefill = core.execute_pass(&mut collector, now_ms);
                core.complete_engine_boundary(prefill.end_ms);
                now_ms = finish_transfer_work(&mut core);
                timing.decode.store(true, Ordering::Relaxed);
            } else {
                timing.prefill.store(true, Ordering::Relaxed);
            }
            let failing = Uuid::from_u128(130_021);
            submit(&mut core, failing, 100, 4, 1, now_ms);
            let error = core
                .try_execute_pass(&mut collector, now_ms)
                .err()
                .expect("timing failure must propagate");
            assert!(error.root_cause().to_string().contains(if fail_decode {
                "decode"
            } else {
                "prefill"
            }));
            assert!(
                capture
                    .events(failing, Transition::StorePrepared)
                    .is_empty()
            );
            assert!(core.next_internal_deadline_ms().is_none());
            assert!(core.kv_manager.num_active_blocks() > 0);
            for request_id in [running, failing] {
                core.apply_command_effects_at(
                    SchedulerCommand::CancelRequest { request_id },
                    true,
                    Some(now_ms),
                )
                .unwrap();
            }
            assert_eq!(core.kv_manager.num_active_blocks(), 0);
            assert!(core.is_drained());

            timing.prefill.store(false, Ordering::Relaxed);
            timing.decode.store(false, Ordering::Relaxed);
            let replacement = Uuid::from_u128(130_022);
            // Four prompt blocks plus one opaque future block need the entire pool.
            submit(&mut core, replacement, 200, 16, 1, now_ms);
            let pass = core.execute_pass(&mut collector, now_ms);
            assert!(
                pass.output_signals
                    .iter()
                    .any(|signal| signal.uuid == replacement && signal.completed)
            );
            core.complete_engine_boundary(pass.end_ms);
            let submitted = capture.events(replacement, Transition::StoreSubmitted);
            assert_eq!(submitted.len(), 1);
            assert!(submitted[0].deadline_ms.unwrap() > now_ms);
            finish_transfer_work(&mut core);
            assert_eq!(core.kv_manager.num_active_blocks(), 0);
            assert!(core.is_drained());
        }
    }
}
