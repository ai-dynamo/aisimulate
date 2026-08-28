// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin neutral contract adapter over the mechanically moved scheduler cores.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{
    DirectRequest, EngineType, KvTransferTimingMode, MockEngineArgs,
    PreemptionMode as CorePreemptionMode, SglangArgs, WorkerType as CoreWorkerType,
};
use crate::engine::generalized::{CommandContext, RankEngine, RankIdentity, RankPass};
use crate::engine::{
    Admission, Backend, Command, CommandEffects, CommandResult, EngineConfig, ForwardPassMetrics,
    HandoffId, HostOffloadObserver, LifecycleEvent, Metrics, Output, PassCompletionEffects,
    PassStartEffects, PendingPass, PreemptionMode, Request, TimingModel, TransferTimingMode,
    WorkerType,
};

use super::{
    EngineCore, EnginePassResult, KvEventVisibility, MockerMetrics,
    SchedulerCommand as CoreCommand, SchedulerCommandEffects as CoreCommandEffects,
    SchedulerCommandResult as CoreCommandResult, SchedulerLifecycleEvent as CoreLifecycle,
    SglangCore, VllmCore,
};

pub fn engine_seed_offset(identity: RankIdentity) -> Result<u64> {
    identity
        .worker_id
        .checked_mul(u64::from(identity.dp_size.get()))
        .and_then(|base| base.checked_add(u64::from(identity.dp_rank)))
        .ok_or_else(|| anyhow!("native mock-engine seed offset overflow"))
}

/// One preserved vLLM/SGLang scheduler rank behind the neutral contract.
pub struct SchedulerRank {
    core: EngineCore,
    handoff_requests: HashMap<HandoffId, Uuid>,
}

impl SchedulerRank {
    pub(crate) fn set_host_offload_observer(&mut self, observer: Arc<dyn HostOffloadObserver>) {
        self.core.set_host_offload_observer(observer);
    }

    pub fn new_with_timing_model(
        identity: RankIdentity,
        config: &EngineConfig,
        timing: Arc<dyn TimingModel>,
        seed_offset: u64,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            config.native_host_offload.is_none() || identity.dp_size.get() == 1,
            "native_host_offload supports only dp_size=1 in the initial implementation"
        );
        let args = core_args(config, timing);
        let capture_kv_events = config.emit_kv_events;
        let core = match config.backend {
            Backend::Vllm | Backend::Trtllm => EngineCore::Vllm(VllmCore::new_with_worker_rank(
                args,
                identity.worker_id,
                identity.dp_rank,
                seed_offset,
                capture_kv_events,
            )),
            Backend::Sglang => EngineCore::Sglang(SglangCore::new_with_worker_rank(
                args,
                identity.worker_id,
                identity.dp_rank,
                seed_offset,
                capture_kv_events,
            )),
        };
        Ok(Self {
            core,
            handoff_requests: HashMap::new(),
        })
    }

    fn core_command(command: Command) -> CoreCommand {
        match command {
            Command::Submit(request) => CoreCommand::Submit(core_request(request)),
            Command::CancelRequest { request_id, .. } => CoreCommand::CancelRequest { request_id },
            Command::SubmitHandoffPrefill {
                handoff_id,
                request,
            } => CoreCommand::SubmitHandoffPrefill {
                handoff_id,
                request: core_request(request),
            },
            Command::ReserveDestination {
                handoff_id,
                request,
            } => CoreCommand::ReserveDestination {
                handoff_id,
                request: core_request(request),
            },
            Command::ActivateDestination { handoff_id } => {
                CoreCommand::ActivateDestination { handoff_id }
            }
            Command::ReleaseSource { handoff_id } => CoreCommand::ReleaseSource { handoff_id },
            Command::CancelSource { handoff_id } => CoreCommand::CancelSource { handoff_id },
            Command::CancelDestination { handoff_id } => {
                CoreCommand::CancelDestination { handoff_id }
            }
        }
    }

    fn metrics(&self) -> Metrics {
        let metrics = match &self.core {
            EngineCore::Vllm(core) => core.mocker_metrics(),
            EngineCore::Sglang(core) => core.mocker_metrics(),
        };
        map_metrics(metrics)
    }
}

impl RankEngine for SchedulerRank {
    type Config = EngineConfig;
    type Command = Command;
    type CommandEffects = CommandEffects;
    type PassStartEffects = PassStartEffects;
    type PendingPass = PendingPass;
    type PassCompletionEffects = PassCompletionEffects;
    type InternalEffects = PassStartEffects;

    fn new(identity: RankIdentity, config: &Self::Config) -> Result<Self> {
        let timing = config.built_in_timing_model()?;
        let seed_offset = engine_seed_offset(identity)?;
        Self::new_with_timing_model(identity, config, timing, seed_offset)
    }

    fn apply_command_effects(
        &mut self,
        command: Self::Command,
        context: CommandContext,
        pending_pass: Option<&mut Self::PendingPass>,
    ) -> Result<Self::CommandEffects> {
        let pending_suppression = pending_output_suppression(&command, &self.handoff_requests);
        let handoff_update = handoff_tracking_update(&command);
        let core_command = Self::core_command(command);
        let mut effects = self.core.apply_command_effects_at(
            core_command,
            context.allow_immediate_admission(),
            context.now_ms,
        )?;
        // Preserve the scheduler's command boundary when no model step is in
        // flight: native G1 mutations produced by the command belong to its
        // returned effects. Mid-pass mutations remain buffered so neither a
        // command nor a due physical transfer can expose KV state before the
        // shared completion boundary.
        if !context.pass_in_flight {
            effects.kv_events.extend(self.core.drain_kv_events());
        }
        let suppressed_pending_output = if let (Some((request_id, discard_on_noop)), Some(pending)) =
            (pending_suppression, pending_pass)
            && (effects.result != CoreCommandResult::Noop || discard_on_noop)
        {
            let before = pending.effects.outputs.len();
            pending
                .effects
                .outputs
                .retain(|output| output.request_id != request_id);
            pending
                .effects
                .lifecycle_events
                .retain(|event| match *event {
                    LifecycleEvent::SourceHeld { request_id: id, .. }
                    | LifecycleEvent::DestinationReserved { request_id: id, .. } => {
                        id != request_id
                    }
                });
            before != pending.effects.outputs.len()
        } else {
            false
        };
        if effects.result != CoreCommandResult::Noop || suppressed_pending_output {
            self.apply_handoff_tracking_update(handoff_update);
        }
        for request_id in &effects.retired_requests {
            self.handoff_requests
                .retain(|_, tracked_request| tracked_request != request_id);
        }
        map_command_effects(effects, self.metrics(), suppressed_pending_output)
    }

    fn is_ready(&self) -> bool {
        self.core.is_ready()
    }

    fn waiting_for_external_command(&self) -> bool {
        self.core.waiting_for_external_command()
    }

    fn execute_pass(
        &mut self,
        now_ms: f64,
    ) -> Result<RankPass<Self::PassStartEffects, Self::PendingPass>> {
        let pass = self.core.try_execute_hidden_pass(now_ms)?;
        let end_ms = pass.end_ms;
        let (same_timestamp_retry, start_effects, completion_effects) = split_pass(pass)?;
        Ok(RankPass {
            end_ms,
            same_timestamp_retry,
            start_effects,
            pending: PendingPass {
                started_at_ms: now_ms,
                effects: completion_effects,
            },
        })
    }

    fn complete_pass(
        &mut self,
        mut pending: Self::PendingPass,
        end_ms: f64,
    ) -> Result<Self::PassCompletionEffects> {
        self.core.complete_engine_boundary(end_ms);
        // The preserved scheduler retries deferred destination reservations
        // when a forward pass releases capacity. Keep that wakeup at the
        // pass-completion boundary: command-time retry is suppressed while a
        // pass is in flight, and waiting until an unrelated later pass can
        // leave disaggregated replay permanently asleep.
        pending.effects.lifecycle_events.extend(
            self.core
                .retry_pending_destinations()
                .into_iter()
                .map(map_lifecycle),
        );
        let completion_kv_events = self.core.drain_kv_events();
        pending.effects.kv_events.extend(completion_kv_events);
        // Occupancy is authoritative at the shared completion boundary, but
        // SGLang cache hit/total are transient observations from this pass.
        // Preserve those fields while refreshing the rest of the snapshot;
        // a later live adapter latches the last non-empty observation.
        let sglang_cache_hit_tokens = pending.effects.metrics.sglang_cache_hit_tokens;
        let sglang_cache_total_tokens = pending.effects.metrics.sglang_cache_total_tokens;
        pending.effects.metrics = self.metrics();
        pending.effects.metrics.sglang_cache_hit_tokens = sglang_cache_hit_tokens;
        pending.effects.metrics.sglang_cache_total_tokens = sglang_cache_total_tokens;
        pending.effects.forward_pass_metrics.duration_ms =
            (end_ms - pending.started_at_ms).max(0.0);
        for output in &pending.effects.outputs {
            if output.completed {
                self.handoff_requests
                    .retain(|_, request_id| *request_id != output.request_id);
            }
        }
        Ok(pending.effects)
    }

    fn complete_idle_group_pass(
        &mut self,
        started_at_ms: f64,
        end_ms: f64,
    ) -> Result<Option<Self::PassCompletionEffects>> {
        self.core.complete_engine_boundary(end_ms);
        let lifecycle_events = self
            .core
            .retry_pending_destinations()
            .into_iter()
            .map(map_lifecycle)
            .collect::<Vec<_>>();
        let kv_events = self.core.drain_kv_events();
        Ok(Some(PassCompletionEffects {
            lifecycle_events,
            kv_events,
            metrics: self.metrics(),
            forward_pass_metrics: ForwardPassMetrics {
                duration_ms: (end_ms - started_at_ms).max(0.0),
                ..Default::default()
            },
            ..PassCompletionEffects::default()
        }))
    }

    fn next_internal_deadline_ms(&self) -> Option<f64> {
        self.core.next_internal_deadline_ms()
    }

    fn process_internal_work(
        &mut self,
        now_ms: f64,
        pass_in_flight: bool,
    ) -> Result<Self::InternalEffects> {
        if pass_in_flight {
            return Ok(PassStartEffects::default());
        }
        self.core.process_internal_work(now_ms);
        Ok(PassStartEffects {
            kv_events: self.core.drain_kv_events(),
            ..PassStartEffects::default()
        })
    }

    fn is_drained(&self) -> bool {
        self.core.is_drained()
    }
}

#[derive(Clone, Copy)]
enum HandoffTrackingUpdate {
    None,
    Insert(HandoffId, Uuid),
    RemoveHandoff(HandoffId),
    RemoveRequest(Uuid),
}

impl SchedulerRank {
    fn apply_handoff_tracking_update(&mut self, update: HandoffTrackingUpdate) {
        match update {
            HandoffTrackingUpdate::None => {}
            HandoffTrackingUpdate::Insert(handoff_id, request_id) => {
                self.handoff_requests.insert(handoff_id, request_id);
            }
            HandoffTrackingUpdate::RemoveHandoff(handoff_id) => {
                self.handoff_requests.remove(&handoff_id);
            }
            HandoffTrackingUpdate::RemoveRequest(request_id) => self
                .handoff_requests
                .retain(|_, tracked_request| *tracked_request != request_id),
        }
    }
}

fn handoff_tracking_update(command: &Command) -> HandoffTrackingUpdate {
    match command {
        Command::SubmitHandoffPrefill {
            handoff_id,
            request,
        }
        | Command::ReserveDestination {
            handoff_id,
            request,
        } => HandoffTrackingUpdate::Insert(*handoff_id, request.request_id),
        Command::ReleaseSource { handoff_id }
        | Command::CancelSource { handoff_id }
        | Command::CancelDestination { handoff_id } => {
            HandoffTrackingUpdate::RemoveHandoff(*handoff_id)
        }
        Command::CancelRequest { request_id, .. } => {
            HandoffTrackingUpdate::RemoveRequest(*request_id)
        }
        Command::Submit(_) | Command::ActivateDestination { .. } => HandoffTrackingUpdate::None,
    }
}

fn core_args(config: &EngineConfig, timing: Arc<dyn TimingModel>) -> MockEngineArgs {
    MockEngineArgs {
        engine_type: match config.backend {
            Backend::Vllm => EngineType::Vllm,
            Backend::Sglang => EngineType::Sglang,
            Backend::Trtllm => EngineType::Trtllm,
        },
        num_gpu_blocks: config.num_gpu_blocks,
        block_size: config.block_size,
        max_model_len: config.max_model_len,
        max_num_seqs: Some(config.max_num_seqs),
        max_num_batched_tokens: Some(config.max_num_batched_tokens),
        enable_prefix_caching: config.enable_prefix_caching,
        enable_chunked_prefill: config.enable_chunked_prefill,
        speedup_ratio: config.speedup_ratio,
        decode_speedup_ratio: config.decode_speedup_ratio,
        worker_type: match config.worker_type {
            WorkerType::Aggregated => CoreWorkerType::Aggregated,
            WorkerType::Prefill => CoreWorkerType::Prefill,
            WorkerType::Decode => CoreWorkerType::Decode,
        },
        perf_model: Arc::new(PerfModel::External { timing }),
        aic_nextn: config.aic_nextn,
        aic_nextn_accept_rates: config.aic_nextn_accept_rates.clone(),
        aic_mtp_seed: config.aic_mtp_seed,
        kv_bytes_per_token: config.kv_bytes_per_token,
        native_host_offload: config.native_host_offload,
        kv_transfer_bandwidth: config.kv_transfer_bandwidth,
        kv_transfer_timing_mode: match config.kv_transfer_timing_mode {
            TransferTimingMode::FullPrompt => KvTransferTimingMode::FullPrompt,
            TransferTimingMode::DestinationMissing => KvTransferTimingMode::DestinationMissing,
        },
        preemption_mode: match config.preemption_mode {
            PreemptionMode::Lifo => CorePreemptionMode::Lifo,
            PreemptionMode::Fifo => CorePreemptionMode::Fifo,
        },
        sglang: Some(SglangArgs {
            schedule_policy: Some(
                match config.sglang.schedule_policy {
                    crate::engine::SglangSchedulePolicy::Fifo => "fifo",
                    crate::engine::SglangSchedulePolicy::Lpm => "lpm",
                }
                .to_string(),
            ),
            page_size: Some(config.block_size),
            max_prefill_tokens: Some(config.sglang.max_prefill_tokens),
            chunked_prefill_size: Some(config.sglang.chunked_prefill_size),
            clip_max_new_tokens: Some(config.sglang.clip_max_new_tokens),
            schedule_conservativeness: Some(config.sglang.schedule_conservativeness),
        }),
        emit_kv_events: config.emit_kv_events,
        emit_kv_token_ids: config.emit_kv_token_ids,
    }
}

fn core_request(request: Request) -> DirectRequest {
    DirectRequest {
        tokens: request.tokens,
        max_output_tokens: request.max_output_tokens,
        output_token_ids: request.output_token_ids,
        uuid: Some(request.request_id),
        arrival_timestamp_ms: None,
    }
}

fn pending_output_suppression(
    command: &Command,
    handoffs: &HashMap<HandoffId, Uuid>,
) -> Option<(Uuid, bool)> {
    match *command {
        Command::CancelRequest {
            request_id,
            discard_pending_output,
        } => Some((request_id, discard_pending_output)),
        Command::CancelSource { handoff_id } | Command::CancelDestination { handoff_id } => {
            handoffs
                .get(&handoff_id)
                .copied()
                .map(|request_id| (request_id, false))
        }
        _ => None,
    }
}

fn map_command_effects(
    effects: CoreCommandEffects,
    metrics: Metrics,
    suppressed_pending_output: bool,
) -> Result<CommandEffects> {
    let result = match effects.result {
        CoreCommandResult::Submitted(id) => CommandResult::Submitted(id),
        CoreCommandResult::DestinationAccepted { request_id } => {
            CommandResult::DestinationAccepted { request_id }
        }
        CoreCommandResult::Applied => CommandResult::Applied,
        CoreCommandResult::Noop => CommandResult::Noop,
    };
    Ok(CommandEffects {
        result,
        lifecycle_events: effects
            .lifecycle_events
            .into_iter()
            .map(map_lifecycle)
            .collect(),
        kv_events: effects.kv_events,
        retired_requests: effects.retired_requests,
        metrics,
        suppressed_pending_output,
    })
}

fn map_lifecycle(event: CoreLifecycle) -> LifecycleEvent {
    match event {
        CoreLifecycle::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        } => LifecycleEvent::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing,
        },
        CoreLifecycle::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        } => LifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens,
        },
    }
}

fn map_metrics(metrics: MockerMetrics) -> Metrics {
    Metrics {
        dp_rank: metrics.dp_rank,
        active_blocks: metrics.active_decode_blocks,
        total_blocks: metrics.total_blocks,
        cache_usage: metrics.gpu_cache_usage_perc,
        running_requests: metrics.running_requests,
        waiting_requests: metrics.waiting_requests,
        preemptions_total: metrics.vllm_preemptions_total,
        sglang_cache_hit_tokens: metrics.sglang_cache_hit_tokens,
        sglang_cache_total_tokens: metrics.sglang_cache_total_tokens,
    }
}

fn split_pass(
    pass: EnginePassResult,
) -> Result<(
    crate::engine::generalized::SameTimestampRetry,
    PassStartEffects,
    PassCompletionEffects,
)> {
    let EnginePassResult {
        same_timestamp_retry,
        output_signals,
        admissions,
        pressure_events,
        lifecycle_events,
        mocker_metrics,
        kv_event_visibility,
        kv_events,
        fpm,
        ..
    } = pass;
    let (start_kv, completion_kv) = match kv_event_visibility {
        KvEventVisibility::PassEnd => (Vec::new(), kv_events),
    };
    let start = PassStartEffects {
        admissions: admissions
            .into_iter()
            .map(|admission| Admission {
                request_id: admission.uuid,
                reused_input_tokens: admission.reused_input_tokens,
                cache_tier_attribution: admission.cache_tier_attribution,
            })
            .collect(),
        pressure_events,
        kv_events: start_kv,
    };
    let completion = PassCompletionEffects {
        outputs: output_signals
            .into_iter()
            .map(|output| Output {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
                cached_tokens: output.cached_tokens,
            })
            .collect(),
        lifecycle_events: lifecycle_events.into_iter().map(map_lifecycle).collect(),
        kv_events: completion_kv,
        metrics: map_metrics(mocker_metrics),
        forward_pass_metrics: fpm.map(map_fpm).unwrap_or_default(),
    };
    Ok((same_timestamp_retry, start, completion))
}

fn map_fpm(fpm: crate::engine::common::protocols::ForwardPassSnapshot) -> ForwardPassMetrics {
    ForwardPassMetrics {
        num_prefill_requests: fpm.num_prefill_requests,
        sum_prefill_tokens: fpm.sum_prefill_tokens,
        var_prefill_length: fpm.var_prefill_length,
        sum_prefill_kv_tokens: fpm.sum_prefill_kv_tokens,
        num_decode_requests: fpm.num_decode_requests,
        sum_decode_kv_tokens: fpm.sum_decode_kv_tokens,
        var_decode_kv_tokens: fpm.var_decode_kv_tokens,
        num_queued_prefill: fpm.num_queued_prefill,
        sum_queued_prefill_tokens: fpm.sum_queued_prefill_tokens,
        var_queued_prefill_length: fpm.var_queued_prefill_length,
        num_queued_decode: fpm.num_queued_decode,
        sum_queued_decode_kv_tokens: fpm.sum_queued_decode_kv_tokens,
        var_queued_decode_kv_tokens: fpm.var_queued_decode_kv_tokens,
        duration_ms: fpm.wall_time_secs * 1_000.0,
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;
    use std::sync::Mutex;

    use super::*;
    use crate::engine::{
        HostOffloadObservation, HostOffloadObservationData, NativeHostOffloadConfig, PressureKind,
        TimingModelConfig,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct CapturedHostEvent {
        request_id: Uuid,
        kind: &'static str,
        at_ms: f64,
    }

    #[derive(Default)]
    struct HostEventCapture(Mutex<Vec<CapturedHostEvent>>);

    impl HostEventCapture {
        fn snapshot(&self) -> Vec<CapturedHostEvent> {
            self.0.lock().expect("host event capture poisoned").clone()
        }
    }

    impl HostOffloadObserver for HostEventCapture {
        fn record(&self, observation: HostOffloadObservation<'_>) {
            let (kind, at_ms) = match observation.event {
                HostOffloadObservationData::LoadQueued { at_ms, .. } => ("load_queued", at_ms),
                HostOffloadObservationData::LoadCompleted { at_ms, .. } => {
                    ("load_completed", at_ms)
                }
                HostOffloadObservationData::LoadCancelled { at_ms, .. } => {
                    ("load_cancelled", at_ms)
                }
                _ => return,
            };
            self.0
                .lock()
                .expect("host event capture poisoned")
                .push(CapturedHostEvent {
                    request_id: observation.request_id,
                    kind,
                    at_ms,
                });
        }
    }

    fn rank() -> SchedulerRank {
        rank_for_worker(WorkerType::Aggregated)
    }

    fn rank_for_worker(worker_type: WorkerType) -> SchedulerRank {
        let config = EngineConfig {
            worker_type,
            num_gpu_blocks: 8,
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 16,
            speedup_ratio: 0.0,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..EngineConfig::default()
        };
        SchedulerRank::new(
            RankIdentity {
                worker_id: 1,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap()
    }

    fn host_rank(observer: Arc<HostEventCapture>) -> SchedulerRank {
        host_rank_with_capacity(observer, 2, 4)
    }

    fn host_rank_with_capacity(
        observer: Arc<HostEventCapture>,
        g1_blocks: usize,
        host_blocks: usize,
    ) -> SchedulerRank {
        let config = EngineConfig {
            num_gpu_blocks: g1_blocks,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 16,
            kv_bytes_per_token: Some(250_000),
            native_host_offload: Some(
                NativeHostOffloadConfig::new(host_blocks).with_bandwidths(1.0, 1.0),
            ),
            speedup_ratio: 0.0,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..EngineConfig::default()
        };
        let mut rank = SchedulerRank::new(
            RankIdentity {
                worker_id: 2,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap();
        rank.set_host_offload_observer(observer);
        rank
    }

    fn submit_completed_prompt(
        rank: &mut SchedulerRank,
        request_id: Uuid,
        tokens: Vec<u32>,
        now_ms: f64,
    ) -> f64 {
        let effects = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id,
                    tokens,
                    max_output_tokens: 0,
                    output_token_ids: None,
                }),
                CommandContext {
                    now_ms,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, CommandResult::Submitted(request_id));
        let pass = rank.execute_pass(now_ms).unwrap();
        let end_ms = pass.end_ms;
        rank.complete_pass(pass.pending, end_ms).unwrap();
        end_ms
    }

    /// Seed G2 with one prompt, evict it from G1, then queue an H2D owned by a
    /// still-pending source handoff. Returns `(handoff_id, request_id, due_ms)`.
    fn queue_source_h2d(rank: &mut SchedulerRank) -> (HandoffId, Uuid, f64) {
        let seed_id = Uuid::from_u128(93_001);
        let seed_end = submit_completed_prompt(rank, seed_id, vec![1, 2, 3, 4], 0.0);
        let store_due = rank.next_internal_deadline_ms().unwrap();
        assert!(store_due >= seed_end);
        rank.process_internal_work(store_due, false).unwrap();

        let evict_id = Uuid::from_u128(93_002);
        let evict_end =
            submit_completed_prompt(rank, evict_id, vec![5, 6, 7, 8, 9, 10, 11, 12], store_due);
        let mut restore_at_ms = evict_end;
        while let Some(deadline) = rank.next_internal_deadline_ms() {
            restore_at_ms = restore_at_ms.max(deadline);
            rank.process_internal_work(restore_at_ms, false).unwrap();
        }

        let handoff_id = HandoffId::from(Uuid::from_u128(93_003));
        let restore_id = Uuid::from_u128(93_004);
        let effects = rank
            .apply_command_effects(
                Command::SubmitHandoffPrefill {
                    handoff_id,
                    request: Request {
                        request_id: restore_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 0,
                        output_token_ids: None,
                    },
                },
                CommandContext {
                    now_ms: restore_at_ms,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, CommandResult::Submitted(restore_id));
        let pass = rank.execute_pass(restore_at_ms).unwrap();
        rank.complete_pass(pass.pending, pass.end_ms).unwrap();
        let due_ms = rank.next_internal_deadline_ms().unwrap();
        (handoff_id, restore_id, due_ms)
    }

    fn start_request_pass(
        rank: &mut SchedulerRank,
        request_id: Uuid,
        output_token_ids: Vec<u32>,
    ) -> PendingPass {
        let effects = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id,
                    tokens: vec![1, 2, 3, 4],
                    max_output_tokens: output_token_ids.len(),
                    output_token_ids: Some(output_token_ids),
                }),
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, CommandResult::Submitted(request_id));
        let pass = rank.execute_pass(0.0).unwrap();
        assert!(
            pass.pending
                .effects
                .outputs
                .iter()
                .any(|output| output.request_id == request_id)
        );
        pass.pending
    }

    #[test]
    fn ordinary_cancel_suppresses_pending_output_when_scheduler_state_is_removed() {
        let request_id = Uuid::from_u128(90_001);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5, 6]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Applied);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn ordinary_noop_cancel_preserves_pending_output() {
        let request_id = Uuid::from_u128(90_002);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: false,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Noop);
        assert!(!effects.suppressed_pending_output);
        assert_eq!(pending.effects.outputs.len(), 1);
    }

    #[test]
    fn explicit_discard_suppresses_pending_output_after_noop_cancellation() {
        let request_id = Uuid::from_u128(90_003);
        let mut rank = rank();
        let mut pending = start_request_pass(&mut rank, request_id, vec![5]);

        let effects = rank
            .apply_command_effects(
                Command::CancelRequest {
                    request_id,
                    discard_pending_output: true,
                },
                CommandContext {
                    now_ms: 1.0,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();

        assert_eq!(effects.result, CommandResult::Noop);
        assert!(effects.suppressed_pending_output);
        assert!(pending.effects.outputs.is_empty());
    }

    #[test]
    fn mid_pass_command_and_internal_call_keep_due_h2d_hidden() {
        let observer = Arc::new(HostEventCapture::default());
        let mut rank = host_rank(Arc::clone(&observer));
        let (_handoff_id, restore_id, h2d_due_ms) = queue_source_h2d(&mut rank);
        assert!(
            observer
                .snapshot()
                .iter()
                .any(|event| { event.request_id == restore_id && event.kind == "load_queued" })
        );

        let busy_id = Uuid::from_u128(93_005);
        let submission = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id: busy_id,
                    tokens: vec![21, 22, 23, 24],
                    max_output_tokens: 0,
                    output_token_ids: None,
                }),
                CommandContext {
                    now_ms: h2d_due_ms - 0.5,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(submission.result, CommandResult::Submitted(busy_id));
        let pass = rank.execute_pass(h2d_due_ms - 0.5).unwrap();
        assert!(pass.end_ms > h2d_due_ms);
        let mut pending = pass.pending;

        let arrival_id = Uuid::from_u128(93_006);
        let arrival = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id: arrival_id,
                    tokens: vec![31, 32, 33, 34],
                    max_output_tokens: 0,
                    output_token_ids: None,
                }),
                CommandContext {
                    now_ms: h2d_due_ms + 0.5,
                    pass_in_flight: true,
                },
                Some(&mut pending),
            )
            .unwrap();
        assert_eq!(arrival.result, CommandResult::Submitted(arrival_id));
        assert!(arrival.kv_events.is_empty());
        assert!(
            !observer
                .snapshot()
                .iter()
                .any(|event| { event.request_id == restore_id && event.kind == "load_completed" })
        );

        let internal = rank.process_internal_work(h2d_due_ms + 0.5, true).unwrap();
        assert_eq!(internal, PassStartEffects::default());
        assert!(
            !observer
                .snapshot()
                .iter()
                .any(|event| { event.request_id == restore_id && event.kind == "load_completed" })
        );

        rank.complete_pass(pending, pass.end_ms).unwrap();
        let events = observer.snapshot();
        assert!(
            events.iter().any(|event| {
                event.request_id == restore_id
                    && event.kind == "load_completed"
                    // The observation retains the physical completion timestamp,
                    // even though it is emitted only at the later model boundary.
                    && event.at_ms == h2d_due_ms
            }),
            "events: {events:?}, pass end: {}",
            pass.end_ms
        );
    }

    #[test]
    fn rejected_submit_at_due_deadline_does_not_settle_host_work() {
        let observer = Arc::new(HostEventCapture::default());
        let mut rank = host_rank(Arc::clone(&observer));
        let (_handoff_id, restore_id, h2d_due_ms) = queue_source_h2d(&mut rank);
        let events_before = observer.snapshot();
        let metrics_before = rank.metrics();
        let deadline_before = rank.next_internal_deadline_ms();

        let error = rank
            .apply_command_effects(
                Command::Submit(Request {
                    request_id: restore_id,
                    tokens: vec![41, 42, 43, 44],
                    max_output_tokens: 0,
                    output_token_ids: None,
                }),
                CommandContext {
                    now_ms: h2d_due_ms + 1.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("already active"));
        assert_eq!(rank.metrics(), metrics_before);
        assert_eq!(rank.next_internal_deadline_ms(), deadline_before);
        assert_eq!(observer.snapshot(), events_before);
        assert!(rank.core.drain_kv_events().is_empty());
    }

    #[test]
    fn same_pass_queues_multiple_host_loads_with_checked_headroom_updates() {
        let observer = Arc::new(HostEventCapture::default());
        let mut rank = host_rank_with_capacity(Arc::clone(&observer), 4, 8);
        let mut now_ms = 0.0;
        for (request_id, tokens) in [
            (Uuid::from_u128(95_001), vec![1, 2, 3, 4]),
            (Uuid::from_u128(95_002), vec![11, 12, 13, 14]),
        ] {
            now_ms = submit_completed_prompt(&mut rank, request_id, tokens, now_ms);
            while let Some(deadline) = rank.next_internal_deadline_ms() {
                now_ms = now_ms.max(deadline);
                rank.process_internal_work(now_ms, false).unwrap();
            }
        }
        now_ms = submit_completed_prompt(
            &mut rank,
            Uuid::from_u128(95_003),
            (100..116).collect(),
            now_ms,
        );
        while let Some(deadline) = rank.next_internal_deadline_ms() {
            now_ms = now_ms.max(deadline);
            rank.process_internal_work(now_ms, false).unwrap();
        }

        let loads = [
            (Uuid::from_u128(95_004), vec![1, 2, 3, 4, 21, 22, 23, 24]),
            (
                Uuid::from_u128(95_005),
                vec![11, 12, 13, 14, 31, 32, 33, 34],
            ),
        ];
        for (request_id, tokens) in &loads {
            let effects = rank
                .apply_command_effects(
                    Command::Submit(Request {
                        request_id: *request_id,
                        tokens: tokens.clone(),
                        max_output_tokens: 0,
                        output_token_ids: None,
                    }),
                    CommandContext {
                        now_ms,
                        pass_in_flight: false,
                    },
                    None,
                )
                .unwrap();
            assert_eq!(effects.result, CommandResult::Submitted(*request_id));
        }

        let pass = rank.execute_pass(now_ms).unwrap();
        let events = observer.snapshot();
        for (request_id, _) in loads {
            assert!(events.iter().any(|event| {
                event.request_id == request_id
                    && event.kind == "load_queued"
                    && event.at_ms == now_ms
            }));
        }
        rank.complete_pass(pass.pending, pass.end_ms).unwrap();
    }

    #[test]
    fn mid_pass_request_and_source_cancel_observe_command_time_without_completing_h2d() {
        for cancel_source in [false, true] {
            let observer = Arc::new(HostEventCapture::default());
            let mut rank = host_rank(Arc::clone(&observer));
            let (handoff_id, restore_id, h2d_due_ms) = queue_source_h2d(&mut rank);

            let busy_id = Uuid::from_u128(94_000 + u128::from(cancel_source));
            rank.apply_command_effects(
                Command::Submit(Request {
                    request_id: busy_id,
                    tokens: vec![21, 22, 23, 24],
                    max_output_tokens: 0,
                    output_token_ids: None,
                }),
                CommandContext {
                    now_ms: h2d_due_ms - 0.5,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
            let pass = rank.execute_pass(h2d_due_ms - 0.5).unwrap();
            assert!(pass.end_ms > h2d_due_ms);
            let mut pending = pass.pending;
            let cancel_at_ms = h2d_due_ms + 0.5;
            let command = if cancel_source {
                Command::CancelSource { handoff_id }
            } else {
                Command::CancelRequest {
                    request_id: restore_id,
                    discard_pending_output: false,
                }
            };
            let effects = rank
                .apply_command_effects(
                    command,
                    CommandContext {
                        now_ms: cancel_at_ms,
                        pass_in_flight: true,
                    },
                    Some(&mut pending),
                )
                .unwrap();
            assert_eq!(effects.result, CommandResult::Applied);
            let events = observer.snapshot();
            assert!(events.iter().any(|event| {
                event.request_id == restore_id
                    && event.kind == "load_cancelled"
                    && event.at_ms == cancel_at_ms
            }));
            assert!(
                !events.iter().any(|event| {
                    event.request_id == restore_id && event.kind == "load_completed"
                })
            );

            rank.complete_pass(pending, pass.end_ms).unwrap();
            assert!(
                !observer.snapshot().iter().any(|event| {
                    event.request_id == restore_id && event.kind == "load_completed"
                })
            );
        }
    }

    #[test]
    fn cancel_source_uses_command_time_outside_a_pass() {
        let observer = Arc::new(HostEventCapture::default());
        let mut rank = host_rank(Arc::clone(&observer));
        let (handoff_id, restore_id, h2d_due_ms) = queue_source_h2d(&mut rank);
        let cancel_at_ms = h2d_due_ms - 0.25;

        let effects = rank
            .apply_command_effects(
                Command::CancelSource { handoff_id },
                CommandContext {
                    now_ms: cancel_at_ms,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(effects.result, CommandResult::Applied);
        assert!(observer.snapshot().iter().any(|event| {
            event.request_id == restore_id
                && event.kind == "load_cancelled"
                && event.at_ms == cancel_at_ms
        }));
    }

    #[test]
    fn handoff_tracking_is_inserted_on_success_and_cleared_by_cancel() {
        let mut rank = rank_for_worker(WorkerType::Decode);
        let handoff_id = HandoffId::from(Uuid::from_u128(91_001));
        let request_id = Uuid::from_u128(91_002);
        let reservation = rank
            .apply_command_effects(
                Command::ReserveDestination {
                    handoff_id,
                    request: Request {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert!(matches!(
            reservation.result,
            CommandResult::DestinationAccepted { .. }
        ));
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let cancellation = rank
            .apply_command_effects(
                Command::CancelDestination { handoff_id },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(cancellation.result, CommandResult::Applied);
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }

    #[test]
    fn pass_start_exposes_vllm_preemption_pressure_event() {
        let config = EngineConfig {
            num_gpu_blocks: 6,
            block_size: 4,
            max_num_seqs: 2,
            max_num_batched_tokens: 16,
            enable_prefix_caching: false,
            enable_chunked_prefill: true,
            speedup_ratio: 0.0,
            preemption_mode: PreemptionMode::Lifo,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            },
            ..EngineConfig::default()
        };
        let mut rank = SchedulerRank::new(
            RankIdentity {
                worker_id: 7,
                dp_rank: 0,
                dp_size: NonZeroU32::MIN,
            },
            &config,
        )
        .unwrap();
        let first = Uuid::from_u128(92_001);
        let second = Uuid::from_u128(92_002);
        for (request_id, tokens) in [
            (first, (0..8).collect::<Vec<_>>()),
            (second, (100..108).collect::<Vec<_>>()),
        ] {
            let effects = rank
                .apply_command_effects(
                    Command::Submit(Request {
                        request_id,
                        tokens,
                        max_output_tokens: 8,
                        output_token_ids: None,
                    }),
                    CommandContext {
                        now_ms: 0.0,
                        pass_in_flight: false,
                    },
                    None,
                )
                .unwrap();
            assert_eq!(effects.result, CommandResult::Submitted(request_id));
        }

        let mut now_ms = 0.0;
        let mut observed = None;
        for _ in 0..16 {
            let pass = rank.execute_pass(now_ms).unwrap();
            if let Some(event) = pass.start_effects.pressure_events.first() {
                assert_eq!(pass.start_effects.pressure_events.len(), 1);
                observed = Some(event.clone());
            }
            let end_ms = pass.end_ms;
            rank.complete_pass(pass.pending, end_ms).unwrap();
            if observed.is_some() {
                break;
            }
            now_ms = end_ms.max(now_ms + 1.0);
        }

        let event = observed.expect("tight native G1 capacity should preempt one vLLM request");
        assert_eq!(event.at_ms, now_ms);
        assert_eq!(event.kind, PressureKind::VllmPreemption);
        assert_eq!(event.request_id, second);
        assert_eq!(event.state_before.running_requests, 2);
        assert_eq!(event.state_before.waiting_requests, Some(0));
        assert_eq!(event.state_after.running_requests, 1);
        assert_eq!(event.state_after.waiting_requests, Some(1));
        assert!(event.request_active_blocks_before > 0);
        assert!(event.state_after.active_blocks < event.state_before.active_blocks);
        assert_eq!(event.logical_available_blocks_before, None);
        assert_eq!(event.required_blocks_before, None);
    }

    #[test]
    fn terminal_handoff_output_clears_tracking() {
        let mut rank = rank_for_worker(WorkerType::Prefill);
        let handoff_id = HandoffId::from(Uuid::from_u128(92_001));
        let request_id = Uuid::from_u128(92_002);
        let submission = rank
            .apply_command_effects(
                Command::SubmitHandoffPrefill {
                    handoff_id,
                    request: Request {
                        request_id,
                        tokens: vec![1, 2, 3, 4],
                        max_output_tokens: 1,
                        output_token_ids: Some(vec![5]),
                    },
                },
                CommandContext {
                    now_ms: 0.0,
                    pass_in_flight: false,
                },
                None,
            )
            .unwrap();
        assert_eq!(submission.result, CommandResult::Submitted(request_id));
        assert_eq!(rank.handoff_requests.get(&handoff_id), Some(&request_id));

        let pass = rank.execute_pass(0.0).unwrap();
        let completion = rank.complete_pass(pass.pending, pass.end_ms).unwrap();
        assert!(
            completion
                .outputs
                .iter()
                .any(|output| output.request_id == request_id && output.completed)
        );
        assert!(!rank.handoff_requests.contains_key(&handoff_id));
    }
}
