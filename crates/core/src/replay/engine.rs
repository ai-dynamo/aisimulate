// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default generalized-engine construction for offline replay.
//!
//! This module contains configuration and conversion helpers only. Scheduler
//! state lives in `aisimulate_core::engine`; virtual time and worker lifecycle live
//! in the moved aggregated/disaggregated replay runtimes.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::engine::generalized::EngineIdentity;
use crate::engine::{Backend, Engine, EngineConfig, EngineFactory, TimingModel, WorkerType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::components::{
    AdmissionQueue, NoReplayMetadata, ReplayEngineObservation, ReplayMode,
};
use crate::replay::core::EngineEventBatch;
use crate::replay::core::round_robin::PoolRoundRobinPlacement;
use crate::replay::disagg::DisaggRuntimeImpl;
use crate::replay::error::runtime_error;
use crate::replay::protocol::DirectRequest;
use crate::replay::{
    ReplayError, ReplayReport, ReplayResult, ReplaySpec, ReplayTopology, Replayer, WorkerStage,
};

fn default_dp_size() -> u32 {
    1
}

fn default_tensor_parallel_size() -> u32 {
    1
}

/// Serializable execution-time descriptor for the AISimulate engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayEngineConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    /// Whether `rank.num_gpu_blocks` came from the user rather than an
    /// upstream capacity estimator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_gpu_blocks_is_explicit: Option<bool>,
    pub rank: EngineConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<ReplayRoleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<ReplayRoleConfig>,
}

impl Default for ReplayEngineConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            num_gpu_blocks_is_explicit: None,
            rank: EngineConfig::default(),
            prefill: None,
            decode: None,
        }
    }
}

/// Rank-group descriptor for one disaggregated role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplayRoleConfig {
    #[serde(default = "default_dp_size")]
    pub dp_size: u32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: u32,
    /// Whether `rank.num_gpu_blocks` came from the user rather than an
    /// upstream capacity estimator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_gpu_blocks_is_explicit: Option<bool>,
    pub rank: EngineConfig,
}

impl Default for ReplayRoleConfig {
    fn default() -> Self {
        Self {
            dp_size: 1,
            tensor_parallel_size: 1,
            num_gpu_blocks_is_explicit: None,
            rank: EngineConfig::default(),
        }
    }
}

impl ReplayEngineConfig {
    pub(crate) fn parse(value: &Value) -> ReplayResult<Self> {
        if value.is_null() {
            return Ok(Self::default());
        }
        let parsed: Self = serde_json::from_value(value.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!("invalid native engine descriptor: {error}"))
        })?;
        parsed.reject_ambiguous_role_topology(value)?;
        Ok(parsed)
    }

    /// Refuse a role block that inherits topology by omission.
    ///
    /// `role()` falls back to the top-level values only when the whole
    /// `prefill`/`decode` block is absent. When the block is *present but
    /// partial*, `ReplayRoleConfig`'s own per-field defaults apply instead, so
    /// `{"dp_size": 8, "prefill": {"rank": {..}}}` silently runs prefill at
    /// `dp_size: 1` -- an 8x capacity difference with no warning, which
    /// `validate_request_dp_ranks` then range-checks against the wrong value and
    /// so confirms rather than catches.
    ///
    /// Which way to resolve that (inherit, or default) is a config-semantics
    /// decision, and either choice silently changes an existing run's numbers.
    /// Refusing the ambiguous authoring instead cannot: it only rejects the
    /// configurations that are already behaving unpredictably, and it is a no-op
    /// whenever the inherited and defaulted values agree.
    fn reject_ambiguous_role_topology(&self, value: &Value) -> ReplayResult<()> {
        let default_role = ReplayRoleConfig::default();
        for role_name in ["prefill", "decode"] {
            let Some(role) = value.get(role_name).and_then(Value::as_object) else {
                continue;
            };
            // Every field `role()` inherits when the block is absent, and so
            // every field a present-but-partial block silently *defaults*
            // instead. `rank` carries the KV budget, backend, and timing model,
            // making it the most damaging omission of the four, not the least:
            // `{"rank": {"num_gpu_blocks": 100000}, "prefill": {"dp_size": 1, "rank": {}}}`
            // parsed cleanly and ran prefill on a default KV budget.
            for (field, inherited_differs, inherited, defaulted) in [
                (
                    "dp_size",
                    self.dp_size != default_role.dp_size,
                    self.dp_size.to_string(),
                    default_role.dp_size.to_string(),
                ),
                (
                    "tensor_parallel_size",
                    self.tensor_parallel_size != default_role.tensor_parallel_size,
                    self.tensor_parallel_size.to_string(),
                    default_role.tensor_parallel_size.to_string(),
                ),
                (
                    "num_gpu_blocks_is_explicit",
                    self.num_gpu_blocks_is_explicit != default_role.num_gpu_blocks_is_explicit,
                    format!("{:?}", self.num_gpu_blocks_is_explicit),
                    format!("{:?}", default_role.num_gpu_blocks_is_explicit),
                ),
                (
                    "rank",
                    self.rank != default_role.rank,
                    "an authored top-level block".to_string(),
                    "an all-default engine config".to_string(),
                ),
            ] {
                if !role.contains_key(field) && inherited_differs {
                    return Err(ReplayError::InvalidSpec(format!(
                        "engine descriptor sets {field} {inherited} but its {role_name} block omits \
                         {field}, which would silently run {role_name} at {defaulted}; author \
                         {role_name}.{field} explicitly"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn role(&self, stage: WorkerStage) -> ReplayRoleConfig {
        let mut role = match stage {
            WorkerStage::Aggregated => ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            },
            WorkerStage::Prefill => self.prefill.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            }),
            WorkerStage::Decode => self.decode.clone().unwrap_or_else(|| ReplayRoleConfig {
                dp_size: self.dp_size,
                tensor_parallel_size: self.tensor_parallel_size,
                num_gpu_blocks_is_explicit: self.num_gpu_blocks_is_explicit,
                rank: self.rank.clone(),
            }),
        };
        role.rank.worker_type = match stage {
            WorkerStage::Aggregated => WorkerType::Aggregated,
            WorkerStage::Prefill => WorkerType::Prefill,
            WorkerStage::Decode => WorkerType::Decode,
        };
        role
    }

    pub(crate) fn validate_topology(&self, topology: &ReplayTopology) -> ReplayResult<()> {
        match topology {
            ReplayTopology::Aggregated { .. } => {
                if self.rank.native_host_offload.is_some() && self.dp_size != 1 {
                    return Err(ReplayError::InvalidSpec(
                        "native_host_offload supports only dp_size=1 in the initial implementation"
                            .to_string(),
                    ));
                }
            }
            ReplayTopology::Disaggregated { .. } => {
                for stage in [WorkerStage::Prefill, WorkerStage::Decode] {
                    let role = self.role(stage);
                    if role.rank.native_host_offload.is_some() {
                        return Err(ReplayError::InvalidSpec(
                            "native_host_offload supports only aggregated replay in the initial implementation"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Reusable construction state for one worker role.
#[doc(hidden)]
#[derive(Clone)]
pub struct ReplayRoleFactory {
    factory: EngineFactory,
    dp_size: NonZeroU32,
    tensor_parallel_size: u32,
    backend: Backend,
    total_blocks: u64,
}

impl ReplayRoleFactory {
    #[doc(hidden)]
    pub fn build(&self, worker_id: usize) -> ReplayResult<Engine> {
        let worker_id = u64::try_from(worker_id).map_err(|_| {
            ReplayError::Engine(format!(
                "worker id {worker_id} exceeds the native engine range"
            ))
        })?;
        self.factory
            .build(EngineIdentity::new(worker_id), self.dp_size)
            .map_err(engine_error)
    }

    #[doc(hidden)]
    pub fn dp_size(&self) -> u32 {
        self.dp_size.get()
    }

    #[doc(hidden)]
    pub fn gpus_per_worker(&self) -> ReplayResult<usize> {
        usize::try_from(self.dp_size.get())
            .ok()
            .and_then(|dp| {
                usize::try_from(self.tensor_parallel_size)
                    .ok()
                    .and_then(|tp| dp.checked_mul(tp))
            })
            .ok_or_else(|| ReplayError::InvalidSpec("engine GPU count overflows usize".into()))
    }

    #[doc(hidden)]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    #[doc(hidden)]
    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }
}

/// Resolves built-in or Runner-provided timing once, then creates role factories.
#[derive(Clone, Default)]
pub struct ReplayEngineFactory {
    timing: Option<Arc<dyn TimingModel>>,
    prefill_timing: Option<Arc<dyn TimingModel>>,
    decode_timing: Option<Arc<dyn TimingModel>>,
}

impl ReplayEngineFactory {
    pub const fn new() -> Self {
        Self {
            timing: None,
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_timing_model(timing: Arc<dyn TimingModel>) -> Self {
        Self {
            timing: Some(timing),
            prefill_timing: None,
            decode_timing: None,
        }
    }

    pub fn with_optional_role_timing_models(
        prefill: Option<Arc<dyn TimingModel>>,
        decode: Option<Arc<dyn TimingModel>>,
    ) -> Self {
        Self {
            timing: None,
            prefill_timing: prefill,
            decode_timing: decode,
        }
    }

    #[doc(hidden)]
    pub fn role_factory(
        &self,
        config: &ReplayEngineConfig,
        stage: WorkerStage,
        emit_kv_events: bool,
    ) -> ReplayResult<ReplayRoleFactory> {
        let mut role = config.role(stage);
        role.rank.emit_kv_events = emit_kv_events;
        let dp_size = NonZeroU32::new(role.dp_size).ok_or_else(|| {
            ReplayError::InvalidSpec("native engine dp_size must be positive".into())
        })?;
        if role.tensor_parallel_size == 0 {
            return Err(ReplayError::InvalidSpec(
                "native tensor_parallel_size must be positive".into(),
            ));
        }
        let timing = match stage {
            WorkerStage::Aggregated => self.timing.as_ref(),
            WorkerStage::Prefill => self.prefill_timing.as_ref().or(self.timing.as_ref()),
            WorkerStage::Decode => self.decode_timing.as_ref().or(self.timing.as_ref()),
        };
        let backend = role.rank.backend;
        let total_blocks = u64::try_from(role.rank.num_gpu_blocks).map_err(|_| {
            ReplayError::InvalidSpec("engine KV block count exceeds the metrics range".into())
        })?;
        let factory = match timing {
            Some(timing) => EngineFactory::with_timing_model(role.rank, Arc::clone(timing)),
            None => EngineFactory::new(role.rank),
        }
        .map_err(engine_error)?;
        Ok(ReplayRoleFactory {
            factory,
            dp_size,
            tensor_parallel_size: role.tensor_parallel_size,
            backend,
            total_blocks,
        })
    }
}

pub fn run_engine_replay(spec: ReplaySpec) -> ReplayResult<ReplayReport> {
    Replayer::new(spec, ReplayEngineFactory::new())?.run()
}

pub fn run_engine_replay_with_timing(
    spec: ReplaySpec,
    timing: Arc<dyn TimingModel>,
) -> ReplayResult<ReplayReport> {
    Replayer::new(spec, ReplayEngineFactory::with_timing_model(timing))?.run()
}

pub fn run_engine_replay_with_optional_role_timing(
    spec: ReplaySpec,
    prefill: Option<Arc<dyn TimingModel>>,
    decode: Option<Arc<dyn TimingModel>>,
) -> ReplayResult<ReplayReport> {
    Replayer::new(
        spec,
        ReplayEngineFactory::with_optional_role_timing_models(prefill, decode),
    )?
    .run()
}

#[derive(Debug, Default)]
struct KvEventBatch(Vec<crate::engine::KvEvent>);

impl EngineEventBatch for KvEventBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
struct KvEventObservation;

impl ReplayEngineObservation for KvEventObservation {
    type Batch = KvEventBatch;

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn observe_engine_events(
        _stage: WorkerStage,
        _worker_id: usize,
        _dp_rank: u32,
        events: Vec<crate::engine::KvEvent>,
    ) -> Self::Batch {
        KvEventBatch(events)
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|event| match &event.data {
                crate::engine::KvEventData::Stored(stored) => stored.blocks.as_slice(),
                crate::engine::KvEventData::Removed { .. } => &[],
            })
            .map(|block| block.tokens_hash)
            .collect()
    }
}

/// Run the engine-neutral half of Dynamo's live/offline handoff conformance
/// fixture without importing or recompiling Replay implementation sources.
#[doc(hidden)]
pub fn run_engine_handoff_conformance(
    config: ReplayEngineConfig,
    factory: ReplayEngineFactory,
    request: DirectRequest,
) -> ReplayResult<crate::replay::NormalizedHandoffConformance> {
    let prefill_factory = factory.role_factory(&config, WorkerStage::Prefill, true)?;
    let decode_factory = factory.role_factory(&config, WorkerStage::Decode, true)?;
    let backend = prefill_factory.backend();
    if backend != decode_factory.backend() {
        return Err(ReplayError::InvalidSpec(
            "handoff conformance requires matching prefill/decode backends".into(),
        ));
    }
    let runtime_config = OfflineDisaggReplayConfig {
        prefill_factory,
        decode_factory,
        prefill_startup_time_ms: None,
        decode_startup_time_ms: None,
        num_prefill_workers: 1,
        num_decode_workers: 1,
        handoff_latency_ms: 0.0,
    };
    DisaggRuntimeImpl::<
        PoolRoundRobinPlacement<KvEventBatch>,
        KvEventObservation,
        NoReplayMetadata,
    >::new_composed(
        &runtime_config,
        AdmissionQueue::new_requests(
            std::collections::VecDeque::from([request]),
            ReplayMode::Trace,
        ),
        true,
        |_, prefill_topology, _, decode_topology| {
            Ok((
                PoolRoundRobinPlacement::new(prefill_topology),
                PoolRoundRobinPlacement::new(decode_topology),
            ))
        },
    )
    .map_err(runtime_error)?
    .run_handoff_conformance(backend)
    .map_err(runtime_error)
}

fn engine_error(error: impl std::fmt::Display) -> ReplayError {
    ReplayError::Engine(error.to_string())
}

#[cfg(test)]
mod engine_config_tests {
    use super::*;

    /// `role()` inherits the top-level topology only when the whole role block is
    /// absent; a present-but-partial block silently fell back to `dp_size: 1`.
    #[test]
    fn a_partial_role_block_cannot_silently_default_the_topology() {
        let error = ReplayEngineConfig::parse(&serde_json::json!({
            "dp_size": 8,
            "prefill": {"rank": {}},
            "decode": {"dp_size": 8, "rank": {}}
        }))
        .expect_err("an omitted prefill dp_size against a non-default top level is ambiguous")
        .to_string();
        assert!(
            error.contains("prefill") && error.contains("dp_size"),
            "the error must name the role and field, got: {error}"
        );

        // Authoring it explicitly resolves the ambiguity, in either direction.
        for authored in [1, 8] {
            ReplayEngineConfig::parse(&serde_json::json!({
                "dp_size": 8,
                "prefill": {"dp_size": authored, "rank": {}},
                "decode": {"dp_size": 8, "rank": {}}
            }))
            .unwrap_or_else(|error| panic!("explicit dp_size {authored} must parse: {error}"));
        }

        // Nothing changes when the inherited and defaulted values agree, which is
        // every configuration leaving the top-level topology at its default.
        let config = ReplayEngineConfig::parse(&serde_json::json!({
            "prefill": {"rank": {"num_gpu_blocks": 17}},
            "decode": {"rank": {}}
        }))
        .expect("an unambiguous partial block must stay accepted");
        assert_eq!(config.role(WorkerStage::Prefill).dp_size, 1);
    }

    /// `role()` inherits every top-level field when the block is absent, so a
    /// present-but-partial block silently defaults every field it omits -- not
    /// just the two the guard originally covered. `rank` is the worst of them:
    /// it carries the KV budget, backend, and timing model, so omitting it
    /// against an authored top-level rank is exactly the "capacity difference
    /// with no warning" harm this guard exists to refuse.
    #[test]
    fn a_partial_role_block_may_not_omit_rank_or_the_explicit_block_flag() {
        for (field, descriptor) in [
            (
                "rank",
                serde_json::json!({
                    "rank": {"num_gpu_blocks": 100000},
                    "prefill": {"dp_size": 1, "rank": {}},
                    "decode": {"dp_size": 1}
                }),
            ),
            (
                "num_gpu_blocks_is_explicit",
                serde_json::json!({
                    "num_gpu_blocks_is_explicit": true,
                    "prefill": {"dp_size": 1, "rank": {}},
                    "decode": {"dp_size": 1, "rank": {}}
                }),
            ),
        ] {
            let error = match ReplayEngineConfig::parse(&descriptor) {
                Err(error) => error.to_string(),
                Ok(_) => panic!("omitting {field} against a non-default top level is ambiguous"),
            };
            assert!(
                error.contains(field) && (error.contains("prefill") || error.contains("decode")),
                "the error must name the role and {field}, got: {error}"
            );
        }

        // Authoring the omitted field explicitly resolves the ambiguity.
        ReplayEngineConfig::parse(&serde_json::json!({
            "rank": {"num_gpu_blocks": 100000},
            "prefill": {"dp_size": 1, "rank": {"num_gpu_blocks": 100000}},
            "decode": {"dp_size": 1, "rank": {}}
        }))
        .expect("an explicitly authored rank block must parse");
    }
}
