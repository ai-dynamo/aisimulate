// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Default generalized-engine construction for offline replay.
//!
//! This module contains configuration and conversion helpers only. Scheduler
//! state lives in `aisimulate_core::engine`; virtual time and worker lifecycle live
//! in the moved aggregated/disaggregated replay runtimes.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::engine::belady::BeladyOracle;
use crate::engine::generalized::EngineIdentity;
use crate::engine::{
    Backend, Engine, EngineConfig, EngineFactory, KvEvictionPolicy, TimingModel, WorkerType,
};
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
    /// Belady ranks worker-local victims by global future input demand. It does
    /// not forecast routing, output reuse, later chunks, or recomputation, and
    /// therefore promises neither worker-local optimality nor higher throughput.
    /// Lookahead changes eviction priority only; execution remains causal.
    pub kv_eviction_policy: KvEvictionPolicy,
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
            kv_eviction_policy: KvEvictionPolicy::Lru,
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
        serde_json::from_value(value.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!("invalid native engine descriptor: {error}"))
        })
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
        if self.kv_eviction_policy == KvEvictionPolicy::Belady {
            if !matches!(topology, ReplayTopology::Aggregated { .. }) || self.dp_size != 1 {
                return Err(ReplayError::InvalidSpec(
                    "belady requires aggregated replay with dp_size=1 per worker".into(),
                ));
            }
            if self.rank.native_host_offload.is_some() || self.rank.g3_offload.is_some() {
                return Err(ReplayError::InvalidSpec(
                    "belady does not support native_host_offload or g3_offload".into(),
                ));
            }
            if !self.rank.enable_prefix_caching {
                return Err(ReplayError::InvalidSpec(
                    "belady requires enable_prefix_caching=true".into(),
                ));
            }
        }
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
    needs_belady_oracle: bool,
    dp_size: NonZeroU32,
    tensor_parallel_size: u32,
    backend: Backend,
    total_blocks: u64,
    kv_cache_capacity_bytes: Option<u64>,
    pub(crate) g3_config: Option<crate::engine::G3OffloadConfig>,
    pub(crate) g3_block_bytes: usize,
    pub(crate) g3_tier: Option<crate::engine::g3_offload::SharedG3Tier>,
}

impl ReplayRoleFactory {
    pub(crate) fn reset_timing_evidence(&self) -> anyhow::Result<()> {
        self.factory.reset_timing_evidence()
    }

    pub(crate) fn with_belady_oracle(mut self, oracle: BeladyOracle) -> Self {
        self.factory = self.factory.with_belady_oracle(oracle);
        self.needs_belady_oracle = false;
        self
    }

    #[doc(hidden)]
    pub fn build(&self, worker_id: usize) -> ReplayResult<Engine> {
        if self.needs_belady_oracle {
            return Err(ReplayError::InvalidSpec(
                "belady requires a prepared input forecast; construct workers through Replayer"
                    .into(),
            ));
        }
        if self.g3_config.is_some() && self.g3_tier.is_none() {
            return Err(ReplayError::InvalidSpec(
                "g3_offload requires a Replay deployment registry".into(),
            ));
        }
        let worker_id = u64::try_from(worker_id).map_err(|_| {
            ReplayError::Engine(format!(
                "worker id {worker_id} exceeds the native engine range"
            ))
        })?;
        let mut engine = self
            .factory
            .build(EngineIdentity::new(worker_id), self.dp_size)
            .map_err(engine_error)?;
        if let Some(registry) = &self.g3_tier {
            registry
                .lock()
                .unwrap()
                .register_worker(worker_id as usize)
                .map_err(|error| ReplayError::Engine(error.to_string()))?;
            for rank in engine.ranks_mut() {
                rank.set_g3_offload(Arc::clone(registry), worker_id as usize);
            }
        }
        Ok(engine)
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

    pub(crate) fn kv_cache_capacity_bytes(&self) -> Option<u64> {
        self.kv_cache_capacity_bytes
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
        let kv_cache_capacity_bytes = role.rank.kv_cache_capacity_bytes;
        let total_blocks = if role.rank.kv_cache_groups.is_empty() {
            u64::try_from(role.rank.num_gpu_blocks).map_err(|_| {
                ReplayError::InvalidSpec("engine KV block count exceeds the metrics range".into())
            })?
        } else {
            0
        };
        role.rank.validate().map_err(engine_error)?;
        let g3_config = role.rank.g3_offload.take();
        let g3_block_bytes = role
            .rank
            .block_size
            .checked_mul(role.rank.kv_cache_bytes_per_token.unwrap_or(0))
            .ok_or_else(|| ReplayError::InvalidSpec("G3 block bytes overflow".into()))?;
        let factory = match timing {
            Some(timing) => EngineFactory::with_timing_model(role.rank, Arc::clone(timing)),
            None => EngineFactory::new(role.rank),
        }
        .map_err(engine_error)?;
        Ok(ReplayRoleFactory {
            factory,
            needs_belady_oracle: config.kv_eviction_policy == KvEvictionPolicy::Belady,
            dp_size,
            tensor_parallel_size: role.tensor_parallel_size,
            backend,
            total_blocks,
            kv_cache_capacity_bytes,
            g3_config,
            g3_block_bytes,
            g3_tier: None,
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
mod belady_tests {
    use super::*;

    #[test]
    fn belady_direct_role_factory_cannot_silently_build_an_lru_engine() {
        let config = ReplayEngineConfig {
            kv_eviction_policy: KvEvictionPolicy::Belady,
            ..Default::default()
        };
        let factory = ReplayEngineFactory::new()
            .role_factory(&config, WorkerStage::Aggregated, false)
            .unwrap();
        let error = match factory.build(0) {
            Ok(_) => panic!("Belady must not construct a worker without its input forecast"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("through Replayer"));
        assert!(
            factory
                .with_belady_oracle(BeladyOracle::new(Vec::new()).unwrap())
                .build(0)
                .is_ok()
        );
    }
}
