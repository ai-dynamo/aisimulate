// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-neutral mock inference schedulers and attention-DP composition.

pub(crate) mod belady;
mod cache;
mod common;
mod config;
pub(crate) mod g3_offload;
pub mod generalized;
mod handoff;
mod host_offload;
mod kv_manager;
mod launch;
mod offload_transfer;
mod protocol;
mod runtime;
mod scheduler;
mod timing;
mod trace;

pub use host_offload::SharedG2Pool;
pub(crate) use host_offload::{
    G2Binding, G2Registry, HostBlockKey, HostOffloadObservation, HostOffloadObservationData,
    HostOffloadObserver,
};
pub use launch::EngineLaunchConfig;

pub use belady::KvEvictionPolicy;
pub(crate) use common::hashing::{
    XXH3_SEED, block_hashes, compute_block_hash_for_tokens, compute_next_sequence_hash,
};
pub use common::running_mean::RunningMean;
pub use common::speculative::normalize_conditional_accept_rates;
pub use config::{
    Backend, EngineConfig, G2Scope, G3OffloadConfig, G3Scope, NativeHostOffloadConfig,
    PreemptionMode, SglangConfig, SglangSchedulePolicy, StateCacheConfig, TrtllmCapacityPolicy,
    TrtllmConfig, WorkerType,
};
pub use g3_offload::{G3IoStats, G3Stats};
pub use handoff::{HandoffId, HandoffTransferTiming, TransferTimingMode, prefill_handoff_delay_ms};
pub use protocol::{
    Admission, CacheTierAttribution, Command, CommandEffects, CommandResult, DecodeAcceptance,
    ForwardPassMetrics, KvBlock, KvEvent, KvEventData, KvEventTier, LifecycleEvent, Metrics,
    Output, PassCompletionEffects, PassStartEffects, PressureEvent, PressureKind, PressureState,
    Request, StoredBlocks,
};
pub use runtime::{Engine, EngineFactory};
pub use scheduler::SchedulerRank;
pub use timing::{
    TimingEvidenceSource, TimingEvidenceSummary, TimingModel, TimingModelConfig,
    TimingOperationEvidence, TimingPhaseEvidence,
};

#[doc(hidden)]
pub use protocol::PendingPass;
pub(in crate::engine) use timing::modeled_duration_ms;
#[doc(hidden)]
pub use trace::g1_parent_chain_events;
