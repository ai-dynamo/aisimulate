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
mod protocol;
mod runtime;
mod scheduler;
mod timing;
mod trace;

pub(crate) use host_offload::{
    HostBlockKey, HostOffloadObservation, HostOffloadObservationData, HostOffloadObserver,
};

pub use belady::KvEvictionPolicy;
pub use common::running_mean::RunningMean;
pub use common::speculative::normalize_conditional_accept_rates;
pub use config::{
    Backend, EngineConfig, G3OffloadConfig, G3Scope, NativeHostOffloadConfig, PreemptionMode,
    SglangConfig, SglangSchedulePolicy, TrtllmCapacityPolicy, TrtllmConfig, WorkerType,
};
pub use g3_offload::{G3IoStats, G3Stats};
pub use handoff::{HandoffId, HandoffTransferTiming, TransferTimingMode, prefill_handoff_delay_ms};
pub use protocol::{
    Admission, CacheTierAttribution, Command, CommandEffects, CommandResult, ForwardPassMetrics,
    KvBlock, KvEvent, KvEventData, LifecycleEvent, Metrics, Output, PassCompletionEffects,
    PassStartEffects, PressureEvent, PressureKind, PressureState, Request, StoredBlocks,
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
