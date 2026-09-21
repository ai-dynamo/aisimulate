// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod arrival;
mod driver;
mod dynamo;
mod generated;
mod lineage;
mod phase;
mod snapshot;
mod steppable;
mod trace;
mod types;
mod weka;

use rand::Rng;
use rand::rngs::StdRng;

pub use driver::{
    AGENTIC_LIFECYCLE_SCHEMA_V1, AgenticLifecycleEvent, AgenticLifecycleEventKind,
    AgenticLifecycleTranscript, AgenticOutputFeedback, AgenticRuntimeFeedback,
    AgenticTerminalFeedback, WorkloadDriver,
};
pub use dynamo::DynamoRequestTrace;
pub use generated::GeneratedRequests;
pub use phase::{
    AGENTIC_PHASE_SCHEMA_V1, AGENTIC_WARMUP_REQUESTS_PER_LANE, AgenticPhaseEvidence,
    AgenticPhaseLane, AgenticPhaseRequest, AgenticPreparationTransition, AgenticReplayPhase,
};
pub use snapshot::{
    AGENTIC_SNAPSHOT_SCHEMA_V1, AgenticPlaySnapshot, AgenticPrimer, AgenticReplayContext,
    AgenticSnapshotEvidence, AgenticSnapshotOptions, AgenticSnapshotRequest,
    PreparedAgenticSnapshots,
};
pub use steppable::{EngineEvent, StepOutcome, SteppableAgg, SteppableEngine, SteppableReplay};
pub use trace::{AgenticGraphBuilder, load_agentic_mooncake, validate_trace_files};
#[doc(hidden)]
pub use types::CompactReadyTurn;
pub use types::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticGraphIdentity, AgenticHashIdScope,
    AgenticMooncakeHeader, AgenticMooncakeRow, AgenticNode, AgenticPlay, AgenticPlayOutcome,
    AgenticPlayStatus, AgenticSourceProvenance, AgenticTrace, AgenticTrajectorySnapshot,
    ArrivalSpec, DelaySpec, LengthSpec, MooncakeRow, OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY,
    OUTPUT_REPLAY_ID_ANNOTATION_KEY, ReadyTurn, ReplayRequestHashes, ReplayRequestPayload,
    SessionPartitionSpec, SessionTrace, SyntheticTraceSpec, Trace, TraceFileFormat, TurnTrace,
    ValidatedAgenticGraph, effective_replay_key, output_replay_id_annotation,
};
pub use weka::{
    WekaImportOptions, WekaImportSummary, WekaImporter, WekaNestedTimestampBasis,
    WekaResolvedTimestampBasis, load_weka_agentic_graph, load_weka_agentic_graph_with_options,
    load_weka_agentic_rows, stream_weka_agentic_rows,
};

pub(super) const SYNTHETIC_OUTPUT_SEED: u64 = 0xD37A_0A7E_5EED;

pub(super) fn planned_output_token_ids(
    authored: Option<Vec<u32>>,
    max_output_tokens: usize,
    output_rng: &mut StdRng,
) -> Vec<u32> {
    authored.unwrap_or_else(|| {
        (0..max_output_tokens)
            .map(|_| output_rng.random::<u32>())
            .collect()
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod snapshot_tests;

#[cfg(test)]
mod dynamo_snapshot_tests;
