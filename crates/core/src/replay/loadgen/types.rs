// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xxhash_rust::xxh3::Xxh3;

use super::trace::synthesize_validated_trace_tokens;
use crate::replay::protocol::DirectRequest;

pub const OUTPUT_REPLAY_ID_ANNOTATION_KEY: &str = "output_replay_id";
pub const OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY: &str = "output_replay_consumer";

pub fn output_replay_id_annotation(replay_key: &str) -> String {
    format!("{OUTPUT_REPLAY_ID_ANNOTATION_KEY}:{replay_key}")
}

pub fn effective_replay_key(
    request_id: Option<&str>,
    session_id: Option<&str>,
    turn_index: usize,
    line_index: usize,
) -> String {
    if let Some(request_id) = request_id.map(str::trim).filter(|value| !value.is_empty()) {
        return request_id.to_string();
    }
    if let Some(session_id) = session_id.map(str::trim).filter(|value| !value.is_empty()) {
        return format!("{session_id}:{turn_index}");
    }
    format!("line:{line_index}")
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub block_size: usize,
    pub sessions: Vec<SessionTrace>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedAgenticGraph {
    pub(super) block_size: usize,
    pub(super) source: AgenticSourceProvenance,
    pub(super) graph_digest: String,
    pub(super) nodes: Vec<AgenticNode>,
    pub(super) plays: Vec<AgenticPlay>,
}

pub type AgenticTrace = ValidatedAgenticGraph;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TraceFileFormat {
    Mooncake,
    /// Mooncake-shaped rows where follow-up turns contain new input deltas.
    /// Offline replay accumulates each generated output and the next input delta
    /// per session before computing engine block hashes. Use this only for delta
    /// traces: it expands compact session turns into cumulative prompts and can
    /// use much more memory than `Mooncake`.
    MooncakeDelta,
    /// Versioned Mooncake request/cache rows plus typed request-level workflow
    /// dependencies.
    AgenticMooncake,
    /// Public Weka/AgentX trace files or directories lowered into the same
    /// validated graph used by `AgenticMooncake`.
    Weka,
    AppliedComputeAgentic,
    Dynamo,
}

/// One producer-neutral Mooncake request row.
///
/// This mirrors the external JSONL schema without depending on Dynamo's
/// producer-side `dynamo-data-gen` crate. Dynamo request-trace lowering lives
/// in the Mocker adapter and converts into this DTO.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MooncakeRow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "created_time"
    )]
    pub timestamp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "delay_ms")]
    pub delay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
}

pub const AGENTIC_MOONCAKE_SCHEMA: &str = "dynamo.agentic_mooncake";
pub const AGENTIC_MOONCAKE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgenticSourceProvenance {
    pub format: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgenticMooncakeHeader {
    pub schema: String,
    pub version: u32,
    pub block_size: usize,
    pub hash_id_scope: AgenticHashIdScope,
    pub source: AgenticSourceProvenance,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgenticHashIdScope {
    Local,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgenticDependencyTrigger {
    Dispatch,
    Completion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgenticDependencyRelation {
    Sequence,
    Spawn,
    Join,
    ReplayBarrier,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgenticDependency {
    pub request_id: String,
    pub trigger: AgenticDependencyTrigger,
    pub delay_ms: f64,
    pub relation: AgenticDependencyRelation,
}

/// One producer-neutral agentic Mooncake request row.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticMooncakeRow {
    pub request_id: String,
    pub play_id: String,
    /// Zero-based source corpus order for this play. When present, every play
    /// must provide one unique contiguous ordinal. This keeps scheduling order
    /// independent of producer-generated play IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_play_ordinal: Option<usize>,
    pub session_id: String,
    pub model: String,
    #[serde(default, alias = "input_tokens")]
    pub input_length: Option<usize>,
    #[serde(default, alias = "output_tokens")]
    pub output_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub hash_ids: Option<Vec<u64>>,
    pub not_before_ms: f64,
    /// Source-recorded API service time. This is provenance for snapshot
    /// reconstruction; replay completion continues to be driven by the
    /// configured engine timing model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_api_time_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_class: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<AgenticDependency>,
}

impl TraceFileFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mooncake => "mooncake",
            Self::MooncakeDelta => "mooncake-delta",
            Self::AgenticMooncake => "agentic_mooncake",
            Self::Weka => "weka",
            Self::AppliedComputeAgentic => "applied_compute_agentic",
            Self::Dynamo => "dynamo",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionTrace {
    pub session_id: String,
    pub first_arrival_timestamp_ms: Option<f64>,
    pub turns: Vec<TurnTrace>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnTrace {
    pub input_length: usize,
    pub max_output_tokens: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub replay_key: Option<String>,
    pub hash_ids: Vec<u32>,
    pub delay_after_previous_ms: f64,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct AgenticNode {
    pub(super) request_id: String,
    pub(super) play_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_play_ordinal: Option<usize>,
    pub(super) session_id: String,
    pub(super) model: String,
    pub(super) input_length: usize,
    pub(super) max_output_tokens: usize,
    pub(super) output_token_ids: Option<Vec<u32>>,
    pub(super) replay_key: Option<String>,
    pub(super) hash_ids: Vec<u64>,
    pub(super) not_before_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recorded_api_time_ms: Option<f64>,
    pub(super) priority: i32,
    pub(super) strict_priority: u32,
    pub(super) policy_class: Option<String>,
    pub(super) dependencies: Vec<AgenticDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgenticPlay {
    pub(super) play_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) source_play_ordinal: Option<usize>,
    pub(super) root_nodes: Vec<usize>,
    pub(super) nodes: Vec<usize>,
}

impl ValidatedAgenticGraph {
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn source(&self) -> &AgenticSourceProvenance {
        &self.source
    }

    pub fn graph_digest(&self) -> &str {
        &self.graph_digest
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn play_count(&self) -> usize {
        self.plays.len()
    }

    pub fn nodes(&self) -> &[AgenticNode] {
        &self.nodes
    }

    pub fn identity(&self) -> AgenticGraphIdentity {
        AgenticGraphIdentity {
            source: self.source.clone(),
            graph_digest: self.graph_digest.clone(),
            block_size: self.block_size,
            node_count: self.nodes.len(),
            play_count: self.plays.len(),
            source_models: self
                .nodes
                .iter()
                .map(|node| node.model.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
        }
    }
}

impl AgenticNode {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn play_id(&self) -> &str {
        &self.play_id
    }

    pub fn source_play_ordinal(&self) -> Option<usize> {
        self.source_play_ordinal
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn input_length(&self) -> usize {
        self.input_length
    }

    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    pub fn hash_ids(&self) -> &[u64] {
        &self.hash_ids
    }

    pub fn dependencies(&self) -> &[AgenticDependency] {
        &self.dependencies
    }

    pub fn not_before_ms(&self) -> f64 {
        self.not_before_ms
    }

    pub fn recorded_api_time_ms(&self) -> Option<f64> {
        self.recorded_api_time_ms
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgenticTrajectorySnapshot {
    pub total_trajectories: usize,
    pub completed_trajectories: usize,
    pub e2e_latencies_ms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgenticGraphIdentity {
    pub source: AgenticSourceProvenance,
    pub graph_digest: String,
    pub block_size: usize,
    pub node_count: usize,
    pub play_count: usize,
    /// Sorted source model labels retained as workload provenance. The replay
    /// execution boundary separately declares how these labels map to the
    /// configured target timing model.
    pub source_models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LengthSpec {
    pub mean: usize,
    pub stddev: f64,
}

#[derive(Debug, Clone)]
pub enum ArrivalSpec {
    Burst,
    ConstantQps { qps: f64 },
    PoissonQps { qps: f64 },
    GammaQps { qps: f64, smoothness: f64 },
}

#[derive(Debug, Clone)]
pub enum DelaySpec {
    None,
    ConstantMs(f64),
    ExponentialMs { mean_ms: f64 },
}

#[derive(Debug, Clone)]
pub struct SyntheticTraceSpec {
    pub block_size: usize,
    pub num_sessions: usize,
    pub turns_per_session: usize,
    pub input_tokens: LengthSpec,
    pub output_tokens: LengthSpec,
    pub shared_prefix_ratio: f64,
    pub num_prefix_groups: usize,
    pub first_turn_arrivals: ArrivalSpec,
    pub inter_turn_delays: DelaySpec,
    pub seed: u64,
    pub arrival_seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum SessionPartitionSpec {
    Random { num_partitions: usize, seed: u64 },
    RoundRobin { num_partitions: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayRequestHashes {
    /// Token-only hashes for complete engine blocks.
    ///
    /// These remain raw identities here; a Dynamo placement adapter converts
    /// them to `LocalBlockHash` at its API boundary.
    pub local_block_hashes: Vec<u64>,
    /// Rolling, sequence-aware identities for the same complete blocks.
    pub sequence_hashes: Vec<u64>,
}

impl ReplayRequestHashes {
    /// Materialize stable replay block identities without importing a concrete
    /// Router implementation or its wire types.
    pub fn from_tokens(tokens: &[u32], engine_block_size: u32) -> Self {
        if engine_block_size == 0 {
            return Self {
                local_block_hashes: Vec::new(),
                sequence_hashes: Vec::new(),
            };
        }

        let block_size = engine_block_size as usize;
        let local_block_hashes = tokens
            .chunks_exact(block_size)
            .map(|block| {
                let mut bytes = Vec::with_capacity(std::mem::size_of_val(block));
                for token in block {
                    bytes.extend_from_slice(&token.to_le_bytes());
                }
                xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, 1337)
            })
            .collect::<Vec<_>>();
        let mut sequence_hashes: Vec<u64> = Vec::with_capacity(local_block_hashes.len());
        for &block_hash in &local_block_hashes {
            let sequence_hash =
                sequence_hashes
                    .last()
                    .copied()
                    .map_or(block_hash, |parent: u64| {
                        let mut bytes = [0_u8; std::mem::size_of::<[u64; 2]>()];
                        bytes[..8].copy_from_slice(&parent.to_le_bytes());
                        bytes[8..].copy_from_slice(&block_hash.to_le_bytes());
                        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, 1337)
                    });
            sequence_hashes.push(sequence_hash);
        }

        Self {
            local_block_hashes,
            sequence_hashes,
        }
    }

    /// Derives engine-block identities directly from compact trace blocks.
    ///
    /// The result is byte-for-byte equivalent to first synthesizing the trace
    /// tokens and passing them to [`Self::from_tokens`], without allocating a
    /// full prompt-sized token vector.
    pub fn from_trace_blocks(
        input_length: usize,
        hash_ids: &[u32],
        trace_block_size: usize,
        engine_block_size: u32,
    ) -> Self {
        if engine_block_size == 0 || trace_block_size == 0 {
            return Self {
                local_block_hashes: Vec::new(),
                sequence_hashes: Vec::new(),
            };
        }

        let engine_block_size = engine_block_size as usize;
        let complete_blocks = input_length / engine_block_size;
        let mut local_block_hashes = Vec::with_capacity(complete_blocks);
        let mut hasher = Xxh3::with_seed(1337);
        let mut engine_tokens = 0;
        let mut remaining_prompt = input_length;
        let mut trace_bytes = vec![0_u8; trace_block_size.saturating_mul(4)];

        for &hash_id in hash_ids {
            if remaining_prompt == 0 {
                break;
            }
            let trace_tokens = remaining_prompt.min(trace_block_size);
            for bytes in trace_bytes[..trace_tokens * 4].chunks_exact_mut(4) {
                bytes.copy_from_slice(&hash_id.to_le_bytes());
            }
            let mut consumed = 0;
            while consumed < trace_tokens {
                let take = (engine_block_size - engine_tokens).min(trace_tokens - consumed);
                hasher.update(&trace_bytes[consumed * 4..(consumed + take) * 4]);
                consumed += take;
                engine_tokens += take;
                if engine_tokens == engine_block_size {
                    local_block_hashes.push(hasher.digest());
                    hasher.reset();
                    engine_tokens = 0;
                }
            }
            remaining_prompt -= trace_tokens;
        }

        let mut sequence_hashes = Vec::with_capacity(local_block_hashes.len());
        for &block_hash in &local_block_hashes {
            let sequence_hash =
                sequence_hashes
                    .last()
                    .copied()
                    .map_or(block_hash, |parent: u64| {
                        let mut bytes = [0_u8; std::mem::size_of::<[u64; 2]>()];
                        bytes[..8].copy_from_slice(&parent.to_le_bytes());
                        bytes[8..].copy_from_slice(&block_hash.to_le_bytes());
                        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, 1337)
                    });
            sequence_hashes.push(sequence_hash);
        }

        Self {
            local_block_hashes,
            sequence_hashes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadyTurn {
    pub request_uuid: Uuid,
    pub authored_request_id: Option<String>,
    pub play_id: Option<String>,
    pub dispatched_at_ms: f64,
    pub session_id: String,
    pub turn_index: usize,
    pub emit_session_metadata: bool,
    pub replay_key: Option<String>,
    pub scheduled_ready_at_ms: f64,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub request: DirectRequest,
}

/// A request whose prompt may still be represented by one hash id per trace
/// block. Offline replay keeps this compact form while an aggregated or
/// prefill router queues the request and materializes tokens only when a
/// worker admits it.
/// Safely exposes compact trace-block IDs whose backing storage is retained
/// until the last owner drops it.
///
/// The steppable runtime retains an [`Arc`] of this lease for every accepted
/// compact request and drops it on terminal completion, cancellation, or
/// replay destruction. Providers can therefore couple their own release
/// action to [`Drop`] without exposing a raw pointer or an unsafe lifetime to
/// the replay core.
pub trait CompactHashIdsLease: std::fmt::Debug {
    /// Immutable canonical trace-block IDs for one compact request.
    fn hash_ids(&self) -> &[u32];

    /// Records that the replay accepted this lease.
    ///
    /// A provider with a release action can use this hook to distinguish a
    /// rejected submission from a retained request. The default is sufficient
    /// for ordinary `Arc`-backed providers.
    fn on_accepted(&self) {}
}

/// Compact trace-block IDs retained by a deferred replay request.
///
/// The owned form preserves the historical compact-submission path. A leased
/// form keeps a provider-owned safe lifecycle guard alive while the replay may
/// still need to materialize the prompt.
#[doc(hidden)]
#[derive(Debug)]
pub enum CompactHashIds {
    Owned(Vec<u32>),
    Leased(Arc<dyn CompactHashIdsLease>),
}

impl CompactHashIds {
    pub(crate) fn as_slice(&self) -> &[u32] {
        match self {
            Self::Owned(ids) => ids,
            Self::Leased(lease) => lease.hash_ids(),
        }
    }

    pub(crate) fn retained_lease(&self) -> Option<Arc<dyn CompactHashIdsLease>> {
        match self {
            Self::Owned(_) => None,
            Self::Leased(lease) => Some(Arc::clone(lease)),
        }
    }
}

impl From<Vec<u32>> for CompactHashIds {
    fn from(value: Vec<u32>) -> Self {
        Self::Owned(value)
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub enum ReplayRequestPayload {
    Materialized(DirectRequest),
    Deferred {
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: CompactHashIds,
        trace_block_size: usize,
    },
}

impl ReplayRequestPayload {
    pub fn materialized(request: DirectRequest) -> Self {
        Self::Materialized(request)
    }

    pub(crate) fn deferred(
        request_metadata: DirectRequest,
        input_length: usize,
        hash_ids: impl Into<CompactHashIds>,
        trace_block_size: usize,
    ) -> Self {
        debug_assert!(request_metadata.tokens.is_empty());
        Self::Deferred {
            request_metadata,
            input_length,
            hash_ids: hash_ids.into(),
            trace_block_size,
        }
    }

    pub fn input_length(&self) -> usize {
        match self {
            Self::Materialized(request) => request.tokens.len(),
            Self::Deferred { input_length, .. } => *input_length,
        }
    }

    pub fn metadata(&self) -> &DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                request_metadata, ..
            } => request_metadata,
        }
    }

    pub fn metadata_mut(&mut self) -> &mut DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                request_metadata, ..
            } => request_metadata,
        }
    }

    pub fn materialized_tokens(&self) -> Option<&[u32]> {
        match self {
            Self::Materialized(request) => Some(&request.tokens),
            Self::Deferred { .. } => None,
        }
    }

    pub fn materialized_request(&self) -> Option<&DirectRequest> {
        match self {
            Self::Materialized(request) => Some(request),
            Self::Deferred { .. } => None,
        }
    }

    pub fn prompt_tokens(&self) -> Vec<u32> {
        match self {
            Self::Materialized(request) => request.tokens.clone(),
            Self::Deferred {
                input_length,
                hash_ids,
                trace_block_size,
                ..
            } => synthesize_validated_trace_tokens(
                *input_length,
                hash_ids.as_slice(),
                *trace_block_size,
            ),
        }
    }

    pub fn replay_hashes(&self, engine_block_size: u32) -> ReplayRequestHashes {
        match self {
            Self::Materialized(request) => {
                ReplayRequestHashes::from_tokens(&request.tokens, engine_block_size)
            }
            Self::Deferred {
                input_length,
                hash_ids,
                trace_block_size,
                ..
            } => ReplayRequestHashes::from_trace_blocks(
                *input_length,
                hash_ids.as_slice(),
                *trace_block_size,
                engine_block_size,
            ),
        }
    }

    pub fn into_direct_request(self) -> DirectRequest {
        match self {
            Self::Materialized(request) => request,
            Self::Deferred {
                mut request_metadata,
                input_length,
                hash_ids,
                trace_block_size,
            } => {
                request_metadata.tokens = synthesize_validated_trace_tokens(
                    input_length,
                    hash_ids.as_slice(),
                    trace_block_size,
                );
                request_metadata
            }
        }
    }

    pub fn materialize(&mut self) -> Option<&DirectRequest> {
        if matches!(self, Self::Deferred { .. }) {
            let payload = std::mem::replace(self, Self::Materialized(DirectRequest::default()));
            *self = Self::Materialized(payload.into_direct_request());
        }
        self.materialized_request()
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct CompactReadyTurn {
    pub request_uuid: Uuid,
    pub authored_request_id: Option<String>,
    pub play_id: Option<String>,
    pub dispatched_at_ms: f64,
    pub session_id: String,
    pub turn_index: usize,
    pub replay_key: Option<String>,
    pub scheduled_ready_at_ms: f64,
    pub replay_hashes: Option<ReplayRequestHashes>,
    pub emit_session_metadata: bool,
    pub request: ReplayRequestPayload,
}

impl CompactReadyTurn {
    #[doc(hidden)]
    pub fn into_ready_turn(self) -> ReadyTurn {
        ReadyTurn {
            request_uuid: self.request_uuid,
            authored_request_id: self.authored_request_id,
            play_id: self.play_id,
            dispatched_at_ms: self.dispatched_at_ms,
            session_id: self.session_id,
            turn_index: self.turn_index,
            emit_session_metadata: self.emit_session_metadata,
            replay_key: self.replay_key,
            scheduled_ready_at_ms: self.scheduled_ready_at_ms,
            replay_hashes: self.replay_hashes,
            request: self.request.into_direct_request(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReplayRequestHashes;
    use crate::replay::loadgen::trace::synthesize_trace_tokens;

    #[test]
    fn compact_trace_hashes_match_synthesized_tokens_when_blocks_misalign() {
        let input_length = 23;
        let trace_block_size = 6;
        let hash_ids = [7, 11, 13, 17];
        let engine_block_size = 8;
        let tokens = synthesize_trace_tokens(input_length, &hash_ids, trace_block_size).unwrap();

        assert_eq!(
            ReplayRequestHashes::from_trace_blocks(
                input_length,
                &hash_ids,
                trace_block_size,
                engine_block_size,
            ),
            ReplayRequestHashes::from_tokens(&tokens, engine_block_size),
        );
    }
}
