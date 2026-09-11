// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::replay::protocol::{DirectRequest, ReplayPromptTokenSource};
use crate::replay::{ReplayError, ReplayResult, SlaThresholds};

pub const CURRENT_REPLAY_SPEC_VERSION: u32 = 1;

/// Serializable input to one replay execution.
///
/// Provider descriptors are data only. A runner resolves them to concrete
/// placement/scaling implementations before constructing [`crate::replay::Replayer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySpec {
    #[serde(default = "default_spec_version")]
    pub version: u32,
    pub topology: ReplayTopology,
    #[serde(default)]
    pub engine: Value,
    #[serde(default)]
    pub adapters: ReplayAdapters,
    /// Soft virtual-time cutoff. Events at the cutoff are processed; replay
    /// stops before the first event after it and leaves in-flight requests
    /// non-terminal in the report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sim_time_ms: Option<f64>,
    /// Optional source-side in-flight cap. Requests whose authored arrival is
    /// ready remain outside the simulated system until an earlier request is
    /// terminal, matching closed-loop and replay-concurrency workloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_in_flight: Option<usize>,
    /// Whether the report should retain one record per arrived request.
    ///
    /// This defaults to `true` to preserve reports produced by version-1
    /// execution specs written before this control was added.
    #[serde(
        default = "default_record_per_request",
        skip_serializing_if = "is_true"
    )]
    pub record_per_request: bool,
    /// Optional latency targets used to calculate goodput.
    #[serde(default, skip_serializing_if = "SlaThresholds::is_unset")]
    pub sla: SlaThresholds,
    pub requests: Vec<ReplayRequest>,
}

impl ReplaySpec {
    pub fn validate(&self) -> ReplayResult<()> {
        if self.version != CURRENT_REPLAY_SPEC_VERSION {
            return Err(ReplayError::InvalidSpec(format!(
                "unsupported version {}; expected {}",
                self.version, CURRENT_REPLAY_SPEC_VERSION
            )));
        }
        self.topology.validate()?;
        if let Some(max_sim_time_ms) = self.max_sim_time_ms {
            validate_time("max_sim_time_ms", max_sim_time_ms)?;
        }
        if self.max_in_flight == Some(0) {
            return Err(ReplayError::InvalidSpec(
                "max_in_flight must be positive".to_string(),
            ));
        }
        self.sla.validate()?;

        let mut ids = BTreeSet::new();
        for request in &self.requests {
            request.validate()?;
            if matches!(self.topology, ReplayTopology::Aggregated { .. })
                && request.prefill_dp_rank.is_some()
            {
                return Err(ReplayError::InvalidSpec(format!(
                    "aggregated request {:?} cannot specify prefill_dp_rank",
                    request.id
                )));
            }
            if !ids.insert(request.id.clone()) {
                return Err(ReplayError::InvalidSpec(format!(
                    "duplicate request id {:?}",
                    request.id
                )));
            }
        }
        Ok(())
    }
}

fn default_spec_version() -> u32 {
    CURRENT_REPLAY_SPEC_VERSION
}

fn default_record_per_request() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayTopology {
    Aggregated {
        workers: WorkerPoolSpec,
    },
    Disaggregated {
        prefill: WorkerPoolSpec,
        decode: WorkerPoolSpec,
        /// Transfer latency used only when the selected engine cannot derive
        /// one from KV bytes and bandwidth.
        #[serde(default)]
        handoff_latency_ms: f64,
    },
}

impl ReplayTopology {
    pub fn aggregated(workers: usize) -> Self {
        Self::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                ..WorkerPoolSpec::default()
            },
        }
    }

    pub fn validate(&self) -> ReplayResult<()> {
        match self {
            Self::Aggregated { workers } => workers.validate("aggregated"),
            Self::Disaggregated {
                prefill,
                decode,
                handoff_latency_ms,
            } => {
                prefill.validate("prefill")?;
                decode.validate("decode")?;
                validate_time("handoff_latency_ms", *handoff_latency_ms)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerPoolSpec {
    pub initial_workers: usize,
    #[serde(default)]
    pub startup_delay_ms: f64,
}

impl Default for WorkerPoolSpec {
    fn default() -> Self {
        Self {
            initial_workers: 1,
            startup_delay_ms: 0.0,
        }
    }
}

impl WorkerPoolSpec {
    fn validate(&self, name: &str) -> ReplayResult<()> {
        if self.initial_workers == 0 {
            return Err(ReplayError::InvalidSpec(format!(
                "{name} pool must start with at least one worker"
            )));
        }
        validate_time(&format!("{name} startup_delay_ms"), self.startup_delay_ms)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub id: String,
    pub arrival_time_ms: f64,
    pub input_tokens: usize,
    /// Materialized prompt content for KV-aware replay. Length-only engine
    /// replay may omit it; adapters must never invent tokens when it is needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_ids: Option<Vec<u32>>,
    pub output_tokens: usize,
    /// Optional exact generated-token plan. When present, its length controls
    /// native generation while `output_tokens` remains the authored maximum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    /// Optional aggregated or disaggregated decode attention-DP rank selected by the workload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,
    /// Optional disaggregated prefill attention-DP rank. When omitted, prefill falls back to
    /// `dp_rank`, matching Dynamo's request-routing contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Zero-based turn index within `session_id`, when the workload has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    /// Provider-neutral request metadata resolved by the selected runner.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl ReplayRequest {
    pub fn validate(&self) -> ReplayResult<()> {
        if self.id.is_empty() {
            return Err(ReplayError::InvalidSpec(
                "request id must not be empty".to_string(),
            ));
        }
        validate_time("request arrival_time_ms", self.arrival_time_ms)?;
        if let Some(input_token_ids) = &self.input_token_ids
            && input_token_ids.len() != self.input_tokens
        {
            return Err(ReplayError::InvalidSpec(format!(
                "request {:?} declares {} input tokens but materializes {} token IDs",
                self.id,
                self.input_tokens,
                input_token_ids.len()
            )));
        }
        // `output_tokens` is documented as the authored maximum, but
        // `DirectRequest::effective_max_output_tokens` returns the plan's
        // length verbatim whenever a plan is present, so a plan longer than
        // that maximum silently generates past the authored ceiling --
        // `output_tokens: 5` with a 100-entry plan emits 100 tokens. Reject
        // the contradiction here, the way the input-token check above rejects
        // a mismatched prompt plan. A plan shorter than the maximum is legal
        // and unaffected.
        if let Some(output_token_ids) = &self.output_token_ids
            && output_token_ids.len() > self.output_tokens
        {
            return Err(ReplayError::InvalidSpec(format!(
                "request {:?} declares at most {} output tokens but plans {} token IDs",
                self.id,
                self.output_tokens,
                output_token_ids.len()
            )));
        }
        self.routing_metadata()?;
        Ok(())
    }

    /// Resolve the small routing subset of provider-neutral metadata while
    /// leaving the authored metadata value intact for reporting and other
    /// adapters.
    pub fn routing_metadata(&self) -> ReplayResult<ReplayRoutingMetadata> {
        if self.metadata.is_null() {
            return Ok(ReplayRoutingMetadata::default());
        }
        serde_json::from_value(self.metadata.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!(
                "request {:?} has invalid routing metadata: {error}",
                self.id
            ))
        })
    }
}

impl TryFrom<(usize, DirectRequest)> for ReplayRequest {
    type Error = ReplayError;

    fn try_from((index, request): (usize, DirectRequest)) -> Result<Self, Self::Error> {
        let DirectRequest {
            tokens,
            max_output_tokens,
            output_token_ids,
            uuid,
            preferred_dp_rank,
            preferred_prefill_dp_rank,
            arrival_timestamp_ms,
            priority,
            strict_priority,
            policy_class,
            replay_context,
            ..
        } = request;
        let input_tokens = tokens.len();
        let arrival_time_ms = arrival_timestamp_ms.ok_or_else(|| {
            ReplayError::InvalidSpec(
                "DirectRequest is missing arrival_timestamp_ms, required by ReplayRequest"
                    .to_string(),
            )
        })?;
        validate_time("request arrival_time_ms", arrival_time_ms)?;
        let (id, session_id, turn_index, mut metadata, input_token_ids) = match replay_context {
            Some(context) => (
                context.authored_id,
                context.session_id,
                context.turn_index,
                context.metadata,
                (context.prompt_token_source != ReplayPromptTokenSource::LengthOnlySynthetic)
                    .then_some(tokens),
            ),
            None => (
                uuid.map(|uuid| uuid.to_string())
                    .unwrap_or_else(|| index.to_string()),
                None,
                None,
                Value::Null,
                Some(tokens),
            ),
        };
        if let Some(metadata_object) = metadata.as_object() {
            let routing: ReplayRoutingMetadata =
                serde_json::from_value(metadata.clone()).map_err(|error| {
                    ReplayError::InvalidSpec(format!(
                        "DirectRequest metadata has invalid routing controls: {error}"
                    ))
                })?;
            if metadata_object.contains_key("priority") && routing.priority != priority {
                return Err(ReplayError::InvalidSpec(
                    "DirectRequest priority conflicts with replay context metadata".to_string(),
                ));
            }
            if metadata_object.contains_key("strict_priority")
                && routing.strict_priority != strict_priority
            {
                return Err(ReplayError::InvalidSpec(
                    "DirectRequest strict_priority conflicts with replay context metadata"
                        .to_string(),
                ));
            }
            if metadata_object.contains_key("policy_class") && routing.policy_class != policy_class
            {
                return Err(ReplayError::InvalidSpec(
                    "DirectRequest policy_class conflicts with replay context metadata".to_string(),
                ));
            }
        }
        if priority != 0 || strict_priority != 0 || policy_class.is_some() {
            if metadata.is_null() {
                metadata = Value::Object(Default::default());
            }
            let object = metadata.as_object_mut().ok_or_else(|| {
                ReplayError::InvalidSpec("DirectRequest metadata must be an object".to_string())
            })?;
            if priority != 0 {
                object.insert("priority".to_string(), Value::from(priority));
            }
            if strict_priority != 0 {
                object.insert("strict_priority".to_string(), Value::from(strict_priority));
            }
            if let Some(policy_class) = policy_class {
                object.insert("policy_class".to_string(), Value::from(policy_class));
            }
        }

        let converted = Self {
            id,
            arrival_time_ms,
            input_tokens,
            input_token_ids,
            output_tokens: max_output_tokens,
            output_token_ids,
            dp_rank: preferred_dp_rank,
            prefill_dp_rank: preferred_prefill_dp_rank,
            session_id,
            turn_index,
            metadata,
        };
        converted.validate()?;
        Ok(converted)
    }
}

/// Provider-neutral routing controls recognized during ReplaySpec lowering.
/// Unknown metadata keys remain available in [`ReplayRequest::metadata`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRoutingMetadata {
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub strict_priority: u32,
    #[serde(default)]
    pub policy_class: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayAdapters {
    #[serde(default = "ProviderSpec::round_robin")]
    pub placement: ProviderSpec,
    #[serde(default = "ProviderSpec::no_scaling")]
    pub scaling: ProviderSpec,
}

impl Default for ReplayAdapters {
    fn default() -> Self {
        Self {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub provider: String,
    #[serde(default)]
    pub config: Value,
}

impl ProviderSpec {
    pub fn round_robin() -> Self {
        Self {
            provider: "round_robin".to_string(),
            config: Value::Null,
        }
    }

    pub fn no_scaling() -> Self {
        Self {
            provider: "none".to_string(),
            config: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStage {
    Aggregated,
    Prefill,
    Decode,
}

fn validate_time(name: &str, value: f64) -> ReplayResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ReplayError::InvalidSpec(format!(
            "{name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod direct_request_conversion_tests {
    use uuid::Uuid;

    use super::*;
    use crate::replay::protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};

    #[test]
    fn direct_request_conversion_requires_arrival_time() {
        let request = DirectRequest {
            tokens: vec![1, 2, 3],
            max_output_tokens: 4,
            ..Default::default()
        };

        let error = ReplayRequest::try_from((0, request)).unwrap_err();
        assert!(
            error.to_string().contains("arrival_timestamp_ms"),
            "{error}"
        );
    }

    #[test]
    fn direct_request_conversion_rejects_invalid_arrival_time() {
        for arrival_timestamp_ms in [-1.0, f64::INFINITY] {
            let request = DirectRequest {
                arrival_timestamp_ms: Some(arrival_timestamp_ms),
                ..Default::default()
            };

            assert!(ReplayRequest::try_from((0, request)).is_err());
        }
    }

    #[test]
    fn direct_request_conversion_uses_authored_id_then_uuid_then_index() {
        let authored = DirectRequest {
            uuid: Some(Uuid::from_u128(7)),
            arrival_timestamp_ms: Some(1.0),
            replay_context: Some(ReplayRequestContext {
                authored_id: "authored-42".to_string(),
                session_id: None,
                turn_index: None,
                metadata: Value::Null,
                prompt_token_source: Default::default(),
            }),
            ..Default::default()
        };
        assert_eq!(
            ReplayRequest::try_from((0, authored)).unwrap().id,
            "authored-42"
        );

        let uuid = DirectRequest {
            uuid: Some(Uuid::from_u128(7)),
            arrival_timestamp_ms: Some(1.0),
            ..Default::default()
        };
        assert_eq!(
            ReplayRequest::try_from((0, uuid)).unwrap().id,
            Uuid::from_u128(7).to_string()
        );

        let index = DirectRequest {
            arrival_timestamp_ms: Some(1.0),
            ..Default::default()
        };
        assert_eq!(ReplayRequest::try_from((3, index)).unwrap().id, "3");
    }

    /// `output_tokens` is the authored maximum, but
    /// `effective_max_output_tokens` returns the plan length verbatim, so a
    /// plan longer than the maximum generates past the authored ceiling. A
    /// plan at or under the maximum stays valid.
    #[test]
    fn validate_rejects_an_output_token_plan_longer_than_the_authored_maximum() {
        let request = |max_output_tokens, plan: Vec<u32>| {
            ReplayRequest::try_from((
                0,
                DirectRequest {
                    max_output_tokens,
                    output_token_ids: Some(plan),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                },
            ))
        };

        let error = request(2, vec![1, 2, 3, 4]).unwrap_err();
        assert!(error.to_string().contains("plans 4 token IDs"), "{error}");

        // A plan at or under the authored maximum stays valid.
        request(4, vec![1, 2, 3, 4]).unwrap().validate().unwrap();
        request(4, vec![1, 2]).unwrap().validate().unwrap();
    }

    #[test]
    fn direct_request_conversion_preserves_execution_fields() {
        let request = DirectRequest {
            tokens: vec![1, 2],
            max_output_tokens: 4,
            output_token_ids: Some(vec![3, 4]),
            arrival_timestamp_ms: Some(10.0),
            preferred_dp_rank: Some(2),
            preferred_prefill_dp_rank: Some(5),
            priority: -7,
            strict_priority: 9,
            policy_class: Some("latency".to_string()),
            replay_context: Some(ReplayRequestContext {
                authored_id: "request".to_string(),
                session_id: Some("session-1".to_string()),
                turn_index: Some(3),
                metadata: serde_json::json!({"k": "v"}),
                prompt_token_source: Default::default(),
            }),
            ..Default::default()
        };

        let converted = ReplayRequest::try_from((0, request)).unwrap();
        assert_eq!(converted.arrival_time_ms, 10.0);
        assert_eq!(converted.input_token_ids, Some(vec![1, 2]));
        assert_eq!(converted.output_tokens, 4);
        assert_eq!(converted.output_token_ids, Some(vec![3, 4]));
        assert_eq!(converted.dp_rank, Some(2));
        assert_eq!(converted.prefill_dp_rank, Some(5));
        assert_eq!(converted.session_id.as_deref(), Some("session-1"));
        assert_eq!(converted.turn_index, Some(3));
        assert_eq!(
            converted.metadata,
            serde_json::json!({
                "k": "v",
                "priority": -7,
                "strict_priority": 9,
                "policy_class": "latency"
            })
        );
    }

    #[test]
    fn direct_request_conversion_preserves_authored_default_routing_metadata() {
        let metadata = serde_json::json!({
            "priority": 0,
            "strict_priority": 0,
            "policy_class": null,
        });
        let request = DirectRequest {
            arrival_timestamp_ms: Some(1.0),
            replay_context: Some(ReplayRequestContext {
                authored_id: "request".to_string(),
                session_id: None,
                turn_index: None,
                metadata: metadata.clone(),
                prompt_token_source: Default::default(),
            }),
            ..Default::default()
        };

        assert_eq!(
            ReplayRequest::try_from((0, request)).unwrap().metadata,
            metadata
        );
    }

    #[test]
    fn direct_request_conversion_rejects_conflicting_routing_metadata() {
        let request = DirectRequest {
            arrival_timestamp_ms: Some(1.0),
            replay_context: Some(ReplayRequestContext {
                authored_id: "request".to_string(),
                session_id: None,
                turn_index: None,
                metadata: serde_json::json!({"priority": 5}),
                prompt_token_source: Default::default(),
            }),
            ..Default::default()
        };

        let error = ReplayRequest::try_from((0, request)).unwrap_err();
        assert!(error.to_string().contains("priority conflicts"), "{error}");
    }

    #[test]
    fn direct_request_conversion_keeps_synthetic_tokens_unmaterialized() {
        let request = DirectRequest {
            tokens: vec![1, 2],
            arrival_timestamp_ms: Some(1.0),
            replay_context: Some(ReplayRequestContext {
                authored_id: "synthetic".to_string(),
                session_id: None,
                turn_index: None,
                metadata: Value::Null,
                prompt_token_source: ReplayPromptTokenSource::LengthOnlySynthetic,
            }),
            ..Default::default()
        };

        let converted = ReplayRequest::try_from((0, request)).unwrap();
        assert_eq!(converted.input_tokens, 2);
        assert_eq!(converted.input_token_ids, None);
    }
}
