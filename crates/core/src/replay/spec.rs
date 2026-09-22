// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::{EncoderShape, TimingModelConfig};
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
    /// Optional encoder pool every request crosses before the language workers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder: Option<EncoderSpec>,
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
        if let Some(encoder) = &self.encoder {
            encoder.validate()?;
        }

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

/// Event-level model of SGLang's dedicated encoder servers (`--encoder-only`)
/// ahead of the language workers.
///
/// A request's images are spread over the instances; each instance runs one
/// serial loop that takes the queued parts up to `max_batch`, preprocesses and
/// encodes them as one batch, then pushes each part's embeddings to the language
/// rank. A batch frees its instance when its GPU work ends; transfers are timed
/// per part, and the request is admitted when its last part arrived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncoderSpec {
    pub instances: usize,
    /// Parts one batch takes at most (`SGLANG_ENCODER_MAX_BATCH_SIZE`).
    pub max_batch: usize,
    /// GPUs one instance occupies (its tensor-parallel width).
    pub gpus_per_instance: usize,
    pub images_per_request: u32,
    /// Geometry of one image as the encoder forward sees it.
    pub shape: EncoderShape,
    /// CPU preprocessing (image decode and processor) per image; a batch costs
    /// its image count times this.
    pub preprocess_ms_per_image: f64,
    /// Embedding bytes one image sends to the language rank(s).
    pub transfer_bytes_per_image: u64,
    /// Network bandwidth in decimal gigabytes per second.
    pub transfer_bandwidth_gb_s: f64,
    /// Timing model pricing the encoder forward, resolved by the runner like a
    /// rank's; `None` shares the language workers' model (tests).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing_model: Option<TimingModelConfig>,
}

impl EncoderSpec {
    pub fn validate(&self) -> ReplayResult<()> {
        for (name, value) in [
            ("instances", self.instances),
            ("max_batch", self.max_batch),
            ("gpus_per_instance", self.gpus_per_instance),
            ("images_per_request", self.images_per_request as usize),
        ] {
            if value == 0 {
                return Err(ReplayError::InvalidSpec(format!(
                    "encoder {name} must be positive"
                )));
            }
        }
        let shape = &self.shape;
        if [
            shape.sequences,
            shape.patch_tokens,
            shape.transformer_tokens,
            shape.output_tokens,
        ]
        .contains(&0)
        {
            return Err(ReplayError::InvalidSpec(
                "encoder shape counts must be positive".to_string(),
            ));
        }
        validate_time(
            "encoder preprocess_ms_per_image",
            self.preprocess_ms_per_image,
        )?;
        if !self.transfer_bandwidth_gb_s.is_finite() || self.transfer_bandwidth_gb_s <= 0.0 {
            return Err(ReplayError::InvalidSpec(format!(
                "encoder transfer_bandwidth_gb_s must be positive and finite, got {}",
                self.transfer_bandwidth_gb_s
            )));
        }
        if let Some(TimingModelConfig::Fixed { .. } | TimingModelConfig::Polynomial) =
            &self.timing_model
        {
            return Err(ReplayError::InvalidSpec(
                "encoder timing_model must price vision batches; fixed and polynomial models do not"
                    .to_string(),
            ));
        }
        Ok(())
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
