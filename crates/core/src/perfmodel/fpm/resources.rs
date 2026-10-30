// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rank-local cache descriptors and timing-independent profile memory sizing.

use serde::{Deserialize, Serialize};

use super::{
    ForwardPassPerfModel, ForwardPassPerfModelConfig, ForwardPassWorkerType, FpmExecutionBinding,
};
use crate::{AicError, BackendKind, MemoryBreakdown};

const MAX_EXACT_BYTES: u64 = 1 << 53;

fn invalid(message: impl Into<String>) -> AicError {
    AicError::InvalidEngineConfig(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmCacheKind {
    Attention,
    Convolution,
    /// Cache whose size does not depend on the request length, such as the
    /// recurrent state of linear-attention or Mamba layers. vLLM holds a fixed
    /// number of pool blocks for each running request.
    State,
}

/// One cache group on one TP rank. `page_size_bytes` includes every layer in
/// this group and any runtime padding; it is not a per-layer or logical-token
/// rate. Window lengths describe retained history, independently of FPM timing
/// query context lengths. A `state` group instead holds `blocks_per_request`
/// blocks of `page_size_bytes` for every running request and has no window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmCacheGroup {
    pub name: String,
    pub kind: FpmCacheKind,
    pub num_layers: u32,
    pub block_size_tokens: u32,
    pub page_size_bytes: u64,
    pub sliding_window: Option<u32>,
    /// Pool blocks held per running request. Required for `state` groups and
    /// invalid for every other kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks_per_request: Option<u32>,
}

impl FpmCacheGroup {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.name.trim().is_empty()
            || self.num_layers == 0
            || self.block_size_tokens == 0
            || self.page_size_bytes == 0
            || self.page_size_bytes > MAX_EXACT_BYTES
            || self.sliding_window == Some(0)
        {
            return Err(invalid(
                "cache group requires a name, positive layers/block/page/window, and page_size_bytes <= 2**53",
            ));
        }
        if self.kind == FpmCacheKind::Convolution && self.sliding_window.is_none() {
            return Err(invalid("convolution cache groups require a sliding_window"));
        }
        if self.kind == FpmCacheKind::State {
            if self.blocks_per_request.is_none_or(|blocks| blocks == 0) {
                return Err(invalid(
                    "state cache groups require a positive blocks_per_request",
                ));
            }
            if self.sliding_window.is_some() {
                return Err(invalid("state cache groups cannot have a sliding_window"));
            }
        } else if self.blocks_per_request.is_some() {
            return Err(invalid(
                "blocks_per_request applies only to state cache groups",
            ));
        }
        Ok(())
    }

    /// Blocks live during one forward pass. All newly scheduled tokens remain
    /// allocated until the next pass, including temporary prefill storage. A
    /// state group holds its fixed blocks for any positive target.
    pub fn block_range(
        &self,
        computed_tokens: u64,
        target_tokens: u64,
    ) -> Result<std::ops::Range<u64>, AicError> {
        self.validate()?;
        if target_tokens < computed_tokens {
            return Err(invalid("cache target_tokens must be >= computed_tokens"));
        }
        if self.kind == FpmCacheKind::State {
            // `validate` guarantees a positive count for state groups.
            let blocks = self.blocks_per_request.map_or(0, u64::from);
            return Ok(0..if target_tokens == 0 { 0 } else { blocks });
        }
        let block = u64::from(self.block_size_tokens);
        let first = self.sliding_window.map_or(0, |window| {
            computed_tokens.saturating_sub(u64::from(window) - 1) / block
        });
        Ok(first..target_tokens.div_ceil(block))
    }

    pub fn bytes_for_forward(
        &self,
        computed_tokens: u64,
        target_tokens: u64,
    ) -> Result<u64, AicError> {
        let range = self.block_range(computed_tokens, target_tokens)?;
        (range.end - range.start)
            .checked_mul(self.page_size_bytes)
            .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
            .ok_or_else(|| invalid("cache allocation exceeds 2**53 bytes"))
    }

    /// Conservative maximum for a request up to `context_length`, with forward
    /// chunks no larger than `max_num_tokens`. The extra block alignment can
    /// coexist with the entire new chunk; clipping only to the retention window
    /// would underestimate prefill memory. A state group needs only its fixed
    /// blocks, whatever the context.
    pub fn peak_request_bytes(
        &self,
        context_length: u64,
        max_num_tokens: u32,
    ) -> Result<u64, AicError> {
        self.validate()?;
        if context_length == 0 || max_num_tokens == 0 {
            return Err(invalid("cache context and max_num_tokens must be positive"));
        }
        let pages = if self.kind == FpmCacheKind::State {
            // `validate` guarantees a positive count for state groups.
            self.blocks_per_request.map_or(0, u64::from)
        } else {
            let block = u64::from(self.block_size_tokens);
            let full = context_length.div_ceil(block);
            if let Some(window) = self.sliding_window {
                let span = u64::from(window) - 1 + u64::from(max_num_tokens) + block - 1;
                full.min(span.div_ceil(block))
            } else {
                full
            }
        };
        pages
            .checked_mul(self.page_size_bytes)
            .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
            .ok_or_else(|| invalid("cache request allocation exceeds 2**53 bytes"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmCacheLayout {
    Linear,
    Grouped,
}

/// How the runtime probe cached the state of its Mamba layers. vLLM's `all`
/// mode is not modeled, so it cannot be recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FpmMambaCacheMode {
    None,
    Align,
}

/// Cache capacity observed for the exact deployment and scheduler settings.
/// This includes the runtime's graph/workspace reservations; it is not a
/// non-KV memory limit or a capacity that can be scaled to another setting.
///
/// The three optional prefix-caching fields record the settings that the probe
/// ran with. They are evidence only: selection, sizing and replay do not read
/// them, and the simulated deployment decides prefix caching.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmRuntimeMemoryConfig {
    pub kv_cache_bytes: u64,
    pub gpu_memory_utilization: f64,
    pub max_model_len: u64,
    pub provenance: String,
    /// The probe's `enable_prefix_caching` setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_prefix_caching: Option<bool>,
    /// The probe's `mamba_cache_mode`. `align` requires prefix caching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba_cache_mode: Option<FpmMambaCacheMode>,
    /// The probe's resolved `hash_block_size`, in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash_block_size_tokens: Option<u32>,
}

impl FpmRuntimeMemoryConfig {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.kv_cache_bytes == 0
            || self.kv_cache_bytes > MAX_EXACT_BYTES
            || !self.gpu_memory_utilization.is_finite()
            || self.gpu_memory_utilization <= 0.0
            || self.gpu_memory_utilization > 1.0
            || self.max_model_len == 0
            || self.provenance.trim().is_empty()
        {
            return Err(invalid(
                "runtime memory requires positive kv_cache_bytes <= 2**53, finite gpu_memory_utilization in (0, 1], positive max_model_len and provenance",
            ));
        }
        if self.mamba_cache_mode == Some(FpmMambaCacheMode::Align)
            && self.enable_prefix_caching != Some(true)
        {
            return Err(invalid(
                "mamba_cache_mode align requires enable_prefix_caching=true",
            ));
        }
        if self.hash_block_size_tokens == Some(0) {
            return Err(invalid("hash_block_size_tokens must be positive"));
        }
        Ok(())
    }
}

/// The typed resource section of the existing profile wire schema.
/// Missing non-KV components leave memory pending until runtime observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmResourceConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activations_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_overhead_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comm_overhead_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_memory: Option<FpmRuntimeMemoryConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_bytes_per_token: Option<u64>,
    pub cache_layout: FpmCacheLayout,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_groups: Vec<FpmCacheGroup>,
    pub max_num_tokens: u32,
    pub max_batch_size: u32,
    pub provenance: String,
}

impl FpmResourceConfig {
    pub fn validate(&self) -> Result<(), AicError> {
        if self.max_num_tokens == 0 || self.max_batch_size == 0 || self.provenance.trim().is_empty()
        {
            return Err(invalid(
                "FPM resources require positive scheduler bounds and provenance",
            ));
        }
        let declared = self.declared_bytes();
        declared
            .iter()
            .flatten()
            .try_fold(0u64, |total, value| total.checked_add(*value))
            .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
            .ok_or_else(|| invalid("total non-KV resource bytes must not exceed 2**53"))?;
        if let Some(runtime) = &self.runtime_memory {
            runtime.validate()?;
            if declared.iter().any(Option::is_some) {
                return Err(invalid(
                    "runtime_memory cannot be combined with declared non-KV resource bytes",
                ));
            }
        }
        match self.cache_layout {
            FpmCacheLayout::Linear
                if self.cache_groups.is_empty()
                    && self
                        .kv_bytes_per_token
                        .is_some_and(|bytes| bytes > 0 && bytes <= MAX_EXACT_BYTES) => {}
            FpmCacheLayout::Grouped
                if self.kv_bytes_per_token.is_none() && !self.cache_groups.is_empty() =>
            {
                let mut names = std::collections::HashSet::new();
                for group in &self.cache_groups {
                    group.validate()?;
                    if !names.insert(group.name.as_str()) {
                        return Err(invalid("cache group names must be unique"));
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "linear cache requires kv_bytes_per_token only; grouped cache requires cache_groups without a scalar token rate",
                ));
            }
        }
        Ok(())
    }

    fn declared_bytes(&self) -> [Option<u64>; 4] {
        [
            self.weights_bytes,
            self.activations_bytes,
            self.runtime_overhead_bytes,
            self.comm_overhead_bytes,
        ]
    }

    fn non_kv_bytes(&self) -> Result<u64, AicError> {
        if self.runtime_memory.is_some() {
            return Err(invalid(
                "runtime memory records cache capacity, not a non-KV byte breakdown",
            ));
        }
        self.require_memory()?;
        self.declared_bytes()
            .into_iter()
            .flatten()
            .try_fold(0u64, |total, value| total.checked_add(value))
            .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
            .ok_or_else(|| invalid("total non-KV resource bytes must not exceed 2**53"))
    }

    pub fn require_memory(&self) -> Result<(), AicError> {
        if self.runtime_memory.is_none() && self.declared_bytes().iter().any(Option::is_none) {
            return Err(invalid(
                "FPM memory is pending runtime profiling; collect and finalize runtime memory or provide all four declared non-KV resource values before simulation",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmCacheBudgetRequest {
    pub total_gpu_capacity_bytes: u64,
    pub memory_fraction_kind: String,
    pub memory_fraction_value: f64,
    pub max_num_tokens: u32,
    pub max_batch_size: u32,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// Optional logical request length for steady, one-token decode occupancy.
    /// Scheduler limits and context_length still control resource admission.
    #[serde(default)]
    pub request_occupancy_tokens: Option<u64>,
    #[serde(default)]
    pub cuda_graph_reserved_bytes: u64,
    #[serde(default)]
    pub tolerance_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmCacheBudgetAdjusted {
    pub tolerance_fraction: f64,
    pub total_kv_size_bytes: u64,
    pub total_kv_size_tokens: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FpmCacheBudget {
    pub total_gpu_capacity_bytes: u64,
    pub total_kv_size_bytes: u64,
    pub kv_size_per_token_bytes: Option<u64>,
    pub total_kv_size_tokens: Option<u64>,
    pub source: String,
    pub memory_breakdown: Option<MemoryBreakdown>,
    pub tolerance_adjusted: Option<FpmCacheBudgetAdjusted>,
    pub resource_provenance: String,
    pub cache_layout: FpmCacheLayout,
    pub cache_groups: Vec<FpmCacheGroup>,
    /// Per-request upper bound, including window block alignment and the
    /// configured prefill chunk. This is not an aggregate token capacity.
    pub request_peak_cache_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_occupancy_cache_bytes: Option<u64>,
}

impl ForwardPassPerfModelConfig {
    /// Select exact profile resources without constructing an analytical model
    /// or reading timings. Identity/schema normalization is the same as for
    /// `best_available`.
    pub fn estimate_cache_budget(
        &self,
        request: &FpmCacheBudgetRequest,
    ) -> Result<FpmCacheBudget, AicError> {
        let resolved = self.clone().resolve()?;
        let resources = resolved
            .fpm_resources()?
            .ok_or_else(|| invalid("cache budget requires fpm_profile resources"))?;
        let context = resolved
            .fpm_profile
            .as_ref()
            .and_then(|p| p.get("context_length"))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| invalid("fpm_profile requires positive context_length"))?;
        let requested_context = request.context_length.unwrap_or(context);
        if requested_context == 0 || requested_context > context {
            return Err(invalid(
                "context_length must be positive and within the FPM profile context bound",
            ));
        }
        resources.estimate_budget(request, requested_context)
    }

    pub fn fpm_resources(&self) -> Result<Option<FpmResourceConfig>, AicError> {
        let Some(profile) = &self.fpm_profile else {
            return Ok(None);
        };
        if profile.get("model").and_then(serde_json::Value::as_str) != Some(self.model.as_str()) {
            return Err(invalid("FPM profile model identity mismatch"));
        }
        let execution = self.validate_fpm_execution()?;
        if self.backend != BackendKind::Vllm
            || self.pp != 1
            || (self.nextn != 0 && execution.is_none())
            || self.speculation.is_some()
        {
            return Err(invalid(
                "FPM resources require vLLM PP1 plain autoregressive execution",
            ));
        }
        let deployment = self
            .fpm_deployment()?
            .ok_or_else(|| invalid("FPM deployment requires a profile"))?;
        let resources: FpmResourceConfig = serde_json::from_value(
            deployment
                .get("resources")
                .cloned()
                .ok_or_else(|| invalid("FPM deployment requires resources"))?,
        )
        .map_err(|error| invalid(format!("FPM resource profile: {error}")))?;
        resources.validate()?;
        if execution.is_some()
            && (resources.cache_layout != FpmCacheLayout::Linear
                || resources.runtime_memory.is_none())
        {
            return Err(invalid(
                "speculative-round FPM resources require observed runtime memory and a linear cache",
            ));
        }
        Ok(Some(resources))
    }

    pub(crate) fn fpm_execution_binding(&self) -> Result<Option<FpmExecutionBinding>, AicError> {
        let Some(profile) = &self.fpm_profile else {
            return Ok(None);
        };
        // Do not change validation order for historical profiles without the
        // optional execution identity.
        let has_execution = profile
            .get("deployments")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|deployments| {
                deployments.iter().any(|deployment| {
                    deployment
                        .get("execution_profile")
                        .is_some_and(|value| !value.is_null())
                })
            });
        if !has_execution {
            return Ok(None);
        }
        let deployment = self
            .fpm_deployment()?
            .ok_or_else(|| invalid("FPM execution identity requires a deployment"))?;
        let Some(execution) = deployment
            .get("execution_profile")
            .filter(|value| !value.is_null())
        else {
            return Ok(None);
        };
        let binding = FpmExecutionBinding {
            model_revision: profile
                .get("model_revision")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("FPM execution identity requires model_revision"))?
                .to_owned(),
            execution_profile: serde_json::from_value(execution.clone())
                .map_err(|error| invalid(format!("FPM execution identity: {error}")))?,
        };
        binding.validate()?;
        Ok(Some(binding))
    }

    fn fpm_deployment(&self) -> Result<Option<&serde_json::Value>, AicError> {
        let Some(profile) = &self.fpm_profile else {
            return Ok(None);
        };
        if profile.get("model").and_then(serde_json::Value::as_str) != Some(self.model.as_str()) {
            return Err(invalid("FPM profile model identity mismatch"));
        }
        let deployments = profile
            .get("deployments")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid("fpm_profile requires deployments"))?;
        let matching_identity: Vec<_> = deployments
            .iter()
            .filter(|d| {
                d.get("system").and_then(serde_json::Value::as_str) == Some(self.system.as_str())
                    && d.get("backend").and_then(serde_json::Value::as_str)
                        == Some(self.backend.as_str())
                    && d.get("backend_version").and_then(serde_json::Value::as_str)
                        == self.backend_version.as_deref()
                    && [
                        ("tp", self.tp),
                        ("pp", self.pp),
                        ("dp", self.attention_dp),
                        ("moe_tp", self.moe_tp_size.unwrap_or(1)),
                        ("moe_ep", self.moe_ep_size.unwrap_or(1)),
                        ("cp", 1),
                    ]
                    .into_iter()
                    .all(|(key, expected)| {
                        d.get(key)
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(if key == "pp" || key == "cp" { 1 } else { 0 })
                            == u64::from(expected)
                    })
            })
            .collect();
        let roles = matching_identity
            .iter()
            .map(|d| {
                d.get("worker_type")
                    .filter(|role| !role.is_null())
                    .cloned()
                    .map(serde_json::from_value::<ForwardPassWorkerType>)
                    .transpose()
                    .map_err(|error| invalid(format!("FPM deployment worker_type: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if roles.iter().any(Option::is_none) && roles.iter().any(Option::is_some) {
            return Err(invalid(
                "ambiguous FPM deployment identity: cannot mix legacy shared and role-specific profiles",
            ));
        }
        let mut matching =
            matching_identity
                .into_iter()
                .zip(roles)
                .filter_map(|(deployment, role)| {
                    role.is_none_or(|role| role == self.worker_type)
                        .then_some(deployment)
                });
        let deployment = matching
            .next()
            .ok_or_else(|| invalid("no matching FPM deployment resources"))?;
        if matching.next().is_some() {
            return Err(invalid("duplicate FPM deployment resources"));
        }
        Ok(Some(deployment))
    }
}

impl FpmResourceConfig {
    fn estimate_budget(
        &self,
        request: &FpmCacheBudgetRequest,
        context: u64,
    ) -> Result<FpmCacheBudget, AicError> {
        self.validate()?;
        if request.memory_fraction_kind != "of_total" {
            return Err(invalid(
                "vLLM FPM resources require memory_fraction_kind='of_total'",
            ));
        }
        let fraction = request.memory_fraction_value;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(invalid(
                "memory_fraction_value must be finite and in [0, 1]",
            ));
        }
        if request
            .tolerance_fraction
            .is_some_and(|t| !t.is_finite() || !(0.0..1.0).contains(&t))
        {
            return Err(invalid("tolerance_fraction must be finite and in [0, 1)"));
        }
        if request.total_gpu_capacity_bytes == 0
            || request.total_gpu_capacity_bytes > MAX_EXACT_BYTES
        {
            return Err(invalid(
                "GPU capacity must be positive and no greater than 2**53",
            ));
        }
        if request.max_num_tokens == 0
            || request.max_num_tokens > self.max_num_tokens
            || request.max_batch_size == 0
            || request.max_batch_size > self.max_batch_size
        {
            return Err(invalid(
                "FPM resource envelope exceeded: provide bounds for the requested positive rank-local scheduler envelope",
            ));
        }
        let (total, memory_breakdown) = if let Some(runtime) = &self.runtime_memory {
            if request.max_num_tokens != self.max_num_tokens
                || request.max_batch_size != self.max_batch_size
            {
                return Err(invalid(
                    "runtime memory requires the exact recorded rank-local scheduler settings; collect new runtime memory for changed settings",
                ));
            }
            if context > runtime.max_model_len {
                return Err(invalid(
                    "context_length exceeds runtime memory max_model_len",
                ));
            }
            if fraction != runtime.gpu_memory_utilization {
                return Err(invalid(
                    "memory_fraction_value must match runtime memory gpu_memory_utilization",
                ));
            }
            if request.cuda_graph_reserved_bytes != 0 {
                return Err(invalid(
                    "runtime memory already includes graph reservations; cuda_graph_reserved_bytes must be 0",
                ));
            }
            if (request.total_gpu_capacity_bytes as f64 * fraction).floor()
                < runtime.kv_cache_bytes as f64
            {
                return Err(invalid(
                    "GPU memory budget is smaller than the recorded runtime KV cache allocation",
                ));
            }
            (runtime.kv_cache_bytes, None)
        } else {
            let non_kv = self
                .non_kv_bytes()?
                .checked_add(request.cuda_graph_reserved_bytes)
                .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
                .ok_or_else(|| {
                    invalid("total FPM non-KV bytes including CUDA graphs must not exceed 2**53")
                })?;
            let available =
                (request.total_gpu_capacity_bytes as f64 * fraction - non_kv as f64).floor();
            if available < 1.0 {
                return Err(invalid(
                    "no KV budget after non-KV resources and CUDA graphs",
                ));
            }
            (
                available as u64,
                Some(MemoryBreakdown {
                    weights_bytes: self.weights_bytes.unwrap(),
                    activations_bytes: self.activations_bytes.unwrap(),
                    runtime_overhead_bytes: self.runtime_overhead_bytes.unwrap(),
                    comm_overhead_bytes: self.comm_overhead_bytes.unwrap(),
                    cuda_graph_reserved_bytes: request.cuda_graph_reserved_bytes,
                }),
            )
        };
        let token_count = |bytes| self.kv_bytes_per_token.map(|rate| bytes / rate);
        let footprint = |tokens: u64, chunk: u32| {
            let bytes = if self.cache_layout == FpmCacheLayout::Linear {
                tokens.checked_mul(self.kv_bytes_per_token.unwrap())
            } else {
                self.cache_groups.iter().try_fold(0u64, |sum, group| {
                    sum.checked_add(group.peak_request_bytes(tokens, chunk).ok()?)
                })
            };
            bytes
                .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
                .ok_or_else(|| invalid("cache request allocation exceeds 2**53 bytes"))
        };
        let request_peak_cache_bytes = footprint(context, request.max_num_tokens)?;
        let request_occupancy_cache_bytes = request
            .request_occupancy_tokens
            .map(|tokens| {
                if tokens == 0 || tokens > context {
                    return Err(invalid(
                        "request_occupancy_tokens must be positive and within context_length",
                    ));
                }
                footprint(tokens, 1)
            })
            .transpose()?;
        Ok(FpmCacheBudget {
            total_gpu_capacity_bytes: request.total_gpu_capacity_bytes,
            total_kv_size_bytes: total,
            kv_size_per_token_bytes: self.kv_bytes_per_token,
            total_kv_size_tokens: token_count(total),
            source: "profile".into(),
            memory_breakdown,
            tolerance_adjusted: request.tolerance_fraction.map(|tolerance| {
                let bytes = (total as f64 * (1.0 - tolerance)).floor() as u64;
                FpmCacheBudgetAdjusted {
                    tolerance_fraction: tolerance,
                    total_kv_size_bytes: bytes,
                    total_kv_size_tokens: token_count(bytes),
                }
            }),
            resource_provenance: self.provenance.clone(),
            cache_layout: self.cache_layout,
            cache_groups: self.cache_groups.clone(),
            request_peak_cache_bytes,
            request_occupancy_cache_bytes,
        })
    }
}

impl ForwardPassPerfModel {
    pub fn estimate_cache_budget(
        &self,
        request: &FpmCacheBudgetRequest,
    ) -> Result<FpmCacheBudget, AicError> {
        self.provenance()
            .ok_or_else(|| invalid("cache resources require canonical model provenance"))?
            .config
            .estimate_cache_budget(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(window: Option<u32>) -> FpmCacheGroup {
        FpmCacheGroup {
            name: "attention".into(),
            kind: FpmCacheKind::Attention,
            num_layers: 2,
            block_size_tokens: 16,
            page_size_bytes: 160,
            sliding_window: window,
            blocks_per_request: None,
        }
    }

    fn state_group(blocks_per_request: Option<u32>) -> FpmCacheGroup {
        FpmCacheGroup {
            name: "state".into(),
            kind: FpmCacheKind::State,
            blocks_per_request,
            ..group(None)
        }
    }

    fn rejection(group: &FpmCacheGroup) -> String {
        match group.validate() {
            Err(AicError::InvalidEngineConfig(message)) => message,
            other => panic!("expected an invalid cache group, got {other:?}"),
        }
    }

    fn memory_rejection(memory: &FpmRuntimeMemoryConfig) -> String {
        match memory.validate() {
            Err(AicError::InvalidEngineConfig(message)) => message,
            other => panic!("expected an invalid runtime memory record, got {other:?}"),
        }
    }

    fn resources() -> FpmResourceConfig {
        FpmResourceConfig {
            weights_bytes: Some(100),
            activations_bytes: Some(20),
            runtime_overhead_bytes: Some(30),
            comm_overhead_bytes: Some(50),
            runtime_memory: None,
            kv_bytes_per_token: None,
            cache_layout: FpmCacheLayout::Grouped,
            cache_groups: vec![group(Some(32))],
            max_num_tokens: 128,
            max_batch_size: 8,
            provenance: "Hand-derived rank-local byte fixture".into(),
        }
    }

    fn request() -> FpmCacheBudgetRequest {
        FpmCacheBudgetRequest {
            total_gpu_capacity_bytes: 1000,
            memory_fraction_kind: "of_total".into(),
            memory_fraction_value: 0.8,
            max_num_tokens: 128,
            max_batch_size: 8,
            context_length: None,
            request_occupancy_tokens: None,
            cuda_graph_reserved_bytes: 20,
            tolerance_fraction: Some(0.1),
        }
    }

    fn runtime_resources() -> FpmResourceConfig {
        FpmResourceConfig {
            weights_bytes: None,
            activations_bytes: None,
            runtime_overhead_bytes: None,
            comm_overhead_bytes: None,
            runtime_memory: Some(FpmRuntimeMemoryConfig {
                kv_cache_bytes: 600,
                gpu_memory_utilization: 0.8,
                max_model_len: 4096,
                provenance: "Synthetic initialized-worker cache allocation".into(),
                enable_prefix_caching: None,
                mamba_cache_mode: None,
                hash_block_size_tokens: None,
            }),
            ..resources()
        }
    }

    fn role_config() -> ForwardPassPerfModelConfig {
        let mut config = ForwardPassPerfModelConfig::new(
            "test/model",
            "test_gpu",
            BackendKind::Vllm,
            ForwardPassWorkerType::Prefill,
        );
        config.backend_version = Some("0.27.0".into());
        let deployment = serde_json::json!({
            "system": "test_gpu", "backend": "vllm", "backend_version": "0.27.0",
            "tp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1,
            "worker_type": "prefill", "resources": runtime_resources(),
        });
        let mut decode = deployment.clone();
        decode["worker_type"] = "decode".into();
        decode["resources"]["runtime_memory"]["kv_cache_bytes"] = 700.into();
        decode["resources"]["max_batch_size"] = 4.into();
        config.fpm_profile = Some(
            serde_json::json!({
                "model": "test/model", "deployments": [deployment, decode],
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        config
    }

    fn round_config() -> ForwardPassPerfModelConfig {
        let mut config = role_config();
        config.nextn = 3;
        config.estimation_mode = super::super::EstimationMode::FpmInterpolation;
        config.estimator_config.fpm_interpolation.method =
            super::super::FpmInterpolationMethod::Direct;
        let profile = config.fpm_profile.as_mut().unwrap();
        profile.insert("model_revision".into(), "1".repeat(40).into());
        for deployment in profile["deployments"].as_array_mut().unwrap() {
            deployment["execution_profile"] = serde_json::json!({
                "kind": "speculative_round", "method": "eagle3", "num_draft_tokens": 3,
                "draft_model": "test/draft", "draft_revision": "2".repeat(40),
                "timing_scope": "model_step_cuda"
            });
            let resources = deployment["resources"].as_object_mut().unwrap();
            resources.insert("cache_layout".into(), "linear".into());
            resources.insert("kv_bytes_per_token".into(), 10.into());
            resources.remove("cache_groups");
        }
        config
    }

    #[test]
    fn speculative_round_keeps_observed_capacity_and_identity_after_reload() {
        let config = round_config();
        config.validate().unwrap();
        let reloaded: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(reloaded.nextn, 3);
        assert_eq!(
            config.fpm_execution_binding().unwrap(),
            reloaded.fpm_execution_binding().unwrap()
        );
        let resources = reloaded.fpm_resources().unwrap().unwrap();
        let mut query = request();
        query.cuda_graph_reserved_bytes = 0;
        query.tolerance_fraction = None;
        let budget = resources.estimate_budget(&query, 4096).unwrap();
        // The recorded usable pool is 600 bytes at 10 bytes/token. SD has
        // already affected that observation; neither draft depth nor TP is a
        // second capacity multiplier or memory deduction.
        assert_eq!(budget.total_kv_size_bytes, 600);
        assert_eq!(budget.total_kv_size_tokens, Some(60));
        assert_eq!(budget.memory_breakdown, None);
        query.cuda_graph_reserved_bytes = 20;
        assert!(resources.estimate_budget(&query, 4096).is_err());
    }

    #[test]
    fn speculative_round_requires_matching_depth_direct_selection_and_observed_memory() {
        let config = round_config();
        for depth in [0, 2, 4] {
            let mut invalid = config.clone();
            invalid.nextn = depth;
            assert!(invalid.validate().is_err());
            assert!(invalid.fpm_resources().is_err());
        }
        let mut invalid = config.clone();
        invalid.estimation_mode = super::super::EstimationMode::Auto;
        assert!(invalid.validate().is_err());
        invalid = config.clone();
        invalid.estimator_config.fpm_interpolation.method =
            super::super::FpmInterpolationMethod::Sol;
        assert!(invalid.validate().is_err());
        invalid = config.clone();
        invalid.fallback_policy = super::super::ForwardPassFallbackPolicy::Allow;
        assert!(invalid.validate().is_err());
        for replacement in [resources(), runtime_resources()] {
            let mut invalid = config.clone();
            invalid.fpm_profile.as_mut().unwrap()["deployments"][0]["resources"] =
                serde_json::to_value(replacement).unwrap();
            assert!(invalid.fpm_resources().is_err());
        }
        let mut invalid = config;
        let resources = invalid.fpm_profile.as_mut().unwrap()["deployments"][0]["resources"]
            .as_object_mut()
            .unwrap();
        resources.remove("runtime_memory");
        assert!(invalid.fpm_resources().is_err());
    }

    #[test]
    fn profile_resources_select_independent_roles_after_round_trip() {
        let config = role_config();
        let mut reloaded: ForwardPassPerfModelConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        let prefill = reloaded.fpm_resources().unwrap().unwrap();
        assert_eq!(prefill.runtime_memory.unwrap().kv_cache_bytes, 600);
        assert_eq!(prefill.max_batch_size, 8);
        reloaded.worker_type = ForwardPassWorkerType::Decode;
        let decode = reloaded.fpm_resources().unwrap().unwrap();
        assert_eq!(decode.runtime_memory.unwrap().kv_cache_bytes, 700);
        assert_eq!(decode.max_batch_size, 4);
        reloaded.worker_type = ForwardPassWorkerType::Aggregated;
        assert!(
            reloaded
                .fpm_resources()
                .unwrap_err()
                .to_string()
                .contains("no matching")
        );
    }

    #[test]
    fn legacy_shared_resources_are_preserved_without_role_inference() {
        let mut config = role_config();
        let deployments = config.fpm_profile.as_mut().unwrap()["deployments"]
            .as_array_mut()
            .unwrap();
        deployments.truncate(1);
        deployments[0]
            .as_object_mut()
            .unwrap()
            .remove("worker_type");
        for role in [
            ForwardPassWorkerType::Prefill,
            ForwardPassWorkerType::Decode,
            ForwardPassWorkerType::Aggregated,
        ] {
            config.worker_type = role;
            assert_eq!(
                config
                    .fpm_resources()
                    .unwrap()
                    .unwrap()
                    .runtime_memory
                    .unwrap()
                    .kv_cache_bytes,
                600
            );
        }
        assert!(
            config.fpm_profile.as_ref().unwrap()["deployments"][0]
                .get("worker_type")
                .is_none()
        );
    }

    #[test]
    fn resource_selection_rejects_ambiguous_shared_and_role_specific_profiles() {
        let mut config = role_config();
        config.fpm_profile.as_mut().unwrap()["deployments"][0]
            .as_object_mut()
            .unwrap()
            .remove("worker_type");
        assert!(
            config
                .fpm_resources()
                .unwrap_err()
                .to_string()
                .contains("cannot mix legacy")
        );
    }

    #[test]
    fn pending_memory_validates_but_cannot_produce_a_cache_budget() {
        let mut pending = resources();
        pending.activations_bytes = None;
        pending.validate().unwrap();
        let saved = serde_json::to_value(&pending).unwrap();
        assert!(saved.get("activations_bytes").is_none());
        assert!(saved.get("runtime_memory").is_none());
        assert_eq!(saved["weights_bytes"], 100);
        assert_eq!(
            serde_json::from_value::<FpmResourceConfig>(saved).unwrap(),
            pending
        );
        assert!(
            pending
                .estimate_budget(&request(), 4096)
                .unwrap_err()
                .to_string()
                .contains("pending runtime profiling")
        );
    }

    #[test]
    fn runtime_memory_preserves_observed_capacity_without_a_breakdown() {
        let mut runtime = runtime_resources();
        runtime.validate().unwrap();
        let mut q = request();
        q.cuda_graph_reserved_bytes = 0;
        let grouped = runtime.estimate_budget(&q, 4096).unwrap();
        assert_eq!(grouped.total_kv_size_bytes, 600);
        assert_eq!(grouped.total_kv_size_tokens, None);
        assert_eq!(grouped.memory_breakdown, None);
        assert_eq!(grouped.request_peak_cache_bytes, 1760);

        runtime.cache_layout = FpmCacheLayout::Linear;
        runtime.cache_groups.clear();
        runtime.kv_bytes_per_token = Some(10);
        q.total_gpu_capacity_bytes = 2000;
        let linear = runtime.estimate_budget(&q, 4096).unwrap();
        // Runtime recorded 600 bytes: a larger device does not double capacity.
        // Ten bytes/token gives 60 raw tokens and 54 with a 10% margin.
        assert_eq!(linear.total_kv_size_bytes, 600);
        assert_eq!(linear.total_kv_size_tokens, Some(60));
        let adjusted = linear.tolerance_adjusted.unwrap();
        assert_eq!(adjusted.total_kv_size_bytes, 540);
        assert_eq!(adjusted.total_kv_size_tokens, Some(54));
        assert_eq!(linear.memory_breakdown, None);
        let saved = serde_json::to_value(&runtime).unwrap();
        assert!(saved.get("weights_bytes").is_none());
        assert_eq!(
            serde_json::from_value::<FpmResourceConfig>(saved).unwrap(),
            runtime
        );
    }

    #[test]
    fn steady_occupancy_keeps_runtime_scheduler_and_context_admission() {
        let runtime = runtime_resources();
        let mut q = request();
        q.cuda_graph_reserved_bytes = 0;
        q.request_occupancy_tokens = Some(1152);
        let budget = runtime.estimate_budget(&q, 4096).unwrap();
        // A 32-token retained window can straddle three 16-token pages.
        // A 128-token prefill chunk needs eleven pages. Each page is 160 bytes.
        assert_eq!(budget.request_occupancy_cache_bytes, Some(480));
        assert_eq!(budget.request_peak_cache_bytes, 1760);
        q.max_num_tokens = 1;
        assert!(runtime.estimate_budget(&q, 4096).is_err());
        q.max_num_tokens = 128;
        assert!(runtime.estimate_budget(&q, 8192).is_err());
        q.request_occupancy_tokens = Some(4097);
        assert!(runtime.estimate_budget(&q, 4096).is_err());
    }

    #[test]
    fn runtime_memory_rejects_changed_settings_and_invalid_evidence() {
        let runtime = runtime_resources();
        let mut q = request();
        q.cuda_graph_reserved_bytes = 0;
        for changed in [
            FpmCacheBudgetRequest {
                max_num_tokens: 64,
                ..q.clone()
            },
            FpmCacheBudgetRequest {
                max_batch_size: 4,
                ..q.clone()
            },
            FpmCacheBudgetRequest {
                memory_fraction_value: 0.9,
                ..q.clone()
            },
            FpmCacheBudgetRequest {
                cuda_graph_reserved_bytes: 1,
                ..q.clone()
            },
            FpmCacheBudgetRequest {
                total_gpu_capacity_bytes: 599,
                ..q.clone()
            },
            FpmCacheBudgetRequest {
                total_gpu_capacity_bytes: 749,
                ..q.clone()
            },
        ] {
            assert!(runtime.estimate_budget(&changed, 4096).is_err());
        }
        assert!(runtime.estimate_budget(&q, 4097).is_err());
        let mut mixed = runtime.clone();
        mixed.weights_bytes = Some(0);
        assert!(mixed.validate().is_err());
        let mut invalid = runtime.runtime_memory.clone().unwrap();
        for bytes in [0, MAX_EXACT_BYTES + 1] {
            invalid.kv_cache_bytes = bytes;
            assert!(invalid.validate().is_err());
        }
        invalid.kv_cache_bytes = MAX_EXACT_BYTES;
        invalid.validate().unwrap();
        for fraction in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            invalid.gpu_memory_utilization = fraction;
            assert!(invalid.validate().is_err());
        }
        invalid.gpu_memory_utilization = 1.0;
        invalid.max_model_len = 0;
        assert!(invalid.validate().is_err());
        invalid.max_model_len = 1;
        invalid.provenance = " ".into();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn runtime_memory_records_prefix_caching_evidence() {
        let plain = runtime_resources().runtime_memory.unwrap();
        // A record without the evidence serializes without its keys, as before.
        let plain_json = r#"{"kv_cache_bytes":600,"gpu_memory_utilization":0.8,"max_model_len":4096,"provenance":"Synthetic initialized-worker cache allocation"}"#;
        assert_eq!(serde_json::to_string(&plain).unwrap(), plain_json);
        let with = |caching, mode, hash_block| FpmRuntimeMemoryConfig {
            enable_prefix_caching: caching,
            mamba_cache_mode: mode,
            hash_block_size_tokens: hash_block,
            ..plain.clone()
        };

        // The probe's settings are valid and round trip, after the existing keys.
        let evidence = with(Some(true), Some(FpmMambaCacheMode::Align), Some(4));
        evidence.validate().unwrap();
        let saved = serde_json::to_string(&evidence).unwrap();
        assert_eq!(
            saved,
            r#"{"kv_cache_bytes":600,"gpu_memory_utilization":0.8,"max_model_len":4096,"provenance":"Synthetic initialized-worker cache allocation","enable_prefix_caching":true,"mamba_cache_mode":"align","hash_block_size_tokens":4}"#
        );
        assert_eq!(
            serde_json::from_str::<FpmRuntimeMemoryConfig>(&saved).unwrap(),
            evidence
        );
        // An explicit null is an absent value and is not written back.
        let mut nulls = serde_json::to_value(&plain).unwrap();
        for key in [
            "enable_prefix_caching",
            "mamba_cache_mode",
            "hash_block_size_tokens",
        ] {
            nulls[key] = serde_json::Value::Null;
        }
        let parsed = serde_json::from_value::<FpmRuntimeMemoryConfig>(nulls).unwrap();
        assert_eq!(parsed, plain);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), plain_json);

        // The resource section validates the record, so it carries the evidence and the rules.
        let mut resources = runtime_resources();
        resources.runtime_memory = Some(evidence.clone());
        resources.validate().unwrap();
        let saved = serde_json::to_value(&resources).unwrap();
        assert_eq!(saved["runtime_memory"]["mamba_cache_mode"], "align");
        assert_eq!(
            serde_json::from_value::<FpmResourceConfig>(saved).unwrap(),
            resources
        );
        resources.runtime_memory = Some(with(Some(false), Some(FpmMambaCacheMode::Align), None));
        assert_eq!(
            resources.validate().unwrap_err().to_string(),
            "invalid engine config: mamba_cache_mode align requires enable_prefix_caching=true"
        );

        // `none` and an absent mode need no prefix caching; `align` does.
        for caching in [None, Some(false), Some(true)] {
            for mode in [None, Some(FpmMambaCacheMode::None)] {
                with(caching, mode, None).validate().unwrap();
            }
        }
        for caching in [None, Some(false)] {
            let align = with(caching, Some(FpmMambaCacheMode::Align), Some(4));
            assert_eq!(
                memory_rejection(&align),
                "mamba_cache_mode align requires enable_prefix_caching=true"
            );
        }
        // A hash block size must be positive; the largest u32 is valid.
        assert_eq!(
            memory_rejection(&with(None, None, Some(0))),
            "hash_block_size_tokens must be positive"
        );
        for size in [1, u32::MAX] {
            with(None, None, Some(size)).validate().unwrap();
        }
        // With both rules broken, the mode rule reports first, as in the Python model.
        let both = with(Some(false), Some(FpmMambaCacheMode::Align), Some(0));
        assert_eq!(
            memory_rejection(&both),
            "mamba_cache_mode align requires enable_prefix_caching=true"
        );

        // The wire names are `none` and `align`. vLLM's `all` is refused, and so is any other text.
        let mut wire = serde_json::to_value(&evidence).unwrap();
        for (text, mode) in [
            ("none", FpmMambaCacheMode::None),
            ("align", FpmMambaCacheMode::Align),
        ] {
            wire["mamba_cache_mode"] = text.into();
            let parsed = serde_json::from_value::<FpmRuntimeMemoryConfig>(wire.clone()).unwrap();
            assert_eq!(parsed.mamba_cache_mode, Some(mode));
        }
        for text in ["all", "Align", ""] {
            wire["mamba_cache_mode"] = text.into();
            assert!(serde_json::from_value::<FpmRuntimeMemoryConfig>(wire.clone()).is_err());
        }
    }

    #[test]
    fn grouped_page_boundaries_keep_full_history_and_evict_windows() {
        let local = group(Some(32));
        // First 32 tokens occupy two physical pages. Token 33 retains parts of
        // pages 0, 1, and 2; page 0 disappears only when its history is obsolete.
        assert_eq!(local.bytes_for_forward(0, 32).unwrap(), 320);
        assert_eq!(local.block_range(32, 33).unwrap(), 0..3);
        assert_eq!(local.block_range(48, 49).unwrap(), 1..4);
        assert_eq!(local.bytes_for_forward(64, 65).unwrap(), 480);
        assert_eq!(group(None).bytes_for_forward(64, 65).unwrap(), 800);
        // A 128-token prefill chunk keeps ten pages here, exceeding the window.
        assert_eq!(local.block_range(512, 640).unwrap(), 30..40);
        assert_eq!(local.bytes_for_forward(512, 640).unwrap(), 1600);
        assert!(local.bytes_for_forward(33, 32).is_err());
    }

    #[test]
    fn cache_budget_keeps_byte_tolerance_without_inventing_token_capacity() {
        let result = resources().estimate_budget(&request(), 4096).unwrap();
        // 80% of 1000, minus 100+20+30+50 non-KV and 20 graph bytes.
        assert_eq!(result.total_kv_size_bytes, 580);
        assert_eq!(
            result
                .tolerance_adjusted
                .as_ref()
                .unwrap()
                .total_kv_size_bytes,
            522
        );
        assert_eq!(result.total_kv_size_tokens, None);
        assert_eq!(result.kv_size_per_token_bytes, None);
        assert_eq!(
            result.tolerance_adjusted.unwrap().total_kv_size_tokens,
            None
        );
        // At most 11 aligned pages cover 31 history + 128 scheduled tokens.
        assert_eq!(result.request_peak_cache_bytes, 1760);
        assert_eq!(
            result.memory_breakdown.unwrap().cuda_graph_reserved_bytes,
            20
        );
    }

    #[test]
    fn grouped_resources_validate_layout_and_exact_bytes() {
        let original = resources();
        assert_eq!(
            serde_json::from_str::<FpmResourceConfig>(&serde_json::to_string(&original).unwrap())
                .unwrap(),
            original
        );
        let mut bad = original.clone();
        bad.kv_bytes_per_token = Some(1);
        assert!(bad.validate().is_err());
        bad = original.clone();
        bad.cache_groups.push(bad.cache_groups[0].clone());
        assert!(bad.validate().is_err());
        bad = original.clone();
        bad.cache_groups[0].page_size_bytes = MAX_EXACT_BYTES;
        assert!(bad.cache_groups[0].bytes_for_forward(0, 17).is_err());
        bad.weights_bytes = Some(u64::MAX);
        assert!(bad.validate().is_err());
        let mut unknown = serde_json::to_value(original).unwrap();
        unknown["cache_groups"][0]["unknown_page_semantics"] = true.into();
        assert!(serde_json::from_value::<FpmResourceConfig>(unknown).is_err());
    }

    #[test]
    fn grouped_budget_rejects_invalid_fraction_envelope_and_capacity() {
        let r = resources();
        for fraction in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut q = request();
            q.memory_fraction_value = fraction;
            assert!(r.estimate_budget(&q, 4096).is_err());
        }
        for tolerance in [f64::NAN, -0.1, 1.0] {
            let mut q = request();
            q.tolerance_fraction = Some(tolerance);
            assert!(r.estimate_budget(&q, 4096).is_err());
        }
        let mut q = request();
        q.max_batch_size = 9;
        assert!(r.estimate_budget(&q, 4096).is_err());
        q = request();
        q.total_gpu_capacity_bytes = 0;
        assert!(r.estimate_budget(&q, 4096).is_err());
        q = request();
        q.memory_fraction_value = 0.1;
        assert!(r.estimate_budget(&q, 4096).is_err());
        q = request();
        q.cuda_graph_reserved_bytes = MAX_EXACT_BYTES;
        assert!(r.estimate_budget(&q, 4096).is_err());
    }

    #[test]
    fn convolution_requires_an_explicit_retention_window() {
        let mut conv = group(None);
        conv.kind = FpmCacheKind::Convolution;
        assert!(conv.validate().is_err());
        conv.sliding_window = Some(4);
        conv.block_size_tokens = 4;
        conv.page_size_bytes = 8;
        // Token 9 uses pages containing 5..8 and 9..12, even for a four-token window.
        assert_eq!(conv.block_range(8, 9).unwrap(), 1..3);
        assert_eq!(conv.bytes_for_forward(8, 9).unwrap(), 16);
    }

    #[test]
    fn state_group_requires_blocks_and_rejects_windows() {
        state_group(Some(1)).validate().unwrap();
        for blocks in [None, Some(0)] {
            let state = state_group(blocks);
            assert_eq!(
                rejection(&state),
                "state cache groups require a positive blocks_per_request"
            );
            // An unusable group fails sizing instead of guessing a block count.
            assert!(state.block_range(0, 1).is_err());
            assert!(state.peak_request_bytes(1, 1).is_err());
        }
        let mut windowed = state_group(Some(1));
        windowed.sliding_window = Some(32);
        assert_eq!(
            rejection(&windowed),
            "state cache groups cannot have a sliding_window"
        );
        for (kind, window) in [
            (FpmCacheKind::Attention, None),
            (FpmCacheKind::Convolution, Some(32)),
        ] {
            let mut counted = group(window);
            counted.kind = kind;
            counted.blocks_per_request = Some(1);
            assert_eq!(
                rejection(&counted),
                "blocks_per_request applies only to state cache groups"
            );
        }
    }

    #[test]
    fn state_group_holds_a_fixed_range_for_any_length() {
        let state = state_group(Some(2));
        // No tokens, no state; any positive target holds the same two blocks.
        assert_eq!(state.block_range(0, 0).unwrap(), 0..0);
        assert_eq!(state.block_range(0, 1).unwrap(), 0..2);
        assert_eq!(state.block_range(4, 8).unwrap(), 0..2);
        assert_eq!(state.block_range(100_000, 100_001).unwrap(), 0..2);
        assert!(state.block_range(5, 4).is_err());
        assert_eq!(state.bytes_for_forward(0, 0).unwrap(), 0);
        assert_eq!(state.bytes_for_forward(100_000, 100_001).unwrap(), 320);
        let two_blocks = 2 * state.page_size_bytes;
        assert_eq!(state.peak_request_bytes(1, 1).unwrap(), two_blocks);
        assert_eq!(
            state.peak_request_bytes(1_000_000, 8_192).unwrap(),
            two_blocks
        );
        // The exact-byte limit still applies: 2**53 passes, twice that fails.
        let mut huge = state_group(Some(1));
        huge.page_size_bytes = MAX_EXACT_BYTES;
        assert_eq!(huge.peak_request_bytes(1, 1).unwrap(), MAX_EXACT_BYTES);
        huge.blocks_per_request = Some(2);
        assert!(huge.bytes_for_forward(0, 1).is_err());
        assert!(huge.peak_request_bytes(1, 1).is_err());
    }

    #[test]
    fn glm_shaped_groups_charge_five_pool_blocks_of_state() {
        // GLM-5.3-Flash at TP 4, fp8, k = 0: one MLA indexer group and five
        // recurrent-state groups. Every group uses 2,176-token pool blocks of
        // 13,045,120 bytes, and each state group holds one block per request.
        let pool_group = |name: &str, kind, num_layers, blocks_per_request| FpmCacheGroup {
            name: name.into(),
            kind,
            num_layers,
            block_size_tokens: 2_176,
            page_size_bytes: 13_045_120,
            sliding_window: None,
            blocks_per_request,
        };
        let attention = pool_group("mla_indexer", FpmCacheKind::Attention, 22, None);
        let states = [
            pool_group("kpool_tail", FpmCacheKind::State, 11, Some(1)),
            pool_group("kda_0", FpmCacheKind::State, 9, Some(1)),
            pool_group("kda_1", FpmCacheKind::State, 9, Some(1)),
            pool_group("kda_2", FpmCacheKind::State, 8, Some(1)),
            pool_group("kda_3", FpmCacheKind::State, 8, Some(1)),
        ];
        let peak = |group: &FpmCacheGroup| group.peak_request_bytes(8_192, 8_192).unwrap();
        // 8,192 tokens fill four attention blocks; the state adds five more.
        assert_eq!(peak(&attention), 52_180_480);
        let state_bytes: u64 = states.iter().map(peak).sum();
        assert_eq!(state_bytes, 65_225_600);
        assert_eq!(peak(&attention) + state_bytes, 117_406_080);

        // The budget sums the same groups for its per-request bounds.
        let mut grouped = resources();
        grouped.cache_groups = std::iter::once(attention).chain(states).collect();
        grouped.max_num_tokens = 8_192;
        let mut query = request();
        query.max_num_tokens = 8_192;
        query.request_occupancy_tokens = Some(8_192);
        let budget = grouped.estimate_budget(&query, 8_192).unwrap();
        assert_eq!(budget.request_peak_cache_bytes, 117_406_080);
        assert_eq!(budget.request_occupancy_cache_bytes, Some(117_406_080));
    }

    #[test]
    fn state_group_serde_round_trip_omits_absent_blocks() {
        let attention = group(Some(32));
        // A group without a block count serializes exactly as before.
        assert_eq!(
            serde_json::to_string(&attention).unwrap(),
            r#"{"name":"attention","kind":"attention","num_layers":2,"block_size_tokens":16,"page_size_bytes":160,"sliding_window":32}"#
        );
        let state = state_group(Some(1));
        let saved = serde_json::to_value(&state).unwrap();
        assert_eq!(saved["kind"], "state");
        assert_eq!(saved["blocks_per_request"], 1);
        for original in [attention, state] {
            let saved = serde_json::to_string(&original).unwrap();
            assert_eq!(
                serde_json::from_str::<FpmCacheGroup>(&saved).unwrap(),
                original
            );
        }
    }
}
