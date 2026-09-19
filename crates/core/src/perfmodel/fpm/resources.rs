// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rank-local cache descriptors and timing-independent profile memory sizing.

use serde::{Deserialize, Serialize};

use super::{ForwardPassPerfModel, ForwardPassPerfModelConfig};
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
}

/// One cache group on one TP rank. `page_size_bytes` includes every layer in
/// this group and any runtime padding; it is not a per-layer or logical-token
/// rate. Window lengths describe retained history, independently of FPM timing
/// query context lengths.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmCacheGroup {
    pub name: String,
    pub kind: FpmCacheKind,
    pub num_layers: u32,
    pub block_size_tokens: u32,
    pub page_size_bytes: u64,
    pub sliding_window: Option<u32>,
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
        Ok(())
    }

    /// Blocks live during one forward pass. All newly scheduled tokens remain
    /// allocated until the next pass, including temporary prefill storage.
    pub fn block_range(
        &self,
        computed_tokens: u64,
        target_tokens: u64,
    ) -> Result<std::ops::Range<u64>, AicError> {
        self.validate()?;
        if target_tokens < computed_tokens {
            return Err(invalid("cache target_tokens must be >= computed_tokens"));
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
    /// would underestimate prefill memory.
    pub fn peak_request_bytes(
        &self,
        context_length: u64,
        max_num_tokens: u32,
    ) -> Result<u64, AicError> {
        self.validate()?;
        if context_length == 0 || max_num_tokens == 0 {
            return Err(invalid("cache context and max_num_tokens must be positive"));
        }
        let block = u64::from(self.block_size_tokens);
        let full = context_length.div_ceil(block);
        let pages = if let Some(window) = self.sliding_window {
            let span = u64::from(window) - 1 + u64::from(max_num_tokens) + block - 1;
            full.min(span.div_ceil(block))
        } else {
            full
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

/// The typed resource section of the existing profile wire schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FpmResourceConfig {
    pub weights_bytes: u64,
    pub activations_bytes: u64,
    pub runtime_overhead_bytes: u64,
    pub comm_overhead_bytes: u64,
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
        self.non_kv_bytes()?;
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

    fn non_kv_bytes(&self) -> Result<u64, AicError> {
        [
            self.weights_bytes,
            self.activations_bytes,
            self.runtime_overhead_bytes,
            self.comm_overhead_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, value| total.checked_add(value))
        .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
        .ok_or_else(|| invalid("total non-KV resource bytes must not exceed 2**53"))
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
    pub memory_breakdown: MemoryBreakdown,
    pub tolerance_adjusted: Option<FpmCacheBudgetAdjusted>,
    pub resource_provenance: String,
    pub cache_layout: FpmCacheLayout,
    pub cache_groups: Vec<FpmCacheGroup>,
    /// Per-request upper bound, including window block alignment and the
    /// configured prefill chunk. This is not an aggregate token capacity.
    pub request_peak_cache_bytes: u64,
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
        if self.backend != BackendKind::Vllm
            || self.pp != 1
            || self.nextn != 0
            || self.speculation.is_some()
        {
            return Err(invalid(
                "FPM resources require vLLM PP1 plain autoregressive execution",
            ));
        }
        let deployments = profile
            .get("deployments")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid("fpm_profile requires deployments"))?;
        let mut matching = deployments.iter().filter(|d| {
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
        });
        let deployment = matching
            .next()
            .ok_or_else(|| invalid("no matching FPM deployment resources"))?;
        if matching.next().is_some() {
            return Err(invalid("duplicate FPM deployment resources"));
        }
        let resources: FpmResourceConfig = serde_json::from_value(
            deployment
                .get("resources")
                .cloned()
                .ok_or_else(|| invalid("FPM deployment requires resources"))?,
        )
        .map_err(|error| invalid(format!("FPM resource profile: {error}")))?;
        resources.validate()?;
        Ok(Some(resources))
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
        let total = available as u64;
        let token_count = |bytes| self.kv_bytes_per_token.map(|rate| bytes / rate);
        let request_peak_cache_bytes = if self.cache_layout == FpmCacheLayout::Linear {
            context.checked_mul(self.kv_bytes_per_token.unwrap())
        } else {
            self.cache_groups.iter().try_fold(0u64, |sum, group| {
                sum.checked_add(
                    group
                        .peak_request_bytes(context, request.max_num_tokens)
                        .ok()?,
                )
            })
        }
        .filter(|bytes| *bytes <= MAX_EXACT_BYTES)
        .ok_or_else(|| invalid("cache request allocation exceeds 2**53 bytes"))?;
        Ok(FpmCacheBudget {
            total_gpu_capacity_bytes: request.total_gpu_capacity_bytes,
            total_kv_size_bytes: total,
            kv_size_per_token_bytes: self.kv_bytes_per_token,
            total_kv_size_tokens: token_count(total),
            source: "profile".into(),
            memory_breakdown: MemoryBreakdown {
                weights_bytes: self.weights_bytes,
                activations_bytes: self.activations_bytes,
                runtime_overhead_bytes: self.runtime_overhead_bytes,
                comm_overhead_bytes: self.comm_overhead_bytes,
                cuda_graph_reserved_bytes: request.cuda_graph_reserved_bytes,
            },
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
        }
    }

    fn resources() -> FpmResourceConfig {
        FpmResourceConfig {
            weights_bytes: 100,
            activations_bytes: 20,
            runtime_overhead_bytes: 30,
            comm_overhead_bytes: 50,
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
            cuda_graph_reserved_bytes: 20,
            tolerance_fraction: Some(0.1),
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
        assert_eq!(result.memory_breakdown.cuda_graph_reserved_bytes, 20);
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
        bad.weights_bytes = u64::MAX;
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
}
