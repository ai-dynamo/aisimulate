// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Neutral engine launch configuration and legacy input conversion.
//!
//! Compatibility mappings adapted and modified from Dynamo's
//! lib/mocker/src/common/protocols.rs at d15ec1dda0e30b5c2513b7dc85bcd372d553035e:
//! https://github.com/ai-dynamo/dynamo/blob/d15ec1dda0e30b5c2513b7dc85bcd372d553035e/lib/mocker/src/common/protocols.rs

use std::ops::{Deref, DerefMut};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{EngineConfig, TimingModelConfig, WorkerType, normalize_conditional_accept_rates};
use crate::ForwardPassPerfModelConfig;

/// Legacy SGLang input overrides; defaults and validation belong to `EngineConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SglangOverrides {
    pub schedule_policy: Option<String>,
    pub page_size: Option<usize>,
    pub max_prefill_tokens: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub clip_max_new_tokens: Option<usize>,
    pub schedule_conservativeness: Option<f64>,
}

/// Legacy TensorRT-LLM input converted into its canonical typed policy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrtllmOverrides {
    pub capacity_scheduler_policy: Option<String>,
}

macro_rules! enum_from_str {
    ($($ty:ty),+ $(,)?) => {$(
        impl std::str::FromStr for $ty {
            type Err = String;

            fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
                serde_json::from_value(json!(value.to_ascii_lowercase())).map_err(|error| error.to_string())
            }
        }
    )+};
}

enum_from_str!(
    super::Backend,
    super::WorkerType,
    super::PreemptionMode,
    super::TransferTimingMode
);

/// An engine rank plus neutral launch and capacity-estimation inputs.
///
/// Engine fields are stored only in [`EngineConfig`]. Integration-specific
/// ports, publishers and output handlers do not belong in this contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineLaunchConfig {
    pub engine: EngineConfig,
    pub dp_size: u32,
    pub tensor_parallel_size: usize,
    pub startup_time: Option<f64>,
    pub gpu_memory_utilization: Option<f64>,
    pub mem_fraction_static: Option<f64>,
    pub free_gpu_memory_fraction: Option<f64>,
}

impl Default for EngineLaunchConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            dp_size: 1,
            tensor_parallel_size: 1,
            startup_time: None,
            gpu_memory_utilization: None,
            mem_fraction_static: None,
            free_gpu_memory_fraction: None,
        }
    }
}

impl Deref for EngineLaunchConfig {
    type Target = EngineConfig;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for EngineLaunchConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

impl EngineLaunchConfig {
    /// Parse a canonical launch object or the historical flat engine arguments.
    /// Unknown fields and conflicting aliases are rejected, never discarded.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut fields = value
            .as_object()
            .context("engine arguments must be an object")?
            .clone();
        if fields.contains_key("engine") {
            return serde_json::from_value::<Self>(Value::Object(fields))?.normalized();
        }
        let mut launch = Map::new();
        for key in [
            "dp_size",
            "tensor_parallel_size",
            "startup_time",
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
        ] {
            if let Some(value) = fields.remove(key) {
                launch.insert(key.to_owned(), value);
            }
        }
        alias(&mut fields, "engine_type", "backend")?;
        alias(&mut fields, "ais_nextn", "aic_nextn")?;
        alias(
            &mut fields,
            "ais_nextn_accept_rates",
            "aic_nextn_accept_rates",
        )?;
        alias(&mut fields, "ais_mtp_seed", "aic_mtp_seed")?;
        alias(
            &mut fields,
            "kv_bytes_per_token",
            "kv_transfer_bytes_per_token",
        )?;
        for key in [
            "backend",
            "worker_type",
            "preemption_mode",
            "kv_transfer_timing_mode",
        ] {
            if let Some(Value::String(value)) = fields.get_mut(key) {
                *value = value.to_ascii_lowercase();
            }
        }
        let is_prefill = take_bool(&mut fields, "is_prefill")?;
        let is_decode = take_bool(&mut fields, "is_decode")?;
        if !fields.contains_key("worker_type") {
            ensure!(
                !(is_prefill && is_decode),
                "is_prefill and is_decode cannot both be true"
            );
            if is_prefill || is_decode {
                fields.insert(
                    "worker_type".into(),
                    json!(if is_prefill { "prefill" } else { "decode" }),
                );
            }
        }
        // The historical CLI uses zero for an unspecified backend-native block size.
        if fields.get("block_size") == Some(&json!(0)) {
            fields.remove("block_size");
        }
        if fields.get("num_gpu_blocks") == Some(&Value::Null) {
            fields.remove("num_gpu_blocks");
        }
        for key in ["max_num_seqs", "max_num_batched_tokens"] {
            if fields.get(key) == Some(&Value::Null) {
                fields.insert(key.into(), json!(usize::MAX));
            }
        }
        for key in ["sglang", "trtllm", "timing_model"] {
            if fields.get(key) == Some(&Value::Null) {
                fields.remove(key);
            }
        }
        if let Some(value) = fields.remove("sglang") {
            let mut sglang = value
                .as_object()
                .context("sglang must be an object")?
                .clone();
            remove_null_overrides(
                &mut sglang,
                &[
                    "schedule_policy",
                    "page_size",
                    "max_prefill_tokens",
                    "chunked_prefill_size",
                    "clip_max_new_tokens",
                    "schedule_conservativeness",
                ],
            );
            if let Some(page_size) = sglang.remove("page_size") {
                insert_unique(&mut fields, "block_size", page_size)?;
            }
            if sglang.get("schedule_policy").and_then(Value::as_str) == Some("fcfs") {
                sglang.insert("schedule_policy".into(), json!("fifo"));
            }
            fields.insert("sglang".into(), Value::Object(sglang));
        }
        if let Some(Value::Object(trtllm)) = fields.get_mut("trtllm") {
            remove_null_overrides(trtllm, &["capacity_scheduler_policy"]);
        }
        if let Some(config) = fields
            .remove("ais_perf_config")
            .filter(|value| !value.is_null())
        {
            let timing = json!({"type":"external", "provider":"ais", "config":config});
            if let Some(existing) = fields.get_mut("timing_model") {
                if existing.get("provider").and_then(Value::as_str) == Some("aic") {
                    existing["provider"] = json!("ais");
                }
            }
            insert_unique(&mut fields, "timing_model", timing)?;
        }
        if let Some(Value::Object(timing)) = fields.get_mut("timing_model") {
            if timing.get("type").and_then(Value::as_str) == Some("default") {
                timing.insert("type".into(), json!("polynomial"));
            }
            if timing.get("type").and_then(Value::as_str) == Some("fixed") {
                for key in ["prefill_ms", "decode_ms"] {
                    if let Some(number) = timing.get(key).and_then(Value::as_f64) {
                        timing.insert(key.into(), json!(number));
                    }
                }
            }
        }
        remove_null_overrides(&mut launch, &["tensor_parallel_size"]);
        let explicit_tp = launch.contains_key("tensor_parallel_size");
        let explicit_dp = launch.contains_key("dp_size");
        let explicit_nextn = fields
            .get("aic_nextn")
            .is_some_and(|value| !value.is_null());
        launch.insert("engine".into(), Value::Object(fields));
        let mut result: Self = serde_json::from_value(Value::Object(launch))?;
        // A canonical performance identity supplies omitted launch topology, but
        // never overrides an explicitly conflicting one.
        if let Some(config) = result.performance_config()? {
            if !explicit_tp {
                result.tensor_parallel_size = config.tp as usize;
            }
            if !explicit_dp {
                result.dp_size = config.attention_dp;
            }
            if !explicit_nextn && config.nextn != 0 {
                result.aic_nextn = Some(config.nextn as usize);
            }
        }
        result.normalized()
    }

    /// Canonical AIS performance identity, when this launch uses AIS timing.
    pub fn performance_config(&self) -> Result<Option<ForwardPassPerfModelConfig>> {
        match &self.timing_model {
            TimingModelConfig::External { provider, config }
                if matches!(provider.as_str(), "ais" | "aic") =>
            {
                Ok(Some(
                    serde_json::from_value(config.clone())
                        .context("invalid AIS performance config")?,
                ))
            }
            _ => Ok(None),
        }
    }

    /// Validate launch inputs and normalize shared engine controls.
    pub fn normalized(mut self) -> Result<Self> {
        self.validate()?;
        if let Some(config) = self.performance_config()? {
            self.timing_model = TimingModelConfig::External {
                provider: "ais".to_owned(),
                config: serde_json::to_value(config)?,
            };
        }
        if let Some(nextn) = self.aic_nextn {
            let rates =
                normalize_conditional_accept_rates(nextn, self.aic_nextn_accept_rates.as_deref())?;
            self.aic_nextn_accept_rates = Some(
                rates
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        self.engine.validate()?;
        ensure!(self.dp_size > 0, "dp_size must be positive");
        ensure!(
            self.tensor_parallel_size > 0,
            "tensor_parallel_size must be positive"
        );
        ensure!(
            u32::try_from(self.tensor_parallel_size).is_ok(),
            "tensor_parallel_size exceeds the engine contract"
        );
        if let Some(seconds) = self.startup_time {
            ensure!(
                seconds.is_finite() && seconds >= 0.0,
                "startup_time must be finite and non-negative"
            );
        }
        for (key, value) in [
            ("gpu_memory_utilization", self.gpu_memory_utilization),
            ("mem_fraction_static", self.mem_fraction_static),
            ("free_gpu_memory_fraction", self.free_gpu_memory_fraction),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "{key} must be finite and in [0, 1]"
                );
            }
        }
        if let Some(config) = self.performance_config()? {
            config.validate()?;
            ensure!(
                config.pp == 1,
                "engine launch supports only pp=1; got pp={}",
                config.pp
            );
            ensure!(
                config.tp as usize == self.tensor_parallel_size,
                "tensor_parallel_size conflicts with canonical tp"
            );
            ensure!(
                config.attention_dp == self.dp_size,
                "dp_size conflicts with canonical attention_dp"
            );
            let identity = serde_json::to_value(&config)?;
            ensure!(
                identity["worker_type"] == serde_json::to_value(self.worker_type)?,
                "AIS worker_type must match engine role"
            );
            ensure!(
                identity
                    .pointer("/speculation/kind")
                    .and_then(Value::as_str)
                    != Some("ngram"),
                "engine launch does not implement ngram draft scheduling"
            );
            ensure!(
                config.nextn as usize == self.aic_nextn.unwrap_or(0),
                "canonical nextn conflicts with scheduler aic_nextn"
            );
        }
        Ok(())
    }

    pub fn is_prefill(&self) -> bool {
        self.worker_type == WorkerType::Prefill
    }

    pub fn is_decode(&self) -> bool {
        self.worker_type == WorkerType::Decode
    }
}

fn alias(fields: &mut Map<String, Value>, old: &str, canonical: &str) -> Result<()> {
    if let Some(value) = fields.remove(old) {
        insert_unique(fields, canonical, value)?;
    }
    Ok(())
}

fn insert_unique(fields: &mut Map<String, Value>, key: &str, value: Value) -> Result<()> {
    if let Some(existing) = fields.get(key) {
        ensure!(*existing == value, "conflicting values for {key}");
    } else {
        fields.insert(key.to_owned(), value);
    }
    Ok(())
}

fn take_bool(fields: &mut Map<String, Value>, key: &str) -> Result<bool> {
    match fields.remove(key) {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(value),
        Some(_) => bail!("{key} must be a boolean"),
    }
}

fn remove_null_overrides(fields: &mut Map<String, Value>, keys: &[&str]) {
    for key in keys {
        if fields.get(*key) == Some(&Value::Null) {
            fields.remove(*key);
        }
    }
}
