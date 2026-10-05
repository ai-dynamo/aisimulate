// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical engine rank, launch topology, and capacity-estimation inputs.

use std::ops::{Deref, DerefMut};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{EngineConfig, TimingModelConfig, WorkerType, normalize_conditional_accept_rates};
use crate::ForwardPassPerfModelConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineLaunchConfig {
    pub engine: EngineConfig,
    pub dp_size: u32,
    pub tensor_parallel_size: usize,
    pub startup_time: Option<f64>,
    /// False keeps inferred capacity automatic through serialization.
    pub num_gpu_blocks_is_explicit: bool,
    pub gpu_memory_utilization: Option<f64>,
    pub mem_fraction_static: Option<f64>,
    pub free_gpu_memory_fraction: Option<f64>,
    pub cuda_graph_reserved_bytes: Option<u64>,
}

impl Default for EngineLaunchConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            dp_size: 1,
            tensor_parallel_size: 1,
            startup_time: None,
            num_gpu_blocks_is_explicit: false,
            gpu_memory_utilization: None,
            mem_fraction_static: None,
            free_gpu_memory_fraction: None,
            cuda_graph_reserved_bytes: None,
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
    /// Parse the canonical launch shape. No downstream argument aliases are accepted.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut result: Self = serde_path_to_error::deserialize(value.clone())
            .context("invalid engine launch config")?;
        if value.get("num_gpu_blocks_is_explicit").is_none() {
            result.num_gpu_blocks_is_explicit = value.pointer("/engine/num_gpu_blocks").is_some();
        }
        if let Some(config) = result.performance_config()? {
            if value.get("tensor_parallel_size").is_none() {
                result.tensor_parallel_size = config.tp as usize;
            }
            if value.get("dp_size").is_none() {
                result.dp_size = config.attention_dp;
            }
            if value.pointer("/engine/aic_nextn").is_none() && config.nextn > 0 {
                result.aic_nextn = Some(config.nextn as usize);
            }
        }
        result.normalized()
    }

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

    /// A scheduling sentinel is not a physical activation-memory budget.
    pub fn capacity_estimation_options(&self) -> Map<String, Value> {
        let mut options = Map::new();
        options.insert("block_size".into(), json!(self.block_size));
        for (key, value) in [
            ("max_num_seqs", self.max_num_seqs),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
        ] {
            if value != usize::MAX {
                options.insert(key.into(), json!(value));
            }
        }
        for (key, value) in [
            ("gpu_memory_utilization", self.gpu_memory_utilization),
            ("mem_fraction_static", self.mem_fraction_static),
            ("free_gpu_memory_fraction", self.free_gpu_memory_fraction),
        ] {
            if let Some(value) = value {
                options.insert(key.into(), json!(value));
            }
        }
        if let Some(value) = self.cuda_graph_reserved_bytes {
            options.insert("cuda_graph_reserved_bytes".into(), json!(value));
        }
        options
    }

    /// Validate launch inputs and normalize shared engine controls.
    pub fn normalized(mut self) -> Result<Self> {
        self.validate()?;
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
