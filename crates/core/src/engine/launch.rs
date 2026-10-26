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
        Self::parse_value(value)?.normalized()
    }

    /// Parse and derive omitted launch controls without validating them.
    fn parse_value(value: Value) -> Result<Self> {
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
        Ok(result)
    }

    /// Check whether `config` can time the engine described by launch
    /// arguments `engine_args`, as when a saved AIS configuration replaces the
    /// engine's timing model. The arguments use the [`Self::from_value`] shape;
    /// their timing model, if any, is replaced by `config`, and omitted launch
    /// controls derive from it exactly as at launch. Returns every launch rule
    /// that would reject the combination, in the order launch validation
    /// checks them; an empty list means the launch would accept it.
    /// Malformed arguments or an invalid `config` are errors, not mismatches.
    pub fn timing_config_mismatches(
        mut engine_args: Value,
        config: &ForwardPassPerfModelConfig,
    ) -> Result<Vec<String>> {
        config.validate()?;
        let engine = engine_args
            .as_object_mut()
            .context("engine launch arguments must be an object")?
            .entry("engine")
            .or_insert_with(|| json!({}));
        engine
            .as_object_mut()
            .context("engine launch arguments field 'engine' must be an object")?
            .insert(
                "timing_model".into(),
                json!({"type": "external", "provider": "ais", "config": config}),
            );
        let launch = Self::parse_value(engine_args)?;
        launch.engine.validate()?;
        launch.timing_identity_mismatches(config)
    }

    /// Launch rules relating a canonical AIS timing identity to this engine.
    fn timing_identity_mismatches(
        &self,
        config: &ForwardPassPerfModelConfig,
    ) -> Result<Vec<String>> {
        let mut mismatches = Vec::new();
        if config.pp != 1 {
            mismatches.push(format!(
                "engine launch supports only pp=1; got pp={}",
                config.pp
            ));
        }
        if config.tp as usize != self.tensor_parallel_size {
            mismatches.push("tensor_parallel_size conflicts with canonical tp".into());
        }
        if config.attention_dp != self.dp_size {
            mismatches.push("dp_size conflicts with canonical attention_dp".into());
        }
        let identity = serde_json::to_value(config)?;
        if identity["worker_type"] != serde_json::to_value(self.worker_type)? {
            mismatches.push("AIS worker_type must match engine role".into());
        }
        if identity
            .pointer("/speculation/kind")
            .and_then(Value::as_str)
            == Some("ngram")
        {
            mismatches.push("engine launch does not implement ngram draft scheduling".into());
        }
        if config.nextn as usize != self.aic_nextn.unwrap_or(0) {
            mismatches.push("canonical nextn conflicts with scheduler aic_nextn".into());
        }
        Ok(mismatches)
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
            if let Some(first) = self.timing_identity_mismatches(&config)?.into_iter().next() {
                anyhow::bail!(first);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendKind, EstimationMode, ForwardPassWorkerType};

    fn timing() -> ForwardPassPerfModelConfig {
        let mut config = ForwardPassPerfModelConfig::new(
            "test/model",
            "test",
            BackendKind::Vllm,
            ForwardPassWorkerType::Aggregated,
        );
        config.estimation_mode = EstimationMode::FpmRegression;
        config
    }

    #[test]
    fn timing_config_mismatches_list_every_launch_rule_in_validation_order() {
        let mut config = timing();
        config.pp = 2;
        config.tp = 2;
        config.attention_dp = 4;
        let args =
            json!({"dp_size": 2, "tensor_parallel_size": 1, "engine": {"worker_type": "decode"}});
        let mismatches =
            EngineLaunchConfig::timing_config_mismatches(args.clone(), &config).unwrap();
        assert_eq!(
            mismatches,
            [
                "engine launch supports only pp=1; got pp=2",
                "tensor_parallel_size conflicts with canonical tp",
                "dp_size conflicts with canonical attention_dp",
                "AIS worker_type must match engine role",
            ]
        );
        // Launch itself rejects the same combination with the first rule.
        let mut launch = args;
        launch["engine"]["timing_model"] =
            json!({"type": "external", "provider": "ais", "config": config});
        let error = EngineLaunchConfig::from_value(launch).unwrap_err();
        assert!(format!("{error:#}").contains(&mismatches[0]), "{error:#}");
    }

    #[test]
    fn timing_config_mismatches_derive_omitted_controls_like_launch() {
        let mut config = timing();
        config.tp = 2;
        config.attention_dp = 4;
        // Omitted tensor_parallel_size and dp_size derive from the timing identity.
        let args = json!({"engine": {}});
        assert!(
            EngineLaunchConfig::timing_config_mismatches(args.clone(), &config)
                .unwrap()
                .is_empty()
        );
        let mut launch = args;
        launch["engine"]["timing_model"] =
            json!({"type": "external", "provider": "ais", "config": config});
        let launched = EngineLaunchConfig::from_value(launch).unwrap();
        assert_eq!((launched.tensor_parallel_size, launched.dp_size), (2, 4));
        // Invalid timing identities and malformed arguments are errors, not mismatches.
        config.tp = 0;
        assert!(EngineLaunchConfig::timing_config_mismatches(json!({}), &config).is_err());
        assert!(EngineLaunchConfig::timing_config_mismatches(json!([]), &timing()).is_err());
        assert!(
            EngineLaunchConfig::timing_config_mismatches(json!({"dp_size": "two"}), &timing())
                .is_err()
        );
    }
}
