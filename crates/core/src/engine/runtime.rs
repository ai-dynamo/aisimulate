// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory for one logical mock engine.

use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::belady::BeladyOracle;
use crate::engine::generalized::{EngineIdentity, GeneralizedMockerEngine, RankIdentity};
use crate::engine::host_offload::{G2Binding, SharedG2Pool};
use crate::engine::scheduler::{SchedulerRank, engine_seed_offset};
use crate::engine::{EngineConfig, TimingModel};

/// A single-rank or attention-DP engine.
pub type Engine = GeneralizedMockerEngine<SchedulerRank>;

/// Constructs engines with one process-local timing provider.
///
/// The serializable engine configuration contains only a provider descriptor.
/// A Runner resolves external timing providers before constructing this
/// factory, which then shares the provider across all attention-DP ranks.
#[derive(Clone)]
pub struct EngineFactory {
    // Runtime-only value; the serializable replay descriptor retains the declaration.
    config: EngineConfig,
    timing: Arc<dyn TimingModel>,
    belady_oracle: Option<BeladyOracle>,
    g2: Option<G2Binding>,
}

impl EngineFactory {
    /// Construct a factory for a built-in timing model.
    pub fn new(config: EngineConfig) -> Result<Self> {
        let config = config.into_runtime_config()?;
        let timing = config.built_in_timing_model()?;
        Ok(Self {
            config,
            timing,
            belady_oracle: None,
            g2: None,
        })
    }

    /// Construct a factory with a process-local timing provider.
    pub fn with_timing_model(config: EngineConfig, timing: Arc<dyn TimingModel>) -> Result<Self> {
        let config = config.into_runtime_config()?;
        Ok(Self {
            config,
            timing,
            belady_oracle: None,
            g2: None,
        })
    }

    pub(crate) fn with_belady_oracle(mut self, oracle: BeladyOracle) -> Self {
        self.belady_oracle = Some(oracle);
        self
    }

    /// Bind cluster-shared G2 ranks to their deployment pool.
    pub(crate) fn with_g2_binding(mut self, binding: G2Binding) -> Self {
        self.g2 = Some(binding);
        self
    }

    /// Join ranks configured with `host_offload.scope = cluster_shared` to
    /// `pool`. `tensor_parallel_size` is part of the pool contract. Ranks with
    /// private or no G2 ignore the pool.
    pub fn with_shared_g2_pool(self, pool: &SharedG2Pool, tensor_parallel_size: u32) -> Self {
        self.with_g2_binding(G2Binding {
            registry: Arc::clone(&pool.0),
            tensor_parallel_size,
        })
    }

    pub(crate) fn reset_timing_evidence(&self) -> Result<()> {
        self.timing.reset_evidence()
    }

    /// Whether any rank built by this factory can produce internal deadlines.
    /// This is a configuration capability, not whether work is currently pending.
    pub(crate) fn can_have_internal_deadlines(&self) -> bool {
        // G3 also uses the native host-offload adapter and requires it during validation.
        // Frontend stages deliver parked requests to the scheduler inbox between passes.
        self.config.native_host_offload.is_some() || self.config.frontend.is_some()
    }

    /// Build one scheduler/KV/timing rank with an explicit identity.
    pub fn build_rank(&self, identity: RankIdentity) -> Result<SchedulerRank> {
        let seed_offset = engine_seed_offset(identity)?;
        let mut rank = SchedulerRank::new_with_g2(
            identity,
            &self.config,
            Arc::clone(&self.timing),
            seed_offset,
            self.g2.as_ref(),
        )?;
        if let Some(oracle) = &self.belady_oracle {
            rank.set_belady_oracle(oracle.clone());
        }
        Ok(rank)
    }

    /// Build a single-rank or attention-DP logical engine.
    pub fn build(&self, identity: EngineIdentity, dp_size: NonZeroU32) -> Result<Engine> {
        GeneralizedMockerEngine::new_with_rank_factory(identity, dp_size, |rank_identity| {
            self.build_rank(rank_identity)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TimingModelConfig;

    fn afd_declaration(window: Option<usize>) -> EngineConfig {
        serde_json::from_value(serde_json::json!({
            "backend": "sglang", "max_model_len": window,
            "afd": {
                "a_pool": {"system": "h200_sxm", "a_tp_size": 1, "n_a_nodes": 1, "batch_size_per_a_replica": 18},
                "f_pool": {"system": "h200_sxm", "f_moe_ep_size": 8, "n_f_nodes": 1},
                "gpus_per_node": 8, "num_microbatches": 3,
                "comm_overhead_factor": 1.0
            },
            "timing_model": {"type": "external", "provider": "afd", "config": {"isl": 16, "osl": 128}}
        })).unwrap()
    }

    #[test]
    fn afd_runtime_limit_does_not_replace_the_serialized_declaration() {
        let timing = EngineConfig::default().built_in_timing_model().unwrap();
        // Some(144) and None resolve to the same scheduler cap but retain
        // different memory reservations, even after saving and reloading.
        for window in [None, Some(144), Some(2048)] {
            let declaration = afd_declaration(window);
            let saved = serde_json::to_value(&declaration).unwrap();
            let factory =
                EngineFactory::with_timing_model(declaration.clone(), Arc::clone(&timing)).unwrap();
            assert_eq!(factory.config.max_model_len, Some(window.unwrap_or(144)));
            assert_eq!(serde_json::to_value(&declaration).unwrap(), saved);
            let restored: EngineConfig = serde_json::from_value(saved.clone()).unwrap();
            assert_eq!(restored.max_model_len, window);
            let factory = EngineFactory::with_timing_model(restored, Arc::clone(&timing)).unwrap();
            assert_eq!(factory.config.max_model_len, Some(window.unwrap_or(144)));
            assert_eq!(serde_json::to_value(&declaration).unwrap(), saved);
        }
    }

    #[test]
    fn legacy_afd_provider_preserves_the_runtime_window_and_explicit_limits() {
        let timing = EngineConfig::default().built_in_timing_model().unwrap();
        for window in [None, Some(144), Some(2048)] {
            let mut saved = serde_json::to_value(afd_declaration(window)).unwrap();
            saved["timing_model"]["provider"] = serde_json::json!("aic-afd");
            let restored: EngineConfig = serde_json::from_value(saved).unwrap();
            assert_eq!(restored.max_model_len, window);
            assert_eq!(
                serde_json::to_value(&restored).unwrap()["timing_model"]["provider"],
                "afd"
            );
            let factory = EngineFactory::with_timing_model(restored, Arc::clone(&timing)).unwrap();
            assert_eq!(factory.config.max_model_len, Some(window.unwrap_or(144)));
        }
        // Process-local callers may still construct the legacy name directly.
        let mut config = afd_declaration(None);
        if let TimingModelConfig::External { provider, .. } = &mut config.timing_model {
            *provider = "aic-afd".into();
        }
        assert_eq!(
            EngineFactory::with_timing_model(config, timing)
                .unwrap()
                .config
                .max_model_len,
            Some(144)
        );
    }

    #[test]
    fn afd_runtime_limit_rejects_invalid_workload_windows() {
        for workload in [
            serde_json::json!({"isl": 0, "osl": 128}),
            serde_json::json!({"isl": 16, "osl": 0}),
            serde_json::json!({"isl": 16}),
            serde_json::json!({"isl": -1, "osl": 128}),
            serde_json::json!({"isl": u64::from(u32::MAX) + 1, "osl": 128}),
        ] {
            let mut declaration = afd_declaration(None);
            declaration.timing_model = TimingModelConfig::External {
                provider: "afd".into(),
                config: workload,
            };
            let timing = EngineConfig::default().built_in_timing_model().unwrap();
            assert!(EngineFactory::with_timing_model(declaration, timing).is_err());
        }
    }

    #[test]
    fn ordinary_runtime_keeps_an_undeclared_limit_unbounded() {
        let factory = EngineFactory::new(EngineConfig::default()).unwrap();
        assert_eq!(factory.config.max_model_len, None);
    }
}
