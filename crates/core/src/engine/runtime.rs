// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory for one logical mock engine.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::engine::generalized::{EngineIdentity, GeneralizedMockerEngine, RankIdentity};
use crate::engine::host_offload::HostCacheDomain;
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
    config: EngineConfig,
    timing: Arc<dyn TimingModel>,
}

impl EngineFactory {
    /// Construct a factory for a built-in timing model.
    pub fn new(config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let timing = config.built_in_timing_model()?;
        Ok(Self { config, timing })
    }

    /// Construct a factory with a process-local timing provider.
    pub fn with_timing_model(config: EngineConfig, timing: Arc<dyn TimingModel>) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, timing })
    }

    /// Build one scheduler/KV/timing rank with an explicit identity.
    pub fn build_rank(&self, identity: RankIdentity) -> Result<SchedulerRank> {
        let seed_offset = engine_seed_offset(identity)?;
        SchedulerRank::new_with_timing_model(
            identity,
            &self.config,
            Arc::clone(&self.timing),
            seed_offset,
        )
    }

    /// Build a single-rank or attention-DP logical engine.
    pub fn build(&self, identity: EngineIdentity, dp_size: NonZeroU32) -> Result<Engine> {
        ensure!(
            self.config.native_host_offload.is_none() || dp_size.get() == 1,
            "attention-DP native_host_offload requires an explicit cache-domain topology"
        );
        GeneralizedMockerEngine::new_with_rank_factory(identity, dp_size, |rank_identity| {
            self.build_rank(rank_identity)
        })
    }

    /// Build a logical engine whose attention-DP ranks map to host-cache domains.
    pub fn build_with_cache_domains(
        &self,
        identity: EngineIdentity,
        dp_size: NonZeroU32,
        cache_domain_ids: &[u32],
    ) -> Result<Engine> {
        ensure!(
            cache_domain_ids.len() == dp_size.get() as usize,
            "cache-domain topology has {} ranks but dp_size is {}",
            cache_domain_ids.len(),
            dp_size
        );
        let mut host_domains = HashMap::new();
        if let Some(config) = &self.config.native_host_offload {
            let kv_bytes_per_token = self
                .config
                .kv_cache_bytes_per_token
                .expect("validated native host offload requires KV byte geometry");
            for cache_domain_id in cache_domain_ids.iter().copied() {
                if host_domains.contains_key(&cache_domain_id) {
                    continue;
                }
                host_domains.insert(
                    cache_domain_id,
                    HostCacheDomain::new(config, self.config.block_size, kv_bytes_per_token)?,
                );
            }
        }
        GeneralizedMockerEngine::new_with_rank_factory(identity, dp_size, |rank_identity| {
            let host_handle = host_domains
                .get(&cache_domain_ids[rank_identity.dp_rank as usize])
                .map(|domain| domain.bind_rank(rank_identity.dp_rank));
            let seed_offset = engine_seed_offset(rank_identity)?;
            SchedulerRank::new_with_timing_model_and_host_handle(
                rank_identity,
                &self.config,
                Arc::clone(&self.timing),
                seed_offset,
                host_handle,
            )
        })
    }
}
