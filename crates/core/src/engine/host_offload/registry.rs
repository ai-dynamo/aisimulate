// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2 pool ownership: private per DP rank, or one cluster-shared pool per
//! Replay deployment or per [`SharedG2Pool`] handle.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Result, ensure};

use super::{HostLink, HostTier, HostTierConfig};
use crate::engine::{G2Scope, NativeHostOffloadConfig};

/// Everything that must agree before ranks may share stored KV blocks.
#[derive(Clone, Debug, PartialEq)]
struct Contract {
    kv_layout_id: String,
    tensor_parallel_size: u32,
    block_size: usize,
    block_bytes: usize,
    capacity_blocks: usize,
    shared_d2h_bandwidth_gbps: f64,
    shared_h2d_bandwidth_gbps: f64,
}

#[derive(Default)]
struct RegistryState {
    pool: Option<(Contract, Arc<Mutex<HostTier>>)>,
    next_client: u64,
}

/// Deployment-owned cluster-shared G2. A fresh Replay starts with a fresh pool.
#[derive(Default)]
pub(crate) struct G2Registry(Mutex<RegistryState>);

impl G2Registry {
    /// Join the deployment pool, creating it from the first participant.
    fn join(&self, contract: Contract, link: HostLink) -> Result<HostClient> {
        let mut state = self.0.lock().unwrap();
        let tier = match &state.pool {
            Some((existing, tier)) => {
                ensure!(
                    *existing == contract,
                    "cluster_shared host_offload participants are incompatible: {existing:?} vs {contract:?}"
                );
                Arc::clone(tier)
            }
            None => {
                let tier = Arc::new(Mutex::new(HostTier::new(HostTierConfig {
                    capacity_blocks: contract.capacity_blocks,
                    block_bytes: contract.block_bytes,
                    shared_gbps: Some((
                        contract.shared_d2h_bandwidth_gbps,
                        contract.shared_h2d_bandwidth_gbps,
                    )),
                })?));
                state.pool = Some((contract, Arc::clone(&tier)));
                tier
            }
        };
        let id = state.next_client;
        state.next_client += 1;
        tier.lock().unwrap().register(id, link)?;
        Ok(HostClient { tier, id })
    }

    /// `(capacity, resident, used)` blocks of the shared pool, if one exists.
    pub(crate) fn occupancy(&self) -> Option<(usize, usize, usize)> {
        let state = self.0.lock().unwrap();
        let tier = state.pool.as_ref()?.1.lock().unwrap();
        let (resident, used) = tier.occupancy();
        Some((tier.capacity_blocks(), resident, used))
    }
}

/// One cluster-shared G2 pool that engines built by separate
/// [`EngineFactory`](crate::engine::EngineFactory) instances can join.
///
/// Replay creates and binds its deployment pool itself. Drivers that run
/// several engines outside Replay, such as live serving with multiple workers
/// in one process, share one handle with
/// [`EngineFactory::with_shared_g2_pool`](crate::engine::EngineFactory::with_shared_g2_pool).
/// Participants must agree on the pool contract; the first engine to join
/// fixes it. Transfer and residency completions are applied when any
/// participant advances the pool, so a driver must re-query each engine's
/// internal deadline after a peer advanced it.
#[derive(Clone, Default)]
pub struct SharedG2Pool(pub(crate) Arc<G2Registry>);

impl SharedG2Pool {
    /// A pool with no participants. Capacity is fixed by the first joiner.
    pub fn new() -> Self {
        Self::default()
    }

    /// `(capacity, resident, used)` blocks, once an engine has joined.
    pub fn occupancy(&self) -> Option<(usize, usize, usize)> {
        self.0.occupancy()
    }
}

/// Deployment binding for ranks whose role selects cluster-shared G2.
#[derive(Clone)]
pub(crate) struct G2Binding {
    pub(crate) registry: Arc<G2Registry>,
    pub(crate) tensor_parallel_size: u32,
}

/// One DP rank's handle on its G2 pool. Dropping it retires the rank.
pub(crate) struct HostClient {
    tier: Arc<Mutex<HostTier>>,
    id: u64,
}

impl HostClient {
    pub(crate) fn new(
        config: &NativeHostOffloadConfig,
        block_size: usize,
        kv_bytes_per_token: usize,
        binding: Option<&G2Binding>,
    ) -> Result<Self> {
        let block_bytes = block_size
            .checked_mul(kv_bytes_per_token)
            .ok_or_else(|| anyhow::anyhow!("native host block byte size overflow"))?;
        let link = HostLink {
            d2h_bandwidth_gbps: config.d2h_bandwidth_gbps,
            h2d_bandwidth_gbps: config.h2d_bandwidth_gbps,
            latency_to_first_byte_ms: config.latency_to_first_byte_ms,
        };
        match config.scope {
            G2Scope::DpRankLocal => {
                let mut tier = HostTier::new(HostTierConfig {
                    capacity_blocks: config.num_host_blocks,
                    block_bytes,
                    shared_gbps: None,
                })?;
                tier.register(0, link)?;
                Ok(Self {
                    tier: Arc::new(Mutex::new(tier)),
                    id: 0,
                })
            }
            G2Scope::ClusterShared => {
                let binding = binding.ok_or_else(|| {
                    anyhow::anyhow!(
                        "cluster_shared host_offload requires a shared G2 pool; construct it through ReplaySpec or EngineFactory::with_shared_g2_pool"
                    )
                })?;
                // Replay validates this; a pool bound directly must not let an
                // invalid participant fix the contract for later joiners.
                ensure!(
                    binding.tensor_parallel_size > 0,
                    "native tensor_parallel_size must be positive"
                );
                binding.registry.join(
                    Contract {
                        kv_layout_id: config.kv_layout_id.clone().unwrap_or_default(),
                        tensor_parallel_size: binding.tensor_parallel_size,
                        block_size,
                        block_bytes,
                        capacity_blocks: config.num_host_blocks,
                        shared_d2h_bandwidth_gbps: config.shared_d2h_bandwidth_gbps,
                        shared_h2d_bandwidth_gbps: config.shared_h2d_bandwidth_gbps,
                    },
                    link,
                )
            }
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, HostTier> {
        self.tier.lock().unwrap()
    }
}

impl Drop for HostClient {
    fn drop(&mut self) {
        if let Ok(mut tier) = self.tier.lock() {
            tier.retire(self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Participant TP size, `(block_size, bytes_per_token)` and host config
    /// edit, then the one pool-contract field that edit changes.
    type ContractCase = (
        u32,
        (usize, usize),
        fn(&mut NativeHostOffloadConfig),
        fn(&mut Contract),
    );

    /// 4-token blocks of 1,000 bytes per token.
    const GEOMETRY: (usize, usize) = (4, 1_000);

    fn join(
        binding: Option<&G2Binding>,
        (block_size, bytes_per_token): (usize, usize),
        edit: impl FnOnce(&mut NativeHostOffloadConfig),
    ) -> Result<HostClient> {
        let mut config = NativeHostOffloadConfig::new(8).cluster_shared("tp1");
        edit(&mut config);
        HostClient::new(&config, block_size, bytes_per_token, binding)
    }

    #[test]
    fn shared_clients_join_one_pool_only_under_an_equal_contract() {
        let binding = G2Binding {
            registry: Arc::default(),
            tensor_parallel_size: 1,
        };
        // Access links and latency are per client and may differ.
        let first = join(Some(&binding), GEOMETRY, |_| {}).unwrap();
        let second = join(Some(&binding), GEOMETRY, |config| {
            config.d2h_bandwidth_gbps = 4.0;
            config.h2d_bandwidth_gbps = 2.0;
            config.latency_to_first_byte_ms = 1.0;
        })
        .unwrap();
        assert!(Arc::ptr_eq(&first.tier, &second.tier));
        assert_eq!((first.id(), second.id()), (0, 1));
        assert_eq!(binding.registry.occupancy(), Some((8, 0, 0)));

        let pool = Contract {
            kv_layout_id: "tp1".into(),
            tensor_parallel_size: 1,
            block_size: 4,
            block_bytes: 4_000,
            capacity_blocks: 8,
            shared_d2h_bandwidth_gbps: 80.0,
            shared_h2d_bandwidth_gbps: 80.0,
        };
        // Each participant differs from the pool contract in exactly one field.
        let cases: &[ContractCase] = &[
            (
                1,
                GEOMETRY,
                |c| c.kv_layout_id = Some("tp2".into()),
                |c| c.kv_layout_id = "tp2".into(),
            ),
            (2, GEOMETRY, |_| {}, |c| c.tensor_parallel_size = 2),
            (1, (8, 500), |_| {}, |c| c.block_size = 8),
            (1, (4, 2_000), |_| {}, |c| c.block_bytes = 8_000),
            (
                1,
                GEOMETRY,
                |c| c.num_host_blocks = 9,
                |c| c.capacity_blocks = 9,
            ),
            (
                1,
                GEOMETRY,
                |c| c.shared_d2h_bandwidth_gbps = 1.0,
                |c| c.shared_d2h_bandwidth_gbps = 1.0,
            ),
            (
                1,
                GEOMETRY,
                |c| c.shared_h2d_bandwidth_gbps = 1.0,
                |c| c.shared_h2d_bandwidth_gbps = 1.0,
            ),
        ];
        for &(tensor_parallel_size, geometry, edit, differ) in cases {
            let binding = G2Binding {
                tensor_parallel_size,
                ..binding.clone()
            };
            let mut other = pool.clone();
            differ(&mut other);
            assert_eq!(
                join(Some(&binding), geometry, edit)
                    .err()
                    .unwrap()
                    .to_string(),
                format!(
                    "cluster_shared host_offload participants are incompatible: {pool:?} vs {other:?}"
                )
            );
        }
        // Rejected participants never registered with the pool.
        assert_eq!(join(Some(&binding), GEOMETRY, |_| {}).unwrap().id(), 2);
        assert_eq!(
            join(None, GEOMETRY, |_| {}).err().unwrap().to_string(),
            "cluster_shared host_offload requires a shared G2 pool; construct it through ReplaySpec or EngineFactory::with_shared_g2_pool"
        );
        // Private caches never join the deployment pool.
        let private = join(Some(&binding), GEOMETRY, |c| c.scope = G2Scope::DpRankLocal).unwrap();
        assert!(!Arc::ptr_eq(&private.tier, &first.tier));
    }
}
