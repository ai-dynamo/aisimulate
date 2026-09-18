// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline input-demand forecasting, separate from causal cache ownership.
//!
//! This deliberately ranks global trace demand, not actual future accesses on a
//! particular worker. A request retires once at its first committed prefill (or
//! terminal removal). Later chunks, preemption retries, and generated outputs do
//! not create forecast occurrences. These are modeling assumptions, not missing
//! refinement steps: the oracle must never predict routing, populate KV, advance
//! arrivals, or change native eligibility and execution rules.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub(crate) use super::common::hashing::SequenceHash;
use super::common::hashing::{
    XXH3_SEED, compute_block_hash_for_tokens, compute_next_sequence_hash,
};

/// Native eviction ranking for offline replay. LRU requires no forecast.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvEvictionPolicy {
    #[default]
    Lru,
    /// Best-effort global future input demand; not worker-local optimal reuse.
    Belady,
}

/// Hash exactly the complete input pages the native cache can identify.
pub(crate) fn input_sequence_hashes(tokens: &[u32], block_size: usize) -> Vec<SequenceHash> {
    assert!(block_size > 0, "Belady block size must be positive");
    let mut parent = None;
    tokens
        .chunks_exact(block_size)
        .map(|block| {
            let local = compute_block_hash_for_tokens(block, XXH3_SEED);
            let hash = parent.map_or(local, |parent| compute_next_sequence_hash(parent, local));
            parent = Some(hash);
            hash
        })
        .collect()
}

#[derive(Debug)]
struct Occurrences {
    requests: Vec<usize>,
    head: usize,
}

impl Occurrences {
    fn next_use(&self) -> usize {
        self.requests.get(self.head).copied().unwrap_or(usize::MAX)
    }
}

#[derive(Debug)]
struct RequestDemand {
    hashes: Vec<SequenceHash>,
    retired: bool,
}

#[derive(Debug)]
struct ForecastState {
    request_indices: FxHashMap<Uuid, usize>,
    demands: Vec<RequestDemand>,
    occurrences: FxHashMap<SequenceHash, Occurrences>,
    // Each append advances at least one input occurrence. Keeping this journal
    // for the run costs O(input occurrences), even when a worker stays idle, and
    // avoids per-worker copies of the forecast or subscriber lifecycle machinery.
    changed_hashes: Vec<SequenceHash>,
}

/// One run-scoped forecast shared by fixed workers, never between replay runs.
/// Native caches own their candidate indexes and journal cursors independently.
#[derive(Clone, Debug)]
pub(crate) struct BeladyOracle {
    state: Arc<Mutex<ForecastState>>,
}

impl BeladyOracle {
    /// Requests must already be in the actual fixed-arrival/source order.
    pub(crate) fn new(requests: Vec<(Uuid, Vec<SequenceHash>)>) -> Result<Self> {
        let mut request_indices = FxHashMap::default();
        let mut demands = Vec::with_capacity(requests.len());
        let mut occurrences: FxHashMap<SequenceHash, Occurrences> = FxHashMap::default();
        for (rank, (uuid, mut hashes)) in requests.into_iter().enumerate() {
            ensure!(
                request_indices.insert(uuid, rank).is_none(),
                "Belady requires unique request identities: {uuid}"
            );
            hashes.sort_unstable();
            hashes.dedup();
            for &hash in &hashes {
                occurrences
                    .entry(hash)
                    .or_insert_with(|| Occurrences {
                        requests: Vec::new(),
                        head: 0,
                    })
                    .requests
                    .push(rank);
            }
            demands.push(RequestDemand {
                hashes,
                retired: false,
            });
        }
        Ok(Self {
            state: Arc::new(Mutex::new(ForecastState {
                request_indices,
                demands,
                occurrences,
                changed_hashes: Vec::new(),
            })),
        })
    }

    pub(crate) fn next_use(&self, hash: SequenceHash) -> usize {
        let state = self.state.lock().expect("Belady forecast lock poisoned");
        state
            .occurrences
            .get(&hash)
            .map_or(usize::MAX, Occurrences::next_use)
    }

    /// Called only after final batch commitment or terminal request removal.
    /// Retire identities, not a global time/rank cursor: an earlier request can
    /// still be queued while a later request commits on another worker.
    pub(crate) fn retire_requests(&self, requests: impl IntoIterator<Item = Uuid>) {
        let mut state = self.state.lock().expect("Belady forecast lock poisoned");
        let mut affected = FxHashSet::default();
        for uuid in requests {
            let Some(&rank) = state.request_indices.get(&uuid) else {
                continue;
            };
            let demand = &mut state.demands[rank];
            if demand.retired {
                continue;
            }
            demand.retired = true;
            affected.extend(std::mem::take(&mut demand.hashes));
        }
        let ForecastState {
            demands,
            occurrences,
            changed_hashes,
            ..
        } = &mut *state;
        for hash in affected {
            let uses = occurrences
                .get_mut(&hash)
                .expect("input demand missing from Belady occurrences");
            let previous = uses.head;
            while uses
                .requests
                .get(uses.head)
                .is_some_and(|&rank| demands[rank].retired)
            {
                uses.head += 1;
            }
            if uses.head != previous {
                changed_hashes.push(hash);
            }
        }
    }

    /// Refresh all affected local candidates before selecting the next victim.
    /// Priorities can increase while buried in an index, so checking only a
    /// lazy heap's current top would select the wrong farthest-use candidate.
    pub(crate) fn changes_since(&self, cursor: &mut usize) -> Vec<(SequenceHash, usize)> {
        let state = self.state.lock().expect("Belady forecast lock poisoned");
        let pending = &state.changed_hashes[*cursor..];
        *cursor = state.changed_hashes.len();
        if pending.is_empty() {
            return Vec::new();
        }
        let changed: FxHashSet<_> = pending.iter().copied().collect();
        changed
            .into_iter()
            .map(|hash| (hash, state.occurrences[&hash].next_use()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use super::*;

    fn id(rank: usize) -> Uuid {
        Uuid::from_u128(rank as u128 + 1)
    }

    #[test]
    fn outstanding_queued_demand_survives_out_of_order_retirement() {
        let oracle = BeladyOracle::new(vec![
            (id(0), vec![10, 20]),
            (id(1), vec![10]),
            (id(2), vec![10, 20]),
        ])
        .unwrap();
        oracle.retire_requests([id(1)]);
        assert_eq!(oracle.next_use(10), 0);
        let mut cursor = 0;
        assert!(oracle.changes_since(&mut cursor).is_empty());
        oracle.retire_requests([id(0), id(0)]);
        assert_eq!(oracle.next_use(10), 2);
        assert_eq!(oracle.next_use(20), 2);
        assert_eq!(oracle.changes_since(&mut cursor).len(), 2);
        oracle.retire_requests([id(2)]);
        assert_eq!(oracle.next_use(10), usize::MAX);
        assert_eq!(oracle.next_use(99), usize::MAX);
    }

    #[test]
    fn worker_cursors_observe_global_demand_independently() {
        let oracle = BeladyOracle::new(vec![(id(0), vec![7]), (id(1), vec![7])]).unwrap();
        let other_worker = oracle.clone();
        let (mut first, mut second) = (0, 0);
        oracle.retire_requests([id(0)]);
        assert_eq!(oracle.changes_since(&mut first), vec![(7, 1)]);
        other_worker.retire_requests([id(1)]);
        assert_eq!(oracle.changes_since(&mut first), vec![(7, usize::MAX)]);
        assert_eq!(
            other_worker.changes_since(&mut second),
            vec![(7, usize::MAX)]
        );
        assert!(oracle.changes_since(&mut first).is_empty());
    }

    #[test]
    fn input_identity_includes_ancestors_and_ignores_incomplete_pages() {
        let first = input_sequence_hashes(&[1, 2, 3, 4, 9], 2);
        let second = input_sequence_hashes(&[5, 6, 3, 4], 2);
        assert_eq!(first.len(), 2);
        assert_ne!(first[1], second[1]);
        assert_eq!(first, input_sequence_hashes(&[1, 2, 3, 4], 2));
    }

    fn optimal_misses(
        trace: &[u64],
        index: usize,
        cached: BTreeSet<u64>,
        memo: &mut HashMap<(usize, Vec<u64>), usize>,
    ) -> usize {
        if index == trace.len() {
            return 0;
        }
        let key = (index, cached.iter().copied().collect());
        if let Some(&value) = memo.get(&key) {
            return value;
        }
        let page = trace[index];
        let result = if cached.contains(&page) {
            optimal_misses(trace, index + 1, cached, memo)
        } else if cached.len() < 2 {
            let mut next = cached;
            next.insert(page);
            1 + optimal_misses(trace, index + 1, next, memo)
        } else {
            1 + cached
                .iter()
                .map(|victim| {
                    let mut next = cached.clone();
                    next.remove(victim);
                    next.insert(page);
                    optimal_misses(trace, index + 1, next, memo)
                })
                .min()
                .unwrap()
        };
        memo.insert(key, result);
        result
    }

    #[test]
    fn sequential_equal_pages_match_exhaustive_optimum() {
        // This optimality claim is deliberately limited to a serial flat page
        // cache. Native batched/prefix/placement behavior has no such guarantee.
        for encoded in 0..3_usize.pow(6) {
            let mut remainder = encoded;
            let trace: Vec<_> = (0..6)
                .map(|_| {
                    let page = (remainder % 3) as u64;
                    remainder /= 3;
                    page
                })
                .collect();
            let oracle = BeladyOracle::new(
                trace
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (id(i), vec![p]))
                    .collect(),
            )
            .unwrap();
            let mut cached = BTreeSet::new();
            let mut misses = 0;
            for (index, &page) in trace.iter().enumerate() {
                if !cached.contains(&page) {
                    misses += 1;
                    if cached.len() == 2 {
                        let victim = *cached.iter().max_by_key(|&&p| oracle.next_use(p)).unwrap();
                        cached.remove(&victim);
                    }
                    cached.insert(page);
                }
                oracle.retire_requests([id(index)]);
            }
            assert_eq!(
                misses,
                optimal_misses(&trace, 0, BTreeSet::new(), &mut HashMap::new()),
                "trace {trace:?}"
            );
        }
    }
}
