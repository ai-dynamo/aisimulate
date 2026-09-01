// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-tier KV-event transactions over the rank-scoped event stream.

use rustc_hash::FxHashMap;

use crate::engine::common::hashing::SequenceHash;
use crate::engine::kv_manager::{BlockRequestLease, SourceReuseDependency};
use crate::engine::{KvEventData, KvEventPublisher, KvEventTier, StoredBlocks};

pub(super) struct HostKvEventTransactions {
    events: KvEventPublisher,
    pending_stores: FxHashMap<SourceReuseDependency, Vec<StoredBlocks>>,
}

impl HostKvEventTransactions {
    pub(super) fn new(events: KvEventPublisher) -> Self {
        Self {
            events,
            pending_stores: FxHashMap::default(),
        }
    }

    pub(super) fn stage_store(
        &mut self,
        dependency: SourceReuseDependency,
        lease: &BlockRequestLease,
        block_indices: &[usize],
        evicted: Vec<SequenceHash>,
    ) {
        if !self.events.is_enabled() {
            return;
        }
        let stores = block_indices
            .iter()
            .map(|&index| lease.stored_block_event(index))
            .collect();
        assert!(
            self.pending_stores.insert(dependency, stores).is_none(),
            "native host store dependency was reused before completion"
        );
        if !evicted.is_empty() {
            self.events.publish(
                KvEventData::Removed {
                    block_hashes: evicted,
                },
                KvEventTier::HostPinned,
                None,
            );
        }
    }

    pub(super) fn complete_store(&mut self, dependency: SourceReuseDependency) {
        let Some(stores) = self.pending_stores.remove(&dependency) else {
            debug_assert!(!self.events.is_enabled());
            return;
        };
        for store in stores {
            self.events
                .publish(KvEventData::Stored(store), KvEventTier::HostPinned, None);
        }
    }
}
