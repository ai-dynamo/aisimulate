// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified adapter derived from ai-dynamo/dynamo, commit
// d9eb42db1168131fdae318eef77255637e4d3495, lib/mocker/src/engine_observations.rs.
// Upstream: https://github.com/ai-dynamo/dynamo/blob/d9eb42db1168131fdae318eef77255637e4d3495/lib/mocker/src/engine_observations.rs
// See ../README.md and the repository THIRD_PARTY_NOTICES.md.

use aisimulate_core::engine::{KvEvent, KvEventData};
use aisimulate_core::replay::{EngineEventBatch, ReplayEngineObservation, WorkerStage};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
};

#[derive(Default)]
pub struct Events(pub Vec<(usize, KvEvent)>);
impl EngineEventBatch for Events {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}
pub struct Observation;
impl ReplayEngineObservation for Observation {
    type Batch = Events;
    const CAPTURE_ENGINE_KV_EVENTS: bool = true;
    fn observe_engine_events(
        _: WorkerStage,
        worker: usize,
        _: u32,
        events: Vec<KvEvent>,
    ) -> Events {
        Events(events.into_iter().map(|event| (worker, event)).collect())
    }
    fn stored_hashes(events: &Events) -> Vec<u64> {
        events
            .0
            .iter()
            .flat_map(|(_, event)| match &event.data {
                KvEventData::Stored(stored) => {
                    stored.blocks.iter().map(|b| b.tokens_hash).collect()
                }
                KvEventData::Removed { .. } => Vec::new(),
            })
            .collect()
    }
}
pub fn convert(worker: usize, event: KvEvent) -> anyhow::Result<RouterEvent> {
    let data = match event.data {
        KvEventData::Stored(stored) => KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: stored.parent_hash.map(ExternalSequenceBlockHash),
            start_position: stored.start_position.map(u32::try_from).transpose()?,
            blocks: stored
                .blocks
                .into_iter()
                .map(|block| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(block.block_hash),
                    tokens_hash: LocalBlockHash(block.tokens_hash),
                    mm_extra_info: None,
                })
                .collect(),
        }),
        KvEventData::Removed { block_hashes } => KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect(),
        }),
    };
    Ok(RouterEvent::with_storage_tier(
        worker.try_into()?,
        KvCacheEvent {
            event_id: event.event_id,
            dp_rank: event.dp_rank,
            data,
        },
        StorageTier::Device,
    ))
}
