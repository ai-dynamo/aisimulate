// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified adapter derived from ai-dynamo/dynamo, commit
// d9eb42db1168131fdae318eef77255637e4d3495, lib/mocker/src/engine_observations.rs.
// Upstream: https://github.com/ai-dynamo/dynamo/blob/d9eb42db1168131fdae318eef77255637e4d3495/lib/mocker/src/engine_observations.rs
// See the repository THIRD_PARTY_NOTICES.md. Modified for AISimulate replay.

use crate::engine::{KvEvent, KvEventData, KvEventTier};
use crate::replay::{EngineEventBatch, KvIngestEventEncoder, ReplayEngineObservation, WorkerStage};

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
    fn kv_ingest_event_count(events: &Events) -> Option<usize> {
        Some(events.0.len())
    }
    fn encode_kv_ingest(
        events: &Events,
        encoder: &mut KvIngestEventEncoder<'_>,
    ) -> anyhow::Result<()> {
        for (worker, event) in &events.0 {
            let (tier, name) = match event.tier {
                KvEventTier::Device => (0, "device"),
                KvEventTier::HostPinned => (1, "host_pinned"),
            };
            encoder.begin_event(
                (*worker).try_into()?,
                event.dp_rank,
                tier,
                name,
                event.event_id,
            );
            match &event.data {
                KvEventData::Stored(stored) => {
                    encoder.begin_kind(0, "stored");
                    encoder.put_optional_u64(stored.parent_hash);
                    encoder.put_optional_u32(stored.start_position.map(u32::try_from).transpose()?);
                    encoder.put_len(stored.blocks.len(), "stored KV block count")?;
                    encoder.add_blocks(stored.blocks.len(), "stored KV block count")?;
                    for block in &stored.blocks {
                        encoder.put_u64(block.block_hash);
                        encoder.put_u64(block.tokens_hash);
                        encoder.put_u8(0); // Engine events do not carry multimodal data.
                    }
                }
                KvEventData::Removed { block_hashes } => {
                    encoder.begin_kind(1, "removed");
                    encoder.put_len(block_hashes.len(), "removed KV block count")?;
                    encoder.add_blocks(block_hashes.len(), "removed KV block count")?;
                    for &hash in block_hashes {
                        encoder.put_u64(hash);
                    }
                }
            }
        }
        Ok(())
    }
    fn stored_hashes(events: &Events) -> Vec<u64> {
        events
            .0
            .iter()
            .filter(|(_, event)| event.tier == KvEventTier::Device)
            .flat_map(|(_, event)| match &event.data {
                KvEventData::Stored(stored) => {
                    stored.blocks.iter().map(|b| b.tokens_hash).collect()
                }
                KvEventData::Removed { .. } => Vec::new(),
            })
            .collect()
    }
}
