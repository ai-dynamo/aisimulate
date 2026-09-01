// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rank-scoped sequencing and publication for tier-neutral KV events.

use std::cell::Cell;
use std::rc::Rc;

use crate::engine::common::protocols::KvEventPublishers;
use crate::engine::{KvEvent, KvEventData, KvEventTier};

/// Cloneable publisher that preserves one event-ID stream across cache tiers.
#[derive(Clone)]
pub(crate) struct KvEventPublisher {
    publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: Option<Rc<Cell<u64>>>,
}

impl KvEventPublisher {
    pub(crate) fn new(publishers: KvEventPublishers, dp_rank: u32) -> Self {
        let next_event_id = (!publishers.is_empty()).then(|| Rc::new(Cell::new(0)));
        Self {
            publishers,
            dp_rank,
            next_event_id,
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.next_event_id.is_some()
    }

    pub(crate) fn dp_rank(&self) -> u32 {
        self.dp_rank
    }

    pub(crate) fn publish(
        &self,
        data: KvEventData,
        tier: KvEventTier,
        token_ids: Option<Vec<Vec<u32>>>,
    ) {
        let Some(next_event_id) = self.next_event_id.as_ref() else {
            return;
        };
        let event_id = next_event_id.get();
        assert_ne!(event_id, u64::MAX, "KV event ID overflow");
        next_event_id.set(event_id + 1);
        let event = KvEvent {
            event_id,
            data,
            dp_rank: self.dp_rank,
            tier,
        };
        if let Err(error) = self.publishers.publish(event, token_ids.as_deref()) {
            tracing::warn!(error = %error, "failed to publish native KV event");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::common::protocols::KvCacheEventSink;

    #[derive(Default)]
    struct CaptureSink(Mutex<Vec<KvEvent>>);

    impl KvCacheEventSink for CaptureSink {
        fn publish(&self, event: KvEvent) -> anyhow::Result<()> {
            self.0.lock().unwrap().push(event);
            Ok(())
        }
    }

    #[test]
    fn cloned_publishers_preserve_rank_stream_order() {
        let sink = Arc::new(CaptureSink::default());
        let publisher = KvEventPublisher::new(KvEventPublishers::new(Some(sink.clone())), 3);

        publisher.publish(
            KvEventData::Removed {
                block_hashes: vec![10],
            },
            KvEventTier::Device,
            None,
        );
        publisher.clone().publish(
            KvEventData::Removed {
                block_hashes: vec![20],
            },
            KvEventTier::HostPinned,
            None,
        );

        let events = sink.0.lock().unwrap();
        assert_eq!(events[0].event_id, 0);
        assert_eq!(events[0].tier, KvEventTier::Device);
        assert_eq!(events[1].event_id, 1);
        assert_eq!(events[1].tier, KvEventTier::HostPinned);
        assert!(events.iter().all(|event| event.dp_rank == 3));
    }
}
