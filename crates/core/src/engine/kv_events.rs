// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rank-scoped sequencing and publication for tier-neutral KV events.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::engine::common::protocols::KvEventPublishers;
use crate::engine::{KvEvent, KvEventData, KvEventTier};

/// Cloneable publisher that preserves one event-ID stream across cache tiers.
#[derive(Clone)]
pub(crate) struct KvEventPublisher {
    publishers: KvEventPublishers,
    dp_rank: u32,
    stream: Arc<Mutex<KvEventStream>>,
}

#[derive(Default)]
struct KvEventStream {
    next_event_id: u64,
    publishing: bool,
    pending: VecDeque<PendingKvEvent>,
}

struct PendingKvEvent {
    event: KvEvent,
    token_ids: Option<Vec<Vec<u32>>>,
}

impl KvEventPublisher {
    pub(crate) fn new(publishers: KvEventPublishers, dp_rank: u32) -> Self {
        Self {
            publishers,
            dp_rank,
            stream: Arc::new(Mutex::new(KvEventStream::default())),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        !self.publishers.is_empty()
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
        if !self.is_enabled() {
            return;
        }
        let should_publish = {
            let mut stream = self.stream.lock().expect("KV event stream mutex poisoned");
            let event_id = stream.next_event_id;
            stream.next_event_id = event_id
                .checked_add(1)
                .unwrap_or_else(|| panic!("KV event ID overflow"));
            stream.pending.push_back(PendingKvEvent {
                event: KvEvent {
                    event_id,
                    data,
                    dp_rank: self.dp_rank,
                    tier,
                },
                token_ids,
            });
            if stream.publishing {
                false
            } else {
                stream.publishing = true;
                true
            }
        };
        if !should_publish {
            return;
        }
        loop {
            let pending = {
                let mut stream = self.stream.lock().expect("KV event stream mutex poisoned");
                let Some(pending) = stream.pending.pop_front() else {
                    stream.publishing = false;
                    return;
                };
                pending
            };
            if let Err(error) = self
                .publishers
                .publish(pending.event, pending.token_ids.as_deref())
            {
                tracing::warn!(error = %error, "failed to publish native KV event");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Barrier, Mutex};

    use super::*;
    use crate::engine::common::protocols::KvCacheEventSink;

    struct BlockingFirstSink {
        started: Arc<Barrier>,
        release: Arc<Barrier>,
        event_ids: Mutex<Vec<u64>>,
    }

    impl KvCacheEventSink for BlockingFirstSink {
        fn publish(&self, event: KvEvent) -> anyhow::Result<()> {
            if event.event_id == 0 {
                self.started.wait();
                self.release.wait();
            }
            self.event_ids.lock().unwrap().push(event.event_id);
            Ok(())
        }
    }

    #[test]
    fn cloned_publishers_preserve_rank_stream_order() {
        let started = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let sink = Arc::new(BlockingFirstSink {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
            event_ids: Mutex::new(Vec::new()),
        });
        let publisher = KvEventPublisher::new(KvEventPublishers::new(Some(sink.clone())), 3);

        let first = publisher.clone();
        let first_thread = std::thread::spawn(move || {
            first.publish(
                KvEventData::Removed {
                    block_hashes: vec![10],
                },
                KvEventTier::Device,
                None,
            );
        });
        started.wait();

        let second = publisher.clone();
        let second_thread = std::thread::spawn(move || {
            second.publish(
                KvEventData::Removed {
                    block_hashes: vec![20],
                },
                KvEventTier::HostPinned,
                None,
            );
        });
        second_thread.join().unwrap();
        release.wait();
        first_thread.join().unwrap();

        assert_eq!(*sink.event_ids.lock().unwrap(), vec![0, 1]);
    }
}
