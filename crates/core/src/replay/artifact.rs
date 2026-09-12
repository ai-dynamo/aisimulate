// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-neutral observations captured from a shared Replay execution.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::{
    HostOffloadObservation, HostOffloadObservationData, HostOffloadObserver, KvEvent,
};
use serde::Serialize;
use uuid::Uuid;

use crate::replay::loadgen::ReplayRequestHashes;
use crate::replay::{ReplayError, ReplayResult};

/// Timestamp policy used when rendering native KV observations as replay
/// artifacts.
///
/// `Native` preserves each backend's publication boundary. The two explicit
/// variants exist for parity fixtures that intentionally normalize all events
/// to one side of a pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplayArtifactKvEventVisibility {
    #[default]
    Native,
    PassStart,
    PassEnd,
}

/// One request released by Replay's workload source.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayArtifactRequest {
    pub request_id: Uuid,
    /// Virtual time at which Replay made the request visible to the engine.
    pub observed_at_ms: f64,
    /// Workload-authored ready time, which can precede `observed_at_ms` while
    /// admission or workload scheduling is gated.
    pub scheduled_ready_at_ms: f64,
    pub input_length: usize,
    pub output_length: usize,
    pub replay_hashes: Option<ReplayRequestHashes>,
}

/// One client-visible output released at a pass-completion boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayArtifactOutput {
    pub request_id: Uuid,
    pub token_id: Option<u32>,
    pub completed: bool,
    pub rejected: bool,
    /// Prompt tokens served from KV cache at first admission. Present only on
    /// the request's first output artifact.
    pub cached_tokens: Option<usize>,
    pub observed_at_ms: f64,
}

/// One native G1 observation at the timestamp selected for the artifact run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayArtifactKvEvent {
    pub event: KvEvent,
    pub observed_at_ms: f64,
}

/// Request-local logical position for one framework-prepared G2 store block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayArtifactHostStoreBlockMapping {
    pub block_hash: u64,
    pub logical_block_index: usize,
}

/// Framework-neutral native host-offload transition retained for parity.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReplayArtifactHostOffloadEventData {
    StorePrepared {
        transfer_id: u64,
        block_hashes: Vec<u64>,
    },
    StoreBlockMappings {
        transfer_id: u64,
        mappings: Vec<ReplayArtifactHostStoreBlockMapping>,
    },
    StoreSubmitted {
        transfer_id: u64,
        block_hashes: Vec<u64>,
        completes_at_ms: f64,
    },
    StoreCompleted {
        transfer_id: u64,
        block_hashes: Vec<u64>,
    },
    LoadQueued {
        transfer_id: u64,
        block_hashes: Vec<u64>,
        completes_at_ms: f64,
    },
    LoadCompleted {
        transfer_id: u64,
        block_hashes: Vec<u64>,
    },
    LoadCancelled {
        transfer_id: u64,
        block_hashes: Vec<u64>,
    },
    Evicted {
        block_hash: u64,
    },
    CapacityRetry {
        block_hashes: Vec<u64>,
        structurally_unfittable: bool,
    },
}

/// One host-offload transition at its native scheduler or transfer boundary.
///
/// Detailed artifacts are restricted to one fixed aggregated DP1 worker, so
/// rank-local transfer IDs are unambiguous within this stream.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayArtifactHostOffloadEvent {
    /// Request that initiated this transition. G2 residency itself remains
    /// globally shared by logical block hash.
    pub request_id: Uuid,
    pub observed_at_ms: f64,
    pub event: ReplayArtifactHostOffloadEventData,
}

/// Optional detailed observations produced by the same virtual-clock/pass
/// loop that generates the normal replay report.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ReplayArtifacts {
    pub requests: Vec<ReplayArtifactRequest>,
    pub outputs: Vec<ReplayArtifactOutput>,
    pub kv_events: Vec<ReplayArtifactKvEvent>,
    /// Native host-tier transitions. Populated only by `run_with_artifacts`.
    pub host_offload_events: Vec<ReplayArtifactHostOffloadEvent>,
}

/// Shared sink retained by the caller while [`crate::replay::Replayer`] owns the run.
///
/// This is intentionally a concrete Replay-owned sink instead of a plugin ABI:
/// it carries only neutral request/output/native-KV values and introduces no
/// dependency on an adapter runtime.
#[derive(Debug, Clone)]
pub(crate) struct ReplayArtifactSink {
    visibility: ReplayArtifactKvEventVisibility,
    shared: Arc<ReplayArtifactShared>,
}

#[derive(Debug, Default)]
struct ReplayArtifactShared {
    state: Mutex<ReplayArtifactState>,
    /// Observer callbacks cannot return errors. Any poisoned-state encounter is
    /// retained here and converted into a Replay invariant at finalization.
    failed: AtomicBool,
}

#[derive(Debug, Default)]
struct ReplayArtifactState {
    artifacts: ReplayArtifacts,
    deferred_pass_start_kv_events: Vec<KvEvent>,
}

impl ReplayArtifactSink {
    pub(crate) fn new(visibility: ReplayArtifactKvEventVisibility) -> Self {
        Self {
            visibility,
            shared: Arc::new(ReplayArtifactShared::default()),
        }
    }

    pub(crate) fn host_offload_observer(&self) -> Arc<dyn HostOffloadObserver> {
        Arc::clone(&self.shared) as Arc<dyn HostOffloadObserver>
    }

    pub(crate) fn record_request(&self, request: ReplayArtifactRequest) -> ReplayResult<()> {
        self.lock()?.artifacts.requests.push(request);
        Ok(())
    }

    pub(crate) fn record_outputs(
        &self,
        observed_at_ms: f64,
        outputs: &[crate::replay::protocol::OutputSignal],
    ) -> ReplayResult<()> {
        self.lock()?
            .artifacts
            .outputs
            .extend(outputs.iter().map(|output| ReplayArtifactOutput {
                request_id: output.uuid,
                token_id: output.token_id,
                completed: output.completed,
                rejected: output.rejected,
                cached_tokens: output.cached_tokens,
                observed_at_ms,
            }));
        Ok(())
    }

    pub(crate) fn record_pass_start_kv_events(
        &self,
        pass_start_ms: f64,
        pass_start_events: &[KvEvent],
    ) -> ReplayResult<()> {
        let mut state = self.lock()?;
        match self.visibility {
            ReplayArtifactKvEventVisibility::Native
            | ReplayArtifactKvEventVisibility::PassStart => {
                state
                    .artifacts
                    .kv_events
                    .extend(
                        pass_start_events
                            .iter()
                            .cloned()
                            .map(|event| ReplayArtifactKvEvent {
                                event,
                                observed_at_ms: pass_start_ms,
                            }),
                    )
            }
            ReplayArtifactKvEventVisibility::PassEnd => {
                if !state.deferred_pass_start_kv_events.is_empty() {
                    return Err(ReplayError::Invariant(
                        "artifact sink observed overlapping passes".to_string(),
                    ));
                }
                state
                    .deferred_pass_start_kv_events
                    .extend_from_slice(pass_start_events);
            }
        }
        Ok(())
    }

    pub(crate) fn record_pass_completion_kv_events(
        &self,
        pass_start_ms: f64,
        pass_end_ms: f64,
        pass_end_events: &[KvEvent],
    ) -> ReplayResult<()> {
        let mut state = self.lock()?;
        let timestamp_ms = match self.visibility {
            ReplayArtifactKvEventVisibility::Native | ReplayArtifactKvEventVisibility::PassEnd => {
                pass_end_ms
            }
            ReplayArtifactKvEventVisibility::PassStart => pass_start_ms,
        };
        if self.visibility == ReplayArtifactKvEventVisibility::PassEnd {
            let pass_start_events = std::mem::take(&mut state.deferred_pass_start_kv_events);
            state
                .artifacts
                .kv_events
                .extend(
                    pass_start_events
                        .into_iter()
                        .map(|event| ReplayArtifactKvEvent {
                            event,
                            observed_at_ms: timestamp_ms,
                        }),
                );
        }
        state
            .artifacts
            .kv_events
            .extend(
                pass_end_events
                    .iter()
                    .cloned()
                    .map(|event| ReplayArtifactKvEvent {
                        event,
                        observed_at_ms: timestamp_ms,
                    }),
            );
        Ok(())
    }

    /// Record KV observations produced by deadline-driven engine work outside
    /// an ordinary model pass, such as native H2D activation.
    pub(crate) fn record_internal_kv_events(
        &self,
        observed_at_ms: f64,
        events: &[KvEvent],
    ) -> ReplayResult<()> {
        self.lock()?
            .artifacts
            .kv_events
            .extend(events.iter().cloned().map(|event| ReplayArtifactKvEvent {
                event,
                observed_at_ms,
            }));
        Ok(())
    }

    /// Drain the accumulated artifacts.
    ///
    /// Fails when a `PassEnd` pass start is still parked in
    /// `deferred_pass_start_kv_events`: those events are flushed only when
    /// that pass's completion is processed, and a run that ends in between --
    /// the `max_sim_time_ms` soft cap leaves exactly this state, with the
    /// start recorded and the completion still on the event heap -- would
    /// otherwise drop that pass's entire G1 batch from an artifact advertised
    /// as a parity fixture, with no error and a complete-looking result.
    pub(crate) fn take(&self) -> ReplayResult<ReplayArtifacts> {
        let mut state = self.lock()?;
        if !state.deferred_pass_start_kv_events.is_empty() {
            return Err(ReplayError::Invariant(format!(
                "artifact sink has {} deferred pass-start KV events from a pass \
                 whose completion was never processed",
                state.deferred_pass_start_kv_events.len()
            )));
        }
        Ok(std::mem::take(&mut state.artifacts))
    }

    fn lock(&self) -> ReplayResult<std::sync::MutexGuard<'_, ReplayArtifactState>> {
        if self.shared.failed.load(Ordering::Acquire) {
            return Err(artifact_sink_poisoned());
        }
        match self.shared.state.lock() {
            Ok(state) => Ok(state),
            Err(_) => {
                self.shared.failed.store(true, Ordering::Release);
                Err(artifact_sink_poisoned())
            }
        }
    }
}

fn artifact_sink_poisoned() -> ReplayError {
    ReplayError::Invariant("replay artifact sink lock was poisoned".to_string())
}

impl HostOffloadObserver for ReplayArtifactShared {
    fn record(&self, observation: HostOffloadObservation<'_>) {
        if self.failed.load(Ordering::Acquire) {
            return;
        }
        let HostOffloadObservation { request_id, event } = observation;
        let (observed_at_ms, data) = match event {
            HostOffloadObservationData::StorePrepared {
                at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::StorePrepared {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                },
            ),
            HostOffloadObservationData::StoreBlockMappings {
                at_ms,
                transfer_id,
                mappings,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::StoreBlockMappings {
                    transfer_id: transfer_id.get(),
                    mappings: mappings
                        .iter()
                        .map(|mapping| ReplayArtifactHostStoreBlockMapping {
                            block_hash: mapping.block.sequence_hash(),
                            logical_block_index: mapping.logical_block_index,
                        })
                        .collect(),
                },
            ),
            HostOffloadObservationData::StoreSubmitted {
                at_ms,
                completes_at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::StoreSubmitted {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                    completes_at_ms,
                },
            ),
            HostOffloadObservationData::StoreCompleted {
                at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::StoreCompleted {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                },
            ),
            HostOffloadObservationData::LoadQueued {
                at_ms,
                completes_at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::LoadQueued {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                    completes_at_ms,
                },
            ),
            HostOffloadObservationData::LoadCompleted {
                at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::LoadCompleted {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                },
            ),
            HostOffloadObservationData::LoadCancelled {
                at_ms,
                transfer_id,
                blocks,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::LoadCancelled {
                    transfer_id: transfer_id.get(),
                    block_hashes: logical_hashes(blocks),
                },
            ),
            HostOffloadObservationData::Evicted { at_ms, block } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::Evicted {
                    block_hash: block.sequence_hash(),
                },
            ),
            HostOffloadObservationData::CapacityRetry {
                at_ms,
                blocks,
                structurally_unfittable,
            } => (
                at_ms,
                ReplayArtifactHostOffloadEventData::CapacityRetry {
                    block_hashes: logical_hashes(blocks),
                    structurally_unfittable,
                },
            ),
        };
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => {
                // The callback ABI is infallible, so retain the error until
                // Replay finalization instead of silently producing a partial
                // artifact stream.
                self.failed.store(true, Ordering::Release);
                return;
            }
        };
        state
            .artifacts
            .host_offload_events
            .push(ReplayArtifactHostOffloadEvent {
                request_id,
                observed_at_ms,
                event: data,
            });
    }
}

fn logical_hashes(blocks: &[crate::engine::HostBlockKey]) -> Vec<u64> {
    blocks.iter().map(|block| block.sequence_hash()).collect()
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::engine::{HostBlockKey, KvEventData};

    #[test]
    fn observer_mutex_poison_is_a_sticky_finalization_error() {
        let sink = ReplayArtifactSink::new(ReplayArtifactKvEventVisibility::Native);
        let shared = Arc::clone(&sink.shared);
        assert!(
            catch_unwind(AssertUnwindSafe(move || {
                let _state = shared.state.lock().unwrap();
                panic!("poison artifact observer state");
            }))
            .is_err()
        );

        sink.host_offload_observer().record(HostOffloadObservation {
            request_id: Uuid::nil(),
            event: HostOffloadObservationData::Evicted {
                at_ms: 1.0,
                block: HostBlockKey::new(7),
            },
        });
        assert!(sink.shared.failed.load(Ordering::Acquire));

        // Even explicit mutex recovery cannot turn the already-dropped
        // callback into an apparently complete artifact stream.
        sink.shared.state.clear_poison();
        for _ in 0..2 {
            let error = sink.take().unwrap_err().to_string();
            assert!(error.contains("replay artifact sink lock was poisoned"));
        }
    }

    /// Under `PassEnd`, a pass start parks its events until that pass's
    /// completion is processed. A run ending in between -- the state
    /// `max_sim_time_ms`'s soft cap produces -- must not silently drop that
    /// pass's whole G1 batch from an artifact advertised as a parity fixture.
    #[test]
    fn take_refuses_to_drop_deferred_pass_start_kv_events() {
        let event = KvEvent {
            event_id: 1,
            dp_rank: 0,
            data: KvEventData::Removed {
                block_hashes: vec![3],
            },
        };
        let sink = ReplayArtifactSink::new(ReplayArtifactKvEventVisibility::PassEnd);
        sink.record_pass_start_kv_events(1.0, std::slice::from_ref(&event))
            .unwrap();

        let error = sink.take().unwrap_err().to_string();
        assert!(error.contains("deferred pass-start KV events"), "{error}");

        // Once the pass completes, the deferred batch flushes and `take`
        // succeeds with the events present.
        sink.record_pass_completion_kv_events(1.0, 2.0, &[])
            .unwrap();
        let artifacts = sink.take().unwrap();
        assert_eq!(artifacts.kv_events.len(), 1);
        assert_eq!(artifacts.kv_events[0].observed_at_ms, 2.0);
    }

    #[test]
    fn observer_encodes_capacity_retry() {
        let sink = ReplayArtifactSink::new(ReplayArtifactKvEventVisibility::Native);
        let request_id = Uuid::from_u128(9);
        let blocks = [HostBlockKey::new(17), HostBlockKey::new(23)];
        sink.host_offload_observer().record(HostOffloadObservation {
            request_id,
            event: HostOffloadObservationData::CapacityRetry {
                at_ms: 4.5,
                blocks: &blocks,
                structurally_unfittable: true,
            },
        });

        assert_eq!(
            sink.take().unwrap().host_offload_events,
            vec![ReplayArtifactHostOffloadEvent {
                request_id,
                observed_at_ms: 4.5,
                event: ReplayArtifactHostOffloadEventData::CapacityRetry {
                    block_hashes: vec![17, 23],
                    structurally_unfittable: true,
                },
            }]
        );
    }
}
