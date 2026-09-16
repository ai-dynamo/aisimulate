// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G3 writes own G2 source pins; promotions own pending G2 destinations.
//! Request termination detaches from admitted promotions rather than aborting I/O.
use crate::engine::g3_offload::{Direction, Probe, SharedG3Tier};
use crate::engine::host_offload::{HostBlockKey, HostTier, Lookup, StoreOutcome, TransferId};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

struct G3StagingState {
    owner: Option<Uuid>,
    reservations: Vec<TransferId>,
}

pub(super) struct VllmG3OffloadAdapter {
    registry: SharedG3Tier,
    worker: usize,
    stages: BTreeMap<u64, G3StagingState>,
    attempted_at: f64,
    attempted: BTreeSet<(Uuid, HostBlockKey)>,
    pub(super) epoch: u64,
    observed_completion_epoch: u64,
}

impl VllmG3OffloadAdapter {
    pub(super) fn new(registry: SharedG3Tier, worker: usize) -> Self {
        Self {
            registry,
            worker,
            stages: BTreeMap::new(),
            attempted_at: 0.0,
            attempted: BTreeSet::new(),
            epoch: 0,
            observed_completion_epoch: 0,
        }
    }

    pub(super) fn store(&mut self, tier: &mut HostTier, keys: &[HostBlockKey], now: f64) {
        let mut registry = self.registry.lock().unwrap();
        if let Some(job) = registry.submit(self.worker, Direction::Write, keys, now) {
            tier.pin_external(&registry.job_keys(job));
            self.epoch += 1;
        }
    }

    /// Return true only when a concrete pending transfer can wake the lookup.
    pub(super) fn stage(
        &mut self,
        tier: &mut HostTier,
        owner: Uuid,
        keys: &[HostBlockKey],
        now: f64,
    ) -> bool {
        if now > self.attempted_at {
            self.attempted_at = now;
            self.attempted.clear();
        }
        let mut missing = Vec::new();
        let mut reservations = Vec::new();
        let mut deferred = false;
        let mut registry = self.registry.lock().unwrap();
        for key in keys.iter().copied() {
            match tier.lookup(key) {
                Lookup::Hit => {}
                Lookup::Pending { .. } => deferred = true,
                Lookup::Miss => match registry.probe(self.worker, key) {
                    Probe::Resident => {
                        // A zero-duration promotion can be evicted by a later
                        // lookup at this same timestamp. Do not re-promote the
                        // same request/key indefinitely while its recoverable
                        // prefix cannot fit. A read-only feasibility check lets
                        // real capacity relief resume promotion immediately.
                        if self.attempted.contains(&(owner, key)) {
                            let prefix = keys
                                .iter()
                                .copied()
                                .take_while(|key| {
                                    !matches!(tier.lookup(*key), Lookup::Miss)
                                        || registry.contains(self.worker, *key)
                                })
                                .collect::<Vec<_>>();
                            if !tier.can_reserve_external_prefix(&prefix) {
                                break;
                            }
                        }
                        let StoreOutcome::Prepared { transfer_id, .. } =
                            tier.reserve_external(owner, &[key], now)
                        else {
                            break;
                        };
                        missing.push(key);
                        reservations.push(transfer_id);
                    }
                    Probe::Pending => deferred = true,
                    Probe::Miss => break,
                },
            }
        }
        if missing.is_empty() {
            return deferred;
        }
        let Some(job) = registry.submit(self.worker, Direction::Read, &missing, now) else {
            for reservation in reservations {
                tier.cancel_external(reservation);
            }
            return deferred;
        };
        self.attempted
            .extend(missing.into_iter().map(|key| (owner, key)));
        self.stages.insert(
            job,
            G3StagingState {
                owner: Some(owner),
                reservations,
            },
        );
        self.epoch += 1;
        true
    }

    pub(super) fn advance(&mut self, tier: &mut HostTier, now: f64) -> bool {
        let (done, completion_epoch) = {
            let mut registry = self.registry.lock().unwrap();
            let done = registry.take_completed(self.worker, now);
            (done, registry.completion_epoch)
        };
        let progressed = !done.is_empty() || completion_epoch != self.observed_completion_epoch;
        self.observed_completion_epoch = completion_epoch;
        for completion in done {
            match completion.direction {
                Direction::Write => tier.unpin_external(&completion.keys),
                Direction::Read => {
                    let stage = self
                        .stages
                        .remove(&completion.id)
                        .expect("promotion owns G2 reservations");
                    for reservation in stage.reservations {
                        tier.complete_external(reservation);
                    }
                }
            }
            self.epoch += 1;
        }
        progressed
    }

    /// Forget a requester without aborting admitted cache promotions. Pending
    /// destinations are job-owned and still become resident on completion.
    pub(super) fn release(&mut self, owner: Uuid) -> bool {
        let mut detached = false;
        for stage in self.stages.values_mut() {
            if stage.owner == Some(owner) {
                stage.owner = None;
                detached = true;
            }
        }
        if detached {
            self.epoch += 1;
        }
        detached
    }

    pub(super) fn has_work(&self) -> bool {
        self.registry.lock().unwrap().has_work(self.worker)
    }

    pub(super) fn next_deadline(&self) -> Option<f64> {
        let registry = self.registry.lock().unwrap();
        // A foreign worker may already have consumed the last completion.
        // Retain a dependency notification until this adapter observes it.
        if registry.completion_epoch != self.observed_completion_epoch {
            Some(registry.current_time_ms())
        } else {
            registry.next_deadline(self.worker)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::g3_offload::G3Tier;
    use crate::engine::host_offload::HostTierConfig;
    use crate::engine::{G3OffloadConfig, G3Scope};

    fn fixture(capacity: usize) -> (VllmG3OffloadAdapter, HostTier, SharedG3Tier) {
        fixture_with_read_bandwidth(capacity, 1.0)
    }

    fn fixture_with_read_bandwidth(
        capacity: usize,
        read_bw: f64,
    ) -> (VllmG3OffloadAdapter, HostTier, SharedG3Tier) {
        let registry = G3Tier::new(
            G3OffloadConfig {
                scope: G3Scope::ClusterShared,
                num_g3_blocks: 8,
                latency_to_first_byte_ms: 0.0,
                read_bandwidth_gbps: read_bw,
                write_bandwidth_gbps: 1.0,
                shared_read_bandwidth_gbps: read_bw,
                shared_write_bandwidth_gbps: 1.0,
            },
            2,
            1_000_000,
        )
        .unwrap();
        let tier = HostTier::new(HostTierConfig {
            capacity_blocks: capacity,
            block_bytes: 1_000_000,
            d2h_bandwidth_gbps: 0.0,
            h2d_bandwidth_gbps: 0.0,
        })
        .unwrap();
        (
            VllmG3OffloadAdapter::new(registry.clone(), 1),
            tier,
            registry,
        )
    }

    fn seed(tier: &mut HostTier, key: HostBlockKey) {
        assert!(matches!(
            tier.prepare_store(Uuid::nil(), &[key], 0.0),
            StoreOutcome::Prepared { .. }
        ));
        tier.submit_prepared_stores(0.0);
        tier.tick(0.0);
    }

    fn seed_secondary(registry: &SharedG3Tier, tier: &mut HostTier, keys: &[HostBlockKey]) -> f64 {
        let now = keys.len() as f64;
        let mut r = registry.lock().unwrap();
        r.submit(0, Direction::Write, keys, 0.0).unwrap();
        r.take_completed(0, now);
        tier.tick(now);
        now
    }

    #[test]
    fn partial_admission_survives_later_miss_and_pending_deduplicates_across_requests() {
        let (mut g3, mut tier, registry) = fixture(2);
        let [x, a, b] = [9, 1, 2].map(HostBlockKey::new);
        seed(&mut tier, x);
        tier.pin_external(&[x]);
        let now = seed_secondary(&registry, &mut tier, &[a, b]);
        let owner = Uuid::from_u128(1);
        assert!(g3.stage(&mut tier, owner, &[a, b], now));
        assert!(matches!(tier.lookup(a), Lookup::Pending { .. }));
        assert_eq!(tier.lookup(b), Lookup::Miss);
        assert!(g3.stage(&mut tier, Uuid::from_u128(2), &[a, b], now));
        assert!(!g3.release(Uuid::from_u128(2)));
        assert_eq!(registry.lock().unwrap().snapshot().read.submitted_jobs, 1);
        assert!(g3.release(owner));
        g3.advance(&mut tier, now + 1.0);
        assert_eq!(tier.lookup(a), Lookup::Hit);
        assert_eq!(
            registry.lock().unwrap().snapshot().read.completed_bytes,
            1_000_000
        );
        // Completion removes promotion ownership; next lookup may evict A for B.
        assert!(g3.stage(&mut tier, owner, &[a, b], now + 1.0));
        assert_eq!(tier.lookup(a), Lookup::Miss);
        g3.advance(&mut tier, now + 2.0);
        assert_eq!(tier.lookup(b), Lookup::Hit);
        tier.unpin_external(&[x]);
        assert!(g3.stage(&mut tier, owner, &[a, b], now + 2.0));
        // Releasing X touches it in LRU, so this retry may re-promote both
        // A and B while evicting X. Account for the full merged2MB job.
        g3.advance(&mut tier, now + 4.0);
        assert_eq!(tier.lookup(a), Lookup::Hit);
        assert_eq!(tier.lookup(b), Lookup::Hit);
    }

    #[test]
    fn pending_key_does_not_prevent_same_request_from_promoting_more_keys() {
        let (mut g3, mut tier, registry) = fixture(3);
        let [a, b, c] = [1, 2, 3].map(HostBlockKey::new);
        let now = seed_secondary(&registry, &mut tier, &[a, b, c]);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a], now));
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a, b, c], now));
        assert_eq!(g3.stages.len(), 2);
        let stats = registry.lock().unwrap().snapshot();
        assert_eq!(stats.read.submitted_jobs, 2); // [A], then one merged [B,C].
        assert!(g3.release(Uuid::nil()));
        assert!(g3.stages.values().all(|s| s.owner.is_none()));
        g3.advance(&mut tier, now + 3.0);
        assert!(g3.stages.is_empty());
        for key in [a, b, c] {
            assert_eq!(tier.lookup(key), Lookup::Hit);
        }
        assert_eq!(
            registry.lock().unwrap().snapshot().read.completed_bytes,
            3_000_000
        );
    }

    #[test]
    fn ordinary_hit_can_be_evicted_but_h2d_pin_protects_it() {
        for pin in [false, true] {
            let (mut g3, mut tier, registry) = fixture(2);
            let [x, a, b] = [9, 1, 2].map(HostBlockKey::new);
            seed(&mut tier, x);
            seed(&mut tier, a);
            tier.pin_external(&[x]);
            let now = seed_secondary(&registry, &mut tier, &[b]);
            let load = if pin {
                Some(tier.schedule_load(Uuid::nil(), &[a], now, now))
            } else {
                None
            };
            assert_eq!(g3.stage(&mut tier, Uuid::nil(), &[a, b], now), !pin);
            assert_eq!(tier.lookup(a), if pin { Lookup::Hit } else { Lookup::Miss });
            if let Some(crate::engine::host_offload::LoadOutcome::Queued(id)) = load {
                assert!(tier.cancel_load(id, now, now));
                assert!(g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
            }
        }
    }

    #[test]
    fn zero_time_repromotion_stops_until_prefix_fits_even_at_the_same_time() {
        let (mut g3, mut tier, registry) = fixture_with_read_bandwidth(2, 0.0);
        let [x, a, b] = [9, 1, 2].map(HostBlockKey::new);
        seed(&mut tier, x);
        tier.pin_external(&[x]);
        let now = seed_secondary(&registry, &mut tier, &[a, b]);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
        g3.advance(&mut tier, now);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
        g3.advance(&mut tier, now);
        assert!(!g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
        assert_eq!(tier.lookup(a), Lookup::Miss); // no fabricated prefix hit.
        assert_eq!(registry.lock().unwrap().snapshot().read.completed_jobs, 2);
        tier.unpin_external(&[x]);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
        g3.advance(&mut tier, now);
        assert!(!g3.stage(&mut tier, Uuid::nil(), &[a, b], now));
        assert_eq!(tier.lookup(a), Lookup::Hit);
        assert_eq!(tier.lookup(b), Lookup::Hit);
    }

    #[test]
    fn prefix_feasibility_does_not_touch_or_evict_entries() {
        let (_, mut tier, _) = fixture(2);
        let [a, x, b] = [1, 9, 2].map(HostBlockKey::new);
        seed(&mut tier, a);
        seed(&mut tier, x);
        tier.pin_external(&[x]);
        assert!(!tier.can_reserve_external_prefix(&[a, b]));
        tier.unpin_external(&[x]);
        for _ in 0..3 {
            assert!(tier.can_reserve_external_prefix(&[a, b]));
        }
        assert_eq!(tier.lookup(a), Lookup::Hit);
        assert_eq!(tier.lookup(x), Lookup::Hit);
        assert!(matches!(
            tier.reserve_external(Uuid::nil(), &[b], 0.0),
            StoreOutcome::Prepared { .. }
        ));
        assert_eq!(tier.lookup(a), Lookup::Miss); // still older than X.
        assert_eq!(tier.lookup(x), Lookup::Hit);
    }

    #[test]
    fn true_missing_tail_allows_completed_partial_prefix_without_new_promotion() {
        let (mut g3, mut tier, registry) = fixture(1);
        let [a, missing] = [1, 2].map(HostBlockKey::new);
        let now = seed_secondary(&registry, &mut tier, &[a]);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[a, missing], now));
        g3.advance(&mut tier, now + 1.0);
        assert!(!g3.stage(&mut tier, Uuid::nil(), &[a, missing], now + 1.0));
        assert_eq!(tier.lookup(a), Lookup::Hit);
        let crate::engine::host_offload::LoadOutcome::Queued(load) =
            tier.schedule_load(Uuid::nil(), &[a], now + 1.0, now + 1.0)
        else {
            panic!("the partial prefix must be available to H2D");
        };
        assert!(matches!(
            tier.prepare_store(Uuid::nil(), &[missing], now + 1.0),
            StoreOutcome::RetryCapacity { .. }
        ));
        assert!(tier.cancel_load(load, now + 1.0, now + 1.0));
        assert_eq!(registry.lock().unwrap().snapshot().read.submitted_jobs, 1);
    }

    #[test]
    fn reservation_rollback_preserves_resident_hits_and_foreign_pending_entries() {
        let (_, mut tier, _) = fixture(4);
        let [x, a, b, foreign] = [9, 1, 2, 3].map(HostBlockKey::new);
        seed(&mut tier, x);
        let reserve = |tier: &mut HostTier, owner, key| {
            let StoreOutcome::Prepared { transfer_id, .. } =
                tier.reserve_external(owner, &[key], 0.0)
            else {
                panic!("free slot")
            };
            transfer_id
        };
        let foreign_id = reserve(&mut tier, Uuid::from_u128(2), foreign);
        let own = [a, b].map(|k| reserve(&mut tier, Uuid::nil(), k));
        // Rollback before a read job is accepted is NOT request termination.
        for id in own {
            tier.cancel_external(id);
        }
        assert_eq!(tier.lookup(x), Lookup::Hit);
        for key in [a, b] {
            assert_eq!(tier.lookup(key), Lookup::Miss);
        }
        assert!(matches!(tier.lookup(foreign), Lookup::Pending { .. }));
        tier.complete_external(foreign_id);
        assert_eq!(tier.lookup(foreign), Lookup::Hit);
    }

    #[test]
    fn resident_hits_are_evictable_and_detached_promotions_still_complete() {
        for finish_before_cancel in [false, true] {
            let (mut g3, mut tier, registry) = fixture(2);
            let a = HostBlockKey::new(1);
            let b = HostBlockKey::new(2);
            let c = HostBlockKey::new(3);
            seed(&mut tier, a);
            registry
                .lock()
                .unwrap()
                .submit(0, Direction::Write, &[b], 0.0)
                .unwrap();
            registry.lock().unwrap().take_completed(0, 1.0);
            tier.tick(1.0);
            let owner = Uuid::from_u128(1);
            assert!(g3.stage(&mut tier, owner, &[a, b], 1.0));
            assert!(matches!(tier.lookup(b), Lookup::Pending { .. }));
            assert!(matches!(
                tier.prepare_store(Uuid::nil(), &[c], 1.0),
                StoreOutcome::Prepared { .. }
            ));
            assert_eq!(tier.lookup(a), Lookup::Miss);
            if finish_before_cancel {
                registry.lock().unwrap().advance(50.0);
            }
            assert!(g3.release(owner));
            assert!(matches!(tier.lookup(b), Lookup::Pending { .. }));
            g3.advance(&mut tier, 50.0);
            assert_eq!(tier.lookup(b), Lookup::Hit);
            assert_eq!(registry.lock().unwrap().snapshot().read.cancelled_jobs, 0);
            assert!(g3.stages.is_empty());
        }
    }

    #[test]
    fn cascade_retains_g2_source_until_g3_completion() {
        let (mut g3, mut tier, _) = fixture(1);
        let a = HostBlockKey::new(1);
        let b = HostBlockKey::new(2);
        seed(&mut tier, a);
        g3.store(&mut tier, &[a], 0.0);
        assert!(matches!(
            tier.prepare_store(Uuid::nil(), &[b], 0.0),
            StoreOutcome::RetryCapacity { .. }
        ));
        assert!(g3.advance(&mut tier, 1.0));
        assert!(matches!(
            tier.prepare_store(Uuid::nil(), &[b], 1.0),
            StoreOutcome::Prepared { .. }
        ));
    }

    #[test]
    fn foreign_completion_retains_a_wakeup_after_its_owner_drains_it() {
        let (mut g3, mut tier, registry) = fixture(2);
        let key = HostBlockKey::new(1);
        registry
            .lock()
            .unwrap()
            .submit(0, Direction::Write, &[key], 0.0)
            .unwrap();
        assert!(g3.stage(&mut tier, Uuid::nil(), &[key], 0.0));
        assert!(
            !g3.has_work(),
            "foreign pending writes do not belong to this worker"
        );
        assert_eq!(
            g3.next_deadline(),
            Some(1.0),
            "dependent requests still wake"
        );
        registry.lock().unwrap().take_completed(0, 1.0);
        assert_eq!(g3.next_deadline(), Some(1.0));
        assert!(g3.advance(&mut tier, 1.0));
        assert!(g3.next_deadline().is_none());
        assert!(g3.stage(&mut tier, Uuid::nil(), &[key], 1.0));
        assert_eq!(g3.stages.len(), 1);
    }

    #[test]
    fn owner_delivery_wakes_even_after_observing_a_future_foreign_epoch() {
        let (mut g3, mut tier, registry) = fixture(2);
        let key = HostBlockKey::new(1);
        registry
            .lock()
            .unwrap()
            .submit(0, Direction::Write, &[key], 0.0)
            .unwrap();
        registry.lock().unwrap().take_completed(0, 1.0);
        g3.advance(&mut tier, 1.0);
        assert!(g3.stage(&mut tier, Uuid::nil(), &[key], 1.0));
        registry.lock().unwrap().advance(50.0);
        assert!(g3.advance(&mut tier, 1.0));
        assert!(matches!(tier.lookup(key), Lookup::Pending { .. }));
        assert!(g3.advance(&mut tier, 2.0));
        assert_eq!(tier.lookup(key), Lookup::Hit);
    }
}
