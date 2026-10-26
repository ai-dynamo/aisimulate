// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::{
    EngineEventBatch, KvFetchAction, KvHints, Placement, PlacementDecision, PlacementEffects,
    PlacementPolicy, RequestIdentity, WorkerTopology,
};
use crate::replay::protocol::AuthoredKvHints;

/// Authored `kv_hints` resolution shared by the round-robin policies.
///
/// A hint is forwarded as its source's scheduler id only while the source
/// worker is live in this pool, advertises router hints and has the named
/// attention-DP rank. Any other authored hint is dropped: workers come and go
/// under scaling, so naming one that is absent is not an error.
#[derive(Debug, Default)]
struct AuthoredKvHintSources {
    /// Live workers that advertise router hints.
    router_hint_workers: BTreeSet<usize>,
}

impl AuthoredKvHintSources {
    fn new(workers: &[WorkerTopology]) -> Self {
        let mut sources = Self::default();
        for worker in workers {
            sources.ready(worker);
        }
        sources
    }

    fn ready(&mut self, worker: &WorkerTopology) {
        if worker.router_hint {
            self.router_hint_workers.insert(worker.worker_id);
        }
    }

    fn retire(&mut self, worker_id: usize) {
        self.router_hint_workers.remove(&worker_id);
    }

    /// `workers` are the policy's live workers, with scheduler ids by DP rank.
    fn resolve(
        &self,
        workers: &BTreeMap<usize, Vec<usize>>,
        hint: Option<AuthoredKvHints>,
    ) -> Option<KvHints> {
        let hint = hint.filter(|hint| self.router_hint_workers.contains(&hint.source_worker))?;
        let source_scheduler_id = *workers
            .get(&hint.source_worker)?
            .get(hint.source_dp_rank as usize)?;
        Some(KvHints {
            fetch: KvFetchAction {
                source_scheduler_id,
            },
        })
    }
}

#[derive(Debug)]
pub struct AggregatedRoundRobin {
    next_worker: usize,
    next_rank_by_worker: FxHashMap<usize, u32>,
    dp_size: u32,
}

#[derive(Debug)]
pub struct AggregatedRoundRobinPlacement<Events: EngineEventBatch> {
    counter: AggregatedRoundRobin,
    workers: BTreeMap<usize, Vec<usize>>,
    hints: AuthoredKvHintSources,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> AggregatedRoundRobinPlacement<Events> {
    pub fn new(dp_size: u32, workers: Vec<WorkerTopology>) -> Self {
        let mut counter = AggregatedRoundRobin::new(dp_size);
        for worker in &workers {
            counter.worker_ready(worker.worker_id);
        }
        Self {
            counter,
            hints: AuthoredKvHintSources::new(&workers),
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            events: PhantomData,
        }
    }
}

impl<Request, Events> PlacementPolicy<Request> for AggregatedRoundRobinPlacement<Events>
where
    Request: RequestIdentity,
    Events: EngineEventBatch,
{
    type Metadata = ();
    type Observation = Events;

    #[inline]
    fn place(
        &mut self,
        request: &Request,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_id = request
            .request_id()
            .ok_or_else(|| anyhow!("round-robin placement requires a request UUID"))?;
        let scheduler_id = self.counter.next(
            self.workers.keys().copied(),
            request.preferred_dp_rank(),
            |worker_id, rank| {
                self.workers
                    .get(&worker_id)
                    .and_then(|ranks| ranks.get(rank as usize))
                    .copied()
            },
        )?;
        let kv_hints = self
            .hints
            .resolve(&self.workers, request.authored_kv_hints());
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: 0,
                cache_sample: None,
                kv_hints,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    #[inline]
    fn observe(&mut self, _observation: Events, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    #[inline]
    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    #[inline]
    fn request_terminal(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    #[inline]
    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.counter.worker_ready(worker.worker_id);
        self.hints.ready(&worker);
        self.workers.insert(worker.worker_id, worker.scheduler_ids);
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        self.hints.retire(worker.worker_id);
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        self.hints.retire(worker.worker_id);
        self.counter.worker_removed(worker.worker_id);
        Ok(Vec::new())
    }

    #[inline]
    fn topology_settled(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

#[derive(Debug)]
pub struct PoolRoundRobinPlacement<Events: EngineEventBatch> {
    next: usize,
    workers: BTreeMap<usize, Vec<usize>>,
    hints: AuthoredKvHintSources,
    events: PhantomData<Events>,
}

impl<Events: EngineEventBatch> PoolRoundRobinPlacement<Events> {
    pub fn new(workers: Vec<WorkerTopology>) -> Self {
        Self {
            next: 0,
            hints: AuthoredKvHintSources::new(&workers),
            workers: workers
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
            events: PhantomData,
        }
    }

    #[cfg(test)]
    pub(crate) fn new_starting_at(workers: Vec<WorkerTopology>, next: usize) -> Self {
        let mut placement = Self::new(workers);
        placement.next = next;
        placement
    }
}

impl<Request, Events> PlacementPolicy<Request> for PoolRoundRobinPlacement<Events>
where
    Request: RequestIdentity,
    Events: EngineEventBatch,
{
    type Metadata = ();
    type Observation = Events;

    fn place(
        &mut self,
        request: &Request,
        _metadata: Self::Metadata,
        _session_id: Option<String>,
        _now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_id = request
            .request_id()
            .ok_or_else(|| anyhow!("round-robin placement requires a request UUID"))?;
        let preferred_dp_rank = request.preferred_dp_rank().map(|rank| rank as usize);
        let active_count = match preferred_dp_rank {
            Some(rank) => self
                .workers
                .values()
                .filter(|scheduler_ids| scheduler_ids.get(rank).is_some())
                .count(),
            None => self.workers.values().map(Vec::len).sum::<usize>(),
        };
        if active_count == 0 {
            if let Some(rank) = preferred_dp_rank {
                return Err(anyhow!(
                    "preferred attention-DP rank {rank} is out of range for the active worker pool"
                ));
            }
            return Err(anyhow!("no active workers for round-robin placement"));
        }
        let index = self.next % active_count;
        let scheduler_id = match preferred_dp_rank {
            Some(rank) => self
                .workers
                .values()
                .filter_map(|scheduler_ids| scheduler_ids.get(rank).copied())
                .nth(index),
            None => self
                .workers
                .values()
                .flat_map(|scheduler_ids| scheduler_ids.iter().copied())
                .nth(index),
        }
        .expect("active round-robin pool must contain a scheduler");
        self.next = index + 1;
        let kv_hints = self
            .hints
            .resolve(&self.workers, request.authored_kv_hints());
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id,
                scheduler_id,
                reported_overlap_tokens: 0,
                cache_sample: None,
                kv_hints,
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }

    fn observe(&mut self, _observation: Events, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn cancel_pending(&mut self, _request_id: Uuid) -> bool {
        false
    }

    fn request_terminal(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn prefill_completed(&mut self, _request_id: Uuid, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }

    fn pending_count(&self) -> usize {
        0
    }

    fn worker_ready(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.hints.ready(&worker);
        self.workers.insert(worker.worker_id, worker.scheduler_ids);
        Ok(Vec::new())
    }

    fn worker_draining(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        self.hints.retire(worker.worker_id);
        Ok(Vec::new())
    }

    fn worker_removed(&mut self, worker: WorkerTopology, _now_ms: f64) -> Result<Vec<Placement>> {
        self.workers.remove(&worker.worker_id);
        self.hints.retire(worker.worker_id);
        Ok(Vec::new())
    }

    fn topology_settled(&mut self, _now_ms: f64) -> Result<Vec<Placement>> {
        Ok(Vec::new())
    }
}

impl AggregatedRoundRobin {
    pub fn new(dp_size: u32) -> Self {
        Self {
            next_worker: 0,
            next_rank_by_worker: FxHashMap::default(),
            dp_size: dp_size.max(1),
        }
    }

    pub(crate) fn next(
        &mut self,
        mut active_workers: impl ExactSizeIterator<Item = usize>,
        preferred_rank: Option<u32>,
        rank_id: impl FnOnce(usize, u32) -> Option<usize>,
    ) -> Result<usize> {
        if active_workers.len() == 0 {
            return Err(anyhow!("no active workers for round-robin placement"));
        }
        let index = self.next_worker % active_workers.len();
        let worker_id = active_workers
            .nth(index)
            .expect("active round-robin worker must exist at the selected index");
        let rank = match preferred_rank {
            Some(rank) if rank >= self.dp_size => {
                return Err(anyhow!(
                    "preferred attention-DP rank {rank} is out of range for dp_size {}",
                    self.dp_size
                ));
            }
            Some(rank) => rank,
            None => {
                self.next_rank_by_worker
                    .get(&worker_id)
                    .copied()
                    .unwrap_or_default()
                    % self.dp_size
            }
        };
        let scheduler_id = rank_id(worker_id, rank).ok_or_else(|| {
            anyhow!("logical worker {worker_id} does not expose preferred attention-DP rank {rank}")
        })?;
        self.next_worker = index + 1;
        if preferred_rank.is_none() {
            self.next_rank_by_worker.insert(worker_id, rank + 1);
        }
        Ok(scheduler_id)
    }

    pub(crate) fn worker_removed(&mut self, worker_id: usize) {
        self.next_rank_by_worker.remove(&worker_id);
    }

    fn worker_ready(&mut self, worker_id: usize) {
        self.next_rank_by_worker.entry(worker_id).or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::protocol::DirectRequest;

    #[derive(Debug)]
    struct TestRequest(Uuid);

    impl RequestIdentity for TestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.0)
        }
    }

    #[derive(Debug)]
    struct RankedTestRequest {
        id: Uuid,
        preferred_dp_rank: u32,
    }

    impl RequestIdentity for RankedTestRequest {
        fn request_id(&self) -> Option<Uuid> {
            Some(self.id)
        }

        fn preferred_dp_rank(&self) -> Option<u32> {
            Some(self.preferred_dp_rank)
        }
    }

    /// A replay request with an authored hint, which placement reads through
    /// the production `RequestIdentity` impl of `DirectRequest`.
    #[derive(Debug)]
    struct HintedTestRequest(DirectRequest);

    impl HintedTestRequest {
        fn numbered(ordinal: u128, kv_hints: Option<AuthoredKvHints>) -> Self {
            Self(DirectRequest {
                uuid: Some(Uuid::from_u128(ordinal)),
                kv_hints,
                ..DirectRequest::default()
            })
        }
    }

    impl RequestIdentity for HintedTestRequest {
        fn request_id(&self) -> Option<Uuid> {
            RequestIdentity::request_id(&self.0)
        }

        fn authored_kv_hints(&self) -> Option<AuthoredKvHints> {
            self.0.authored_kv_hints()
        }
    }

    fn immediate(effects: PlacementEffects) -> Placement {
        let PlacementDecision::Immediate(placement) = effects.decision else {
            panic!("round-robin placement must be immediate");
        };
        placement
    }

    fn scheduler_id(
        policy: &mut impl PlacementPolicy<TestRequest, Metadata = ()>,
        ordinal: u128,
    ) -> usize {
        let effects = PlacementPolicy::<TestRequest>::place(
            policy,
            &TestRequest(Uuid::from_u128(ordinal)),
            (),
            None,
            0.0,
        )
        .unwrap();
        let PlacementDecision::Immediate(placement) = effects.decision else {
            panic!("round-robin placement must be immediate");
        };
        placement.scheduler_id
    }

    #[test]
    fn pool_rotation_preserves_position_after_topology_change() {
        let mut policy = PoolRoundRobinPlacement::<()>::new(vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10],
                router_hint: false,
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![11],
                router_hint: false,
            },
            WorkerTopology {
                worker_id: 2,
                scheduler_ids: vec![12],
                router_hint: false,
            },
        ]);

        assert_eq!(
            (1..=4)
                .map(|ordinal| scheduler_id(&mut policy, ordinal))
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 10]
        );
        PlacementPolicy::<TestRequest>::worker_draining(
            &mut policy,
            WorkerTopology {
                worker_id: 2,
                scheduler_ids: vec![12],
                router_hint: false,
            },
            0.0,
        )
        .unwrap();

        assert_eq!(scheduler_id(&mut policy, 5), 11);
    }

    #[test]
    fn aggregated_rotation_preserves_position_and_ranks_after_topology_changes() {
        let worker = |worker_id| WorkerTopology {
            worker_id,
            scheduler_ids: vec![worker_id * 10, worker_id * 10 + 1],
            router_hint: false,
        };
        let mut policy =
            AggregatedRoundRobinPlacement::<()>::new(2, vec![worker(1), worker(2), worker(3)]);
        assert_eq!(
            (1..=4)
                .map(|ordinal| scheduler_id(&mut policy, ordinal))
                .collect::<Vec<_>>(),
            vec![10, 20, 30, 11]
        );
        PlacementPolicy::<TestRequest>::worker_draining(&mut policy, worker(3), 0.0).unwrap();
        assert_eq!(scheduler_id(&mut policy, 5), 21);
        PlacementPolicy::<TestRequest>::worker_removed(&mut policy, worker(3), 0.0).unwrap();
        PlacementPolicy::<TestRequest>::worker_ready(&mut policy, worker(3), 0.0).unwrap();
        // The re-added worker starts again at rank zero; retained workers keep
        // their independent rank rotations.
        assert_eq!(scheduler_id(&mut policy, 6), 30);
        assert_eq!(scheduler_id(&mut policy, 7), 10);
        PlacementPolicy::<TestRequest>::worker_ready(&mut policy, worker(4), 0.0).unwrap();
        assert_eq!(
            (8..=10)
                .map(|ordinal| scheduler_id(&mut policy, ordinal))
                .collect::<Vec<_>>(),
            vec![20, 31, 40]
        );
    }

    #[test]
    fn aggregated_failed_rank_lookup_does_not_advance_worker_or_rank() {
        let mut counter = AggregatedRoundRobin::new(2);
        assert!(
            counter
                .next([1, 2].into_iter(), Some(2), |_, _| None)
                .is_err()
        );
        assert!(counter.next([1, 2].into_iter(), None, |_, _| None).is_err());
        let rank_id = |worker, rank| Some(worker * 10 + rank as usize);
        assert_eq!(counter.next([1, 2].into_iter(), None, rank_id).unwrap(), 10);
        assert_eq!(counter.next([1, 2].into_iter(), None, rank_id).unwrap(), 20);
        assert_eq!(counter.next([1, 2].into_iter(), None, rank_id).unwrap(), 11);
    }

    #[test]
    fn empty_pool_returns_an_error_instead_of_dividing_by_zero() {
        let mut policy = PoolRoundRobinPlacement::<()>::new(Vec::new());
        let error = PlacementPolicy::<TestRequest>::place(
            &mut policy,
            &TestRequest(Uuid::from_u128(1)),
            (),
            None,
            0.0,
        )
        .unwrap_err();

        assert!(error.to_string().contains("no active workers"));
    }

    #[test]
    fn pool_round_robin_honors_authored_dp_rank_across_workers() {
        let mut policy = PoolRoundRobinPlacement::<()>::new(vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10, 11],
                router_hint: false,
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![20, 21],
                router_hint: false,
            },
        ]);
        let place = |policy: &mut PoolRoundRobinPlacement<()>, ordinal, rank| {
            let effects = PlacementPolicy::<RankedTestRequest>::place(
                policy,
                &RankedTestRequest {
                    id: Uuid::from_u128(ordinal),
                    preferred_dp_rank: rank,
                },
                (),
                None,
                0.0,
            )?;
            let PlacementDecision::Immediate(placement) = effects.decision else {
                panic!("round-robin placement must be immediate");
            };
            Ok::<_, anyhow::Error>(placement.scheduler_id)
        };

        assert_eq!(place(&mut policy, 1, 1).unwrap(), 11);
        assert_eq!(place(&mut policy, 2, 1).unwrap(), 21);
        assert!(
            place(&mut policy, 3, 2)
                .unwrap_err()
                .to_string()
                .contains("out of range")
        );
    }

    #[test]
    fn aggregated_round_robin_honors_authored_dp_rank_within_each_worker() {
        let mut policy = AggregatedRoundRobinPlacement::<()>::new(
            2,
            vec![
                WorkerTopology {
                    worker_id: 0,
                    scheduler_ids: vec![10, 11],
                    router_hint: false,
                },
                WorkerTopology {
                    worker_id: 1,
                    scheduler_ids: vec![20, 21],
                    router_hint: false,
                },
            ],
        );

        let place = |policy: &mut AggregatedRoundRobinPlacement<()>, ordinal, rank| {
            let effects = PlacementPolicy::<RankedTestRequest>::place(
                policy,
                &RankedTestRequest {
                    id: Uuid::from_u128(ordinal),
                    preferred_dp_rank: rank,
                },
                (),
                None,
                0.0,
            )?;
            let PlacementDecision::Immediate(placement) = effects.decision else {
                panic!("round-robin placement must be immediate");
            };
            Ok::<_, anyhow::Error>(placement.scheduler_id)
        };

        assert_eq!(place(&mut policy, 1, 1).unwrap(), 11);
        assert_eq!(place(&mut policy, 2, 1).unwrap(), 21);
        assert!(
            place(&mut policy, 3, 2)
                .unwrap_err()
                .to_string()
                .contains("out of range")
        );
    }

    /// Two workers with two ranks each, whose engines accept router hints or not.
    fn kv_hint_topology(router_hint: bool) -> Vec<WorkerTopology> {
        vec![
            WorkerTopology {
                worker_id: 0,
                scheduler_ids: vec![10, 11],
                router_hint,
            },
            WorkerTopology {
                worker_id: 1,
                scheduler_ids: vec![20, 21],
                router_hint,
            },
        ]
    }

    fn hinted_placement(
        policy: &mut impl PlacementPolicy<HintedTestRequest, Metadata = ()>,
        ordinal: u128,
        source: Option<(usize, u32)>,
    ) -> Placement {
        let kv_hints = source.map(|(source_worker, source_dp_rank)| AuthoredKvHints {
            source_worker,
            source_dp_rank,
        });
        let request = HintedTestRequest::numbered(ordinal, kv_hints);
        immediate(policy.place(&request, (), None, 0.0).unwrap())
    }

    fn fetch_from(source_scheduler_id: usize) -> Option<KvHints> {
        Some(KvHints {
            fetch: KvFetchAction {
                source_scheduler_id,
            },
        })
    }

    fn assert_forwards_authored_kv_hints(
        policy: &mut impl PlacementPolicy<HintedTestRequest, Metadata = ()>,
    ) {
        // The source is resolved from its authored (worker, rank) to a scheduler
        // id, independent of which rank receives the request.
        assert_eq!(
            hinted_placement(policy, 1, Some((1, 0))).kv_hints,
            fetch_from(20)
        );
        assert_eq!(
            hinted_placement(policy, 2, Some((0, 1))).kv_hints,
            fetch_from(11)
        );
        // Round-robin never creates a hint.
        assert_eq!(hinted_placement(policy, 3, None).kv_hints, None);
    }

    #[test]
    fn authored_kv_hint_is_forwarded_as_scheduler_id() {
        assert_forwards_authored_kv_hints(&mut AggregatedRoundRobinPlacement::<()>::new(
            2,
            kv_hint_topology(true),
        ));
        assert_forwards_authored_kv_hints(&mut PoolRoundRobinPlacement::<()>::new(
            kv_hint_topology(true),
        ));
        // A pool without router-hint workers, such as P/D decode, forwards nothing.
        let mut plain = PoolRoundRobinPlacement::<()>::new(kv_hint_topology(false));
        assert_eq!(hinted_placement(&mut plain, 4, Some((1, 0))).kv_hints, None);
    }

    /// A hint whose source has drained or been removed, or that names a worker
    /// or rank the pool lacks, is dropped, never an error, and the request is
    /// still placed. A worker that joins later is a source.
    fn assert_drops_authored_kv_hints_without_a_live_source(
        policy: &mut impl PlacementPolicy<HintedTestRequest, Metadata = ()>,
    ) {
        let drained = kv_hint_topology(true).remove(1);
        policy.worker_draining(drained.clone(), 0.0).unwrap();
        let placement = hinted_placement(policy, 1, Some((1, 1)));
        assert_eq!(placement.kv_hints, None);
        assert_eq!(placement.scheduler_id / 10, 1, "worker 0 serves it");
        policy.worker_removed(drained, 0.0).unwrap();
        for source in [(1, 0), (0, 2), (7, 0)] {
            assert_eq!(hinted_placement(policy, 2, Some(source)).kv_hints, None);
        }
        policy
            .worker_ready(
                WorkerTopology {
                    worker_id: 2,
                    scheduler_ids: vec![30, 31],
                    router_hint: true,
                },
                0.0,
            )
            .unwrap();
        assert_eq!(
            hinted_placement(policy, 3, Some((2, 1))).kv_hints,
            fetch_from(31)
        );
    }

    #[test]
    fn authored_kv_hint_without_a_live_source_is_dropped() {
        assert_drops_authored_kv_hints_without_a_live_source(
            &mut AggregatedRoundRobinPlacement::<()>::new(2, kv_hint_topology(true)),
        );
        assert_drops_authored_kv_hints_without_a_live_source(
            &mut PoolRoundRobinPlacement::<()>::new(kv_hint_topology(true)),
        );
    }
}
