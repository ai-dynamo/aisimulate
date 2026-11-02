// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM CPU-offload connector behavior over the framework-neutral host tier.
//!
//! The default profile is vLLM 0.24's `OffloadingConnector`. It applies
//! ordinary resident touches in exact reverse logical order. Store and load
//! completion keys, however, travel through Python sets; their
//! within-completion order is process/version dependent and is not a stable
//! framework contract. The host tier therefore gives each completion cohort
//! one recency epoch and uses logical identity only as a deterministic
//! tie-break inside that epoch.
//!
//! Two vLLM 0.32.0 profiles ([`G2Connector`]) change these rules:
//!
//! - `offloading_v0_32`, `OffloadingConnector` with the CPU backend: lookups
//!   and stores touch nothing; a block is ranked when its store is admitted;
//!   a finishing request offers its remaining prompt blocks and re-ranks every
//!   block it stored, offered or loaded, its first block newest
//!   (`vllm/v1/kv_offload/cpu/policies/lru.py`).
//! - `simple_cpu_v0_32`, `SimpleCPUOffloadConnector` in eager mode
//!   (`vllm/v1/simple_kv_offload/manager.py`): every block computed before the
//!   current pass is stored from a per-request cursor that starts at block 0,
//!   decode blocks included, and a finishing request stores the rest; a block
//!   whose store is in flight misses; a hit stays pinned until the request
//!   loads it or looks up again; peers load the same block at once; a block
//!   becomes the newest when its last pin drops (a completed store, a released
//!   load or lookup pin), the CPU block pool's free queue. A store holds its G1
//!   source blocks until the copy completes, and a preemption waits for every
//!   transfer in flight (`vllm/v1/simple_kv_offload/worker.py`).
//!
//! With a state cache in G1 (a hybrid model), both 0.32.0 profiles also keep
//! recurrent states, one key per state slot ([`HostBlockKey::state`]). A state
//! is offered when G1 publishes it, before the pass's blocks (the scheduler's
//! boundary hand-off), and a hit ends at the deepest block boundary that holds
//! the attention prefix and every slot of the state there.

use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::sync::Arc;
use uuid::Uuid;

use crate::engine::common::protocols::PrefillCost;
use crate::engine::common::sequence::RequestSequence;
use crate::engine::host_offload::{
    CompletedTransfer, G2Binding, HostBlockKey, HostBlockMeta, HostClient, HostOffloadObservation,
    HostOffloadObservationData, HostStoreBlockMapping, HostTierHandle, LoadOutcome, Lookup,
    StoreOutcome, TransferId,
};
use crate::engine::kv_manager::{
    BlockRequestLease, DestinationReservation, G1Acquire, G1Manager, SourceReuseDependency,
    StateHandoff,
};
use crate::engine::{G2Connector, HostOffloadObserver, KvHintSource, NativeHostOffloadConfig};

struct LoadingPrefix {
    transfer_id: TransferId,
    keys: Vec<HostBlockKey>,
    /// State slots at the head of `keys`, restored with the prefix.
    state_slots: usize,
    reservation: DestinationReservation,
    original_g1_tokens: usize,
    loaded_prefix_blocks: usize,
    request_footprint_blocks: usize,
}

#[derive(Clone, Copy)]
struct ActivatedPrefix {
    original_g1_tokens: usize,
    loaded_prefix_blocks: usize,
    loaded_blocks: usize,
    request_footprint_blocks: usize,
}

enum LoadState {
    Loading(LoadingPrefix),
    Activated(ActivatedPrefix),
}

/// Restore rounds (lookups that submit G3 promotions or KVCR pulls) a waiting
/// request may make while its DP rank schedules no work and emits no output. A
/// prefix larger than G2 can evict its own restored blocks forever; past this
/// bound the request stops restoring and computes what G2 cannot supply. A
/// request merely waiting behind running work never reaches it.
pub(super) const MAX_RESTORE_ROUNDS_WITHOUT_PROGRESS: usize = 1024;

/// Request-local G3 or KVCR restore progress.
#[derive(Default)]
pub(super) struct RestoreRounds {
    pub(super) rounds_without_progress: usize,
    /// The rank's progress count at this request's last round.
    rank_progress: u64,
}

impl RestoreRounds {
    pub(super) fn bypassed(&self) -> bool {
        self.rounds_without_progress >= MAX_RESTORE_ROUNDS_WITHOUT_PROGRESS
    }

    /// The request was scheduled or preempted: its restores start over.
    pub(super) fn reset(&mut self) {
        self.rounds_without_progress = 0;
    }

    /// Count one restore round; the count restarts whenever the rank made
    /// progress. Returns true when this round reaches the bound.
    pub(super) fn record_round(&mut self, rank_progress: u64) -> bool {
        if self.rank_progress != rank_progress {
            self.rank_progress = rank_progress;
            self.rounds_without_progress = 0;
        }
        self.rounds_without_progress += 1;
        self.bypassed()
    }
}

/// Request-local connector cursor and transient load ownership.
pub(super) struct VllmHostRequestState {
    owner: Uuid,
    /// One key per complete prompt block; `simple_cpu_v0_32` appends the keys
    /// of blocks completed during decode as it stores them.
    prompt_keys: Vec<HostBlockKey>,
    /// Router identities, retained only for `HostPinned` residency events.
    prompt_meta: Option<Vec<HostBlockMeta>>,
    /// Peer rank the router named as holding a prefix of this prompt.
    kv_hint: Option<KvHintSource>,
    /// G3 or KVCR restore rounds; the two are exclusive in v1.
    restore_rounds: RestoreRounds,
    next_store_block: usize,
    latest_store: Option<TransferId>,
    load: Option<LoadState>,
    first_admission_recorded: bool,
    /// `simple_cpu_v0_32`: tokens computed before the latest pass, the prefix
    /// the store may copy.
    confirmed_tokens: usize,
    /// `simple_cpu_v0_32`: keys the latest lookup hit, pinned until the load.
    lookup_pin: Option<Vec<HostBlockKey>>,
    /// `offloading_v0_32`: blocks stored, offered or loaded, re-ranked when
    /// the request finishes.
    accessed_blocks: BTreeSet<usize>,
    /// `offloading_v0_32`: boundaries, in blocks, of the states stored,
    /// offered or loaded, re-ranked with the blocks.
    accessed_states: BTreeSet<usize>,
}

impl VllmHostRequestState {
    pub(super) fn new(
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
        block_size: usize,
        retain_meta: bool,
        kv_hint: Option<KvHintSource>,
    ) -> Self {
        let prompt_blocks = sequence.num_input_tokens() / block_size;
        let prompt_keys: Vec<_> = (0..prompt_blocks)
            .map(|index| {
                HostBlockKey::new(
                    lease
                        .sequence_hash(index)
                        .expect("complete prompt block must retain its sequence hash"),
                )
            })
            .collect();
        let prompt_meta = retain_meta.then(|| {
            let session_id = lease.session_id.as_deref().map(Arc::<str>::from);
            (0..prompt_blocks)
                .map(|index| HostBlockMeta {
                    parent: index
                        .checked_sub(1)
                        .map(|parent| prompt_keys[parent].sequence_hash()),
                    tokens_hash: lease
                        .local_hash(index)
                        .expect("shared G2 metadata requires retained local hashes"),
                    position: index,
                    session_id: session_id.clone(),
                })
                .collect()
        });
        Self {
            owner: lease.owner(),
            prompt_keys,
            prompt_meta,
            kv_hint,
            restore_rounds: RestoreRounds::default(),
            next_store_block: 0,
            latest_store: None,
            load: None,
            first_admission_recorded: false,
            confirmed_tokens: 0,
            lookup_pin: None,
            accessed_blocks: BTreeSet::new(),
            accessed_states: BTreeSet::new(),
        }
    }

    /// Append the keys (and router identities) of blocks the lease completed
    /// past the known ones, up to `blocks`.
    fn extend_keys(&mut self, lease: &BlockRequestLease, blocks: usize) {
        while self.prompt_keys.len() < blocks {
            let index = self.prompt_keys.len();
            let Some(hash) = lease.sequence_hash(index) else {
                return;
            };
            if let Some(meta) = &mut self.prompt_meta {
                let Some(tokens_hash) = lease.local_hash(index) else {
                    return;
                };
                meta.push(HostBlockMeta {
                    parent: index
                        .checked_sub(1)
                        .map(|parent| self.prompt_keys[parent].sequence_hash()),
                    tokens_hash,
                    position: index,
                    session_id: lease.session_id.as_deref().map(Arc::<str>::from),
                });
            }
            self.prompt_keys.push(HostBlockKey::new(hash));
        }
    }

    /// A completed G2 load is activated and awaits admission.
    pub(super) fn has_activated_load(&self) -> bool {
        self.activated().is_some()
    }

    fn activated(&self) -> Option<ActivatedPrefix> {
        match self.load {
            Some(LoadState::Activated(load)) => Some(load),
            _ => None,
        }
    }

    pub(super) fn reserved_blocks(&self) -> usize {
        match &self.load {
            Some(LoadState::Loading(load)) => load.loaded_prefix_blocks,
            Some(LoadState::Activated(load)) => load.loaded_prefix_blocks,
            None => 0,
        }
    }

    pub(super) fn unreserved_tail_blocks(&self) -> usize {
        match &self.load {
            Some(LoadState::Loading(load)) => load
                .request_footprint_blocks
                .saturating_sub(load.loaded_prefix_blocks),
            Some(LoadState::Activated(load)) => load
                .request_footprint_blocks
                .saturating_sub(load.loaded_prefix_blocks),
            None => 0,
        }
    }
}

pub(super) struct HostLookupHit {
    /// The state slots at the hit boundary (a hybrid hit), then the blocks
    /// after the G1 prefix. SimpleCPU loads and frees the Mamba groups first.
    keys: Vec<HostBlockKey>,
    state_slots: usize,
    base_g1_blocks: usize,
    original_g1_tokens: usize,
    request_footprint_blocks: usize,
}

pub(super) enum HostLookup {
    Miss,
    Hit(HostLookupHit),
    Deferred,
}

pub(super) enum StartLoad {
    Queued,
    Deferred,
    Retry,
    CapacityBlocked,
}

/// Transfer completions that require request-owned G1 activation.
pub(super) struct CompletedLoad {
    pub(super) uuid: Uuid,
    pub(super) transfer_id: TransferId,
}

pub(super) struct HostTransferProgress {
    pub(super) completed_loads: Vec<CompletedLoad>,
    pub(super) completed_any: bool,
}

/// Recurrent-state geometry of a hybrid model's G1 state cache.
#[derive(Clone, Copy)]
struct HybridStates {
    /// Slots per state, one per Mamba KV cache group.
    slots: usize,
}

impl HybridStates {
    fn keys(self, prefix: HostBlockKey) -> Vec<HostBlockKey> {
        (0..self.slots)
            .map(|slot| HostBlockKey::state(prefix.sequence_hash(), slot))
            .collect()
    }
}

pub(super) struct VllmHostOffloadAdapter {
    host: HostClient,
    shared: bool,
    connector: G2Connector,
    /// Set when G1 has a state cache: G2 keeps recurrent states too.
    hybrid: Option<HybridStates>,
    /// The rank publishes G2 residency as `HostPinned` KV events.
    publishes_residency: bool,
    g3: Option<super::g3_offload::VllmG3OffloadAdapter>,
    kvcr: Option<super::kvcr::VllmKvcrAdapter>,
    load_epoch: u64,
    /// Keys being loaded, which defer a peer's lookup. SimpleCPU loads a block
    /// into several requests at once, so it tracks none.
    load_by_key: FxHashMap<HostBlockKey, TransferId>,
    compute_not_before_ms: f64,
    /// Present only for detailed artifact capture. Ordinary runs neither retain
    /// request-local mapping state nor allocate mapping payloads.
    observer: Option<Arc<dyn HostOffloadObserver>>,
}

impl VllmHostOffloadAdapter {
    /// Shared pools are sampled by the deployment registry, not once per rank.
    pub(super) fn per_rank_cache_metrics(&self) -> Option<crate::engine::HostCacheMetrics> {
        (!self.shared).then(|| {
            self.host
                .lock()
                .metrics(crate::engine::G2Scope::DpRankLocal)
        })
    }

    pub(super) fn set_g3(
        &mut self,
        registry: crate::engine::g3_offload::SharedG3Tier,
        node: usize,
    ) {
        self.host.lock().hold_completed_sources(self.host.id());
        self.g3 = Some(super::g3_offload::VllmG3OffloadAdapter::new(registry, node));
    }

    /// Returns this rank's G2, which peers pull from when a hint names it.
    pub(super) fn set_kvcr(
        &mut self,
        registry: crate::engine::kvcr::SharedKvcrRegistry,
        rank: KvHintSource,
    ) -> HostTierHandle {
        self.kvcr = Some(super::kvcr::VllmKvcrAdapter::new(registry, rank));
        self.host.tier_handle()
    }

    /// Changes whenever G3 or KVCR stages, lands or loads; drives
    /// same-timestamp retries. `None` without G3 until this rank's first KVCR
    /// pull, so an idle KVCR schedules exactly like G2 alone.
    pub(super) fn retry_epoch(&self) -> Option<u64> {
        let kvcr_epoch = self.kvcr.as_ref().map_or(0, |kvcr| kvcr.epoch);
        (self.g3.is_some() || kvcr_epoch > 0)
            .then(|| self.g3.as_ref().map_or(0, |g3| g3.epoch) + kvcr_epoch + self.load_epoch)
    }

    pub(super) fn note_rank_progress(&mut self) {
        if let Some(g3) = &mut self.g3 {
            g3.note_rank_progress();
        }
        if let Some(kvcr) = &mut self.kvcr {
            kvcr.note_rank_progress();
        }
    }

    pub(super) fn set_observer(&mut self, observer: Arc<dyn HostOffloadObserver>) {
        self.host.lock().set_observer(Arc::clone(&observer));
        self.observer = Some(observer);
    }

    /// `subscribe` delivers the pool's `HostPinned` residency events to this
    /// rank: its private cache, or every block of a shared pool.
    pub(super) fn new(
        config: &NativeHostOffloadConfig,
        block_size: usize,
        kv_bytes_per_token: usize,
        binding: Option<&G2Binding>,
        subscribe: bool,
    ) -> anyhow::Result<Self> {
        let host = HostClient::new(config, block_size, kv_bytes_per_token, binding)?;
        let shared = host.lock().is_shared();
        if subscribe {
            host.lock().subscribe(host.id());
        }
        Ok(Self {
            host,
            shared,
            connector: config.connector,
            hybrid: None,
            publishes_residency: subscribe,
            g3: None,
            kvcr: None,
            load_epoch: 0,
            load_by_key: FxHashMap::default(),
            compute_not_before_ms: 0.0,
            observer: None,
        })
    }

    fn tracks_loads(&self) -> bool {
        self.connector != G2Connector::SimpleCpuV032
    }

    /// Whether a store keeps a reference on the G1 blocks it copies until the
    /// copy completes: a cluster-shared pool, like Mooncake Store and LMCache
    /// MP, or SimpleCPU, which touches its GPU source blocks for the copy. A
    /// hybrid G2 does too, as state allocations are not fenced behind a
    /// pending copy.
    pub(super) fn holds_store_sources(&self) -> bool {
        self.shared || self.connector == G2Connector::SimpleCpuV032 || self.hybrid.is_some()
    }

    /// G1 has a state cache: lookups, loads and stores carry recurrent states.
    pub(super) fn set_hybrid(&mut self, blocks_per_state: usize) {
        self.hybrid = Some(HybridStates {
            slots: blocks_per_state,
        });
    }

    pub(super) fn is_hybrid(&self) -> bool {
        self.hybrid.is_some()
    }

    /// Whether requests must retain router identities for residency events.
    pub(super) fn publishes_residency(&self) -> bool {
        self.shared || self.publishes_residency
    }

    /// Query G2 after the scheduler's one authoritative G1 prefix lookup.
    pub(super) fn lookup(
        &mut self,
        request: &mut VllmHostRequestState,
        sequence: &RequestSequence,
        g1_cost: &PrefillCost,
        block_size: usize,
        now_ms: f64,
    ) -> HostLookup {
        if request.activated().is_some() {
            return HostLookup::Miss;
        }
        // vLLM skips a request waiting for its async connector load
        // (`WAITING_FOR_REMOTE_KVS`) without a lookup or LRU touch until the
        // load finishes; a landing pull or H2D completion wakes it.
        if matches!(request.load, Some(LoadState::Loading(_)))
            || self
                .kvcr
                .as_ref()
                .is_some_and(|kvcr| kvcr.is_pulling(request.owner))
        {
            return HostLookup::Deferred;
        }
        let hybrid = self.hybrid;
        let Self {
            host,
            connector,
            g3,
            kvcr,
            load_by_key,
            ..
        } = self;
        let simple = *connector == G2Connector::SimpleCpuV032;
        let mut tier = host.lock();
        // SimpleCPU drops the previous lookup's pin before looking up again.
        if let Some(pinned) = request.lookup_pin.take() {
            tier.unpin_external(&pinned);
        }
        if let Some(transfer_id) = request.latest_store {
            if !tier.has_transfer(transfer_id) {
                request.latest_store = None;
            } else if !simple {
                // The offloading connectors wait for the request's own stores.
                return HostLookup::Deferred;
            }
        }

        debug_assert_eq!(g1_cost.cached_tokens % block_size, 0);
        let base_g1_blocks = g1_cost.cached_tokens / block_size;
        // SimpleCPU must recompute at least the last token, and so must the
        // offloading connector of a hybrid model, whose Mamba groups count as
        // sliding-window groups there.
        let end = if simple || hybrid.is_some() {
            ((sequence.len() - 1) / block_size).min(request.prompt_keys.len())
        } else {
            request.prompt_keys.len()
        };
        let outcome = if base_g1_blocks >= request.prompt_keys.len() {
            if let Some(g3) = g3 {
                g3.release(request.owner);
            }
            if let Some(kvcr) = kvcr {
                kvcr.release(request.owner);
            }
            HostLookup::Miss
        } else if let Some(g3) = g3
            && !request.restore_rounds.bypassed()
            && g3.stage(
                &mut tier,
                host.id(),
                request.owner,
                &mut request.restore_rounds,
                &request.prompt_keys[base_g1_blocks..],
                request
                    .prompt_meta
                    .as_deref()
                    .map(|meta| &meta[base_g1_blocks..]),
                now_ms,
            )
        {
            HostLookup::Deferred
        } else if let Some(kvcr) = kvcr
            && let Some(source) = request.kv_hint
            && !request.restore_rounds.bypassed()
            && kvcr.stage(
                &mut tier,
                host.id(),
                request.owner,
                source,
                &mut request.restore_rounds,
                &request.prompt_keys[base_g1_blocks..],
                request
                    .prompt_meta
                    .as_deref()
                    .map(|meta| &meta[base_g1_blocks..]),
                now_ms,
            )
        {
            HostLookup::Deferred
        } else {
            // SimpleCPU registers a block only when its store completes; the
            // offloading connectors count a block being stored, then defer.
            let present = |key: &HostBlockKey| match tier.lookup(*key) {
                Lookup::Hit => true,
                Lookup::Pending { .. } => !simple,
                Lookup::Miss => false,
            };
            let mut matched = request.prompt_keys[base_g1_blocks..end]
                .iter()
                .take_while(|key| present(key))
                .count();
            // A hybrid hit ends at the deepest boundary that also holds every
            // slot of its state.
            let mut keys = Vec::new();
            if let Some(hybrid) = hybrid {
                (matched, keys) = (1..=matched)
                    .rev()
                    .find_map(|blocks| {
                        let states = hybrid.keys(request.prompt_keys[base_g1_blocks + blocks - 1]);
                        states.iter().all(present).then_some((blocks, states))
                    })
                    .unwrap_or_default();
            }
            let state_slots = keys.len();
            keys.extend_from_slice(&request.prompt_keys[base_g1_blocks..base_g1_blocks + matched]);
            if matched == 0 {
                HostLookup::Miss
            } else if keys.iter().any(|key| {
                load_by_key.contains_key(key) || matches!(tier.lookup(*key), Lookup::Pending { .. })
            }) {
                HostLookup::Deferred
            } else {
                HostLookup::Hit(HostLookupHit {
                    keys,
                    state_slots,
                    base_g1_blocks,
                    original_g1_tokens: g1_cost.cached_tokens,
                    // The first pass after a restore writes a new working state.
                    request_footprint_blocks: sequence.current_known_blocks()
                        + hybrid.map_or(0, |hybrid| hybrid.slots),
                })
            }
        };

        match connector {
            // vLLM 0.24 passes this ordered logical list after lookup to
            // LRUCachePolicy.touch(), which explicitly iterates it in reverse.
            G2Connector::OffloadingV024 => {
                for key in request.prompt_keys.iter().rev().copied() {
                    tier.touch(key);
                }
            }
            G2Connector::OffloadingV032 => {}
            // SimpleCPU pins its hit until the request loads it or looks up
            // again.
            G2Connector::SimpleCpuV032 => {
                if let HostLookup::Hit(hit) = &outcome {
                    tier.pin_external(&hit.keys);
                    request.lookup_pin = Some(hit.keys.clone());
                }
            }
        }
        outcome
    }

    /// Reserve real G1 capacity before queuing the H2D.
    pub(super) fn start_load(
        &mut self,
        uuid: Uuid,
        request: &mut VllmHostRequestState,
        lease: &BlockRequestLease,
        hit: HostLookupHit,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> StartLoad {
        if hit
            .keys
            .iter()
            .any(|key| self.load_by_key.contains_key(key))
        {
            return StartLoad::Deferred;
        }
        let hashes = hit.keys[hit.state_slots..]
            .iter()
            .map(|key| key.sequence_hash())
            .collect::<Vec<_>>();
        let reservation = match kv_manager.reserve_native_host_destination(
            uuid,
            lease,
            hit.base_g1_blocks,
            &hashes,
            hit.state_slots > 0,
        ) {
            G1Acquire::Ready(reservation) => reservation,
            G1Acquire::CapacityExhausted => return StartLoad::CapacityBlocked,
        };

        let mut tier = self.host.lock();
        // Only a private lane hands out G1 capacity that is still being
        // copied; a shared pool holds it until the copy completes.
        let dependencies = kv_manager.native_destination_pending_dependencies(&reservation);
        let mut not_before_ms = now_ms;
        for dependency in dependencies {
            assert!(kv_manager.is_native_source_dependency_pending(dependency));
            let deadline = tier
                .transfer_deadline(transfer_id(dependency))
                .expect("pending source dependency must retain a submitted D2H");
            not_before_ms = not_before_ms.max(deadline);
        }

        // Transfer bookkeeping follows the logical prefix, unlike LRU touches.
        let transfer_id =
            match tier.schedule_load(self.host.id(), uuid, &hit.keys, now_ms, not_before_ms) {
                LoadOutcome::Queued(transfer_id) => transfer_id,
                LoadOutcome::Miss => {
                    kv_manager.cancel_destination(reservation);
                    return StartLoad::Retry;
                }
            };
        // The load holds the hit now.
        if let Some(pinned) = request.lookup_pin.take() {
            tier.unpin_external(&pinned);
        }
        if let Some(kvcr) = &self.kvcr {
            kvcr.note_loaded(tier.take_kvcr_origin(&hit.keys));
        }
        drop(tier);

        if self.tracks_loads() {
            for key in &hit.keys {
                assert!(self.load_by_key.insert(*key, transfer_id).is_none());
            }
        }
        let loaded_prefix_blocks = hit.base_g1_blocks + hashes.len();
        if self.connector == G2Connector::OffloadingV032 {
            request
                .accessed_blocks
                .extend(hit.base_g1_blocks..loaded_prefix_blocks);
            if hit.state_slots > 0 {
                request.accessed_states.insert(loaded_prefix_blocks);
            }
        }
        if let Some(g3) = &mut self.g3 {
            g3.release(uuid);
        }
        if let Some(kvcr) = &mut self.kvcr {
            kvcr.release(uuid);
        }
        self.load_epoch += 1;
        request.load = Some(LoadState::Loading(LoadingPrefix {
            transfer_id,
            keys: hit.keys,
            state_slots: hit.state_slots,
            reservation,
            original_g1_tokens: hit.original_g1_tokens,
            loaded_prefix_blocks,
            request_footprint_blocks: hit.request_footprint_blocks,
        }));
        StartLoad::Queued
    }

    /// Complete due transfers. Store dependencies become terminal before any
    /// same-timestamp H2D activation is handed back to the scheduler.
    pub(super) fn advance(
        &mut self,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> HostTransferProgress {
        let mut tier = self.host.lock();
        let client = self.host.id();
        let (completed, mut completed_any) = tier.tick(client, now_ms);
        self.load_epoch += completed.len() as u64;
        for transfer in &completed {
            let CompletedTransfer::Store {
                request_id: _,
                transfer_id,
                blocks,
            } = transfer
            else {
                continue;
            };
            assert!(kv_manager.satisfy_native_source_dependency(source_dependency(*transfer_id)));
            if let Some(g3) = &mut self.g3 {
                g3.store(&mut tier, blocks, now_ms);
            }
        }

        if let Some(g3) = &mut self.g3 {
            completed_any |= g3.advance(&mut tier, now_ms);
        }
        if let Some(kvcr) = &mut self.kvcr {
            completed_any |= kvcr.advance(&mut tier, now_ms);
        }
        // Drain after G3 and KVCR: a promotion or pull landing in G2 above
        // publishes residency that no later deadline would otherwise deliver.
        for event in tier.take_events(client) {
            kv_manager.publish_host_pinned_event(event);
        }
        let mut loads = Vec::new();
        for transfer in completed {
            let CompletedTransfer::Load {
                request_id: uuid,
                transfer_id,
                blocks,
            } = transfer
            else {
                continue;
            };
            if self.tracks_loads() {
                for key in &blocks {
                    assert_eq!(self.load_by_key.remove(key), Some(transfer_id));
                }
            }
            loads.push(CompletedLoad { uuid, transfer_id });
        }
        HostTransferProgress {
            completed_loads: loads,
            completed_any,
        }
    }

    /// Submit the stores prepared by the completed scheduler pass, then settle
    /// configured zero-latency transfers at the same boundary.
    pub(super) fn complete_engine_boundary(
        &mut self,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) -> HostTransferProgress {
        let mut progress = self.advance(kv_manager, now_ms);
        self.host
            .lock()
            .submit_prepared_stores(self.host.id(), now_ms);
        let settled = self.advance(kv_manager, now_ms);
        progress.completed_loads.extend(settled.completed_loads);
        progress.completed_any |= settled.completed_any;
        progress
    }
    pub(super) fn activate_completed_load(
        &mut self,
        uuid: Uuid,
        transfer_id: TransferId,
        request: &mut VllmHostRequestState,
        sequence: &RequestSequence,
        lease: &mut BlockRequestLease,
        kv_manager: &mut G1Manager,
    ) {
        let Some(LoadState::Loading(load)) = request.load.take() else {
            panic!("completed H2D lost its request reservation")
        };
        assert_eq!(load.transfer_id, transfer_id);
        assert!(
            kv_manager
                .native_destination_pending_dependencies(&load.reservation)
                .is_empty(),
            "H2D completed before its destination dependencies"
        );
        kv_manager.activate_native_destination(uuid, sequence, lease, load.reservation);
        request.load = Some(LoadState::Activated(ActivatedPrefix {
            original_g1_tokens: load.original_g1_tokens,
            loaded_prefix_blocks: load.loaded_prefix_blocks,
            loaded_blocks: load.keys.len() - load.state_slots,
            request_footprint_blocks: load.request_footprint_blocks,
        }));
    }

    /// Freeze the first post-H2D retry to the prefix owned by its reservation.
    pub(super) fn freeze_activated_cost(
        request: &VllmHostRequestState,
        mut cost: PrefillCost,
        block_size: usize,
    ) -> PrefillCost {
        let Some(load) = request.activated() else {
            return cost;
        };
        let loaded_tokens = load.loaded_prefix_blocks * block_size;
        assert!(cost.cached_tokens >= loaded_tokens);
        let newly_visible = cost.cached_tokens - loaded_tokens;
        assert_eq!(newly_visible % block_size, 0);
        cost.cached_tokens = loaded_tokens;
        cost.active_cached_tokens = loaded_tokens;
        cost.new_tokens += newly_visible;
        cost.new_blocks += newly_visible / block_size;
        cost
    }

    /// Advance the persistent store cursor and return first-admission tier attribution.
    pub(super) fn on_admitted(
        &self,
        request: &mut VllmHostRequestState,
        reused_tokens: usize,
        block_size: usize,
    ) -> Option<(usize, usize)> {
        request.restore_rounds.reset();
        let attribution = match request.load.take() {
            Some(LoadState::Activated(load)) => {
                assert!(reused_tokens / block_size <= load.loaded_prefix_blocks);
                let g1 = load.original_g1_tokens.min(reused_tokens);
                let g2 = reused_tokens - g1;
                assert!(g2 <= load.loaded_blocks * block_size);
                request.next_store_block = reused_tokens / block_size;
                (g1, g2)
            }
            Some(LoadState::Loading(_)) => panic!("loading request cannot be admitted"),
            None => (reused_tokens, 0),
        };
        if self.connector == G2Connector::SimpleCpuV032 {
            // SimpleCPU scans every admitted request from block 0, a loaded or
            // preempted one included; its reused prefix is already computed.
            request.next_store_block = 0;
            request.confirmed_tokens = reused_tokens;
        }
        if request.first_admission_recorded {
            None
        } else {
            request.first_admission_recorded = true;
            Some(attribution)
        }
    }

    /// Prepare one missing-only, per-request write-through cohort: the prompt
    /// blocks computed so far, or for SimpleCPU every block computed before
    /// this pass (vLLM stores a step's blocks when the next step is scheduled).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe_completed_blocks(
        &mut self,
        uuid: Uuid,
        request: &mut VllmHostRequestState,
        lease: &BlockRequestLease,
        computed_tokens: usize,
        prompt_tokens: usize,
        block_size: usize,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) {
        let end = if self.connector == G2Connector::SimpleCpuV032 {
            let confirmed_blocks =
                std::mem::replace(&mut request.confirmed_tokens, computed_tokens) / block_size;
            request.extend_keys(lease, confirmed_blocks);
            confirmed_blocks.min(request.prompt_keys.len())
        } else {
            (computed_tokens.min(prompt_tokens) / block_size).min(request.prompt_keys.len())
        };
        if end <= request.next_store_block {
            return;
        }
        let start = request.next_store_block;
        let mut tier = self.host.lock();
        if self.connector == G2Connector::OffloadingV032 {
            // `prepare_store` records the blocks it is offered, except those
            // still being stored, even when the store fails.
            request.accessed_blocks.extend((start..end).filter(|index| {
                !matches!(
                    tier.lookup(request.prompt_keys[*index]),
                    Lookup::Pending { .. }
                )
            }));
        }
        let missing_indices = (start..end)
            .filter(|index| matches!(tier.lookup(request.prompt_keys[*index]), Lookup::Miss))
            .collect::<Vec<_>>();
        if missing_indices.is_empty() {
            request.next_store_block = end;
            return;
        }
        let Some(snapshot) =
            kv_manager.snapshot_native_store_sources(uuid, lease, &missing_indices)
        else {
            return;
        };
        assert_eq!(snapshot.len(), missing_indices.len());
        assert!(
            snapshot
                .sequence_hashes()
                .zip(missing_indices.iter().copied())
                .all(|(hash, index)| HostBlockKey::new(hash) == request.prompt_keys[index])
        );
        match tier.prepare_store(
            self.host.id(),
            uuid,
            &request.prompt_keys[start..end],
            request.prompt_meta.as_deref().map(|meta| &meta[start..end]),
            now_ms,
        ) {
            StoreOutcome::AlreadyPresent => request.next_store_block = end,
            StoreOutcome::RetryCapacity { .. } => {}
            StoreOutcome::Prepared {
                transfer_id,
                stored_blocks,
            } => {
                assert_eq!(stored_blocks, snapshot.len());
                kv_manager.attach_native_store_source_dependency(
                    uuid,
                    lease,
                    snapshot,
                    source_dependency(transfer_id),
                );
                if let Some(observer) = &self.observer {
                    let mappings = missing_indices
                        .iter()
                        .map(|index| HostStoreBlockMapping {
                            block: request.prompt_keys[*index],
                            logical_block_index: *index,
                        })
                        .collect::<Vec<_>>();
                    observer.record(HostOffloadObservation {
                        request_id: uuid,
                        event: HostOffloadObservationData::StoreBlockMappings {
                            at_ms: now_ms,
                            transfer_id,
                            mappings: &mappings,
                        },
                    });
                }
                request.latest_store = Some(transfer_id);
                request.next_store_block = end;
                // Only vLLM 0.24 touches the prompt after a store.
                if self.connector == G2Connector::OffloadingV024 {
                    for key in request.prompt_keys.iter().rev().copied() {
                        tier.touch(key);
                    }
                }
            }
        }
    }

    /// Offer a state G1 published this pass, as the scheduler's boundary
    /// hand-off does. SimpleCPU stores the missing slots together and drops
    /// the hand-off when the pool is full; `offloading_v0_32` stores each slot
    /// as its own job, and only inside the prompt.
    pub(super) fn store_state_handoff(
        &mut self,
        handoff: &StateHandoff,
        request: &mut VllmHostRequestState,
        prompt_tokens: usize,
        block_size: usize,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) {
        let hybrid = self.hybrid.expect("only a hybrid G1 hands off states");
        // A state retracted with its pass (a same-pass preemption) is gone.
        if !kv_manager.state_handoff_is_live(handoff) {
            return;
        }
        let keys = hybrid.keys(HostBlockKey::new(handoff.hash));
        let mut tier = self.host.lock();
        if self.connector == G2Connector::SimpleCpuV032 {
            let missing = (0..keys.len())
                .filter(|slot| tier.lookup(keys[*slot]) == Lookup::Miss)
                .collect::<Vec<_>>();
            if missing.is_empty() {
                return;
            }
            if let StoreOutcome::Prepared { transfer_id, .. } =
                tier.prepare_store(self.host.id(), handoff.owner, &keys, None, now_ms)
            {
                assert!(kv_manager.attach_state_handoff_dependency(
                    handoff,
                    &missing,
                    source_dependency(transfer_id)
                ));
                request.latest_store = Some(transfer_id);
            }
            return;
        }
        if handoff.position > prompt_tokens {
            return;
        }
        let mut offered = false;
        for (slot, key) in keys.iter().enumerate() {
            match tier.lookup(*key) {
                Lookup::Pending { .. } => continue,
                Lookup::Hit => offered = true,
                Lookup::Miss => {
                    offered = true;
                    if let StoreOutcome::Prepared { transfer_id, .. } =
                        tier.prepare_store(self.host.id(), handoff.owner, &[*key], None, now_ms)
                    {
                        assert!(kv_manager.attach_state_handoff_dependency(
                            handoff,
                            &[slot],
                            source_dependency(transfer_id)
                        ));
                        request.latest_store = Some(transfer_id);
                    }
                }
            }
        }
        if offered {
            request
                .accessed_states
                .insert(handoff.position / block_size);
        }
    }

    /// The connector's work when a request leaves the scheduler, before its
    /// G1 blocks are released. SimpleCPU drops its lookup pin and stores the
    /// blocks computed since its last store (`request_finished_all_groups`).
    /// `offloading_v0_32` offers its remaining prompt blocks (`_build_store_jobs`
    /// takes `finished_req_ids`), then re-ranks every block the request stored,
    /// offered or loaded, its first block newest (`on_request_finished`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish_request(
        &mut self,
        uuid: Uuid,
        request: &mut VllmHostRequestState,
        lease: &BlockRequestLease,
        computed_tokens: usize,
        prompt_tokens: usize,
        block_size: usize,
        kv_manager: &mut G1Manager,
        now_ms: f64,
    ) {
        match self.connector {
            G2Connector::OffloadingV024 => {}
            G2Connector::OffloadingV032 => {
                self.observe_completed_blocks(
                    uuid,
                    request,
                    lease,
                    computed_tokens,
                    prompt_tokens,
                    block_size,
                    kv_manager,
                    now_ms,
                );
                // Head to tail by end position (`order_request_keys`); a
                // state's slots precede the block ending at its boundary, as
                // vLLM lists a hybrid model's Mamba groups first.
                let mut keys = request
                    .accessed_blocks
                    .iter()
                    .map(|&index| (index + 1, 1, request.prompt_keys[index]))
                    .collect::<Vec<_>>();
                if let Some(hybrid) = self.hybrid {
                    for &boundary in &request.accessed_states {
                        let states = hybrid.keys(request.prompt_keys[boundary - 1]);
                        keys.extend(states.into_iter().map(|key| (boundary, 0, key)));
                    }
                }
                keys.sort_by_key(|&(end, group, _)| (end, group));
                let mut tier = self.host.lock();
                for (_, _, key) in keys.into_iter().rev() {
                    tier.touch(key);
                }
            }
            G2Connector::SimpleCpuV032 => {
                if let Some(pinned) = request.lookup_pin.take() {
                    self.host.lock().unpin_external(&pinned);
                }
                // Everything computed is confirmed once the request leaves.
                request.confirmed_tokens = computed_tokens;
                self.observe_completed_blocks(
                    uuid,
                    request,
                    lease,
                    computed_tokens,
                    prompt_tokens,
                    block_size,
                    kv_manager,
                    now_ms,
                );
            }
        }
    }

    /// Fence and authorize newly acquired G1 capacity whose prior owner is
    /// still being copied to host by a private lane.
    pub(super) fn fence_allocation(
        &mut self,
        uuid: Uuid,
        lease: &mut BlockRequestLease,
        dependencies: &[SourceReuseDependency],
        kv_manager: &mut G1Manager,
    ) {
        if dependencies.is_empty() {
            return;
        }
        let tier = self.host.lock();
        for dependency in dependencies {
            let deadline = tier
                .transfer_deadline(transfer_id(*dependency))
                .expect("pending dependency must retain a submitted D2H");
            self.compute_not_before_ms = self.compute_not_before_ms.max(deadline);
        }
        kv_manager.authorize_native_compute_after_dependencies(uuid, lease, dependencies);
    }

    /// vLLM flushes this request's prepared stores before releasing its G1
    /// capacity, and SimpleCPU synchronizes every transfer in flight
    /// (`handle_preemptions`). A shared pool instead holds the capacity until
    /// each copy completes.
    pub(super) fn preempt_request(&mut self, request: &mut VllmHostRequestState) {
        request.restore_rounds.reset();
        let deadline = {
            let tier = self.host.lock();
            if self.connector == G2Connector::SimpleCpuV032 {
                tier.last_deadline(self.host.id())
            } else {
                request
                    .latest_store
                    .and_then(|transfer_id| tier.transfer_deadline(transfer_id))
            }
        };
        if let Some(deadline) = deadline {
            self.compute_not_before_ms = self.compute_not_before_ms.max(deadline);
        }
    }

    pub(super) fn cancel_request(
        &mut self,
        request: &mut VllmHostRequestState,
        kv_manager: &mut G1Manager,
        mutation_now_ms: f64,
        observed_at_ms: f64,
    ) -> bool {
        let detached = self.g3.as_mut().is_some_and(|g3| g3.release(request.owner))
            | self
                .kvcr
                .as_mut()
                .is_some_and(|kvcr| kvcr.release(request.owner));
        let Some(load) = request.load.take() else {
            return detached;
        };
        let LoadState::Loading(load) = load else {
            return false;
        };
        if self.tracks_loads() {
            for key in &load.keys {
                assert_eq!(self.load_by_key.remove(key), Some(load.transfer_id));
            }
        }
        assert!(self.host.lock().cancel_load(
            self.host.id(),
            load.transfer_id,
            mutation_now_ms,
            observed_at_ms
        ));
        kv_manager.cancel_destination(load.reservation);
        true
    }

    pub(super) fn compute_not_before_ms(&self, now_ms: f64) -> f64 {
        self.compute_not_before_ms.max(now_ms)
    }

    pub(super) fn next_deadline(&self) -> Option<f64> {
        self.host
            .lock()
            .next_deadline(self.host.id())
            .into_iter()
            .chain(self.g3.as_ref().and_then(|g3| g3.next_deadline()))
            .chain(self.kvcr.as_ref().and_then(|kvcr| kvcr.next_deadline()))
            .min_by(f64::total_cmp)
    }

    pub(super) fn current_time_ms(&self) -> f64 {
        self.host.lock().current_time_ms()
    }

    /// Stores a request prepared when it left between passes. Only the 0.32
    /// connectors store at request finish. vLLM builds those stores in the
    /// next step, and `Scheduler.has_requests` keeps the engine stepping until
    /// it has, so an engine boundary submits them.
    pub(super) fn has_unsubmitted_stores(&self) -> bool {
        self.connector != G2Connector::OffloadingV024
            && self.host.lock().has_prepared_stores(self.host.id())
    }

    pub(super) fn has_work(&self) -> bool {
        self.host.lock().has_pending_work(self.host.id())
            || self.g3.as_ref().is_some_and(|g3| g3.has_work())
            || self.kvcr.as_ref().is_some_and(|kvcr| kvcr.has_work())
    }
}

fn source_dependency(transfer_id: TransferId) -> SourceReuseDependency {
    SourceReuseDependency::from_adapter_id(transfer_id.get())
}

fn transfer_id(dependency: SourceReuseDependency) -> TransferId {
    TransferId::new(dependency.adapter_id())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::engine::common::protocols::{DirectRequest, KvEventPublishers, MockEngineArgs};
    use crate::engine::host_offload::C;
    use crate::engine::kv_manager::NativeAllocation;
    use crate::engine::scheduler::vllm::core::VllmCore;
    use crate::engine::trace::TraceCollector;
    use crate::engine::{HostOffloadObservation, HostOffloadObservationData};

    fn adapter(capacity_blocks: usize) -> VllmHostOffloadAdapter {
        VllmHostOffloadAdapter::new(
            &NativeHostOffloadConfig::new(capacity_blocks).with_bandwidths(1.0, 1.0),
            4,
            250_000,
            None,
            false,
        )
        .unwrap()
    }

    fn request(
        owner: Uuid,
        tokens: Vec<u32>,
    ) -> (RequestSequence, BlockRequestLease, VllmHostRequestState) {
        let (sequence, identities) =
            RequestSequence::new(owner, tokens, 0, 0, 4, true, false, false, None);
        let lease = BlockRequestLease::new(owner, identities);
        let host = VllmHostRequestState::new(&sequence, &lease, 4, false, None);
        (sequence, lease, host)
    }

    fn seed_host(adapter: &mut VllmHostOffloadAdapter, key: HostBlockKey) -> f64 {
        let StoreOutcome::Prepared { transfer_id, .. } =
            adapter
                .host
                .lock()
                .prepare_store(C, Uuid::nil(), &[key], None, 0.0)
        else {
            panic!("host seed was prechecked as absent")
        };
        assert_eq!(adapter.host.lock().submit_prepared_stores(C, 0.0), 1);
        let deadline = adapter.host.lock().transfer_deadline(transfer_id).unwrap();
        assert!(matches!(
            adapter.host.lock().tick(C, deadline).0.as_slice(),
            [CompletedTransfer::Store { .. }]
        ));
        deadline
    }

    fn raw_cost(
        manager: &G1Manager,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
    ) -> PrefillCost {
        manager.get_native_prefill_cost(sequence, lease)
    }

    /// Compute one fresh block for `id`; the allocation must not wait on a
    /// source-reuse dependency.
    fn computed(
        manager: &mut G1Manager,
        id: Uuid,
        tokens: Vec<u32>,
    ) -> (RequestSequence, BlockRequestLease, VllmHostRequestState) {
        let (mut sequence, mut lease, host) = request(id, tokens);
        assert!(matches!(
            manager.allocate_native(id, &mut lease, 4, 0),
            NativeAllocation::Ready { dependencies, .. } if dependencies.is_empty()
        ));
        manager.finalize_native_computed_prefix(id, 0, 4, &mut sequence, &mut lease);
        (sequence, lease, host)
    }

    /// [`computed`], then offer the completed block to G2 at `now`.
    fn stored(
        adapter: &mut VllmHostOffloadAdapter,
        manager: &mut G1Manager,
        id: Uuid,
        tokens: Vec<u32>,
        now: f64,
    ) -> (RequestSequence, BlockRequestLease, VllmHostRequestState) {
        let (sequence, lease, mut host) = computed(manager, id, tokens);
        adapter.observe_completed_blocks(id, &mut host, &lease, 4, 4, 4, manager, now);
        (sequence, lease, host)
    }

    #[derive(Default)]
    struct EvictionCapture(Mutex<Vec<HostBlockKey>>);

    impl EvictionCapture {
        fn take(&self) -> Vec<HostBlockKey> {
            std::mem::take(&mut *self.0.lock().expect("eviction capture poisoned"))
        }
    }

    impl HostOffloadObserver for EvictionCapture {
        fn record(&self, observation: HostOffloadObservation<'_>) {
            if let HostOffloadObservationData::Evicted { block, .. } = observation.event {
                self.0
                    .lock()
                    .expect("eviction capture poisoned")
                    .push(block);
            }
        }
    }

    #[test]
    fn exhausted_pass_budget_does_not_touch_a_waiting_requests_host_prefix() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(64)
            .block_size(4)
            .max_num_seqs(Some(3))
            .max_num_batched_tokens(Some(4))
            .enable_chunked_prefill(true)
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(
                NativeHostOffloadConfig::new(4).with_bandwidths(0.0, 0.0),
            ))
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new(args);
        let evictions = Arc::new(EvictionCapture::default());
        core.set_host_offload_observer(evictions.clone());
        let mut collector = TraceCollector::default();
        let mut now_ms = 0.0;

        let mut complete_request =
            |core: &mut VllmCore, uuid: Uuid, tokens: Vec<u32>, now_ms: &mut f64| {
                core.receive(DirectRequest {
                    tokens,
                    max_output_tokens: 0,
                    uuid: Some(uuid),
                    arrival_timestamp_ms: Some(*now_ms),
                    ..Default::default()
                });
                for _ in 0..4 {
                    let pass = core.execute_pass(&mut collector, *now_ms);
                    *now_ms = pass.end_ms;
                    core.complete_engine_boundary(*now_ms);
                    if core.is_empty() {
                        return;
                    }
                }
                panic!("seed request did not complete");
            };

        // The two-block lineage supplies a non-prefix G2 candidate `a`: after
        // its first block is evicted, connector lookup still touches `a` before
        // discovering the prefix miss.
        let shared_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (_, _, shared_host) = request(Uuid::from_u128(20), shared_tokens.clone());
        let [x, a] = shared_host.prompt_keys.as_slice() else {
            panic!("shared request must have two prompt blocks")
        };
        let x = *x;
        let a = *a;
        complete_request(
            &mut core,
            Uuid::from_u128(20),
            shared_tokens.clone(),
            &mut now_ms,
        );

        // Fill the remainder of G2 in a known LRU order: X, A, B, D.
        complete_request(
            &mut core,
            Uuid::from_u128(21),
            vec![9, 10, 11, 12],
            &mut now_ms,
        );
        complete_request(
            &mut core,
            Uuid::from_u128(22),
            vec![13, 14, 15, 16],
            &mut now_ms,
        );
        assert!(evictions.take().is_empty());

        // The producer's first chunk stores C0 and evicts X, leaving A as LRU.
        let producer = Uuid::from_u128(23);
        core.receive(DirectRequest {
            tokens: vec![21, 22, 23, 24, 25, 26, 27, 28],
            max_output_tokens: 0,
            uuid: Some(producer),
            arrival_timestamp_ms: Some(now_ms),
            ..Default::default()
        });
        let first_chunk = core.execute_pass(&mut collector, now_ms);
        now_ms = first_chunk.end_ms;
        core.complete_engine_boundary(now_ms);
        assert_eq!(evictions.take(), vec![x]);
        assert!(!core.is_empty());

        // The producer is running and consumes the complete four-token budget.
        // vLLM must not query the waiting request afterward. If it does, the
        // lookup touches A despite X missing, promotes A to MRU, and the
        // producer's C1 store evicts B instead of A.
        core.receive(DirectRequest {
            tokens: shared_tokens,
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(24)),
            arrival_timestamp_ms: Some(now_ms),
            ..Default::default()
        });
        core.execute_pass(&mut collector, now_ms);

        assert_eq!(evictions.take(), vec![a]);
    }

    #[test]
    fn h2d_reserves_real_g1_capacity_and_cancel_releases_it() {
        let mut adapter = adapter(2);
        let destination_id = Uuid::from_u128(1);
        let (destination_sequence, destination_lease, mut destination_host) =
            request(destination_id, vec![1, 2, 3, 4]);
        let key = destination_host.prompt_keys[0];
        let now_ms = seed_host(&mut adapter, key);

        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let source_id = Uuid::from_u128(2);
        let (_, source_lease, _) = computed(&mut manager, source_id, vec![5, 6, 7, 8]);

        let HostLookup::Hit(hit) = adapter.lookup(
            &mut destination_host,
            &destination_sequence,
            &raw_cost(&manager, &destination_sequence, &destination_lease),
            4,
            adapter.current_time_ms(),
        ) else {
            panic!("seeded host block must hit")
        };
        assert!(matches!(
            adapter.start_load(
                destination_id,
                &mut destination_host,
                &destination_lease,
                hit,
                &mut manager,
                now_ms,
            ),
            StartLoad::CapacityBlocked
        ));

        manager.finish_native(source_id, source_lease);
        let HostLookup::Hit(hit) = adapter.lookup(
            &mut destination_host,
            &destination_sequence,
            &raw_cost(&manager, &destination_sequence, &destination_lease),
            4,
            adapter.current_time_ms(),
        ) else {
            panic!("host block must remain resident")
        };
        assert!(matches!(
            adapter.start_load(
                destination_id,
                &mut destination_host,
                &destination_lease,
                hit,
                &mut manager,
                now_ms,
            ),
            StartLoad::Queued
        ));
        let follower_id = Uuid::from_u128(8);
        let (follower_sequence, follower_lease, mut follower_host) =
            request(follower_id, vec![1, 2, 3, 4]);
        assert!(matches!(
            adapter.lookup(
                &mut follower_host,
                &follower_sequence,
                &raw_cost(&manager, &follower_sequence, &follower_lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Deferred
        ));
        assert_eq!(manager.num_active_blocks(), 1);
        assert!(adapter.cancel_request(&mut destination_host, &mut manager, now_ms, now_ms));
        assert_eq!(manager.num_active_blocks(), 0);
        // Cancellation leaves no due work that would reject a same-time command.
        assert_eq!(adapter.next_deadline(), None);
        assert!(adapter.host.lock().is_resident(key));
        assert!(matches!(
            adapter.lookup(
                &mut follower_host,
                &follower_sequence,
                &raw_cost(&manager, &follower_sequence, &follower_lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Hit(_)
        ));
    }

    #[test]
    fn completed_h2d_activates_the_reserved_request_lease_as_a_real_hit() {
        let mut adapter = adapter(2);
        let owner = Uuid::from_u128(3);
        let (sequence, mut lease, mut host) = request(owner, vec![1, 2, 3, 4]);
        let now_ms = seed_host(&mut adapter, host.prompt_keys[0]);
        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let HostLookup::Hit(hit) = adapter.lookup(
            &mut host,
            &sequence,
            &raw_cost(&manager, &sequence, &lease),
            4,
            adapter.current_time_ms(),
        ) else {
            panic!("seeded host block must hit")
        };
        assert!(matches!(
            adapter.start_load(owner, &mut host, &lease, hit, &mut manager, now_ms),
            StartLoad::Queued
        ));
        let transfer_id = match &host.load {
            Some(LoadState::Loading(load)) => load.transfer_id,
            _ => panic!("request must be loading"),
        };
        let deadline = adapter.host.lock().transfer_deadline(transfer_id).unwrap();
        let completed = adapter.advance(&mut manager, deadline).completed_loads;
        assert_eq!(completed.len(), 1);
        adapter.activate_completed_load(
            owner,
            transfer_id,
            &mut host,
            &sequence,
            &mut lease,
            &mut manager,
        );
        assert_eq!(raw_cost(&manager, &sequence, &lease).cached_tokens, 4);
        assert_eq!(lease.resident_block_count(), 1);
        assert_eq!(adapter.host.lock().resident_snapshot().len(), 1);
    }

    #[test]
    fn reused_store_source_fences_but_does_not_pin_g1_capacity() {
        let mut adapter = adapter(2);
        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let source_id = Uuid::from_u128(6);
        let (source_sequence, source_lease, mut source_host) =
            stored(&mut adapter, &mut manager, source_id, vec![1, 2, 3, 4], 0.0);
        let store_id = source_host.latest_store.unwrap();
        assert!(manager.is_native_source_dependency_pending(source_dependency(store_id)));
        adapter.complete_engine_boundary(&mut manager, 0.0);
        assert_eq!(adapter.host.lock().transfer_deadline(store_id), Some(1.0));
        assert!(matches!(
            adapter.lookup(
                &mut source_host,
                &source_sequence,
                &raw_cost(&manager, &source_sequence, &source_lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Deferred
        ));

        manager.finish_native(source_id, source_lease);
        assert_eq!(manager.num_active_blocks(), 0);
        let destination_id = Uuid::from_u128(7);
        let (mut destination_sequence, mut destination_lease, _) =
            request(destination_id, vec![5, 6, 7, 8]);
        let allocation = manager.allocate_native(destination_id, &mut destination_lease, 4, 0);
        let NativeAllocation::Ready { dependencies, .. } = allocation else {
            panic!("released source capacity must be reusable")
        };
        assert_eq!(dependencies, vec![source_dependency(store_id)]);
        adapter.fence_allocation(
            destination_id,
            &mut destination_lease,
            &dependencies,
            &mut manager,
        );
        manager.finalize_native_computed_prefix(
            destination_id,
            0,
            4,
            &mut destination_sequence,
            &mut destination_lease,
        );
        assert_eq!(adapter.compute_not_before_ms(0.0), 1.0);
        adapter.advance(&mut manager, 1.0);
        assert!(!manager.is_native_source_dependency_pending(source_dependency(store_id)));
    }

    #[test]
    fn shared_store_holds_its_g1_source_until_the_actual_d2h_completes() {
        use crate::engine::host_offload::{G2Binding, G2Registry};
        let registry: Arc<G2Registry> = Arc::default();
        let join = || {
            let mut config = NativeHostOffloadConfig::new(4)
                .with_bandwidths(1.0, 1.0)
                .cluster_shared("tp1");
            config.shared_d2h_bandwidth_gbps = 1.0;
            let binding = G2Binding {
                registry: Arc::clone(&registry),
                tensor_parallel_size: 1,
            };
            VllmHostOffloadAdapter::new(&config, 4, 250_000, Some(&binding), false).unwrap()
        };
        let (mut adapter, mut peer) = (join(), join());
        let mut manager = G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        manager.hold_native_store_sources();
        let source_id = Uuid::from_u128(6);
        let (_, source_lease, source_host) =
            stored(&mut adapter, &mut manager, source_id, vec![1, 2, 3, 4], 0.0);
        let store = source_dependency(source_host.latest_store.unwrap());
        adapter.complete_engine_boundary(&mut manager, 0.0);
        // Alone on the 1 GB/s pool the 1 MB D2H would finish at 1 ms; a peer
        // store joining later halves its share and moves completion to 1.5 ms.
        assert_eq!(adapter.next_deadline(), Some(1.0));
        let mut peer_manager =
            G1Manager::new_with_caching(1, 4, KvEventPublishers::default(), 0, true);
        let _peer = stored(
            &mut peer,
            &mut peer_manager,
            Uuid::from_u128(7),
            vec![5, 6, 7, 8],
            0.5,
        );
        peer.complete_engine_boundary(&mut peer_manager, 0.5);

        manager.finish_native(source_id, source_lease);
        assert_eq!(manager.num_active_blocks(), 1, "the D2H still holds G1");
        let destination_id = Uuid::from_u128(8);
        let (_, mut destination_lease, _) = request(destination_id, vec![9, 10, 11, 12]);
        for (now, pending) in [(1.0, true), (1.5, false)] {
            assert!(matches!(
                manager.allocate_native(destination_id, &mut destination_lease, 4, 0),
                NativeAllocation::CapacityExhausted
            ));
            adapter.advance(&mut manager, now);
            assert_eq!(
                manager.is_native_source_dependency_pending(store),
                pending,
                "{now}"
            );
        }
        // The completed copy returned the block, so its next owner neither
        // inherits a dependency nor waits for a projected fence.
        assert!(matches!(
            manager.allocate_native(destination_id, &mut destination_lease, 4, 0),
            NativeAllocation::Ready { dependencies, .. } if dependencies.is_empty()
        ));
        assert_eq!(adapter.compute_not_before_ms(1.5), 1.5);
    }

    #[test]
    fn shared_residency_reaches_subscribed_ranks_as_host_pinned_events() {
        use crate::engine::host_offload::{G2Binding, G2Registry};
        use crate::engine::{KvBlock, KvEvent, KvEventData, KvEventTier, StoredBlocks};
        let mut host = NativeHostOffloadConfig::new(4)
            .with_bandwidths(0.0, 0.0)
            .cluster_shared("tp1");
        (
            host.shared_d2h_bandwidth_gbps,
            host.shared_h2d_bandwidth_gbps,
        ) = (0.0, 0.0);
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(16)
            .block_size(4)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(host))
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // The router identity a subscriber receives must not depend on whether
        // the rank that stored the blocks publishes events itself.
        let (_, identities) = RequestSequence::new(
            Uuid::nil(),
            tokens.clone(),
            0,
            0,
            4,
            true,
            true,
            false,
            None,
        );
        let lease = BlockRequestLease::new(Uuid::nil(), identities);
        let block = |index: usize| KvBlock {
            block_hash: lease.sequence_hash(index).unwrap(),
            tokens_hash: lease.local_hash(index).unwrap(),
            token_ids: None,
        };
        let expected = KvEvent {
            event_id: 0,
            dp_rank: 0,
            data: KvEventData::Stored(StoredBlocks {
                parent_hash: None,
                start_position: Some(0),
                blocks: vec![block(0), block(1)],
                session_id: Some("producer-session".into()),
            }),
            tier: KvEventTier::HostPinned,
        };
        for producer_publishes in [true, false] {
            let binding = G2Binding {
                registry: Arc::<G2Registry>::default(),
                tensor_parallel_size: 1,
            };
            let rank = |id, capture| {
                VllmCore::new_with_worker_rank(args.clone(), id, 0, id, capture, Some(&binding))
                    .unwrap()
            };
            let (mut producer, mut peer) = (rank(0, producer_publishes), rank(1, true));
            producer.receive(DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(1)),
                arrival_timestamp_ms: Some(0.0),
                session_id: Some("producer-session".into()),
                ..Default::default()
            });
            let mut device = Vec::new();
            while !producer.is_empty() {
                let pass = producer.execute_pass(&mut TraceCollector::default(), 0.0);
                producer.complete_engine_boundary(pass.end_ms);
                device.extend(pass.kv_events);
            }
            if producer_publishes {
                assert!(device.iter().all(|event| event.tier == KvEventTier::Device));
                let KvEventData::Stored(stored) = &device[0].data else {
                    panic!("the first device event stores the prompt");
                };
                assert_eq!(stored.blocks[..2], [block(0), block(1)]);
            } else {
                assert!(device.is_empty());
            }

            peer.process_internal_work(0.0);
            assert_eq!(peer.drain_kv_events(), vec![expected.clone()]);
        }
        // Device events keep their JSON shape; a missing tier reads as device.
        let mut json = serde_json::to_value(&expected).unwrap();
        assert_eq!(json["tier"], "host_pinned");
        json.as_object_mut().unwrap().remove("tier");
        let device_event: KvEvent = serde_json::from_value(json).unwrap();
        assert_eq!(device_event.tier, KvEventTier::Device);
        assert!(
            serde_json::to_value(&device_event)
                .unwrap()
                .get("tier")
                .is_none()
        );
    }

    /// Serve each arrival wave to completion and drain its host transfers
    /// before the next wave. Returns every KV event per wave and each
    /// admission's host-reused prompt tokens.
    fn served_kv_events(
        host: Option<NativeHostOffloadConfig>,
        waves: &[&[&[u32]]],
    ) -> (Vec<Vec<crate::engine::KvEvent>>, Vec<usize>) {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(4)
            .block_size(4)
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(host)
            .speedup_ratio(0.0)
            .build()
            .unwrap();
        let mut core = VllmCore::new_with_worker_rank(args, 0, 0, 0, true, None).unwrap();
        let (mut now_ms, mut next_uuid) = (0.0, 0u128);
        let mut host_reused = Vec::new();
        let events = waves
            .iter()
            .map(|wave| {
                for prompt in *wave {
                    next_uuid += 1;
                    core.receive(DirectRequest {
                        tokens: prompt.to_vec(),
                        max_output_tokens: 1,
                        uuid: Some(Uuid::from_u128(next_uuid)),
                        arrival_timestamp_ms: Some(now_ms),
                        ..Default::default()
                    });
                }
                let mut events = Vec::new();
                for _ in 0..8 {
                    if core.is_ready() {
                        let pass = core.execute_pass(&mut TraceCollector::default(), now_ms);
                        now_ms = pass.end_ms;
                        core.complete_engine_boundary(now_ms);
                        events.extend(pass.kv_events);
                        host_reused.extend(pass.admissions.iter().map(|admission| {
                            admission
                                .cache_tier_attribution
                                .map_or(0, |tiers| tiers.host_reused_input_tokens)
                        }));
                    } else if let Some(deadline) = core.next_internal_deadline_ms() {
                        now_ms = now_ms.max(deadline);
                        core.process_internal_work(now_ms);
                    } else {
                        assert!(core.is_empty());
                        return events;
                    }
                    events.extend(core.drain_kv_events());
                }
                panic!("wave did not drain")
            })
            .collect();
        (events, host_reused)
    }

    #[test]
    fn per_rank_g2_publishes_host_evictions() {
        use crate::engine::{KvEventData, KvEventTier};
        let (a, x, y) = ([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25]);
        // A two-block per-rank G2 holds A and X; storing Y evicts A, the LRU block.
        let waves: [&[&[u32]]; 3] = [&[&a], &[&x], &[&y]];
        let (events, _) = served_kv_events(
            Some(NativeHostOffloadConfig::new(2).with_bandwidths(1.0, 1.0)),
            &waves,
        );
        let block = |tokens: &[u32]| {
            let (_, _, host) = request(Uuid::nil(), tokens.to_vec());
            host.prompt_keys[0].sequence_hash()
        };
        let (a0, x0, y0) = (block(&a), block(&x), block(&y));
        let host = events
            .iter()
            .flatten()
            .filter(|event| event.tier == KvEventTier::HostPinned)
            .map(|event| match &event.data {
                KvEventData::Stored(stored) => (
                    "stored",
                    stored.blocks.iter().map(|block| block.block_hash).collect(),
                ),
                KvEventData::Removed { block_hashes } => ("removed", block_hashes.clone()),
            })
            .collect::<Vec<(&str, Vec<u64>)>>();
        assert_eq!(
            host,
            vec![
                ("stored", vec![a0]),
                ("stored", vec![x0]),
                ("removed", vec![a0]),
                ("stored", vec![y0]),
            ]
        );
    }

    #[test]
    fn per_rank_g2_publishes_host_residency_without_changing_device_residency() {
        use crate::engine::{KvEvent, KvEventData, KvEventTier};
        let (a, x, y) = ([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25]);
        // X and Y together evict A's only full block from the four-block G1
        // and then free their partial blocks, so A's return either recomputes
        // or restores that block into free capacity without another eviction.
        let waves: [&[&[u32]]; 3] = [&[&a], &[&x, &y], &[&a]];
        let (without_g2, g1_host_reused) = served_kv_events(None, &waves);
        let (per_rank_g2, host_reused) = served_kv_events(
            Some(NativeHostOffloadConfig::new(4).with_bandwidths(1.0, 1.0)),
            &waves,
        );
        assert_eq!(
            (g1_host_reused, host_reused),
            (vec![0; 4], vec![0, 0, 0, 4]),
            "only the per-rank-G2 run restores A's block by H2D"
        );

        let block = |tokens: &[u32]| {
            let (_, _, host) = request(Uuid::nil(), tokens.to_vec());
            host.prompt_keys[0].sequence_hash()
        };
        let (a0, x0, y0) = (block(&a), block(&x), block(&y));
        let residency = |events: &[Vec<KvEvent>], tier: KvEventTier| {
            events
                .iter()
                .flatten()
                .filter(|event| event.tier == tier)
                .map(|event| match &event.data {
                    KvEventData::Stored(stored) => (
                        "stored",
                        stored.blocks.iter().map(|block| block.block_hash).collect(),
                    ),
                    KvEventData::Removed { block_hashes } => ("removed", block_hashes.clone()),
                })
                .collect::<Vec<(&str, Vec<u64>)>>()
        };
        // G1 residency is exactly the G2-off stream; event IDs are shared with
        // the interleaved HostPinned events, as in one vLLM KV event stream.
        let device_residency = vec![
            ("stored", vec![a0]),
            ("stored", vec![x0]),
            ("removed", vec![a0]),
            ("stored", vec![y0]),
            ("stored", vec![a0]),
        ];
        assert_eq!(
            residency(&without_g2, KvEventTier::Device),
            device_residency
        );
        assert_eq!(
            residency(&per_rank_g2, KvEventTier::Device),
            device_residency
        );
        assert!(residency(&without_g2, KvEventTier::HostPinned).is_empty());
        // Native vLLM's OffloadingConnector reports its per-rank CPU cache the
        // same way, so a KV router can credit A while it is host-only.
        assert_eq!(
            residency(&per_rank_g2, KvEventTier::HostPinned),
            vec![
                ("stored", vec![a0]),
                ("stored", vec![x0]),
                ("stored", vec![y0])
            ]
        );
        let ids = per_rank_g2
            .iter()
            .flatten()
            .map(|event| event.event_id)
            .collect::<Vec<_>>();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]), "{ids:?}");
    }

    #[test]
    fn mixed_g2_g3_prefix_stages_then_acquires_and_activates_real_g1() {
        use crate::engine::g3_offload::{Direction, G3Tier};
        use crate::engine::{G3OffloadConfig, G3Scope};
        let mut adapter = adapter(2);
        let owner = Uuid::from_u128(900);
        let (sequence, mut lease, mut host) = request(owner, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let first = host.prompt_keys[0];
        let second = host.prompt_keys[1];
        let now = seed_host(&mut adapter, first);
        let registry = G3Tier::new(
            G3OffloadConfig {
                scope: G3Scope::ClusterShared,
                num_g3_blocks: 2,
                latency_to_first_byte_ms: 0.0,
                read_bandwidth_gbps: 1.0,
                write_bandwidth_gbps: 0.0,
                shared_read_bandwidth_gbps: 1.0,
                shared_write_bandwidth_gbps: 0.0,
            },
            2,
            1_000_000,
        )
        .unwrap();
        registry
            .lock()
            .unwrap()
            .submit(0, Direction::Write, &[second], now)
            .unwrap();
        registry.lock().unwrap().take_completed(0, now);
        adapter.set_g3(registry.clone(), 1);
        let mut manager = G1Manager::new_with_caching(2, 4, KvEventPublishers::default(), 0, true);
        adapter.advance(&mut manager, now);
        assert!(matches!(
            adapter.lookup(
                &mut host,
                &sequence,
                &raw_cost(&manager, &sequence, &lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Deferred
        ));
        assert_eq!(manager.num_active_blocks(), 0, "G3 staging owns no G1");
        let staged_at = adapter.next_deadline().unwrap();
        assert!(adapter.advance(&mut manager, staged_at).completed_any);
        let HostLookup::Hit(hit) = adapter.lookup(
            &mut host,
            &sequence,
            &raw_cost(&manager, &sequence, &lease),
            4,
            adapter.current_time_ms(),
        ) else {
            panic!("mixed prefix must now hit G2");
        };
        assert_eq!(hit.keys, vec![first, second]);
        assert!(matches!(
            adapter.start_load(owner, &mut host, &lease, hit, &mut manager, staged_at),
            StartLoad::Queued
        ));
        assert_eq!(
            manager.num_active_blocks(),
            2,
            "H2D owns the real G1 reservation"
        );
        let completed_at = adapter.next_deadline().unwrap();
        assert_eq!(
            completed_at - staged_at,
            2.0,
            "H2D transfers both 1MB G2 blocks"
        );
        let completed = adapter.advance(&mut manager, completed_at).completed_loads;
        assert_eq!(completed.len(), 1);
        adapter.activate_completed_load(
            owner,
            completed[0].transfer_id,
            &mut host,
            &sequence,
            &mut lease,
            &mut manager,
        );
        assert_eq!(raw_cost(&manager, &sequence, &lease).cached_tokens, 8);
        let stats = registry.lock().unwrap().snapshot();
        assert_eq!(
            stats.read.completed_bytes, 1_000_000,
            "only the missing G2 block crosses G3"
        );
        assert_eq!(stats.cross_worker_read_blocks, 1);
    }

    #[test]
    fn independent_review_external_reservation_publishes_real_eviction() {
        let mut adapter = adapter(1);
        let a = HostBlockKey::new(1);
        let b = HostBlockKey::new(2);
        let now = seed_host(&mut adapter, a);
        let observed = Arc::new(EvictionCapture::default());
        adapter.set_observer(observed.clone());
        assert!(matches!(
            adapter
                .host
                .lock()
                .reserve_external(C, Uuid::nil(), &[b], None, now),
            StoreOutcome::Prepared { .. }
        ));
        assert_eq!(adapter.host.lock().lookup(a), Lookup::Miss);
        assert_eq!(
            observed.take(),
            vec![a],
            "actual G2 eviction must remain observable"
        );
    }

    #[test]
    fn requester_termination_preserves_promotion_without_activating_g1() {
        for observed_at_ms in [0.5, 1.0, 2.0] {
            for foreign_watermark in [None, Some(0.75), Some(3.0)] {
                check_g3_cancellation(observed_at_ms, foreign_watermark);
            }
        }
    }

    #[test]
    fn g3_promotion_into_per_rank_g2_publishes_host_residency_in_the_same_advance() {
        use crate::engine::g3_offload::{Direction, G3Tier};
        use crate::engine::scheduler::capture_kv_event_sink;
        use crate::engine::{G3OffloadConfig, G3Scope, KvEventData, KvEventTier};
        // A rank that publishes KV events subscribes to its per-rank G2.
        let mut adapter = VllmHostOffloadAdapter::new(
            &NativeHostOffloadConfig::new(2).with_bandwidths(1.0, 1.0),
            4,
            250_000,
            None,
            true,
        )
        .unwrap();
        let owner = Uuid::from_u128(902);
        let (sequence, identities) =
            RequestSequence::new(owner, vec![1, 2, 3, 4, 5], 0, 0, 4, true, true, false, None);
        let mut lease = BlockRequestLease::new(owner, identities);
        lease.session_id = Some("restoring-session".into());
        let mut host = VllmHostRequestState::new(&sequence, &lease, 4, true, None);
        let key = host.prompt_keys[0];
        let registry = G3Tier::new(
            G3OffloadConfig {
                scope: G3Scope::ClusterShared,
                num_g3_blocks: 4,
                latency_to_first_byte_ms: 0.0,
                read_bandwidth_gbps: 1.0,
                write_bandwidth_gbps: 0.0,
                shared_read_bandwidth_gbps: 1.0,
                shared_write_bandwidth_gbps: 0.0,
            },
            2,
            1_000_000,
        )
        .unwrap();
        registry
            .lock()
            .unwrap()
            .submit(0, Direction::Write, &[key], 0.0)
            .unwrap();
        registry.lock().unwrap().take_completed(0, 0.0);
        adapter.set_g3(registry, 1);
        let (buffer, sink) = capture_kv_event_sink();
        let mut manager =
            G1Manager::new_with_caching(4, 4, KvEventPublishers::new(Some(sink)), 0, true);
        adapter.advance(&mut manager, 0.0);
        assert!(matches!(
            adapter.lookup(
                &mut host,
                &sequence,
                &raw_cost(&manager, &sequence, &lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Deferred
        ));
        // The requester leaves; its detached G3 promotion still lands in G2
        // during the final advance, which leaves no later deadline.
        assert!(adapter.cancel_request(&mut host, &mut manager, 0.0, 0.5));
        buffer.drain();
        adapter.advance(&mut manager, 4.0);
        assert_eq!(adapter.host.lock().lookup(key), Lookup::Hit);
        assert!(adapter.next_deadline().is_none());
        let host_pinned = buffer
            .drain()
            .into_iter()
            .filter(|event| event.tier == KvEventTier::HostPinned)
            .map(|event| match event.data {
                KvEventData::Stored(stored) => {
                    assert_eq!(stored.session_id.as_deref(), Some("restoring-session"));
                    stored
                        .blocks
                        .iter()
                        .map(|block| block.block_hash)
                        .collect::<Vec<_>>()
                }
                KvEventData::Removed { .. } => panic!("promotion evicts nothing"),
            })
            .collect::<Vec<_>>();
        assert_eq!(host_pinned, vec![vec![key.sequence_hash()]]);
    }

    fn check_g3_cancellation(observed_at_ms: f64, foreign_watermark: Option<f64>) {
        use crate::engine::g3_offload::{Direction, G3Tier};
        use crate::engine::{G3OffloadConfig, G3Scope};
        let mut adapter = adapter(2);
        let owner = Uuid::from_u128(901);
        let (sequence, lease, mut host) = request(owner, vec![1, 2, 3, 4, 5]);
        let key = host.prompt_keys[0];
        let registry = G3Tier::new(
            G3OffloadConfig {
                scope: G3Scope::ClusterShared,
                num_g3_blocks: 4,
                latency_to_first_byte_ms: 0.0,
                read_bandwidth_gbps: 1.0,
                write_bandwidth_gbps: 0.0,
                shared_read_bandwidth_gbps: 1.0,
                shared_write_bandwidth_gbps: 0.0,
            },
            2,
            1_000_000,
        )
        .unwrap();
        registry
            .lock()
            .unwrap()
            .submit(0, Direction::Write, &[key], 0.0)
            .unwrap();
        registry.lock().unwrap().take_completed(0, 0.0);
        adapter.set_g3(registry.clone(), 1);
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        adapter.advance(&mut manager, 0.0);
        assert!(matches!(
            adapter.lookup(
                &mut host,
                &sequence,
                &raw_cost(&manager, &sequence, &lease),
                4,
                adapter.current_time_ms(),
            ),
            HostLookup::Deferred
        ));
        // Read completes at 1ms. The committed pass retains G2 mutation time
        // 0ms, independently of observation time or a foreign watermark.
        if let Some(now) = foreign_watermark {
            registry.lock().unwrap().advance(now);
        }
        assert!(adapter.cancel_request(&mut host, &mut manager, 0.0, observed_at_ms));
        assert_eq!(adapter.host.lock().current_time_ms(), 0.0);
        assert!(matches!(
            adapter.host.lock().lookup(key),
            Lookup::Pending { .. }
        ));
        assert_eq!(manager.num_active_blocks(), 0);
        assert_eq!(host.reserved_blocks(), 0);
        let completed = foreign_watermark.is_some_and(|t| t >= 1.0);
        let stats = registry.lock().unwrap().snapshot();
        assert_eq!(
            stats.read.completed_bytes,
            if completed { 1_000_000 } else { 0 }
        );
        assert_eq!(stats.read.completed_jobs, u64::from(completed));
        assert_eq!(stats.read.cancelled_jobs, 0);
        // The detached promotion may finish into G2, never into canceled G1.
        assert!(
            adapter
                .advance(&mut manager, 4.0)
                .completed_loads
                .is_empty()
        );
        assert_eq!(adapter.host.lock().lookup(key), Lookup::Hit);
        assert_eq!(manager.num_active_blocks(), 0);
        assert!(adapter.next_deadline().is_none());
    }

    /// Single-rank workers 0 and 1 with private G2s of `blocks` blocks, both
    /// registered with one KVCR registry.
    fn kvcr_workers(
        config: crate::engine::KvcrConfig,
        blocks: [usize; 2],
    ) -> (
        crate::engine::kvcr::SharedKvcrRegistry,
        [KvHintSource; 2],
        [VllmHostOffloadAdapter; 2],
    ) {
        let registry = crate::engine::kvcr::KvcrRegistry::new(config);
        let ranks = [0, 1].map(|worker_id| KvHintSource {
            worker_id,
            dp_rank: 0,
        });
        let adapters = [0, 1].map(|index| {
            let mut adapter = adapter(blocks[index]);
            let g2 = adapter.set_kvcr(Arc::clone(&registry), ranks[index]);
            registry
                .lock()
                .unwrap()
                .register_worker(ranks[index].worker_id, vec![g2], 1)
                .unwrap();
            adapter
        });
        (registry, ranks, adapters)
    }

    #[test]
    fn hinted_lookup_pulls_a_peer_prefix_and_counts_only_its_first_load() {
        let config = crate::engine::KvcrConfig {
            latency_to_first_byte_ms: 1.0,
            bandwidth_gbps: 1.0,
        };
        // Worker 1 pulls from worker 0.
        let (registry, ranks, [mut source, mut target]) = kvcr_workers(config, [4, 4]);
        let owner = Uuid::from_u128(910);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (sequence, lease, mut host) = request(owner, tokens.clone());
        for key in host.prompt_keys.clone() {
            seed_host(&mut source, key);
        }
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        let cost = raw_cost(&manager, &sequence, &lease);

        // Without a hint the target misses exactly as G2 alone does.
        assert!(matches!(
            target.lookup(&mut host, &sequence, &cost, 4, 0.0),
            HostLookup::Miss
        ));
        assert_eq!(target.retry_epoch(), None);
        host.kv_hint = Some(ranks[0]);
        assert!(matches!(
            target.lookup(&mut host, &sequence, &cost, 4, 0.0),
            HostLookup::Deferred
        ));
        assert!(target.retry_epoch().is_some());
        // Replay starts the pull once the instant's passes have run.
        registry.lock().unwrap().start_parked();
        // The deferred request wakes at the first byte (1 ms), then when two
        // 1 MB blocks at 1 GB/s have landed; both workers stay busy until then.
        assert!(source.has_work() && target.has_work());
        assert_eq!(target.next_deadline(), Some(1.0));
        assert!(!target.advance(&mut manager, 1.0).completed_any);
        assert_eq!(target.next_deadline(), Some(3.0));
        assert!(target.advance(&mut manager, 3.0).completed_any);
        assert_eq!(target.next_deadline(), None);
        assert!(!source.has_work() && !target.has_work());
        let HostLookup::Hit(hit) = target.lookup(&mut host, &sequence, &cost, 4, 3.0) else {
            panic!("the pulled prefix must hit G2");
        };
        assert_eq!(hit.keys, host.prompt_keys);
        assert!(matches!(
            target.start_load(owner, &mut host, &lease, hit, &mut manager, 3.0),
            StartLoad::Queued
        ));
        let loaded_blocks = || registry.lock().unwrap().stats().loaded_blocks;
        assert_eq!(loaded_blocks(), 2);

        // A second request loading the same blocks does not count them again.
        assert!(target.cancel_request(&mut host, &mut manager, 3.0, 3.0));
        let other = Uuid::from_u128(911);
        let (sequence, lease, mut host) = request(other, tokens);
        let cost = raw_cost(&manager, &sequence, &lease);
        let HostLookup::Hit(hit) = target.lookup(&mut host, &sequence, &cost, 4, 3.0) else {
            panic!("the pulled prefix stays in G2");
        };
        assert!(matches!(
            target.start_load(other, &mut host, &lease, hit, &mut manager, 3.0),
            StartLoad::Queued
        ));
        assert_eq!(loaded_blocks(), 2);
    }

    #[test]
    fn a_thrashing_hinted_prefix_stops_pulling_after_rounds_without_rank_progress() {
        let config = crate::engine::KvcrConfig {
            latency_to_first_byte_ms: 0.0,
            bandwidth_gbps: 0.0,
        };
        // A pull never evicts its own prefix, but the target's one-block G2
        // holds only one of two one-block prefixes, so each request's pull
        // evicts the block the other request pulled just before.
        let (registry, ranks, [mut source, mut target]) = kvcr_workers(config, [2, 1]);
        let mut requests =
            [(912, vec![1, 2, 3, 4]), (914, vec![5, 6, 7, 8])].map(|(id, tokens)| {
                let (sequence, lease, mut host) = request(Uuid::from_u128(id), tokens);
                for key in host.prompt_keys.clone() {
                    seed_host(&mut source, key);
                }
                host.kv_hint = Some(ranks[0]);
                (sequence, lease, host)
            });
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        let costs = requests
            .each_ref()
            .map(|(sequence, lease, _)| raw_cost(&manager, sequence, lease));
        // Rank progress before round 10 starts the count over.
        let rounds = 10 + MAX_RESTORE_ROUNDS_WITHOUT_PROGRESS;
        for round in 0..rounds {
            if round == 10 {
                target.note_rank_progress();
            }
            let now = round as f64;
            for ((sequence, _, host), cost) in requests.iter_mut().zip(&costs) {
                let lookup = target.lookup(host, sequence, cost, 4, now);
                assert!(matches!(lookup, HostLookup::Deferred), "round {round}");
                registry.lock().unwrap().start_parked();
                assert!(target.advance(&mut manager, now).completed_any);
            }
        }
        // Past the bound neither request pulls again: the first computes its
        // evicted block, the second loads the block its last pull left.
        let now = rounds as f64;
        let [first, second] = &mut requests;
        assert!(matches!(
            target.lookup(&mut first.2, &first.0, &costs[0], 4, now),
            HostLookup::Miss
        ));
        assert!(matches!(
            target.lookup(&mut second.2, &second.0, &costs[1], 4, now),
            HostLookup::Hit(_)
        ));
        let stats = registry.lock().unwrap().stats();
        assert_eq!(stats.bypassed, 2);
        assert_eq!(stats.pulls.success + stats.pulls.partial, 2 * rounds as u64);
    }

    #[test]
    fn a_pull_in_flight_defers_its_owner_without_a_lookup_or_lru_touch() {
        let config = crate::engine::KvcrConfig {
            latency_to_first_byte_ms: 1.0,
            bandwidth_gbps: 0.0,
        };
        let (registry, ranks, [mut source, mut target]) = kvcr_workers(config, [4, 4]);
        let owner = Uuid::from_u128(915);
        let (sequence, lease, mut host) = request(owner, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let [local, pulled] = [host.prompt_keys[0], host.prompt_keys[1]];
        let other = HostBlockKey::new(77);
        // `other` is newer than `local`; a lookup would move `local` behind it.
        let t = seed_host(&mut source, pulled)
            .max(seed_host(&mut target, local))
            .max(seed_host(&mut target, other));
        host.kv_hint = Some(ranks[0]);
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        let cost = raw_cost(&manager, &sequence, &lease);
        assert!(matches!(
            target.lookup(&mut host, &sequence, &cost, 4, t),
            HostLookup::Deferred
        ));
        registry.lock().unwrap().start_parked();
        assert!(target.kvcr.as_ref().unwrap().is_pulling(owner));
        target.host.lock().touch(other);
        let recency = |target: &VllmHostOffloadAdapter| target.host.lock().lru_order();
        let before = recency(&target);
        assert_eq!(before, [local, other]);
        let epoch = target.kvcr.as_ref().unwrap().epoch;
        for now in [t, t + 0.5] {
            assert!(matches!(
                target.lookup(&mut host, &sequence, &cost, 4, now),
                HostLookup::Deferred
            ));
        }
        assert_eq!(recency(&target), before, "no LRU touch while pulling");
        assert_eq!(target.kvcr.as_ref().unwrap().epoch, epoch, "no new stage");
        assert_eq!(registry.lock().unwrap().stats().pulls.success, 0);
        // The landing wakes the owner, whose next lookup hits the whole prefix.
        assert!(target.advance(&mut manager, t + 1.0).completed_any);
        assert!(!target.kvcr.as_ref().unwrap().is_pulling(owner));
        let HostLookup::Hit(hit) = target.lookup(&mut host, &sequence, &cost, 4, t + 1.0) else {
            panic!("the landed pull extends the local hit");
        };
        assert_eq!(hit.keys, vec![local, pulled]);
        let stats = registry.lock().unwrap().stats();
        assert_eq!((stats.pulls.success, stats.blocks), (1, 1));
    }

    #[test]
    fn a_load_that_must_retry_keeps_the_kvcr_origin_marks() {
        let config = crate::engine::KvcrConfig {
            latency_to_first_byte_ms: 0.0,
            bandwidth_gbps: 0.0,
        };
        let (registry, ranks, [mut source, mut target]) = kvcr_workers(config, [2, 2]);
        let owner = Uuid::from_u128(913);
        let (sequence, lease, mut host) = request(owner, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        for key in host.prompt_keys.clone() {
            seed_host(&mut source, key);
        }
        host.kv_hint = Some(ranks[0]);
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        let cost = raw_cost(&manager, &sequence, &lease);
        assert!(matches!(
            target.lookup(&mut host, &sequence, &cost, 4, 0.0),
            HostLookup::Deferred
        ));
        registry.lock().unwrap().start_parked();
        assert!(target.advance(&mut manager, 0.0).completed_any);
        let HostLookup::Hit(hit) = target.lookup(&mut host, &sequence, &cost, 4, 0.0) else {
            panic!("the pulled prefix must hit G2");
        };
        // Another reservation evicts the second block, which the lookup touched
        // least recently, before the H2D starts.
        let other = [HostBlockKey::new(99)];
        let evicting = target
            .host
            .lock()
            .reserve_external(C, Uuid::nil(), &other, None, 0.0);
        assert!(matches!(evicting, StoreOutcome::Prepared { .. }));
        assert!(matches!(
            target.start_load(owner, &mut host, &lease, hit, &mut manager, 0.0),
            StartLoad::Retry
        ));
        assert_eq!(registry.lock().unwrap().stats().loaded_blocks, 0);
        assert_eq!(target.host.lock().take_kvcr_origin(&host.prompt_keys), 1);
    }

    fn connector_adapter(connector: G2Connector, capacity_blocks: usize) -> VllmHostOffloadAdapter {
        VllmHostOffloadAdapter::new(
            &NativeHostOffloadConfig::new(capacity_blocks)
                .with_bandwidths(1.0, 1.0)
                .with_connector(connector),
            4,
            250_000,
            None,
            false,
        )
        .unwrap()
    }

    fn connector_args(connector: G2Connector) -> MockEngineArgs {
        MockEngineArgs::builder()
            .num_gpu_blocks(16)
            .block_size(4)
            .enable_prefix_caching(true)
            .kv_cache_bytes_per_token(Some(250_000))
            .native_host_offload(Some(
                NativeHostOffloadConfig::new(16)
                    .with_bandwidths(0.0, 0.0)
                    .with_connector(connector),
            ))
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    /// Hashes of the `HostPinned` blocks stored among `events`.
    fn host_stored(events: impl IntoIterator<Item = crate::engine::KvEvent>) -> Vec<u64> {
        use crate::engine::{KvEventData, KvEventTier};
        events
            .into_iter()
            .filter(|event| event.tier == KvEventTier::HostPinned)
            .flat_map(|event| match event.data {
                KvEventData::Stored(stored) => stored.blocks,
                KvEventData::Removed { .. } => Vec::new(),
            })
            .map(|block| block.block_hash)
            .collect()
    }

    /// The G2 blocks each pass stored while one request of `prompt` and
    /// `output` tokens ran to completion.
    fn host_stores_per_pass(
        connector: G2Connector,
        prompt: Vec<u32>,
        output: usize,
    ) -> Vec<Vec<u64>> {
        let mut core =
            VllmCore::new_with_worker_rank(connector_args(connector), 0, 0, 0, true, None).unwrap();
        core.receive(DirectRequest {
            tokens: prompt,
            max_output_tokens: output,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        });
        let mut now_ms = 0.0;
        let mut stores = Vec::new();
        while !core.is_empty() {
            let pass = core.execute_pass(&mut TraceCollector::default(), now_ms);
            now_ms = pass.end_ms;
            core.complete_engine_boundary(now_ms);
            stores.push(host_stored(
                pass.kv_events.into_iter().chain(core.drain_kv_events()),
            ));
        }
        stores
    }

    #[test]
    fn simple_cpu_stores_every_computed_block_one_pass_later() {
        // Eager SimpleCPU stores the blocks computed before the pass it is
        // scheduling, decode blocks included, and a finishing request's last
        // blocks at its finish (manager.py `_prepare_eager_store_specs`,
        // `_queue_finished_eager_store`). The offloading connector stores the
        // prompt as each pass computes it.
        let prompt = (1..=8).collect::<Vec<u32>>();
        let (_, _, host) = request(Uuid::nil(), prompt.clone());
        let prompt_blocks = host
            .prompt_keys
            .iter()
            .map(|key| key.sequence_hash())
            .collect::<Vec<_>>();
        // Nine passes compute the KV of tokens 0..16. The prompt blocks are
        // stored in pass 2; block 2 fills in pass 5 and is stored in pass 6;
        // block 3 fills in the last pass and is stored at the finish.
        let simple = host_stores_per_pass(G2Connector::SimpleCpuV032, prompt.clone(), 9);
        assert_eq!(
            simple.iter().map(Vec::len).collect::<Vec<_>>(),
            [0, 2, 0, 0, 0, 1, 0, 0, 1]
        );
        assert_eq!(simple[1], prompt_blocks);
        let offloading = host_stores_per_pass(G2Connector::OffloadingV032, prompt, 9);
        assert_eq!(offloading[0], prompt_blocks);
        assert_eq!(offloading.concat(), prompt_blocks);
    }

    #[test]
    fn a_hit_whose_store_is_in_flight_misses_for_simple_cpu_and_defers_offloading() {
        // SimpleCPU registers a block in its CPU pool only when the copy
        // completes; the offloading connector reports `HIT_PENDING` and defers
        // the request, as it does while the request's own store is in flight.
        for (connector, deferred) in [
            (G2Connector::SimpleCpuV032, false),
            (G2Connector::OffloadingV032, true),
        ] {
            let mut adapter = connector_adapter(connector, 4);
            let manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
            let (sequence, lease, mut writer) = request(Uuid::from_u128(1), vec![1, 2, 3, 4, 5]);
            let (_, _, mut reader) = request(Uuid::from_u128(2), vec![1, 2, 3, 4, 5]);
            let StoreOutcome::Prepared { transfer_id, .. } =
                adapter
                    .host
                    .lock()
                    .prepare_store(C, writer.owner, &writer.prompt_keys, None, 0.0)
            else {
                panic!("the block was absent")
            };
            writer.latest_store = Some(transfer_id);
            for host in [&mut reader, &mut writer] {
                let outcome = adapter.lookup(
                    host,
                    &sequence,
                    &raw_cost(&manager, &sequence, &lease),
                    4,
                    0.0,
                );
                assert_eq!(
                    (
                        matches!(outcome, HostLookup::Deferred),
                        matches!(outcome, HostLookup::Miss)
                    ),
                    (deferred, !deferred),
                    "{connector:?}"
                );
            }
        }
    }

    #[test]
    fn simple_cpu_pins_a_hit_until_the_request_loads_it_or_looks_up_again() {
        // `get_num_new_matched_tokens` pins the CPU hit and drops the previous
        // pin; `update_state_after_alloc` hands the blocks to the load. A
        // prompt ending on a block boundary keeps its last block to compute.
        let mut adapter = connector_adapter(G2Connector::SimpleCpuV032, 2);
        let mut manager = G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
        let id = Uuid::from_u128(1);
        let (sequence, lease, mut host) = request(id, vec![1, 2, 3, 4, 5]);
        let key = host.prompt_keys[0];
        let now = seed_host(&mut adapter, key);
        let (boundary, boundary_lease, mut boundary_host) = request(id, vec![1, 2, 3, 4]);
        assert!(matches!(
            adapter.lookup(
                &mut boundary_host,
                &boundary,
                &raw_cost(&manager, &boundary, &boundary_lease),
                4,
                now,
            ),
            HostLookup::Miss
        ));
        for _ in 0..2 {
            let cost = raw_cost(&manager, &sequence, &lease);
            assert!(matches!(
                adapter.lookup(&mut host, &sequence, &cost, 4, now),
                HostLookup::Hit(_)
            ));
            assert_eq!(adapter.host.lock().pins(key), 1);
        }
        let cost = raw_cost(&manager, &sequence, &lease);
        let HostLookup::Hit(hit) = adapter.lookup(&mut host, &sequence, &cost, 4, now) else {
            panic!("the pinned block must hit")
        };
        assert!(matches!(
            adapter.start_load(id, &mut host, &lease, hit, &mut manager, now),
            StartLoad::Queued
        ));
        assert_eq!(adapter.host.lock().pins(key), 1, "only the load holds it");
    }

    #[test]
    fn simple_cpu_loads_a_block_into_two_requests_at_once() {
        // A SimpleCPU load only references the CPU blocks
        // (`update_state_after_alloc`); the offloading connector defers a
        // request whose hit another request is loading.
        for (connector, concurrent) in [
            (G2Connector::SimpleCpuV032, true),
            (G2Connector::OffloadingV032, false),
        ] {
            let mut adapter = connector_adapter(connector, 2);
            let mut manager =
                G1Manager::new_with_caching(4, 4, KvEventPublishers::default(), 0, true);
            let mut requests = [1, 2].map(|id| request(Uuid::from_u128(id), vec![1, 2, 3, 4, 5]));
            let now = seed_host(&mut adapter, requests[0].2.prompt_keys[0]);
            let outcomes = requests.each_mut().map(|(sequence, lease, host)| {
                let cost = raw_cost(&manager, sequence, lease);
                match adapter.lookup(host, sequence, &cost, 4, now) {
                    HostLookup::Hit(hit) => matches!(
                        adapter.start_load(host.owner, host, lease, hit, &mut manager, now),
                        StartLoad::Queued
                    ),
                    HostLookup::Deferred => false,
                    HostLookup::Miss => panic!("the seeded block must hit"),
                }
            });
            assert_eq!(outcomes, [true, concurrent], "{connector:?}");
        }
    }

    #[test]
    fn simple_cpu_holds_a_finished_request_s_blocks_until_their_copy_completes() {
        // SimpleCPU touches the GPU blocks it copies (manager.py
        // `_prepare_eager_store_specs`) and frees them when the copy completes
        // (`_process_store_completion`); the offloading connector frees them
        // with the request and fences their next owner.
        for (connector, held) in [
            (G2Connector::SimpleCpuV032, 2),
            (G2Connector::OffloadingV032, 0),
        ] {
            let mut args = connector_args(connector);
            args.native_host_offload = Some(
                NativeHostOffloadConfig::new(16)
                    .with_bandwidths(1.0, 1.0)
                    .with_connector(connector),
            );
            let mut core = VllmCore::new_with_worker_rank(args, 0, 0, 0, true, None).unwrap();
            core.receive(DirectRequest {
                tokens: (1..=8).collect(),
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(1)),
                arrival_timestamp_ms: Some(0.0),
                ..Default::default()
            });
            let pass = core.execute_pass(&mut TraceCollector::default(), 0.0);
            core.complete_engine_boundary(pass.end_ms);
            assert!(core.is_empty());
            assert_eq!(
                core.mocker_metrics().active_decode_blocks,
                held,
                "{connector:?}"
            );
            core.process_internal_work(core.next_internal_deadline_ms().unwrap());
            assert_eq!(
                core.mocker_metrics().active_decode_blocks,
                0,
                "{connector:?}"
            );
        }
    }

    #[test]
    fn simple_cpu_preemption_waits_for_every_transfer_in_flight() {
        // A preemption makes SimpleCPU's worker synchronize every transfer in
        // flight (worker.py `handle_preemptions`); the offloading connector
        // flushes only the preempted request's stores.
        for (connector, compute_not_before) in [
            (G2Connector::SimpleCpuV032, 2.0),
            (G2Connector::OffloadingV032, 1.0),
        ] {
            let mut adapter = connector_adapter(connector, 2);
            let [first, _] = [[1, 2, 3, 4], [5, 6, 7, 8]].map(|tokens| {
                let (_, _, host) = request(Uuid::nil(), tokens.to_vec());
                let StoreOutcome::Prepared { transfer_id, .. } =
                    adapter
                        .host
                        .lock()
                        .prepare_store(C, Uuid::nil(), &host.prompt_keys, None, 0.0)
                else {
                    panic!("the block was absent")
                };
                transfer_id
            });
            assert_eq!(adapter.host.lock().submit_prepared_stores(C, 0.0), 2);
            let (_, _, mut preempted) = request(Uuid::nil(), vec![1, 2, 3, 4]);
            preempted.latest_store = Some(first);
            adapter.preempt_request(&mut preempted);
            assert_eq!(
                adapter.compute_not_before_ms(0.0),
                compute_not_before,
                "{connector:?}"
            );
        }
    }

    #[test]
    fn offloading_v0_32_evicts_a_finished_prefix_tail_first() {
        use crate::engine::{KvEventData, KvEventTier};
        // When a request finishes, the 0.32 offloading connector re-ranks its
        // blocks with the first one newest (lru.py `on_request_finished`), so
        // the next eviction takes the tail of A's prefix. SimpleCPU's free
        // queue takes its head, the block stored first.
        let (a, x, y) = (
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
        );
        let (_, _, a_host) = request(Uuid::nil(), a.to_vec());
        let [a0, a1] = [0, 1].map(|index| a_host.prompt_keys[index].sequence_hash());
        for (connector, evicted) in [
            (G2Connector::OffloadingV032, a1),
            (G2Connector::SimpleCpuV032, a0),
        ] {
            let (events, _) = served_kv_events(
                Some(
                    NativeHostOffloadConfig::new(3)
                        .with_bandwidths(1.0, 1.0)
                        .with_connector(connector),
                ),
                &[&[&a], &[&x], &[&y]],
            );
            let removed = events
                .iter()
                .flatten()
                .filter(|event| event.tier == KvEventTier::HostPinned)
                .filter_map(|event| match &event.data {
                    KvEventData::Removed { block_hashes } => Some(block_hashes.clone()),
                    KvEventData::Stored(_) => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(removed, [vec![evicted]], "{connector:?}");
        }
    }

    #[test]
    fn a_request_cancelled_between_passes_still_stores_its_blocks() {
        use crate::engine::scheduler::SchedulerCommand;
        // SimpleCPU stores a leaving request's computed blocks, and vLLM keeps
        // stepping while that store is pending: an otherwise idle rank runs a
        // pass to submit it.
        let mut core = VllmCore::new_with_worker_rank(
            connector_args(G2Connector::SimpleCpuV032),
            0,
            0,
            0,
            true,
            None,
        )
        .unwrap();
        let id = Uuid::from_u128(1);
        let prompt = (1..=8).collect::<Vec<u32>>();
        core.receive(DirectRequest {
            tokens: prompt.clone(),
            max_output_tokens: 16,
            uuid: Some(id),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        });
        let pass = core.execute_pass(&mut TraceCollector::default(), 0.0);
        core.complete_engine_boundary(pass.end_ms);
        core.drain_kv_events();
        core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
            .unwrap();
        assert!(core.is_empty() && core.is_ready());
        let pass = core.execute_pass(&mut TraceCollector::default(), pass.end_ms);
        core.complete_engine_boundary(pass.end_ms);
        let (_, _, host) = request(Uuid::nil(), prompt);
        assert_eq!(
            host_stored(pass.kv_events.into_iter().chain(core.drain_kv_events())),
            host.prompt_keys
                .iter()
                .map(|key| key.sequence_hash())
                .collect::<Vec<_>>()
        );
        assert!(!core.is_ready() && core.is_drained());
    }

    #[test]
    fn a_hybrid_hit_ends_at_the_deepest_state_with_every_slot() {
        // vLLM's hybrid lookup needs the attention prefix and every Mamba
        // group's block at one boundary (`_lookup_complete_chunks`, and the
        // SimpleCPU coordinator's `find_longest_cache_hit`).
        for connector in [G2Connector::SimpleCpuV032, G2Connector::OffloadingV032] {
            let mut adapter = connector_adapter(connector, 16);
            adapter.set_hybrid(2);
            let manager = G1Manager::new_with_caching(16, 4, KvEventPublishers::default(), 0, true);
            let (sequence, lease, mut host) = request(Uuid::from_u128(1), (1..=21).collect());
            let blocks = host.prompt_keys.clone();
            // G2 holds blocks 0..5, both slots of the states after blocks 0
            // and 1, and one slot of the state after block 3.
            let state =
                |block: usize, slot| HostBlockKey::state(blocks[block].sequence_hash(), slot);
            for key in blocks.iter().copied().chain([
                state(0, 0),
                state(0, 1),
                state(1, 0),
                state(1, 1),
                state(3, 0),
            ]) {
                seed_host(&mut adapter, key);
            }
            let cost = raw_cost(&manager, &sequence, &lease);
            let HostLookup::Hit(hit) = adapter.lookup(&mut host, &sequence, &cost, 4, 0.0) else {
                panic!("{connector:?}: the state after block 1 is a hit")
            };
            assert_eq!(hit.state_slots, 2, "{connector:?}");
            assert_eq!(hit.keys, [state(1, 0), state(1, 1), blocks[0], blocks[1]]);
        }
    }

    /// A hybrid model in miniature: 64-byte blocks, a two-slot state, the
    /// internal prefill checkpoint, and a G2 that copies instantly.
    fn hybrid_args(connector: G2Connector, gpu_blocks: usize) -> MockEngineArgs {
        MockEngineArgs::builder()
            .num_gpu_blocks(gpu_blocks)
            .block_size(4)
            .max_num_seqs(Some(16))
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
            .kv_cache_bytes_per_token(Some(16))
            .state_cache(Some(crate::engine::StateCacheConfig {
                bytes_per_request: 128,
                num_speculative_states: None,
                prefill_checkpoint: true,
                junction_checkpoints: false,
            }))
            .prefix_match_unit(Some(4))
            .native_host_offload(Some(
                NativeHostOffloadConfig::new(64)
                    .with_bandwidths(0.0, 0.0)
                    .with_connector(connector),
            ))
            .build()
            .unwrap()
    }

    /// Run `core` from `now` until it is idle. Returns `(reused, from G1, from
    /// G2)` prompt tokens of each admission.
    fn run_to_idle(core: &mut VllmCore, now: &mut f64) -> Vec<(usize, usize, usize)> {
        let mut admissions = Vec::new();
        for _ in 0..1_000 {
            if core.is_ready() {
                let pass = core.execute_pass(&mut TraceCollector::default(), *now);
                *now = pass.end_ms.max(*now) + 1.0;
                core.complete_engine_boundary(*now);
                admissions.extend(pass.admissions.iter().map(|admission| {
                    let tiers = admission.cache_tier_attribution.unwrap();
                    (
                        admission.reused_input_tokens,
                        tiers.g1_reused_input_tokens,
                        tiers.host_reused_input_tokens,
                    )
                }));
            } else if let Some(deadline) = core.next_internal_deadline_ms() {
                *now = now.max(deadline);
                core.process_internal_work(*now);
            } else {
                return admissions;
            }
        }
        panic!("the rank did not go idle");
    }

    /// Run each `(id, prompt, output)` turn to idle on one core. Returns the
    /// admissions of each turn.
    fn run_turns(
        args: MockEngineArgs,
        turns: &[(u128, std::ops::Range<u32>, usize)],
    ) -> Vec<Vec<(usize, usize, usize)>> {
        let mut core = VllmCore::new(args);
        let mut now = 0.0;
        let admissions = turns
            .iter()
            .map(|(id, tokens, output)| {
                core.receive(DirectRequest {
                    tokens: tokens.clone().collect(),
                    max_output_tokens: *output,
                    uuid: Some(Uuid::from_u128(*id)),
                    arrival_timestamp_ms: Some(now),
                    ..Default::default()
                });
                run_to_idle(&mut core, &mut now)
            })
            .collect();
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        admissions
    }

    #[test]
    fn a_follow_up_restores_attention_and_state_from_g2() {
        // A one-token turn finishes in the pass whose prefill checkpoint
        // publishes its state at 20, and still offers it: vLLM takes a pass's
        // boundary hand-offs before a request can finish in it. A filler then
        // evicts the turn from G1, and the follow-up loads blocks 0..5 and
        // that state. 20 is the follow-up's last boundary before its final
        // token, so the load also caches the state in G1 (vLLM's post-load
        // `cache_blocks`), and the next turn resumes there without a load.
        for connector in [G2Connector::SimpleCpuV032, G2Connector::OffloadingV032] {
            let turns = [
                (1, 0..22, 1),
                (2, 1000..1040, 1),
                (3, 0..21, 1),
                (4, 0..30, 2),
            ];
            let admissions = run_turns(hybrid_args(connector, 16), &turns);
            assert_eq!(
                admissions[2..],
                [[(20, 0, 20)], [(20, 20, 0)]],
                "{connector:?}"
            );
        }
    }

    #[test]
    fn a_restored_state_brings_its_speculative_blocks_to_admission() {
        // The load allocates the restored state's speculative blocks, so the
        // follow-up's admission charges only its new slot (vLLM 43b4aaea's
        // MambaManager adds them only for a request without blocks): 3 token
        // blocks, 2 for the slot and 2 for the checkpoint fit the 7 free.
        let mut args = hybrid_args(G2Connector::SimpleCpuV032, 20);
        args.aic_nextn = Some(3);
        args.aic_nextn_accept_rates = Some("1,1,1".to_string());
        args.disable_eagle_block_drop = true;
        args.state_cache.as_mut().unwrap().num_speculative_states = Some(3);
        let turns = [(1, 0..22, 1), (2, 1000..1040, 1), (3, 0..30, 2)];
        assert_eq!(run_turns(args, &turns)[2], [(20, 0, 20)]);
    }

    #[test]
    fn a_state_retracted_by_a_draft_slot_preemption_is_not_stored() {
        // Replay reserves MTP draft slots while emitting tokens, after the
        // pass published its states (vLLM reserves them while scheduling). A
        // request preempted there retracts its pass, so a hybrid G2 stores
        // after emission: the second request's prefill checkpoint at 4 is
        // retracted and never stored.
        for connector in [G2Connector::SimpleCpuV032, G2Connector::OffloadingV032] {
            let mut args = hybrid_args(connector, 12);
            args.aic_nextn = Some(3);
            args.aic_nextn_accept_rates = Some("1,1,1".to_string());
            args.disable_eagle_block_drop = true;
            let mut core = VllmCore::new(args);
            let prompts: [Vec<u32>; 2] = [(0..8).collect(), (100..108).collect()];
            for (id, tokens) in prompts.iter().enumerate() {
                core.receive(DirectRequest {
                    tokens: tokens.clone(),
                    max_output_tokens: 8,
                    uuid: Some(Uuid::from_u128(id as u128)),
                    arrival_timestamp_ms: Some(0.0),
                    ..Default::default()
                });
            }
            let pass = core.execute_pass(&mut TraceCollector::default(), 0.0);
            core.complete_engine_boundary(pass.end_ms);
            assert_eq!(
                (
                    pass.admissions.len(),
                    pass.mocker_metrics.vllm_preemptions_total
                ),
                (2, 1),
                "{connector:?}"
            );
            let tier = core.host_offload().unwrap().host.lock();
            let checkpoint = |tokens: &Vec<u32>| {
                let (_, _, host) = request(Uuid::nil(), tokens.clone());
                tier.lookup(HostBlockKey::state(host.prompt_keys[0].sequence_hash(), 0))
            };
            assert_eq!(checkpoint(&prompts[0]), Lookup::Hit, "{connector:?}");
            assert_eq!(checkpoint(&prompts[1]), Lookup::Miss, "{connector:?}");
        }
    }

    #[test]
    fn offloading_v0_32_re_ranks_a_finished_request_s_states_with_its_blocks() {
        // At finish the 0.32 connector re-ranks the request's keys with its
        // head newest (lru.py `on_request_finished`), ordered by end position
        // with Mamba groups first (`order_request_keys`). A 10-token prompt
        // stores blocks 0 and 1 and its prefill checkpoint at 8, so eviction
        // takes the tail block first, then the state, then the head.
        let mut core = VllmCore::new(hybrid_args(G2Connector::OffloadingV032, 16));
        core.receive(DirectRequest {
            tokens: (0..10).collect(),
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(1)),
            arrival_timestamp_ms: Some(0.0),
            ..Default::default()
        });
        run_to_idle(&mut core, &mut 0.0);
        let (_, _, host) = request(Uuid::nil(), (0..10).collect());
        let [head, tail] = [host.prompt_keys[0], host.prompt_keys[1]];
        let state = |slot| HostBlockKey::state(tail.sequence_hash(), slot);
        let tier = core.host_offload().unwrap().host.lock();
        assert_eq!(tier.lru_order(), [tail, state(1), state(0), head]);
    }
}
