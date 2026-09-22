// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Optional composition: AISimulate owns simulation; unmodified Dynamo owns policy.
mod events;

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use aisimulate_core::engine::EngineConfig;
use aisimulate_core::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use aisimulate_core::replay::{
    AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1, Placement, PlacementCacheSample, PlacementDecision,
    PlacementEffects, PlacementPolicy, ReplayAdmissionMetadata, ReplayComposition,
    ReplayDeterminism, ReplayEngineConfig, ReplayError, ReplayPromptTokenSource, ReplaySpec,
    WorkerTopology,
};
use anyhow::{Context, Result, bail, ensure};
use dynamo_kv_router::protocols::{LocalBlockHash, RoutingConstraints, WorkerAffinityTarget};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::sequences::ReplicaRequestLeaseObserver;
use dynamo_kv_router::services::indexer::{backend::Indexer, registry::WorkerRegistry};
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
};
use dynamo_kv_router::services::selection::{
    HostCache, HostReplication, KvEventIngress, KvIndexSource, PromptRequest, RouterPluginRegistry,
    SelectionAdmission, SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome,
    SelectionService, SelectionServiceBuilder, SessionBinding, WorkerCatalogRecord, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionId, WorkerType};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::runtime::Runtime;
use uuid::Uuid;

pub const DYNAMO_REVISION: &str = "d9eb42db1168131fdae318eef77255637e4d3495";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterConfig {
    policy: String,
    #[serde(default)]
    affinity: Option<AffinityConfig>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AffinityConfig {
    mode: AffinityMode,
    #[serde(default = "default_affinity_ttl_seconds")]
    ttl_seconds: f64,
}
fn default_affinity_ttl_seconds() -> f64 {
    3600.0
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AffinityMode {
    Session,
    SiblingGroup,
}
impl RouterConfig {
    fn parse(input: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(input).context("invalid Dynamo router configuration")?;
        ensure!(
            config.policy == "kv_router",
            "Dynamo plugin supports router.policy: kv_router only"
        );
        if let Some(affinity) = &config.affinity {
            ensure!(
                affinity.ttl_seconds.is_finite()
                    && affinity.ttl_seconds >= 1.0
                    && affinity.ttl_seconds <= 31_536_000.0,
                "affinity.ttl_seconds must be between 1 and 31536000"
            );
        }
        Ok(config)
    }
}

/// Real engine events are supplied directly by Replay; no sockets or estimated cache.
struct ReplayIngress;
#[async_trait::async_trait]
impl KvEventIngress for ReplayIngress {
    fn open(
        &self,
        registry: &WorkerRegistry,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Indexer {
        registry.get_or_create_indexer(key.clone(), block_size)
    }
    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
        let key = RoutingPartitionId::new(&record.model_name, &record.routing_group);
        for rank in record.dp_ranks() {
            registry
                .remove_dp_rank_blocks(record.worker_id, rank, &key)
                .await;
        }
    }
}
/// Replay terminals, rather than production request timeout sweepers, own bookings.
struct ReplayLeases;
impl ReplicaRequestLeaseObserver for ReplayLeases {
    fn admitted(&self, _: SchedulerBookingDescriptor) {}
    fn progressed(&self, _: &SchedulerBookingDescriptor) {}
    fn completed(&self, _: &SchedulerBookingDescriptor) {}
}

#[derive(Clone)]
pub struct Metadata(Option<ReplayRequestHashes>, Option<usize>);
impl ReplayAdmissionMetadata for Metadata {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self {
        Self(hashes, None)
    }
    fn for_prefill(self) -> Self {
        Self(self.0, Some(1))
    }
    fn max_output_tokens_override(&self) -> Option<usize> {
        self.1
    }
    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        self.0
    }
}

#[derive(Default)]
struct Evidence {
    capture_decisions: bool,
    decision_count: u64,
    decisions_by_role: BTreeMap<&'static str, u64>,
    decisions: Vec<Value>,
    physical_kv_events: u64,
}
type SharedEvidence = Arc<Mutex<Evidence>>;

struct Composition {
    config: RouterConfig,
    engine: RefCell<Option<ReplayEngineConfig>>,
    evidence: SharedEvidence,
}
impl ReplayComposition for Composition {
    type Metadata = Metadata;
    type Observation = events::Observation;
    type AggregatedPlacement = NativePlacement;
    type DisaggregatedPlacement = NativePlacement;
    fn validate_spec(&self, spec: &ReplaySpec) -> aisimulate_core::replay::ReplayResult<()> {
        let check = || -> Result<()> {
            ensure!(
                matches!(
                    spec.adapters.placement.provider.as_str(),
                    "round_robin" | "dynamo_kv_router"
                ),
                "unsupported placement descriptor for Dynamo plugin"
            );
            ensure!(
                spec.adapters.scaling.provider == "none",
                "Dynamo policy plugin does not provide dynamic scaling"
            );
            let engine = if spec.engine.is_null() {
                ReplayEngineConfig::default()
            } else {
                serde_json::from_value(spec.engine.clone())?
            };
            *self.engine.borrow_mut() = Some(engine);
            self.evidence
                .lock()
                .map_err(|_| anyhow::anyhow!("routing evidence lock poisoned"))?
                .capture_decisions = spec.record_per_request;
            Ok(())
        };
        check().map_err(|e| ReplayError::InvalidSpec(format!("{e:#}")))
    }
    fn set_determinism(
        &mut self,
        determinism: ReplayDeterminism,
    ) -> aisimulate_core::replay::ReplayResult<()> {
        if determinism.selector_seed().is_some() {
            return Err(ReplayError::InvalidSpec("native Dynamo SelectionCore does not expose seeded selection; workload seeds remain supported".into()));
        }
        Ok(())
    }
    fn create_aggregated_placement(
        &mut self,
        dp: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<NativePlacement> {
        let rank = self
            .engine
            .borrow()
            .as_ref()
            .context("missing validated engine")?
            .rank
            .clone();
        NativePlacement::new(
            "aggregated",
            rank,
            dp,
            topology,
            self.config.clone(),
            self.evidence.clone(),
        )
    }
    fn create_disaggregated_placements(
        &mut self,
        pdp: u32,
        ptop: Vec<WorkerTopology>,
        ddp: u32,
        dtop: Vec<WorkerTopology>,
    ) -> Result<(NativePlacement, NativePlacement)> {
        let engine = self
            .engine
            .borrow()
            .as_ref()
            .context("missing validated engine")?
            .clone();
        let prefill = engine
            .prefill
            .map_or_else(|| engine.rank.clone(), |r| r.rank);
        let decode = engine.decode.map_or(engine.rank, |r| r.rank);
        Ok((
            NativePlacement::new(
                "prefill",
                prefill,
                pdp,
                ptop,
                self.config.clone(),
                self.evidence.clone(),
            )?,
            NativePlacement::new(
                "decode",
                decode,
                ddp,
                dtop,
                self.config.clone(),
                self.evidence.clone(),
            )?,
        ))
    }
}

struct ActiveRequest {
    group_key: Option<String>,
    hold: Option<Hold>,
    lease: Option<AffinityLease>,
    target: WorkerAffinityTarget,
}
struct PendingRequest {
    request: ReplayRequestPayload,
    metadata: Metadata,
    session_id: Option<String>,
}
pub struct NativePlacement {
    runtime: Runtime,
    // A live blocking task inhibits Tokio's automatic paused-clock advancement.
    // Drop the sender before the runtime, including on construction failures.
    stop_clock_guard: Option<mpsc::Sender<()>>,
    core: Arc<SelectionCore>,
    _service: SelectionService,
    key: RoutingPartitionId,
    affinity: Option<SessionAffinity>,
    config: RouterConfig,
    rank: EngineConfig,
    dp: u32,
    topology: HashMap<usize, Vec<usize>>,
    active: HashMap<Uuid, ActiveRequest>,
    pending: VecDeque<PendingRequest>,
    pending_ready: bool,
    bindings: HashMap<String, WorkerAffinityTarget>,
    // This index only supports membership invalidation. Native affinity owns
    // expiration; periodically discard index entries it no longer considers live.
    binding_prune_interval: Duration,
    next_binding_prune: Option<Duration>,
    now: Duration,
    // Preserve the replay instant for same-time wakeups; Duration quantizes it.
    replay_now_ms: f64,
    role: &'static str,
    evidence: SharedEvidence,
}
impl NativePlacement {
    fn new(
        role: &'static str,
        rank: EngineConfig,
        dp: u32,
        topology: Vec<WorkerTopology>,
        config: RouterConfig,
        evidence: SharedEvidence,
    ) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        let (stop, receiver) = mpsc::channel();
        let built = runtime.block_on(async {
            tokio::time::pause();
            tokio::task::spawn_blocking(move || {
                let _ = receiver.recv();
            });
            let router_config = KvRouterConfig {
                use_kv_events: true,
                router_queue_threshold: None,
                ..Default::default()
            };
            let host = SelectionHost {
                cache: HostCache {
                    index: KvIndexSource::Owned(Arc::new(ReplayIngress)),
                    ..Default::default()
                },
                replication: HostReplication {
                    request_leases: Some(Arc::new(ReplayLeases)),
                    ..Default::default()
                },
                ..Default::default()
            };
            let worker_type = match role {
                "prefill" => WorkerType::Prefill,
                "decode" => WorkerType::Decode,
                _ => WorkerType::Aggregated,
            };
            let service = SelectionServiceBuilder::new(
                router_config,
                worker_type,
                RouterPluginRegistry::default(),
            )
            .host(host)
            .indexer_threads(1)
            .build()
            .await?;
            let core = service.core().clone();
            let key = RoutingPartitionId::new("aisimulate", role);
            let partition =
                core.ensure_partition(key.clone(), rank.block_size.try_into()?, false)?;
            let affinity = config
                .affinity
                .as_ref()
                .map(|a| {
                    partition.session_affinity(SessionAffinityConfig::new(Duration::from_secs_f64(
                        a.ttl_seconds,
                    )))
                })
                .transpose()?;
            Ok::<_, anyhow::Error>((core, key, affinity, service))
        });
        let (core, key, affinity, service) = match built {
            Ok(value) => value,
            Err(error) => {
                drop(stop);
                return Err(error);
            }
        };
        let binding_prune_interval = config
            .affinity
            .as_ref()
            .map_or(Duration::from_secs(60), |affinity| {
                Duration::from_secs_f64(affinity.ttl_seconds.min(60.0))
            });
        let mut placement = Self {
            runtime,
            stop_clock_guard: Some(stop),
            core,
            _service: service,
            key,
            affinity,
            config,
            rank,
            dp,
            topology: HashMap::new(),
            active: HashMap::new(),
            pending: VecDeque::new(),
            pending_ready: false,
            bindings: HashMap::new(),
            binding_prune_interval,
            next_binding_prune: Some(binding_prune_interval),
            now: Duration::ZERO,
            replay_now_ms: 0.0,
            role,
            evidence,
        };
        for worker in topology {
            placement.add_worker(worker)?;
        }
        Ok(placement)
    }
    fn advance(&mut self, now_ms: f64) -> Result<()> {
        ensure!(now_ms.is_finite() && now_ms >= 0.0, "invalid replay clock");
        ensure!(
            now_ms >= self.replay_now_ms,
            "Dynamo replay clock moved backwards"
        );
        let next = Duration::try_from_secs_f64(now_ms / 1000.0).context("replay clock overflow")?;
        ensure!(next >= self.now, "Dynamo replay clock moved backwards");
        let delta = next - self.now;
        if !delta.is_zero() {
            self.runtime.block_on(tokio::time::advance(delta));
            self.now = next;
        }
        self.replay_now_ms = now_ms;
        if self
            .next_binding_prune
            .is_some_and(|deadline| self.now >= deadline)
        {
            self.prune_bindings()?;
            self.next_binding_prune = self.now.checked_add(self.binding_prune_interval);
        }
        Ok(())
    }
    fn add_worker(&mut self, worker: WorkerTopology) -> Result<()> {
        ensure!(
            worker.scheduler_ids.len() == self.dp as usize,
            "worker scheduler topology does not match attention DP"
        );
        let request: WorkerRequest = serde_json::from_value(json!({
            "worker_id":worker.worker_id, "model_name":"aisimulate", "routing_group":self.role,
            "endpoint":format!("http://simulation-worker-{}:1", worker.worker_id),
            "block_size":self.rank.block_size, "data_parallel_start_rank":0, "data_parallel_size":self.dp,
            "max_num_batched_tokens":self.rank.max_num_batched_tokens, "total_kv_blocks":self.rank.num_gpu_blocks,
        }))?;
        self.runtime.block_on(self.core.upsert_worker(request))?;
        self.topology.insert(worker.worker_id, worker.scheduler_ids);
        Ok(())
    }
    fn group_key(
        &self,
        request: &ReplayRequestPayload,
        session: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(affinity) = &self.config.affinity else {
            return Ok(None);
        };
        let context = request.metadata().replay_context.as_ref();
        let agentic = context.and_then(|c| c.agentic.as_ref());
        let session = session
            .or_else(|| context.and_then(|c| c.session_id.as_deref()))
            .or_else(|| agentic.map(|a| a.conversation_id.as_str()))
            .context("conversation affinity requires session identity")?;
        let parts = match affinity.mode {
            AffinityMode::Session => json!(["session", agentic.map(|a| &a.play_id), session]),
            AffinityMode::SiblingGroup => {
                let a = agentic
                    .context("sibling_group affinity requires Agentic conversation lineage")?;
                let lineage = a
                    .lineage
                    .as_ref()
                    .context("sibling_group affinity requires unambiguous conversation lineage")?;
                ensure!(
                    lineage.schema == AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1,
                    "unsupported conversation lineage schema"
                );
                match &lineage.parent_conversation_id {
                    Some(parent) => {
                        json!(["siblings", a.play_id, lineage.root_conversation_id, parent])
                    }
                    None => json!([
                        "root",
                        a.play_id,
                        lineage.root_conversation_id,
                        a.conversation_id
                    ]),
                }
            }
        };
        Ok(Some(
            blake3::hash(serde_json::to_string(&parts)?.as_bytes())
                .to_hex()
                .to_string(),
        ))
    }
    fn release(&mut self, id: Uuid) -> Result<()> {
        if let Some(active) = self.active.remove(&id) {
            self.runtime.block_on(async {
                self.core.free_reservation(&id.to_string()).await?;
                // Native hold/lease Drop computes idle expiry using this runtime's clock.
                drop(active);
                Ok::<_, anyhow::Error>(())
            })?;
            self.pending_ready = true;
        }
        Ok(())
    }
    fn prune_bindings(&mut self) -> Result<()> {
        let Some(table) = &self.affinity else {
            return Ok(());
        };
        self.runtime.block_on(async {
            let mut error = None;
            self.bindings
                .retain(|key, _| match table.query_target(key, None) {
                    Ok(target) => target.is_some(),
                    Err(failure) => {
                        error = Some(failure);
                        true
                    }
                });
            error.map_or(Ok(()), |error| Err(error.into()))
        })
    }
    fn invalidate_worker_bindings(&mut self, worker_id: usize) -> Result<()> {
        let keys = self
            .bindings
            .iter()
            .filter(|(_, target)| target.worker_id == worker_id as u64)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        self.runtime.block_on(async {
            if let Some(table) = &self.affinity {
                for key in &keys {
                    if let AcquireStep::Held(hold) = table.try_acquire(key, None)? {
                        hold.invalidate();
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        })?;
        for key in keys {
            self.bindings.remove(&key);
        }
        self.pending_ready = true;
        Ok(())
    }
    fn retry_pending(&mut self) -> Result<Vec<Placement>> {
        self.pending_ready = false;
        let mut released = Vec::new();
        for _ in 0..self.pending.len() {
            let pending = self
                .pending
                .pop_front()
                .expect("pending count bounded above");
            let effects = self.place(
                &pending.request,
                pending.metadata,
                pending.session_id,
                self.replay_now_ms,
            )?;
            if let PlacementDecision::Immediate(placement) = effects.decision {
                released.push(placement);
            }
        }
        Ok(released)
    }
}
impl Drop for NativePlacement {
    fn drop(&mut self) {
        self.core.shutdown();
        self.stop_clock_guard.take();
    }
}
impl PlacementPolicy<ReplayRequestPayload> for NativePlacement {
    type Metadata = Metadata;
    type Observation = events::Events;
    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        self.advance(now_ms)?;
        let request_meta = request.metadata();
        let id = request_meta
            .uuid
            .context("native policy requires request UUID")?;
        ensure!(
            !self.active.contains_key(&id),
            "duplicate native policy request UUID"
        );
        ensure!(
            request_meta.preferred_dp_rank.is_none()
                && request_meta.preferred_prefill_dp_rank.is_none(),
            "native Dynamo policy plugin does not support authored DP pins; affinity selects and binds the native worker/DP pair"
        );
        ensure!(
            request_meta.policy_class.is_none(),
            "native Dynamo policy plugin does not provide custom policy classes"
        );
        ensure!(
            !request_meta.replay_context.as_ref().is_some_and(
                |c| c.prompt_token_source == ReplayPromptTokenSource::LengthOnlySynthetic
            ),
            "KV routing requires materialized token identities, not length-only prompt placeholders"
        );
        let group_key = self.group_key(request, session_id.as_deref())?;
        let hold = self.runtime.block_on(async {
            match (&self.affinity, &group_key) {
                (Some(table), Some(key)) => table.try_acquire(key, None).map(Some),
                _ => Ok(None),
            }
        })?;
        let hold = match hold {
            Some(AcquireStep::Held(hold)) => Some(hold),
            Some(AcquireStep::Wait(_)) => {
                let mut owned = request_meta.clone();
                owned.tokens = request.prompt_tokens();
                self.pending.push_back(PendingRequest {
                    request: ReplayRequestPayload::materialized(owned),
                    metadata,
                    session_id,
                });
                return Ok(PlacementEffects {
                    decision: PlacementDecision::Queued,
                    released: Vec::new(),
                });
            }
            None => None,
        };
        let affinity_target = hold.as_ref().and_then(Hold::target);
        let expected_output_tokens = metadata
            .max_output_tokens_override()
            .unwrap_or(request_meta.max_output_tokens);
        let hashes = metadata.0.unwrap_or_else(|| {
            ReplayRequestHashes::from_tokens(&request.prompt_tokens(), self.rank.block_size as u32)
        });
        let prompt = PromptRequest {
            block_hashes: Some(
                hashes
                    .local_block_hashes
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            ),
            sequence_hashes: Some(hashes.sequence_hashes.iter().map(|&v| v as i64).collect()),
            isl_tokens: Some(request.input_length()),
            ..Default::default()
        };
        let partition = self
            .core
            .partition(&self.key)
            .context("missing native routing partition")?;
        let output_tokens: u32 = expected_output_tokens.try_into()?;
        let (selected, best_overlap) = self.runtime.block_on(async {
            let overlaps = partition
                .indexer()
                .find_matches(
                    hashes
                        .local_block_hashes
                        .iter()
                        .copied()
                        .map(LocalBlockHash)
                        .collect(),
                )
                .await?;
            let best_overlap = overlaps
                .scores
                .iter()
                .filter(|(w, _)| self.topology.contains_key(&(w.worker_id as usize)))
                .map(|(_, &overlap)| overlap)
                .max()
                .unwrap_or(0);
            let outcome = self
                .core
                .run_selection(SelectionOperation {
                    key: self.key.clone(),
                    prompt: prompt.view(),
                    router_config_override: None,
                    expected_output_tokens: Some(output_tokens),
                    priority_jump: request_meta.priority as f64,
                    strict_priority: request_meta.strict_priority,
                    policy_class: None,
                    session_context: None,
                    session: SessionBinding::None,
                    affinity_target,
                    pinned_worker: None,
                    allowed_worker_ids: Some(
                        self.topology
                            .keys()
                            .map(|&id| id as u64)
                            .collect::<HashSet<_>>(),
                    ),
                    routing_constraints: RoutingConstraints::default(),
                    admission: SelectionAdmission::Book {
                        selection_id: id.to_string(),
                    },
                    track_active_blocks: true,
                    return_routing_hashes: false,
                    replay_id: None,
                })
                .await
                .result?;
            match outcome {
                SelectionOutcome::Selected(selected) => Ok((selected, best_overlap)),
                SelectionOutcome::QueueRejected { rejection } => {
                    bail!("native Dynamo policy rejected request: {rejection:?}")
                }
            }
        })?;
        let worker = selected.response.best_worker;
        let scheduler = *self
            .topology
            .get(&(worker.worker_id as usize))
            .and_then(|r| r.get(worker.dp_rank as usize))
            .context("native policy selected unavailable worker/DP")?;
        self.active.insert(
            id,
            ActiveRequest {
                group_key: group_key.clone(),
                hold,
                lease: None,
                target: worker.into(),
            },
        );
        let mut evidence = self
            .evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("routing evidence lock poisoned"))?;
        evidence.decision_count += 1;
        *evidence.decisions_by_role.entry(self.role).or_default() += 1;
        if evidence.capture_decisions {
            evidence.decisions.push(json!({
            "request_id":id, "authored_request_id":request_meta.replay_context.as_ref().map(|c| &c.authored_id),
            "session_id":session_id.or_else(|| request_meta.replay_context.as_ref().and_then(|c|c.session_id.clone())), "group_key":group_key, "role":self.role,
            "worker_id":worker.worker_id, "dp_rank":worker.dp_rank, "scheduler_id":scheduler,
            "at_ms":now_ms, "overlap_blocks":selected.response.target_cached_prefix_blocks,
            "best_available_overlap_blocks":best_overlap, "native_policy":"dynamo.SelectionCore",
        }));
        }
        Ok(PlacementEffects {
            decision: PlacementDecision::Immediate(Placement {
                request_id: id,
                scheduler_id: scheduler,
                reported_overlap_tokens: selected.response.cached_tokens,
                cache_sample: Some(PlacementCacheSample {
                    overlap_blocks: selected.response.target_cached_prefix_blocks,
                    best_available_overlap_blocks: best_overlap,
                    isl_blocks: (request.input_length() / self.rank.block_size).try_into()?,
                }),
                placement_replica_id: None,
            }),
            released: Vec::new(),
        })
    }
    fn observe(&mut self, observation: events::Events, now_ms: f64) -> Result<Vec<Placement>> {
        self.advance(now_ms)?;
        let count = observation.0.len() as u64;
        let partition = self
            .core
            .partition(&self.key)
            .context("missing native routing partition")?;
        self.runtime.block_on(async {
            for (worker, event) in observation.0 {
                partition
                    .indexer()
                    .try_apply_event(events::convert(worker, event)?)
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        })?;
        self.evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("routing evidence lock poisoned"))?
            .physical_kv_events += count;
        Ok(Vec::new())
    }
    fn dispatch_committed(&mut self, id: Uuid, now: f64) -> Result<()> {
        self.advance(now)?;
        if let Some(active) = self.active.get_mut(&id)
            && let (Some(table), Some(hold)) = (&self.affinity, active.hold.take())
        {
            active.lease = Some(
                self.runtime
                    .block_on(async { table.commit(hold, active.target) })?,
            );
        }
        if let Some(active) = self.active.get(&id)
            && let Some(key) = &active.group_key
        {
            self.bindings.insert(key.clone(), active.target);
        }
        self.pending_ready = true;
        Ok(())
    }
    fn dispatch_aborted(&mut self, id: Uuid, now: f64) -> Result<()> {
        self.advance(now)?;
        self.release(id)
    }
    fn advance_clock(&mut self, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        if self.pending_ready {
            self.retry_pending()
        } else {
            Ok(Vec::new())
        }
    }
    fn next_wakeup_ms(&self) -> Option<f64> {
        (self.pending_ready && !self.pending.is_empty()).then_some(self.replay_now_ms)
    }
    fn cancel_pending(&mut self, id: Uuid) -> bool {
        let before = self.pending.len();
        self.pending
            .retain(|p| p.request.metadata().uuid != Some(id));
        self.pending.len() != before
    }
    fn request_terminal(&mut self, id: Uuid, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        self.release(id)?;
        Ok(Vec::new())
    }
    fn prefill_completed(&mut self, id: Uuid, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        if self.active.contains_key(&id) {
            self.runtime
                .block_on(self.core.prefill_complete(&id.to_string()))?;
        }
        Ok(Vec::new())
    }
    fn pending_count(&self) -> usize {
        self.pending.len()
    }
    fn worker_ready(&mut self, worker: WorkerTopology, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        self.add_worker(worker)?;
        Ok(Vec::new())
    }
    fn worker_draining(&mut self, worker: WorkerTopology, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        self.topology.remove(&worker.worker_id);
        self.invalidate_worker_bindings(worker.worker_id)?;
        Ok(Vec::new())
    }
    fn worker_removed(&mut self, worker: WorkerTopology, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        self.topology.remove(&worker.worker_id);
        self.invalidate_worker_bindings(worker.worker_id)?;
        self.runtime
            .block_on(self.core.delete_worker(worker.worker_id as u64))?;
        Ok(Vec::new())
    }
    fn topology_settled(&mut self, now: f64) -> Result<Vec<Placement>> {
        self.advance(now)?;
        Ok(Vec::new())
    }
}

pub fn execute(payload: &str, router_config: &str) -> Result<String> {
    let config = RouterConfig::parse(router_config)?;
    let evidence = SharedEvidence::default();
    let composition = Composition {
        config,
        engine: RefCell::new(None),
        evidence: evidence.clone(),
    };
    let serialized =
        aisimulate_core::execute_replay_json_with_composition(payload, false, composition)?;
    let mut result: Value = serde_json::from_str(&serialized)?;
    let evidence = evidence
        .lock()
        .map_err(|_| anyhow::anyhow!("routing evidence lock poisoned"))?;
    result["dynamo_policy"] = json!({"dynamo_revision":DYNAMO_REVISION, "native_policy":"dynamo.SelectionCore",
        "physical_kv_events":evidence.physical_kv_events,
        "decision_count":evidence.decision_count, "decisions_by_role":evidence.decisions_by_role,
        "decisions_captured":evidence.capture_decisions, "decisions":evidence.decisions});
    Ok(serde_json::to_string(&result)?)
}
#[pyfunction]
fn run_replay_json(
    py: Python<'_>,
    payload: String,
    router_config_json: String,
) -> PyResult<String> {
    py.allow_threads(move || execute(&payload, &router_config_json))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(format!("{error:#}")))
}
#[pyfunction]
fn native_contract() -> String {
    let mut value = aisimulate_core::native_replay_contract();
    value["plugin_version"] = json!(env!("CARGO_PKG_VERSION"));
    value["dynamo_revision"] = json!(DYNAMO_REVISION);
    value.to_string()
}
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_replay_json, module)?)?;
    module.add_function(wrap_pyfunction!(native_contract, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisimulate_core::engine::{KvBlock, KvEvent, KvEventData, StoredBlocks};
    use aisimulate_core::replay::{
        AgenticConversationLineage, AgenticRuntimeIdentity, DirectRequest, ReplayRequestContext,
    };

    fn placement(mode: &str) -> NativePlacement {
        let rank = EngineConfig {
            block_size: 4,
            num_gpu_blocks: 128,
            ..Default::default()
        };
        NativePlacement::new(
            "aggregated",
            rank,
            2,
            vec![
                WorkerTopology {
                    worker_id: 3,
                    scheduler_ids: vec![0, 1],
                },
                WorkerTopology {
                    worker_id: 7,
                    scheduler_ids: vec![2, 3],
                },
            ],
            RouterConfig::parse(&format!(
                r#"{{"policy":"kv_router","affinity":{{"mode":"{mode}","ttl_seconds":10}}}}"#
            ))
            .unwrap(),
            SharedEvidence::default(),
        )
        .unwrap()
    }
    fn request(session: &str, parent: Option<&str>) -> ReplayRequestPayload {
        let id = Uuid::new_v4();
        ReplayRequestPayload::materialized(DirectRequest {
            tokens: (1..=16).collect(),
            max_output_tokens: 4,
            uuid: Some(id),
            replay_context: Some(ReplayRequestContext {
                authored_id: id.to_string(),
                session_id: Some(session.into()),
                turn_index: None,
                metadata: Value::Null,
                prompt_token_source: ReplayPromptTokenSource::Materialized,
                agentic: Some(AgenticRuntimeIdentity {
                    request_id: id.to_string(),
                    play_id: "play-1".into(),
                    conversation_id: session.into(),
                    lane_id: None,
                    root_id: None,
                    parent_id: None,
                    cache_id: None,
                    lineage: Some(AgenticConversationLineage {
                        schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                        root_conversation_id: "root".into(),
                        parent_conversation_id: parent.map(str::to_owned),
                    }),
                }),
            }),
            ..Default::default()
        })
    }
    fn choose(policy: &mut NativePlacement, request: &ReplayRequestPayload, now: f64) -> Placement {
        match policy
            .place(request, Metadata(None, None), None, now)
            .unwrap()
            .decision
        {
            PlacementDecision::Immediate(p) => p,
            PlacementDecision::Queued => panic!("unexpected queue"),
        }
    }
    #[test]
    fn same_time_waiter_wakeup_preserves_replay_clock_precision() {
        // Duration rounds this instant down to a nanosecond. Returning that
        // rounded instant to replay would violate its nondecreasing clock.
        let now_ms = 0.1234561;
        let rounded_ms = Duration::try_from_secs_f64(now_ms / 1000.0)
            .unwrap()
            .as_secs_f64()
            * 1000.0;
        assert!(rounded_ms < now_ms);
        let mut policy = placement("sibling_group");
        let first = request("first", Some("root"));
        let sibling = request("sibling", Some("root"));
        let selected = choose(&mut policy, &first, now_ms);
        assert!(matches!(
            policy
                .place(&sibling, Metadata(None, None), None, now_ms)
                .unwrap()
                .decision,
            PlacementDecision::Queued
        ));
        policy
            .dispatch_committed(selected.request_id, now_ms)
            .unwrap();
        assert_eq!(policy.next_wakeup_ms(), Some(now_ms));
        let released = policy.advance_clock(now_ms).unwrap();
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].scheduler_id, selected.scheduler_id);
        policy
            .dispatch_committed(released[0].request_id, now_ms)
            .unwrap();
        assert_eq!(policy.next_wakeup_ms(), None);
    }

    #[test]
    fn native_sibling_binding_idle_ttl_and_abort_are_virtual() {
        let mut policy = placement("sibling_group");
        let a = request("child-a", Some("root"));
        let b = request("child-b", Some("root"));
        let root = request("root", None);
        let key = policy.group_key(&a, None).unwrap().unwrap();
        assert_eq!(Some(key.clone()), policy.group_key(&b, None).unwrap());
        assert_ne!(Some(key.clone()), policy.group_key(&root, None).unwrap());
        let pa = choose(&mut policy, &a, 0.0);
        policy.dispatch_committed(pa.request_id, 0.0).unwrap();
        let pb = choose(&mut policy, &b, 1.0);
        assert_eq!(pa.scheduler_id, pb.scheduler_id);
        policy.dispatch_committed(pb.request_id, 1.0).unwrap();
        policy.advance(100_000.0).unwrap();
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_some()
        }));
        policy.request_terminal(pa.request_id, 100_000.0).unwrap();
        policy.request_terminal(pb.request_id, 100_000.0).unwrap();
        policy.advance(109_000.0).unwrap();
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_some()
        }));
        policy.advance(111_000.0).unwrap();
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_none()
        }));
        let aborted = choose(&mut policy, &a, 111_000.0);
        policy
            .dispatch_aborted(aborted.request_id, 111_000.0)
            .unwrap();
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_none()
        }));
    }
    #[test]
    fn tentative_binding_is_not_visible_and_abort_releases_waiter() {
        let mut policy = placement("sibling_group");
        let first = request("first", Some("root"));
        let sibling = request("sibling", Some("root"));
        let key = policy.group_key(&first, None).unwrap().unwrap();
        let tentative = choose(&mut policy, &first, 0.0);
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_none()
        }));
        assert!(matches!(
            policy
                .place(&sibling, Metadata(None, None), None, 0.0)
                .unwrap()
                .decision,
            PlacementDecision::Queued
        ));
        assert_eq!(policy.pending_count(), 1);
        // Remove the rejected tentative worker. The waiter must select a live
        // worker instead of inheriting a binding that was never dispatched.
        let rejected_worker = *policy
            .topology
            .iter()
            .find(|(_, schedulers)| schedulers.contains(&tentative.scheduler_id))
            .unwrap()
            .0;
        policy.dispatch_aborted(tentative.request_id, 0.0).unwrap();
        policy
            .worker_draining(
                WorkerTopology {
                    worker_id: rejected_worker,
                    scheduler_ids: vec![],
                },
                0.0,
            )
            .unwrap();
        let released = policy.advance_clock(0.0).unwrap();
        assert_eq!(released.len(), 1);
        assert_ne!(released[0].scheduler_id / 2, tentative.scheduler_id / 2);
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_none()
        }));
        policy
            .dispatch_committed(released[0].request_id, 0.0)
            .unwrap();
        assert!(policy.runtime.block_on(async {
            policy
                .affinity
                .as_ref()
                .unwrap()
                .query_target(&key, None)
                .unwrap()
                .is_some()
        }));
        policy
            .request_terminal(released[0].request_id, 0.0)
            .unwrap();
    }
    #[test]
    fn cancelled_initializer_waiter_stays_cancelled_and_removed_binding_reinitializes() {
        let mut policy = placement("session");
        let first = request("session", None);
        let waiting = request("session", None);
        let selected = choose(&mut policy, &first, 0.0);
        assert!(matches!(
            policy
                .place(&waiting, Metadata(None, None), None, 0.0)
                .unwrap()
                .decision,
            PlacementDecision::Queued
        ));
        assert!(policy.cancel_pending(waiting.metadata().uuid.unwrap()));
        assert!(!policy.cancel_pending(waiting.metadata().uuid.unwrap()));
        policy.dispatch_committed(selected.request_id, 0.0).unwrap();
        assert!(policy.advance_clock(0.0).unwrap().is_empty());
        policy.request_terminal(selected.request_id, 1.0).unwrap();
        let retired = *policy
            .topology
            .iter()
            .find(|(_, schedulers)| schedulers.contains(&selected.scheduler_id))
            .unwrap()
            .0;
        policy
            .worker_removed(
                WorkerTopology {
                    worker_id: retired,
                    scheduler_ids: vec![],
                },
                1.0,
            )
            .unwrap();
        let replacement = choose(&mut policy, &request("session", None), 1.0);
        assert_ne!(replacement.scheduler_id / 2, selected.scheduler_id / 2);
        policy
            .dispatch_committed(replacement.request_id, 1.0)
            .unwrap();
        policy
            .request_terminal(replacement.request_id, 2.0)
            .unwrap();
    }
    #[test]
    fn invalid_router_and_affinity_inputs_fail_explicitly() {
        for mode in ["session", "sibling_group"] {
            let config = RouterConfig::parse(&format!(
                r#"{{"policy":"kv_router","affinity":{{"mode":"{mode}"}}}}"#
            ))
            .unwrap();
            assert_eq!(config.affinity.unwrap().ttl_seconds, 3600.0);
        }
        for config in [
            r#"{"policy":"round_robin"}"#,
            r#"{"policy":"kv_router","affinity":{"mode":"session","ttl_seconds":0.5}}"#,
            r#"{"policy":"kv_router","affinity":{"mode":"other","ttl_seconds":3600}}"#,
            r#"{"policy":"kv_router","unexpected":true}"#,
        ] {
            assert!(RouterConfig::parse(config).is_err());
        }
        assert!(
            RouterConfig::parse(
                r#"{"policy":"kv_router","affinity":{"mode":"session","ttl_seconds":1.5}}"#
            )
            .is_ok()
        );
        let mut policy = placement("sibling_group");
        let mut missing = request("session", None);
        missing
            .metadata_mut()
            .replay_context
            .as_mut()
            .unwrap()
            .agentic
            .as_mut()
            .unwrap()
            .lineage = None;
        assert!(
            policy
                .place(&missing, Metadata(None, None), None, 0.0)
                .unwrap_err()
                .to_string()
                .contains("unambiguous conversation lineage")
        );
    }
    #[test]
    fn native_index_observes_physical_stores_and_removals() {
        let mut policy = placement("session");
        let a = request("a", None);
        let hashes = ReplayRequestHashes::from_tokens(&a.prompt_tokens(), 4);
        policy
            .observe(
                events::Events(vec![(
                    7,
                    KvEvent {
                        event_id: 1,
                        dp_rank: 1,
                        data: KvEventData::Stored(StoredBlocks {
                            parent_hash: None,
                            start_position: None,
                            blocks: hashes
                                .sequence_hashes
                                .iter()
                                .zip(&hashes.local_block_hashes)
                                .map(|(&block_hash, &tokens_hash)| KvBlock {
                                    block_hash,
                                    tokens_hash,
                                    token_ids: None,
                                })
                                .collect(),
                        }),
                    },
                )]),
                0.0,
            )
            .unwrap();
        let chosen = choose(&mut policy, &a, 0.0);
        assert_eq!(
            chosen.scheduler_id, 3,
            "native selector must choose the physical warm worker/DP"
        );
        assert_eq!(chosen.cache_sample.unwrap().overlap_blocks, 4);
        policy.dispatch_committed(chosen.request_id, 0.0).unwrap();
        policy.request_terminal(chosen.request_id, 1.0).unwrap();
        policy
            .observe(
                events::Events(vec![(
                    7,
                    KvEvent {
                        event_id: 2,
                        dp_rank: 1,
                        data: KvEventData::Removed {
                            block_hashes: hashes.sequence_hashes,
                        },
                    },
                )]),
                2.0,
            )
            .unwrap();
        let next = choose(&mut policy, &request("b", None), 2.0);
        assert_eq!(next.cache_sample.unwrap().best_available_overlap_blocks, 0);
    }
    #[test]
    fn binding_index_prunes_native_expired_entries_but_preserves_active_leases() {
        let mut policy = placement("session");
        let live = request("live", None);
        let idle = request("idle", None);
        let live_key = policy.group_key(&live, None).unwrap().unwrap();
        let idle_key = policy.group_key(&idle, None).unwrap().unwrap();
        let live = choose(&mut policy, &live, 0.0);
        policy.dispatch_committed(live.request_id, 0.0).unwrap();
        let idle = choose(&mut policy, &idle, 0.0);
        policy.dispatch_committed(idle.request_id, 0.0).unwrap();
        policy.request_terminal(idle.request_id, 0.0).unwrap();
        assert_eq!(policy.bindings.len(), 2);
        policy.advance(9_000.0).unwrap();
        assert_eq!(policy.bindings.len(), 2);
        policy.advance(11_000.0).unwrap();
        assert!(policy.bindings.contains_key(&live_key));
        assert!(!policy.bindings.contains_key(&idle_key));
        assert_eq!(policy.bindings.len(), 1);
        policy.request_terminal(live.request_id, 11_000.0).unwrap();
        policy.advance(22_000.0).unwrap();
        assert!(policy.bindings.is_empty());
        assert_eq!(policy.pending_count(), 0);
        assert!(
            policy.next_wakeup_ms().is_none(),
            "idle housekeeping cannot prolong replay"
        );
    }
    #[test]
    fn routing_evidence_respects_request_capture_and_preserves_bounded_counters() {
        for (topology, expected) in [
            (
                json!({"kind":"aggregated","workers":{"initial_workers":2}}),
                json!({"aggregated":32}),
            ),
            (
                json!({"kind":"disaggregated","prefill":{"initial_workers":2},"decode":{"initial_workers":2}}),
                json!({"prefill":32,"decode":32}),
            ),
        ] {
            let mut summaries = Vec::new();
            for capture in [false, true] {
                let payload = json!({"topology":topology,"record_per_request":capture,
                    "engine":{"dp_size":2,"rank":{"block_size":4,"num_gpu_blocks":128}},
                    "requests":(0..32).map(|i|json!({"id":i.to_string(),"arrival_time_ms":i as f64 * 100.0,
                        "input_tokens":16,"input_token_ids":(1..=16).collect::<Vec<_>>(),"output_tokens":2,
                        "session_id":"session"})).collect::<Vec<_>>()});
                let result: Value = serde_json::from_str(
                    &execute(
                        &payload.to_string(),
                        r#"{"policy":"kv_router","affinity":{"mode":"session","ttl_seconds":10}}"#,
                    )
                    .unwrap(),
                )
                .unwrap();
                let evidence = &result["dynamo_policy"];
                let count = evidence["decision_count"].as_u64().unwrap();
                assert_eq!(evidence["decisions_by_role"], expected);
                assert_eq!(
                    count,
                    expected
                        .as_object()
                        .unwrap()
                        .values()
                        .map(|n| n.as_u64().unwrap())
                        .sum::<u64>()
                );
                assert_eq!(evidence["decisions_captured"], capture);
                assert_eq!(
                    evidence["decisions"].as_array().unwrap().len() as u64,
                    if capture { count } else { 0 }
                );
                assert!(evidence["physical_kv_events"].as_u64().unwrap() > 0);
                summaries.push((
                    evidence["decision_count"].clone(),
                    evidence["decisions_by_role"].clone(),
                    evidence["physical_kv_events"].clone(),
                ));
            }
            assert_eq!(
                summaries[0], summaries[1],
                "capture changes storage only, never execution"
            );
        }
    }
    #[test]
    fn canonical_engine_execution_uses_native_policy_in_both_topologies() {
        for topology in [
            json!({"kind":"aggregated","workers":{"initial_workers":2}}),
            json!({"kind":"disaggregated","prefill":{"initial_workers":2},"decode":{"initial_workers":2}}),
        ] {
            let payload = json!({"topology":topology,"engine":{"dp_size":2,"rank":{"block_size":4,"num_gpu_blocks":128}},"requests":[
                {"id":"first","arrival_time_ms":0.0,"input_tokens":16,"input_token_ids":(1..=16).collect::<Vec<_>>(),"output_tokens":4,"session_id":"session"},
                {"id":"next","arrival_time_ms":100.0,"input_tokens":16,"input_token_ids":(1..=16).collect::<Vec<_>>(),"output_tokens":4,"session_id":"session"}
            ]});
            let result: Value = serde_json::from_str(
                &execute(
                    &payload.to_string(),
                    r#"{"policy":"kv_router","affinity":{"mode":"session"}}"#,
                )
                .unwrap(),
            )
            .unwrap();
            assert!(
                result["per_request"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|r| r["reused_input_tokens"].as_u64().unwrap_or(0) > 0),
                "engine must actually reuse physical cache, {result}"
            );
            let policy = &result["dynamo_policy"];
            assert!(policy["physical_kv_events"].as_u64().unwrap() > 0);
            let decisions = policy["decisions"].as_array().unwrap();
            assert!(
                decisions
                    .iter()
                    .any(|d| d["overlap_blocks"].as_u64().unwrap() > 0),
                "{result}"
            );
            for role in ["aggregated", "prefill", "decode"] {
                let ds = decisions
                    .iter()
                    .filter(|d| d["role"] == role)
                    .collect::<Vec<_>>();
                if ds.len() == 2 {
                    assert_eq!(ds[0]["worker_id"], ds[1]["worker_id"]);
                    assert_eq!(ds[0]["dp_rank"], ds[1]["dp_rank"]);
                }
            }
        }
    }
}
