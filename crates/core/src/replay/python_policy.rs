// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Python policy bridge for optional routing providers. Engines and replay stay
//! in this extension; providers receive owned metadata and native KV events.

mod events;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::replay::{
    AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1, Placement, PlacementCacheSample, PlacementDecision,
    PlacementEffects, PlacementPolicy, ReplayAdmissionMetadata, ReplayComposition,
    ReplayDeterminism, ReplayEngineConfig, ReplayError, ReplayPromptTokenSource, ReplaySpec,
    WorkerTopology,
};

/// Version of the owned JSON protocol, independent of either native library ABI.
pub const POLICY_API_VERSION: u32 = 1;

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

/// Neutral provider evidence retained after replay consumes the composition.
#[derive(Clone, Default)]
pub struct PythonPolicyEvidence(Arc<Mutex<Vec<ProviderEvidence>>>);
struct ProviderEvidence {
    role: &'static str,
    provider: Py<PyAny>,
    fault: Arc<Mutex<Option<String>>>,
}
impl PythonPolicyEvidence {
    pub fn snapshot(&self) -> Result<Value> {
        let providers = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("policy evidence lock poisoned"))?;
        let mut roles = serde_json::Map::new();
        for ProviderEvidence {
            role,
            provider,
            fault,
        } in providers.iter()
        {
            let callback_fault = fault
                .lock()
                .map_err(|_| anyhow::anyhow!("policy error lock poisoned"))?;
            ensure!(
                callback_fault.is_none(),
                "routing provider failed: {}",
                callback_fault.as_deref().unwrap_or_default()
            );
            drop(callback_fault);
            let payload: String =
                Python::with_gil(|py| provider.call_method0(py, "evidence")?.extract(py))
                    .with_context(|| format!("{role} policy evidence failed"))?;
            roles.insert(
                (*role).into(),
                serde_json::from_str(&payload).context("invalid policy evidence JSON")?,
            );
        }
        Ok(json!({"api_version": POLICY_API_VERSION, "roles": roles}))
    }
}

/// One provider factory creates independent aggregated or prefill/decode hosts.
/// The factory never receives an AISimulate engine handle.
pub struct PythonPolicyComposition {
    factory: Py<PyAny>,
    config_json: String,
    engine: RefCell<Option<ReplayEngineConfig>>,
    capture_decisions: RefCell<bool>,
    evidence: PythonPolicyEvidence,
}
impl PythonPolicyComposition {
    pub fn new(factory: Py<PyAny>, config_json: String) -> Self {
        Self {
            factory,
            config_json,
            engine: RefCell::new(None),
            capture_decisions: RefCell::new(false),
            evidence: PythonPolicyEvidence::default(),
        }
    }
    pub fn evidence(&self) -> PythonPolicyEvidence {
        self.evidence.clone()
    }
    fn create(
        &self,
        role: &'static str,
        rank: &crate::engine::EngineConfig,
        dp: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<PythonPlacement> {
        let block_size: u32 = rank
            .block_size
            .try_into()
            .context("routing block size exceeds wire protocol range")?;
        ensure!(block_size > 0, "routing block size must be positive");
        ensure!(
            topology
                .iter()
                .all(|worker| worker.scheduler_ids.len() == dp as usize),
            "worker scheduler topology does not match attention DP"
        );
        let workers = serde_json::to_string(&json!({
            "api_version": POLICY_API_VERSION, "dp_size": dp,
            "block_size": rank.block_size, "total_kv_blocks": rank.num_gpu_blocks,
            "max_num_batched_tokens": rank.max_num_batched_tokens,
            "host_offload": rank.native_host_offload.is_some(), "g3_offload": rank.g3_offload.is_some(),
            "capture_decisions": *self.capture_decisions.borrow(),
            "workers": topology.iter().map(|worker| json!({"worker_id": worker.worker_id})).collect::<Vec<_>>(),
        }))?;
        let provider =
            Python::with_gil(|py| self.factory.call1(py, (role, &self.config_json, workers)))
                .with_context(|| format!("creating {role} routing policy failed"))?;
        let fault = Arc::new(Mutex::new(None));
        self.evidence
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("policy evidence lock poisoned"))?
            .push(ProviderEvidence {
                role,
                provider: Python::with_gil(|py| provider.clone_ref(py)),
                fault: fault.clone(),
            });
        Ok(PythonPlacement {
            provider,
            fault,
            now_ms: 0.0,
            block_size,
            topology: topology
                .into_iter()
                .map(|worker| (worker.worker_id, worker.scheduler_ids))
                .collect(),
        })
    }
}
impl ReplayComposition for PythonPolicyComposition {
    type Metadata = Metadata;
    type Observation = events::Observation;
    type AggregatedPlacement = PythonPlacement;
    type DisaggregatedPlacement = PythonPlacement;
    fn validate_spec(&self, spec: &ReplaySpec) -> crate::replay::ReplayResult<()> {
        let validate = || -> Result<()> {
            let empty = |value: &Value| {
                value.is_null() || value.as_object().is_some_and(|object| object.is_empty())
            };
            ensure!(
                matches!(
                    spec.adapters.placement.provider.as_str(),
                    "round_robin" | "external_policy"
                ) && empty(&spec.adapters.placement.config),
                "unsupported placement descriptor for Python policy integration; pass policy configuration through the provider argument"
            );
            ensure!(
                spec.adapters.scaling.provider == "none" && empty(&spec.adapters.scaling.config),
                "Python routing policy integration does not support dynamic scaling"
            );
            let engine = if spec.engine.is_null() {
                ReplayEngineConfig::default()
            } else {
                serde_json::from_value(spec.engine.clone())?
            };
            *self.engine.borrow_mut() = Some(engine);
            *self.capture_decisions.borrow_mut() = spec.record_per_request;
            Ok(())
        };
        validate().map_err(|error| ReplayError::InvalidSpec(format!("{error:#}")))
    }
    fn set_determinism(
        &mut self,
        determinism: ReplayDeterminism,
    ) -> crate::replay::ReplayResult<()> {
        if determinism.selector_seed().is_some() {
            return Err(ReplayError::InvalidSpec(
                "Python routing policy integration does not support seeded selection".into(),
            ));
        }
        Ok(())
    }
    fn create_aggregated_placement(
        &mut self,
        dp: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<PythonPlacement> {
        let engine = self.engine.borrow();
        self.create(
            "aggregated",
            &engine.as_ref().context("missing validated engine")?.rank,
            dp,
            topology,
        )
    }
    fn create_disaggregated_placements(
        &mut self,
        pdp: u32,
        ptop: Vec<WorkerTopology>,
        ddp: u32,
        dtop: Vec<WorkerTopology>,
    ) -> Result<(PythonPlacement, PythonPlacement)> {
        let engine = self.engine.borrow();
        let engine = engine.as_ref().context("missing validated engine")?;
        Ok((
            self.create(
                "prefill",
                engine
                    .prefill
                    .as_ref()
                    .map_or(&engine.rank, |role| &role.rank),
                pdp,
                ptop,
            )?,
            self.create(
                "decode",
                engine
                    .decode
                    .as_ref()
                    .map_or(&engine.rank, |role| &role.rank),
                ddp,
                dtop,
            )?,
        ))
    }
}

#[derive(Deserialize)]
struct Selected {
    request_id: Uuid,
    worker_id: usize,
    dp_rank: u32,
    cached_tokens: usize,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    isl_blocks: u32,
}
#[derive(Deserialize)]
struct Effects {
    decision: Option<Selected>,
    #[serde(default)]
    released: Vec<Selected>,
}

pub struct PythonPlacement {
    provider: Py<PyAny>,
    fault: Arc<Mutex<Option<String>>>,
    now_ms: f64,
    block_size: u32,
    topology: HashMap<usize, Vec<usize>>,
}
impl PythonPlacement {
    fn check_fault(&self) -> Result<()> {
        let fault = self
            .fault
            .lock()
            .map_err(|_| anyhow::anyhow!("policy error lock poisoned"))?;
        ensure!(
            fault.is_none(),
            "routing provider failed: {}",
            fault.as_deref().unwrap_or_default()
        );
        Ok(())
    }
    fn infallible<T>(&self, method: &str, result: PyResult<T>, fallback: T) -> T {
        match result {
            Ok(value) => value,
            Err(error) => {
                if let Ok(mut fault) = self.fault.lock() {
                    if fault.is_none() {
                        *fault = Some(format!("{method}: {error}"));
                    }
                }
                fallback
            }
        }
    }

    fn placement(&self, selected: Selected) -> Result<Placement> {
        let scheduler_id = *self
            .topology
            .get(&selected.worker_id)
            .and_then(|ranks| ranks.get(selected.dp_rank as usize))
            .context("routing provider selected unavailable worker/DP")?;
        Ok(Placement {
            request_id: selected.request_id,
            scheduler_id,
            reported_overlap_tokens: selected.cached_tokens,
            cache_sample: Some(PlacementCacheSample {
                overlap_blocks: selected.overlap_blocks,
                best_available_overlap_blocks: selected.best_available_overlap_blocks,
                isl_blocks: selected.isl_blocks,
            }),
            placement_replica_id: None,
        })
    }
    fn placements(&self, json: &str) -> Result<Vec<Placement>> {
        serde_json::from_str::<Vec<Selected>>(json)
            .context("invalid routing provider placements JSON")?
            .into_iter()
            .map(|selected| self.placement(selected))
            .collect()
    }
    fn lifecycle(&self, method: &str, id: Uuid, now: f64) -> Result<()> {
        self.check_fault()?;
        Python::with_gil(|py| {
            self.provider
                .call_method1(py, method, (id.to_string(), now))
        })
        .with_context(|| format!("routing provider {method} failed"))?;
        Ok(())
    }
    fn lifecycle_released(&self, method: &str, id: Uuid, now: f64) -> Result<Vec<Placement>> {
        self.check_fault()?;
        let result: String = Python::with_gil(|py| {
            self.provider
                .call_method1(py, method, (id.to_string(), now))?
                .extract(py)
        })
        .with_context(|| format!("routing provider {method} failed"))?;
        self.placements(&result)
    }
}
impl PlacementPolicy<ReplayRequestPayload> for PythonPlacement {
    type Metadata = Metadata;
    type Observation = events::Events;
    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        self.check_fault()?;
        self.now_ms = now_ms;
        let meta = request.metadata();
        let id = meta
            .uuid
            .context("routing provider requires request UUID")?;
        let context = meta.replay_context.as_ref();
        let agentic = context.and_then(|context| context.agentic.as_ref());
        let lineage = agentic
            .and_then(|agentic| agentic.lineage.as_ref())
            .filter(|lineage| lineage.schema == AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1);
        let session = agentic
            .map(|agentic| agentic.conversation_id.as_str())
            .or(session_id.as_deref())
            .or_else(|| context.and_then(|context| context.session_id.as_deref()));
        let hashes = metadata.0.unwrap_or_else(|| {
            ReplayRequestHashes::from_tokens(&request.prompt_tokens(), self.block_size)
        });
        let payload = serde_json::to_string(&json!({
            "request_id": id, "input_tokens": request.input_length(), "output_tokens": metadata.1.unwrap_or(meta.max_output_tokens),
            "local_block_hashes": hashes.local_block_hashes, "sequence_hashes": hashes.sequence_hashes,
            "priority": meta.priority, "strict_priority": meta.strict_priority, "policy_class": meta.policy_class,
            "preferred_dp_rank": meta.preferred_dp_rank, "preferred_prefill_dp_rank": meta.preferred_prefill_dp_rank,
            "prompt_token_source": context.map_or(ReplayPromptTokenSource::Materialized, |context| context.prompt_token_source),
            "authored_request_id": context.map(|context| &context.authored_id), "session_id": session,
            "identity": {"scope": agentic.map(|agentic| &agentic.play_id), "session": session,
                "root": lineage.map(|lineage| &lineage.root_conversation_id),
                "parent": lineage.and_then(|lineage| lineage.parent_conversation_id.as_ref()), "lineage_available": lineage.is_some()},
        }))?;
        let result: String = Python::with_gil(|py| {
            self.provider
                .call_method1(py, "place", (payload, now_ms))?
                .extract(py)
        })
        .context("routing provider placement failed")?;
        let effects: Effects =
            serde_json::from_str(&result).context("invalid routing provider effects JSON")?;
        let decision = if let Some(selected) = effects.decision {
            ensure!(
                selected.request_id == id,
                "routing provider returned wrong request identity"
            );
            PlacementDecision::Immediate(self.placement(selected)?)
        } else {
            PlacementDecision::Queued
        };
        Ok(PlacementEffects {
            decision,
            released: effects
                .released
                .into_iter()
                .map(|selected| self.placement(selected))
                .collect::<Result<_>>()?,
        })
    }
    fn observe(&mut self, observation: events::Events, now_ms: f64) -> Result<Vec<Placement>> {
        self.check_fault()?;
        self.now_ms = now_ms;
        let payload = serde_json::to_string(
            &observation
                .0
                .iter()
                .map(|(worker_id, event)| json!({"worker_id": worker_id, "event": event}))
                .collect::<Vec<_>>(),
        )?;
        let result: String = Python::with_gil(|py| {
            self.provider
                .call_method1(py, "observe", (payload, now_ms))?
                .extract(py)
        })
        .context("routing provider KV observation failed")?;
        self.placements(&result)
    }
    fn dispatch_committed(&mut self, id: Uuid, now_ms: f64) -> Result<()> {
        self.now_ms = now_ms;
        self.lifecycle("dispatch_committed", id, now_ms)
    }
    fn dispatch_aborted(&mut self, id: Uuid, now_ms: f64) -> Result<()> {
        self.now_ms = now_ms;
        self.lifecycle("dispatch_aborted", id, now_ms)
    }
    fn advance_clock(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.check_fault()?;
        self.now_ms = now_ms;
        let result: String = Python::with_gil(|py| {
            self.provider
                .call_method1(py, "advance_clock", (now_ms,))?
                .extract(py)
        })
        .context("routing provider clock advance failed")?;
        self.placements(&result)
    }
    fn next_wakeup_ms(&self) -> Option<f64> {
        if self.check_fault().is_err() {
            return Some(self.now_ms);
        }
        self.infallible(
            "next_wakeup_ms",
            Python::with_gil(|py| {
                self.provider
                    .call_method0(py, "next_wakeup_ms")?
                    .extract(py)
            }),
            Some(self.now_ms),
        )
    }
    fn cancel_pending(&mut self, id: Uuid) -> bool {
        self.infallible(
            "cancel_pending",
            Python::with_gil(|py| {
                self.provider
                    .call_method1(py, "cancel_pending", (id.to_string(),))?
                    .extract(py)
            }),
            false,
        )
    }
    fn request_terminal(&mut self, id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.now_ms = now_ms;
        self.lifecycle_released("request_terminal", id, now_ms)
    }
    fn prefill_completed(&mut self, id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.now_ms = now_ms;
        self.lifecycle_released("prefill_completed", id, now_ms)
    }
    fn pending_count(&self) -> usize {
        self.infallible(
            "pending_count",
            Python::with_gil(|py| self.provider.call_method0(py, "pending_count")?.extract(py)),
            1,
        )
    }
    fn worker_ready(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        anyhow::bail!(
            "Python routing policy integration does not support dynamic worker membership"
        )
    }
    fn worker_draining(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        anyhow::bail!(
            "Python routing policy integration does not support dynamic worker membership"
        )
    }
    fn worker_removed(&mut self, _: WorkerTopology, _: f64) -> Result<Vec<Placement>> {
        anyhow::bail!(
            "Python routing policy integration does not support dynamic worker membership"
        )
    }
    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.advance_clock(now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{
        AgenticConversationLineage, AgenticRuntimeIdentity, DirectRequest, ReplayCaptureOptions,
        ReplayRequestContext,
    };
    use std::ffi::CString;

    const PROVIDER: &str = r#"
import json
class Policy:
    live = 0
    def __del__(self): type(self).live -= 1
    def __init__(self, role, config, workers):
        type(self).live += 1
        self.role, self.workers = role, json.loads(workers)
        self.requests, self.events, self.commits, self.terminals = [], [], [], []
    def place(self, payload, now):
        request = json.loads(payload)
        self.requests.append(request)
        return json.dumps({'decision': {'request_id': request['request_id'],
            'worker_id': self.workers['workers'][-1]['worker_id'], 'dp_rank': 1,
            'cached_tokens': 0, 'overlap_blocks': 0, 'best_available_overlap_blocks': 0,
            'isl_blocks': request['input_tokens']//self.workers['block_size']}, 'released': []})
    def observe(self, payload, now):
        self.events += json.loads(payload)
        return '[]'
    def dispatch_committed(self, request_id, now): self.commits.append(request_id)
    def dispatch_aborted(self, request_id, now): pass
    def request_terminal(self, request_id, now):
        self.terminals.append(request_id)
        return '[]'
    def prefill_completed(self, request_id, now): return '[]'
    def advance_clock(self, now): return '[]'
    def next_wakeup_ms(self): return None
    def pending_count(self): return 0
    def cancel_pending(self, request_id): return False
    def evidence(self): return json.dumps({'requests':self.requests,'events':self.events,'commits':self.commits,'terminals':self.terminals})
class BrokenWakeup(Policy):
    def next_wakeup_ms(self): raise ValueError('provider wakeup failed')
class BrokenPlace(Policy):
    def place(self, payload, now): raise ValueError('provider placement failed')
class BrokenCount(Policy):
    def pending_count(self): raise ValueError('provider pending count failed')
class WrongWorker(Policy):
    def place(self, payload, now):
        selected = json.loads(super().place(payload, now))
        selected['decision']['worker_id'] = 999
        return json.dumps(selected)
class WrongDp(Policy):
    def place(self, payload, now):
        selected = json.loads(super().place(payload, now))
        selected['decision']['dp_rank'] = 999
        return json.dumps(selected)
class WrongIdentity(Policy):
    def place(self, payload, now):
        selected = json.loads(super().place(payload, now))
        selected['decision']['request_id'] = '00000000-0000-0000-0000-000000000000'
        return json.dumps(selected)
"#;
    fn factory(name: &str) -> Py<PyAny> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            PyModule::from_code(
                py,
                CString::new(PROVIDER).unwrap().as_c_str(),
                c"policy_test.py",
                c"policy_test",
            )
            .unwrap()
            .getattr(name)
            .unwrap()
            .unbind()
        })
    }
    fn execute(topology: Value, provider: &str) -> Result<(crate::ReplayExecutionResult, Value)> {
        let payload = json!({"topology":topology, "record_per_request":true,
            "engine":{"dp_size":2,"rank":{"block_size":4,"num_gpu_blocks":128}},
            "requests":[
                {"id":"first","arrival_time_ms":0.0,"input_tokens":16,"input_token_ids":(1..=16).collect::<Vec<_>>(),"output_tokens":4},
                {"id":"next","arrival_time_ms":100.0,"input_tokens":16,"input_token_ids":(1..=16).collect::<Vec<_>>(),"output_tokens":4}]});
        let composition = PythonPolicyComposition::new(factory(provider), "{}".into());
        let evidence = composition.evidence();
        let report = crate::execute_replay_with_composition(
            &payload.to_string(),
            composition,
            ReplayCaptureOptions::default(),
            None,
        )?;
        Ok((report, evidence.snapshot()?))
    }
    #[test]
    fn bridge_delivers_real_events_and_dispatch_lifecycle_in_both_topologies() {
        for (topology, roles) in [
            (
                json!({"kind":"aggregated","workers":{"initial_workers":2}}),
                vec!["aggregated"],
            ),
            (
                json!({"kind":"disaggregated","prefill":{"initial_workers":2},"decode":{"initial_workers":2}}),
                vec!["prefill", "decode"],
            ),
        ] {
            let (result, evidence) = execute(topology, "Policy").unwrap();
            assert!(
                result
                    .report
                    .per_request
                    .iter()
                    .any(|record| record.reused_input_tokens > 0)
            );
            for role in roles {
                let data = &evidence["roles"][role];
                assert_eq!(data["requests"].as_array().unwrap().len(), 2);
                assert_eq!(data["commits"].as_array().unwrap().len(), 2);
                assert!(!data["events"].as_array().unwrap().is_empty());
                assert!(
                    data["events"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|entry| entry["event"]["dp_rank"] == 1)
                );
                assert_eq!(
                    data["requests"][0]["output_tokens"],
                    if role == "prefill" { 1 } else { 4 }
                );
            }
        }
    }
    #[test]
    fn provider_errors_fail_execution_without_fallback_or_panics() {
        for (provider, expected) in [
            ("BrokenPlace", "provider placement failed"),
            ("BrokenWakeup", "provider wakeup failed"),
            ("BrokenCount", "provider pending count failed"),
            ("WrongWorker", "selected unavailable worker/DP"),
            ("WrongDp", "selected unavailable worker/DP"),
            ("WrongIdentity", "returned wrong request identity"),
        ] {
            let error = execute(
                json!({"kind":"aggregated","workers":{"initial_workers":2}}),
                provider,
            )
            .unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
    }
    #[test]
    fn bridge_preserves_conversation_lineage_and_authored_hashes() {
        let composition = PythonPolicyComposition::new(factory("Policy"), "{}".into());
        let evidence = composition.evidence();
        let rank = crate::engine::EngineConfig {
            block_size: 4,
            ..Default::default()
        };
        let mut policy = composition
            .create(
                "aggregated",
                &rank,
                2,
                vec![WorkerTopology {
                    worker_id: 7,
                    scheduler_ids: vec![11, 19],
                }],
            )
            .unwrap();
        let request = ReplayRequestPayload::materialized(DirectRequest {
            tokens: vec![1; 8],
            uuid: Some(Uuid::new_v4()),
            max_output_tokens: 5,
            replay_context: Some(ReplayRequestContext {
                authored_id: "authored".into(),
                session_id: Some("transport".into()),
                turn_index: None,
                metadata: Value::Null,
                prompt_token_source: ReplayPromptTokenSource::Materialized,
                agentic: Some(AgenticRuntimeIdentity {
                    request_id: "authored".into(),
                    play_id: "play".into(),
                    conversation_id: "child".into(),
                    lane_id: None,
                    root_id: None,
                    parent_id: None,
                    cache_id: None,
                    lineage: Some(AgenticConversationLineage {
                        schema: AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1.into(),
                        root_conversation_id: "root".into(),
                        parent_conversation_id: Some("parent".into()),
                    }),
                }),
            }),
            ..Default::default()
        });
        let hashes = ReplayRequestHashes::from_tokens(&request.prompt_tokens(), 4);
        let placement = policy
            .place(
                &request,
                Metadata::from_hashes(Some(hashes.clone())),
                Some("ignored-transport".into()),
                0.0,
            )
            .unwrap();
        assert!(matches!(
            placement.decision,
            PlacementDecision::Immediate(Placement {
                scheduler_id: 19,
                ..
            })
        ));
        let snapshot = evidence.snapshot().unwrap();
        let sent = &snapshot["roles"]["aggregated"]["requests"][0];
        assert_eq!(
            sent["identity"],
            json!({"scope":"play","session":"child","root":"root","parent":"parent","lineage_available":true})
        );
        assert_eq!(sent["local_block_hashes"], json!(hashes.local_block_hashes));
        assert_eq!(sent["sequence_hashes"], json!(hashes.sequence_hashes));
    }
    #[test]
    fn provider_lifetime_ends_after_report_and_on_error() {
        for provider in ["Policy", "BrokenPlace"] {
            let factory = factory(provider);
            let composition = PythonPolicyComposition::new(
                Python::with_gil(|py| factory.clone_ref(py)),
                "{}".into(),
            );
            let evidence = composition.evidence();
            let payload = json!({"topology":{"kind":"aggregated","workers":{"initial_workers":1}},
                "engine":{"dp_size":2},"requests":[{"id":"r","arrival_time_ms":0.0,"input_tokens":4,"input_token_ids":[1,2,3,4],"output_tokens":1}]});
            let result = crate::execute_replay_with_composition(
                &payload.to_string(),
                composition,
                ReplayCaptureOptions::default(),
                None,
            );
            assert_eq!(result.is_ok(), provider == "Policy");
            evidence.snapshot().unwrap();
            drop(evidence);
            Python::with_gil(|py| {
                assert_eq!(
                    factory
                        .getattr(py, "live")
                        .unwrap()
                        .extract::<usize>(py)
                        .unwrap(),
                    0
                )
            });
        }
    }
    #[test]
    fn unrelated_adapter_configuration_is_rejected() {
        for adapters in [
            json!({"placement":{"provider":"unknown"}}),
            json!({"placement":{"provider":"round_robin","config":{"ignored":true}}}),
            json!({"scaling":{"provider":"none","config":{"ignored":true}}}),
        ] {
            let composition = PythonPolicyComposition::new(factory("Policy"), "{}".into());
            let spec: ReplaySpec =
                serde_json::from_value(json!({"topology":{"kind":"aggregated","workers":{"initial_workers":1}},"adapters":adapters,"requests":[]})).unwrap();
            assert!(composition.validate_spec(&spec).is_err());
        }
    }
}
