// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport for independently installed placement policies. The replay owns
//! execution and event visibility; the external object owns placement state.

use std::cell::RefCell;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::engine::{KvEvent, KvEventData};
use crate::replay::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::replay::{
    EngineEventBatch, Placement, PlacementDecision, PlacementEffects, PlacementPolicy,
    ReplayAdmissionMetadata, ReplayCaptureOptions, ReplayComposition, ReplayDeterminism,
    ReplayEngineConfig, ReplayEngineObservation, ReplayError, ReplayResult, ReplaySpec, Replayer,
    WorkerStage, WorkerTopology,
};

/// Keep every owned reference's destruction on the synchronous replay thread.
/// Dropping a plain `Py` without the GIL queues its decref in PyO3's global
/// reference pool, which another thread can flush. External native policies may
/// be unsendable, so even error cleanup must acquire the GIL before decref.
struct PolicyHandle(Option<Py<PyAny>>);

impl PolicyHandle {
    fn new(object: Py<PyAny>) -> Self {
        Self(Some(object))
    }

    fn clone_owned(&self) -> Self {
        Python::with_gil(|py| Self::new(self.clone_ref(py)))
    }
}

impl Deref for PolicyHandle {
    type Target = Py<PyAny>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live policy handle")
    }
}

impl Drop for PolicyHandle {
    fn drop(&mut self) {
        Python::with_gil(|_| drop(self.0.take()));
    }
}

#[derive(Default)]
struct SharedState {
    error: Option<String>,
    evidence: Map<String, Value>,
    policies: Vec<(&'static str, PolicyHandle)>,
}

type Shared = Arc<Mutex<SharedState>>;

fn check_state(shared: &Shared) -> Result<()> {
    let state = shared
        .lock()
        .map_err(|_| anyhow::anyhow!("policy state lock poisoned"))?;
    if let Some(error) = &state.error {
        bail!("external placement policy failed: {error}");
    }
    Ok(())
}

/// Execute through the same preparation, event loop and report finalization as
/// the built-in composition. No engine handles or Rust traits cross modules.
pub(super) fn execute(payload: &str, factory: Py<PyAny>) -> Result<String> {
    let shared = Shared::default();
    let composition = ExternalComposition {
        factory: PolicyHandle::new(factory),
        spec: RefCell::new(None),
        determinism: ReplayDeterminism::default(),
        shared: shared.clone(),
    };
    let mut output = super::execute_replay_with_dispatch(
        payload,
        false,
        move |spec, factory, input, _, determinism, tokenization| {
            let mut replayer = Replayer::with_composition(spec, factory, composition)?
                .with_capture_options(ReplayCaptureOptions {
                    determinism,
                    ..Default::default()
                });
            if let Some(input) = input {
                replayer = replayer.with_runtime_input(input);
            }
            if let Some(tokenization) = tokenization {
                replayer = replayer.with_tokenization(
                    tokenization.workers,
                    tokenization.base_ms,
                    tokenization.per_input_token_ms,
                )?;
            }
            replayer.run().map(|report| (report, None))
        },
    )?;
    // cancel_pending has an infallible legacy trait signature. An exception in
    // that callback must still invalidate an otherwise successful final report.
    check_state(&shared)?;
    // Request the potentially large decision evidence once, after execution.
    // Returning it with every lifecycle callback would make capture quadratic.
    let policies = std::mem::take(&mut shared.lock().unwrap().policies);
    for (role, object) in policies {
        ExternalPlacement {
            object,
            role,
            pending_count: 0,
            next_wakeup_ms: None,
            shared: shared.clone(),
        }
        .call::<()>(json!({"op": "snapshot"}))?;
    }
    output.report_fields.insert(
        "external_policy".into(),
        Value::Object(std::mem::take(&mut shared.lock().unwrap().evidence)),
    );
    output.into_json()
}

struct ExternalComposition {
    factory: PolicyHandle,
    spec: RefCell<Option<(ReplayEngineConfig, Value, bool)>>,
    determinism: ReplayDeterminism,
    shared: Shared,
}

impl ExternalComposition {
    fn placement(
        &self,
        stage: WorkerStage,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<ExternalPlacement> {
        check_state(&self.shared)?;
        let spec = self.spec.borrow();
        let (engine, placement, capture) =
            spec.as_ref().context("policy spec was not validated")?;
        let role = stage_name(stage);
        let context = json!({
            "schema_version": 1,
            "role": role,
            "dp_size": dp_size,
            "topology": topology,
            "engine": engine.role(stage),
            "placement": placement,
            "selector_seed": self.determinism.selector_seed(),
            "capture_decisions": capture,
        });
        let object = Python::with_gil(|py| self.factory.call1(py, (context.to_string(),)))
            .with_context(|| format!("constructing external {role} placement policy"))?;
        let mut policy = ExternalPlacement {
            object: PolicyHandle::new(object),
            role,
            pending_count: 0,
            next_wakeup_ms: None,
            shared: self.shared.clone(),
        };
        policy.call::<()>(json!({"op": "status"}))?;
        self.shared
            .lock()
            .unwrap()
            .policies
            .push((role, policy.object.clone_owned()));
        Ok(policy)
    }
}

impl ReplayComposition for ExternalComposition {
    type Metadata = ExternalMetadata;
    type Observation = ExternalObservation;
    type AggregatedPlacement = ExternalPlacement;
    type DisaggregatedPlacement = ExternalPlacement;

    fn validate_spec(&self, spec: &ReplaySpec) -> ReplayResult<()> {
        if spec.adapters.scaling.provider != "none" || !spec.adapters.scaling.config.is_null() {
            return Err(ReplayError::InvalidSpec(
                "external placement composition does not provide scaling".into(),
            ));
        }
        *self.spec.borrow_mut() = Some((
            ReplayEngineConfig::parse(&spec.engine)?,
            serde_json::to_value(&spec.adapters.placement)
                .map_err(|error| ReplayError::InvalidSpec(error.to_string()))?,
            spec.record_per_request,
        ));
        Ok(())
    }

    fn create_aggregated_placement(
        &mut self,
        dp_size: u32,
        topology: Vec<WorkerTopology>,
    ) -> Result<ExternalPlacement> {
        self.placement(WorkerStage::Aggregated, dp_size, topology)
    }

    fn create_disaggregated_placements(
        &mut self,
        prefill_dp_size: u32,
        prefill_topology: Vec<WorkerTopology>,
        decode_dp_size: u32,
        decode_topology: Vec<WorkerTopology>,
    ) -> Result<(ExternalPlacement, ExternalPlacement)> {
        Ok((
            self.placement(WorkerStage::Prefill, prefill_dp_size, prefill_topology)?,
            self.placement(WorkerStage::Decode, decode_dp_size, decode_topology)?,
        ))
    }

    fn set_determinism(&mut self, determinism: ReplayDeterminism) -> ReplayResult<()> {
        self.determinism = determinism;
        Ok(())
    }
}

struct ExternalMetadata {
    hashes: Option<ReplayRequestHashes>,
    output_limit: Option<usize>,
}

impl ReplayAdmissionMetadata for ExternalMetadata {
    fn from_hashes(hashes: Option<ReplayRequestHashes>) -> Self {
        Self {
            hashes,
            output_limit: None,
        }
    }

    fn for_prefill(mut self) -> Self {
        self.output_limit = Some(1);
        self
    }

    fn max_output_tokens_override(&self) -> Option<usize> {
        self.output_limit
    }

    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        self.hashes
    }
}

#[derive(Default, Serialize)]
struct ObservationBatch(Vec<Observation>);

#[derive(Serialize)]
struct Observation {
    stage: &'static str,
    worker_id: usize,
    dp_rank: u32,
    events: Vec<KvEvent>,
}

impl EngineEventBatch for ObservationBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

struct ExternalObservation;

impl ReplayEngineObservation for ExternalObservation {
    type Batch = ObservationBatch;
    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn observe_engine_events(
        stage: WorkerStage,
        worker_id: usize,
        dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> ObservationBatch {
        if events.is_empty() {
            ObservationBatch::default()
        } else {
            ObservationBatch(vec![Observation {
                stage: stage_name(stage),
                worker_id,
                dp_rank,
                events,
            }])
        }
    }

    fn stored_hashes(batch: &ObservationBatch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|observation| &observation.events)
            .flat_map(|event| match &event.data {
                KvEventData::Stored(stored) => stored.blocks.as_slice(),
                KvEventData::Removed { .. } => &[],
            })
            .map(|block| block.tokens_hash)
            .collect()
    }
}

#[derive(Deserialize)]
struct Envelope {
    result: Value,
    pending_count: usize,
    next_wakeup_ms: Option<f64>,
    evidence: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
struct WireEffects {
    placement: Option<Placement>,
    released: Vec<Placement>,
}

struct ExternalPlacement {
    object: PolicyHandle,
    role: &'static str,
    pending_count: usize,
    next_wakeup_ms: Option<f64>,
    shared: Shared,
}

impl ExternalPlacement {
    fn call<T: DeserializeOwned>(&mut self, command: Value) -> Result<T> {
        let invoke = || -> Result<(T, Envelope)> {
            // Cleanup callbacks remain callable after a partial policy commit
            // failed. The original poison still prevents a successful report.
            if !matches!(
                command["op"].as_str(),
                Some("dispatch_aborted" | "cancel_pending")
            ) {
                check_state(&self.shared)?;
            }
            let encoded = Python::with_gil(|py| -> PyResult<String> {
                self.object
                    .call_method1(py, "call_json", (command.to_string(),))?
                    .extract(py)
            })?;
            let value: Value =
                serde_json::from_str(&encoded).context("invalid external policy response JSON")?;
            ensure!(
                value.get("next_wakeup_ms").is_some(),
                "external policy response omitted next_wakeup_ms"
            );
            let envelope: Envelope = serde_json::from_value(value)?;
            if command["op"] == "snapshot" {
                ensure!(
                    envelope.evidence.is_some(),
                    "external policy snapshot omitted evidence"
                );
            }
            if let Some(wakeup) = envelope.next_wakeup_ms {
                ensure!(
                    wakeup.is_finite() && wakeup >= 0.0,
                    "external policy wakeup must be finite and nonnegative"
                );
            }
            let result = serde_json::from_value(envelope.result.clone())?;
            Ok((result, envelope))
        };
        match invoke().with_context(|| format!("external {} policy {}", self.role, command["op"])) {
            Ok((result, envelope)) => {
                self.pending_count = envelope.pending_count;
                self.next_wakeup_ms = envelope.next_wakeup_ms;
                if let Some(evidence) = envelope.evidence {
                    self.shared
                        .lock()
                        .unwrap()
                        .evidence
                        .insert(self.role.into(), Value::Object(evidence));
                }
                Ok(result)
            }
            Err(error) => {
                self.shared
                    .lock()
                    .unwrap()
                    .error
                    .get_or_insert_with(|| format!("{error:#}"));
                Err(error)
            }
        }
    }
}

impl PlacementPolicy<ReplayRequestPayload> for ExternalPlacement {
    type Metadata = ExternalMetadata;
    type Observation = ObservationBatch;

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: ExternalMetadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        let effects: WireEffects = self.call(json!({
            "op": "place", "now_ms": now_ms, "request": request.metadata(),
            "input_length": request.input_length(), "hashes": metadata.hashes,
            "max_output_tokens_override": metadata.output_limit, "session_id": session_id,
        }))?;
        Ok(PlacementEffects {
            decision: effects
                .placement
                .map_or(PlacementDecision::Queued, PlacementDecision::Immediate),
            released: effects.released,
        })
    }

    fn observe(&mut self, observation: ObservationBatch, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "observe", "now_ms": now_ms, "events": observation}))
    }

    fn dispatch_committed(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.call(json!({"op": "dispatch_committed", "request_id": request_id, "now_ms": now_ms}))
    }

    fn dispatch_aborted(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.call(json!({"op": "dispatch_aborted", "request_id": request_id, "now_ms": now_ms}))
    }

    fn advance_clock(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "advance_clock", "now_ms": now_ms}))
    }

    fn next_wakeup_ms(&self) -> Option<f64> {
        self.next_wakeup_ms
    }

    fn pending_count(&self) -> usize {
        self.pending_count
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.call(json!({"op": "cancel_pending", "request_id": request_id}))
            .unwrap_or(false)
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "request_terminal", "request_id": request_id, "now_ms": now_ms}))
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "prefill_completed", "request_id": request_id, "now_ms": now_ms}))
    }

    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "worker_ready", "now_ms": now_ms, "worker": worker}))
    }

    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "worker_draining", "now_ms": now_ms, "worker": worker}))
    }

    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "worker_removed", "now_ms": now_ms, "worker": worker}))
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.call(json!({"op": "topology_settled", "now_ms": now_ms}))
    }
}

fn stage_name(stage: WorkerStage) -> &'static str {
    match stage {
        WorkerStage::Aggregated => "aggregated",
        WorkerStage::Prefill => "prefill",
        WorkerStage::Decode => "decode",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{KvBlock, KvEventTier, StoredBlocks};
    use crate::replay::{DirectRequest, ReplayTopology, WorkerPoolSpec};
    use pyo3::exceptions::PyValueError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct DropCounts {
        factories: AtomicUsize,
        policies: AtomicUsize,
    }

    #[pyclass(unsendable)]
    struct RecordingFactory {
        fail_op: Option<String>,
        drops: Option<Arc<DropCounts>>,
    }

    impl Drop for RecordingFactory {
        fn drop(&mut self) {
            if let Some(drops) = &self.drops {
                drops.factories.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[pymethods]
    impl RecordingFactory {
        fn __call__(&self, py: Python<'_>, context: &str) -> PyResult<Py<RecordingPolicy>> {
            if self.fail_op.as_deref() == Some("construct") {
                return Err(PyValueError::new_err("injected construction failure"));
            }
            Py::new(
                py,
                RecordingPolicy {
                    context: serde_json::from_str(context).unwrap(),
                    calls: Vec::new(),
                    fail_op: self.fail_op.clone(),
                    drops: self.drops.clone(),
                },
            )
        }
    }

    #[pyclass(unsendable)]
    struct RecordingPolicy {
        context: Value,
        calls: Vec<Value>,
        fail_op: Option<String>,
        drops: Option<Arc<DropCounts>>,
    }

    impl Drop for RecordingPolicy {
        fn drop(&mut self) {
            if let Some(drops) = &self.drops {
                drops.policies.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[pymethods]
    impl RecordingPolicy {
        fn call_json(&mut self, command: &str) -> PyResult<String> {
            let command: Value = serde_json::from_str(command).unwrap();
            let op = command["op"].as_str().unwrap();
            if self.fail_op.as_deref() == Some(op) {
                return Err(PyValueError::new_err(format!("injected {op} failure")));
            }
            let result = match op {
                "place" => json!({"placement": {
                    "request_id": command["request"]["uuid"],
                    "scheduler_id": self.context["topology"][0]["scheduler_ids"][0],
                    "reported_overlap_tokens": 0,
                }, "released": []}),
                "status" | "snapshot" | "dispatch_committed" | "dispatch_aborted" => Value::Null,
                "cancel_pending" => json!(true),
                _ => json!([]),
            };
            let snapshot = op == "snapshot";
            self.calls.push(command);
            let mut envelope =
                json!({"result": result, "pending_count": 0, "next_wakeup_ms": null});
            if snapshot {
                envelope["evidence"] = json!({"construction": self.context, "calls": self.calls});
            }
            Ok(envelope.to_string())
        }
    }

    fn factory(fail_op: Option<&str>) -> Py<PyAny> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            Py::new(
                py,
                RecordingFactory {
                    fail_op: fail_op.map(str::to_owned),
                    drops: None,
                },
            )
            .unwrap()
            .into_any()
        })
    }

    fn policy(fail_op: Option<&str>) -> ExternalPlacement {
        let composition = ExternalComposition {
            factory: PolicyHandle::new(factory(fail_op)),
            spec: RefCell::new(Some((
                ReplayEngineConfig::default(),
                json!({"provider": "test_policy", "config": {}}),
                true,
            ))),
            determinism: ReplayDeterminism::default(),
            shared: Shared::default(),
        };
        composition
            .placement(
                WorkerStage::Aggregated,
                1,
                vec![WorkerTopology {
                    worker_id: 0,
                    scheduler_ids: vec![0],
                    router_hint: false,
                }],
            )
            .unwrap()
    }

    fn snapshot(policy: &mut ExternalPlacement) -> Value {
        policy.call::<()>(json!({"op": "snapshot"})).unwrap();
        policy.shared.lock().unwrap().evidence[policy.role].clone()
    }

    #[test]
    fn external_policy_uses_shared_execution_for_both_topologies() {
        for topology in [
            ReplayTopology::aggregated(1),
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            },
        ] {
            let payload = json!({"spec": {
                "topology": topology,
                "adapters": {"placement": {"provider": "test_policy", "config": {"opaque": 42}}},
                "engine": {"rank": {"num_gpu_blocks": 16, "block_size": 4, "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
                "requests": [{"id": "turn", "arrival_time_ms": 0.0, "input_tokens": 8, "output_tokens": 2}],
            }, "determinism": "canonical_v1", "tokenization": {"workers": 1, "base_ms": 2.0, "per_input_token_ms": 0.0}}).to_string();
            let output: Value =
                serde_json::from_str(&execute(&payload, factory(None)).unwrap()).unwrap();
            assert_eq!(output["per_request"].as_array().unwrap().len(), 1);
            for (role, evidence) in output["external_policy"].as_object().unwrap() {
                assert_eq!(evidence["construction"]["role"], *role);
                assert_eq!(
                    evidence["construction"]["selector_seed"],
                    crate::replay::CANONICAL_SELECTOR_SEED
                );
                assert_eq!(
                    evidence["construction"]["placement"]["config"]["opaque"],
                    42
                );
                assert_eq!(
                    evidence["construction"]["engine"]["rank"]["num_gpu_blocks"],
                    16
                );
                let calls = evidence["calls"].as_array().unwrap();
                let commit = calls
                    .iter()
                    .position(|call| call["op"] == "dispatch_committed")
                    .unwrap();
                let terminal = calls
                    .iter()
                    .position(|call| call["op"] == "request_terminal")
                    .unwrap();
                assert!(commit < terminal);
                assert_eq!(calls.last().unwrap()["op"], "snapshot");
                assert_eq!(
                    calls.iter().filter(|call| call["op"] == "snapshot").count(),
                    1
                );
                if role != "decode" {
                    assert!(calls.iter().any(|call| call["op"] == "observe"));
                }
            }
        }
    }

    #[test]
    fn external_policy_preserves_deferred_hashes_and_prefill_limit() {
        let mut policy = policy(None);
        let request = ReplayRequestPayload::deferred(
            DirectRequest {
                uuid: Some(Uuid::from_u128(7)),
                max_output_tokens: 12,
                ..Default::default()
            },
            1_000_000,
            vec![9],
            1_000_000,
        );
        let hashes = ReplayRequestHashes {
            local_block_hashes: vec![81],
            sequence_hashes: vec![82],
        };
        let metadata = ExternalMetadata::from_hashes(Some(hashes)).for_prefill();
        policy
            .place(&request, metadata, Some("conversation".into()), 7.5)
            .unwrap();
        let evidence = snapshot(&mut policy);
        let command = &evidence["calls"][1];
        assert_eq!(command["request"]["tokens"], json!([]));
        assert_eq!(command["input_length"], 1_000_000);
        assert_eq!(command["hashes"]["sequence_hashes"], json!([82]));
        assert_eq!(command["max_output_tokens_override"], 1);
        assert_eq!(command["session_id"], "conversation");
        assert_eq!(command["now_ms"], 7.5);
    }

    #[test]
    fn external_policy_preserves_physical_kv_event_identity_and_tiers() {
        let mut policy = policy(None);
        let events = vec![
            KvEvent {
                event_id: 6,
                dp_rank: 2,
                tier: KvEventTier::HostPinned,
                data: KvEventData::Stored(StoredBlocks {
                    parent_hash: Some(5),
                    start_position: None,
                    session_id: Some("session".into()),
                    blocks: vec![KvBlock {
                        block_hash: 7,
                        tokens_hash: 8,
                        token_ids: Some(vec![9]),
                    }],
                }),
            },
            KvEvent {
                event_id: 7,
                dp_rank: 2,
                tier: KvEventTier::Device,
                data: KvEventData::Removed {
                    block_hashes: vec![10],
                },
            },
        ];
        let expected = serde_json::to_value(&events).unwrap();
        let observation =
            ExternalObservation::observe_engine_events(WorkerStage::Prefill, 3, 2, events);
        assert_eq!(ExternalObservation::stored_hashes(&observation), vec![8]);
        policy.observe(observation, 4.5).unwrap();
        let evidence = snapshot(&mut policy);
        let observation = &evidence["calls"][1]["events"][0];
        assert_eq!(observation["events"], expected);
        assert_eq!(observation["stage"], "prefill");
        assert_eq!(observation["worker_id"], 3);
        assert_eq!(observation["dp_rank"], 2);
    }

    #[test]
    fn external_policy_infallible_cancel_failure_poison_is_not_hidden() {
        let mut policy = policy(Some("cancel_pending"));
        assert!(!policy.cancel_pending(Uuid::from_u128(1)));
        assert!(
            check_state(&policy.shared)
                .unwrap_err()
                .to_string()
                .contains("injected cancel_pending failure")
        );
        assert!(policy.advance_clock(1.0).is_err());
    }

    #[test]
    fn external_policy_partial_commit_still_allows_abort_cleanup() {
        let mut policy = policy(Some("dispatch_committed"));
        assert!(policy.dispatch_committed(Uuid::from_u128(1), 0.0).is_err());
        policy.dispatch_aborted(Uuid::from_u128(1), 0.0).unwrap();
        assert!(check_state(&policy.shared).is_err());
        Python::with_gil(|py| {
            let recorder = policy
                .object
                .bind(py)
                .extract::<PyRef<'_, RecordingPolicy>>()
                .unwrap();
            assert_eq!(recorder.calls.last().unwrap()["op"], "dispatch_aborted");
        });
    }

    #[test]
    fn external_policy_drops_unsendable_handles_before_leaving_execution_thread() {
        pyo3::prepare_freethreaded_python();
        let payload = json!({
            "topology": ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(), decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            },
            "adapters": {"placement": {"provider": "test_policy", "config": {}}},
            "engine": {"rank": {"num_gpu_blocks": 16, "block_size": 4,
                "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}}},
            "requests": [{"id": "turn", "arrival_time_ms": 0.0, "input_tokens": 8, "output_tokens": 2}],
        }).to_string();
        for (fail_op, expected_policies) in [
            (None, 2),
            (Some("construct"), 0),
            (Some("status"), 1),
            (Some("place"), 2),
            (Some("dispatch_committed"), 2),
            (Some("snapshot"), 2),
        ] {
            let drops = Arc::new(DropCounts::default());
            Python::with_gil(|py| {
                let factory = Py::new(
                    py,
                    RecordingFactory {
                        fail_op: fail_op.map(str::to_owned),
                        drops: Some(drops.clone()),
                    },
                )
                .unwrap()
                .into_any();
                py.allow_threads(|| {
                    let output = execute(&payload, factory);
                    assert_eq!(output.is_ok(), fail_op.is_none(), "{fail_op:?}");
                    // Assert before reacquiring the caller's GIL: deferred PyO3
                    // decrefs must not await another thread's reference-pool flush.
                    assert_eq!(drops.factories.load(Ordering::SeqCst), 1, "{fail_op:?}");
                    assert_eq!(
                        drops.policies.load(Ordering::SeqCst),
                        expected_policies,
                        "{fail_op:?}"
                    );
                });
            });
        }
    }
}
