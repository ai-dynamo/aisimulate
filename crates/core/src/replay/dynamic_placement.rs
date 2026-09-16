// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic placement-policy loading for the neutral replay seam.
//!
//! The public loader validates the exported descriptor before it copies the
//! vtable, and the resulting policy keeps the shared library alive until after
//! its opaque instance is destroyed. The adapter carries Replay's routing-safe
//! admission metadata and forwards router-visible KV observations through the
//! negotiated lossless packet callback.

use std::path::Path;
use std::sync::Arc;

use aisimulate_placement_abi::{
    AdmissionDecisionV1, AdmissionMetadataFormatV1, BlockHashSliceV1, ByteSliceV1,
    CAPABILITY_LOSSLESS_KV_EVENTS_V1, KvEventSliceV1, KvEventV1, KvStorageTierV1, KvStoredBlockV1,
    MAX_ADMISSION_METADATA_BYTES_V1, MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1,
    MAX_ADMISSION_PROMPT_TOKEN_IDS_V1, PlacementAdmissionV1, PlacementBatchResultV1,
    PlacementCacheSampleV1, PlacementCreateRequestV1, PlacementDiagnosticSliceV1,
    PlacementHandleV1, PlacementLimitsV1, PlacementMetadataV1, PlacementMutationKindV1,
    PlacementMutationPayloadV1, PlacementMutationSliceV1, PlacementMutationV1,
    PlacementResultSliceV1, PlacementSliceV1, PlacementV1, PluginEntryV1, PluginVTableV1,
    PromptIdentityV1, RequestLifecycleV1, SchedulerIdSliceV1, StatusV1, TokenIdSliceV1,
    WorkerCapacityV1, WorkerTopologySliceV1, WorkerTopologyV1, validate_create_request_v1,
    validate_descriptor_v1,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
use uuid::Uuid;

use super::components::{ReplayAdmissionMetadata, ReplayEngineObservation};
use super::core::{
    EngineEventBatch, Placement, PlacementCacheSample, PlacementDecision, PlacementEffects,
    PlacementPolicy, WorkerTopology,
};
use super::loadgen::{ReplayRequestHashes, ReplayRequestPayload};
use crate::engine::{KvEvent, KvEventData};
use crate::replay::WorkerStage;

/// Fixed export name every V1 placement plugin must provide.
const PLACEMENT_PLUGIN_ENTRY_V1: &[u8] = b"aisimulate_placement_plugin_v1\0";

/// Provider inputs used when the neutral replay adapter creates a policy.
#[derive(Debug, Clone)]
pub struct DynamicPlacementConfig {
    /// Deterministic seed supplied to the placement provider.
    pub selector_seed: [u8; 32],
    /// Capacity facts for every initial worker, keyed by `worker_id`.
    pub capacities: Vec<WorkerCapacityV1>,
    /// Namespace of `provider_options`.
    pub options_namespace: Vec<u8>,
    /// Provider-defined option bytes in `options_namespace`.
    pub provider_options: Vec<u8>,
    /// Host-selected bounds for provider outputs.
    pub limits: PlacementLimitsV1,
}

/// Replay admission metadata retained by the dynamic placement adapter.
///
/// The admission queue supplies precomputed hashes here, including for
/// deferred trace payloads. Keeping them separate from the payload preserves
/// Replay's compact queue representation while making both hash forms
/// available to a dynamic KV-aware placement policy.
#[derive(Debug, Clone, Default)]
pub struct DynamicPlacementMetadata {
    replay_hashes: Option<ReplayRequestHashes>,
}

/// One router-visible AISimulate KV event tagged with its emitting worker.
#[derive(Debug)]
struct DynamicKvEvent {
    worker_id: usize,
    event: KvEvent,
}

/// Ordered KV observations that a dynamically loaded placement provider sees.
#[derive(Debug, Default)]
pub struct DynamicKvEventBatch(Vec<DynamicKvEvent>);

impl EngineEventBatch for DynamicKvEventBatch {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

/// Observation flavor that retains exactly the KV events needed by a dynamic
/// KV-aware placement provider.
#[derive(Debug, Default)]
pub struct DynamicKvEventObservation;

impl ReplayEngineObservation for DynamicKvEventObservation {
    type Batch = DynamicKvEventBatch;

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn capture_engine_kv_events(stage: WorkerStage) -> bool {
        !matches!(stage, WorkerStage::Decode)
    }

    fn observe_engine_events(
        stage: WorkerStage,
        worker_id: usize,
        _dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> Self::Batch {
        if matches!(stage, WorkerStage::Decode) {
            return DynamicKvEventBatch::default();
        }
        DynamicKvEventBatch(
            events
                .into_iter()
                .map(|event| DynamicKvEvent { worker_id, event })
                .collect(),
        )
    }
}

impl ReplayAdmissionMetadata for DynamicPlacementMetadata {
    fn from_hashes(replay_hashes: Option<ReplayRequestHashes>) -> Self {
        Self { replay_hashes }
    }

    fn for_prefill(self) -> Self {
        self
    }

    fn max_output_tokens_override(&self) -> Option<usize> {
        None
    }

    fn into_hashes(self) -> Option<ReplayRequestHashes> {
        self.replay_hashes
    }
}

/// A loaded placement plugin that can create one or more policy instances.
///
/// Each created policy retains a shared library owner until after its opaque
/// plugin handle is destroyed, so a provider can safely supply distinct
/// policies for the prefill and decode roles of one disaggregated replay.
pub struct DynamicPlacementPlugin {
    library: Arc<Library>,
    vtable: PluginVTableV1,
    capabilities: u64,
}

impl DynamicPlacementPlugin {
    /// Loads and validates a V1 placement plugin from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // Safety: keeping `library` in this value retains the entry point,
        // descriptor, and vtable for the lifetime in which they are read.
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("loading placement plugin {}", path.display()))?;
        // Safety: the symbol name is NUL-terminated and the library is live.
        let entry = unsafe { library.get::<PluginEntryV1>(PLACEMENT_PLUGIN_ENTRY_V1) }
            .with_context(|| format!("loading placement entry point from {}", path.display()))?;
        // Safety: the plugin entry point is called while its library is live.
        let descriptor = unsafe { entry() };
        // Safety: descriptor and vtable remain live through the retained library.
        let vtable = unsafe { validate_descriptor_v1(descriptor) }.map_err(|error| {
            anyhow!(
                "invalid placement plugin descriptor in {}: {error:?}",
                path.display()
            )
        })?;
        // Safety: `validate_descriptor_v1` checked this non-null readable table.
        let vtable = unsafe { *vtable };
        // Safety: descriptor validation established that this immutable record
        // remains readable while the library is retained.
        let capabilities = unsafe { (*descriptor).capabilities };

        Ok(Self {
            library: Arc::new(library),
            vtable,
            capabilities,
        })
    }

    /// Creates the plugin's placement instance for an already-resolved worker topology.
    pub fn create(
        &self,
        workers: Vec<WorkerTopology>,
        config: DynamicPlacementConfig,
    ) -> Result<DynamicPlacementPolicy> {
        ensure!(
            self.capabilities & CAPABILITY_LOSSLESS_KV_EVENTS_V1 != 0
                && self.vtable.supports_lossless_kv_events(),
            "dynamic placement plugin does not negotiate lossless KV observations"
        );
        validate_initial_topology(&workers, &config.capacities)?;
        let scheduler_ids = workers
            .iter()
            .map(|worker| {
                worker
                    .scheduler_ids
                    .iter()
                    .copied()
                    .map(u64::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .context("scheduler ID does not fit the placement ABI")
            })
            .collect::<Result<Vec<_>>>()?;
        let worker_records = workers
            .iter()
            .zip(&scheduler_ids)
            .map(|(worker, scheduler_ids)| {
                Ok(WorkerTopologyV1 {
                    worker_id: u64::try_from(worker.worker_id)
                        .context("worker ID does not fit the placement ABI")?,
                    scheduler_ids: SchedulerIdSliceV1 {
                        data: scheduler_ids.as_ptr(),
                        len: scheduler_ids.len() as u64,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let request = PlacementCreateRequestV1 {
            struct_size: std::mem::size_of::<PlacementCreateRequestV1>() as u32,
            payload_version: aisimulate_placement_abi::PLACEMENT_CREATE_PAYLOAD_VERSION_V1,
            flags: 0,
            reserved: 0,
            selector_seed: config.selector_seed,
            workers: WorkerTopologySliceV1 {
                data: worker_records.as_ptr(),
                len: worker_records.len() as u64,
            },
            capacities: aisimulate_placement_abi::WorkerCapacitySliceV1 {
                data: config.capacities.as_ptr(),
                len: config.capacities.len() as u64,
            },
            options_namespace: bytes(&config.options_namespace),
            provider_options: bytes(&config.provider_options),
            limits: config.limits,
        };
        validate_create_request_v1(&request)
            .map_err(|error| anyhow!("invalid dynamic placement create request: {error:?}"))?;

        let mut handle = PlacementHandleV1(std::ptr::null_mut());
        let mut error = ByteSliceV1::EMPTY;
        // Safety: the ABI request borrows the vectors above for this call only;
        // descriptor validation established that `create` is present.
        let status = unsafe { required_create(&self.vtable)(request, &mut handle, &mut error) };
        let error_text = unsafe { take_plugin_bytes(&self.vtable, error, "create error") };
        if status != StatusV1::OK {
            if !handle.0.is_null() {
                // Safety: the provider returned this handle through its own
                // create callback, so the same validated table owns its cleanup.
                unsafe { required_destroy(&self.vtable)(handle) };
            }
            bail!(
                "placement plugin create failed with status {}{}",
                status.0,
                error_suffix(error_text.as_deref())
            );
        }
        ensure!(
            !handle.0.is_null(),
            "placement plugin create succeeded without returning an instance handle"
        );

        Ok(DynamicPlacementPolicy {
            instance: PlacementInstance {
                handle,
                vtable: self.vtable,
            },
            pending_count: 0,
            last_now_ms: 0.0,
            limits: config.limits,
            supports_lossless_kv_events: true,
            // Field order intentionally destroys `instance` before unloading the
            // library: its Drop invokes the provider's destroy callback.
            _library: Some(Arc::clone(&self.library)),
        })
    }
}

/// One V1 plugin instance adapted to neutral aggregated replay placement.
pub struct DynamicPlacementPolicy {
    instance: PlacementInstance,
    pending_count: usize,
    last_now_ms: f64,
    limits: PlacementLimitsV1,
    supports_lossless_kv_events: bool,
    _library: Option<Arc<Library>>,
}

impl DynamicPlacementPolicy {
    /// Builds a fixture-only proxy for focused ABI adapter tests.
    ///
    /// # Safety
    ///
    /// This is excluded from production builds. `vtable` must remain valid for
    /// the returned policy's lifetime and every callback must accept the
    /// fixture's non-null sentinel handle.
    #[cfg(feature = "placement-test-fixture")]
    #[doc(hidden)]
    pub unsafe fn from_test_vtable(vtable: PluginVTableV1, limits: PlacementLimitsV1) -> Self {
        Self {
            instance: PlacementInstance {
                handle: PlacementHandleV1(std::ptr::dangling_mut()),
                vtable,
            },
            pending_count: 0,
            last_now_ms: 0.0,
            limits,
            supports_lossless_kv_events: vtable.supports_lossless_kv_events(),
            _library: None,
        }
    }

    fn apply(&mut self, mutation: PlacementMutationV1) -> Result<AppliedBatch> {
        ensure!(
            mutation.now_ms.is_finite(),
            "dynamic placement mutation time must be finite"
        );
        ensure!(
            mutation.now_ms >= self.last_now_ms,
            "dynamic placement mutation time {} precedes the previous time {}",
            mutation.now_ms,
            self.last_now_ms
        );
        let batch = PlacementMutationSliceV1 {
            data: &mutation,
            len: 1,
        };
        let mut result = empty_batch_result();
        // Safety: `instance` is live, `batch` is valid for the call, and the
        // result output points to host-owned initialized storage.
        let status = unsafe {
            required_apply_batch(&self.instance.vtable)(self.instance.handle, batch, &mut result)
        };
        if status != StatusV1::OK {
            let error = self.instance.last_error()?;
            bail!(
                "placement plugin apply_batch failed with status {}{}",
                status.0,
                error_suffix(error.as_deref())
            );
        }
        let result_guard = BatchResultGuard {
            vtable: self.instance.vtable,
            result,
        };
        result_guard.validate(self.limits)?;
        ensure!(
            result_guard.result.applied_mutations == 1,
            "placement plugin committed {} of 1 requested mutation(s)",
            result_guard.result.applied_mutations
        );
        self.pending_count = usize::try_from(result_guard.result.pending_count)
            .context("placement plugin pending count does not fit this host")?;
        self.last_now_ms = mutation.now_ms;
        let admissions =
            unsafe { copy_admissions(result_guard.result.admission_results, self.limits)? };
        let released = unsafe { copy_placements(result_guard.result.released, self.limits)? };
        Ok(AppliedBatch {
            admissions,
            released,
        })
    }

    fn apply_kv_event(&mut self, event: DynamicKvEvent, now_ms: f64) -> Result<Vec<Placement>> {
        ensure!(
            self.supports_lossless_kv_events,
            "dynamic placement plugin does not negotiate lossless KV observations"
        );
        ensure!(
            now_ms.is_finite(),
            "dynamic placement observation time must be finite"
        );
        ensure!(
            now_ms >= self.last_now_ms,
            "dynamic placement observation time {now_ms} precedes the previous time {}",
            self.last_now_ms
        );
        let worker_id = u64::try_from(event.worker_id)
            .context("KV observation worker ID does not fit the placement ABI")?;
        let event_id = event.event.event_id;
        let dp_rank = event.event.dp_rank;
        let (_stored_blocks, packet) = match &event.event.data {
            KvEventData::Stored(stored) => {
                let stored_blocks: Vec<KvStoredBlockV1> = stored
                    .blocks
                    .iter()
                    .map(|block| KvStoredBlockV1 {
                        sequence_hash: block.block_hash,
                        token_hash: block.tokens_hash,
                    })
                    .collect();
                let packet = KvEventV1::stored(
                    worker_id,
                    dp_rank,
                    KvStorageTierV1::DEVICE,
                    event_id,
                    stored.parent_hash,
                    stored
                        .start_position
                        .map(u64::try_from)
                        .transpose()
                        .context("KV observation start position does not fit the placement ABI")?,
                    &stored_blocks,
                );
                (stored_blocks, packet)
            }
            KvEventData::Removed { block_hashes } => (
                Vec::new(),
                KvEventV1::removed(
                    worker_id,
                    dp_rank,
                    KvStorageTierV1::DEVICE,
                    event_id,
                    block_hashes,
                ),
            ),
        };
        let batch = KvEventSliceV1 {
            data: std::ptr::from_ref(&packet),
            len: 1,
        };
        let mut result = empty_batch_result();
        // Safety: the packet and its nested vectors remain live for the full
        // synchronous callback; capability negotiation established the tail.
        let status = unsafe {
            required_apply_kv_events(&self.instance.vtable)(
                self.instance.handle,
                batch,
                now_ms,
                &mut result,
            )
        };
        if status != StatusV1::OK {
            let error = self.instance.last_error()?;
            bail!(
                "placement plugin apply_kv_events failed with status {}{}",
                status.0,
                error_suffix(error.as_deref())
            );
        }
        let result_guard = BatchResultGuard {
            vtable: self.instance.vtable,
            result,
        };
        result_guard.validate(self.limits)?;
        ensure!(
            result_guard.result.applied_mutations == 1,
            "placement plugin committed {} of 1 KV observation(s)",
            result_guard.result.applied_mutations
        );
        ensure!(
            result_guard.result.admission_results.len == 0,
            "placement plugin returned admission decisions for a KV observation"
        );
        self.pending_count = usize::try_from(result_guard.result.pending_count)
            .context("placement plugin pending count does not fit this host")?;
        self.last_now_ms = now_ms;
        unsafe { copy_placements(result_guard.result.released, self.limits) }
    }

    fn lifecycle(
        &mut self,
        kind: PlacementMutationKindV1,
        request_id: Uuid,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        let result = self.apply(PlacementMutationV1 {
            struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
            kind,
            flags: 0,
            sequence: 0,
            now_ms,
            payload: PlacementMutationPayloadV1 {
                request_lifecycle: RequestLifecycleV1 {
                    request_id: request_id.into_bytes(),
                    flags: 0,
                    reserved: 0,
                },
            },
        })?;
        ensure!(
            result.admissions.is_empty(),
            "placement plugin returned admission results for a lifecycle mutation"
        );
        Ok(result.released)
    }

    fn worker_lifecycle(
        &mut self,
        kind: PlacementMutationKindV1,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        let scheduler_ids = worker
            .scheduler_ids
            .iter()
            .copied()
            .map(u64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("scheduler ID does not fit the placement ABI")?;
        let worker = WorkerTopologyV1 {
            worker_id: u64::try_from(worker.worker_id)
                .context("worker ID does not fit the placement ABI")?,
            scheduler_ids: SchedulerIdSliceV1 {
                data: scheduler_ids.as_ptr(),
                len: scheduler_ids.len() as u64,
            },
        };
        let result = self.apply(PlacementMutationV1 {
            struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
            kind,
            flags: 0,
            sequence: 0,
            now_ms,
            payload: PlacementMutationPayloadV1 { worker },
        })?;
        ensure!(
            result.admissions.is_empty(),
            "placement plugin returned admission results for a worker lifecycle mutation"
        );
        Ok(result.released)
    }
}

impl PlacementPolicy<ReplayRequestPayload> for DynamicPlacementPolicy {
    type Metadata = DynamicPlacementMetadata;
    type Observation = DynamicKvEventBatch;

    fn place(
        &mut self,
        request: &ReplayRequestPayload,
        metadata: Self::Metadata,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<PlacementEffects> {
        let request_id = request
            .metadata()
            .uuid
            .ok_or_else(|| anyhow!("dynamic placement requires a request UUID"))?;
        ensure!(
            session_id.is_none(),
            "dynamic placement V1 cannot encode a session ID until its admission flag is defined"
        );
        let prompt_identity = admission_prompt_identity(request, metadata.replay_hashes.as_ref())?;
        let metadata_bytes = request
            .metadata()
            .replay_context
            .as_ref()
            .map(serde_json::to_vec)
            .transpose()
            .context("serializing dynamic placement admission metadata")?;
        if let Some(metadata_bytes) = metadata_bytes.as_ref() {
            ensure!(
                metadata_bytes.len() <= MAX_ADMISSION_METADATA_BYTES_V1 as usize,
                "dynamic placement admission metadata exceeds the ABI limit of {} bytes",
                MAX_ADMISSION_METADATA_BYTES_V1
            );
        }
        let admission_metadata = metadata_bytes
            .as_deref()
            .map(|metadata_bytes| PlacementMetadataV1 {
                format: AdmissionMetadataFormatV1::JSON_UTF8,
                flags: 0,
                bytes: bytes(metadata_bytes),
            })
            .unwrap_or(PlacementMetadataV1::EMPTY);
        let result = self.apply(PlacementMutationV1 {
            struct_size: std::mem::size_of::<PlacementMutationV1>() as u32,
            kind: PlacementMutationKindV1::ADMIT,
            flags: 0,
            sequence: 0,
            now_ms,
            payload: PlacementMutationPayloadV1 {
                admission: PlacementAdmissionV1 {
                    request_id: request_id.into_bytes(),
                    flags: 0,
                    priority: request.metadata().priority,
                    prompt_tokens: request.input_length() as u64,
                    max_output_tokens: request.metadata().effective_max_output_tokens() as u64,
                    prompt_identity,
                    metadata: admission_metadata,
                    session_id: ByteSliceV1::EMPTY,
                },
            },
        })?;
        ensure!(
            result.admissions.len() == 1,
            "placement plugin returned {} admission results for one admission",
            result.admissions.len()
        );
        let admission = result
            .admissions
            .into_iter()
            .next()
            .expect("length checked");
        ensure!(
            admission.request_id == request_id,
            "placement plugin returned an admission result for a different request"
        );
        let decision = match admission.decision {
            AdmissionDecisionV1::IMMEDIATE => PlacementDecision::Immediate(admission.placement),
            AdmissionDecisionV1::QUEUED => PlacementDecision::Queued,
            value => bail!(
                "placement plugin returned unsupported admission decision {}",
                value.0
            ),
        };
        Ok(PlacementEffects {
            decision,
            released: result.released,
        })
    }

    fn observe(&mut self, observation: Self::Observation, now_ms: f64) -> Result<Vec<Placement>> {
        let mut released = Vec::new();
        for event in observation.0 {
            released.extend(self.apply_kv_event(event, now_ms)?);
        }
        Ok(released)
    }

    fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.lifecycle(
            PlacementMutationKindV1::CANCEL_PENDING,
            request_id,
            self.last_now_ms,
        )
        .is_ok()
    }

    fn request_terminal(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.lifecycle(
            PlacementMutationKindV1::REQUEST_TERMINAL,
            request_id,
            now_ms,
        )
    }

    fn prefill_completed(&mut self, request_id: Uuid, now_ms: f64) -> Result<Vec<Placement>> {
        self.lifecycle(
            PlacementMutationKindV1::PREFILL_COMPLETED,
            request_id,
            now_ms,
        )
    }

    fn pending_count(&self) -> usize {
        self.pending_count
    }

    fn worker_ready(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.worker_lifecycle(PlacementMutationKindV1::WORKER_READY, worker, now_ms)
    }

    fn worker_draining(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.worker_lifecycle(PlacementMutationKindV1::WORKER_DRAINING, worker, now_ms)
    }

    fn worker_removed(&mut self, worker: WorkerTopology, now_ms: f64) -> Result<Vec<Placement>> {
        self.worker_lifecycle(PlacementMutationKindV1::WORKER_REMOVED, worker, now_ms)
    }

    fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        let result = self.apply(PlacementMutationV1::topology_settled(now_ms))?;
        ensure!(
            result.admissions.is_empty(),
            "placement plugin returned admission results for topology_settled"
        );
        Ok(result.released)
    }
}

struct PlacementInstance {
    handle: PlacementHandleV1,
    vtable: PluginVTableV1,
}

impl PlacementInstance {
    fn last_error(&self) -> Result<Option<String>> {
        let mut error = ByteSliceV1::EMPTY;
        // Safety: the handle is live and the vtable was descriptor-validated.
        let status = unsafe { required_last_error(&self.vtable)(self.handle, &mut error) };
        let message = unsafe { take_plugin_bytes(&self.vtable, error, "last error") };
        ensure!(
            status == StatusV1::OK,
            "placement plugin last_error failed with status {}",
            status.0
        );
        Ok(message)
    }
}

impl Drop for PlacementInstance {
    fn drop(&mut self) {
        // Safety: the handle was returned by this vtable and remains owned here.
        unsafe { required_destroy(&self.vtable)(self.handle) };
    }
}

struct BatchResultGuard {
    vtable: PluginVTableV1,
    result: PlacementBatchResultV1,
}

impl BatchResultGuard {
    fn validate(&self, limits: PlacementLimitsV1) -> Result<()> {
        ensure!(
            self.result.struct_size as usize >= std::mem::size_of::<PlacementBatchResultV1>(),
            "placement plugin returned a too-small batch result"
        );
        validate_output_slice(
            self.result.admission_results.data,
            self.result.admission_results.len,
            limits.max_admission_results,
            "admission results",
        )?;
        validate_output_slice(
            self.result.released.data,
            self.result.released.len,
            limits.max_released,
            "released placements",
        )?;
        validate_output_slice(
            self.result.diagnostics.data,
            self.result.diagnostics.len,
            limits.max_mutations,
            "diagnostics",
        )?;
        Ok(())
    }
}

impl Drop for BatchResultGuard {
    fn drop(&mut self) {
        // Safety: a successful apply_batch transfers one release obligation for
        // every output slice in this exact result.
        unsafe { required_release_results(&self.vtable)(self.result) };
    }
}

struct AppliedBatch {
    admissions: Vec<AdmissionResult>,
    released: Vec<Placement>,
}

fn empty_batch_result() -> PlacementBatchResultV1 {
    PlacementBatchResultV1 {
        struct_size: 0,
        flags: 0,
        applied_mutations: 0,
        pending_count: 0,
        admission_results: PlacementResultSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        released: PlacementSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        diagnostics: PlacementDiagnosticSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    }
}

struct AdmissionResult {
    request_id: Uuid,
    decision: AdmissionDecisionV1,
    placement: Placement,
}

fn validate_initial_topology(
    workers: &[WorkerTopology],
    capacities: &[WorkerCapacityV1],
) -> Result<()> {
    ensure!(
        !workers.is_empty(),
        "dynamic placement requires at least one worker"
    );
    ensure!(
        workers.len() == capacities.len(),
        "dynamic placement requires one capacity record per worker"
    );
    for worker in workers {
        let worker_id =
            u64::try_from(worker.worker_id).context("worker ID does not fit the placement ABI")?;
        ensure!(
            capacities
                .iter()
                .filter(|capacity| capacity.worker_id == worker_id)
                .count()
                == 1,
            "dynamic placement requires exactly one capacity record for worker {}",
            worker.worker_id
        );
    }
    Ok(())
}

fn admission_prompt_identity(
    request: &ReplayRequestPayload,
    replay_hashes: Option<&ReplayRequestHashes>,
) -> Result<PromptIdentityV1> {
    let mut identity = PromptIdentityV1::OMITTED;
    if let Some(tokens) = request.materialized_tokens() {
        ensure!(
            tokens.len() <= MAX_ADMISSION_PROMPT_TOKEN_IDS_V1 as usize,
            "dynamic placement materialized prompt has {} token IDs, exceeding the ABI limit of {}",
            tokens.len(),
            MAX_ADMISSION_PROMPT_TOKEN_IDS_V1
        );
        identity.flags |= PromptIdentityV1::MATERIALIZED_TOKEN_IDS_PRESENT;
        identity.materialized_token_ids = TokenIdSliceV1 {
            data: tokens.as_ptr(),
            len: tokens.len() as u64,
        };
    }
    if let Some(replay_hashes) = replay_hashes {
        ensure!(
            replay_hashes.local_block_hashes.len() == replay_hashes.sequence_hashes.len(),
            "dynamic placement received mismatched local ({}) and sequence ({}) replay hash counts",
            replay_hashes.local_block_hashes.len(),
            replay_hashes.sequence_hashes.len()
        );
        ensure!(
            replay_hashes.local_block_hashes.len() <= MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1 as usize,
            "dynamic placement received {} local block hashes, exceeding the ABI limit of {}",
            replay_hashes.local_block_hashes.len(),
            MAX_ADMISSION_PROMPT_BLOCK_HASHES_V1
        );
        identity.flags |= PromptIdentityV1::LOCAL_BLOCK_HASHES_PRESENT;
        identity.local_block_hashes = BlockHashSliceV1 {
            data: replay_hashes.local_block_hashes.as_ptr(),
            len: replay_hashes.local_block_hashes.len() as u64,
        };
        identity.flags |= PromptIdentityV1::SEQUENCE_BLOCK_HASHES_PRESENT;
        identity.sequence_block_hashes = BlockHashSliceV1 {
            data: replay_hashes.sequence_hashes.as_ptr(),
            len: replay_hashes.sequence_hashes.len() as u64,
        };
    }
    Ok(identity)
}

fn bytes(value: &[u8]) -> ByteSliceV1 {
    ByteSliceV1 {
        data: value.as_ptr(),
        len: value.len() as u64,
    }
}

fn validate_output_slice<T>(data: *const T, len: u64, maximum: u64, name: &str) -> Result<()> {
    ensure!(
        len <= maximum,
        "placement plugin returned {len} {name}, exceeding negotiated bound {maximum}"
    );
    ensure!(
        len <= usize::MAX as u64,
        "placement plugin returned {name} too large for this host"
    );
    ensure!(
        len == 0 || !data.is_null(),
        "placement plugin returned a null {name} pointer with nonzero length"
    );
    Ok(())
}

unsafe fn copy_admissions(
    slice: PlacementResultSliceV1,
    limits: PlacementLimitsV1,
) -> Result<Vec<AdmissionResult>> {
    validate_output_slice(
        slice.data,
        slice.len,
        limits.max_admission_results,
        "admission results",
    )?;
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    // Safety: `validate_output_slice` checked the pointer and the ABI owns the
    // result until the surrounding BatchResultGuard is dropped.
    let values = unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) };
    values
        .iter()
        .map(|value| {
            Ok(AdmissionResult {
                request_id: Uuid::from_bytes(value.request_id),
                decision: value.decision,
                placement: placement_from_abi(value.placement)?,
            })
        })
        .collect()
}

unsafe fn copy_placements(
    slice: PlacementSliceV1,
    limits: PlacementLimitsV1,
) -> Result<Vec<Placement>> {
    validate_output_slice(
        slice.data,
        slice.len,
        limits.max_released,
        "released placements",
    )?;
    if slice.len == 0 {
        return Ok(Vec::new());
    }
    // Safety: as in `copy_admissions`, the provider keeps the result alive
    // until `release_results` runs through the enclosing guard.
    let values = unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) };
    values.iter().copied().map(placement_from_abi).collect()
}

fn placement_from_abi(value: PlacementV1) -> Result<Placement> {
    let cache_sample = match value.cache_sample.flags {
        0 => None,
        PlacementCacheSampleV1::PRESENT => Some(PlacementCacheSample {
            overlap_blocks: value.cache_sample.overlap_blocks,
            best_available_overlap_blocks: value.cache_sample.best_available_overlap_blocks,
            isl_blocks: value.cache_sample.isl_blocks,
        }),
        flags => bail!("placement plugin returned unsupported cache-sample flags {flags}"),
    };
    Ok(Placement {
        request_id: Uuid::from_bytes(value.request_id),
        scheduler_id: usize::try_from(value.scheduler_id)
            .context("placement scheduler ID does not fit this host")?,
        reported_overlap_tokens: usize::try_from(value.reported_overlap_tokens)
            .context("placement overlap-token count does not fit this host")?,
        cache_sample,
        placement_replica_id: (value.placement_replica_id != 0)
            .then(|| usize::try_from(value.placement_replica_id))
            .transpose()
            .context("placement replica ID does not fit this host")?,
    })
}

unsafe fn take_plugin_bytes(
    vtable: &PluginVTableV1,
    value: ByteSliceV1,
    purpose: &str,
) -> Option<String> {
    if value.len == 0 {
        return None;
    }
    if value.data.is_null() || value.len > aisimulate_placement_abi::MAX_CREATE_OPTION_BYTES_V1 {
        return Some(format!("{purpose} was not a bounded byte slice"));
    }
    // Safety: the ABI guarantees the byte slice remains valid until released.
    let copied = unsafe { std::slice::from_raw_parts(value.data, value.len as usize) }.to_vec();
    // Safety: descriptor validation established this callback and the returned
    // bytes belong to the same provider table.
    unsafe { required_release_bytes(vtable)(value) };
    Some(String::from_utf8_lossy(&copied).into_owned())
}

fn error_suffix(error: Option<&str>) -> String {
    error.map(|error| format!(": {error}")).unwrap_or_default()
}

fn required_create(vtable: &PluginVTableV1) -> aisimulate_placement_abi::CreateFnV1 {
    vtable
        .create
        .expect("descriptor validation requires create")
}
fn required_apply_batch(
    vtable: &PluginVTableV1,
) -> unsafe extern "C" fn(
    PlacementHandleV1,
    PlacementMutationSliceV1,
    *mut PlacementBatchResultV1,
) -> StatusV1 {
    vtable
        .apply_batch
        .expect("descriptor validation requires apply_batch")
}
fn required_apply_kv_events(
    vtable: &PluginVTableV1,
) -> unsafe extern "C" fn(
    PlacementHandleV1,
    KvEventSliceV1,
    f64,
    *mut PlacementBatchResultV1,
) -> StatusV1 {
    vtable
        .apply_kv_events
        .expect("lossless-KV capability validation requires apply_kv_events")
}
fn required_release_results(
    vtable: &PluginVTableV1,
) -> unsafe extern "C" fn(PlacementBatchResultV1) {
    vtable
        .release_results
        .expect("descriptor validation requires release_results")
}
fn required_release_bytes(vtable: &PluginVTableV1) -> unsafe extern "C" fn(ByteSliceV1) {
    vtable
        .release_bytes
        .expect("descriptor validation requires release_bytes")
}
fn required_last_error(
    vtable: &PluginVTableV1,
) -> unsafe extern "C" fn(PlacementHandleV1, *mut ByteSliceV1) -> StatusV1 {
    vtable
        .last_error
        .expect("descriptor validation requires last_error")
}
fn required_destroy(vtable: &PluginVTableV1) -> unsafe extern "C" fn(PlacementHandleV1) {
    vtable
        .destroy
        .expect("descriptor validation requires destroy")
}
