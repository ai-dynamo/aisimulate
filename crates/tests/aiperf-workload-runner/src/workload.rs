// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-library composition. AgentX reconstruction and scheduling remain upstream-owned.

use crate::{dispatch::Dispatcher, model::ModelExecution};
use aiperf_clock_runtime::{
    cancellation::RunCancellation,
    clock::{SimClock, driver::drive_sim_with_source},
};
use aiperf_engine_phase::{
    agentic_replay::prepare::{prepare_legacy_agentic, prepare_legacy_agentic_phases},
    execute::{
        dataset_build::{
            PreparedNativeConversationSourceFactory,
            build_native_scheduled_phase_plan_with_source_factory, phase_seamless_to_next,
            user_centric_phase_scope,
        },
        plan::native_scheduled_resources,
    },
    phase_runtime::run_scheduled_phases_with_cancellation,
};
use aiperf_exchange_model::TextTokenizer;
use aiperf_protocol::protocol_dto::PhaseSpec;
use anyhow::{Context, Result, ensure};
use std::{path::Path, rc::Rc, sync::Arc};

/// Validate and retain the run's authored default endpoint policy.
pub(crate) fn default_endpoint_profile(
    run: &aiperf_config::model::BenchmarkRun,
    registry: &aiperf_endpoints::EndpointRegistry,
) -> Result<aiperf_endpoints::profile_validation::ValidatedEndpointProfileV2> {
    let profiles =
        aiperf_endpoints::profile_validation::validate_endpoint_profiles_v2(run, registry)?;
    let profile = profiles
        .into_iter()
        .find(|profile| {
            profile.profile_id == aiperf_engine_kernel::endpoint_plan::DEFAULT_ENDPOINT_PROFILE_ID
        })
        .context("default endpoint profile missing")?;
    ensure!(
        !profile.config.use_server_token_count,
        "offline simulation cannot supply server token counts"
    );
    Ok(profile)
}

/// Authored run projection shared by every execution binding (owned; no borrows).
pub(crate) struct Head {
    pub(crate) input: serde_json::Value,
    pub(crate) raw_input: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>>,
    pub(crate) phases: Vec<PhaseSpec>,
    pub(crate) models: aiperf_protocol::protocol_dto::ModelsSpec,
    pub(crate) tokenizer: Arc<dyn TextTokenizer>,
    pub(crate) kind: aiperf_config::model::workload_kind::WorkloadKind,
    pub(crate) legacy_weka: bool,
    pub(crate) input_projection: crate::participant_adapter::NativeInputProjection,
}

pub(crate) fn parse_head(
    run: &aiperf_config::model::BenchmarkRun,
    input_projection: crate::participant_adapter::NativeInputProjection,
) -> Result<Head> {
    aiperf_engine_input::execution_input::validate_run_scenario(run)?;
    let (_, raw) = aiperf_engine_input::execution_input::factory_workload_input_resolved(run)?;
    let input: serde_json::Value = serde_json::from_str(raw.get())?;
    input.get("dataset").context("dataset missing")?;
    let phases: Vec<PhaseSpec> = serde_json::from_value(input["phases"].clone())?;
    ensure!(!phases.is_empty(), "explicit phases required");
    let models: aiperf_protocol::protocol_dto::ModelsSpec = run
        .cfg
        .models
        .clone()
        .context("explicit model required")?
        .into();
    ensure!(models.items.len() == 1, "proof requires exactly one model");
    let raw_input: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>> =
        serde_json::from_str(raw.get())?;
    let tokenizer: Arc<dyn TextTokenizer> = match input_projection {
        crate::participant_adapter::NativeInputProjection::Text => resolve_tokenizer(run)?,
        crate::participant_adapter::NativeInputProjection::TraceHashes => Arc::new(NoTextTokenizer),
    };
    let kind = aiperf_config::model::workload_kind::workload_kind(&run.cfg);
    let legacy_weka = aiperf_config::model::workload_kind::weka_legacy_agentic(&run.cfg)
        .map_err(anyhow::Error::msg)?;
    Ok(Head {
        input,
        raw_input,
        phases,
        models,
        tokenizer,
        kind,
        legacy_weka,
        input_projection,
    })
}

/// Direct trace geometry must never enter a text encoder or decoder. A violated
/// preparation invariant fails immediately rather than being folded into a zero count.
struct NoTextTokenizer;
impl TextTokenizer for NoTextTokenizer {
    fn name(&self) -> &str {
        "trace-hashes:no-text-tokenizer"
    }
    fn encode(&self, _text: &str) -> aiperf_exchange_model::TokenizerResult<Vec<u32>> {
        panic!("direct trace hash invariant: text encoding must not be called")
    }
    fn decode(&self, _tokens: &[u32]) -> aiperf_exchange_model::TokenizerResult<String> {
        panic!("direct trace hash invariant: text reconstruction must not be called")
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
    fn eos_token_id(&self) -> Option<u32> {
        None
    }
}

/// Resolve the authored tokenizer without substituting an unsupported source.
///
/// Supported forms are the exact `builtin` sentinel, an existing local
/// `tokenizer.json` file or model directory, and an authored Hugging Face
/// repository id whose exact revision is already staged in the Hugging Face
/// cache (`offline_snapshot_directory`). The proof never substitutes a different
/// revision, downloads an unpinned `main`, or falls back to the built-in
/// tokenizer. Repository acquisition is a synchronous cache read, so it happens
/// before `native_hosted` enters its local Tokio runtime.
fn resolve_tokenizer(run: &aiperf_config::model::BenchmarkRun) -> Result<Arc<dyn TextTokenizer>> {
    let config = run.cfg.tokenizer.as_ref().context(
        "proof requires an explicit tokenizer; set tokenizer.name to \"builtin\", an existing \
         local tokenizer file or directory, or a Hugging Face repository id",
    )?;
    ensure!(
        !config.apply_chat_template,
        "proof does not apply tokenizer chat templates; set tokenizer.apply_chat_template=false"
    );
    ensure!(
        config.server_url.is_none(),
        "proof does not support server-side tokenization; tokenizer.server_url must be absent"
    );
    ensure!(
        config.per_model.is_empty(),
        "proof supports one tokenizer for the single model; tokenizer.per_model must be empty"
    );
    let name = config.name.as_deref().context(
        "proof requires an explicit tokenizer.name; set \"builtin\", an existing local tokenizer \
         file or directory, or a Hugging Face repository id",
    )?;
    let revision = config.revision.as_str();
    ensure!(
        !revision.trim().is_empty(),
        "tokenizer.revision must not be empty"
    );

    if name == "builtin" {
        return Ok(Arc::new(aiperf_tokenizers::TiktokenTokenizer::builtin()));
    }
    if name.parse::<aiperf_tokenizers::TiktokenEncoding>().is_ok() {
        anyhow::bail!(
            "unsupported tokenizer {name:?}: the proof accepts \"builtin\", an existing local \
             tokenizer file or directory, or a Hugging Face repository id; other built-in \
             tiktoken encodings are not supported"
        );
    }

    let load_directory = |directory: &Path| -> Result<Arc<dyn TextTokenizer>> {
        aiperf_tokenizers::HuggingFaceTokenizer::from_directory(directory)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .with_context(|| format!("loading tokenizer directory {}", directory.display()))
    };

    let path = Path::new(name);
    if path.is_dir() {
        return load_directory(path);
    }
    // The oracle's local source is directory-only; the AISim proof also accepts
    // an explicit `tokenizer.json` file so a calibration can stage one file.
    if path.is_file() {
        return aiperf_tokenizers::HuggingFaceTokenizer::from_file(path)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn TextTokenizer>)
            .with_context(|| format!("loading local tokenizer file {name:?}"));
    }
    if aiperf_tokenizers::is_local_tokenizer_path(name) {
        anyhow::bail!(
            "local tokenizer path {name:?} is not an existing tokenizer file or directory"
        );
    }
    ensure!(
        aiperf_tokenizers::is_hugging_face_repository_id(name),
        "unsupported tokenizer {name:?}: expected \"builtin\", an existing local tokenizer file or \
         directory, or a Hugging Face repository id"
    );

    let context = format!(
        "loading tokenizer {name:?} revision {revision:?}: the exact revision must be pre-staged \
         in the Hugging Face cache or passed as an explicit local tokenizer file or directory"
    );
    match aiperf_tokenizers::offline_snapshot_directory(name, revision) {
        Ok(directory) => load_directory(&directory).context(context),
        Err(error) => Err(anyhow::Error::new(error).context(context)),
    }
}

pub fn execute(
    run: &aiperf_config::model::BenchmarkRun,
    cancel: RunCancellation,
    observer: Rc<dyn aiperf_timing::PhaseObserver>,
    model_factory: Option<(&aisimulate_core::engine::EngineFactory, usize)>,
    identity_source: Rc<dyn aiperf_rng::identity::IdentitySource>,
    deployment: Option<(
        Box<dyn aisimulate_core::replay::loadgen::SteppableReplay>,
        usize,
    )>,
) -> Result<serde_json::Value> {
    let head = parse_head(run, crate::participant_adapter::NativeInputProjection::Text)?;
    if head.kind == aiperf_config::model::workload_kind::WorkloadKind::Graph && !head.legacy_weka {
        let prepared = crate::graph::prepare(
            run,
            head.raw_input
                .get("dataset")
                .context("raw graph dataset missing")?,
            head.tokenizer.as_ref(),
            &head.phases,
        )?;
        return crate::graph::execute(
            run,
            prepared,
            head.phases,
            cancel,
            observer,
            model_factory,
            deployment,
        );
    }
    let prep = lower_scheduled(run, &head, cancel.clone(), identity_source)?;
    let clock = Rc::new(SimClock::new());
    let engine = if let Some((deployment, capacity)) = deployment {
        ModelExecution::from_deployment(clock.clone(), deployment, capacity)?
    } else {
        match model_factory {
            Some((factory, capacity)) => {
                ModelExecution::from_factory(clock.clone(), factory, capacity)?
            }
            None => ModelExecution::fixed(clock.clone(), 1.0, 1.0, 64)?,
        }
    };
    let dispatcher = Rc::new(Dispatcher::new(engine.clone()));
    let plans = build_plans(run, &prep, clock.clone())?;
    run_with_engine(plans, clock, engine, dispatcher, observer, cancel)
}

/// Everything the scheduler needs that does not depend on a backend or clock.
pub(crate) struct ScheduledPrep {
    pub(crate) prepared: aiperf_engine_input::dataset_input::PreparedDatasetInput,
    pub(crate) phases: Vec<PhaseSpec>,
    pub(crate) origins: Vec<Option<usize>>,
    pub(crate) model: String,
    pub(crate) tokenizer: Arc<dyn TextTokenizer>,
    pub(crate) identity_source: Rc<dyn aiperf_rng::identity::IdentitySource>,
    pub(crate) resolver: Rc<dyn aiperf_multiturn::PreparedTurnEndpointResolver>,
}

/// Dataset acquisition, endpoint binding and resident-dataset lowering through
/// the owning native adapters. Moved verbatim from `execute`; no policy copy.
pub(crate) fn lower_scheduled(
    run: &aiperf_config::model::BenchmarkRun,
    head: &Head,
    cancel: RunCancellation,
    identity_source: Rc<dyn aiperf_rng::identity::IdentitySource>,
) -> Result<ScheduledPrep> {
    let Head {
        input,
        raw_input,
        phases,
        models,
        tokenizer,
        kind,
        legacy_weka,
        input_projection,
    } = head;
    let legacy_weka = *legacy_weka;
    let dataset = input.get("dataset").context("dataset missing")?;
    let phases = phases.clone();
    ensure!(
        *kind == aiperf_config::model::workload_kind::WorkloadKind::Scheduled || legacy_weka,
        "typed workload {kind:?} has no simulator execution binding"
    );
    tracing::debug!(
        ?input_projection,
        ?kind,
        legacy_weka,
        authored_phases = phases.len(),
        "preparing shared workload input"
    );
    let (mut prepared, phases, origins) = if legacy_weka {
        tracing::debug!(?input_projection, "acquiring complete trace source");
        let traces = acquire(
            dataset,
            raw_input.get("dataset").context("raw dataset missing")?,
            *input_projection,
        )?;
        tracing::debug!(
            traces = traces.len(),
            ?input_projection,
            "trace source admitted; lowering shared AgentX metadata"
        );
        use aiperf_engine_phase::agentic_replay::prepare::LegacyAgenticPreparation;
        let projection = match input_projection {
            crate::participant_adapter::NativeInputProjection::TraceHashes => {
                LegacyAgenticPreparation::TraceHashes
            }
            crate::participant_adapter::NativeInputProjection::Text => {
                let seed = run.random_seed.unwrap_or(0);
                let (corpus, hash_base_seed) =
                    aiperf_engine_input::legacy_weka::legacy_weka_corpus(
                        dataset
                            .pointer("/prompts/corpus")
                            .and_then(|value| value.as_str()),
                        tokenizer.as_ref(),
                        seed,
                    )?;
                LegacyAgenticPreparation::Text {
                    corpus: corpus.into(),
                    hash_base_seed,
                    tokenizer: tokenizer.clone(),
                    use_server_token_count: false,
                }
            }
        };
        let prepared = prepare_legacy_agentic(run, &phases, dataset, &traces, projection)?;
        tracing::debug!(
            roots = prepared.agentic_trees.len(),
            lanes = prepared.agentic_lanes.lanes.len(),
            "shared source geometry and lane selection prepared"
        );
        let (phases, origins) = prepare_legacy_agentic_phases(
            run,
            &phases,
            dataset,
            input["system_idle_gap_cap_seconds"].as_f64(),
            &prepared,
        )?;
        for (index, phase) in phases.iter().enumerate() {
            let common = phase.common();
            tracing::debug!(phase_index = index, phase_name = %common.name, warmup = common.is_warmup(), duration_s = ?common.duration, requests = ?common.requests, grace_s = ?common.grace_period, origin = ?origins[index], "shared phase program lowered");
        }
        (prepared, phases, origins)
    } else {
        ensure!(
            *input_projection == crate::participant_adapter::NativeInputProjection::Text,
            "direct trace hashes require an authored trace source"
        );
        let prepared = crate::scheduled::prepare(
            run,
            raw_input.get("dataset").context("raw dataset missing")?,
            models,
            tokenizer.as_ref(),
            cancel.clone(),
        )?;
        let origins = (0..phases.len()).map(Some).collect::<Vec<_>>();
        (prepared, phases, origins)
    };
    let registry = aiperf_endpoints::EndpointRegistry::builtin()?;
    let mut profile = default_endpoint_profile(run, &registry)?;
    if *input_projection == crate::participant_adapter::NativeInputProjection::TraceHashes {
        profile.endpoint_id = "aisimulate".parse()?;
    }
    let endpoint_id = profile.endpoint_id;
    let endpoint = registry.prepare(&endpoint_id, profile.config)?;
    let mut table = aiperf_endpoints::PreparedEndpointTable::new();
    let key = table.push(endpoint)?;
    let binding = aiperf_engine_input::dataset_input::ResidentDatasetEndpointBinding::new(
        Arc::new(table),
        aiperf_exchange_model::credit::PreparedEndpointReference { key, endpoint_id },
    );
    let resolver = binding.resolver()?;
    let model = &models.items.first().context("one model required")?.name;
    let source = Arc::try_unwrap(prepared.dataset)
        .map_err(|_| anyhow::anyhow!("dataset shared before preparation"))?;
    prepared.dataset = aiperf_multiturn::prepare_resident_dataset_with_timing(
        source,
        resolver.as_ref(),
        model,
        &aiperf_multiturn::DEFAULT_RESIDENT_DATASET_LOWERER,
        None,
        phases.iter().any(|phase| phase.common().is_warmup()),
    )?;
    prepared.resident_endpoint_binding = Some(binding);
    Ok(ScheduledPrep {
        prepared,
        phases,
        origins,
        model: model.clone(),
        tokenizer: tokenizer.clone(),
        identity_source,
        resolver,
    })
}

/// Build every phase plan against the supplied clock. Plans own all state
/// (`Rc`, no borrows), so they are `'static` and can be spawned or hosted.
pub(crate) fn build_plans(
    run: &aiperf_config::model::BenchmarkRun,
    prep: &ScheduledPrep,
    clock: Rc<SimClock>,
) -> Result<Vec<aiperf_engine_phase::phase_runtime::ScheduledPhasePlan>> {
    let ScheduledPrep {
        prepared,
        phases,
        origins,
        model,
        tokenizer,
        identity_source,
        resolver,
    } = prep;
    let samplers = aiperf_dataset::sampler::SamplerRegistry::with_builtin_strategies()?;
    let factory = PreparedNativeConversationSourceFactory {
        identity_source: identity_source.clone(),
        endpoint_resolver: resolver.clone(),
        samplers: &samplers,
        materializer: Arc::new(aiperf_dataset::TraceHashAwareRequestMaterializer),
        defer_grpc_operations: false,
        ignore_trace_delays: run.cfg.ignore_trace_delays,
        end_conversation_on_truncation: prepared.end_conversation_on_truncation,
        worker_group_partition: None,
        position_addressed: false,
        whole_corpus_positions: false,
        run_sampler: aiperf_multiturn::RunScopedSampler::new(),
    };
    let resources = native_scheduled_resources(phases);
    let root = aiperf_rng::RngRoot::new(run.random_seed);
    let dataset_root = aiperf_rng::RngRoot::new(prepared.random_seed.or(run.random_seed));
    let mut plans = Vec::new();
    let mut profiling_index = 0;
    for (index, phase) in phases.iter().enumerate() {
        let measured_index = if phase.common().exclude_from_results {
            None
        } else {
            let current = profiling_index;
            profiling_index += 1;
            Some(current)
        };
        plans.push(build_native_scheduled_phase_plan_with_source_factory(
            index,
            measured_index,
            origins[index],
            phase,
            phase_seamless_to_next(phases, index),
            &prepared.full_view(),
            model,
            prepared.default_output_tokens,
            dataset_root,
            root,
            root,
            &factory,
            tokenizer.clone(),
            clock.clone(),
            0,
            &run.benchmark_id,
            &run.artifact_dir,
            &[],
            &resources,
            None,
            aiperf_config::failure::OnFailure::scheduled_or_default(None),
            prepared.agentic_cache_bust_target,
            prepared.cache_bust_target,
            prepared.agentic_trees.clone(),
            prepared.agentic_lanes.clone(),
            prepared.warmup_handoff.clone(),
            false,
            0,
            aiperf_engine_kernel::request_rate::SlotDrawAddress::for_worker_group(0, 0, 1),
            None,
            aiperf_distributed_model::ModuloWorkerGroupPartition::direct(),
            user_centric_phase_scope(phases, index),
        )?);
    }
    Ok(plans)
}

fn run_with_engine(
    plans: Vec<aiperf_engine_phase::phase_runtime::ScheduledPhasePlan>,
    clock: Rc<SimClock>,
    engine: Rc<ModelExecution>,
    dispatcher: Rc<Dispatcher>,
    observer: Rc<dyn aiperf_timing::PhaseObserver>,
    cancel: RunCancellation,
) -> Result<serde_json::Value> {
    let mut result = None;
    let event_source: Rc<dyn aiperf_clock_runtime::clock::driver::SimEventSource> = engine.clone();
    drive_sim_with_source(clock.clone(), event_source, |_| async {
        result = Some(
            run_scheduled_phases_with_cancellation(
                plans,
                clock.clone(),
                dispatcher.clone(),
                observer,
                Some(cancel),
                None,
            )
            .await,
        );
    })?;
    let report = result.context("driver returned without a report")??;
    ensure!(
        engine.is_drained(),
        "simulator retained request/pass ownership after run"
    );
    let backend_report = engine.take_backend_report(clock.now_ns() as f64 / 1_000_000.0)?;
    let model_trace:Vec<serde_json::Value> = engine.trace().into_iter().map(|event| match event {
        crate::model::ModelEvent::CapacityWait {request_id,at_ns} => serde_json::json!({"event":"capacity_wait","request_id":request_id.to_string(),"at_ns":at_ns}),
        crate::model::ModelEvent::Submitted {request_id,at_ns} => serde_json::json!({"event":"submitted","request_id":request_id.to_string(),"at_ns":at_ns}),
        crate::model::ModelEvent::PassStarted {pass_id,at_ns,request_ids} => serde_json::json!({"event":"pass_started","pass_id":pass_id,"at_ns":at_ns,"request_ids":request_ids.iter().map(u128::to_string).collect::<Vec<_>>()}),
        crate::model::ModelEvent::PassCompleted {pass_id,at_ns} => serde_json::json!({"event":"pass_completed","pass_id":pass_id,"at_ns":at_ns}),
        crate::model::ModelEvent::Observed {request_id,at_ns,is_final} => serde_json::json!({"event":"observed","request_id":request_id.to_string(),"at_ns":at_ns,"is_final":is_final}),
        crate::model::ModelEvent::Cancelled {request_id,at_ns} => serde_json::json!({"event":"cancelled","request_id":request_id.to_string(),"at_ns":at_ns}),
    }).collect();
    let submitted_count = model_trace
        .iter()
        .filter(|event| event["event"] == "submitted")
        .count();
    let cancelled_count = model_trace
        .iter()
        .filter(|event| event["event"] == "cancelled")
        .count();
    // Assign occurrence labels after execution: instrumentation cannot affect
    // source sampling, dispatch order, callbacks, or the scheduler's frontier.
    // Distinct runtime sessions sharing a source template get distinct labels;
    // all turns in the same runtime session keep that occurrence.
    let mut records = dispatcher.records.borrow_mut();
    label_occurrences(&mut records);
    Ok(serde_json::json!({
        "format": "aisim-aiperf-library-proof-v1",
        "capture_format": "aisim-runtime-values-raw-v1",
        "native_phase_report": report,
        "backend_report": backend_report,
        "backend_per_request": backend_report.per_request,
        "backend_report_source": "actual Engine submissions, admissions, outputs and provisioned-clock intervals folded by AISim TraceCollector",
        "requests": records.as_slice(),
        "requests_coverage": "all dispatcher-issued requests, including cancelled None and dropped futures; absent observations remain null",
        "model_trace": model_trace,
        "model_submitted_count": submitted_count,
        "model_cancelled_count": cancelled_count,
        "virtual_end_ns": clock.now_ns(),
        "engine_drained": true,
    }))
}

/// Post-run occurrence labels; shared by every execution binding.
pub(crate) fn label_occurrences(records: &mut [serde_json::Value]) {
    let mut occurrences = std::collections::BTreeMap::<String, Vec<String>>::new();
    for row in records.iter_mut() {
        let template = row["request"]["conversation_id"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let session = row["request"]["correlation_id"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        let sessions = occurrences.entry(template).or_default();
        let occurrence = match sessions.iter().position(|existing| *existing == session) {
            Some(index) => index,
            None => {
                sessions.push(session);
                sessions.len() - 1
            }
        };
        row["runtime_occurrence"] = serde_json::json!(occurrence);
    }
}

fn acquire(
    dataset: &serde_json::Value,
    descriptor: &serde_json::value::RawValue,
    projection: crate::participant_adapter::NativeInputProjection,
) -> Result<Vec<(String, aiperf_agentx::trace::WekaTrace)>> {
    use aiperf_engine_input::legacy_weka::{
        LegacyWekaSource, legacy_weka_source, load_local_legacy_weka_traces,
    };
    ensure!(
        tokio::runtime::Handle::try_current().is_err(),
        "prepare the native workload before entering an executor"
    );
    let (source, _) = legacy_weka_source(descriptor)?;
    let max_context = dataset
        .pointer("/synthesis/max_context_length")
        .and_then(|value| value.as_i64());
    let max_osl = dataset
        .pointer("/synthesis/max_osl")
        .and_then(|value| value.as_i64());
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            match source {
                LegacyWekaSource::File {
                    path,
                    records,
                    entries,
                    basis,
                } => {
                    load_local_legacy_weka_traces(
                        path,
                        records,
                        entries,
                        max_context,
                        max_osl,
                        basis,
                    )
                    .await
                }
                LegacyWekaSource::HuggingFace {
                    name,
                    subset,
                    split,
                    revision,
                    entries,
                    basis,
                } => {
                    let source = aiperf_agentx::hf_dataset::HfDatasetRef {
                        name,
                        subset,
                        split,
                        revision,
                        max_rows: None,
                    };
                    let result = match projection {
                        crate::participant_adapter::NativeInputProjection::Text => {
                            aiperf_agentx::hf_dataset::load_hf_weka_traces(
                                source,
                                entries,
                                max_context,
                                max_osl,
                                basis,
                            )
                            .await
                        }
                        crate::participant_adapter::NativeInputProjection::TraceHashes => {
                            aiperf_agentx::hf_dataset::load_hf_weka_trace_hashes(
                                source,
                                entries,
                                max_context,
                                max_osl,
                                basis,
                            )
                            .await
                        }
                    };
                    let (traces, _) = result.map_err(anyhow::Error::msg)?;
                    Ok(traces)
                }
            }
        })
}
