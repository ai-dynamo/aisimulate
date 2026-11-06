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
use std::{cell::RefCell, rc::Rc, sync::Arc};

/// Validate and retain the run's authored default endpoint policy.
pub(crate) fn default_endpoint_profile(
    run: &aiperf_config::model::BenchmarkRun,
    registry: &aiperf_endpoints::EndpointRegistry,
) -> Result<aiperf_endpoints::profile_validation::ValidatedEndpointProfileV2> {
    let profiles = aiperf_endpoints::profile_validation::validate_endpoint_profiles_v2(run, registry)?;
    let profile = profiles.into_iter().find(|profile| {
        profile.profile_id == aiperf_engine_kernel::endpoint_plan::DEFAULT_ENDPOINT_PROFILE_ID
    }).context("default endpoint profile missing")?;
    ensure!(!profile.config.use_server_token_count, "offline simulation cannot supply server token counts");
    Ok(profile)
}

pub fn execute(
    run: &aiperf_config::model::BenchmarkRun,
    cancel: RunCancellation,
    observer: Rc<dyn aiperf_timing::PhaseObserver>,
    proof_wait: Option<(
        aiperf_engine_kernel::event_writer::EventWriterHandle,
        Arc<aiperf_engine_kernel::event_writer::Sequencer>,
    )>,
    model_factory: Option<(&aisimulate_core::engine::EngineFactory, usize)>,
    identity_source: Rc<dyn aiperf_rng::identity::IdentitySource>,
) -> Result<serde_json::Value> {
    let (_, raw) = aiperf_engine_input::execution_input::factory_workload_input_resolved(run)?;
    let input: serde_json::Value = serde_json::from_str(raw.get())?;
    let dataset = input.get("dataset").context("dataset missing")?;
    let phases: Vec<PhaseSpec> = serde_json::from_value(input["phases"].clone())?;
    ensure!(!phases.is_empty(), "explicit phases required");
    ensure!(
        input.pointer("/tokenizer/name").and_then(|v| v.as_str()) == Some("builtin"),
        "proof requires explicit builtin tokenizer"
    );
    let models: aiperf_protocol::protocol_dto::ModelsSpec = run
        .cfg
        .models
        .clone()
        .context("explicit model required")?
        .into();
    ensure!(models.items.len() == 1, "proof requires exactly one model");
    let raw_input: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>> =
        serde_json::from_str(raw.get())?;
    let tokenizer: Arc<dyn TextTokenizer> =
        Arc::new(aiperf_tokenizers::TiktokenTokenizer::builtin());
    let kind = aiperf_config::model::workload_kind::workload_kind(&run.cfg);
    let legacy_weka = aiperf_config::model::workload_kind::weka_legacy_agentic(&run.cfg)
        .map_err(anyhow::Error::msg)?;
    if kind == aiperf_config::model::workload_kind::WorkloadKind::Graph && !legacy_weka {
        ensure!(proof_wait.is_none(), "graph control rendezvous is not implemented");
        let prepared = crate::graph::prepare(
            run, raw_input.get("dataset").context("raw graph dataset missing")?,
            tokenizer.as_ref(), &phases,
        )?;
        return crate::graph::execute(run, prepared, phases, cancel, observer, model_factory);
    }
    ensure!(
        kind == aiperf_config::model::workload_kind::WorkloadKind::Scheduled || legacy_weka,
        "typed workload {kind:?} has no simulator execution binding"
    );
    let (mut prepared, phases, origins) = if legacy_weka {
    let raw_dataset: std::collections::BTreeMap<String, Box<serde_json::value::RawValue>> =
        serde_json::from_str(
            raw_input
                .get("dataset")
                .context("raw dataset missing")?
                .get(),
        )?;
    let traces = acquire(
        dataset,
        raw_dataset
            .get("records")
            .context("inline records required")?,
    )?;
    let seed = run.random_seed.unwrap_or(0);
    let (corpus, hash_namespace) = match dataset
        .pointer("/prompts/corpus")
        .and_then(|value| value.as_str())
    {
        Some("coding") => (
            aiperf_dataset_synthesis::coding::build_coding_corpus(tokenizer.as_ref(), seed)?,
            aiperf_rng::namespace::DATASET_CODING_CONTENT_CORPUS,
        ),
        Some("sonnet") => (
            aiperf_dataset_synthesis::corpus::tokenize_sonnet_corpus(tokenizer.as_ref())?,
            aiperf_rng::namespace::DATASET_PROMPT_CORPUS,
        ),
        _ => anyhow::bail!("proof requires explicit prompts.corpus coding or sonnet"),
    };
    ensure!(
        dataset
            .get("weka_nested_timestamp_basis")
            .is_none_or(|basis| basis.is_null() || basis == "auto"),
        "proof currently supports only automatic nested timestamp basis; explicit absolute/relative is not silently ignored"
    );
    let hash_seed = aiperf_rng::PythonRandomGenerator::derive_child_seed(seed, hash_namespace);
    let prepared = prepare_legacy_agentic(
        run,
        &phases,
        dataset,
        &traces,
        corpus.into(),
        hash_seed,
        tokenizer.clone(),
        false,
    )?;
    let (phases, origins) = prepare_legacy_agentic_phases(
        run,
        &phases,
        dataset,
        input["system_idle_gap_cap_seconds"].as_f64(),
        &prepared,
    )?;
    (prepared, phases, origins)
    } else {
        let prepared = crate::scheduled::prepare(
            run,
            raw_input.get("dataset").context("raw dataset missing")?,
            &models,
            tokenizer.as_ref(),
            cancel.clone(),
        )?;
        let origins = (0..phases.len()).map(Some).collect::<Vec<_>>();
        (prepared, phases, origins)
    };
    let registry = aiperf_endpoints::EndpointRegistry::builtin()?;
    let profile = default_endpoint_profile(run, &registry)?;
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
    )?;
    prepared.resident_endpoint_binding = Some(binding);
    let samplers = aiperf_dataset::sampler::SamplerRegistry::with_builtin_strategies()?;
    let factory = PreparedNativeConversationSourceFactory {
        identity_source,
        endpoint_resolver: resolver,
        samplers: &samplers,
        materializer: Arc::new(aiperf_dataset::EndpointRequestMaterializer),
        defer_grpc_operations: false,
        ignore_trace_delays: run.cfg.ignore_trace_delays,
        end_conversation_on_truncation: prepared.end_conversation_on_truncation,
        worker_group_partition: None,
        position_addressed: false,
        whole_corpus_positions: false,
        run_sampler: aiperf_multiturn::RunScopedSampler::new(),
    };
    let clock = Rc::new(SimClock::new());
    let engine = match model_factory {
        Some((factory, capacity)) => {
            ModelExecution::from_factory(clock.clone(), factory, capacity)?
        }
        None => ModelExecution::fixed(clock.clone(), 1.0, 1.0, 64)?,
    };
    let dispatcher = Rc::new(Dispatcher {
        execution: engine.clone(),
        records: RefCell::new(Vec::new()),
    });
    let resources = native_scheduled_resources(&phases);
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
            phase_seamless_to_next(&phases, index),
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
            user_centric_phase_scope(&phases, index),
        )?);
    }
    let mut result = None;
    let rendezvous = proof_wait.map(|(events, sequencer)| {
        Rc::new(crate::rendezvous::Rendezvous {
            model: engine.clone(),
            clock: clock.clone(),
            cancellation: cancel.clone(),
            events,
            sequencer,
            benchmark_id: run.benchmark_id.clone(),
            entered: std::cell::Cell::new(false),
            resume_application: std::cell::Cell::new(false),
        })
    });
    let event_source: Rc<dyn aiperf_clock_runtime::clock::driver::SimEventSource> =
        match &rendezvous {
            Some(source) => source.clone(),
            None => engine.clone(),
        };
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
    let backend_report = engine.take_backend_report(clock.now_ns() as f64 / 1_000_000.0);
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
    let mut occurrences = std::collections::BTreeMap::<String, Vec<String>>::new();
    let mut records = dispatcher.records.borrow_mut();
    for row in records.iter_mut() {
        let template = row["request"]["conversation_id"].as_str().unwrap_or("").to_owned();
        let session = row["request"]["correlation_id"].as_str().unwrap_or("").to_owned();
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
        "proof_rendezvous_entered": rendezvous.as_ref().is_some_and(|source|source.entered.get()),
    }))
}

fn acquire(
    dataset: &serde_json::Value,
    records: &serde_json::value::RawValue,
) -> Result<Vec<(String, aiperf_agentx::trace::WekaTrace)>> {
    // Source payload never round-trips through Value: WEKA hash IDs are i128.
    let documents =
        match serde_json::from_str::<Vec<Box<serde_json::value::RawValue>>>(records.get()) {
            Ok(documents) => documents,
            Err(_) => vec![records.to_owned()],
        };
    let (traces, _) = aiperf_agentx::loader::load_weka_traces_from_documents(
        documents,
        "inline AISim proof",
        dataset["entries"].as_u64().map(|n| n as usize),
        dataset
            .pointer("/synthesis/max_context_length")
            .and_then(|v| v.as_i64()),
        dataset
            .pointer("/synthesis/max_osl")
            .and_then(|v| v.as_i64()),
        None,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(traces)
}
