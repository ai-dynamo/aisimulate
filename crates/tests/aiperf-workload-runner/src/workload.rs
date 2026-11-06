// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-library composition. AgentX reconstruction and scheduling remain upstream-owned.

use std::{rc::Rc, sync::Arc, cell::RefCell};
use anyhow::{Context, Result, ensure};
use aiperf_clock_runtime::{clock::{SimClock, driver::drive_sim_with_source}, cancellation::RunCancellation};
use aiperf_engine_phase::{agentic_replay::prepare::{prepare_legacy_agentic,prepare_legacy_agentic_phases}, execute::{dataset_build::{PreparedNativeConversationSourceFactory,build_native_scheduled_phase_plan_with_source_factory,phase_seamless_to_next,user_centric_phase_scope},plan::native_scheduled_resources}, phase_runtime::run_scheduled_phases_with_cancellation};
use aiperf_exchange_model::TextTokenizer;
use aiperf_protocol::protocol_dto::PhaseSpec;
use crate::{dispatch::Dispatcher, model::ModelExecution};

pub fn execute(run: &aiperf_config::model::BenchmarkRun, cancel: RunCancellation, observer: Rc<dyn aiperf_timing::PhaseObserver>, proof_wait: Option<(aiperf_engine_kernel::event_writer::EventWriterHandle,Arc<aiperf_engine_kernel::event_writer::Sequencer>)>) -> Result<serde_json::Value> {
    let (_, raw) = aiperf_engine_input::execution_input::factory_workload_input_resolved(run)?;
    let input: serde_json::Value = serde_json::from_str(raw.get())?;
    let dataset = input.get("dataset").context("dataset missing")?;
    let phases: Vec<PhaseSpec> = serde_json::from_value(input["phases"].clone())?;
    ensure!(!phases.is_empty(), "explicit phases required");
    ensure!(input.pointer("/tokenizer/name").and_then(|v|v.as_str()) == Some("builtin"), "proof requires explicit builtin tokenizer");
    ensure!(dataset["format"] == "weka_trace", "proof supports explicit weka_trace format only");
    let raw_input: std::collections::BTreeMap<String,Box<serde_json::value::RawValue>> = serde_json::from_str(raw.get())?;
    let raw_dataset: std::collections::BTreeMap<String,Box<serde_json::value::RawValue>> = serde_json::from_str(raw_input.get("dataset").context("raw dataset missing")?.get())?;
    let traces = acquire(dataset,raw_dataset.get("records").context("inline records required")?)?;
    let tokenizer: Arc<dyn TextTokenizer> = Arc::new(aiperf_tokenizers::TiktokenTokenizer::builtin());
    let seed = run.random_seed.unwrap_or(0);
    let corpus = match dataset.pointer("/prompts/corpus").and_then(|value|value.as_str()) {
        Some("coding") => aiperf_dataset_synthesis::coding::build_coding_corpus(tokenizer.as_ref(), seed)?,
        Some("sonnet") => aiperf_dataset_synthesis::corpus::tokenize_sonnet_corpus(tokenizer.as_ref())?,
        _ => anyhow::bail!("proof requires explicit prompts.corpus coding or sonnet"),
    };
    ensure!(dataset.get("weka_nested_timestamp_basis").is_none_or(|basis|basis.is_null() || basis == "auto"),"proof currently supports only automatic nested timestamp basis; explicit absolute/relative is not silently ignored");
    let hash_seed = aiperf_rng::PythonRandomGenerator::derive_child_seed(seed, aiperf_rng::namespace::DATASET_CODING_CONTENT_CORPUS);
    let mut prepared = prepare_legacy_agentic(run, &phases, dataset, &traces, corpus.into(), hash_seed, tokenizer.clone(), false)?;
    let (phases, origins) = prepare_legacy_agentic_phases(run, &phases, dataset, input["system_idle_gap_cap_seconds"].as_f64(), &prepared)?;
    let endpoint_id = aiperf_registry::registry_id::RegistryId::new("chat")?;
    let endpoint = aiperf_endpoints::EndpointRegistry::builtin()?.prepare(&endpoint_id, aiperf_endpoints::RawEndpointConfig { streaming: true, use_server_token_count: false, ..Default::default() })?;
    let mut table = aiperf_endpoints::PreparedEndpointTable::new();
    let key = table.push(endpoint)?;
    let binding = aiperf_engine_input::dataset_input::ResidentDatasetEndpointBinding::new(Arc::new(table), aiperf_exchange_model::credit::PreparedEndpointReference { key, endpoint_id });
    let resolver = binding.resolver()?;
    let models: aiperf_protocol::protocol_dto::ModelsSpec = run.cfg.models.clone().context("explicit model required")?.into();
    let model = &models.items.first().context("one model required")?.name;
    let source = Arc::try_unwrap(prepared.dataset).map_err(|_|anyhow::anyhow!("dataset shared before preparation"))?;
    prepared.dataset = aiperf_multiturn::prepare_resident_dataset_with_timing(source, resolver.as_ref(), model, &aiperf_multiturn::DEFAULT_RESIDENT_DATASET_LOWERER, None)?;
    prepared.resident_endpoint_binding = Some(binding);
    let samplers = aiperf_dataset::sampler::SamplerRegistry::with_builtin_strategies()?;
    let factory = PreparedNativeConversationSourceFactory { endpoint_resolver: resolver, samplers: &samplers, materializer: Arc::new(aiperf_dataset::EndpointRequestMaterializer), defer_grpc_operations:false, ignore_trace_delays:false, end_conversation_on_truncation:prepared.end_conversation_on_truncation, worker_group_partition:None,position_addressed:false,whole_corpus_positions:false,run_sampler:aiperf_multiturn::RunScopedSampler::new() };
    let clock = Rc::new(SimClock::new());
    let engine = ModelExecution::fixed(clock.clone(), 1.0, 1.0, 64)?;
    let dispatcher = Rc::new(Dispatcher { execution: engine.clone(), records: RefCell::new(Vec::new()) });
    let resources = native_scheduled_resources(&phases);
    let root = aiperf_rng::RngRoot::new(run.random_seed);
    let dataset_root = aiperf_rng::RngRoot::new(prepared.random_seed.or(run.random_seed));
    let mut plans = Vec::new();
    for (index, phase) in phases.iter().enumerate() {
        plans.push(build_native_scheduled_phase_plan_with_source_factory(index,Some(index),origins[index],phase,phase_seamless_to_next(&phases,index),&prepared.full_view(),model,prepared.default_output_tokens,dataset_root,root,root,&factory,tokenizer.clone(),clock.clone(),0,&run.benchmark_id,&run.artifact_dir,&[],&resources,None,aiperf_config::failure::OnFailure::scheduled_or_default(None),prepared.agentic_cache_bust_target,None,prepared.agentic_trees.clone(),prepared.agentic_lanes.clone(),prepared.warmup_handoff.clone(),false,0,aiperf_engine_kernel::request_rate::SlotDrawAddress::for_worker_group(0,0,1),None,aiperf_distributed_model::ModuloWorkerGroupPartition::direct(),user_centric_phase_scope(&phases,index))?);
    }
    let mut result = None;
    let rendezvous = proof_wait.map(|(events,sequencer)|Rc::new(crate::rendezvous::Rendezvous { model:engine.clone(),clock:clock.clone(),cancellation:cancel.clone(),events,sequencer,benchmark_id:run.benchmark_id.clone(),entered:std::cell::Cell::new(false),resume_application:std::cell::Cell::new(false) }));
    let event_source: Rc<dyn aiperf_clock_runtime::clock::driver::SimEventSource> = match &rendezvous { Some(source)=>source.clone(),None=>engine.clone() };
    drive_sim_with_source(clock.clone(), event_source, |_| async {
        result = Some(run_scheduled_phases_with_cancellation(plans,clock.clone(),dispatcher.clone(),observer,Some(cancel),None).await);
    })?;
    let report = result.context("driver returned without a report")??;
    ensure!(engine.is_drained(), "simulator retained request/pass ownership after run");
    Ok(serde_json::json!({"format":"aisim-aiperf-library-proof-v1","native_phase_report":report,"requests":dispatcher.records.borrow().as_slice(),"virtual_end_ns":clock.now_ns(),"engine_drained":true,"proof_rendezvous_entered":rendezvous.as_ref().is_some_and(|source|source.entered.get())}))
}

fn acquire(dataset: &serde_json::Value,records:&serde_json::value::RawValue) -> Result<Vec<(String,aiperf_agentx::trace::WekaTrace)>> {
    // Source payload never round-trips through Value: WEKA hash IDs are i128.
    let documents = match serde_json::from_str::<Vec<Box<serde_json::value::RawValue>>>(records.get()) {
        Ok(documents)=>documents,
        Err(_)=>vec![records.to_owned()],
    };
    let (traces, _) = aiperf_agentx::loader::load_weka_traces_from_documents(documents,"inline AISim proof",dataset["entries"].as_u64().map(|n|n as usize),dataset.pointer("/synthesis/max_context_length").and_then(|v|v.as_i64()),dataset.pointer("/synthesis/max_osl").and_then(|v|v.as_i64()),None).map_err(anyhow::Error::msg)?;
    Ok(traces)
}
