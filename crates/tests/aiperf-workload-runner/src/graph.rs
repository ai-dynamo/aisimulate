// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Graph source acquisition and lowering remain owned by the native graph input library.

use aiperf_engine_input::graph_input::{
    BuiltinRunnerGraphInputAdapterResolver, GraphInputContext, GraphInputExecutionPolicy,
    PreparedRunnerGraphInput, load_execution_graph_input,
};
use aiperf_exchange_model::TextTokenizer;
use anyhow::{Context, Result, ensure};
use aiperf_clock::clock::Clock;
use aiperf_clock_runtime::clock::{SimClock, driver::drive_sim_with_source};
use aiperf_engine_graph::{
    graph_execution::{GraphExecutionEvent, GraphExecutionEventSink, GraphNodeIssueAssociation, GraphNodeFinished},
    graph_phase_runtime::{GraphPhaseBackendConfig, GraphPhaseBackendFactory, PreparedGraphPhaseBackend, run_graph_phases},
};
use aiperf_exchange_model::{dispatch::{RequestObserver, ReplayTerminalStatus}, wire::OpenAiChatMessage};
use aiperf_graph::{
    execution::{LocalGraphTraceExecutionBackend, TracePlacement, TraceCancellation, TraceCompletion, TraceFinalization},
    materialize::{GraphPrompt, MaterializedGraphRequest, SegmentItemsMaterializer},
    policy::{CompositeNodeDispatchPolicy, NodeDispatchPolicy, PrefillSlotNodePolicy, CancellationNodePolicy, StopConditionNodePolicy, GraphNodeIssueObserver, NodeDispatchInfo, AbortTraceNodeFailurePolicy, ResilientNodeFailurePolicy},
    sink::{GraphSink, GraphReply, GraphDispatchOptions, GraphDispatchContext, GraphNodeIssueObservation},
};
use aiperf_metrics::observer::NativeMetricsObserver;
use aiperf_metrics_model::{Phase, observer_metadata::{RequestMetricMetadata, NativeResponseMetadata}};
use aiperf_request_execution::{
    dispatch::{PreparedTurn, PreparedEndpointBinding, OperationSettlement, OperationControlFacts},
    execution::{Execution, Observation, Request},
};
use async_trait::async_trait;
use std::{cell::{Cell, RefCell}, collections::BTreeMap, rc::{Rc, Weak}, sync::Arc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Prepare the first bounded graph formats without translating their request programs.
pub(crate) fn prepare(
    run: &aiperf_config::model::BenchmarkRun,
    dataset: &serde_json::value::RawValue,
    tokenizer: &dyn TextTokenizer,
    phases: &[aiperf_protocol::protocol_dto::PhaseSpec],
) -> Result<PreparedRunnerGraphInput> {
    let format = aiperf_engine_input::graph_input::graph_input_format(dataset)?;
    ensure!(
        matches!(format.as_str(), "dag_jsonl" | "dynamo_trace"),
        "graph simulator proof has no execution coverage for format {format:?}"
    );
    let context = GraphInputContext {
        tokenizer,
        run_random_seed: run.random_seed,
    };
    let policy = GraphInputExecutionPolicy {
        has_warmup_phase: phases.iter().any(|phase| phase.common().exclude_from_results),
        ..Default::default()
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating graph source preparation runtime")?;
    tokio::task::LocalSet::new().block_on(&runtime, async {
        load_execution_graph_input(
            &BuiltinRunnerGraphInputAdapterResolver::new(),
            dataset,
            &context,
            "chat",
            policy,
            None,
        )
        .await
    })
}

struct IssueEvents(Arc<dyn GraphExecutionEventSink>);
impl GraphNodeIssueObserver for IssueEvents {
    fn on_issue(&self, info: &NodeDispatchInfo, observation: GraphNodeIssueObservation) -> std::result::Result<(), aiperf_graph::policy::GraphPolicyError> {
        self.0.emit(GraphExecutionEvent::NodeIssued {
            trace_id: info.trace_id.clone(), node_id: info.node_id.clone(),
            issue: Some(GraphNodeIssueAssociation { worker_id: 0, observation }),
        }).map_err(|error| aiperf_graph::policy::GraphPolicyError::Other(error.to_string()))
    }
}

struct BackendFactory {
    clock: Rc<SimClock>,
    model: Rc<crate::model::ModelExecution>,
    segments: Arc<dyn aiperf_segment_store::segment::SegmentStore>,
    model_name: String,
    default_max_tokens: usize,
    next_uuid: Rc<Cell<u128>>,
    records: Rc<RefCell<Vec<serde_json::Value>>>,
    on_failure: aiperf_config::failure::OnFailure,
    ignore_trace_delays: bool,
}

impl GraphPhaseBackendFactory for BackendFactory {
    fn prepare_backend(&self, config: GraphPhaseBackendConfig) -> Result<PreparedGraphPhaseBackend> {
        let mut policies = Vec::<Rc<dyn NodeDispatchPolicy>>::new();
        let slots = config.prefill_concurrency.map(|limit| Rc::new(aiperf_adaptive::SlotPool::new(limit)));
        if let Some(pool) = &slots { policies.push(Rc::new(PrefillSlotNodePolicy::new(pool.clone()))); }
        if let Some(cancellation) = config.cancellation {
            policies.push(Rc::new(CancellationNodePolicy::new(Box::new(aiperf_timing::BernoulliFixedDelay::new(
                Some(cancellation.rate), cancellation.delay_seconds,
                cancellation.rng_root.derive_indexed_root(aiperf_rng::namespace::GRAPH_NODE_CANCELLATION_WORKER, 0),
            )?), cancellation.phase)));
        }
        if let Some(stop) = config.dag_child_stop {
            policies.push(Rc::new(StopConditionNodePolicy::new(
                stop.checker, stop.run_state, self.clock.clone(), stop.truncations,
                stop.request_budget.clone(), stop.request_budget.map(|_| Arc::new(IssueEvents(config.events.clone())) as Arc<dyn GraphNodeIssueObserver>),
            )));
        }
        let placement = Rc::new(Placement {
            clock: self.clock.clone(), model: self.model.clone(),
            materializer: Rc::new(SegmentItemsMaterializer::new(self.segments.clone())),
            model_name: self.model_name.clone(), default_max_tokens: self.default_max_tokens,
            next_uuid: self.next_uuid.clone(), records: self.records.clone(),
            phase: config.metrics_phase, events: config.events,
            policy: Rc::new(CompositeNodeDispatchPolicy::new(policies)),
            slots, active: RefCell::new(BTreeMap::new()), cancelled: Cell::new(false),
            next_trace: Cell::new(0), on_failure: self.on_failure,
            ignore_trace_delays: self.ignore_trace_delays,
        });
        Ok(PreparedGraphPhaseBackend::new(
            aiperf_engine_kernel::distributed_hooks::PreparedGraphAllocation::new(1)?, placement, true,
        ))
    }
}

struct Placement {
    clock: Rc<SimClock>, model: Rc<crate::model::ModelExecution>,
    materializer: Rc<SegmentItemsMaterializer>, model_name: String, default_max_tokens: usize,
    next_uuid: Rc<Cell<u128>>, records: Rc<RefCell<Vec<serde_json::Value>>>, phase: Phase,
    events: Arc<dyn GraphExecutionEventSink>, policy: Rc<dyn NodeDispatchPolicy>,
    slots: Option<Rc<aiperf_adaptive::SlotPool>>,
    active: RefCell<BTreeMap<u64, Weak<LocalGraphTraceExecutionBackend<OpenAiChatMessage>>>>,
    cancelled: Cell<bool>, next_trace: Cell<u64>, on_failure: aiperf_config::failure::OnFailure,
    ignore_trace_delays: bool,
}

#[async_trait(?Send)]
impl TracePlacement for Placement {
    async fn execute_trace(&self, program: aiperf_graph_model::model::GraphTraceProgram, cancellation: Option<TraceCancellation>) -> std::result::Result<TraceCompletion, aiperf_graph::errors::TraceError> {
        use aiperf_graph::errors::TraceError;
        if self.cancelled.get() { return Err(TraceError::Cancelled("graph phase cancelled".into())); }
        if !program.is_static_graph_program() { return Err(TraceError::UnsupportedDriver(program.driver.kind)); }
        let input_tokens = program.profiling.graph.nodes.iter().filter_map(|(id, node)| {
            node.as_llm().and_then(|node| node.metadata.get("input_tokens")).and_then(serde_json::Value::as_u64).map(|tokens| (id.clone(), tokens as usize))
        }).collect();
        let sink = Rc::new(ModelSink {
            clock: self.clock.clone(), model: self.model.clone(), model_name: self.model_name.clone(),
            default_max_tokens: self.default_max_tokens, next_uuid: self.next_uuid.clone(),
            records: self.records.clone(), phase: self.phase, trace_id: program.profiling.trace.id.clone(),
            events: self.events.clone(), native: Rc::new(NativeMetricsObserver::new(self.clock.clone(), 0, Default::default())),
            input_tokens, cancellation: cancellation.clone(),
        });
        let backend = LocalGraphTraceExecutionBackend::new(self.clock.clone(), self.materializer.clone(), sink)
            .with_node_policy(self.policy.clone())
            .with_node_failure(match self.on_failure {
                aiperf_config::failure::OnFailure::Abort => Rc::new(AbortTraceNodeFailurePolicy) as Rc<dyn aiperf_graph::policy::NodeFailurePolicy>,
                aiperf_config::failure::OnFailure::Continue => Rc::new(ResilientNodeFailurePolicy),
            })
            .with_executor_flags(aiperf_graph::executor::ExecutorFlags { ignore_edge_delays: self.ignore_trace_delays, ..Default::default() });
        let backend = Rc::new(backend);
        let id = self.next_trace.get(); self.next_trace.set(id + 1);
        self.active.borrow_mut().insert(id, Rc::downgrade(&backend));
        let result = backend.execute_trace(program, cancellation).await;
        self.active.borrow_mut().remove(&id);
        result
    }
    async fn finish_trace_reports(&self, _trace_id: &str, execution: std::result::Result<TraceCompletion, aiperf_graph::errors::TraceError>) -> TraceFinalization {
        match self.model.finish_reports().await {
            Ok(()) => TraceFinalization::reportless(execution),
            Err(error) => TraceFinalization::unacknowledged(execution, aiperf_graph::errors::TraceError::Other(error.to_string())),
        }
    }
    fn cancel_inflight(&self) -> std::result::Result<(), aiperf_graph::errors::TraceError> {
        self.cancelled.set(true);
        for backend in self.active.borrow().values().filter_map(Weak::upgrade) { backend.cancel_inflight()?; }
        Ok(())
    }
    fn truncate_inflight(&self) -> std::result::Result<(), aiperf_graph::errors::TraceError> {
        for backend in self.active.borrow().values().filter_map(Weak::upgrade) { backend.truncate_inflight()?; }
        Ok(())
    }
    fn set_prefill_limit(&self, limit: usize) -> std::result::Result<(), aiperf_graph::errors::TraceError> {
        let slots = self.slots.as_ref().ok_or_else(|| aiperf_graph::errors::TraceError::Other("graph phase has no prefill slots".into()))?;
        if limit == 0 { return Err(aiperf_graph::errors::TraceError::Other("prefill limit must be positive".into())); }
        slots.set_limit(limit); Ok(())
    }
}

struct ModelSink {
    clock: Rc<SimClock>, model: Rc<crate::model::ModelExecution>, model_name: String,
    default_max_tokens: usize, next_uuid: Rc<Cell<u128>>, records: Rc<RefCell<Vec<serde_json::Value>>>,
    phase: Phase, trace_id: String, events: Arc<dyn GraphExecutionEventSink>,
    native: Rc<NativeMetricsObserver>, input_tokens: BTreeMap<String, usize>, cancellation: Option<TraceCancellation>,
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for ModelSink {
    fn publish_request(&self, _node_id: &str, mut request: MaterializedGraphRequest, _context: Option<&GraphDispatchContext>) -> MaterializedGraphRequest {
        request.prompt.publish_original(); request
    }
    async fn dispatch(&self, _node_id: &str, _prompt: GraphPrompt, _max_tokens: Option<usize>, _first: &dyn Fn()) -> Result<GraphReply<OpenAiChatMessage>> {
        anyhow::bail!("graph execution requires the full native materialized request")
    }
    async fn dispatch_request_with_context(&self, node_id: &str, request: MaterializedGraphRequest, _context: &GraphDispatchContext, options: GraphDispatchOptions, first: &dyn Fn()) -> Result<GraphReply<OpenAiChatMessage>> {
        self.infer(node_id, request, options, first).await
    }
}

impl ModelSink {
    async fn infer(&self, node_id: &str, request: MaterializedGraphRequest, options: GraphDispatchOptions, first: &dyn Fn()) -> Result<GraphReply<OpenAiChatMessage>> {
        request.prompt.validate_publication()?;
        let wires = request.prompt.message_wires()?;
        let encoder = aiperf_request_execution::model_input::OpenAiRequestEncoder::default();
        let mut ids = match self.input_tokens.get(node_id).copied() {
            Some(tokens) => encoder.encode_graph_messages(&wires, tokens)?,
            None => encoder.encode_graph_messages_exact(&wires)?,
        };
        if ids.is_empty() { ids.push(0); }
        let input_tokens = ids.len();
        let cap = request.max_tokens.unwrap_or(self.default_max_tokens).max(1);
        let ordinal = self.next_uuid.get();
        self.next_uuid.set(ordinal.checked_add(1).context("graph request identity exhausted")?);
        let uuid = Uuid::from_u128(ordinal);
        let model = request.model.as_deref().unwrap_or(&self.model_name);
        let mut turn = aiperf_engine_graph::graph_execution::materialized_graph_messages_turn(
            node_id, &request.prompt, request.uses_raw_messages, false, None,
            aiperf_engine_input::graph_input::CacheBustTarget::None,
        )?;
        turn.max_tokens = request.max_tokens.map(|value| value as u32);
        turn.raw_tools = request.tools.map(|wire| serde_json::from_slice(&wire)).transpose()?;
        if let Some(extra) = request.additional_body { turn.extra_body = Some(serde_json::from_slice(&extra)?); }
        let registry = aiperf_endpoints::EndpointRegistry::builtin()?;
        let endpoint_id = aiperf_registry::registry_id::RegistryId::new("chat")?;
        let endpoint = registry.prepare(&endpoint_id, aiperf_endpoints::RawEndpointConfig {
            streaming: request.streaming, use_server_token_count: false, ..Default::default()
        })?;
        let credit_phase = if self.phase == Phase::Warmup { aiperf_exchange_model::credit::CreditPhase::Warmup } else { aiperf_exchange_model::credit::CreditPhase::Profiling };
        let correlation = format!("{}/{node_id}/{uuid}", self.trace_id);
        let turns = [turn];
        let formatted = endpoint.format_payload(&aiperf_endpoints::registry::PreparedRequest::new(
            model, &turns, None, None, credit_phase, None, Some(&self.trace_id), Some(&self.trace_id),
        ))?;
        let wire = formatted.materialize_standalone()?;
        let mut table = aiperf_endpoints::PreparedEndpointTable::new();
        let key = table.push(endpoint)?;
        let prepared = PreparedTurn {
            runtime_session_id: correlation.clone(), parent_correlation_id: None, is_spawn: false,
            responses_continuation_keys: None,
            request: aiperf_request_execution::dispatch::Request {
                uuid, input_length: input_tokens, max_output_tokens: cap, prompt_text: None,
                body: Some(aiperf_segment_store::body_plan::RequestBody::wire(wire.clone())),
                headers: Default::default(), parameters: Default::default(), endpoint_path: None,
                streaming: request.streaming, x_correlation_id: Some(correlation.clone()), is_final_turn: true,
                response_capture: aiperf_exchange_model::transport::ResponseCapture::Auto,
                cancel_after_ns: options.cancel_after_ns, url_index: None, image_count: Some(0),
                recorded_api_time_ns: None, recorded_ttft_ns: None, current_turn_boundary: None,
            },
            model: model.into(), endpoint: PreparedEndpointBinding::Prepared(aiperf_exchange_model::credit::PreparedEndpointReference { key, endpoint_id }),
            endpoint_aware: true, allows_response_chaining: false, data_policy: Default::default(), deferred: None,
        };
        self.native.register_metadata(uuid, RequestMetricMetadata {
            request_index: usize::try_from(ordinal.saturating_sub(1)).ok(), phase: self.phase,
            correlation_id: Some(correlation.clone()), conversation_id: Some(self.trace_id.clone()),
            dimensions: aiperf_metrics_model::ModelEndpoint { endpoint_url: None, model: Some(model.into()) },
            ..Default::default()
        });
        let started = self.clock.now_ns();
        self.native.on_arrival(uuid, started, input_tokens, cap);
        let index = self.records.borrow().len();
        self.records.borrow_mut().push(serde_json::json!({
            "issue_order": index, "issue_ns":started, "status":"issued",
            "request":{"uuid":uuid,"correlation_id":correlation,"conversation_id":self.trace_id,"node_id":node_id,"credit_phase":credit_phase,"body":serde_json::from_slice::<serde_json::Value>(&wire)?},
            "wire":{"body_bytes":wire.as_ref()},"input_token_ids":ids,"observations":[]
        }));
        let cancel = CancellationToken::new();
        let event_error = RefCell::new(None);
        let mut callback = |observation: Observation| {
            self.records.borrow_mut()[index]["observations"].as_array_mut().map(|rows| rows.push(serde_json::json!(format!("{observation:?}"))));
            if matches!(observation, Observation::FirstToken { .. }) {
                if let Err(error) = self.events.emit(GraphExecutionEvent::FirstToken { trace_id: self.trace_id.clone(), uuid }) { *event_error.borrow_mut() = Some(error); }
                first();
            }
        };
        let inference = self.model.infer(Request::from_prepared(prepared, ids, Vec::new()), &cancel, self.native.as_ref(), &mut callback);
        tokio::pin!(inference);
        let result = if let Some(cancellation) = &self.cancellation {
            tokio::select! {
                biased;
                result = &mut inference => result,
                () = cancellation.cancelled() => { cancel.cancel(); inference.await }
            }
        } else { inference.await };
        let observations = self.native.request_scalars(uuid);
        let terminal = match &result {
            Ok(Some(outcome)) => outcome.dispatch.terminal,
            Ok(None) => ReplayTerminalStatus::Canceled,
            Err(_) => ReplayTerminalStatus::Failed,
        };
        let at_ns = result.as_ref().ok().and_then(|outcome| outcome.as_ref()).map_or(self.clock.now_ns(), |outcome| outcome.dispatch.end_ns);
        self.events.emit(GraphExecutionEvent::NodeFinished {
            trace_id: self.trace_id.clone(), node_id: Some(node_id.into()), issue: None,
            finished: GraphNodeFinished {
                uuid, phase: self.phase, request_started_ns: started,
                latency_started_ns: observations.as_ref().map_or(started, |facts| facts.dispatch_start_ns.unwrap_or(facts.credit_issued_ns)),
                run_origin_ns: 0, settlement: OperationSettlement { terminal, at_ns },
                control: OperationControlFacts { context_overflow: result.as_ref().ok().and_then(|outcome| outcome.as_ref()).is_some_and(|outcome| outcome.dispatch.model_response.context_overflow) },
                observations, has_response: result.as_ref().is_ok_and(|outcome| outcome.is_some()),
            },
        })?;
        let response = result.as_ref().ok().and_then(|outcome| outcome.as_ref());
        let text = response.map_or_else(String::new, |outcome| outcome.dispatch.response_text.clone());
        self.native.record_response(uuid, NativeResponseMetadata {
            start_ns: response.map(|outcome| outcome.dispatch.start_ns), end_ns: response.map(|outcome| outcome.dispatch.end_ns),
            last_content_ns: response.and_then(|outcome| outcome.dispatch.last_content_ns),
            prompt_tokens: response.and_then(|outcome| outcome.dispatch.prompt_tokens), completion_tokens: response.and_then(|outcome| outcome.dispatch.completion_tokens),
            http: response.map_or_else(Default::default, |outcome| outcome.dispatch.http), trace_data: None, cancellation_ns: None,
        });
        let ingest = self.native.snapshot_record(uuid, u64::try_from(ordinal.saturating_sub(1)).context("graph record ordinal exceeds u64")?).context("graph inference did not produce native metrics")?;
        self.events.emit(GraphExecutionEvent::Record {
            trace_id: self.trace_id.clone(), node_id: Some(node_id.into()), issue: None, provider_origin: None,
            record: Box::new(aiperf_records::CapturedRecord { uuid, x_correlation_id: self.trace_id.clone(),
                output: aiperf_records::CapturedModelOutput::from_parts(&text, None, None), raw: None, ingest }),
        })?;
        {
            let mut records = self.records.borrow_mut(); let row = &mut records[index];
            row["status"] = serde_json::json!(if response.is_some() { "settled" } else { "cancelled_or_failed" });
            row["end_ns"] = serde_json::json!(at_ns); row["response_text"] = serde_json::json!(text);
            row["output_token_ids"] = serde_json::json!(response.and_then(|outcome| outcome.output_token_ids.clone()));
        }
        if let Some(error) = event_error.borrow_mut().take() { return Err(error.into()); }
        Ok(match terminal { ReplayTerminalStatus::Completed => GraphReply::from_text(text), ReplayTerminalStatus::Canceled => GraphReply::cancelled(), _ => GraphReply::failed() })
    }
}

/// Execute graph phases through the native graph driver and the same actual model binding.
pub(crate) fn execute(
    run: &aiperf_config::model::BenchmarkRun, input: PreparedRunnerGraphInput,
    phases: Vec<aiperf_protocol::protocol_dto::PhaseSpec>, cancellation: aiperf_clock_runtime::cancellation::RunCancellation,
    observer: Rc<dyn aiperf_timing::PhaseObserver>, model_factory: Option<(&aisimulate_core::engine::EngineFactory, usize)>,
) -> Result<serde_json::Value> {
    let clock = Rc::new(SimClock::new());
    let model = match model_factory { Some((factory, capacity)) => crate::model::ModelExecution::from_factory(clock.clone(), factory, capacity)?, None => crate::model::ModelExecution::fixed(clock.clone(), 1.0, 1.0, 64)? };
    let models = run.cfg.models.as_ref().context("explicit graph model required")?;
    ensure!(models.items.len() == 1, "graph proof requires exactly one model");
    ensure!(!input.cache_bust_target.is_enabled(), "graph cache-bust presentation binding is not implemented");
    let records = Rc::new(RefCell::new(Vec::new()));
    let on_failure = aiperf_config::failure::OnFailure::graph_or_default(None);
    let factory = BackendFactory { clock: clock.clone(), model: model.clone(), segments: input.bundle.segments.clone(), model_name: models.items[0].name.clone(), default_max_tokens: input.default_output_tokens, next_uuid: Rc::new(Cell::new(1)), records: records.clone(), on_failure, ignore_trace_delays: run.cfg.ignore_trace_delays };
    let mut result = None;
    drive_sim_with_source(clock.clone(), model.clone(), |_| async {
        result = Some(run_graph_phases(
            &phases, &run.benchmark_id, &run.artifact_dir, &input.bundle, clock.clone(),
            aiperf_rng::RngRoot::new(input.random_seed.or(run.random_seed)), input.allow_dataset_wrap,
            input.cache_bust_target.is_enabled(), input.t_star_window, phases.iter().map(|_| Vec::new()).collect(),
            &factory, on_failure, None, None, observer, Some(cancellation), None, None,
        ).await);
    })?;
    let report = result.context("graph driver returned without report")??;
    ensure!(model.is_drained(), "graph model retained requests after drain");
    let backend = model.take_backend_report(clock.now_ns() as f64 / 1_000_000.0);
    Ok(serde_json::json!({
        "format":"aisim-aiperf-library-proof-v1", "capture_format":"aisim-runtime-values-raw-v1",
        "native_phase_report":{"phases":report.phases,"workload":{"admitted":report.workload.admitted,"completed":report.workload.completed,"cancelled":report.workload.cancelled,"failed":report.workload.failed,"traces_debug":format!("{:?}",report.workload.traces)}},
        "backend_report":backend,"backend_per_request":backend.per_request,
        "requests":records.borrow().as_slice(),"model_submitted_count":records.borrow().len(),
        "virtual_end_ns":clock.now_ns(),"engine_drained":true,
        "coverage":"dag_jsonl and dynamo_trace through native graph lowering and phase driver; other graph families remain unproven"
    }))
}
