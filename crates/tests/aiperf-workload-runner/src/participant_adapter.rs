// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Replayer hosting of the production AIPerf workload.
//!
//! The AISim Replayer event loop owns time and every engine decision. This
//! module hosts the unchanged AIPerf scheduler futures (`run_scheduled_phases`)
//! as a [`WorkloadParticipant`]: `infer` submits through a mailbox instead of
//! driving an engine, and the Replayer reports tokens/terminals back as facts.
//! No `drive_sim*` pump and no `SteppableReplay` is involved; the participant
//! only polls a private `LocalSet` until it parks, and treats `SimClock` as a
//! slave of the Replayer's logical time.
//!
//! Concurrency, arrival and stop policy stay in AIPerf's plans. Backend
//! capacity is the engine's own queueing (`max_num_seqs`); this adapter adds no
//! admission bound and no per-request script/capacity semaphore.

use crate::{
    dispatch::Dispatcher,
    workload::{build_plans, lower_scheduled, parse_head},
};
use aiperf_clock::clock::Clock;
use aiperf_clock_runtime::{cancellation::RunCancellation, clock::SimClock};
use aiperf_engine_kernel::phase_report::PhasedScheduledRunReport;
use aiperf_engine_kernel::scheduled::NativeRecordSink;
use aiperf_engine_phase::phase_runtime::run_scheduled_phases_with_cancellation;
use aiperf_exchange_model::{
    dispatch::{ReplayTerminalStatus as AiperfTerminal, RequestObserver},
    measure::{TurnDispatchOutcome, observe::CompletionError},
};
use aiperf_metrics_model::{Phase, ingest::RecordIngest};
use aiperf_request_execution::execution::{
    Execution, ExecutionError, Observation, Request, RequestOutcome,
};
use aisimulate_core::engine::RejectionCause;
use aisimulate_core::replay::{
    DirectRequest, ParticipantStep, ReplayTerminalStatus as AisimTerminal, WorkloadParticipant,
};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};
use tokio::{sync::Notify, task::LocalSet};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Phase identity and typed-record capture for one scheduled plan.
///
/// The phase-local `NativeMetricsObserver` is the only observer that receives
/// the engine's request metadata. This sink is attached per plan and receives
/// that observer's post-drain `RecordIngest` rows, so no second observer and no
/// per-request phase lookup are involved.
struct HostedRecordSink {
    phase: Phase,
    phase_index: usize,
    authored_phase_index: Option<usize>,
    profiling_index: Option<usize>,
    phase_name: Rc<str>,
    phase_kind: &'static str,
    captured: Rc<RefCell<Vec<HostedPhaseRecords>>>,
}

impl NativeRecordSink for HostedRecordSink {
    fn on_native_records(&self, mut records: Vec<(Uuid, RecordIngest)>) {
        for (_, record) in &mut records {
            record.phase_index = self.authored_phase_index;
            record.profiling_index = self.profiling_index;
            record.phase_name = self
                .authored_phase_index
                .map(|_| self.phase_name.to_string());
            record.phase_kind = Some(Cow::Borrowed(self.phase_kind));
        }
        self.captured.borrow_mut().push(HostedPhaseRecords {
            phase: self.phase,
            phase_index: self.phase_index,
            phase_name: self.phase_name.to_string(),
            phase_kind: self.phase_kind,
            records,
        });
    }
}

/// One phase's exact typed native records, captured after drain.
pub struct HostedPhaseRecords {
    /// Engine-recorded credit phase for this plan.
    pub phase: Phase,
    /// Zero-based position in the resolved phase program.
    pub phase_index: usize,
    /// Stable phase identifier from the plan.
    pub phase_name: String,
    /// `warmup` or `profiling`.
    pub phase_kind: &'static str,
    /// Engine `RecordIngest` rows with the phase position stamped.
    pub records: Vec<(Uuid, RecordIngest)>,
}

/// A same-instant re-poll loop that never parks on the clock is a livelock.
const MAX_SAME_INSTANT_POLLS: u32 = 100_000;

#[derive(Default)]
struct Mailbox {
    /// `(clock ns, token id)` facts delivered by the Replayer.
    outputs: RefCell<VecDeque<(i64, u32)>>,
    terminal: RefCell<Option<(i64, AisimTerminal)>>,
    prompt_cache_read_tokens: RefCell<Option<usize>>,
    rejection_cause: RefCell<Option<RejectionCause>>,
    ready: Notify,
}

#[derive(Default)]
struct ExecState {
    live: BTreeMap<u128, Rc<Mailbox>>,
    /// Identities whose AIPerf future ended first; late facts are tolerated.
    retired: BTreeSet<u128>,
    submits: Vec<DirectRequest>,
    cancels: Vec<Uuid>,
    total: usize,
}

/// Mailbox-backed `Execution`: submits are drained by the participant, facts are
/// routed back by request identity.
pub struct ParticipantExecution {
    clock: Rc<SimClock>,
    state: RefCell<ExecState>,
    skip_context_overflow: bool,
}

impl ParticipantExecution {
    fn new(clock: Rc<SimClock>, skip_context_overflow: bool) -> Rc<Self> {
        Rc::new(Self {
            clock,
            state: RefCell::default(),
            skip_context_overflow,
        })
    }

    /// Cancel intent for a request whose AIPerf future is ending early. Only a
    /// still-live engine request needs a native cancel; facts that race it are
    /// tolerated through `retired`.
    fn retire(&self, id: u128) {
        let mut state = self.state.borrow_mut();
        if state.live.remove(&id).is_some() {
            state.retired.insert(id);
            state.cancels.push(Uuid::from_u128(id));
        }
    }

    fn token(&self, uuid: Uuid, at_ns: i64, token: u32) -> Result<()> {
        let state = self.state.borrow();
        match state.live.get(&uuid.as_u128()) {
            Some(mailbox) => {
                mailbox.outputs.borrow_mut().push_back((at_ns, token));
                mailbox.ready.notify_one();
                Ok(())
            }
            None if state.retired.contains(&uuid.as_u128()) => Ok(()),
            None => bail!("output token for unknown request {uuid}"),
        }
    }

    fn terminal(&self, uuid: Uuid, at_ns: i64, status: AisimTerminal) -> Result<()> {
        let state = self.state.borrow();
        match state.live.get(&uuid.as_u128()) {
            Some(mailbox) => {
                *mailbox.terminal.borrow_mut() = Some((at_ns, status));
                mailbox.ready.notify_one();
                Ok(())
            }
            None if state.retired.contains(&uuid.as_u128()) => Ok(()),
            None => bail!("terminal for unknown request {uuid}"),
        }
    }
}

/// Retires the engine request if the AIPerf future ends before its terminal.
struct RequestGuard<'a> {
    execution: &'a ParticipantExecution,
    id: u128,
}
impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.execution.retire(self.id);
    }
}

fn aiperf_terminal(status: AisimTerminal) -> AiperfTerminal {
    match status {
        AisimTerminal::Completed => AiperfTerminal::Completed,
        AisimTerminal::Rejected => AiperfTerminal::Rejected,
        AisimTerminal::Canceled => AiperfTerminal::Canceled,
        AisimTerminal::Failed => AiperfTerminal::Failed,
    }
}

impl Execution for ParticipantExecution {
    fn now(&self) -> i64 {
        self.clock.now_ns()
    }
    fn sleep_until(&self, deadline_ns: i64) -> Pin<Box<dyn Future<Output = ()> + '_>> {
        self.clock.clone().sleep_until(deadline_ns)
    }
    fn supports_transport_start(&self) -> bool {
        true
    }
    fn discard_request(&self, uuid: Uuid) {
        self.retire(uuid.as_u128());
    }

    fn infer<'a>(
        &'a self,
        request: Request,
        cancel: &'a CancellationToken,
        observer: &'a dyn RequestObserver,
        callback: &'a mut dyn FnMut(Observation),
    ) -> Pin<Box<dyn Future<Output = Result<Option<RequestOutcome>, ExecutionError>> + 'a>> {
        Box::pin(async move {
            let uuid = request.prepared.request.uuid;
            let id = uuid.as_u128();
            if cancel.is_cancelled() {
                return Ok(None);
            }
            if !request.images.is_empty() {
                return Err(ExecutionError::Failed(
                    "native-hosted AgentX proof is text-only; modeled image geometry unsupported"
                        .into(),
                ));
            }
            if self.state.borrow().live.contains_key(&id)
                || self.state.borrow().retired.contains(&id)
            {
                return Err(ExecutionError::Failed("duplicate request identity".into()));
            }
            let start_ns = self.now();
            observer.on_admit(uuid, start_ns, 0);
            callback(Observation::TransportStarted { at_ns: start_ns });
            // Admission/start callbacks may cancel synchronously; nothing awaits
            // between this check and registration.
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let input_tokens = request.input_token_ids.len();
            let mailbox = Rc::new(Mailbox::default());
            {
                let mut state = self.state.borrow_mut();
                state.live.insert(id, mailbox.clone());
                state.total += 1;
                state.submits.push(DirectRequest {
                    tokens: request.input_token_ids,
                    max_output_tokens: request.prepared.request.max_output_tokens,
                    uuid: Some(uuid),
                    arrival_timestamp_ms: Some(start_ns as f64 / 1_000_000.0),
                    ..Default::default()
                });
            }
            let _guard = RequestGuard {
                execution: self,
                id,
            };
            let mut tokens = Vec::new();
            let mut last_content_ns = None;
            let (end_ns, status) = loop {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let notified = mailbox.ready.notified();
                let output = mailbox.outputs.borrow_mut().pop_front();
                if let Some((at_ns, token)) = output {
                    if tokens.is_empty() {
                        callback(Observation::FirstToken {
                            ttft_ns: at_ns.saturating_sub(start_ns),
                        });
                    }
                    tokens.push(token);
                    last_content_ns = Some(at_ns);
                    observer.on_output_tokens(uuid, at_ns, 1);
                    callback(Observation::OutputTokenIds {
                        at_ns,
                        ids: vec![token],
                    });
                    continue;
                }
                if let Some(terminal) = *mailbox.terminal.borrow() {
                    break terminal;
                }
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => return Ok(None),
                    () = notified => {},
                }
            };
            // Terminal reached: engine ownership ended; nothing to cancel on drop.
            self.state.borrow_mut().live.remove(&id);
            self.state.borrow_mut().retired.insert(id);
            let terminal = aiperf_terminal(status);
            let context_overflow = status == AisimTerminal::Rejected
                && *mailbox.rejection_cause.borrow() == Some(RejectionCause::ContextOverflow);
            if context_overflow {
                observer.on_context_overflow(uuid, self.skip_context_overflow);
            }
            let completion_error = match status {
                AisimTerminal::Rejected | AisimTerminal::Failed => CompletionError::Present,
                _ => CompletionError::Absent,
            };
            let response_text = if status == AisimTerminal::Completed {
                aiperf_request_execution::model_input::synthetic_reply(uuid, tokens.len())
            } else {
                String::new()
            };
            observer.on_input_tokens(uuid, Some(input_tokens));
            observer.on_usage(
                uuid,
                aiperf_exchange_model::dispatch::ObservedUsage {
                    prompt_tokens: Some(input_tokens),
                    completion_tokens: Some(tokens.len()),
                    prompt_cache_read_tokens: *mailbox.prompt_cache_read_tokens.borrow(),
                    ..Default::default()
                },
            );
            let dispatch = TurnDispatchOutcome {
                completion_error,
                start_ns,
                end_ns,
                last_content_ns,
                terminal,
                response_text,
                model_response: aiperf_exchange_model::measure::observe::ModelResponseMetadata {
                    context_overflow,
                    ..Default::default()
                },
                chat_capture: None,
                prompt_tokens: Some(input_tokens as u64),
                completion_tokens: Some(tokens.len() as u64),
                http: Default::default(),
            };
            observer.on_operation_settled(uuid, terminal, start_ns, end_ns);
            observer.on_response_settled(uuid);
            observer.on_terminal_at(uuid, terminal, end_ns);
            Ok(Some(RequestOutcome {
                dispatch,
                output_token_ids: Some(tokens),
            }))
        })
    }
}

struct FlagWaker(Arc<AtomicBool>);
impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct Shared {
    dispatcher: Rc<Dispatcher>,
    execution: Rc<ParticipantExecution>,
    clock: Rc<SimClock>,
    captured_records: Rc<RefCell<Vec<HostedPhaseRecords>>>,
    coverage_plan: aiperf_engine_kernel::profile_metric_coverage::ProfileMetricCoveragePlan,
    metrics_config: aiperf_metrics_model::MetricsConfig,
    prepared_roots: usize,
    result: RefCell<Option<Result<PhasedScheduledRunReport>>>,
}

/// Typed AIPerf-side results retained after the Replayer consumes the participant.
///
/// Records come from each phase's own `NativeMetricsObserver`; their metadata is
/// engine-produced and only the phase position is stamped by the capture sink.
pub struct HostedTypedReport {
    /// AIPerf's phase report, including per-phase native metric summaries.
    pub phased: PhasedScheduledRunReport,
    /// Shared coverage projection of the exact lowered phase program.
    pub coverage_plan: aiperf_engine_kernel::profile_metric_coverage::ProfileMetricCoveragePlan,
    /// Resolved metric policy for typed record reductions.
    pub metrics_config: aiperf_metrics_model::MetricsConfig,
    /// Complete root trees retained by source preparation.
    pub prepared_roots: usize,
    /// Exact typed records grouped by phase, in resolved phase order.
    pub records: Vec<HostedPhaseRecords>,
    /// Logical run end at finalization.
    pub virtual_end_ns: i64,
    /// Participant dispatch provenance, retained independently of the metrics.
    pub dispatch: HostedDispatchEvidence,
}

/// Dispatch-side evidence retained with a typed hosted report.
pub struct HostedDispatchEvidence {
    /// Dispatcher request rows exactly as captured, in issue order.
    pub requests: Vec<serde_json::Value>,
    /// Requests submitted to the participant execution loop.
    pub submitted_requests: usize,
}

/// AIPerf-side results retained after the Replayer consumes the participant.
pub struct HostedRunHandle {
    shared: Rc<Shared>,
}

impl HostedRunHandle {
    /// Opt into compact dispatch evidence before any workload request is submitted.
    pub fn use_compact_dispatch_evidence(&self) -> Result<()> {
        ensure!(
            self.shared.execution.state.borrow().total == 0,
            "compact dispatch evidence must be selected before execution"
        );
        self.shared.dispatcher.set_compact_capture();
        Ok(())
    }

    /// AIPerf's own phase report plus every dispatcher-issued request row.
    /// Backend measurements belong to the Replayer's `ReplayReport`.
    pub fn into_report(self) -> Result<serde_json::Value> {
        let report = self
            .shared
            .result
            .borrow_mut()
            .take()
            .context("native-hosted AIPerf run did not complete")??;
        let mut records = self.shared.dispatcher.records.borrow_mut();
        crate::workload::label_occurrences(&mut records);
        Ok(serde_json::json!({
            "format": "aisim-aiperf-native-hosted-v1",
            "native_phase_report": report,
            "requests": records.as_slice(),
            "virtual_end_ns": self.shared.clock.now_ns(),
        }))
    }

    /// Consume the run into typed phase and native-record results.
    ///
    /// Each phase's records are the rows its own `NativeMetricsObserver`
    /// produced; per-phase summaries remain in `phased.reports`. The dispatcher
    /// rows are retained as dispatch evidence and are never reparsed.
    pub fn into_typed_report(self) -> Result<HostedTypedReport> {
        let phased = self
            .shared
            .result
            .borrow_mut()
            .take()
            .context("native-hosted AIPerf run did not complete")??;
        let virtual_end_ns = self.shared.clock.now_ns();
        let mut records = std::mem::take(&mut *self.shared.captured_records.borrow_mut());
        records.sort_by_key(|phase| phase.phase_index);
        let mut requests = std::mem::take(&mut *self.shared.dispatcher.records.borrow_mut());
        crate::workload::label_occurrences(&mut requests);
        let submitted_requests = self.shared.execution.state.borrow().total;
        Ok(HostedTypedReport {
            phased,
            coverage_plan: self.shared.coverage_plan.clone(),
            metrics_config: self.shared.metrics_config.clone(),
            prepared_roots: self.shared.prepared_roots,
            records,
            virtual_end_ns,
            dispatch: HostedDispatchEvidence {
                requests,
                submitted_requests,
            },
        })
    }
}

/// The production AgentX workload as a Replayer-hosted participant.
pub struct AgentXParticipant {
    // Field order is drop order: tasks (borrowing nothing) die before the runtime.
    local: LocalSet,
    runtime: tokio::runtime::Runtime,
    shared: Rc<Shared>,
    flag: Arc<AtomicBool>,
    waker: Waker,
    /// Facts arrived after the last `advance_to`; the host polls us again.
    dirty: bool,
    last_now_ms: f64,
}

impl AgentXParticipant {
    fn pump(&mut self) -> Result<()> {
        let _enter = self.runtime.enter();
        let mut stalled = 0_u32;
        let mut mark = (
            self.shared.clock.now_ns(),
            self.shared.clock.scheduled_count(),
        );
        loop {
            self.flag.store(false, Ordering::SeqCst);
            let mut cx = Context::from_waker(&self.waker);
            match Pin::new(&mut self.local).poll(&mut cx) {
                Poll::Ready(()) => break,
                Poll::Pending => {}
            }
            let now = (
                self.shared.clock.now_ns(),
                self.shared.clock.scheduled_count(),
            );
            if now == mark {
                stalled += 1;
                ensure!(
                    stalled < MAX_SAME_INSTANT_POLLS,
                    "AIPerf workload self-woke at {}ns without parking on the clock",
                    now.0
                );
            } else {
                mark = now;
                stalled = 0;
            }
            if self.flag.load(Ordering::SeqCst) {
                continue;
            }
            // Parked. A sleeper registered for an already-due instant fires now;
            // a later one is reported through `next_wakeup_ms`.
            if self.shared.clock.next_event_time() == Some(self.shared.clock.now_ns()) {
                self.shared.clock.advance_to(self.shared.clock.now_ns());
                continue;
            }
            break;
        }
        Ok(())
    }
}

/// Least integer ns whose forward conversion reaches `ms` (never runs early).
fn ns_from_ms(ms: f64) -> Result<i64> {
    crate::model::deployment_deadline_ns(ms).map_err(|error| anyhow!("{error}"))
}

impl WorkloadParticipant for AgentXParticipant {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        if self.dirty || self.flag.load(Ordering::SeqCst) {
            return Some(self.last_now_ms);
        }
        self.shared
            .clock
            .next_event_time()
            .map(|ns| ns as f64 / 1_000_000.0)
    }

    fn advance_to(&mut self, now_ms: f64, _cluster_in_flight: usize) -> Result<ParticipantStep> {
        let now_ns = ns_from_ms(now_ms)?;
        self.last_now_ms = now_ms;
        self.dirty = false;
        self.shared.clock.advance_to(now_ns);
        self.pump()?;
        if matches!(&*self.shared.result.borrow(), Some(Err(_))) {
            return Err(self.shared.result.borrow_mut().take().unwrap().unwrap_err());
        }
        let mut state = self.shared.execution.state.borrow_mut();
        let mut submits = std::mem::take(&mut state.submits);
        let mut cancels = std::mem::take(&mut state.cancels);
        // A request submitted and cancelled inside one pump never reaches the
        // engine (the host cannot cancel a same-step submission).
        cancels.retain(|id| {
            let before = submits.len();
            submits.retain(|submit| submit.uuid != Some(*id));
            submits.len() == before
        });
        Ok(ParticipantStep { submits, cancels })
    }

    fn on_output_token(&mut self, request_id: Uuid, token_id: u32, now_ms: f64) -> Result<()> {
        self.dirty = true;
        self.last_now_ms = now_ms;
        self.shared
            .execution
            .token(request_id, ns_from_ms(now_ms)?, token_id)
    }

    fn on_prompt_cache_read_tokens(
        &mut self,
        request_id: Uuid,
        tokens: usize,
        _now_ms: f64,
    ) -> Result<()> {
        let state = self.shared.execution.state.borrow();
        if let Some(mailbox) = state.live.get(&request_id.as_u128()) {
            // The first reporting role owns prompt cache usage. A later
            // decode-side zero describes transferred KV, not prefill hits.
            mailbox
                .prompt_cache_read_tokens
                .borrow_mut()
                .get_or_insert(tokens);
        } else if !state.retired.contains(&request_id.as_u128()) {
            bail!("cache usage for unknown request {request_id}");
        }
        Ok(())
    }

    fn on_rejection_cause(
        &mut self,
        request_id: Uuid,
        cause: RejectionCause,
        _now_ms: f64,
    ) -> Result<()> {
        let state = self.shared.execution.state.borrow();
        if let Some(mailbox) = state.live.get(&request_id.as_u128()) {
            mailbox.rejection_cause.borrow_mut().get_or_insert(cause);
        } else if !state.retired.contains(&request_id.as_u128()) {
            bail!("rejection cause for unknown request {request_id}");
        }
        Ok(())
    }

    fn on_terminal(&mut self, request_id: Uuid, now_ms: f64, status: AisimTerminal) -> Result<()> {
        self.dirty = true;
        self.last_now_ms = now_ms;
        self.shared
            .execution
            .terminal(request_id, ns_from_ms(now_ms)?, status)
    }

    fn is_drained(&self) -> bool {
        let state = self.shared.execution.state.borrow();
        self.shared.result.borrow().is_some()
            && state.live.is_empty()
            && state.submits.is_empty()
            && state.cancels.is_empty()
    }

    fn total_requests(&self) -> usize {
        self.shared.execution.state.borrow().total
    }
}

/// Input preparation selected once before hosting the shared workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeInputProjection {
    /// Reconstruct authored live endpoint text for semantic comparison controls.
    Text,
    /// Preserve trace hash identity and recorded token geometry without text/tokenization.
    TraceHashes,
}

/// Prepare the real run through the shared lowering and host its plans.
///
/// Everything here is constructed on the calling thread, which must also be the
/// thread that calls `Replayer::run`; every future owns its state (`Rc`), so
/// the spawned task is `'static` without borrowing this function's locals.
pub fn native_hosted(
    run: &aiperf_config::model::BenchmarkRun,
    input: NativeInputProjection,
    cancel: RunCancellation,
    observer: Rc<dyn aiperf_timing::PhaseObserver>,
    identity_source: Rc<dyn aiperf_rng::identity::IdentitySource>,
) -> Result<(AgentXParticipant, HostedRunHandle)> {
    let head = parse_head(run, input)?;
    ensure!(
        head.kind != aiperf_config::model::workload_kind::WorkloadKind::Graph || head.legacy_weka,
        "native hosting currently binds the production AgentX scheduled path only"
    );
    let prep = lower_scheduled(run, &head, cancel.clone(), identity_source)?;
    let clock = Rc::new(SimClock::new());
    let mut plans = build_plans(run, &prep, clock.clone())?;
    // Typed capture uses each phase's own native-metrics observer. The sink is
    // per phase and receives the post-drain records; it never adds an observer.
    let captured_records = Rc::new(RefCell::new(Vec::new()));
    let mut profiling_index = 0;
    for (phase_index, plan) in plans.iter_mut().enumerate() {
        let measured_index = if prep.phases[phase_index].common().exclude_from_results {
            None
        } else {
            let index = profiling_index;
            profiling_index += 1;
            Some(index)
        };
        plan.native_record_sink = Some(Rc::new(HostedRecordSink {
            phase: match plan.ancillary.phase {
                aiperf_timing::Phase::Warmup => aiperf_metrics_model::Phase::Warmup,
                aiperf_timing::Phase::Profiling => aiperf_metrics_model::Phase::Profiling,
            },
            phase_index,
            authored_phase_index: prep.origins[phase_index],
            profiling_index: measured_index,
            phase_name: Rc::from(plan.config.id.as_str()),
            phase_kind: plan.config.kind.as_str(),
            captured: captured_records.clone(),
        }));
    }
    let execution = ParticipantExecution::new(
        clock.clone(),
        aiperf_agentx::scenario::scenario_drops_agentic_overflow_records(
            run.cfg.scenario.as_deref(),
        ),
    );
    let dispatcher = Rc::new(Dispatcher::new(execution.clone()));
    let shared = Rc::new(Shared {
        dispatcher: dispatcher.clone(),
        execution,
        clock: clock.clone(),
        captured_records,
        coverage_plan:
            aiperf_engine_kernel::profile_metric_coverage::ProfileMetricCoveragePlan::new(
                &prep.phases,
                &prep.origins,
            ),
        metrics_config: plans
            .first()
            .context("native hosted phase program is empty")?
            .metrics_config
            .clone(),
        prepared_roots: prep.prepared.agentic_trees.len(),
        result: RefCell::new(None),
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .context("creating participant runtime")?;
    let local = LocalSet::new();
    {
        let _enter = runtime.enter();
        let shared = shared.clone();
        local.spawn_local(async move {
            let outcome = run_scheduled_phases_with_cancellation(
                plans,
                clock,
                dispatcher,
                observer,
                Some(cancel),
                None,
            )
            .await;
            *shared.result.borrow_mut() = Some(outcome);
        });
    }
    let flag = Arc::new(AtomicBool::new(true));
    let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
    Ok((
        AgentXParticipant {
            local,
            runtime,
            shared: shared.clone(),
            flag,
            waker,
            dirty: true,
            last_now_ms: 0.0,
        },
        HostedRunHandle { shared },
    ))
}
