// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Simulator-owned model binding, modified from NVIDIA AIPerf.
//! Upstream: https://github.com/NVIDIA-dev/aiperf-rs/blob/d453b97b45c815f08776865498295b9fcd147cb5/crates/engine/simulate/src/simple_execution.rs
//! Apache-2.0; NVIDIA CORPORATION & AFFILIATES, 2026.
//! Modified to consume the shared execution contract and the current AISim core.
//! This is an independent proof runner, not the product execution path.
//!
//! A model pass starts and completes in separate source steps. Completion only
//! queues observations: the existing SimClock driver drains runnable application
//! futures (including callbacks and their successors) before starting another
//! pass. There is no replay driver, steppable ABI, or guessed yield count here.

use aiperf_clock::clock::Clock;
use aiperf_clock_runtime::clock::{
    driver::{SimDriveError, SimEventSource, SimStep},
    sim_clock::SimClock,
};
use aiperf_exchange_model::TextTokenizer;
use aiperf_exchange_model::dispatch::{ReplayTerminalStatus, RequestObserver};
use aiperf_exchange_model::measure::observe::CompletionError;
use aiperf_exchange_model::measure::{ModelResponseMetadata, TurnDispatchOutcome};
use aiperf_request_execution::execution::{
    Execution, ExecutionError, Observation, Outcome, Request,
};
use aisimulate_core::engine::{
    Command, Engine, EngineConfig, EngineFactory, Output, Request as ModelRequest,
    TimingModelConfig,
    generalized::{EngineIdentity, PassId, SchedulerCommand},
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    future::Future,
    num::NonZeroU32,
    pin::Pin,
    rc::Rc,
};
use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Explicit modeled response capabilities, not a replacement physical scheduler.
#[derive(Clone, Debug)]
pub struct ResponseScript {
    /// Exact desired population, independently clipped to the request cap.
    /// Absence selects a deterministic zero-token-ID population of that cap.
    pub output_token_ids: Option<Vec<u32>>,
    /// Original operation terminal, including provider-failed successful dispatches.
    pub terminal: ReplayTerminalStatus,
    /// Authoritative error evidence, never derived from terminal or diagnostics.
    pub completion_error: CompletionError,
    /// Optional explicit response content; otherwise emitted IDs are decoded.
    pub response_text: Option<String>,
    /// Explicit provider metadata and context-overflow classification.
    pub model_response: ModelResponseMetadata,
    /// Explicit bounded native capture, including sticky paid-prefix evidence.
    pub chat_capture: Option<Box<aiperf_exchange_model::measure::observe::ChatCapture>>,
    /// Physical output is tool payload rather than text timing evidence.
    pub is_tool_only: bool,
    /// Stop after this many actual emitted tokens and cancel remaining model work.
    pub fail_after_tokens: Option<usize>,
    /// Delay before the modeled transport start.
    pub pre_send_delay_ns: i64,
    /// Delay between last model output and response settlement.
    pub terminal_tail_ns: i64,
    /// Post-settlement accounting delay; cancellation cannot erase the result.
    pub accounting_delay_ns: i64,
    /// Explicit later report failure, not an inference/provider error.
    pub report_error: Option<String>,
}
impl Default for ResponseScript {
    fn default() -> Self {
        Self {
            output_token_ids: None,
            terminal: ReplayTerminalStatus::Completed,
            completion_error: CompletionError::Absent,
            response_text: None,
            model_response: ModelResponseMetadata::default(),
            chat_capture: None,
            is_tool_only: false,
            fail_after_tokens: None,
            pre_send_delay_ns: 0,
            terminal_tail_ns: 0,
            accounting_delay_ns: 0,
            report_error: None,
        }
    }
}

/// Actual model/application boundary facts, retained for this experimental proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelEvent {
    /// A caller encountered the application admission bound.
    CapacityWait { request_id: u128, at_ns: i64 },
    /// A Submit command was accepted by the actual Engine.
    Submitted { request_id: u128, at_ns: i64 },
    /// The actual Engine committed a pass and selected these requests.
    PassStarted {
        pass_id: u64,
        at_ns: i64,
        request_ids: Vec<u128>,
    },
    /// The actual Engine completed a pass, before application delivery.
    PassCompleted { pass_id: u64, at_ns: i64 },
    /// An application callback returned.
    Observed {
        request_id: u128,
        at_ns: i64,
        is_final: bool,
    },
    /// Request ownership was cancelled, including future-drop cleanup.
    Cancelled { request_id: u128, at_ns: i64 },
}

#[derive(Default)]
struct Mailbox {
    outputs: RefCell<VecDeque<Output>>,
    ready: Notify,
}
struct State {
    engine: Engine,
    pending: Option<(PassId, f64, i64)>,
    requests: BTreeMap<u128, Rc<Mailbox>>,
    active: BTreeSet<u128>,
    scripts: BTreeMap<Uuid, ResponseScript>,
    report_errors: Vec<String>,
    pending_reports: usize,
    retained_settled: BTreeMap<Uuid, Outcome>,
    failure: Option<String>,
}

/// Single-threaded actual AISimulate Engine binding to AIPerf's source pump.
/// The admission permit lives through final callback delivery, not just Submit.
pub struct ModelExecution {
    clock: Rc<SimClock>,
    state: RefCell<State>,
    capacity: Semaphore,
    events: RefCell<Vec<ModelEvent>>,
    reports_ready: Notify,
}

impl ModelExecution {
    /// Build an actual pinned EngineFactory model with fixed pass timing.
    /// `max_sequences` bounds both application admission and engine sequences.
    pub fn fixed(
        clock: Rc<SimClock>,
        prefill_ms: f64,
        decode_ms: f64,
        max_sequences: usize,
    ) -> Result<Rc<Self>, ExecutionError> {
        if max_sequences == 0 || max_sequences > Semaphore::MAX_PERMITS {
            return Err(ExecutionError::Failed("invalid admission capacity".into()));
        }
        let config = EngineConfig {
            timing_model: TimingModelConfig::Fixed {
                prefill_ms,
                decode_ms,
            },
            max_num_seqs: max_sequences,
            enable_prefix_caching: false,
            enable_chunked_prefill: false,
            // The simple proof models one committed pass, not vLLM's default
            // async lookahead queue. A callback-issued successor must be eligible
            // for the next pass rather than skip already-committed batches.
            async_scheduling: Some(false),
            ..Default::default()
        };
        let engine = EngineFactory::new(config)
            .and_then(|factory| factory.build(EngineIdentity::new(0), NonZeroU32::MIN))
            .map_err(|error| ExecutionError::Failed(error.to_string()))?;
        Ok(Self::with_capacity(clock, engine, max_sequences))
    }

    /// Bind a caller-built engine with capacity-one application admission.
    /// This experimental adapter supports engines without internal-work deadlines.
    pub fn new(clock: Rc<SimClock>, engine: Engine) -> Rc<Self> {
        Self::with_capacity(clock, engine, 1)
    }

    fn with_capacity(clock: Rc<SimClock>, engine: Engine, capacity: usize) -> Rc<Self> {
        Rc::new(Self {
            clock,
            state: RefCell::new(State {
                engine,
                pending: None,
                requests: BTreeMap::new(),
                active: BTreeSet::new(),
                scripts: BTreeMap::new(),
                report_errors: Vec::new(),
                pending_reports: 0,
                retained_settled: BTreeMap::new(),
                failure: None,
            }),
            capacity: Semaphore::new(capacity),
            events: RefCell::new(Vec::new()),
            reports_ready: Notify::new(),
        })
    }

    /// Snapshot actual model and callback boundary ordering.
    pub fn trace(&self) -> Vec<ModelEvent> {
        self.events.borrow().clone()
    }

    /// True only when engine, committed pass, and application ownership drained.
    pub fn is_drained(&self) -> bool {
        let state = self.state.borrow();
        state.failure.is_none()
            && state.engine.is_drained()
            && state.pending.is_none()
            && state.requests.is_empty()
    }

    /// Register one explicit response capability by native request identity.
    /// An active or already-scripted identity cannot be silently overwritten.
    pub fn set_response_script(
        &self,
        uuid: Uuid,
        script: ResponseScript,
    ) -> Result<(), ExecutionError> {
        if script.pre_send_delay_ns < 0
            || script.terminal_tail_ns < 0
            || script.accounting_delay_ns < 0
        {
            return Err(ExecutionError::Failed("negative script delay".into()));
        }
        let mut state = self.state.borrow_mut();
        if state.active.contains(&uuid.as_u128()) || state.scripts.contains_key(&uuid) {
            return Err(ExecutionError::Failed(
                "duplicate response script identity".into(),
            ));
        }
        state.scripts.insert(uuid, script);
        Ok(())
    }

    /// Original settled operation retained when its accounting future was dropped.
    /// Its retention proves operation custody, not successful report processing.
    pub fn retained_settled_outcome(&self, uuid: Uuid) -> Option<Outcome> {
        self.state.borrow().retained_settled.get(&uuid).cloned()
    }

    fn cancel_request(&self, id: u128) {
        let mut state = self.state.borrow_mut();
        if state.requests.remove(&id).is_some() {
            // Fixed models have no internal-work deadline and cancellation is
            // valid even with a committed pass. Latch unexpected failures for
            // the source pump rather than panic from a future's Drop.
            if let Err(error) = state.engine.apply_command_effects(
                SchedulerCommand::new(
                    0,
                    Command::CancelRequest {
                        request_id: Uuid::from_u128(id),
                        discard_pending_output: true,
                    },
                ),
                self.now() as f64 / 1_000_000.0,
            ) {
                state.failure = Some(format!(
                    "model cancellation failed for request {id}: {error}"
                ));
            }
            self.events.borrow_mut().push(ModelEvent::Cancelled {
                request_id: id,
                at_ns: self.now(),
            });
        }
    }
}

/// Identity remains reserved through terminal-tail and report accounting, even
/// after physical Engine ownership has ended. It outlives RequestGuard.
struct IdentityGuard<'a> {
    execution: &'a ModelExecution,
    id: u128,
}
impl Drop for IdentityGuard<'_> {
    fn drop(&mut self) {
        self.execution.state.borrow_mut().active.remove(&self.id);
    }
}

struct RequestGuard<'a> {
    execution: &'a ModelExecution,
    id: u128,
}
impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.execution.cancel_request(self.id);
    }
}

/// Local accounting custody: future drop cannot erase a settled operation or
/// silently satisfy a report obligation. No background task or timing is owned.
struct SettledReportGuard<'a> {
    execution: &'a ModelExecution,
    uuid: Uuid,
    outcome: Option<Outcome>,
}
impl Drop for SettledReportGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.execution.state.borrow_mut();
        if let Some(outcome) = self.outcome.take() {
            state.retained_settled.insert(self.uuid, outcome);
            state
                .report_errors
                .push("post-settlement accounting future dropped".into());
        }
        state.pending_reports -= 1;
        self.execution.reports_ready.notify_waiters();
    }
}

impl Execution for ModelExecution {
    fn now(&self) -> i64 {
        self.clock.now_ns()
    }
    fn sleep_until(&self, deadline_ns: i64) -> Pin<Box<dyn Future<Output = ()> + '_>> {
        self.clock.clone().sleep_until(deadline_ns)
    }
    fn supports_transport_start(&self) -> bool {
        true
    }

    fn discard_turn(&self, uuid: Uuid) {
        self.state.borrow_mut().scripts.remove(&uuid);
        self.cancel_request(uuid.as_u128());
    }

    fn finish_reports(&self) -> Pin<Box<dyn Future<Output = Result<(), ExecutionError>> + '_>> {
        Box::pin(async move {
            loop {
                let notified = self.reports_ready.notified();
                if self.state.borrow().pending_reports == 0 {
                    return match self.state.borrow().report_errors.first() {
                        Some(error) => Err(ExecutionError::Failed(format!(
                            "report processing failed: {error}"
                        ))),
                        None => Ok(()),
                    };
                }
                notified.await;
            }
        })
    }

    fn infer<'a>(
        &'a self,
        request: Request,
        cancel: &'a CancellationToken,
        observer: &'a dyn RequestObserver,
        callback: &'a mut dyn FnMut(Observation),
    ) -> Pin<Box<dyn Future<Output = Result<Option<Outcome>, ExecutionError>> + 'a>> {
        // Transfer script custody before the future is polled: dropping even an
        // unpolled inference cannot leave an orphaned registered response.
        let script = self
            .state
            .borrow_mut()
            .scripts
            .remove(&request.turn.uuid)
            .unwrap_or_default();
        Box::pin(async move {
            let uuid = request.turn.uuid;
            let id = uuid.as_u128();
            if cancel.is_cancelled() {
                return Ok(None);
            }
            if !self.state.borrow_mut().active.insert(id) {
                return Err(ExecutionError::Failed(
                    "duplicate active request identity".into(),
                ));
            }
            let _identity = IdentityGuard {
                execution: self,
                id,
            };
            let image_count = if let Some(body) = &request.turn.request_body {
                let wire = body
                    .to_wire()
                    .map_err(|error| ExecutionError::Failed(error.to_string()))?;
                let value: serde_json::Value = serde_json::from_slice(&wire)
                    .map_err(|error| ExecutionError::Failed(error.to_string()))?;
                value
                    .get("messages")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|message| {
                        message.get("content").and_then(serde_json::Value::as_array)
                    })
                    .flatten()
                    .filter(|part| {
                        matches!(
                            part.get("type").and_then(serde_json::Value::as_str),
                            Some("image_url" | "input_image" | "image")
                        )
                    })
                    .count()
            } else {
                request.turn.image_count.unwrap_or(0) as usize
            };
            if image_count != request.images.len() {
                return Err(ExecutionError::Failed(
                    "modeled image inputs require one explicitly supplied processor geometry per native image".into(),
                ));
            }
            let mut previous_end = 0;
            for image in &request.images {
                if image.token_start < previous_end
                    || image.token_start >= image.token_end
                    || image.token_end > request.input_token_ids.len()
                    || image.encoder.sequences == 0
                    || image.encoder.patch_tokens == 0
                    || image.encoder.transformer_tokens == 0
                    || image.encoder.output_tokens == 0
                    || image.embedding_bytes == 0
                {
                    return Err(ExecutionError::Failed(
                        "invalid explicit modeled image geometry".into(),
                    ));
                }
                previous_end = image.token_end;
            }
            if self.capacity.available_permits() == 0 {
                self.events.borrow_mut().push(ModelEvent::CapacityWait {
                    request_id: id,
                    at_ns: self.now(),
                });
            }
            let _permit = tokio::select! {
                biased;
                () = cancel.cancelled() => return Ok(None),
                permit = self.capacity.acquire() => permit.map_err(|error| ExecutionError::Failed(error.to_string()))?,
            };
            if self.state.borrow().requests.contains_key(&id) {
                return Err(ExecutionError::Failed(
                    "duplicate active request identity".into(),
                ));
            }
            let cap = request.turn.max_output_tokens;
            // This experimental population-producing adapter has an explicit bound;
            // do not silently truncate a supported source request or risk OOM.
            const MAX_POPULATION: usize = 1_048_576;
            let population_len = script
                .output_token_ids
                .as_ref()
                .map_or(cap, Vec::len)
                .min(cap);
            if population_len > MAX_POPULATION {
                return Err(ExecutionError::Failed(
                    "experimental modeled output population exceeds 1048576 tokens".into(),
                ));
            }
            let mut population = script.output_token_ids.as_ref().map_or_else(
                || vec![0; population_len],
                |ids| ids.iter().take(cap).copied().collect(),
            );
            let capped = script
                .output_token_ids
                .as_ref()
                .is_some_and(|ids| ids.len() > cap);
            population.truncate(cap);
            if script
                .fail_after_tokens
                .is_some_and(|cut| cut > population.len())
            {
                return Err(ExecutionError::Failed(
                    "script failure cut exceeds capped output population".into(),
                ));
            }
            // Validate token vocabulary before any operation exists. A partial
            // UTF-8 prefix is valid model output and is decoded lossily later.
            if script.response_text.is_none() {
                aiperf_tokenizers::TiktokenTokenizer::builtin()
                    .decode_lossy(&population)
                    .map_err(|error| ExecutionError::Failed(error.to_string()))?;
            }
            if script.pre_send_delay_ns > 0 {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => return Ok(None),
                    () = self.sleep_until(self.now().saturating_add(script.pre_send_delay_ns)) => {},
                }
            }
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let start_ns = self.now();
            let input_tokens = request.input_token_ids.len();
            observer.on_admit(uuid, start_ns, 0);
            callback(Observation::TransportStarted { at_ns: start_ns });
            // Admission/start callbacks may synchronously cancel. No await or
            // external callback may intervene between this check and Submit.
            if cancel.is_cancelled() {
                return Ok(None);
            }
            let mailbox = Rc::new(Mailbox::default());
            {
                let mut state = self.state.borrow_mut();
                state
                    .engine
                    .apply_command_effects(
                        SchedulerCommand::new(
                            0,
                            Command::Submit(ModelRequest {
                                request_id: uuid,
                                tokens: request.input_token_ids,
                                max_output_tokens: population.len(),
                                output_token_ids: Some(population),
                                session_id: Some(request.turn.x_correlation_id.clone()),
                                images: request
                                    .images
                                    .into_iter()
                                    .map(|image| aisimulate_core::engine::ImageSpec {
                                        identity: image.identity,
                                        token_start: image.token_start,
                                        token_end: image.token_end,
                                        encoder: aisimulate_core::engine::EncoderShape {
                                            sequences: image.encoder.sequences,
                                            patch_tokens: image.encoder.patch_tokens,
                                            transformer_tokens: image.encoder.transformer_tokens,
                                            output_tokens: image.encoder.output_tokens,
                                        },
                                        embedding_bytes: image.embedding_bytes,
                                    })
                                    .collect(),
                                kv_hint: None,
                            }),
                        ),
                        start_ns as f64 / 1_000_000.0,
                    )
                    .map_err(|error| ExecutionError::Failed(error.to_string()))?;
                // CommandEffects contains ownership/lifecycle deltas, not output;
                // actual outputs arrive exclusively from complete_pass.
                state.requests.insert(id, mailbox.clone());
            }
            let _guard = RequestGuard {
                execution: self,
                id,
            };
            self.events.borrow_mut().push(ModelEvent::Submitted {
                request_id: id,
                at_ns: start_ns,
            });
            let mut tokens = Vec::new();
            let mut last_content_ns = None;
            let mut rejected = false;
            loop {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                if script.fail_after_tokens == Some(tokens.len()) {
                    self.cancel_request(id);
                    break;
                }
                let notified = mailbox.ready.notified();
                let output = mailbox.outputs.borrow_mut().pop_front();
                if let Some(output) = output {
                    rejected = output.rejected;
                    if let Some(token) = output.token_id {
                        let at_ns = self.now();
                        if tokens.is_empty() && !script.is_tool_only {
                            callback(Observation::FirstToken {
                                ttft_ns: at_ns.saturating_sub(start_ns),
                            });
                        }
                        tokens.push(token);
                        if !script.is_tool_only {
                            last_content_ns = Some(at_ns);
                            observer.on_output_tokens(uuid, at_ns, 1);
                        }
                        callback(Observation::OutputTokenIds {
                            at_ns,
                            ids: vec![token],
                        });
                    }
                    self.events.borrow_mut().push(ModelEvent::Observed {
                        request_id: id,
                        at_ns: self.now(),
                        is_final: output.completed || rejected,
                    });
                    if output.completed || rejected {
                        // Final output delivery and immediate response settlement
                        // are one poll, matching native dispatch-first arbitration.
                        // A token callback's cancellation does not rewrite this
                        // completed batch; an awaited terminal tail still can cancel.
                        self.state.borrow_mut().requests.remove(&id);
                        break;
                    }
                } else {
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => return Ok(None),
                        () = notified => {},
                    }
                }
            }
            if script.terminal_tail_ns > 0 {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => return Ok(None),
                    () = self.sleep_until(self.now().saturating_add(script.terminal_tail_ns)) => {},
                }
            }
            let end_ns = self.now();
            let terminal = if rejected {
                ReplayTerminalStatus::Rejected
            } else {
                script.terminal
            };
            let completion_error = if rejected {
                CompletionError::Present
            } else {
                script.completion_error
            };
            let response_text = match script.response_text {
                Some(text) => text,
                None => aiperf_tokenizers::TiktokenTokenizer::builtin()
                    .decode_lossy(&tokens)
                    .map_err(|error| ExecutionError::Failed(error.to_string()))?,
            };
            let mut metadata = script.model_response;
            metadata.output_token_ids = Some(tokens.clone());
            if metadata.content.is_none() {
                metadata.content = Some(response_text.clone());
            }
            if capped
                && terminal == ReplayTerminalStatus::Completed
                && metadata.finish_reason.is_none()
            {
                metadata.finish_reason = Some("length".into());
            }
            observer.on_input_tokens(uuid, Some(input_tokens));
            observer.on_usage(
                uuid,
                aiperf_exchange_model::dispatch::ObservedUsage {
                    prompt_tokens: Some(input_tokens),
                    completion_tokens: Some(tokens.len()),
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
                model_response: metadata,
                chat_capture: script.chat_capture,
                prompt_tokens: Some(input_tokens as u64),
                completion_tokens: Some(tokens.len() as u64),
                http: Default::default(),
            };
            self.state.borrow_mut().pending_reports += 1;
            let mut report = SettledReportGuard {
                execution: self,
                uuid,
                outcome: Some(Outcome {
                    dispatch,
                    output_token_ids: Some(tokens),
                }),
            };
            // Custody is installed before callbacks: even callback unwinding
            // must retain the frozen operation and fail its report obligation.
            observer.on_operation_settled(uuid, terminal, start_ns, end_ns);
            observer.on_response_settled(uuid);
            if let Some(error) = script.report_error {
                self.state.borrow_mut().report_errors.push(error);
            }
            if script.accounting_delay_ns > 0 {
                self.sleep_until(end_ns.saturating_add(script.accounting_delay_ns))
                    .await;
            }
            observer.on_terminal_at(uuid, terminal, end_ns);
            Ok(report.outcome.take())
        })
    }
}

fn source_error(error: impl fmt::Display) -> SimDriveError {
    SimDriveError::EventSource(error.to_string())
}

impl SimEventSource for ModelExecution {
    fn next_event_ns(&self) -> Result<Option<i64>, SimDriveError> {
        let state = self.state.borrow();
        if let Some(error) = &state.failure {
            return Err(source_error(error));
        }
        if state.engine.next_internal_deadline_ms().is_some() {
            return Err(source_error(
                "experimental adapter does not support internal-work deadlines",
            ));
        }
        Ok(state
            .pending
            .map(|(_, _, ns)| ns.max(self.now()))
            .or_else(|| state.engine.is_ready().then(|| self.now())))
    }
    fn set_time_ns(&self, now_ns: i64) -> Result<(), SimDriveError> {
        if now_ns != self.now() {
            return Err(source_error("source and SimClock disagree"));
        }
        Ok(())
    }
    fn step(&self, now_ns: i64) -> Result<SimStep, SimDriveError> {
        let mut state = self.state.borrow_mut();
        if let Some((pass_id, end_ms, end_ns)) = state.pending.take() {
            if now_ns < end_ns {
                state.pending = Some((pass_id, end_ms, end_ns));
                return Err(source_error(
                    "model completion requested before its deadline",
                ));
            }
            let completed = state
                .engine
                .complete_pass(pass_id, end_ms)
                .map_err(source_error)?;
            self.events.borrow_mut().push(ModelEvent::PassCompleted {
                pass_id: pass_id.get(),
                at_ns: now_ns,
            });
            for rank in completed.effects.by_rank {
                for output in rank.effects.outputs {
                    let id = output.request_id.as_u128();
                    if let Some(mailbox) = state.requests.get(&id) {
                        mailbox.outputs.borrow_mut().push_back(output);
                        mailbox.ready.notify_one();
                    }
                }
            }
            // Crucial: return to the existing quiescence pump. Do not execute
            // another pass while these observations await application delivery.
        } else if let Some(started) = state
            .engine
            .execute_pass(now_ns as f64 / 1_000_000.0)
            .map_err(source_error)?
        {
            let end_ns_f = (started.end_ms * 1_000_000.0).ceil();
            if !end_ns_f.is_finite() || end_ns_f < now_ns as f64 || end_ns_f >= i64::MAX as f64 {
                return Err(source_error("invalid modeled pass deadline"));
            }
            let request_ids = started
                .by_rank
                .iter()
                .flat_map(|rank| rank.effects.committed_requests.iter())
                .map(|id| id.as_u128())
                .collect::<Vec<_>>();
            self.events.borrow_mut().push(ModelEvent::PassStarted {
                pass_id: started.pass_id.get(),
                at_ns: now_ns,
                request_ids,
            });
            state.pending = Some((started.pass_id, started.end_ms, end_ns_f as i64));
        } else {
            return Ok(SimStep {
                end_ns: now_ns,
                made_progress: false,
            });
        }
        Ok(SimStep {
            end_ns: now_ns,
            made_progress: true,
        })
    }
    fn is_idle(&self) -> bool {
        self.is_drained()
    }
}
