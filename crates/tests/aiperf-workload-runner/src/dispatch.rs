// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local request adaptation only; all arrival, branching and stop policy lives in AIPerf.

use aiperf_dispatch::turn::TurnToSend;
use aiperf_engine_kernel::scheduled::TurnDispatcher;
use aiperf_exchange_model::{dispatch::RequestObserver, measure::TurnDispatchOutcome};
use aiperf_request_execution::execution::{Execution, Observation, Request};
use aiperf_request_execution::model_input::OpenAiRequestEncoder;
use async_trait::async_trait;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    Complete,
    Compact,
}

pub struct Dispatcher {
    pub execution: Rc<dyn Execution>,
    pub records: RefCell<Vec<serde_json::Value>>,
    capture: Cell<CaptureMode>,
}

// A cancelled phase may drop dispatch rather than poll infer to Ok(None).
// Retain that issued row and every observation delivered before the drop.
struct CaptureGuard<'a> {
    dispatcher: &'a Dispatcher,
    index: usize,
    returned: bool,
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if !self.returned {
            let mut records = self.dispatcher.records.borrow_mut();
            records[self.index]["status"] = serde_json::json!("future_dropped");
            records[self.index]["drop_ns"] = serde_json::json!(self.dispatcher.execution.now());
        }
    }
}

impl Dispatcher {
    pub(crate) fn new(execution: Rc<dyn Execution>) -> Self {
        Self {
            execution,
            records: RefCell::new(Vec::new()),
            capture: Cell::new(CaptureMode::Complete),
        }
    }

    /// Retain identities, scalar outcomes and cryptographic input witnesses,
    /// not repeated context bodies or per-token JSON. Observer delivery is unchanged.
    pub(crate) fn set_compact_capture(&self) {
        self.capture.set(CaptureMode::Compact);
    }

    async fn execute(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
        started: &dyn Fn(i64),
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<TurnDispatchOutcome>> {
        let compact = self.capture.get() == CaptureMode::Compact;
        let index = self.records.borrow().len();
        let identity = serde_json::json!({
            "uuid": turn.uuid,
            "correlation_id": turn.x_correlation_id,
            "request_correlation_id": turn.request_correlation_id,
            "conversation_id": turn.conversation_id,
            "parent_correlation_id": turn.parent_correlation_id,
            "root_correlation_id": turn.root_correlation_id,
            "turn_index": turn.turn_index,
            "num_turns": turn.num_turns,
            "agent_depth": turn.agent_depth,
            "is_spawn": turn.is_spawn,
            "source_kind": turn.source_kind,
            "replay_source": turn.replay_source,
            "credit_phase": turn.credit_phase,
            "timestamp_ms": turn.timestamp_ms,
            "delay_ms": turn.delay_ms,
            "input_length": turn.input_length,
            "input_tokens": turn.input_tokens,
            "max_output_tokens": turn.max_output_tokens,
            "max_output_tokens_authored": turn.max_output_tokens_authored,
            "effective_model": turn.effective_model,
            "cache_bust_marker": turn.cache_bust_marker,
            "cache_bust_target": turn.cache_bust_target,
            "target": turn.endpoint_path,
        });
        self.records.borrow_mut().push(serde_json::json!({
            "request": identity,
            "issue_order": index,
            "issue_ns": self.execution.now(),
            "status": "issued",
            "start_ns": null,
            "first_content_ns": null,
            "first_ttft_ns": null,
            "end_ns": null,
            "return_ns": null,
            "drop_ns": null,
            "output_token_ids": null,
            "observations": [],
        }));
        let mut capture = CaptureGuard {
            dispatcher: self,
            index,
            returned: false,
        };
        let wire = turn
            .request_body
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prepared request body absent"))?
            .to_wire()?;
        let body: serde_json::Value = serde_json::from_slice(&wire)?;
        let model = turn
            .effective_model
            .as_deref()
            .or_else(|| body["model"].as_str())
            .ok_or_else(|| anyhow::anyhow!("prepared request model absent"))?
            .to_owned();
        let ids = OpenAiRequestEncoder::default().encode_turn(
            &turn,
            &model,
            |hash_ids, block_size, input_length| {
                let hash_ids = hash_ids
                    .iter()
                    .copied()
                    .map(|hash_id| {
                        u32::try_from(hash_id).map_err(|_| {
                            anyhow::anyhow!("trace hash_id {hash_id} cannot be represented as u32")
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                aisimulate_core::replay::loadgen::TurnTrace {
                    input_length,
                    hash_ids,
                    ..Default::default()
                }
                .synthesize_tokens(block_size)
            },
        )?;
        {
            let mut records = self.records.borrow_mut();
            let row = &mut records[index];
            if compact {
                let mut tokens = blake3::Hasher::new();
                for id in &ids {
                    tokens.update(&id.to_le_bytes());
                }
                row["capture"] = serde_json::json!({
                    "mode": "compact",
                    "omitted": ["request_body", "wire_bytes", "input_token_ids", "output_token_ids", "per_token_observations", "response_content", "http_debug", "chat_capture_debug"],
                    "wire_bytes": wire.len(),
                    "wire_blake3": blake3::hash(&wire).to_hex().to_string(),
                    "input_token_count": ids.len(),
                    "input_token_ids_le_u32_blake3": tokens.finalize().to_hex().to_string(),
                });
            } else {
                row["request"]["body"] = body;
                // Exact bytes are authoritative; UTF-8 is a convenience view only.
                // FNV is explicitly labelled, not passed off as a cryptographic hash.
                let hash = wire.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
                });
                row["wire"] = serde_json::json!({
                    "body_bytes": wire.as_ref(),
                    "body_utf8": std::str::from_utf8(&wire).ok(),
                    "fnv1a64": format!("{hash:016x}"),
                });
                row["input_token_ids"] = serde_json::json!(ids);
            }
            row["status"] = serde_json::json!("infer_pending");
        }
        let mut callback = |event| {
            let observed_ns = self.execution.now();
            {
                let mut records = self.records.borrow_mut();
                let row = &mut records[index];
                let retain_observation =
                    !compact || !matches!(&event, Observation::OutputTokenIds { .. });
                let observation = match &event {
                    Observation::TransportStarted { at_ns } => {
                        row["start_ns"] = serde_json::json!(at_ns);
                        serde_json::json!({"event":"transport_started","at_ns":at_ns,"callback_ns":observed_ns})
                    }
                    Observation::FirstToken { ttft_ns } => {
                        row["first_content_ns"] = serde_json::json!(observed_ns);
                        row["first_ttft_ns"] = serde_json::json!(ttft_ns);
                        serde_json::json!({"event":"first_token","ttft_ns":ttft_ns,"callback_ns":observed_ns})
                    }
                    Observation::OutputTokenIds { at_ns, ids } => {
                        if compact {
                            serde_json::Value::Null
                        } else {
                            serde_json::json!({"event":"output_token_ids","at_ns":at_ns,"ids":ids,"callback_ns":observed_ns})
                        }
                    }
                };
                if retain_observation && let Some(observations) = row["observations"].as_array_mut()
                {
                    observations.push(observation);
                }
            }
            // Invoke the same callbacks synchronously in their original order.
            match event {
                Observation::TransportStarted { at_ns } => started(at_ns),
                Observation::FirstToken { ttft_ns } => first(ttft_ns),
                Observation::OutputTokenIds { .. } => {}
            }
        };
        // The native request owns its prepared body. Diagnostic wire bytes are
        // no longer needed across the response future, even in complete mode.
        drop(wire);
        let outcome = self
            .execution
            .infer(
                Request::from_scheduled_request(turn, &model, ids, Vec::new()),
                &cancel,
                observer,
                &mut callback,
            )
            .await;
        {
            let mut records = self.records.borrow_mut();
            let row = &mut records[index];
            row["return_ns"] = serde_json::json!(self.execution.now());
            match &outcome {
                Ok(Some(outcome)) => {
                    let dispatch = &outcome.dispatch;
                    let metadata = &dispatch.model_response;
                    row["status"] = serde_json::json!("settled");
                    row["start_ns"] = serde_json::json!(dispatch.start_ns);
                    row["end_ns"] = serde_json::json!(dispatch.end_ns);
                    row["last_content_ns"] = serde_json::json!(dispatch.last_content_ns);
                    if compact {
                        row["output_token_count"] =
                            serde_json::json!(outcome.output_token_ids.as_ref().map(Vec::len));
                    } else {
                        row["output_token_ids"] = serde_json::json!(outcome.output_token_ids);
                    }
                    row["terminal"] = serde_json::json!(dispatch.terminal);
                    row["completion_error"] = serde_json::json!(match dispatch.completion_error {
                        aiperf_exchange_model::measure::observe::CompletionError::Unknown =>
                            "unknown",
                        aiperf_exchange_model::measure::observe::CompletionError::Absent =>
                            "absent",
                        aiperf_exchange_model::measure::observe::CompletionError::Present =>
                            "present",
                    });
                    if !compact {
                        row["response_text"] = serde_json::json!(dispatch.response_text);
                    }
                    row["prompt_tokens"] = serde_json::json!(dispatch.prompt_tokens);
                    row["completion_tokens"] = serde_json::json!(dispatch.completion_tokens);
                    if compact {
                        row["model_response"] = serde_json::json!({
                            "cached_prompt_tokens": metadata.cached_prompt_tokens,
                            "error_kind": metadata.error_kind,
                            "context_overflow": metadata.context_overflow,
                            "content_omitted": true,
                        });
                    } else {
                        row["model_response"] = serde_json::json!({
                            "content": metadata.content,
                            "client_output_tokens": metadata.client_output_tokens,
                            "reasoning": metadata.reasoning,
                            "cached_prompt_tokens": metadata.cached_prompt_tokens,
                            "response_id": metadata.response_id,
                            "finish_reason": metadata.finish_reason,
                            "output_token_ids": metadata.output_token_ids,
                            "prompt_token_ids": metadata.prompt_token_ids,
                            "assistant_message": metadata.assistant_message,
                            "error_kind": metadata.error_kind,
                            "error_message": metadata.error_message,
                            "context_overflow": metadata.context_overflow,
                        });
                        // These native types intentionally have no wire serializer.
                        // Retain the complete Debug witness rather than omit fields
                        // or invent absent HTTP/capture measurements.
                        row["http_debug"] = serde_json::json!(format!("{:?}", dispatch.http));
                        row["chat_capture_debug"] = serde_json::json!(
                            dispatch
                                .chat_capture
                                .as_ref()
                                .map(|capture| format!("{capture:?}"))
                        );
                    }
                }
                Ok(None) => row["status"] = serde_json::json!("cancelled_none"),
                Err(error) => {
                    row["status"] = serde_json::json!("execution_error");
                    row["execution_error"] = serde_json::json!(error.to_string());
                }
            }
        }
        capture.returned = true;
        Ok(outcome?.map(|outcome| outcome.dispatch))
    }
}

#[async_trait(?Send)]
impl TurnDispatcher for Dispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        self.dispatch_turn_with_start(turn, observer, first, &|_| {})
            .await
    }
    async fn dispatch_turn_with_start(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
        started: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        self.execute(turn, observer, first, started, CancellationToken::new())
            .await?
            .ok_or_else(|| anyhow::anyhow!("uncancelled execution returned no outcome"))
    }
    async fn dispatch_turn_cancellable(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<TurnDispatchOutcome>> {
        self.execute(turn, observer, first, &|_| {}, cancel).await
    }
    async fn dispatch_turn_cancellable_with_start(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
        started: &dyn Fn(i64),
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<TurnDispatchOutcome>> {
        self.execute(turn, observer, first, started, cancel).await
    }
    fn supports_transport_start(&self) -> bool {
        self.execution.supports_transport_start()
    }
    fn close_session(&self, session: &str) {
        self.execution.close_session(session);
    }
    fn discard_turn(&self, uuid: uuid::Uuid) {
        self.execution.discard_request(uuid);
    }
    async fn finish_reports(&self) -> anyhow::Result<()> {
        Ok(self.execution.finish_reports().await?)
    }
}
