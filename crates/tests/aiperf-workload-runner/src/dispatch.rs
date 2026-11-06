// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local request adaptation only; all arrival, branching and stop policy lives in AIPerf.

use aiperf_dispatch::turn::TurnToSend;
use aiperf_engine_kernel::scheduled::TurnDispatcher;
use aiperf_exchange_model::{
    TextTokenizer, dispatch::RequestObserver, measure::TurnDispatchOutcome,
};
use aiperf_request_execution::execution::{Execution, Observation, Request};
use async_trait::async_trait;
use std::{cell::RefCell, rc::Rc};
use tokio_util::sync::CancellationToken;

pub struct Dispatcher {
    pub execution: Rc<dyn Execution>,
    pub records: RefCell<Vec<serde_json::Value>>,
}

impl Dispatcher {
    async fn execute(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        first: &dyn Fn(i64),
        started: &dyn Fn(i64),
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<TurnDispatchOutcome>> {
        let wire = turn
            .request_body
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prepared request body absent"))?
            .to_wire()?;
        let body: serde_json::Value = serde_json::from_slice(&wire)?;
        let tokenizer = aiperf_tokenizers::TiktokenTokenizer::builtin();
        let mut ids = Vec::new();
        let messages = body["messages"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("proof supports chat messages only"))?;
        for message in messages {
            match &message["content"] {
                serde_json::Value::String(text) => ids.extend(tokenizer.encode(text)?),
                serde_json::Value::Array(parts) => {
                    for part in parts {
                        anyhow::ensure!(
                            part["type"] == "text",
                            "proof supports text only; no implicit image geometry"
                        );
                        ids.extend(
                            tokenizer.encode(
                                part["text"]
                                    .as_str()
                                    .ok_or_else(|| anyhow::anyhow!("text part lacks text"))?,
                            )?,
                        );
                    }
                }
                serde_json::Value::Null => {}
                _ => anyhow::bail!("unsupported message content"),
            }
        }
        let identity = serde_json::json!({"uuid":turn.uuid,"correlation_id":turn.x_correlation_id,"conversation_id":turn.conversation_id,"parent_correlation_id":turn.parent_correlation_id,"root_correlation_id":turn.root_correlation_id,"body":body});
        let mut callback = |event| match event {
            Observation::TransportStarted { at_ns } => started(at_ns),
            Observation::FirstToken { ttft_ns } => first(ttft_ns),
            Observation::OutputTokenIds { .. } => {}
        };
        let outcome = self
            .execution
            .infer(
                Request {
                    turn,
                    input_token_ids: ids,
                    images: Vec::new(),
                },
                &cancel,
                observer,
                &mut callback,
            )
            .await?;
        if let Some(outcome) = &outcome {
            self.records.borrow_mut().push(serde_json::json!({"request":identity,"start_ns":outcome.dispatch.start_ns,"end_ns":outcome.dispatch.end_ns,"output_token_ids":outcome.output_token_ids}));
        }
        Ok(outcome.map(|outcome| outcome.dispatch))
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
        self.execution.discard_turn(uuid);
    }
    async fn finish_reports(&self) -> anyhow::Result<()> {
        Ok(self.execution.finish_reports().await?)
    }
}
