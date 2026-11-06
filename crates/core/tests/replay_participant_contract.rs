// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! External participant contract checks: a malformed step fails before any
//! admission side effect or callback, an unparked or non-finite deadline is
//! rejected with its own context, and a drained participant never ends the run
//! while the engine still owns physical work.

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{
    DirectRequest, ParticipantStep, ReplayEngineConfig, ReplayEngineFactory, ReplayRuntimeInput,
    ReplaySpec, ReplayTerminalStatus, ReplayTopology, Replayer, WorkloadParticipant,
};
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;
use uuid::Uuid;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scenario {
    MissingUuid,
    UnconsumedDeadline,
    NanAfterAdvance,
    DrainedDuringPhysicalWork,
}

struct ContractProbe {
    scenario: Scenario,
    started: bool,
    terminals: Rc<RefCell<Vec<(u128, f64)>>>,
}

fn request(id: u128) -> DirectRequest {
    DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens: vec![1, 2, 3, 4],
        max_output_tokens: 3,
        output_token_ids: Some(vec![17, 18, 19]),
        ..Default::default()
    }
}

impl WorkloadParticipant for ContractProbe {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        if !self.started {
            return Some(0.0);
        }
        match self.scenario {
            // The fired deadline is deliberately left at the current instant.
            Scenario::UnconsumedDeadline => Some(0.0),
            Scenario::NanAfterAdvance => Some(f64::NAN),
            Scenario::MissingUuid | Scenario::DrainedDuringPhysicalWork => None,
        }
    }

    fn advance_to(&mut self, _now_ms: f64, _cluster_in_flight: usize) -> Result<ParticipantStep> {
        let mut step = ParticipantStep::default();
        if self.started {
            return Ok(step);
        }
        self.started = true;
        match self.scenario {
            Scenario::MissingUuid => {
                // A valid request precedes the malformed one; validation must
                // run over the whole step before anything is admitted.
                step.submits.push(request(1));
                let mut missing = request(2);
                missing.uuid = None;
                step.submits.push(missing);
            }
            Scenario::DrainedDuringPhysicalWork => step.submits.push(request(1)),
            Scenario::UnconsumedDeadline | Scenario::NanAfterAdvance => {}
        }
        Ok(step)
    }

    fn on_output_token(&mut self, _request_id: Uuid, _token_id: u32, _now_ms: f64) -> Result<()> {
        Ok(())
    }

    fn on_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
        _status: ReplayTerminalStatus,
    ) -> Result<()> {
        self.terminals
            .borrow_mut()
            .push((request_id.as_u128(), now_ms));
        Ok(())
    }

    fn is_drained(&self) -> bool {
        // Claims drained as soon as the request is submitted, while the engine
        // still owns its physical pass.
        self.scenario == Scenario::DrainedDuringPhysicalWork && self.started
    }

    fn total_requests(&self) -> usize {
        1
    }
}

fn run(scenario: Scenario) -> (Result<()>, Vec<(u128, f64)>) {
    let timing_ms = if scenario == Scenario::DrainedDuringPhysicalWork {
        10.0
    } else {
        0.0
    };
    let rank = EngineConfig {
        block_size: 4,
        num_gpu_blocks: 128,
        max_num_seqs: 2,
        max_num_batched_tokens: 128,
        enable_prefix_caching: false,
        async_scheduling: Some(false),
        prefill_schedule_interval: 1,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: timing_ms,
            decode_ms: timing_ms,
        },
        ..EngineConfig::for_backend(Backend::Vllm)
    };
    let spec = ReplaySpec {
        version: 1,
        encoder: None,
        topology: ReplayTopology::aggregated(1),
        engine: serde_json::to_value(ReplayEngineConfig {
            rank,
            ..Default::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    let terminals = Rc::new(RefCell::new(Vec::new()));
    let result = Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(ContractProbe {
            scenario,
            started: false,
            terminals: terminals.clone(),
        })))
        .run()
        .map(|_| ());
    (result, terminals.borrow().clone())
}

#[test]
fn missing_uuid_fails_before_any_terminal_callback() {
    let (result, terminals) = run(Scenario::MissingUuid);
    let message = format!(
        "{:#}",
        result.expect_err("a submit without a uuid must fail the run")
    );
    assert!(message.contains("without a uuid"), "{message}");
    assert!(
        terminals.is_empty(),
        "validation must precede admission and callbacks: {terminals:?}"
    );
}

#[test]
fn same_instant_unconsumed_deadline_fails_with_its_own_error() {
    let (result, _) = run(Scenario::UnconsumedDeadline);
    let message = format!(
        "{:#}",
        result.expect_err("a deadline left at the current instant must fail instead of looping")
    );
    assert!(
        message.contains("did not park after advance_to(0ms)"),
        "{message}"
    );
    assert!(message.contains("next wakeup 0ms"), "{message}");
}

#[test]
fn non_finite_wakeup_after_advance_fails_with_context() {
    let (result, _) = run(Scenario::NanAfterAdvance);
    let message = format!(
        "{:#}",
        result.expect_err("a non-finite wakeup after advance_to must fail the run")
    );
    assert!(
        message.contains("did not park after advance_to(0ms)"),
        "{message}"
    );
    assert!(message.contains("next wakeup NaN"), "{message}");
}

#[test]
fn drained_participant_does_not_end_the_run_while_physical_work_remains() {
    let (result, terminals) = run(Scenario::DrainedDuringPhysicalWork);
    result.expect("the host must keep running while the engine owns a submitted request");
    assert!(
        terminals.iter().any(|(id, at_ms)| *id == 1 && *at_ms > 0.0),
        "the terminal must be delivered after the physical pass: {terminals:?}"
    );
}
