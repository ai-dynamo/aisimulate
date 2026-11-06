// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Reactive workload facts settle before the next native engine decision.

use aisimulate_core::engine::{Backend, EngineConfig, TimingModelConfig};
use aisimulate_core::replay::{DirectRequest, ReplayEngineConfig, ReplayEngineFactory};
use aisimulate_core::replay::{
    ParticipantStep, ReplayRuntimeInput, ReplaySpec, ReplayTerminalStatus, ReplayTopology,
    Replayer, WorkloadParticipant,
};
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;
use uuid::Uuid;

#[derive(Clone, Debug)]
enum Fact {
    Token(u128, f64),
    Terminal(u128, f64, ReplayTerminalStatus),
    Submit(u128, f64),
    Cancel(f64),
}
struct Probe {
    successor: bool,
    eager_successor: bool,
    initialized: bool,
    react_at: Option<f64>,
    successor_at: Option<f64>,
    canceled: bool,
    done: bool,
    log: Rc<RefCell<Vec<Fact>>>,
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
impl WorkloadParticipant for Probe {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        if !self.initialized {
            Some(0.0)
        } else {
            self.react_at.or(self.successor_at)
        }
    }
    fn advance_to(&mut self, now: f64, _: usize) -> Result<ParticipantStep> {
        let mut step = ParticipantStep::default();
        if !self.initialized {
            self.initialized = true;
            step.submits.push(request(1));
            if self.successor {
                step.submits.push(request(2));
            }
        } else if self.react_at.is_some_and(|at| at <= now) {
            self.react_at = None;
            self.canceled = true;
            self.log.borrow_mut().push(Fact::Cancel(now));
            step.cancels.push(Uuid::from_u128(1));
            if self.eager_successor {
                self.log.borrow_mut().push(Fact::Submit(3, now));
                step.submits.push(request(3));
            }
        } else if self.successor_at.is_some_and(|at| at <= now) {
            self.successor_at = None;
            self.log.borrow_mut().push(Fact::Submit(3, now));
            step.submits.push(request(3));
        }
        Ok(step)
    }
    fn on_output_token(&mut self, id: Uuid, _: u32, now: f64) -> Result<()> {
        self.log.borrow_mut().push(Fact::Token(id.as_u128(), now));
        if id.as_u128() == 1 && !self.canceled {
            self.react_at = Some(now);
        }
        Ok(())
    }
    fn on_terminal(&mut self, id: Uuid, now: f64, status: ReplayTerminalStatus) -> Result<()> {
        self.log
            .borrow_mut()
            .push(Fact::Terminal(id.as_u128(), now, status));
        if self.successor && !self.eager_successor && id.as_u128() == 1 {
            self.successor_at = Some(now);
        }
        if (!self.successor && id.as_u128() == 1) || (self.successor && id.as_u128() == 3) {
            self.done = true;
        }
        Ok(())
    }
    fn is_drained(&self) -> bool {
        self.done
    }
    fn total_requests(&self) -> usize {
        if self.successor { 3 } else { 1 }
    }
}
fn run(successor: bool, eager_successor: bool) -> Vec<Fact> {
    let ms = if successor { 10.0 } else { 0.0 };
    let rank = EngineConfig {
        block_size: 4,
        num_gpu_blocks: 128,
        max_num_seqs: 2,
        max_num_batched_tokens: 128,
        enable_prefix_caching: false,
        async_scheduling: Some(false),
        prefill_schedule_interval: 1,
        timing_model: TimingModelConfig::Fixed {
            prefill_ms: ms,
            decode_ms: ms,
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
    let log = Rc::new(RefCell::new(Vec::new()));
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::External(Box::new(Probe {
            successor,
            eager_successor,
            initialized: false,
            react_at: None,
            successor_at: None,
            canceled: false,
            done: false,
            log: log.clone(),
        })))
        .run()
        .unwrap();
    let result = log.borrow().clone();
    println!("successor={successor} eager_successor={eager_successor} {result:?}");
    result
}
#[test]
fn zero_duration_first_token_reaction_precedes_next_pass() {
    let log = run(false, false);
    let tokens = log
        .iter()
        .filter(|f| matches!(f, Fact::Token(1, _)))
        .count();
    assert_eq!(
        tokens, 1,
        "must poll first-token cancellation before another zero-duration pass"
    );
    assert!(
        log.iter()
            .any(|f| matches!(f, Fact::Terminal(1, 0.0, ReplayTerminalStatus::Canceled)))
    );
}
#[test]
fn cancel_terminal_successor_joins_next_batch() {
    let control = run(true, true);
    let log = run(true, false);
    let control_c_token = control
        .iter()
        .find_map(|f| {
            if let Fact::Token(3, at) = f {
                Some(*at)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(
        control_c_token, 30.0,
        "control mixed prefill+decode batch starts at 10ms and costs 20ms"
    );
    assert!(log.iter().any(|f| matches!(f, Fact::Cancel(10.0))));
    assert!(
        log.iter()
            .any(|f| matches!(f, Fact::Terminal(1, 10.0, ReplayTerminalStatus::Canceled)))
    );
    assert!(log.iter().any(|f| matches!(f, Fact::Submit(3, 10.0))));
    let first_c_token = log
        .iter()
        .find_map(|f| {
            if let Fact::Token(3, at) = f {
                Some(*at)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(
        first_c_token, control_c_token,
        "terminal-triggered C must join B's next batch just like same-step C"
    );
}
