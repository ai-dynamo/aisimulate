// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Participant cancellation remains causal across frontend and committed-pass boundaries.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use aisimulate_core::engine::{
    Backend, EncoderShape, EngineConfig, SglangConfig, TimingModel, TimingModelConfig, VisionShape,
};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, DirectRequest, EncoderSpec, ParticipantStep, PerRequestRecord,
    ReplayEngineConfig, ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec,
    ReplayTerminalStatus, ReplayTopology, Replayer, WorkerPoolSpec, WorkloadParticipant,
};
use anyhow::Result;
use uuid::Uuid;

const REQUEST: Uuid = Uuid::from_u128(101);

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Plain,
    Encoder,
    Receipt,
    Decode,
}

#[derive(Clone, Copy, Debug)]
enum Fact {
    Token(f64),
    Cancel(f64),
    Terminal(f64, ReplayTerminalStatus),
    Quiescent(f64),
}

struct DeadlineParticipant {
    output_tokens: usize,
    submitted: bool,
    cancel_at_ms: Option<f64>,
    cancel_issued: bool,
    terminal_seen: bool,
    quiescent_seen: bool,
    facts: Rc<RefCell<Vec<Fact>>>,
}

impl WorkloadParticipant for DeadlineParticipant {
    fn next_wakeup_ms(&mut self) -> Option<f64> {
        if !self.submitted {
            Some(0.0)
        } else if !self.cancel_issued {
            self.cancel_at_ms
        } else {
            None
        }
    }

    fn advance_to(&mut self, now_ms: f64, _cluster_in_flight: usize) -> Result<ParticipantStep> {
        let mut step = ParticipantStep::default();
        if !self.submitted {
            self.submitted = true;
            step.submits.push(DirectRequest {
                uuid: Some(REQUEST),
                tokens: vec![1, 2, 3, 4],
                max_output_tokens: self.output_tokens,
                output_token_ids: Some(vec![17; self.output_tokens]),
                ..Default::default()
            });
        } else if !self.cancel_issued && self.cancel_at_ms.is_some_and(|at| at <= now_ms) {
            self.cancel_issued = true;
            self.facts.borrow_mut().push(Fact::Cancel(now_ms));
            step.cancels.push(REQUEST);
        }
        Ok(step)
    }

    fn on_output_token(&mut self, id: Uuid, _token_id: u32, now_ms: f64) -> Result<()> {
        assert_eq!(id, REQUEST);
        self.facts.borrow_mut().push(Fact::Token(now_ms));
        Ok(())
    }

    fn on_terminal(&mut self, id: Uuid, now_ms: f64, status: ReplayTerminalStatus) -> Result<()> {
        assert_eq!(id, REQUEST);
        self.terminal_seen = true;
        self.facts.borrow_mut().push(Fact::Terminal(now_ms, status));
        Ok(())
    }

    fn on_quiescent(&mut self, id: Uuid, now_ms: f64) -> Result<()> {
        assert_eq!(id, REQUEST);
        self.quiescent_seen = true;
        self.facts.borrow_mut().push(Fact::Quiescent(now_ms));
        Ok(())
    }

    fn is_drained(&self) -> bool {
        self.terminal_seen && self.quiescent_seen
    }

    fn total_requests(&self) -> usize {
        1
    }
}

/// Synthetic latency-only provider: one image takes 10ms of encoder GPU work.
struct EncoderTiming;

impl TimingModel for EncoderTiming {
    fn predict_vision_ms(&self, shapes: &[VisionShape]) -> Result<Option<f64>> {
        Ok(Some(
            shapes
                .iter()
                .map(|shape| f64::from(shape.count) * 10.0)
                .sum(),
        ))
    }

    fn predict_prefill_ms(
        &self,
        _batch_size: usize,
        _mean_isl: usize,
        _mean_prefix: usize,
    ) -> Result<f64> {
        Ok(10.0)
    }

    fn predict_decode_ms(
        &self,
        _batch_size: usize,
        _active_kv_tokens: usize,
        _mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        Ok(10.0)
    }
}

fn run(disagg: bool, scenario: Scenario, cancel_at_ms: Option<f64>) -> (ReplayReport, Vec<Fact>) {
    let encoding = matches!(scenario, Scenario::Encoder);
    let backend = if encoding {
        Backend::Sglang
    } else {
        Backend::Vllm
    };
    let rank = EngineConfig {
        block_size: 4,
        num_gpu_blocks: 128,
        max_num_seqs: 2,
        max_num_batched_tokens: 128,
        enable_prefix_caching: false,
        async_scheduling: (!encoding).then_some(false),
        sglang: SglangConfig {
            num_reserved_decode_tokens: 0,
            ..Default::default()
        },
        timing_model: if encoding {
            TimingModelConfig::External {
                provider: "aic".into(),
                config: serde_json::json!({"model": "test"}),
            }
        } else {
            TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 10.0,
            }
        },
        ..EngineConfig::for_backend(backend)
    };
    let spec = ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        encoder: encoding.then_some(EncoderSpec {
            instances: 1,
            max_batch: 1,
            gpus_per_instance: 1,
            images_per_request: 1,
            shape: EncoderShape {
                sequences: 1,
                patch_tokens: 4,
                transformer_tokens: 4,
                output_tokens: 1,
            },
            preprocess_ms_per_image: 1.0,
            transfer_bytes_per_image: 1_000_000,
            transfer_bandwidth_gb_s: 1.0,
            timing_model: None,
        }),
        topology: if disagg {
            ReplayTopology::Disaggregated {
                prefill: WorkerPoolSpec::default(),
                decode: WorkerPoolSpec::default(),
                handoff_latency_ms: 0.0,
            }
        } else {
            ReplayTopology::aggregated(1)
        },
        engine: serde_json::to_value(ReplayEngineConfig {
            rank,
            ..Default::default()
        })
        .unwrap(),
        adapters: Default::default(),
        max_sim_time_ms: Some(1000.0),
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    };
    let factory = if encoding {
        let timing = Arc::new(EncoderTiming);
        ReplayEngineFactory::with_timing_model(timing.clone()).with_encoder_timing(timing)
    } else {
        ReplayEngineFactory::new()
    };
    let facts = Rc::new(RefCell::new(Vec::new()));
    let mut replay =
        Replayer::new(spec, factory)
            .unwrap()
            .with_runtime_input(ReplayRuntimeInput::External(Box::new(
                DeadlineParticipant {
                    output_tokens: if matches!(scenario, Scenario::Decode) {
                        4
                    } else {
                        1
                    },
                    submitted: false,
                    cancel_at_ms,
                    cancel_issued: false,
                    terminal_seen: false,
                    quiescent_seen: false,
                    facts: facts.clone(),
                },
            )));
    if matches!(scenario, Scenario::Receipt) {
        replay = replay.with_detokenization(1, 50.0, 0.0).unwrap();
    }
    let report = replay.run().unwrap();
    let facts = facts.borrow().clone();
    (report, facts)
}

fn record(report: &ReplayReport) -> &PerRequestRecord {
    let uuid = REQUEST.to_string();
    report
        .per_request
        .iter()
        .find(|record| record.uuid == uuid)
        .unwrap()
}

fn assert_canceled_once(report: &ReplayReport, facts: &[Fact], at_ms: f64) -> f64 {
    let terminals = facts
        .iter()
        .filter_map(|fact| match fact {
            Fact::Terminal(at, status) => Some((*at, *status)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        terminals,
        vec![(at_ms, ReplayTerminalStatus::Canceled)],
        "{facts:?}"
    );
    let cancels = facts
        .iter()
        .filter_map(|fact| match fact {
            Fact::Cancel(at) => Some(*at),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(cancels, vec![at_ms], "{facts:?}");
    let quiescent = facts
        .iter()
        .filter_map(|fact| match fact {
            Fact::Quiescent(at) => Some(*at),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(quiescent.len(), 1, "{facts:?}");
    assert!(quiescent[0] >= at_ms, "{facts:?}");
    let terminal_index = facts
        .iter()
        .position(|fact| matches!(fact, Fact::Terminal(..)))
        .unwrap();
    let quiescent_index = facts
        .iter()
        .position(|fact| matches!(fact, Fact::Quiescent(_)))
        .unwrap();
    assert!(terminal_index < quiescent_index, "{facts:?}");
    assert!(
        facts[terminal_index..]
            .iter()
            .all(|fact| !matches!(fact, Fact::Token(_))),
        "{facts:?}"
    );
    assert_eq!(
        record(report).terminal_status,
        ReplayTerminalStatus::Canceled
    );
    assert_eq!(record(report).terminal_time_ms, at_ms);
    quiescent[0]
}

fn encoder_cancellation(disagg: bool) {
    let (control, _) = run(disagg, Scenario::Encoder, None);
    // 1ms preprocessing + 10ms encoder forward + 1ms embedding transfer.
    assert!(record(&control).first_admit_ms.unwrap() >= 12.0);
    let (report, facts) = run(disagg, Scenario::Encoder, Some(4.0));
    assert_canceled_once(&report, &facts, 4.0);
    assert_eq!(record(&report).first_admit_ms, None, "{facts:?}");
    assert_eq!(record(&report).output_length, 0, "{facts:?}");
    assert!(
        facts.iter().all(|fact| !matches!(fact, Fact::Token(_))),
        "{facts:?}"
    );
}

#[test]
fn aggregated_participant_cancels_request_owned_by_encoder() {
    encoder_cancellation(false);
}

#[test]
fn disaggregated_participant_cancels_request_owned_by_encoder() {
    encoder_cancellation(true);
}

fn receipt_cancellation(disagg: bool) {
    let (engine_control, _) = run(disagg, Scenario::Plain, None);
    let engine_end_ms = record(&engine_control).terminal_time_ms;
    let (receipt_control, _) = run(disagg, Scenario::Receipt, None);
    assert_eq!(
        record(&receipt_control).terminal_time_ms,
        engine_end_ms + 50.0
    );
    // The engine has finished, but the client has not received its terminal.
    let cancel_at_ms = engine_end_ms + 1.0;
    assert!(cancel_at_ms < record(&receipt_control).terminal_time_ms);
    let (report, facts) = run(disagg, Scenario::Receipt, Some(cancel_at_ms));
    assert!(record(&report).first_admit_ms.unwrap() < cancel_at_ms);
    assert_canceled_once(&report, &facts, cancel_at_ms);
}

#[test]
fn aggregated_participant_cancels_response_awaiting_detokenization() {
    receipt_cancellation(false);
}

#[test]
fn disaggregated_participant_cancels_response_awaiting_detokenization() {
    receipt_cancellation(true);
}

#[test]
fn disaggregated_decode_cancel_waits_for_committed_pass_before_quiescence() {
    let (_, control) = run(true, Scenario::Decode, None);
    let tokens = control
        .iter()
        .filter_map(|fact| match fact {
            Fact::Token(at) => Some(*at),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tokens.len(), 4, "{control:?}");
    let (first_token_ms, next_pass_end_ms) = (tokens[0], tokens[1]);
    assert_eq!(next_pass_end_ms, first_token_ms + 10.0);
    // By the first decode output the zero-latency handoff has completed. The
    // next 10ms decode pass commits at that instant; cancel 4ms into it.
    let cancel_at_ms = first_token_ms + 4.0;
    assert!(cancel_at_ms < next_pass_end_ms);
    let (report, facts) = run(true, Scenario::Decode, Some(cancel_at_ms));
    let delivered_tokens = facts
        .iter()
        .filter_map(|fact| match fact {
            Fact::Token(at) => Some(*at),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(delivered_tokens, vec![first_token_ms], "{facts:?}");
    let quiescent_ms = assert_canceled_once(&report, &facts, cancel_at_ms);
    assert_eq!(
        quiescent_ms, next_pass_end_ms,
        "the causal terminal must not release a committed decode pass early: {facts:?}"
    );
}
