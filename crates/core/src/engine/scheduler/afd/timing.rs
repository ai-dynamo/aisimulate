// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD replay timing. Prefill uses actual geometry; only fixed synthetic
//! decode caches a stride-averaged pass. Timing and evidence share each query.

use std::sync::Mutex;

use anyhow::Result;

use crate::engine::scheduler::afd::pipeline::{
    AfdEvalPhase, AfdPipelineKnobs, evaluate_afd_metrics,
};
#[cfg(feature = "python")]
use crate::engine::timing::{
    TimingEvidenceAccumulator, TimingEvidenceSource, TimingOperationEvidence, ValidatedTimingPhase,
};
use crate::engine::timing::{TimingEvidenceSummary, TimingModel};
#[cfg(feature = "python")]
use crate::perfmodel::afd::model::AfdOpEvidence;
use crate::perfmodel::afd::model::{AfdForwardPassPerfModel, AfdPhase};

/// Replay timing provider for AFD ranks.
pub struct AfdTimingModel {
    perf: AfdForwardPassPerfModel,
    knobs: AfdPipelineKnobs,
    isl: u32,
    osl: u32,
    /// Full-batch sampled decode identity (fixed synthetic workloads); trace and
    /// agentic workloads use the instantaneous per-pass evaluation.
    frozen_decode_identity: bool,
    decode: Mutex<Option<FrozenDecode>>,
    /// Power evidence accumulator: per-op energy from the two
    /// pool engines flows into the standard replay power statistics.
    #[cfg(feature = "python")]
    evidence: Mutex<TimingEvidenceAccumulator>,
}

struct FrozenDecode {
    step_ms: f64,
    #[cfg(feature = "python")]
    evidence: ValidatedTimingPhase,
}

impl AfdTimingModel {
    /// The decode identity chooses fixed KV sampling or the current KV point.
    pub fn new(perf: AfdForwardPassPerfModel, isl: u32, osl: u32, frozen: bool) -> Self {
        let config = perf.config();
        let a_replica_count = config.topology.a_pool.n_a_nodes * config.topology.gpus_per_node
            / config.topology.a_pool.a_tp_size;
        let f_rank_count =
            config.topology.f_pool.n_f_nodes * config.topology.effective_f_gpus_per_node();
        Self {
            knobs: AfdPipelineKnobs {
                pipeline_layout: config.topology.pipeline_layout,
                total_batch_size: a_replica_count * config.topology.a_pool.batch_size_per_a_replica,
                total_gpus: a_replica_count * config.topology.a_pool.a_tp_size + f_rank_count,
                latency_correction: 1.0,
            },
            perf,
            isl,
            osl,
            frozen_decode_identity: frozen,
            decode: Mutex::new(None),
            #[cfg(feature = "python")]
            evidence: Mutex::new(TimingEvidenceAccumulator::default()),
        }
    }

    #[cfg(feature = "python")]
    fn operation_evidence(&self, measured: Vec<AfdOpEvidence>) -> Result<ValidatedTimingPhase> {
        let mut operations = Vec::with_capacity(measured.len());
        for op in measured {
            operations.push(TimingOperationEvidence::new(
                op.name,
                op.latency_ms,
                (op.energy_wms > 0.0).then_some(op.energy_wms),
                TimingEvidenceSource::from_native(op.source.as_str()),
            )?);
        }
        ValidatedTimingPhase::from_operations(operations)
    }
}

fn checked_timing_u32(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("AFD timing {field} exceeds u32 range"))
}

impl TimingModel for AfdTimingModel {
    fn predict_prefill_ms(
        &self,
        batch_size: usize,
        mean_isl: usize,
        mean_prefix: usize,
    ) -> Result<f64> {
        let batch_size = checked_timing_u32(batch_size, "batch_size")?;
        let mean_isl = checked_timing_u32(mean_isl, "mean_isl")?;
        let mean_prefix = checked_timing_u32(mean_prefix, "mean_prefix")?;
        #[cfg(feature = "python")]
        let (metrics, operations) =
            self.perf
                .predict_prefill_step_with_operations(batch_size, mean_isl, mean_prefix)?;
        #[cfg(not(feature = "python"))]
        let metrics = self
            .perf
            .predict_prefill_step_ms(batch_size, mean_isl, mean_prefix)?;
        let evaluation = evaluate_afd_metrics(
            &AfdPipelineKnobs {
                total_batch_size: batch_size,
                ..self.knobs
            },
            &metrics,
            AfdEvalPhase::Prefill,
            mean_isl,
            self.osl,
        )?;
        #[cfg(feature = "python")]
        self.evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD evidence accumulator poisoned"))?
            .record(&self.operation_evidence(operations)?, true)?;
        Ok(evaluation.step_latency_ms)
    }

    fn predict_decode_ms(
        &self,
        batch_size: usize,
        _active_kv_tokens: usize,
        mean_context_length: usize,
        _total_kv_tokens: usize,
    ) -> Result<f64> {
        if !self.frozen_decode_identity {
            let batch_size = checked_timing_u32(batch_size, "batch_size")?;
            let mean_context_length =
                checked_timing_u32(mean_context_length.max(1), "mean_context_length")?;
            // Trace/agentic: instantaneous pipeline evaluation at the current
            // mean KV position (per-pass semantics; the frozen stride average
            // only makes sense for the fixed synthetic identity). The
            // configured microbatch count is kept unless an A replica has
            // fewer sequences than microbatches. The frozen identity keeps
            // charging the full configured batch and pipeline depth.
            #[cfg(feature = "python")]
            let (metrics, operations) = self.perf.predict_decode_step_with_operations(
                batch_size,
                mean_context_length,
                0,
            )?;
            #[cfg(not(feature = "python"))]
            let metrics = self
                .perf
                .predict_decode_step_ms(batch_size, mean_context_length, 0)?;
            let knobs = AfdPipelineKnobs {
                total_batch_size: batch_size,
                ..self.knobs
            };
            let evaluation =
                evaluate_afd_metrics(&knobs, &metrics, AfdEvalPhase::Decode, self.isl, self.osl)?;
            #[cfg(feature = "python")]
            {
                let phase = self.operation_evidence(operations)?;
                self.evidence
                    .lock()
                    .map_err(|_| anyhow::anyhow!("AFD timing evidence accumulator was poisoned"))?
                    .record(&phase, false)?;
            }
            return Ok(evaluation.step_latency_ms);
        }
        let mut cache = self
            .decode
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD decode cache poisoned"))?;
        if cache.is_none() {
            #[cfg(feature = "python")]
            let (metrics, operations) =
                self.perf
                    .predict_phase_with_operations(AfdPhase::Decode, self.isl, self.osl, 0)?;
            #[cfg(not(feature = "python"))]
            let metrics =
                self.perf
                    .predict_phase_metrics(AfdPhase::Decode, self.isl, self.osl, 0)?;
            let evaluation = evaluate_afd_metrics(
                &self.knobs,
                &metrics,
                AfdEvalPhase::Decode,
                self.isl,
                self.osl,
            )?;
            *cache = Some(FrozenDecode {
                step_ms: evaluation.step_latency_ms,
                #[cfg(feature = "python")]
                evidence: self.operation_evidence(operations)?,
            });
        }
        let cached = cache.as_ref().expect("prepared decode");
        #[cfg(feature = "python")]
        self.evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD evidence accumulator poisoned"))?
            .record(&cached.evidence, false)?;
        Ok(cached.step_ms)
    }

    fn evidence_summary(&self) -> Option<TimingEvidenceSummary> {
        #[cfg(feature = "python")]
        {
            return self
                .evidence
                .lock()
                .ok()
                .map(|evidence| evidence.snapshot());
        }
        #[cfg(not(feature = "python"))]
        None
    }

    #[cfg(feature = "python")]
    fn reset_evidence(&self) -> Result<()> {
        *self
            .evidence
            .lock()
            .map_err(|_| anyhow::anyhow!("AFD timing evidence accumulator was poisoned"))? =
            TimingEvidenceAccumulator::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::PerfDatabase;
    use crate::perfmodel::afd::model::{
        AfdAPoolSpec, AfdEstimatorPlumbing, AfdFPoolSpec, AfdForwardPassModelConfig,
        AfdForwardPassPerfModel, AfdQuantOverrides,
    };
    use crate::perfmodel::engine::Engine;
    use crate::perfmodel::engine::spec::EngineSpec;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn timing_conversion_checks_the_u32_boundary() {
        assert_eq!(
            checked_timing_u32(u32::MAX as usize, "test").unwrap(),
            u32::MAX
        );
        if usize::BITS > 32 {
            assert!(
                checked_timing_u32(u32::MAX as usize + 1, "test")
                    .unwrap_err()
                    .to_string()
                    .contains("test")
            );
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn timing_overflow_does_not_mutate_evidence_or_decode_cache() {
        let timing = provider(false);
        let overflow = u32::MAX as usize + 1;
        let before = format!("{:?}", timing.evidence_summary());
        for (batch, isl, prefix, field) in [
            (overflow, 1, 0, "batch_size"),
            (1, overflow, 0, "mean_isl"),
            (1, 1, overflow, "mean_prefix"),
        ] {
            assert!(
                timing
                    .predict_prefill_ms(batch, isl, prefix)
                    .unwrap_err()
                    .to_string()
                    .contains(field)
            );
        }
        for (batch, context, field) in [
            (overflow, 1, "batch_size"),
            (1, overflow, "mean_context_length"),
        ] {
            assert!(
                timing
                    .predict_decode_ms(batch, 0, context, 0)
                    .unwrap_err()
                    .to_string()
                    .contains(field)
            );
        }
        assert!(timing.decode.lock().unwrap().is_none());
        assert_eq!(format!("{:?}", timing.evidence_summary()), before);
        assert!(
            timing
                .predict_decode_ms(1, usize::MAX, 1, usize::MAX)
                .unwrap()
                .is_finite()
        );
    }

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems")
    }

    fn golden_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("parity_tests/perfmodel/goldens")
    }

    fn build_engine(spec_file: &str, db: &Arc<PerfDatabase>) -> Arc<Engine> {
        let path = golden_dir().join("afd_specs").join(spec_file);
        let json = std::fs::read_to_string(&path).expect("read spec fixture");
        let spec: EngineSpec = serde_json::from_str(&json).expect("parse spec");
        Arc::new(Engine::build(spec, db.clone()).expect("engine build"))
    }

    /// The golden-case provider (pure h200 pool pair, frozen identity at
    /// isl=1024/osl=128) shared by the parity and evidence tests.
    fn provider(frozen: bool) -> AfdTimingModel {
        provider_for(1, 18, 3, frozen)
    }

    fn provider_for(
        n_a_nodes: u32,
        batch_size_per_a_replica: u32,
        num_microbatches: u32,
        frozen: bool,
    ) -> AfdTimingModel {
        let db = Arc::new(
            PerfDatabase::load(&systems_root(), "h200_sxm", "sglang", "0.5.14").expect("load db"),
        );
        let a_engine = build_engine("moe_pure_h200_a_sglang.json", &db);
        let f_engine = build_engine("moe_pure_h200_f_sglang.json", &db);
        let config = AfdForwardPassModelConfig {
            model: "Qwen/Qwen3-30B-A3B".into(),
            backend: crate::BackendKind::Sglang,
            backend_version: Some("0.5.14".into()),
            attention_backend: None,
            moe_kernel_source: None,
            topology: Arc::new(crate::engine::AfdEngineConfig {
                a_pool: AfdAPoolSpec {
                    system: "h200_sxm".into(),
                    a_tp_size: 1,
                    n_a_nodes,
                    batch_size_per_a_replica,
                },
                f_pool: AfdFPoolSpec {
                    system: "h200_sxm".into(),
                    f_moe_ep_size: 8,
                    n_f_nodes: 1,
                },
                gpus_per_node: 8,
                f_gpus_per_node: None,
                num_microbatches,
                pipeline_layout: crate::perfmodel::afd::model::AfdPipelineLayout::FourStage,
                comm_overhead_factor: 1.0,
                routing_location: crate::AfdRoutingLocation::A,
            }),
            num_layers: 48,
            hidden_size: 2048,
            num_experts: 128,
            topk: 8,
            model_shape: Some(crate::perfmodel::afd::AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 48,
            }),

            nextn: 0,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        };
        let perf = AfdForwardPassPerfModel::from_engines(config, a_engine, f_engine)
            .expect("afd perf model");
        AfdTimingModel::new(perf, 1024, 128, frozen)
    }

    #[test]
    fn native_replay_retracts_readmits_and_finishes_deterministically() {
        use crate::engine::{Backend, EngineConfig, TimingModelConfig};
        use crate::replay::{
            CanonicalReplayCoverage, CanonicalReplayRecord, ReplayCaptureOptions,
            ReplayDeterminism, ReplayEngineConfig, ReplayEngineFactory, ReplayRequest, ReplaySpec,
            ReplayTerminalStatus, ReplayTopology, Replayer,
        };

        for frozen in [false, true] {
            let mut rank = EngineConfig::for_backend(Backend::Sglang);
            rank.block_size = 1;
            rank.num_gpu_blocks = 330;
            rank.max_num_seqs = 3;
            rank.max_num_batched_tokens = 512;
            rank.enable_prefix_caching = false;
            rank.sglang.schedule_conservativeness = 0.0;
            rank.timing_model = TimingModelConfig::Fixed {
                prefill_ms: 1.0,
                decode_ms: 1.0,
            };
            let spec = ReplaySpec {
                version: 1,
                topology: ReplayTopology::aggregated(1),
                engine: serde_json::to_value(ReplayEngineConfig {
                    rank,
                    ..Default::default()
                })
                .unwrap(),
                encoder: None,
                adapters: Default::default(),
                max_sim_time_ms: None,
                max_in_flight: Some(3),
                record_per_request: true,
                sla: Default::default(),
                requests: (0..3)
                    .map(|i| ReplayRequest {
                        id: format!("pressure-{i}"),
                        arrival_time_ms: 0.0,
                        input_tokens: 96,
                        input_token_ids: Some(vec![i; 96]),
                        output_tokens: 24,
                        output_token_ids: None,
                        dp_rank: None,
                        prefill_dp_rank: None,
                        kv_hints: None,
                        session_id: None,
                        turn_index: None,
                        metadata: serde_json::Value::Null,
                    })
                    .collect(),
            };
            let capture = ReplayCaptureOptions {
                determinism: ReplayDeterminism::CanonicalV1,
                capture_canonical_evidence: true,
                ..Default::default()
            };
            let run = || {
                let timing = Arc::new(provider_for(1, 3, 1, frozen));
                let report =
                    Replayer::new(spec.clone(), ReplayEngineFactory::with_timing_model(timing))
                        .unwrap()
                        .with_capture_options(capture)
                        .run()
                        .unwrap();
                assert_eq!(report.request_counts.completed_requests, 3);
                assert_eq!(report.per_request.len(), 3);
                assert!(report.per_request.iter().all(|r| {
                    r.terminal_status == ReplayTerminalStatus::Completed && r.output_length == 24
                }));
                let pressure = report.runtime_evidence.pressure.as_ref().unwrap();
                assert!(
                    (1..=3).contains(&pressure.sglang_retractions_total),
                    "fixture must exercise bounded pressure: {pressure:?}"
                );
                assert!(
                    pressure
                        .records
                        .iter()
                        .all(|r| r.readmitted_at_ms.is_some())
                );
                assert!(report.committed_prefill_tokens > 288);
                CanonicalReplayRecord::build(
                    &report,
                    serde_json::json!({"timing": "afd", "frozen": frozen}),
                    &CanonicalReplayCoverage::from_report(&report, capture),
                    serde_json::Value::Null,
                )
                .unwrap()
                .into_json_line()
                .unwrap()
            };
            assert_eq!(run(), run());
        }
    }

    #[test]
    fn prefill_uses_actual_batch_for_both_decode_identities() {
        for frozen in [false, true] {
            let small = provider_for(1, 18, 3, frozen);
            let large = provider_for(1, 36, 3, frozen);
            let a = small.predict_prefill_ms(1, 128, 0).unwrap();
            let b = large.predict_prefill_ms(1, 128, 0).unwrap();
            assert!(a > 0.);
            assert_eq!(a, b);
            // Capacity still controls the separate frozen decode identity.
            assert_ne!(
                small
                    .perf
                    .predict_phase_metrics(AfdPhase::Prefill, 128, 8, 0)
                    .unwrap()
                    .billed_step_ms,
                large
                    .perf
                    .predict_phase_metrics(AfdPhase::Prefill, 128, 8, 0)
                    .unwrap()
                    .billed_step_ms
            );
        }
    }

    /// Instantaneous decode keeps 3 microbatches for partial batches with at
    /// least 3 sequences and reduces the count only below that threshold.
    /// The frozen identity stays batch-independent.
    #[test]
    fn instantaneous_decode_limits_microbatches_only_by_sequence_count() {
        let timing = provider(false);
        for (batch, expected_microbatches) in [(18, 3), (8, 3), (3, 3), (2, 2), (1, 1)] {
            let actual = timing
                .predict_decode_ms(batch * 8, batch * 8 * 1100, 1100, 1 << 20)
                .expect("decode");
            let metrics = timing
                .perf
                .predict_decode_step_ms(batch as u32 * 8, 1100, 0)
                .expect("metrics");
            assert_eq!(
                timing
                    .perf
                    .effective_microbatch_geometry(batch as u32 * 8)
                    .0,
                expected_microbatches
            );
            let expected = metrics.billed_step_ms;
            assert_eq!(actual, expected, "batch={batch}");
        }

        let frozen = provider(true);
        let frozen_full = frozen
            .predict_decode_ms(18, 18 * 1024, 1024, 1 << 20)
            .expect("frozen full");
        let frozen_partial = frozen
            .predict_decode_ms(6, 6 * 1024, 1024, 1 << 20)
            .expect("frozen partial");
        assert_eq!(frozen_full, frozen_partial);
    }

    /// Pool-wide batch contract: with two A replicas the provider splits the
    /// pool-wide batch per A replica, so identical per-worker micro geometry
    /// prices identically. (Before the contract fix the instantaneous path
    /// treated the pool-wide count as the per-worker micro batch, inflating
    /// both sides quadratically in a_replica_count.) The dense FFN is
    /// weight-bound at these batch sizes, so the assertions key on equal
    /// geometry producing equal price rather than on latency slopes.
    #[test]
    fn instantaneous_decode_splits_pool_batch_across_a_replicas() {
        let single = provider_for(1, 18, 3, false);
        let dual = provider_for(2, 18, 3, false);
        assert_eq!(single.perf.effective_microbatch_geometry(144), (3, 6));
        assert_eq!(dual.perf.effective_microbatch_geometry(288), (3, 6));
        assert_eq!(dual.perf.effective_microbatch_geometry(144), (3, 3));
        assert_eq!(dual.perf.effective_microbatch_geometry(128), (3, 3));
        assert_eq!(dual.perf.effective_microbatch_geometry(32), (2, 1));

        let one = single
            .perf
            .predict_decode_step_ms(144, 1100, 0)
            .expect("single-worker metrics");
        let two = dual
            .perf
            .predict_decode_step_ms(288, 1100, 0)
            .expect("dual-worker metrics");
        assert!(
            (one.layer_times.attention_ms - two.layer_times.attention_ms).abs() < 1e-12,
            "same per-worker micro geometry must price the A side identically: {} vs {}",
            one.layer_times.attention_ms,
            two.layer_times.attention_ms
        );

        // The dual pool's F side serves b = a_replica_count * a_micro = 12: the
        // same total the single-worker pool reaches only at a_micro = 12.
        let single_b12 = single
            .perf
            .predict_decode_step_ms(288, 1100, 0)
            .expect("single-worker b=12 metrics");
        assert!(
            (two.layer_times.ffn_ms - single_b12.layer_times.ffn_ms).abs() < 1e-12,
            "dual-pool F side must run at b=12 like the single pool at a_micro=12: {} vs {}",
            two.layer_times.ffn_ms,
            single_b12.layer_times.ffn_ms
        );
        assert_eq!(single.perf.effective_microbatch_geometry(288), (3, 12));
    }

    /// The frozen batch and KV sampling remain constant across scheduler
    /// batches and repeated calls. Every call pays a complete pass.
    #[test]
    fn provider_uses_complete_pass_for_the_frozen_identity() {
        let timing = provider(true);

        // Both batches have a busiest A replica of one sequence, but F
        // consumes exactly 8 versus 3 sequences. The fixed measured fixture
        // therefore has a smaller partial-prefill pass; idle A replicas must
        // not pad the F workload.
        let prefill = timing.predict_prefill_ms(8, 1024, 0).expect("prefill");
        assert!(prefill > 0.0);
        let prefill_partial = timing
            .predict_prefill_ms(3, 1024, 0)
            .expect("prefill partial");
        assert!(prefill_partial < prefill);

        // Decode: the stride-averaged frozen step, constant per pass.
        let decode = timing
            .predict_decode_ms(8, 8 * 1024, 1024, 16384)
            .expect("decode");
        assert_eq!(
            decode,
            timing
                .perf
                .predict_phase_metrics(AfdPhase::Decode, 1024, 128, 0)
                .unwrap()
                .billed_step_ms
        );
        let decode_later = timing
            .predict_decode_ms(8, 8 * 1100, 1100, 16384)
            .expect("decode later");
        assert_eq!(decode, decode_later);
    }

    /// The frozen decode-step cache must not swallow evidence: like the
    /// monolithic `AicTimingModel`, every predicted pass accumulates into
    /// the summary (power statistics are per-replay totals).
    #[cfg(feature = "python")]
    #[test]
    fn frozen_decode_records_evidence_per_call() {
        let timing = provider(true);
        timing.reset_evidence().expect("reset");
        timing
            .predict_decode_ms(8, 8 * 1024, 1024, 16384)
            .expect("decode");
        let one_pass = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;
        assert!(one_pass > 0.0, "one decode pass must record evidence");
        for _ in 0..4 {
            timing
                .predict_decode_ms(8, 8 * 1024, 1024, 16384)
                .expect("decode");
        }
        let total = timing
            .evidence_summary()
            .expect("summary")
            .decode
            .latency_ms;
        assert!(
            (total - 5.0 * one_pass).abs() < 1e-9 * one_pass,
            "five decode calls must accumulate five passes: total={total} one={one_pass}"
        );
    }
}
