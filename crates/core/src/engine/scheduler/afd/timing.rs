// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AFD replay timing uses actual per-pass geometry and records each query.

#[cfg(feature = "python")]
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
use crate::perfmodel::afd::model::AfdForwardPassPerfModel;
#[cfg(feature = "python")]
use crate::perfmodel::afd::model::AfdOpEvidence;

/// Replay timing provider for AFD ranks.
pub struct AfdTimingModel {
    perf: AfdForwardPassPerfModel,
    knobs: AfdPipelineKnobs,
    isl: u32,
    osl: u32,
    /// Power evidence accumulator: per-op energy from the two
    /// pool engines flows into the standard replay power statistics.
    #[cfg(feature = "python")]
    evidence: Mutex<TimingEvidenceAccumulator>,
}

impl AfdTimingModel {
    /// Price every replay pass from its current batch and KV context.
    pub fn new(perf: AfdForwardPassPerfModel, isl: u32, osl: u32) -> Self {
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
        let batch_size = checked_timing_u32(batch_size, "batch_size")?;
        let mean_context_length =
            checked_timing_u32(mean_context_length.max(1), "mean_context_length")?;
        #[cfg(feature = "python")]
        let (metrics, operations) =
            self.perf
                .predict_decode_step_with_operations(batch_size, mean_context_length, 0)?;
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
        Ok(evaluation.step_latency_ms)
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
pub(crate) mod tests {
    use super::*;
    use crate::perf_database::PerfDatabase;
    use crate::perfmodel::afd::model::AfdPhase;
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
    fn timing_overflow_does_not_mutate_evidence_() {
        let timing = provider();
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
        assert_eq!(format!("{:?}", timing.evidence_summary()), before);
        assert!(
            timing
                .predict_decode_ms(1, usize::MAX, 1, usize::MAX)
                .unwrap()
                .is_finite()
        );
    }

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/aisim/src/aisim_core/systems")
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

    /// Pure h200 pool pair shared by parity and evidence tests.
    fn provider() -> AfdTimingModel {
        provider_for(1, 18, 3)
    }

    pub(crate) fn provider_for(
        n_a_nodes: u32,
        batch_size_per_a_replica: u32,
        num_microbatches: u32,
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
        AfdTimingModel::new(perf, 1024, 128)
    }

    #[test]
    fn prefill_uses_actual_batch_independently_of_capacity() {
        {
            let small = provider_for(1, 18, 3);
            let large = provider_for(1, 36, 3);
            let a = small.predict_prefill_ms(1, 128, 0).unwrap();
            let b = large.predict_prefill_ms(1, 128, 0).unwrap();
            assert!(a > 0.);
            assert_eq!(a, b);
            // Static forward queries continue to use configured capacity.
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
    #[test]
    fn instantaneous_decode_limits_microbatches_only_by_sequence_count() {
        let timing = provider();
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
    }

    /// Pool-wide batch contract: with two A replicas the provider splits the
    /// pool-wide batch per A replica, so identical per-worker micro geometry
    /// prices identically. (Before the contract fix the instantaneous path
    /// treated the pool-wide count as the per-worker micro batch, inflating
    /// both sides quadratically in a_replica_count.) The dense FFN is
    /// weight-bound at these batch sizes, so the assertions key on equal
    /// geometry producing equal price rather than on latency slopes.
    #[test]
    fn prefill_retained_buffers_share_one_pool_token_budget() {
        let single = provider_for(1, 32, 1);
        let deep = provider_for(1, 32, 16);
        let a = single.perf.prefill_plan_memory_bytes(4096, 4096);
        let b = deep.perf.prefill_plan_memory_bytes(4096, 4096);
        // TP1 has no slice padding. Only F probability rounding may add
        // at most one hidden shard per additional microbatch.
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.3, b.3);
        assert!(b.2 <= a.2 + 15 * 2048 * 2);
        let capped = provider_for(1, 8, 3)
            .perf
            .prefill_plan_memory_bytes(2048, 128);
        assert_eq!(capped.4, 3 * 128); // ceil(8 / 3) sequences per A microbatch
        let half = deep.perf.prefill_plan_memory_bytes(2048, 4096);
        assert!(half.0 < b.0 && half.3 < b.3);
    }

    #[test]
    fn instantaneous_decode_splits_pool_batch_across_a_replicas() {
        let single = provider_for(1, 18, 3);
        let dual = provider_for(2, 18, 3);
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

    /// Every decode pass records evidence: like the
    /// monolithic `AicTimingModel`, every predicted pass accumulates into
    /// the summary (power statistics are per-replay totals).
    #[cfg(feature = "python")]
    #[test]
    fn dynamic_decode_records_evidence_per_call() {
        let timing = provider();
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
