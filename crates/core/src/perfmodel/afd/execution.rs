// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified native port of heterogeneous AFD/fabric behavior; source revision and
// original paths: root THIRD_PARTY_NOTICES.md (AISimulate heterogeneous AFD).
// Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

//! Prepared heterogeneous-layer execution. All numeric queries use native ops.
use super::super::batch::{BatchShape, Microbatches};
use super::*;
use crate::operators::Op;
use crate::operators::communication::NcclOp;
use crate::operators::elementwise::ElementwiseOp;
use crate::perfmodel::afd::plan::{AfdExecutionPlan, AfdPhasePlan, token_shards};

type Communication = [(&'static str, usize, PerformanceResult, f64); 7];

fn phase_plan(plan: &AfdExecutionPlan, phase: AfdPhase) -> &AfdPhasePlan {
    match phase {
        AfdPhase::Prefill => &plan.context,
        AfdPhase::Decode => &plan.generation,
    }
}
fn phase_ops(engine: &Engine, phase: AfdPhase) -> &[Op] {
    match phase {
        AfdPhase::Prefill => engine.context_ops(),
        AfdPhase::Decode => engine.generation_ops(),
    }
}

pub(super) fn physical_weights(engine: &Engine, plan: &AfdExecutionPlan, phase: AfdPhase) -> f64 {
    let p = phase_plan(plan, phase);
    let ops = phase_ops(engine, phase);
    let mut count = vec![0_u32; p.templates.len()];
    for &i in &plan.layers {
        count[i] += 1;
    }
    let mut weights = 0.0;
    for (t, n) in p.templates.iter().zip(count) {
        weights += t
            .ops
            .iter()
            .chain(&t.rank_ops)
            .map(|&i| ops[i].weight_bytes())
            .sum::<f64>()
            / t.work_scale
            * f64::from(n);
    }
    // Generation one-time work may have MTP work scaling too.
    let scale = p.templates.first().map_or(1.0, |t| t.work_scale);
    weights += p
        .prefix
        .iter()
        .chain(&p.suffix)
        .map(|&i| ops[i].weight_bytes())
        .sum::<f64>()
        / scale;
    weights / (1_u64 << 30) as f64
}

fn query(
    engine: &Engine,
    phase: AfdPhase,
    index: usize,
    batch: u32,
    seq: u32,
    prefix: u32,
    tokens: u32,
) -> Result<PerformanceResult, AicError> {
    let op = &phase_ops(engine, phase)[index];
    match phase {
        AfdPhase::Prefill => {
            query_context_op(op, engine.database(), batch, seq, prefix, 1.0, Some(tokens))
        }
        AfdPhase::Decode => query_generation_op(
            op,
            engine.database(),
            batch,
            1,
            seq,
            1.0,
            prefix,
            Some(tokens),
        ),
    }
}

fn rank_result(
    mut query: impl FnMut(u32) -> Result<PerformanceResult, AicError>,
    tokens: u32,
    tp: u32,
) -> Result<(PerformanceResult, f64), AicError> {
    let mut combined = PerformanceResult::zero();
    let mut active_gpu_ms = 0.0;
    let mut first = true;
    for (slice, ranks) in token_shards(tokens, tp) {
        if ranks == 0 || slice == 0 {
            continue;
        }
        let r = query(slice)?;
        combined.latency_ms = combined.latency_ms.max(r.latency_ms);
        active_gpu_ms += r.latency_ms * f64::from(ranks);
        combined.energy_wms += r.energy_wms * f64::from(ranks);
        combined.source = if first {
            r.source
        } else {
            combined.source.combine(r.source)
        };
        combined.moe_comm_fallbacks.extend(r.moe_comm_fallbacks);
        first = false;
    }
    Ok((combined, active_gpu_ms))
}

#[derive(Clone)]
struct AWork {
    branches: [f64; 4],
    result: PerformanceResult,
    active: f64,
}
fn a_work(
    engine: &Engine,
    phase: AfdPhase,
    index: usize,
    shape: BatchShape,
    seq: u32,
    prefix: u32,
    width: u32,
    rank_local: bool,
) -> Result<AWork, AicError> {
    let tp = engine.parallelism().0;
    let mut w = AWork {
        branches: [0.; 4],
        result: PerformanceResult::zero(),
        active: 0.,
    };
    let mut seen = false;
    for (g, (batch, copies)) in shape.groups.into_iter().enumerate() {
        if batch == 0 || copies == 0 {
            continue;
        }
        let batch = batch
            .checked_mul(width)
            .ok_or_else(|| AicError::InvalidEngineConfig("AFD batch overflow".into()))?;
        let tokens = if phase == AfdPhase::Prefill {
            batch.checked_mul(seq)
        } else {
            Some(batch)
        }
        .ok_or_else(|| AicError::InvalidEngineConfig("AFD token overflow".into()))?;
        let shards = if rank_local {
            token_shards(tokens, tp)
        } else {
            [(tokens, tp), (0, 0)]
        };
        for (i, (slice, ranks)) in shards.into_iter().enumerate() {
            if slice == 0 || ranks == 0 {
                continue;
            }
            let r = query(engine, phase, index, batch, seq, prefix, slice)?;
            w.branches[g * 2 + i] = r.latency_ms;
            // Broadcast work runs on every rank; both branch slots must
            // include it before adding rank-local work to either branch.
            if !rank_local {
                w.branches[g * 2 + 1] = r.latency_ms;
            }
            w.active += r.latency_ms * f64::from(ranks * copies);
            w.result.energy_wms += r.energy_wms * f64::from(ranks * copies);
            w.result.source = if seen {
                w.result.source.combine(r.source)
            } else {
                r.source
            };
            seen = true;
            w.result.moe_comm_fallbacks.extend(r.moe_comm_fallbacks);
        }
    }
    w.result.latency_ms = w.branches.into_iter().fold(0., f64::max);
    Ok(w)
}

fn cached(
    cache: &mut [Option<PerformanceResult>],
    i: usize,
    query: impl FnOnce() -> Result<PerformanceResult, AicError>,
) -> Result<PerformanceResult, AicError> {
    if let Some(result) = &cache[i] {
        return Ok(result.clone());
    }
    let result = query()?;
    cache[i] = Some(result.clone());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AfdRoutingLocation;
    use crate::perf_database::PerfDatabase;

    #[test]
    fn physical_weights_count_source_layers_and_copies_without_work_scaling() {
        use crate::common::enums::GemmQuantMode;
        use crate::operators::GemmOp;
        use crate::perfmodel::afd::plan::AfdLayerTemplate;
        use crate::perfmodel::engine::spec::EngineSpec;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut spec: EngineSpec = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("parity_tests/perfmodel/goldens/afd_specs/moe_pure_h200_a_sglang.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let db = Arc::new(
            PerfDatabase::load(
                &root.join("../../python/aisimulate/src/aisimulate_core/systems"),
                "h200_sxm",
                "sglang",
                "0.5.14",
            )
            .unwrap(),
        );
        let mut attention = GemmOp::new("attention", 4, 8, GemmQuantMode::Bfloat16);
        let mut shared = GemmOp::new("shared", 16, 8, GemmQuantMode::Bfloat16);
        let mut head = GemmOp::new("logits", 8, 2, GemmQuantMode::Bfloat16);
        for op in [&mut attention, &mut shared, &mut head] {
            op.scale_factor = 0.5;
        }
        spec.context_ops = vec![Op::Gemm(attention), Op::Gemm(shared), Op::Gemm(head)];
        spec.generation_ops = spec.context_ops.clone();
        let phase = AfdPhasePlan {
            templates: vec![
                AfdLayerTemplate {
                    attention_kind: "global".into(),
                    routed: false,
                    ops: vec![0],
                    rank_ops: vec![],
                    work_scale: 0.5,
                },
                AfdLayerTemplate {
                    attention_kind: "global".into(),
                    routed: true,
                    ops: vec![0],
                    rank_ops: vec![1],
                    work_scale: 0.5,
                },
            ],
            prefix: vec![],
            suffix: vec![2],
        };
        let plan = AfdExecutionPlan {
            role: crate::AfdPoolRole::A,
            model: spec.engine.model_name.clone(),
            hidden_size: 8,
            num_experts: 4,
            topk: 2,
            shared_inter_size: 16,
            layers: vec![0, 1, 1],
            context: phase.clone(),
            generation: phase,
        };
        spec.engine.extra.insert("afd_pool_role".into(), "a".into());
        spec.afd_plan = Some(plan.clone());
        let mut invalid = spec.clone();
        invalid.engine.extra.remove("afd_pool_role");
        assert!(Engine::build(invalid, db.clone()).is_err());
        let mut invalid = spec.clone();
        invalid
            .engine
            .extra
            .insert("afd_pool_role".into(), "f".into());
        assert!(Engine::build(invalid, db.clone()).is_err());
        let engine = Engine::build(spec, db).unwrap();
        // BF16: 3 attention copies (4*8*2), 2 complete shared copies
        // (16*8*2), one head (8*2*2). No microbatch/execution multiplier.
        assert_eq!(
            physical_weights(&engine, &plan, AfdPhase::Prefill) * (1_u64 << 30) as f64,
            736.
        );
    }

    /// Measured GEMMs: m=3 costs 2 ms, m=8 costs 3 ms, and m=9 costs
    /// 9 ms. B=8/m=3 must query the prefix at 8, never padded 9, and run
    /// each of its three ops once. Routed A=2/F=1 gives a 7 ms pipeline.
    #[test]
    fn dense_prefix_uses_exact_full_batch_for_latency_energy_and_memory() {
        use crate::common::enums::{GemmQuantMode, MoeQuantMode};
        use crate::operators::{GemmOp, MoeOp};
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        use crate::perfmodel::afd::plan::AfdLayerTemplate;
        use crate::perfmodel::engine::spec::EngineSpec;

        let tmp = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(tmp.path());
        // This fixture supplies measured compute and explicit communication
        // availability; its pipeline tests inject zero communication timings.
        std::fs::create_dir_all(tmp.path().join("data/nccl/test")).unwrap();
        write_parquet(
            &tmp.path().join("data/nccl/test/nccl_perf.parquet"),
            &[
                Col::Str("nccl_dtype", vec!["half"; 2]),
                Col::Str("op_name", vec!["all_gather", "reduce_scatter"]),
                Col::I64("num_gpus", vec![8; 2]),
                Col::I64("message_size", vec![1024; 2]),
                Col::F64("latency", vec![1.; 2]),
            ],
        );

        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["bfloat16"; 6]),
                Col::I64("m", vec![1, 3, 8, 9, 12, 32]),
                Col::I64("n", vec![256; 6]),
                Col::I64("k", vec![256; 6]),
                Col::F64("latency", vec![1., 2., 3., 9., 5., 6.]),
                Col::F64("power", vec![100.; 6]),
            ],
        );
        write_parquet(
            &data.join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["bfloat16"; 2]),
                Col::I64("num_tokens", vec![1, 256]),
                Col::I64("hidden_size", vec![256; 2]),
                Col::I64("inter_size", vec![128; 2]),
                Col::I64("topk", vec![2; 2]),
                Col::I64("num_experts", vec![32; 2]),
                Col::I64("moe_tp_size", vec![1; 2]),
                Col::I64("moe_ep_size", vec![8; 2]),
                Col::Str("distribution", vec!["uniform"; 2]),
                Col::F64("latency", vec![1.; 2]),
                Col::F64("power", vec![100.; 2]),
            ],
        );
        let db = Arc::new(PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").unwrap());
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("parity_tests/perfmodel/goldens/afd_specs/moe_pure_h200_a_sglang.json");
        let mut engines = Vec::new();
        for role in [AfdPoolRole::A, AfdPoolRole::F] {
            let a = role == AfdPoolRole::A;
            let mut spec: EngineSpec =
                serde_json::from_str(&std::fs::read_to_string(&fixture).unwrap()).unwrap();
            spec.schema_version = crate::ENGINE_SPEC_SCHEMA_VERSION;
            spec.engine.system_name = "testsys".into();
            spec.engine.backend = BackendKind::Vllm;
            spec.engine.backend_version = Some("1.0".into());
            spec.engine.parallel.tp_size = if a { 1 } else { 8 };
            spec.engine.parallel.moe_tp_size = Some(1);
            spec.engine.parallel.moe_ep_size = Some(if a { 1 } else { 8 });
            spec.engine
                .extra
                .insert("afd_pool_role".into(), if a { "a" } else { "f" }.into());
            spec.context_ops = if a {
                ["attention", "afd_dense_gate_up", "embedding"]
                    .into_iter()
                    .map(|name| Op::Gemm(GemmOp::new(name, 256, 256, GemmQuantMode::Bfloat16)))
                    .collect()
            } else {
                vec![Op::Moe(MoeOp::new(
                    "routed",
                    256,
                    128,
                    2,
                    32,
                    1,
                    8,
                    MoeQuantMode::Bfloat16,
                    "uniform",
                ))]
            };
            spec.generation_ops = spec.context_ops.clone();
            let template = |routed| AfdLayerTemplate {
                attention_kind: "global".into(),
                routed,
                ops: if a {
                    if routed { vec![0] } else { vec![0, 1] }
                } else if routed {
                    vec![0]
                } else {
                    vec![]
                },
                rank_ops: vec![],
                work_scale: 1.,
            };
            let phase = AfdPhasePlan {
                templates: vec![template(false), template(true)],
                prefix: if a { vec![2] } else { vec![] },
                suffix: vec![],
            };
            spec.afd_plan = Some(AfdExecutionPlan {
                role,
                model: spec.engine.model_name.clone(),
                hidden_size: 256,
                num_experts: 32,
                topk: 2,
                layers: vec![0, 1],
                shared_inter_size: 0,
                context: phase.clone(),
                generation: phase,
            });
            // Saved engine specs must reject a dense layer after a routed layer.
            let mut invalid = spec.clone();
            invalid.afd_plan.as_mut().unwrap().layers = vec![1, 0];
            let invalid = EngineSpec::from_bincode(&invalid.to_bincode().unwrap()).unwrap();
            assert!(matches!(
                Engine::build(invalid, db.clone()),
                Err(AicError::NotImplemented(_))
            ));
            engines.push(Arc::new(Engine::build(spec, db.clone()).unwrap()));
        }
        let wire = AfdTimingConfig {
            model: "Qwen/Qwen3-30B-A3B".into(),
            backend_version: Some("1.0".into()),
            attention_backend: None,
            moe_kernel_source: None,
            isl: 1,
            osl: 2,
            num_layers: 2,
            hidden_size: 256,
            num_experts: 32,
            topk: 2,
            model_shape: Some(AfdModelShape {
                num_shared_experts: 0,
                num_moe_layers: 1,
            }),
            frozen_decode_identity: false,
            quant: AfdQuantOverrides::default(),
            estimator: AfdEstimatorPlumbing::default(),
        };
        let topology = Arc::new(AfdEngineConfig {
            a_pool: AfdAPoolSpec {
                system: "testsys".into(),
                a_tp_size: 1,
                n_a_nodes: 1,
                batch_size_per_a_replica: 8,
            },
            f_pool: AfdFPoolSpec {
                system: "testsys".into(),
                f_moe_ep_size: 8,
                n_f_nodes: 1,
            },
            gpus_per_node: 8,
            f_gpus_per_node: None,
            num_microbatches: 3,
            pipeline_layout: AfdPipelineLayout::FourStage,
            comm_overhead_factor: 1.,
            routing_location: AfdRoutingLocation::A,
        });
        for nextn in [0, 3] {
            let config = wire
                .clone()
                .into_perf_config(topology.clone(), BackendKind::Vllm, nextn)
                .unwrap();
            let mut model = AfdForwardPassPerfModel::from_engines(
                config,
                engines[0].clone(),
                engines[1].clone(),
            )
            .unwrap();
            let comms: Communication =
                std::array::from_fn(|_| ("unused", 1, PerformanceResult::zero(), 0.));
            let run = |phase, micro_tokens, expected, prefix_energy| {
                let mut evidence = Vec::new();
                let result = model
                    .measure_plan(phase, micro_tokens, 8, 1, 0, 3, Some(&comms), |name, r| {
                        evidence.push((name.to_owned(), r.energy_wms));
                    })
                    .unwrap();
                assert_eq!(result.billed_step_ms, expected);
                for name in ["afd_dense_gate_up", "embedding"] {
                    let entries: Vec<_> = evidence.iter().filter(|(n, _)| n == name).collect();
                    assert_eq!(entries.len(), 1);
                    // Eight A ranks out of 16 total, each consuming 100 W.
                    assert_eq!(entries[0].1, prefix_energy);
                }
            };
            // Exact 3+3+2 tail: A runs 2+2+1.5ms, F drains 1ms later.
            // The 9ms full prefix gives 15.5ms, rather than padded 16ms.
            run(AfdPhase::Prefill, 3, 15.5, 150.);
            run(
                AfdPhase::Decode,
                if nextn == 0 { 3 } else { 12 },
                if nextn == 0 { 15.5 } else { 32. },
                if nextn == 0 { 150. } else { 300. },
            );
            // Full-batch dense intermediate: 256 * 3 * 8, plus 256*2*8 hidden.
            assert_eq!(model.plan_activation_bytes(3, 8), 10240);
            model.config.topology = Arc::new(AfdEngineConfig {
                pipeline_layout: AfdPipelineLayout::TwoStage,
                ..(*topology).clone()
            });
            assert_eq!(
                model
                    .measure_plan(AfdPhase::Prefill, 3, 8, 1, 0, 3, Some(&comms), |_, _| {})
                    .unwrap()
                    .billed_step_ms,
                15.5
            );
        }
    }

    #[test]
    fn rank_shapes_conserve_tokens_and_sum_participating_energy() {
        for (tokens, tp, shapes, slowest) in [
            (7, 1, vec![7], 7.),
            (7, 2, vec![4, 3], 4.),
            (7, 4, vec![2, 1], 2.),
            (2, 4, vec![1], 1.),
            (0, 4, vec![], 0.),
        ] {
            let mut queried = Vec::new();
            let (r, active_gpu_ms) = rank_result(
                |slice| {
                    queried.push(slice);
                    let mut r = PerformanceResult::zero();
                    r.latency_ms = f64::from(slice);
                    r.energy_wms = f64::from(slice) * 3.;
                    Ok(r)
                },
                tokens,
                tp,
            )
            .unwrap();
            assert_eq!(queried, shapes);
            assert_eq!(r.latency_ms, slowest);
            assert_eq!(r.energy_wms, f64::from(tokens) * 3.);
            assert_eq!(active_gpu_ms, f64::from(tokens));
        }
    }
}

impl AfdForwardPassPerfModel {
    /// Extra A buffers retained during remote expert work; used by both OOM
    /// preflight and admission budgeting, in bytes per physical A rank.
    pub(crate) fn plan_workspace_bytes(&self, tokens: u32) -> u64 {
        let plan = self.a_pool.engine.afd_plan().expect("validated A plan");
        let slice = u64::from(tokens.div_ceil(self.config.topology.a_pool.a_tp_size));
        let hidden = u64::from(self.config.hidden_size);
        let held = if plan.shared_inter_size > 0 {
            slice * hidden * 2 * u64::from(self.config.topology.num_microbatches)
        } else {
            0
        };
        // Rank-local residuals for all in-flight microbatches, send/return
        // staging and padded gathered output (bf16 local tensors).
        held + slice
            * hidden
            * 2
            * (u64::from(self.config.topology.num_microbatches)
                + 2
                + u64::from(self.config.topology.a_pool.a_tp_size))
    }

    pub(crate) fn f_plan_buffers(&self, tokens: u32) -> (u64, u64) {
        let p = crate::operators::afd_comm::afd_send_prob(
            self.config.num_experts,
            self.config.topk,
            self.config.topology.f_pool.n_f_nodes,
        );
        let local = u64::from(self.config.topology.effective_f_gpus_per_node());
        let rank_tokens = (p * self.token_mapping.busiest(tokens) as f64).ceil() as u64;
        let node_tokens = rank_tokens * local;
        let hidden = u64::from(self.config.hidden_size);
        // Full node input/output during compute; rank-local return staging
        // can be retained for every circulating microbatch.
        let peak = node_tokens * hidden * 4;
        let held = rank_tokens * hidden * 2 * u64::from(self.config.topology.num_microbatches);
        (peak, held)
    }

    #[cfg(not(feature = "python"))]
    pub(crate) fn predict_prefill_step_ms(
        &self,
        pool_batch: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<AfdPhaseMetrics, AicError> {
        self.measure_prefill_step(pool_batch, isl, prefix, |_, _| {})
    }

    fn measure_prefill_step(
        &self,
        pool_batch: u32,
        isl: u32,
        prefix: u32,
        visit: impl FnMut(&str, &PerformanceResult),
    ) -> Result<AfdPhaseMetrics, AicError> {
        if pool_batch == 0 || isl == 0 || prefix > isl {
            return Err(AicError::InvalidEngineConfig(
                "AFD prefill requires a positive batch/ISL and prefix <= ISL".into(),
            ));
        }
        let (m, _batch) = self.effective_microbatch_geometry(pool_batch);
        self.measure_pool_plan(
            AfdPhase::Prefill,
            pool_batch,
            isl.saturating_sub(prefix).max(1),
            prefix,
            m,
            None,
            visit,
        )
    }

    #[cfg(feature = "python")]
    pub(crate) fn predict_prefill_step_with_operations(
        &self,
        pool_batch: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<(AfdPhaseMetrics, Vec<AfdOpEvidence>), AicError> {
        let mut operations = Vec::new();
        let metrics = self.measure_prefill_step(pool_batch, isl, prefix, |name, r| {
            operations.push(AfdOpEvidence::new(name, r))
        })?;
        Ok((metrics, operations))
    }

    pub(crate) fn plan_activation_bytes(&self, tokens: u32, full_tokens: u32) -> u64 {
        let plan = self.a_pool.engine.afd_plan().expect("validated A plan");
        let shared = u64::from(tokens.div_ceil(self.config.topology.a_pool.a_tp_size))
            * u64::from(plan.shared_inter_size)
            * 6;
        let dense = self
            .a_pool
            .engine
            .context_ops()
            .iter()
            .filter_map(|op| match op {
                Op::Gemm(g) if g.name.contains("afd_dense") && g.name.contains("gate_up") => {
                    Some(u64::from(g.n) * 3 * u64::from(full_tokens))
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let routed = shared + u64::from(tokens) * u64::from(self.config.hidden_size) * 2;
        if self.dense_prefix_layers == 0 {
            routed
        } else {
            routed.max(dense + u64::from(full_tokens) * u64::from(self.config.hidden_size) * 2)
        }
    }

    fn plan_communication(
        &self,
        phase: AfdPhase,
        shape: BatchShape,
        tp: u32,
        shared: bool,
    ) -> Result<Communication, AicError> {
        let cfg = &self.config;
        let f_ranks = cfg.topology.f_pool.n_f_nodes * cfg.topology.effective_f_gpus_per_node();
        let a_db = self.a_pool.engine.database();
        let f_db = self.f_pool.engine.database();
        let tokens = shape.max();
        let receiver = self.token_mapping.busiest_groups(shape.groups);
        let mut outgoing =
            self.a2f
                .query_with_receiver_load(a_db, tokens, tp, receiver, Some(self.fabric))?;
        let mut incoming =
            self.f2a
                .query_with_receiver_load(a_db, tokens, tp, receiver, Some(self.fabric))?;
        let mut ag = self.f_ag.query_with_receiver_load(f_db, receiver)?;
        let mut rs = self.f_rs.query_with_receiver_load(f_db, receiver)?;
        ag.energy_wms *= f64::from(f_ranks);
        rs.energy_wms *= f64::from(f_ranks);
        // F-node reduce-scatter already sums local expert ranks. A receives
        // one partial per F node, not one per expert GPU.
        let mut combine = self.a_combine.clone();
        combine.a_tp_size = 1;
        combine.f_moe_ep_size = cfg.topology.f_pool.n_f_nodes;
        let mut combined = PerformanceResult::zero();
        let mut merged = PerformanceResult::zero();
        let mut gathered = PerformanceResult::zero();
        let (mut combine_active_ms, mut merge_active_ms, mut gathered_active_ms) = (0., 0., 0.);
        let mut seen = false;
        for (tokens, copies) in shape.groups {
            if tokens == 0 || copies == 0 {
                continue;
            }
            let (c, active) = rank_result(|slice| combine.query(a_db, slice), tokens, tp)?;
            combined.latency_ms = combined.latency_ms.max(c.latency_ms);
            combined.energy_wms += c.energy_wms * f64::from(copies);
            combined.source = if seen {
                combined.source.combine(c.source)
            } else {
                c.source
            };
            combine_active_ms += active * f64::from(copies);
            if shared {
                let merge =
                    ElementwiseOp::new("afd_a_shared_merge", f64::from(cfg.hidden_size) * 6.);
                let (r, active) = rank_result(|slice| merge.query(a_db, slice), tokens, tp)?;
                merged.latency_ms = merged.latency_ms.max(r.latency_ms);
                merged.energy_wms += r.energy_wms * f64::from(copies);
                merged.source = if seen {
                    merged.source.combine(r.source)
                } else {
                    r.source
                };
                merge_active_ms += active * f64::from(copies);
            }
            if tp > 1 {
                let gather = NcclOp {
                    name: "afd_a_token_allgather".into(),
                    scale_factor: 1.,
                    hidden_size: f64::from(tokens.div_ceil(tp)) * f64::from(cfg.hidden_size),
                    num_gpus: tp,
                    dtype: CommQuantMode::Half,
                    operation: "all_gather".into(),
                    seq_split: 1,
                };
                let r = gather.query(a_db, 1)?;
                gathered.latency_ms = gathered.latency_ms.max(r.latency_ms);
                gathered.energy_wms += r.energy_wms * f64::from(tp * copies);
                gathered.source = if seen {
                    gathered.source.combine(r.source)
                } else {
                    r.source
                };
                gathered_active_ms += r.latency_ms * f64::from(tp * copies);
            }
            seen = true;
        }
        let active_replicas = shape
            .groups
            .iter()
            .filter(|(tokens, _)| *tokens > 0)
            .map(|(_, n)| *n)
            .sum::<u32>();
        let all_ranks = f64::from(tp * active_replicas) + f64::from(f_ranks);
        outgoing.energy_wms *= all_ranks;
        incoming.energy_wms *= all_ranks;
        let outgoing_active_ms = outgoing.latency_ms * all_ranks;
        let incoming_active_ms = incoming.latency_ms * all_ranks;
        let ag_active_ms = ag.latency_ms * f64::from(f_ranks);
        let rs_active_ms = rs.latency_ms * f64::from(f_ranks);
        let results = [
            ("afd_a2f_transfer", 1, outgoing, outgoing_active_ms),
            ("afd_f2a_transfer", 3, incoming, incoming_active_ms),
            ("afd_f_node_allgather", 2, ag, ag_active_ms),
            ("afd_f_node_reducescatter", 2, rs, rs_active_ms),
            ("afd_a_side_combine", 4, combined, combine_active_ms),
            ("afd_a_shared_merge", 4, merged, merge_active_ms),
            ("afd_a_token_allgather", 4, gathered, gathered_active_ms),
        ];
        // These operations replace the aggregate graph's communication and
        // return-side work. Scale their results once, before pipeline grouping
        // and evidence. Compiled A-local TP ops already carry their work scale.
        let scale = self.communication_work_scale(phase);
        Ok(if scale == 1.0 {
            results
        } else {
            results.map(|(name, stage, r, active_ms)| {
                (name, stage, r.scaled(scale), active_ms * scale)
            })
        })
    }

    // Compatibility for full-capacity callers and explicit communication fixtures.
    pub(super) fn measure_plan(
        &self,
        phase: AfdPhase,
        _batch: u32,
        full_batch: u32,
        seq: u32,
        prefix: u32,
        m: u32,
        communication: Option<&Communication>,
        visit: impl FnMut(&str, &PerformanceResult),
    ) -> Result<AfdPhaseMetrics, AicError> {
        let pool = full_batch
            .checked_mul(a_replica_count(&self.config))
            .ok_or_else(|| AicError::InvalidEngineConfig("AFD pool batch overflow".into()))?;
        self.measure_pool_plan(
            phase,
            pool,
            seq,
            prefix,
            m,
            communication.map(std::slice::from_ref),
            visit,
        )
    }

    pub(super) fn measure_pool_plan(
        &self,
        phase: AfdPhase,
        pool_batch: u32,
        seq: u32,
        prefix: u32,
        m: u32,
        communication: Option<&[Communication]>,
        mut visit: impl FnMut(&str, &PerformanceResult),
    ) -> Result<AfdPhaseMetrics, AicError> {
        let cfg = &self.config;
        let replicas = a_replica_count(cfg);
        let batches = Microbatches::new(pool_batch, replicas, m);
        let full = BatchShape::full(pool_batch, replicas);
        let ap = self.a_pool.engine.afd_plan().expect("prepared A plan");
        let fp = self.f_pool.engine.afd_plan().expect("prepared F plan");
        let a = phase_plan(ap, phase);
        let f = phase_plan(fp, phase);
        let tp = cfg.topology.a_pool.a_tp_size;
        let f_ranks = cfg.topology.f_pool.n_f_nodes * cfg.topology.effective_f_gpus_per_node();
        let gpu_count = f64::from(tp * replicas + f_ranks);
        let width = if phase == AfdPhase::Decode {
            cfg.nextn + 1
        } else {
            1
        };
        let token_width = if phase == AfdPhase::Prefill {
            seq
        } else {
            width
        };
        let mut emit = |name: &str, mut r: PerformanceResult, active: f64, count: f64| {
            r.latency_ms = active;
            visit(name, &r.scaled(count / gpu_count));
        };
        // Dense prefix runs at the exact full replica populations once.
        let mut full_prefix_ms = 0.;
        if self.dense_prefix_layers > 0 {
            let mut branches = [0.; 4];
            for &j in &a.prefix {
                let w = a_work(
                    &self.a_pool.engine,
                    phase,
                    j,
                    full,
                    seq,
                    prefix,
                    width,
                    false,
                )?;
                for (b, t) in branches.iter_mut().zip(w.branches) {
                    *b += t;
                }
                emit(
                    phase_ops(&self.a_pool.engine, phase)[j].name(),
                    w.result,
                    w.active,
                    1.,
                );
            }
            for (i, t) in a.templates.iter().enumerate().filter(|(_, t)| !t.routed) {
                for &j in &t.ops {
                    let w = a_work(
                        &self.a_pool.engine,
                        phase,
                        j,
                        full,
                        seq,
                        prefix,
                        width,
                        false,
                    )?;
                    for (b, t) in branches.iter_mut().zip(w.branches) {
                        *b += t * f64::from(self.repetitions[i]);
                    }
                    emit(
                        phase_ops(&self.a_pool.engine, phase)[j].name(),
                        w.result,
                        w.active,
                        f64::from(self.repetitions[i]),
                    );
                }
            }
            full_prefix_ms = branches.into_iter().fold(0., f64::max);
        }
        let mut measurements = Vec::with_capacity(batches.len);
        let mut sums = [0.; 5];
        let mut endpoints = 0.;
        for (ri, &(shape, repeats)) in batches.runs[..batches.len].iter().enumerate() {
            let tokens = shape.scaled(token_width)?;
            let owned;
            let comms = match communication {
                Some(c) => &c[ri.min(c.len() - 1)],
                None => {
                    owned = self.plan_communication(phase, tokens, tp, ap.shared_inter_size > 0)?;
                    &owned
                }
            };
            let mut durations = vec![[0.; 5]; a.templates.len()];
            let mut cache_inline: [Option<AWork>; 32] = std::array::from_fn(|_| None);
            let mut cache_heap;
            let len = phase_ops(&self.a_pool.engine, phase).len();
            let cache = if len <= 32 {
                &mut cache_inline[..len]
            } else {
                cache_heap = vec![None; len];
                &mut cache_heap
            };
            let mut f_inline: [Option<PerformanceResult>; 8] = std::array::from_fn(|_| None);
            let mut f_heap;
            let flen = phase_ops(&self.f_pool.engine, phase).len();
            let fcache = if flen <= 8 {
                &mut f_inline[..flen]
            } else {
                f_heap = vec![None; flen];
                &mut f_heap
            };
            for (i, (at, ft)) in a.templates.iter().zip(&f.templates).enumerate() {
                if !at.routed {
                    continue;
                }
                let count = f64::from(self.repetitions[i]) * f64::from(repeats);
                let mut branches = [0.; 4];
                for (indices, rank_local) in [(&at.ops, false), (&at.rank_ops, true)] {
                    for &j in indices {
                        if cache[j].is_none() {
                            cache[j] = Some(a_work(
                                &self.a_pool.engine,
                                phase,
                                j,
                                shape,
                                seq,
                                prefix,
                                width,
                                rank_local,
                            )?);
                        }
                        let w = cache[j].as_ref().unwrap();
                        for (b, t) in branches.iter_mut().zip(w.branches) {
                            *b += t;
                        }
                        emit(
                            phase_ops(&self.a_pool.engine, phase)[j].name(),
                            w.result.clone(),
                            w.active,
                            count,
                        );
                    }
                }
                durations[i][0] = branches.into_iter().fold(0., f64::max);
                for &j in &ft.ops {
                    let mut r = cached(fcache, j, || {
                        query(
                            &self.f_pool.engine,
                            phase,
                            j,
                            shape.total().checked_mul(width).ok_or_else(|| {
                                AicError::InvalidEngineConfig("AFD batch overflow".into())
                            })?,
                            seq,
                            prefix,
                            tokens.total(),
                        )
                    })?;
                    durations[i][2] += r.latency_ms;
                    let active = r.latency_ms * f64::from(f_ranks);
                    r.energy_wms *= f64::from(f_ranks);
                    emit(
                        phase_ops(&self.f_pool.engine, phase)[j].name(),
                        r,
                        active,
                        count,
                    );
                }
                for (name, stage, r, active) in comms {
                    durations[i][*stage] += r.latency_ms;
                    emit(name, r.clone(), *active, count);
                }
            }
            let mut one_time = [0.; 2];
            for (slot, indices) in [&a.prefix, &a.suffix].into_iter().enumerate() {
                if slot == 0 && self.dense_prefix_layers > 0 {
                    continue;
                }
                let mut branches = [0.; 4];
                for &j in indices {
                    // Prefill logits consumes one hidden state per sequence.
                    let endpoint_seq = if slot == 1 && phase == AfdPhase::Prefill {
                        1
                    } else {
                        seq
                    };
                    let w = a_work(
                        &self.a_pool.engine,
                        phase,
                        j,
                        shape,
                        endpoint_seq,
                        prefix,
                        width,
                        false,
                    )?;
                    for (b, t) in branches.iter_mut().zip(w.branches) {
                        *b += t;
                    }
                    emit(
                        phase_ops(&self.a_pool.engine, phase)[j].name(),
                        w.result,
                        w.active,
                        f64::from(repeats),
                    );
                }
                one_time[slot] = branches.into_iter().fold(0., f64::max);
            }
            for &i in &ap.layers {
                for (sum, d) in sums.iter_mut().zip(durations[i]) {
                    *sum += d * f64::from(repeats);
                }
            }
            endpoints += (one_time[0] + one_time[1]) * f64::from(repeats);
            measurements.push((durations, one_time));
        }
        let pass = |factor| {
            cfg.topology
                .pipeline_layout
                .schedule(
                    &ap.layers[self.dense_prefix_layers..],
                    batches.count,
                    |micro, template| measurements[batches.run(micro)].0.get(template).copied(),
                    |micro| measurements[batches.run(micro)].1[0],
                    |micro| measurements[batches.run(micro)].1[1],
                    factor,
                    |_, _, _, _, _, _| {},
                )
                .map(|t| t + full_prefix_ms)
        };
        let t_step_ms = pass(1.)?;
        let billed_step_ms = if cfg.topology.comm_overhead_factor == 1. {
            t_step_ms
        } else {
            pass(cfg.topology.comm_overhead_factor)?
        };
        let denom = f64::from(batches.count) * f64::from(cfg.num_layers);
        Ok(AfdPhaseMetrics {
            layer_times: AfdLayerTimes {
                attention_ms: (sums[0] + sums[4] + full_prefix_ms + endpoints) / denom,
                ffn_ms: sums[2] / denom,
                a_to_f_ms: sums[1] / denom,
                f_to_a_ms: sums[3] / denom,
                num_layers: cfg.num_layers,
            },
            t_step_ms,
            billed_step_ms,
        })
    }

    pub(super) fn measure_phase(
        &self,
        phase: AfdPhase,
        isl: u32,
        osl: u32,
        prefix: u32,
        mut visit: impl FnMut(&str, &PerformanceResult),
    ) -> Result<AfdPhaseMetrics, AicError> {
        if isl == 0 || osl == 0 || prefix > isl {
            return Err(AicError::InvalidEngineConfig(
                "AFD isl/osl must be positive and prefix <= isl".into(),
            ));
        }
        let m = self.config.topology.num_microbatches;
        let batch = self
            .config
            .topology
            .a_pool
            .batch_size_per_a_replica
            .div_ceil(m);
        if phase == AfdPhase::Prefill {
            return self.measure_plan(
                phase,
                batch,
                self.config.topology.a_pool.batch_size_per_a_replica,
                isl.saturating_sub(prefix).max(1),
                prefix,
                m,
                None,
                visit,
            );
        }
        let plan = self.a_pool.engine.afd_plan().expect("validated A plan");
        let pool_batch =
            self.config.topology.a_pool.batch_size_per_a_replica * a_replica_count(&self.config);
        let batches = Microbatches::new(pool_batch, a_replica_count(&self.config), m);
        let communication = batches.runs[..batches.len]
            .iter()
            .map(|(shape, _)| {
                self.plan_communication(
                    phase,
                    shape.scaled(self.config.nextn + 1)?,
                    self.config.topology.a_pool.a_tp_size,
                    plan.shared_inter_size > 0,
                )
            })
            .collect::<Result<Vec<_>, AicError>>()?;
        let mut average = PhaseAverage::default();
        let repeats = f64::from(osl.saturating_sub(1).max(1));
        for_each_decode_sample(isl, osl, |s, repeat| {
            let sample = self.measure_pool_plan(
                phase,
                pool_batch,
                s,
                prefix,
                m,
                Some(&communication),
                |name, r| {
                    visit(name, &r.clone().scaled(f64::from(repeat) / repeats));
                },
            )?;
            average.push(&sample, repeat);
            Ok(())
        })?;
        Ok(average.finish(self.config.num_layers))
    }
}
