// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction-time table availability. Shape, slice and interpolation-domain
//! coverage remains a query-time check; no synthetic workload is queried here.

use std::collections::HashMap;

use crate::AicError;
use crate::common::enums::{DatabaseMode, GemmQuantMode, MoeQuantMode, TransferKind};
use crate::config::PerfSource;
use crate::operators::Op;
use crate::perf_database::moe::MoeKernel;
use crate::perf_database::{PerfDatabase, kernel_source_ok, parquet_loader::PerfReader};

pub(super) fn validate<'a>(
    db: &PerfDatabase,
    ops: impl Iterator<Item = &'a Op>,
) -> Result<(), AicError> {
    let mut check = Availability {
        db,
        tables: HashMap::new(),
    };
    for op in ops {
        check.op(op)?;
    }
    Ok(())
}

struct Availability<'a> {
    db: &'a PerfDatabase,
    tables: HashMap<String, bool>,
}

impl Availability<'_> {
    fn has(&mut self, name: &str) -> Result<bool, AicError> {
        if let Some(found) = self.tables.get(name) {
            return Ok(*found);
        }
        if name == "moe_perf.parquet" {
            let found = match self.db.moe.available_quants(MoeKernel::Standard) {
                Ok(quants) => !quants.is_empty(),
                Err(error) if error.is_missing_perf_data() => false,
                Err(error) => return Err(error),
            };
            self.tables.insert(name.to_owned(), found);
            return Ok(found);
        }
        let sources = match name {
            "nccl_perf.parquet" => self
                .db
                .communication
                .nccl_root()
                .map(|root| vec![PerfSource(root.join(name), None)])
                .unwrap_or_default(),
            "oneccl_perf.parquet" => self
                .db
                .communication
                .oneccl_root()
                .map(|root| vec![PerfSource(root.join(name), None)])
                .unwrap_or_default(),
            "dsv4_megamoe_module_perf.parquet" => {
                vec![PerfSource(
                    self.db.dsv4_megamoe.primary_path().to_owned(),
                    None,
                )]
            }
            _ => self
                .db
                .source_resolver
                .sources_for(name, &self.db.data_root)?,
        };
        let mut found = false;
        for source in sources {
            if !source.path().is_file() {
                continue;
            }
            let reader = PerfReader::open(source.path())?;
            let kernel = reader.col_optional("kernel_source");
            for row in reader.rows()? {
                if kernel_source_ok(source.kernel_sources(), kernel, &row?)? {
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        self.tables.insert(name.to_owned(), found);
        Ok(found)
    }

    fn any(&mut self, names: &[&str]) -> Result<(), AicError> {
        for name in names {
            if self.has(name)? {
                return Ok(());
            }
        }
        Err(AicError::PerfDatabase(format!(
            "required op-level data unavailable for {:?}: no rows in {} at {}",
            self.db.database_mode,
            names.join(" or "),
            self.db.data_root.display(),
        )))
    }

    fn op(&mut self, op: &Op) -> Result<(), AicError> {
        use Op::*;
        // CSA context parallelism consumes both sparse correction tables even
        // when its base module uses the analytic SOL. HCA never consumes them.
        if let Dsv4Context(module) = op
            && module.cp_size > 1
            && module.attn_kind == crate::perf_database::dsv4::AttnKind::Csa
        {
            self.any(&["dsv4_paged_mqa_logits_module_perf.parquet"])?;
            self.any(&["dsv4_csa_topk_calib_perf.parquet"])?;
        }
        let sol = matches!(
            self.db.database_mode,
            DatabaseMode::Sol | DatabaseMode::SolFull
        );
        match op {
            Dsv41Stage(stage) => {
                for child in &stage.children {
                    self.op(child)?;
                }
                return Ok(());
            }
            Overlap(group) => {
                for child in group.group_a.iter().chain(&group.group_b) {
                    self.op(child)?;
                }
                return Ok(());
            }
            Fallback(group) => {
                let primary = if self.db.database_mode == DatabaseMode::Hybrid {
                    validate(&self.db.silicon_view(), [&*group.primary].into_iter())
                } else {
                    self.op(&group.primary)
                };
                match primary {
                    Ok(()) => return Ok(()),
                    Err(error)
                        if error.is_missing_perf_data() || matches!(error, AicError::Io { .. }) =>
                    {
                        for child in &group.fallback {
                            self.op(child)?;
                        }
                        return Ok(());
                    }
                    Err(error) => return Err(error),
                }
            }
            TokenScale(scale) => return self.op(&scale.op),
            FpmForward(fpm) => {
                self.db
                    .fpm_forward
                    .select_cell(&fpm.match_identity, &fpm.model_path)?;
                return Ok(());
            }
            // These families have no analytic/empirical implementation.
            MoeAllToAll(_) | MoeExpertCompute(_) | Dsv4MegaMoe(_)
                if !matches!(
                    self.db.database_mode,
                    DatabaseMode::Silicon | DatabaseMode::Hybrid
                ) =>
            {
                return Err(AicError::UnsupportedModel(format!(
                    "{} requires silicon data; {:?} is unsupported",
                    op.name(),
                    self.db.database_mode,
                )));
            }
            _ if sol => return Ok(()),
            _ => {}
        }
        match op {
            Gemm(gemm) => {
                self.any(&["gemm_perf.parquet"])?;
                if gemm.quant_mode == GemmQuantMode::Fp8Static {
                    self.any(&["computescale_perf.parquet"])?;
                    if gemm.low_precision_input {
                        self.any(&["scale_matrix_perf.parquet"])?;
                    }
                }
                Ok(())
            }
            ContextAttention(_) => self.any(&["context_attention_perf.parquet"]),
            GenerationAttention(_) => self.any(&["generation_attention_perf.parquet"]),
            EncoderAttention(_) => self.any(&["encoder_attention_perf.parquet"]),
            ContextMla(_) => self.any(&["context_mla_perf.parquet"]),
            GenerationMla(_) => self.any(&["generation_mla_perf.parquet"]),
            MlaModuleContext(_) => self.any(&["mla_context_module_perf.parquet"]),
            MlaModuleGeneration(_) => self.any(&["mla_generation_module_perf.parquet"]),
            MlaBmm(_) => self.any(&["mla_bmm_perf.parquet"]),
            Moe(moe) => {
                let found = match moe.moe_kernel_source.as_deref() {
                    Some(source) => !self
                        .db
                        .moe
                        .available_quants_for_kernel_source(source)?
                        .is_empty(),
                    None => {
                        self.has("moe_perf.parquet")?
                            || (moe.is_gated
                                && moe.quant_mode == MoeQuantMode::Nvfp4
                                && self.db.moe.low_latency_available()?)
                    }
                };
                if found {
                    Ok(())
                } else {
                    Err(AicError::PerfDatabase(format!(
                        "required MoE data unavailable for kernel_source={:?} at {}",
                        moe.moe_kernel_source,
                        self.db.data_root.display()
                    )))
                }
            }
            // State-space kernels explicitly fall back to their analytic SOL
            // on missing tables in every database mode.
            Mamba2(_) | Gdn(_) | Kda(_) => Ok(()),
            Mhc(_) => self.any(&["mhc_module_perf.parquet"]),
            WideEpContextMla(_) => self.any(&["wideep_context_mla_perf.parquet"]),
            WideEpGenerationMla(_) => self.any(&["wideep_generation_mla_perf.parquet"]),
            DsaContext(module) => {
                self.any(&["dsa_context_module_perf.parquet"])?;
                if module.cp_size > 1 {
                    let prefix =
                        crate::perf_database::dsa::dsa_sparse_file_prefix(&module.architecture);
                    self.any(&[&format!("{prefix}_mqa_logits_module_perf.parquet")])?;
                    self.any(&[&format!("{prefix}_topk_module_perf.parquet")])?;
                }
                Ok(())
            }
            DsaGeneration(_) => self.any(&["dsa_generation_module_perf.parquet"]),
            MsaContext(_) | MsaGeneration(_) => {
                let phase = if matches!(op, MsaContext(_)) {
                    "context"
                } else {
                    "generation"
                };
                let own = format!("msa_{phase}_module_perf.parquet");
                if self.db.database_mode != DatabaseMode::Empirical && self.has(&own)? {
                    return Ok(());
                }
                if self.db.database_mode != DatabaseMode::Silicon
                    && self.db.transfer_policy.contains(TransferKind::XOp)
                {
                    return self.any(&[&format!("dsa_{phase}_module_perf.parquet")]);
                }
                self.any(&[&own])
            }
            Dsv4Context(module) | Dsv4Generation(module) => {
                use crate::perf_database::dsv4::AttnKind;
                let kind = match module.attn_kind {
                    AttnKind::Csa => "csa",
                    AttnKind::Hca => "hca",
                };
                let phase = if matches!(op, Dsv4Context(_)) {
                    "context"
                } else {
                    "generation"
                };
                self.any(&[&format!("dsv4_{kind}_{phase}_module_perf.parquet")])
            }
            Dsv4MegaMoe(_) => self.any(&["dsv4_megamoe_module_perf.parquet"]),
            MoeAllToAll(_) => self.any(&[
                "moe_a2a_perf.parquet",
                "wideep_deepep_ll_perf.parquet",
                "wideep_deepep_normal_perf.parquet",
                "trtllm_alltoall_perf.parquet",
            ]),
            MoeExpertCompute(_) => self.any(&[
                "moe_expert_compute_perf.parquet",
                "wideep_context_moe_perf.parquet",
                "wideep_generation_moe_perf.parquet",
                "wideep_moe_perf.parquet",
                "moe_perf.parquet",
            ]),
            CustomAllReduce(comm) if comm.tp_size > 1 => {
                if self.db.system_spec.node.num_gpus_per_node == 72
                    && comm.tp_size > 4
                    && matches!(
                        self.db.database_mode,
                        DatabaseMode::Silicon | DatabaseMode::Hybrid
                    )
                {
                    self.any(&["nccl_perf.parquet", "oneccl_perf.parquet"])
                } else {
                    self.any(&["custom_allreduce_perf.parquet"])
                }
            }
            Nccl(comm) if comm.num_gpus > 1 => {
                self.any(&["nccl_perf.parquet", "oneccl_perf.parquet"])
            }
            // Dispatch chooses its comm legs from topology at query time. A
            // single-rank graph needs no communication table.
            MoeDispatch(dispatch) if dispatch.moe_tp_size * dispatch.moe_ep_size > 1 => {
                self.any(&[
                    "custom_allreduce_perf.parquet",
                    "nccl_perf.parquet",
                    "oneccl_perf.parquet",
                    "trtllm_alltoall_perf.parquet",
                ])
            }
            Vision(_) => {
                self.any(&["gemm_perf.parquet"])?;
                self.any(&["encoder_attention_perf.parquet"])
            }
            Embedding(_) | Elementwise(_) | P2P(_) | CustomAllReduce(_) | Nccl(_)
            | MoeDispatch(_) => Ok(()),
            Dsv41Attention(_) | Dsv41Mhc(_) | Dsv41Engram(_) | Dsv41Linear(_) => {
                match self.db.database_mode {
                    DatabaseMode::Silicon => self.any(&["dsv41_module_perf.parquet"]),
                    DatabaseMode::Empirical => Err(AicError::EmpiricalNotImplemented(format!(
                        "DeepSeek-V4.1 {} has no empirical anchor",
                        op.name()
                    ))),
                    _ => Ok(()),
                }
            }
            Overlap(_) | Fallback(_) | TokenScale(_) | FpmForward(_) | Dsv41Stage(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use crate::config::PerfDbSources;
    use crate::operators::dsv4::{Dsv4MegaMoeOp, Dsv4ModuleOp};
    use crate::operators::op::{FallbackOp, OverlapOp};
    use crate::operators::{EmbeddingOp, GemmOp};
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
    use crate::perfmodel::engine::{Engine, spec::EngineSpec};
    use std::sync::Arc;

    fn systems() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aisimulate_core/systems/b200_sxm.yaml");
        std::fs::copy(spec, root.path().join("b200_sxm.yaml")).unwrap();
        std::fs::create_dir_all(root.path().join("data/b200_sxm/vllm/0.24.0")).unwrap();
        root
    }

    fn engine_readiness(db: PerfDatabase, op: Op) -> Result<(), AicError> {
        let config = serde_json::from_value(serde_json::json!({
            "schema_version": crate::ENGINE_CONFIG_SCHEMA_VERSION,
            "model_name": "readiness-fixture",
            "system_name": "b200_sxm", "backend": "vllm", "backend_version": "0.24.0",
            "tp_size": 1, "pp_size": 1, "database_mode": db.database_mode,
        }))
        .unwrap();
        let (context, generation) = if matches!(op, Op::Dsv4Generation(_)) {
            (vec![], vec![op])
        } else {
            (vec![op], vec![])
        };
        Engine::build(EngineSpec::new(config, context, generation), Arc::new(db))?
            .validate_forward_pass_readiness()
    }

    fn dsv4_op(kind: &str, context: bool, cp: u32) -> Op {
        let module: Dsv4ModuleOp = serde_json::from_value(serde_json::json!({
            "name": "attention", "scale_factor": 1.0, "attn_kind": kind,
            "num_heads": 64, "native_heads": 64, "tp_size": 1, "cp_size": cp,
            "kv_cache_dtype": "bfloat16", "fmha_quant_mode": "bfloat16",
            "gemm_quant_mode": "bfloat16", "architecture": "DeepseekV4ForCausalLM",
        }))
        .unwrap();
        if context {
            Op::Dsv4Context(module)
        } else {
            Op::Dsv4Generation(module)
        }
    }

    fn table(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_parquet(path, &[Col::Str("kernel_source", vec!["fixture"])]);
    }

    #[test]
    fn dsv41_stage_requires_measured_module_for_silicon() {
        use crate::operators::dsv41::{Dsv41LinearOp, Dsv41StageOp};
        let root = systems();
        let stage = Op::Dsv41Stage(Dsv41StageOp {
            name: "stage".into(),
            is_context: true,
            decoder_replay: true,
            bounded: true,
            window_size: 128,
            children: vec![Op::Dsv41Linear(Dsv41LinearOp {
                name: "projection".into(),
                n: 16,
                k: 16,
                quant_mode: GemmQuantMode::Fp8Block,
            })],
        });
        for mode in [
            DatabaseMode::Silicon,
            DatabaseMode::Empirical,
            DatabaseMode::Hybrid,
            DatabaseMode::Sol,
            DatabaseMode::SolFull,
        ] {
            let db = PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0")
                .unwrap()
                .with_mode(mode, TransferPolicy::ALL);
            assert_eq!(
                validate(&db, [&stage].into_iter()).is_ok(),
                !matches!(mode, DatabaseMode::Silicon | DatabaseMode::Empirical)
            );
        }
        table(
            &root
                .path()
                .join("data/b200_sxm/vllm/0.24.0/dsv41_module_perf.parquet"),
        );
        let db = PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0")
            .unwrap()
            .with_mode(DatabaseMode::Silicon, TransferPolicy::ALL);
        assert!(validate(&db, [&stage].into_iter()).is_ok());
    }

    #[test]
    fn dsv4_requires_its_phase_module_not_an_auxiliary_table() {
        for kind in ["Csa", "Hca"] {
            for context in [true, false] {
                let root = systems();
                let data = root.path().join("data/b200_sxm/vllm/0.24.0");
                table(&data.join("dsv4_paged_mqa_logits_module_perf.parquet"));
                let load =
                    || PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0").unwrap();
                let phase = if context { "context" } else { "generation" };
                let required = format!("dsv4_{}_{phase}_module_perf.parquet", kind.to_lowercase());
                let error = engine_readiness(load(), dsv4_op(kind, context, 1)).unwrap_err();
                assert!(error.to_string().contains(&required), "{error}");
                table(&data.join(required));
                engine_readiness(load(), dsv4_op(kind, context, 1)).unwrap();
            }
        }
    }

    #[test]
    fn csa_context_parallel_requires_both_sparse_tables() {
        let mqa = "dsv4_paged_mqa_logits_module_perf.parquet";
        let topk = "dsv4_csa_topk_calib_perf.parquet";
        for mode in [
            DatabaseMode::Silicon,
            DatabaseMode::Hybrid,
            DatabaseMode::Empirical,
            DatabaseMode::Sol,
        ] {
            for present in [vec![], vec![mqa], vec![topk], vec![mqa, topk]] {
                let root = systems();
                let data = root.path().join("data/b200_sxm/vllm/0.24.0");
                if mode != DatabaseMode::Sol {
                    table(&data.join("dsv4_csa_context_module_perf.parquet"));
                    table(&data.join("dsv4_hca_context_module_perf.parquet"));
                }
                for name in &present {
                    table(&data.join(name));
                }
                let load = || {
                    PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0")
                        .unwrap()
                        .with_mode(mode, TransferPolicy::ALL)
                };
                assert_eq!(
                    engine_readiness(load(), dsv4_op("Csa", true, 2)).is_ok(),
                    present.len() == 2,
                    "{mode:?}, {present:?}"
                );
                engine_readiness(load(), dsv4_op("Hca", true, 2)).unwrap();
            }
        }
    }

    #[test]
    fn megamoe_readiness_uses_only_the_resolved_primary() {
        let root = systems();
        let name = "dsv4_megamoe_module_perf.parquet";
        let family = root.path().join("data/b200_sxm/moe/vllm/0.24.0").join(name);
        table(&family);
        let op = || {
            Op::Dsv4MegaMoe(Dsv4MegaMoeOp {
                name: "megamoe".into(),
                scale_factor: 1.0,
                hidden_size: 4096,
                inter_size: 2048,
                topk: 8,
                num_experts: 256,
                moe_tp_size: 1,
                moe_ep_size: 8,
                quant_mode: crate::common::enums::MoeQuantMode::Fp8,
                workload_distribution: "balanced".into(),
                is_context: true,
                source_policy: "primary".into(),
                pre_dispatch: "none".into(),
                num_fused_shared_experts: 0,
                kernel_source: "fixture".into(),
                kernel_dtype: "fp8".into(),
            })
        };
        let db = PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0").unwrap();
        assert_eq!(db.dsv4_megamoe.primary_path(), family);
        engine_readiness(db, op()).unwrap();

        let custom = root.path().join("custom.parquet");
        table(&custom);
        for primary in [&custom, &root.path().join("missing.parquet")] {
            // MegaMoE ignores row filters and additional sources at query time.
            let sources = PerfDbSources::from([(
                name.into(),
                vec![
                    PerfSource(primary.clone(), Some(vec!["excluded".into()])),
                    PerfSource(family.clone(), None),
                ],
            )]);
            let db = PerfDatabase::load_with_sources(
                root.path(),
                "b200_sxm",
                "vllm",
                "0.24.0",
                &sources,
            )
            .unwrap();
            assert_eq!(db.dsv4_megamoe.primary_path(), primary);
            assert_eq!(engine_readiness(db, op()).is_ok(), primary == &custom);
        }
    }

    #[test]
    fn absent_tables_respect_modes_and_composite_fallbacks() {
        let root = systems();
        let gemm = Op::Gemm(GemmOp::new("gemm", 16, 16, GemmQuantMode::Bfloat16));
        for mode in [
            DatabaseMode::Silicon,
            DatabaseMode::Hybrid,
            DatabaseMode::Empirical,
            DatabaseMode::Sol,
        ] {
            let db = PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0")
                .unwrap()
                .with_mode(mode, TransferPolicy::ALL);
            assert_eq!(
                validate(&db, [&gemm].into_iter()).is_ok(),
                mode == DatabaseMode::Sol
            );
            let fallback = Op::Fallback(FallbackOp::new(
                "fallback",
                gemm.clone(),
                vec![Op::Embedding(EmbeddingOp::new(
                    "embedding",
                    16,
                    16,
                    GemmQuantMode::Bfloat16,
                ))],
            ));
            validate(&db, [&fallback].into_iter()).unwrap();
            let overlap = Op::Overlap(OverlapOp::new(
                "overlap",
                vec![fallback],
                vec![gemm.clone()],
            ));
            assert_eq!(
                validate(&db, [&overlap].into_iter()).is_ok(),
                mode == DatabaseMode::Sol
            );
        }
    }

    #[test]
    fn table_presence_honors_source_filters_and_empty_tables() {
        let root = systems();
        let data = root
            .path()
            .join("data/b200_sxm/vllm/0.24.0/gemm_perf.parquet");
        let gemm = Op::Gemm(GemmOp::new("gemm", 16, 16, GemmQuantMode::Bfloat16));
        for rows in [vec![], vec!["allowed"]] {
            write_parquet(&data, &[Col::Str("kernel_source", rows.clone())]);
            for filter in ["allowed", "excluded"] {
                let sources = PerfDbSources::from([(
                    "gemm_perf.parquet".into(),
                    vec![PerfSource(data.clone(), Some(vec![filter.into()]))],
                )]);
                let db = PerfDatabase::load_with_sources(
                    root.path(),
                    "b200_sxm",
                    "vllm",
                    "0.24.0",
                    &sources,
                )
                .unwrap();
                assert_eq!(
                    validate(&db, [&gemm].into_iter()).is_ok(),
                    !rows.is_empty() && filter == "allowed"
                );
            }
        }
    }

    #[test]
    fn moe_readiness_requires_eligible_or_exact_source_rows() {
        use crate::common::enums::MoeQuantMode;
        use crate::operators::MoeOp;

        let root = systems();
        let data = root
            .path()
            .join("data/b200_sxm/vllm/0.24.0/moe_perf.parquet");
        for eligibility in [None, Some(true), Some(false)] {
            let mut columns = vec![
                Col::Str("moe_dtype", vec!["fp8_block"]),
                Col::I64("num_tokens", vec![32]),
                Col::I64("hidden_size", vec![8192]),
                Col::I64("inter_size", vec![2048]),
                Col::I64("topk", vec![8]),
                Col::I64("num_experts", vec![256]),
                Col::I64("moe_tp_size", vec![1]),
                Col::I64("moe_ep_size", vec![1]),
                Col::Str("distribution", vec!["uniform"]),
                Col::Str("kernel_source", vec!["exact"]),
                Col::F64("latency", vec![0.25]),
            ];
            if let Some(eligible) = eligibility {
                columns.push(Col::Bool("default_eligible", vec![eligible]));
            }
            write_parquet(&data, &columns);
            for source in [None, Some("exact")] {
                let db = PerfDatabase::load(root.path(), "b200_sxm", "vllm", "0.24.0").unwrap();
                let mut op = MoeOp::new(
                    "moe",
                    8192,
                    2048,
                    8,
                    256,
                    1,
                    1,
                    MoeQuantMode::Fp8Block,
                    "uniform",
                );
                op.moe_kernel_source = source.map(str::to_owned);
                let expected = source.is_some() || eligibility != Some(false);
                assert_eq!(engine_readiness(db, Op::Moe(op)).is_ok(), expected);
            }
        }
    }
}
