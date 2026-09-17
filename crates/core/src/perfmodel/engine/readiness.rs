// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Construction-time table availability. Shape, slice and interpolation-domain
//! coverage remains a query-time check; no synthetic workload is queried here.

use std::collections::HashMap;

use crate::AicError;
use crate::common::enums::{DatabaseMode, GemmQuantMode, TransferKind};
use crate::config::PerfSource;
use crate::operators::Op;
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
                vec![PerfSource(self.db.data_root.join(name), None)]
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
        let sol = matches!(
            self.db.database_mode,
            DatabaseMode::Sol | DatabaseMode::SolFull
        );
        match op {
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
            Moe(_) => self.any(&["moe_perf.parquet"]),
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
                self.any(&[
                    &format!("dsv4_{kind}_{phase}_module_perf.parquet"),
                    "dsv4_paged_mqa_logits_module_perf.parquet",
                ])
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
            Overlap(_) | Fallback(_) | TokenScale(_) | FpmForward(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use crate::config::PerfDbSources;
    use crate::operators::op::{FallbackOp, OverlapOp};
    use crate::operators::{EmbeddingOp, GemmOp};
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn systems() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/aisimulate/src/aiconfigurator_core/systems/b200_sxm.yaml");
        std::fs::copy(spec, root.path().join("b200_sxm.yaml")).unwrap();
        std::fs::create_dir_all(root.path().join("data/b200_sxm/vllm/0.24.0")).unwrap();
        root
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
}
