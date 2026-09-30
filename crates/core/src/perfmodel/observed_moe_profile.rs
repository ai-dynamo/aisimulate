// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity and scope of the qualified observed decode MoE profiles.
//!
//! These are provenance identities, never latency constants. The collector's
//! authored closure binds its checked-in producer/capture identity; the existing
//! collection events bind each fixed corpus to its complete physical node set.

use std::path::Path;

use serde_yaml::Value;

use crate::common::enums::DatabaseMode;
use crate::common::error::AicError;
use crate::perfmodel::{BackendKind, DataType, EngineConfig};

const DISTRIBUTION: &str = "observed_glm52_nvfp4_decode_1ab2c747975e_v1";
pub(crate) const VERSION: &str = "0.5.18+nvinternal.rubin.0.8full.66997102";
pub(crate) const KERNEL: &str = "sglang_flashinfer_trtllm_moe";
const IMAGE: &str = "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci";
const IMAGE_DIGEST: &str =
    "sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab";
const COMPOSITE_DISTRIBUTION: &str = "observed_glm52_nvfp4_decode_composite_v2";

/// Internal selection only: the existing distribution string remains the wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeProfile {
    V1,
    V2,
}

impl DecodeProfile {
    pub(crate) fn from_distribution(distribution: &str) -> Result<Self, AicError> {
        match distribution {
            DISTRIBUTION => Ok(Self::V1),
            COMPOSITE_DISTRIBUTION => Ok(Self::V2),
            _ => Err(error(distribution, "unsupported observed decode profile")),
        }
    }

    pub(crate) fn distribution(self) -> &'static str {
        match self {
            Self::V1 => DISTRIBUTION,
            Self::V2 => COMPOSITE_DISTRIBUTION,
        }
    }

    pub(crate) fn cache_index(self) -> usize {
        match self {
            Self::V1 => 0,
            Self::V2 => 1,
        }
    }

    pub(crate) fn physical_nodes(self) -> &'static [u32] {
        match self {
            Self::V1 => &[1, 8, 32],
            Self::V2 => &[1, 4, 8, 32],
        }
    }

    fn collection_identity(self) -> (&'static str, &'static str, &'static str, u64) {
        match self {
            Self::V1 => (
                "collector.sglang_rubin.publish_observed_moe",
                "sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332",
                "sha256:dd11934097a3bc664e82c7081af5ceac3161666eb689e736b47765d7ea955cde",
                864,
            ),
            Self::V2 => (
                "collector.sglang_rubin.publish_observed_moe_v2",
                "sha256:bbe3ebf2d450053f524d38b0a8ef97f0e55df000b6c6f61430b0207afc622eaf",
                "sha256:8a1c8ca3136fcf2289ff49e7f07d1d95e3d23dea9301db0eafeb07436d529519",
                1152,
            ),
        }
    }

    pub(crate) fn error(self, detail: impl std::fmt::Display) -> AicError {
        error(self.distribution(), detail)
    }
}

pub(crate) fn error(distribution: &str, detail: impl std::fmt::Display) -> AicError {
    AicError::DecodeMoeProfile(format!("{distribution}: {detail}"))
}

/// The native graph ceiling buckets are distinct from the admitted logical sizes.
fn native_decode_bucket(logical: u32) -> Option<u32> {
    if logical == 0 {
        return None;
    }
    [1, 2, 4, 8, 12, 16, 24, 32]
        .into_iter()
        .find(|&physical| physical >= logical)
}

pub(crate) fn v2_physical_node(logical: u32) -> Result<u32, AicError> {
    // The composite corpus covers these logical occupancies only. In particular,
    // physical N4 does not admit L4; its evidence is L3 at ISL32768. This mapping
    // applies to routed MoE only, never to whole-model token counts.
    if ![1, 3, 8, 29, 31, 32].contains(&logical) {
        return Err(DecodeProfile::V2.error(format!(
            "logical tokens {logical} are not in the qualified set 1/3/8/29/31/32"
        )));
    }
    native_decode_bucket(logical)
        .ok_or_else(|| DecodeProfile::V2.error("no native physical bucket"))
}

pub(crate) fn validate_runtime(
    selected: DecodeProfile,
    system: &str,
    backend: &str,
    version: &str,
    mode: DatabaseMode,
) -> Result<(), AicError> {
    if (system, backend, version) != ("vr200_hecate", "sglang", VERSION)
        || mode != DatabaseMode::Silicon
    {
        return Err(selected.error("requires the measured VR200 SGLang runtime and SILICON mode"));
    }
    Ok(())
}

pub(crate) fn validate_engine(
    selected: DecodeProfile,
    config: &EngineConfig,
) -> Result<(), AicError> {
    validate_runtime(
        selected,
        &config.system_name,
        config.backend.as_str(),
        config.backend_version.as_deref().unwrap_or(""),
        config.database_mode,
    )?;
    if config.moe_kernel_source.is_some() {
        return Err(selected.error("moe_kernel_source cannot override an observed decode profile"));
    }
    let parallel = &config.parallel;
    if config.model_name != "nvidia/GLM-5.2-NVFP4"
        || config.backend != BackendKind::Sglang
        || config.quantization.moe_dtype != Some(DataType::Nvfp4)
        || config.quantization.weight_dtype != Some(DataType::Bfloat16)
        || config.quantization.activation_dtype != Some(DataType::Bfloat16)
        || config.quantization.kv_cache_dtype != Some(DataType::Fp8)
        || parallel.tp_size != 4
        || parallel.moe_tp_size != Some(4)
        || parallel.moe_ep_size != Some(1)
        || parallel.pp_size != 1
        || parallel.attention_dp_size.unwrap_or(1) != 1
        || parallel.cp_size.unwrap_or(1) != 1
        || config
            .speculative
            .as_ref()
            .and_then(|spec| spec.nextn)
            .unwrap_or(0)
            != 0
        || config.forward_model.as_deref().unwrap_or("op_level") != "op_level"
    {
        return Err(
            selected.error("unsupported model, parallelism, quantization or speculative/FPM mode")
        );
    }
    Ok(())
}

// Communication precision lives on the serialized operators, not EngineConfig.
pub(crate) fn validate_communication(
    selected: DecodeProfile,
    ops: &[crate::operators::Op],
) -> Result<(), AicError> {
    use crate::common::enums::CommQuantMode;
    use crate::operators::Op;
    for op in ops {
        match op {
            Op::CustomAllReduce(op) if op.quant != CommQuantMode::Half => {
                return Err(selected.error("requires half communication"));
            }
            Op::Nccl(op) if op.dtype != CommQuantMode::Half => {
                return Err(selected.error("requires half communication"));
            }
            Op::MoeDispatch(op) if op.comm_quant != CommQuantMode::Half => {
                return Err(selected.error("requires half communication"));
            }
            Op::Overlap(op) => {
                validate_communication(selected, &op.group_a)?;
                validate_communication(selected, &op.group_b)?;
            }
            Op::Fallback(op) => {
                validate_communication(selected, std::slice::from_ref(&op.primary))?;
                validate_communication(selected, &op.fallback)?;
            }
            Op::TokenScale(op) => validate_communication(selected, std::slice::from_ref(&op.op))?,
            Op::Dsv41Stage(op) => validate_communication(selected, &op.children)?,
            _ => {}
        }
    }
    Ok(())
}

fn runtime_matches(runtime: &Value) -> bool {
    let Some(mapping) = runtime.as_mapping() else {
        return false;
    };
    let allowed = [
        "framework",
        "version",
        "image",
        "image_variant",
        "image_digest",
    ];
    mapping
        .keys()
        .all(|key| key.as_str().is_some_and(|key| allowed.contains(&key)))
        && [
            ("framework", "sglang"),
            ("version", VERSION),
            ("image", IMAGE),
            ("image_digest", IMAGE_DIGEST),
        ]
        .iter()
        .all(|(key, expected)| runtime.get(*key).and_then(Value::as_str) == Some(*expected))
}

pub(crate) fn validate_metadata(
    selected: DecodeProfile,
    table_path: &Path,
) -> Result<(), AicError> {
    let path = table_path.with_file_name("collection_meta.yaml");
    let data =
        std::fs::read(&path).map_err(|err| selected.error(format!("{}: {err}", path.display())))?;
    let meta: Value = serde_yaml::from_slice(&data)
        .map_err(|err| selected.error(format!("{}: {err}", path.display())))?;
    if meta.get("schema_version").and_then(Value::as_u64) != Some(2)
        || !runtime_matches(&meta["runtime"])
    {
        return Err(selected.error(format!(
            "{}: expected schema2 measured runtime",
            path.display()
        )));
    }
    let table = &meta["tables"]["moe_perf"];
    let events = table
        .get("collections")
        .and_then(Value::as_sequence)
        .ok_or_else(|| {
            selected.error(format!("{}: missing MoE collection events", path.display()))
        })?;
    let (collector_ref, collector_hash, case_plan_hash, campaign_rows) =
        selected.collection_identity();
    let node_count = selected.physical_nodes().len() as u64;
    let matching: Vec<_> = events
        .iter()
        .filter(|event| {
            event["collector_ref"].as_str() == Some(collector_ref)
                && event["collector_hash"].as_str() == Some(collector_hash)
                && event["case_plan_hash"].as_str() == Some(case_plan_hash)
        })
        .collect();
    if matching.len() != 1 {
        return Err(selected.error(format!(
            "{}: exact approved source/case-plan event missing or ambiguous",
            path.display()
        )));
    }
    let event = matching[0];
    if event["rows"].as_u64() != Some(node_count)
        || event["source_campaign_rows"].as_u64() != Some(campaign_rows)
        || event["status"].as_str() != Some("complete")
        || event["source_campaign_status"].as_str() != Some("complete")
        || event["collected_at"]
            .as_str()
            .is_none_or(|text| text.trim().is_empty())
        || !runtime_matches(&event["runtime"])
        || table["rows"]
            .as_u64()
            .is_none_or(|count| count < node_count)
        || table["status"].as_str() != Some("complete")
    {
        return Err(selected.error(format!(
            "{}: incomplete or mismatched observed collection event",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_buckets_and_admitted_occupancies_are_separate() {
        for (logical, physical) in [
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (8, 8),
            (9, 12),
            (12, 12),
            (13, 16),
            (17, 24),
            (24, 24),
            (25, 32),
            (32, 32),
        ] {
            assert_eq!(native_decode_bucket(logical), Some(physical));
        }
        for logical in [0, 33, u32::MAX] {
            assert_eq!(native_decode_bucket(logical), None);
        }
        for (logical, physical) in [(1, 1), (3, 4), (8, 8), (29, 32), (31, 32), (32, 32)] {
            assert_eq!(v2_physical_node(logical).unwrap(), physical);
        }
        for logical in (0..=33).chain([u32::MAX]) {
            if ![1, 3, 8, 29, 31, 32].contains(&logical) {
                assert!(matches!(
                    v2_physical_node(logical),
                    Err(AicError::DecodeMoeProfile(_))
                ));
            }
        }
    }
}
