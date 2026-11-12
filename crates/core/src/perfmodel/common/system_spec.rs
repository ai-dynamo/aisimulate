// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified native port of heterogeneous AFD/fabric behavior; source revision and
// original paths: root THIRD_PARTY_NOTICES.md (AISimulate heterogeneous AFD).
// Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

//! Typed system hardware spec parsed from `python/aisimulate/src/aisimulate_core/systems/*.yaml`.
//!
//! Mirrors `aisimulate.sdk.system_spec.SystemSpec`. The Python type
//! subclasses `dict`; the Rust port uses a typed struct because every
//! downstream consumer here needs typed field access. The on-disk YAML schema
//! is the contract; struct layout follows the YAML shape one-to-one.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::common::enums::{ComputeDtype, QuantMapping};
use crate::common::error::AicError;

/// Hardware system spec deserialized from `systems/<system>.yaml`.
#[derive(Clone, Debug, Deserialize)]
pub struct SystemSpec {
    /// Path to per-system perf data, relative to the systems directory.
    pub data_dir: PathBuf,
    pub gpu: GpuSpec,
    pub node: NodeSpec,
    #[serde(default)]
    pub misc: MiscSpec,
}

/// GPU hardware spec.
#[derive(Clone, Debug, Deserialize)]
pub struct GpuSpec {
    /// HBM memory bandwidth, bytes/s.
    pub mem_bw: f64,
    /// Empirical scaling factor applied to `mem_bw`.
    #[serde(default = "default_mem_bw_scaling")]
    pub mem_bw_empirical_scaling_factor: f64,
    /// Constant latency added to memory-bound ops (seconds).
    #[serde(default)]
    pub mem_empirical_constant_latency: f64,
    /// HBM capacity in bytes.
    #[serde(default)]
    pub mem_capacity: Option<u64>,
    /// Peak scalar FP32 throughput (not tensor-core throughput).
    #[serde(default)]
    pub fp32_flops: Option<f64>,
    /// Peak TC-FLOPS at bf16.
    #[serde(default)]
    pub bfloat16_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at int8.
    #[serde(default)]
    pub int8_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at fp8.
    #[serde(default)]
    pub fp8_tc_flops: Option<f64>,
    /// Peak TC-FLOPS at fp4/nvfp4.
    #[serde(default)]
    pub fp4_tc_flops: Option<f64>,
    /// Per-GPU TDP, watts.
    #[serde(default)]
    pub power: Option<f64>,
    /// CUDA SM architecture version (e.g. 100 for Blackwell SM_100).
    #[serde(default)]
    pub sm_version: Option<u32>,
}

/// Strict tensor-core FLOPS resolver. Mirrors Python
/// `common.get_quant_tc_flops`: the quant mode's `compute_dtype` selects the
/// `<dtype>_tc_flops` spec entry; a missing OR non-positive entry is an error
/// (the platform either lacks hardware for that dtype or the YAML is
/// incomplete / holds a placeholder) — never a `bfloat16_tc_flops *
/// compute_factor` extrapolation, and never a zero divisor that would turn
/// SOL into inf and load-time clamps into silent data corruption.
pub(crate) fn quant_tc_flops(spec: &SystemSpec, mapping: QuantMapping) -> Result<f64, AicError> {
    let Some(dtype) = mapping.compute_dtype else {
        return Err(AicError::MissingSystemFlops(format!(
            "quant mode '{}' is memory-only and has no compute FLOPS",
            mapping.name
        )));
    };
    let value = match dtype {
        ComputeDtype::Bfloat16 => spec.gpu.bfloat16_tc_flops,
        ComputeDtype::Int8 => spec.gpu.int8_tc_flops,
        ComputeDtype::Fp8 => spec.gpu.fp8_tc_flops,
        ComputeDtype::Fp4 => spec.gpu.fp4_tc_flops,
    };
    value.filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| {
        AicError::MissingSystemFlops(format!(
            "quant mode '{}' needs '{}', which this system's YAML does not define (or defines \
             as a non-positive placeholder): either the platform does not support {:?} compute \
             and the quant mode cannot be modeled on it, or the system YAML is missing the \
             entry.",
            mapping.name,
            dtype.flops_key(),
            dtype
        ))
    })
}

/// Resolve storage-format requests to a supported execution lane. This is
/// hardware normalization, never a data-dependent precision fallback.
pub(crate) fn execution_quantization(
    spec: &SystemSpec,
    gemm: Option<crate::common::enums::GemmQuantMode>,
    moe: Option<crate::common::enums::MoeQuantMode>,
) -> Result<
    (
        Option<crate::common::enums::GemmQuantMode>,
        Option<crate::common::enums::MoeQuantMode>,
    ),
    AicError,
> {
    use crate::common::enums::{GemmQuantMode as G, MoeQuantMode as M};
    if spec.gpu.sm_version.is_none() && (gemm == Some(G::Nvfp4) || moe == Some(M::Nvfp4)) {
        return Err(AicError::InvalidEngineConfig(
            "nvfp4 execution mapping requires a known GPU SM architecture".into(),
        ));
    }
    let blackwell = spec.gpu.sm_version.is_some_and(|sm| sm >= 100);
    let gemm = gemm.map(|q| {
        if q == G::Nvfp4 && !blackwell {
            G::Nvfp4Wo
        } else {
            q
        }
    });
    let moe = moe.map(|q| {
        if q == M::Nvfp4 && !blackwell {
            M::Nvfp4Wo
        } else {
            q
        }
    });
    for mapping in gemm.map(G::mapping).into_iter().chain(moe.map(M::mapping)) {
        if mapping.compute_dtype == Some(ComputeDtype::Fp8)
            && spec.gpu.sm_version.is_some_and(|sm| sm < 89)
        {
            return Err(AicError::InvalidEngineConfig(format!(
                "GPU SM{:?} does not support FP8 execution for {}",
                spec.gpu.sm_version, mapping.name
            )));
        }
        if mapping.compute_dtype.is_some() {
            quant_tc_flops(spec, mapping).map_err(|error| {
                AicError::InvalidEngineConfig(format!(
                    "unsupported execution precision {}: {error}",
                    mapping.name
                ))
            })?;
        }
    }
    Ok((gemm, moe))
}

/// Node-level topology spec.
#[derive(Clone, Debug, Deserialize)]
pub struct NodeSpec {
    pub num_gpus_per_node: u32,
    /// Inter-node bandwidth, bytes/s per GPU, single direction.
    pub inter_node_bw: f64,
    /// Intra-node (NVLink) bandwidth, bytes/s per GPU, single direction.
    pub intra_node_bw: f64,
    /// PCIe bandwidth, bytes/s, single direction.
    #[serde(default)]
    pub pcie_bw: Option<f64>,
    /// Point-to-point latency (seconds).
    #[serde(default)]
    pub p2p_latency: f64,
    /// Rack-level fan-out (GB200/GB300). Optional.
    #[serde(default)]
    pub num_gpus_per_rack: Option<u32>,
    /// Inter-rack bandwidth, bytes/s. Optional.
    #[serde(default)]
    pub inter_rack_bw: Option<f64>,
    /// Cross-rack propagation latency in seconds; absent preserves p2p_latency.
    #[serde(default)]
    pub inter_rack_latency: Option<f64>,
}

/// Miscellaneous, mostly empirical, configuration.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct MiscSpec {
    /// NCCL memory overhead by per-rank count.
    #[serde(default)]
    pub nccl_mem: BTreeMap<u32, u64>,
    /// Other memory overhead in bytes.
    #[serde(default)]
    pub other_mem: Option<u64>,
    /// NCCL version string used during data collection. The corresponding
    /// `nccl_perf.parquet` lives under `<systems_root>/<data_dir>/nccl/
    /// <nccl_version>/` and is system-wide (NOT per backend/version),
    /// mirroring Python `sdk/operations/communication.py:294`.
    #[serde(default)]
    pub nccl_version: Option<String>,
    /// OneCCL version string used during data collection. Set on XPU
    /// systems (e.g. `b60`) where Intel oneCCL is the comm backend; the
    /// corresponding `oneccl_perf.parquet` lives under
    /// `<systems_root>/<data_dir>/oneccl/<oneccl_version>/`. Mirrors
    /// Python `sdk/operations/communication.py:301-303`.
    #[serde(default)]
    pub oneccl_version: Option<String>,
}

fn default_mem_bw_scaling() -> f64 {
    1.0
}

impl SystemSpec {
    /// Load a system YAML from disk.
    pub fn load(path: &Path) -> Result<Self, AicError> {
        let text = fs::read_to_string(path).map_err(|source| AicError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_yaml::from_str(&text).map_err(|source| AicError::Yaml {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Point-to-point bandwidth (bytes/s) for a given collective fan-out.
    ///
    /// Mirrors Python's `SystemSpec.get_p2p_bandwidth` three-tier selection:
    /// `intra_node_bw` within a node, `inter_node_bw` within a rack,
    /// `inter_rack_bw` (or `inter_node_bw` fallback) across racks.
    pub fn get_p2p_bandwidth(&self, num_gpus: u32) -> f64 {
        let node = &self.node;
        if num_gpus <= node.num_gpus_per_node {
            return node.intra_node_bw;
        }
        let per_rack = node.num_gpus_per_rack.unwrap_or(u32::MAX);
        if num_gpus <= per_rack {
            return node.inter_node_bw;
        }
        node.inter_rack_bw.unwrap_or(node.inter_node_bw)
    }

    /// Propagation latency for the chosen scale-up/scale-out domain.
    pub fn get_p2p_latency(&self, num_gpus: u32) -> f64 {
        let node = &self.node;
        if num_gpus <= node.num_gpus_per_rack.unwrap_or(u32::MAX) {
            return node.p2p_latency;
        }
        node.inter_rack_latency
            .unwrap_or(node.p2p_latency)
            .max(node.p2p_latency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_precision_has_only_explicit_hardware_mappings() {
        use crate::common::enums::{GemmQuantMode as G, MoeQuantMode as M};
        let mut hopper = SystemSpec::load(&systems_root().join("h200_sxm.yaml")).unwrap();
        assert_eq!(
            execution_quantization(&hopper, Some(G::Nvfp4), Some(M::Nvfp4)).unwrap(),
            (Some(G::Nvfp4Wo), Some(M::Nvfp4Wo))
        );
        let blackwell = SystemSpec::load(&systems_root().join("b200_sxm.yaml")).unwrap();
        assert_eq!(
            execution_quantization(&blackwell, Some(G::Nvfp4), Some(M::Nvfp4)).unwrap(),
            (Some(G::Nvfp4), Some(M::Nvfp4))
        );
        assert_eq!(
            execution_quantization(&blackwell, Some(G::Nvfp4Wo), None)
                .unwrap()
                .0,
            Some(G::Nvfp4Wo)
        );
        hopper.gpu.bfloat16_tc_flops = None;
        assert!(execution_quantization(&hopper, Some(G::Nvfp4), None).is_err());
        hopper.gpu.sm_version = Some(80);
        assert!(execution_quantization(&hopper, Some(G::Fp8), None).is_err());
    }

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn systems_root() -> PathBuf {
        // CARGO_MANIFEST_DIR points at aic-core/rust/aiconfigurator-core; the systems
        // dir is two levels up.
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("python/aisimulate/src/aisimulate_core/systems")
    }

    #[test]
    fn parse_b200_sxm() {
        let spec = SystemSpec::load(&systems_root().join("b200_sxm.yaml"))
            .expect("b200_sxm.yaml parse must succeed");
        assert_eq!(spec.data_dir, PathBuf::from("data/b200_sxm"));
        // 7.7 TB/s since PR #1246 (was 8.0; this pin went stale unnoticed
        // because cargo test was not in the gating CI).
        assert_eq!(spec.gpu.mem_bw, 7_700_000_000_000.0);
        assert_eq!(spec.gpu.mem_bw_empirical_scaling_factor, 0.8);
        assert_eq!(spec.gpu.bfloat16_tc_flops, Some(2_250_000_000_000_000.0));
        assert_eq!(spec.gpu.sm_version, Some(100));
        assert_eq!(spec.node.num_gpus_per_node, 8);
        assert!(spec.node.num_gpus_per_rack.is_none()); // b200_sxm has no rack tier
        // InferenceX CX7 400GbE reference, single-direction bandwidth per GPU.
        assert_eq!(spec.get_p2p_bandwidth(8), 900_000_000_000.0);
        assert_eq!(spec.get_p2p_bandwidth(16), 50_000_000_000.0);
        assert_eq!(spec.misc.nccl_mem.get(&8), Some(&411_041_792));
        assert_eq!(spec.misc.nccl_version.as_deref(), Some("2.27.3"));
    }

    #[test]
    fn parse_gb200_rack_tier() {
        let spec = SystemSpec::load(&systems_root().join("gb200.yaml"))
            .expect("gb200.yaml parse must succeed");
        assert_eq!(spec.node.num_gpus_per_node, 4);
        assert_eq!(spec.node.num_gpus_per_rack, Some(72));
        // NVLink stays within the rack; CX7 NDR400 provides 400 / 8 = 50 GB/s per GPU.
        assert_eq!(spec.get_p2p_bandwidth(4), 900_000_000_000.0);
        assert_eq!(spec.get_p2p_bandwidth(72), 900_000_000_000.0);
        assert_eq!(spec.get_p2p_bandwidth(73), 50_000_000_000.0);
        assert_eq!(spec.get_p2p_bandwidth(128), 50_000_000_000.0);
    }

    #[test]
    fn p2p_bandwidth_three_tier_selection() {
        // Build a minimal in-memory spec with all three tiers set.
        let spec = SystemSpec {
            data_dir: PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                fp32_flops: None,
                mem_bw: 1.0,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: None,
                int8_tc_flops: None,
                fp8_tc_flops: None,
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900.0,
                inter_node_bw: 100.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: Some(72),
                inter_rack_bw: Some(10.0),
                inter_rack_latency: None,
            },
            misc: MiscSpec::default(),
        };

        // Within node -> intra
        assert_eq!(spec.get_p2p_bandwidth(1), 900.0);
        assert_eq!(spec.get_p2p_bandwidth(8), 900.0);
        // Within rack -> inter-node
        assert_eq!(spec.get_p2p_bandwidth(9), 100.0);
        assert_eq!(spec.get_p2p_bandwidth(72), 100.0);
        // Across racks -> inter-rack
        assert_eq!(spec.get_p2p_bandwidth(73), 10.0);
    }

    #[test]
    fn p2p_bandwidth_falls_back_to_inter_node_when_inter_rack_unset() {
        let spec = SystemSpec {
            data_dir: PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                fp32_flops: None,
                mem_bw: 1.0,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: None,
                int8_tc_flops: None,
                fp8_tc_flops: None,
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900.0,
                inter_node_bw: 100.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: Some(72),
                inter_rack_bw: None,
                inter_rack_latency: None, // unset
            },
            misc: MiscSpec::default(),
        };
        assert_eq!(spec.get_p2p_bandwidth(100), 100.0);
    }
}
