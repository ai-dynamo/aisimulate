# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint index reuse survives the model-to-native operation transfer."""

import json
from pathlib import Path

import pandas as pd
import pytest

import aisimulate_core
from aisimulate.sdk import common, config, engine
from aisimulate.sdk.models import get_model
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize(
    "model_path,layers,full_layers",
    [
        ("nvidia/GLM-5.2-NVFP4", 78, 21),
        ("nvidia/GLM-5.3-NVFP4", 78, 21),
        ("deepseek-ai/DeepSeek-V3.2", 61, 61),
        ("nvidia/GLM-5-NVFP4", 78, 78),
    ],
)
def test_vllm_checkpoint_layer_counts_reach_native_ops(model_path, layers, full_layers, phase):
    model = get_model(
        model_path,
        config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        backend_name="vllm",
    )
    specs = json.loads(engine._ops_json(getattr(model, f"{phase}_ops")))
    attention = next(fields for spec in specs for tag, fields in spec.items() if tag == f"Dsa{phase.capitalize()}")
    # GLM-5.2/5.3: the three-layer prefix plus 18 periodic producer layers
    # gives 21 full and 57 reuse layers. DeepSeek-V3.2 has no reuse pattern.
    assert attention["scale_factor"] == layers
    assert attention["full_frac"] == pytest.approx(full_layers / layers)
    assert attention["full_frac"] * attention["scale_factor"] == pytest.approx(full_layers)


def test_trtllm_without_a_qualified_skip_producer_retains_all_full():
    model = get_model(
        "nvidia/GLM-5.2-NVFP4",
        config.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8),
        backend_name="trtllm",
    )
    assert model.extra_params["dsa_full_layer_fraction"] == 1.0


def test_public_model_preserves_equivalent_regex_projection_exclusions(tmp_path):
    raw = json.loads(
        (Path(aisimulate_core.__file__).parent / "model_configs/zai-org--GLM-5.2-FP8_config.json").read_text()
    )
    latencies = []
    for index, suffix in enumerate(("", "$")):
        checkpoint = tmp_path / str(index)
        checkpoint.mkdir()
        raw["quantization_config"]["ignore"] = [rf"re:.*self_attn\..*{suffix}"]
        (checkpoint / "config.json").write_text(json.dumps(raw))
        graph = get_model(
            str(checkpoint),
            config.ModelConfig(
                tp_size=8,
                moe_tp_size=1,
                moe_ep_size=8,
                gemm_quant_mode=common.GEMMQuantMode.fp8,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            ),
            backend_name="vllm",
        )
        model = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=str(checkpoint),
                system="b200_sxm",
                backend="vllm",
                backend_version="current",
                worker_type="aggregated",
                tp=8,
                moe_tp_size=1,
                moe_ep_size=8,
                gemm_quant_mode="fp8",
                kvcache_quant_mode="fp8",
                estimation_mode="op_level",
                database_mode="SILICON",
                enable_shared_layer=False,
            )
        )
        phase_latencies = []
        for phase in ("context", "generation"):
            specs = json.loads(engine._ops_json(getattr(graph, f"{phase}_ops")))
            attention = next(spec[f"Dsa{phase.capitalize()}"] for spec in specs if f"Dsa{phase.capitalize()}" in spec)
            assert attention["gemm_quant_mode"] == "bfloat16"
            assert attention["attn_projection_quant_modes"] == dict.fromkeys(("q", "kv", "o", "indexer"), "bfloat16")
            ops = model.static_phase_diagnostics(batch_size=1, context_length=8192, prefill=phase == "context")
            latency = next(op["latency_ms"] for op in ops if op["name"] == f"{phase}_attention")
            assert latency > 0
            phase_latencies.append(latency)
        latencies.append(phase_latencies)
    # An optional end anchor must not change either native table selection or
    # the projection weight modes used by the Rust attention estimate.
    assert latencies[0] == pytest.approx(latencies[1])


@pytest.mark.parametrize("skip_kv", [None, "bfloat16", "fp8"])
def test_public_model_uses_only_matching_vllm_reuse_measurements(tmp_path, skip_kv):
    """Independent ledger: 21 * 10 + 57 * 2 = 324 ms; all-full = 780 ms.

    Synthetic tables deliberately use large latencies to keep the exact
    measured cells above the hardware roofline. Other operations retain the
    shipped measurements; only the attention component is asserted here.
    """
    systems = Path(aisimulate_core.__file__).parent / "systems"
    (tmp_path / "b200_sxm.yaml").symlink_to(systems / "b200_sxm.yaml")
    (tmp_path / "query_versions.yaml").symlink_to(systems / "query_versions.yaml")
    target = tmp_path / "data/b200_sxm"
    target.mkdir(parents=True)
    for family in (systems / "data/b200_sxm").iterdir():
        if family.name != "sparse_attention":
            (target / family.name).symlink_to(family, target_is_directory=True)
    table_dir = target / "sparse_attention/vllm/0.24.0"
    table_dir.mkdir(parents=True)
    for phase in ("context", "generation"):
        base = {
            "op_name": f"dsa_{phase}_module",
            "kernel_source": "FLASHINFER_MLA_SPARSE",
            "architecture": "GlmMoeDsaForCausalLM",
            "mla_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "gemm_type": "bfloat16",
            "num_heads": 8,
            "batch_size": 1,
            "isl": 8192,
            "step": 0,
            "latency": 10.0,
        }
        rows = [base]
        if skip_kv is not None:
            rows.append(dict(base, op_name=f"dsa_{phase}_module_skip_indexer", kv_cache_dtype=skip_kv, latency=2.0))
        pd.DataFrame(rows).to_parquet(table_dir / f"dsa_{phase}_module_perf.parquet", index=False)
    model = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            backend="vllm",
            backend_version="current",
            worker_type="aggregated",
            tp=8,
            moe_tp_size=1,
            moe_ep_size=8,
            gemm_quant_mode="nvfp4",
            kvcache_quant_mode="bfloat16",
            estimation_mode="op_level",
            database_mode="SILICON",
            enable_shared_layer=False,
            systems_paths=(str(tmp_path),),
        )
    )
    for phase in ("context", "generation"):
        # The decode API includes the next token in its table coordinate.
        ops = model.static_phase_diagnostics(
            batch_size=1, context_length=8192 if phase == "context" else 8191, prefill=phase == "context"
        )
        attention = next(op for op in ops if op["name"] == f"{phase}_attention")
        expected = 324.0 if skip_kv == "bfloat16" else 780.0
        assert attention["latency_ms"] == pytest.approx(expected)
