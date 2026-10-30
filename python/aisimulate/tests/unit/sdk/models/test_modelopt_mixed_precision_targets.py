# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targets-only ModelOpt MIXED_PRECISION exports (nvidia/GLM-5.3-NVFP4 shape).

A ModelOpt MIXED_PRECISION export quantizes exactly the modules its
``quantized_layers`` map lists. vLLM v0.24.0 (ee0da84a,
model_executor/layers/quantization/modelopt.py L2392-2490) and SGLang v0.5.14
(49e384ce, srt/layers/quantization/modelopt_quant.py L734-792) apply the
exclusion list first, then resolve a module by exact key, packed shards or a
descendant key (one that starts with the module path + "."), and leave any
linear module without an entry unquantized. An ancestor key quantizes nothing.
nvidia/GLM-5.3-NVFP4 lists only its three dense FP8 MLP layers and the NVFP4
routed experts, so its attention, indexer and shared experts run BF16.

Synthetic fixtures use the packaged GLM-5.2-NVFP4 / DeepSeek-V3 geometry
with a generated ModelOpt header shaped like that export. The packaged
nvidia/GLM-5.3-NVFP4 itself is an unmodified copy of the upstream export at
e3b8f9f2 (model_configs/GLM53_NVFP4_PROVENANCE.md); the graph and SILICON
tests run on both.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import aisimulate_core
from aisimulate.sdk.task_v2 import Task
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, engine
from aisimulate_core.sdk.common import DefaultHFModels
from aisimulate_core.sdk.config import ModelConfig
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.models import deepseek_v32, get_model
from aisimulate_core.sdk.models import helpers as H
from aisimulate_core.sdk.models.deepseek_v32 import _dense_mlp_groups
from aisimulate_core.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit

_CONFIGS = Path(aisimulate_core.__file__).parent / "model_configs"
_GLM_BASE = "nvidia--GLM-5.2-NVFP4"
_DSV3_BASE = "deepseek-ai--DeepSeek-V3"
_PACKAGED = "nvidia/GLM-5.3-NVFP4"
# nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf, byte for byte.
_PACKAGED_SHA256 = {
    "config": "b0b6b9ab922a82e3688bc39a4117bc443e106fcf88969ef48b3627778f4bdf7b",
    "hf_quant_config": "13799d4fed353c20dde2264fbdd7f68a780b514782f8d80bba96ac5e44aa9bba",
}

GLM_LAYERS = 78
DENSE_LAYERS = 3
ALL_GROUPS = frozenset(H.ATTENTION_PROJECTION_GROUPS)
BF16 = common.GEMMQuantMode.bfloat16
FP8_STATIC = common.GEMMQuantMode.fp8_static
NVFP4 = common.GEMMQuantMode.nvfp4

_DENSE_PROJECTIONS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
_GLM_ATTENTION = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "indexer.wq_b",
    "indexer.wk",
    "indexer.weights_proj",
)
_GLM_FULL_INDEXER_LAYERS = [layer for layer in range(GLM_LAYERS) if max(layer - 2, 0) % 4 == 0]


def _glm53_layers(*, dense_algo="FP8", dense_projections=_DENSE_PROJECTIONS, extra=None):
    """The GLM-5.3-NVFP4 layout: FP8 dense prefix, NVFP4 routed experts, nothing else."""
    layers = {
        f"model.layers.{layer}.mlp.{projection}": {"quant_algo": dense_algo}
        for layer in range(DENSE_LAYERS)
        for projection in dense_projections
    }
    layers.update(
        {
            f"model.layers.{layer}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}
            for layer in range(DENSE_LAYERS, GLM_LAYERS)
        }
    )
    layers.update(extra or {})
    return layers


def _listed(modules, layers=None, algo="NVFP4", block="self_attn"):
    layers = range(GLM_LAYERS) if layers is None else layers
    return {f"model.layers.{layer}.{block}.{module}": {"quant_algo": algo} for layer in layers for module in modules}


def _shared_experts_listed(algo="FP8"):
    projections = ("gate_proj", "up_proj", "down_proj")
    return _listed(projections, range(DENSE_LAYERS, GLM_LAYERS), algo, "mlp.shared_experts")


def _quant_sections(layers, *, exclude=("model.layers.78*",), quant_method="modelopt"):
    packed = {"gate_up_proj": ["gate_proj", "up_proj"]}
    language = {
        "quant_method": quant_method,
        "quant_algo": "MIXED_PRECISION",
        "ignore": list(exclude),
        "quantized_layers": layers,
        "packed_modules_mapping": packed,
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    }
    hf_quant = {
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": "FP8",
            "quantized_layers": layers,
            "exclude_modules": list(exclude),
            "packed_modules_mapping": packed,
        },
    }
    return language, hf_quant


def _raw(layers=None, **kwargs):
    language, hf_quant = _quant_sections(_glm53_layers() if layers is None else layers, **kwargs)
    return {
        "num_hidden_layers": GLM_LAYERS,
        "first_k_dense_replace": DENSE_LAYERS,
        "hidden_act": "silu",
        "quantization_config": language,
        "hf_quant_config": hf_quant,
    }


def _write_export(directory: Path, base: str, layers: dict, **kwargs) -> str:
    checkpoint = json.loads((_CONFIGS / f"{base}_config.json").read_text())
    checkpoint["quantization_config"], hf_quant = _quant_sections(layers, **kwargs)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(checkpoint))
    (directory / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    return str(directory)


@pytest.fixture(scope="module")
def glm53(tmp_path_factory):
    return _write_export(tmp_path_factory.mktemp("glm53") / "export", _GLM_BASE, _glm53_layers())


@pytest.fixture(params=["synthetic", "packaged"])
def glm53_export(request, glm53):
    """The synthetic export and the packaged upstream copy resolve alike."""
    return glm53 if request.param == "synthetic" else _PACKAGED


def _native(model, phase):
    """Name -> (tag, fields) of the native op specs, overlap groups flattened."""
    result = {}

    def walk(specs):
        for spec in specs:
            tag, fields = next(iter(spec.items()))
            if tag == "Overlap":
                walk(fields["group_a"])
                walk(fields["group_b"])
            else:
                result[fields["name"]] = (tag, fields)

    walk(json.loads(engine._ops_json(getattr(model, f"{phase}_ops"))))
    return result


def _gemm_modes(native):
    return {name: fields["quant_mode"] for name, (tag, fields) in native.items() if tag == "Gemm"}


def _build(path, backend, **kwargs):
    kwargs.setdefault("tp_size", 4)
    kwargs.setdefault("moe_tp_size", 1)
    kwargs.setdefault("moe_ep_size", 4)
    return get_model(path, ModelConfig(**kwargs), backend)


class TestScope:
    def test_glm53_shaped_export_is_in_scope(self, glm53):
        targets = H.modelopt_mixed_precision_targets(get_model_config_from_model_path(glm53)["raw_config"])
        # 3 dense layers x 4 projection keys + 75 routed-expert blocks.
        assert len(targets) == 87
        assert targets["model.layers.0.mlp.gate_up_proj"] == "fp8"
        assert targets["model.layers.77.mlp.experts"] == "nvfp4"

    def test_hf_quant_config_alone_is_in_scope(self):
        # config.json without quantization_config: the loader synthesizes
        # quant_method=mixed_precision from hf_quant_config.json.
        raw = _raw()
        raw["quantization_config"] = {"quant_method": "mixed_precision", "kv_cache_quant_method": "fp8"}
        assert len(H.modelopt_mixed_precision_targets(raw)) == 87

    def test_base_lane_requant_is_out_of_scope(self):
        raw = _raw(quant_method="fp8")
        raw["quantization_config"]["weight_block_size"] = [128, 128]
        assert H.modelopt_mixed_precision_targets(raw) is None
        # The packaged DeepSeek-V4 expert-only requants keep the FP8-block base lane.
        for path in ("nvidia/DeepSeek-V4-Flash-NVFP4", "nvidia/DeepSeek-V4-Pro-NVFP4"):
            assert H.modelopt_mixed_precision_targets(get_model_config_from_model_path(path)["raw_config"]) is None

    def test_exclusion_style_nvfp4_checkpoint_is_out_of_scope(self):
        raw = get_model_config_from_model_path("nvidia/GLM-5.2-NVFP4")["raw_config"]
        assert H.modelopt_mixed_precision_targets(raw) is None
        assert H.modelopt_shared_experts_unlisted(raw) is False
        assert H.modelopt_dense_projection_listed(raw, 0, "down_proj") is None

    def test_mixed_precision_without_a_layer_map_is_out_of_scope(self):
        assert H.modelopt_mixed_precision_targets(_raw({})) is None

    def test_conflicting_algorithms_for_one_module_raise(self):
        raw = _raw()
        raw["hf_quant_config"]["quantization"]["quantized_layers"] = {
            "model.layers.0.mlp.down_proj": {"quant_algo": "NVFP4"}
        }
        with pytest.raises(ValueError, match="conflicting algorithms for 'model.layers.0.mlp.down_proj'"):
            H.modelopt_mixed_precision_targets(raw)


_ATTENTION_CASES = {
    "nothing_listed": ({}, (), ALL_GROUPS),
    "all_listed": (_listed(_GLM_ATTENTION), (), frozenset()),
    "only_o_proj": (_listed(("o_proj",)), (), frozenset({"q", "kv", "indexer"})),
    # Ancestor keys quantize nothing beneath them on either runtime.
    "whole_block_parent_key": (
        {f"model.layers.{layer}.self_attn": {"quant_algo": "NVFP4"} for layer in range(GLM_LAYERS)},
        (),
        ALL_GROUPS,
    ),
    "ancestor_indexer_key": (
        {
            **_listed(_GLM_ATTENTION[:5]),
            **{f"model.layers.{layer}.self_attn.indexer": {"quant_algo": "NVFP4"} for layer in range(GLM_LAYERS)},
        },
        (),
        frozenset({"indexer"}),
    ),
    # A key inside an excluded module (the MTP layer) lists nothing.
    "excluded_mtp_key": ({"model.layers.78.self_attn.o_proj": {"quant_algo": "NVFP4"}}, (), ALL_GROUPS),
    # A key naming the fused q_a/kv_a runtime module lists both groups.
    "fused_qkv_a_key": (
        _listed(("fused_qkv_a_proj_with_mqa",)),
        (),
        frozenset({"o", "indexer"}),
    ),
    "exclusion_glob_wins": (_listed(_GLM_ATTENTION), ("model.layers.*.self_attn*",), ALL_GROUPS),
    "indexer_listed_in_full_layers_only": (
        {
            **_listed(_GLM_ATTENTION[:5]),
            **_listed(_GLM_ATTENTION[5:], layers=_GLM_FULL_INDEXER_LAYERS),
        },
        (),
        frozenset(),
    ),
}


@pytest.mark.parametrize("precise", [False, True])
@pytest.mark.parametrize("case", list(_ATTENTION_CASES))
def test_attention_groups_absent_from_quantized_layers_are_excluded(case, precise):
    extra, exclude, expected = _ATTENTION_CASES[case]
    raw = _raw(_glm53_layers(extra=extra), exclude=("model.layers.78*", *exclude))
    assert H.attention_projection_exclusions(raw, precise_module_paths=precise) == expected


def test_keys_beyond_the_decoder_layers_list_nothing():
    # MTP/draft layers sit at index >= num_hidden_layers even without an exclusion glob.
    raw = _raw(_glm53_layers(extra=_listed(_GLM_ATTENTION, layers=[GLM_LAYERS])), exclude=())
    assert H.attention_projection_exclusions(raw) == ALL_GROUPS
    raw = _raw(_glm53_layers(extra=_listed(_GLM_ATTENTION, layers=[GLM_LAYERS - 1])), exclude=())
    assert H.attention_projection_exclusions(raw) == frozenset()


class TestSharedExpertsAndDensePrefix:
    def test_shared_experts_are_unquantized_unless_listed(self):
        assert H.modelopt_shared_experts_unlisted(_raw()) is True
        shared = _shared_experts_listed()
        assert H.modelopt_shared_experts_unlisted(_raw(_glm53_layers(extra=shared))) is False

    def test_ancestor_and_mtp_shared_expert_keys_list_nothing(self):
        ancestor = {
            f"model.layers.{layer}.mlp.shared_experts": {"quant_algo": "FP8"}
            for layer in range(DENSE_LAYERS, GLM_LAYERS)
        }
        assert H.modelopt_shared_experts_unlisted(_raw(_glm53_layers(extra=ancestor))) is True
        mtp = _listed(("gate_proj", "up_proj", "down_proj"), [GLM_LAYERS], "FP8", "mlp.shared_experts")
        assert H.modelopt_shared_experts_unlisted(_raw(_glm53_layers(extra=mtp))) is True

    @pytest.mark.parametrize(
        "projections,expected",
        [
            (_DENSE_PROJECTIONS, [(3, FP8_STATIC, FP8_STATIC)]),
            # SGLang resolves the fused gate/up GEMM by its own key.
            (("gate_up_proj", "down_proj"), [(3, FP8_STATIC, FP8_STATIC)]),
            (("gate_proj", "up_proj", "gate_up_proj"), [(3, FP8_STATIC, BF16)]),
            ((), [(3, BF16, BF16)]),
        ],
    )
    def test_dense_prefix_precision_follows_the_listing(self, projections, expected):
        raw = _raw(_glm53_layers(dense_projections=projections))
        assert _dense_mlp_groups(raw, GLM_LAYERS, FP8_STATIC) == expected

    def test_half_listed_fused_gate_up_is_rejected(self):
        raw = _raw(_glm53_layers(dense_projections=("gate_proj", "down_proj")))
        with pytest.raises(ValueError, match="same quantization for their fused GEMM"):
            _dense_mlp_groups(raw, GLM_LAYERS, FP8_STATIC)

    def test_shards_without_the_fused_gate_up_key_are_rejected(self):
        # SGLang's DeepseekV2/GlmMoeDsa packed_modules_mapping has no
        # gate_up_proj: listed shards alone leave the fused GEMM unquantized,
        # a layout it cannot load, so there is nothing faithful to price.
        raw = _raw(_glm53_layers(dense_projections=("gate_proj", "up_proj", "down_proj")))
        assert H.modelopt_dense_projection_listed(raw, 0, "gate_proj") is True
        assert H.modelopt_dense_projection_listed(raw, 0, "gate_up_proj") is False
        with pytest.raises(ValueError, match="but not the fused gate_up_proj"):
            _dense_mlp_groups(raw, GLM_LAYERS, FP8_STATIC)


class TestGemmGuard:
    def test_listed_gemms_with_another_dtype_fail_unless_gemm_is_explicit(self, tmp_path):
        # NVFP4 attention next to FP8 dense layers: the coarse mode (fp8_static)
        # cannot price both, so inference fails instead of mispricing attention.
        path = _write_export(tmp_path / "export", _GLM_BASE, _glm53_layers(extra=_listed(_GLM_ATTENTION)))
        with pytest.raises(ValueError, match=r"gemm_quant_mode=fp8_static \(nvfp4: model\.layers\.0\.self_attn"):
            _build(path, "vllm")
        model = _build(path, "vllm", gemm_quant_mode=NVFP4)
        attention = _native(model, "context")["context_attention"][1]
        assert attention["attn_projection_quant_modes"] == dict.fromkeys(H.ATTENTION_PROJECTION_GROUPS, "nvfp4")

    def test_non_blackwell_weight_only_lane_matches_nvfp4_entries(self):
        raw = _raw(_glm53_layers(dense_algo="NVFP4", extra=_listed(_GLM_ATTENTION)))
        H.validate_modelopt_targets_only_gemm_mode(raw, NVFP4)
        H.validate_modelopt_targets_only_gemm_mode(raw, common.GEMMQuantMode.nvfp4_wo)
        with pytest.raises(ValueError, match="gemm_quant_mode=fp8_static"):
            H.validate_modelopt_targets_only_gemm_mode(raw, FP8_STATIC)

    def test_glm53_layout_passes(self):
        H.validate_modelopt_targets_only_gemm_mode(_raw(), FP8_STATIC)

    def test_keys_that_quantize_no_modeled_gemm_are_ignored(self):
        # Ancestor keys and keys inside the excluded MTP layer quantize no
        # decoder GEMM, so their dtype cannot conflict with the coarse mode.
        extra = {
            **{f"model.layers.{layer}.self_attn": {"quant_algo": "NVFP4"} for layer in range(GLM_LAYERS)},
            **_listed(_GLM_ATTENTION, layers=[GLM_LAYERS]),
            **_listed(("gate_proj", "up_proj", "down_proj"), [GLM_LAYERS], "NVFP4", "mlp.shared_experts"),
        }
        H.validate_modelopt_targets_only_gemm_mode(_raw(_glm53_layers(extra=extra)), FP8_STATIC)


@pytest.mark.parametrize(
    "algo,quant_dynamic,expected",
    [
        ("mxfp8", None, common.GEMMQuantMode.fp8_block),
        ("fp8", None, FP8_STATIC),
        ("fp8", False, FP8_STATIC),
        ("fp8", True, common.GEMMQuantMode.fp8),
        ("w4a16_nvfp4", None, common.GEMMQuantMode.w4a16_nvfp4),
        ("nvfp4", None, NVFP4),
        ("int8", None, None),
    ],
)
def test_per_algorithm_gemm_mode(algo, quant_dynamic, expected):
    assert H._mixed_precision_algo_gemm_mode(algo, quant_dynamic) is expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("nvidia/DeepSeek-V4-Flash-NVFP4", ("fp8_block", "nvfp4", "fp8", None)),
        ("nvidia/DeepSeek-V4-Pro-NVFP4", ("fp8_block", "nvfp4", "fp8", None)),
        ("nvidia/MiniMax-M3-NVFP4", ("fp8_block", "nvfp4", None, None)),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", ("fp8_static", "w4a16_nvfp4", "fp8", "fp8")),
        ("nvidia/Qwen3.6-27B-NVFP4", ("fp8_static", None, "fp8", "fp8")),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", ("fp8_static", "w4a16_nvfp4", "fp8", "fp8")),
    ],
)
def test_coarse_inference_of_packaged_mixed_precision_bundles_is_unchanged(path, expected):
    info = get_model_config_from_model_path(path)
    modes = H._infer_quant_modes_from_raw_config(info["raw_config"], info["architecture"])
    keys = ("gemm_quant_mode", "moe_quant_mode", "kvcache_quant_mode", "fmha_quant_mode")
    assert tuple(getattr(modes.get(key), "name", None) for key in keys) == expected


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_native_graph_prices_unlisted_modules_bf16(glm53_export, backend, phase):
    model = _build(glm53_export, backend)
    assert (model.config.gemm_quant_mode, model.config.moe_quant_mode, model.config.kvcache_quant_mode) == (
        FP8_STATIC,
        common.MoEQuantMode.nvfp4,
        common.KVCacheQuantMode.fp8,
    )
    native = _native(model, phase)
    attention = native[f"{phase}_attention"][1]
    assert attention["gemm_quant_mode"] == "bfloat16"
    assert attention["attn_projection_quant_modes"] == dict.fromkeys(H.ATTENTION_PROJECTION_GROUPS, "bfloat16")
    assert attention["kv_cache_dtype"] == "fp8"
    gemms = _gemm_modes(native)
    for name in ("shared_gate_up_gemm", "shared_ffn2_gemm", "router_gemm", "logits_gemm"):
        assert gemms[f"{phase}_{name}"] == "bfloat16", name
    assert native[f"{phase}_moe"][1]["quant_mode"] == "nvfp4"
    dense = {name: mode for name, mode in gemms.items() if "_dense" in name}
    if backend == "sglang":
        # Only SGLang models the dense prefix; it runs the listed FP8 lane.
        assert dense == {f"{phase}_dense_gate_up_gemm": "fp8_static", f"{phase}_dense_down_gemm": "fp8_static"}
        assert native[f"{phase}_dense_gate_up_gemm"][1]["scale_factor"] == DENSE_LAYERS
    else:
        assert dense == {}


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_explicit_gemm_mode_reaches_listed_modules_only(glm53_export, phase):
    model = _build(glm53_export, "sglang", gemm_quant_mode=NVFP4, moe_quant_mode=common.MoEQuantMode.nvfp4)
    native = _native(model, phase)
    assert native[f"{phase}_attention"][1]["gemm_quant_mode"] == "bfloat16"
    gemms = _gemm_modes(native)
    assert gemms[f"{phase}_shared_gate_up_gemm"] == gemms[f"{phase}_shared_ffn2_gemm"] == "bfloat16"
    assert gemms[f"{phase}_dense_gate_up_gemm"] == gemms[f"{phase}_dense_down_gemm"] == "nvfp4"


def test_listed_shared_experts_keep_the_coarse_mode(tmp_path):
    shared = _shared_experts_listed()
    path = _write_export(tmp_path / "export", _GLM_BASE, _glm53_layers(extra=shared))
    gemms = _gemm_modes(_native(_build(path, "vllm"), "context"))
    assert gemms["context_shared_gate_up_gemm"] == gemms["context_shared_ffn2_gemm"] == "fp8_static"


_LARGE_EP = {
    "trtllm": ({"context": "nvlink_two_sided", "generation": "nvlink_two_sided"}, 4, "shared_gate_up_gemm"),
    "sglang": ({"context": "deepep_ht", "generation": "deepep_ll"}, 8, "gate_ffn1_gemm"),
    "vllm": ({"context": "deepep_ht", "generation": "deepep_ll"}, 8, "gate_ffn1_gemm"),
}


def _large_ep(path, backend):
    comm, gpus_per_node, _ = _LARGE_EP[backend]
    return _build(
        path,
        backend,
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
        attention_dp_size=16,
        moe_backend="deepep_moe" if backend == "sglang" else None,
        moe_comm_backend=dict(comm),
        num_gpus_per_node=gpus_per_node,
    )


@pytest.mark.parametrize("backend", list(_LARGE_EP))
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_large_ep_shared_experts_follow_the_listing(glm53_export, backend, phase):
    shared = f"{phase}_{_LARGE_EP[backend][2]}"
    assert _gemm_modes(_native(_large_ep(glm53_export, backend), phase))[shared] == "bfloat16"
    # Exclusion-style GLM-5.2-NVFP4 keeps its transcribed legacy asymmetry:
    # BF16 on trtllm, the model-wide nvfp4 lane elsewhere.
    legacy = _gemm_modes(_native(_large_ep("nvidia/GLM-5.2-NVFP4", backend), phase))[shared]
    assert legacy == ("bfloat16" if backend == "trtllm" else "nvfp4")


class TestDeepSeekFamily:
    """The DeepSeek/Kimi MLA family resolves the same rule through the shared helpers."""

    DSV3_LAYERS = 61
    MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj")

    def _export(self, tmp_path, attention=MLA_PROJECTIONS, dense_algo="NVFP4"):
        layers = {
            f"model.layers.{layer}.mlp.{projection}": {"quant_algo": dense_algo}
            for layer in range(DENSE_LAYERS)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        layers.update(
            {
                f"model.layers.{layer}.mlp.experts": {"quant_algo": "NVFP4"}
                for layer in range(DENSE_LAYERS, self.DSV3_LAYERS)
            }
        )
        layers.update(_listed(attention, range(self.DSV3_LAYERS)))
        return _write_export(tmp_path / "export", _DSV3_BASE, layers, exclude=("model.layers.61*",))

    def test_unlisted_o_proj_and_shared_experts_run_bf16(self, tmp_path):
        path = self._export(tmp_path)
        raw = get_model_config_from_model_path(path)["raw_config"]
        assert H.attention_projection_exclusions(raw) == frozenset({"o", "indexer"})
        model = _build(path, "sglang", tp_size=8, moe_ep_size=8)
        assert model.config.gemm_quant_mode is NVFP4
        assert model._shared_expert_quant_mode is BF16
        # The family keeps its explicit-gemm gating for shared experts.
        explicit = _build(path, "sglang", tp_size=8, moe_ep_size=8, gemm_quant_mode=NVFP4)
        assert explicit._shared_expert_quant_mode is NVFP4

    def test_task_attention_identity_follows_the_listing(self, tmp_path):
        mixed = get_model_config_from_model_path(self._export(tmp_path / "mixed"))["raw_config"]
        unlisted = get_model_config_from_model_path(self._export(tmp_path / "unlisted", attention=()))["raw_config"]

        def identity_mixed(raw):
            task = SimpleNamespace(_raw_config=raw, _role_attr=lambda role, name: NVFP4)
            return Task._attention_quant_identity_mixed(task, "agg")

        assert identity_mixed(mixed) is True
        assert identity_mixed(unlisted) is False

    def test_guard_applies_to_the_family(self, tmp_path):
        path = self._export(tmp_path, dense_algo="FP8")
        with pytest.raises(ValueError, match="gemm_quant_mode=fp8_static"):
            _build(path, "sglang", tp_size=8, moe_ep_size=8)


class TestPackagedExport:
    """The packaged nvidia/GLM-5.3-NVFP4 is the upstream targets-only export."""

    @pytest.mark.parametrize("suffix", list(_PACKAGED_SHA256))
    def test_bundle_is_the_pinned_upstream_bytes(self, suffix):
        data = (_CONFIGS / f"nvidia--GLM-5.3-NVFP4_{suffix}.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == _PACKAGED_SHA256[suffix]

    def test_bundle_resolves_the_listed_modes(self):
        info = get_model_config_from_model_path(_PACKAGED)
        raw = info["raw_config"]
        modes = H._infer_quant_modes_from_raw_config(raw, info["architecture"])
        assert (modes["gemm_quant_mode"], modes["moe_quant_mode"], modes["kvcache_quant_mode"]) == (
            FP8_STATIC,
            common.MoEQuantMode.nvfp4,
            common.KVCacheQuantMode.fp8,
        )
        assert len(H.modelopt_mixed_precision_targets(raw)) == 87
        assert H.attention_projection_exclusions(raw) == ALL_GROUPS
        assert H.modelopt_shared_experts_unlisted(raw) is True
        assert _dense_mlp_groups(raw, GLM_LAYERS, FP8_STATIC) == [(DENSE_LAYERS, FP8_STATIC, FP8_STATIC)]
        # The upstream config carries head_dim 64 (the rope width); it is read
        # as is. For DSA it only feeds the activation-memory width (all
        # backends); latency op keys and KV bytes per token are unaffected.
        assert info["d"] == 64


def test_packaged_targets_only_checkpoints_are_reviewed():
    """Every packaged checkpoint the targets-only rule reaches, by family.

    The changed resolvers run only for the DEEPSEEK/KIMIK25/DEEPSEEKV32
    families. nvidia/GLM-5.3-NVFP4 is the only member of those families in
    scope, so the other predictions here do not move. A new member needs the
    same review.
    """
    in_scope = {}
    for path in sorted(DefaultHFModels):
        info = get_model_config_from_model_path(path)
        if H.modelopt_mixed_precision_targets(info["raw_config"]) is not None:
            in_scope[path] = H._architecture_to_model_family(info["architecture"])
    assert in_scope == {
        _PACKAGED: "DEEPSEEKV32",
        "nvidia/MiniMax-M3-NVFP4": "MINIMAXM3",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": "NEMOTRONH",
        # QWEN35 family: outside the changed DEEPSEEK/KIMIK25/DEEPSEEKV32 resolvers.
        "nvidia/Qwen3.8-2.4T-A95B-NVFP4": "QWEN35",
        "nvidia/Qwen3.6-27B-NVFP4": "QWEN35",
        "nvidia/Qwen3.6-35B-A3B-NVFP4": "QWEN35",
    }


_SILICON_SYSTEMS = ("b200_sxm", "b300_sxm", "gb200", "gb300")
_SILICON_RUNTIMES = (("sglang", "0.5.14"), ("vllm", "0.24.0"), ("trtllm", "1.3.0rc20"))
_SILICON_LAYOUTS = ((4, 1, 4), (8, 1, 8), (4, 4, 1))


def _silicon(model, system, backend, version, layout):
    tp, moe_tp, moe_ep = layout
    return ForwardPassPerfModelConfig(
        model=model,
        system=system,
        backend=backend,
        backend_version=version,
        worker_type="aggregated",
        tp=tp,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        database_mode="SILICON",
        estimation_mode="op_level",
        fallback_policy="deny",
        enable_shared_layer=False,
    )


def _phase_attention(model):
    result = []
    for prefill, batch_size in ((True, 1), (False, 32)):
        ops = model.static_phase_diagnostics(prefill=prefill, batch_size=batch_size, context_length=4096)
        (attention,) = [op for op in ops if op["name"].endswith("_attention")]
        result.append(attention)
    return result


@pytest.mark.parametrize("layout", _SILICON_LAYOUTS, ids=lambda layout: "tp{}-moetp{}-ep{}".format(*layout))
@pytest.mark.parametrize(("backend", "version"), _SILICON_RUNTIMES)
@pytest.mark.parametrize("system", _SILICON_SYSTEMS)
def test_packaged_silicon_covers_the_checkpoint_precision(glm53_export, system, backend, version, layout):
    # Public contract: the canonical constructor with fallback denied. The
    # DSA rows exist for (fmha bf16, kv fp8, gemm bf16), which is what the
    # checkpoint runs; no new collection is needed.
    tp, moe_tp, moe_ep = layout
    graph = _build(glm53_export, backend, tp_size=tp, moe_tp_size=moe_tp, moe_ep_size=moe_ep)
    assert _native(graph, "context")["context_attention"][1]["gemm_quant_mode"] == "bfloat16"
    model = RustForwardPassPerfModel.best_available(_silicon(glm53_export, system, backend, version, layout))
    try:
        for attention in _phase_attention(model):
            assert attention["source"] == "silicon"
            assert attention["latency_ms"] > 0
    finally:
        model.close()


def test_coarse_attention_mode_has_no_silicon_rows(glm53, monkeypatch):
    # Negative control: pricing attention at the coarse fp8_static mode (the
    # behaviour without the targets-only rule) selects DSA rows that do not exist.
    monkeypatch.setattr(deepseek_v32, "attention_projection_exclusions", lambda raw, **kwargs: frozenset())
    model = RustForwardPassPerfModel.best_available(_silicon(glm53, "b200_sxm", "vllm", "0.24.0", (4, 1, 4)))
    try:
        with pytest.raises(PerfDataNotAvailableError, match='gemm_quant: "fp8_static"'):
            _phase_attention(model)
    finally:
        model.close()
