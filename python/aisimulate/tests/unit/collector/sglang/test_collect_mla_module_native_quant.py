# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``_model_native_gemm_quant`` and the ModelOpt ``MIXED_PRECISION`` export.

A ``MIXED_PRECISION`` export names one quantization lane per module in
``quantized_layers`` instead of one top-level algorithm. The collector reads the
cached export files from ``_MODEL_CONFIG_DIR``. These tests point that directory
at ``tmp_path`` and write the files themselves.
"""

import json
import sys
import types
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

MODEL_ID = "org/m"

# Shape of nvidia/GLM-5.3-NVFP4: static per-tensor FP8 dense MLP in the leading
# layers and NVFP4 routed experts. No attention module is listed, so attention
# stays BF16.
GLM53_LANES = {
    "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
    "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
    "model.layers.0.mlp.down_proj": {"quant_algo": "FP8"},
    "model.layers.0.mlp.gate_up_proj": {"quant_algo": "FP8"},
    "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
}


@pytest.fixture(autouse=True)
def _mock_helper_imports(monkeypatch):
    """Ensure the module can be imported without CUDA / SGLang installed."""

    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90  # default Hopper
    fake_helper.log_perf = lambda **kw: None
    fake_helper.benchmark_with_power = lambda **kw: None
    monkeypatch.setitem(sys.modules, "helper", fake_helper)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.float8_e4m3fn = "float8_e4m3fn"
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        get_device_capability=lambda *_a, **_kw: (9, 0),
        get_device_name=lambda *_a, **_kw: "Fake GPU",
        is_available=lambda: False,
        set_device=lambda *_a, **_kw: None,
    )
    fake_torch.randn = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _import_module():
    """Import collect_mla_module after mocking."""
    import importlib.util

    mod_name = "collector.sglang.collect_mla_module"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        "collector/sglang/collect_mla_module.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod(monkeypatch, tmp_path):
    """The collector module reading its cached export files from ``tmp_path``."""
    module = _import_module()
    monkeypatch.setattr(module, "_MODEL_CONFIG_DIR", str(tmp_path))
    return module


def _write_hf_quant(tmp_path, quantization):
    """Write ``org--m_hf_quant_config.json`` around a ModelOpt ``quantization`` section."""
    payload = {"producer": {"name": "modelopt"}, "quantization": quantization}
    (tmp_path / "org--m_hf_quant_config.json").write_text(json.dumps(payload))


def _write_config(tmp_path, quantization_config=None):
    """Write ``org--m_config.json``, with no ``quantization_config`` key when ``None``."""
    payload = {"architectures": ["GlmMoeDsaForCausalLM"]}
    if quantization_config is not None:
        payload["quantization_config"] = quantization_config
    (tmp_path / "org--m_config.json").write_text(json.dumps(payload))


def _mixed_hf_quant(lanes):
    return {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": "FP8",
        "quantized_layers": lanes,
        "exclude_modules": ["model.layers.78*"],
    }


def _mixed_config(lanes, **overrides):
    config = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": lanes,
        "ignore": ["model.layers.78*"],
        "kv_cache_scheme": {"type": "float", "num_bits": 8},
    }
    config.update(overrides)
    return config


def _write_mixed_export(tmp_path, lanes):
    """Write both files of a ModelOpt MIXED_PRECISION export with the given lanes."""
    _write_hf_quant(tmp_path, _mixed_hf_quant(lanes))
    _write_config(tmp_path, _mixed_config(lanes))


class TestMixedPrecisionExport:
    def test_glm53_shape_with_both_files_selects_the_nvfp4_loader(self, mod, tmp_path):
        _write_mixed_export(tmp_path, GLM53_LANES)
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    def test_config_json_only(self, mod, tmp_path):
        _write_config(tmp_path, _mixed_config(GLM53_LANES))
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    def test_config_json_without_quant_method_is_modelopt(self, mod, tmp_path):
        # SGLang parses a section with no quant_method as ModelOpt.
        config = _mixed_config(GLM53_LANES)
        del config["quant_method"]
        _write_config(tmp_path, config)
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    def test_hf_quant_config_only(self, mod, tmp_path):
        _write_hf_quant(tmp_path, _mixed_hf_quant(GLM53_LANES))
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    def test_hf_quant_config_is_read_when_config_json_has_no_quantization_config(self, mod, tmp_path):
        # Shape of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4: config.json carries no
        # quantization section and hf_quant_config.json holds the mixed export.
        lanes = {
            "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
            "backbone.layers.1.mixer.experts.0.down_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        }
        _write_config(tmp_path)
        _write_hf_quant(tmp_path, _mixed_hf_quant(lanes))
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    @pytest.mark.parametrize("fp4_lane", ["NVFP4", "nvfp4", "W4A16_NVFP4", "w4a16_nvfp4"])
    def test_fp8_and_fp4_lanes_select_the_nvfp4_loader(self, mod, tmp_path, fp4_lane):
        lanes = {
            "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
            "model.layers.1.mlp.experts": {"quant_algo": fp4_lane},
        }
        _write_mixed_export(tmp_path, lanes)
        assert mod._model_native_gemm_quant(MODEL_ID) == "nvfp4"

    @pytest.mark.parametrize(
        "lanes",
        [
            pytest.param({"model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"}}, id="fp8-lanes-only"),
            pytest.param({}, id="empty-quantized-layers"),
            pytest.param(None, id="null-quantized-layers"),
        ],
    )
    def test_export_without_an_fp4_lane_raises(self, mod, tmp_path, lanes):
        # SGLang resolves such an export to w4afp8, which this collector has no path for.
        _write_mixed_export(tmp_path, lanes)
        with pytest.raises(ValueError, match=rf"{MODEL_ID}.*w4afp8"):
            mod._model_native_gemm_quant(MODEL_ID)

    def test_fp8_only_raise_names_the_config_json_section(self, mod, tmp_path):
        # SGLang reads the config.json section and never looks at the NVFP4 lanes that
        # hf_quant_config.json lists, so the message has to say which section was read.
        fp8_lanes = {"model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"}}
        _write_hf_quant(tmp_path, _mixed_hf_quant(GLM53_LANES))
        _write_config(tmp_path, _mixed_config(fp8_lanes))
        with pytest.raises(ValueError, match=r"no NVFP4 lane in its config\.json quantization_config:"):
            mod._model_native_gemm_quant(MODEL_ID)

    def test_fp8_only_raise_names_the_hf_quant_config_section(self, mod, tmp_path):
        fp8_lanes = {"model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"}}
        _write_hf_quant(tmp_path, _mixed_hf_quant(fp8_lanes))
        with pytest.raises(ValueError, match=r"no NVFP4 lane in its hf_quant_config\.json quantization:"):
            mod._model_native_gemm_quant(MODEL_ID)

    def test_export_that_quantizes_attention_raises(self, mod, tmp_path):
        # The bfloat16 DSA label assumes the attention projections are unquantized, so an export
        # that lists a self_attn lane has no valid label.
        lanes = {**GLM53_LANES, "model.layers.3.self_attn.q_b_proj": {"quant_algo": "NVFP4", "group_size": 16}}
        _write_mixed_export(tmp_path, lanes)
        with pytest.raises(ValueError, match=rf"{MODEL_ID}.*quantizes attention modules.*self_attn\.q_b_proj"):
            mod._model_native_gemm_quant(MODEL_ID)

    def test_quant_method_other_than_modelopt_is_not_the_mixed_recipe(self, mod, tmp_path):
        _write_config(tmp_path, _mixed_config(GLM53_LANES, quant_method="compressed-tensors"))
        assert mod._model_native_gemm_quant(MODEL_ID) is None

    @pytest.mark.parametrize("section", ["config.json", "hf_quant_config.json"])
    def test_lower_case_quant_algo_is_not_the_mixed_recipe(self, mod, tmp_path, section):
        # SGLang compares quant_algo to "MIXED_PRECISION" exactly (configs/model_config.py:1145@v0.5.14,
        # :1153@v0.5.15), so a lower-case value is not the mixed recipe to the framework either.
        if section == "config.json":
            _write_config(tmp_path, _mixed_config(GLM53_LANES, quant_algo="mixed_precision"))
        else:
            _write_hf_quant(tmp_path, {**_mixed_hf_quant(GLM53_LANES), "quant_algo": "mixed_precision"})
        assert mod._model_native_gemm_quant(MODEL_ID) is None

    def test_config_json_section_takes_precedence_over_hf_quant_config(self, mod, tmp_path):
        # SGLang reads hf_quant_config.json only when config.json has no quantization section.
        _write_hf_quant(tmp_path, _mixed_hf_quant(GLM53_LANES))
        _write_config(tmp_path, {"quant_method": "awq"})
        assert mod._model_native_gemm_quant(MODEL_ID) is None

    def test_deepseek_v4_shape_stays_fp8_block(self, mod, tmp_path):
        # config.json declares block-FP8 and SGLang reads that section first, so the NVFP4
        # lanes in hf_quant_config.json do not change the loader.
        lanes = {"layers.0.ffn.experts": {"quant_algo": "NVFP4", "group_size": 16}}
        _write_hf_quant(tmp_path, _mixed_hf_quant(lanes))
        _write_config(
            tmp_path,
            _mixed_config(lanes, quant_method="fp8", weight_block_size=[128, 128]),
        )
        assert mod._model_native_gemm_quant(MODEL_ID) == "fp8_block"

    def test_dsa_operator_gemm_type_stays_bfloat16(self, mod, tmp_path):
        # The export lists no attention module, so the timed DSA projections stay BF16.
        _write_mixed_export(tmp_path, GLM53_LANES)
        assert mod._model_dsa_operator_gemm_type(MODEL_ID) == "bfloat16"

    def test_hopper_loader_rejects_the_export_before_sglang_is_imported(self, mod, tmp_path, monkeypatch):
        _write_mixed_export(tmp_path, GLM53_LANES)
        # An import of SGLang would raise ImportError, so the ValueError below can only come
        # from the SM guard that runs ahead of it.
        monkeypatch.setitem(sys.modules, "sglang", None)
        with (
            patch.object(mod, "get_sm_version", return_value=90),
            pytest.raises(ValueError, match=r"no collector-supported native NVFP4.*SM90"),
        ):
            mod.load_model_runner(MODEL_ID, 8, "bfloat16", "dsa")


class TestUnchangedExports:
    @pytest.mark.parametrize(
        ("hf_quant", "quantization_config", "expected"),
        [
            pytest.param({"quant_algo": "NVFP4", "group_size": 16}, None, "nvfp4", id="top-level-nvfp4"),
            pytest.param({"quant_algo": "FP8"}, None, "fp8_block", id="top-level-fp8"),
            pytest.param(
                None,
                {"quant_method": "fp8", "weight_block_size": [128, 128]},
                "fp8_block",
                id="block-fp8-config",
            ),
        ],
    )
    def test_quantized_exports(self, mod, tmp_path, hf_quant, quantization_config, expected):
        if hf_quant is not None:
            _write_hf_quant(tmp_path, hf_quant)
        if quantization_config is not None:
            _write_config(tmp_path, quantization_config)
        assert mod._model_native_gemm_quant(MODEL_ID) == expected

    def test_bf16_config(self, mod, tmp_path):
        _write_config(tmp_path)
        assert mod._model_native_gemm_quant(MODEL_ID) is None

    def test_no_files(self, mod):
        assert mod._model_native_gemm_quant(MODEL_ID) is None


def test_bundled_glm53_nvfp4_export_selects_the_nvfp4_loader():
    # Reads the bundled files, not tmp_path. The check passes on the pure-NVFP4 copy
    # (top-level quant_algo NVFP4) through the top-level branch, and on the MIXED_PRECISION
    # export only through the mixed rule.
    mod = _import_module()
    assert mod._model_native_gemm_quant("nvidia/GLM-5.3-NVFP4") == "nvfp4"
