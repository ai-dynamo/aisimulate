# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MIXED_PRECISION export names what it quantizes; everything else is unquantized."""

from copy import deepcopy

import pytest

from aisimulate_core.sdk import common, config, models
from aisimulate_core.sdk.models import deepseek_v32, helpers

pytestmark = pytest.mark.unit

GLM53_LIKE = {
    "quant_algo": "mixed_precision",
    "hf_quant_config": {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                **{
                    f"model.layers.{i}.mlp.{p}": {"quant_algo": "FP8"}
                    for i in range(3)
                    for p in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
                },
                **{f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for i in range(3, 78)},
            },
            "exclude_modules": ["model.layers.78*"],
        }
    },
    "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": ["model.layers.78*"],
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "float", "dynamic": False},
                "targets": ["model.layers.0.mlp.gate_proj"],
            }
        },
    },
}


def _export(*, hf_layers=(), language_layers=(), targets=()):
    """A MIXED_PRECISION raw config that lists the given entries in each of the three sources."""
    return {
        "quant_algo": "mixed_precision",
        "hf_quant_config": {"quantization": {"quantized_layers": {layer: {} for layer in hf_layers}}},
        "quantization_config": {
            "quantized_layers": {layer: {} for layer in language_layers},
            "config_groups": {"group_0": {"targets": list(targets)}},
        },
    }


def test_quantized_paths_collects_both_sources_lower_cased():
    paths = helpers.mixed_precision_quantized_paths(GLM53_LIKE)
    assert "model.layers.0.mlp.gate_proj" in paths and "model.layers.77.mlp.experts" in paths
    assert len(paths) == 87


def test_quantized_paths_is_none_outside_mixed_precision():
    assert (
        helpers.mixed_precision_quantized_paths(
            {"quant_algo": "nvfp4", "hf_quant_config": GLM53_LIKE["hf_quant_config"]}
        )
        is None
    )


def test_quantized_paths_is_none_for_class_name_targets():
    raw = {
        "quant_algo": "mixed_precision",
        "quantization_config": {
            "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}, "targets": ["Linear"]}}
        },
    }
    assert helpers.mixed_precision_quantized_paths(raw) is None


def test_quantized_paths_reads_each_of_the_three_sources():
    raw = _export(
        hf_layers=["model.layers.0.mlp.gate_proj"],
        language_layers=["model.layers.1.mlp.up_proj"],
        targets=["model.layers.2.mlp.down_proj"],
    )
    assert helpers.mixed_precision_quantized_paths(raw) == frozenset(
        {"model.layers.0.mlp.gate_proj", "model.layers.1.mlp.up_proj", "model.layers.2.mlp.down_proj"}
    )


def test_quantized_paths_drops_class_names_next_to_dotted_paths():
    paths = helpers.mixed_precision_quantized_paths(_export(targets=["Linear", "model.layers.0.mlp.gate_proj"]))
    assert paths == frozenset({"model.layers.0.mlp.gate_proj"})


def test_quantized_paths_keeps_dot_less_modules_next_to_a_dotted_path():
    paths = helpers.mixed_precision_quantized_paths(_export(targets=["lm_head", "model.layers.0.mlp.gate_proj"]))
    assert paths == frozenset({"lm_head", "model.layers.0.mlp.gate_proj"})
    assert helpers.mixed_precision_covers(paths, "lm_head")


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(["Linear", "re:.*self_attn.*", "model.layers.0.mlp.gate_proj"], id="class-name-and-regex"),
        pytest.param([r"re:model\.layers\.1\.mlp\.up_proj", "model.layers.0.mlp.gate_proj"], id="regex-without-star"),
        pytest.param(["model.layers.*.mlp.experts", "model.layers.0.mlp.gate_proj"], id="glob-with-dots"),
    ],
)
def test_quantized_paths_drops_regex_and_glob_targets(targets):
    paths = helpers.mixed_precision_quantized_paths(_export(targets=targets))
    assert paths == frozenset({"model.layers.0.mlp.gate_proj"})


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(["re:.*self_attn.*"], id="regex-only"),
        pytest.param(["model.layers.*.mlp.experts"], id="glob-only"),
        pytest.param(["lm_head"], id="dot-less-only"),
    ],
)
def test_quantized_paths_is_none_without_a_kept_dotted_entry(targets):
    assert helpers.mixed_precision_quantized_paths(_export(targets=targets)) is None


def test_quantized_paths_ignores_entries_that_are_not_strings():
    raw = _export(
        language_layers=[7, "model.layers.0.mlp.gate_proj"],
        targets=[1.5, None, "model.layers.1.mlp.up_proj"],
    )
    assert helpers.mixed_precision_quantized_paths(raw) == frozenset(
        {"model.layers.0.mlp.gate_proj", "model.layers.1.mlp.up_proj"}
    )


def test_covers_matches_exact_and_prefix_paths():
    paths = frozenset({"model.layers.3.mlp.experts", "model.layers.0.mlp.gate_up_proj"})
    assert helpers.mixed_precision_covers(paths, "model.layers.3.mlp.experts.17.down_proj")
    assert helpers.mixed_precision_covers(paths, "model.layers.0.mlp.gate_up_proj")
    assert not helpers.mixed_precision_covers(paths, "model.layers.0.mlp.gate_proj")
    assert not helpers.mixed_precision_covers(paths, "model.layers.30.mlp.shared_experts.gate_proj")


def test_covers_matches_whole_components_not_string_prefixes():
    paths = frozenset({"model.layers.3"})
    assert helpers.mixed_precision_covers(paths, "model.layers.3.mlp.experts.0.up_proj")
    assert not helpers.mixed_precision_covers(paths, "model.layers.30.mlp.experts.0.up_proj")


def test_quantized_paths_and_covers_ignore_case():
    # The first letter stays lower case on purpose: an entry that starts with a capital is a class name.
    paths = helpers.mixed_precision_quantized_paths(_export(hf_layers=["model.layers.5.MLP.Experts"]))
    assert paths == frozenset({"model.layers.5.mlp.experts"})
    assert helpers.mixed_precision_covers(paths, "Model.Layers.5.MLP.Experts.0.gate_proj")


def test_attention_exclusions_are_the_unlisted_groups():
    paths = helpers.mixed_precision_quantized_paths(GLM53_LIKE)
    assert helpers.mixed_precision_attention_exclusions(paths) == frozenset({"q", "kv", "o", "indexer"})
    listed = frozenset({"model.layers.0.self_attn.q_b_proj", "model.layers.0.self_attn.o_proj"})
    assert helpers.mixed_precision_attention_exclusions(listed) == frozenset({"kv", "indexer"})
    assert helpers.mixed_precision_attention_exclusions(frozenset({"model.layers.0.self_attn"})) == frozenset()


@pytest.mark.parametrize("child", ["q_a_layernorm", "kv_a_layernorm", "indexer.k_norm", "q_a"])
def test_attention_exclusions_ignore_children_that_are_not_projections(child):
    listed = frozenset({f"model.layers.0.self_attn.{child}"})
    assert helpers.mixed_precision_attention_exclusions(listed) == frozenset({"q", "kv", "o", "indexer"})


def test_attention_exclusions_read_the_indexer_block_as_the_indexer_group():
    listed = frozenset({"model.layers.0.self_attn.indexer"})
    assert helpers.mixed_precision_attention_exclusions(listed) == frozenset({"q", "kv", "o"})


@pytest.mark.parametrize("projection,group", [("kv_a_proj_with_mqa", "kv"), ("indexer.wq_b", "indexer")])
def test_attention_exclusions_read_the_group_of_a_listed_projection(projection, group):
    listed = frozenset({f"model.layers.0.self_attn.{projection}"})
    assert helpers.mixed_precision_attention_exclusions(listed) == frozenset({"q", "kv", "o", "indexer"}) - {group}


def test_shared_experts_quantized_only_when_listed():
    assert not helpers.mixed_precision_shared_experts_quantized(helpers.mixed_precision_quantized_paths(GLM53_LIKE))
    assert helpers.mixed_precision_shared_experts_quantized(frozenset({"model.layers.4.mlp.shared_experts.gate_proj"}))


def test_shared_experts_quantized_for_the_singular_module_name():
    assert helpers.mixed_precision_shared_experts_quantized(frozenset({"model.layers.4.mlp.shared_expert.up_proj"}))


DENSE_RAW = {
    **GLM53_LIKE,
    "first_k_dense_replace": 3,
    "num_hidden_layers": 78,
    "moe_layer_freq": 1,
    "hidden_act": "silu",
}
GLM53_LAYERS = GLM53_LIKE["hf_quant_config"]["quantization"]["quantized_layers"]
FP8 = {"quant_algo": "FP8"}


def test_dense_groups_follow_the_targets():
    groups = deepseek_v32._dense_mlp_groups(
        DENSE_RAW,
        78,
        common.GEMMQuantMode.fp8_static,
        quantized_paths=helpers.mixed_precision_quantized_paths(DENSE_RAW),
    )
    assert groups == [(3, common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.fp8_static)]
    unlisted = frozenset({"model.layers.0.mlp.gate_up_proj"})  # only layer 0 gate/up listed
    groups = deepseek_v32._dense_mlp_groups(DENSE_RAW, 78, common.GEMMQuantMode.fp8_static, quantized_paths=unlisted)
    assert groups == [
        (1, common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.bfloat16),
        (2, common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.bfloat16),
    ]
    with pytest.raises(ValueError, match="same quantization"):
        deepseek_v32._dense_mlp_groups(
            DENSE_RAW,
            78,
            common.GEMMQuantMode.fp8_static,
            quantized_paths=frozenset({"model.layers.0.mlp.gate_proj"}),
        )


def test_dense_glob_still_wins_over_a_listed_layer():
    raw = {
        **DENSE_RAW,
        "hf_quant_config": {
            "quantization": {**DENSE_RAW["hf_quant_config"]["quantization"], "exclude_modules": ["model.layers.1.*"]}
        },
    }
    groups = deepseek_v32._dense_mlp_groups(
        raw,
        78,
        common.GEMMQuantMode.fp8_static,
        quantized_paths=helpers.mixed_precision_quantized_paths(raw),
    )
    assert sorted(groups) == sorted(
        [
            (2, common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.fp8_static),
            (1, common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.bfloat16),
        ]
    )


def test_shared_experts_excluded_when_unlisted():
    assert deepseek_v32._shared_experts_excluded_from_quant(GLM53_LIKE)
    listed = {
        **GLM53_LIKE,
        "hf_quant_config": {
            "quantization": {"quantized_layers": {"model.layers.4.mlp.shared_experts.gate_proj": FP8}},
        },
    }
    assert not deepseek_v32._shared_experts_excluded_from_quant(listed)


def test_shared_experts_glob_still_excludes_a_listed_module():
    raw = {
        **GLM53_LIKE,
        "hf_quant_config": {
            "quantization": {
                "quantized_layers": {"model.layers.4.mlp.shared_experts.gate_proj": FP8},
                "exclude_modules": ["*.shared_experts*"],
            }
        },
    }
    assert deepseek_v32._shared_experts_excluded_from_quant(raw)


def test_class_name_targets_keep_the_glob_reading():
    raw = {
        "quant_algo": "mixed_precision",
        "hf_quant_config": {"quantization": {"exclude_modules": ["*.shared_experts*"]}},
        "quantization_config": {
            "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}, "targets": ["Linear"]}}
        },
    }
    assert deepseek_v32._shared_experts_excluded_from_quant(raw)  # by the glob, not by omission
    assert helpers.mixed_precision_quantized_paths(raw) is None


# An expert-only requant of a natively FP8 checkpoint keeps the base FP8 header next to a MIXED_PRECISION
# list of its experts, as the DeepSeek-V4 NVFP4 exports do. The lists describe the expert lane only.
DEEPSEEK_V4_LIKE = {
    "quant_algo": "mixed_precision",
    "hf_quant_config": {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for i in range(3, 6)
            },
        }
    },
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
}


@pytest.mark.parametrize("quant_method", ["fp8", "FP8"])
def test_expert_only_requant_of_an_fp8_checkpoint_keeps_the_glob_reading(quant_method):
    raw = deepcopy(DEEPSEEK_V4_LIKE)
    raw["quantization_config"]["quant_method"] = quant_method
    assert helpers.mixed_precision_quantized_paths(raw) is None
    assert deepseek_v32._shared_experts_excluded_from_quant(raw) is False  # no glob names a shared expert


def test_the_same_lists_under_a_modelopt_header_keep_the_targets_reading():
    raw = deepcopy(DEEPSEEK_V4_LIKE)
    raw["quantization_config"]["quant_method"] = "modelopt"
    listed = frozenset(f"model.layers.{i}.mlp.experts" for i in range(3, 6))
    assert helpers.mixed_precision_quantized_paths(raw) == listed
    assert deepseek_v32._shared_experts_excluded_from_quant(raw) is True


def _glm52_model(monkeypatch, quantized_layers, globs=()):
    """GLM-5.2 on SGLang through get_model, its quantization swapped for GLM53_LIKE.

    The export lists ``quantized_layers`` and excludes only ``globs``.
    """
    export = deepcopy(GLM53_LIKE)
    export["hf_quant_config"]["quantization"].update(quantized_layers=quantized_layers, exclude_modules=list(globs))
    export["quantization_config"]["ignore"] = list(globs)
    info = deepcopy(models._get_model_info("nvidia/GLM-5.2-NVFP4"))
    info["raw_config"].update(export)
    monkeypatch.setattr(models, "_get_model_info", lambda model_path: info)
    model_config = config.ModelConfig(tp_size=4, moe_tp_size=4, moe_ep_size=1)
    return models.get_model("nvidia/GLM-5.2-NVFP4", model_config, "sglang")


def test_listed_projections_stay_quantized(monkeypatch):
    attention = {f"model.layers.{i}.self_attn.{child}": FP8 for i in range(78) for child in ("q_b_proj", "o_proj")}
    model = _glm52_model(monkeypatch, {**GLM53_LAYERS, **attention})
    assert model.extra_params["dsa_attn_quant_exclusions"] == frozenset({"kv", "indexer"})


def test_block_level_target_covers_every_group(monkeypatch):
    model = _glm52_model(monkeypatch, {**GLM53_LAYERS, "model.layers.0.self_attn": FP8})
    assert model.extra_params["dsa_attn_quant_exclusions"] == frozenset()


def test_attention_globs_are_unioned_with_the_targets(monkeypatch):
    # The targets leave kv unlisted. The glob keeps o_proj out although a target lists it.
    attention = {f"model.layers.{i}.self_attn.{child}": FP8 for i in range(78) for child in ("q_b_proj", "o_proj")}
    attention.update({f"model.layers.{i}.self_attn.indexer": FP8 for i in range(78)})
    model = _glm52_model(monkeypatch, {**GLM53_LAYERS, **attention}, globs=["*.o_proj"])
    assert model.extra_params["dsa_attn_quant_exclusions"] == frozenset({"kv", "o"})


def test_create_reads_dense_and_shared_expert_precision_from_the_targets(monkeypatch):
    layer_zero_only = {
        path: entry
        for path, entry in GLM53_LAYERS.items()
        if not path.startswith(("model.layers.1.", "model.layers.2."))
    }
    model = _glm52_model(monkeypatch, layer_zero_only)
    assert model.extra_params["dsa_dense_mlp_groups"] == [
        (1, common.GEMMQuantMode.fp8_static, common.GEMMQuantMode.fp8_static),
        (2, common.GEMMQuantMode.bfloat16, common.GEMMQuantMode.bfloat16),
    ]
    assert model.extra_params["dsa_shared_expert_quant_mode"] == common.GEMMQuantMode.bfloat16
