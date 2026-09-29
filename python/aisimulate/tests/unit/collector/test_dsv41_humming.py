# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from types import SimpleNamespace

import pytest
from collector.sglang.dsv41_humming import loaded_humming_geometry, validate_humming_observations

pytestmark = pytest.mark.unit


def native_experts():
    # Native H100 TP4 metadata observed at SGLang 1aa0e962: local576 becomes
    # physical640 in Humming; the collector must preserve the native padding.
    precision = dict(
        a_dtype="bfloat16",
        b_dtype="float4e2m1",
        c_dtype="bfloat16",
        bs_dtype="float8e8m0",
        as_dtype=None,
        input_scale_group_size=0,
        weight_scale_group_size=32,
        weight_scale_group_size_n=0,
        weight_scale_type="group",
        weight_scale_2_type="none",
        num_experts=384,
    )
    metas = dict(
        w13=dict(shape_n=1280, shape_k=5120, pad_shape_n=128, pad_shape_k=0),
        w2=dict(shape_n=5120, shape_k=640, pad_shape_n=0, pad_shape_k=64),
    )
    return SimpleNamespace(
        moe_tp_size=4,
        _dsv4_mxfp4_backend="humming",
        input_schemas={
            name: SimpleNamespace(a_dtype=None, input_scale_dtype=None, input_scale_group_size=0) for name in metas
        },
        humming_metas={
            name: SimpleNamespace(_config_str=json.dumps({**precision, **shape})) for name, shape in metas.items()
        },
    )


def test_native_padding_preserves_logical_v41_shape():
    experts = native_experts()
    before = copy.deepcopy(experts.humming_metas)
    geometry = loaded_humming_geometry(experts)
    assert geometry["w2"]["shape_k"] == 640
    assert geometry["w2"]["shape_k"] - geometry["w2"]["pad_shape_k"] == 576
    assert experts.humming_metas == before


@pytest.mark.parametrize("mismatch", ["schema", "precision", "logical_shape", "unloaded"])
def test_loaded_humming_rejects_incompatible_execution(mismatch):
    experts = native_experts()
    if mismatch == "schema":
        experts.input_schemas["w2"].a_dtype = "float8e4m3"
    elif mismatch == "unloaded":
        experts._dsv4_mxfp4_backend = None
    else:
        meta = json.loads(experts.humming_metas["w2"]._config_str)
        meta["a_dtype" if mismatch == "precision" else "pad_shape_k"] = "float8e4m3" if mismatch == "precision" else 0
        experts.humming_metas["w2"]._config_str = json.dumps(meta)
    with pytest.raises(RuntimeError, match="native baseline Humming"):
        loaded_humming_geometry(experts)


@pytest.mark.parametrize("mismatch", [None, "missing_projection", "fp8_operand", "scale", "half_accum", "two_runners"])
def test_admission_requires_actual_unquantized_bf16_operands(mismatch):
    calls = [
        dict(
            sublayer=name,
            input_dtype="torch.bfloat16",
            has_input_scale=False,
            compute_config={"gemm_type": "indexed", "use_f16_accum": False},
        )
        for name in ("w13", "w2")
    ]
    configs = [dict(core_id=12, config={})]
    if mismatch == "missing_projection":
        calls.pop()
    elif mismatch == "fp8_operand":
        calls[1]["input_dtype"] = "torch.float8_e4m3fn"
    elif mismatch == "scale":
        calls[1]["has_input_scale"] = True
    elif mismatch == "half_accum":
        calls[0]["compute_config"]["use_f16_accum"] = True
    elif mismatch == "two_runners":
        configs.append(dict(core_id=13, config={}))
    if mismatch is None:
        validate_humming_observations(calls, configs)
    else:
        with pytest.raises(RuntimeError, match="native Humming qualification"):
            validate_humming_observations(calls, configs)
