# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.deepseek_v41 import (
    MODEL_PATH,
    DeepSeekV41Config,
    V41RequestWorkload,
    resolve_execution_profile,
    stage_workloads,
)
from aiconfigurator_core.sdk.models.helpers import _infer_quant_modes_from_raw_config
from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

pytestmark = pytest.mark.unit


@pytest.fixture
def descriptor():
    return get_model_config_from_model_path(MODEL_PATH)["extra_params"]


def test_real_v41_config_and_quant(descriptor):
    info = get_model_config_from_model_path(MODEL_PATH)
    assert isinstance(descriptor, DeepSeekV41Config)
    assert (info["layers"], info["hidden_size"], info["n"]) == (40, 5120, 64)
    assert len(descriptor.compress_ratios) == 40
    assert Counter(descriptor.layer_role(i) for i in range(40)) == {"swa": 2, "full": 4, "reindex": 4, "reuse": 30}
    quant = _infer_quant_modes_from_raw_config(info["raw_config"], info["architecture"])
    assert quant["moe_quant_mode"] == common.MoEQuantMode.w4a8_mxfp4_mxfp8


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
@pytest.mark.parametrize("total_gpus", [4, 32])
def test_default_task_enumerates_matching_moe_parallel_widths(backend, total_gpus):
    from aiconfigurator.sdk.task_v2 import Task

    task = Task(
        model_path=MODEL_PATH,
        system_name="gb300",
        backend_name=backend,
        total_gpus=total_gpus,
        database_mode="SOL",
        isl=1024,
        osl=128,
        nextn=0,
    )
    assert task.model_family == "DEEPSEEKV41"
    assert task.is_moe
    parallel = [tuple(choice) for choice in task.iter_parallel("agg")]
    assert (4, 1, 1, 1, 4, 1) in parallel
    assert all(tp * dp * cp == moe_tp * moe_ep for tp, _, dp, moe_tp, moe_ep, cp in parallel)


def test_shared_pool_memory_slope_and_odd_decode(descriptor):
    assert descriptor.compressed_entry_bytes == 288
    assert descriptor.index_entry_bytes == 68
    assert descriptor.kvcache_bytes(130) - descriptor.kvcache_bytes(128) == 2 * 890
    # Only the ratio-one owner grows on odd tokens; the three ratio-two owners
    # publish their next compressed entry on the following even token.
    assert descriptor.kvcache_bytes(129) - descriptor.kvcache_bytes(128) == 356
    assert descriptor.engram_table_bytes(1) == 202758032400


def test_replay_tail_is_bounded_per_actual_request(descriptor):
    requests = (V41RequestWorkload(256, 1024), V41RequestWorkload(3, 4096))
    full = stage_workloads(descriptor, "full", requests, 39)
    assert [(r.query_tokens, r.prefix_tokens) for r in full] == [(256, 1024), (3, 4096)]
    tail = stage_workloads(descriptor, "decoder_bounded", requests, 21)
    assert [(r.query_tokens, r.prefix_tokens) for r in tail] == [(128, 1152), (3, 4096)]
    assert stage_workloads(descriptor, "decoder_bounded", requests, 20) == full


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
def test_unverified_replay_backend_fails(backend):
    assert resolve_execution_profile(False, backend) == "full"
    with pytest.raises(NotImplementedError, match="not verified"):
        resolve_execution_profile(True, backend)


def _build_model(*, replay=False, backend="sglang", tp=4):
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import get_model

    return get_model(
        MODEL_PATH,
        ModelConfig(tp_size=tp, pp_size=1, attention_dp_size=1, moe_tp_size=1, moe_ep_size=tp, decoder_replay=replay),
        backend,
    )


@pytest.mark.parametrize("backend", ["sglang", "vllm", "trtllm"])
def test_v41_text_graph_full_profile_all_backends(backend):
    import json

    model = _build_model(backend=backend)
    assert model.execution_profile == "full"
    assert not model.encoder_ops
    specs = [json.loads(op._spec_json()) for op in model.context_ops]
    stages = [op["Dsv41Stage"] for op in specs if "Dsv41Stage" in op]
    assert len(stages) == 40
    assert sum(stage["bounded"] for stage in stages) == 19
    assert all(not stage["decoder_replay"] for stage in stages)
    roles = Counter(
        next(c["Dsv41Attention"]["role"] for c in stage["children"] if "Dsv41Attention" in c) for stage in stages
    )
    assert roles == {"swa": 2, "full": 4, "reindex": 4, "reuse": 30}


def test_replay_keeps_inventory_and_kv_pool_capacity():
    full, replay = _build_model(), _build_model(replay=True)
    assert replay.execution_profile == "decoder_bounded"
    assert full.get_resident_weights_bytes() == replay.get_resident_weights_bytes()
    assert full.get_resident_weights_bytes() > full.extra_params.engram_table_bytes(4)
    assert full.get_kvcache_bytes_per_sequence(4096) == replay.get_kvcache_bytes_per_sequence(4096)
    assert replay.get_kvcache_max_tokens(replay.get_kvcache_bytes_per_sequence(4096)) == 4096


def test_batch_capacity_reserves_each_request_window(descriptor):
    model = _build_model()
    fixed = 40 * 128 * 512 + 3 * 2 * 2 * 512 * 4
    assert model.get_kvcache_batch_capacity(4 * fixed + 4096 * 890, 4) == 4096
    assert model.get_kvcache_batch_capacity(4 * fixed - 1, 4) == 0
    assert model.get_additional_activation_bytes(1024) > 1024 * 4 * 5120 * 2


@pytest.mark.parametrize(("dp", "pp"), [(2, 1), (1, 2)])
def test_unmodeled_parallel_cache_ownership_fails(dp, pp):
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import get_model

    with pytest.raises(NotImplementedError, match="DP Engram collectives and PP cache ownership"):
        get_model(
            MODEL_PATH,
            ModelConfig(tp_size=4, pp_size=pp, attention_dp_size=dp, moe_tp_size=1, moe_ep_size=4 * dp),
            "sglang",
        )


def test_native_sol_replay_decode_and_short_extend_contract():
    from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine

    def engine(replay):
        return EngineHandle(
            compile_engine(
                MODEL_PATH,
                "gb300",
                "sglang",
                tp_size=4,
                moe_tp_size=1,
                moe_ep_size=4,
                decoder_replay=replay,
                database_mode="SOL",
            )
        )

    full, replay = engine(False), engine(True)
    assert replay.predict_prefill_latency(1, 1024) < full.predict_prefill_latency(1, 1024)
    assert replay.predict_decode_latency(1, 1024) == full.predict_decode_latency(1, 1024)
    # Both modes process the same three actual query tokens; the bounded SWA
    # intentionally does not read the pre-existing ring prefix.
    assert replay.predict_prefill_latency(1, 4099, 4096) > 0
    assert replay.mixed_step_latency(2048, 1, 1024, 2) > 0


def test_native_block32_shared_projections_have_distinct_perf_identity():
    import json

    model = _build_model()
    first_layer = next(
        json.loads(op._spec_json())["Dsv41Stage"]
        for op in model.context_ops
        if "Dsv41Stage" in json.loads(op._spec_json())
    )
    linears = [c["Dsv41Linear"] for c in first_layer["children"] if "Dsv41Linear" in c]
    assert {(op["n"], op["k"]) for op in linears} == {(1152, 5120), (5120, 576)}
    assert all(c["Gemm"]["quant_mode"] == "bfloat16" for c in first_layer["children"] if "Gemm" in c)
