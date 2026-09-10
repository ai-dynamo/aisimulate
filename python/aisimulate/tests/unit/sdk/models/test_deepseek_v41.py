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
