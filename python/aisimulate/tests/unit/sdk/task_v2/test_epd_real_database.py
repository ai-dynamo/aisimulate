# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD single-point evaluation against a real packaged perf database.

Synthetic databases route the engine step to the Python path, so only a
real database exercises the default compiled-engine path where the encode
worker's ``EncoderOnlyModel`` must satisfy the full-engine spec contract.
"""

import json

import pytest

from aisimulate.sdk import config
from aisimulate.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
_SYSTEM = "h200_sxm"
_BACKEND = "sglang"
_WORKLOAD = dict(
    isl=2048,
    osl=256,
    image_height=768,
    image_width=768,
    num_images_per_request=2,
)


def test_encoder_only_model_satisfies_engine_spec_contract():
    from aisimulate_core.sdk.engine import build_engine_spec_json
    from aisimulate_core.sdk.models.vit_ops import EncoderOnlyModel

    model = EncoderOnlyModel(
        encoder_ops=[],
        encoder_config=None,
        config=config.ModelConfig(tp_size=2, enable_encoder_dp=False),
    )
    spec = json.loads(
        build_engine_spec_json(
            model,
            model_path="epd-encoder-only",
            system=_SYSTEM,
            backend=_BACKEND,
            backend_version=None,
            kv_block_size=None,
            systems_path=None,
            nextn=0,
        )
    )
    assert spec["context_ops"] == []
    assert spec["generation_ops"] == []


@pytest.mark.parametrize("engine_step_backend", [None, "rust"])
def test_run_single_agg_epd_real_database(engine_step_backend):
    task = Task(
        serving_mode="agg",
        model_path=_MODEL,
        system_name=_SYSTEM,
        backend_name=_BACKEND,
        enable_epd=True,
        engine_step_backend=engine_step_backend,
        **_WORKLOAD,
    )
    row = task.run_single_agg(tp=1, batch_size=1, encoder_tp=1)
    assert row["(e)workers"] == 1
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]


def test_run_single_agg_epd_with_nested_kimi_k3_vision_config():
    task = Task(
        serving_mode="agg",
        model_path="moonshotai/Kimi-K3",
        system_name="b200_sxm",
        backend_name="trtllm",
        database_mode="SOL",
        enable_epd=True,
        isl=128,
        osl=2,
        image_height=224,
        image_width=224,
        num_images_per_request=1,
        engine_step_backend="rust",
    )
    row = task.run_single_agg(tp=16, moe_tp=16, batch_size=1, encoder_tp=1)

    assert (row["(e)workers"], row["(a)workers"]) == (1, 1)
    assert (row["(e)tp"], row["(e)bs"]) == (1, 1)
    assert row["num_total_gpus"] == 17
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]
    assert row["encoder_memory"] == 0  # The language worker no longer hosts the encoder.


@pytest.mark.parametrize("engine_step_backend", [None, "rust"])
def test_run_single_disagg_epd_real_database(engine_step_backend):
    task = Task(
        serving_mode="disagg",
        prefill_model_path=_MODEL,
        decode_model_path=_MODEL,
        prefill_system_name=_SYSTEM,
        decode_system_name=_SYSTEM,
        prefill_backend_name=_BACKEND,
        decode_backend_name=_BACKEND,
        enable_epd=True,
        engine_step_backend=engine_step_backend,
        **_WORKLOAD,
    )
    row = task.run_single_disagg(
        prefill_tp=1,
        decode_tp=1,
        decode_batch_size=8,
        encoder_tp=1,
    )
    assert row["(e)workers"] == 1
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]


@pytest.mark.parametrize("video", [False, True])
def test_run_single_disagg_epd_kimi_k3_keeps_vision_off_language_workers(video):
    visual_fields = (
        dict(num_images_per_request=0, num_videos_per_request=1, video_frames=4, video_height=224, video_width=224)
        if video
        else dict(image_height=224, image_width=224, num_images_per_request=1)
    )
    task = Task(
        serving_mode="disagg",
        prefill_model_path="moonshotai/Kimi-K3",
        decode_model_path="moonshotai/Kimi-K3",
        prefill_system_name="b200_sxm",
        decode_system_name="b200_sxm",
        prefill_backend_name="trtllm",
        decode_backend_name="trtllm",
        database_mode="SOL",
        enable_epd=True,
        isl=128,
        osl=2,
        engine_step_backend="rust",
        **visual_fields,
    )
    # The language model supports TP16; the 12-head vision tower does not.
    # Both language workers must omit that separately hosted tower.
    row = task.run_single_disagg(
        prefill_tp=16,
        prefill_moe_tp=16,
        decode_tp=16,
        decode_moe_tp=16,
        decode_batch_size=2,
        encoder_tp=4,
        encoder_batch_size=2,
    )

    assert (row["(e)workers"], row["(p)workers"], row["(d)workers"]) == (1, 1, 1)
    assert (row["(e)tp"], row["(p)tp"], row["(d)tp"]) == (4, 16, 16)
    assert row["(e)bs"] == 2
    assert row["num_total_gpus"] == 36
    assert row["encoder_latency"] > 0
    assert row["ttft"] > row["encoder_latency"]
    assert row["tpot"] > 0
    assert row["seq/s"] > 0
    assert row["(e)memory"] > 0
