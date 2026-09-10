# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.utils import get_model_config_from_model_path
from tools.support_matrix.support_matrix import (
    SUPPORT_MATRIX_IMAGE_WORKLOAD,
    SupportMatrix,
    TestConstraints,
    _get_encoder_coverage,
    _support_matrix_row_command,
)

pytestmark = pytest.mark.unit

SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
MAVERICK = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
CONSTRAINTS = TestConstraints(total_gpus=32, isl=256, osl=256, prefix=128, ttft=2000.0, tpot=50.0)


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_llama4_matrix_recognizes_implemented_encoder(model_id):
    coverage = _get_encoder_coverage(model_id)

    assert coverage.checkpoint_declares_encoder
    assert coverage.aic_encoder_implemented
    assert coverage.workload == SUPPORT_MATRIX_IMAGE_WORKLOAD


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_llama4_matrix_tasks_enable_nonzero_image_work(mode, model_id):
    workload = _get_encoder_coverage(model_id).workload
    task = SupportMatrix._create_task(
        mode=mode,
        model=model_id,
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        constraints=CONSTRAINTS,
        image_workload=workload,
    )

    assert task.image_height == 1024
    assert task.image_width == 1024
    assert task.num_images_per_request == 1
    enc = get_model_config_from_model_path(model_id)["extra_params"].vision_config
    runtime = RuntimeConfig(
        image_height=task.image_height, image_width=task.image_width, num_images_per_request=task.num_images_per_request
    )
    resolved = BaseBackend._encoder_workload_per_visual(runtime, enc)
    assert resolved.sequences_per_image == 17
    assert BaseBackend._visual_context_tokens_from_encoder_config(enc, runtime) == 17 * 144 + 16 + 3


@pytest.mark.parametrize("model_id", [SCOUT, MAVERICK])
def test_llama4_replay_command_cannot_skip_encoder(model_id):
    workload = _get_encoder_coverage(model_id).workload
    command = _support_matrix_row_command(
        model=model_id,
        system="h200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        constraints=CONSTRAINTS,
        image_workload=workload,
    )

    assert "--image-height 1024" in command
    assert "--image-width 1024" in command
    assert "--num-images 1" in command
