# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment startup inputs survive the public collection command."""

import json

import pytest

from aisimulate.support.fpm import fpm_cli_args
from aisimulate.support.schema import FPMDeployment, SupportRequest

pytestmark = pytest.mark.unit


def test_environment_reaches_formal_generator_arguments(tmp_path):
    request = SupportRequest.model_validate(
        {
            "identity": {
                "model": "fixture/model",
                "model_revision": "a" * 40,
                "model_kind": "moe",
                "framework": "vllm",
                "framework_version": "0.28.0",
                "gpu": "gb300",
                "interconnect": "nvlink",
            },
        }
    )
    env = [{"name": "PYTHONPATH", "value": "/runtime/dynamo:/runtime/wheel"}]
    deployment = FPMDeployment(executor="slurm", image="fixture.sqsh", extra_env=env)
    command = fpm_cli_args(request, output_dir=tmp_path, plan_only=True, deployment=deployment)
    assert "K8sConfig.extra_env=" + json.dumps(env) in command
    assert "--extra-env" not in command
    assert "extra_env" not in FPMDeployment().model_dump(exclude_none=True)


@pytest.mark.parametrize(
    "env",
    [
        [{"name": "BAD-NAME", "value": "x"}],
        [{"name": "PYTHONPATH", "value": "x"}, {"name": "PYTHONPATH", "value": "y"}],
        [{"name": "PYTHONPATH", "value": "x", "unknown": "x"}],
        [{"name": "PYTHONPATH", "value": "x\x00y"}],
    ],
)
def test_deployment_rejects_ambiguous_environment(env):
    with pytest.raises(ValueError, match="extra_env"):
        FPMDeployment(extra_env=env)
