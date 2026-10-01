# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct the archived native ReplaySpec for an independent fresh replay."""

from copy import deepcopy

from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import BackendDeploymentSpec, ForwardPassEstimatorSpec, ReplaySpec


def replay_spec_from_dict(payload):
    value = deepcopy(payload)
    deployment = value["backend_deployment"]
    if deployment.get("encoder") is not None:
        raise ValueError("This experiment does not use an encoder pool")
    deployment["forward_pass_estimators"] = {
        role: ForwardPassEstimatorSpec(**entry)
        for role, entry in deployment.get("forward_pass_estimators", {}).items()
    }
    value["backend_deployment"] = BackendDeploymentSpec(**deployment)
    value["adapters"] = {
        name: AdapterReplaySpec(
            config=entry["config"],
            runtime_hooks=tuple(RuntimeHookSpec(**hook) for hook in entry.get("runtime_hooks", [])),
        )
        for name, entry in value.get("adapters", {}).items()
    }
    return ReplaySpec(**value)
