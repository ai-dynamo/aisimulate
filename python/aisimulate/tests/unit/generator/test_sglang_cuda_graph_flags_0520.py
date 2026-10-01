# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang >= 0.5.20 renamed `--cuda-graph-bs` / `--cuda-graph-max-bs` to the
`*-decode` spellings (ServerArgs is a msgspec Struct; the old aliases are gone,
0.5.21 rejects `--cuda-graph-bs` as ambiguous). cli_args.0.5.20.j2 renders the
new spelling; older versions keep the old one (op-probe harness, 2026-10-01)."""
import copy

import pytest
import yaml

from aisimulate.generator.api import generate_backend_artifacts

pytestmark = pytest.mark.unit

_PARAMS = {
    "ServiceConfig": {
        "model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_path": "deepseek-ai/DeepSeek-V3",
        "served_model_name": "DeepSeek-V3",
        "include_frontend": True,
    },
    "K8sConfig": {"name_prefix": "test", "k8s_namespace": "default"},
    "DynConfig": {"mode": "agg"},
    "WorkerConfig": {"agg_workers": 1, "agg_gpus_per_worker": 8, "prefill_workers": 0, "decode_workers": 0},
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 1024, "osl": 256},
    "ModelConfig": {"is_moe": True, "prefix": 0, "nextn": 0},
    "BenchConfig": {},
    "params": {
        "agg": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "moe_expert_parallel_size": 8,
            "max_batch_size": 64,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
        }
    },
}


def _worker_args(version: str) -> str:
    artifacts = generate_backend_artifacts(copy.deepcopy(_PARAMS), "sglang",
                                           backend_version=version, deployment_target="dynamo-j2")
    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    worker = next(svc for name, svc in k8s["spec"]["services"].items() if name != "Frontend")
    return " ".join(worker["extraPodSpec"]["mainContainer"]["args"])


@pytest.mark.parametrize("version", ["0.5.20", "0.5.21"])
def test_new_versions_render_decode_spelling(version):
    args = _worker_args(version)
    assert "--cuda-graph-bs-decode " in args
    assert "--cuda-graph-bs " not in args
    assert "--cuda-graph-max-bs " not in args


@pytest.mark.parametrize("version", ["0.5.16", "0.5.19"])
def test_old_versions_keep_old_spelling(version):
    args = _worker_args(version)
    assert "--cuda-graph-bs " in args
    assert "--cuda-graph-bs-decode" not in args
