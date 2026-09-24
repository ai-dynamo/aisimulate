# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shlex

import pytest
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.runner import _render_cell

from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS

pytestmark = pytest.mark.unit


def plan(tmp_path, backend, model):
    points = tmp_path / "points.json"
    points.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 1024, "total_kv_read_tokens": 0}],
                "decode": [{"batch_size": 1, "total_kv_read_tokens": 1024}],
            }
        )
    )
    args = _parser().parse_args(
        [
            "--backend",
            backend,
            "--model-path",
            model,
            "--gpu",
            "gb300",
            "--fpm-max-gpus",
            "4",
            "--fpm-gpu-counts",
            "2",
            "4",
            "--fpm-parallel-presets",
            "pure_tp",
            "--fpm-max-model-len",
            "131072",
            "--fpm-benchmark-points-file",
            str(points),
        ]
    )
    return build_collection_plan(
        backend=backend,
        model_path=model,
        system="gb300",
        selected_ops=set(),
        options=FPMCollectionOptions.from_args(args),
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("model", list(MODEL_REVISIONS))
def test_all_required_glm_deployments_render_native_precision_and_scope(tmp_path, backend, model):
    campaign = plan(tmp_path, backend, model)
    assert len(campaign.cells) == 4
    frozen = campaign.to_dict()
    assert frozen["options"]["warmup_repeats"] == 5
    assert frozen["options"]["measurement_repeats"] == 10
    assert frozen["options"]["graph_policy"] == "native_graph_policy"
    for cell in campaign.cells:
        assert cell.topology.tp in (2, 4)
        assert (cell.topology.pp, cell.topology.dp, cell.topology.cp, cell.topology.moe_ep) == (1, 1, 1, 1)
        assert cell.topology.moe_tp == cell.topology.tp
        assert cell.gemm_quant_mode == ("nvfp4" if "NVFP4" in model else "fp8_block")
        directory = tmp_path / cell.cell_id
        directory.mkdir()
        _render_cell(campaign, cell, directory, {"generator_dynamo_version": "1.3.0"})
        script = (directory / "run.sh").read_text()
        command = next(line for line in script.splitlines() if line.startswith("engine_command=("))
        argv = shlex.split(command.removeprefix("engine_command=(").removesuffix(")"))
        assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
        assert argv[argv.index("--revision") + 1] == MODEL_REVISIONS[model]
        tp_flag = "--tensor-parallel-size" if backend == "vllm" else "--tp-size"
        assert argv[argv.index(tp_flag) + 1] == str(cell.topology.tp)
        assert "--enable-expert-parallel" not in argv
        if backend == "vllm":
            assert "--cudagraph-metrics" in argv and "--language-model-only" in argv
            assert "--no-enable-prefix-caching" in argv
            assert "--compilation-config" not in argv
        else:
            assert "collector.fpm_forward.sglang_driver" in argv
            assert "--disable-radix-cache" in argv
            assert argv[argv.index("--context-length") + 1] == "131079"
            assert argv[argv.index("--benchmark-max-context-length") + 1] == "131072"
            assert "--max-model-len" not in argv
            assert "--enable-mixed-chunk" not in argv
            assert argv[argv.index("--moe-runner-backend") + 1] == "auto"
            assert "--cuda-graph-bs" not in argv
            graph_start = argv.index("--cuda-graph-bs-decode") + 1
            graph_sizes = []
            for value in argv[graph_start:]:
                if value.startswith("--"):
                    break
                graph_sizes.append(int(value))
            assert graph_sizes == list(range(1, 33))
            if "NVFP4" in model:
                assert argv[argv.index("--quantization") + 1] == "modelopt_fp4"
