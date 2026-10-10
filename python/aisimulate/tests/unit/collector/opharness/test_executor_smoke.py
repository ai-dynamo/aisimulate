# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""executor_smoke judges collect.py's real path by its artifacts and checks a
pipeline shard plan against the collector's case plan without a GPU."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture()
def es():
    spec = importlib.util.spec_from_file_location("executor_smoke_t", COMPONENTS / "executor_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_judge_run_statuses(es, tmp_path):
    d = tmp_path / "op"
    d.mkdir()
    assert es.judge_run(0, d)["status"] == "fail"            # exit 0 but nothing staged, nothing reported
    (d / "errors_sglang.moe.json").write_text(json.dumps([{"e": 1}, {"e": 2}]))
    v = es.judge_run(0, d)
    assert v["status"] == "all_failed" and v["errors"] == 2   # int4_wo / job 469988017 shape
    (d / "moe_perf.parquet").write_bytes(b"")
    assert es.judge_run(0, d)["status"] == "ok"
    assert es.judge_run(1, d)["status"] == "fail"


def test_shard_ops_dedups_shards_of_one_op(es, tmp_path):
    p = tmp_path / "shards.yaml"
    p.write_text(yaml.safe_dump({"shards": [{"op": "gemm", "name": "bf16"}, {"op": "gemm", "name": "fp8"},
                                            {"op": "dsv4_csa_attn_module", "name": "all"}]}))
    assert es.shard_ops(p) == ["gemm", "dsv4_csa_attn_module"]


def test_not_in_plan_message_is_recognised(es):
    m = es.NOT_IN_PLAN_RX.search("collect.py: error: Requested ops are not present in the collector v2 case plan: a, b")
    assert m and m.group(1) == "a, b"


def test_plan_check_reproduces_the_pipeline_refusal_on_a_cpu(es):
    """The three research-only dsv4 ops the upstream shard plan carried are refused by
    --model-cases-full (case file: 'do not schedule them in the default model plan');
    gemm is in the plan. No framework, no GPU."""
    assert es.plan_check_one("sglang", "dsv4_csa_attn_module", python=sys.executable)["status"] == "not_in_plan"
    assert es.plan_check_one("sglang", "gemm", python=sys.executable)["status"] == "in_plan"
