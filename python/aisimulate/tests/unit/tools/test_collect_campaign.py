# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/perf_database/collect_campaign.py — the invariants a hand-rolled launcher drifted on (sm120, 2026-09-30)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "collect_campaign.py"


def _load():
    spec = importlib.util.spec_from_file_location("collect_campaign", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_campaign"] = mod
    spec.loader.exec_module(mod)
    return mod


def _plan(tmp_path, shards):
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"backend": "vllm", "sm": 120, "shards": shards}))
    return p


def test_collector_argv_always_resumes_and_never_keeps_csv(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16", "case_filter": "['bfloat16', "}, {"op": "kda"}]))
    for shard in plan.shards:
        argv = c.collector_argv(plan, shard)
        assert "--resume" in argv and "--keep-csv" not in argv
        assert argv[argv.index("--sm") + 1] == "120"
        assert argv[argv.index("--ops") + 1] == shard.op
    assert "--case-filter" in c.collector_argv(plan, plan.shards[0])
    assert "--case-filter" not in c.collector_argv(plan, plan.shards[1])
    assert plan.shards[1].name == "all"


def test_one_namespace_per_shard_and_readonly_checkout(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16"}, {"op": "gemm", "name": "fp8"}]))
    dirs = {c.shard_dir(tmp_path, s) for s in plan.shards}
    assert len(dirs) == 2
    argv = c.docker_argv(
        plan, plan.shards[0], image="img", gpu=3, checkout=Path("/ck"), shard_dir=dirs.pop(), container_name="x"
    )
    assert '"device=3"' in argv and "/ck:/ais:ro" in argv and argv[argv.index("-w") + 1] == "/out"


def test_duplicate_shard_keys_are_rejected(tmp_path):
    c = _load()
    with pytest.raises(ValueError):
        c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "a"}, {"op": "gemm", "name": "a"}]))


def test_image_is_the_manifest_pin_plus_finalize_deps():
    c = _load()
    base = c.pinned_image("vllm", "gemm")
    assert base.startswith("vllm/vllm-openai:")  # framework_manifest default pin, never a hand-typed tag
    df = c.dockerfile(base)
    assert df.startswith(f"FROM {base}\n") and "pyarrow" in df and "pandas" in df
    assert c.collect_image_tag(base).startswith("aisim-collect:")


def test_shard_state_reads_the_namespace(tmp_path):
    c = _load()
    plan = c.load_plan(_plan(tmp_path, [{"op": "gemm", "name": "bf16"}]))
    s = plan.shards[0]
    assert c.shard_state(tmp_path, s) == "pending"
    d = c.shard_dir(tmp_path, s)
    (d / ".ckpt").mkdir(parents=True)
    assert c.shard_state(tmp_path, s) == "started"
    (d / "DONE").touch()
    assert c.shard_state(tmp_path, s) == "done-no-parquet"  # the --keep-csv failure mode, picked up by `finalize`
    (d / "gemm_perf.parquet").write_bytes(b"")
    assert c.shard_state(tmp_path, s) == "finalized"
