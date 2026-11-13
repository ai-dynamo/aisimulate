# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks of the capture harness: smoke gate and server command.

Run with: uv run --with pytest python -m pytest -c /dev/null test_collect.py
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import collect

E2E_COMMAND_TAIL = ["--worker-extension-cls", "receipt_probe.CampaignCacheProbe"]


def _write(tmp_path, records, endpoint_stats=None):
    steps = tmp_path / "steps.jsonl.gz"
    with gzip.open(steps, "wt") as stream:
        for seq, (prefill, decode, wall, lists) in enumerate(records, 1):
            sched = {"num_prefill_requests": prefill, "num_decode_requests": decode}
            if lists:
                n = prefill + decode
                sched.update(extend_lengths=[1] * n, past_kv_lengths=[10] * n)
            stream.write(json.dumps({"seq": seq, "fpm": {"wall_time": wall, "scheduled_requests": sched}}) + "\n")
    stats = {"messages": len(records), "missing_messages": 0, "out_of_order": 0, "undecodable": 0}
    stats.update(endpoint_stats or {})
    summary = tmp_path / "steps.jsonl.gz.summary.json"
    summary.write_text(json.dumps({"endpoints": {"tcp://127.0.0.1:20380": stats}}))
    return steps, summary


def test_smoke_check_passes_with_lists_and_both_kinds(tmp_path):
    receipt = collect.check_steps(*_write(tmp_path, [(0, 2, 0.01, True), (1, 2, 0.05, True), (0, 0, 0.0, False)]))
    assert receipt["status"] == "passed"
    assert receipt["steps"] == 2 and receipt["step_kinds"] == {"prefill": 0, "decode": 1, "mixed": 1}


@pytest.mark.parametrize(
    ("records", "stats", "failed"),
    [
        ([(0, 2, 0.01, False), (1, 2, 0.05, False)], None, "per_request_lists"),
        ([(0, 2, 0.01, True), (0, 3, 0.01, True)], None, "decode_and_mixed_steps"),
        ([(0, 2, 0.01, True), (1, 2, 0.05, True)], {"missing_messages": 3}, "no_sequence_gaps"),
    ],
)
def test_smoke_check_fails(tmp_path, records, stats, failed):
    receipt = collect.check_steps(*_write(tmp_path, records, stats))
    assert receipt["status"] == "failed" and receipt["checks"][failed] is False


def test_server_command_is_the_e2e_command_plus_the_scheduler():
    command = collect.server_command(Path("/model"))
    assert command[-2:] == ["--scheduler-cls", collect.SCHEDULER]
    assert command[-4:-2] == E2E_COMMAND_TAIL
    assert "--no-async-scheduling" in command and "--language-model-only" in command


def test_server_env_puts_the_hook_directory_first(tmp_path):
    env = collect.server_env(Path("/code"), Path("/aisim/src"), Path("/dynamo-site"), tmp_path / "run", "seed42-l4")
    assert env["PYTHONPATH"].split(":") == ["/aisim/src/aisim_core/fpm_hooks", "/code", "/aisim/src", "/dynamo-site"]
    assert env["DYN_FORWARDPASS_METRIC_PORT"] == str(collect.FPM_PORT)
