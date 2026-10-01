# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import psutil
import pytest
from check_readme_commands import ROOT, assert_outputs, read_blocks, run_shell


def test_every_readme_command_is_registered():
    manifest = json.loads((ROOT / "scripts/readme_commands.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert len(read_blocks(readme, manifest)) == len(manifest)
    with pytest.raises(ValueError, match="needs a readme-check"):
        read_blocks(readme + "\n```bash\nunregistered\n```\n", manifest)
    with pytest.raises(ValueError, match="Manifest drift"):
        read_blocks(readme, {**manifest, "stale": {}})


def test_no_requests_is_failure_even_with_valid_json(tmp_path):
    (tmp_path / "prediction.json").write_text('{"completed_requests": 0}')
    with pytest.raises(ValueError, match="no requests"):
        assert_outputs({"prediction": "prediction.json"}, tmp_path, tmp_path / "log")


def test_unresolved_recommendation_is_failure(tmp_path):
    (tmp_path / "0001.yaml").write_text("engine:\n  backend:\n    choices: [vllm, sglang]\n")
    with pytest.raises(ValueError, match="Unresolved"):
        assert_outputs({"recommendations": "."}, tmp_path, tmp_path / "log")


def test_nonzero_and_timeout_are_distinct(tmp_path):
    assert run_shell("exit 7", tmp_path, dict(os.environ), tmp_path / "failed", 10) == (
        7,
        False,
    )
    code, timeout = run_shell("sleep 20", tmp_path, dict(os.environ), tmp_path / "timeout", 0.3)
    assert timeout and code != 0


def test_timeout_kills_workers_in_new_sessions(tmp_path):
    command = (
        "python3 -c 'import subprocess,time; "
        'p=subprocess.Popen(["sleep","60"], start_new_session=True); '
        'open("pid","w").write(str(p.pid)); time.sleep(60)' + "'"
    )
    _, timeout = run_shell(command, tmp_path, dict(os.environ), tmp_path / "log", 1)
    assert timeout
    pid = int((tmp_path / "pid").read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_request_capture_must_match_override(tmp_path):
    (tmp_path / "requests.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete"):
        assert_outputs({"requests": "requests.jsonl", "completed_requests": 4}, tmp_path, tmp_path / "log")
