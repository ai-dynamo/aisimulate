# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution identity (review 2026-09-25 P1 #3): the run id names the CASE; the fingerprint
names what one emit hands the job (image, probe script and arguments, every input file of the
bundle). The spec carries it into the job and --import-raws files a raw only when the job ran
this emit's fingerprint."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture(scope="module")
def pd(tmp_path_factory):
    os.environ["AIS_PROBE_WORKSPACE"] = str(tmp_path_factory.mktemp("ws"))
    spec = importlib.util.spec_from_file_location("probe_driver_fp", COMPONENTS / "probe_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_driver_fp"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(pd, budget=6012, dummy_cfg='{"a": 1}', image="vllm/vllm-openai:v0.32.0"):
    ws = Path(pd.ROOT)
    (ws / "dummy_models" / "m__rep").mkdir(parents=True, exist_ok=True)
    (ws / "dummy_models" / "m__rep" / "config.json").write_text(dummy_cfg)
    (ws / "inputs").mkdir(exist_ok=True)
    (ws / "inputs" / "r.sh").write_text(f"engine_command=(python3 -m dynamo.vllm --max-num-batched-tokens {budget})\n")
    run = {
        "id": "9d000f77e125",
        "image": image,
        "probe_script": "probe_vllm.py",
        "probe_args": ["--run-sh", f"{pd.WORK}/inputs/r.sh", "--model-override", f"{pd.WORK}/dummy_models/m__rep"],
        "probe_env": {},
    }
    return pd.exec_fingerprint(run, ["inputs/r.sh", "dummy_models/m__rep"])


def test_a_changed_token_budget_changes_the_fingerprint_not_the_id(pd):
    """The rendered config is an input file of the bundle: the same case with another engine
    budget is another emit."""
    assert _run(pd) == _run(pd)
    assert _run(pd, budget=6016) != _run(pd, budget=6012)


def test_the_dummy_the_image_and_the_arguments_are_part_of_the_fingerprint(pd):
    base = _run(pd)
    assert _run(pd, dummy_cfg='{"a": 2}') != base
    assert _run(pd, image="vllm/vllm-openai:v0.33.0") != base
    run = {"image": "i", "probe_script": "probe_vllm.py", "probe_env": {}}
    with_kv = pd.exec_fingerprint({**run, "probe_args": ["--kv-cache-dtype", "fp8"]}, [])
    assert with_kv != pd.exec_fingerprint({**run, "probe_args": []}, [])
