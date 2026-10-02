# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution identity (review 2026-09-25 P1 #3): the run id names the CASE; the
execution fingerprint names what actually shaped the run (engine invocation,
dummy config, image, kv override). A raw is evidence for the current plan only
when its fingerprint (sidecar or the argv it recorded) matches."""
import importlib.util
import json
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


RUN_SH = """#!/bin/bash
engine_command=(python3 -m dynamo.vllm --model org/m --served-model-name org/m --tensor-parallel-size 1 --max-num-seqs 512 --max-num-batched-tokens {budget} --dump-config-to /results/resolved.json)
"""


def _plan_run(pd, tmp_path, budget, dummy_cfg='{"a": 1}'):
    ws = Path(pd.ROOT)
    (ws / "archive" / "run_sh").mkdir(parents=True, exist_ok=True)
    (ws / "dummy_models" / "generic" / "m__rep").mkdir(parents=True, exist_ok=True)
    (ws / "dummy_models" / "generic" / "m__rep" / "config.json").write_text(dummy_cfg)
    rsh = ws / "archive" / "run_sh" / f"r{budget}.sh"
    rsh.write_text(RUN_SH.format(budget=budget))
    return {"id": "9d000f77e125", "repo": "org/m", "backend": "vllm", "version": "0.29.0",
            "image": "vllm/vllm-openai:v0.29.0", "kv_dtype": None, "cli_extra_args": [],
            "run_sh": str(rsh), "model_dir": f"{pd.WORK}/dummy_models/generic/m__rep"}


def test_review_p1_changed_token_budget_changes_the_fingerprint_not_the_id(pd, tmp_path):
    a = _plan_run(pd, tmp_path, 6012)
    b = _plan_run(pd, tmp_path, 6016)
    assert a["id"] == b["id"]  # the CASE is the same
    fa, fb = pd.exec_fingerprint(a), pd.exec_fingerprint(b)
    assert fa["fingerprint"] != fb["fingerprint"] and fa["engine"] != fb["engine"]
    assert fa["dummy"] == fb["dummy"]


def test_dummy_and_image_are_part_of_the_fingerprint(pd, tmp_path):
    a = _plan_run(pd, tmp_path, 6016)
    fa = pd.exec_fingerprint(a)
    b = _plan_run(pd, tmp_path, 6016, dummy_cfg='{"a": 2}')
    assert pd.exec_fingerprint(b)["fingerprint"] != fa["fingerprint"]
    c = dict(_plan_run(pd, tmp_path, 6016), image="vllm/vllm-openai:v0.30.0")
    assert pd.exec_fingerprint(c)["fingerprint"] != fa["fingerprint"]


def test_engine_tokens_drop_deployment_specific_values_only(pd):
    toks = ["--model", "/work/dummy_models/x", "--served-model-name=org/m", "--tensor-parallel-size", "1",
            "--max-num-batched-tokens", "6016", "--dump-config-to", "/results/r.json"]
    assert pd.engine_tokens("vllm", toks) == ["--tensor-parallel-size", "1", "--max-num-batched-tokens", "6016"]


def test_evidence_status_sidecar_is_decisive(pd, tmp_path):
    run = _plan_run(pd, tmp_path, 6016)
    run["exec_fingerprint"] = pd.exec_fingerprint(run)
    fp = run["exec_fingerprint"]["fingerprint"]
    assert pd.evidence_status(run, fp, {}) == "current"
    assert pd.evidence_status(run, "000000000000", {}) == "stale"


def test_evidence_status_legacy_raw_checked_on_its_recorded_argv(pd, tmp_path):
    run = _plan_run(pd, tmp_path, 6016)
    run["exec_fingerprint"] = pd.exec_fingerprint(run)
    # the probe substitutes the dummy dir for --model; everything else identical -> current
    raw_same = {"engine_argv": ["python3", "-m", "dynamo.vllm", "--model", "/work/dummy_models/generic/m__rep",
                                "--served-model-name", "org/m", "--tensor-parallel-size", "1",
                                "--max-num-seqs", "512", "--max-num-batched-tokens", "6016",
                                "--dump-config-to", "/results/resolved.json"]}
    assert pd.evidence_status(run, None, raw_same) == "current"
    raw_old = {"engine_argv": [t if t != "6016" else "6012" for t in raw_same["engine_argv"]]}
    assert pd.evidence_status(run, None, raw_old) == "stale"
    # no sidecar, nothing recorded: never 'current'
    assert pd.evidence_status(run, None, {}) == "unverified"
    assert pd.evidence_status({"backend": "vllm"}, None, raw_same) == "unverified"


def test_trtllm_engine_yaml_compared_semantically(pd, tmp_path):
    ws = Path(pd.ROOT)
    (ws / "archive" / "run_sh").mkdir(parents=True, exist_ok=True)
    y = ws / "archive" / "run_sh" / "t.engine.yaml"
    y.write_text("backend: pytorch\nmax_num_tokens: 4672\nkv_cache_config:\n  dtype: auto\n")
    run = {"id": "t", "repo": "org/m", "backend": "trtllm", "version": "1.3.0rc23", "image": "img",
           "kv_dtype": None, "cli_extra_args": [], "render_artifact": str(y), "model_dir": ""}
    run["exec_fingerprint"] = pd.exec_fingerprint(run)
    same = {"engine_yaml": {"kv_cache_config": {"dtype": "auto"}, "max_num_tokens": 4672, "backend": "pytorch"}}
    assert pd.evidence_status(run, None, same) == "current"
    other = {"engine_yaml": {"kv_cache_config": {"dtype": "auto"}, "max_num_tokens": 4608, "backend": "pytorch"}}
    assert pd.evidence_status(run, None, other) == "stale"
