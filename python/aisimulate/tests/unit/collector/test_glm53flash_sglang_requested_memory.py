# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY consumer subset of FPM 04e5797a; no partial planner migration."""

import hashlib
import json

import pytest

from collector.fpm_forward import glm53flash_validation as validation
from collector.fpm_forward.config import validate_sglang_mem_fraction_static
from collector.fpm_forward.sglang_artifact import file_receipt, validate_sglang_repetitions
from tests.unit.collector.test_fpm_glm53flash_sglang_artifact import artifact
from tests.unit.collector.test_glm53flash_validation import write_plan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [False, True, "0.82", 0, 1, float("nan"), float("inf")])
def test_reader_rejects_invalid_requested_fraction(value):
    with pytest.raises(ValueError, match="strictly between"):
        validate_sglang_mem_fraction_static(value)


@pytest.mark.parametrize("kind", ["declared_config", "resolved_config"])
@pytest.mark.parametrize("actual", [None, 0.9062999999999999, "0.82", True])
def test_sha_valid_native_receipt_must_match_frozen_requested_value(tmp_path, kind, actual):
    cell, payload = artifact(tmp_path)
    cell.sglang_mem_fraction_static = 0.82
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for name in ("declared_config", "resolved_config"):
        path = tmp_path / evidence[name]["file"]
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = actual if kind == name else 0.82
        path.write_text(json.dumps(config))
        evidence[name] = file_receipt(path)
    with pytest.raises(ValueError, match=f"{kind.removesuffix('_config')} memory fraction"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


def test_matching_explicit_native_setting_is_accepted(tmp_path):
    cell, payload = artifact(tmp_path)
    cell.sglang_mem_fraction_static = 0.82
    evidence = payload["input_provenance"]["native_forward_manifest"]
    for name in ("declared_config", "resolved_config"):
        path = tmp_path / evidence[name]["file"]
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = 0.82
        path.write_text(json.dumps(config))
        evidence[name] = file_receipt(path)
    validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


@pytest.mark.parametrize("cell_fraction", [None, 0.82, 0.83])
def test_acceptance_reconstructs_and_crossbinds_requested_plan_value(tmp_path, cell_fraction):
    spec = write_plan(tmp_path, ("sglang", "fp8", 2, "prefill"), "holdout")
    path = tmp_path / spec["plan"]["path"]
    frozen = json.loads(path.read_text())
    frozen["options"]["sglang_mem_fraction_static"] = 0.82
    if cell_fraction is not None:
        frozen["cells"][0]["sglang_mem_fraction_static"] = cell_fraction
    path.write_text(json.dumps(frozen))
    spec["plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cell_fraction == 0.82:
        run = validation._plan_run(spec, tmp_path, "holdout")
        assert run["runtime_cell"].sglang_mem_fraction_static == 0.82
    else:
        with pytest.raises(ValueError, match="between frozen plan and cell"):
            validation._plan_run(spec, tmp_path, "holdout")


@pytest.mark.parametrize("missing", ["plan", "cell", "runtime_cell"])
def test_ops_cannot_lose_requested_fraction_at_its_private_cell_boundary(missing):
    from types import SimpleNamespace

    from collector.glm53flash_validation import requested_sglang_memory

    run = {
        "key": ("sglang", "fp8", 2, "prefill"),
        "plan": {"options": {"sglang_mem_fraction_static": 0.82}},
        "cell": {"sglang_mem_fraction_static": 0.82},
        "runtime_cell": SimpleNamespace(sglang_mem_fraction_static=0.82),
    }
    assert requested_sglang_memory(run) == 0.82
    if missing == "plan":
        run["plan"]["options"].clear()
    elif missing == "cell":
        run["cell"].clear()
    else:
        del run["runtime_cell"].sglang_mem_fraction_static
    with pytest.raises(ValueError, match="memory fraction|frozen plan and cell"):
        requested_sglang_memory(run)


def test_ops_load_native_checks_requested_fraction_against_actual_receipts(tmp_path, monkeypatch):
    from collector.glm53flash_validation import load_native
    from tests.unit.collector.test_glm53flash_graph_export import fixture, put

    run, root = fixture(tmp_path, monkeypatch, "holdout")
    run.setdefault("plan", {})["options"] = {"sglang_mem_fraction_static": 0.82}
    run.setdefault("cell", {})["sglang_mem_fraction_static"] = 0.82
    for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
        path = root / name
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = 0.9063
        put(path, config)
    with pytest.raises(ValueError, match="declared memory fraction"):
        load_native(run, root)
    for name in ("sglang-declared-config.json", "sglang-resolved-config.json"):
        path = root / name
        config = json.loads(path.read_text())
        config["mem_fraction_static"] = 0.82
        put(path, config)
    native = load_native(run, root)
    assert native["execution_policy"]["sha256"]
