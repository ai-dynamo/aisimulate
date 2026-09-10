# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit point transport preserves the native manifest and attempt identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.config import FPMCollectionOptions, add_fpm_arguments
from collector.fpm_forward.native_artifact import COLLECTOR_PROVENANCE_FILENAME
from collector.fpm_forward.planner import BackendPolicy, FPMCell
from collector.fpm_forward.runner import (
    POINTS_FILENAME,
    POINTS_RECEIPT_FILENAME,
    REMOTE_WORKDIR,
    _cell_generator_overrides,
    _configured_sampling_metadata,
    _record_points_receipts,
    _stage_points_file,
    _validate_points_receipts,
    run_collection,
)
from collector.fpm_forward.types import ParallelTopology

pytestmark = pytest.mark.unit


def _payload():
    return {
        "schema_version": 3,
        "prefill": [
            {"batch_size": 2, "total_prefill_tokens": 8, "total_kv_read_tokens": 256, "rows": [[3, 128], [5, 128]]}
        ],
        "decode": [{"batch_size": 2, "total_kv_read_tokens": 256}],
    }


def _options(tmp_path, payload=None, **extra):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(_payload() if payload is None else payload, indent=2))
    options = FPMCollectionOptions.from_args(
        argparse.Namespace(fpm_max_gpus=4, fpm_benchmark_points_file=str(path), **extra)
    )
    return options, path


def _cell(phase="prefill", policy=None):
    return FPMCell(
        cell_id="cell",
        workload_kind=phase,
        topology=ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        weight_quantization="bfloat16",
        kv_cache_dtype="auto",
        backend_policy=policy or BackendPolicy("baseline", {}, {}),
        parallel_strategy="pure_tp",
        gemm_quant_mode="bfloat16",
        moe_quant_mode="bfloat16",
        fmha_quant_mode="bfloat16",
        comm_quant_mode="half",
    )


def _plan(options):
    return SimpleNamespace(options=options, sha256="plan-sha", model_path="text/model")


def test_cli_freezes_canonical_payload_rows_and_source_mutation(tmp_path):
    options, source = _options(tmp_path)
    canonical = json.dumps(_payload(), sort_keys=True, separators=(",", ":"))
    assert options.benchmark_points_json == canonical
    assert options.benchmark_points_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert options.to_dict()["benchmark_points"] == {
        "payload": _payload(),
        "sha256": options.benchmark_points_sha256,
    }
    source.write_text("{}")
    staged = _stage_points_file(_plan(options), tmp_path)
    assert len(staged) == 1
    assert staged[0].name == POINTS_FILENAME
    assert staged[0].read_text() == canonical
    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)
    args = parser.parse_args(["--fpm-max-gpus", "4", "--fpm-benchmark-points-file", str(staged[0])])
    assert FPMCollectionOptions.from_args(args) == options


def test_manifest_formatting_ignores_paths_but_point_order_changes_identity(tmp_path):
    options, source = _options(tmp_path)
    source.write_text(options.benchmark_points_json)
    same = FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4, fpm_benchmark_points_file=str(source)))
    assert same.to_dict() == options.to_dict()
    changed = _payload()
    changed["prefill"][0]["rows"].reverse()
    other, _ = _options(tmp_path, changed)
    assert other.benchmark_points_sha256 != options.benchmark_points_sha256
    assert other.to_dict() != options.to_dict()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"schema_version": True, "prefill": [], "decode": []},
        {"schema_version": 4, "prefill": [], "decode": []},
        {"schema_version": 3, "prefill": [], "decode": []},
        {"schema_version": 3, "prefill": [1], "decode": []},
    ],
)
def test_invalid_envelope_rejected_before_planning(tmp_path, payload):
    with pytest.raises(ValueError, match="benchmark-points"):
        _options(tmp_path, payload)


def test_duplicate_json_fields_are_not_silently_canonicalized(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"schema_version":2,"schema_version":3,"prefill":[],"decode":[]}')
    with pytest.raises(ValueError, match="duplicate benchmark-points field"):
        FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4, fpm_benchmark_points_file=str(source)))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_generator_owns_explicit_argument_rendering(tmp_path, phase):
    options, _ = _options(tmp_path)
    plan, cell = _plan(options), _cell(phase)
    args = _cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
    assert args.count("--benchmark-points-file") == 1
    assert args[args.index("--benchmark-points-file") + 1] == f"{REMOTE_WORKDIR}/{POINTS_FILENAME}"
    assert _configured_sampling_metadata(plan, cell, smoke=False) == {
        "benchmark_points_sha256": options.benchmark_points_sha256,
        "requested_point_count": 1,
    }


def test_default_grid_and_smoke_remain_explicitly_separate(tmp_path):
    legacy = FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4))
    assert "benchmark_points" not in legacy.to_dict()
    assert _stage_points_file(_plan(legacy), tmp_path) == []
    args = _cell_generator_overrides(_plan(legacy), _cell(), {}, smoke=True)["params"]["agg"]["extra_cli_args"]
    assert "--benchmark-points-file" not in args
    with pytest.raises(ValueError, match="cannot be combined with --smoke"):
        _options(tmp_path, smoke=True)
    options, _ = _options(tmp_path)
    with pytest.raises(ValueError, match="cannot be combined with --smoke"):
        run_collection(
            _plan(options),
            generator_overrides={},
            checkpoint_dir=str(tmp_path / "checkpoint"),
            artifact_root=str(tmp_path / "artifacts"),
            resume=False,
            retry_failed=False,
            smoke=True,
        )
    assert not (tmp_path / "artifacts").exists()


def test_backend_policy_cannot_replace_frozen_points(tmp_path):
    options, _ = _options(tmp_path)
    cell = _cell(
        policy=BackendPolicy("bad", {"params": {"agg": {"extra_cli_args": ["--benchmark-points-file=other"]}}}, {})
    )
    with pytest.raises(ValueError, match="must be supplied through"):
        _cell_generator_overrides(_plan(options), cell, {})


def test_staging_rejects_modified_frozen_payload(tmp_path):
    options, _ = _options(tmp_path)
    from dataclasses import replace

    plan = _plan(replace(options, benchmark_points_json="{}"))
    with pytest.raises(ValueError, match="payload and SHA256 disagree"):
        _stage_points_file(plan, tmp_path)


@pytest.mark.parametrize("transport", ["_exec", "_exec_checked"])
def test_runtime_hash_and_attempt_receipt_are_checked_in_both_transports(tmp_path, transport):
    options, _ = _options(tmp_path)
    plan, cell = _plan(options), _cell()
    staged = _stage_points_file(plan, tmp_path)[0]
    raw = tmp_path / "raw"
    unit = raw / "unit"
    unit.mkdir(parents=True)
    (unit / COLLECTOR_PROVENANCE_FILENAME).write_text("{}")

    def execute(pod, command, timeout):
        assert pod == "unit" and timeout == 300
        local = [sys.executable, *command[1:-2], str(staged), str(unit / POINTS_RECEIPT_FILENAME)]
        return subprocess.run(local, check=True, capture_output=True, text=True)

    resource = SimpleNamespace(**{transport: execute})
    _record_points_receipts(resource, ["unit"], plan, cell, "attempt", phase="before")
    with pytest.raises(ValueError, match="receipt mismatch"):
        _validate_points_receipts(plan, cell, raw, "attempt")
    _record_points_receipts(resource, ["unit"], plan, cell, "attempt", phase="after")
    _validate_points_receipts(plan, cell, raw, "attempt")
    with pytest.raises(ValueError, match="receipt mismatch"):
        _validate_points_receipts(plan, cell, raw, "another-attempt")
    staged.write_text("{}")
    with pytest.raises(subprocess.CalledProcessError):
        _record_points_receipts(resource, ["unit"], plan, cell, "attempt", phase="after")
    (unit / POINTS_RECEIPT_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        _validate_points_receipts(plan, cell, raw, "attempt")


def test_checked_in_study_manifests_freeze_exact_phase_counts():
    root = Path(__file__).resolve().parents[5]
    for name, counts in (("calibration", (100, 26)), ("heldout", (28, 10))):
        source = root / "data/experimental/deepseek-v41/verification-plan" / f"{name}.json"
        options = FPMCollectionOptions.from_args(
            argparse.Namespace(fpm_max_gpus=4, fpm_benchmark_points_file=str(source))
        )
        payload = json.loads(options.benchmark_points_json)
        assert (len(payload["prefill"]), len(payload["decode"])) == counts


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_eager_is_frozen_and_rendered_without_capture_configuration(tmp_path, phase):
    options, _ = _options(tmp_path, fpm_enforce_eager=True)
    plan, cell = _plan(options), _cell(phase)
    assert options.to_dict()["enforce_eager"] is True
    args = _cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
    assert args.count("--enforce-eager") == 1
    assert "--compilation-config" not in args
    assert "--benchmark-points-file" in args
    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)
    parsed = parser.parse_args(["--fpm-max-gpus", "4", "--fpm-enforce-eager"])
    assert FPMCollectionOptions.from_args(parsed).enforce_eager is True
    assert "enforce_eager" not in FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4)).to_dict()


@pytest.mark.parametrize("argument", ["--enforce-eager", "--no-enforce-eager", "--enforce-eager=false"])
def test_policy_cannot_hide_eager_identity(tmp_path, argument):
    options, _ = _options(tmp_path, fpm_enforce_eager=True)
    cell = _cell(policy=BackendPolicy("conflict", {"params": {"agg": {"extra_cli_args": [argument]}}}, {}))
    with pytest.raises(ValueError, match="must be supplied through --fpm-enforce-eager"):
        _cell_generator_overrides(_plan(options), cell, {})
