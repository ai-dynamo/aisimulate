# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python contract for forward-pass model state snapshots.

Expected values are never recomputed in Python: each check compares a restored
model with the uninterrupted original on identical inputs, by exact float bits.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

from .test_external_execution_profile import profile  # noqa: F401

pytestmark = pytest.mark.unit


class _Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def between(self, low: int, high: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return low + (self.state >> 33) % (high - low)


def _rank(prefill: int, decode: int, kv: int, wall_ms: float) -> dict:
    return {
        "version": 1,
        "wall_time": wall_ms / 1000.0,
        "scheduled_requests": {
            "num_prefill_requests": int(prefill > 0),
            "sum_prefill_tokens": prefill,
            "sum_prefill_kv_tokens": prefill // 4,
            "num_decode_requests": decode,
            "sum_decode_kv_tokens": kv if decode else 0,
        },
    }


def _stream(seed: int, count: int, worker: str = "aggregated") -> list[list[dict]]:
    lcg = _Lcg(seed)
    iterations = []
    for index in range(count):
        prefill, decode = lcg.between(64, 16384), lcg.between(1, 700)
        kv = decode * lcg.between(64, 4096)
        # Hand-specified positive surface; Rust tests own fitting accuracy.
        wall = 2.0 + 4e-4 * prefill + 2e-5 * kv + 1e-2 * decode + lcg.between(0, 97) * 1e-3
        shape = 0 if worker == "decode" else index % 4
        if shape == 0:
            iterations.append([_rank(0, decode, kv, wall)])
        elif shape == 1:
            iterations.append([_rank(prefill, 0, 0, wall)])
        elif shape == 2:
            iterations.append([_rank(prefill, decode, kv, wall)])
        else:
            iterations.append([_rank(prefill, 0, 0, wall / 2), _rank(0, decode, kv, wall)])
    return iterations


def _queries(worker: str = "aggregated") -> list[list[dict]]:
    queries = [[_rank(0, 0, 0, 0.0)]]
    for decode, kv in [(1, 64), (32, 40_000), (600, 2_400_000)]:
        queries.append([_rank(0, decode, kv, 0.0)])
        if worker == "aggregated":
            for prefill in (128, 12_000):
                queries += [
                    [_rank(prefill, 0, 0, 0.0)],
                    [_rank(prefill, decode, kv, 0.0)],
                    [_rank(prefill, 0, 0, 0.0), _rank(0, decode, kv, 0.0)],
                ]
    return queries


def _bits(model: RustForwardPassPerfModel, queries: list[list[dict]]) -> list[str | None]:
    out = []
    for query in queries:
        try:
            value = model.estimate_forward_pass_time_ms(query)
        except ValueError as error:
            out.append(f"error: {error}")
            continue
        out.append(None if value is None else struct.pack(">d", value).hex())
    return out


def _state(model: RustForwardPassPerfModel) -> dict:
    diagnostics = model.diagnostics()
    diagnostics.pop("provenance")
    return {
        "diagnostics": diagnostics,
        "stores": model.regression_store_diagnostics(),
        "readiness": model.readiness_report(),
    }


def _regression_config(worker: str = "aggregated") -> ForwardPassPerfModelConfig:
    return ForwardPassPerfModelConfig(
        model="test/model",
        system="test",
        backend="vllm",
        worker_type=worker,
        estimation_mode="fpm_regression",
        estimator_config={"fpm_regression": {"fit": {"kind": "spline"}}},
    )


def _continue(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    config["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
    return config


def test_to_file_round_trips_through_file_config_and_dataclass(tmp_path):
    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(1, 120))
    queries = _queries()
    pinned = model.to_file(tmp_path / "snap")
    assert pinned == json.loads((tmp_path / "snap" / "config.json").read_text())
    snapshot = pinned["estimator_config"]["state_snapshot"]
    assert snapshot["path"] == str((tmp_path / "snap").resolve())
    assert snapshot["on_tune"] == "reject"
    assert len(snapshot["sha256"]) == 64
    for restored in (
        RustForwardPassPerfModel.best_available(file=tmp_path / "snap"),
        RustForwardPassPerfModel.best_available(pinned),
        RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**pinned)),
    ):
        assert _bits(restored, queries) == _bits(model, queries)
        assert _state(restored) == _state(model)
        # The saved configuration is a fixed point of construction.
        assert restored.diagnostics()["provenance"]["config"] == pinned
        with pytest.raises(ValueError, match="on_tune='reject'"):
            restored.tune_with_fpms(_stream(2, 1))
    with pytest.raises(TypeError, match="exactly one"):
        RustForwardPassPerfModel.best_available()
    with pytest.raises(TypeError, match="exactly one"):
        RustForwardPassPerfModel.best_available(pinned, file=tmp_path / "snap")
    with pytest.raises(ValueError, match="already exists"):
        model.to_file(tmp_path / "snap")


def test_snapshot_mismatches_fail_instead_of_selecting_another_model(tmp_path):
    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(3, 40))
    pinned = model.to_file(tmp_path / "snap")
    changed = json.loads(json.dumps(pinned))
    changed["estimator_config"]["fpm_regression"]["min_observations"] = 6
    with pytest.raises(ValueError, match="pinned identity at estimator_config.fpm_regression.min_observations"):
        RustForwardPassPerfModel.best_available(changed)
    for policy in ("allow", "regression"):
        with pytest.raises(ValueError, match="fallback_policy='deny'"):
            RustForwardPassPerfModel.best_available({**pinned, "fallback_policy": policy})
    with pytest.raises(ValueError, match="explicit estimation_mode"):
        RustForwardPassPerfModel.best_available({**pinned, "estimation_mode": "auto"})
    stale = json.loads(json.dumps(pinned))
    stale["estimator_config"]["state_snapshot"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match manifest.json"):
        RustForwardPassPerfModel.best_available(stale)
    (tmp_path / "snap" / "state.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="state.bin does not match its manifest"):
        RustForwardPassPerfModel.best_available(file=tmp_path / "snap")


_CHILD = """
import hashlib, json, struct, sys
from pathlib import Path
from aisimulate_core.sdk import RustForwardPassPerfModel
request = json.loads(sys.argv[1])
model = RustForwardPassPerfModel.best_available(request["config"])
for chunk in request["next"]:
    model.tune_with_fpms(chunk)
bits = []
for query in request["queries"]:
    value = model.estimate_forward_pass_time_ms(query)
    bits.append(None if value is None else struct.pack(">d", value).hex())
if request["save"]:
    model.to_file(request["save"])
print(json.dumps({"bits": bits, "readiness": model.readiness_report()}))
"""


def _child(config: dict, next_chunks: list, queries: list, save: Path | None) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    request = json.dumps({"config": config, "next": next_chunks, "queries": queries, "save": save and str(save)})
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, request], env=env, capture_output=True, text=True, check=False, timeout=600
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("worker", ["aggregated", "decode"])
def test_continued_tuning_in_another_process_matches_the_uninterrupted_model(tmp_path, worker):
    model = RustForwardPassPerfModel.best_available(_regression_config(worker))
    model.tune_with_fpms(_stream(5, 90, worker))
    pinned = model.to_file(tmp_path / "snap")
    queries = _queries(worker)
    next_stream = _stream(6, 60, worker)
    chunks = [next_stream[index : index + 7] for index in range(0, len(next_stream), 7)]
    child = _child(_continue(pinned), chunks, queries, tmp_path / "child")
    for chunk in chunks:
        model.tune_with_fpms(chunk)
    assert child["bits"] == _bits(model, queries)
    assert child["readiness"] == model.readiness_report()
    model.to_file(tmp_path / "parent")
    assert _digest(tmp_path / "child" / "state.bin") == _digest(tmp_path / "parent" / "state.bin")
    # A cold model fed only the continuation differs, so equality is evidence.
    cold = RustForwardPassPerfModel.best_available(_regression_config(worker))
    for chunk in chunks:
        cold.tune_with_fpms(chunk)
    assert _bits(cold, queries) != _bits(model, queries)


def test_from_learned_snapshot_restores_without_identity(tmp_path):
    stump = {
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "feature": [0, -1, -1],
        "threshold": [64.0, 0.0, 0.0],
        "value": [0.0, 0.5, 2.0],
        "missing_left": [True, True, True],
    }
    artifact = {
        "schema": "aic_fpm_learned_forward_perf",
        "schema_version": 1,
        "worker_type": "decode",
        "target": "ms",
        "features": ["num_decode_requests"],
        "stores": {"pure_decode": {"baseline": 4.0, "trees": [stump]}},
        "metadata": {},
    }
    model = RustForwardPassPerfModel.from_learned(artifact, {"min_observations": 2})
    model.tune_with_fpms(_stream(7, 30, "decode"))
    queries = _queries("decode")
    model.to_file(tmp_path / "snap")
    summary = RustForwardPassPerfModel.inspect_state_snapshot(tmp_path / "snap")
    assert summary["kind"] == "learned_without_identity"
    assert summary["identity"] is None and summary["topology"] is None
    restored = RustForwardPassPerfModel.best_available(file=tmp_path / "snap")
    assert _bits(restored, queries) == _bits(model, queries)
    assert restored.deployment_topology() is None


def test_readiness_topology_and_launch_compatibility_without_loading(tmp_path):
    config = ForwardPassPerfModelConfig(
        model="test/model",
        system="test",
        backend="vllm",
        worker_type="aggregated",
        estimation_mode="fpm_regression",
    )
    model = RustForwardPassPerfModel.best_available(config)
    cold = model.readiness_report()
    assert cold["ready"] is False and cold["corrections"] == []
    assert [store["not_ready_reason"] for store in cold["stores"]] == ["insufficient_data"] * 4
    # No cross-rank work: that store has no data while the others retain 20.
    model.tune_with_fpms([iteration for iteration in _stream(9, 80) if len(iteration) == 1])
    report = model.readiness_report()
    legacy = model.regression_store_diagnostics()
    assert [(store["workload_kind"], store["ready"], store["retained_observations"]) for store in report["stores"]] == [
        (store["workload_kind"], store["ready"], store["retained_observations"]) for store in legacy
    ]
    assert report["stores"][0]["ready"] and report["stores"][3]["ready"]
    assert report["stores"][2]["not_ready_reason"] == "insufficient_data"
    # Enough samples without a usable fit report a different reason.
    assert {store["not_ready_reason"] for store in report["stores"] if not store["ready"]} <= {
        "insufficient_data",
        "no_usable_fit",
    }
    topology = model.deployment_topology()
    assert (topology["worker_type"], topology["tp"], topology["attention_dp"]) == ("aggregated", 1, 1)
    assert topology["correction_limits"] is None
    model.to_file(tmp_path / "snap")
    summary = RustForwardPassPerfModel.inspect_state_snapshot(tmp_path / "snap")
    assert summary["kind"] == "regression"
    assert summary["readiness"] == report
    assert summary["topology"] == topology
    # The same Rust rules engine launch enforces, reported as a list.
    assert RustForwardPassPerfModel.config_engine_launch_mismatches(
        {"dp_size": 8, "tensor_parallel_size": 2}, file=tmp_path / "snap"
    ) == ["tensor_parallel_size conflicts with canonical tp", "dp_size conflicts with canonical attention_dp"]
    assert model.engine_launch_mismatches({"engine": {"worker_type": "decode"}}) == [
        "AIS worker_type must match engine role"
    ]
    assert model.engine_launch_mismatches({}) == []
    with pytest.raises(ValueError):
        model.engine_launch_mismatches({"dp_size": "eight"})


def _systems_copy(destination: Path, system: str) -> Path:
    """Copy one system's fingerprinted inputs: top-level files and its data dir."""
    packaged = Path(str(resources.files("aisimulate_core") / "systems"))
    destination.mkdir()
    for entry in packaged.iterdir():
        if entry.is_file():
            shutil.copy2(entry, destination / entry.name)
    shutil.copytree(packaged / "data" / system, destination / "data" / system)
    return destination


def _native_stream(model: RustForwardPassPerfModel, seed: int, count: int) -> list[list[dict]]:
    iterations = _stream(seed, count)
    for index, iteration in enumerate(iterations):
        native = model.estimate_forward_pass_time_ms(iteration)
        for rank in iteration:
            rank["wall_time"] = native * (0.7 + (index % 9) * 0.11) / 1000.0
    return iterations


@pytest.mark.integration
def test_native_snapshot_pins_engine_and_perf_data_across_processes(tmp_path):
    root = _systems_copy(tmp_path / "systems", "h200_sxm")
    config = ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        worker_type="aggregated",
        estimation_mode="op_level",
        systems_paths=(str(root),),
        estimator_config={"correction": {"mode": "hybrid", "min_observations": 3}},
    )
    model = RustForwardPassPerfModel.best_available(config)
    cold = RustForwardPassPerfModel.best_available(config)
    model.tune_with_fpms(_native_stream(cold, 11, 48))
    queries = _queries()
    reference = model.to_file(tmp_path / "reference")
    packed = model.to_file(tmp_path / "packed", include_perf_db=True)
    assert packed["systems_paths"] == [str((tmp_path / "packed" / "perf_db" / "systems").resolve())]
    assert reference["systems_paths"] == [str(root.resolve())]
    expected = _bits(model, queries)
    # A fresh interpreter recompiles the engine; its spec digest must match.
    assert _child(packed, [], queries, None)["bits"] == expected
    assert _bits(RustForwardPassPerfModel.best_available(file=tmp_path / "reference"), queries) == expected
    # Changing referenced data invalidates only the reference snapshot.
    data = next((root / "data" / "h200_sxm").rglob("*.parquet"))
    data.write_bytes(data.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="fingerprint"):
        RustForwardPassPerfModel.best_available(file=tmp_path / "reference")
    assert _bits(RustForwardPassPerfModel.best_available(file=tmp_path / "packed"), queries) == expected


def test_execution_profile_is_copied_and_perf_data_packed(profile, tmp_path):  # noqa: F811
    config, _, _, root, path = profile
    model = RustForwardPassPerfModel.best_available(config)
    # Synthetic oracle from the profile fixture: three repetitions of 2 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    pinned = model.to_file(tmp_path / "packed", include_perf_db=True)
    selection = pinned["estimator_config"]["op_level"]["execution_profile"]
    assert selection == {
        "path": str((tmp_path / "packed" / "execution_profile.json").resolve()),
        "sha256": config["estimator_config"]["op_level"]["execution_profile"]["sha256"],
    }
    assert pinned["systems_paths"] == [str((tmp_path / "packed" / "perf_db" / "systems").resolve())]
    packed = tmp_path / "packed" / "perf_db" / "systems"
    assert sorted(str(p.relative_to(packed)) for p in packed.rglob("*") if p.is_file()) == sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )
    path.unlink()
    restored = RustForwardPassPerfModel.best_available(file=tmp_path / "packed")
    assert restored.predict_prefill_latency(1, 4, 0) == 6.0
    assert restored.readiness_report() == model.readiness_report()
