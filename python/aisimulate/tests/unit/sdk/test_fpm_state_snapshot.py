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


def test_state_snapshot_round_trips_through_directory_config_and_dataclass(tmp_path):
    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(1, 120))
    queries = _queries()
    pinned = model.save_state_snapshot(tmp_path / "snap")
    assert pinned == json.loads((tmp_path / "snap" / "config.json").read_text())
    snapshot = pinned["estimator_config"]["state_snapshot"]
    # Pinned paths are absolute and lexically normalized; symlinks are kept.
    assert snapshot["path"] == os.path.abspath(tmp_path / "snap")
    assert snapshot["on_tune"] == "reject"
    assert len(snapshot["sha256"]) == 64
    for restored in (
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "snap"),
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
        RustForwardPassPerfModel.best_available(pinned, state_snapshot=tmp_path / "snap")
    with pytest.raises(ValueError, match="already exists"):
        model.save_state_snapshot(tmp_path / "snap")


def test_snapshot_mismatches_fail_instead_of_selecting_another_model(tmp_path):
    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(3, 40))
    pinned = model.save_state_snapshot(tmp_path / "snap")
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
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "snap")


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
    model.save_state_snapshot(request["save"])
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
    pinned = model.save_state_snapshot(tmp_path / "snap")
    queries = _queries(worker)
    next_stream = _stream(6, 60, worker)
    chunks = [next_stream[index : index + 7] for index in range(0, len(next_stream), 7)]
    child = _child(_continue(pinned), chunks, queries, tmp_path / "child")
    for chunk in chunks:
        model.tune_with_fpms(chunk)
    assert child["bits"] == _bits(model, queries)
    assert child["readiness"] == model.readiness_report()
    model.save_state_snapshot(tmp_path / "parent")
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
    model.save_state_snapshot(tmp_path / "snap")
    summary = RustForwardPassPerfModel.inspect_state_snapshot(tmp_path / "snap")
    assert summary["kind"] == "learned_without_identity"
    assert summary["identity"] is None and summary["topology"] is None
    restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "snap")
    assert _bits(restored, queries) == _bits(model, queries)
    assert restored.deployment_topology() is None
    # A missing learned member is a broken snapshot, not unavailable performance data.
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    broken = model.save_state_snapshot(tmp_path / "broken")
    (tmp_path / "broken" / "learned" / "artifact.json").unlink()
    for call in (
        lambda: RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "broken"),
        lambda: RustForwardPassPerfModel.best_available(broken),
    ):
        with pytest.raises(FileNotFoundError) as caught:
            call()
        assert not isinstance(caught.value, PerfDataNotAvailableError)


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
    model.save_state_snapshot(tmp_path / "snap")
    summary = RustForwardPassPerfModel.inspect_state_snapshot(tmp_path / "snap")
    assert summary["kind"] == "regression"
    assert summary["readiness"] == report
    assert summary["topology"] == topology
    # The same Rust rules engine launch enforces, reported as a list.
    assert RustForwardPassPerfModel.config_engine_launch_mismatches(
        {"dp_size": 8, "tensor_parallel_size": 2}, state_snapshot=tmp_path / "snap"
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
    reference = model.save_state_snapshot(tmp_path / "reference")
    packed = model.save_state_snapshot(tmp_path / "packed", include_perf_db=True)
    assert packed["systems_paths"] == [os.path.abspath(tmp_path / "packed" / "perf_db" / "systems")]
    assert reference["systems_paths"] == [os.path.abspath(root)]
    expected = _bits(model, queries)
    # A fresh interpreter recompiles the engine; its spec digest must match.
    assert _child(packed, [], queries, None)["bits"] == expected
    assert _bits(RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "reference"), queries) == expected
    # Changing referenced data invalidates only the reference snapshot.
    data = next((root / "data" / "h200_sxm").rglob("*.parquet"))
    data.write_bytes(data.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="fingerprint"):
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "reference")
    assert _bits(RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "packed"), queries) == expected


@pytest.mark.parametrize("profile_version", [1, 2])
def test_execution_profile_is_copied_and_perf_data_packed(profile, tmp_path, profile_version):  # noqa: F811
    config, payload, publish, root, path = profile
    payload["schema_version"] = profile_version
    publish()
    model = RustForwardPassPerfModel.best_available(config)
    # Synthetic oracle from the profile fixture: three repetitions of 2 ms.
    assert model.predict_prefill_latency(1, 4, 0) == 6.0
    pinned = model.save_state_snapshot(tmp_path / "packed", include_perf_db=True)
    selection = pinned["estimator_config"]["op_level"]["execution_profile"]
    assert selection == {
        "path": os.path.abspath(tmp_path / "packed" / "execution_profile.json"),
        "sha256": config["estimator_config"]["op_level"]["execution_profile"]["sha256"],
    }
    assert pinned["systems_paths"] == [os.path.abspath(tmp_path / "packed" / "perf_db" / "systems")]
    packed = tmp_path / "packed" / "perf_db" / "systems"
    assert sorted(str(p.relative_to(packed)) for p in packed.rglob("*") if p.is_file()) == sorted(
        str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
    )
    path.unlink()
    restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "packed")
    assert restored.predict_prefill_latency(1, 4, 0) == 6.0
    query = [{"version": 1, "scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 4}}]
    if profile_version == 1:
        with pytest.raises(ValueError, match="schema_version 2"):
            restored.estimate_forward_pass_time_ms(query)
    else:
        assert restored.estimate_forward_pass_time_ms(query) == 6.0
    assert restored.readiness_report() == model.readiness_report()


def _op_level_config(root: Path) -> ForwardPassPerfModelConfig:
    return ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-32B",
        system="h200_sxm",
        backend="vllm",
        worker_type="aggregated",
        estimation_mode="op_level",
        systems_paths=(str(root),),
    )


def _scale_gemm_latency(root: Path, factor: float) -> None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for path in (root / "data" / "h200_sxm" / "gemm" / "vllm").rglob("gemm_perf.parquet"):
        table = pq.read_table(path)
        column = table.schema.get_field_index("latency")
        pq.write_table(table.set_column(column, "latency", pc.multiply(table["latency"], factor)), path)


@pytest.mark.integration
def test_native_models_never_adopt_tables_parsed_before_their_fingerprint(tmp_path):
    from aisimulate_core.sdk.engine import EngineHandle

    queries = _queries()
    roots = [_systems_copy(tmp_path / name, "h200_sxm") for name in ("saved", "restored")]
    before = _child(_op_level_config(roots[0]).to_dict(), [], queries, None)["bits"]
    # Engines in this process parse the original tables, then the data changes.
    engines = [EngineHandle.compile("Qwen/Qwen3-32B", "h200_sxm", "vllm", systems_path=str(root)) for root in roots]
    for engine, root in zip(engines, roots, strict=True):
        engine.predict_prefill_latency(1, 4096)
        _scale_gemm_latency(root, 2.0)
    # A model constructed now fingerprints and reads the changed data.
    model = RustForwardPassPerfModel.best_available(_op_level_config(roots[0]))
    expected = _bits(model, queries)
    assert expected != before
    pinned = model.save_state_snapshot(tmp_path / "saved-snapshot")
    assert _child(pinned, [], queries, None)["bits"] == expected
    # Restoring here a snapshot saved from the changed data elsewhere reads it too.
    saved = _child(_op_level_config(roots[1]).to_dict(), [], queries, tmp_path / "restored-snapshot")
    assert saved["bits"] == expected
    restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "restored-snapshot")
    assert _bits(restored, queries) == expected


@pytest.mark.integration
def test_data_dir_with_parent_components_restores_by_reference_but_cannot_be_packed(tmp_path):
    root = _systems_copy(tmp_path / "systems", "h200_sxm")
    (root / "data" / "b200_sxm").mkdir()
    yaml = root / "h200_sxm.yaml"
    yaml.write_text(yaml.read_text().replace("data_dir: data/h200_sxm", "data_dir: data/b200_sxm/../h200_sxm", 1))
    model = RustForwardPassPerfModel.best_available(_op_level_config(root))
    queries = _queries()
    reference = model.save_state_snapshot(tmp_path / "reference")
    assert _child(reference, [], queries, None)["bits"] == _bits(model, queries)
    # The packed YAML is copied unchanged, and packing does not copy `data/b200_sxm`.
    with pytest.raises(ValueError, match="plain directory names"):
        model.save_state_snapshot(tmp_path / "packed", include_perf_db=True)
    assert not (tmp_path / "packed").exists()


def test_state_snapshot_must_name_the_pinned_directory_and_io_errors_are_os_errors(tmp_path):
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    model = RustForwardPassPerfModel.best_available(_regression_config())
    model.tune_with_fpms(_stream(13, 40))
    pinned = model.save_state_snapshot(tmp_path / "snap")
    queries = _queries()
    # Restoration reads the pinned path, so a copy must not silently restore it.
    shutil.copytree(tmp_path / "snap", tmp_path / "copy")
    with pytest.raises(ValueError, match="pinned to .*snap"):
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "copy")
    # Another name for the same directory is the same snapshot.
    (tmp_path / "alias").symlink_to(tmp_path / "snap", target_is_directory=True)
    restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "alias")
    assert _bits(restored, queries) == _bits(model, queries)
    # Missing snapshots raise the same OSError subclass through state_snapshot= and
    # config=, never the skippable missing-performance-data class.
    missing = json.loads(json.dumps(pinned))
    missing["estimator_config"]["state_snapshot"]["path"] = str(tmp_path / "missing")
    for call in (
        lambda: RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "missing"),
        lambda: RustForwardPassPerfModel.best_available(missing),
        lambda: RustForwardPassPerfModel.inspect_state_snapshot(tmp_path / "missing"),
    ):
        with pytest.raises(FileNotFoundError) as caught:
            call()
        assert not isinstance(caught.value, PerfDataNotAvailableError)
    (tmp_path / "not-a-directory").write_text("")
    with pytest.raises(OSError) as caught:
        model.save_state_snapshot(tmp_path / "not-a-directory" / "snap")
    assert not isinstance(caught.value, (PerfDataNotAvailableError, ValueError))


# ---------------------------------------------------------------------------
# Native FPM interpolation from an external parquet, with no operation tables
# ---------------------------------------------------------------------------


def _interpolation_config(tmp_path: Path, mode: str) -> ForwardPassPerfModelConfig:
    from .test_fpm_forward import _median_pair_request

    # The systems root holds only gb300.yaml: its data_dir does not exist.
    base, _, _ = _median_pair_request(tmp_path, "vllm")
    data = base.to_dict()
    data["estimator_config"] = {
        **data["estimator_config"],
        "correction": {"mode": mode, "min_observations": 3},
    }
    return ForwardPassPerfModelConfig(**data)


def _interpolation_queries() -> list[list[dict]]:
    from .test_fpm_forward import _median_metrics

    return [[_median_metrics(phase, tokens)] for tokens in (512, 640, 768, 1024) for phase in ("prefill", "decode")]


def _interpolation_stream(cold: RustForwardPassPerfModel, seed: int, count: int, factor: float) -> list[list[dict]]:
    from .test_fpm_forward import _median_metrics

    lcg = _Lcg(seed)
    iterations = []
    for index in range(count):
        metrics = _median_metrics("prefill" if index % 2 else "decode", lcg.between(512, 1025))
        native = cold.estimate_forward_pass_time_ms([metrics])
        # Distinct warm-up and continuation regimes keep the cold control
        # (fed only the continuation) away from the tuned model.
        metrics["wall_time"] = native * (factor + (index % 5) * 0.03) / 1000.0
        iterations.append([metrics])
    return iterations


@pytest.mark.parametrize("mode", ["fixed", "dynamic_local", "hybrid"])
def test_fpm_interpolation_snapshots_restore_continue_and_pack_external_data(tmp_path, mode):
    config = _interpolation_config(tmp_path, mode)
    model = RustForwardPassPerfModel.best_available(config)
    cold = RustForwardPassPerfModel.best_available(config)
    assert model.diagnostics()["provenance"]["selected_estimation_mode"] == "fpm_interpolation"
    queries = _interpolation_queries()
    untuned = _bits(cold, queries)
    model.tune_with_fpms(_interpolation_stream(cold, 11, 60, 0.8))
    saved = _bits(model, queries)
    assert saved != untuned
    pinned = model.save_state_snapshot(tmp_path / "snap")
    packed = model.save_state_snapshot(tmp_path / "packed", include_perf_db=True)
    source = config.to_dict()["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    packed_parquet = packed["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    assert packed_parquet == os.path.join(os.path.abspath(tmp_path / "packed"), "perf_db", "fpm", Path(source).name)
    assert Path(packed_parquet).with_suffix(".metadata.json").is_file()
    assert pinned["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"] == source

    frozen = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "snap")
    assert _bits(frozen, queries) == saved
    assert _state(frozen) == _state(model)

    # Continue in a fresh interpreter: it recompiles the engine (spec digest
    # independent of the parquet location) and must match bit for bit.
    next_stream = _interpolation_stream(cold, 12, 21, 1.4)
    chunks = [next_stream[index : index + 7] for index in range(0, len(next_stream), 7)]
    child = _child(_continue(pinned), chunks, queries, tmp_path / "child")
    for chunk in chunks:
        model.tune_with_fpms(chunk)
        cold.tune_with_fpms(chunk)
    assert child["bits"] == _bits(model, queries)
    assert child["readiness"] == model.readiness_report()
    model.save_state_snapshot(tmp_path / "parent")
    assert _digest(tmp_path / "child" / "state.bin") == _digest(tmp_path / "parent" / "state.bin")
    assert _bits(cold, queries) != _bits(model, queries)

    # Change the source parquet: the reference snapshot is rejected by
    # fingerprint before the changed file is read, and this process can no
    # longer save; the packed copy still restores in a fresh process.
    with open(source, "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="differs from the snapshot fingerprint"):
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "snap")
    with pytest.raises(ValueError, match="changed after this process loaded it"):
        model.save_state_snapshot(tmp_path / "after-change")
    assert not (tmp_path / "after-change").exists()
    assert _child(packed, [], queries, None)["bits"] == saved


def _replay_latencies(timing: dict, capture: bool) -> dict:
    from aisimulate import _runtime

    rank = {
        "backend": "vllm",
        "block_size": 64,
        "num_gpu_blocks": 4096,
        "timing_model": {"type": "external", "provider": "aic", "config": timing},
    }
    payload = {
        "spec": {
            "version": 1,
            "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
            "engine": {"rank": rank},
            "requests": [{"id": "r0", "arrival_time_ms": 0.0, "input_tokens": 512, "output_tokens": 3}],
        },
        "capture_performance_diagnostics": capture,
    }
    report = json.loads(_runtime.run_replay_json(json.dumps(payload)))
    return {key: report[key] for key in ("mean_ttft_ms", "mean_ttst_ms", "mean_e2e_latency_ms")}


def test_replay_applies_restored_corrections_with_and_without_diagnostics(tmp_path):
    config = _interpolation_config(tmp_path, "fixed")
    model = RustForwardPassPerfModel.best_available(config)
    cold = RustForwardPassPerfModel.best_available(config)
    model.tune_with_fpms(_interpolation_stream(cold, 31, 20, 2.0))
    pinned = model.save_state_snapshot(tmp_path / "snap")
    untuned = _replay_latencies(config.to_dict(), capture=False)
    assert _replay_latencies(config.to_dict(), capture=True) == untuned
    restored = _replay_latencies(pinned, capture=False)
    assert _replay_latencies(pinned, capture=True) == restored
    # Observations ran about twice the native latency in both phases.
    assert all(restored[key] > untuned[key] for key in untuned), (restored, untuned)


def test_perf_data_fingerprint_is_taken_when_this_process_loads_it(tmp_path):
    config = _interpolation_config(tmp_path, "hybrid")
    model = RustForwardPassPerfModel.best_available(config)
    cold = RustForwardPassPerfModel.best_available(config)
    model.tune_with_fpms(_interpolation_stream(cold, 21, 30, 0.8))
    queries = _interpolation_queries()
    before = model.save_state_snapshot(tmp_path / "before")
    # A semantically neutral edit to the sidecar still changes the inputs.
    source = config.to_dict()["estimator_config"]["fpm_interpolation"]["fpm_parquet_path"]
    sidecar = Path(source).with_suffix(".metadata.json")
    sidecar.write_text(sidecar.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after this process loaded it"):
        model.save_state_snapshot(tmp_path / "after")
    with pytest.raises(ValueError, match="differs from the snapshot fingerprint"):
        RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "before")
    # A fresh process loads the changed data, so its own fingerprint can save.
    stream = _interpolation_stream(cold, 22, 14, 0.8)
    child = _child(config.to_dict(), [stream], queries, tmp_path / "fresh")
    assert child["bits"]
    fresh = json.loads((tmp_path / "fresh" / "manifest.json").read_text())["native"]["perf_db"]
    stale = json.loads((tmp_path / "before" / "manifest.json").read_text())["native"]["perf_db"]
    assert fresh["fpm_parquet"]["sha256"] == stale["fpm_parquet"]["sha256"]
    assert fresh["fpm_parquet"]["sidecar_sha256"] != stale["fpm_parquet"]["sidecar_sha256"]
    assert before["estimator_config"]["state_snapshot"]["path"] == os.path.abspath(tmp_path / "before")


# ---------------------------------------------------------------------------
# Every kind continues in another process
# ---------------------------------------------------------------------------

_STORES = {
    "aggregated": ["pure_decode", "contains_locally_mixed", "cross_rank_aggregated", "pure_prefill"],
    "decode": ["pure_decode"],
}


def _stump(feature: int, threshold: float, left: float, right: float) -> dict:
    return {
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "feature": [feature, -1, -1],
        "threshold": [threshold, 0.0, 0.0],
        "value": [0.0, left, right],
        "missing_left": [True, True, True],
    }


def _forest(worker: str) -> dict:
    trees = [_stump(2, 1024.0, 0.5, 3.0), _stump(1, 50_000.0, 0.25, 2.5), _stump(0, 64.0, 0.125, 1.0)]
    return {
        "schema": "aic_fpm_learned_forward_perf",
        "schema_version": 1,
        "worker_type": worker,
        "target": "ms",
        "features": ["num_decode_requests", "sum_decode_kv_tokens", "sum_prefill_tokens"],
        "stores": {kind: {"baseline": 5.0 + index, "trees": trees} for index, kind in enumerate(_STORES[worker])},
        "metadata": {"trainer": "snapshot-test"},
    }


_IDENTITY_KEYS = ("model", "system", "backend", "worker_type")


def _checkpoint(directory: Path, worker: str, adaptation: str) -> dict:
    """A seed checkpoint whose split ignores the feature that explains its rows.

    Defaults are deliberately left unset, so the learner uses the public
    structural recipe and can rebuild the split.
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    directory.mkdir()
    config = {
        "model": "structural-test",
        "system": "test-gpu",
        "backend": "vllm",
        "worker_type": worker,
        "estimation_mode": "fpm_learned",
        "estimator_config": {
            "correction": {"enabled": adaptation != "online"},
            "fpm_learned": {"checkpoint_path": str(directory), "adaptation": adaptation},
        },
    }
    wall = [3.0, 9.0, 3.0, 9.0]
    table = pa.table(
        {
            "row_id": pa.array([0, 1, 2, 3], type=pa.uint64()),
            "y": pa.array(wall, type=pa.float64()),
            "wall_time_ms": pa.array(wall, type=pa.float64()),
            "x0": pa.array([8.0, 8.0, 256.0, 256.0], type=pa.float64()),
            "x1": pa.array([20_000.0, 400_000.0, 20_000.0, 400_000.0], type=pa.float64()),
        }
    )
    pq.write_table(table, directory / "rows.parquet", compression="zstd")
    identity = RustForwardPassPerfModel.learned_identity({key: config[key] for key in _IDENTITY_KEYS})
    features = ["num_decode_requests", "sum_decode_kv_tokens"]
    tree = _stump(0, 64.5, 6.0, 6.0)
    artifact = {
        "schema": "aic_fpm_learned_forward_perf",
        "schema_version": 1,
        "worker_type": worker,
        "target": "ms",
        "features": features,
        "stores": {kind: {"baseline": 0.0, "trees": [tree]} for kind in _STORES[worker]},
        "metadata": {"identity": identity},
    }
    (directory / "model.json").write_text(json.dumps(artifact))
    seed = {
        "rows_file": "rows.parquet",
        "rows_sha256": _digest(directory / "rows.parquet"),
        "row_count": 4,
        "bin_thresholds": [[64.5], [100_000.5]],
        "n_bins_non_missing": [2, 2],
        "missing_values_bin_idx": 255,
        "loss": "squared_error",
        "learning_rate": 1.0,
        "l2_regularization": 0.0,
        "min_samples_leaf": 1,
        "max_depth": None,
        "max_leaf_nodes": 2,
    }
    manifest = {
        "schema": "aisimulate_fpm_training_checkpoint",
        "schema_version": 1,
        "artifact_file": "model.json",
        "artifact_sha256": _digest(directory / "model.json"),
        "identity": identity,
        "features": features,
        "target": "ms",
        "shrinkage": "included_in_leaf_values",
        "stores": dict.fromkeys(_STORES[worker], seed),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return config


def _regression(worker: str, regression: dict) -> dict:
    return {
        "model": "test/model",
        "system": "test",
        "backend": "vllm",
        "worker_type": worker,
        "estimation_mode": "fpm_regression",
        "estimator_config": {"fpm_regression": regression},
    }


_ERROR_THRESHOLD = {
    "kind": "error_threshold",
    "relative_tolerance": 0.02,
    "absolute_tolerance_ms": 0.01,
    "window": 6,
    "trigger": 2,
    "cooldown": 3,
    "startup_observations": 4,
}


def _kind_case(tmp_path: Path, kind: str):
    """Return (build, worker, warm, next) for one snapshot kind."""
    if kind == "error_threshold":
        config = _regression("aggregated", {"fit": {"linear": {"update_policy": _ERROR_THRESHOLD}}})
        return (lambda: RustForwardPassPerfModel.best_available(config)), "aggregated", _stream(31, 90), _stream(32, 30)
    if kind == "selected":
        config = _regression(
            "aggregated",
            {
                "fit": {"linear": {"feature_axes": ["attention", "moe", "n"], "non_negative": False}},
                "sampling": {"axes": ["attention", "moe", "logN"], "bins_per_axis": [3, 3, 2]},
            },
        )
        return (lambda: RustForwardPassPerfModel.best_available(config)), "aggregated", _stream(33, 90), _stream(34, 30)
    if kind == "fpm5":
        config = _regression("aggregated", {"fit": {"linear": {"feature_set": "fpm5"}}})
        return (lambda: RustForwardPassPerfModel.best_available(config)), "aggregated", _stream(39, 90), _stream(40, 30)
    if kind in ("learned_frozen", "from_learned"):
        artifact = tmp_path / "forest.json"
        artifact.write_text(json.dumps(_forest("aggregated")))
        if kind == "from_learned":
            options = {"min_observations": 3}
            build = lambda: RustForwardPassPerfModel.from_learned(json.loads(artifact.read_text()), options)
        else:
            config = {
                "model": "learned/model",
                "system": "test-gpu",
                "backend": "vllm",
                "worker_type": "aggregated",
                "estimation_mode": "fpm_learned",
                "estimator_config": {
                    "correction": {"mode": "hybrid", "min_observations": 3},
                    "fpm_learned": {"artifact_path": str(artifact), "adaptation": "frozen"},
                },
            }
            build = lambda: RustForwardPassPerfModel.best_available(config)
        return build, "aggregated", _stream(35, 70), _stream(36, 30)
    # Online experts: warm long enough for default eager structural refreshes
    # (every 32 batches), continue for less than the recent window.
    config = _checkpoint(tmp_path / "checkpoint", "decode", kind)
    warm, next_count = (40, 23) if kind == "online" else (140, 6)
    return (
        (lambda: RustForwardPassPerfModel.best_available(config)),
        "decode",
        _stream(37, warm, "decode"),
        _stream(38, next_count, "decode"),
    )


_STATE_FILES = ("state.bin", "learned/online/state.bin", "learned/online/model.json", "learned/stable.json")


@pytest.mark.parametrize(
    "kind", ["error_threshold", "selected", "fpm5", "learned_frozen", "from_learned", "online", "causal_selector"]
)
def test_every_kind_continues_in_another_process_like_the_uninterrupted_model(tmp_path, kind):
    build, worker, warm, next_stream = _kind_case(tmp_path, kind)
    model, cold = build(), build()
    queries = _queries(worker)
    untuned = _bits(cold, queries)
    model.tune_with_fpms(warm)
    assert _bits(model, queries) != untuned
    if kind in ("online", "causal_selector"):
        stores = model.learned_online_diagnostics()["stores"].values()
        assert sum(store["rebuilt_nodes"] for store in stores) > 0
    pinned = model.save_state_snapshot(tmp_path / "snap")
    chunks = [next_stream[index : index + 7] for index in range(0, len(next_stream), 7)]
    child = _child(_continue(pinned), chunks, queries, tmp_path / "child")
    for chunk in chunks:
        model.tune_with_fpms(chunk)
        cold.tune_with_fpms(chunk)
    assert child["bits"] == _bits(model, queries)
    assert child["readiness"] == model.readiness_report()
    model.save_state_snapshot(tmp_path / "parent")
    compared = [name for name in _STATE_FILES if (tmp_path / "parent" / name).exists()]
    assert compared and compared == [name for name in _STATE_FILES if (tmp_path / "child" / name).exists()]
    for name in compared:
        assert _digest(tmp_path / "child" / name) == _digest(tmp_path / "parent" / name), name
    # A cold model fed only the continuation estimates differently.
    assert _bits(cold, queries) != _bits(model, queries)
