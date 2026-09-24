# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY allocator configuration and original receipt integrity; no GPU use."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward import glm53flash_validation as validation
from collector.fpm_forward import sglang_allocator as allocator
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import FPMCollectionOptions, reject_fpm_arguments_without_fpm
from collector.fpm_forward.database import _validate_allocator_deployments
from collector.fpm_forward.runner import _cell_generator_overrides
from collector.fpm_forward.sglang_artifact import file_receipt, validate_sglang_repetitions
from collector.fpm_forward.shards import make_shards

from tests.unit.collector.test_fpm_glm53flash_planning import plan
from tests.unit.collector.test_fpm_glm53flash_sglang_artifact import artifact
from tests.unit.collector.test_fpm_sglang_memory_option import rebuild
from tests.unit.collector.test_glm53flash_validation import write_plan

pytestmark = pytest.mark.unit
MODEL = "zai-org/GLM-5.3-Flash"


def worker(rank, policy, identity, hardware):
    return {
        "schema": allocator.SCHEMA,
        "rank": rank,
        "pid": 100 + rank,
        "run_id": "run",
        "execution_identity": identity,
        "hardware": hardware,
        "requested_policy": policy,
        "environment": allocator.configured_environment(policy["max_split_size_mb"]),
        "allocator_backend": "native",
        "max_split_size_bytes": -1 if policy["max_split_size_mb"] is None else policy["max_split_size_mb"] * (1 << 20),
        "torch": {
            "version": "TEST_ONLY_TORCH",
            "git_revision": "a" * 40,
            "cuda_version": "13.0",
            "files": {
                name: {"path": "/TEST_ONLY/torch/" + (name if name != "_C" else "_C.so"), "sha256": "b" * 64}
                for name in (*allocator.SOURCE_FILES, *allocator.LIBRARY_FILES, "_C")
            },
        },
    }


def with_allocator(tmp_path, requested=16384):
    cell, payload = artifact(tmp_path)
    cell.sglang_allocator_max_split_size_mb = requested
    policy = allocator.request_policy(requested)
    payload["allocator_policy"] = policy
    evidence = payload["input_provenance"]["native_forward_manifest"]
    evidence["allocator_identities"] = []
    for rank in range(cell.topology.tp):
        hardware = json.loads((tmp_path / f"state-layout-rank-{rank}.json").read_text())["hardware"]
        data = worker(rank, policy, payload["execution_identity"], hardware)
        path = tmp_path / f"allocator-identity-rank-{rank}.json"
        path.write_text(json.dumps(data))
        ref = {"tp_rank": rank, **file_receipt(path)}
        evidence["allocator_identities"].append(ref)
        trace = evidence["traces"][rank]
        path = tmp_path / trace["file"]
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for row in records:
            row["allocator_policy"] = policy
            row["allocator_identity_sha256"] = ref["sha256"]
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        trace.update(file_receipt(path))
    return cell, payload


@pytest.mark.parametrize("value", [True, False, 19, 0, -1, 20.0, "20", 1 << 63])
def test_exact_integer_native_limit_and_cli_negative(value):
    with pytest.raises(ValueError, match="integer"):
        FPMCollectionOptions.from_args(argparse.Namespace(fpm_max_gpus=4, sglang_allocator_max_split_size_mb=value))
    if value != "20":
        with pytest.raises(SystemExit):
            _parser().parse_args(["--gpu", "gb300", f"{allocator.OPTION}={value}"])


def test_minimum_and_public_flag_and_non_fpm_scope():
    args = _parser().parse_args(["--gpu", "gb300", "--fpm-max-gpus", "4", allocator.OPTION, "20"])
    assert FPMCollectionOptions.from_args(args).sglang_allocator_max_split_size_mb == 20
    with pytest.raises(ValueError, match="FPM-only"):
        reject_fpm_arguments_without_fpm(argparse.Namespace(ops=["gemm"], sglang_allocator_max_split_size_mb=20))


def test_plan_identity_render_shards_and_backend(tmp_path):
    base = plan(tmp_path, "sglang", MODEL)
    assert rebuild(base, sglang_allocator_max_split_size_mb=None).to_dict() == base.to_dict()
    assert "sglang_allocator_max_split_size_mb" not in base.to_dict()["options"]
    candidates = [
        rebuild(base, sglang_allocator_max_split_size_mb=n, shard_token_budget=1_000_000) for n in (16384, 32768)
    ]
    assert len({base.sha256, *(c.sha256 for c in candidates)}) == 3
    assert not {c.cell_id for c in candidates[0].cells} & {c.cell_id for c in candidates[1].cells}
    for campaign in candidates:
        for shard in make_shards(campaign):
            assert (
                shard.plan.options.sglang_allocator_max_split_size_mb
                == campaign.options.sglang_allocator_max_split_size_mb
            )
            cell = shard.plan.cells[0]
            args = _cell_generator_overrides(shard.plan, cell, {})["params"]["agg"]["extra_cli_args"]
            assert args.count(allocator.OPTION) == 1
            assert int(args[args.index(allocator.OPTION) + 1]) == cell.sglang_allocator_max_split_size_mb
    with pytest.raises(ValueError, match="between frozen plan and cell"):
        _cell_generator_overrides(candidates[0], base.cells[0], {})
    with pytest.raises(ValueError, match="requires backend=sglang"):
        rebuild(plan(tmp_path, "vllm", MODEL), sglang_allocator_max_split_size_mb=16384)
    with pytest.raises(ValueError, match="conflicting"):
        _cell_generator_overrides(
            candidates[0],
            candidates[0].cells[0],
            {"K8sConfig": {"extra_env": [{"name": "PYTORCH_ALLOC_CONF", "value": "expandable_segments:True"}]}},
        )


@pytest.mark.parametrize("key", allocator.ENV_KEYS)
@pytest.mark.parametrize("value", ["", "backend:cudaMallocAsync", "max_split_size_mb:512"])
def test_inherited_other_or_empty_settings_never_silently_mix(monkeypatch, key, value):
    for name in allocator.ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="inherited"):
        allocator.prepare_environment([allocator.OPTION, "16384"])


def test_setting_precedes_any_framework_import_in_fresh_process():
    env = {k: v for k, v in os.environ.items() if k not in allocator.ENV_KEYS}
    package = Path(allocator.__file__).parents[2]
    script = """import os,sys
from collector.fpm_forward.sglang_allocator import prepare_environment
assert 'torch' not in sys.modules and 'sglang' not in sys.modules
assert prepare_environment(['--sglang-allocator-max-split-size-mb','16384'])['max_split_size_mb']==16384
assert os.environ['PYTORCH_CUDA_ALLOC_CONF']=='backend:native,max_split_size_mb:16384'
assert 'torch' not in sys.modules and 'sglang' not in sys.modules
"""
    env["PYTHONPATH"] = str(package)
    result = subprocess.run([sys.executable, "-c", script], cwd="/tmp", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_late_torch_and_duplicate_request_reject(monkeypatch):
    for name in allocator.ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="before Torch"):
        allocator.prepare_environment([allocator.OPTION, "16384"])
    with pytest.raises(ValueError, match="duplicate"):
        allocator.prepare_environment([allocator.OPTION, "16384", allocator.OPTION, "32768"])


@pytest.mark.parametrize("wrong_mapping", [False, True])
def test_observer_uses_existing_native_apis_and_exact_loaded_libraries(tmp_path, monkeypatch, wrong_mapping):
    root = tmp_path / "torch"
    for name in (*allocator.SOURCE_FILES, *allocator.LIBRARY_FILES, "_C.so"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"TEST_ONLY original file " + name.encode())
    for name in allocator.ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native,max_split_size_mb:16384")
    policy = allocator.prepare_environment([allocator.OPTION, "16384"])
    calls = []

    def backend():
        calls.append("backend")
        return "native"

    def stats():
        calls.append("stats")
        return {"max_split_size": 16384 * (1 << 20)}

    torch = SimpleNamespace(
        __file__=str(root / "__init__.py"),
        __version__="TEST_ONLY_TORCH",
        version=SimpleNamespace(git_version="a" * 40, cuda="13.0"),
        _C=SimpleNamespace(__file__=str(root / "_C.so")),
        cuda=SimpleNamespace(memory=SimpleNamespace(get_allocator_backend=backend), memory_stats=stats),
    )
    original_read = Path.read_text
    maps = "\n".join(f"0-1 r--p 0 0:0 0 {root / name}" for name in allocator.LIBRARY_FILES)
    if wrong_mapping:
        maps += "\n0-1 r--p 0 0:0 0 /another/libtorch_cuda.so"
    monkeypatch.setattr(
        Path, "read_text", lambda p, *a, **kw: maps if str(p) == "/proc/self/maps" else original_read(p, *a, **kw)
    )
    if wrong_mapping:
        with pytest.raises(ValueError, match="mapping differs"):
            allocator.observe_worker(torch, rank=0, run_id="run", execution_identity={}, policy=policy, hardware={})
        assert calls == []
    else:
        result = allocator.observe_worker(
            torch, rank=0, run_id="run", execution_identity={}, policy=policy, hardware={}
        )
        assert calls == ["backend", "stats"]
        for ref in result["torch"]["files"].values():
            assert ref["sha256"] == hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()
        assert result["pid"] == os.getpid()


@pytest.mark.parametrize("abbreviated", [False, True])
def test_direct_driver_allocator_request_matches_early_parser_before_server_construction(monkeypatch, abbreviated):
    from collector.fpm_forward import sglang_driver

    from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS

    for name in allocator.ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    if not abbreviated:
        # Track the mutation so this positive test cannot leak allocator state.
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native,max_split_size_mb:16384")
    calls = []

    class ServerArgs:
        @staticmethod
        def add_cli_args(parser):
            return None

        @staticmethod
        def from_cli_args(args):
            calls.append(args.sglang_allocator_max_split_size_mb)
            raise RuntimeError("TEST_ONLY stop before constructing native ServerArgs")

    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=ServerArgs))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=None))
    flag = allocator.OPTION.removesuffix("-mb") if abbreviated else allocator.OPTION
    argv = [
        flag,
        "16384",
        "--benchmark-mode",
        "prefill",
        "--benchmark-points-file",
        "TEST_ONLY.json",
        "--tokenizer-revision",
        MODEL_REVISIONS[MODEL],
    ]
    if abbreviated:
        with pytest.raises(ValueError, match="pre-import policy"):
            sglang_driver.main(argv)
        assert calls == []
    else:
        with pytest.raises(RuntimeError, match="TEST_ONLY stop"):
            sglang_driver.main(argv)
        assert calls == [16384]


@pytest.mark.parametrize("cell_value", [None, 16384, 32768])
def test_acceptance_binds_frozen_plan_and_cell_allocator(tmp_path, cell_value):
    spec = write_plan(tmp_path, ("sglang", "fp8", 2, "prefill"), "holdout")
    path = tmp_path / spec["plan"]["path"]
    frozen = json.loads(path.read_text())
    frozen["options"]["sglang_allocator_max_split_size_mb"] = 16384
    if cell_value is not None:
        frozen["cells"][0]["sglang_allocator_max_split_size_mb"] = cell_value
    path.write_text(json.dumps(frozen))
    spec["plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if cell_value == 16384:
        assert (
            validation._plan_run(spec, tmp_path, "holdout")["runtime_cell"].sglang_allocator_max_split_size_mb == 16384
        )
    else:
        with pytest.raises(ValueError, match="between frozen plan and cell"):
            validation._plan_run(spec, tmp_path, "holdout")


@pytest.mark.parametrize("requested", [None, 20, 16384, 32768])
def test_complete_original_rank_receipts_and_forward_binding(tmp_path, requested):
    cell, payload = with_allocator(tmp_path, requested)
    normalized = validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")
    assert normalized["requested_policy"]["max_split_size_mb"] == requested
    assert normalized["allocator_backend"] == "native"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "wrong_plan",
        "rank",
        "boolean_rank",
        "pid",
        "hardware",
        "environment",
        "backend",
        "effective",
        "torch_source",
        "source_missing",
        "trace",
        "run",
        "precision",
    ],
)
def test_sha_valid_but_wrong_actual_evidence_rejected(tmp_path, damage):
    cell, payload = with_allocator(tmp_path)
    evidence = payload["input_provenance"]["native_forward_manifest"]
    ref = evidence["allocator_identities"][1]
    path = tmp_path / ref["file"]
    value = json.loads(path.read_text())
    if damage == "missing":
        del evidence["allocator_identities"]
    elif damage == "wrong_plan":
        cell.sglang_allocator_max_split_size_mb = 32768
    elif damage == "rank":
        value["rank"] = 0
    elif damage == "boolean_rank":
        value["rank"] = True
    elif damage == "pid":
        value["pid"] = 100
    elif damage == "hardware":
        value["hardware"]["uuid"] = "GPU-wrong"
    elif damage == "environment":
        value["environment"]["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    elif damage == "backend":
        value["allocator_backend"] = "cudaMallocAsync"
    elif damage == "effective":
        value["max_split_size_bytes"] = 32768 * (1 << 20)
    elif damage == "torch_source":
        value["torch"]["files"]["version.py"]["sha256"] = "c" * 64
    elif damage == "source_missing":
        del value["torch"]["files"]["version.py"]
    elif damage == "run":
        value["run_id"] = "another_run"
    elif damage == "precision":
        value["execution_identity"]["model_config_sha256"] = "c" * 64
    elif damage == "trace":
        trace = evidence["traces"][1]
        f = tmp_path / trace["file"]
        records = [json.loads(line) for line in f.read_text().splitlines()]
        records[0]["allocator_identity_sha256"] = "d" * 64
        f.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        trace.update(file_receipt(f))
    if damage not in ("missing", "wrong_plan", "trace"):
        path.write_text(json.dumps(value))
        ref.update(file_receipt(path))
    with pytest.raises(ValueError, match="allocator|Torch"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


def test_legacy_is_readable_but_cannot_satisfy_requested_profile(tmp_path):
    cell, payload = artifact(tmp_path)
    assert validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json") is None
    cell.sglang_allocator_max_split_size_mb = 16384
    with pytest.raises(ValueError, match="allocator"):
        validate_sglang_repetitions(cell, payload, tmp_path / "benchmark.json")


def test_policy_pair_gate_includes_actual_allocator_not_only_server_args():
    server = {"random_seed": 1, "mem_fraction_static": 0.82}
    actual = worker(0, allocator.request_policy(16384), {}, {})
    normalized = allocator.validate_worker(
        actual, rank=0, run_id="run", execution_identity={}, policy=actual["requested_policy"]
    )
    first = validation._sglang_execution_policy(server, normalized)
    for other in (None, {**normalized, "max_split_size_bytes": 32768 * (1 << 20)}):
        with pytest.raises(ValueError, match="execution policies differ"):
            validation._same_sglang_policy(
                first, validation._sglang_execution_policy(server, other), "calibration/holdout"
            )
    validation._same_sglang_policy(
        first, validation._sglang_execution_policy({**server, "random_seed": 9}, normalized), "shards"
    )


@pytest.mark.parametrize("different", [False, True])
def test_shard_union_preserves_and_checks_actual_allocator(tmp_path, monkeypatch, different):
    children = [
        {
            "key": ("sglang", "fp8", 2, "prefill"),
            "role": "calibration",
            "cell": {"cell_id": str(i)},
            "plan": {"sha256": str(i)},
            "original_point_ids": {1: i + 1},
        }
        for i in range(2)
    ]
    parent = {
        "children": children,
        "key": children[0]["key"],
        "role": "calibration",
        "points": [{"benchmark_id": i} for i in (1, 2)],
    }

    def native(run, base):
        size = 32768 if different and run is children[1] else 16384
        raw = worker(0, allocator.request_policy(size), {}, {})
        actual = allocator.validate_worker(
            raw, rank=0, run_id="run", execution_identity={}, policy=raw["requested_policy"]
        )
        return {
            "values": {1: 12},
            "request_ids": {run["cell"]["cell_id"]},
            "backend_version": "0.5.20",
            "receipts": [],
            **validation._sglang_execution_policy(
                {"random_seed": int(run["cell"]["cell_id"]), "mem_fraction_static": 0.82}, actual
            ),
        }

    monkeypatch.setattr(validation, "_native_run", native)
    if different:
        with pytest.raises(ValueError, match="across native shards"):
            validation._load_native(parent, tmp_path, "fpm")
    else:
        result = validation._load_native(parent, tmp_path, "fpm")
        assert result["values"] == {1: 12, 2: 12}
        assert result["_allocator_policy"]["requested_policy"]["max_split_size_mb"] == 16384
        assert "_allocator_policy" not in json.dumps(result["shards"])


def test_disjoint_cells_and_phases_cannot_mix_policies_in_one_deployment():
    original = {
        "backend": "sglang",
        "model_path": MODEL,
        "tp": 2,
        "cell_id": "first",
        "workload_kind": "prefill",
        "total_prefill_tokens": 128,
        "sglang_allocator_max_split_size_mb": 16384,
        "sglang_allocator_policy_sha256": "a" * 64,
    }
    another = {**original, "cell_id": "second", "workload_kind": "decode", "total_prefill_tokens": 0}
    _validate_allocator_deployments([original, another])
    for field, value in (("sglang_allocator_max_split_size_mb", 32768), ("sglang_allocator_policy_sha256", "b" * 64)):
        with pytest.raises(ValueError, match="mixes"):
            _validate_allocator_deployments([original, {**another, field: value}])
    with pytest.raises(ValueError, match="mixes"):
        _validate_allocator_deployments(
            [original, {**another, "sglang_allocator_max_split_size_mb": None, "sglang_allocator_policy_sha256": None}]
        )


def test_publication_keeps_policy_columns_and_rejects_disjoint_new_policy(tmp_path):
    import pyarrow.parquet as pq
    from collector.fpm_forward.database import aggregate_cell, write_formal_database

    from tests.unit.collector.test_fpm_forward import _synthetic_plan_and_cell

    campaign, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(campaign, cell, cell_dir, expected_attempt_id="attempt")
    campaign.backend = "sglang"
    for row in rows:
        row.update(backend="sglang", sglang_allocator_max_split_size_mb=16384, sglang_allocator_policy_sha256="a" * 64)
    path, _, _ = write_formal_database(campaign, rows, systems_root=tmp_path / "systems")
    assert pq.read_table(path).to_pylist()[0]["sglang_allocator_policy_sha256"] == "a" * 64
    unchanged = path.read_bytes()
    alternate = copy.deepcopy(rows)
    for row in alternate:
        row.update(
            cell_id="different",
            total_prefill_tokens=row["total_prefill_tokens"] + 1,
            sglang_allocator_policy_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="mixes"):
        write_formal_database(campaign, alternate, systems_root=tmp_path / "systems")
    assert path.read_bytes() == unchanged


@pytest.mark.parametrize("tamper", [False, True])
def test_consumer_rows_bind_actual_allocator_policy(tmp_path, tamper):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.unit.collector.test_glm53flash_validation import calibration_table

    path, run, native, rows = calibration_table(tmp_path)
    run["key"] = ("sglang", *run["key"][1:])
    run["cell"]["sglang_allocator_max_split_size_mb"] = 16384
    value = worker(0, allocator.request_policy(16384), {}, {})
    native["_allocator_policy"] = allocator.validate_worker(
        value, rank=0, run_id="run", execution_identity={}, policy=value["requested_policy"]
    )
    for row in rows:
        row.update(
            backend="sglang",
            timing_boundary=validation.TIMING_BOUNDARIES["sglang"],
            sglang_allocator_max_split_size_mb=16384,
            sglang_allocator_policy_sha256=allocator.digest(native["_allocator_policy"]),
        )
    if tamper:
        rows[0]["sglang_allocator_policy_sha256"] = "c" * 64
    pq.write_table(pa.Table.from_pylist(rows), path)
    if tamper:
        with pytest.raises(ValueError, match="not bound"):
            validation._bind_fpm_rows([path], run, native)
    else:
        assert validation._bind_fpm_rows([path], run, native)["rows"] == 3
