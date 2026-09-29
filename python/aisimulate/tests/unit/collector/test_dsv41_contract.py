# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import shutil
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest
from collector.sglang.collect_dsv41_module import aggregate_rank_records, get_dsv41_module_test_cases
from collector.sglang.dsv41_contract import (
    build_manifest,
    operation_geometry,
    validate_attention_geometry,
    validate_attention_manifest,
    validate_row,
    write_full_with_bounded_attention,
    write_parquet,
)

pytestmark = pytest.mark.unit


def indexer_shape_fixture():
    manifest = build_manifest(4, False)
    geometry = json.loads(
        next(e["geometry"] for e in manifest["phases"]["context"] if e["component"] == "attention" and e["layer"] == 2)
    )
    weight = lambda shape: SimpleNamespace(weight=SimpleNamespace(shape=shape))
    indexer = SimpleNamespace(
        n_heads=32,
        n_local_heads=32,
        index_head_dim=128,
        index_topk=512,
        wq_b=weight((4096, 1280)),
        weights_proj=weight((32, 5120)),
    )
    attention = SimpleNamespace(
        n_local_heads=16,
        n_local_groups=2,
        head_dim=512,
        q_lora_rank=1280,
        o_lora_rank=1024,
        compress_ratio=2,
        indexer=indexer,
    )
    return attention, geometry


def test_loaded_indexer_heads_and_projection_shapes_match_consumer_identity():
    attention, geometry = indexer_shape_fixture()
    validate_attention_geometry(attention, geometry, 2)
    with pytest.raises(RuntimeError, match="indexer n_heads"):
        validate_attention_geometry(attention, geometry | {"index_n_heads": 8}, 2)
    attention.indexer.n_local_heads = 8
    with pytest.raises(RuntimeError, match="indexer n_local_heads"):
        validate_attention_geometry(attention, geometry, 2)


@pytest.mark.parametrize(("name", "shape"), [("wq_b", (1024, 1280)), ("weights_proj", (8, 5120))])
def test_loaded_indexer_projection_shards_cannot_masquerade_as_replicated(name, shape):
    attention, geometry = indexer_shape_fixture()
    getattr(attention.indexer, name).weight.shape = shape
    with pytest.raises(RuntimeError, match=f"indexer {name} weight shape"):
        validate_attention_geometry(attention, geometry, 2)


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_attention_manifest_rejects_wrong_heads_in_either_phase(phase):
    attention, geometry = indexer_shape_fixture()
    entry = {"component": "attention", "layer": 0, "geometry": operation_geometry(geometry)}
    manifest = {"phases": {"context": [entry.copy()], "generation": [entry.copy()]}}
    manifest["phases"][phase][0]["geometry"] = operation_geometry(geometry | {"index_n_heads": 8})
    with pytest.raises(RuntimeError, match="indexer n_heads"):
        validate_attention_manifest([SimpleNamespace(self_attn=attention)], manifest)


def test_attention_manifest_binds_nonowner_labels_to_actual_owner():
    attention, geometry = indexer_shape_fixture()
    reuse = SimpleNamespace(**vars(attention) | {"indexer": None})
    entries = [
        {"component": "attention", "layer": 0, "geometry": operation_geometry(geometry)},
        {
            "component": "attention",
            "layer": 1,
            "geometry": operation_geometry(geometry | {"role": "reuse", "index_n_heads": 8}),
        },
    ]
    manifest = {"phases": {"context": entries, "generation": entries}}
    with pytest.raises(RuntimeError, match="indexer n_heads.*layer 1"):
        validate_attention_manifest([SimpleNamespace(self_attn=attention), SimpleNamespace(self_attn=reuse)], manifest)


@pytest.mark.parametrize("entrypoint", ["module", "forward"])
def test_both_collection_paths_validate_geometry_before_wrapping(monkeypatch, entrypoint):
    import sys

    from collector.sglang.dsv41_native_runner import ComponentRecorder, validate_forward_contract

    # No CUDA/module methods exist: a malformed identity must fail before
    # any instrumentation or forward execution can use them.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", SimpleNamespace(tensor_model_parallel_all_reduce=None))
    layer_type = type("DeepseekV4DecoderLayer", (), {})
    layers = [layer_type() for _ in range(40)]
    attention, geometry = indexer_shape_fixture()
    for layer in layers:
        layer.self_attn = attention
    bad = operation_geometry(geometry | {"index_n_heads": 8})
    manifest = {
        "phases": {
            phase: [{"component": "attention", "layer": i, "geometry": bad} for i in range(40)]
            for phase in ("context", "generation")
        }
    }
    runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: layers))
    with pytest.raises(RuntimeError, match="indexer n_heads"):
        (ComponentRecorder if entrypoint == "module" else validate_forward_contract)(runner, manifest)


def row():
    manifest = build_manifest(4, False)
    attention = next(e for e in manifest["phases"]["generation"] if e["component"] == "attention")
    return {
        "component": "attention",
        "geometry": attention["geometry"],
        "batch_size": 1,
        "prefix": 0,
        "x": 129,
        "latency": 0.125,
        "kernel_source": "test.native_dispatch",
        "measurement_scope": "local_compute",
        "source_sha256": "a" * 64,
        "config_sha256": manifest["config_sha256"],
        "runtime_digest": "sha256:" + "b" * 64,
        "used_cuda_graph": False,
        "sample_count": 5,
        "kv_seed_regime": "real_kv",
        "execution_profile": "full",
    }


def test_manifest_matches_native_graph_and_profiles(monkeypatch):
    monkeypatch.delenv("DSV41_EXECUTION_PROFILE", raising=False)
    full, replay = build_manifest(4, False), build_manifest(4, True)
    assert full["config_sha256"] == replay["config_sha256"]
    assert full["config_sha256"] == "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
    for entry in replay["phases"]["context"]:
        if entry["component"] == "attention":
            assert json.loads(entry["geometry"])["bounded_prefill"] == (entry["layer"] >= 21)
    for manifest in (full, replay):
        for entries in manifest["phases"].values():
            for entry in entries:
                if entry["component"] == "attention":
                    geometry = json.loads(entry["geometry"])
                    assert "kv_cache_layout" not in geometry
                    assert geometry["index_n_heads"] == 32
                    assert geometry["fmha_quant_mode"] == "fp8"
    assert operation_geometry({"name": "display", "n": 32, "k": 64}) == '{"k":64,"n":32}'
    assert len(get_dsv41_module_test_cases()) == 8


@pytest.mark.parametrize("profile", ["full", "decoder_bounded"])
def test_case_selection_includes_only_requested_execution_profile(monkeypatch, profile):
    monkeypatch.setenv("DSV41_EXECUTION_PROFILE", profile)
    cases = get_dsv41_module_test_cases()
    assert len(cases) == 4
    assert {case["params"][2] for case in cases} == {profile}


def test_case_selection_rejects_unknown_execution_profile(monkeypatch):
    monkeypatch.setenv("DSV41_EXECUTION_PROFILE", "unqualified")
    with pytest.raises(ValueError, match="DSV41_EXECUTION_PROFILE"):
        get_dsv41_module_test_cases()


@pytest.mark.parametrize(
    "mutation",
    [
        {"kv_seed_regime": "zero_filled"},
        {"measurement_scope": "whole_module_includes_collective"},
        {"runtime_digest": "latest"},
        {"prefix": 3},
        {"batch_size": 1.5},
        {"latency": float("nan")},
    ],
)
def test_rejects_unqualified_measurements(mutation):
    with pytest.raises(ValueError):
        validate_row(row() | mutation)


def test_parquet_contract_rejects_duplicate_and_mixed_provenance(tmp_path):
    target = tmp_path / "dsv41_module_perf.parquet"
    write_parquet([row()], target)
    schema = pq.read_schema(target)
    assert str(schema.field("x").type) == "int64"
    assert str(schema.field("used_cuda_graph").type) == "bool"
    with pytest.raises(ValueError, match="duplicate"):
        write_parquet([row(), row()], target)
    with pytest.raises(ValueError, match="one immutable"):
        write_parquet([row(), row() | {"x": 130, "source_sha256": "c" * 64}], target)
    with pytest.raises(ValueError, match="one immutable"):
        write_parquet([row(), row() | {"x": 130, "execution_profile": "decoder_bounded"}], target)


def test_bounded_context_requires_actual_tail_and_real_prefix():
    manifest = build_manifest(4, True)
    entry = next(e for e in manifest["phases"]["context"] if e["component"] == "attention" and e["layer"] == 21)
    point = row() | {"geometry": entry["geometry"], "prefix": 1152, "x": 128, "execution_profile": "decoder_bounded"}
    validate_row(point)
    for change in ({"x": 129}, {"execution_profile": "full"}, {"kv_seed_regime": "n/a"}):
        with pytest.raises(ValueError):
            validate_row(point | change)


def joint_export_fixture():
    # Synthetic selection witnesses only, never GPU observations or accuracy.
    full, bounded = build_manifest(4, False), build_manifest(4, True)

    def point(manifest, layer, prefix, latency):
        entry = next(e for e in manifest["phases"]["context"] if e["component"] == "attention" and e["layer"] == layer)
        return row() | {
            "geometry": entry["geometry"],
            "x": 128,
            "prefix": prefix,
            "latency": latency,
            "execution_profile": manifest["execution_profile"],
        }

    return [point(full, 0, 64, 0.125), point(full, 21, 64, 0.25)], [
        point(bounded, 0, 320, 0.375),
        point(bounded, 21, 64, 0.75),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "empty_base",
        "empty_additions",
        "base_profile",
        "addition_profile",
        "unknown_profile",
        "primitive",
        "duplicate_base",
        "duplicate_addition",
        "overlap",
        "source_sha256",
        "config_sha256",
        "runtime_digest",
        "used_cuda_graph",
        "missing_geometry",
        "invalid_tail",
        "different_columns",
    ],
)
def test_joint_export_rejects_unqualified_cohorts_without_writing(tmp_path, failure):
    full, bounded = joint_export_fixture()
    if failure == "empty_base":
        full.clear()
    elif failure == "empty_additions":
        bounded.clear()
    elif failure == "base_profile":
        full = [r | {"execution_profile": "decoder_bounded"} for r in full]
    elif failure in ("addition_profile", "unknown_profile"):
        bounded = [bounded[0] | {"execution_profile": "full" if failure == "addition_profile" else "unknown"}]
    elif failure == "primitive":
        bounded = [
            bounded[0]
            | {
                "component": "mhc",
                "geometry": '{"hc_mult":4,"hidden_size":5120,"sinkhorn_iters":20}',
                "prefix": 0,
                "kv_seed_regime": "n/a",
            }
        ]
    elif failure == "duplicate_base":
        full.append(full[0].copy())
    elif failure == "duplicate_addition":
        bounded.append(bounded[0].copy())
    elif failure == "overlap":
        bounded = [full[0] | {"execution_profile": "decoder_bounded"}]
    elif failure in ("source_sha256", "config_sha256", "runtime_digest", "used_cuda_graph"):
        value = True if failure == "used_cuda_graph" else ("sha256:" if failure == "runtime_digest" else "") + "c" * 64
        bounded = [r | {failure: value} for r in bounded]
    elif failure == "missing_geometry":
        bounded[0]["geometry"] = operation_geometry(json.loads(bounded[0]["geometry"]) | {"num_heads": 8})
    elif failure == "invalid_tail":
        bounded[1]["x"] = 129
    elif failure == "different_columns":
        bounded[0]["extra_evidence"] = "must not be silently dropped"
    target = tmp_path / "joint.parquet"
    with pytest.raises(ValueError):
        write_full_with_bounded_attention(full, bounded, target)
    assert not target.exists()


def test_joint_export_preserves_profiles_and_native_silicon_selection(tmp_path):
    import aisimulate_core._native as core
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import PerfDatabase

    full, bounded = joint_export_fixture()
    original = json.loads(json.dumps([*full, *bounded]))
    with pytest.raises(ValueError, match="one immutable"):
        write_parquet([*full, *bounded], tmp_path / "ordinary.parquet")
    databases = []
    for name in ("full", "joint"):
        root = tmp_path / name
        data = root / "data/gb300/sglang/dev-test-joint-export"
        data.mkdir(parents=True)
        package = Path(__file__).resolve().parents[3]
        shutil.copy(package / "src/aisimulate_core/systems/gb300.yaml", root / "gb300.yaml")
        target = data / "dsv41_module_perf.parquet"
        if name == "full":
            write_parquet(full, target)
        else:
            write_full_with_bounded_attention(full, bounded, target)
            assert pq.read_table(target).to_pylist() == original
        databases.append(
            PerfDatabase(
                "gb300", "sglang", "dev-test-joint-export", str(root), database_mode="SILICON", strict_provenance=False
            )
        )
    assert [*full, *bounded] == original

    def predict(database, point):
        operation = core.op_from_spec_json(
            json.dumps({"Dsv41Attention": json.loads(point["geometry"]) | {"name": "context_attention"}})
        )
        result = _evaluate_single_op(
            database, operation, is_context=True, batch_size=1, s=point["x"], prefix=point["prefix"]
        )
        assert result.source == "silicon"
        return float(result)

    for point in full:
        assert predict(databases[0], point) == pytest.approx(point["latency"])
        assert predict(databases[1], point) == pytest.approx(point["latency"])
    for point in bounded:
        with pytest.raises(RuntimeError, match="no measured SILICON data"):
            predict(databases[0], point)
        assert predict(databases[1], point) == pytest.approx(point["latency"])


def test_rank_aggregation_requires_complete_distinct_invocations(tmp_path):
    paths = [tmp_path / f"rank-{rank}.jsonl" for rank in range(2)]
    for rank, path in enumerate(paths):
        path.write_text(
            "\n".join(
                json.dumps(row() | {"sample": 1, "invocation": invocation, "tp_rank": rank, "latency": value + rank})
                for invocation, value in ((1, 1.0), (2, 3.0))
            )
        )
    result = aggregate_rank_records(paths, 2)
    assert result[0]["latency"] == 3.0  # median(max(1,2), max(3,4))
    assert result[0]["sample_count"] == 2
    with pytest.raises(ValueError, match="rank files"):
        aggregate_rank_records(paths[:1], 2)
    paths[0].write_text(paths[0].read_text() + "\n" + paths[0].read_text().splitlines()[0])
    with pytest.raises(ValueError, match="duplicate TP rank"):
        aggregate_rank_records(paths, 2)


@pytest.mark.parametrize(
    ("measured_change", "error"),
    [
        ({}, None),
        ({"index_n_heads": 8}, RuntimeError),
        ({"head_dim": 256}, RuntimeError),
        ({"fmha_quant_mode": "bfloat16"}, RuntimeError),
        ({"kv_cache_layout": "logical_fp4"}, ValueError),
        ({"unknown_dimension": 1}, ValueError),
    ],
)
def test_writer_to_native_silicon_query_requires_exact_measured_identity(tmp_path, measured_change, error):
    import aisimulate_core._native as core
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.perf_database import PerfDatabase

    package = Path(__file__).resolve().parents[3]
    shutil.copy(package / "src/aisimulate_core/systems/gb300.yaml", tmp_path / "gb300.yaml")
    data = tmp_path / "data/gb300/sglang/0.0.0.dev0"
    data.mkdir(parents=True)
    point = row()
    measured = point | {"geometry": operation_geometry(json.loads(point["geometry"]) | measured_change)}
    write_parquet([measured], data / "dsv41_module_perf.parquet")
    operation = core.op_from_spec_json(
        json.dumps(
            {
                "Dsv41Attention": json.loads(point["geometry"])
                | {"name": "generation_attention", "kv_cache_layout": "sglang_fp8_bf16"}
            }
        )
    )
    database = PerfDatabase(
        "gb300", "sglang", "0.0.0.dev0", str(tmp_path), database_mode="SILICON", strict_provenance=False
    )
    if error is not None:
        message = "noncanonical" if error is ValueError else "[Dd]sv41|[Dd]eep[Ss]eek|[Mm]issing"
        with pytest.raises(error, match=message):
            _evaluate_single_op(database, operation, is_context=False, batch_size=1, s=129, x=1)
        return
    result = _evaluate_single_op(database, operation, is_context=False, batch_size=1, s=129, x=1)
    assert float(result) == pytest.approx(0.125)
    assert result.source == "silicon"


@pytest.mark.parametrize("dtype", ["half", "bfloat16", "float32", "float8"])
@pytest.mark.parametrize("message_size", [10240, 10241])
def test_baseline_rank_admission_converts_nccl_bytes_to_elements(tmp_path, dtype, message_size):
    from collector.sglang.collect_dsv41_module import aggregate_baseline_records

    paths = []
    for rank in range(2):
        path = tmp_path / f"baseline-rank-{rank}.jsonl"
        point = {
            "kind": "nccl",
            "nccl_dtype": dtype,
            "num_gpus": 2,
            "op_name": "all_reduce",
            "message_size": message_size,
            "latency": rank + 1.0,
            "sample": 2,
            "tp_rank": rank,
            "source_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "runtime_digest": "sha256:" + "c" * 64,
            "used_cuda_graph": False,
            "kernel_source": "torch.distributed.nccl.all_reduce",
            "execution_profile": "decoder_bounded",
        }
        path.write_text(json.dumps(point) + "\n")
        paths.append(path)
    if dtype not in ("half", "bfloat16"):
        with pytest.raises(ValueError, match="supported 16-bit dtype"):
            aggregate_baseline_records(paths, 2)
        return
    if message_size % 2:
        with pytest.raises(ValueError, match="whole elements"):
            aggregate_baseline_records(paths, 2)
        return
    result = aggregate_baseline_records(paths, 2)["nccl"][0]
    assert result["message_size"] == 5120
    assert result["wire_dtype"] == "bfloat16"
    assert result["latency"] == 2.0
    paths[0].write_text(paths[0].read_text() * 2)
    with pytest.raises(ValueError, match="duplicate baseline rank/sample"):
        aggregate_baseline_records(paths, 2)


def test_output_profile_binding_rejects_mixed_campaigns(tmp_path):
    from collector.sglang.collect_dsv41_module import bind_output_profile

    destination = str(tmp_path / "dsv41_module_perf.txt")
    bind_output_profile(destination, "full")
    bind_output_profile(destination, "full")
    with pytest.raises(ValueError, match="separate output tables"):
        bind_output_profile(destination, "decoder_bounded")


@pytest.fixture
def module_worker_case(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV41_CHECKPOINT", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("DSV41_RUNTIME_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv("DSV41_PROMPT_FILE", str(tmp_path / "prompts.json"))
    return {
        "model_path": "deepseek-ai/DeepSeek-V4.1-Flash",
        "tp_size": 4,
        "execution_profile": "full",
        "sweep": {"query_lengths": [128], "prefix_lengths": [0], "batch_sizes": [1], "decode_steps": 1},
        "perf_filename": str(tmp_path / "dsv41_module_perf.txt"),
    }


@pytest.mark.parametrize("installed_version", ["0.5.4", "0.0.0.dev0+g1aa0e962"])
def test_module_worker_persists_installed_sglang_version(monkeypatch, module_worker_case, installed_version):
    from collector.sglang import collect_dsv41_module

    get_version = Mock(return_value=installed_version)
    monkeypatch.setattr(metadata, "version", get_version)

    def invoke_native(command, **kwargs):
        get_version.assert_called_once_with("sglang")
        output = Path(command[command.index("--output") + 1])
        for rank in range(module_worker_case["tp_size"]):
            point = row() | {"sample": 0, "invocation": 0, "tp_rank": rank}
            (output / f"rank-{rank}.jsonl").write_text(json.dumps(point) + "\n")
        (output / "COMPLETE").touch()

    native_run = Mock(side_effect=invoke_native)
    monkeypatch.setattr(collect_dsv41_module.subprocess, "run", native_run)
    collect_dsv41_module.run_dsv41_module_worker(**module_worker_case)

    native_run.assert_called_once()
    with Path(module_worker_case["perf_filename"]).open(newline="") as handle:
        measured = list(csv.DictReader(handle))
    assert len(measured) == 1
    assert measured[0]["framework"] == "sglang"
    assert measured[0]["version"] == installed_version
    assert float(measured[0]["latency"]) == 0.125


def test_module_worker_missing_sglang_metadata_fails_before_gpu(monkeypatch, module_worker_case):
    from collector.sglang import collect_dsv41_module

    missing = metadata.PackageNotFoundError("sglang")
    get_version = Mock(side_effect=missing)
    monkeypatch.setattr(metadata, "version", get_version)
    native_run = Mock()
    monkeypatch.setattr(collect_dsv41_module.subprocess, "run", native_run)

    with pytest.raises(metadata.PackageNotFoundError) as error:
        collect_dsv41_module.run_dsv41_module_worker(**module_worker_case)

    assert error.value is missing
    get_version.assert_called_once_with("sglang")
    native_run.assert_not_called()
    assert not Path(module_worker_case["perf_filename"]).exists()


def test_registry_provenance_covers_execution_and_geometry_dependencies(tmp_path):
    from collector import provenance
    from collector.op_catalog import family_for_perf_file, load_family_map

    package_root = Path(__file__).resolve().parents[3]
    module = "collector.sglang.collect_dsv41_module"
    closures = provenance.load_closures(package_root / "collector/hash_closures.yaml")
    assert module in provenance.enumerate_provenance_modules()
    assert family_for_perf_file("dsv41_module_perf.parquet", load_family_map()) == "dsv41"
    # Use the real closure, then change the executed runner and model geometry
    # independently: either must invalidate the collector identity.
    dependencies = {module.replace(".", "/") + ".py", *provenance.SHARED_CORE}
    dependencies.update(provenance._expand_closure_files(package_root, closures[module]))
    for relative in dependencies:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root / relative, target)
    baseline = provenance.collector_hash(module, tmp_path, closures)
    for relative in (
        "collector/sglang/dsv41_native_runner.py",
        "collector/sglang/dsv41_humming.py",
        "collector/sglang/dsv41_workloads.py",
        "src/aisimulate_core/sdk/models/deepseek_v41.py",
        "src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4.1-Flash_config.json",
    ):
        target = tmp_path / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n ")
        assert provenance.collector_hash(module, tmp_path, closures) != baseline, relative
        target.write_bytes(original)
    assert provenance.collector_hash(module, tmp_path, closures) == baseline
