# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from collector.sglang.collect_dsv41_module import aggregate_rank_records, get_dsv41_module_test_cases
from collector.sglang.dsv41_contract import (
    build_manifest,
    operation_geometry,
    validate_attention_geometry,
    validate_attention_manifest,
    validate_row,
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


def test_manifest_matches_native_graph_and_profiles():
    full, replay = build_manifest(4, False), build_manifest(4, True)
    assert full["config_sha256"] == replay["config_sha256"]
    assert full["config_sha256"] == "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
    for entry in replay["phases"]["context"]:
        if entry["component"] == "attention":
            assert json.loads(entry["geometry"])["bounded_prefill"] == (entry["layer"] >= 21)
    assert operation_geometry({"name": "display", "n": 32, "k": 64}) == '{"k":64,"n":32}'
    assert len(get_dsv41_module_test_cases()) == 8


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


def test_bounded_context_requires_actual_tail_and_real_prefix():
    manifest = build_manifest(4, True)
    entry = next(e for e in manifest["phases"]["context"] if e["component"] == "attention" and e["layer"] == 21)
    point = row() | {"geometry": entry["geometry"], "prefix": 1152, "x": 128, "execution_profile": "decoder_bounded"}
    validate_row(point)
    for change in ({"x": 129}, {"execution_profile": "full"}, {"kv_seed_regime": "n/a"}):
        with pytest.raises(ValueError):
            validate_row(point | change)


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


def test_writer_to_native_silicon_query_roundtrip(tmp_path):
    import aiconfigurator_core._aiconfigurator_core as core
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    package = Path(__file__).resolve().parents[3]
    shutil.copy(package / "src/aiconfigurator_core/systems/gb300.yaml", tmp_path / "gb300.yaml")
    data = tmp_path / "data/gb300/sglang/0.0.0.dev0"
    data.mkdir(parents=True)
    point = row()
    write_parquet([point], data / "dsv41_module_perf.parquet")
    operation = core.op_from_spec_json(
        json.dumps({"Dsv41Attention": json.loads(point["geometry"]) | {"name": "generation_attention"}})
    )
    database = PerfDatabase(
        "gb300", "sglang", "0.0.0.dev0", str(tmp_path), database_mode="SILICON", strict_provenance=False
    )
    result = _evaluate_single_op(database, operation, is_context=False, batch_size=1, s=129, x=1)
    assert float(result) == pytest.approx(0.125)
    assert result.source == "silicon"


def test_baseline_rank_admission_converts_nccl_bytes_to_elements(tmp_path):
    from collector.sglang.collect_dsv41_module import aggregate_baseline_records

    paths = []
    for rank in range(2):
        path = tmp_path / f"baseline-rank-{rank}.jsonl"
        point = {
            "kind": "nccl",
            "nccl_dtype": "half",
            "num_gpus": 2,
            "op_name": "all_reduce",
            "message_size": 10240,
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
    result = aggregate_baseline_records(paths, 2)["nccl"][0]
    assert result["message_size"] == 5120
    assert result["wire_dtype"] == "bfloat16"
    assert result["latency"] == 2.0
    paths[0].write_text(paths[0].read_text() * 2)
    with pytest.raises(ValueError, match="duplicate baseline rank/sample"):
        aggregate_baseline_records(paths, 2)


def test_checked_in_gb300_tables_support_complete_strict_tp_forward():
    from aiconfigurator_core.sdk.engine import EngineHandle

    root = Path(__file__).resolve().parents[5] / "data/experimental/deepseek-v41/gb300-silicon"
    assert root.is_dir()
    for profile in ("full", "decoder_bounded"):
        systems = root / "indexer-identity-v2" / "initial" / profile / "systems"
        assert not list(systems.rglob("custom_allreduce_perf.parquet"))
        handle = EngineHandle.compile(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "gb300",
            "sglang",
            backend_version="0.0.0.dev0",
            tp_size=4,
            moe_tp_size=4,
            moe_ep_size=1,
            decoder_replay=profile == "decoder_bounded",
            systems_path=str(systems),
            database_mode="SILICON",
            shared_layer=False,
            strict_provenance=True,
        )
        context, generation = handle.run_static_per_op(batch_size=2, isl=385, prefix=256, osl=2)
        assert sum(item[1] for item in context) > 0
        assert sum(item[1] for item in generation) > 0
        assert any(item[3] == "mixed" for item in context)


def test_historical_eight_head_labels_do_not_satisfy_corrected_silicon_queries():
    from aiconfigurator_core.sdk.engine import EngineHandle

    root = Path(__file__).resolve().parents[5] / "data/experimental/deepseek-v41/gb300-silicon"
    handle = EngineHandle.compile(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "gb300",
        "sglang",
        backend_version="0.0.0.dev0",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        systems_path=str(root / "full/systems"),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    with pytest.raises(RuntimeError, match="[Dd]sv41|[Dd]eep[Ss]eek|[Mm]issing"):
        handle.run_static_per_op(batch_size=2, isl=385, prefix=256, osl=2)


@pytest.mark.parametrize("cohort", ["initial", "study", "prefix-refinement"])
@pytest.mark.parametrize("profile", ["full", "decoder_bounded"])
def test_every_derived_observation_roundtrips_through_strict_native_silicon(cohort, profile):
    import aiconfigurator_core._aiconfigurator_core as native
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    root = Path(__file__).resolve().parents[5] / "data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2"
    systems = root / cohort / profile / "systems"
    database = PerfDatabase(
        "gb300",
        "sglang",
        "0.0.0.dev0",
        str(systems),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    rows = pq.read_table(systems / "data/gb300/dsv41/sglang/0.0.0.dev0/dsv41_module_perf.parquet").to_pylist()
    variants = {"attention": "Dsv41Attention", "mhc": "Dsv41Mhc", "engram": "Dsv41Engram", "linear": "Dsv41Linear"}
    for point in rows:
        geometry = json.loads(point["geometry"])
        op = native.op_from_spec_json(json.dumps({variants[point["component"]]: geometry | {"name": "identity_check"}}))
        result = _evaluate_single_op(
            database,
            op,
            is_context=geometry.get("is_context", True),
            batch_size=point["batch_size"],
            s=point["x"],
            prefix=point["prefix"],
            x=point["x"],
        )
        assert result.source == "silicon"
        assert float(result) == pytest.approx(point["latency"], rel=0, abs=1e-12)


def test_versioned_indexer_derivation_preserves_every_observed_value_and_original():
    import yaml

    original_root = Path(__file__).resolve().parents[5] / "data/experimental/deepseek-v41/gb300-silicon"
    root = original_root / "indexer-identity-v2"
    helpers = runpy.run_path(str(root / "derive.py"))
    file_sha, digest = helpers["file_sha"], helpers["digest"]
    receipt = json.loads((root / "derivation.json").read_text())
    assert receipt["derivation_source_sha256"] == file_sha(root / "derive.py")
    assert receipt["source_proof_sha256"] == file_sha(root / "source-proof.json")
    assert len(receipt["datasets"]) == 6
    assert sum(item["rows"] for item in receipt["datasets"]) == 4080
    assert sum(item["changed_attention_rows"] for item in receipt["datasets"]) == 3670
    for item in receipt["datasets"]:
        old_path, new_path = original_root / item["original_path"], root / item["derived_path"]
        assert file_sha(old_path) == item["original_sha256"]
        assert file_sha(new_path) == item["derived_sha256"]
        old, new = pq.read_table(old_path), pq.read_table(new_path)
        assert old.schema == new.schema
        for column in old.column_names:
            if column != "geometry":
                assert old[column].equals(new[column])
        assert len(old) == len(new) == len(item["rows_mapping"]) == item["rows"]
        for before, after, mapping in zip(old.to_pylist(), new.to_pylist(), item["rows_mapping"], strict=True):
            assert digest(before) == mapping["original_sha256"]
            assert digest(after) == mapping["derived_sha256"]
            assert struct.pack("!d", before["latency"]).hex() == mapping["latency_f64_bits"]
            assert struct.pack("!d", after["latency"]).hex() == mapping["latency_f64_bits"]
            if before["component"] == "attention":
                before_geometry, after_geometry = json.loads(before["geometry"]), json.loads(after["geometry"])
                assert before_geometry["index_n_heads"] == 8
                assert after_geometry == before_geometry | {"index_n_heads": 32}
            else:
                assert before == after
        old_meta = yaml.safe_load(old_path.with_name("collection_meta.yaml").read_text())
        new_meta = yaml.safe_load(new_path.with_name("collection_meta.yaml").read_text())
        old_meta["tables"]["dsv41_module_perf"]["data_sha256"] = item["derived_sha256"]
        assert old_meta == new_meta
        for artifact in item["unchanged_artifacts"]:
            original = original_root / artifact["path"]
            relative = original.relative_to(old_path.parents[5])
            copied = new_path.parents[5] / relative
            assert file_sha(original) == file_sha(copied) == artifact["sha256"]
    for item in receipt["manifests"]:
        assert file_sha(original_root / item["original_path"]) == item["original_sha256"]
        assert file_sha(root / item["derived_path"]) == item["derived_sha256"]
        manifest = json.loads((root / item["derived_path"]).read_text())
        assert manifest == build_manifest(4, manifest["execution_profile"] == "decoder_bounded")


@pytest.mark.parametrize("heads", [32, 8.0, True])
def test_indexer_derivation_rejects_unexpected_source_identity(heads):
    root = Path(__file__).resolve().parents[5] / "data/experimental/deepseek-v41/gb300-silicon/indexer-identity-v2"
    correct = runpy.run_path(str(root / "derive.py"))["corrected_row"]
    point = row()
    point["geometry"] = operation_geometry(json.loads(point["geometry"]) | {"index_n_heads": heads})
    with pytest.raises(ValueError, match="unexpected original index heads"):
        correct(point)


def test_output_profile_binding_rejects_mixed_campaigns(tmp_path):
    from collector.sglang.collect_dsv41_module import bind_output_profile

    destination = str(tmp_path / "dsv41_module_perf.txt")
    bind_output_profile(destination, "full")
    bind_output_profile(destination, "full")
    with pytest.raises(ValueError, match="separate output tables"):
        bind_output_profile(destination, "decoder_bounded")


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
        "collector/sglang/dsv41_workloads.py",
        "src/aiconfigurator_core/sdk/models/deepseek_v41.py",
        "src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V4.1-Flash_config.json",
    ):
        target = tmp_path / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n ")
        assert provenance.collector_hash(module, tmp_path, closures) != baseline, relative
        target.write_bytes(original)
    assert provenance.collector_hash(module, tmp_path, closures) == baseline
