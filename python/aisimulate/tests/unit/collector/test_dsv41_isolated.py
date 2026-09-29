# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from collector.sglang.dsv41_contract import build_manifest, canonical_json
from collector.sglang.dsv41_isolated_runner import (
    FRAMEWORK_COMMIT,
    REQUIRED_SOURCES,
    WEIGHT_INITIALIZER,
    aggregate_isolated_records,
    sha,
    validate_plan,
)

pytestmark = pytest.mark.unit


def declared_plan():
    return dict(
        schema="dsv41.isolated-collection.v1",
        purpose="calibration",
        tp_size=2,
        execution_profile="full",
        token_counts=[1, 32],
        components=["engram"],
        warmup=2,
        iterations=5,
        moe_runner_backend="flashinfer_mxfp4",
        expected_gpu="H100",
        expected_sm=90,
        framework_commit=FRAMEWORK_COMMIT,
        collector_revision="e" * 40,
        source_pins=dict.fromkeys(REQUIRED_SOURCES, "a" * 64),
        metadata_pins=dict.fromkeys(("config.json", "tokenizer.json", "tokenizer_config.json"), "b" * 64),
        runtime_digest="sha256:" + "c" * 64,
        image_sha256="d" * 64,
        weight_initializer=WEIGHT_INITIALIZER.copy(),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"tp_size": 4},
        {"token_counts": [32, 1]},
        {"token_counts": [True]},
        {"components": ["attention"]},
        {"purpose": "promoted_canary"},
        {"source_pins": {}},
        {"metadata_pins": {}},
        {"expected_sm": 100},
        {"image_sha256": "latest"},
        {"collector_revision": "dirty"},
        {"weight_initializer": {"name": "zero_weights"}},
    ],
)
def test_isolated_plan_rejects_unqualified_identity(change):
    with pytest.raises(ValueError):
        validate_plan(declared_plan() | change, build_manifest(2, False))


def test_isolated_admission_requires_both_engram_tables_and_every_sample(tmp_path):
    plan, manifest = declared_plan(), build_manifest(2, False)
    plan_path, manifest_path = tmp_path / "plan.json", tmp_path / "manifest.json"
    plan_path.write_text(json.dumps(plan))
    manifest_path.write_text(json.dumps(manifest))
    for rank in range(2):
        (tmp_path / f"isolated-rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    state="complete_pending_admission",
                    tp_rank=rank,
                    plan_sha256=sha(plan_path),
                    manifest_sha256=sha(manifest_path),
                    runtime_digest=plan["runtime_digest"],
                    purpose="calibration",
                    source_hashes=plan["source_pins"],
                    full_model=False,
                    checkpoint_weights_loaded=False,
                )
            )
        )
        rows = []
        for entry in manifest["phases"]["context"]:
            if entry["component"] != "engram":
                continue
            for tokens in plan["token_counts"]:
                for sample in range(2, 7):
                    rows.append(
                        dict(
                            component="engram",
                            geometry=entry["geometry"],
                            batch_size=1,
                            prefix=0,
                            x=tokens,
                            latency=0.125,
                            kernel_source="test.fixture",
                            measurement_scope="local_compute",
                            source_sha256=hashlib.sha256(canonical_json(plan["source_pins"]).encode()).hexdigest(),
                            config_sha256=manifest["config_sha256"],
                            runtime_digest=plan["runtime_digest"],
                            execution_profile="full",
                            used_cuda_graph=False,
                            sample_count=1,
                            kv_seed_regime="n/a",
                            sample=sample,
                            invocation=tokens,
                            tp_rank=rank,
                            case_plan_sha256=sha(plan_path),
                            collection_purpose="calibration",
                        )
                    )
        (tmp_path / f"rank-{rank}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    measured, baselines = aggregate_isolated_records(tmp_path, plan_path, manifest_path)
    assert len(measured) == 4 and not baselines
    # A different run with identical physical keys cannot borrow this receipt.
    path = tmp_path / "rank-0.jsonl"
    original = path.read_text()
    path.write_text(original.replace(sha(plan_path), "f" * 64))
    with pytest.raises(ValueError, match="raw measurement differs"):
        aggregate_isolated_records(tmp_path, plan_path, manifest_path)
    path.write_text(original)
    # A complete rank set for only one table must still fail the declared grid.
    for rank in range(2):
        path = tmp_path / f"rank-{rank}.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:10]))
    with pytest.raises(ValueError, match="coverage is incomplete"):
        aggregate_isolated_records(tmp_path, plan_path, manifest_path)


def test_canary_cannot_be_admitted_by_changing_its_description(tmp_path):
    plan_path, manifest_path = tmp_path / "plan.json", tmp_path / "manifest.json"
    plan_path.write_text(json.dumps(declared_plan() | {"purpose": "smoke"}))
    manifest_path.write_text(json.dumps(build_manifest(2, False)))
    with pytest.raises(ValueError, match="smoke measurements cannot"):
        aggregate_isolated_records(tmp_path, plan_path, manifest_path)


def test_isolated_hash_closure_tracks_reader_and_native_dependencies(tmp_path):
    from collector import provenance

    root = Path(__file__).resolve().parents[3]
    module = "collector.sglang.dsv41_isolated_runner"
    closures = provenance.load_closures(root / "collector/hash_closures.yaml")
    assert module in provenance.enumerate_provenance_modules()
    dependencies = {module.replace(".", "/") + ".py", *provenance.SHARED_CORE}
    dependencies.update(provenance._expand_closure_files(root, closures[module]))
    for relative in dependencies:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    original_hash = provenance.collector_hash(module, tmp_path, closures)
    for relative in (
        "collector/sglang/dsv41_native_runner.py",
        "collector/sglang/dsv41_humming.py",
        "collector/sglang/collect_dsv41_module.py",
        "src/aisimulate_core/sdk/models/deepseek_v41.py",
    ):
        target = tmp_path / relative
        original = target.read_bytes()
        target.write_bytes(original + b"\n ")
        assert provenance.collector_hash(module, tmp_path, closures) != original_hash, relative
        target.write_bytes(original)
