# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real planner/sharder schemas, TEST_ONLY tiny raw files; no native acceptance."""

import copy
import sys
from pathlib import Path

import pytest

from collector.fpm_forward.config import FPMCollectionOptions
from collector.fpm_forward.glm53flash_validation import _plan_run
from collector.fpm_forward.planner import build_collection_plan
from collector.fpm_forward.shards import canonical, digest, make_shards, shard_manifest
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration
from tools.glm53flash_hf import raw_archive, raw_campaign

pytestmark = pytest.mark.unit


def real_sharded_controls(stage_root, work):
    """Use actual production planners/serializers; raw latencies are not fabricated."""
    work.mkdir()
    stage = policy.read(stage_root / "stage.json")
    report = policy.read(stage_root / stage["acceptance"]["path"])
    original = policy.read(stage_root / stage["input_manifest"]["path"])
    cells = {tuple(c[k] for k in raw_campaign.FIELDS[:-1]): c for c in report["cells"]}
    entries = {ident: {} for ident in cells}
    for source in stage["sources"]:
        source.setdefault("original_consumer_paths", ["original/" + Path(source["path"]).name])
    roots = {}
    for backend, quant, tp in sorted({key[:3] for key in raw_campaign.KEYS}):
        model = "zai-org/GLM-5.3-Flash" if quant == "fp8" else "nvidia/GLM-5.3-Flash-NVFP4"
        for role in ("calibration", "holdout"):
            payload = {
                "schema_version": 3,
                "prefill": [
                    {"batch_size": 1, "total_prefill_tokens": 32, "total_kv_read_tokens": n} for n in (1024, 65536)
                ],
                "decode": [{"batch_size": 1, "total_kv_read_tokens": 131071}],
            }
            corpus = work / (role + ".txt")
            corpus.write_text("TEST_ONLY " + role + " corpus")
            options = FPMCollectionOptions(
                input_text_path=str(corpus),
                input_text_sha256=policy.sha(corpus),
                max_gpus=tp,
                gpu_counts=(tp,),
                parallel_presets=("pure_tp",),
                parallel_axes=(),
                moe_backend="auto",
                attention_backend="auto",
                enable_wideep="false",
                enable_eplb="false",
                weight_quantizations=(),
                kv_cache_dtypes=("fp8",),
                vllm_max_model_len=131072,
                benchmark_points_json=canonical(payload),
                benchmark_points_sha256=digest(payload),
                shard_token_budget=100000,
                dataset_role=role,
            )
            parent = build_collection_plan(
                backend=backend, model_path=model, system="gb300", selected_ops=set(), options=options
            )
            shards = make_shards(parent)
            inventory = shard_manifest(parent, shards)
            closed = work / f"{backend}-{quant}-{tp}-{role}"
            parent_root = closed / "artifacts" / parent.sha256[:16]
            parent_path = parent_root / "collection-plan.json"
            manifest_path = parent_root / "shard-manifest.json"
            integration.write(parent_path, parent.to_dict())
            integration.write(manifest_path, inventory)
            (closed / "checkpoints").mkdir()
            (closed / "checkpoints/TEST_ONLY.json").write_text('{"test_only":true}')
            for cell in parent.cells:
                ident = (backend, quant, tp, cell.workload_kind)
                roots[(*ident, role)] = closed
                specs, observations = [], []
                for shard in shards:
                    if shard.identity["parent_cell_id"] != cell.cell_id:
                        continue
                    cid = shard.identity["child_cell_id"]
                    child_path = parent_root / "plans" / (cid + ".json")
                    integration.write(child_path, shard.plan.to_dict())
                    child_dir = parent_root / "shards" / shard.plan.sha256[:16] / "cells" / cid
                    raw = child_dir / "raw"
                    raw.mkdir(parents=True)
                    (raw / "TEST_ONLY-native.log").write_text("TEST_ONLY raw bytes, not native inference")
                    failed = child_dir / "attempts/TEST_ONLY-failed"
                    failed.mkdir(parents=True)
                    (failed / "traceback.txt").write_text("TEST_ONLY failure retained")
                    specs.append(
                        {
                            "plan": {"path": child_path.relative_to(work).as_posix(), "sha256": policy.sha(child_path)},
                            "cell_id": cid,
                            "attempt_id": "TEST_ONLY-attempt",
                            "raw_root": raw.relative_to(work).as_posix(),
                        }
                    )
                    observations.append(
                        {
                            "child_cell_id": cid,
                            "source_plan_sha256": shard.plan.sha256,
                            "original_point_ids": {
                                str(p["native_benchmark_id"]): p["original_point_id"]
                                for p in shard.identity["point_map"]
                            },
                            "receipts": [{"path": p.name, "sha256": policy.sha(p)} for p in raw.iterdir()],
                            "runtime_run_id": "TEST_ONLY-run",
                            "runtime_grid_digest": "a" * 64,
                            "backend_version": cells[ident][role + "_evidence"]["backend_version"],
                        }
                    )
                spec = {
                    "plan": {"path": parent_path.relative_to(work).as_posix(), "sha256": policy.sha(parent_path)},
                    "cell_id": cell.cell_id,
                    "shard_manifest": {
                        "path": manifest_path.relative_to(work).as_posix(),
                        "sha256": policy.sha(manifest_path),
                    },
                    "shards": specs,
                }
                # The strict native validator's production parser verifies all
                # parent/child/point-map identities without pretending raw ran.
                parsed = _plan_run(spec, work, role)
                assert len(parsed["children"]) == len(observations)
                entries[ident][role] = spec
                cells[ident][role + "_evidence"].update(shards=observations)
    for ident, entry in entries.items():
        part = next(
            p for p in stage["configurations"] if (p["backend"], p["weight_quantization"], p["tp"]) == ident[:3]
        )
        data = [
            {"path": s["original_consumer_paths"][0], "sha256": s["sha256"]}
            for s in stage["sources"]
            if s["sha256"] in (part["source_partition"]["parquet_sha256"], part["source_partition"]["metadata_sha256"])
        ]
        entry.update(consumer_config={}, consumer_data=data)
        cells[ident]["prediction_provenance"] = {"data_receipts": data}
    original.update(entries=list(entries.values()), test_only=True)
    integration.write(stage_root / stage["input_manifest"]["path"], original)
    report["input_manifest_sha256"] = digest(original)
    integration.write(stage_root / stage["acceptance"]["path"], report)
    stage["input_manifest"]["sha256"] = policy.sha(stage_root / stage["input_manifest"]["path"])
    stage["input_manifest_sha256"] = stage["input_manifest"]["sha256"]
    stage["acceptance"]["sha256"] = policy.sha(stage_root / stage["acceptance"]["path"])
    integration.write(stage_root / "stage.json", stage)
    return roots


@pytest.fixture
def shared(tmp_path, request, monkeypatch):
    from tests.unit.tools.test_glm53flash_hf_publication import staged

    stage, _ = staged.__wrapped__(tmp_path, request)
    work = tmp_path / "TEST_ONLY_NATIVE"
    roots = real_sharded_controls(stage, work)
    monkeypatch.setattr(raw_campaign, "production", lambda _: None)  # No production bypass exists.
    monkeypatch.setitem(sys.modules, "raw_campaign", raw_campaign)
    plan = raw_campaign.make_plan(stage, work)
    for job in plan["jobs"]:
        source = roots[raw_campaign.key(job)]
        job.update(source_root=str(source), uri="ssh://ocijhb/TEST_ONLY/" + source.name + "/campaign.tar.gz")
    bundles, archived = {}, {}
    target = tmp_path / "TEST_ONLY_ARCHIVES"
    target.mkdir()
    for job in plan["jobs"]:
        if job["uri"] not in archived:
            path = target / Path(job["source_root"]).name
            raw_campaign.archive_one(stage, plan, raw_campaign.name(job), path)
            archived[job["uri"]] = path
        bundles[raw_campaign.name(job)] = str(archived[job["uri"]])
    return stage, plan, bundles


def test_real_sharder_parent_layout_binds_32_roles_to_16_physical_archives(shared, tmp_path, monkeypatch):
    stage, plan, bundles = shared
    original = raw_archive.verify_bundle
    calls = []

    def verify(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(raw_archive, "verify_bundle", verify)
    output = tmp_path / "TEST_ONLY_BOUND"
    records = raw_campaign.bind(stage, plan, bundles, output)
    assert len(records) == 32 and len(calls) == len(set(calls)) == 16
    assert len({r["sha256"] for r in records}) == 16
    assert sum(len(r["native_roots"]) for r in records) == 48
    assert not list(output.rglob("*.tar.gz"))
    for record in records:
        receipt = policy.read(output / record["archive_receipt"]["path"])
        assert len(receipt["labels"]) == 2
        entries = list(raw_archive.inventory_records(output / record["source_inventory"]["path"]))
        assert any("/attempts/TEST_ONLY-failed/" in e["path"] for e in entries)
        assert any(e["path"] == "checkpoints/TEST_ONLY.json" for e in entries)
    policy.validate_external_receipts(records, stage_root=stage, evidence_root=output)


@pytest.mark.parametrize(
    "mutation", ["different_source", "missing_label", "different_content", "different_label", "different_receipt"]
)
def test_shared_archive_requires_exact_physical_and_logical_identity(shared, tmp_path, mutation):
    stage, plan, bundles = shared
    if mutation == "different_source":
        first, second = plan["jobs"][:2]
        assert first["source_root"] != second["source_root"]
        second["uri"] = first["uri"]
        with pytest.raises(ValueError, match="different source roots"):
            raw_campaign.bind(stage, plan, bundles, tmp_path / "uncreated")
        return
    output = tmp_path / "TEST_ONLY_BOUND"
    records = raw_campaign.bind(stage, plan, bundles, output)
    changed = copy.deepcopy(records)
    if mutation == "missing_label":
        changed.pop()
    elif mutation == "different_content":
        changed[0]["sha256"] = "0" * 64
    elif mutation == "different_label":
        changed[0]["role"] = "holdout"
    else:
        receipt = changed[0]["archive_receipt"]
        path = output / receipt["path"]
        # Even otherwise valid metadata cannot differ between shared references.
        attestation = policy.read(path)
        attestation["comment"] = "TEST_ONLY changed sidecar"
        integration.write(path, attestation)
        receipt.update(sha256=policy.sha(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError):
        raw_campaign.validate(changed, stage, output)


@pytest.mark.parametrize("mutation", ["tar", "receipt", "inventory", "replacement", "symlink"])
def test_verification_cache_rejects_changes_before_reuse(tmp_path, mutation):
    source = tmp_path / "TEST_ONLY_SOURCE"
    source.mkdir()
    (source / "raw.bin").write_bytes(b"TEST_ONLY original")
    output = tmp_path / "TEST_ONLY_ARCHIVE"
    label = {"backend": "vllm", "weight_quantization": "fp8", "tp": 2, "phase": "prefill", "role": "calibration"}
    raw_archive.create_archive(source, output, "ssh://ocijhb/TEST_ONLY.tar.gz", [label])
    verified = {}
    raw_campaign._verify_once(output, verified)
    filename = {"tar": raw_archive.ARCHIVE, "receipt": "receipt.json", "inventory": raw_archive.INVENTORY}.get(
        mutation, "receipt.json"
    )
    path = output / filename
    if mutation == "replacement":
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif mutation == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError, match="changed|symlink"):
        raw_campaign._verify_once(output, verified)


def test_final_source_identity_recheck_preserves_failure(shared, tmp_path, monkeypatch):
    stage, plan, bundles = shared
    validate = raw_campaign.validate
    source = Path(next(iter(bundles.values()))) / "receipt.json"

    def mutate(*args):
        result = validate(*args)
        with source.open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(raw_campaign, "validate", mutate)
    output = tmp_path / "TEST_ONLY_FAILED_BIND"
    with pytest.raises(ValueError, match="changed before final binding"):
        raw_campaign.bind(stage, plan, bundles, output)
    assert not (output / "external-raw-evidence.json").exists()
    assert policy.read(output / "failure.json")["partial_output_preserved"] is True


def test_bundle_change_during_full_verification_is_not_cached(tmp_path, monkeypatch):
    source = tmp_path / "TEST_ONLY_SOURCE"
    source.mkdir()
    (source / "raw.bin").write_bytes(b"TEST_ONLY original")
    output = tmp_path / "TEST_ONLY_ARCHIVE"
    label = {"backend": "vllm", "weight_quantization": "fp8", "tp": 2, "phase": "prefill", "role": "calibration"}
    raw_archive.create_archive(source, output, "ssh://ocijhb/TEST_ONLY.tar.gz", [label])
    original = raw_archive.verify_bundle

    def mutate_after_read(bundle):
        original(bundle)
        with (bundle / "receipt.json").open("ab") as stream:
            stream.write(b"\n")

    monkeypatch.setattr(raw_archive, "verify_bundle", mutate_after_read)
    verified = {}
    with pytest.raises(ValueError, match="changed during verification"):
        raw_campaign._verify_once(output, verified)
    assert not verified
