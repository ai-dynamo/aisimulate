# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic contract tests; mock Hub pins never leave pytest outputs."""

import copy
import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collector.fpm_forward.glm53flash_publication import partition_table
from tests.unit.tools import test_glm53flash_hf_publication as fixtures
from tools.glm53flash_hf import glm53flash as policy
from tools.glm53flash_hf import import_glm53flash as integration
from tools.glm53flash_hf import profile

pytestmark = pytest.mark.unit
MOCK_REVISION = "e" * 40  # Only the explicitly mocked Hub confirms this fixture revision.


@pytest.fixture
def profile_context(tmp_path, request):
    previous, external = fixtures.staged.__wrapped__(tmp_path, request)
    stage_root = tmp_path / "PROFILE_TEST_ONLY_STAGE"
    stage_root.mkdir()
    report = policy.read(previous / "validation/acceptance.json")
    sources, parts, entries = {}, [], []
    for backend in ("vllm", "sglang"):
        source_root = tmp_path / backend
        table = source_root / "fpm_forward_perf.parquet"
        metadata = table.with_suffix(".metadata.json")
        rows = pq.read_table(table).to_pylist()
        for row in rows:
            row.update(
                gemm_quant_mode="fp8_block",
                moe_quant_mode=row["weight_quantization"],
                fmha_quant_mode="fp8",
                fmha_resolution="checkpoint_native",
                comm_quant_mode="half",
                moe_backend="auto",
                attention_backend="auto",
                enable_wideep=False,
                enable_eplb=False,
                model_config_sha256="a" * 64,
                execution_profile="full",
                engram_residency="none",
                input_modality="text",
            )
        pq.write_table(pa.Table.from_pylist(rows), table)
        meta = policy.read(metadata)
        meta["parquet_sha256"] = policy.sha(table)
        integration.write(metadata, meta)
        # Deliberately non-default data_dir proves the tool follows accepted YAML bytes.
        system = source_root / "gb300.yaml"
        system.write_text("data_dir: accepted-data/gb300\ngpu:\n  sm_version: 103\n")
        receipts = [
            {"path": path.relative_to(tmp_path).as_posix(), "sha256": policy.sha(path)}
            for path in (table, metadata, system)
        ]
        for path, receipt in zip((table, metadata, system), receipts, strict=True):
            if receipt["sha256"] in sources:
                sources[receipt["sha256"]]["original_consumer_paths"].append(receipt["path"])
                continue
            archive = "sources/" + receipt["sha256"] + path.suffix
            (stage_root / archive).parent.mkdir(exist_ok=True)
            (stage_root / archive).write_bytes(path.read_bytes())
            sources[receipt["sha256"]] = {
                "path": archive,
                "sha256": receipt["sha256"],
                "original_consumer_paths": [receipt["path"]],
            }
        parts.extend(partition_table(table, metadata, stage_root))
        for cell in report["cells"]:
            if cell["backend"] != backend:
                continue
            model = "nvidia/GLM-5.3-Flash-NVFP4" if cell["weight_quantization"] == "nvfp4" else "zai-org/GLM-5.3-Flash"
            config = {
                "model": model,
                "system": "gb300",
                "backend": backend,
                "backend_version": meta["backend_version"],
                "tp": cell["tp"],
                "kvcache_quant_mode": "fp8",
                "systems_paths": [backend],
            }
            cell["prediction_provenance"] = {"config": config, "data_receipts": receipts}
            entries.append({"consumer_config": config, "consumer_data": receipts})
    input_manifest = {
        "schema": "glm53flash_independent_holdout_v1",
        "mode": "fpm",
        "entries": entries,
        "test_only": True,
    }
    integration.write(stage_root / "validation/input-manifest.json", input_manifest)
    report["input_manifest_sha256"] = hashlib.sha256(
        json.dumps(input_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    integration.write(stage_root / "validation/acceptance.json", report)
    stage = {
        "schema": "glm53flash_fpm_publication_stage_v1",
        "status": "STAGED_NOT_PUBLISHED",
        "repo_id": profile.REPO_ID,
        "configurations": parts,
        "sources": list(sources.values()),
        "acceptance": {
            "path": "validation/acceptance.json",
            "sha256": policy.sha(stage_root / "validation/acceptance.json"),
        },
        "input_manifest": {
            "path": "validation/input-manifest.json",
            "sha256": policy.sha(stage_root / "validation/input-manifest.json"),
        },
        "input_manifest_sha256": policy.sha(stage_root / "validation/input-manifest.json"),
    }
    integration.write(stage_root / "stage.json", stage)
    policy.validate_stage(stage_root)
    dataset = tmp_path / "TEST_ONLY_DATASET"
    campaign = Path("campaigns") / policy.CAMPAIGN / policy.sha(stage_root / "stage.json")
    archive = dataset / campaign / "stage"
    shutil.copytree(stage_root, archive)
    manifests = {}
    for part in parts:
        version = re.sub(r"[^a-z0-9.-]+", "-", part["backend_version"].lower()).strip("-")
        leaf = (
            Path("data")
            / part["model_id"].replace("/", "--")
            / "gb300"
            / part["backend"]
            / version
            / f"pure-tp{part['tp']}"
        )
        target = leaf / "fpm/fpm_forward_perf.parquet"
        (dataset / target).parent.mkdir(parents=True)
        shutil.copyfile(stage_root / part["parquet"]["path"], dataset / target)
        metadata = target.with_suffix(".metadata.json")
        shutil.copyfile(stage_root / part["metadata"]["path"], dataset / metadata)
        manifests[(part["backend"], part["weight_quantization"], part["tp"])] = {
            "fpm": [
                {
                    "path": target.as_posix(),
                    "metadata_path": metadata.as_posix(),
                    "sha256": policy.sha(dataset / target),
                }
            ]
        }
    import_result = dataset / campaign / "import-result.json"
    integration.write(import_result, {"test_only": True})
    controls = {(campaign / "stage/stage.json").as_posix(): policy.sha(stage_root / "stage.json")}
    context = stage, report, input_manifest, manifests, controls, archive
    return stage_root, dataset, import_result, context, external


def fake_hub(dataset, *, actual_revision=MOCK_REVISION, corrupt_path=None, requests=None):
    def request(request, timeout):
        assert timeout == 120
        url = request.full_url
        if requests is not None:
            requests.append(url)
        if "/api/datasets/" in url:
            return io.BytesIO(json.dumps({"id": profile.REPO_ID, "sha": actual_revision}).encode())
        prefix = f"https://huggingface.co/datasets/{profile.REPO_ID}/resolve/{MOCK_REVISION}/"
        assert url.startswith(prefix)
        path = unquote(url[len(prefix) :])
        raw = (dataset / path).read_bytes()
        return io.BytesIO(raw + b"corruption" if path == corrupt_path else raw)

    return request


def test_production_entry_rejects_marked_fixture_before_hub_or_output(profile_context, tmp_path, monkeypatch):
    stage, dataset, imported, _, _ = profile_context
    monkeypatch.setattr(profile, "verify_hub_revision", lambda *_: pytest.fail("fixture reached Hub verification"))
    destination = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="synthetic/test evidence"):
        profile.generate(stage, dataset, imported, MOCK_REVISION, destination)
    assert not destination.exists()


def test_eight_profiles_preserve_exact_runtime_yaml_and_offline_sdk_contract(profile_context, tmp_path, monkeypatch):
    stage, dataset, imported, context, _ = profile_context
    # Only tests bypass synthetic admission. Production has no bypass argument.
    monkeypatch.setattr(profile, "_require_production_evidence", lambda _: None)
    monkeypatch.setattr(profile, "_load_import", lambda *_: context)
    requests = []
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", fake_hub(dataset, requests=requests))
    output = tmp_path / "TEST_ONLY_PROFILES_DO_NOT_PUBLISH"
    result = profile.generate(stage, dataset, imported, MOCK_REVISION, output)
    assert len(result["profiles"]) == 8
    assert policy.read(output / "generation.json")["status"] == "PIN_VERIFIED_OFFLINE_VALIDATION_PENDING"
    cache = tmp_path / "cache"
    for name, entry in result["profiles"].items():
        assert len(entry["files"]) == 3
        assert entry["identity"]["kv_cache_dtype"] == "fp8"
        if entry["identity"]["backend"] == "vllm":
            assert "0.30.0-test.candidate" in entry["files"][0]["path"]
            assert "0.30.0+test.CANDIDATE" in entry["files"][0]["target"]
        assert entry["files"][0]["target"].startswith("accepted-data/gb300/")
        root = profile.fpm_dataset.materialize_fpm_profile(output / "hf_dataset.json", name, cache_dir=cache)
        assert (root / "gb300.yaml").read_bytes() == (dataset / entry["files"][2]["path"]).read_bytes()
    count = len(requests)
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", lambda *_a, **_k: pytest.fail("offline attempted network"))
    for name in result["profiles"]:
        profile.fpm_dataset.materialize_fpm_profile(
            output / "hf_dataset.json", name, cache_dir=cache, local_files_only=True
        )
    assert len(requests) == count


@pytest.mark.parametrize(
    "mutation",
    ["missing_origin", "missing_yaml", "wrong_cache_precision", "phase_yaml_bytes", "unsafe_data_dir", "wrong_sm"],
)
def test_system_yaml_must_be_exact_accepted_unambiguous_bytes(profile_context, mutation):
    root, _, _, context, _ = profile_context
    stage, report, original, *_ = copy.deepcopy(context)
    cells = [
        c for c in report["cells"] if c["backend"] == "vllm" and c["weight_quantization"] == "fp8" and c["tp"] == 2
    ]
    if mutation == "missing_origin":
        for source in stage["sources"]:
            source["original_consumer_paths"] = []
    elif mutation == "missing_yaml":
        for cell in cells:
            cell["prediction_provenance"]["data_receipts"] = [
                r for r in cell["prediction_provenance"]["data_receipts"] if not r["path"].endswith(".yaml")
            ]
    elif mutation == "wrong_cache_precision":
        cells[0]["prediction_provenance"]["config"]["kvcache_quant_mode"] = "bf16"
    elif mutation == "phase_yaml_bytes":
        alternate = root / "sources/alternate.yaml"
        alternate.write_text("data_dir: different-data/gb300\ngpu:\n  sm_version: 103\n")
        origin = "vllm/alternate/gb300.yaml"
        receipt = {"path": origin, "sha256": policy.sha(alternate)}
        stage["sources"].append(
            {"path": "sources/alternate.yaml", "sha256": receipt["sha256"], "original_consumer_paths": [origin]}
        )
        prediction = cells[0]["prediction_provenance"]
        prediction["data_receipts"] = [
            copy.deepcopy(r) for r in prediction["data_receipts"] if not r["path"].endswith(".yaml")
        ]
        prediction["data_receipts"].append(receipt)
        original["entries"].append(
            {"consumer_config": prediction["config"], "consumer_data": prediction["data_receipts"]}
        )
    else:
        source = next(s for s in stage["sources"] if s["path"].endswith(".yaml"))
        path = root / source["path"]
        path.write_text(
            "data_dir: /unsafe\ngpu:\n  sm_version: 103\n"
            if mutation == "unsafe_data_dir"
            else "data_dir: data/gb300\ngpu:\n  sm_version: 100\n"
        )
        digest = policy.sha(path)
        old_digest = source["sha256"]
        source["sha256"] = digest
        for container in [report, original]:

            def replace(value):
                if isinstance(value, dict):
                    if value.get("sha256") == old_digest:
                        value["sha256"] = digest
                    for nested in value.values():
                        replace(nested)
                elif isinstance(value, list):
                    for nested in value:
                        replace(nested)

            replace(container)
    with pytest.raises(ValueError):
        profile._select_system_yaml(root, stage, cells, original)


@pytest.mark.parametrize("revision", ["main", "v1", "e" * 39, "E" * 40])
def test_arbitrary_revision_labels_never_reach_hub(revision, monkeypatch):
    monkeypatch.setattr(
        profile.fpm_dataset, "urlopen", lambda *_a, **_k: pytest.fail("invalid revision reached network")
    )
    with pytest.raises(ValueError, match="40-character"):
        profile.verify_hub_revision(revision, {})


def test_hub_must_confirm_commit_and_exact_file_bytes(tmp_path, monkeypatch):
    data = tmp_path / "data.bin"
    data.write_bytes(b"TEST_ONLY")
    files = {"data.bin": policy.sha(data)}
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", fake_hub(tmp_path, actual_revision="f" * 40))
    with pytest.raises(ValueError, match="confirm"):
        profile.verify_hub_revision(MOCK_REVISION, files)
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", fake_hub(tmp_path, corrupt_path="data.bin"))
    with pytest.raises(ValueError, match="differs"):
        profile.verify_hub_revision(MOCK_REVISION, files)


def test_failed_remote_verification_leaves_no_profile_output(profile_context, tmp_path, monkeypatch):
    stage, dataset, imported, context, _ = profile_context
    monkeypatch.setattr(profile, "_require_production_evidence", lambda _: None)
    monkeypatch.setattr(profile, "_load_import", lambda *_: context)
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", fake_hub(dataset, actual_revision="f" * 40))
    output = tmp_path / "UNCREATED_PROFILES"
    with pytest.raises(ValueError, match="confirm"):
        profile.generate(stage, dataset, imported, MOCK_REVISION, output)
    assert not output.exists()


@pytest.mark.skipif(not os.environ.get("GLM_TEST_BASE"), reason="requires immutable local dataset base")
@pytest.mark.timeout(300)
def test_complete_import_to_profile_pipeline_with_mock_hub(profile_context, tmp_path, monkeypatch):
    from tests.unit.tools.test_glm53flash_raw_campaign import build_bound_fixture

    stage, _, _, _, _ = profile_context
    _, _, root, _ = build_bound_fixture(stage, tmp_path, monkeypatch)
    external = root / "external-raw-evidence.json"
    dataset = tmp_path / "TEST_ONLY_FULL_BASELINE_COPY"
    imported = integration.prepare(Path(os.environ["GLM_TEST_BASE"]), stage, dataset, external, "1" * 40, "2026-09-23")
    result_path = dataset / "campaigns" / policy.CAMPAIGN / imported["stage_sha256"] / "import-result.json"
    monkeypatch.setattr(profile, "_require_production_evidence", lambda _: None)  # Explicitly synthetic test pipeline.
    monkeypatch.setattr(profile.fpm_dataset, "urlopen", fake_hub(dataset))
    result = profile.generate(stage, dataset, result_path, MOCK_REVISION, tmp_path / "TEST_ONLY_FULL_PROFILES")
    assert len(result["profiles"]) == 8
