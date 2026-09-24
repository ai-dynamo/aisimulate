# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate eight consumer pins after verifying an actual immutable HF commit.

This tool reads the Hub; it never uploads. Source calibration, system YAML and
canonical paths are taken from accepted receipts rather than reconstructed from
repository defaults. No production option permits synthetic qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk import fpm_dataset

if __package__:
    from . import glm53flash as policy
    from . import import_glm53flash as integration
else:
    import glm53flash as policy
    import import_glm53flash as integration

REPO_ID = "nvidia/aisimulate-fpm-dataset"
IDENTITY_FIELDS = [
    *policy.IDENTITY,
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "fmha_resolution",
    "comm_quant_mode",
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
    "model_config_sha256",
    "execution_profile",
    "engram_residency",
    "input_modality",
]


def _require_production_evidence(value):
    """Reject explicit synthetic/test markers without providing an override."""
    if isinstance(value, dict):
        policy.require(
            not any(value.get(key) for key in ("test_only", "test_fixture", "synthetic")),
            "synthetic/test evidence cannot produce serving profiles",
        )
        for item in value.values():
            _require_production_evidence(item)
    elif isinstance(value, list):
        for item in value:
            _require_production_evidence(item)
    elif isinstance(value, str):
        policy.require("TEST_ONLY" not in value, "synthetic/test evidence cannot produce serving profiles")


def _key(cell):
    return tuple(cell[field] for field in ("backend", "weight_quantization", "tp", "phase"))


def _load_import(stage_root, dataset_root, import_result_path):
    stage = policy.validate_stage(stage_root)
    report = policy.read(policy.checked(stage_root, stage["acceptance"]))
    input_manifest = policy.read(policy.checked(stage_root, stage["input_manifest"]))
    for value in (stage, report, input_manifest):
        _require_production_evidence(value)
    policy.require(import_result_path.resolve().is_relative_to(dataset_root.resolve()), "import result escapes dataset")
    result = policy.read(import_result_path)
    policy.require(
        result.get("status") == "CANONICAL_LOCAL_NOT_PUBLISHED"
        and result.get("policy") == policy.POLICY
        and result.get("stage_sha256") == policy.sha(stage_root / "stage.json"),
        "canonical import does not identify this accepted stage",
    )
    paths = result.get("added_manifests", [])
    policy.require(len(paths) == len(set(paths)) == 8, "canonical import must contain exactly eight configurations")
    index = policy.read(dataset_root / "catalog/index.json")
    bundle = policy.read(dataset_root / "catalog/canonical-bundle.json")
    controls = {
        "catalog/index.json": policy.sha(dataset_root / "catalog/index.json"),
        "catalog/canonical-bundle.json": policy.sha(dataset_root / "catalog/canonical-bundle.json"),
        import_result_path.relative_to(dataset_root).as_posix(): policy.sha(import_result_path),
    }
    for receipt in bundle["catalog_files"]:
        policy.checked(dataset_root, receipt)
        controls[receipt["path"]] = receipt["sha256"]
    registered = {item["path"]: item for item in bundle["configuration_manifests"]}
    manifests = {}
    for path in paths:
        policy.require(
            path in index["configuration_manifests"] and path in registered,
            "configuration is missing from the canonical catalog",
        )
        manifest = policy.read(policy.checked(dataset_root, registered[path]))
        policy.validate_snapshot(dataset_root, manifest)
        receipt = policy.read(dataset_root / manifest["provenance"]["import_receipt"])
        policy.require(receipt["stage"]["sha256"] == result["stage_sha256"], "mixed canonical stages")
        archive_stage = policy.checked(dataset_root, receipt["stage"]).parent
        policy.require(archive_stage == import_result_path.parent / "stage", "import result and archive disagree")
        quant = "nvfp4" if manifest["weight_quantization"] == "nvfp4" else "fp8"
        key = (manifest["framework"], quant, manifest["tp"])
        policy.require(
            key not in manifests and manifest["snapshot_status"] == "current", "duplicate or historical cell"
        )
        manifests[key] = manifest
        controls[path] = registered[path]["sha256"]
        for field in ("stage", "external_raw_evidence"):
            policy.checked(dataset_root, receipt[field])
            controls[receipt[field]["path"]] = receipt[field]["sha256"]
        provenance_path = manifest["provenance"]["import_receipt"]
        controls[provenance_path] = policy.sha(dataset_root / provenance_path)
    policy.require(set(manifests) == {key[:3] for key in policy.KEYS}, "canonical matrix differs from required matrix")
    archive = import_result_path.parent / "stage"
    partition_receipts = [part[field] for part in stage["configurations"] for field in ("parquet", "metadata")]
    for receipt in [stage["acceptance"], stage["input_manifest"], *stage["sources"], *partition_receipts]:
        path = policy.checked(archive, receipt)
        controls[path.relative_to(dataset_root).as_posix()] = receipt["sha256"]
    policy.require(
        policy.sha(dataset_root / "scripts/glm53flash.py") == policy.sha(Path(policy.__file__)),
        "canonical dataset does not contain the reviewed GLM policy",
    )
    for path in ("scripts/glm53flash.py", "scripts/manage_dataset.py"):
        controls[path] = policy.sha(dataset_root / path)
    integration.load_manager(dataset_root).validate_dataset(dataset_root, write_report=False)
    return stage, report, input_manifest, manifests, controls, archive


def _select_system_yaml(stage_root, stage, cells, input_manifest):
    """Select exact accepted YAML receipts; refuse ambiguous overlays."""
    selected = {}
    for cell in cells:
        prediction = cell["prediction_provenance"]
        config = prediction["config"]
        expected_model = (
            "nvidia/GLM-5.3-Flash-NVFP4" if cell["weight_quantization"] == "nvfp4" else "zai-org/GLM-5.3-Flash"
        )
        policy.require(
            config.get("model") == expected_model
            and config.get("backend") == cell["backend"]
            and config.get("tp") == cell["tp"]
            and config.get("system") == "gb300"
            and config.get("kvcache_quant_mode") == "fp8",
            "accepted consumer identity mismatch",
        )
        receipts = prediction["data_receipts"]
        matches = [
            entry
            for entry in input_manifest["entries"]
            if entry.get("consumer_data") == receipts
            and entry.get("consumer_config", {}).get("backend_version") == config["backend_version"]
        ]
        policy.require(matches, "accepted data receipts are absent from the original acceptance input")
        found = {}
        for receipt in receipts:
            if PurePosixPath(receipt["path"]).name != "gb300.yaml":
                continue
            sources = [
                source
                for source in stage["sources"]
                if source["sha256"] == receipt["sha256"]
                and receipt["path"] in source.get("original_consumer_paths", [])
            ]
            policy.require(len(sources) == 1, "system YAML has no exact accepted source origin")
            policy.checked(stage_root, sources[0])
            found[receipt["sha256"]] = sources[0]
        policy.require(len(found) == 1, "accepted system YAML is absent or ambiguous")
        selected.update(found)
    policy.require(len(selected) == 1, "prefill and decode used different system YAML bytes")
    source = next(iter(selected.values()))
    data = yaml.safe_load(policy.checked(stage_root, source).read_text())
    policy.require(
        isinstance(data, dict) and isinstance(data.get("gpu"), dict) and data["gpu"].get("sm_version") == 103,
        "accepted system YAML does not describe GB300 sm103",
    )
    data_dir = fpm_dataset._relative_path(data.get("data_dir"))
    return source, data_dir


def _build_profiles(stage_root, dataset_root, context):
    stage, report, input_manifest, manifests, controls, archive = context
    cells = {_key(cell): cell for cell in report["cells"]}
    profiles = {}
    for part in stage["configurations"]:
        key = (part["backend"], part["weight_quantization"], part["tp"])
        manifest = manifests[key]
        entry = manifest["fpm"][0]
        table = pq.read_table(policy.checked(dataset_root, entry))
        rows = table.to_pylist()
        for row in rows:
            _require_production_evidence(row)
        identity = {field: rows[0][field] for field in IDENTITY_FIELDS}
        policy.require(
            all(all(row[field] == value for field, value in identity.items()) for row in rows),
            "profile contains mixed execution identities",
        )
        policy.require(
            identity["execution_profile"] == "full"
            and identity["input_modality"] == "text"
            and identity["engram_residency"] == "none"
            and identity["enable_eplb"] is False
            and identity["enable_wideep"] is False,
            "profile is outside the accepted full-text pure-TP scope",
        )
        selected_cells = [cells[(*key, phase)] for phase in ("prefill", "decode")]
        policy.require(
            all(
                cell["prediction_provenance"]["config"]["backend_version"] == part["backend_version"]
                for cell in selected_cells
            ),
            "profile runtime differs from accepted consumer",
        )
        system_source, data_dir = _select_system_yaml(stage_root, stage, selected_cells, input_manifest)
        system_path = policy.checked(archive, system_source).relative_to(dataset_root).as_posix()
        version = part["backend_version"]
        policy.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version), "unsafe exact runtime version")
        target = PurePosixPath(data_dir) / part["backend"] / version
        files = [
            {"path": entry["path"], "target": str(target / "fpm_forward_perf.parquet"), "sha256": entry["sha256"]},
            {
                "path": entry["metadata_path"],
                "target": str(target / "fpm_forward_perf.metadata.json"),
                "sha256": policy.sha(dataset_root / entry["metadata_path"]),
            },
            {"path": system_path, "target": "gb300.yaml", "sha256": system_source["sha256"]},
        ]
        for item in files:
            fpm_dataset._relative_path(item["path"])
            fpm_dataset._relative_path(item["target"])
            policy.checked(dataset_root, item)
            controls[item["path"]] = item["sha256"]
        name = f"gb300-{part['backend']}-{part['weight_quantization']}-tp{part['tp']}-full"
        profiles[name] = {
            "admission": "serving",
            "identity": identity,
            "model_revision": part["model_revision"],
            "files": files,
            "acceptance_sha256": stage["acceptance"]["sha256"],
            "source_partition": part["source_partition"],
        }
    policy.require(len(profiles) == 8, "exactly eight profiles are required")
    return profiles, controls


def verify_hub_revision(revision, files):
    """Read actual Hub commit identity and hash its immutable response bytes."""
    policy.require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Hub revision must be an immutable 40-character SHA",
    )
    api = f"https://huggingface.co/api/datasets/{REPO_ID}/revision/{revision}"
    with fpm_dataset.urlopen(fpm_dataset._download_request(api), timeout=120) as response:
        info = json.load(response)
    policy.require(
        info.get("sha") == revision and info.get("id") == REPO_ID, "Hub did not confirm the requested commit"
    )
    verified = []
    for path, expected in sorted(files.items()):
        fpm_dataset._relative_path(path)
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{quote(path, safe='/')}"
        digest, size = hashlib.sha256(), 0
        with fpm_dataset.urlopen(fpm_dataset._download_request(url), timeout=120) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        policy.require(digest.hexdigest() == expected, f"published file differs from accepted local bytes: {path}")
        verified.append({"path": path, "sha256": expected, "bytes": size})
    return {"repo_id": REPO_ID, "revision": revision, "verified_files": verified}


def generate(stage_root, dataset_root, import_result, revision, destination):
    stage_root, dataset_root, import_result, destination = (
        Path(path).resolve() for path in (stage_root, dataset_root, import_result, destination)
    )
    policy.require(not destination.exists(), "profile output directory must not exist")
    policy.require(
        not destination.is_relative_to(dataset_root) and not destination.is_relative_to(stage_root),
        "profile output must not modify immutable dataset inputs",
    )
    context = _load_import(stage_root, dataset_root, import_result)
    profiles, files = _build_profiles(stage_root, dataset_root, context)
    verified = verify_hub_revision(revision, files)
    destination.mkdir(parents=True)
    manifest = {"format_version": 1, "repo_id": REPO_ID, "revision": revision, "profiles": profiles}
    integration.write(destination / "hf_dataset.json", manifest)
    integration.write(
        destination / "generation.json",
        {
            "schema": "glm53flash_consumer_profiles_v1",
            "status": "PIN_VERIFIED_OFFLINE_VALIDATION_PENDING",
            "stage_sha256": policy.sha(stage_root / "stage.json"),
            "canonical_import_result_sha256": policy.sha(import_result),
            "hf_dataset_sha256": policy.sha(destination / "hf_dataset.json"),
            **verified,
            "remaining": [
                "check in consumer pin",
                "installed-wheel prediction reproduction",
                "offline cache acceptance",
            ],
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "dataset", "import-result", "destination"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = generate(args.stage, args.dataset, args.import_result, args.revision, args.destination)
    print(json.dumps({"revision": result["revision"], "profiles": sorted(result["profiles"])}, indent=2))


if __name__ == "__main__":
    main()
