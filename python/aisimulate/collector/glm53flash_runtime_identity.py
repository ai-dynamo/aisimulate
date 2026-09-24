# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact deployment identities admitted by GLM native evidence consumers.

A local version suffix is a different runtime. Repair candidates remain
unqualified until their native Engine, source and binary evidence is reviewed;
neither a version prefix nor a matching upstream tag admits them implicitly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

BASELINE_VERSIONS = {"vllm": "0.30.0", "sglang": "0.5.20"}
VLLM_KPOOL_CANDIDATE = "0.30.0+glm53kpool.bf5f6b0e689d"
# Deliberately empty until native Engine qualification has passed and its
# immutable receipt has been reviewed. A build receipt or version suffix alone
# cannot promote a repair. Values will be reviewed qualification receipt hashes.
ADMITTED_VLLM_REPAIRS: dict[str, str] = {}
_BUILD_SHA256 = "3b72d70800e2ea244944580c1ce6a4faaa3dedf68af41b323aa690999abd9444"
_WHEEL_SHA256 = "a3b63cb3c95cf976f717077102e8172a33501c7d05092bc84cc58e3aaef47d36"
_ENGINE_IDENTITY_SHA256 = "d412233edffae84ae4b36a2e44d08bc4d3652a7e7f2bc62e8fc1193967a1cb22"
_V2_SOURCE_SHA256 = "48a6f6689d176be79d0ac38db0e68078050e63f73669c649b332d32dff076f1e"


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _candidate_build() -> dict:
    path = Path(__file__).parent / "fpm_forward/runtime/glm53flash_vllm_kpool_candidate/build-receipt.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _BUILD_SHA256:
        raise ValueError("GLM repair build receipt differs from the reviewed immutable build")
    build = json.loads(raw)
    if build["version"] != VLLM_KPOOL_CANDIDATE or build["wheel_sha256"] != _WHEEL_SHA256:
        raise ValueError("GLM repair wheel identity differs from its reviewed build")
    return build


_QUALIFICATION_SCOPE = "native_Engine_functional_correctness_for_frozen_geometry_suite_only"
_QUALIFICATION_PROFILES = (("stock", "reference"), ("candidate", "reference"), ("candidate", "split"))
_QUALIFICATION_SOURCES = (
    "validate.py",
    "probe.py",
    "worker_probe.py",
    "request-id-source.json",
    "cohort-source.json",
)


def _qualification_root() -> Path:
    return Path(__file__).parent / "fpm_forward/runtime/glm53flash_vllm_kpool_candidate/qualification"


def _qualification_sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("GLM qualification requires a lowercase SHA256")
    return value


def _qualification_name(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise ValueError("GLM qualification file must be a flat safe basename")
    return value


def _qualification_json(root: Path, reference: dict) -> dict:
    name = _qualification_name(reference["path"])
    path = root / name
    if path.suffix != ".json" or path.is_symlink() or not path.is_file():
        raise ValueError("GLM qualification requires an original regular JSON file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _qualification_sha(reference["sha256"]):
        raise ValueError(f"GLM qualification file SHA256 differs: {name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("GLM qualification receipt must be a JSON object")
    return value


def _qualification_uri(value: str) -> str:
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError("GLM qualification external URI is unsafe")
    uri = urlsplit(value)
    path = unquote(uri.path)
    if (
        uri.scheme not in {"https", "s3", "gs", "ssh"}
        or not uri.hostname
        or uri.username is not None
        or uri.password is not None
        or uri.query
        or uri.fragment
        or not path.startswith("/")
        or path in {"", "/"}
        or any(part in {".", ".."} for part in path.split("/"))
        or "//" in path
        or any(c.isspace() or ord(c) < 32 for c in path)
        or any(c in path for c in ("\\", "%", "?", "#"))
    ):
        raise ValueError("GLM qualification requires a stable credential-free external URI")
    return value


def _qualification_inventory(rows: list) -> dict[str, str]:
    if not isinstance(rows, list):
        raise ValueError("GLM qualification raw file inventory is missing")
    result = {}
    for row in rows:
        name = _qualification_name(row["path"])
        if name in result:
            raise ValueError("GLM qualification raw inventory repeats a file")
        result[name] = _qualification_sha(row["sha256"])
    return result


def _qualification_profile(root: Path, profile: dict, evidence: dict, expected: dict, checkpoint: str, tp: int):
    """Check small original receipts against a full-raw validator result.

    This verifies the packaged evidence chain, not the externally retained token
    arrays. Promotion still requires running qualification/validate.py on every
    original raw file before pinning the resulting comparison and summary bytes.
    """
    from .fpm_forward.runtime.glm53flash_vllm_kpool_candidate.qualification.validate import (
        validate_checkpoint_identity,
    )

    runtime_kind, mode = profile["runtime_kind"], profile["mode"]
    _qualification_uri(profile["raw_uri"])
    _qualification_sha(profile["original_validator"]["sha256"])
    _qualification_uri(profile["original_validator"]["source_uri"])
    preflight = _qualification_json(root, profile["preflight"])
    receipt = _qualification_json(root, profile["native_receipt"])
    if any(
        receipt.get(key) != value
        for key, value in {
            "status": "native_requests_and_histories_verified",
            "runtime_kind": runtime_kind,
            "mode": mode,
            "checkpoint": checkpoint,
            "tp": tp,
            "policy": "production",
            "correctness_comparison": "NOT_EVALUATED",
            "accuracy_acceptance": "NOT_EVALUATED",
        }.items()
    ):
        raise ValueError("GLM qualification original native profile identity differs")
    checkpoint_identity = validate_checkpoint_identity(preflight, expected, checkpoint)
    runtime = preflight["runtime"]
    wanted_files = {
        **expected["source_pins"],
        **expected["native_binaries"],
        expected["helper_path"]: expected["helper_sha256"][runtime_kind],
    }
    if (
        preflight.get("status") != "passed"
        or runtime.get("version") != expected["versions"][runtime_kind]
        or runtime.get("loaded_files") != wanted_files
        or runtime.get("candidate_wheel_sha256") != (_WHEEL_SHA256 if runtime_kind == "candidate" else None)
    ):
        raise ValueError("GLM qualification actual runtime/source/native binary closure differs")
    settings = {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "mamba_cache_mode": "none",
        "kv_cache_dtype": "fp8_e4m3",
        "max_model_len": 131079,
        "max_num_batched_tokens": 16398,
        "long_prefill_token_threshold": 4097 if mode == "split" else 0,
        "enforce_eager": False,
    }
    args = preflight["public_engine_args"]
    if any(type(args.get(key)) is not type(value) or args[key] != value for key, value in settings.items()):
        raise ValueError("GLM qualification public Engine arguments differ from the production profile")
    for field, value, source_name in (
        ("request_identity", "native_assign_request_id_v1", "request-id-source.json"),
        ("cohort_admission", "native_scheduling_pause_enqueue_v1", "cohort-source.json"),
    ):
        sources = json.loads((root / source_name).read_bytes())["sources"]
        if (
            preflight.get(field + "_protocol") != value
            or evidence.get(field + "_protocol") != value
            or preflight.get(field + "_source_sha256") != {row["path"]: row["sha256"] for row in sources}
        ):
            raise ValueError("GLM qualification native request/cohort protocol differs")
    modes = evidence.get("native_modes", [])
    mapping = evidence.get("external_to_native_request_ids", {})
    splits = evidence.get("actual_prefill_splits", {})
    if (
        evidence.get("requests") != 20
        or evidence.get("checkpoint_identity") != checkpoint_identity
        or "FULL" not in modes
        or sorted(set(modes)) != modes
        or not set(modes) <= {"NONE", "PIECEWISE", "FULL"}
        or len(mapping) != 20
        or len(set(mapping.values())) != 20
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items())
        or set(splits) != {str(rank) for rank in range(tp)}
        or any(set(rows) != set(mapping.values()) for rows in splits.values())
    ):
        raise ValueError("GLM qualification lacks the complete production native request/rank evidence")
    _qualification_sha(evidence["all_tp_trace_digest"])
    inventory = _qualification_inventory(evidence["files"])
    required = {
        "preflight.json",
        "native-receipt.json",
        "outputs.jsonl",
        "effective-native-config.json",
        "worker-installation.json",
        "worker-completion.json",
        "request-id-map.jsonl",
        "cohort-admission.jsonl",
        *(
            f"{prefix}-rank-{rank}.{suffix}"
            for rank in range(tp)
            for prefix, suffix in (("worker", "json"), ("forward", "jsonl"), ("prompts", "jsonl"))
        ),
    }
    if not required <= inventory.keys() or any(
        re.fullmatch(r"(worker|forward|prompts)-rank-\d+\.(json|jsonl)", name) and name not in required
        for name in inventory
    ):
        raise ValueError("GLM qualification raw inventory lacks exact TP rank coverage")
    for field, name in (("preflight", "preflight.json"), ("native_receipt", "native-receipt.json")):
        if inventory[name] != profile[field]["sha256"]:
            raise ValueError("GLM qualification packaged receipt differs from the original raw inventory")
    original = receipt["evidence"]
    original_inventory = _qualification_inventory(original["files"])
    if "native-receipt.json" in original_inventory:
        raise ValueError("GLM qualification original receipt cannot include its own future file hash")
    additions = (["checkpoint_identity"] if "checkpoint_identity" not in original else []) + [
        "files.native-receipt.json"
    ]
    if profile.get("revalidation_added_fields") != additions:
        raise ValueError("GLM qualification must identify exactly the newly derived validation fields")
    if inventory != {**original_inventory, "native-receipt.json": profile["native_receipt"]["sha256"]}:
        raise ValueError("GLM qualification revalidation changed the original raw file inventory")
    # Old original receipts precede checkpoint crossbinding. The new validator
    # may add that proof, never rewrite the original profile or token evidence.
    old_evidence = {k: v for k, v in original.items() if k not in {"files", "checkpoint_identity"}}
    new_evidence = {k: v for k, v in evidence.items() if k not in {"files", "checkpoint_identity"}}
    if old_evidence != new_evidence or original.get("checkpoint_identity", checkpoint_identity) != checkpoint_identity:
        raise ValueError("GLM qualification revalidation changed original native evidence")


def _validate_qualification_summary(expected_sha256: str) -> dict:
    """Require a reviewed immutable, packaged four-cell Engine qualification."""
    root = _qualification_root()
    summary = _qualification_json(root, {"path": "admission-summary.json", "sha256": expected_sha256})
    _candidate_build()
    expected = _qualification_json(root, {"path": "expected-runtime.json", "sha256": _ENGINE_IDENTITY_SHA256})
    fixed = {
        "schema_version": 1,
        "status": "native_engine_qualification_passed",
        "scope": _QUALIFICATION_SCOPE,
        "backend_version": VLLM_KPOOL_CANDIDATE,
        "build_receipt_sha256": _BUILD_SHA256,
        "wheel_sha256": _WHEEL_SHA256,
        "expected_runtime_sha256": _ENGINE_IDENTITY_SHA256,
        "accuracy_acceptance": "NOT_EVALUATED",
        "formal_8_cell_coverage": "NOT_EVALUATED",
    }
    if any(type(summary.get(k)) is not type(v) or summary[k] != v for k, v in fixed.items()):
        raise ValueError("GLM qualification summary identity or scope differs")
    if (
        not isinstance(summary.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", summary["source_commit"]) is None
    ):
        raise ValueError("GLM qualification must identify its immutable validator source commit")
    sources = summary.get("validator_sources", {})
    if set(sources) != set(_QUALIFICATION_SOURCES) or any(
        (root / name).is_symlink()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != _qualification_sha(sources[name])
        for name in sources
    ):
        raise ValueError("GLM qualification validator source bytes differ from the packaged implementation")
    cells = summary.get("cells", [])
    wanted = {(precision, tp) for precision in ("fp8", "nvfp4") for tp in (2, 4)}
    if len(cells) != 4 or {(c.get("checkpoint"), c.get("tp")) for c in cells} != wanted:
        raise ValueError("GLM qualification summary must contain exactly four FP8/NVFP4 TP2/TP4 cells")
    paths, uris = set(), set()
    for cell in cells:
        result = _qualification_json(root, cell["comparison"])
        _qualification_uri(cell["comparison"]["source_uri"])
        constraints = {
            "status": "passed",
            "scope": _QUALIFICATION_SCOPE,
            "checkpoint": cell["checkpoint"],
            "tp": cell["tp"],
            "policy": "production",
            "requests_per_mode": 20,
            "greedy_output_tokens_per_request": 32,
            "comparisons": 40,
            "differences": [],
            "accuracy_acceptance": "NOT_EVALUATED",
            "formal_8_cell_coverage": "NOT_EVALUATED",
        }
        if any(type(result.get(k)) is not type(v) or result[k] != v for k, v in constraints.items()):
            raise ValueError("GLM qualification comparison is not a complete production 20x32/40 pass")
        profiles = cell.get("profiles", [])
        evidence = result.get("native_receipts", [])
        if len(evidence) != 3 or [(p.get("runtime_kind"), p.get("mode")) for p in profiles] != list(
            _QUALIFICATION_PROFILES
        ):
            raise ValueError("GLM qualification must preserve all three ordered native profiles")
        names = [cell["comparison"]["path"]]
        for profile, proof in zip(profiles, evidence, strict=True):
            if profile["raw_uri"] in uris:
                raise ValueError("GLM qualification profiles alias the same external raw directory")
            uris.add(profile["raw_uri"])
            names.extend(profile[field]["path"] for field in ("preflight", "native_receipt"))
            _qualification_profile(root, profile, proof, expected, cell["checkpoint"], cell["tp"])
        for name in names:
            if name in paths:
                raise ValueError("GLM qualification profiles alias the same packaged original receipt")
            paths.add(name)
    return summary


def validate_backend_version(backend: str, version: str) -> str:
    repaired = backend == "vllm" and version == VLLM_KPOOL_CANDIDATE and version in ADMITTED_VLLM_REPAIRS
    if backend not in BASELINE_VERSIONS or (version != BASELINE_VERSIONS[backend] and not repaired):
        raise ValueError(f"unqualified GLM backend runtime: {backend} {version!r}")
    if repaired:
        _validate_qualification_summary(ADMITTED_VLLM_REPAIRS[version])
    return version


def vllm_source_pins(version: str, manifest: Path) -> dict[str, str]:
    """Return the effective source closure for an admitted exact runtime."""
    validate_backend_version("vllm", version)
    pins = json.loads(manifest.read_bytes())
    if version == VLLM_KPOOL_CANDIDATE:
        patch = _candidate_build()["patch"]
        if pins.get(patch["source_path"]) != patch["base_sha256"]:
            raise ValueError("GLM repair source base differs from its reviewed build")
        pins[patch["source_path"]] = patch["patched_sha256"]
        root = Path(__file__).parent / "fpm_forward/runtime/glm53flash_vllm_kpool_candidate"
        native_identity = (root / "qualification/expected-runtime.json").read_bytes()
        v2_source = (root / "v2-source-sha256.json").read_bytes()
        if (
            hashlib.sha256(native_identity).hexdigest() != _ENGINE_IDENTITY_SHA256
            or hashlib.sha256(v2_source).hexdigest() != _V2_SOURCE_SHA256
        ):
            raise ValueError("GLM repaired Engine/V2 source manifest differs")
        for name, sha in {**json.loads(native_identity)["source_pins"], **json.loads(v2_source)}.items():
            if name in pins and pins[name] != sha:
                raise ValueError("GLM repair source closure has conflicting identities")
            pins[name] = sha
    return pins


def vllm_source_manifest_sha256(version: str, manifest: Path) -> str:
    pins = vllm_source_pins(version, manifest)
    if version == BASELINE_VERSIONS["vllm"]:
        # Preserve the already published stock protocol byte-for-byte.
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    return _canonical_sha256(pins)


def vllm_runtime_closure(version: str, manifest: Path) -> dict | None:
    """Expected repaired-runtime files; construction grants no admission."""
    sources = vllm_source_pins(version, manifest)
    if version == BASELINE_VERSIONS["vllm"]:
        return None
    build = _candidate_build()
    return {
        "schema_version": 1,
        "backend_version": version,
        "wheel_sha256": build["wheel_sha256"],
        "build_receipt_sha256": _BUILD_SHA256,
        "qualification_receipt_sha256": ADMITTED_VLLM_REPAIRS[version],
        "runtime_source_manifest_sha256": vllm_source_manifest_sha256(version, manifest),
        "files": {
            **{path: sha for path, sha in sources.items() if path.startswith("vllm/")},
            **build["unchanged_native_binaries"],
        },
    }


def observe_vllm_runtime_closure(version: str, manifest: Path) -> dict | None:
    """Read actual per-worker source and native binary bytes before timing."""
    closure = vllm_runtime_closure(version, manifest)
    if closure is None:
        return None
    import vllm

    if vllm.__version__ != version or importlib.metadata.version("vllm") != version:
        raise ValueError("GLM repair imported package and distribution versions differ")
    package = Path(vllm.__file__).resolve().parent
    observed = {}
    for name, expected in closure["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "vllm":
            raise ValueError("GLM repair closure contains a non-package file")
        path = package.joinpath(*relative.parts[1:]).resolve(strict=True)
        if not path.is_relative_to(package):
            raise ValueError("GLM repair closure file resolves outside the imported package")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"GLM repair source/native binary differs: {name}")
        observed[name] = actual
    return {"contract_sha256": _canonical_sha256(closure), "observed_files": observed}


def validate_vllm_runtime_closure(version: str, manifest: Path, observed: dict | None) -> None:
    closure = vllm_runtime_closure(version, manifest)
    expected = (
        None if closure is None else {"contract_sha256": _canonical_sha256(closure), "observed_files": closure["files"]}
    )
    if observed != expected:
        raise ValueError("GLM vLLM native hardware repair source/binary closure is missing or differs")


def vllm_unaligned_prefill_admitted(version: str) -> bool:
    validate_backend_version("vllm", version)
    return version in ADMITTED_VLLM_REPAIRS


def validate_vllm_source_identity(producer: dict, manifest: Path) -> dict[str, str]:
    pins = vllm_source_pins(producer.get("vllm_package_version"), manifest)
    if producer.get("runtime_source_manifest_sha256") != vllm_source_manifest_sha256(
        producer.get("vllm_package_version"), manifest
    ):
        raise ValueError("GLM vLLM runtime source manifest differs from the admitted source closure")
    return pins


def validate_runtime_pair(backend: str, calibration: dict, holdout: dict) -> str:
    version = validate_backend_version(backend, calibration.get("backend_version"))
    if holdout.get("backend_version") != version:
        raise ValueError("GLM calibration and holdout use different native runtime versions")
    return version
