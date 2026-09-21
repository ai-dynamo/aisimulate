# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory and preflight for the image-specific Rubin SGLang collectors.

Image declarations come from Dynamo CI pipeline 68286189, job 443229440:
https://gitlab-master.nvidia.com/dl/ai-dynamo/dynamo-ci/-/jobs/443229440
The image digest must be enforced by the launcher. Neither a caller-supplied
image reference nor environment build declarations attest the running image.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

IMAGE_REPOSITORY = "gitlab-master.nvidia.com:5005/dl/ai-dynamo/dynamo-ci"
IMAGE_TAG = f"{IMAGE_REPOSITORY}:8a8fb0687160e73169fd6236d93ed19dec80ef54-68286189-rubin-sglang-arm64"
IMAGE_INDEX_DIGEST = "sha256:53299500a280c8de34bd484507a45b2f83b4d5e7c999b77284fa31930f7e63ab"
IMAGE_ARM64_DIGEST = "sha256:1c7ffbccde1dd9a3ec894db29a1aa3721e3ba393fc7337b2148b2f184aa0bcab"
IMAGE_REF = f"{IMAGE_REPOSITORY}@{IMAGE_INDEX_DIGEST}"
DYNAMO_COMMIT = "60fd5c53e66c3718a7ae2118927e9dd294c8d9e9"
EXPECTED_BUILD_ENV = {
    "SGLANG_VERSION": "0.5.18+02c5a855",
    "NVIDIA_SGLANG_VERSION": "rubin.0.8full",
    "CUDA_VERSION": "13.5.0.012",
}
# Native serving constraints at SGLang 02c5a855aceb968c310e6fbc6632270e26edc84b,
# relative to python/sglang/srt. MoE's phase-independent perf key requires the
# finalized path: models/deepseek_v2.py:904-1023 otherwise defers finalization
# during capture; layers/moe/fused_moe_triton/layer.py:387-391 reads this flag.
# Prefill graphs change DSA dispatch in layers/attention/dsa_backend.py:3313-3339
# and layers/attention/dsa/dsa_indexer.py:1601-1608. Keep eager prefill in both
# serving and collection; decode CUDA graphs retain their native behavior.
REQUIRED_SERVING_ENV = {"SGLANG_ENABLE_MOE_DEFERRED_FINALIZE": "0"}
REQUIRED_SERVER_ARGS = {"disable_prefill_cuda_graph": True}
_DISTRIBUTIONS = ("sglang", "torch", "flashinfer-python", "flashinfer", "sgl-kernel", "aisimulate", "nixl")
_IMPORTS = ("sglang", "aisimulate", "aisimulate._runtime")
_CHECKPOINT_FILES = ("config.json", "hf_quant_config.json", "quantization_config.json", "quantize_config.json")


def declared_serving_configuration() -> dict[str, Any]:
    """Required launch settings; this declaration does not attest a live server."""
    return {"environment": dict(REQUIRED_SERVING_ENV), "server_args": dict(REQUIRED_SERVER_ARGS)}


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _package_versions() -> dict[str, dict[str, str | None]]:
    packages = {}
    for name in _DISTRIBUTIONS:
        try:
            packages[name] = {"version": importlib.metadata.version(name), "error": None, "error_type": None}
        except Exception as error:
            packages[name] = {"version": None, "error": _error(error), "error_type": type(error).__name__}
    return packages


def _cuda_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"available": None, "torch_version": None, "torch_cuda_version": None, "devices": []}
    try:
        torch = importlib.import_module("torch")
        result["torch_version"] = str(torch.__version__)
        result["torch_cuda_version"] = torch.version.cuda
        result["available"] = torch.cuda.is_available()
        if result["available"]:
            for index in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(index)
                result["devices"].append(
                    {
                        "index": index,
                        "name": device.name,
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_bytes": device.total_memory,
                        "uuid": str(device.uuid) if getattr(device, "uuid", None) is not None else None,
                    }
                )
    except Exception as error:
        # A broken framework/driver is an observation, not an omitted device.
        result["error"] = _error(error)
    return result


def _import_inventory() -> dict[str, dict[str, str | None]]:
    imports = {}
    for name in _IMPORTS:
        try:
            module = importlib.import_module(name)
            imports[name] = {"file": getattr(module, "__file__", None), "error": None}
        except Exception as error:
            imports[name] = {"file": None, "error": _error(error)}
    return imports


def _checkpoint_inventory(checkpoint_dir: str | Path) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    files = {}
    for name in _CHECKPOINT_FILES:
        try:
            payload = (root / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        except FileNotFoundError as error:
            files[name] = {"error": _error(error)} if name == "config.json" else {"present": False}
        except OSError as error:
            files[name] = {"error": _error(error)}
    return {"directory": str(root), "files": files}


def collect_inventory(
    *, checkpoint_dir: str | Path | None = None, launcher_image: str | None = None, check_imports: bool = False
) -> dict[str, Any]:
    """Observe packages and CUDA without importing optional frameworks at module load.

    Only explicitly named build/serving variables and checkpoint metadata files are read.
    Distribution versions are observations; the image's SGLANG_VERSION declaration
    does not establish an expected installed SGLang distribution version.
    """
    observed: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "package_versions": _package_versions(),
        "reported_build_environment": {name: os.environ.get(name) for name in EXPECTED_BUILD_ENV},
        "serving_environment": {name: os.environ.get(name) for name in REQUIRED_SERVING_ENV},
        "cuda": _cuda_inventory(),
    }
    if check_imports:
        observed["imports"] = _import_inventory()
    if checkpoint_dir is not None:
        observed["checkpoint"] = _checkpoint_inventory(checkpoint_dir)
    return {
        "declared_image": {
            "tag": IMAGE_TAG,
            "reference": IMAGE_REF,
            "index_digest": IMAGE_INDEX_DIGEST,
            "arm64_manifest_digest": IMAGE_ARM64_DIGEST,
            "dynamo_commit": DYNAMO_COMMIT,
            "platform": "linux/arm64",
            "build_environment": dict(EXPECTED_BUILD_ENV),
            "sglang_distribution_version": None,
        },
        "declared_serving_configuration": declared_serving_configuration(),
        "launcher_provenance": {"image": launcher_image, "image_identity_verified": False},
        "observed": observed,
    }


def validate_runtime(inventory: dict[str, Any]) -> list[str]:
    """Return unmet preflight requirements, without claiming image attestation.

    This checks the hardware, reported build metadata, and required serving
    environment, not kernel compatibility or a serving run. Optional import and
    checkpoint checks apply when requested during inventory collection. Missing
    observations fail rather than imply success.
    """
    errors = []
    observed = inventory.get("observed", {})
    host = observed.get("platform", {})
    if host.get("system") != "Linux" or host.get("machine") != "aarch64":
        errors.append("Rubin collectors require an observed Linux/aarch64 host")
    for name, expected in EXPECTED_BUILD_ENV.items():
        actual = observed.get("reported_build_environment", {}).get(name)
        if actual != expected:
            errors.append(f"Build declaration {name}: expected {expected!r}, observed {actual!r}")
    for name, expected in REQUIRED_SERVING_ENV.items():
        actual = observed.get("serving_environment", {}).get(name)
        if actual != expected:
            errors.append(
                f"Serving environment {name}: expected {expected!r}, observed {actual!r}; "
                "set the required value before launching Python"
            )
    for name in _DISTRIBUTIONS:
        package = observed.get("package_versions", {}).get(name, {})
        if package.get("error") and (
            name in ("sglang", "torch") or package.get("error_type") != "PackageNotFoundError"
        ):
            errors.append(f"Package metadata {name} failed: {package['error']}")
        elif name in ("sglang", "torch") and not package.get("version"):
            errors.append(f"Missing package version observation for {name}: {package.get('error')}")
    cuda = observed.get("cuda", {})
    if cuda.get("error"):
        errors.append(f"CUDA inventory failed: {cuda['error']}")
    if cuda.get("available") is not True or not cuda.get("devices"):
        errors.append("At least one visible CUDA device is required")
    if not cuda.get("torch_cuda_version"):
        errors.append("Missing torch CUDA runtime version")
    for device in cuda.get("devices", []):
        if device.get("capability") != [10, 7]:
            errors.append(f"CUDA device {device.get('index')} requires SM107, observed {device.get('capability')}")
    for name, result in observed.get("imports", {}).items():
        if result.get("error"):
            errors.append(f"Import {name} failed: {result['error']}")
    for name, result in observed.get("checkpoint", {}).get("files", {}).items():
        if result.get("error"):
            errors.append(f"Checkpoint metadata {name} failed: {result['error']}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, help="Hash config.json and available quantization sidecars")
    parser.add_argument(
        "--launcher-image", help="Record the launcher's image reference; this is not verified in-container"
    )
    parser.add_argument(
        "--check-imports", action="store_true", help="Probe SGLang and AISimulate/native runtime imports"
    )
    parser.add_argument("--validate", action="store_true", help="Exit nonzero if the observed runtime fails preflight")
    args = parser.parse_args(argv)
    # Flush pre-existing C output before redirecting, and import diagnostics before
    # restoring descriptor 1. Otherwise libc may emit buffered diagnostics after JSON.
    flush_native_streams = ctypes.CDLL(None).fflush
    flush_native_streams.argtypes = [ctypes.c_void_p]
    flush_native_streams.restype = ctypes.c_int
    sys.stdout.flush()
    flush_native_streams(None)
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            inventory = collect_inventory(
                checkpoint_dir=args.checkpoint_dir,
                launcher_image=args.launcher_image,
                check_imports=args.check_imports,
            )
    finally:
        flush_native_streams(None)
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    errors = validate_runtime(inventory) if args.validate else None
    inventory["validation"] = {"requested": args.validate, "errors": errors}
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
