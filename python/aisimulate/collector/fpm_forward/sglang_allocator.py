# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native allocator policy and observed worker identity, without GPU mutation.

Original integration of PyTorch public APIs; no upstream implementation copied.
Configuration semantics reviewed at pytorch/pytorch@cf30153c4c131c8164ee7798e5022d810682e2cb,
c10/core/AllocatorConfig.cpp:108-123 and c10/cuda/CUDACachingAllocator.cpp.
PyTorch license/attribution: see repository THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "sglang_native_allocator_policy_v1"
OPTION = "--sglang-allocator-max-split-size-mb"
ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_NO_CUDA_MEMORY_CACHING")
SOURCE_FILES = ("__init__.py", "version.py", "cuda/memory.py")
LIBRARY_FILES = ("lib/libc10_cuda.so", "lib/libtorch_cuda.so", "lib/libtorch_cpu.so")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_max_split_size(value):
    # Exact pinned parser permits >= the default20MiB large segment. Bound the
    # byte product to the signed memory_stats field, rather than wrapping it.
    if value is not None and (type(value) is not int or not 20 <= value <= ((1 << 63) - 1) // (1 << 20)):
        raise ValueError(f"{OPTION} requires an integer >=20 MiB representable in native memory statistics")


def cli_max_split_size(value):
    try:
        parsed = int(value)
        validate_max_split_size(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return parsed


def request_policy(value):
    validate_max_split_size(value)
    return {"schema": SCHEMA, "backend": "native", "max_split_size_mb": value}


def configured_environment(value):
    validate_max_split_size(value)
    return {
        key: (
            f"backend:native,max_split_size_mb:{value}"
            if key == "PYTORCH_CUDA_ALLOC_CONF" and value is not None
            else None
        )
        for key in ENV_KEYS
    }


def prepare_environment(argv=None):
    """Run before serving imports; reject inherited competing allocator settings."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if sum(item == OPTION or item.startswith(OPTION + "=") for item in argv) > 1:
        raise ValueError("duplicate native allocator request")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(OPTION, type=cli_max_split_size, default=None)
    args, _ = parser.parse_known_args(argv)
    value = args.sglang_allocator_max_split_size_mb
    expected = configured_environment(value)
    for key in ENV_KEYS:
        actual = os.environ.get(key)
        if actual is not None and actual != expected[key]:
            raise ValueError(f"unexpected inherited allocator environment {key}")
    if value is not None:
        if "torch" in sys.modules:
            raise ValueError("explicit allocator policy must be installed before Torch/serving imports")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = expected["PYTORCH_CUDA_ALLOC_CONF"]
    return request_policy(value)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observe_worker(torch, *, rank, run_id, execution_identity, policy, hardware):
    """Observe existing allocator and loaded package; do not initialize/change it."""
    if not isinstance(policy, dict) or policy != request_policy(policy.get("max_split_size_mb")):
        raise ValueError("invalid declared allocator policy")
    root = Path(torch.__file__).resolve().parent
    files = {
        name: {"path": str(root / name), "sha256": _file_sha(root / name)} for name in (*SOURCE_FILES, *LIBRARY_FILES)
    }
    extension = Path(torch._C.__file__).resolve()
    if extension.parent != root:
        raise ValueError("Torch native extension is outside its observed package")
    files["_C"] = {"path": str(extension), "sha256": _file_sha(extension)}
    mapped = {
        line.split(maxsplit=5)[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(line.split(maxsplit=5)) == 6
    }
    for name in LIBRARY_FILES:
        actual = {path for path in mapped if Path(path).name == Path(name).name}
        if actual != {str(root / name)}:
            raise ValueError(f"Torch allocator dependency mapping differs: {name}")
    identity = {
        "schema": SCHEMA,
        "rank": rank,
        "pid": os.getpid(),
        "run_id": run_id,
        "execution_identity": execution_identity,
        "hardware": hardware,
        "requested_policy": policy,
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "max_split_size_bytes": torch.cuda.memory_stats()["max_split_size"],
        "torch": {
            "version": str(torch.__version__),
            "git_revision": torch.version.git_version,
            "cuda_version": torch.version.cuda,
            "files": files,
        },
    }
    validate_worker(
        identity, rank=rank, run_id=run_id, execution_identity=execution_identity, policy=policy, hardware=hardware
    )
    return identity


def validate_worker(value, *, rank, run_id, execution_identity, policy, hardware=None):
    """Validate exact provenance and return rank-independent actual policy."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("native allocator worker receipt missing or invalid")
    if not isinstance(policy, dict) or policy != request_policy(policy.get("max_split_size_mb")):
        raise ValueError("native allocator requested policy differs")
    if (
        type(value.get("rank")) is not int
        or value["rank"] != rank
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or value.get("run_id") != run_id
        or value.get("execution_identity") != execution_identity
        or value.get("requested_policy") != policy
    ):
        raise ValueError("native allocator worker rank/run/checkpoint/request identity differs")
    if hardware is not None and value.get("hardware") != hardware:
        raise ValueError("native allocator receipt differs from actual rank hardware")
    if value.get("environment") != configured_environment(policy["max_split_size_mb"]):
        raise ValueError("actual allocator environment differs from the requested native policy")
    expected = -1 if policy["max_split_size_mb"] is None else policy["max_split_size_mb"] * (1 << 20)
    if (
        value.get("allocator_backend") != "native"
        or type(value.get("max_split_size_bytes")) is not int
        or value["max_split_size_bytes"] != expected
    ):
        raise ValueError("actual allocator backend/effective max split differs from the requested policy")
    torch = value.get("torch", {})
    if (
        not isinstance(torch, dict)
        or not isinstance(torch.get("version"), str)
        or not torch["version"]
        or not isinstance(torch.get("git_revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", torch.get("git_revision", ""))
        or not isinstance(torch.get("cuda_version"), str)
        or not torch["cuda_version"]
    ):
        raise ValueError("actual Torch build identity missing")
    files = torch.get("files")
    if not isinstance(files, dict) or set(files) != {*SOURCE_FILES, *LIBRARY_FILES, "_C"}:
        raise ValueError("actual Torch allocator source/library closure incomplete")
    for name, ref in files.items():
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not Path(ref["path"]).is_absolute()
            or not isinstance(ref["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])
        ):
            raise ValueError(f"actual Torch file identity invalid: {name}")
    root = Path(files["__init__.py"]["path"]).parent
    if any(Path(files[name]["path"]) != root / name for name in (*SOURCE_FILES, *LIBRARY_FILES)) or (
        Path(files["_C"]["path"]).parent != root
        or not Path(files["_C"]["path"]).name.startswith("_C.")
        or not Path(files["_C"]["path"]).name.endswith(".so")
    ):
        raise ValueError("actual Torch allocator files are outside the observed package")
    return {
        key: value[key]
        for key in ("schema", "requested_policy", "environment", "allocator_backend", "max_split_size_bytes", "torch")
    }
