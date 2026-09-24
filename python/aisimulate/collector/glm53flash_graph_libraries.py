# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify the already-bound PyTorch CUDA runtime without choosing by path order.

Original bindings to documented Linux dlopen/dlinfo/dladdr and ELF relocations.
No CUDA function is invoked. See README.glm53flash.md for qualification scope.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path


class _LinkMap(ctypes.Structure):
    _fields_ = (("address", ctypes.c_void_p), ("name", ctypes.c_char_p))


class _DlInfo(ctypes.Structure):
    _fields_ = (
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("symbol_address", ctypes.c_void_p),
    )


def _maps():
    rows = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        start, end = (int(value, 16) for value in fields[0].split("-"))
        rows.append(
            {"start": start, "end": end, "permissions": fields[1], "path": fields[5] if len(fields) == 6 else ""}
        )
    return rows


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _loader():
    loader = ctypes.CDLL(None)
    signatures = {
        "dlopen": ([ctypes.c_char_p, ctypes.c_int], ctypes.c_void_p),
        "dlclose": ([ctypes.c_void_p], ctypes.c_int),
        "dlinfo": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
        "dladdr": ([ctypes.c_void_p, ctypes.POINTER(_DlInfo)], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(loader, name)
        function.argtypes, function.restype = arguments, result
    return loader


def _existing(loader, path):
    # ctypes.CDLL(path, mode=...) unconditionally adds RTLD_NOW. Use dlopen
    # directly to avoid promoting the original caller's lazy relocations.
    handle = loader.dlopen(os.fsencode(path), os.RTLD_NOLOAD | os.RTLD_LAZY)
    if not handle:
        raise RuntimeError(f"CUDA provider inspection requires an already loaded object: {path}")
    return handle


def _identity(loader, path):
    handle = _existing(loader, path)
    try:
        link = ctypes.POINTER(_LinkMap)()
        if loader.dlinfo(handle, 2, ctypes.byref(link)) != 0 or not link:
            raise RuntimeError("CUDA provider inspection could not obtain RTLD_DI_LINKMAP")
        actual = Path(link.contents.name.decode()).resolve(strict=True)
        requested = Path(path).resolve(strict=True)
        if not actual.samefile(requested):
            raise RuntimeError("loader object differs from the observed CUDA library")
        stat = actual.stat()
        return {
            "path": str(actual),
            "sha256": _sha(actual),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "load_bias": link.contents.address or 0,
        }
    finally:
        if loader.dlclose(handle) != 0:
            raise RuntimeError("could not release the provider inspection reference")


def _relocations(loader, caller, rows):
    process = subprocess.run(["readelf", "-rW", caller["path"]], capture_output=True, text=True, check=True, timeout=60)
    observations = []
    with open("/proc/self/mem", "rb", buffering=0) as memory:
        for line in process.stdout.splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[2] not in {
                "R_AARCH64_JUMP_SLOT",
                "R_AARCH64_GLOB_DAT",
                "R_X86_64_JUMP_SLOT",
                "R_X86_64_GLOB_DAT",
            }:
                continue
            symbol = fields[4].split("@", 1)[0]
            if not symbol.startswith(("cudaGraph", "cudaStream", "cudaEvent")):
                continue
            slot = caller["load_bias"] + int(fields[0], 16)
            if not any(row["start"] <= slot and slot + 8 <= row["end"] and "r" in row["permissions"] for row in rows):
                raise RuntimeError("CUDA relocation is outside an observed readable mapping")
            raw = os.pread(memory.fileno(), 8, slot)
            if len(raw) != 8:
                raise RuntimeError("CUDA relocation pointer read is incomplete")
            pointer = int.from_bytes(raw, "little")
            info = _DlInfo()
            if not pointer or not loader.dladdr(pointer, ctypes.byref(info)) or not info.filename:
                raise RuntimeError("CUDA relocation has no resolved provider")
            observations.append(
                {
                    "symbol": symbol,
                    "relocation": fields[2],
                    "slot": slot,
                    "target": pointer,
                    "provider_path": str(Path(info.filename.decode()).resolve(strict=True)),
                    "provider_load_bias": info.base,
                }
            )
    return observations


def _select_provider(candidates, callers):
    """Reject unresolved/mixed bindings, including byte-identical second instances."""
    identities = {(item["path"], item["load_bias"]): item for item in candidates}
    if len(identities) != len(candidates) or len({item["sha256"] for item in candidates}) != 1:
        raise RuntimeError("co-loaded CUDA runtimes differ from the qualified identical-binary case")
    required = {
        "libtorch_cuda.so": {
            "cudaGraphLaunch",
            "cudaGraphInstantiateWithFlags",
            "cudaStreamBeginCapture",
            "cudaStreamEndCapture",
        },
        "libc10_cuda.so": {"cudaStreamGetCaptureInfo"},
    }
    providers = set()
    for caller in callers:
        name = Path(caller["path"]).name
        if name not in required or not required[name].issubset({row["symbol"] for row in caller["relocations"]}):
            raise RuntimeError("PyTorch CUDA caller lacks the required actual graph/stream relocations")
        for row in caller["relocations"]:
            identity = (row["provider_path"], row["provider_load_bias"])
            if identity not in identities:
                raise RuntimeError("PyTorch CUDA relocation is unresolved or bound outside the observed runtimes")
            providers.add(identity)
    if len(callers) != 2 or {Path(item["path"]).name for item in callers} != required.keys() or len(providers) != 1:
        raise RuntimeError("PyTorch CUDA graph/stream callers do not share one actual runtime provider")
    return identities[providers.pop()]


def resolve_torch_cudart(mapped):
    """Return the actual PyTorch provider and all observations for multiple mappings.

    GB300 qualification observed two separate mappings of the same CUDA13 binary.
    Both PyTorch CUDA callers must already bind all inspected graph/stream/event
    relocations to one of those instances. Unknown binary mixtures remain rejected.
    """
    if (
        platform.machine() not in {"aarch64", "x86_64"}
        or ctypes.sizeof(ctypes.c_void_p) != 8
        or sys.byteorder != "little"
    ):
        raise RuntimeError("CUDA provider inspection requires qualified little-endian Linux ELF64")
    torch = sys.modules.get("torch")
    if torch is None or not getattr(torch, "__file__", None):
        raise RuntimeError("CUDA graph provider requires the already imported PyTorch package")
    rows = _maps()
    observed_paths = {str(Path(row["path"]).resolve()) for row in rows if row["path"].startswith("/")}
    library_dir = Path(torch.__file__).resolve().parent / "lib"
    loader = _loader()
    candidates = [_identity(loader, path) for path in sorted(mapped)]
    callers = []
    for name in ("libtorch_cuda.so", "libc10_cuda.so"):
        path = (library_dir / name).resolve(strict=True)
        if str(path) not in observed_paths:
            raise RuntimeError("PyTorch CUDA caller is not already mapped from the active package")
        caller = _identity(loader, path)
        caller["relocations"] = _relocations(loader, caller, rows)
        callers.append(caller)
    selected = _select_provider(candidates, callers)
    after = {
        row["path"]
        for row in _maps()
        if row["path"].startswith("/") and Path(row["path"]).name.startswith("libcudart.so")
    }
    if after != set(mapped):
        raise RuntimeError("CUDA runtime mappings changed during provider inspection")
    return selected["path"], {
        "policy": "actual_torch_relocations_identical_cudart_v1",
        "candidates": candidates,
        "callers": callers,
        "selected_path": selected["path"],
        "selected_load_bias": selected["load_bias"],
    }
