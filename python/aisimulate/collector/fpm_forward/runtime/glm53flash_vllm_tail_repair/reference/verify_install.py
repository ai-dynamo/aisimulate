# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Actual isolated native CPU import/spec proof; no GPU or model qualification."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU qualification requires CUDA_VISIBLE_DEVICES empty before import")
    root = Path(__file__).resolve().parent
    identity = json.loads((root / "patch-identity.json").read_bytes())
    build = json.loads(args.build_receipt.read_bytes())
    sources = json.loads((root / "expected-source-sha256.json").read_bytes())
    binaries = json.loads((root / "expected-native-binaries.json").read_bytes())
    if (
        build["version"] != identity["version"]
        or build["patch"] != identity
        or build["unchanged_native_binaries"] != binaries
        or len(binaries) != 19
        or sha(args.wheel) != build["wheel_sha256"]
    ):
        raise RuntimeError("build/wheel/patch/native closure differs")
    import torch
    import vllm
    from vllm.v1.kv_cache_interface import KpoolTailSpec, SlidingWindowSpec

    package = Path(vllm.__file__).resolve().parent.parent
    if (
        package != args.install_root.resolve()
        or vllm.__version__ != identity["version"]
        or importlib.metadata.version("vllm") != identity["version"]
    ):
        raise RuntimeError("actual imported distribution differs from isolated installation")
    observed = {name: sha(package / name) for name in {**sources, **binaries}}
    if observed != {**sources, **binaries}:
        raise RuntimeError("actual installed source/native binary bytes differ")
    distribution = importlib.metadata.distribution("vllm")
    actual_binary_names = {
        str(item)
        for item in distribution.files
        if str(item).startswith("vllm/") and (str(item).endswith(".so") or str(item) == "vllm/vllm-rs")
    }
    if actual_binary_names != set(binaries):
        raise RuntimeError("installed RECORD binary membership differs")
    fields = dict(block_size=4, num_kv_heads=2, head_size=128, head_size_v=0, dtype=torch.bfloat16, sliding_window=4)
    tail = KpoolTailSpec(**fields)
    ordinary = SlidingWindowSpec(**fields)
    if (
        tail.uses_slot_mapping is not False
        or ordinary.uses_slot_mapping is not True
        or tail.prefix_cacheable is not False
        or tail.max_admission_blocks_per_request(8192, 131079) != 1
        or tail.max_num_blocks_per_req(None, 131079) != 1
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("actual native spec/CPU contract differs")
    result = {
        "status": "ACTUAL_CPU_INSTALL_SOURCE_AND_NATIVE_SPEC_PASS",
        "version": vllm.__version__,
        "build_receipt_sha256": sha(args.build_receipt),
        "wheel_sha256": sha(args.wheel),
        "installed_root": str(package),
        "observed_files": observed,
        "native_binary_names": sorted(actual_binary_names),
        "native_spec_source": inspect.getsourcefile(KpoolTailSpec),
        "tail_spec": str(tail),
        "tail_uses_slot_mapping": tail.uses_slot_mapping,
        "ordinary_sliding_window_uses_slot_mapping": ordinary.uses_slot_mapping,
        "pytorch_cuda_initialized": False,
        "actual_model_cache_groups": "NOT_EVALUATED",
        "model_correctness": "NOT_EVALUATED",
        "formal_admission": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
