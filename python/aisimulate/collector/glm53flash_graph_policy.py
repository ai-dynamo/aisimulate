# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound ordinary decode dispatch; no holdout-derived answer lookup.

Original inspection wrappers. Native predicates: sgl-project/sglang at
94602c9c2b7cbdb8efd5c52802dac6a1c180089e, model_executor/runner/
{decode_cuda_graph_runner,base_cuda_graph_runner}.py. See README.glm53flash.md.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import re
from pathlib import Path

from collector.glm53flash_contract import BACKENDS, CHECKPOINTS, canonical_json, sha256_json

NATIVE_SOURCE_SHA256 = "401b762a863931720b2b5cdc7b64246fac11cb215dbf6ea0fd19f3db24ce7e49"

SOURCE_PINS = {
    "srt/model_executor/runner/decode_cuda_graph_runner.py": (
        "55892739b9c577ae43a60d5d31eac53f81e2b4aeca57ef5368b9c881117889d8"
    ),
    "srt/model_executor/runner/base_cuda_graph_runner.py": (
        "03098df21a963d28f8075e0630a2bc1c356560449f9ca5f9e74dfb5f9892b7ed"
    ),
    "srt/model_executor/runner_backend/full_cuda_graph_backend.py": (
        "0dc52a9a581636a20f5070cbb81d921bc56e4fb3394a9a1cf601747271c6905b"
    ),
    "srt/model_executor/runner/shape_key.py": "26e3f15209b654345a35966bd817ff8d0eb6c4c118527e78ca1e89942d5ea2c5",
}
DIRECT_FLAGS = (
    "enable_torch_compile",
    "is_encoder_decoder",
    "require_mlp_tp_gather",
    "require_attn_tp_gather",
    "require_mlp_sync",
    "enable_two_batch_overlap",
    "enable_pdmux",
    "ragged_verify_mode",
    "enable_prefill_cp",
)
EXTRA_FLAGS = ("enable_lora", "speculative_enabled", "metadata_glue_enabled", "reuse_output_buffer")


def validate_snapshot(value):
    if value.get("source_pins") != SOURCE_PINS or value.get("backend") != "sglang":
        raise ValueError("native graph dispatch source is unqualified")
    if (value.get("backend_version"), value.get("backend_revision")) != BACKENDS["sglang"]:
        raise ValueError("native graph dispatch version is unqualified")
    flags = value.get("native_flags", {})
    if set(flags) != set(DIRECT_FLAGS + EXTRA_FLAGS) or any(v is not False for v in flags.values()):
        raise ValueError("native graph dispatch has unsupported actual eligibility predicates")
    sizes = value.get("capture_sizes")
    if (
        not isinstance(sizes, list)
        or not sizes
        or any(type(s) is not int or s < 1 for s in sizes)
        or sizes != sorted(set(sizes))
        or type(value.get("max_bs")) is not int
        or value.get("max_bs") != sizes[-1]
        or type(value.get("disable_padding")) is not bool
        or type(value.get("captured_req_width")) is not int
        or value["captured_req_width"] != 1
    ):
        raise ValueError("native graph capture sizes/width are not an exact ordinary decode policy")
    expected = [{"size": size, "stream_idx": None, "variant_label": None, "attention_variant": None} for size in sizes]
    if value.get("captured_keys") != expected or any(
        type(key.get("size")) is not int for key in value.get("captured_keys", [])
    ):
        raise ValueError("actual native graph keys contain missing captures or unsupported variants")
    if type(value.get("tp_rank")) is not int or value["tp_rank"] < 0:
        raise ValueError("native graph dispatch lacks an exact rank")
    return value


def snapshot_native(runner):
    """Read actual initialized runner predicates and graph keys, before timing."""
    import sglang

    if (
        type(runner).__module__ != "sglang.srt.model_executor.runner.decode_cuda_graph_runner"
        or type(runner).__name__ != "DecodeCudaGraphRunner"
    ):
        raise ValueError("dispatch snapshot requires the actual pinned native decode runner")
    package = Path(sglang.__file__).resolve().parent
    hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    model = runner.model_runner
    glue = runner._metadata_glue
    flags = {name: getattr(runner, name) for name in DIRECT_FLAGS}
    flags.update(
        enable_lora=model.lora_manager is not None,
        speculative_enabled=not model.spec_algorithm.is_none(),
        metadata_glue_enabled=glue is not None and not glue.disabled,
        reuse_output_buffer=runner.backend._reuse_output_buffer,
    )
    value = {
        "backend": "sglang",
        "backend_version": sglang.__version__,
        "backend_revision": BACKENDS["sglang"][1],
        "source_pins": hashes,
        "native_flags": flags,
        "capture_sizes": sorted(runner.capture_bs),
        "max_bs": runner.max_bs,
        "captured_req_width": runner.captured_req_width,
        "disable_padding": runner.disable_padding,
        "captured_keys": sorted(
            (dataclasses.asdict(key) for key in runner.backend._graphs), key=lambda key: key["size"]
        ),
        "tp_rank": model.ps.tp_rank,
    }
    return validate_snapshot(value)


def padded_batch(snapshot, batch):
    validate_snapshot(snapshot)
    if type(batch) is not int or batch < 1:
        raise ValueError("native graph needs a positive exact batch")
    sizes = snapshot["capture_sizes"]
    index = bisect.bisect_left(sizes, batch)
    if index == len(sizes) or (snapshot["disable_padding"] and sizes[index] != batch):
        raise ValueError("batch is outside the initialized native capture policy")
    return sizes[index]


def build_policy(
    snapshots,
    *,
    checkpoint_format,
    tp_size,
    provenance,
    resolved_config_sha256,
    state_layout_sha256,
    capture_registry_sha256,
):
    """Assemble only checked calibration receipts; no holdout coordinates enter."""
    if type(tp_size) is not int or tp_size not in (2, 4) or checkpoint_format not in CHECKPOINTS:
        raise ValueError("unqualified graph topology/checkpoint")
    if set(snapshots) != set(range(tp_size)):
        raise ValueError("native graph policy requires all TP rank snapshots")
    first = None
    for rank, snapshot in snapshots.items():
        validate_snapshot(snapshot)
        if snapshot["tp_rank"] != rank:
            raise ValueError("native policy rank mismatch")
        common = {k: v for k, v in snapshot.items() if k != "tp_rank"}
        if first is not None and common != first:
            raise ValueError("TP ranks selected different native graph policies")
        first = common
    assert first is not None
    if provenance.get("checkpoint_revision") != CHECKPOINTS[checkpoint_format][1]:
        raise ValueError("graph policy checkpoint differs from its actual calibration provenance")
    if (
        provenance.get("source_sha256") != NATIVE_SOURCE_SHA256
        or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("config_sha256", ""))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", provenance.get("runtime_digest", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", resolved_config_sha256)
    ):
        raise ValueError("native graph calibration source/config/runtime identity is incomplete")
    for values in (state_layout_sha256, capture_registry_sha256):
        if set(values) != set(range(tp_size)):
            raise ValueError("native graph policy requires complete state/capture receipts")
    for rank in range(tp_size):
        hashes = [state_layout_sha256[rank], *capture_registry_sha256[rank]]
        if len(capture_registry_sha256[rank]) != len(first["capture_sizes"]) or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in hashes
        ):
            raise ValueError("native graph state/capture hashes are incomplete")
    return {
        "schema_version": 1,
        "backend": "sglang",
        "backend_version": first["backend_version"],
        "backend_revision": first["backend_revision"],
        "checkpoint_format": checkpoint_format,
        "checkpoint_revision": CHECKPOINTS[checkpoint_format][1],
        "tp_size": tp_size,
        "phase": "generation",
        "runtime_mode": "FULL",
        "capture_sizes": first["capture_sizes"],
        "disable_padding": first["disable_padding"],
        "captured_req_width": first["captured_req_width"],
        "native_flags": first["native_flags"],
        "source_pins": first["source_pins"],
        "source_sha256": provenance["source_sha256"],
        "config_sha256": provenance["config_sha256"],
        "runtime_digest": provenance["runtime_digest"],
        "resolved_config_sha256": resolved_config_sha256,
        "native_policy_receipt_sha256": sha256_json({str(rank): value for rank, value in sorted(snapshots.items())}),
        "state_layout_sha256": {str(rank): value for rank, value in sorted(state_layout_sha256.items())},
        "capture_registry_sha256": {str(rank): value for rank, value in sorted(capture_registry_sha256.items())},
    }


def persist_snapshot(runner, output):
    value = snapshot_native(runner)
    path = Path(output) / f"graph-policy-rank-{value['tp_rank']}.json"
    encoded = canonical_json(value).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("native initialized graph policy changed during collection")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return {"file": path.name, "sha256": hashlib.sha256(encoded).hexdigest()}
