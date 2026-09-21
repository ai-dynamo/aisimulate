# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native V4.1 component collection and strict physical-key aggregation."""

from __future__ import annotations

import fcntl
import json
import os
import statistics
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from collector.case_generator import _framework_specific_model_case_values, get_base_common_case_values
from collector.sglang.dsv41_contract import build_manifest, validate_row

__compat__ = "sglang@1aa0e962b206102b7c439a4a0c4981cfec6e87bc"


def get_dsv41_module_test_cases() -> list[dict]:
    values = _framework_specific_model_case_values("dsv41_module", "sglang")
    sweep = get_base_common_case_values("dsv41_module")
    if not sweep:
        raise ValueError("missing V4.1 workload sweep")
    selected_profile = os.environ.get("DSV41_EXECUTION_PROFILE")
    if selected_profile not in (None, "full", "decoder_bounded"):
        raise ValueError("DSV41_EXECUTION_PROFILE must be full or decoder_bounded")
    cases = []
    for value in values:
        for tp in value["tensor_parallel_sizes"]:
            for profile in value["execution_profiles"]:
                if selected_profile is not None and selected_profile != profile:
                    continue
                cases.append(
                    {
                        "id": f"dsv41_{profile}_tp{tp}_{value['model_path'].replace('/', '_')}",
                        "params": [value["model_path"], tp, profile, sweep],
                    }
                )
    return cases


def aggregate_rank_records(paths: list[Path], tp_size: int) -> list[dict]:
    """Reduce repeated actual invocations to median of per-sample rank maximum.

    Raw rank samples remain adjacent evidence. No cross-runtime/config or
    cross-profile rows merge; duplicate execution of a physical shape within
    one sample is rejected unless its workload identity is explicitly equal.
    """
    groups = defaultdict(list)
    if {path.name for path in paths} != {f"rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError("missing or unexpected TP rank files")
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            validate_row(row)
            key = tuple(row[k] for k in ("component", "geometry", "batch_size", "prefix", "x"))
            groups[key].append(row)
    output = []
    for rows in groups.values():
        signatures = {
            tuple(
                row[k]
                for k in (
                    "source_sha256",
                    "config_sha256",
                    "runtime_digest",
                    "used_cuda_graph",
                    "execution_profile",
                    "kernel_source",
                )
            )
            for row in rows
        }
        if len(signatures) != 1:
            raise ValueError("incompatible invocations share a physical V4.1 key")
        samples = defaultdict(dict)
        for row in rows:
            sample = samples[(row["sample"], row["invocation"])]
            if row["tp_rank"] in sample:
                raise ValueError("duplicate TP rank within one physical invocation")
            sample[row["tp_rank"]] = row["latency"]
        if any(set(sample) != set(range(tp_size)) for sample in samples.values()):
            raise ValueError("incomplete TP rank set within physical invocation")
        result = {k: v for k, v in rows[0].items() if k not in ("sample", "invocation", "tp_rank")}
        result["latency"] = statistics.median(max(values.values()) for values in samples.values())
        result["sample_count"] = len(samples)
        output.append(result)
    return output


def aggregate_baseline_records(paths: list[Path], tp_size: int) -> dict[str, list[dict]]:
    """Admit measured native baselines only after every rank/sample agrees."""
    import math

    if {p.name for p in paths} != {f"baseline-rank-{rank}.jsonl" for rank in range(tp_size)}:
        raise ValueError("missing baseline rank files")
    columns = {
        "gemm": ("gemm_dtype", "m", "n", "k"),
        "moe": (
            "moe_dtype",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "topk",
            "num_experts",
            "moe_tp_size",
            "moe_ep_size",
            "distribution",
        ),
        "nccl": ("op_name", "nccl_dtype", "num_gpus", "message_size"),
    }
    groups = defaultdict(list)
    for path in paths:
        humming_qualification = None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            kind = row["kind"]
            if kind not in columns or not math.isfinite(row["latency"]) or row["latency"] <= 0:
                raise ValueError("invalid native baseline observation")
            if kind == "moe" and row["moe_dtype"] == "w4a16_mxfp4_humming":
                from collector.sglang.dsv41_humming import validate_humming_observations

                if humming_qualification is None:
                    receipt = path.parent / f"humming-qualification-rank-{row['tp_rank']}.json"
                    if not receipt.is_file():
                        raise ValueError("missing actual native Humming qualification")
                    humming_qualification = json.loads(receipt.read_text())
                if (
                    humming_qualification.get("state") != "actual_bf16_native_humming_calls_verified"
                    or humming_qualification.get("tp_rank") != row["tp_rank"]
                    or row["kernel_source"] != "sglang_mxfp4_humming_moe"
                    or any(
                        humming_qualification.get("provenance", {}).get(key) != row[key]
                        for key in ("source_sha256", "config_sha256", "runtime_digest", "execution_profile")
                    )
                ):
                    raise ValueError("native Humming qualification provenance differs")
                validate_humming_observations(humming_qualification["calls"], humming_qualification["configs"])
            key = (kind, *(row[c] for c in columns[kind]))
            groups[key].append(row)
    result = defaultdict(list)
    for key, rows in groups.items():
        signatures = {
            tuple(
                r[k]
                for k in (
                    "source_sha256",
                    "config_sha256",
                    "runtime_digest",
                    "used_cuda_graph",
                    "kernel_source",
                    "execution_profile",
                )
            )
            for r in rows
        }
        if len(signatures) != 1:
            raise ValueError("mixed baseline provenance")
        samples = defaultdict(dict)
        for row in rows:
            sample = samples[row["sample"]]
            if row["tp_rank"] in sample:
                raise ValueError("duplicate baseline rank/sample")
            sample[row["tp_rank"]] = row["latency"]
        if any(set(sample) != set(range(tp_size)) for sample in samples.values()):
            raise ValueError("incomplete baseline rank/sample")
        if key[0] == "moe":
            histograms = {tuple(r["routing_histogram"]) for r in rows}
            if len(histograms) != 1 or len(next(iter(histograms))) != rows[0]["num_experts"]:
                raise ValueError("baseline uniform routing differs between ranks")
            if sum(next(iter(histograms))) != rows[0]["num_tokens"] * rows[0]["topk"]:
                raise ValueError("baseline routing does not cover every token slot")
        measured = {c: rows[0][c] for c in columns[key[0]]}
        if key[0] == "nccl":
            # Native raw evidence records physical bytes. NcclOp and the
            # existing nccl-tests collector key their table by ELEMENTS.
            if measured["nccl_dtype"] not in ("half", "bfloat16"):
                raise ValueError("NCCL baseline requires a supported 16-bit dtype: half or bfloat16")
            if measured["message_size"] % 2:
                raise ValueError("16-bit NCCL payload must contain whole elements")
            measured["message_size"] //= 2
            measured["wire_dtype"] = "bfloat16"
        measured.update(
            latency=statistics.median(max(s.values()) for s in samples.values()),
            kernel_source=rows[0]["kernel_source"],
            sample_count=len(samples),
        )
        result[key[0]].append(measured)
    return dict(result)


def bind_output_profile(perf_filename: str, execution_profile: str) -> None:
    """Called under the GPU/output lock; never mix profiles in one table."""
    destination = Path(perf_filename)
    marker = destination.parent / f".{destination.name}.dsv41-profile.json"
    identity = {"execution_profile": execution_profile}
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("V4.1 profiles require separate output tables; select DSV41_EXECUTION_PROFILE")
    marker.write_text(json.dumps(identity))


def run_dsv41_module_worker(
    model_path: str, tp_size: int, execution_profile: str, sweep: dict, *, perf_filename: str, device: str = "cuda:0"
) -> None:
    from importlib.metadata import version as get_version

    from collector.helper import log_perf

    version = get_version("sglang")
    del device  # The native framework owns all ranks in this TP invocation.
    output = Path(perf_filename).parent / f"dsv41-tp{tp_size}-{execution_profile}" / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(tp_size, execution_profile == "decoder_bounded")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    required = ("DSV41_CHECKPOINT", "DSV41_RUNTIME_DIGEST", "DSV41_PROMPT_FILE")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"native V4.1 collection requires {', '.join(missing)}")
    if model_path != "deepseek-ai/DeepSeek-V4.1-Flash":
        raise ValueError("no unverified checkpoint alias is permitted")
    command = [
        sys.executable,
        "-m",
        "collector.sglang.dsv41_native_runner",
        "--manifest",
        str(manifest_path),
        "--runtime-digest",
        os.environ["DSV41_RUNTIME_DIGEST"],
        "--prompt-file",
        os.environ["DSV41_PROMPT_FILE"],
        "--output",
        str(output),
        "--model-path",
        os.environ["DSV41_CHECKPOINT"],
        "--tp-size",
        str(tp_size),
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--moe-a2a-backend",
        "none",
        "--disable-shared-experts-fusion",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--lengths",
        *map(str, sweep["query_lengths"]),
        "--prefixes",
        *map(str, sweep["prefix_lengths"]),
        "--batches",
        *map(str, sweep["batch_sizes"]),
        "--decode-steps",
        str(sweep["decode_steps"]),
    ]
    if execution_profile == "decoder_bounded":
        command.append("--enable-decoder-swa-bounded-replay")
    # The generic executor assigns one worker per GPU. A native TP invocation
    # owns the visible GPU group, so serialize these grouped workers locally.
    with (Path(perf_filename).parent / ".dsv41-native-gpus.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bind_output_profile(perf_filename, execution_profile)
        with (output / "native.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not (output / "COMPLETE").is_file():
        raise RuntimeError("fresh native attempt did not produce completion receipt")
    rows = aggregate_rank_records(sorted(output.glob("rank-*.jsonl")), tp_size)
    for row in rows:
        log_perf(
            [{k: v for k, v in row.items() if k != "kernel_source"}],
            "sglang",
            version,
            "cuda",
            "dsv41_module",
            row["kernel_source"],
            perf_filename,
        )
