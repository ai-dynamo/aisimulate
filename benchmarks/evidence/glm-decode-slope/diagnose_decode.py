# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query native GLM decode timing; this script contains no timing estimator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.parquet as pq

import aisimulate_core
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_identity(path: Path) -> dict:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "root": git("rev-parse", "--show-toplevel"),
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
    }


def file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def measured_dsa_rows(systems: Path, output: Path) -> dict:
    """Preserve measurements, not an independent interpolation implementation."""
    table = (
        systems
        / "data/b300_sxm/sparse_attention/sglang/0.5.14/dsa_generation_module_perf.parquet"
    )
    rows = [
        row
        for row in pq.read_table(table).to_pylist()
        if row["architecture"] == "GlmMoeDsaForCausalLM"
        and row["num_heads"] == 8
        and row["kv_cache_dtype"] == "fp8"
        and row["gemm_type"] == "bfloat16"
    ]
    with (output / "measured-dsa-rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    domains: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        domains.setdefault((row["op_name"], row["batch_size"]), []).append(
            row["isl"] + row["step"]
        )
    return {
        "table": file_identity(table),
        "filter": {
            "architecture": "GlmMoeDsaForCausalLM",
            "num_heads": 8,
            "kv_cache_dtype": "fp8",
            "gemm_type": "bfloat16",
        },
        "rows": len(rows),
        "domains": [
            {
                "op_name": op,
                "batch_size": batch,
                "min_sequence_tokens": min(points),
                "max_sequence_tokens": max(points),
                "measured_coordinates": sorted(set(points)),
            }
            for (op, batch), points in sorted(domains.items())
        ],
    }


def model_variants(
    model_dir: Path, output: Path, variants: list[str]
) -> tuple[dict, list]:
    source = model_dir / "nvidia--GLM-5.3-NVFP4_config.json"
    quant = model_dir / "nvidia--GLM-5.3-NVFP4_hf_quant_config.json"
    identities = {"checkpoint": "nvidia/GLM-5.3-NVFP4"}
    evidence = []
    for variant in variants:
        if variant == "checkpoint":
            continue
        root = output / "synthetic-models" / variant
        root.mkdir(parents=True)
        raw = json.loads(source.read_text())
        pattern = ("F" if variant == "all_full" else "S") * raw["num_hidden_layers"]
        previous = raw["index_topk_pattern"]
        raw["index_topk_pattern"] = pattern
        write_json(root / "config.json", raw)
        (root / "hf_quant_config.json").write_bytes(quant.read_bytes())
        identities[variant] = str(root.resolve())
        evidence.append(
            {
                "variant": variant,
                "kind": "synthetic_layer_reuse_ablation",
                "source": file_identity(source),
                "quantization_source": file_identity(quant),
                "changes": {
                    "index_topk_pattern": {"before": previous, "after": pattern}
                },
                "files": [
                    file_identity(root / "config.json"),
                    file_identity(root / "hf_quant_config.json"),
                ],
            }
        )
    return identities, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--label",
        required=True,
        help="Human-readable run label; not a verified native revision",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[32768, 65536, 131072, 262144, 393216, 524288, 600000],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 13, 14])
    parser.add_argument("--nextn", nargs="+", type=int, default=[0, 3])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["checkpoint", "all_full", "all_skip"],
        default=["checkpoint", "all_full", "all_skip"],
    )
    parser.add_argument(
        "--expected-native-sha256",
        help="Fail if the imported native extension has a different digest",
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.contexts + args.batches) or any(
        value < 0 for value in args.nextn
    ):
        parser.error("contexts/batches must be positive and nextn must be nonnegative")
    if args.output_dir.exists():
        parser.error("output directory must be new")

    native = importlib.import_module("aisimulate._runtime")
    native_path = Path(native.__file__).resolve()
    native_identity = file_identity(native_path)
    if (
        args.expected_native_sha256
        and native_identity["sha256"] != args.expected_native_sha256
    ):
        parser.error("native extension SHA256 does not match --expected-native-sha256")
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    core_root = Path(aisimulate_core.__file__).resolve().parent
    systems = core_root / "systems"
    model_dir = core_root / "model_configs"
    data_files = sorted(
        path for path in (systems / "data/b300_sxm").rglob("*") if path.is_file()
    )
    data_files += sorted(path for path in systems.glob("*") if path.is_file())
    model_files = sorted(model_dir.glob("nvidia--GLM-5.3-NVFP4*.json"))
    manifest = {
        "schema_version": 1,
        "label": args.label,
        "status": "running",
        "units": {"latency": "milliseconds", "context": "tokens"},
        "scope": "Static native op-level decode queries; no scheduler, cache lookup, or GPU measurement",
        "semantics": {
            "context_length": "Past context; native decode query evaluates context_length + 1",
            "cycle_latency_ms": "One configured native generation timing round, including MTP composition",
            "full_skip_attribution": "Checkpoint uses native blended DSA; all_full/all_skip change only index_topk_pattern in synthetic copies",
            "measured_dsa_table": "Explicit baseline candidate table; presence is not proof of execution at every query",
            "acceptance": "No accepted-token conversion; cycle latency is not TPOT",
        },
        "python": sys.version,
        "executable": sys.executable,
        "native": native_identity,
        "script": file_identity(Path(__file__)),
        "source_checkout": git_identity(core_root),
        "script_checkout": git_identity(Path(__file__).parent),
        "model_files": [file_identity(path) for path in model_files],
        "data_files": [file_identity(path) for path in data_files],
        "grid": {
            "contexts": args.contexts,
            "batches": args.batches,
            "nextn": args.nextn,
            "variants": args.variants,
        },
        "models": [],
        "record_count": 0,
    }
    identities, variants = model_variants(model_dir, args.output_dir, args.variants)
    manifest["synthetic_model_variants"] = variants
    manifest["measured_dsa"] = measured_dsa_rows(systems, args.output_dir)
    write_json(args.output_dir / "manifest.json", manifest)
    summary = []
    non_attention = {}
    with (args.output_dir / "native-queries.jsonl").open("w") as stream:
        for variant, nextn in product(args.variants, args.nextn):
            config = ForwardPassPerfModelConfig(
                model=identities[variant],
                system="b300_sxm",
                backend="sglang",
                backend_version="0.5.20",
                worker_type="aggregated",
                tp=8,
                attention_dp=1,
                moe_tp_size=8,
                moe_ep_size=1,
                nextn=nextn,
                kv_block_size=64,
                estimation_mode="op_level",
                fallback_policy="deny",
                database_mode="SILICON",
                systems_paths=(str(systems),),
                estimator_config={"correction": {"enabled": False}},
            )
            model = RustForwardPassPerfModel.best_available(config)
            manifest["models"].append(
                {
                    "variant": variant,
                    "nextn": nextn,
                    "requested_config": config.to_dict(),
                    "diagnostics": model.diagnostics(),
                    "topology": model.deployment_topology(),
                }
            )
            for batch in args.batches:
                for context in args.contexts:
                    ops = model.static_phase_diagnostics(
                        batch_size=batch, context_length=context, prefill=False
                    )
                    latency = model.static_phase_latency(
                        batch_size=batch,
                        input_tokens=context,
                        output_tokens=2,
                        prefill=False,
                    )
                    op_sum = sum(op["latency_ms"] for op in ops)
                    if not math.isclose(latency, op_sum, rel_tol=1e-10, abs_tol=1e-10):
                        raise RuntimeError(
                            f"native scalar/op decomposition differs: {latency} versus {op_sum}"
                        )
                    attention = sum(
                        op["latency_ms"]
                        for op in ops
                        if op["name"] == "generation_attention"
                    )
                    row = {
                        "variant": variant,
                        "nextn": nextn,
                        "batch_size": batch,
                        "context_length": context,
                        "cycle_latency_ms": latency,
                        "attention_latency_ms": attention,
                        "other_ops_latency_ms": op_sum - attention,
                    }
                    stream.write(
                        json.dumps({**row, "operations": ops}, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    summary.append(row)
                    non_attention[variant, nextn, batch, context] = {
                        op["name"]: op["latency_ms"]
                        for op in ops
                        if op["name"] != "generation_attention"
                    }
            print(
                f"finished {variant} nextn={nextn}: {len(summary)} native queries",
                flush=True,
            )
    comparisons = 0
    for (variant, nextn, batch, context), ops in non_attention.items():
        baseline = non_attention.get(("checkpoint", nextn, batch, context))
        if variant != "checkpoint" and baseline is not None:
            if ops != baseline:
                raise RuntimeError(
                    f"synthetic {variant} changed non-attention operations at {(nextn, batch, context)}"
                )
            comparisons += 1
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest.update(
        status="completed",
        record_count=len(summary),
        wall_seconds=time.perf_counter() - start,
        non_attention_ablation_comparisons=comparisons,
    )
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
