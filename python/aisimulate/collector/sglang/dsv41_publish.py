# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish one DeepSeek-V4.1 campaign into a perf-database systems tree.

Inputs are the ADMITTED artifacts the two producers leave behind:

* isolated runs: ``<raw>/`` holding ``plan.json``, ``manifest.json``, ``rank-*.jsonl``,
  ``baseline-rank-*.jsonl`` (+ humming receipts) — one directory per TP;
* attention tables: ``dsv41_attention_runner --admit`` parquet files — one per TP x profile.

Outputs under ``<systems_root>/data/<system>/``:

* ``dsv41/<backend>/<version>/dsv41_module_perf.parquet`` (attention + linear + engram + mhc),
* ``gemm/<backend>/<version>/gemm_perf.parquet`` and ``moe/<backend>/<version>/moe_perf.parquet``
  (the native router / LM-head / expert baselines),
* ``comm/nccl/<nccl_version>/nccl_perf.parquet`` (the native all-reduce baseline),

each with its ``collection_meta.yaml`` (collector/provenance.py design-§5 writer).

A physical key measured by more than one isolated run (mHC and the router GEMM are
TP-independent) is pooled: the per-sample rank maxima of every run are concatenated and the
published latency is their median, ``sample_count`` their total. Rows are never selected by
latency and runs with a different runtime/config/source identity never merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

from .collect_dsv41_module import aggregate_baseline_records
from .dsv41_contract import canonical_json, validate_row, write_parquet

# backend -> producer module holding FRAMEWORK_COMMIT / FRAMEWORK_VERSION / aggregate_isolated_records
PRODUCERS = {
    "sglang": ("collector.sglang.dsv41_isolated_runner", "collector.sglang.dsv41_attention_runner"),
    "vllm": ("collector.vllm.dsv41_isolated_runner", "collector.vllm.dsv41_attention_runner"),
}
IDENTITY = ("source_sha256", "config_sha256", "runtime_digest", "used_cuda_graph", "execution_profile")
BASELINE_COLUMNS = {
    "gemm": ("gemm_dtype", "m", "n", "k"),
    "moe": ("moe_dtype", "num_tokens", "hidden_size", "inter_size", "topk", "num_experts", "moe_tp_size", "moe_ep_size", "distribution"),
    "nccl": ("op_name", "nccl_dtype", "num_gpus", "message_size"),
}
BASELINE_INTEGERS = {"m", "n", "k", "num_tokens", "hidden_size", "inter_size", "topk", "num_experts", "moe_tp_size", "moe_ep_size", "num_gpus", "message_size", "sample_count"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank_maxima(raw: Path, pattern: str, tp: int, key_of, strip: tuple[str, ...]):
    """Per physical key: the row template and the per-sample rank maxima of one run."""
    groups = defaultdict(list)
    for rank in range(tp):
        for line in (raw / pattern.format(rank=rank)).read_text().splitlines():
            row = json.loads(line)
            groups[key_of(row)].append(row)
    out = {}
    for key, rows in groups.items():
        samples = defaultdict(dict)
        for row in rows:
            samples[(row["sample"], row.get("invocation", 0))][row["tp_rank"]] = row["latency"]
        if any(set(s) != set(range(tp)) for s in samples.values()):
            raise ValueError(f"incomplete rank set for {key}")
        template = {k: v for k, v in rows[0].items() if k not in strip}
        out[key] = (template, [max(s.values()) for s in samples.values()])
    return out


def pool_runs(per_run: list[dict], signature_keys: tuple[str, ...]) -> list[dict]:
    """Merge per-run (template, maxima) maps; identical signatures required for a shared key."""
    merged = {}
    for run in per_run:
        for key, (template, maxima) in run.items():
            if key in merged:
                base, pooled = merged[key]
                if any(base[k] != template[k] for k in signature_keys):
                    raise ValueError(f"runs disagree on the measurement identity of {key}")
                pooled.extend(maxima)
            else:
                merged[key] = (dict(template), list(maxima))
    rows = []
    for template, maxima in merged.values():
        row = dict(template)
        row["latency"] = statistics.median(maxima)
        row["sample_count"] = len(maxima)
        rows.append(row)
    return rows


def load_isolated_runs(raw_dirs: list[Path], producer):
    """Validate every isolated run through the producer's own admission, then pool."""
    module_runs, baseline_runs, plans = [], [], []
    for raw in raw_dirs:
        plan_path, manifest_path = raw / "plan.json", raw / "manifest.json"
        plan = json.loads(plan_path.read_text())
        producer.aggregate_isolated_records(raw, plan_path, manifest_path)  # fail-closed admission per run
        aggregate_baseline_records([raw / f"baseline-rank-{r}.jsonl" for r in range(plan["tp_size"])], plan["tp_size"])
        tp = plan["tp_size"]
        strip = ("sample", "invocation", "tp_rank", "case_plan_sha256", "collection_purpose")
        module_runs.append(
            _rank_maxima(raw, "rank-{rank}.jsonl", tp, lambda r: tuple(r[k] for k in ("component", "geometry", "batch_size", "prefix", "x")), strip)
        )
        baseline_runs.append(
            _rank_maxima(
                raw, "baseline-rank-{rank}.jsonl", tp,
                lambda r: (r["kind"], *(r[c] for c in BASELINE_COLUMNS[r["kind"]])),
                ("sample", "tp_rank", "routing_histogram", "routing_seed", "physical_local_intermediate", "case_plan_sha256", "collection_purpose"),
            )
        )
        plans.append((raw, plan, len(module_runs[-1])))
    module_rows = pool_runs(module_runs, (*IDENTITY, "kernel_source"))
    for row in module_rows:
        validate_row(row)
    baseline_rows = pool_runs(baseline_runs, (*IDENTITY, "kernel_source"))
    tables = defaultdict(list)
    for row in baseline_rows:
        kind = row.pop("kind")
        measured = {c: row[c] for c in BASELINE_COLUMNS[kind]}
        if kind == "nccl":
            # raw evidence records physical bytes; NcclOp keys by 16-bit ELEMENTS (collect_dsv41_module.py)
            if measured["nccl_dtype"] not in ("half", "bfloat16") or measured["message_size"] % 2:
                raise ValueError("NCCL baseline requires whole 16-bit elements")
            measured["message_size"] //= 2
            measured["wire_dtype"] = "bfloat16"
        measured.update(latency=row["latency"], kernel_source=row["kernel_source"], sample_count=row["sample_count"])
        tables[kind].append(measured)
    return module_rows, dict(tables), plans


def load_attention_tables(raw_dirs: list[Path]):
    """Admitted attention tables (`--admit` output stored as <raw>/admitted.parquet) with their plans."""
    import pyarrow.parquet as pq

    rows, runs = [], []
    for raw in raw_dirs:
        table = pq.read_table(raw / "admitted.parquet").to_pylist()
        plan = json.loads((raw / "plan.json").read_text())
        for row in table:
            validate_row(row)
            if row["component"] != "attention":
                raise ValueError(f"{raw}: admitted attention table carries {row['component']} rows")
            if row["runtime_digest"] != plan["runtime_digest"] or row["execution_profile"] != plan["execution_profile"]:
                raise ValueError(f"{raw}: admitted rows do not match plan.json")
        rows.extend(table)
        runs.append((raw, plan, len(table)))
    return rows, runs


def write_baseline_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fields = [(k, pa.int64() if k in BASELINE_INTEGERS else pa.float64() if k == "latency" else pa.string()) for k in rows[0]]
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(fields)), path)


def main(argv=None):
    from collector import provenance

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, help="systems entry, e.g. h20_3e")
    parser.add_argument("--backend", default="sglang", choices=sorted(PRODUCERS), help="framework whose producers made the artifacts")
    parser.add_argument("--systems-root", required=True, type=Path, help="…/aisimulate_core/systems")
    parser.add_argument("--isolated", nargs="+", required=True, type=Path, help="isolated raw dirs (plan.json + manifest.json inside)")
    parser.add_argument("--attention", nargs="*", default=[], type=Path, help="attention raw dirs (plan.json + admitted.parquet inside)")
    parser.add_argument("--image", required=True, help="OCI image reference, e.g. lmsysorg/sglang:v0.5.21")
    parser.add_argument("--torch", required=True, help="torch version string inside the image")
    parser.add_argument("--nccl", required=True, help="runtime NCCL version from an ncclGetVersion audit, e.g. 2.30.7")
    parser.add_argument("--collected-at", default=date.today().isoformat())
    parser.add_argument("--overwrite-nccl", action="store_true", help="replace an existing comm/nccl/<ver> table from this campaign")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2], help="python/aisimulate checkout")
    args = parser.parse_args(argv)

    import importlib

    producer = importlib.import_module(PRODUCERS[args.backend][0])
    isolated_module, attention_module = PRODUCERS[args.backend]
    framework_commit, framework_version = producer.FRAMEWORK_COMMIT, producer.FRAMEWORK_VERSION
    module_rows, baselines, isolated_runs = load_isolated_runs(args.isolated, producer)
    attention_rows, attention_runs = load_attention_tables(args.attention)
    all_rows = [*attention_rows, *module_rows]
    if len({tuple(r[k] for k in IDENTITY) for r in all_rows}) != 1:
        raise ValueError("attention tables and isolated runs carry different measurement identities")
    runtime_digest = all_rows[0]["runtime_digest"]
    if {plan["runtime_digest"] for _, plan, _ in (*isolated_runs, *attention_runs)} != {runtime_digest}:
        raise ValueError("plans and rows disagree on the runtime digest")

    closures = provenance.load_closures(args.repo_root / "collector/hash_closures.yaml")
    abi = dict(
        torch=args.torch,
        nccl=args.nccl,
        nccl_identity_evidence="ncclGetVersion audit of the libnccl.so.2 mapped by a torch.distributed all_reduce inside the pinned image",
    )
    runtime = dict(framework=args.backend, version=framework_version, image=args.image, image_digest=runtime_digest, source_commit=framework_commit, abi=abi)

    def event(module, run, rows, runtime_meta):
        raw, plan, _ = run
        return dict(
            collector_ref=plan["collector_revision"],
            collector_hash=provenance.collector_hash(module, args.repo_root, closures),
            case_plan_hash="sha256:" + sha(raw / "plan.json"),
            collected_at=args.collected_at,
            rows=rows,
            status="complete",
            runtime=runtime_meta,
        )

    def publish(dest, table, rows, writer, runtime_meta, events):
        dest.mkdir(parents=True, exist_ok=True)
        writer(rows, dest / f"{table}.parquet")
        provenance.write_collection_meta(dest, runtime_meta, {table: dict(rows=len(rows), status="complete", collections=events)}, provenance_tier="collected")
        print(f"wrote {dest / table}.parquet rows={len(rows)}")

    data = args.systems_root / "data" / args.system
    iso = isolated_module
    publish(
        data / "dsv41" / args.backend / framework_version, "dsv41_module_perf", all_rows, write_parquet, runtime,
        [*(event(iso, run, run[2], runtime) for run in isolated_runs),
         *(event(attention_module, run, run[2], runtime) for run in attention_runs)],
    )
    for kind, table in (("gemm", "gemm_perf"), ("moe", "moe_perf")):
        publish(data / kind / args.backend / framework_version, table, baselines[kind], write_baseline_parquet, runtime,
                [event(iso, run, len(baselines[kind]), runtime) for run in isolated_runs])
    nccl_dest = data / "comm" / "nccl" / args.nccl
    if (nccl_dest / "nccl_perf.parquet").exists() and not args.overwrite_nccl:
        # comm/nccl is keyed by the NCCL library version, not by the framework image: a sibling
        # campaign in another image (same ncclGetVersion) already published this table.
        print(f"kept existing {nccl_dest / 'nccl_perf.parquet'} (same NCCL {args.nccl}; --overwrite-nccl to replace)")
    else:
        nccl_runtime = dict(framework="nccl", version=args.nccl, image=args.image, image_digest=runtime_digest, source_commit=framework_commit, abi=abi)
        publish(nccl_dest, "nccl_perf", baselines["nccl"], write_baseline_parquet, nccl_runtime,
                [event(iso, run, len(baselines["nccl"]), nccl_runtime) for run in isolated_runs])


if __name__ == "__main__":
    main()
