#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity gate: op database loaded from the Hugging Face layout versus the packaged database (AISIM-2063).

Local and offline: reads a locally built publish tree (``publish_op_latency_tree.py build``), never uploads,
never changes a default. Run::

    python tools/perf_database/op_database_parity.py run --tree /tmp/op-tree --workdir /tmp/parity --json parity.json

Checks (any failure makes the exit code 1):

1. ``rows``: every packaged ``systems/data`` file has a loaded counterpart with the same bytes (copied files),
   the same Arrow row hash and column types (op tables), and no loaded file is unaccounted for.
2. ``lookup``: the rebuilt ``perf_data_reuse_manifest.yaml`` / ``query_versions.yaml`` are byte-identical.
3. ``predictions``: per-op latency, source, fallbacks and totals from ``static_phase_latency`` /
   ``static_phase_diagnostics`` (op-level estimator, the production ``best_available`` path) are bit-identical for
   a representative model/system/backend set. Both runs happen in fresh processes with sockets blocked.
4. ``offline``: the pinned ``ensure_data`` path with ``offline=True`` makes no network call, and an offline
   Hugging Face pin with an empty cache fails without falling back to packaged data.
5. ``startup``: process wall time, build time, first-prediction time and peak RSS, median of ``--repeats`` fresh
   processes per source, with the one-time load (verify + materialize) cost reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PACKAGED_NAME = "packaged"
TREE_NAME = "hf-layout"
DEFAULT_ROOT = "default"  # worker root meaning "no systems_paths, no pin": the packaged path a rollback returns to

# (model, system, backend, extra identity). Only bundled model configs, so no network is needed to resolve a model.
SCENARIOS: list[dict] = [
    dict(model="Qwen/Qwen3-32B", system="h200_sxm", backend="vllm", tp=2),
    dict(model="Qwen/Qwen3-32B", system="h100_sxm", backend="trtllm", tp=2),
    dict(model="Qwen/Qwen3-32B", system="b200_sxm", backend="sglang", tp=2),
    dict(model="Qwen/Qwen3-32B", system="gb300", backend="trtllm", tp=2),
    dict(model="Qwen/Qwen3-32B", system="l40s", backend="vllm", tp=4),
    dict(model="meta-llama/Meta-Llama-3.1-8B", system="a100_sxm", backend="trtllm", tp=1),
    dict(model="meta-llama/Meta-Llama-3.1-70B", system="h100_sxm", backend="vllm", tp=4),
    dict(model="Qwen/Qwen3-30B-A3B", system="h200_sxm", backend="vllm", tp=2, moe_tp_size=2, moe_ep_size=1),
    dict(model="Qwen/Qwen3-235B-A22B", system="b200_sxm", backend="trtllm", tp=8, moe_tp_size=1, moe_ep_size=8),
    dict(
        model="deepseek-ai/DeepSeek-V3",
        system="b200_sxm",
        backend="sglang",
        tp=1,
        attention_dp=8,
        moe_tp_size=1,
        moe_ep_size=8,
    ),
    dict(
        model="deepseek-ai/DeepSeek-V3",
        system="h200_sxm",
        backend="trtllm",
        tp=1,
        attention_dp=8,
        moe_tp_size=1,
        moe_ep_size=8,
    ),
]
PREFILL_SHAPES = [(1, 128), (8, 1024), (32, 4096)]  # (batch size, input tokens)
DECODE_SHAPES = [(1, 1024), (16, 2048), (128, 8192)]  # (batch size, total sequence length)
STARTUP_SCENARIO = SCENARIOS[0]


# --------------------------------------------------------------------------- worker (fresh process)


def _block_network() -> None:
    def refuse(*_a, **_k):
        raise OSError("network blocked by the parity harness")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def _ops(diagnostics: list[dict]) -> list[list]:
    return [
        [d["name"], d["latency_ms"], d.get("energy_wms"), d["source"], d.get("details", {}).get("fallbacks")]
        for d in diagnostics
    ]


def _packaged_systems() -> Path:
    from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

    return PACKAGED_SYSTEMS_DIR


def _peak_rss_mib() -> float:
    """Peak resident set of this process (``VmHWM``; ``ru_maxrss`` would inherit the parent's peak across exec)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    _block_network()
    from aisim_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    imported = time.perf_counter()
    scenarios = [STARTUP_SCENARIO] if args.startup else SCENARIOS
    out: dict = {"root": args.root, "imports_s": imported - started, "scenarios": []}
    for scenario in scenarios:
        entry: dict = {"scenario": scenario, "build_s": None, "error": None, "cases": {}}
        t0 = time.perf_counter()
        try:
            config = ForwardPassPerfModelConfig(
                worker_type="aggregated",
                estimation_mode="op_level",
                fallback_policy="deny",
                systems_paths=() if args.root == DEFAULT_ROOT else (args.root,),
                **scenario,
            )
            model = RustForwardPassPerfModel.best_available(config)
            entry["build_s"] = time.perf_counter() - t0
            entry["selected_root"] = model.diagnostics()["provenance"]["selected_systems_root"]
        except Exception as exc:  # recorded, and compared between the two sources
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            out["scenarios"].append(entry)
            continue
        try:
            first = None
            for phase, shapes in (("prefill", PREFILL_SHAPES), ("decode", DECODE_SHAPES)):
                for batch, length in shapes:
                    key = f"{phase}:b{batch}:l{length}"
                    try:
                        t1 = time.perf_counter()
                        if phase == "prefill":
                            total = model.static_phase_latency(
                                batch_size=batch, input_tokens=length, output_tokens=1, prefill=True
                            )
                        else:
                            total = model.static_phase_latency(
                                batch_size=batch, input_tokens=length, output_tokens=2, prefill=False
                            )
                        first = first if first is not None else time.perf_counter() - t1
                        entry["cases"][key] = {"total_ms": total}
                        if not args.startup:
                            diag = model.static_phase_diagnostics(
                                batch_size=batch, context_length=length, prefix=0, prefill=phase == "prefill"
                            )
                            entry["cases"][key]["ops"] = _ops(diag)
                    except Exception as exc:
                        entry["cases"][key] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    if args.startup:
                        break
                if args.startup:
                    break
            entry["first_prediction_s"] = first
        finally:
            model.close()
        out["scenarios"].append(entry)
    out["total_s"] = time.perf_counter() - started
    out["peak_rss_mib"] = _peak_rss_mib()
    Path(args.out).write_text(json.dumps(out, sort_keys=True))
    return 0


def _run_worker(root: Path | str, out: Path, *, startup: bool, hub_offline: bool = True) -> tuple[dict, float]:
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    env.pop("AICONFIGURATOR_SYSTEMS_PATH", None)
    if hub_offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AISIM_DATA_OFFLINE="1")
    cmd = [sys.executable, __file__, "worker", "--root", str(root), "--out", str(out)]
    if startup:
        cmd.append("--startup")
    t0 = time.perf_counter()
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    if done.returncode != 0:
        raise RuntimeError(f"worker failed for {root}: {done.stderr[-2000:]}")
    return json.loads(out.read_text()), wall


# --------------------------------------------------------------------------- comparisons


def compare_files(packaged: Path, loaded: Path, tree: Path, runtime_paths: frozenset[str]) -> dict:
    """Row/hash equality of every packaged file against the loaded root (``data/`` only).

    Files outside every version manifest (system-level README, collection reports, upstream power provenance) are
    published but are not runtime inputs, so the loaded root does not carry them; each is instead checked against
    the tree itself (same SHA-256 as the packaged file at its published path). A path a manifest declares as a
    runtime input is never excused this way.
    """
    import pyarrow.parquet as pq

    from aisim_core.data.op_schema import DATA_PREFIX, SKIPPED_NAMES, file_sha256, table_content_sha256

    result = {
        "packaged_files": 0,
        "skipped_placeholders": 0,
        "not_runtime_published": [],
        "byte_equal": 0,
        "row_equal_tables": 0,
        "rows": 0,
        "problems": [],
    }
    seen = set()
    for source in sorted(p for p in (packaged / "data").rglob("*") if p.is_file()):
        rel = source.relative_to(packaged).as_posix()
        result["packaged_files"] += 1
        if source.name in SKIPPED_NAMES:
            result["skipped_placeholders"] += 1
            continue
        target = loaded / rel
        seen.add(rel)
        if not target.is_file():
            published = tree / DATA_PREFIX / rel.removeprefix("data/")
            if rel not in runtime_paths and published.is_file() and file_sha256(published) == file_sha256(source):
                result["not_runtime_published"].append(rel)
            else:
                result["problems"].append(f"missing in loaded root and not published byte-identical: {rel}")
            continue
        if source.suffix == ".parquet":
            a, b = pq.read_table(source), pq.read_table(target)
            same_types = [(f.name, str(f.type)) for f in a.schema] == [(f.name, str(f.type)) for f in b.schema]
            if table_content_sha256(a) != table_content_sha256(b) or a.num_rows != b.num_rows or not same_types:
                result["problems"].append(f"rows differ: {rel}")
            else:
                result["row_equal_tables"] += 1
                result["rows"] += a.num_rows
        elif file_sha256(source) != file_sha256(target):
            result["problems"].append(f"bytes differ: {rel}")
        else:
            result["byte_equal"] += 1
    for target in sorted(p for p in (loaded / "data").rglob("*") if p.is_file()):
        rel = target.relative_to(loaded).as_posix()
        if rel not in seen:
            result["problems"].append(f"unaccounted for in loaded root: {rel}")
    result["ok"] = not result["problems"]
    return result


def compare_lookup(packaged: Path, loaded: Path) -> dict:
    from aisim_core.data.op_schema import file_sha256

    names = ("perf_data_reuse_manifest.yaml", "query_versions.yaml")
    rows = {n: (loaded / n).is_file() and file_sha256(packaged / n) == file_sha256(loaded / n) for n in names}
    return {"byte_identical": rows, "ok": all(rows.values())}


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def compare_predictions(packaged: dict, loaded: dict) -> dict:
    report = {"scenarios": [], "compared_values": 0, "mismatches": [], "max_abs_diff_ms": 0.0, "max_rel_diff": 0.0}
    for left, right in zip(packaged["scenarios"], loaded["scenarios"], strict=True):
        name = "{model} {system} {backend}".format(**left["scenario"])
        row = {"name": name, "scenario": left["scenario"], "build_error": left["error"], "cases": 0, "ops": 0}
        if left["error"] != right["error"]:
            report["mismatches"].append(f"{name}: build outcome differs: {left['error']!r} vs {right['error']!r}")
        row["usable"] = left["error"] is None
        for key, a in left["cases"].items():
            b = right["cases"].get(key)
            report["compared_values"] += 1
            row["cases"] += 1
            if "total_ms" in a and b and "total_ms" in b:
                diff = abs(a["total_ms"] - b["total_ms"])
                report["max_abs_diff_ms"] = max(report["max_abs_diff_ms"], diff)
                if a["total_ms"]:
                    report["max_rel_diff"] = max(report["max_rel_diff"], diff / abs(a["total_ms"]))
            row["ops"] += len(a.get("ops", []))
            report["compared_values"] += len(a.get("ops", []))
            if not _same(a, b):
                report["mismatches"].append(f"{name} {key}")
        report["scenarios"].append(row)
    report["usable_scenarios"] = sum(1 for r in report["scenarios"] if r["usable"])
    report["ok"] = not report["mismatches"] and report["usable_scenarios"] > 0
    return report


# --------------------------------------------------------------------------- offline


def check_offline(tree: Path, scratch: Path) -> dict:
    """The pinned loading path makes no network call and never substitutes packaged data."""
    from aisim_core.data.ensure import DataConflictError, DataPin, MissingDataError, ensure_data
    from aisim_core.data.op_tree import op_latency_entries

    entries = op_latency_entries(tree, hardware=["h200_sxm"], framework=["vllm"])
    entry = sorted(entries, key=lambda e: e.manifest.path)[0]
    identity = entry.identity
    config = {
        "system": identity.hardware,
        "backend": identity.framework,
        "backend_version": identity.framework_version,
        "operation_family": identity.operation_family,
    }
    original = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
    _block_network()
    out: dict = {}
    try:
        ensured = ensure_data(
            config, DataPin(local_root=tree, categories=("op-latency",), offline=True, cache_dir=scratch / "cache")
        )
        out["local_pin_single_entry"] = {
            "ok": True,
            "configuration_id": ensured.datasets["op-latency"]["configuration_id"],
            "systems_paths": ensured.config["systems_paths"],
        }
        try:
            ensure_data(
                {k: v for k, v in config.items() if k != "operation_family"},
                DataPin(local_root=tree, categories=("op-latency",), offline=True, cache_dir=scratch / "cache"),
            )
            out["local_pin_without_family"] = {"ok": False, "detail": "expected a conflict"}
        except DataConflictError as exc:
            out["local_pin_without_family"] = {"ok": True, "detail": "DataConflictError: " + str(exc)[:120]}
        before = dict(config)
        try:
            ensure_data(
                before,
                DataPin(
                    repo="nvidia/aisim-data",
                    revision="0" * 40,
                    categories=("op-latency",),
                    offline=True,
                    cache_dir=scratch / "empty-cache",
                ),
            )
            out["hf_pin_offline_empty_cache"] = {"ok": False, "detail": "expected MissingDataError"}
        except MissingDataError as exc:
            out["hf_pin_offline_empty_cache"] = {
                "ok": "systems_paths" not in before,
                "detail": "MissingDataError (no packaged fallback): " + str(exc)[:140],
            }
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = original
    out["ok"] = all(v["ok"] for v in out.values())
    return out


# --------------------------------------------------------------------------- orchestration


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run(args: argparse.Namespace) -> int:
    from aisim_core.data.op_tree import load_op_latency_tree

    packaged = _packaged_systems()
    tree, workdir = Path(args.tree).absolute(), Path(args.workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    loaded_root = workdir / "loaded-systems"
    if loaded_root.exists():
        shutil.rmtree(loaded_root)
    load = load_op_latency_tree(tree, loaded_root)
    report: dict = {
        "tree": str(tree),
        "schema": "aisim-op-parity-report/v1",
        "tree_catalog_sha256": _sha256(tree / "catalog" / "dataset.json"),
        "tree_op_index_sha256": _sha256(tree / "catalog" / "op-latency.json"),
        "packaged_systems": str(packaged),
        "load": {
            "entries": load.entries,
            "files": load.files,
            "bytes": load.bytes,
            "lookup_files": list(load.lookup_files),
            "seconds": load.seconds,
            "root_bytes": _du(loaded_root),
        },
    }
    report["rows"] = compare_files(packaged, loaded_root, tree, load.runtime_paths)
    report["lookup"] = compare_lookup(packaged, loaded_root)

    packaged_run, _ = _run_worker(packaged, workdir / "pred-packaged.json", startup=False)
    tree_run, _ = _run_worker(loaded_root, workdir / "pred-tree.json", startup=False)
    report["predictions"] = compare_predictions(packaged_run, tree_run)
    used = {s.get("selected_root") for s in tree_run["scenarios"] if s.get("selected_root")}
    report["predictions"]["tree_run_selected_roots"] = sorted(used)
    report["predictions"]["tree_run_used_loaded_root_only"] = used == {str(loaded_root)}
    report["predictions"]["ok"] = (
        report["predictions"]["ok"] and report["predictions"]["tree_run_used_loaded_root_only"]
    )

    report["offline"] = check_offline(tree, workdir / "offline")

    startup: dict = {"repeats": args.repeats, "scenario": STARTUP_SCENARIO}
    for name, root in ((PACKAGED_NAME, packaged), (TREE_NAME, loaded_root)):
        runs = []
        for i in range(args.repeats):
            data, wall = _run_worker(root, workdir / f"startup-{name}-{i}.json", startup=True)
            s = data["scenarios"][0]
            runs.append(
                {
                    "wall_s": wall,
                    "imports_s": data["imports_s"],
                    "build_s": s["build_s"],
                    "first_prediction_s": s["first_prediction_s"],
                    "peak_rss_mib": data["peak_rss_mib"],
                }
            )
        startup[name] = {k: statistics.median(r[k] for r in runs) for k in runs[0]} | {"runs": runs}
    for key in ("wall_s", "build_s", "first_prediction_s", "peak_rss_mib"):
        base, new = startup[PACKAGED_NAME][key], startup[TREE_NAME][key]
        startup.setdefault("delta", {})[key] = {"abs": new - base, "pct": 100.0 * (new - base) / base if base else None}
    report["startup"] = startup
    # Full sweep memory/time for both sources (all scenarios in one process).
    report["sweep"] = {
        PACKAGED_NAME: {"total_s": packaged_run["total_s"], "peak_rss_mib": packaged_run["peak_rss_mib"]},
        TREE_NAME: {"total_s": tree_run["total_s"], "peak_rss_mib": tree_run["peak_rss_mib"]},
    }
    report["gates"] = {
        "rows": report["rows"]["ok"],
        "lookup": report["lookup"]["ok"],
        "predictions": report["predictions"]["ok"],
        "offline": report["offline"]["ok"],
    }
    report["ok"] = all(report["gates"].values())
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if not args.keep:
        shutil.rmtree(loaded_root, ignore_errors=True)
    return 0 if report["ok"] else 1


def drill(args: argparse.Namespace) -> int:
    """Rollback drill (local, no production state touched): pinned op data, then the pin removed.

    Predictions with the HF-layout root and with no pin at all (the packaged default a rollback returns to)
    must be bit-identical, and every pin variable must be inert when unset.
    """
    from aisim_core.data.ensure import ensure_data, pin_from_environment
    from aisim_core.data.op_tree import load_op_latency_tree

    packaged = _packaged_systems()
    tree, workdir = Path(args.tree).absolute(), Path(args.workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    root = workdir / "drill-systems"
    if root.exists():
        shutil.rmtree(root)
    load_op_latency_tree(tree, root)
    pin_vars = {"AISIM_DATASET": "ai-dynamo/aisim-data@" + "a" * 40, "AISIM_DATA_CATEGORIES": "fpm,op-latency"}
    config = {"system": "h200_sxm", "backend": "vllm", "backend_version": "0.24.0"}
    checks = {
        "pin_set_builds_a_pin": pin_from_environment(pin_vars) is not None,
        "pin_unset_is_no_pin": pin_from_environment({}) is None,
        "no_pin_leaves_config_unchanged": ensure_data(config, None).config == config,
    }
    pinned, _ = _run_worker(root, workdir / "drill-pinned.json", startup=False)
    rolled_back, _ = _run_worker(DEFAULT_ROOT, workdir / "drill-rolled-back.json", startup=False)
    predictions = compare_predictions(rolled_back, pinned)
    checks["rolled_back_predictions_identical"] = predictions["ok"]
    default_roots = {s.get("selected_root") for s in rolled_back["scenarios"] if s.get("selected_root")}
    checks["rolled_back_uses_packaged_root"] = default_roots == {str(packaged)}
    report = {
        "schema": "aisim-op-rollback-drill/v1",
        "checks": checks,
        "compared_values": predictions["compared_values"],
        "max_abs_diff_ms": predictions["max_abs_diff_ms"],
        "ok": all(checks.values()),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    shutil.rmtree(root, ignore_errors=True)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="load the tree, compare with the packaged database, measure")
    r.add_argument("--tree", type=Path, required=True, help="local op-latency publish tree")
    r.add_argument("--workdir", type=Path, default=None)
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--json", type=Path)
    r.add_argument("--keep", action="store_true", help="keep the loaded systems root")
    d = sub.add_parser("drill", help="rollback drill: pinned op data versus the packaged default")
    d.add_argument("--tree", type=Path, required=True)
    d.add_argument("--workdir", type=Path, required=True)
    d.add_argument("--json", type=Path)
    w = sub.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("--root", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--startup", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "worker":
        return _worker(args)
    if args.command == "drill":
        return drill(args)
    if args.workdir is None:
        args.workdir = Path(tempfile.mkdtemp(prefix="op-parity-"))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
