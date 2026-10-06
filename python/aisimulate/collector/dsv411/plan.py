# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a dsv411 collection plan + SDK manifest (CPU).

    python -m collector.dsv411.plan --backend sglang --tp 2 --purpose smoke \
        --model-path /work/ckpt-meta --pins sglang_0521_pins.txt --image-digest <sha256 hex> \
        --collector-revision <git sha> --components attention_core indexer engram mhc shared_linear \
        --overrides '{"context": {"query_lengths": [128, 2048], ...}}' --out plans/sglang-tp2-smoke

Writes ``<out>/plan.json`` and ``<out>/manifest.json``; the producers refuse anything that
differs from these frozen files (plan sha is carried by every raw row).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path

from . import contract

PRODUCERS = {"sglang": "collector.sglang.collect_dsv411_module", "vllm": "collector.vllm.collect_dsv411_module"}
DEFAULT_POOL = {
    # SGLang ModelRunner pool / vLLM config limits; sized for the full grid (1M past kv, batch 1024).
    # sglang: max_total_tokens must hold the largest resident context case, batch 8 x (1048575 + 1) =
    # 8,388,608 full tokens (plus one chunk of query tokens); on V4.1-Flash a full token costs ~1.6 KB
    # (kv_source layers 2/8/14 at ratio 2, 20 at ratio 1, FlashMLA 584 B rows + fp4 indexer keys), so the
    # pool is ~14 GB. The SWA pool (40 layers x 584 B per slot) only holds the slid windows plus one
    # extend's new tokens: max_new_tokens 262,144 + 1024 x (128 window + 64 page) < 8650752 x 1/16.
    "sglang": dict(
        context_length=1048576,
        max_total_tokens=8650752,
        max_requests=1024,
        mem_fraction_static=0.85,
        max_new_tokens=262144,
        swa_full_tokens_ratio=0.0625,
    ),
    "vllm": dict(context_length=1048576, max_requests=1024, max_new_tokens=262144),
}


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r"([0-9a-f]{64})\s+(\S+\.py)$", line.strip())
        if m:
            pins[m.group(2)] = m.group(1)
    if not pins:
        raise SystemExit(f"{path}: no '<sha256> <relative/path.py>' lines")
    return pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=sorted(PRODUCERS))
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--purpose", required=True, choices=contract.PURPOSES)
    parser.add_argument(
        "--model-path", type=Path, required=True, help="checkpoint metadata dir (config.json, tokenizer*)"
    )
    parser.add_argument(
        "--pins", type=Path, required=True, help="'<sha256> <path.py>' lines of the installed framework package"
    )
    parser.add_argument("--image-digest", required=True, help="OCI image sha256 hex (64 chars)")
    parser.add_argument("--collector-revision", required=True)
    parser.add_argument("--components", nargs="+", default=list(contract.STRUCTURE_FIELDS))
    parser.add_argument(
        "--overrides", default=None, help="JSON: per-section axis overrides of the grid (smoke subsets)"
    )
    parser.add_argument("--grid", type=Path, default=contract.GRID_PATH)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--expected-gpu", default="H20")
    parser.add_argument(
        "--moe-backend", default=None, help="sglang moe_runner_backend (default flashinfer_mxfp4 tp2 / humming tp4)"
    )
    parser.add_argument("--pool", default=None, help="JSON overriding the default pool limits")
    parser.add_argument(
        "--kv-seed",
        choices=list(contract.KV_SEED_REGIMES),
        default="synthetic_kv",
        help="how cached-prefill / decode KV is seeded (stored per row): synthetic_kv = serving allocation "
        "bookkeeping + bounded random cache contents (default); real_kv = chunked prefills of corpus tokens",
    )
    parser.add_argument(
        "--regime-exception",
        action="append",
        default=[],
        help="component=reason: declare an eager generation exception",
    )
    parser.add_argument("--shard", default=None, help="index/count: run only every count-th case starting at index")
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="JSON list of case ids (or a receipt whose failed_cases name them): keep only these expanded cases - "
        "a complement plan for cells another run recorded as failed (run it in its own output dir; pooled at publish)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    producer = importlib.import_module(PRODUCERS[args.backend])
    manifest = contract.build_manifest(args.tp, args.backend)
    grid = contract.load_grid(args.grid)
    overrides = json.loads(args.overrides) if args.overrides else None
    cases, drops = contract.expand_cases(grid, args.components, overrides=overrides)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    cases = contract.shard_cases(cases, shard)
    if args.cases is not None:
        selected = json.loads(args.cases.read_text())
        if isinstance(selected, dict):
            selected = sorted(selected.get("failed_cases") or {})
        wanted = set(selected)
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing or not cases:
            raise SystemExit(f"--cases: not in the expanded grid/shard: {sorted(missing)[:5]} (selected {len(wanted)})")
    pool = dict(DEFAULT_POOL[args.backend])
    if args.pool:
        pool.update(json.loads(args.pool))
    pins = read_pins(args.pins)
    missing = set(producer.REQUIRED_SOURCES) - set(pins)
    if missing:
        raise SystemExit(f"pins file lacks required sources: {sorted(missing)}")
    metadata_pins = {
        n: hashlib.sha256((args.model_path / n).read_bytes()).hexdigest()
        for n in ("config.json", "tokenizer.json", "tokenizer_config.json")
    }
    plan = dict(
        schema=contract.PLAN_SCHEMA,
        purpose=args.purpose,
        backend=args.backend,
        tp_size=args.tp,
        components=list(args.components),
        cases=cases,
        case_drops=drops,
        grid_sha256=contract.sha256_file(args.grid),
        grid_overrides=overrides,
        shard=list(shard) if shard else None,
        chunk_prefill_size=grid["context"]["chunk_prefill_size"],
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        regimes=dict(grid["regimes"]),
        kv_seed_regime=args.kv_seed,
        regime_exceptions=[
            dict(component=item.split("=", 1)[0], phase="generation", reason=item.split("=", 1)[1])
            for item in args.regime_exception
        ],
        expected_gpu=args.expected_gpu,
        expected_sm=producer.EXPECTED_SM[args.expected_gpu],
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        runtime_digest="sha256:" + args.image_digest,
        image_sha256=args.image_digest,
        source_pins={k: pins[k] for k in sorted(producer.REQUIRED_SOURCES)},
        metadata_pins=metadata_pins,
        collector_revision=args.collector_revision,
        weight_initializer=producer.WEIGHT_INITIALIZER,
        pool=pool,
    )
    if args.backend == "sglang":
        plan["moe_runner_backend"] = args.moe_backend or ("humming" if args.tp == 4 else "flashinfer_mxfp4")
    else:
        plan["moe_backend"] = args.moe_backend or "marlin"
    contract.validate_plan(
        plan,
        manifest,
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        expected_sm=producer.EXPECTED_SM,
        required_sources=producer.REQUIRED_SOURCES,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    keys = contract.expected_keys(plan, manifest)
    print(
        json.dumps(
            dict(
                out=str(args.out),
                cases=len(cases),
                drops=drops,
                physical_keys=len(keys),
                representatives={p: {c: len(s) for c, s in r.items()} for p, r in manifest["representatives"].items()},
            )
        )
    )


if __name__ == "__main__":
    main()
