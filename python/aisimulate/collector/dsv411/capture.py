# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-cell, single-process (TP1) dsv411 producer runs for opharness ``path_diff.py --capture``.

A capture script under ``collector/opharness/components/captures/`` is one line::

    from collector.dsv411.capture import run_cell; run_cell("vllm", "attention", "context")

It freezes a one-case plan in place (purpose ``smoke``, source pins hashed from the installed
framework package, metadata pins from the checkpoint directory), loads the SDK manifest the host
exported into the probe workspace (``facts/manifests/dsv411_<backend>_tp1.json``, written by
``e2e_align.py --sdk-manifest``; the framework image has no SDK), and runs the producer's ``run()``
in this process with a torchrun-equivalent single-rank environment so the capturing profiler sees
every kernel. The cells mirror the serving probe (isl 4096 prefill; one decode token after it).

Environment: ``DSV411_MODEL_PATH`` (checkpoint metadata dir), ``DSV411_PROMPT_FILE`` (corpus),
``AIS_PROBE_WORKSPACE`` (manifests), optional ``DSV411_LAUNCH_IMAGE_SHA256`` / ``DSV411_COLLECTOR_REVISION``
(recorded as unverified placeholders when absent — a capture is kernel-path evidence, never perf data),
optional ``DSV411_PRIVATE_CACHE`` (defaults to a scratch root whose home-cache is this user's ~/.cache),
optional ``DSV411_CAPTURE_ENGRAM_ROWS`` (engram cells only: a TP1 rank cannot hold the 384M-row fp8 hash
tables on 140 GB, so the capture patches ``engram_num_embeddings`` in a scratch copy of the checkpoint
config AND in the manifest's engram structures — memory only, the lookup / wkv / gate kernels are the same —
and records the override in the plan).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import socket
import tempfile
import time
import traceback
from pathlib import Path

from . import contract
from .plan import PRODUCERS

CELLS = {
    # kind -> (components, case builder)
    "attention": (["attention_core", "indexer"], None),
    "engram": (["engram"], None),
    "mhc": (["mhc"], None),
    "shared_linear": (["shared_linear"], None),
}
PROBE_ISL = 4096
POOL = {
    "sglang": dict(
        context_length=16384,
        max_total_tokens=65536,
        max_requests=1,
        mem_fraction_static=0.8,
        max_new_tokens=8192,
        swa_full_tokens_ratio=1.0,
    ),
    "vllm": dict(context_length=16384, max_requests=1, max_new_tokens=8192),
}


def _case(kind: str, phase: str) -> dict:
    if kind == "attention":
        if phase == "context":
            case = dict(kind="attention", phase="context", batch_size=1, query=PROBE_ISL, past_kv=0)
        else:
            case = dict(kind="attention", phase="generation", batch_size=1, query=1, past_kv=PROBE_ISL)
        case["case_id"] = f"{phase}-b1-q{case['query']}-kv{case['past_kv']}"
    else:
        tokens = PROBE_ISL if phase == "context" else 1
        case = dict(kind="tokens", phase=phase, tokens=tokens, case_id=f"tokens-{phase}-t{tokens}")
    case["index"] = 0
    return case


def _single_rank_env(backend: str) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if backend == "sglang":
        os.environ.setdefault("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", "env://")
    else:
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("DSV411_ALLOCATION_ID", "capture")
    if "DSV411_PRIVATE_CACHE" not in os.environ:
        root = Path(tempfile.mkdtemp(prefix="dsv411_capture_cache_"))
        (root / "home-cache").symlink_to(Path.home() / ".cache")
        os.environ["DSV411_PRIVATE_CACHE"] = str(root)


def _shrink_engram_tables(model_path: Path, manifest: dict, rows: int) -> tuple[Path, dict]:
    """Scratch checkpoint-metadata copy with ``engram_num_embeddings`` = rows per engram layer, and the
    manifest re-keyed to it (config sha + engram structures). Capture-only; never a measurement input."""
    scratch = Path(tempfile.mkdtemp(prefix="dsv411_capture_ckpt_"))
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(model_path / name, scratch / name)
    config = json.loads((scratch / "config.json").read_text())

    def patch(node):
        if isinstance(node, dict):
            if "engram_num_embeddings" in node and isinstance(node["engram_num_embeddings"], list):
                node["engram_num_embeddings"] = [rows] * len(node["engram_num_embeddings"])
                # bucket primes are drawn just below engram_vocab_size per hash column and must sum
                # to at most the table rows: scale the vocab with the rows (compressed vocab untouched)
                # (primes are searched UPWARD from vocab-1, so leave headroom below rows/columns)
                columns = (int(node["engram_max_ngram_size"]) - 1) * int(node["engram_n_heads"])
                vocab = max(int(node["engram_compressed_vocab_size"]) + 1, int(rows // columns * 0.9))
                node["engram_vocab_size"] = vocab
            for value in node.values():
                patch(value)
        elif isinstance(node, list):
            for item in node:
                patch(item)

    patch(config)
    (scratch / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    manifest = copy.deepcopy(manifest)
    manifest["config_sha256"] = contract.sha256_json(json.loads((scratch / "config.json").read_text()))
    for entry in manifest["entries"]:
        if entry["component"] == "engram":
            entry["structure"]["num_embeddings"] = rows
            entry["structure_key"] = contract.structure_key("engram", entry["structure"])
    reps = {}
    for entry in manifest["entries"]:
        reps.setdefault(entry["phase"], {}).setdefault(entry["component"], {}).setdefault(
            entry["structure_key"], entry["layer"]
        )
    manifest["representatives"] = reps
    manifest["engram_rows_override"] = rows
    return scratch, manifest


def run_cell(backend: str, kind: str, phase: str, *, output: Path | None = None) -> Path:
    if backend not in PRODUCERS or kind not in CELLS or phase not in ("context", "generation"):
        raise ValueError(f"unknown cell {backend}/{kind}/{phase}")
    producer = importlib.import_module(PRODUCERS[backend])
    model_path = Path(os.environ["DSV411_MODEL_PATH"])
    prompt_file = Path(os.environ["DSV411_PROMPT_FILE"])
    workspace = Path(os.environ.get("AIS_PROBE_WORKSPACE") or os.environ.get("AIC_PROBE_WORKSPACE") or ".")
    manifest_path = workspace / "facts" / "manifests" / f"dsv411_{backend}_tp1.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path}: export it on the host with components/e2e_align.py --sdk-manifest")
    manifest = json.loads(manifest_path.read_text())
    engram_rows = int(os.environ.get("DSV411_CAPTURE_ENGRAM_ROWS") or 0)
    if kind == "engram" and engram_rows:
        model_path, manifest = _shrink_engram_tables(model_path, manifest, engram_rows)
    _single_rank_env(backend)
    # pins: the installed package IS the pinned source for a capture
    package = Path(importlib.import_module("vllm" if backend == "vllm" else "sglang").__file__).resolve().parent
    sources = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob("*.py"))
    }
    image = (
        os.environ.get("DSV411_LAUNCH_IMAGE_SHA256") or hashlib.sha256(b"dsv411-capture-unverified-image").hexdigest()
    )
    os.environ["DSV411_LAUNCH_IMAGE_SHA256"] = image
    revision = os.environ.get("DSV411_COLLECTOR_REVISION") or "0" * 40
    components, _ = CELLS[kind]
    grid = contract.load_grid()
    plan = dict(
        schema=contract.PLAN_SCHEMA,
        purpose="smoke",
        backend=backend,
        tp_size=1,
        components=components,
        cases=[_case(kind, phase)],
        grid_sha256=contract.sha256_file(contract.GRID_PATH),
        chunk_prefill_size=grid["context"]["chunk_prefill_size"],
        warmup=2,
        iterations=5,
        seed=20261003,
        regimes=dict(grid["regimes"]),
        regime_exceptions=[],
        expected_gpu=os.environ.get("DSV411_EXPECTED_GPU", "H20"),
        expected_sm=producer.EXPECTED_SM[os.environ.get("DSV411_EXPECTED_GPU", "H20")],
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        runtime_digest="sha256:" + image,
        image_sha256=image,
        image_sha256_source="env"
        if os.environ.get("DSV411_LAUNCH_IMAGE_SHA256") == image and "DSV411_LAUNCH_IMAGE_SHA256" in os.environ
        else "unverified-capture",
        source_pins={k: sources[k] for k in sorted(producer.REQUIRED_SOURCES)},
        metadata_pins={
            n: hashlib.sha256((model_path / n).read_bytes()).hexdigest()
            for n in ("config.json", "tokenizer.json", "tokenizer_config.json")
        },
        collector_revision=revision,
        weight_initializer=producer.WEIGHT_INITIALIZER,
        pool=dict(POOL[backend]),
        capture_cell=dict(
            kind=kind, phase=phase, probe_isl=PROBE_ISL, engram_rows_override=manifest.get("engram_rows_override")
        ),
    )
    if backend == "sglang":
        plan["moe_runner_backend"] = "flashinfer_mxfp4"
    else:
        plan["moe_backend"] = "marlin"
    contract.validate_plan(
        plan,
        manifest,
        framework_commit=producer.FRAMEWORK_COMMIT,
        framework_version=producer.FRAMEWORK_VERSION,
        expected_sm=producer.EXPECTED_SM,
        required_sources=producer.REQUIRED_SOURCES,
    )
    # evidence (plan, manifest, receipt with traceback, raw rows) persists in the workspace facts
    out = output or (workspace / "facts" / "dsv411_captures" / f"{backend}_{kind}_{phase}_{int(time.time())}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    args = argparse.Namespace(
        plan=out / "plan.json",
        manifest=out / "manifest.json",
        model_path=model_path,
        prompt_file=prompt_file,
        output=out,
        runtime_digest=plan["runtime_digest"],
        admit=None,
    )
    from .runtime import new_receipt, write_receipt

    receipt = new_receipt(0, backend)
    try:
        producer.run(args, receipt)
    except BaseException as error:
        receipt.update(
            state="failed_preserved", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
        raise
    finally:
        write_receipt(out / "rank-0.json", receipt)
    print(f"dsv411 capture cell {backend}/{kind}/{phase}: {receipt.get('state')} rows={receipt.get('rows')} -> {out}")
    return out
