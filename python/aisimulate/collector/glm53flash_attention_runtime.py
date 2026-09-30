# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral GPU helpers shared by the GLM attention runners.

Both runners time the loaded attention module with CUDA events. Prefill
repeats the module call eagerly inside the real forward (serving runs prefill
eagerly); decode replays a CUDA graph of the module captured from the exact
arguments, forward context and persistent metadata buffers that the
framework's own decode-graph capture used (serving runs decode under full CUDA
graphs), after a real framework decode replay refreshed those buffers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from collector.glm53flash_attention_contract import (
    TIMING_METHODS,
    attention_body,
    geometry_key,
    indexer_regime,
    sha256_json,
)


def package_source_sha256(package_root: Path) -> tuple[str, dict]:
    """Hash every Python source of the imported framework package."""
    sources = {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }
    return sha256_json(sources), sources


def config_sha256(checkpoint: Path) -> str:
    return sha256_json(json.loads((Path(checkpoint) / "config.json").read_text()))


def corpus_tokens(tokenizer, corpus: Path, minimum: int) -> tuple[list[int], dict]:
    text = Path(corpus).read_text()
    tokens = tokenizer.encode(text)
    if len(tokens) < minimum:
        raise RuntimeError(f"corpus has {len(tokens)} tokens, fewer than the {minimum} a request needs")
    return tokens, {
        "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
    }


def request_tokens(tokens: list[int], request: int, length: int) -> list[int]:
    """Distinct real-text window per request (deterministic offset)."""
    start = (request * 4099) % (len(tokens) - length + 1)
    return tokens[start : start + length]


class EventTimer:
    """Enqueue every repetition back to back, then read all intervals."""

    def __init__(self, torch):
        self.torch = torch
        self.pairs = []

    def __call__(self, fn):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.pairs.append((start, end))
        return result

    def read(self) -> list[float]:
        self.torch.cuda.synchronize()
        values = [float(start.elapsed_time(end)) for start, end in self.pairs]
        self.pairs.clear()
        return values


class RawWriter:
    """Per-rank raw JSONL stream of repetition samples and diagnostics."""

    def __init__(self, output: Path, rank: int, key_base: dict, provenance: dict):
        self.path = Path(output) / f"rank-{rank}.jsonl"
        self.rank = rank
        self.key_base = key_base
        self.provenance = provenance

    def samples(self, target: dict, latencies: list[float], warmup: int, kernel_source: str, extra: dict) -> None:
        phase = target["phase"]
        timing_method, used_graph = TIMING_METHODS[phase]
        key = {
            "geometry": geometry_key(attention_body(self.key_base, phase == "context")),
            "batch_size": target["batch_size"],
            "prefix": target["prefix"],
            "x": target["x"],
            "indexer_regime": indexer_regime(phase, target["prefix"], target["x"], self.key_base["index_topk"]),
        }
        with self.path.open("a") as stream:
            for repetition, latency in enumerate(latencies):
                if repetition < warmup:
                    continue
                stream.write(
                    json.dumps(
                        {
                            "record": "sample",
                            "target_id": target["target_id"],
                            "tp_rank": self.rank,
                            "repetition": repetition - warmup,
                            "latency_ms": latency,
                            "key": key,
                            "provenance": self.provenance,
                            "kernel_source": kernel_source,
                            "timing_method": timing_method,
                            "used_cuda_graph": used_graph,
                            "extra": extra,
                        }
                    )
                    + "\n"
                )

    def diagnostic(self, payload: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps({"record": "diagnostic", "tp_rank": self.rank, **payload}) + "\n")


def target_id(phase: str, batch: int, prefix: int, x: int) -> str:
    return f"{phase}-b{batch}-p{prefix}-x{x}"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
