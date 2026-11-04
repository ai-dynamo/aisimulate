# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: the per-layer decode cost vLLM's own engine pays, with CUDA graphs and eager.

Op tables time components in isolation; this measures what serving runs. vLLM builds the
model through its own loader (dummy weights, the first N layers of the packaged config),
allocates its own KV cache and captures its own decode graphs; the probe only feeds token
ids and times steps. The difference of the step time between two layer counts, divided
by their difference, is the per-layer decode cost under each execution mode. It answers
"is this eager-timed table measuring kernels or launch overhead" (the SGLang DSV4.1 module
tables are eager; vLLM 0.33.0 jobs 527491679 / 527491831 measured 2.9 ms per layer eager vs
0.08 ms with graphs at batch 1 on B200).

Arguments (``--case-filter key=value,...``):
  model     HF repo with a packaged config in src/aisimulate_core/model_configs (required)
  layers    layer counts, ``+``-separated (default ``4+8``)
  modes     ``graph``, ``eager`` or ``graph+eager`` (default)
  points    ``<batch>x<kv_len>`` pairs, ``+``-separated (default ``1x1024+8x1024+32x1024``)
  steps     decode steps per timed generate (default 128)
  attempts  timed attempts per point, median reported (default 5)

Per (layers, mode, point): ``generate`` with 1 and with 1 + ``steps`` output tokens for
``batch`` random prompts of ``kv_len`` tokens (prefix caching off, EOS ignored); the step
time is their difference over ``steps``. Both generates prefill ``batch * kv_len`` tokens, so
the prefill's run-to-run jitter lands in the difference; ``steps`` must be large enough for
the decode to dominate it (32 steps left batch 8/32 points noisy on B200, job 527771824).
Each (layers, mode) engine runs in its own child process, as serving runs one engine per
process.

Config edits, kept so the measured layers stay as served: ``num_hidden_layers`` (top
level or ``text_config``); Engram tables emptied when the config declares them (hundreds
of GB, and an early layer is in every layer count, so the per-layer difference excludes
it); the vision tower is not loaded (image limit 0) when the config has one.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parents[2] / "src" / "aisimulate_core" / "model_configs"
_CHILD_TIMEOUT_S = 3600


def _points(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split("+"):
        batch, _, kv = item.partition("x")
        out.append((int(batch), int(kv)))
    return out


def run(args: dict[str, str], out_dir: Path) -> dict:
    model = args["model"]
    layers = [int(n) for n in args.get("layers", "4+8").split("+")]
    modes = args.get("modes", "graph+eager").split("+")
    points = _points(args.get("points", "1x1024+8x1024+32x1024"))
    steps, attempts = int(args.get("steps", 128)), int(args.get("attempts", 5))
    if not set(modes) <= {"graph", "eager"} or len(layers) < 2:
        raise ValueError(f"modes {modes} must be graph/eager and layers {layers} needs two or more counts")
    runs = []
    for n_layers in layers:
        for mode in modes:
            spec = {
                "model": model,
                "layers": n_layers,
                "mode": mode,
                "points": points,
                "steps": steps,
                "attempts": attempts,
            }
            runs.append(_engine(spec))
    per_layer = []
    lo, hi = min(layers), max(layers)
    for mode in modes:
        by = {(r["layers"], tuple(p["point"])): p for r in runs if r["mode"] == mode for p in r["points"]}
        for point in points:
            a, b = by.get((lo, point)), by.get((hi, point))
            if a and b and a.get("step_ms") is not None and b.get("step_ms") is not None:
                per_layer.append(
                    {
                        "mode": mode,
                        "batch": point[0],
                        "kv_len": point[1],
                        "ms": (b["step_ms"] - a["step_ms"]) / (hi - lo),
                    }
                )
    # every requested (mode, point) must yield a per-layer value: the probe exists to compare them
    return {"runs": runs, "per_layer_ms": per_layer, "failed": len(per_layer) < len(modes) * len(points)}


def _engine(spec: dict) -> dict:
    """One engine in a child process; its JSON result, or the child's failure."""
    with tempfile.TemporaryDirectory(prefix="vllm_decode_step_") as tmp:
        spec_path, out_path = Path(tmp) / "spec.json", Path(tmp) / "out.json"
        spec_path.write_text(json.dumps({**spec, "workdir": tmp}))
        cmd = [sys.executable, "-m", "collector.probes.vllm_decode_step", str(spec_path), str(out_path)]
        cwd = Path(__file__).resolve().parents[2]  # python/aisimulate: `collector` is importable
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:  # keep the engines that finished; record this one's failure
            return {
                **spec,
                "points": [],
                "error": f"child timed out after {_CHILD_TIMEOUT_S}s: {str(exc.stderr)[-2000:]}",
            }
        if out_path.exists():
            return json.loads(out_path.read_text())
        return {**spec, "points": [], "error": f"child exit {proc.returncode}: {proc.stderr[-2000:]}"}


def _vocab_size(config: dict) -> int:
    text = config.get("text_config") or config
    size = text.get("vocab_size") or config.get("vocab_size")
    if not size:
        raise ValueError("the packaged config declares no vocab_size; the probe cannot draw valid token ids")
    return int(size)


def _model_dir(model: str, layers: int, workdir: str) -> tuple[str, bool]:
    config = json.loads((_CONFIGS / f"{model.replace('/', '--')}_config.json").read_text())
    text = config.get("text_config") or config
    if layers > text["num_hidden_layers"]:
        raise ValueError(f"{model} has {text['num_hidden_layers']} layers, asked for {layers}")
    text["num_hidden_layers"] = layers
    for key in ("engram_layer_ids", "engram_num_embeddings"):
        if key in text:
            text[key] = []
    out = Path(workdir) / "model"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return str(out), "vision_config" in config


def _child(spec: dict) -> dict:
    import random

    import torch
    from vllm import LLM, SamplingParams

    points, steps = [tuple(p) for p in spec["points"]], spec["steps"]
    model_dir, has_vision = _model_dir(spec["model"], spec["layers"], spec["workdir"])
    llm = LLM(
        model=model_dir,
        load_format="dummy",
        enforce_eager=spec["mode"] == "eager",
        max_model_len=max(kv for _b, kv in points) + steps + 64,
        max_num_seqs=max(b for b, _kv in points),
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        gpu_memory_utilization=0.85,
        seed=0,
        **({"limit_mm_per_prompt": {"image": 0}} if has_vision else {}),
    )
    import vllm

    compilation = llm.llm_engine.vllm_config.compilation_config
    result = {
        **{k: spec[k] for k in ("model", "layers", "mode", "steps")},
        "vllm_version": vllm.__version__,
        "device": torch.cuda.get_device_name(0),
        "cudagraph_mode": str(getattr(compilation, "cudagraph_mode", None)),
        "points": [],
    }
    rng = random.Random(0)
    vocab = _vocab_size(json.loads((Path(model_dir) / "config.json").read_text()))

    def timed(batch: int, kv_len: int, tokens: int) -> float:
        prompts = [{"prompt_token_ids": [rng.randrange(vocab) for _ in range(kv_len)]} for _ in range(batch)]
        params = SamplingParams(
            max_tokens=tokens, min_tokens=tokens, ignore_eos=True, temperature=0.0, detokenize=False
        )
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        if {len(o.outputs[0].token_ids) for o in outputs} != {tokens}:
            raise RuntimeError(f"asked for {tokens} tokens per request, the engine produced a different count")
        return elapsed

    for batch, kv_len in points:
        row: dict = {"point": [batch, kv_len]}
        try:
            timed(batch, kv_len, 1 + steps)  # warm-up: graphs, JIT, allocator
            ms = [
                (timed(batch, kv_len, 1 + steps) - timed(batch, kv_len, 1)) / steps * 1e3
                for _ in range(spec["attempts"])
            ]
            row.update(step_ms=statistics.median(ms), attempts_ms=ms)
        except Exception as exc:  # the other points still run
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        result["points"].append(row)
    return result


if __name__ == "__main__":
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    spec_file, out_file = sys.argv[1], sys.argv[2]
    try:
        res = _child(json.loads(Path(spec_file).read_text()))
    except Exception as exc:
        res = {**json.loads(Path(spec_file).read_text()), "points": [], "error": f"{type(exc).__name__}: {exc}"[:2000]}
    Path(out_file).write_text(json.dumps(res))
