#!/usr/bin/env python3
"""TRT-LLM identity probe — SIMPLE version (agreed: try simple first).

llmapi LLM(load_format='dummy') -> object-graph search for the torch model
(A-level introspection) -> one tiny generate under torch.profiler (C-level
kernels). Expected failure mode: the executor lives in a subprocess and the
model/kernels are invisible in-process — if so, this records exactly that,
which is the datapoint that justifies the complex design.

Runs INSIDE trtllm-probe:1.3.0rc20-onnxfix with LD_LIBRARY_PATH set by the
image entrypoint (invoke via `bash -lc`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from collections import Counter, defaultdict


import importlib.abc
import importlib.util


class _CutlassWalkGuard(importlib.abc.MetaPathFinder):
    """trtllm's warmup pkgutil.walk_packages force-imports every cutlass
    submodule, including `cutlass._mlir_helpers` — a module the normal flow
    never imports because `cutlass.base_dsl._mlir_helpers` already registered
    the same MLIR value casters -> fatal double registration. Raising
    ImportError here is safe: walk_packages ignores ImportError by design,
    and any legitimate later import of this module would have crashed anyway."""

    # only the never-legitimately-imported duplicate-caster module is blocked;
    # blocking wider cutlass._mlir broke legit cute-dsl runner imports.
    # SCOPE (2026-09-30, TensorRT-LLM 1.3.0rc29): the block is active only
    # while a pkgutil.walk_packages walk is running (_safe_walk below). On
    # rc29 flashinfer 0.6.18 imports cutlass._mlir_helpers LEGITIMATELY at
    # import time (and it no longer double-registers there); a global block
    # made `import flashinfer` fail silently inside
    # tensorrt_llm._torch.attention.backends, so FlashInferAttentionMetadata
    # was never exported and every model load died with an ImportError —
    # while the plain image imported fine. The warmup walk is the only place
    # the duplicate-caster import ever happened, so that is the only place
    # to block it.
    BLOCK = ("cutlass._mlir_helpers",)

    def find_spec(self, name, path=None, target=None):
        if not _WALKING:
            return None
        for b in self.BLOCK:
            if name == b or name.startswith(b + "."):
                raise ImportError(f"blocked by AIC probe walk-guard: {name}")
        return None


_WALKING = False


sys.meta_path.insert(0, _CutlassWalkGuard())

# cutlass DSL hashes its own module tree via pkgutil.walk_packages for a JIT
# cache key (cutlass.py:512). walk_packages only forgives ImportError; broken
# generated dialect modules raise AttributeError and kill the load. The MLIR
# dialect imports bypass meta_path (file-based loaders), so guard at the
# walk itself: truncate on ANY exception — a shorter hash input is harmless.
import pkgutil

_orig_walk = pkgutil.walk_packages


def _safe_walk(*a, **k):
    global _WALKING
    it = _orig_walk(*a, **k)
    _WALKING = True  # the cutlass walk-guard blocks only inside a walk
    try:
        while True:
            try:
                yield next(it)
            except StopIteration:
                return
            except Exception:
                return
    finally:
        _WALKING = False


pkgutil.walk_packages = _safe_walk


def find_torch_model(root, max_depth: int = 8):
    import torch.nn as nn

    seen, queue, best = set(), [(root, 0)], None
    while queue:
        obj, d = queue.pop(0)
        if id(obj) in seen or d > max_depth:
            continue
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            n = sum(1 for _ in obj.parameters(recurse=True))
            if best is None or n > best[1]:
                best = (obj, n)
            continue
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, nn.Module):
                queue.append((child, d + 1))
            elif callable(child) or isinstance(child, (str, int, float, bool, bytes)):
                continue
            elif isinstance(child, (list, tuple)) and len(child) < 32:
                queue.extend((c, d + 1) for c in child)
            elif isinstance(child, dict) and len(child) < 32:
                queue.extend((c, d + 1) for c in child.values())
            elif not isinstance(child, set):
                queue.append((child, d + 1))
    return best[0] if best else None



def vision_dummy_forward(model, model_dir: str, rec: dict, attr_names=("visual", "mm_encoder", "vision_tower")) -> None:
    """vLLM's engine construction runs a profile_run that pushes dummy images through the vision
    encoder, so probe_vllm records encoder kernels for free; this framework's text-only probe never
    executes the vision tower, and the encoder_attention gate had NO serving evidence (sm89 2026-10-04:
    DIVERGED for want of a kernel, not a path difference). Mirror the profile_run: one synthetic image
    (grid 1 x 32 x 32 patches = 512 px square at patch 16, HF pixel_values layout
    [num_patches, in_channels * temporal_patch * patch * patch]) through the vision module under the
    device profiler, recorded as the same `profile_run_kernels` table path_diff reads for that phase."""
    import inspect
    import json as _json
    from pathlib import Path

    import torch
    vis = next((getattr(model, a, None) for a in attr_names if getattr(model, a, None) is not None), None)

    def _takes_pixels_and_grid(m) -> bool:
        # only the Qwen-VL family's layout is synthesised here: forward(pixel_values | x, grid_thw) over HF flat
        # patches. SigLIP-style towers (Gemma-4: forward(pixel_values) over a 4-D image) and MoonViT (Kimi-K2.5)
        # take other inputs and are recorded as skipped, never attempted — a wrong guess is not evidence.
        try:
            ps = [p for p in inspect.signature(m.forward).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            return False
        return len(ps) >= 2 and ps[1].default is inspect.Parameter.empty and ps[1].name in ("grid_thw", "image_grid_thw")
    # trtllm wraps the ViT (forward(pixel_values, grid_thw)) in a Qwen3VisionModelBase whose forward takes the
    # executor's multimodal_params list: descend to the child that takes the raw patches
    for _ in range(3):
        if vis is None or _takes_pixels_and_grid(vis):
            break
        vis = next((getattr(vis, a, None) for a in attr_names + ("vision_model", "encoder")
                    if getattr(vis, a, None) is not None), None)
    if vis is None:
        return
    if not _takes_pixels_and_grid(vis):
        try:
            sig = str(inspect.signature(vis.forward))
        except (TypeError, ValueError):
            sig = "?"
        rec["vision_probe"] = {"skipped": f"unsupported vision input layout: {type(vis).__name__}.forward{sig}"[:300]}
        return
    try:
        cfg = _json.loads((Path(model_dir) / "config.json").read_text())
        vc = cfg.get("vision_config") or (cfg.get("text_config") or {}).get("vision_config") or {}
        p, t, c = int(vc.get("patch_size", 14)), int(vc.get("temporal_patch_size", 2)), int(vc.get("in_channels", 3))
        merge = int(vc.get("spatial_merge_size", 2))
        side = 32 - (32 % merge)
        grid = torch.tensor([[1, side, side]], dtype=torch.int64)
        dev = next(vis.parameters()).device
        dt = next(vis.parameters()).dtype
        pix = torch.randn(int(grid[0].prod()), c * t * p * p, dtype=dt, device=dev)
        from torch.profiler import ProfilerActivity, profile
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            vis(pix, grid)
            torch.cuda.synchronize()
        from torch.autograd import DeviceType
        acc: dict = {}
        for ev in prof.profiler.kineto_results.events():
            try:
                if ev.device_type() != DeviceType.CUDA:
                    continue
            except Exception:
                continue
            a = acc.setdefault(ev.name(), {"us": 0.0, "launches": 0})
            a["us"] += (ev.duration_ns() / 1e3 if hasattr(ev, "duration_ns") else ev.duration_us())
            a["launches"] += 1
        rec["profile_run_kernels"] = sorted(
            ({"kernel": k, "us": round(v["us"], 1), "launches": v["launches"]} for k, v in acc.items()),
            key=lambda r: -r["us"])
        rec["vision_probe"] = {"module": f"{type(vis).__module__}.{type(vis).__name__}", "grid_thw": grid.tolist(),
                               "pixel_values_shape": list(pix.shape), "dtype": str(dt), "kernels": len(acc)}
    except Exception as e:
        # evidence for ONE gate, never a verdict on the cell: a failed vision probe is recorded here, not in errors
        rec["vision_probe"] = {"error": f"{type(e).__name__}: {e}"[:400]}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    # prompt length of the profiled request (same knob/default as the vllm and
    # sglang probes; recorded as probe_isl). The historical 16-token prompt
    # sat below every length-conditional dispatch threshold.
    ap.add_argument("--isl", type=int,
                    default=int(os.environ.get("AIS_PROBE_ISL") or os.environ.get("AIC_PROBE_ISL") or "4096"))
    ap.add_argument("--kv-dtype", default=None,
                    help="override kv_cache_config.dtype (e.g. fp8): the kv-cache dtype variant "
                         "axis the collector sweeps; recorded as probe_kv_cache_dtype")
    ap.add_argument("--eager", action="store_true",
                    help="drop the rendered cuda_graph_config (A/B only; default keeps the framework's own graph mode)")
    ap.add_argument("--engine-yaml", default=None,
                    help="generator-rendered extra_engine_args yaml (dynamo.trtllm contract)")
    args = ap.parse_args()
    rec: dict = {"model_path": args.model, "errors": {}}
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
    except Exception:
        rec["device_capability"] = None

    import tensorrt_llm

    rec["trtllm_version"] = tensorrt_llm.__version__
    # This image's _cutlass_ir C lib predates register_traceback_file_exclusion,
    # so the generated `_iket_ops_gen` module crashes on import — but the iket
    # dialect IS legitimately imported (SM100 cute-dsl runners) even on SM90,
    # where its ops never execute. Give ONLY the generated-ops module a
    # permissive stub (PEP 562 __getattr__): imports succeed, attribute
    # accesses yield dummies, nothing SM90 actually runs is affected.
    # Only stub when the real module is actually broken (rc20 image had a
    # stale _cutlass_ir C lib); newer images import it fine and must not be
    # shadowed.
    import types
    try:
        import cutlass._mlir.dialects._iket_ops_gen  # noqa: F401
        rec["cutlass_stub_modules"] = []
    except Exception:
        _gen = types.ModuleType("cutlass._mlir.dialects._iket_ops_gen")
        _gen.__all__ = []
        _gen.__getattr__ = lambda name: type(name, (object,), {"__init__": lambda self, *a, **k: None})
        sys.modules["cutlass._mlir.dialects._iket_ops_gen"] = _gen
        rec["cutlass_stub_modules"] = ["cutlass._mlir.dialects._iket_ops_gen"]
    try:
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        kwargs = dict(
            model=args.model,
            load_format="dummy",
            trust_remote_code=args.trust_remote_code,
            # cache-cold prefill: block reuse (default on) would turn the
            # profiled request into a residual of the warmup prompt — same
            # policy as vllm enable_prefix_caching=False / sglang disable_radix_cache
            kv_cache_config=KvCacheConfig(max_tokens=max(16384, args.isl + 256),
                                          enable_block_reuse=False,
                                          **({"dtype": args.kv_dtype} if args.kv_dtype else {})),
            max_batch_size=8,
            max_seq_len=args.isl + 64,
        )
        rec["probe_kv_cache_dtype"] = args.kv_dtype  # None = the engine yaml's own value
        if args.engine_yaml:
            import yaml as _yaml
            eng = _yaml.safe_load(open(args.engine_yaml)) or {}
            rec["engine_yaml"] = dict(eng)
            probe_overrides = {}
            # empty template slots render as None — dropping them mirrors
            # dynamo.trtllm, which also skips unset keys
            eng = {k: v for k, v in eng.items() if v is not None}
            # backend: pytorch is llmapi's constructor CHOICE, not an arg
            if eng.pop("backend", None) not in (None, "pytorch"):
                probe_overrides["backend"] = "non-pytorch backend requested; probe uses llmapi pytorch LLM"
            kvc = dict(eng.pop("kv_cache_config", {}) or {})
            # identity probe: cap the pool (tiny dummies + fraction sizing OOMed
            # sglang at 138GB; same hazard here) — keep dtype/block identity keys
            if kvc.pop("free_gpu_memory_fraction", None) is not None:
                probe_overrides["kv_cache_config.free_gpu_memory_fraction"] = "replaced by max_tokens=16384 cap"
            kvc["max_tokens"] = max(16384, args.isl + 256)
            if kvc.get("enable_block_reuse", True):
                probe_overrides["kv_cache_config.enable_block_reuse"] = "False: profiled prefill must be cache-cold"
            kvc["enable_block_reuse"] = False
            if args.kv_dtype:
                # kv-cache dtype variant (probe_driver kv_variants): the
                # collector sweeps fp8-KV on every backend; the rendered
                # engine yaml says `auto` for most profiles
                probe_overrides["kv_cache_config.dtype"] = f"{kvc.get('dtype', 'auto')} -> {args.kv_dtype} (kv variant probe)"
                kvc["dtype"] = args.kv_dtype
            kwargs["kv_cache_config"] = KvCacheConfig(**kvc)
            # the rendered cuda_graph_config IS serving's execution mode; keep
            # it (owner decision 2026-09-24) — --eager drops it for A/B only
            if args.eager and eng.pop("cuda_graph_config", None) is not None:
                probe_overrides["cuda_graph_config"] = "dropped: --eager A/B run"
            rec["probe_eager"] = bool(args.eager)
            kwargs.update(eng)
            if (kwargs.get("max_seq_len") or 0) < args.isl + 64:
                probe_overrides["max_seq_len"] = f"raised to {args.isl + 64} to fit the {args.isl}-token probe prompt"
                kwargs["max_seq_len"] = args.isl + 64
            rec["probe_overrides"] = probe_overrides
        # unknown kwargs are drift facts (template ahead of / behind llmapi)
        rec["engine_yaml_unknown_args"] = []
        for _ in range(8):
            try:
                llm = LLM(**kwargs)
                break
            except TypeError as te:
                import re as _re
                m = _re.search(r"unexpected keyword argument '([^']+)'", str(te))
                if not m or m.group(1) not in kwargs:
                    raise
                rec["engine_yaml_unknown_args"].append(m.group(1))
                kwargs.pop(m.group(1))
        else:
            raise RuntimeError("LLM kwargs never converged")
        rec["llm_class"] = type(llm).__qualname__
        rec["executor_class"] = type(getattr(llm, "_executor", None)).__qualname__

        model = find_torch_model(llm)
        if model is None:
            rec["in_process_model"] = False  # THE datapoint: subprocess executor
        else:
            rec["in_process_model"] = True
            rec["model_class"] = type(model).__qualname__
            qm = defaultdict(list)
            for name, mod in model.named_modules():
                q = getattr(mod, "quant_method", None) or getattr(mod, "quant_config", None)
                if q is not None:
                    qm[type(q).__name__].append(name)
            rec["quant_methods"] = {k: {"count": len(v), "modules": v[:4]} for k, v in qm.items()}
            rec["param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
            vision_dummy_forward(model, args.model, rec)   # encoder_attention evidence (profile_run table)

        # ground truth for kv dtype: what the KV cache manager actually
        # allocates with (llmapi 'auto' resolves inside the executor)
        kvres = {"configured": str(getattr(kwargs.get("kv_cache_config"), "dtype", None))}
        seen_kv, queue_kv = set(), [(llm, 0)]
        while queue_kv:
            obj, d = queue_kv.pop(0)
            if id(obj) in seen_kv or d > 8:
                continue
            seen_kv.add(id(obj))
            tn = type(obj).__name__
            if "CacheManager" in tn:  # KVCacheManager, MambaHybridCacheManager, ...
                for a in ("dtype", "kv_cache_dtype", "kv_dtype", "kv_cache_type",
                          "mamba_ssm_cache_dtype"):
                    v = getattr(obj, a, None)
                    if v is not None:
                        kvres[f"manager.{a}"] = str(v)
                kvres.setdefault("manager_class", tn)
                if any(k.startswith("manager.") for k in kvres):
                    break
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                try:
                    child = getattr(obj, name)
                except Exception:
                    continue
                if not isinstance(child, (str, int, float, bool, bytes, type(None))):
                    queue_kv.append((child, d + 1))
        rec["kv_cache_resolved"] = kvres

        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            from tensorrt_llm import SamplingParams

            rec["probe_isl"] = args.isl
            rec["probe_prefix_caching"] = False
            # warmup on a DIFFERENT prompt (block reuse is off, but never rely on
            # one guard): lazy JIT / autotune happen off-profile
            _ = llm.generate([list(range(1, args.isl + 1))], SamplingParams(max_tokens=2))
            prompt = [list(range(args.isl))]

            def _table(prof):
                rows = []
                for e in prof.key_averages():
                    dt = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
                    if dt > 0:
                        rows.append({"kernel": e.key, "launches": e.count, "us": round(dt, 1)})
                return sorted(rows, key=lambda r: -r["us"])  # no count cap: identity evidence

            # TWO phases, like the vllm/sglang probes: the llmapi has no step()
            # so prefill = a max_tokens=1 run; decode = what a max_tokens=2 run
            # executes beyond it (per-kernel launch/time difference)
            _kw = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            with profile(**_kw) as p1:
                llm.generate(prompt, SamplingParams(max_tokens=1))
                torch.cuda.synchronize()
            with profile(**_kw) as p2:
                llm.generate(prompt, SamplingParams(max_tokens=2))
                torch.cuda.synchronize()
            pre = _table(p1)
            both = {r["kernel"]: r for r in _table(p2)}
            pre_by = {r["kernel"]: r for r in pre}
            dec = []
            for k, r in both.items():
                d_l = r["launches"] - pre_by.get(k, {}).get("launches", 0)
                if d_l > 0:
                    dec.append({"kernel": k, "launches": d_l,
                                "us": round(max(r["us"] - pre_by.get(k, {}).get("us", 0.0), 0.0), 1)})
            rec["prefill_kernels"] = pre
            rec["decode_kernels"] = sorted(dec, key=lambda r: -r["us"])
            rec["kernels_visible_in_process"] = bool(pre)  # False == subprocess executor
            rec["kernels"] = _table(p2)  # whole-run view kept for older consumers
        except Exception:
            rec["errors"]["generate"] = traceback.format_exc()[-2000:]
    except Exception:
        rec["errors"]["load"] = traceback.format_exc()

    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, "errors:", list(rec["errors"]),
          "in_process:", rec.get("in_process_model"), "kernels:", rec.get("kernels_visible_in_process"))


if __name__ == "__main__":
    main()
