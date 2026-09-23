#!/usr/bin/env python3
"""FPM run.sh -> framework parser -> identity probe (vLLM path).

Bridges AISim's FPM artifacts to the probe: the engine command the
generator rendered IS the probe input — parsed by vLLM's own CLI parser so
there is zero translation drift between "what a deployment runs" and "what
the probe runs". The only mutations: model_path may be swapped to a dummy
variant, load_format forced to dummy, eager mode forced (identity probe).

Runs INSIDE the vllm image.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import traceback
from collections import Counter, defaultdict

# flags owned by dynamo/FPM orchestration, not vLLM engine args
_NON_ENGINE_FLAGS_WITH_VALUE = {
    "--benchmark-mode", "--dump-config-to", "--benchmark-output-path",
    "--nnodes", "--node-rank", "--master-addr", "--master-port",
    "--data-parallel-size-local", "--data-parallel-start-rank",
    "--data-parallel-address", "--data-parallel-rpc-port",
}
_NON_ENGINE_FLAGS_BARE = {"--headless", "--data-parallel-hybrid-lb"}


def parse_run_sh(path: str) -> tuple[list[str], dict[str, str]]:
    """Extract the engine argv (launcher + FPM flags stripped) and exported env."""
    text = open(path).read()
    m = re.search(r"engine_command=\((.*?)\)\s*$", text, re.M | re.S)
    if m:
        argv = shlex.split(m.group(1))
    else:
        # dynamo target (generator falls back to it when FPM preconditions
        # do not hold): a multi-line `python3 -m dynamo.vllm \` invocation
        # with shell variables. Take that block and resolve $MODEL_PATH.
        # take ONLY the backslash-continued invocation, not the shell plumbing
        # that follows it (pipes, subshell/loop tails)
        blk = re.search(r"python3 -m dynamo\.vllm((?:[^\n]*\\\n)*[^\n]*)", text)
        if not blk:
            raise ValueError(f"no engine command found in {path}")
        line = blk.group(1).replace("\\\n", " ")
        line = re.split(r"\s(?:2>&1|\||&|;|\))", line)[0]
        model = re.search(r'^export MODEL_PATH=\$\{MODEL_PATH:-"([^"]+)"\}', text, re.M)
        line = line.replace('"$MODEL_PATH"', model.group(1) if model else "")
        line = re.sub(r'"?\$\{?[A-Z_]+\}?"?', "", line)  # drop unresolved vars
        argv = shlex.split(line)
    # strip launcher prefix: python3 -m dynamo.vllm / vllm serve ...
    while argv and not argv[0].startswith("--"):
        argv.pop(0)
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in _NON_ENGINE_FLAGS_BARE:
            i += 1
        elif a in _NON_ENGINE_FLAGS_WITH_VALUE:
            i += 2
        else:
            out.append(a)
            i += 1
    env = dict(re.findall(r"^export ([A-Za-z_][A-Za-z0-9_]*)=(\S+)$", text, re.M))
    return out, env


def find_torch_model(root, max_depth: int = 8):
    """BFS the object graph for the biggest nn.Module — version-agnostic."""
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
            continue  # don't descend into modules
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, nn.Module):
                queue.append((child, d + 1))  # Modules are callable — check first
            elif callable(child) or isinstance(child, (str, int, float, bool, bytes)):
                continue
            elif isinstance(child, (list, tuple)) and len(child) < 32:
                queue.extend((c, d + 1) for c in child)
            elif isinstance(child, dict) and len(child) < 32:
                queue.extend((c, d + 1) for c in child.values())
            elif not isinstance(child, set):
                queue.append((child, d + 1))
    return best[0] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-sh", required=True)
    ap.add_argument("--model-override", default=None, help="swap --model to this dummy variant dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--py-paths", action="store_true")
    # Probe prompt length. Default 4096 (owner decision 2026-09-19): sparse
    # models gate their real path on sequence length (M3 sparse-vs-dense
    # threshold at topk_blocks*block_size = 2048 tokens, FA split-KV), so a
    # short prompt records the wrong serving truth. 32 was the historical
    # default and hid those paths; records carry probe_isl so evidence from
    # both eras stays distinguishable. Coverage is a parameter, never a
    # per-model special case.
    ap.add_argument("--isl", type=int, default=int(os.environ.get("AIS_PROBE_ISL") or os.environ.get("AIC_PROBE_ISL") or "4096"))
    # Sweep the collector-relevant serving config. kv-cache-dtype is the
    # typical one (fp8 is a common deployment); the collector sweeps it
    # universally, so the probe must too or path_diff can never compare the
    # fp8 path (the Gemma-4 fp8+head512 backend divergence went uncaught for
    # exactly this reason, 2026-09-22). Overrides the rendered run.sh value.
    ap.add_argument("--kv-cache-dtype", default=None,
                    help="override serving --kv-cache-dtype (e.g. fp8) to probe that config")
    args = ap.parse_args()

    rec: dict = {"run_sh": args.run_sh, "errors": {}, "probe_isl": None}
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
    except Exception:
        rec["device_capability"] = None
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep EngineCore in-process

    argv, sh_env = parse_run_sh(args.run_sh)
    for k, v in sh_env.items():  # generator-owned env is part of the contract
        if k not in os.environ and not k.startswith(("HF_", "DYN_FPM", "FPM_")):
            os.environ[k] = v
    if args.model_override:
        i = argv.index("--model")
        argv[i + 1] = args.model_override
    if args.kv_cache_dtype:
        if "--kv-cache-dtype" in argv:
            argv[argv.index("--kv-cache-dtype") + 1] = args.kv_cache_dtype
        else:
            argv += ["--kv-cache-dtype", args.kv_cache_dtype]
    rec["probe_kv_cache_dtype"] = args.kv_cache_dtype
    rec["engine_argv"] = argv

    import vllm

    rec["vllm_version"] = vllm.__version__

    try:
        from vllm.engine.arg_utils import EngineArgs
        try:
            from vllm.utils import FlexibleArgumentParser
        except ImportError:  # 0.24 moved utils into submodules
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        parser = FlexibleArgumentParser()
        EngineArgs.add_cli_args(parser)
        ns = parser.parse_args(argv)
        # config delta: every rendered flag that differs from the framework's
        # OWN default (same parser, argv = --model only) is recorded. The
        # generator's deltas are liabilities to audit — the DSV4 crash on
        # 0.29 was triggered by one (--max-num-batched-tokens 6012, a value
        # the framework itself would never produce; owner decision
        # 2026-09-20: record the delta on every probe). Identity args are
        # not deltas. Computed at the parser layer, so the probe's own
        # injections below (load_format/enforce_eager) never appear.
        try:
            _defaults = parser.parse_args(["--model", ns.model])
            _skip = {"model", "served_model_name"}
            def _enc(v):
                return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
            rec["config_delta"] = {
                k: {"rendered": _enc(v), "default": _enc(getattr(_defaults, k, None))}
                for k, v in vars(ns).items()
                if k not in _skip and v != getattr(_defaults, k, None)
            }
        except Exception as e:
            rec["errors"]["config_delta"] = f"{type(e).__name__}: {e}"[:200]
        ea = EngineArgs.from_cli_args(ns)
        ea.load_format = "dummy"
        ea.enforce_eager = True  # identity probe: no graph capture
        # The profiled request must be a CACHE-COLD prefill. vllm enables
        # prefix caching by default and the warmup request below uses the
        # same prompt, so with caching on the "prefill" step recomputed only
        # the last block: a query of <= block_size tokens, which DSA's
        # metadata builder classifies as decode (reorder threshold 256 for
        # 128 heads, flashmla_sparse.py:244 @0.29.0) — the serving evidence
        # then showed the fp8 DECODE kernel for prefill and path_diff flagged
        # the collector's (correct) sparse prefill as drift. Same policy as
        # the sglang probe's --disable-radix-cache. Found 2026-09-23.
        ea.enable_prefix_caching = False
        rec["probe_prefix_caching"] = False
        rec["engine_args_resolved"] = {
            k: v for k, v in vars(ea).items()
            if isinstance(v, (str, int, float, bool, type(None)))
            and any(s in k for s in ("quant", "dtype", "parallel", "block", "model",
                                     "attention", "kv", "moe", "backend", "eager", "load"))
        }
    except Exception:
        rec["errors"]["parse"] = traceback.format_exc()
        json.dump(rec, open(args.out, "w"), indent=1, default=str)
        return

    try:
        try:
            from vllm.v1.engine.llm_engine import LLMEngine
        except ImportError:
            from vllm import LLMEngine
        # Engine construction runs vLLM's OWN profile_run: a full-model dummy
        # forward at max_num_batched_tokens which, for multimodal models, also
        # pushes vLLM's max-size dummy images through the vision encoder. The
        # probe prompt is text-only, so this is the only place vision-encoder
        # kernels (encoder_attention) execute — capture it as a third evidence
        # table. Same device-stream method as the phase tables (2026-09-24).
        from torch.profiler import ProfilerActivity as _PA, profile as _profile
        with _profile(activities=[_PA.CPU, _PA.CUDA]) as _p_init:
            engine = LLMEngine.from_engine_args(ea)
        try:
            from torch.autograd import DeviceType as _DT
            _acc: dict = {}
            for _kev in _p_init.profiler.kineto_results.events():
                try:
                    if _kev.device_type() != _DT.CUDA:
                        continue
                except Exception:
                    continue
                _a = _acc.setdefault(_kev.name(), {"us": 0.0, "launches": 0})
                _a["us"] += (_kev.duration_ns() / 1e3 if hasattr(_kev, "duration_ns") else _kev.duration_us())
                _a["launches"] += 1
            rec["profile_run_kernels"] = sorted(
                ({"kernel": k, "us": round(v["us"], 1), "launches": v["launches"]} for k, v in _acc.items()),
                key=lambda r: -r["us"])
        except Exception as e:
            rec["errors"]["profile_run_kernels"] = f"{type(e).__name__}: {e}"[:200]
        model = find_torch_model(engine)
        if model is None:
            raise RuntimeError("no nn.Module found via object-graph search")
        rec["model_class"] = type(model).__qualname__

        qm = defaultdict(list)
        for name, mod in model.named_modules():
            q = getattr(mod, "quant_method", None)
            if q is not None:
                qm[f"{type(q).__module__}.{type(q).__name__}"].append(name)
        rec["quant_methods"] = {k: {"count": len(v), "modules": v[:6]} for k, v in qm.items()}
        rec["param_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        samples = {}
        for name, p in model.named_parameters():
            for key in ("experts", "qkv", "kv_b", "o_proj", "gate_up", "down_proj", "indexer"):
                if key in name and key not in {s.split("::")[0] for s in samples}:
                    samples[f"{key}::{name}"] = f"{p.dtype} {tuple(p.shape)}"
        rec["weight_samples"] = samples

        kvres = {}
        vcfg = getattr(engine, "vllm_config", None)
        if vcfg is not None:
            kvres["cache_config_dtype"] = str(getattr(vcfg.cache_config, "cache_dtype", None))
        for _name, mod in model.named_modules():
            kd = getattr(mod, "kv_cache_dtype", None)
            if kd is not None:
                kvres["attn_kv_cache_dtype"] = str(kd)
                break
        # ground truth: the torch dtype the model runner allocates the cache
        # with (this is where vllm's 'auto' finally resolves)
        seen, queue = set(), [(engine, 0)]
        while queue:
            obj, d = queue.pop(0)
            if id(obj) in seen or d > 8:
                continue
            seen.add(id(obj))
            if type(obj).__name__.endswith("ModelRunner"):
                kvres["runner_kv_cache_dtype"] = str(getattr(obj, "kv_cache_dtype", None))
                spec = None
                try:
                    spec = obj.get_kv_cache_spec()
                except Exception:
                    pass
                if spec:
                    kvres["kv_cache_spec_dtypes"] = sorted(
                        {str(getattr(s, "dtype", None)) for s in spec.values()})
                break
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                try:
                    child = getattr(obj, name)
                except Exception:
                    continue
                if not isinstance(child, (str, int, float, bool, bytes, type(None))):
                    queue.append((child, d + 1))
        rec["kv_cache_resolved"] = kvres or None
    except Exception:
        rec["errors"]["load"] = traceback.format_exc()

    if args.trace and not rec["errors"]:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile, record_function

            def wrap_span(cls, meth, label_fn):
                orig = getattr(cls, meth)
                if getattr(orig, "_aic_wrapped", False):
                    return

                def wrapped(self, *a, _o=orig, **k):
                    with record_function(label_fn(self)):
                        return _o(self, *a, **k)

                wrapped._aic_wrapped = True
                setattr(cls, meth, wrapped)

            for _n, m in model.named_modules():
                q = getattr(m, "quant_method", None)
                if q is not None and hasattr(type(q), "apply"):
                    wrap_span(type(q), "apply", lambda s: f"AIC::quant_apply::{type(s).__name__}")
            try:  # attention boundaries: scan the LOADED model for attention-ish
                # module classes (Attention/MLA/Mixer/linear-attn) — generic across
                # model families, no hardcoded module-path list to maintain
                import torch.nn as _nn
                wrapped = set()
                for _n, m in model.named_modules():
                    t = type(m)
                    if t in wrapped:
                        continue
                    if any(k in t.__name__ for k in ("Attention", "Attn", "MLA", "Mixer", "SSM", "Compressor", "Indexer")):
                        # forward may be inherited (DSV4 classes) — wrap the
                        # class in the MRO that actually defines it; the span
                        # label reads the runtime type, so sharing a base is fine
                        holder = next((c for c in t.__mro__
                                       if "forward" in vars(c) and c is not _nn.Module), None)
                        if holder is None or holder in wrapped:
                            continue
                        wrap_span(holder, "forward",
                                  lambda s: f"AIC::attn::{type(getattr(s, 'impl', s)).__name__}")
                        wrapped.add(t)
                        wrapped.add(holder)
                rec["attn_classes_wrapped"] = sorted(t.__name__ for t in wrapped)
                if not wrapped:
                    rec["errors"]["attn_hook"] = "no attention-ish module classes found in model"
            except Exception as e:
                rec["errors"]["attn_hook"] = f"{type(e).__name__}: {e}"[:200]

            from vllm import SamplingParams
            rec["probe_isl"] = args.isl
            prompt = {"prompt_token_ids": list(range(args.isl))}
            # warmup request: lazy JIT / autotune happen off-profile
            engine.add_request("warm0", {"prompt_token_ids": list(range(args.isl))},
                               SamplingParams(max_tokens=2, temperature=0))
            while engine.has_unfinished_requests():
                engine.step()
            try:
                _exp = torch._C._profiler._ExperimentalConfig(verbose=True)
            except Exception:
                _exp = None
            _prof_kw = dict(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            with_stack=True,
                            **({"experimental_config": _exp} if _exp else {}))
            # TWO phases, both profiled: prefill kernels (indexer quant/rope,
            # context attention) are DIFFERENT from decode's — excluding
            # prefill from the profile blinded the path-alignment gate
            engine.add_request("probe0", prompt, SamplingParams(max_tokens=2, temperature=0))
            with profile(**_prof_kw) as p_pre:
                engine.step()  # prefill
                torch.cuda.synchronize()
            with profile(**_prof_kw) as p:
                while engine.has_unfinished_requests():
                    engine.step()
                torch.cuda.synchronize()

            def fw_frames(ev):
                frames = []
                for fr in getattr(ev, "stack", None) or []:
                    if "/vllm/" in fr or "flash" in fr or "triton" in fr or "marlin" in fr:
                        frames.append(fr.split("site-packages/")[-1])
                return tuple(frames[:6])

            spans: dict = {}

            def collect(ev, acc, paths, seen):
                if id(ev) in seen:
                    return
                seen.add(id(ev))
                kerns = getattr(ev, "kernels", None) or []
                for kern in kerns:
                    a = acc.setdefault(kern.name, {"us": 0.0, "launches": 0})
                    a["us"] += kern.duration
                    a["launches"] += 1
                if kerns:
                    key = (ev.name, fw_frames(ev))
                    pth = paths.setdefault(key, {"kernels": set(), "launches": 0})
                    pth["kernels"].update(k.name.split("(")[0][:60] for k in kerns)
                    pth["launches"] += len(kerns)
                for c in getattr(ev, "cpu_children", None) or []:
                    collect(c, acc, paths, seen)

            # whole-run kernel table: span attribution goes blind when a
            # release moves kernel launches out of module.forward (vllm 0.29
            # sparse-MLA did exactly that) — and the CPU event tree goes blind
            # to kernels replayed inside CUDA graphs (no CPU launch event:
            # vllm 0.29 M3 runs its whole graph-safe attend+indexer under
            # full cudagraph, so serving evidence showed ZERO attend kernels).
            # Ground truth is the kineto DEVICE event stream — every executed
            # kernel appears there, graph-replayed or not; the CPU-tree walk
            # stays as fallback. This is the orphan source build_ops consumes
            # (decode_kernels).
            def device_kernel_table(prof):
                acc: dict = {}
                try:
                    from torch.autograd import DeviceType
                    for kev in prof.profiler.kineto_results.events():
                        try:
                            if kev.device_type() != DeviceType.CUDA:
                                continue
                        except Exception:
                            continue
                        name = kev.name()
                        dur = (kev.duration_ns() / 1e3
                               if hasattr(kev, "duration_ns") else kev.duration_us())
                        a = acc.setdefault(name, {"us": 0.0, "launches": 0})
                        a["us"] += dur
                        a["launches"] += 1
                except Exception:
                    return None
                return acc or None

            for phase_key, prof in (("prefill_kernels", p_pre), ("decode_kernels", p)):
                _acc = device_kernel_table(prof)
                if _acc is None:
                    _acc = {}
                    _seen: set = set()
                    for ev in prof.profiler.function_events:
                        collect(ev, _acc, {}, _seen)
                rec[phase_key] = [
                    {"kernel": n, "us": round(a["us"], 1), "launches": a["launches"]}
                    for n, a in sorted(_acc.items(), key=lambda kv: -kv[1]["us"])
                ]
            for ev in list(p_pre.profiler.function_events) + list(p.profiler.function_events):
                if ev.name.startswith("AIC::"):
                    slot = spans.setdefault(ev.name, {"calls": 0, "kernels": {}, "py_paths": {}})
                    slot["calls"] += 1
                    acc: dict = {}
                    paths: dict = {}
                    collect(ev, acc, paths, set())
                    for n, agg in acc.items():
                        k = slot["kernels"].setdefault(n, {"us": 0.0, "launches": 0})
                        k["us"] += agg["us"]
                        k["launches"] += agg["launches"]
                    for (opname, frames), pth in paths.items():
                        key = opname + (" <- " + " <- ".join(frames) if frames else "")
                        s = slot["py_paths"].setdefault(key, {"kernels": set(), "launches": 0})
                        s["kernels"].update(pth["kernels"])
                        s["launches"] += pth["launches"]
            for slot in spans.values():
                slot["kernels"] = dict(sorted(slot["kernels"].items(), key=lambda kv: -kv[1]["us"])[:10])
                slot["py_paths"] = {k: {"kernels": sorted(v["kernels"])[:6], "launches": v["launches"]}
                                    for k, v in list(slot["py_paths"].items())[:10]}
            rec["api_trace"] = spans
        except Exception:
            rec["errors"]["trace"] = traceback.format_exc()

    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    print("WROTE", args.out, "errors:", list(rec["errors"]))


if __name__ == "__main__":
    main()
