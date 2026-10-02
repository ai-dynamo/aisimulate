"""Vision-tower evidence probe for TensorRT-LLM (runs INSIDE the trtllm image,
with TLLM_WORKER_USE_SINGLE_PROCESS=1 like probe_trtllm.py so the model lives in
this process).

Builds the llmapi LLM from the golden engine yaml like probe_trtllm.py, finds the
torch model by object-graph search, and pushes one synthetic image through
``model.mm_encoder.visual`` — the Qwen3VisionModel that serving's aggregated and
mm-encoder-only paths both run (_torch/models/modeling_qwen3vl.py:1206
``self.visual(pixel_values, grid_thw=grid_thw)``, :1358 mm_encoder owned by the
normal worker @1.3.0rc29) — under torch.profiler; writes ``profile_run_kernels``
into the --out sidecar.
"""
import argparse
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_common import profile_vision_forward, synthetic_image_inputs, write_sidecar  # noqa: E402


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
            except Exception:  # noqa: BLE001
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--isl", type=int, default=4096)
    ap.add_argument("--engine-yaml", default=None)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    rec: dict = {"vision_probe": {"backend": "trtllm", "model_path": args.model}, "errors": {}}
    try:
        import torch
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        kwargs = dict(model=args.model, load_format="dummy", trust_remote_code=args.trust_remote_code,
                      kv_cache_config=KvCacheConfig(max_tokens=max(16384, args.isl + 256), enable_block_reuse=False,
                                                    **({"dtype": args.kv_dtype} if args.kv_dtype else {})),
                      max_batch_size=8, max_seq_len=args.isl + 64)
        if args.engine_yaml:
            import yaml
            eng = {k: v for k, v in (yaml.safe_load(open(args.engine_yaml)) or {}).items() if v is not None}
            eng.pop("backend", None)
            kvc = dict(eng.pop("kv_cache_config", {}) or {})
            kvc.pop("free_gpu_memory_fraction", None)
            kvc["max_tokens"] = max(16384, args.isl + 256)
            kvc["enable_block_reuse"] = False
            if args.kv_dtype:
                kvc["dtype"] = args.kv_dtype
            kwargs["kv_cache_config"] = KvCacheConfig(**kvc)
            kwargs.update(eng)
            if (kwargs.get("max_seq_len") or 0) < args.isl + 64:
                kwargs["max_seq_len"] = args.isl + 64
        dropped = []
        for _ in range(8):
            try:
                llm = LLM(**kwargs)
                break
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if not m or m.group(1) not in kwargs:
                    raise
                dropped.append(m.group(1))
                kwargs.pop(m.group(1))
        else:
            raise RuntimeError("LLM kwargs never converged")
        rec["vision_probe"]["engine_yaml_unknown_args"] = dropped
        model = find_torch_model(llm)
        if model is None:
            raise RuntimeError("no in-process torch model (subprocess executor?)")
        rec["vision_probe"]["model_class"] = type(model).__qualname__
        enc = getattr(model, "mm_encoder", None)
        visual = getattr(enc, "visual", None) if enc is not None else None
        if visual is None:
            # fall back to any submodule whose class says Vision and takes (pixel_values, grid_thw)
            for name, mod in model.named_modules():
                if type(mod).__name__.endswith("VisionModel") and hasattr(mod, "forward"):
                    visual = mod
                    break
        if visual is None:
            raise RuntimeError(f"{type(model).__qualname__} exposes no vision encoder (mm_encoder.visual)")
        rec["vision_probe"]["vision_module"] = type(visual).__qualname__
        _proc, inputs = synthetic_image_inputs(args.model)
        pixel_values, grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
        rec["vision_probe"].update({"image_processor": type(_proc).__name__,
                                    "patches": int(pixel_values.shape[0]), "grid_thw": grid_thw.tolist()})
        p0 = next(visual.parameters())
        pv = pixel_values.to(device=p0.device, dtype=p0.dtype)
        gt = grid_thw.to(p0.device)

        def fwd():
            with torch.inference_mode():
                return visual(pv, grid_thw=gt)

        rec["profile_run_kernels"] = profile_vision_forward(fwd)
    except Exception:
        rec["errors"]["vision"] = traceback.format_exc()[-3000:]
    write_sidecar(args.out, rec, __file__)


if __name__ == "__main__":
    main()
