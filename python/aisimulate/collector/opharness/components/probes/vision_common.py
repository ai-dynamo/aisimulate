"""Shared pieces of the vision-tower evidence probes (vision_sglang.py / vision_trtllm.py).

Why a separate script: the plan probes are text-only (token-id prompts), so a
VL checkpoint's vision encoder never runs there and the encoder_attn gate has no
serving evidence on sglang / trtllm (vLLM gets it for free from its own
multimodal profile_run). The probe code hash is part of every run's execution
fingerprint, so this lives OUTSIDE probe_<backend>.py: it adds a third evidence
table (``profile_run_kernels`` — same name/phase as vLLM's) through a sidecar
``archive/raw/<id>.vision.json`` that build_records merges, without
invalidating the text-probe evidence of every other run.
"""
import hashlib
import json
import pathlib
import sys


def synthetic_image_inputs(model_dir: str, side: int = 448):
    """One synthetic RGB image through the checkpoint's OWN image processor
    (preprocessor_config.json travels with the dummy dir): returns the processor
    output dict (Qwen-VL family: pixel_values [patches, C*T*ph*pw] float32,
    image_grid_thw [N, 3])."""
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(model_dir)
    rng = np.random.default_rng(1234)
    img = Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8))
    out = proc(images=[img], return_tensors="pt")
    return proc, out


def kernel_table(prof) -> list[dict]:
    rows = []
    for e in prof.key_averages():
        dt = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if dt > 0:
            rows.append({"kernel": e.key, "calls": e.count, "us": round(dt, 1)})
    return sorted(rows, key=lambda r: -r["us"])


def profile_vision_forward(fn, warmup: int = 1) -> list[dict]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return kernel_table(p)


def write_sidecar(out: str, rec: dict, script: str) -> None:
    rec.setdefault("vision_probe", {})["script_sha"] = hashlib.sha256(
        pathlib.Path(script).read_bytes()).hexdigest()[:12]
    rec["vision_probe"]["common_sha"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:12]
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("WROTE", out, "kernels:", len(rec.get("profile_run_kernels") or []), "errors:", list(rec.get("errors", {})),
          file=sys.stderr)
