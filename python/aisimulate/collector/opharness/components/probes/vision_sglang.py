"""Vision-tower evidence probe for sglang (runs INSIDE the sglang image).

Loads the dummy checkpoint exactly like probe_sglang.py stage 2 (bench_one_batch
load_model with the golden engine CLI), then pushes one synthetic image through
``model.visual`` — the module serving runs for every image item
(srt/models/qwen3_vl.py:1459 ``self.visual(pixel_values, grid_thw=grid_thw)``
@0.5.21) — under torch.profiler and writes the kernel table as
``profile_run_kernels`` into the sidecar given by --out.
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_common import profile_vision_forward, synthetic_image_inputs, write_sidecar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--isl", type=int, default=4096)
    ap.add_argument("--engine-cli-file", default=None, help="file holding the golden engine CLI (as plan engine_cli)")
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--override", default=None)
    args = ap.parse_args()
    rec: dict = {"vision_probe": {"backend": "sglang", "model_path": args.model}, "errors": {}}
    try:
        import torch
        from sglang.srt.server_args import PortArgs, ServerArgs

        graph_off = ["disable_cuda_graph"]
        if args.engine_cli_file:
            import shlex
            argv = shlex.split(open(args.engine_cli_file).read().strip())
            argv += ["--model-path", args.model, "--load-format", "dummy", "--trust-remote-code",
                     "--disable-radix-cache", "--max-total-tokens", str(max(16384, 16 * args.isl)),
                     "--max-running-requests", "32"]
            argv += [f"--{f.replace('_', '-')}" for f in graph_off]
            if args.kv_dtype:
                argv += ["--kv-cache-dtype", args.kv_dtype]
            if args.quantization:
                argv += ["--quantization", args.quantization]
            if args.override:
                argv += ["--json-model-override-args", args.override]
            cli_parser = argparse.ArgumentParser()
            ServerArgs.add_cli_args(cli_parser)
            ns, unknown = cli_parser.parse_known_args(argv)
            rec["vision_probe"]["engine_cli_unknown_args"] = unknown
            sa = ServerArgs.from_cli_args(ns)
        else:
            sa = ServerArgs(model_path=args.model, load_format="dummy", trust_remote_code=True, tp_size=1,
                            disable_radix_cache=True, max_running_requests=32,
                            max_total_tokens=max(16384, 16 * args.isl), disable_cuda_graph=True,
                            **({"json_model_override_args": args.override} if args.override else {}),
                            **({"quantization": args.quantization} if args.quantization else {}),
                            **({"kv_cache_dtype": args.kv_dtype} if args.kv_dtype else {}))
        try:
            from sglang.bench_one_batch import load_model
        except ModuleNotFoundError:
            from sglang.benchmark.one_batch import load_model
        from sglang.srt.layers.moe import initialize_moe_config
        import inspect as _inspect
        if _inspect.signature(initialize_moe_config).parameters:
            initialize_moe_config(sa)
        else:
            from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
            publish(sa, role="scheduler",
                    ranks=SpawnRanks(world_rank=spawn_world_rank(sa, tp_rank=0, pp_rank=0), gpu_id=0))
            initialize_moe_config()
        for _fn in ("initialize_fp8_gemm_config", "initialize_fp4_gemm_config"):
            try:
                _f = getattr(__import__("sglang.benchmark.one_batch", fromlist=[_fn]), _fn)
                _f(sa) if _inspect.signature(_f).parameters else _f()
            except Exception as _e:  # noqa: BLE001
                rec["vision_probe"].setdefault("init_warnings", []).append(f"{_fn}: {type(_e).__name__}")
        ret, _tok = load_model(sa, PortArgs.init_new(sa), 0, 0)
        model_runner = getattr(ret, "torch_runner", ret)
        model = model_runner.model
        rec["vision_probe"]["model_class"] = type(model).__qualname__
        visual = getattr(model, "visual", None)
        if visual is None:
            raise RuntimeError(f"{type(model).__qualname__} has no .visual module (not a VL checkpoint?)")
        rec["vision_probe"]["vision_module"] = type(visual).__qualname__
        _proc, inputs = synthetic_image_inputs(args.model)
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]
        rec["vision_probe"].update({"image_processor": type(_proc).__name__,
                                    "patches": int(pixel_values.shape[0]), "grid_thw": grid_thw.tolist()})
        device = torch.device(model_runner.device if isinstance(model_runner.device, str) else "cuda:0")
        pv = pixel_values.to(device)
        gt = grid_thw.to(device)

        def fwd():
            with torch.inference_mode():
                return model.visual(pv, grid_thw=gt)

        rec["profile_run_kernels"] = profile_vision_forward(fwd)
    except Exception:
        rec["errors"]["vision"] = traceback.format_exc()[-3000:]
    write_sidecar(args.out, rec, __file__)


if __name__ == "__main__":
    main()
