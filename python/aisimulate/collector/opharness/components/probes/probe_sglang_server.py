"""Backend-identity probe for sglang at tp>1 — the launch_server route.

probe_sglang.py stage 2 drives ModelRunner in-process (bench_one_batch
load_model) and is single-process by design, so it cannot host tp>1. This
route runs the REAL server (`python -m sglang.launch_server` with the golden
engine CLI, which already carries --tensor-parallel-size N) inside one
container that owns N GPUs, and takes its evidence through the server's own
surfaces:

  * identity  -> GET /get_server_info (resolved ServerArgs + scheduler state
                 + version; http_server.py:842 server_info @0.5.21)
  * kernels   -> POST /start_profile with profile_by_stage=true, num_steps=1
                 (the scheduler's torch.profiler exports one trace per stage per
                 TP rank: <profile_id>-TP-<r>-{prefill,decode}.trace.json.gz,
                 srt/utils/profile_utils.py:95-183, 355-375 @0.5.21), then one
                 /generate request; rank-0 traces are parsed into the same
                 prefill_kernels / decode_kernels tables probe_sglang writes.

What this route cannot see (and leaves None, flagged by identity_source):
quant_methods / param_dtypes / weight_samples (in-process module walk) and the
api span maps (prefill_api / decode_api) — path_diff still grades from the
device-stream kernel tables (orphans), exactly like cudagraph-replayed kernels.
"""
import argparse
import gzip
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import defaultdict


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _http(method: str, url: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode()
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        return txt


def _kernel_table_from_trace(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    acc: dict[str, dict] = defaultdict(lambda: {"calls": 0, "us": 0.0})
    for ev in tr.get("traceEvents", []):
        if ev.get("cat") != "kernel" or ev.get("ph") != "X":
            continue
        a = acc[ev["name"]]
        a["calls"] += 1
        a["us"] += float(ev.get("dur", 0.0))
    rows = [{"kernel": k, "calls": v["calls"], "us": round(v["us"], 1)} for k, v in acc.items() if v["us"] > 0]
    return sorted(rows, key=lambda r: -r["us"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-cli", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--isl", type=int, default=4096)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--launch-timeout", type=int, default=1500)
    args = ap.parse_args()
    rec: dict = {"model_path": args.model, "errors": {}, "identity_source": "launch_server(tp>1)",
                 "probe_route": "server", "probe_tp": args.tp}
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{os.path.basename(args.out)[:-5]}.profile")
    os.makedirs(out_dir, exist_ok=True)
    argv = shlex.split(args.engine_cli)
    pool = str(max(16384, 16 * args.isl))
    argv += ["--model-path", args.model, "--load-format", "dummy", "--trust-remote-code",
             "--disable-radix-cache", "--max-total-tokens", pool, "--max-running-requests", "32",
             "--host", "127.0.0.1", "--port", str(port)]
    if args.kv_dtype:
        argv += ["--kv-cache-dtype", args.kv_dtype]
    rec["engine_cli"] = args.engine_cli
    rec["launch_argv"] = argv
    rec["probe_isl"] = args.isl
    rec["probe_prefix_caching"] = False
    rec["probe_cuda_graph"] = True
    rec["probe_eager"] = False
    rec["probe_kv_cache_dtype"] = args.kv_dtype
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
        rec["probe_visible_gpus"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        rec["device_capability"] = None
    try:
        cfg = json.load(open(os.path.join(args.model, "config.json")))
        tc = cfg.get("text_config", cfg)
        qc = cfg.get("quantization_config") or tc.get("quantization_config")
        rec["model_config"] = {"architectures": cfg.get("architectures"),
                               "hf_quant_method": qc.get("quant_method") if isinstance(qc, dict) else None,
                               "num_hidden_layers": tc.get("num_hidden_layers")}
    except Exception:  # noqa: BLE001
        rec["errors"]["model_config"] = traceback.format_exc()[-800:]
    log_path = os.path.join(out_dir, "server.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    env.setdefault("SGLANG_TORCH_PROFILER_DIR", out_dir)
    proc = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", *argv], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    try:
        t0 = time.time()
        up = False
        while time.time() - t0 < args.launch_timeout:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"{base}/health", timeout=5).read()
                up = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        if not up:
            tail = open(log_path, errors="replace").read()[-3000:]
            rec["errors"]["launch"] = (f"server exited rc={proc.returncode}" if proc.poll() is not None
                                       else f"health timeout after {args.launch_timeout}s") + "\n" + tail
            return
        rec["server_startup_s"] = round(time.time() - t0, 1)
        info = _http("GET", f"{base}/get_server_info")
        if isinstance(info, dict):
            rec["internal_states"] = info.pop("internal_states", None)
            rec["sglang_version"] = info.get("version")
            rec["server_args_resolved"] = info
            rec["attn_backend"] = info.get("attention_backend")
            rec["model_runner.prefill_attention_backend_str"] = info.get("prefill_attention_backend") or info.get("attention_backend")
            rec["model_runner.decode_attention_backend_str"] = info.get("decode_attention_backend") or info.get("attention_backend")
            rec["kv_cache_resolved"] = {"server_arg": info.get("kv_cache_dtype")}
            rec["effective_moe_runner_backend"] = info.get("moe_runner_backend")
        prompt = list(range(1, args.isl + 1))
        gen = {"input_ids": prompt, "sampling_params": {"max_new_tokens": 2, "temperature": 0}}
        _http("POST", f"{base}/generate", gen, timeout=900)  # warm-up (JIT, graph capture already done at startup)
        pid = "aic"
        _http("POST", f"{base}/start_profile", {"output_dir": out_dir, "num_steps": 1, "activities": ["CPU", "CUDA"],
                                                  "profile_by_stage": True, "profile_id": pid}, timeout=120)
        _http("POST", f"{base}/generate", {"input_ids": prompt, "sampling_params": {"max_new_tokens": 4, "temperature": 0}},
              timeout=900)
        want = {st: os.path.join(out_dir, f"{pid}-TP-0-{st}.trace.json.gz") for st in ("prefill", "decode")}
        t1 = time.time()
        while time.time() - t1 < 300 and not all(os.path.exists(p) for p in want.values()):
            time.sleep(3)
        try:
            _http("POST", f"{base}/stop_profile", {}, timeout=60)
        except Exception:  # noqa: BLE001
            pass
        rec["trace_files"] = sorted(os.listdir(out_dir))
        trace: dict = {}
        for st, p in want.items():
            if os.path.exists(p):
                time.sleep(2)  # export finishes before the file is closed; give it a beat
                trace[f"{st}_kernels"] = _kernel_table_from_trace(p)
            else:
                rec["errors"][f"trace_{st}"] = f"no {os.path.basename(p)} within 300s; files={rec['trace_files']}"
        rec["trace"] = trace
    except Exception:
        rec["errors"]["probe"] = traceback.format_exc()[-3000:]
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        log.close()
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("WROTE", args.out, "errors:", list(rec["errors"]))


if __name__ == "__main__":
    main()
