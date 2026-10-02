"""Backend-identity probe for TensorRT-LLM at tp>1 — the trtllm-serve route.

probe_trtllm.py keeps the llmapi LLM in-process (TLLM_WORKER_USE_SINGLE_PROCESS=1),
which only exists for tp1. For tp>1 this route runs the real server
(`trtllm-serve <model> --backend pytorch --tp_size N --extra_llm_api_options
<golden agg_config.yaml>`) inside one container that owns N GPUs and takes the
evidence through the server's own surfaces:

  * kernels  -> POST /start_profile (serve/openai_server.py:1450; body
                output_dir / num_steps, serve/openai_protocol.py:2372
                StartProfileRequest) schedules the executor's torch.profiler on
                EVERY rank for the next num_steps iterations and exports
                ``trtllm-trace-<profile_id>-rank-<r>.json`` chrome traces
                (_torch/pyexecutor/profiling.py:480 @1.3.0rc29). Two windows,
                one iteration (prefill) and two iterations (prefill+decode), are
                diffed per kernel exactly like probe_trtllm.py does in-process.
  * identity -> the resolved engine yaml the server was given plus /v1/models;
                the in-process module walk (quant_methods, param_dtypes,
                model_class) is not reachable from outside, left None and
                flagged by identity_source.
"""
import argparse
import glob
import json
import os
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


def _kernel_table_from_trace(path: str) -> dict[str, dict]:
    with open(path) as f:
        tr = json.load(f)
    acc: dict[str, dict] = defaultdict(lambda: {"launches": 0, "us": 0.0})
    for ev in (tr.get("traceEvents", tr) if isinstance(tr, dict) else tr):
        if not isinstance(ev, dict) or ev.get("cat") != "kernel" or ev.get("ph") != "X":
            continue
        a = acc[ev["name"]]
        a["launches"] += 1
        a["us"] += float(ev.get("dur", 0.0))
    return acc


def _newest_rank0(out_dir: str, after: float) -> str | None:
    cands = [p for p in glob.glob(os.path.join(out_dir, "trtllm-trace-*-rank-0.json")) if os.path.getmtime(p) >= after]
    return max(cands, key=os.path.getmtime) if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-yaml", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--isl", type=int, default=4096)
    ap.add_argument("--kv-dtype", default=None)
    ap.add_argument("--launch-timeout", type=int, default=1500)
    args = ap.parse_args()
    rec: dict = {"model_path": args.model, "errors": {}, "identity_source": "trtllm-serve(tp>1)",
                 "probe_route": "server", "probe_tp": args.tp, "probe_isl": args.isl,
                 "probe_prefix_caching": False, "probe_kv_cache_dtype": args.kv_dtype}
    import yaml
    eng = {k: v for k, v in (yaml.safe_load(open(args.engine_yaml)) or {}).items() if v is not None}
    rec["engine_yaml"] = dict(eng)
    eng.pop("backend", None)
    kvc = dict(eng.pop("kv_cache_config", {}) or {})
    kvc["enable_block_reuse"] = False
    if args.kv_dtype:
        kvc["dtype"] = args.kv_dtype
    eng["kv_cache_config"] = kvc
    eng["load_format"] = "dummy"
    eng.pop("tensor_parallel_size", None)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"{os.path.basename(args.out)[:-5]}.profile")
    os.makedirs(out_dir, exist_ok=True)
    serve_yaml = os.path.join(out_dir, "serve_options.yaml")
    yaml.safe_dump(eng, open(serve_yaml, "w"), sort_keys=False)
    rec["serve_options"] = eng
    cmd = ["trtllm-serve", args.model, "--backend", "pytorch", "--tp_size", str(args.tp), "--host", "127.0.0.1",
           "--port", str(port), "--trust_remote_code", "--extra_llm_api_options", serve_yaml,
           "--max_seq_len", str(max(int(eng.get("max_seq_len") or 0), args.isl + 64))]
    rec["launch_argv"] = cmd
    try:
        import torch
        rec["device_capability"] = "sm%d%d" % torch.cuda.get_device_capability()
        rec["probe_visible_gpus"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        rec["device_capability"] = None
    try:
        import tensorrt_llm
        rec["trtllm_version"] = tensorrt_llm.__version__
    except Exception:  # noqa: BLE001
        pass
    log_path = os.path.join(out_dir, "server.log")
    log = open(log_path, "w")
    env = dict(os.environ)
    env["TLLM_TORCH_PROFILER_DIR"] = out_dir
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
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
        try:
            models = _http("GET", f"{base}/v1/models")
            rec["served_model"] = (models.get("data") or [{}])[0].get("id") if isinstance(models, dict) else None
        except Exception:  # noqa: BLE001
            pass
        prompt = list(range(1, args.isl + 1))

        def completion(max_tokens: int):
            return _http("POST", f"{base}/v1/completions",
                         {"model": rec.get("served_model") or os.path.basename(args.model.rstrip("/")),
                          "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}, timeout=900)

        completion(2)  # warm-up
        tables = {}
        for label, steps, max_tokens in (("p1", 1, 1), ("p2", 2, 2)):
            mark = time.time()
            _http("POST", f"{base}/start_profile", {"output_dir": out_dir, "num_steps": steps}, timeout=120)
            completion(max_tokens)
            path = None
            t1 = time.time()
            while time.time() - t1 < 300 and path is None:
                path = _newest_rank0(out_dir, mark)
                if path is None:
                    time.sleep(3)
            if path is None:
                rec["errors"][f"trace_{label}"] = f"no rank-0 trace within 300s; files={sorted(os.listdir(out_dir))}"
                break
            time.sleep(3)
            tables[label] = _kernel_table_from_trace(path)
            rec.setdefault("trace_files", []).append(os.path.basename(path))
        if "p1" in tables:
            pre = tables["p1"]
            rec["prefill_kernels"] = sorted(({"kernel": k, "launches": v["launches"], "us": round(v["us"], 1)}
                                             for k, v in pre.items()), key=lambda r: -r["us"])
        if "p1" in tables and "p2" in tables:
            dec = []
            for k, v in tables["p2"].items():
                d_l = v["launches"] - tables["p1"].get(k, {}).get("launches", 0)
                if d_l > 0:
                    dec.append({"kernel": k, "launches": d_l,
                                "us": round(max(v["us"] - tables["p1"].get(k, {}).get("us", 0.0), 0.0), 1)})
            rec["decode_kernels"] = sorted(dec, key=lambda r: -r["us"])
            rec["kernels_visible_in_process"] = bool(rec.get("prefill_kernels"))
    except Exception:
        rec["errors"]["probe"] = traceback.format_exc()[-3000:]
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=90)
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
