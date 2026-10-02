#!/usr/bin/env python3
"""Component: e2e admission — SDK prediction vs live measurement, same config.

onboard_model step 10. One question: for the golden deployment of a
checkpoint (the exact config `cli generate` rendered — tp, quant modes,
framework, version) does the SDK's prediction agree with what the deployment
measured, within a declared tolerance?

Inputs are explicit files, never re-derived from prose:
  --golden <dir>        a golden render (generator_config.yaml + run.sh) — the
                        model path, system_name and parallelism come from HERE
  --measurement <json>  what the live deployment measured at ONE operating
                        point. Contract (all numbers per request, milliseconds):
                            {"isl": 4096, "osl": 512, "batch_size": 32,
                             "ttft_ms": 812.0, "tpot_ms": 27.5,
                             "source": "<who/what produced it>"}
                        aiperf/genai-perf profile exports are accepted too
                        (time_to_first_token.avg / inter_token_latency.avg,
                        input/output sequence length and concurrency fields).
  --backend/--version   the probe pins (framework identity of the measurement)
  --db-version          the SDK perf-data version to predict with; default =
                        newest available for (system, backend). Recorded in the
                        verdict: a gap between measured framework version and
                        perf-data version is part of the result, not hidden.

Verdict = every compared metric within --tolerance (relative, default 0.25):
  aligned | diverged | not-comparable (system has no perf data, model not
  buildable, measurement incomplete). Written to
  results/<sm>/e2e/<framework>-<version>/<org>_<name>[__b<bs>_i<isl>_o<osl>].json

What this component does NOT do: run the deployment. The measurement comes
from the golden run.sh's own benchmark (FPM benchmark mode) executed on a GPU
that matches an SDK system entry — the probe box (H20 proxied as h200_sxm) is
not such a GPU, so admission campaigns run elsewhere (see TODO.md).

Usage:
  python3 e2e_align.py --golden <dir> --measurement m.json --backend vllm --version 0.29.0 [--sm sm90]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE")
            or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
            or Path.cwd())

METRICS = ("ttft_ms", "tpot_ms")


# ------------------------------------------------------------- measurement
def load_measurement(path: Path) -> dict:
    """Normalize the measurement file to the contract; aiperf/genai-perf exports
    are mapped field by field (avg values, milliseconds)."""
    raw = json.loads(path.read_text())
    m = {"source": raw.get("source") or path.name}
    if "ttft_ms" in raw or "tpot_ms" in raw:  # our contract
        for k in ("isl", "osl", "batch_size", "ttft_ms", "tpot_ms"):
            if k in raw:
                m[k] = raw[k]
        return m

    def _avg(key):  # aiperf / genai-perf: {"metric": {"avg": x, "unit": "ms"}}
        v = raw.get(key)
        if isinstance(v, dict):
            val = v.get("avg")
            unit = str(v.get("unit", "ms")).lower()
            if val is not None and unit in ("ns", "us", "s"):
                val = val / 1e6 if unit == "ns" else val / 1e3 if unit == "us" else val * 1e3
            return val
        return v
    ttft, itl = _avg("time_to_first_token"), _avg("inter_token_latency")
    if ttft is not None:
        m["ttft_ms"] = ttft
    if itl is not None:
        m["tpot_ms"] = itl
    for src, dst in (("input_sequence_length", "isl"), ("output_sequence_length", "osl"),
                     ("request_concurrency", "batch_size"), ("concurrency", "batch_size")):
        v = _avg(src) if isinstance(raw.get(src), dict) else raw.get(src)
        if v is not None:
            m[dst] = int(round(v))
    for src, dst in (("isl", "isl"), ("osl", "osl"), ("batch_size", "batch_size")):  # mixed files
        if dst not in m and src in raw:
            m[dst] = raw[src]
    return m


# ------------------------------------------------------------------ golden
def load_golden(golden: Path) -> dict:
    """model path, system, tp/pp/dp from the generator's own config; the run.sh
    engine line is kept as the identity of what was deployed."""
    sub = golden
    if not (golden / "generator_config.yaml").exists():
        sub = next((d for d in golden.iterdir() if (d / "generator_config.yaml").exists()), None)
        if sub is None:
            raise SystemExit(f"no generator_config.yaml under {golden}")
    cfg = yaml.safe_load((sub / "generator_config.yaml").read_text())
    svc, k8s = cfg.get("ServiceConfig") or {}, cfg.get("K8sConfig") or {}
    run_sh = next((p for p in (sub / "run.sh", sub / "run_0.sh") if p.exists()), None)
    engine_line = None
    tp = pp = dp = 1
    if run_sh is not None:
        text = run_sh.read_text()
        mline = re.search(r"engine_command=\((.*?)\)\n", text, re.S) or re.search(r"python3 -m dynamo\.\w+.*", text)
        engine_line = " ".join((mline.group(1) if mline.lastindex else mline.group(0)).split()) if mline else None
        if engine_line:
            def _flag(name, default):
                m = re.search(rf"--{name}(?:=|\s+)(\d+)", engine_line)
                return int(m.group(1)) if m else default
            tp = _flag("tensor-parallel-size", _flag("tp-size", _flag("tp", 1)))
            pp = _flag("pipeline-parallel-size", _flag("pp-size", 1))
            dp = _flag("data-parallel-size", _flag("dp-size", 1))
    return {"model_path": svc.get("model_path") or svc.get("model_name"),
            "system": k8s.get("system_name"), "tp": tp, "pp": pp, "dp": dp,
            "engine_line": engine_line, "golden_dir": str(sub)}


# -------------------------------------------------------------- prediction
def predict(model_path: str, system: str, backend: str, db_version: str | None,
            tp: int, pp: int, isl: int, osl: int, batch_size: int) -> dict:
    """SDK static prediction for one operating point. Returns ttft_ms / tpot_ms
    plus the per-op breakdowns the SDK exposes, and the perf-data version used."""
    from aisimulate.sdk import config, perf_database
    from aisimulate.sdk.backends.factory import get_backend
    from aisimulate.sdk.inference_session import InferenceSession
    from aisimulate.sdk.models import get_model

    version = db_version or perf_database.get_latest_database_version(system, backend)
    database = perf_database.get_database(system=system, backend=backend, version=version)
    model = get_model(model_path, config.ModelConfig(tp_size=tp, pp_size=pp), backend)
    session = InferenceSession(model, database, get_backend(backend))
    rt = config.RuntimeConfig(batch_size=batch_size, beam_width=1, isl=isl, osl=osl)
    summary = session.run_static(runtime_config=rt, mode="static", stride=32)
    df = summary.get_summary_df()
    out = {"db_version": str(version), "system": system,
           "ttft_ms": float(df.loc[0, "ttft"]) if "ttft" in df.columns else None,
           "tpot_ms": float(df.loc[0, "tpot"]) if "tpot" in df.columns else None}
    for attr, key in (("_context_latency_dict", "context_breakdown_ms"),
                      ("_generation_latency_dict", "generation_breakdown_ms")):
        d = getattr(summary, attr, None)
        if isinstance(d, dict) and d:
            out[key] = {str(k): round(float(v), 4) for k, v in d.items()}
    return out


# ------------------------------------------------------------------- grade
def grade(measured: dict, predicted: dict, tolerance: float) -> dict:
    """Pure: per-metric relative error vs tolerance -> verdict."""
    metrics = {}
    for k in METRICS:
        m, p = measured.get(k), predicted.get(k)
        if m is None or p is None:
            metrics[k] = {"measured": m, "predicted": p, "rel_error": None, "within": None}
            continue
        rel = (p - m) / m if m else None
        metrics[k] = {"measured": m, "predicted": p, "rel_error": None if rel is None else round(rel, 4),
                      "within": None if rel is None else abs(rel) <= tolerance}
    compared = [v["within"] for v in metrics.values() if v["within"] is not None]
    if not compared:
        verdict = "not-comparable"
    else:
        verdict = "aligned" if all(compared) else "diverged"
    return {"verdict": verdict, "tolerance": tolerance, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--measurement", type=Path, required=True)
    ap.add_argument("--backend", required=True, choices=["vllm", "sglang", "trtllm"])
    ap.add_argument("--version", required=True, help="framework version the measurement ran on")
    ap.add_argument("--db-version", default=None, help="SDK perf-data version (default: newest for system/backend)")
    ap.add_argument("--system", default=None, help="override the golden's system_name")
    ap.add_argument("--tolerance", type=float, default=0.25)
    ap.add_argument("--sm", default="sm90")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    golden = load_golden(args.golden)
    measured = load_measurement(args.measurement)
    system = args.system or golden["system"]
    missing = [k for k in ("isl", "osl", "batch_size") if measured.get(k) is None]
    result = {"repo": golden["model_path"], "backend": args.backend, "framework_version": args.version,
              "system": system, "golden": golden, "measurement": measured,
              "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if missing or not system:
        result.update({"verdict": "not-comparable",
                       "reason": f"measurement lacks {missing}" if missing else "golden has no system_name"})
    else:
        try:
            pred = predict(golden["model_path"], system, args.backend, args.db_version,
                           golden["tp"], golden["pp"], int(measured["isl"]), int(measured["osl"]),
                           int(measured["batch_size"]))
            result["prediction"] = pred
            result.update(grade(measured, pred, args.tolerance))
            if str(pred["db_version"]) != str(args.version):
                result["caveat"] = (f"perf data {pred['db_version']} predicts a {args.version} deployment "
                                    f"(no {args.version} perf data for {system}/{args.backend})")
        except Exception as e:  # SDK cannot build/predict this config: a terminal fact, recorded
            result.update({"verdict": "not-comparable", "reason": f"{type(e).__name__}: {str(e)[:300]}"})
    tag = (golden["model_path"] or "unknown").replace("/", "_")
    point = f"__b{measured.get('batch_size')}_i{measured.get('isl')}_o{measured.get('osl')}"
    out = args.out or HARNESS / "results" / args.sm / "e2e" / f"{args.backend}-{args.version}" / f"{tag}{point}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"{result['verdict']:15s} {golden['model_path']} {args.backend}-{args.version} on {system}: "
          + ", ".join(f"{k} meas={v['measured']} pred={v['predicted']} err={v['rel_error']}"
                      for k, v in (result.get("metrics") or {}).items())
          + (f" [{result['reason']}]" if result.get("reason") else "")
          + (f" caveat: {result['caveat']}" if result.get("caveat") else ""))
    print(f"wrote {out}")
    return 0 if result["verdict"] == "aligned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
