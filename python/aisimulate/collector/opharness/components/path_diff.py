#!/usr/bin/env python3
"""Component: execution-path alignment — collector op path vs serving,
PROFILER-MEASURED ON BOTH SIDES (owner rule 2026-09-19: selector/class names
are leads, never the verdict; only captured kernels count).

Two modes:

  --capture   (runs INSIDE the framework container, same interpreter as the
              collector) Execute a collector command under torch.profiler and
              dump the executed CUDA kernel names to JSON. torch.profiler is
              chosen because it is what the serving probes use — same capture,
              same normalization, so the two sides are directly comparable.
              nsys is the escalation tool for cases torch.profiler cannot see.

  --diff      (host) Translate BOTH kernel sets — the collector capture and
              the serving probe's records.jsonl entry — through the ONE
              vocabulary (kernel_taxonomy_<sm>.yaml, AIS_SM, default sm90) and report the verdict:
                aligned   collector's canonical backend set matches serving's
                          for the op's kernels (subset relation: the collector
                          exercises one op, serving runs the whole model)
                diverged  collector executed backend families serving never
                          did (the wrong-path signal), or vice versa for the
                          op under test

Usage (capture, in-container):
  python3 path_diff.py --capture --out /out/cap.json -- \
      python3 collector/vllm/collect_mla_module.py --mode context ...

Usage (diff, host):
  AIS_PROBE_WORKSPACE=<ws> python3 path_diff.py --diff \
      --capture-file facts/pathdiff/cap.json --repo deepseek-ai/DeepSeek-V3.2 \
      --framework vllm --version 0.29.0 [--op-hint 'attn|mla|sparse']
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE")
            or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
            or Path.cwd())


# --------------------------------------------------------------------------
# capture (in-container; only stdlib + torch)

def capture(argv: list[str], out: str) -> int:
    import runpy
    import torch
    from torch.profiler import ProfilerActivity, profile

    script, args = argv[0], argv[1:]
    sys.argv = [script] + args
    err = None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                err = f"exit={e.code}"
        except Exception as e:  # capture is still evidence on failure
            err = f"{type(e).__name__}: {e}"
        torch.cuda.synchronize()
    kernels = sorted({
        ev.key for ev in prof.key_averages()
        if (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) > 0
        and not ev.key.startswith(("aten::", "Memcpy", "Memset", "cuda", "Cuda"))
    })
    json.dump({"cmd": argv, "error": err, "kernels": kernels},
              open(out, "w"), indent=1)
    print(f"path_diff capture: {len(kernels)} kernels -> {out}"
          + (f" (RUN ERROR: {err})" if err else ""))
    return 0


# --------------------------------------------------------------------------
# diff (host; shares the probes' normalization + taxonomy)

def _load_labeler():
    sys.path.insert(0, str(HERE))
    from probe_driver import label_kernels, normalize_kernel  # noqa: E402
    return label_kernels, normalize_kernel


def _record_from_raw(raw_path: str) -> dict:
    """Turn a raw probe JSON that is NOT in the plan (an A/B run such as a
    different --isl) into a minimal serving record with the same shape
    build_records emits, so a verdict can be computed against explicit
    evidence and the report names that file."""
    from probe_driver import build_ops  # noqa: E402
    f = json.loads(Path(raw_path).read_text())
    ops, orphans = build_ops(f)
    return {"id": f"raw:{Path(raw_path).name}", "ops": ops, "orphan_kernels": orphans,
            "runtime": {"tp": None, "kv_cache_dtype": f.get("probe_kv_cache_dtype"),
                        "isl": f.get("probe_isl"), "prefix_caching": f.get("probe_prefix_caching")}}


def _kv_equiv(a, b):
    # auto resolves to the model dtype (bf16 for these models); treat the
    # unquantized aliases as one class, fp8 as its own.
    norm = lambda x: "fp8" if (x or "").startswith("fp8") else "unquant"
    return norm(a) == norm(b)


def diff(capture_file: str, repo: str, framework: str, version: str,
         op_hint: str | None, save: str | None = None, kv_dtype: str | None = None,
         isl: int | None = None, serving_raw: str | None = None) -> int:
    label_kernels, normalize_kernel = _load_labeler()
    cap = json.loads(Path(capture_file).read_text())
    # custom-op LAUNCHERS shadow their kernels under a second name
    # (_vllm_fa3_C::fwd wraps flash::FlashAttnFwdSm90) — same exclusion
    # build_ops applies to serving orphans, applied here to both sides
    # trtllm's custom ops surface as `trtllm::<op>` launchers over
    # `tensorrt_llm::_v1::kernels::...` kernels (causal_conv1d_fwd vs
    # causal_conv1d::causal_conv1d_fwd_kernel) — same exclusion
    _launcher = re.compile(r"^(_\w*C\w*|sglang|sgl_kernel|triton_|vllm|trtllm)::(?!.*_kernel)")
    def _is_launcher(name: str) -> bool:
        return bool(_launcher.match(name)) and "kernel" not in name.split("::")[-1].lower()
    col_kernels = sorted({n for k in cap["kernels"]
                          if (n := normalize_kernel(k)) and not _is_launcher(n)})
    col_backends, col_unmatched = label_kernels(col_kernels)

    candidates = []
    if serving_raw:  # explicit evidence file (A/B run outside the plan)
        candidates = [_record_from_raw(serving_raw)]
    else:
        for line in (ROOT / "archive" / "records.jsonl").open():
            r = json.loads(line)
            if not (r["target"].get("repo") == repo and r["runtime"]["backend"] == framework
                    and r["runtime"].get("version") == version
                    and (r.get("outcome") or {}).get("status") == "ok"):
                continue
            if kv_dtype is not None and not _kv_equiv(r["runtime"].get("kv_cache_dtype"), kv_dtype):
                continue
            if isl is not None and r["runtime"].get("isl") != isl:
                continue
            candidates.append(r)
    # Several records can match one (repo, framework, version, kv) — tp/ep
    # variants, re-probes, older runs. Pick by EVIDENCE QUALITY, never by file
    # order: cache-cold prefill (prefix_caching False) beats a cached
    # residual, a record with both phase tables beats a decode-only one, a
    # known isl beats an unknown one. Found 2026-09-23: "last match" picked a
    # decode-only Qwen3.5 record over the cache-cold one and produced a false
    # collector-only `flashinfer` (GDN prefill) verdict.
    def _quality(r):
        rt = r["runtime"]
        return (rt.get("prefix_caching") is False,
                bool(r.get("ops")) and any((o.get("phase") == "prefill") for o in r.get("ops") or []),
                rt.get("isl") is not None)
    candidates.sort(key=_quality)
    serving = candidates[-1] if candidates else None
    if len(candidates) > 1:  # several serving configs match (tp/ep/kv variants): say which one won
        print(f"[note] {len(candidates)} serving records match; using {serving['id']} "
              f"(tp={serving['runtime'].get('tp')} kv={serving['runtime'].get('kv_cache_dtype')}); "
              f"others: {[c['id'] for c in candidates[:-1]]}", file=sys.stderr)
    if serving is None:
        _hint = f" kv={kv_dtype}" if kv_dtype else ""
        print(f"[no-serving-record] {repo} {framework}-{version}{_hint} — probe that config first")
        return 2
    srv_backends: set = set()
    srv_kernels: set = set()
    orphans = serving.get("orphan_kernels") or []
    if orphans:  # kernels the span attribution missed still count as executed
        labels, _ = label_kernels(orphans)
        srv_backends |= labels
        srv_kernels |= set(orphans)
    for op in serving.get("ops") or []:
        if op_hint and not re.search(op_hint, " ".join(
                [op.get("label") or ""] + (op.get("kernels") or [])), re.I):
            continue
        srv_backends |= set(op.get("backends") or [])
        srv_kernels |= {k for k in (op.get("kernels") or []) if not _is_launcher(k)}
    srv_kernels = {k for k in srv_kernels if not _is_launcher(k)}

    # `triton` is deliberately NOT here: in the taxonomy it now names only the
    # Triton attention backend (routing/activation are framework_native), and
    # it is the sole label of vllm's fp8 head_dim>256 path — ignoring it left
    # that path uncompared (2026-09-23).
    infra = {"cublas", "vllm_kernel", "sgl_kernel", "torch"}
    col_sig = col_backends - infra
    srv_sig = srv_backends - infra
    only_col = sorted(col_sig - srv_sig)
    # family-level match is NECESSARY, not sufficient: the same canonical
    # family can hide different kernels (0.29 DSA indexer did exactly this) —
    # for every signal family on both sides, the collector's kernels must
    # name-overlap serving's, else it is kernel drift and the gate stays red
    def fam_kernels(kerns):
        out = {}
        for k in kerns:
            labels, _ = label_kernels([k])
            for b in labels - infra:
                out.setdefault(b, set()).add(k.split("<")[0])
        return out
    col_fam = fam_kernels(col_kernels)
    srv_fam = fam_kernels(srv_kernels)
    kernel_drift = {}
    for fam in (col_sig & srv_sig):
        ck, sk = col_fam.get(fam, set()), srv_fam.get(fam, set())
        hits = {c for c in ck if any(c in s or s in c for s in sk)}
        misses = sorted(ck - hits)
        if misses:
            kernel_drift[fam] = {"collector_only_kernels": misses,
                                 "serving_kernels": sorted(sk)}
    # A capture that crashed, or that executed no signal-family kernel at all,
    # is not evidence: the empty set is a subset of everything and would read
    # as "aligned". Found 2026-09-24 when six broken sglang captures (mock
    # runner drift, subprocess collectors invisible to the parent profiler)
    # all came back aligned with col=[] — refuse to grade them.
    if cap.get("error"):
        verdict = "invalid-capture"
    elif not col_sig:
        verdict = "no-collector-signal"
    else:
        verdict = "aligned" if not only_col and not kernel_drift else "diverged"
    report = {
        "verdict": verdict,
        "repo": repo, "framework": framework, "version": version,
        # reproducibility: a verdict must name both inputs and the scoping it
        # was computed under, or it cannot be re-derived after a taxonomy change
        "capture_file": str(capture_file), "capture_cmd": cap.get("cmd"),
        "capture_run_error": cap.get("error"),
        "kv_dtype": kv_dtype, "op_hint": op_hint,
        "serving_record": {"id": serving["id"], "tp": serving["runtime"].get("tp"),
                           "kv_cache_dtype": serving["runtime"].get("kv_cache_dtype"),
                           "isl": serving["runtime"].get("isl"),
                           "prefix_caching": serving["runtime"].get("prefix_caching"),
                           "raw_file": serving_raw, "candidates": len(candidates)},
        "collector_backends": sorted(col_backends),
        "serving_backends": sorted(srv_backends),
        "collector_only_signal": only_col,
        "kernel_drift": kernel_drift or None,
        "serving_kernels_matched_in_collector": sorted(
            k for k in srv_kernels if any(k.split("<")[0] in c or c in k for c in col_kernels))[:10],
        "collector_unmatched_kernels": sorted(col_unmatched)[:10],
        "note": "subset semantics: the collector exercises ONE op; serving runs "
                "the whole model. 'diverged' = the collector executed a signal "
                "backend family serving never did.",
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if verdict == "aligned" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--out", help="capture output json")
    ap.add_argument("--capture-file")
    ap.add_argument("--repo")
    ap.add_argument("--framework", default="vllm")
    ap.add_argument("--version")
    ap.add_argument("--save-verdict", default=None,
                    help="persist the verdict json (the workflow gate reads these)")
    ap.add_argument("--op-hint", default=None,
                    help="regex over serving op labels/kernels to scope the comparison")
    ap.add_argument("--kv-dtype", default=None,
                    help="select the serving record whose kv-cache dtype matches this "
                         "capture (fp8|auto|bf16); auto and bf16 are treated as equivalent")
    ap.add_argument("--isl", type=int, default=None,
                    help="select the serving record probed at this prompt length (records carry "
                         "runtime.isl); prefill dispatch is length-conditional")
    ap.add_argument("--serving-raw", default=None,
                    help="diff against this raw probe JSON instead of archive/records.jsonl "
                         "(A/B evidence outside the plan); the report names the file")
    ap.add_argument("cmd", nargs="*", help="capture mode: script + args (after --)")
    args = ap.parse_args()
    if args.capture:
        cmd = args.cmd
        if cmd and cmd[0] == "python3":
            cmd = cmd[1:]
        return capture(cmd, args.out)
    if args.diff:
        return diff(args.capture_file, args.repo, args.framework, args.version,
                    args.op_hint, args.save_verdict, args.kv_dtype, args.isl, args.serving_raw)
    ap.error("pass --capture or --diff")


if __name__ == "__main__":
    raise SystemExit(main())
