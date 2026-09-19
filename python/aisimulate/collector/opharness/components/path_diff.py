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
              vocabulary (kernel_taxonomy.yaml) and report the verdict:
                aligned   collector's canonical backend set matches serving's
                          for the op's kernels (subset relation: the collector
                          exercises one op, serving runs the whole model)
                diverged  collector executed backend families serving never
                          did (the wrong-path signal), or vice versa for the
                          op under test

Usage (capture, in-container):
  python3 path_diff.py --capture --out /out/cap.json -- \
      python3 collector/vllm/collect_mla_module_029.py --mode context ...

Usage (diff, host):
  AIC_PROBE_WORKSPACE=<ws> python3 path_diff.py --diff \
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
ROOT = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))


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


def diff(capture_file: str, repo: str, framework: str, version: str,
         op_hint: str | None, save: str | None = None) -> int:
    label_kernels, normalize_kernel = _load_labeler()
    cap = json.loads(Path(capture_file).read_text())
    # custom-op LAUNCHERS shadow their kernels under a second name
    # (_vllm_fa3_C::fwd wraps flash::FlashAttnFwdSm90) — same exclusion
    # build_ops applies to serving orphans, applied here to both sides
    _launcher = re.compile(r"^(_\w*C\w*|sglang|sgl_kernel|triton_|vllm)::(?!.*_kernel)")
    def _is_launcher(name: str) -> bool:
        return bool(_launcher.match(name)) and "kernel" not in name.split("::")[-1].lower()
    col_kernels = sorted({n for k in cap["kernels"]
                          if (n := normalize_kernel(k)) and not _is_launcher(n)})
    col_backends, col_unmatched = label_kernels(col_kernels)

    serving = None
    for line in (ROOT / "archive" / "records.jsonl").open():
        r = json.loads(line)
        if (r["target"].get("repo") == repo and r["runtime"]["backend"] == framework
                and r["runtime"].get("version") == version
                and (r.get("outcome") or {}).get("status") == "ok"):
            serving = r
    if serving is None:
        print(f"[no-serving-record] {repo} {framework}-{version} — probe first")
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

    infra = {"cublas", "vllm_kernel", "sgl_kernel", "torch", "triton"}
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
    verdict = "aligned" if not only_col and not kernel_drift else "diverged"
    report = {
        "verdict": verdict,
        "repo": repo, "framework": framework, "version": version,
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
    ap.add_argument("cmd", nargs="*", help="capture mode: script + args (after --)")
    args = ap.parse_args()
    if args.capture:
        cmd = args.cmd
        if cmd and cmd[0] == "python3":
            cmd = cmd[1:]
        return capture(cmd, args.out)
    if args.diff:
        return diff(args.capture_file, args.repo, args.framework, args.version,
                    args.op_hint, args.save_verdict)
    ap.error("pass --capture or --diff")


if __name__ == "__main__":
    raise SystemExit(main())
