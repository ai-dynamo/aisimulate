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
                no-collector-signal  the capture ran no kernel the vocabulary
                          can attribute (empty set = subset of everything —
                          refused). GEMM-class ops whose only families are
                          infra (cublas/vllm_kernel/...) are graded on
                          kernel-NAME overlap inside those families instead.
                invalid-capture  the capture command itself failed

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
import hashlib
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

def capture(argv: list[str], out: str, env: list[str] | None = None) -> int:
    import runpy
    import torch
    from torch.profiler import ProfilerActivity, profile

    # Cell selection. A whole-sweep capture is a UNION of every
    # length/prefix-conditional path the collector can take and can only be
    # compared with a serving record that unions them — none does. The
    # collectors already expose cell filters through environment variables
    # (AIC_DSA_CONTEXT_{SEQ_LENS,PREFIX_LENS,BATCH_SIZES}, op_smoke
    # --case-index/--case-prefix on argv); `--env K=V` sets them here so the
    # capture file records the cell it measured and the verdict can name it.
    for kv in env or []:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    cell_env = {k: v for k, v in os.environ.items() if k.startswith(("AIC_", "AIS_"))
                and k not in ("AIS_PROBE_WORKSPACE", "AIS_GENERATOR_SRC", "AIC_GENERATOR_SRC")}
    script, args = argv[0], argv[1:]
    sys.argv = [script] + args
    err = None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as e:
            # 10 is collector.helper.EXIT_CODE_RESTART: the collector asks the
            # executor to recycle its worker AFTER the case ran — a protocol
            # signal, not a failure (trtllm collect_moe.py:830).
            if e.code not in (0, None, 10):
                err = f"exit={e.code}"
        except Exception as e:  # capture is still evidence on failure
            err = f"{type(e).__name__}: {e}"
        torch.cuda.synchronize()
    kernels = sorted({
        ev.key for ev in prof.key_averages()
        if (getattr(ev, "self_device_time_total", 0) or getattr(ev, "self_cuda_time_total", 0)) > 0
        and not ev.key.startswith(("aten::", "Memcpy", "Memset", "cuda", "Cuda"))
    })
    json.dump({"cmd": argv, "env": cell_env or None, "error": err, "kernels": kernels},
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
    from probe_driver import build_orphan_phases  # noqa: E402
    ops, orphans = build_ops(f)
    return {"id": f"raw:{Path(raw_path).name}", "ops": ops, "orphan_kernels": orphans,
            "orphan_phases": {k: v for k, v in build_orphan_phases(f).items() if k in set(orphans)},
            "runtime": {"tp": None, "kv_cache_dtype": f.get("probe_kv_cache_dtype"),
                        "isl": f.get("probe_isl"), "prefix_caching": f.get("probe_prefix_caching")}}


def _kv_equiv(a, b):
    # auto resolves to the model dtype (bf16 for these models); treat the
    # unquantized aliases as one class, fp8 as its own.
    norm = lambda x: "fp8" if (x or "").startswith("fp8") else "unquant"
    return norm(a) == norm(b)


# ---------------------------------------------------------------- the target
# A gate proves ONE op. The taxonomy carries a `role` per kernel (attention,
# gemm, moe_gemm, linear_attention, dsa_indexer, quant, mhc, routing, kvcache,
# norm_rope, infra, ...); the gate's target roles say which kernels are the
# evidence and which are auxiliary work. Auxiliary overlap (a shared quant
# kernel next to two different GEMMs) never carries a verdict (review
# 2026-09-25 P1). Target roles and phase come from the gate name unless given
# explicitly; both are recorded in the verdict.
_GATE_TARGETS = (
    # (regex on the gate/capture name, target roles, phase or None)
    (r"^encoder_attn", ("attention",), "profile_run"),      # vision encoder runs in vLLM's profile run
    (r"^compute_scale", ("quant",), None),
    (r"^gemm_", ("gemm",), None),
    (r"^mla_bmm", ("gemm",), None),
    (r"^moe_", ("moe_gemm", "routing"), None),
    (r"^mhc", ("mhc",), None),
    (r"^(gdn|kda)_ctx", ("linear_attention",), "prefill"),
    (r"^(gdn|kda)_gen", ("linear_attention",), "decode"),
    (r"^(attn|mla|dsa|dsv4|msa|trt_attn|trt_dsa|trt_mla)\w*?_ctx", ("attention", "dsa_indexer"), "prefill"),
    (r"^(attn|mla|dsa|dsv4|msa|trt_attn|trt_dsa|trt_mla)\w*?_gen", ("attention", "dsa_indexer"), "decode"),
    (r"^(attn|mla|dsa|dsv4|msa)", ("attention", "dsa_indexer"), None),   # hca_attn, paged_mqa_logits, ...
)
# roles that are never a target: they are the glue around every op
_NEVER_TARGET = frozenset({"infra", "kvcache", "norm_rope", "activation", "moe_infra"})


# Defined kernel equivalence (review 2026-09-25): a GEMM library picks its tile
# / cluster / split-K instantiation from the problem shape, so two nvjet or
# xmma kernels that differ only in the tile blob are the SAME path. Only the
# gemm roles use it; attention/moe kernels are graded on their full names.
# Every equivalence is explicit and role-scoped; anything not listed is
# compared by full normalized name.
_EQUIV_RULES = (
    # role, pattern, replacement
    ("gemm", r"^(nvjet_sm\d+_\w+?)_\d[\w]*?_([A-Z]{3})$", r"\1_\2"),          # nvjet tile blob (layout kept)
    ("moe_gemm", r"^(nvjet_sm\d+_\w+?)_\d[\w]*?_([A-Z]{3})$", r"\1_\2"),
    ("gemm", r"_tilesize\d+x\d+x\d+\S*", ""),                              # xmma tile
    ("gemm", r"_\d+x\d+x\d+(x\d+)?(_|$)", r"\2"),                           # cutlass tile shapes
    # sgl_kernel DSA top-k selection dispatches on batch size (small_batch vs
    # main): one op, two instantiations — the probe's decode batch is small,
    # the collector's case batch is not
    ("dsa_indexer", r"^topk_(main|small_batch)_kernel$", "topk_kernel"),
)


def equiv_key(kernel: str, role: str | None) -> str:
    k = kernel.split("<")[0]
    for r, pat, rep in _EQUIV_RULES:
        if r == role:
            k = re.sub(pat, rep, k)
    return k


def tile_equiv(kernel: str) -> str:  # kept for callers/tests: the gemm-role view
    return equiv_key(kernel, "gemm")


def infer_target(gate_name: str | None):
    """(target_roles, phase) from the gate name; (None, None) when unknown."""
    if not gate_name:
        return None, None
    for rx, roles, phase in _GATE_TARGETS:
        if re.match(rx, gate_name):
            return tuple(roles), phase
    return None, None


def grade(col_kernels, srv_kernels, target_roles, cap_error, role_of) -> dict:
    """The verdict rule, pure (unit-tested in tests/unit/collector/opharness).

    col_kernels / srv_kernels: normalized, launcher-filtered kernel names of
    the collector capture and the (phase-selected) serving record.
    target_roles: the op roles this gate must prove; None = every labeled
    non-glue role the collector executed (unscoped gate, recorded as such).
    role_of(kernel) -> (backend, role) | None, from the taxonomy.

      aligned              every collector kernel of every target role name-
                           matches a serving kernel of the same role, and the
                           collector launched no target-role backend family
                           serving did not
      diverged             a target role the collector ran is absent from
                           serving (wrong phase/op), a collector-only backend
                           family inside a target role, or kernel drift inside
                           a role (same family, different kernel)
      no-collector-signal  the capture executed no kernel of any target role:
                           it proves nothing about the op (auxiliary quant /
                           norm / kv-cache overlap does not count)
      invalid-capture      the capture command itself failed
    """
    def by_role(kerns):
        out: dict = {}
        for k in kerns:
            hit = role_of(k)
            if hit:
                out.setdefault(hit[1], {})[k.split("<")[0]] = hit[0]
        return out
    col_roles, srv_roles = by_role(col_kernels), by_role(srv_kernels)
    if target_roles is None:
        target_roles = tuple(sorted(r for r in col_roles if r not in _NEVER_TARGET))
    col_target = {r: col_roles[r] for r in target_roles if col_roles.get(r)}

    def name_hits(ck, sk, role=None):
        eq = (lambda k: equiv_key(k, role))
        sk2 = {eq(s) for s in sk}
        return {c for c in ck if any(eq(c) in s or s in eq(c) for s in sk2)}

    evidence, drift, missing_roles, only_col = {}, {}, [], []
    for role, ck in col_target.items():
        sk = srv_roles.get(role, {})
        if not sk:
            missing_roles.append(role)
            evidence[role] = {"collector": sorted(ck), "serving": [], "matched": []}
            continue
        hits = name_hits(set(ck), set(sk), role)
        misses = sorted(set(ck) - hits)
        if misses:
            drift[role] = {"collector_only_kernels": misses, "serving_kernels": sorted(sk)}
        fam_only = sorted(set(ck.values()) - set(sk.values()))
        if fam_only:
            only_col.extend(f"{role}:{b}" for b in fam_only)
        evidence[role] = {"collector": sorted(ck), "serving": sorted(sk), "matched": sorted(hits)}
    # auxiliary roles the collector ran that serving did not: information, not a verdict
    aux_only = sorted(r for r in col_roles if r not in target_roles and r not in _NEVER_TARGET
                      and r not in srv_roles)
    if cap_error:
        verdict = "invalid-capture"
    elif not col_target:
        verdict = "no-collector-signal"
    else:
        verdict = "aligned" if not (missing_roles or drift or only_col) else "diverged"
    return {"verdict": verdict, "target_roles": list(target_roles), "only_col": sorted(only_col),
            "missing_roles": missing_roles, "kernel_drift": drift or None,
            "role_evidence": evidence, "aux_collector_only_roles": aux_only}


def select_serving(record: dict, phase: str | None, op_hint: str | None, is_launcher) -> tuple[set, bool]:
    """Serving kernels for the comparison, scoped to `phase`.

    Phase truth is the record's `kernel_phases` (every kernel the device
    tables saw, by phase) — span attribution is by NAME across phases, so a
    kernel launched in both phases but attributed to one phase's span must
    still count for the other. Records without `kernel_phases` (legacy)
    fall back to op phases + unscoped orphans and report phase_scoped=False.
    Returns (kernels, phase_scoped)."""
    kernels: set = set()
    kphases = record.get("kernel_phases")
    scoped = phase is not None
    if phase is not None and isinstance(kphases, dict) and kphases:
        pool = set(record.get("orphan_kernels") or [])
        for op in record.get("ops") or []:
            if op_hint and not re.search(op_hint, " ".join([op.get("op") or op.get("label") or ""]
                                                           + (op.get("kernels") or [])), re.I):
                continue
            pool |= set(op.get("kernels") or [])
        for k in pool:
            ph = kphases.get(k)
            if ph is None:
                kernels.add(k)          # not in a device table (span-only): keep, unproven
                scoped = False
            elif phase in ph:
                kernels.add(k)
        return {k for k in kernels if not is_launcher(k)}, scoped
    # legacy path (no kernel_phases): op phases where known, orphans unscoped
    phases = record.get("orphan_phases") or {}
    for k in record.get("orphan_kernels") or []:
        if phase is None:
            kernels.add(k)
        elif k in phases:
            if phase in phases[k]:
                kernels.add(k)
        else:
            kernels.add(k)
            scoped = False
    for op in record.get("ops") or []:
        op_phase = op.get("phase") if op.get("phase") in ("prefill", "decode", "profile_run") else None
        if phase is not None and op_phase not in (phase, None):
            continue
        if phase is not None and op_phase is None:
            scoped = False
        if op_hint and not re.search(op_hint, " ".join([op.get("op") or op.get("label") or ""]
                                                       + (op.get("kernels") or [])), re.I):
            continue
        kernels |= set(op.get("kernels") or [])
    return {k for k in kernels if not is_launcher(k)}, scoped


def diff(capture_file: str, repo: str, framework: str, version: str,
         op_hint: str | None, save: str | None = None, kv_dtype: str | None = None,
         isl: int | None = None, serving_raw: str | None = None, gate_name: str | None = None,
         target_roles: tuple | None = None, phase: str | None = None) -> int:
    label_kernels, normalize_kernel = _load_labeler()
    from probe_driver import kernel_role  # noqa: E402
    inferred_roles, inferred_phase = infer_target(gate_name or (Path(save).stem if save else None))
    target_roles = target_roles or inferred_roles
    phase = phase or inferred_phase
    cap = json.loads(Path(capture_file).read_text())
    # custom-op LAUNCHERS shadow their kernels under a second name
    # (_vllm_fa3_C::fwd wraps flash::FlashAttnFwdSm90) — same exclusion
    # build_ops applies to serving orphans, applied here to both sides
    # trtllm's custom ops surface as `trtllm::<op>` launchers over
    # `tensorrt_llm::_v1::kernels::...` kernels (causal_conv1d_fwd vs
    # causal_conv1d::causal_conv1d_fwd_kernel) — same exclusion
    _launcher = re.compile(r"^(_\w*C\w*|sglang|sgl_kernel|triton_|vllm|trtllm)::(?!.*_kernel)")
    def _is_launcher(name: str) -> bool:
        # launchers are <ext module>::<op> names (fwd, scaled_mm, causal_conv1d_fwd);
        # a tail naming a kernel or a GEMM instantiation is the kernel itself —
        # vllm::cutlass_3x_gemm_sm90_fp8 IS the CUTLASS GEMM (found 2026-09-25: the
        # fp8 GEMM gate had been graded on the quant kernel alone because its GEMM
        # was dropped here)
        tail = name.split("::")[-1].lower()
        return bool(_launcher.match(name)) and "kernel" not in tail and "gemm" not in tail
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
                # framework-mode probe (CUDA graphs / compile = serving truth,
                # owner decision 2026-09-24) beats an eager or pre-flag record
                rt.get("probe_eager") is False,
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
    srv_kernels, phase_scoped = select_serving(serving, phase, op_hint, _is_launcher)
    srv_backends, _ = label_kernels(srv_kernels)
    g = grade(col_kernels, srv_kernels, target_roles, cap.get("error"), kernel_role)
    verdict, only_col, kernel_drift = g["verdict"], g["only_col"], g["kernel_drift"]
    report = {
        "verdict": verdict,
        "repo": repo, "framework": framework, "version": version,
        # reproducibility: a verdict must name both inputs and the scoping it
        # was computed under, or it cannot be re-derived after a taxonomy change
        "capture_file": str(capture_file), "capture_cmd": cap.get("cmd"),
        "capture_env": cap.get("env"),  # the collector cell filters the capture ran under
        "capture_run_error": cap.get("error"),
        "kv_dtype": kv_dtype, "op_hint": op_hint,
        # what this gate proves: the target op roles and the execution phase
        # the serving evidence was selected from (phase_scoped False = legacy
        # record without phase information, evidence unscoped)
        "gate_name": gate_name or (Path(save).stem if save else None), "target_roles": g["target_roles"], "phase": phase,
        "phase_scoped": phase_scoped, "role_evidence": g["role_evidence"],
        "missing_roles": g["missing_roles"], "aux_collector_only_roles": g["aux_collector_only_roles"],
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
        "note": "subset semantics: the collector exercises ONE op; serving runs the whole "
                "model. Graded on the target roles only: every collector kernel of a target "
                "role must name-match a serving kernel of that role in the selected phase.",
    }
    report["capture_sha256"] = hashlib.sha256(Path(capture_file).read_bytes()).hexdigest()[:16]
    report["serving_record"]["exec_fingerprint"] = serving.get("exec_fingerprint")
    report["serving_record"]["evidence_status"] = serving.get("evidence_status")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if save:
        # the committed verdict is the CONCLUSION: what was compared (ids, sha,
        # fingerprint, roles, phase) and what came out (verdict, counts, the
        # drifting names when red). Kernel lists are evidence and go to the
        # workspace (archive/evidence/pathdiff/...) — owner decision 2026-09-26.
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(summarize_report(report), indent=1, ensure_ascii=False))
        ev = ROOT / "archive" / "evidence" / "pathdiff" / Path(save).parent.name / Path(save).name
        ev.parent.mkdir(parents=True, exist_ok=True)
        ev.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if verdict == "aligned" else 1


def summarize_report(report: dict) -> dict:
    """Committed view of a verdict: identity of both inputs, the scoping, the
    verdict, per-role counts, and — only when not aligned — the names that
    decide it (missing roles, collector-only families, drifting kernels)."""
    keep = ("verdict", "repo", "framework", "version", "gate_name", "target_roles", "phase", "phase_scoped",
            "kv_dtype", "op_hint", "capture_file", "capture_sha256", "capture_env", "capture_run_error",
            "serving_record", "collector_backends", "serving_backends", "missing_roles", "aux_collector_only_roles")
    out = {k: report.get(k) for k in keep if k in report}
    out["role_counts"] = {role: {"collector": len(ev["collector"]), "serving": len(ev["serving"]), "matched": len(ev["matched"])}
                          for role, ev in (report.get("role_evidence") or {}).items()}
    if report.get("verdict") != "aligned":
        out["collector_only_signal"] = report.get("collector_only_signal")
        out["kernel_drift"] = report.get("kernel_drift")
    out["evidence"] = "archive/evidence/pathdiff/<framework-version>/<gate>.json in the probe workspace (full kernel lists)"
    return out


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
    ap.add_argument("--gate-name", default=None,
                    help="gate name (e.g. gemm_fp8_Llama-3.1-70B-FP8); target roles and phase are inferred "
                         "from it unless --target-roles/--phase are given; default: the --save-verdict stem")
    ap.add_argument("--target-roles", default=None, help="comma list of taxonomy roles the gate must prove")
    ap.add_argument("--phase", default=None, choices=["prefill", "decode", "profile_run"],
                    help="select serving evidence from this execution phase only")
    ap.add_argument("--env", action="append", default=None, metavar="K=V",
                    help="capture mode: set a collector cell filter (e.g. "
                         "AIC_DSA_CONTEXT_SEQ_LENS=4096) before running; recorded in the capture")
    ap.add_argument("cmd", nargs="*", help="capture mode: script + args (after --)")
    args = ap.parse_args()
    if args.capture:
        cmd = args.cmd
        if cmd and cmd[0] == "python3":
            cmd = cmd[1:]
        return capture(cmd, args.out, args.env)
    if args.diff:
        return diff(args.capture_file, args.repo, args.framework, args.version,
                    args.op_hint, args.save_verdict, args.kv_dtype, args.isl, args.serving_raw,
                    args.gate_name, tuple(args.target_roles.split(",")) if args.target_roles else None,
                    args.phase)
    ap.error("pass --capture or --diff")


if __name__ == "__main__":
    raise SystemExit(main())
