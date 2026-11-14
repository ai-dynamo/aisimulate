#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: execution-path alignment — collector op path vs serving,
PROFILER-MEASURED ON BOTH SIDES (owner rule 2026-09-19: selector/class names
are leads, never the verdict; only captured kernels count).

The collector side is the result of `collect.py --ops probe.trace_case` (one planned
case run under torch.profiler). The serving side is a probe record of the same SM from
a campaign (probe_driver.py). Both kernel sets go through the ONE vocabulary
(kernel_taxonomy.yaml rules for that SM); components/gates.py calls diff() for every gate.

  aligned              every collector kernel of the gate's target roles name-matches a
                       serving kernel of that role, the collector launched no target-role
                       backend family serving did not (subset relation: the collector
                       exercises one op, serving runs the whole model), and every other
                       kernel of the timed runs is one serving runs. A kernel that differs
                       from serving's only in its tactic (tile, stages) is the same path:
                       it is listed under tactic_drift and does not diverge the gate
  diverged             a target role the collector ran is absent from serving, a
                       collector-only backend family inside a target role, kernel drift
                       inside a role, or auxiliary work in the timed runs serving never does
  unlabeled-kernels    the capture ran kernels the vocabulary cannot label on this SM
  no-collector-signal  the capture ran no kernel of any target role
  invalid-capture      the capture itself failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_driver


def _kv_equiv(a, b):
    return probe_driver.kv_class(a) == probe_driver.kv_class(b)


# ---------------------------------------------------------------- the target
# A gate proves ONE op. The taxonomy carries a `role` per kernel (attention, gemm, moe_gemm,
# linear_attention, dsa_indexer, quant, mhc, routing, kvcache, norm_rope, infra, ...); the
# gate's target roles (gates.gate_target, from its op) say which kernels are the evidence and
# which are auxiliary work. Auxiliary overlap (a shared quant kernel next to two different
# GEMMs) never proves a gate (review 2026-09-25 P1), but auxiliary work serving never does
# fails it: the row pays for it (aux_drift below).
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
    ("gemm", r"^(nvjet_sm\d+_\w+?)_\d[\w]*?_([A-Z]{3})$", r"\1_\2"),  # nvjet tile blob (layout kept)
    ("moe_gemm", r"^(nvjet_sm\d+_\w+?)_\d[\w]*?_([A-Z]{3})$", r"\1_\2"),
    ("gemm", r"_tilesize\d+x\d+x\d+\S*", ""),  # xmma tile
    ("gemm", r"_\d+x\d+x\d+(x\d+)?(_|$)", r"\2"),  # cutlass tile shapes
    ("gemm", r"_cta\d+x\d+x\d+_2cta\d", ""),  # CuTe DSL tile + 2-CTA cluster (SGLang TGV)
    # sgl_kernel DSA top-k selection dispatches on batch size (small_batch vs
    # main): one op, two instantiations — the probe's decode batch is small,
    # the collector's case batch is not
    ("dsa_indexer", r"^(sglang::)?topk_(main|small_batch(_cluster)?)_kernel$", "topk_kernel"),
    # flashinfer -> TRT-LLM FMHA cubins (Blackwell dense attention / MLA decode): the family is
    # (sm, qkv dtypes, head dims, paged-kv kind, PAGE SIZE); the tail — HVPerCta<n>, MultiCtasKv[Cga],
    # VarSeqQ<n>Kv<n>, Persistent/Static/Grouped/SwapsAbForGen — is the tile/CTA split the runtime picks
    # from kv length and batch (B200 2026-10-03: attn_gen MultiCtasKvCga vs MultiCtasKv, mla_gen
    # HVPerCta128 vs 256 at identical page size; owner decision 2026-10-04: relax to the family).
    # The page size P<n> stays in the key on purpose: a page mismatch IS a collector defect.
    (
        "attention",
        r"^(fmhaSm\d+[a-z]?Kernel_Qkv[A-Za-z0-9]+?H(?:Qk\d+HV\d+|\d+))(?:HVPerCta\d+)?(PagedKv(?:Causal|Dense)(?:P\d+|Paged)?)\w*$",
        r"\1_\2",
    ),
)


def equiv_key(kernel: str, role: str | None) -> str:
    k = kernel.split("<")[0]
    for r, pat, rep in _EQUIV_RULES:
        if r == role:
            k = re.sub(pat, rep, k)
    return k


# A tactic is the instantiation a tuner or heuristic picks for a shape: a tile, a stage count,
# a cluster or load scheme. Two kernels that differ only in it run the same path; near-ties
# flip between runs (TRT-LLM-gen MoE at 4096 tokens, 2026-11-08), so a tactic never diverges
# a gate. TRT-LLM-gen batched GEMMs name it after their dtypes (bmm_E4m3_..._Fp32 then
# _t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_...); Triton tiles and cute-DSL parameters are
# stripped at normalization already (probe_driver.normalize_kernel).
_TACTIC_RULES = ((re.compile(r"^(bmm_\w+?)_t\d+x\d+x\d+\w*$"), r"\1"),)


def tactic_key(kernel: str, role: str | None) -> str:
    k = equiv_key(kernel, role)
    for rx, rep in _TACTIC_RULES:
        k = rx.sub(rep, k)
    return k


def grade(col_kernels, srv_kernels, target_roles, cap_error, role_of, col_measured=None, srv_every=None) -> dict:
    """The verdict rule, pure (unit-tested in tests/unit/collector/opharness).

    col_kernels / srv_kernels: normalized CUDA kernel names of
    the collector capture and the (phase-selected) serving record.
    target_roles: the op roles this gate must prove; None = every labeled
    non-glue role the collector executed (unscoped gate, recorded as such).
    role_of(kernel) -> (backend, role) | None, from the taxonomy.
    col_measured: the kernels of the collector's timed runs alone; when given they
    replace col_kernels for every check (None: the capture did not record them,
    the timed range is graded and auxiliary work goes unchecked).
    srv_every: every serving kernel, any phase (default srv_kernels).
    Kernel drift that is only a tactic (tactic_key matches a serving kernel of the role) is
    reported under tactic_drift and does not diverge the gate.

      aligned              every collector kernel of every target role name-
                           matches a serving kernel of the same role, the
                           collector launched no target-role backend family
                           serving did not, and every other kernel of its timed
                           runs name-matches a serving kernel of its role
      diverged             a target role the collector ran is absent from
                           serving (wrong phase/op), a collector-only backend
                           family inside a target role, kernel drift inside
                           a role (same family, different kernel), or a timed-
                           run kernel outside the target roles serving never
                           launches (aux_drift)
      unlabeled-kernels    the capture ran kernels kernel_taxonomy.yaml cannot label on this
                           SM: label them before the gate can pass
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

    # The timed runs alone are what the row pays for; a warmup's one-time work (a DeepGEMM
    # scale pack on the first call, autotune candidates) is in the timed range but not in them.
    # Without them (an older trace), the whole timed range stands in.
    if col_measured:
        col_kernels = col_measured
    col_roles, srv_roles = by_role(col_kernels), by_role(srv_kernels)
    if target_roles is None:
        target_roles = tuple(sorted(r for r in col_roles if r not in _NEVER_TARGET))
    col_target = {r: col_roles[r] for r in target_roles if col_roles.get(r)}

    def name_hits(ck, sk, role=None):
        eq = lambda k: equiv_key(k, role)
        sk2 = {eq(s) for s in sk}
        return {c for c in ck if any(eq(c) in s or s in eq(c) for s in sk2)}

    evidence, drift, tactic_drift, missing_roles, only_col = {}, {}, {}, [], []
    for role, ck in col_target.items():
        sk = srv_roles.get(role, {})
        if not sk:
            missing_roles.append(role)
            evidence[role] = {"collector": sorted(ck), "serving": [], "matched": []}
            continue
        hits = name_hits(set(ck), set(sk), role)
        misses = sorted(set(ck) - hits)
        if misses:
            srv_tactics = {tactic_key(k, role) for k in sk}
            if all(tactic_key(k, role) in srv_tactics for k in misses):
                tactic_drift[role] = {"collector_only_kernels": misses, "serving_kernels": sorted(sk)}
            else:
                drift[role] = {"collector_only_kernels": misses, "serving_kernels": sorted(sk)}
        fam_only = sorted(set(ck.values()) - set(sk.values()))
        if fam_only:
            only_col.extend(f"{role}:{b}" for b in fam_only)
        evidence[role] = {"collector": sorted(ck), "serving": sorted(sk), "matched": sorted(hits)}
    # Auxiliary work the row pays for (review 2026-11-08): a kernel of the timed runs outside the
    # target roles must be one serving launches, in any phase, under the same role. The vLLM B200
    # fp8_block rows timed an inductor-compiled activation quant and a DeepGEMM fp32->ue8m0 scale
    # pack where serving runs one CUDA quant (7-28% slower rows); a gemm-only gate passed them.
    # Only the timed runs count: the warmups' one-time work (JIT, autotune inputs) is not paid
    # per call.
    aux_drift = None
    if col_measured:
        every = by_role(srv_kernels if srv_every is None else srv_every)
        aux: dict = {}
        for k in col_measured:
            hit = role_of(k)
            if hit is None or hit[1] in target_roles:
                continue
            if not name_hits({k.split("<")[0]}, set(every.get(hit[1], {})), hit[1]):
                aux.setdefault(hit[1], []).append(k.split("<")[0])
        aux_drift = {r: sorted(v) for r, v in sorted(aux.items())}
    # a kernel the vocabulary cannot name has no role, so it would be invisible to the rule above:
    # an op whose own kernels are unlabeled (the Blackwell flashinfer GDN cubins under the sm100
    # rules, 2026-11-06) passed on its conv1d alone
    unlabeled = sorted(k for k in col_kernels if role_of(k) is None)
    if cap_error:
        verdict = "invalid-capture"
    elif not col_target:
        verdict = "unlabeled-kernels" if unlabeled else "no-collector-signal"
    else:
        verdict = "diverged" if (missing_roles or drift or only_col or aux_drift) else "aligned"
        if verdict == "aligned" and unlabeled:
            verdict = "unlabeled-kernels"
    return {
        "verdict": verdict,
        "target_roles": list(target_roles),
        "only_col": sorted(only_col),
        "missing_roles": missing_roles,
        "kernel_drift": drift or None,
        # same kernel, another tactic: a diagnostic, never a divergence
        "tactic_drift": tactic_drift or None,
        "role_evidence": evidence,
        # None: the capture carries no timed-run kernels, so auxiliary work was not checked
        "aux_drift": aux_drift,
        "unlabeled_kernels": unlabeled,
    }


def select_serving(record: dict, phase: str | None) -> tuple[set, bool]:
    """Serving kernels for the comparison, scoped to `phase`.

    Phase truth is the record's `kernel_phases` (every kernel the device tables saw, by
    phase): span attribution is by NAME across phases, so a kernel launched in both phases
    but attributed to one phase's span must still count for the other. A kernel seen only
    in a span (no device-table entry) is kept but leaves the selection unscoped.
    Returns (kernels, phase_scoped)."""
    kphases = record.get("kernel_phases") or {}
    pool = set(record.get("orphan_kernels") or [])
    for op in record.get("ops") or []:
        pool |= set(op.get("kernels") or [])
    kernels, scoped = set(), phase is not None
    for k in pool:
        ph = kphases.get(k)
        if phase is None or (ph is not None and phase in ph):
            kernels.add(k)
        elif ph is None:
            kernels.add(k)
            scoped = False
    return kernels, scoped


def load_collector_side(files) -> dict:
    """The collector side of a gate: the ``probe_trace_case.json`` of each of its cases
    (``{"result": {"kernels", "measured_kernels", "error", "env", "row", ...}}``). A gate's
    ``also`` cases union their kernels; the first error wins; the timed-run kernels are None
    unless every case recorded them."""
    files = [files] if isinstance(files, (str, Path)) else list(files)
    kernels, measured, errors, cmds, envs = set(), set(), [], [], {}
    for f in files:
        doc = json.loads(Path(f).read_text())
        part = doc.get("result") or {}
        errors.append(doc.get("error") or part.get("error"))
        cmds.append(f"probe.{doc.get('probe')} {part.get('op')} {part.get('row')}")
        envs.update(part.get("env") or {})
        kernels.update(part.get("kernels") or [])
        if measured is not None and part.get("measured_kernels") is not None:
            measured.update(part["measured_kernels"])
        else:
            measured = None
    return {
        "kernels": sorted(kernels),
        "measured": None if measured is None else sorted(measured),
        "error": next((e for e in errors if e), None),
        "cmd": cmds,
        "env": envs or None,
    }


def serving_records(
    repo: str, framework: str, version: str, sm: str, kv_dtype: str | None, isl: int = probe_driver.DEFAULT_ISL
) -> list[dict]:
    """The probe records of every campaign in the workspace for (repo, framework, version)
    on this SM with a clean outcome, the kv class and prompt length matching, newest first.
    A gate that names no prompt length reads the default one: an --isl campaign is newer
    but answers a different shape."""
    out = []
    for path in sorted((probe_driver.ROOT / "campaigns").glob("*/records.jsonl")):
        for r in map(json.loads, path.open()):
            rt = r["runtime"]
            if (
                r["target"].get("repo") == repo
                and rt.get("backend") == framework
                and rt.get("version") == version
                and rt.get("sm") == sm
                and (r.get("outcome") or {}).get("status") == "ok"
                and (kv_dtype is None or _kv_equiv(rt.get("kv_cache_dtype"), kv_dtype))
                and (rt.get("isl") or probe_driver.DEFAULT_ISL) == isl
            ):
                out.append({**r, "campaign": path.parent.name})
    # no kv asked: the RENDERED configuration is the serving truth, a kv-variant record never
    # stands in for it; then the most recent import wins
    out.sort(
        key=lambda r: (
            kv_dtype is not None or r["runtime"].get("kv_cache_dtype") in (None, "auto"),
            r.get("imported_at", 0),
        )
    )
    return out[::-1]


def diff(
    capture_files,
    repo: str,
    framework: str,
    version: str,
    sm: str,
    save: str | None = None,
    kv_dtype: str | None = None,
    gate_name: str | None = None,
    target_roles: tuple | None = None,
    phase: str | None = None,
    isl: int = probe_driver.DEFAULT_ISL,
) -> int:
    cap = load_collector_side(capture_files)
    col_kernels = sorted({n for k in cap["kernels"] if (n := probe_driver.normalize_kernel(k))})
    col_backends, _ = probe_driver.label_kernels(col_kernels, sm)
    candidates = serving_records(repo, framework, version, sm, kv_dtype, isl)
    if not candidates:
        _hint = (f" kv={kv_dtype}" if kv_dtype else "") + f" isl={isl}"
        print(f"[no-serving-record] {repo} {framework}-{version} {sm}{_hint} — probe that config first")
        return 2
    serving = candidates[0]
    if len(candidates) > 1:  # several serving configs match (tp / kv variants, re-probes): say which one won
        print(
            f"[note] {len(candidates)} serving records match; using {serving['id']} from {serving['campaign']} "
            f"(tp={serving['runtime'].get('tp')} kv={serving['runtime'].get('kv_cache_dtype')})",
            file=sys.stderr,
        )
    col_measured = (
        None
        if cap["measured"] is None
        else sorted({n for k in cap["measured"] if (n := probe_driver.normalize_kernel(k))})
    )
    srv_kernels, phase_scoped = select_serving(serving, phase)
    srv_backends, _ = probe_driver.label_kernels(srv_kernels, sm)
    role_of = lambda k: probe_driver.kernel_role(k, sm)
    g = grade(
        col_kernels,
        srv_kernels,
        target_roles,
        cap.get("error"),
        role_of,
        col_measured,
        select_serving(serving, None)[0],
    )
    files = [capture_files] if isinstance(capture_files, (str, Path)) else list(capture_files)
    report = {
        "verdict": g["verdict"],
        "repo": repo,
        "framework": framework,
        "version": version,
        "sm": sm,
        # reproducibility: both inputs and the scoping the verdict was computed under
        "capture_files": [str(f) for f in files],
        "capture_sha256": hashlib.sha256(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:16],
        "capture_cmd": cap.get("cmd"),
        "capture_env": cap.get("env"),
        "capture_run_error": cap.get("error"),
        "kv_dtype": kv_dtype,
        "gate_name": gate_name,
        "target_roles": g["target_roles"],
        "phase": phase,
        "phase_scoped": phase_scoped,
        "missing_roles": g["missing_roles"],
        "collector_only_signal": g["only_col"],
        "kernel_drift": g["kernel_drift"],
        "tactic_drift": g["tactic_drift"],
        "role_evidence": g["role_evidence"],
        "aux_drift": g["aux_drift"],
        "unlabeled_kernels": g["unlabeled_kernels"],
        "serving_record": {
            "id": serving["id"],
            "campaign": serving["campaign"],
            "tp": serving["runtime"].get("tp"),
            "kv_cache_dtype": serving["runtime"].get("kv_cache_dtype"),
            "isl": serving["runtime"].get("isl"),
            "exec_fingerprint": serving.get("exec_fingerprint"),
            "candidates": len(candidates),
        },
        "collector_backends": sorted(col_backends),
        "serving_backends": sorted(srv_backends),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if g["verdict"] == "aligned" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="one collector capture vs the serving record of the same SM")
    ap.add_argument(
        "--capture-file", action="append", required=True, help="probe_trace_case.json (repeat for also cases)"
    )
    ap.add_argument("--repo", required=True)
    ap.add_argument("--framework", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--sm", required=True, help="e.g. sm100: the SM both sides ran on")
    ap.add_argument("--kv-dtype", default=None, help="fp8|auto|bf16 (auto and bf16 are one class)")
    ap.add_argument("--target-roles", required=True, help="comma list of taxonomy roles the capture must prove")
    ap.add_argument("--phase", default=None, choices=["prefill", "decode", "profile_run"])
    ap.add_argument(
        "--isl", type=int, default=probe_driver.DEFAULT_ISL, help="serving records probed at this prompt length"
    )
    ap.add_argument("--save", default=None, help="write the verdict json here")
    args = ap.parse_args()
    return diff(
        args.capture_file,
        args.repo,
        args.framework,
        args.version,
        args.sm,
        args.save,
        args.kv_dtype,
        None,
        tuple(args.target_roles.split(",")),
        args.phase,
        args.isl,
    )


if __name__ == "__main__":
    raise SystemExit(main())
