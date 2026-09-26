#!/usr/bin/env python3
"""Component: decompose observed execution into op families + residue.

onboard_model step 5. Input is the curated probe records (archive/records.jsonl,
framework-mode identity probes); every kernel a record observed — the ones
attached to an API span and the taxonomy-labeled orphans — is translated
through the SM's kernel taxonomy (the SAME file the path_diff gate uses) into
(role, backend-label). The output per (framework, version) is one YAML:

    results/<sm>/decompose/<framework>-<version>.yaml            (committed SUMMARY)
      results:
        <repo>:
          record: <id>  variant: <dummy cut>  kv: <rendered|fp8...>
          families: {<role>: {<label>: <kernel count>}}   # covered execution
          residue:  [kernels no taxonomy rule labels]     # NOT covered
          ops_observed: [<api-span ops>]
    <workspace>/archive/evidence/decompose/<sm>/<framework>-<version>.yaml
      the same, with the kernel NAMES per role/label (evidence, not committed)

`residue` is the whole point: a non-empty residue means the model runs
execution no op family names yet, and deciding "new family / new table /
absorb into an existing one" is the owner's granularity call (step 6). An
empty residue makes step 6 a no-op. Nothing here interprets kernels — a
kernel is either matched by a taxonomy rule or it is residue; growing the
taxonomy is the only way to shrink the residue, and that edit is reviewed.

Representative record per (repo, framework, version): the rendered-KV,
framework-mode (non-eager) probe of the checkpoint's representative dummy
cut; fp8-KV variants are listed under `kv_variants` when they add kernels.

Usage:
  AIS_PROBE_WORKSPACE=<ws> python3 decompose.py --sm sm90 [--repo org/name]
      [--records <ws>/archive/records.jsonl] [--out results/<sm>/decompose]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE")
            or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
            or Path.cwd())


def load_rules(sm: str, taxonomy: Path | None = None) -> list[tuple[re.Pattern, str, str]]:
    path = taxonomy or HERE / f"kernel_taxonomy_{sm}.yaml"
    doc = yaml.safe_load(path.read_text())
    return [(re.compile(r["match"]), str(r["backend"]), str(r.get("role") or "unknown"))
            for r in doc["rules"]]


def label(kernel: str, rules) -> tuple[str, str] | None:
    """First matching taxonomy rule -> (role, backend label); None = residue."""
    for rx, backend, role in rules:
        if rx.search(kernel):
            return role, backend
    return None


def record_kernels(rec: dict) -> set[str]:
    ks: set[str] = set()
    for op in rec.get("ops") or []:
        ks.update(op.get("kernels") or [])
    ks.update(rec.get("orphan_kernels") or [])
    return {k for k in ks if k}


def _quality(rec: dict) -> tuple:
    """Representative first: rendered KV, framework mode (graphs+compile), deepest cut."""
    rt = rec.get("runtime") or {}
    kv = str(rt.get("kv_cache_dtype"))
    rendered = kv in ("None", "auto", "bf16", "bfloat16")
    return (rendered, rt.get("probe_eager") is False, -int(str(rec["target"].get("variant", ""))[5:] or 0)
            if str(rec["target"].get("variant", "")).startswith("depth") else 0)


def decompose_record(rec: dict, rules) -> dict:
    families: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    residue: list[str] = []
    for k in sorted(record_kernels(rec)):
        hit = label(k, rules)
        if hit is None:
            residue.append(k)
        else:
            role, backend = hit
            families[role][backend].append(k)
    ops = sorted({op["op"] for op in (rec.get("ops") or []) if op.get("op")})
    rt = rec.get("runtime") or {}
    return {
        "record": rec["id"],
        "variant": rec["target"].get("variant"),
        "kv": rt.get("kv_cache_dtype"),
        "probe_eager": rt.get("probe_eager"),
        "families": {role: dict(sorted(labels.items())) for role, labels in sorted(families.items())},
        "residue": residue,
        "ops_observed": ops,
        "coverage": {"labeled": sum(len(v) for labels in families.values() for v in labels.values()),
                     "residue": len(residue)},
    }


def decompose(records_path: Path, sm: str, repo_filter: str | None, rules) -> dict[tuple[str, str], dict]:
    """(framework, version) -> {repo: decomposition}; representative record per repo,
    fp8-KV siblings folded in as kv_variants when they add kernels."""
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (rec.get("outcome") or {}).get("status") != "ok":
            continue
        rt = rec.get("runtime") or {}
        if str(rt.get("sm_measured") or sm) != sm:
            continue
        repo = rec["target"]["repo"]
        if repo_filter and repo_filter not in repo:
            continue
        by_key[(rt["backend"], str(rt["version"]), repo)].append(rec)
    out: dict[tuple[str, str], dict] = defaultdict(dict)
    for (fw, ver, repo), recs in sorted(by_key.items()):
        recs.sort(key=_quality, reverse=True)
        rep = decompose_record(recs[0], rules)
        base = record_kernels(recs[0])
        extras = {}
        for other in recs[1:]:
            add = sorted(record_kernels(other) - base)
            if add:
                extras[str((other.get("runtime") or {}).get("kv_cache_dtype"))] = {
                    "record": other["id"], "added_kernels": add,
                    "added_residue": [k for k in add if label(k, rules) is None]}
        if extras:
            rep["kv_variants"] = extras
        out[(fw, ver)][repo] = rep
    return out


def summarize(entry: dict) -> dict:
    """The committed view of a decomposition: role -> backend -> kernel COUNT,
    the residue list (the decision input), the ops seen and the record id.
    Kernel names live in the evidence file next to the archive; the summary
    is what the workflow predicates and reviewers read (owner decision
    2026-09-26: the repo carries conclusions, evidence stays out)."""
    out = {k: entry[k] for k in ("record", "variant", "kv", "probe_eager") if k in entry}
    out["families"] = {role: {b: len(ks) for b, ks in labels.items()} for role, labels in entry["families"].items()}
    out["residue"] = list(entry.get("residue") or [])
    out["ops_observed"] = list(entry.get("ops_observed") or [])
    out["coverage"] = entry.get("coverage")
    if entry.get("kv_variants"):
        out["kv_variants"] = {kv: {"record": v["record"], "added_kernels": len(v["added_kernels"]),
                                   "added_residue": list(v["added_residue"])} for kv, v in entry["kv_variants"].items()}
    return out


def _merge_doc(path: Path, repos: dict, sm: str, fw: str, ver: str, taxonomy_name: str, kind: str) -> dict:
    existing = yaml.safe_load(path.read_text()) if path.exists() else None
    results = dict((existing or {}).get("results") or {})
    results.update(repos)  # a filtered run refreshes its repos, keeps the rest
    residue_repos = sorted(r for r, d in results.items() if d.get("residue"))
    return {
        "_meta": {"platform": sm, "framework": fw, "version": ver, "taxonomy": taxonomy_name, "kind": kind,
                  "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "summary": {"repos": len(results), "repos_with_residue": len(residue_repos),
                              "residue_kernels": sorted({k for d in results.values() for k in d.get("residue", [])})}},
        "results": dict(sorted(results.items())),
    }


def write_outputs(decomp: dict[tuple[str, str], dict], sm: str, out_dir: Path, taxonomy_name: str,
                  evidence_dir: Path | None = None) -> list[Path]:
    """results/<sm>/decompose/<fw>-<ver>.yaml = SUMMARY (committed);
    <evidence_dir>/<fw>-<ver>.yaml = full kernel lists (workspace evidence)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (fw, ver), repos in sorted(decomp.items()):
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            full = evidence_dir / f"{fw}-{ver}.yaml"
            full.write_text(yaml.safe_dump(_merge_doc(full, repos, sm, fw, ver, taxonomy_name, "evidence:kernels"),
                                           sort_keys=False, width=110, allow_unicode=True))
        path = out_dir / f"{fw}-{ver}.yaml"
        doc = _merge_doc(path, {r: summarize(e) for r, e in repos.items()}, sm, fw, ver, taxonomy_name, "summary")
        # leaf collections in flow style: one line per role / list, a few lines per repo
        path.write_text(yaml.safe_dump(doc, sort_keys=False, width=200, allow_unicode=True, default_flow_style=None))
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sm", default="sm90")
    ap.add_argument("--records", type=Path, default=None, help="default <workspace>/archive/records.jsonl")
    ap.add_argument("--taxonomy", type=Path, default=None, help="default components/kernel_taxonomy_<sm>.yaml")
    ap.add_argument("--repo", default=None, help="substring filter on org/name")
    ap.add_argument("--out", type=Path, default=None, help="default results/<sm>/decompose/")
    ap.add_argument("--all-versions", action="store_true",
                    help="also decompose records of retired framework pins (default: targets.yaml pins only)")
    args = ap.parse_args()
    records = args.records or ROOT / "archive" / "records.jsonl"
    if not records.exists():
        raise SystemExit(f"no records at {records} (run probe_driver --records first)")
    taxonomy = args.taxonomy or HERE / f"kernel_taxonomy_{args.sm}.yaml"
    rules = load_rules(args.sm, taxonomy)
    decomp = decompose(records, args.sm, args.repo, rules)
    if not args.all_versions:
        pins = {(fw, str(v)) for fw, be in yaml.safe_load((HARNESS / "targets.yaml").read_text())["backends"].items()
                for v in (be.get("versions") or [])}
        decomp = {k: v for k, v in decomp.items() if k in pins}
    if not decomp:
        raise SystemExit("no ok records matched")
    out_dir = args.out or HARNESS / "results" / args.sm / "decompose"
    evidence_dir = ROOT / "archive" / "evidence" / "decompose" / args.sm
    for path in write_outputs(decomp, args.sm, out_dir, taxonomy.name, evidence_dir):
        doc = yaml.safe_load(path.read_text())
        s = doc["_meta"]["summary"]
        print(f"wrote {path}: {s['repos']} repos, {s['repos_with_residue']} with residue, "
              f"{len(s['residue_kernels'])} distinct residue kernels")
        for k in s["residue_kernels"][:20]:
            print(f"   residue: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
