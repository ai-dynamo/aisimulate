#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component: pack a campaign's EVIDENCE into one addressable bundle.

The repository carries only curated findings (published under
benchmarks/evidence/collector/). Everything else stays out of git: the
campaign's intermediate conclusions in the harness's local results/ (matrices,
verdict summaries, decompose summaries, retests, campaign ledgers; owner
decision 2026-10-30) and the evidence they rest on — raw probe JSONs with their
fingerprint sidecars, records.jsonl, collector captures, the full path_diff
reports and kernel-level decompositions (owner decision 2026-09-26: tens of
thousands of kernel lines are not a review surface). This component packs
both into one tar.gz per campaign, named by content (results/ members under
`results/`), and indexes it in results/evidence_index.yaml (campaign id,
sha256, size, member counts, the harness commit that produced it, where the
bundle was put).

Usage:
  AIS_PROBE_WORKSPACE=<ws> python3 evidence_bundle.py --campaign 2026-09-26_sm90 \\
      [--out <ws>/archive/bundles] [--location "s3://... or a path the team agrees on"]
  python3 evidence_bundle.py --verify <bundle.tar.gz>      # sha256 against the index
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
ROOT = Path(
    os.environ.get("AIS_PROBE_WORKSPACE")
    or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
    or Path.cwd()
)

# what a campaign's evidence is, relative to the workspace
MEMBERS = (
    ("archive/records.jsonl", "records"),
    ("archive/plan*.json", "plans"),
    ("archive/raw/*.json", "raw"),
    ("archive/raw/*.fp", "fingerprints"),
    ("archive/run_sh/*", "rendered_engine_configs"),
    # console tail of every probe (emit_queues `tee`s it): the only evidence for a probe that died by a native
    # abort before writing a raw; build_matrix classifies those cells from it (sm89 Gemma-4: MMHA head size 512)
    ("archive/logs/*.tail", "console_tails"),
    ("archive/evidence/**/*", "full_reports"),
    ("facts/pathdiff/opcov_*.json", "captures"),
)


def collect(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for pattern, kind in MEMBERS:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
        if files:
            out[kind] = files
    return out


def collect_results(harness: Path) -> list[Path]:
    """The harness's local results/ tree; the index is rewritten by build() itself."""
    results = harness / "results"
    if not results.is_dir():
        return []
    return sorted(p for p in results.rglob("*") if p.is_file() and p.name != "evidence_index.yaml")


def build(root: Path, campaign: str, out_dir: Path, location: str | None) -> dict:
    members = collect(root)
    if not members.get("raw"):
        raise SystemExit(f"no raw evidence under {root}/archive/raw — nothing to bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{campaign}.tar.gz.part"
    counts: dict[str, int] = {}
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for kind, files in members.items():
            counts[kind] = len(files)
            for f in files:
                tar.add(f, arcname=str(f.relative_to(root)), recursive=False)
        results = collect_results(HARNESS)
        if results:
            counts["results"] = len(results)
            for f in results:
                tar.add(f, arcname=str(f.relative_to(HARNESS)), recursive=False)
        manifest = json.dumps(
            {
                "campaign": campaign,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "workspace": str(root),
                "counts": counts,
            },
            indent=1,
        ).encode()
        info = tarfile.TarInfo("EVIDENCE_MANIFEST.json")
        info.size = len(manifest)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(manifest))
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = out_dir / f"{campaign}.{sha[:12]}.tar.gz"
    tmp.rename(final)
    try:
        commit = subprocess.run(
            ["git", "-C", str(HARNESS), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    entry = {
        "campaign": campaign,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness_commit": commit,
        "bundle": final.name,
        "sha256": sha,
        "size_bytes": final.stat().st_size,
        "location": location or str(final),
        "counts": counts,
    }
    index = HARNESS / "results" / "evidence_index.yaml"
    doc = yaml.safe_load(index.read_text()) if index.exists() else None
    doc = doc or {
        "_meta": {
            "purpose": "conclusions in results/ point at evidence bundles by campaign id + sha256; "
            "bundles live outside git (location column)"
        },
        "bundles": [],
    }
    doc["bundles"] = [b for b in doc["bundles"] if b.get("campaign") != campaign] + [entry]
    index.write_text(yaml.safe_dump(doc, sort_keys=False, width=110))
    return entry


def verify(bundle: Path) -> int:
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    index = yaml.safe_load((HARNESS / "results" / "evidence_index.yaml").read_text())
    hit = next((b for b in index.get("bundles", []) if b.get("sha256") == sha), None)
    if hit is None:
        print(f"NOT INDEXED: {bundle.name} sha256={sha}")
        return 1
    print(f"ok: {bundle.name} = campaign {hit['campaign']} ({hit['harness_commit']}), {hit['counts']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", default=None, help="campaign id, e.g. 2026-09-26_sm90")
    ap.add_argument("--out", type=Path, default=None, help="default <workspace>/archive/bundles")
    ap.add_argument("--location", default=None, help="where the bundle is/will be stored (recorded in the index)")
    ap.add_argument("--verify", type=Path, default=None)
    args = ap.parse_args()
    if args.verify:
        return verify(args.verify)
    if not args.campaign:
        ap.error("--campaign or --verify")
    entry = build(ROOT, args.campaign, args.out or ROOT / "archive" / "bundles", args.location)
    print(json.dumps(entry, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
