"""Retire pre-kv-tag raws that the framework-mode full re-probe replaces.

Only (backend, pinned version) runs whose repo the current plans also cover
are retired; other versions (vllm 0.24, sglang 0.5.14) keep their records.
Usage: python3 retire_old_ids.py [--apply]"""
import json, glob, os, shutil, sys
import os
ROOT = os.environ.get("AIS_PROBE_WORKSPACE") or os.getcwd()
pins = {"vllm": "0.29.0", "sglang": "0.5.16", "trtllm": "1.3.0rc23"}
cur = {}
for pf in ("plan_full_sgl_trt.json", "plan_full_vllm.json", "plan_glm53.json", "plan_trt_kv.json", "plan_sgl_kvfp8.json", "plan.json"):
    for x in json.load(open(f"{ROOT}/archive/{pf}")):
        if isinstance(x, dict) and "skip" not in x:
            cur.setdefault(x["id"], x)
cur_ids = set(cur); cur_repos = {(x["backend"], x["repo"]) for x in cur.values()}
apply = "--apply" in sys.argv
dest = f"{ROOT}/facts/raw_superseded/pre_kvtag_ids_2026-09-24"; os.makedirs(dest, exist_ok=True)
moved = 0; kept_other_version = 0; kept_uncovered = 0
for pf in sorted(glob.glob(f"{ROOT}/archive/plan*.json")):
    for x in json.load(open(pf)):
        if not isinstance(x, dict) or "skip" in x or x["id"] in cur_ids:
            continue
        raw = f"{ROOT}/archive/raw/{x['id']}.json"
        if not os.path.exists(raw):
            continue
        if x.get("version") != pins.get(x.get("backend")):
            kept_other_version += 1; continue
        if (x.get("backend"), x.get("repo")) not in cur_repos:
            kept_uncovered += 1; continue
        moved += 1
        if apply:
            shutil.move(raw, f"{dest}/{x['id']}.json")
print(f"{'moved' if apply else 'would move'} {moved} pre-tag raws at the pins; kept {kept_other_version} other-version, {kept_uncovered} uncovered-repo")
