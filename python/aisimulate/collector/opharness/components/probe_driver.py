#!/usr/bin/env python3
"""Facts driver: given a generator-rendered config, what does the framework
actually deploy? targets.yaml (input) -> golden `cli generate` renders ->
per-GPU probe queues -> evidence (archive/raw, records.jsonl) -> results.

Runs on the host. Golden configs are rendered by the generator CLI; the
workspace currently drives the predecessor aiconfigurator toolchain's venv
for that (a factual pin, see render_cmd), overridable via AIS_GENERATOR_SRC.

  --plan            enumerate runs (use --backends / --only to scope)
  --emit-queues     render goldens + write per-GPU queue scripts
  --records         raw probe JSONs -> archive/records.jsonl (curated evidence)
  --matrix          consolidated results/<sm>/<framework>-<version>.yaml
  --check-coverage  collector-mentioned repos are a coverage LOWER bound
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

import yaml

# workspace: where dummy_models/, archive/ and probe outputs live
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE")
            or os.environ.get("AIC_PROBE_WORKSPACE")  # legacy name
            or Path.cwd())
# generator source: this repo by default; override to pin a specific checkout
AIS_SRC = os.environ.get("AIS_GENERATOR_SRC") or os.environ.get("AIC_GENERATOR_SRC",
                         str(Path(__file__).resolve().parents[3] / "src"))
if AIS_SRC not in sys.path:
    sys.path.insert(0, AIS_SRC)
WORK = "/work"  # container mount of ROOT
# The probes run FROM THIS CHECKOUT, never from a copy in the workspace (review
# 2026-09-25 P2 #5: twin scripts drift). When the checkout lives inside the
# workspace it is already visible under /work; otherwise it is mounted read-only.
_HERE = Path(__file__).resolve().parent
try:
    PROBES_IN_CONTAINER = f"{WORK}/{(_HERE / 'probes').relative_to(ROOT)}"
    _EXTRA_MOUNT = ""
except ValueError:
    PROBES_IN_CONTAINER = "/harness/probes"
    _EXTRA_MOUNT = f"-v {_HERE}:/harness:ro "
SCRATCH_QUEUES = ROOT / "archive" / "queues"

GOLDEN_TARGET = {"sglang": "dynamo-python", "vllm": "fpm", "trtllm": "dynamo-python"}
# The golden renderer is THIS repository's generator (aisimulate), run through
# its own console script from an environment that has the native runtime
# built (uv pip install -e python/aisimulate). Until 2026-09-25 it was the
# aic checkout's `aiconfigurator.main` — every generator change had to be
# mirrored there, and models only aisimulate knows (Qwen3.8) rendered as
# "generator rejects". AIS_GENERATOR_CLI overrides the binary.
GEN_CLI = Path(os.environ.get("AIS_GENERATOR_CLI") or ROOT / "venv_ais" / "bin" / "aiconfigurator")


_FACTS_FAIL_RX = re.compile(r"Fact resolution failed|Unknown hardware profile")


def golden_facts_status(render_output: str) -> dict:
    """Did the golden render apply the generator's MODEL facts (facts/models.yaml
    defaults: block-size, trust-remote-code, autotune flags ...)?

    generator/pipeline.run_pipeline swallows a facts-resolution failure ("Fact
    resolution failed; continuing without facts.") and renders with facts=None,
    so the engine args silently lack every model default. Found 2026-10-01: all
    RTX PRO renders (no hardware profile for rtx_pro_6000_server) lost them, and
    the V4.1 block-size fact that rescues H20 never reached sm120. The status is
    recorded on the run, in records and in the matrix cell so a render without
    facts is never mistaken for a golden one."""
    if _FACTS_FAIL_RX.search(render_output or ""):
        m = re.search(r"KeyError: [^\n]*", render_output) or re.search(r"Fact resolution failed[^\n]*", render_output)
        return {"applied": False, "reason": (m.group(0) if m else "fact resolution failed").strip()[:200]}
    return {"applied": True}


_GEN_STAMP: str | None = None


def _generator_stamp() -> str:
    """HEAD plus a hash of the uncommitted generator-source diff. The golden-render cache was keyed on HEAD
    alone, so an uncommitted generator/facts edit re-rendered nothing and the first fixed-fact emit silently
    served the stale goldens (B200 handoff 2026-10-04, defect 3; found again on sm89 when A/B-ing a facts change)."""
    global _GEN_STAMP
    if _GEN_STAMP is None:
        import subprocess as _sp
        repo = str(Path(AIS_SRC).parent)
        head = _sp.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
        diff = _sp.run(["git", "-C", repo, "diff", "HEAD", "--", AIS_SRC], capture_output=True, text=True).stdout
        # untracked generator files (a new facts yaml / template) are invisible to `git diff`: hash their names + sizes
        untracked = _sp.run(["git", "-C", repo, "ls-files", "--others", "--exclude-standard", "--", AIS_SRC],
                            capture_output=True, text=True).stdout.split()
        diff += "".join(f"\n{u}:{(Path(repo) / u).stat().st_size}" for u in untracked if (Path(repo) / u).exists())
        _GEN_STAMP = head + (("+dirty-" + hashlib.sha256(diff.encode()).hexdigest()[:8]) if diff.strip() else "")
    return _GEN_STAMP


def render_golden(run: dict) -> Path | None:
    """Invoke the REAL user-facing generator command and archive it verbatim.

    golden/<id>/command.txt is the exact generator `cli generate` argv —
    the thing we converge on and guarantee. Artifacts are stored untouched;
    every probe-side adaptation happens later as a RECORDED post-process.
    Owner decisions: --system comes from targets.platform (h200_sxm proxies
    the H20 probe box — same VRAM, h20 deliberately not added to code); per-checkpoint extra args live in targets.yaml
    checkpoint_overrides.cli_extra_args and are spliced into the command.
    """
    import shutil
    import subprocess
    gdir = ROOT / "archive" / "golden" / run["id"]
    cmd = [str(GEN_CLI), "cli", "generate",
           "--model-path", run["repo"],
           "--total-gpus", str(run["tp"]),
           "--system", run["system"],
           "--backend", run["backend"],
           "--deployment-target", GOLDEN_TARGET[run["backend"]],
           "--config-template-version", run["version"],
           "--save-dir", str(gdir)]
    cmd += list(run.get("cli_extra_args") or [])
    cmd_txt = shlex.join(cmd)
    gen_commit = _generator_stamp()
    stamp = gdir / "command.txt"
    # cache valid only for the SAME command rendered by the SAME generator code
    if stamp.exists() and stamp.read_text().splitlines()[:2] == [cmd_txt, f"# generator={gen_commit}"]:
        sub = next((d for d in gdir.iterdir() if d.is_dir()), None)
        if sub is not None:
            log = gdir / "render.log"
            run["golden_facts"] = golden_facts_status(log.read_text() if log.exists() else "")
            return sub  # cached golden for the identical command
    if gdir.exists():
        shutil.rmtree(gdir)
    gdir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = AIS_SRC
    # cwd = the WORKSPACE: aic-core's safe_mkdir accepts paths under cwd / $HOME / tmp only, so with the
    # harness checkout as cwd a workspace on another mount (B200 /raid, B300 NFS) rendered 124/124 as
    # silent 'generator rejects' (B300 2026-09-20, B200 2026-09-29 handoffs).
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd=str(ROOT))
    stamp.write_text(cmd_txt + f"\n# generator={gen_commit}\n# exit={r.returncode}\n")
    (gdir / "render.log").write_text((r.stdout or "")[-8000:] + (r.stderr or "")[-8000:])
    run["golden_facts"] = golden_facts_status((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        run["golden_error"] = (r.stderr or r.stdout or "").strip().splitlines()[-1][:200] if (r.stderr or r.stdout) else "no output"
        return None
    return next((d for d in gdir.iterdir() if d.is_dir()), None)


def extract_sglang_cli_from_run_sh(run_sh: Path) -> str:
    """Post-process: lift the dynamo.sglang engine args out of golden run_0.sh.
    Drops only wrapper plumbing ($MODEL_PATH placeholder, host/metrics/shell);
    engine-selection flags pass through verbatim."""
    import re
    text = run_sh.read_text()
    m = re.search(r"python3 -m dynamo\.sglang((?:[^\n]*\\\n)*[^\n]*)", text)
    if not m:
        raise SystemExit(f"golden run_0.sh has no dynamo.sglang block: {run_sh}")
    block = m.group(1).replace("\\\n", " ")
    block = re.split(r"\s(?:2>&1|\||&|;|\))", block)[0]
    toks = shlex.split(block)
    out, skip = [], False
    DROP = {"--model-path", "--served-model-name", "--host", "--port"}
    FLAG_DROP = {"--enable-metrics"}
    for i, tk in enumerate(toks):
        if skip:
            skip = False
            continue
        if tk in DROP:
            skip = True
            continue
        if tk in FLAG_DROP:
            continue
        out.append(tk)
    return " ".join(out)


def _cea(v, sm: str | None = None):
    """cli_extra_args entry: plain list, or {args: [...], fact: "<evidence>",
    sms: [sm120, ...]} — the fact field cites the probe evidence this generator
    input derives from; ``sms`` (optional) limits the customization to those
    SMs (RTX 6000 / 5000 handoffs 2026-09-30: an entry keyed (repo, fw) rode
    along to every SM; sm120 needs bf16 KV for NVFP4 MLA checkpoints, which
    sm90/sm100 must not inherit). No ``sms`` = every SM."""
    if not v:
        return []
    if isinstance(v, dict):
        if v.get("sms") and (sm or current_sm()) not in [str(x) for x in v["sms"]]:
            return []
        return list(v["args"])
    return list(v)



_MENTION_ORGS = (r"(?:deepseek-ai|zai-org|moonshotai|nvidia|openai|meta-llama|"
                 r"mistralai|google|Qwen|XiaomiMiMo|MiniMaxAI|sgl-project)")


def collector_mentioned_repos() -> set[str]:
    """Every HF repo the collector's case yamls mention (the coverage floor).
    Brace-expansion prose like org/Name-{A,B}-X truncates at '{' and .py
    paths false-match — both filtered."""
    mentioned: set[str] = set()
    cases = Path(AIS_SRC).parent / "collector" / "cases" / "models"
    for f in cases.glob("*_cases.yaml"):
        for m in re.findall(rf"\b({_MENTION_ORGS}/[\w.\-]+)", f.read_text()):
            if not m.endswith("-") and not m.endswith(".py"):
                mentioned.add(m)
    return mentioned

def derive_roster_checkpoints(fam: dict, targets: dict) -> list[dict]:
    """Roster checkpoints DERIVED from the collector's own case declarations:
    every org/repo its cases yamls mention, minus gated repos and repos owned
    by other (special-adapter) families, plus probe-only extra_repos. Profile
    comes from the checkpoint's quant metadata; variants from the dummy
    manifest. targets.yaml holds only the overlay (checkpoint_overrides)."""
    mentioned = collector_mentioned_repos()
    # owner-decided exclusions live in targets.yaml roster.excluded (each entry
    # carries decided_by/reason); any OTHER missing config is a hard stop that
    # goes back to the owner — there is no self-service escape hatch.
    excluded = {e["repo"] for e in (fam.get("excluded") or [])}
    owned_elsewhere = {ck["repo"] for fname, f in targets["families"].items()
                      if not f.get("derive") for ck in f.get("checkpoints", [])}
    repos = sorted((mentioned - excluded - owned_elsewhere)
                   | set(fam.get("extra_repos") or []))
    manifest = json.loads((ROOT / "dummy_models" / "variants_manifest.json").read_text())
    variants_of: dict[str, list[str]] = {}
    for v in manifest["variants"]:
        variants_of.setdefault(v["repo"], []).append(v["variant"].split("__", 1)[1])
    # representative-first ordering: index 0 is the default probe variant
    _head = {"rep": 0, "all_kinds": 1, "rep_mix": 2, "interleave_pair": 3}

    def _rank(n: str):
        if n.startswith("depth"):  # deeper = more faithful; depth8 before depth4
            return (4, -int(n[5:]))
        return (_head.get(n, 9), n)
    for vs in variants_of.values():
        vs.sort(key=_rank)
    overrides = fam.get("checkpoint_overrides") or {}
    out = []
    for repo in repos:
        profile = derive_profile(repo, ROOT / "configs")
        if profile == "MISSING":
            raise SystemExit(f"derive_roster: no fetched config for {repo} — OWNER DECISION NEEDED "
                             f"(fetch the config, or the owner records an exclusion in targets.yaml roster.excluded)")
        ck = {"repo": repo, "profile": profile, "variants": variants_of.get(repo, [])}
        ov = dict(overrides.get(repo) or {})
        if "profile" in ov:  # explicit pin wins, but derivation drift is loud
            if ov["profile"] != profile:
                print(f"derive_roster: {repo} profile pinned {ov['profile']} != derived {profile}")
            ck["profile"] = ov.pop("profile")
        ck.update(ov)
        out.append(ck)
    return out


def _run_id(ck: dict, variant: str, backend: str, version: str, tp: int, kv, platform: str | None = None) -> str:
    """Run id = sha1 of the case identity. ``platform`` (targets platform.name,
    e.g. h20_sm90) is part of it since 2026-09-30 (owner decision after the
    B200 / sm120 campaigns produced the SAME id as the H20 for the same case,
    so a shared workspace or evidence bundle would overwrite raws). Passing
    platform=None yields the pre-2026-09-30 id (used only by the migration)."""
    key = f"{ck['repo']}|{variant}|{backend}|{version}|{ck['profile']}|tp{tp}|kv{kv or 'rendered'}"
    if platform:
        key += f"|{platform}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def migrate_run_ids(root: Path, platform: str, apply: bool = False) -> dict:
    """Rename every pre-2026-09-30 artifact of this workspace to the
    platform-bearing run id: archive/raw/<id>.json + .fp, run_sh/<id>.sh,
    golden/<id>/, and the id fields in plan*.json, records.jsonl,
    archive.jsonl. Idempotent; dry-run unless apply=True."""
    import shutil
    mapping: dict[str, str] = {}
    for pf in sorted((root / "archive").glob("plan*.json")):
        try:
            runs = json.loads(pf.read_text())
        except Exception:
            continue
        for r in runs:
            if not isinstance(r, dict) or r.get("skip") or "repo" not in r:
                continue
            ck = {"repo": r["repo"], "profile": r.get("profile")}
            plat = r.get("platform") or platform
            old = _run_id(ck, r["variant"], r["backend"], r["version"], r["tp"], r.get("kv_dtype"))
            new = _run_id(ck, r["variant"], r["backend"], r["version"], r["tp"], r.get("kv_dtype"), plat)
            if r["id"] == old:
                mapping[old] = new
    stats = {"runs": len(mapping), "renamed": 0, "rewritten_lines": 0, "files": 0}
    if not apply:
        return stats | {"dry_run": True}
    for old, new in mapping.items():
        for rel in (f"raw/{old}.json", f"raw/{old}.fp", f"run_sh/{old}.sh", f"golden/{old}"):
            src, dst = root / "archive" / rel, root / "archive" / rel.replace(old, new)
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst)); stats["renamed"] += 1
        raw = root / "archive" / "raw" / f"{new}.json"
        if raw.exists():  # the raw carries its own id (provenance.id): --records rebuilds from it
            text = raw.read_text()
            if f'"id": "{old}"' in text:
                raw.write_text(text.replace(f'"id": "{old}"', f'"id": "{new}"')); stats["rewritten_lines"] += 1
    for pf in sorted((root / "archive").glob("plan*.json")):
        text = pf.read_text(); n = 0
        for old, new in mapping.items():
            if f'"id": "{old}"' in text:
                text = text.replace(f'"id": "{old}"', f'"id": "{new}"'); n += 1
        if n:
            pf.write_text(text); stats["files"] += 1
    for name in ("records.jsonl", "archive.jsonl"):
        jp = root / "archive" / name
        if not jp.exists():
            continue
        out = []
        for line in jp.read_text().splitlines():
            for old, new in mapping.items():
                if f'"id": "{old}"' in line:
                    line = line.replace(f'"id": "{old}"', f'"id": "{new}"'); stats["rewritten_lines"] += 1
                    break
            out.append(line)
        jp.write_text("\n".join(out) + "\n"); stats["files"] += 1
    return stats


def _oom_at_load(rid: str) -> bool:
    """True when the raw probe for this run id failed at engine LOAD with CUDA OOM
    (weights of the dummy cut do not fit the probe GPU)."""
    p = ROOT / "archive" / "raw" / f"{rid}.json"
    if not p.exists():
        return False
    try:
        err = str((json.loads(p.read_text()).get("errors") or {}).get("load") or "")
    except (OSError, ValueError):
        return False
    # torch / vllm / sglang: OutOfMemoryError; trtllm executor: "insufficient GPU memory"
    return "OutOfMemoryError" in err or "CUDA out of memory" in err or "insufficient GPU memory" in err


# ------------------------------------------------------------ execution identity
# The run id names the CASE (repo, cut, backend, version, profile, tp, kv). It
# does not change when the generator renders different engine args, the dummy
# changes or the image moves — so a raw produced under an older configuration
# must not count as evidence for the current one (review 2026-09-25 P1 #3).
# The execution fingerprint covers what shapes the execution: the engine
# invocation the probe consumed, the dummy checkpoint's config, the image and
# the kv override. The queue skips a run only when a sidecar recorded THIS
# fingerprint; records carry both sides and the matrix refuses stale evidence.
# deployment-wrapper / path values: the dynamo benchmark flags and result paths
# are consumed by the wrapper, the probe strips them before the engine sees argv
_ENGINE_DROP = {"--model", "--model-path", "--served-model-name", "--dump-config-to",
                "--benchmark-output-path", "--benchmark-mode", "--tokenizer", "--tokenizer-path"}


def engine_tokens(backend: str, tokens: list[str]) -> list[str]:
    """Engine invocation tokens with the deployment-specific values (model
    path, served name, result paths) removed, so the rendered command and
    the probe's argv (which substitutes the dummy dir) compare equal."""
    # both sides start at the first flag: the render carries the interpreter and
    # module (python3 -m dynamo.vllm), the probe's argv may not
    first = next((i for i, t in enumerate(tokens) if str(t).startswith("--")), 0)
    out, skip = [], False
    for t in tokens[first:]:
        if skip:
            skip = False
            continue
        key = t.split("=", 1)[0]
        if key in _ENGINE_DROP:
            skip = "=" not in t
            continue
        out.append(t)
    return out


def _run_sh_engine_tokens(text: str) -> list[str]:
    m = re.search(r"engine_command=\((.*?)\)\n", text, re.S)
    if not m:
        m = re.search(r"python3 -m dynamo\.\w+[^\n]*", text)
        return shlex.split(m.group(0)) if m else []
    return shlex.split(m.group(1).replace("\\\n", " "))


def _sha(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode()); h.update(b"\0")
    return h.hexdigest()[:12]


def plan_engine_tokens(run: dict) -> list[str] | None:
    """The engine invocation the probe will consume, from the render artifact."""
    be = run["backend"]
    if be == "vllm" and run.get("run_sh"):
        return engine_tokens(be, _run_sh_engine_tokens(Path(run["run_sh"]).read_text()))
    if be == "sglang" and run.get("engine_cli"):
        return engine_tokens(be, shlex.split(run["engine_cli"]))
    if be == "trtllm" and run.get("render_artifact"):
        # yaml compared semantically (canonical json), so formatting never counts
        return [json.dumps(yaml.safe_load(Path(run["render_artifact"]).read_text()), sort_keys=True)]
    return None


def dummy_fingerprint(model_dir_in_container: str) -> str | None:
    d = ROOT / str(model_dir_in_container).replace(f"{WORK}/", "", 1)
    parts = []
    for name in ("config.json", "hf_quant_config.json"):
        f = d / name
        if f.exists():
            parts.append(f"{name}:{f.read_text()}")
    return _sha(*parts) if parts else None


def exec_fingerprint(run: dict) -> dict:
    """{fingerprint, engine, dummy, image, kv} — engine = sha of the engine
    tokens (None when the render artifact is unreadable), dummy = sha of the
    dummy's config files."""
    toks = plan_engine_tokens(run)
    eng = _sha(*toks) if toks else None
    dum = dummy_fingerprint(run.get("model_dir", ""))
    probe = _HERE / "probes" / (f"probe_{run['backend']}_server.py"
                                if run["backend"] in ("sglang", "trtllm") and int(run.get("tp", 1)) > 1
                                else f"probe_{run['backend']}.py")
    code = _sha(probe.read_text()) if probe.exists() else None   # the probe code that will run
    fp = _sha(eng or "", dum or "", code or "", str(run.get("image")), str(run.get("kv_dtype")),
              json.dumps(run.get("cli_extra_args") or [], sort_keys=True))
    return {"fingerprint": fp, "engine": eng, "dummy": dum, "probe_code": code,
            "image": run.get("image"), "kv": run.get("kv_dtype")}


def raw_engine_fingerprint(backend: str, raw: dict) -> str | None:
    """Engine fingerprint of what a raw probe actually ran (vllm: argv; sglang:
    the engine cli it was given); trtllm raws carry the yaml as a dict, so
    None — the sidecar is the only evidence there."""
    if backend == "vllm" and isinstance(raw.get("engine_argv"), list):
        return _sha(*engine_tokens(backend, [str(t) for t in raw["engine_argv"]]))
    if backend == "sglang" and isinstance(raw.get("engine_cli"), str):
        return _sha(*engine_tokens(backend, shlex.split(raw["engine_cli"])))
    if backend == "trtllm" and raw.get("engine_yaml") is not None:
        ey = raw["engine_yaml"]
        ey = yaml.safe_load(ey) if isinstance(ey, str) else ey
        return _sha(json.dumps(ey, sort_keys=True))
    return None


def evidence_status(run: dict, sidecar_fp: str | None, raw: dict | None) -> str:
    """current | stale | unverified — does this raw belong to THIS execution
    configuration? Sidecar (written by the queue next to the raw) is decisive;
    without one, vllm/sglang raws are checked on the engine invocation they
    recorded; anything else is unverified (legacy evidence, never 'current')."""
    fp = run.get("exec_fingerprint") or {}
    if sidecar_fp:
        return "current" if sidecar_fp == fp.get("fingerprint") else "stale"
    if raw is not None and fp.get("engine"):
        eng = raw_engine_fingerprint(run["backend"], raw)
        if eng is not None:
            return "current" if eng == fp["engine"] else "stale"
    return "unverified"


def select_dummy_dir(dummy_root: Path, repo_tag: str, variant: str, fam_name: str,
                     fam: dict, ck: dict) -> tuple[Path | None, str | None]:
    """(dummy dir, skip reason). Dummy dirs are keyed by ADAPTER family.

    When targets declare ``dummy_overrides.family`` for the checkpoint (or its
    family), ONLY that adapter's dir is this checkpoint's dummy: a dir built by
    another adapter — typically the generic depth cut that predates the override
    — is a different model (DeepSeek-V4.1 2026-10-01: the generic 2-layer cut, two
    sliding-window layers with stale source pointers, was probed under a dsv41
    override until the plan was rebuilt). Without an override: the targets family
    dir, then generic, then any adapter dir (a roster repo may use a special one)."""
    want = ((ck.get("dummy_overrides") or {}).get("family")) or ((fam.get("dummy_overrides") or {}).get("family"))
    leaf = f"{repo_tag}__{variant}"
    if want:
        vdir = dummy_root / want / leaf
        if vdir.exists():
            return vdir, None
        other = next((d.name for d in sorted(dummy_root.iterdir()) if d.is_dir() and (d / leaf).exists()), None)
        if other:
            return None, (f"dummy {leaf} was built by adapter '{other}' but targets declare "
                          f"dummy_overrides.family '{want}' — rebuild with components/dummies.py")
        return None, f"no dummy dir {want}/{leaf} (dummy_overrides.family '{want}'; run components/dummies.py)"
    order = [fam.get("dummy_dir") or fam_name, "generic"] + sorted(d.name for d in dummy_root.iterdir() if d.is_dir())
    for famdir in dict.fromkeys(order):
        vdir = dummy_root / famdir / leaf
        if vdir.exists():
            return vdir, None
    return None, f"no dummy dir {leaf}"


def enumerate_runs(targets: dict, full: bool, backends: list[str]) -> list[dict]:
    runs = []
    topos = [t for t in targets["topologies"] if t["evidence"] == "real" and (full or t["tp"] == 1)]
    # topology_policy.tp_required: checkpoints whose representative probe must run
    # at a declared tp (the dummy does not fit one GPU, e.g. Qwen3.8-2.4T). They
    # replace the tp1 topology for that (repo, backend) only; the route is the
    # backend's multi-GPU probe (sglang: probe_sglang_server.py).
    tp_required = {(e["repo"], e.get("backend", "sglang")): int(e["tp"])
                   for e in (targets.get("topology_policy") or {}).get("tp_required") or []}

    def _topos_for(repo: str, backend: str) -> list[dict]:
        req = tp_required.get((repo, backend))
        if req is None:
            return topos
        picked = [t for t in targets["topologies"] if t["evidence"] == "real" and t["tp"] == req and t.get("ep", 1) == 1]
        if not picked:
            raise ValueError(f"topology_policy.tp_required asks tp{req} for {repo}/{backend} but targets.topologies has no such real topology")
        return picked
    for backend in backends:
        be = targets["backends"][backend]
        versions = be["versions"] if full else [be["versions"][-1]]
        for fam_name, fam in targets["families"].items():
            if fam.get("derive") and "checkpoints" not in fam:
                fam["checkpoints"] = derive_roster_checkpoints(fam, targets)
            variants = fam.get("dummy_variants") or []
            if not variants and not any(c.get("variants") for c in fam["checkpoints"]):
                continue  # adapter pending (kimi_k3)
            override = (fam.get("variant_overrides") or {}).get(backend)
            for ck in fam["checkpoints"]:
                repo_tag = ck["repo"].split("/")[-1]
                # per-checkpoint variants win (architectures in a mixed family
                # each have their own layer kinds); else the family list
                ck_variants = ck.get("variants") or variants
                if not ck_variants:
                    # roster repo with no dummy variant yet (config fetched, dummy
                    # not generated): loud, not a crash — the plan still covers
                    # every other checkpoint and the gap is visible in the log
                    print(f"enumerate_runs: {ck['repo']} has no dummy variants — skipped "
                          f"(run gen_dummy_models.py)", file=sys.stderr)
                    continue
                ck_override = (ck.get("variant_overrides") or {}).get(backend) or override
                fallback_from = None
                if full or ck_override:
                    use_variants = ck_variants if full else [ck_override]
                else:
                    # capacity fallback (observed, never predicted): when the
                    # representative cut OOMed at LOAD on this backend, the next
                    # smaller faithful cut becomes the representative (depth8 ->
                    # depth4, all_kinds -> single-kind). The switch is recorded
                    # on the run so records/matrix show which cut was probed.
                    # Found 2026-09-25: Qwen3.8-2.4T-A95B depth8 (8 x 26B params)
                    # cannot fit one GPU; the driver only ever queued index 0.
                    chosen = ck_variants[0]
                    for i, v in enumerate(ck_variants):
                        chosen = v
                        rid0 = _run_id(ck, v, backend, versions[-1], topos[0]["tp"], None,
                                       (targets.get("platform") or {"name": "h20_sm90"})["name"])  # same default as the run ids below
                        if _oom_at_load(rid0) and i + 1 < len(ck_variants):
                            fallback_from = v
                            continue
                        break
                    use_variants = [chosen]
                for variant in use_variants:
                    # dummy dirs are keyed by ADAPTER family (a roster repo may
                    # still use a special adapter) — search every adapter dir,
                    # preferring the targets family, then generic, then the rest
                    vdir, _why = select_dummy_dir(ROOT / "dummy_models", repo_tag, variant, fam_name, fam, ck)
                    if vdir is None:
                        runs.append({"skip": _why, "repo": ck["repo"], "variant": variant})
                        continue
                    for version in versions:
                        for topo in _topos_for(ck["repo"], backend):
                            # kv-cache dtype is a first-class serving-config axis
                            # the collector sweeps; probe it on every backend
                            # whose probe can override it (vllm --kv-cache-dtype,
                            # sglang --kv-dtype, trtllm --kv-dtype over the engine
                            # yaml's kv_cache_config.dtype). The golden sglang CLI for fp8 profiles does
                            # NOT render a kv dtype, so without this variant the
                            # sglang fp8-KV DSA/MLA paths were never probed
                            # (found 2026-09-24: the old records' fp8 KV came
                            # from a retired probe injection, not the generator).
                            kv_variants = {"vllm": [None, "fp8"], "sglang": [None, "fp8_e4m3"],
                                           "trtllm": [None, "fp8"]}.get(backend, [None])
                            for kv in kv_variants:
                                plat = targets.get("platform") or {"name": "h20_sm90", "sm": 90, "system": "h200_sxm"}
                                rid = _run_id(ck, variant, backend, version, topo["tp"], kv, plat["name"])
                                runs.append({
                                    "id": rid, "family": fam_name, "repo": ck["repo"], "profile": ck["profile"],
                                    "platform": plat["name"], "sm": plat["sm"], "system": plat["system"],
                                    "variant": variant, "backend": backend, "version": version,
                                    "image": be["images"][version], "tp": topo["tp"],
                                    "kv_dtype": kv,
                                    "model_dir": f"{WORK}/{vdir.relative_to(ROOT)}",
                                    "aic_registered": ck.get("aic_registered", False),
                                    **({"capacity_fallback_from": fallback_from} if fallback_from else {}),
                                    "cli_extra_args": (list(be.get("cli_extra_args") or [])
                                                       + _cea((ck.get("cli_extra_args") or {}).get(backend)
                                                              or (fam.get("cli_extra_args") or {}).get(backend))),
                                })
    return runs



def _generator_src_commit() -> dict:
    """Record WHICH generator code rendered the engine args, so archive
    provenance survives checkout/branch changes."""
    import subprocess

    repo = str(Path(AIS_SRC).parent)
    try:
        rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        branch = subprocess.run(["git", "-C", repo, "branch", "--show-current"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = branch = "unknown"
    return {"generator_src": repo, "generator_commit": rev, "generator_branch": branch or "detached"}


def emit_queues(runs: list[dict], gpu_list: list[int], plan_name: str) -> None:
    SCRATCH_QUEUES.mkdir(parents=True, exist_ok=True)
    src_info = _generator_src_commit()
    for run in runs:
        if "skip" not in run:
            run.update(src_info)
    (ROOT / "archive" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "archive" / "run_sh").mkdir(parents=True, exist_ok=True)
    queues: dict[int, list[str]] = {g: [] for g in gpu_list}
    for i, run in enumerate(r for r in runs if "skip" not in r):
        g = gpu_list[i % len(gpu_list)]
        dev = str(g)
        if int(run.get("tp", 1)) > 1:
            # a tp>1 run owns tp GPUs: the first tp of --gpu-list, queued on the
            # first GPU's queue. Run such plans ALONE (emit with --only), the other
            # per-GPU queues would otherwise share those devices.
            if run["backend"] not in ("sglang", "trtllm"):
                run["skip"] = f"tp{run['tp']} probe route not implemented for {run['backend']}"
                continue
            if len(gpu_list) < int(run["tp"]):
                run["skip"] = f"tp{run['tp']} needs {run['tp']} GPUs in --gpu-list, got {len(gpu_list)}"
                continue
            g = gpu_list[0]
            dev = ",".join(str(x) for x in gpu_list[: int(run["tp"])])
        # skip only when a sidecar recorded THIS execution fingerprint next to the
        # raw: a raw from an older render / dummy / image is not evidence for the
        # current plan (review 2026-09-25); the sidecar is written after the run
        head = (f"[ -f {ROOT}/archive/raw/{run['id']}.json ] && "
                f"[ \"$(cat {ROOT}/archive/raw/{run['id']}.fp 2>/dev/null)\" = \"__FP__\" ] && "
                f"echo 'skip [{run['id']}] (done)' || {{ "
                f"echo '### [{run['id']}] {run['backend']} {run['repo']} {run['variant']} "
                f"{run['version']} tp{run['tp']}' && timeout 1500 docker run --rm "
                f"--gpus '\"device={dev}\"' --shm-size 16g -e HF_HUB_OFFLINE=1 "
                # host runs an MPS daemon; probes must NOT attach (a fake pipe
                # dir makes the CUDA client fall back to a normal context)
                f"-e CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps "
                f"-v {ROOT}:{WORK} -v {ROOT}/jitcache:/root/.cache {_EXTRA_MOUNT}"
                f"-e TRITON_CACHE_DIR=/root/.cache/triton -e DG_JIT_CACHE_DIR=/root/.cache/deep_gemm ")
        if run["backend"] == "sglang":
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            cli = extract_sglang_cli_from_run_sh(art / "run_0.sh")
            run["engine_cli"] = cli
            run["engine_args_fidelity"] = "cli-golden"
            run["golden_dir"] = str(art)
            _kv = f" --kv-dtype {run['kv_dtype']}" if run.get("kv_dtype") else ""
            if int(run.get("tp", 1)) > 1:
                run["probe_route"] = "launch_server"
                cmd = (head + f"{run['image']} python3 {PROBES_IN_CONTAINER}/probe_sglang_server.py "
                       f"--model {run['model_dir']} --engine-cli {shlex.quote(cli)} --tp {run['tp']}{_kv} "
                       f"--out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
            else:
                cmd = (head + f"{run['image']} python3 {PROBES_IN_CONTAINER}/probe_sglang.py "
                       f"--model {run['model_dir']} --engine-cli {shlex.quote(cli)} --trace{_kv} "
                       f"--out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        elif run["backend"] == "vllm":  # golden fpm run.sh, consumed verbatim
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            src = next((p for p in (art / "run.sh", art / "run_0.sh") if p.exists()), None)
            if src is None:
                run["skip"] = f"golden has no run.sh: {art}"
                continue
            rsh = ROOT / "archive" / "run_sh" / f"{run['id']}.sh"
            rsh.write_text(src.read_text())
            run["run_sh"] = str(rsh)
            run["engine_args_fidelity"] = "cli-golden"
            run["golden_dir"] = str(art)
            _kv = f" --kv-cache-dtype {run['kv_dtype']}" if run.get("kv_dtype") else ""
            cmd = (head + f"--entrypoint python3 {run['image']} {PROBES_IN_CONTAINER}/probe_vllm.py "
                   f"--run-sh {WORK}/archive/run_sh/{run['id']}.sh --model-override {run['model_dir']}{_kv} "
                   f"--trace --out {WORK}/archive/raw/{run['id']}.json 2>&1 | tail -1 ; }}")
        else:  # trtllm: golden extra_engine_args (agg_config.yaml), consumed verbatim
            art = render_golden(run)
            if art is None:
                run["skip"] = f"golden render failed: {run.get('golden_error')}"
                continue
            src = art / "agg_config.yaml"
            if not src.exists():
                run["skip"] = f"golden has no agg_config.yaml: {art}"
                continue
            eyml = ROOT / "archive" / "run_sh" / f"{run['id']}.engine.yaml"
            eyml.write_text(src.read_text())
            run["engine_args_fidelity"] = "cli-golden"
            run["render_artifact"] = str(eyml)
            run["golden_dir"] = str(art)
            # any checkpoint with custom code (auto_map) needs it; cheapest
            # correct rule is to always pass it for dummy probing
            trc = "--trust-remote-code "
            _kv = f"--kv-dtype {run['kv_dtype']} " if run.get("kv_dtype") else ""
            if int(run.get("tp", 1)) > 1:
                run["probe_route"] = "trtllm-serve"
                cmd = (head + f"{run['image']} bash -lc 'python3 {PROBES_IN_CONTAINER}/probe_trtllm_server.py "
                       f"--model {run['model_dir']} --tp {run['tp']} {trc}{_kv}"
                       f"--engine-yaml {WORK}/archive/run_sh/{run['id']}.engine.yaml "
                       f"--out {WORK}/archive/raw/{run['id']}.json' "
                       f"2>&1 | tail -1 ; }}")
            else:
                cmd = (head.replace("docker run --rm ",
                                    "docker run --rm -e TLLM_WORKER_USE_SINGLE_PROCESS=1 ")
                       + f"{run['image']} bash -lc 'python3 {PROBES_IN_CONTAINER}/probe_trtllm.py "
                       f"--model {run['model_dir']} {trc}{_kv}"
                       f"--engine-yaml {WORK}/archive/run_sh/{run['id']}.engine.yaml "
                       f"--out {WORK}/archive/raw/{run['id']}.json' "
                       f"2>&1 | tail -1 ; }}")
        run["exec_fingerprint"] = exec_fingerprint(run)
        _fp = run["exec_fingerprint"]["fingerprint"]
        cmd = cmd.replace("__FP__", _fp)
        # keep the console tail of every probe: a hard native abort (trtllm attentionOp assert, sglang argparse) kills
        # the probe before it can write a raw, and the matrix could only say "crashed before dump" (sm89 Gemma-4 NVFP4:
        # "Head size 512 is not supported by MMHA"; B200 handoff defect 1). build_matrix classifies from this file.
        (ROOT / "archive" / "logs").mkdir(parents=True, exist_ok=True)
        cmd = cmd.replace("2>&1 | tail -1 ;", f"2>&1 | tail -n 400 | tee {ROOT}/archive/logs/{run['id']}.tail | tail -1 ;")
        if cmd.rstrip().endswith("; }"):
            cmd = cmd.rstrip()[:-3] + f"; echo {_fp} > {ROOT}/archive/raw/{run['id']}.fp ; }}"
        queues[g].append(cmd)
    for g, cmds in queues.items():
        p = SCRATCH_QUEUES / f"gpu{g}.sh"
        p.write_text("#!/bin/bash\n" + "\n".join(cmds) + f"\necho ARCHIVE_QUEUE_GPU{g}_DONE\n")
        print(f"{p}: {len(cmds)} jobs")
    (ROOT / "archive" / plan_name).write_text(json.dumps(runs, indent=1))
    print(f"plan: {ROOT / 'archive' / plan_name} ({sum(1 for r in runs if 'skip' not in r)} runs, "
          f"{sum(1 for r in runs if 'skip' in r)} skipped)")
    _nofacts = [r for r in runs if "skip" not in r and (r.get("golden_facts") or {}).get("applied") is False]
    if _nofacts:
        print(f"WARNING: {len(_nofacts)} runs rendered WITHOUT generator model facts (golden_facts.applied=false; "
              f"model defaults such as block-size / trust-remote-code are absent from their engine args) — "
              f"e.g. {_nofacts[0]['id']}: {(_nofacts[0]['golden_facts'] or {}).get('reason')}", file=sys.stderr)


def check_coverage(targets: dict) -> None:
    """Coverage floor: the collector's declared model roster is a LOWER bound
    for probe targets (targets may exceed it, never trail it)."""
    mentioned = collector_mentioned_repos()
    for fam in targets["families"].values():
        if fam.get("derive") and "checkpoints" not in fam:
            fam["checkpoints"] = derive_roster_checkpoints(fam, targets)
    covered = {ck["repo"] for fam in targets["families"].values() for ck in fam["checkpoints"]}
    excluded = {e["repo"] for f in targets["families"].values()
                for e in (f.get("excluded") or [])}
    missing = sorted(mentioned - covered - excluded)
    print(f"collector mentions {len(mentioned)} repos; targets cover {len(covered)}; "
          f"owner-excluded {len(mentioned & excluded)}")
    if missing:
        print("MISSING FROM TARGETS (coverage floor violated):")
        for r in missing:
            print("  ", r)
        raise SystemExit(1)
    print("coverage floor satisfied")



# ---------------------------------------------------------------------------
# checkpoint quant profile (merged from derive_profile.py): derived from the
# checkpoint's own quant metadata, NEVER the repo name (dummy fidelity rule 4)
def derive_profile(repo: str, configs_dir: Path) -> str:
    stem = repo.replace('/', '_')
    p = configs_dir / f'{stem}.json'
    if not p.exists():
        return 'MISSING'
    c = json.loads(p.read_text())
    qc = c.get('quantization_config') or (c.get('text_config') or {}).get('quantization_config')
    hq_p = configs_dir / f'{stem}_hfquant.json'
    if hq_p.exists():
        algo = ((json.loads(hq_p.read_text()).get('quantization') or {}).get('quant_algo') or '').upper()
        if 'NVFP4' in algo or 'FP4' in algo:
            return 'nvfp4'
        if 'MXFP8' in algo:
            return 'mxfp8'
        if 'FP8' in algo:
            return 'fp8'
    if not qc:
        return 'bfloat16'

    def groups_have_4bit_float() -> bool:
        for g in (qc.get('config_groups') or {}).values():
            for part in ('weights', 'input_activations'):
                w = g.get(part) or {}
                if isinstance(w, dict) and w.get('num_bits') == 4 and w.get('type') == 'float':
                    return True
        # modelopt's other MIXED_PRECISION shape: flat per-layer dict
        for v in (qc.get('quantized_layers') or {}).values():
            if isinstance(v, dict) and 'FP4' in str(v.get('quant_algo', '')).upper():
                return True
        return False

    m = (qc.get('quant_method') or '').lower()
    algo = (qc.get('quant_algo') or '').upper()
    if m == 'mxfp8':
        return 'mxfp8'
    if m == 'mxfp4':
        return 'mxfp4'
    if m in ('modelopt', 'modelopt_mixed') or algo == 'MIXED_PRECISION':
        return 'nvfp4' if (groups_have_4bit_float() or 'FP4' in algo) else 'fp8'
    if m == 'compressed-tensors':
        # 4-bit float groups = fp4-family weights (e.g. Kimi-K3 native: w4 float
        # group-32); 4-bit int = packed w4 (marlin path), served fp8-activation
        if groups_have_4bit_float():
            return 'nvfp4'
        return 'fp8'
    if m == 'fp8':
        return 'fp8'
    return f'?{m}'

# ---------------------------------------------------------------------------
# records stage (merged from make_records.py): raw probe JSONs -> curated
# records.jsonl — kernel normalization, taxonomy labeling, error compression

# kernels that are infrastructure, never op identity
# attention-ish module classes whose forward spans define the attention identity
ATTN_CLASS_RE = re.compile(r"Attention|Attn|MLA|Mixer|SSM|DeltaRule|GatedDelta|KDA|Mamba|Impl$|Backend$|Flash|Indexer|Compressor|Sparse")
# linear-attention / SSM kernel families (fla, KDA, mamba2) — identity of hybrid models
LINEAR_ATTN_KERNEL_RE = re.compile(r"chunk_gated_delta|delta_rule|gdn_decode|gdn_prefill|fused_recurrent|_kda_|kda_|mamba|_chunk_scan|_chunk_state|_state_passing|selective_state|_fwd_recompute_w_u", re.I)
KERNEL_DENY = re.compile(
    r"Memcpy|Memset|^memcpy\d|^memset\d|Lazy Function Loading|Runtime Triggered Module Loading|"
    r"at::native::(vectorized_elementwise|elementwise|index_elementwise|"
    r"unrolled_elementwise|reduce_kernel|distribution_|fill|indexSelect|index_put|"
    r"vectorized_layer_norm|CatArrayBatchedCopy|write_indices|cunn_|sort|radix|"
    r"mbtopk|gatherTopK|arange|triu_tril|masked_scale|_scatter_gather|"
    r"bitonic|cumsum|tensor_kernel_scan|upsample|multi_tensor_apply)|aten::(fill_|copy_|zero_)|"
    # a device symbol literally named "kernel" (vllm 0.30 cute-DSL glue) carries no identity
    r"^kernel$|"
    # trtllm MoE tactic profiler + stream-delay glue: the autotuner, not the serving path
    r"cutlass_kernels::populateRandomBufferKernel|cutlass_kernels::prepareFakeRouterBuffers|kernels::delayStreamKernel|"
    r"^void at::native::.*FillFunctor"
)
# wrapper identifiers to skip when extracting a meaningful kernel name
NAME_WRAPPERS = {"void", "cutlass::device_kernel", "flash::enable_sm90_or_later",
                 "cute", "std", "c10", "at", "at::native", "int", "bool", "float",
                 "unsigned", "long", "char",
                 # "(anonymous namespace)::kernel<...>" — the namespace words are not the kernel
                 "anonymous", "namespace"}
FRAME_DENY = re.compile(r"_inductor/runtime|pybind11_detail|<built-in method")
FW_FRAME = re.compile(r"(sglang|vllm|tensorrt_llm|cutlass|flashinfer|deep_gemm|sgl_kernel|flash)")


_TRITON_TILE_SUFFIX = re.compile(r"_(?:\d+x){3}\d+(?=(?:_[a-z]\w*)?$)")
# cute-DSL symbols append every tensor spec ("_tensorptrbf16gmemalign32o1291612…",
# "_object_a…") to the kernel name; identity ends at the kernel name.
_CUTEDSL_PARAM_SUFFIX = re.compile(r"_(tensorptr|object).*$")


def normalize_kernel(name: str) -> str | None:
    # "(anonymous namespace)::" carries no identity and, left in place, hides
    # the at::native:: glue from KERNEL_DENY and the kernel name from the
    # identifier scan (K3's KDA decode kernel read as "namespace", 2026-09-25)
    name = name.replace("(anonymous namespace)::", "")
    if KERNEL_DENY.search(name):
        return None
    name = name.split("(")[0] if "<" not in name else name  # plain symbol: drop the (ParamType) tail
    if "<" not in name and " " not in name.strip():
        # Triton autotuned kernels (triton_kernels matmul_ogs) bake the tile
        # config into the name — _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1
        # at M=1 vs ..._128x256x128x1 at M=4096 — the same kernel, different
        # BLOCK sizes chosen from the token count. Identity is the kernel, not
        # the tile: strip the suffix (an epilogue marker like _swiglu survives).
        if name.startswith("kernel_cutlass_"):
            # cute-DSL instantiations bake shape parameters into the symbol
            # (…gdn_decode_bf16state_mtp_ilp4_kernel_tensorptrbf16gmemalign32o1291612812828);
            # identity is the kernel, so drop the trailing parameter blob.
            return _CUTEDSL_PARAM_SUFFIX.sub("", name)[:80]
        return _TRITON_TILE_SUFFIX.sub("", name)[:80]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]{5,}", name)
    for ident in idents:
        if (ident in NAME_WRAPPERS or ident.split("::")[0] in NAME_WRAPPERS
                or "anonymous" in ident):
            continue
        return ident[:80]
    return name.split("(")[0][:80]


def clean_path(path: str) -> str:
    parts = [p.strip() for p in path.split("<-")]
    kept = [p for p in parts if not FRAME_DENY.search(p)]
    return " <- ".join(kept[:5])


def current_sm() -> str:
    """The SM every per-SM artifact is keyed by (kernel_taxonomy_<sm>.yaml,
    results/pathdiff/<sm>/, results/retests/<sm>/, results/<sm>/). Source of
    truth is targets.yaml `platform.sm` (the box this checkout is pinned to);
    AIS_SM only overrides it (grading another SM's evidence from here). Until
    2026-09-30 the default was a hard-coded sm90, so a forgotten export on an
    sm120 box labelled its records with the sm90 vocabulary without a word
    (RTX 5000 handoff §1.2 A)."""
    env = os.environ.get("AIS_SM")
    if env:
        return env
    try:
        import yaml
        plat = yaml.safe_load((Path(__file__).resolve().parents[1] / "targets.yaml").read_text()).get("platform") or {}
        if plat.get("sm"):
            return f"sm{int(plat['sm'])}"
    except Exception:
        pass
    return "sm90"


def load_taxonomy():
    import yaml
    # Per-SM vocabulary files (owner decision 2026-09-20: the verdict plane
    # forks by SM — a platform's kernel names are that platform's data, and
    # separate files delete the cross-arch audit discipline a shared file
    # demanded). Discoverability rule: kernel_taxonomy_<sm>.yaml, enumerable
    # by glob; a session working sm103 never touches the sm90 file.
    sm = current_sm()
    path = Path(__file__).parent / f"kernel_taxonomy_{sm}.yaml"
    rules = yaml.safe_load(path.read_text())["rules"]
    return [(re.compile(r["match"]), r["backend"], r["role"]) for r in rules]


_TAXONOMY = load_taxonomy()


def attn_identity_label(kernel: str) -> str:
    """Identity-column label for an attention kernel: the taxonomy BACKEND
    label (fa3, trtllm_fmha, triton_mla, flashmla_sparse, ...) when the
    vocabulary knows the kernel, else the normalized kernel name. Raw names
    are fine on SM90 (FlashAttnFwdSm90) but on Blackwell the FMHA cubin name
    is an 80-char shape blob (B200 handoff 2026-09-29: fmhaSm100fKernel_Qkv
    Bfloat16...P16MultiCtasKvVarSeqQ8Kv25) — unreadable and never equal
    across models, so the column was noise. The raw kernel stays in the
    record's kernel tables; only the column changes."""
    name = normalize_kernel(kernel)
    for rx, backend, role in _TAXONOMY:
        if role in ("attention", "dsa_indexer", "linear_attention") and rx.search(name):
            return backend
    return name

# Orphan keep-rule (build_ops): labels that carry no path identity — a kernel
# whose ONLY labels are these is noise for path_diff and may be capped.
_ORPHAN_REST_CAP = 24  # unlabeled orphans kept per record, by device time


def label_kernels(kernels):
    """kernel names -> ({backend labels}, {unmatched kernels})."""
    labels, unmatched = set(), set()
    for k in kernels:
        for rx, backend, role in _TAXONOMY:
            if rx.search(k):
                if backend not in ("infra", "framework_native"):
                    labels.add(backend)
                break
        else:
            unmatched.add(k)
    return labels, unmatched


def kernel_role(kernel: str):
    """First matching taxonomy rule -> (backend, role); None when unlabeled.
    The role axis is what path_diff grades on (target op vs auxiliary work)."""
    for rx, backend, role in _TAXONOMY:
        if rx.search(kernel):
            return backend, role
    return None


def label_kernel_roles(kernels) -> dict:
    """kernel -> {"backend", "role"} for every labeled kernel."""
    out = {}
    for k in kernels:
        hit = kernel_role(k)
        if hit:
            out[k] = {"backend": hit[0], "role": hit[1]}
    return out


_ORPHAN_TABLES = (("prefill_kernels", "prefill"), ("decode_kernels", "decode"),
                  ("profile_run_kernels", "profile_run"))


def _orphan_table_name_ok(name: str) -> bool:
    return not (name.startswith(("AIC::", "step", "aten::")) or name.isupper()
                or re.match(r"^(sglang|sgl_kernel|_\w*C\w*|triton_)\w*::", name))


def build_orphan_phases(facts: dict) -> dict:
    """normalized kernel -> sorted phases it ran in (prefill / decode /
    profile_run), from the device-stream tables. Phase is identity: a
    prefill gate must not borrow decode evidence (review 2026-09-25)."""
    trace = facts.get("trace") or facts
    phases: dict[str, set] = {}
    for tbl, phase in _ORPHAN_TABLES:
        for k in (trace.get(tbl) or []):
            name = k.get("kernel", "")
            if not _orphan_table_name_ok(name):
                continue
            n = normalize_kernel(name)
            if n:
                phases.setdefault(n, set()).add(phase)
    return {k: sorted(v) for k, v in phases.items()}


def build_ops(facts: dict) -> tuple[list[dict], list[str]]:
    """Merge api_trace spans into ops; return (ops, orphan_kernels)."""
    ops: list[dict] = []
    attributed: set[str] = set()
    trace = facts.get("trace") or facts  # sglang nests under trace; vllm flat
    # vllm's single api_trace table is not split by phase: those ops carry
    # phase None (unknown), never a borrowed "decode" (review 2026-09-25)
    for phase_key, phase in (("prefill_api", "prefill"), ("decode_api", "decode"), ("api_trace", None)):
        spans = trace.get(phase_key) or {}
        merged: dict[frozenset, dict] = {}
        for span, s in spans.items():
            _, kind, detail = span.split("::", 2)
            kerns = set(filter(None, (normalize_kernel(k) for k in s.get("kernels", {}))))
            attributed |= kerns
            key = frozenset(kerns) or frozenset({span})
            slot = merged.setdefault(key, {"phase": phase, "op": None, "quant": None,
                                           "api": None, "kernels": sorted(kerns),
                                           "calls": s.get("calls", 0)})
            if kind == "quant_apply":
                slot["quant"] = detail
            else:
                slot["op"] = f"{kind}:{detail}"
            paths = [clean_path(p) for p in (s.get("py_paths") or {})
                     if not FRAME_DENY.search(p) and not KERNEL_DENY.search(p.split("<-")[0])]
            if paths and (slot["api"] is None or kind != "quant_apply"):
                slot["api"] = paths[0]
        for v in merged.values():
            if v["kernels"] or v["op"]:
                labels, unmatched = label_kernels(v["kernels"])
                v["backends"] = sorted(labels) or None
                if unmatched:
                    v["unclassified_kernels"] = sorted(unmatched)
                ops.append(v)
    # trtllm probe: flat kernels list, no spans
    if not ops and facts.get("kernels"):
        kerns = sorted(set(filter(None, (normalize_kernel(k["kernel"]) for k in facts["kernels"]))))
        attributed |= set(kerns)
        labels, unmatched = label_kernels(kerns)
        ops.append({"phase": "generate", "op": "all", "quant": None, "api": None,
                    "kernels": kerns, "calls": 1, "backends": sorted(labels) or None,
                    "unclassified_kernels": sorted(unmatched) or None})
    # Orphan kernels are those the span attribution missed — critically, the
    # cudagraph-replayed attention kernels (no CPU launch event, so no span),
    # which live only in the device-stream prefill/decode tables. The list is
    # path_diff's serving-side identity evidence, so the keep rule is
    # structural, not a count: EVERY orphan the taxonomy labels with an
    # identity-bearing backend is kept, and only the unlabeled/noise remainder
    # is capped (by device time, for the needs-taxonomy backlog). History: an
    # alphabetical [:15] cut dropped flash_fwd_splitkv_mla_fp8_sparse /
    # get_mla_metadata on half the records (137/274 raws exceed 15 orphans),
    # so the fp8 attention path never reached records.jsonl and path_diff saw
    # an empty serving signal (fp8-KV probe sweep 2026-09-23; the B300 rerun
    # hit the same bug independently). A time-ranked top-40 fixed today's
    # data but was still a count; this rule cannot lose a labeled kernel.
    orphans_t: dict[str, float] = {}
    # profile_run_kernels: vLLM's own dummy full-model forward at engine
    # construction (the only place vision-encoder kernels run for a text
    # probe) — legitimate serving execution, so it counts as identity evidence
    for tbl in ("prefill_kernels", "decode_kernels", "profile_run_kernels"):
        for k in (trace.get(tbl) or []):
            name = k.get("kernel", "")
            if (name.startswith(("AIC::", "step", "aten::")) or name.isupper()
                    or re.match(r"^(sglang|sgl_kernel|_\w*C\w*|triton_)\w*::", name)):
                continue  # spans, phase markers, custom-op launchers — not kernels
            n = normalize_kernel(name)
            if n and n not in attributed:
                orphans_t[n] = orphans_t.get(n, 0.0) + float(k.get("us") or 0.0)
    # Every taxonomy-labeled orphan is kept — including the cublas /
    # vllm_kernel / sgl_kernel / torch families: under CUDA graphs and
    # torch.compile (framework-mode probes, 2026-09-24) the Python spans that
    # used to attribute GEMM and quant kernels vanish, so those kernels arrive
    # here, and path_diff grades gemm-class ops on exactly their names.
    # Only unlabeled kernels fall under the time-ranked cap.
    signal = {n for n in orphans_t if label_kernels([n])[0]}
    rest = sorted((n for n in orphans_t if n not in signal), key=lambda n: -orphans_t[n])
    return ops, sorted(signal | set(rest[:_ORPHAN_REST_CAP]))


def compress_error(stage: str, tb: str) -> dict:
    lines = tb.strip().splitlines()
    frames = [ln.strip()[:110] for ln in lines
              if ln.strip().startswith("File") and FW_FRAME.search(ln)][-5:]
    return {"stage": stage, "exc": lines[-1][:400], "frames": frames}


def _sidecar(rid: str) -> str | None:
    p = ROOT / "archive" / "raw" / f"{rid}.fp"
    return p.read_text().strip() if p.exists() else None


def build_records() -> None:
    # plan files oldest -> newest; the newest plan's run for an id wins, and a
    # run that carries an execution fingerprint always beats one that does not
    plan: dict = {}
    for pf in sorted((ROOT / "archive").glob("plan*.json"), key=lambda q: q.stat().st_mtime):
        for r in json.loads(pf.read_text()):
            if not isinstance(r, dict) or "skip" in r or not r.get("id"):
                continue
            cur = plan.get(r["id"])
            if cur is None or r.get("exec_fingerprint") or not cur.get("exec_fingerprint"):
                plan[r["id"]] = r
    out = ROOT / "archive" / "records.jsonl"
    n = 0
    stale_prefill = 0
    with out.open("w") as fh:
        for rid, run in plan.items():
            raw = ROOT / "archive" / "raw" / f"{rid}.json"
            if not raw.exists():
                continue
            f = json.loads(raw.read_text())
            # vision-tower sidecar (probes/vision_<backend>.py): the plan probe is
            # text-only, so VL checkpoints' encoder kernels come from this separate
            # evidence table, merged under the same profile_run phase vLLM's own
            # profile_run already uses. The sidecar's script hash is its provenance.
            vraw = ROOT / "archive" / "raw" / f"{rid}.vision.json"
            if vraw.exists():
                v = json.loads(vraw.read_text())
                if v.get("profile_run_kernels"):
                    tbl_home = f["trace"] if isinstance(f.get("trace"), dict) else f
                    tbl_home.setdefault("profile_run_kernels", v["profile_run_kernels"])
                    f["vision_evidence"] = v.get("vision_probe")
            ops, orphans = build_ops(f)
            sa = f.get("server_args_resolved") or f.get("engine_args_resolved") or {}
            keep = re.compile(
                r"^(kv_cache_dtype|page_size|block_size|quantization|attention_backend|"
                r"(prefill|decode)_attention_backend|dsa_(prefill|decode|topk|paged_mqa_logits)_backend|"
                r"moe_(runner|a2a)_backend|fp8_gemm_runner_backend|fp4_gemm_runner_backend|"
                r"bf16_gemm_backend|linear_attn_backend|mamba_backend|dtype|load_format|"
                r"tensor_parallel_size|max_model_len|context_length)$")
            sa = {k: v for k, v in sa.items() if keep.match(k)}
            rec = {
                "id": rid,
                "target": {k: run.get(k) for k in ("repo", "family", "variant", "profile",
                                                   "kvcache_quant_mode", "aic_registered")},
                "runtime": {**{k: run.get(k) for k in ("backend", "version", "image", "tp")},
                            # parallel dims are identity keys: ep sharding is
                            # kernel-invariant on the a2a=none path (measured,
                            # facts/tepdep/), but a2a=deepep changes the route
                            "ep": run.get("ep", 1), "dp": run.get("dp", 1),
                            "a2a": run.get("a2a", "none"),
                            "engine_cli": run.get("engine_cli"),
                            "unknown_args": f.get("engine_cli_unknown_args") or None,
                            "platform": run.get("platform", "h20_sm90"),
                            "sm_measured": f.get("device_capability"),
                            # kv-cache dtype is a first-class serving-config
                            # dimension: the collector sweeps fp8-KV, so
                            # path_diff must compare captures against the
                            # SAME-kv serving record (owner 2026-09-23).
                            "kv_cache_dtype": (
                                f.get("probe_kv_cache_dtype")
                                or sa.get("kv_cache_dtype")
                                or (f.get("kv_cache_resolved") or {}).get("attn_kv_cache_dtype")
                                or "auto"),
                            # prompt length and cache state decide WHICH prefill
                            # path serving took (DSA: query<=256 -> decode
                            # kernel; prefix-cache hit -> block residual), so a
                            # verdict must be able to select on them
                            "isl": f.get("probe_isl"),
                            "prefix_caching": f.get("probe_prefix_caching"),
                            # execution mode: probes default to the framework's
                            # own mode (torch.compile + CUDA graphs); eager is
                            # an A/B. None = pre-2026-09-24 probe (eager).
                            "probe_eager": f.get("probe_eager"),
                            "probe_cuda_graph": f.get("probe_cuda_graph"),
                            "evidence": "real"},
                "resolved": {k: v for k, v in sa.items() if v is not None},
                # generator-rendered flags that differ from the framework's own
                # parser defaults (probe-computed; owner decision 2026-09-20:
                # every delta is an auditable liability — the DSV4 0.29 crash
                # was triggered by one)
                "config_delta": f.get("config_delta") or None,
                "identity": {
                    "model_class": f.get("model_class"),
                    # sglang exposes the backend object; vllm/trtllm only reveal
                    # it through the wrapped attention spans — take either.
                    # sglang: backend object; vllm: wrapped attention spans;
                    # trtllm: no spans — fall back to the attention kernel family
                    "attn_backend": ((f.get("attn_backend") or "").rsplit(".", 1)[-1]
                                     or next((s.split("::")[2] for s in (f.get("api_trace") or {})
                                              if "::attn::" in s
                                              # older probes wrapped vllm's CustomOp base, so
                                              # activations/quant methods carry attn spans too
                                              and ATTN_CLASS_RE.search(s.split("::")[2])), None)
                                     # graph-replayed / compiled forwards emit no
                                     # Python spans (framework-mode probes,
                                     # 2026-09-24): the attention kernel family
                                     # in the device-stream tables is the identity
                                     or next((attn_identity_label(k["kernel"])
                                              for k in ((f.get("kernels") or [])
                                                        + (f.get("decode_kernels") or [])
                                                        + (f.get("prefill_kernels") or []))
                                              if re.search(r"fmha|flash_?attn|flash_fwd|"
                                                           r"mla_|attention_kernel|paged_kv|"
                                                           r"sparse_attn_fwd|mqa_logits|unified_attention|"
                                                           # vllm TRITON_MLA decode (the only MLA decode
                                                           # backend on sm120): _fwd_grouped_kernel_stage1
                                                           r"^_fwd_grouped_kernel_stage1",
                                                           k["kernel"], re.I)
                                              and "norm" not in k["kernel"].lower()
                                              # kv-cache insert glue matched `mla_` and posed as the
                                              # attention identity on sm120 TRITON_MLA records
                                              and "concat_and_cache" not in k["kernel"]), None)),
                    # hybrids (GDN / KDA / mamba + attention): the linear-attention
                    # kernel family is part of the identity too; recorded separately
                    # so the attention column can show "fa3 + chunk_gated_delta_rule"
                    "linear_attn_kernel": next((normalize_kernel(k["kernel"])
                                                for k in ((f.get("kernels") or [])
                                                          + (f.get("decode_kernels") or [])
                                                          + (f.get("prefill_kernels") or []))
                                                if LINEAR_ATTN_KERNEL_RE.search(k["kernel"])), None),
                    "modules": {k.rsplit(".", 1)[-1]: v.get("modules", v.get("examples", []))
                                for k, v in (f.get("quant_methods") or {}).items()},
                    "param_dtypes": f.get("param_dtypes"),
                    "weight_samples": f.get("weight_samples") or None,
                },
                # execution identity: the plan's fingerprint, the sidecar the queue
                # wrote next to this raw, and whether the evidence is current for
                # the plan (stale = the render/dummy/image changed since the probe)
                "exec_fingerprint": (run.get("exec_fingerprint") or {}).get("fingerprint"),
                "raw_fingerprint": _sidecar(rid),
                # did the golden render apply generator model facts (golden_facts_status)
                "golden_facts": run.get("golden_facts"),
                "evidence_status": evidence_status(run, _sidecar(rid), f),
                "ops": ops or None,
                "orphan_kernels": orphans or None,
                # phase of every kept orphan (prefill/decode/profile_run): the
                # comparison selects serving evidence by phase, never by name alone
                "orphan_phases": ({k: v for k, v in build_orphan_phases(f).items() if k in set(orphans)} or None),
                # phase of EVERY kernel the device tables saw (span-attributed
                # ones included): span attribution is by name across phases, so a
                # kernel both phases launch would otherwise vanish from the phase
                # its span did not cover (sglang decode FA3, 2026-09-25)
                "kernel_phases": (build_orphan_phases(f) or None),
                "outcome": ({"status": "ok"} if not f.get("errors") else
                            compress_error(*next(iter(f["errors"].items())))),
            }
            fh.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
            n += 1
            if (rec["runtime"].get("backend") == "vllm" and f.get("prefill_kernels")
                    and rec["runtime"].get("prefix_caching") is not False):
                stale_prefill += 1
    raw_bytes = sum(p.stat().st_size for p in (ROOT / "archive" / "raw").glob("*.json"))
    print(f"wrote {out}: {n} records, {out.stat().st_size // 1024}KB (raw evidence: {raw_bytes // 1024}KB)")
    if stale_prefill:
        # evidence-quality signal, printed on every rebuild so it cannot be
        # forgotten: these prefill tables predate the cache-cold probe fix and
        # hold a prefix-cache residual (query <= block_size), not an isl-token
        # prefill — identity of the prefill kernel may be wrong (DSA/MLA fp8)
        print(f"NOTE: {stale_prefill} vllm records still carry cached-residual prefill "
              f"evidence (runtime.prefix_caching != False); re-probe before drawing prefill conclusions")


# ---------------------------------------------------------------------------
# results matrix: THE consolidated output file — per (checkpoint x backend):
# can it boot, under what command, and what identity the framework actually
# deployed. Machine facts stay in archive/records.jsonl (per-run evidence);
# this is their consolidation.
def _fail_cause(note: str) -> str:
    rules = [
        ("not a valid Hugg", "generator rejects"),
        ("OutOfMemoryError|CUDA out of memory|insufficient GPU memory", "capacity (no faithful cut fits one probe GPU)"),
        ("Cannot find model module|not a registered|not supported for now|Unknown architecture|pydantic.*value_", "arch not registered"),
        ("NotImplementedError", "tied-embedding quant gap"),
        ("Only gated SiLU", "NVFP4 x gelu-MoE: no kernel path"),
        ("pre-blackwell|Arch unsupported|use Blackwell|TllmGenFmhaRunner|Minimum ca|COMPRESS pool|No supported MoE GEMM tactic|mxfp8 is not supported|NVFP4 quantization with the selected", "platform floor (needs Blackwell)"),
        # sm120 (consumer Blackwell, 101376 B opt-in smem/block): vllm 0.30 TRITON_MLA
        # fp8-KV decode asks for 102400 B -> every MLA checkpoint that resolves fp8 KV
        # (probe --kv-cache-dtype fp8, or NVFP4 artifacts whose hf_quant pins fp8 KV)
        # dies at CUDA-graph capture; sparse-MLA sm120 decode has an enumerated shape table
        # Ada (sm89) has the same 101376 B opt-in smem/block (L40 probe 2026-10-04: identical
        # Required 102400 / Hardware limit 101376 on the NVFP4 MLA cells), so the label names both.
        ("out of resource: shared memory", "platform limit (sm89/sm120 smem 101376 B: TRITON_MLA fp8-KV decode)"),
        ("SM120 sparse-MLA has no decode kernel", "platform gap (sm120 sparse-MLA decode shape table)"),
        # DSV4 hyperconnection kernels are DeepGEMM SM90 / SM10x only (L40 vllm 0.30.0 probe: 11
        # DeepSeek-V4 cells, `hyperconnection.hpp:56 Unsupported architecture` at model init)
        ("deepgemm.*Unsupported architecture", "platform floor (DeepGEMM needs SM90+)"),
        # sparse-attention / FP4 kernels with no Ada build (L40 sglang 0.5.21 + vllm probes)
        ("Sparse Attention Forward Kernel is only supported on|Unsupported architecture for sparse decode|requires Blackwell|"
         "requires Hopper or newer|Invalid backend: \\d+",
         "platform floor (SM90+/Blackwell-only kernel: sparse attention, MXFP, FP4, DeepSeek-V4)"),
        # the framework image, not the SM: rc29's transformers / python env lacks what the checkpoint or model needs
        ("update Transformers with the command|No module named '(fla|flash_linear_attention)'",
         "image gap (framework image lacks the transformers version / module this model needs)"),
        ("reduce max_num_tokens|shard the model weights across GPUs",
         "capacity (weights + engine buffers exceed one GPU)"),
        ("leave no GPU memory for the KV cache|exceeds available Mamba cache blocks",
         "capacity (weights + state leave no KV room on this GPU)"),
        # sglang's tp memory-balance check: another probe occupies a GPU of the group (the queue
        # puts tp>1 cells in ONE GPU's queue without reserving the group) — not a framework fact
        ("memory capacity is unbalanced", "harness: busy GPU in the tp group (rerun alone)"),
        # trtllm 1.3.0rc29 MoE resolution turned every candidate down for SM (L40: CutlassFusedMoE FP8_BLOCK_SCALES
        # SM90/SM120 only, TritonFusedMoE SM90 only, Marlin implements nvfp4/w4a16_nvfp4 only): a platform floor of the
        # framework, not a generator or model fact — except NVFP4, which MARLIN serves (targets.yaml sm89 customization)
        ("no MoE implementation can serve this layer",
         "platform floor (trtllm MoE: no implementation for this quant on this SM)"),
        ("requires an fp8 prefill query", "config gap (needs --attention-config use_prefill_query_quantization)"),
        # single-kind dummy cuts forced by capacity (72GB box): a cut with no attention
        # layer, or one that stripped every quantized layer, is a dummy artifact
        ("no attention-ish module classes|MIXED_PRECISION quant_algo requires a non-empty",
         "capacity (single-kind cut is not a faithful identity probe)"),
        ("Mismatched Tensor", "flake (flashinfer; env workaround exists)"),
        ("sparse forward|KVCacheManagerV2", "rc23 M3-sparse not wired"),
        # sparse (DSA) MLA has no backend below SM90 whatever the kv dtype: every GLM-5.x /
        # DeepSeek-V3.2 cell on L40 failed here with kv auto AND fp8, and the catch-all below
        # filed all 14 as "ckpt-forced fp8-KV" (a config cause that no flag can fix)
        ("No valid attention backend found.*use_sparse=True",
         "platform gap (no sparse-MLA attention backend on this SM)"),
        ("frame #|No valid attention backend", "ckpt-forced fp8-KV"),
        ("NoneType|QuantAlgo", "quant parser gap"),
    ]
    for pat, tag in rules:
        if re.search(pat, note):
            return tag
    return "framework gap"


# Deciding line is not always the LAST one (sglang wraps the cause in a "Capture cuda graph failed" /
# kill_process_tree epilogue): a second pass over the whole traceback, ONLY for causes the last-line rules
# left as "framework gap", and only with patterns specific enough not to fire on incidental earlier lines.
_DEEP_RULES = [
    ("memory capacity is unbalanced", "harness: busy GPU in the tp group (rerun alone)"),
    # sglang's launch route ends on a kill_process_tree epilogue; the OOM that killed the worker is earlier in the log
    ("torch.OutOfMemoryError: CUDA out of memory", "capacity (no faithful cut fits one probe GPU)"),
    # a JIT-compiled kernel whose ptx needs a newer target than the card (flashinfer cute-DSL RMSNormFP4Quant, rc29 L40)
    ("requires \\.target sm_\\d+ or higher", "platform floor (JIT kernel needs a newer SM target)"),
    # trtllm native aborts (no Python traceback, so only the console tail is known): attentionOp asserts
    ("Head size \\d+ is not supported by MMHA",
     "platform floor (trtllm MMHA has no kernel for this head size on this SM)"),
    ("Deepseek should be supported by fmha", "platform floor (trtllm MLA has no FMHA kernel on this SM)"),
    ("Unsupported architecture for sparse decode|Sparse Attention Forward Kernel is only supported on",
     "platform floor (SM90+/Blackwell-only kernel: sparse attention, MXFP, FP4)"),
    # DSV4 sparse decode: DeepGEMM attention.hpp asserts arch_major in {9, 10, 12} (L40 sglang 0.5.21)
    ("deepgemm/csrc/apis/[a-z_]+\\.hpp:\\d+\\): .*arch_major", "platform floor (DeepGEMM needs SM90+)"),
    # flashinfer 0.6.18's cutlass fused-MoE JIT does not compile for sm_89: tensorrt_llm::kernels::
    # cutlass_kernels has no Fp4Type there (nemotron_h bf16/fp8 MoE on sglang 0.5.21, L40) — a framework
    # JIT defect, not a model or config fact
    ("cutlass_kernels\" has no member \"Fp4Type\"|has no member .Fp4Type",
     "framework defect (flashinfer cutlass fused-MoE JIT fails to build for sm89: Fp4Type)"),
]


def _fail_cause_full(last_line: str, traceback_text: str) -> str:
    cause = _fail_cause(last_line)
    if cause == "framework gap":
        for pat, tag in _DEEP_RULES:
            if re.search(pat, traceback_text):
                return tag
    return cause


def build_matrix(targets: dict) -> None:
    # resolve each backend's plan by the PINNED version (hardcoded names went
    # stale on the first version bump): pick the plan file whose runs carry
    # (backend, pinned version); ties break to the most recently written
    pins = {be: (cfg.get("versions") or ["?"])[0] for be, cfg in targets["backends"].items()}
    plans: dict = {}
    for pf in sorted((ROOT / "archive").glob("plan*.json"), key=lambda q: q.stat().st_mtime):
        try:
            runs = json.loads(pf.read_text())
        except Exception:
            continue
        for be, ver in pins.items():
            if any(isinstance(r, dict) and r.get("backend") == be and r.get("version") == ver
                   and "skip" not in r for r in runs):
                # EVERY plan file that carries this (backend, pin) contributes:
                # the archive accumulates plan files (roster, re-plans, kv
                # variants, onboarding subsets) and a matrix built from the
                # newest one alone silently dropped every other model
                # (found 2026-09-24: an 18-run GLM-5.3 plan produced an
                # 18-cell matrix stamped with another backend's version)
                plans.setdefault(be, []).append(pf.name)
    missing = sorted(set(pins) - set(plans))
    if missing and not plans:
        raise SystemExit(f"--matrix: no plan file matches the pinned version for {missing} — emit queues first")
    if missing:
        # a single-backend campaign (an sm120 box probing only vllm) is a legal
        # workspace: backends without a plan get no matrix file, not an abort
        # (found 2026-09-30 on the first non-sm90 campaign; the H20 workspace
        # always carried all three backends so the abort never fired)
        print(f"--matrix: no plan for {missing} at the pinned version — skipped (matrix only for {sorted(plans)})")
    # pass+custom means PER-CHECKPOINT facts-derived args (backend-level
    # generator-sets like --benchmark-mode apply to every run and are not
    # a customization of this model)
    custom: dict = {}  # entries scoped with `sms:` count only on this SM (_cea)
    for fam in targets["families"].values():
        for ck in fam.get("checkpoints") or []:
            for be2, v2 in (ck.get("cli_extra_args") or {}).items():
                if _cea(v2):
                    custom[(ck["repo"], be2)] = " ".join(_cea(v2))
        for repo2, o2 in (fam.get("checkpoint_overrides") or {}).items():
            for be2, v2 in ((o2 or {}).get("cli_extra_args") or {}).items():
                if _cea(v2):
                    custom[(repo2, be2)] = " ".join(_cea(v2))
    # records by id: a cell reads the record of ITS run, never a sibling's
    # (review 2026-09-25: (repo, backend) indexing let an older version's
    # record lend its identity to the new version's matrix)
    recs_by_id: dict = {}
    for line in (ROOT / "archive" / "records.jsonl").open():
        r = json.loads(line)
        recs_by_id[r["id"]] = r
    kvcap = {}
    for p in (ROOT / "facts" / "kvcap").glob("*.json"):
        kvcap[p.stem] = (json.loads(p.read_text()) or {}).get("kv_cache_resolved") or {}
    short = {"torch.bfloat16": "bf16", "torch.float16": "fp16",
             "torch.float8_e4m3fn": "fp8_e4m3", "torch.uint8": "fp8(u8)"}
    out: dict = {}
    counts: dict = {}
    versions: dict = {}  # measured: the version the plan actually ran

    def _matrix_runs(be, pfs):
        """Runs that define a matrix cell: this backend at its pin (a mixed
        plan file carries all three backends) and the RENDERED config only —
        kv-dtype variant runs are path_diff evidence, not the deployed
        identity of the model x backend cell. Ids dedup across plan files."""
        # a RENDERED run wins over a 'skip' of the same id from an older plan
        # file (a checkpoint the generator rejected before it was bundled keeps
        # its stale skip in the old plan; the re-emitted plan renders it)
        # the CURRENT plan decides a cell: plan files are visited oldest ->
        # newest and a later rendered run for the same (repo, backend, pin)
        # replaces an earlier one (a stale skip never replaces a rendered run)
        by_cell: dict = {}
        for pf in pfs:  # pfs sorted by mtime, oldest first
            for r in json.loads((ROOT / "archive" / pf).read_text()):
                if (isinstance(r, dict) and r.get("backend") == be and r.get("version") == pins[be]
                        and not r.get("kv_dtype") and r.get("id")):
                    cur = by_cell.get(r["repo"])
                    if cur is None or "skip" not in r or "skip" in cur:
                        by_cell[r["repo"]] = r
        yield from by_cell.values()
    for be, pfs in plans.items():
        for run in _matrix_runs(be, pfs):
            repo = run.get("repo")
            versions[be] = run["version"]
            cell: dict = {}
            raw = ROOT / "archive" / "raw" / f"{run.get('id','')}.json"
            if "skip" in run:
                _skip_cause = ("dummy not built" if run["skip"].startswith(("dummy ", "no dummy"))
                               else "generator rejects")
                cell = {"verdict": "fail", "cause": _skip_cause, "error": run["skip"][:160]}
            elif not raw.exists():
                cell = {"verdict": "fail", "cause": "no raw (crashed before dump)"}
                tail = ROOT / "archive" / "logs" / f"{run.get('id','')}.tail"
                if tail.exists():
                    txt = tail.read_text(errors="replace")
                    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                    # the deciding line: a native assert (`what():` / `Assertion failed`) beats a Python error line,
                    # and the LAST line of a class beats earlier ones (framework warnings mention "error:" too)
                    deciding = next((ln for pat in (r"what\(\):", r"Assertion failed", r"(?:Error|Exception)\b[^\n]*:", r"error:")
                                     for ln in reversed(lines) if re.search(pat, ln)),
                                    lines[-1] if lines else "")
                    cause = _fail_cause_full(deciding, txt)
                    if deciding:
                        cell = {"verdict": "fail",
                                "cause": cause if cause != "framework gap" else "no raw (crashed before dump)",
                                "error": deciding[:200]}
            elif evidence_status(run, _sidecar(run["id"]), json.loads(raw.read_text())) == "stale":
                cell = {"verdict": "fail", "cause": "stale evidence",
                        "error": "raw probed under an earlier render/dummy/image — re-emit queues and re-probe"}
            else:
                f = json.loads(raw.read_text())
                err = f.get("errors") or {}
                if err:
                    tb = next(iter(err.values())).strip()
                    full = tb.splitlines()[-1]
                    # classified on the FULL line: selector errors carry the deciding flag (use_sparse) late
                    note = full[:200]
                    cell = {"verdict": "fail", "cause": _fail_cause_full(full, tb), "error": note}
                else:
                    ca = custom.get((repo, be))
                    cell = {"verdict": "pass+custom" if ca else "pass"}
                    if ca:
                        cell["extra_args"] = ca
                    rec = recs_by_id.get(run["id"])
                    cell["evidence"] = evidence_status(run, _sidecar(run["id"]), f)
                    if rec and (rec.get("outcome") or {}).get("status") == "ok":
                        ident = rec.get("identity") or {}
                        res = rec.get("resolved") or {}
                        cell["attention"] = " + ".join(x for x in (ident.get("attn_backend"), ident.get("linear_attn_kernel")) if x) or None
                        moe_q = next((k for k in (ident.get("modules") or {}) if "MoE" in k), None)
                        moe_b = set()
                        _moe_rx = re.compile(r"moe|Marlin|marlin|grouped|expert")
                        for op in rec.get("ops") or []:
                            if _moe_rx.search(" ".join(op.get("kernels") or [])):
                                moe_b |= set(op.get("backends") or [])
                        # framework-mode records (CUDA graphs / compile) carry
                        # the MoE kernels as orphans, not under spans
                        moe_orph = [k for k in rec.get("orphan_kernels") or [] if _moe_rx.search(k)]
                        if moe_orph:
                            moe_b |= label_kernels(moe_orph)[0]
                        moe_b -= {"cublas", "vllm_kernel", "sgl_kernel", "torch"}
                        if moe_q:
                            cell["moe"] = (moe_q.replace("Method", "")
                                           + ("->" + "/".join(sorted(moe_b)) if moe_b else ""))
                        kv = kvcap.get(rec["id"], {})
                        act = (kv.get("runner_kv_cache_dtype") or kv.get("pool.dtype")
                               or next((v for k2, v in kv.items()
                                        if k2.startswith("manager.") and "dtype" in k2), None)
                               or res.get("kv_cache_dtype"))
                        act = short.get(str(act), str(act).replace("DataType.", "").lower()) if act else None
                        cell["kv_allocated"] = act
                        cell["topology"] = f"tp{run.get('tp',1)}"
            # several rendered runs can exist for one (repo, backend) across
            # plan files (re-plans, id-formula changes): a run that passed
            # is the cell; a later run with no raw or a crash never
            # overwrites it
            _gf = run.get("golden_facts")
            if isinstance(_gf, dict) and _gf.get("applied") is False:
                # the engine args this cell was probed with lack every model default
                cell["golden_facts"] = "not_applied: " + str(_gf.get("reason", ""))[:120]
            prev = out.get(repo, {}).get(be)
            if prev and prev["verdict"] != "fail" and cell["verdict"] == "fail":
                continue
            out.setdefault(repo, {})[be] = cell
    for repo, cells in out.items():
        for be, cell in cells.items():
            c = counts.setdefault(be, {"pass": 0, "pass+custom": 0, "fail": 0})
            c[cell["verdict"]] += 1
    # one file per (SM, framework); version pinned inside and OVERWRITTEN on
    # bumps — git diff of a re-run IS the upgrade audit. Future SMs are
    # sibling dirs (results/sm100/...).
    sm = (targets.get("platform") or {}).get("name", "sm90").split("_")[-1]
    # results/ is the HARNESS's committed output (README layout; workflow_check
    # reads results/<sm>/ from the checkout) — not workspace data. Writing under
    # ROOT only coincided with the checkout when the two were the same dir
    # (found 2026-09-30: an external workspace produced a matrix workflow_check
    # could not see).
    outdir = _HERE.parent / "results" / sm
    outdir.mkdir(parents=True, exist_ok=True)
    for be in plans:
        rows = {repo: cells[be] for repo, cells in sorted(out.items()) if be in cells}
        doc = {"_meta": {"platform": (targets.get("platform") or {}).get("name"),
                         "framework": be, "version": versions.get(be),
                         "verdicts": "pass = plain `cli generate` output boots+runs; "
                                     "pass+custom = boots with facts-derived extra generate args; "
                                     "fail = root-caused (see results/findings.yaml)",
                         "summary": counts.get(be),
                         # cells whose golden render lost the generator model facts (see golden_facts_status)
                         "golden_facts_not_applied": sum(1 for c in rows.values() if c.get("golden_facts"))},
               "results": rows}
        p = outdir / f"{be}-{versions.get(be)}.yaml"
        p.write_text(yaml.safe_dump(doc, width=200, sort_keys=False, allow_unicode=True))
        print(f"wrote {p} ({p.stat().st_size//1024} KB) {counts.get(be)}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, default=Path(__file__).resolve().parents[1] / "targets.yaml")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--emit-queues", action="store_true")
    ap.add_argument("--check-coverage", action="store_true",
                    help="every model repo the collector's case yamls mention must be a target")
    ap.add_argument("--full", action="store_true", help="all variants x all pinned versions (default: representative)")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--gpu-list", default=None,
                    help="comma list of GPU indices (overrides --gpus/--gpu-offset; for boxes with busy GPUs)")
    ap.add_argument("--backends", default="sglang", help="comma list: sglang,vllm")
    ap.add_argument("--only", default=None,
                    help="comma list of repo substrings — plan/queues cover only matching checkpoints")
    ap.add_argument("--plan-name", default="plan.json")
    ap.add_argument("--records", action="store_true", help="raw probe JSONs -> archive/records.jsonl")
    ap.add_argument("--migrate-run-ids", action="store_true",
                    help="rename pre-2026-09-30 artifacts (raw/.fp/run_sh/golden, plan/records/archive ids) to platform-bearing run ids; add --apply to write")
    ap.add_argument("--apply", action="store_true", help="with --migrate-run-ids: perform the renames (default dry-run)")
    ap.add_argument("--matrix", action="store_true", help="consolidated results: matrix.yaml (verdict + deployed identity per model x backend)")
    args = ap.parse_args()

    if args.migrate_run_ids:
        plat = (yaml.safe_load(args.targets.read_text()).get("platform") or {}).get("name") or "h20_sm90"
        st = migrate_run_ids(ROOT, plat, apply=args.apply)
        print(json.dumps({"platform": plat, **st}))
        if not args.apply:
            print("dry run — add --apply to rename")
        return
    if args.records:
        build_records()
        return
    if args.matrix:
        build_matrix(yaml.safe_load(args.targets.read_text()))
        return
    if args.check_coverage:
        check_coverage(yaml.safe_load(args.targets.read_text()))
        return
    targets = yaml.safe_load(args.targets.read_text())
    runs = enumerate_runs(targets, args.full, args.backends.split(","))
    if args.only:
        pats = [p.strip() for p in args.only.split(",") if p.strip()]
        runs = [r for r in runs if any(p in (r.get("repo") or "") for p in pats)]
        print(f"--only {args.only}: {len(runs)} runs")
    if args.plan:
        for r in runs:
            print(json.dumps(r))
        return
    if args.emit_queues:
        gpu_list = ([int(x) for x in args.gpu_list.split(",")] if args.gpu_list
                    else list(range(args.gpu_offset, args.gpu_offset + args.gpus)))
        emit_queues(runs, gpu_list, args.plan_name)


if __name__ == "__main__":
    main()
