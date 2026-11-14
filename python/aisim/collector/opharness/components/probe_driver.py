#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving probes: given a generator-rendered config, what does the framework deploy?

A campaign is one system. targets.yaml (the roster overlay) -> golden `cli generate`
renders for that system -> one ``probe.serving`` shard per run for the collection
pipeline -> the job's raws -> records. A campaign lives in
``<workspace>/campaigns/<name>/`` on the control machine and is never committed:

  --plan --system S                     print the runs (--backends, --only, --full, --isl, --version)
  --gates [--op OP]                     the runs instead: what the path gates in scope read (repo, kv,
                                        isl, release), every backend unless --backends names some
  --emit-shards --system S --url-base U render, bundle and upload every run; shard plans in
                                        shards/tp<N>/<backend>-<version>.yaml, one job each
  --import-raws DIR                     file a job result's raws into raw/ (the job must have run this emit)
  --records                             raw/ -> records.jsonl (kernels labeled with the SM each raw ran on),
                                        and one line per run: ok, or why it has no serving evidence

Re-emitting a campaign starts it over (its raws and records are removed).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
HARNESS = _HERE.parent
# The workspace holds everything generated: configs/, dummy_models/, campaigns/, results/.
ROOT = Path(os.environ.get("AIS_PROBE_WORKSPACE") or HARNESS / "workspace")
# generator source: this repo by default; override to pin a specific checkout
AIS_SRC = os.environ.get("AIS_GENERATOR_SRC") or str(HARNESS.parents[1] / "src")
SYSTEMS = HARNESS.parents[1] / "src" / "aisim_core" / "systems"
# A probe's argv names its inputs under BUNDLE (the unpacked bundle in the job) and its raw
# under OUT (the shard directory); collector/probes/serving.py substitutes both.
WORK = "{BUNDLE}"
OUT = "{OUT}"

GOLDEN_TARGET = {"sglang": "dynamo-python", "vllm": "fpm", "trtllm": "dynamo-python"}
# The golden renderer is THIS repository's generator, run through its console script from an
# environment that has the native runtime built (uv sync). AIS_GENERATOR_CLI overrides it.
GEN_CLI = Path(os.environ.get("AIS_GENERATOR_CLI") or ROOT / "venv_ais" / "bin" / "aiconfigurator")


def campaign_dir(name: str) -> Path:
    return ROOT / "campaigns" / name


@functools.cache
def system_sm(system: str) -> str:
    """``b200_sxm`` -> ``sm100``, from the system spec the generator renders for."""
    spec = yaml.safe_load((SYSTEMS / f"{system}.yaml").read_text())
    return f"sm{int(spec['gpu']['sm_version'])}"


_FACTS_FAIL_RX = re.compile(r"Fact resolution failed|Unknown hardware profile")


def golden_facts_status(render_output: str) -> dict:
    """Did the golden render apply the generator's MODEL facts (facts/models.yaml
    defaults: block-size, trust-remote-code, autotune flags ...)?

    generator/pipeline.run_pipeline swallows a facts-resolution failure ("Fact
    resolution failed; continuing without facts.") and renders with facts=None,
    so the engine args silently lack every model default. Found 2026-10-01: all
    RTX PRO renders (no hardware profile for rtx_pro_6000_server) lost them.
    The status is recorded on the run and in its record so a
    render without facts is never mistaken for a golden one."""
    if _FACTS_FAIL_RX.search(render_output or ""):
        m = re.search(r"KeyError: [^\n]*", render_output) or re.search(r"Fact resolution failed[^\n]*", render_output)
        return {"applied": False, "reason": (m.group(0) if m else "fact resolution failed").strip()[:200]}
    return {"applied": True}


def _sha(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


# one render per distinct command within an emit (the kv variants of a run share it)
_RENDERED: dict[str, tuple[Path | None, dict, str | None]] = {}


def render_golden(run: dict, cdir: Path) -> Path | None:
    """Invoke the REAL user-facing generator command for the run's system and keep its
    artifacts untouched under ``<campaign>/golden/``; every probe-side adaptation happens
    later as a recorded post-process. Per-checkpoint extra args (targets.yaml
    cli_extra_args) are spliced into the command."""
    cmd = [
        str(GEN_CLI),
        "cli",
        "generate",
        "--model-path",
        run["repo"],
        "--total-gpus",
        str(run["tp"]),
        "--system",
        run["system"],
        "--backend",
        run["backend"],
        "--deployment-target",
        GOLDEN_TARGET[run["backend"]],
        "--config-template-version",
        run["version"],
        *(run.get("cli_extra_args") or []),
    ]
    key = _sha(*cmd)
    if key not in _RENDERED:
        gdir = cdir / "golden" / key
        shutil.rmtree(gdir, ignore_errors=True)
        gdir.mkdir(parents=True)
        env = {**os.environ, "PYTHONPATH": AIS_SRC}
        # cwd = the WORKSPACE: aic-core's safe_mkdir accepts paths under cwd / $HOME / tmp only, so a
        # save dir on another mount rendered every run as a silent 'generator rejects' (2026-09-20).
        r = subprocess.run(
            [*cmd, "--save-dir", str(gdir)], capture_output=True, text=True, env=env, timeout=900, cwd=str(ROOT)
        )
        output = (r.stdout or "") + (r.stderr or "")
        (gdir / "command.txt").write_text(shlex.join(cmd) + f"\n# exit={r.returncode}\n")
        (gdir / "render.log").write_text(output[-16000:])
        art = next((d for d in gdir.iterdir() if d.is_dir()), None) if r.returncode == 0 else None
        error = None if art else ((output.strip().splitlines() or ["no output"])[-1][:200])
        _RENDERED[key] = (art, golden_facts_status(output), error)
    art, facts, error = _RENDERED[key]
    run["golden_facts"] = facts
    if art is None:
        run["golden_error"] = error
    return art


def extract_sglang_cli_from_run_sh(run_sh: Path) -> str:
    """Post-process: lift the dynamo.sglang engine args out of golden run_0.sh.
    Drops only wrapper plumbing ($MODEL_PATH placeholder, host/metrics/shell);
    engine-selection flags pass through verbatim."""
    text = run_sh.read_text()
    m = re.search(r"python3 -m dynamo\.sglang((?:[^\n]*\\\n)*[^\n]*)", text)
    if not m:
        raise SystemExit(f"golden run_0.sh has no dynamo.sglang block: {run_sh}")
    block = m.group(1).replace("\\\n", " ")
    block = re.split(r"\s(?:2>&1|\||&|;|\))", block)[0]
    toks = shlex.split(block)
    out, skip = [], False
    drop = {"--model-path", "--served-model-name", "--host", "--port"}
    flag_drop = {"--enable-metrics"}
    for tk in toks:
        if skip:
            skip = False
            continue
        if tk in drop:
            skip = True
            continue
        if tk in flag_drop:
            continue
        out.append(tk)
    return " ".join(out)


def _cea(v, sm: str):
    """cli_extra_args entry: plain list, or {args: [...], fact: "<evidence>",
    sms: [sm120, ...]} — ``sms`` (optional) limits the customization to those
    SMs (an entry keyed (repo, fw) used to ride along to every SM; sm120 needs
    bf16 KV for NVFP4 MLA checkpoints, which sm90/sm100 must not inherit). A list of
    such dicts applies each entry on its own SMs (DeepSeek-V4's page size holds on every
    SM, its fp8_ds_mla KV only on Hopper)."""
    if not v:
        return []
    if isinstance(v, dict):
        if v.get("sms") and sm not in [str(x) for x in v["sms"]]:
            return []
        return list(v["args"])
    if all(isinstance(x, dict) for x in v):
        return [a for x in v for a in _cea(x, sm)]
    return list(v)


_REPO_KEYS = frozenset({"model_path", "model_paths", "model_aliases", "supported_model_paths", "default_model_paths"})


def collector_mentioned_repos() -> set[str]:
    """Every checkpoint the collector's case yamls declare (the coverage floor): the values of
    their model-path keys. A text scan with a fixed list of orgs missed four declared models
    (RadixArk, meta-models, stepfun-ai) and took one from a comment that says it does not exist."""
    found: set[str] = set()

    def walk(node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for x in node:
                walk(x, key)
        elif key in _REPO_KEYS and isinstance(node, str) and "/" in node:
            found.add(node)

    for f in sorted((Path(AIS_SRC).parent / "collector" / "cases" / "models").glob("*_cases.yaml")):
        walk(yaml.safe_load(f.read_text()))
    return found


def derived_repos(fam: dict, targets: dict) -> list[str]:
    """The repos of a derived family: every org/repo the collector's case yamls
    mention, minus owner exclusions and repos owned by other (special-adapter)
    families, plus probe-only extra_repos."""
    mentioned = collector_mentioned_repos()
    # owner-decided exclusions live in targets.yaml roster.excluded (each entry
    # carries decided_by/reason).
    excluded = {e["repo"] for e in (fam.get("excluded") or [])}
    owned_elsewhere = {
        ck["repo"] for f in targets["families"].values() if not f.get("derive") for ck in f.get("checkpoints", [])
    }
    return sorted((mentioned - excluded - owned_elsewhere) | set(fam.get("extra_repos") or []))


def derive_roster_checkpoints(fam: dict, targets: dict, only: list[str] | None = None) -> list[dict]:
    """Roster checkpoints DERIVED from the collector's own case declarations
    (derived_repos). Profile comes from the checkpoint's quant metadata;
    variants from the dummy manifest. targets.yaml holds only the overlay
    (checkpoint_overrides)."""
    repos = [r for r in derived_repos(fam, targets) if not only or any(p in r for p in only)]
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
        if repo not in variants_of:  # no dummy yet: never probed here, enumerate_runs reports it
            out.append({"repo": repo, "profile": None, "variants": []})
            continue
        profile = derive_profile(repo, ROOT / "configs")
        if profile == "MISSING":
            raise SystemExit(f"derive_roster: {repo} has dummies but no fetched config — fetch_inputs.py {repo}")
        ck = {"repo": repo, "profile": profile, "variants": variants_of[repo]}
        ov = dict(overrides.get(repo) or {})
        if "profile" in ov:  # explicit pin wins, but derivation drift is loud
            if ov["profile"] != profile:
                print(f"derive_roster: {repo} profile pinned {ov['profile']} != derived {profile}")
            ck["profile"] = ov.pop("profile")
        ck.update(ov)
        out.append(ck)
    return out


DEFAULT_ISL = 4096  # the probes' prompt length unless a campaign asks for another


def _run_id(
    ck: dict, variant: str, backend: str, version: str, tp: int, kv, system: str, isl: int = DEFAULT_ISL
) -> str:
    """Run id = sha1 of the case identity, the system and prompt length included (the same
    case on two systems, or at two lengths, is two runs)."""
    key = f"{ck['repo']}|{variant}|{backend}|{version}|{ck['profile']}|tp{tp}|kv{kv or 'rendered'}|{system}"
    if isl != DEFAULT_ISL:
        key += f"|isl{isl}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ------------------------------------------------------------ execution identity
# The run id names the CASE. The fingerprint names what one emit hands the job: the image,
# the probe script and its arguments, and every input file the bundle carries (the dummy
# checkpoint and the rendered engine config). The spec carries it into the job, and
# --import-raws files a raw only when the job ran this emit's fingerprint.
def probe_script(run: dict) -> str:
    """tp1 runs the in-process probe; sglang and trtllm tp>1 run the real server."""
    server = run["backend"] in ("sglang", "trtllm") and int(run.get("tp", 1)) > 1
    return f"probe_{run['backend']}{'_server' if server else ''}.py"


def exec_fingerprint(run: dict, bundle_files: list[str]) -> str:
    """``bundle_files`` are the bundle's input paths relative to the workspace."""
    h = hashlib.sha1()
    script = _HERE / "probes" / run["probe_script"]
    spec = [run.get("image"), run["probe_script"], run["probe_args"], run.get("probe_env")]
    h.update(json.dumps(spec, sort_keys=True).encode())
    h.update(script.read_bytes() if script.exists() else b"")
    for rel in sorted(bundle_files):
        root = ROOT / rel
        for f in sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file()):
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def select_dummy_dir(
    dummy_root: Path, repo_tag: str, variant: str, fam_name: str, fam: dict, ck: dict
) -> tuple[Path | None, str | None]:
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
            return None, (
                f"dummy {leaf} was built by adapter '{other}' but targets declare "
                f"dummy_overrides.family '{want}' — rebuild with components/dummies.py"
            )
        return None, f"no dummy dir {want}/{leaf} (dummy_overrides.family '{want}'; run components/dummies.py)"
    order = [fam.get("dummy_dir") or fam_name, "generic"] + sorted(d.name for d in dummy_root.iterdir() if d.is_dir())
    for famdir in dict.fromkeys(order):
        vdir = dummy_root / famdir / leaf
        if vdir.exists():
            return vdir, None
    return None, f"no dummy dir {leaf}"


def enumerate_runs(
    targets: dict,
    system: str,
    full: bool,
    backends: list[str],
    only: list[str] | None = None,
    isl: int = DEFAULT_ISL,
    version: str | None = None,
) -> list[dict]:
    """The runs of a campaign for one system. ``isl`` is the probes' prompt length: shape-
    conditional paths (DSA prefill below 256 query tokens) are visible only at their length.
    ``version`` is the release probed (default: the framework default; ``full``: every pin)."""
    sm = system_sm(system)
    topos = [t for t in targets["topologies"] if t["evidence"] == "real" and (full or t["tp"] == 1)]
    # topology_policy.tp_required: checkpoints whose representative probe must run at a
    # declared tp (the dummy does not fit one GPU, e.g. Qwen3.8-2.4T), replacing the tp1
    # topology for that (repo, backend). An entry may carry `sms: [100]`: the tp applies on
    # those SMs only (the Kimi-K3 96-head trtllm-gen decode gap exists at tp1 on sm100 and not
    # on sm90; the probe must see the per-rank head count serving would use).
    tp_required = {
        (e["repo"], e.get("backend", "sglang")): int(e["tp"])
        for e in (targets.get("topology_policy") or {}).get("tp_required") or []
        if not e.get("sms") or sm in [f"sm{int(x)}" for x in e["sms"]]
    }

    def _topos_for(repo: str, backend: str) -> list[dict]:
        req = tp_required.get((repo, backend))
        if req is None:
            return topos
        picked = [
            t for t in targets["topologies"] if t["evidence"] == "real" and t["tp"] == req and t.get("ep", 1) == 1
        ]
        if not picked:
            raise ValueError(
                f"topology_policy.tp_required asks tp{req} for {repo}/{backend} "
                "but targets.topologies has no such real topology"
            )
        return picked

    runs = []
    for backend in backends:
        be = targets["backends"][backend]
        versions = be["versions"] if full else [version or be["default_version"]]
        if any(v not in be["images"] for v in versions):
            raise ValueError(f"{backend} pins no release {versions}; the manifest pins {be['versions']}")
        for fam_name, fam in targets["families"].items():
            if fam.get("derive") and "checkpoints" not in fam:
                fam["checkpoints"] = derive_roster_checkpoints(fam, targets, only)
            variants = fam.get("dummy_variants") or []
            if not variants and not any(c.get("variants") for c in fam["checkpoints"]):
                continue  # adapter pending (kimi_k3)
            override = (fam.get("variant_overrides") or {}).get(backend)
            for ck in fam["checkpoints"]:
                repo_tag = ck["repo"].split("/")[-1]
                # per-checkpoint variants win (architectures in a mixed family each have
                # their own layer kinds); else the family list
                ck_variants = ck.get("variants") or variants
                if not ck_variants:
                    print(
                        f"enumerate_runs: {ck['repo']} has no dummy variants — skipped (run components/dummies.py)",
                        file=sys.stderr,
                    )
                    continue
                ck_override = (ck.get("variant_overrides") or {}).get(backend) or override
                use_variants = ck_variants if full else [ck_override or ck_variants[0]]
                for variant in use_variants:
                    vdir, why = select_dummy_dir(ROOT / "dummy_models", repo_tag, variant, fam_name, fam, ck)
                    if vdir is None:
                        runs.append({"skip": why, "repo": ck["repo"], "variant": variant, "backend": backend})
                        continue
                    for version in versions:
                        for topo in _topos_for(ck["repo"], backend):
                            # kv-cache dtype is a first-class serving-config axis the collector
                            # sweeps; the golden sglang CLI for fp8 profiles renders no kv dtype,
                            # so without this variant the fp8-KV DSA/MLA paths were never probed
                            kv_variants = {"vllm": [None, "fp8"], "sglang": [None, "fp8_e4m3"], "trtllm": [None, "fp8"]}
                            for kv in kv_variants.get(backend, [None]):
                                runs.append(
                                    {
                                        "id": _run_id(ck, variant, backend, version, topo["tp"], kv, system, isl),
                                        "family": fam_name,
                                        "repo": ck["repo"],
                                        "profile": ck["profile"],
                                        "system": system,
                                        "sm": sm,
                                        "variant": variant,
                                        "backend": backend,
                                        "version": version,
                                        "image": be["images"][version],
                                        "tp": topo["tp"],
                                        "kv_dtype": kv,
                                        "isl": isl,
                                        "model_dir": f"{WORK}/{vdir.relative_to(ROOT)}",
                                        "aic_registered": ck.get("aic_registered", False),
                                        "cli_extra_args": (
                                            list(be.get("cli_extra_args") or [])
                                            + _cea(
                                                (ck.get("cli_extra_args") or {}).get(backend)
                                                or (fam.get("cli_extra_args") or {}).get(backend),
                                                sm,
                                            )
                                        ),
                                    }
                                )
    return runs


def kv_class(kv) -> str:
    """auto resolves to the model dtype (bf16 for these models): the unquantized aliases are
    one class, fp8 (fp8, fp8_e4m3) its own."""
    return "fp8" if (kv or "").startswith("fp8") else "unquant"


def gate_runs(
    targets: dict, system: str, backends: list[str], op: str | None = None, derive: bool = False
) -> list[dict]:
    """The serving runs the path gates in scope read on this system's SM: per gate, its repo
    at its kv class, prompt length (``isl``) and release (gates.gate_version). A gate with no
    roster run is reported; a ``no_serving`` floor needs none. ``derive`` adds the gates
    derived for the paths no row names (gates.paths)."""
    import copy

    sys.path.insert(0, str(_HERE))
    import gates

    sm = system_sm(system)
    need: dict[tuple[str, str, int], dict[tuple[str, str], list[str]]] = {}
    for g in gates.gates_in_scope(",".join(backends), sm, op=op, derive=derive):
        if sm in (g.get("no_serving") or {}):
            continue
        key = (g["backend"], gates.gate_version(g), int(g.get("isl", DEFAULT_ISL)))
        need.setdefault(key, {}).setdefault((g["repo"], kv_class(g["kv"])), []).append(g["gate"])
    runs: dict[str, dict] = {}
    for (backend, version, isl), wanted in sorted(need.items()):
        repos = sorted({repo for repo, _ in wanted})
        got = [
            r
            for r in enumerate_runs(copy.deepcopy(targets), system, False, [backend], repos, isl, version)
            if "skip" not in r and (r["repo"], kv_class(r["kv_dtype"])) in wanted
        ]
        runs.update((r["id"], r) for r in got)
        for (repo, kv), names in sorted(wanted.items()):
            if not any((r["repo"], kv_class(r["kv_dtype"])) == (repo, kv) for r in got):
                print(f"gate_runs: no {backend} {version} run for {repo} kv={kv} ({', '.join(names)})", file=sys.stderr)
    return list(runs.values())


def _generator_src_commit() -> dict:
    """Record WHICH generator code rendered the engine args."""
    repo = str(Path(AIS_SRC).parent)
    try:
        rev = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--", AIS_SRC], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        rev, dirty = "unknown", ""
    return {"generator_src": repo, "generator_commit": rev + ("+dirty" if dirty else "")}


def _prepare_probe(run: dict, cdir: Path) -> tuple[str, list[str], dict] | None:
    """Render the run's golden config and return (probe script, argv, extra env) with
    inputs under WORK and the raw under OUT; None (with run["skip"] set) when the run
    cannot be probed."""
    raw = f"{OUT}/raw_{run['id']}.json"
    art = render_golden(run, cdir)
    if art is None:
        run["skip"] = f"golden render failed: {run.get('golden_error')}"
        return None
    run["golden_dir"] = str(art)
    probe = _probe_argv(run, art, cdir, raw)
    if probe is not None:
        probe[1].extend(["--isl", str(run.get("isl", DEFAULT_ISL))])
    return probe


def _probe_argv(run: dict, art: Path, cdir: Path, raw: str) -> tuple[str, list[str], dict] | None:
    tp = int(run.get("tp", 1))
    script = probe_script(run)
    if run["backend"] == "sglang":
        cli = extract_sglang_cli_from_run_sh(art / "run_0.sh")
        run["engine_cli"] = cli
        kv = ["--kv-dtype", run["kv_dtype"]] if run.get("kv_dtype") else []
        if tp > 1:
            return script, ["--model", run["model_dir"], "--engine-cli", cli, "--tp", str(tp), *kv, "--out", raw], {}
        return script, ["--model", run["model_dir"], "--engine-cli", cli, "--trace", *kv, "--out", raw], {}
    if run["backend"] == "vllm":  # golden fpm run.sh, consumed verbatim
        src = next((p for p in (art / "run.sh", art / "run_0.sh") if p.exists()), None)
        if src is None:
            run["skip"] = f"golden has no run.sh: {art}"
            return None
        rsh = cdir / "inputs" / f"{run['id']}.sh"
        rsh.write_text(src.read_text())
        run["run_sh"] = str(rsh)
        kv = ["--kv-cache-dtype", run["kv_dtype"]] if run.get("kv_dtype") else []
        argv = ["--run-sh", f"{WORK}/{rsh.relative_to(ROOT)}", "--model-override", run["model_dir"], *kv]
        return script, [*argv, "--trace", "--out", raw], {}
    # trtllm: golden extra_engine_args (agg_config.yaml), consumed verbatim
    src = art / "agg_config.yaml"
    if not src.exists():
        run["skip"] = f"golden has no agg_config.yaml: {art}"
        return None
    eyml = cdir / "inputs" / f"{run['id']}.engine.yaml"
    eyml.write_text(src.read_text())
    run["render_artifact"] = str(eyml)
    # any checkpoint with custom code (auto_map) needs it; cheapest correct rule is to always pass it
    kv = ["--kv-dtype", run["kv_dtype"]] if run.get("kv_dtype") else []
    common = ["--trust-remote-code", *kv, "--engine-yaml", f"{WORK}/{eyml.relative_to(ROOT)}"]
    if tp > 1:
        return script, ["--model", run["model_dir"], "--tp", str(tp), *common, "--out", raw], {}
    return script, ["--model", run["model_dir"], *common, "--out", raw], {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}


_SHARD_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def emit_shards(runs: list[dict], cdir: Path, url_base: str, upload: bool = True) -> None:
    """One ``probe.serving`` shard per run.

    A run's inputs (its dummy model dir and rendered engine config, the files its argv names
    under WORK) go into ``bundles/<id>.tgz``; ``bundles/<id>.json`` names the probe script, its
    argv and the execution fingerprint. Both go to ``url_base`` (the WebDAV share);
    collector/probes/serving.py fetches them in the job and writes ``raw_<id>.json`` into the
    shard dir. Shard plans land in ``shards/tp<N>/<backend>-<version>.yaml``: one job per plan,
    in that release's image, with that many GPUs for a tp>1 run (AISIM_GPU_NUM)."""
    import tarfile
    import urllib.error
    import urllib.request

    shutil.rmtree(cdir, ignore_errors=True)
    _RENDERED.clear()
    for sub in ("inputs", "bundles"):
        (cdir / sub).mkdir(parents=True)
    src_info = _generator_src_commit()
    plans: dict[tuple[int, str, str], list[dict]] = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if upload:  # WebDAV refuses a PUT (409) into a missing collection: create each level; an existing one answers 405
        scheme, _, rest = url_base.partition("://")
        host, *parts = rest.split("/")
        for i in range(1, len(parts) + 1):
            try:
                url = f"{scheme}://{host}/{'/'.join(parts[:i])}/"
                opener.open(urllib.request.Request(url, method="MKCOL"), timeout=60).close()
            except urllib.error.HTTPError as exc:
                if exc.code != 405:
                    raise
    for run in (r for r in runs if "skip" not in r):
        run.update(src_info)
        probe = _prepare_probe(run, cdir)
        if probe is None:
            continue
        script, argv, env = probe
        run.update(probe_script=script, probe_args=argv, probe_env=env)
        rels = sorted({a[len(WORK) + 1 :] for a in argv if a.startswith(WORK + "/")})
        run["exec_fingerprint"] = exec_fingerprint(run, rels)
        tgz = cdir / "bundles" / f"{_SHARD_NAME.sub('_', run['id'])}.tgz"
        with tarfile.open(tgz, "w:gz", dereference=True) as tar:
            for rel in rels:
                tar.add(ROOT / rel, arcname=rel)
        spec = {
            "run_id": run["id"],
            "backend": run["backend"],
            "script": script,
            "args": argv,
            "env": env,
            "bundle": f"{url_base}/{tgz.name}",
            "fingerprint": run["exec_fingerprint"],
        }
        spec_file = tgz.with_suffix(".json")
        spec_file.write_text(json.dumps(spec, indent=1))
        if upload:
            for f in (tgz, spec_file):
                req = urllib.request.Request(f"{url_base}/{f.name}", data=f.read_bytes(), method="PUT")
                opener.open(req, timeout=600).close()
        shard = {"op": "probe.serving", "name": tgz.stem[:120], "case_filter": f"spec={url_base}/{spec_file.name}"}
        plans.setdefault((int(run.get("tp", 1)), run["backend"], run["version"]), []).append(shard)
    for (tp, backend, version), shards in sorted(plans.items()):
        d = cdir / "shards" / f"tp{tp}"
        d.mkdir(parents=True, exist_ok=True)
        plan_file = d / f"{backend}-{version}.yaml"
        plan_file.write_text(yaml.safe_dump({"backend": backend, "shards": shards}, sort_keys=False, width=1000))
        print(f"{plan_file}: {len(shards)} probe.serving shards (tp{tp}; --frameworks {backend}:{version})")
    (cdir / "plan.json").write_text(json.dumps(runs, indent=1))
    skipped = [r for r in runs if "skip" in r]
    print(f"plan: {cdir / 'plan.json'} ({len(runs) - len(skipped)} runs, {len(skipped)} skipped)")
    nofacts = [r for r in runs if "skip" not in r and (r.get("golden_facts") or {}).get("applied") is False]
    if nofacts:
        print(
            f"WARNING: {len(nofacts)} runs rendered WITHOUT generator model facts (golden_facts.applied=false) — "
            f"e.g. {nofacts[0]['id']}: {nofacts[0]['golden_facts'].get('reason')}",
            file=sys.stderr,
        )


def load_plan(cdir: Path) -> list[dict]:
    plan = cdir / "plan.json"
    if not plan.exists():
        raise SystemExit(f"no campaign plan {plan} — emit the campaign first")
    return json.loads(plan.read_text())


def import_raws(result_dir: Path, cdir: Path) -> None:
    """File the raws of a pipeline job's ``probe.serving`` shards into ``<campaign>/raw/``.
    A shard counts only when the job ran this emit (its spec fingerprint equals the plan's);
    a probe that died before writing a raw leaves its console tail as ``raw/<id>.tail``."""
    runs = {r["id"]: r for r in load_plan(cdir) if "skip" not in r}
    raw_dir = cdir / "raw"
    raw_dir.mkdir(exist_ok=True)
    n = 0
    for res_file in sorted(result_dir.rglob("probe_serving.json")):
        doc = json.loads(res_file.read_text())
        res = doc.get("result") or {}
        rid = res.get("run_id")
        run = runs.get(rid)
        if run is None:
            print(f"skip {res_file.parent.name}: run {rid} is not in this campaign", file=sys.stderr)
            continue
        if res.get("fingerprint") != run["exec_fingerprint"]:
            print(f"skip {rid}: the job ran another emit of this run (fingerprint differs)", file=sys.stderr)
            continue
        raw = res_file.parent / str(res.get("raw"))
        if res.get("raw") and raw.exists():
            (raw_dir / f"{rid}.json").write_text(raw.read_text())
        else:
            (raw_dir / f"{rid}.tail").write_text(res.get("tail") or doc.get("error") or "")
        n += 1
    print(f"imported {n} shard results into {raw_dir}")


def load_targets(path) -> dict:
    """targets.yaml with each backend's releases and images taken from
    collector/framework_manifest.yaml, the one home of a pin: the framework default and every
    family pin (a family collected at another release is probed at that release)."""
    targets = yaml.safe_load(Path(path).read_text())
    manifest = yaml.safe_load((HARNESS.parent / "framework_manifest.yaml").read_text())
    for fw, be in targets["backends"].items():
        spec = manifest["frameworks"][fw]
        pins = [spec["default"], *(spec.get("families") or {}).values()]
        be["default_version"] = str(spec["default"]["version"])
        be["images"] = {str(pin["version"]): pin["images"]["default"] for pin in pins}
        be["versions"] = list(be["images"])
    return targets


def roster_repos(targets: dict) -> list[str]:
    """Every checkpoint repo the probe roster names (targets.yaml families, derived ones
    included). Needs no dummies: fetch_inputs.py --from-roster runs before they exist."""
    repos: set[str] = set()
    for fam in targets["families"].values():
        if fam.get("derive") and "checkpoints" not in fam:
            repos.update(derived_repos(fam, targets))
        else:
            repos.update(ck["repo"] for ck in fam.get("checkpoints") or [])
    return sorted(repos)


# ---------------------------------------------------------------------------
# checkpoint quant profile: derived from the checkpoint's own quant metadata,
# NEVER the repo name
def derive_profile(repo: str, configs_dir: Path) -> str:
    stem = repo.replace("/", "_")
    p = configs_dir / f"{stem}.json"
    if not p.exists():
        return "MISSING"
    c = json.loads(p.read_text())
    qc = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config")
    hq_p = configs_dir / f"{stem}_hfquant.json"
    if hq_p.exists():
        algo = ((json.loads(hq_p.read_text()).get("quantization") or {}).get("quant_algo") or "").upper()
        if "NVFP4" in algo or "FP4" in algo:
            return "nvfp4"
        if "MXFP8" in algo:
            return "mxfp8"
        if "FP8" in algo:
            return "fp8"
    if not qc:
        return "bfloat16"

    def groups_have_4bit_float() -> bool:
        for g in (qc.get("config_groups") or {}).values():
            for part in ("weights", "input_activations"):
                w = g.get(part) or {}
                if isinstance(w, dict) and w.get("num_bits") == 4 and w.get("type") == "float":
                    return True
        # modelopt's other MIXED_PRECISION shape: flat per-layer dict
        for v in (qc.get("quantized_layers") or {}).values():
            if isinstance(v, dict) and "FP4" in str(v.get("quant_algo", "")).upper():
                return True
        return False

    m = (qc.get("quant_method") or "").lower()
    algo = (qc.get("quant_algo") or "").upper()
    if m == "mxfp8":
        return "mxfp8"
    if m == "mxfp4":
        return "mxfp4"
    if m in ("modelopt", "modelopt_mixed") or algo == "MIXED_PRECISION":
        return "nvfp4" if (groups_have_4bit_float() or "FP4" in algo) else "fp8"
    if m == "compressed-tensors":
        # 4-bit float groups = fp4-family weights (e.g. Kimi-K3 native: w4 float
        # group-32); 4-bit int = packed w4 (marlin path), served fp8-activation
        if groups_have_4bit_float():
            return "nvfp4"
        return "fp8"
    if m == "fp8":
        return "fp8"
    return f"?{m}"


# ---------------------------------------------------------------------------
# records: raw probe JSONs -> records.jsonl — kernel normalization, taxonomy labeling

# kernels that are infrastructure, never op identity
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
NAME_WRAPPERS = {
    "void",
    "cutlass::device_kernel",
    "flash::enable_sm90_or_later",
    "cute",
    "std",
    "c10",
    "at",
    "at::native",
    "int",
    "bool",
    "float",
    "unsigned",
    "long",
    "char",
    # "(anonymous namespace)::kernel<...>" — the namespace words are not the kernel
    "anonymous",
    "namespace",
}
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
            # cute-DSL instantiations bake shape parameters into the symbol; drop the blob
            return _CUTEDSL_PARAM_SUFFIX.sub("", name)[:80]
        return _TRITON_TILE_SUFFIX.sub("", name)[:80]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]{5,}", name)
    for ident in idents:
        if ident in NAME_WRAPPERS or ident.split("::")[0] in NAME_WRAPPERS or "anonymous" in ident:
            continue
        return ident[:80]
    return name.split("(")[0][:80]


def clean_path(path: str) -> str:
    parts = [p.strip() for p in path.split("<-")]
    kept = [p for p in parts if not FRAME_DENY.search(p)]
    return " <- ".join(kept[:5])


@functools.cache
def taxonomy(sm: str) -> tuple:
    """(regex, backend, role) rules for one SM from kernel_taxonomy.yaml: the rules whose
    `sms` names this SM, plus every rule without `sms`, in file order."""
    rules = yaml.safe_load((_HERE / "kernel_taxonomy.yaml").read_text())["rules"]
    return tuple((re.compile(r["match"]), r["backend"], r["role"]) for r in rules if sm in r.get("sms", [sm]))


# Orphan keep-rule (build_ops): unlabeled orphans kept per record, by device time
_ORPHAN_REST_CAP = 24


def label_kernels(kernels, sm: str):
    """kernel names -> ({backend labels}, {unmatched kernels})."""
    labels, unmatched = set(), set()
    for k in kernels:
        for rx, backend, role in taxonomy(sm):
            if rx.search(k):
                if backend not in ("infra", "framework_native"):
                    labels.add(backend)
                break
        else:
            unmatched.add(k)
    return labels, unmatched


def kernel_role(kernel: str, sm: str):
    """First matching taxonomy rule -> (backend, role); None when unlabeled.
    The role axis is what path_diff grades on (target op vs auxiliary work)."""
    for rx, backend, role in taxonomy(sm):
        if rx.search(kernel):
            return backend, role
    return None


_ORPHAN_TABLES = (("prefill_kernels", "prefill"), ("decode_kernels", "decode"), ("profile_run_kernels", "profile_run"))


def _orphan_table_name_ok(name: str) -> bool:
    return not (
        name.startswith(("AIC::", "step", "aten::"))
        or name.isupper()
        or re.match(r"^(sglang|sgl_kernel|_\w*C\w*|triton_)\w*::", name)
    )


def build_orphan_phases(facts: dict) -> dict:
    """normalized kernel -> sorted phases it ran in (prefill / decode /
    profile_run), from the device-stream tables. Phase is identity: a
    prefill gate must not borrow decode evidence."""
    trace = facts.get("trace") or facts
    phases: dict[str, set] = {}
    for tbl, phase in _ORPHAN_TABLES:
        for k in trace.get(tbl) or []:
            name = k.get("kernel", "")
            if not _orphan_table_name_ok(name):
                continue
            n = normalize_kernel(name)
            if n:
                phases.setdefault(n, set()).add(phase)
    return {k: sorted(v) for k, v in phases.items()}


def build_ops(facts: dict, sm: str) -> tuple[list[dict], list[str]]:
    """Merge api_trace spans into ops; return (ops, orphan_kernels)."""
    ops: list[dict] = []
    attributed: set[str] = set()
    trace = facts.get("trace") or facts  # sglang nests under trace; vllm flat
    # vllm's single api_trace table is not split by phase: those ops carry
    # phase None (unknown), never a borrowed "decode"
    for phase_key, phase in (("prefill_api", "prefill"), ("decode_api", "decode"), ("api_trace", None)):
        spans = trace.get(phase_key) or {}
        merged: dict[frozenset, dict] = {}
        for span, s in spans.items():
            _, kind, detail = span.split("::", 2)
            kerns = set(filter(None, (normalize_kernel(k) for k in s.get("kernels", {}))))
            attributed |= kerns
            key = frozenset(kerns) or frozenset({span})
            slot = merged.setdefault(
                key,
                {
                    "phase": phase,
                    "op": None,
                    "quant": None,
                    "api": None,
                    "kernels": sorted(kerns),
                    "calls": s.get("calls", 0),
                },
            )
            if kind == "quant_apply":
                slot["quant"] = detail
            else:
                slot["op"] = f"{kind}:{detail}"
            paths = [
                clean_path(p)
                for p in (s.get("py_paths") or {})
                if not FRAME_DENY.search(p) and not KERNEL_DENY.search(p.split("<-")[0])
            ]
            if paths and (slot["api"] is None or kind != "quant_apply"):
                slot["api"] = paths[0]
        for v in merged.values():
            if v["kernels"] or v["op"]:
                labels, unmatched = label_kernels(v["kernels"], sm)
                v["backends"] = sorted(labels) or None
                if unmatched:
                    v["unclassified_kernels"] = sorted(unmatched)
                ops.append(v)
    # trtllm probe: flat kernels list, no spans
    if not ops and facts.get("kernels"):
        kerns = sorted(set(filter(None, (normalize_kernel(k["kernel"]) for k in facts["kernels"]))))
        attributed |= set(kerns)
        labels, unmatched = label_kernels(kerns, sm)
        ops.append(
            {
                "phase": "generate",
                "op": "all",
                "quant": None,
                "api": None,
                "kernels": kerns,
                "calls": 1,
                "backends": sorted(labels) or None,
                "unclassified_kernels": sorted(unmatched) or None,
            }
        )
    # Orphan kernels are those the span attribution missed — critically, the
    # cudagraph-replayed attention kernels (no CPU launch event, so no span),
    # which live only in the device-stream prefill/decode tables. The list is
    # path_diff's serving-side identity evidence, so the keep rule is
    # structural, not a count: EVERY orphan the taxonomy labels is kept, and
    # only the unlabeled remainder is capped (by device time). An alphabetical
    # [:15] cut once dropped the fp8 sparse-MLA kernels from half the records.
    orphans_t: dict[str, float] = {}
    # profile_run_kernels: vLLM's own dummy full-model forward at engine
    # construction, and the probes' vision forward (the only place vision-encoder
    # kernels run for a text probe) — serving execution, so identity evidence
    for tbl in ("prefill_kernels", "decode_kernels", "profile_run_kernels"):
        for k in trace.get(tbl) or []:
            name = k.get("kernel", "")
            if not _orphan_table_name_ok(name):
                continue  # spans, phase markers, custom-op launchers — not kernels
            n = normalize_kernel(name)
            if n and n not in attributed:
                orphans_t[n] = orphans_t.get(n, 0.0) + float(k.get("us") or 0.0)
    # Every taxonomy-labeled orphan is kept — including the cublas / vllm_kernel /
    # sgl_kernel / torch families: under CUDA graphs and torch.compile the Python
    # spans that used to attribute GEMM and quant kernels vanish, so those kernels
    # arrive here, and path_diff grades gemm-class ops on exactly their names.
    # Labeled means any taxonomy match, framework_native and infra included: the
    # aux-work check looks for the collector's routing / norm_rope / kvcache kernels
    # among these (a capped-out cooperative_topk decode kernel read as aux drift).
    signal = {n for n in orphans_t if kernel_role(n, sm)}
    rest = sorted((n for n in orphans_t if n not in signal), key=lambda n: -orphans_t[n])
    return ops, sorted(signal | set(rest[:_ORPHAN_REST_CAP]))


def compress_error(stage: str, tb: str) -> dict:
    lines = tb.strip().splitlines()
    frames = [ln.strip()[:110] for ln in lines if ln.strip().startswith("File") and FW_FRAME.search(ln)][-5:]
    return {"stage": stage, "exc": lines[-1][:400], "frames": frames}


_RESOLVED_KEEP = re.compile(
    r"^(kv_cache_dtype|page_size|block_size|quantization|attention_backend|"
    r"(prefill|decode)_attention_backend|dsa_(prefill|decode|topk|paged_mqa_logits)_backend|"
    r"moe_(runner|a2a)_backend|fp8_gemm_runner_backend|fp4_gemm_runner_backend|"
    r"bf16_gemm_backend|linear_attn_backend|mamba_backend|dtype|load_format|"
    r"tensor_parallel_size|max_model_len|context_length)$"
)


def build_record(run: dict, f: dict, imported_at: float) -> dict:
    """One record: the run's target and runtime, the resolved serving config, the
    deployed identity and the kernels by op and phase. The SM is the one the raw ran on."""
    sm = f.get("device_capability") or run["sm"]
    ops, orphans = build_ops(f, sm)
    sa = f.get("server_args_resolved") or f.get("engine_args_resolved") or {}
    sa = {k: v for k, v in sa.items() if _RESOLVED_KEEP.match(k)}
    rec = {
        "id": run["id"],
        "target": {k: run.get(k) for k in ("repo", "family", "variant", "profile", "aic_registered")},
        "runtime": {
            **{k: run.get(k) for k in ("backend", "version", "image", "tp", "system")},
            "sm": sm,
            "engine_cli": run.get("engine_cli"),
            "unknown_args": f.get("engine_cli_unknown_args") or None,
            # the collector sweeps fp8-KV, so path_diff compares against the SAME-kv record
            "kv_cache_dtype": (
                f.get("probe_kv_cache_dtype")
                or sa.get("kv_cache_dtype")
                or (f.get("kv_cache_resolved") or {}).get("attn_kv_cache_dtype")
                or "auto"
            ),
            # prompt length decides WHICH prefill path serving took (DSA: query<=256 -> decode kernel)
            "isl": f.get("probe_isl"),
            # probes run the framework's own mode (torch.compile + CUDA graphs); eager is an A/B
            "probe_eager": f.get("probe_eager"),
            "probe_cuda_graph": f.get("probe_cuda_graph"),
            # weight content of the dummy model: the kernel times are a timing reference only
            # on dense low-bit weights (probes/_dense_dummy.py; TRT-LLM's own dummy draw is dense)
            "dense_weights": f.get("probe_dense_weights"),
            "framework_dummy": f.get("probe_framework_dummy"),
        },
        "resolved": {k: v for k, v in sa.items() if v is not None},
        # generator-rendered flags that differ from the framework's own parser defaults:
        # every delta is an auditable liability (the DSV4 0.29 crash was triggered by one)
        "config_delta": f.get("config_delta") or None,
        "identity": {
            "model_class": f.get("model_class"),
            "modules": {
                k.rsplit(".", 1)[-1]: v.get("modules", v.get("examples", []))
                for k, v in (f.get("quant_methods") or {}).items()
            },
            "param_dtypes": f.get("param_dtypes"),
            "weight_samples": f.get("weight_samples") or None,
            # what the engine actually allocated (runner / pool / manager dtypes)
            "kv_cache_resolved": f.get("kv_cache_resolved") or None,
        },
        "exec_fingerprint": run.get("exec_fingerprint"),
        "golden_facts": run.get("golden_facts"),
        "imported_at": imported_at,
        "ops": ops or None,
        "orphan_kernels": orphans or None,
        # phase of every kernel the device tables saw (span-attributed ones included): span
        # attribution is by name across phases, so a kernel both phases launch would otherwise
        # vanish from the phase its span did not cover (sglang decode FA3, 2026-09-25)
        "kernel_phases": build_orphan_phases(f) or None,
        "outcome": ({"status": "ok"} if not f.get("errors") else compress_error(*next(iter(f["errors"].items())))),
    }
    return {k: v for k, v in rec.items() if v is not None}


def run_status(run: dict, cdir: Path, record: dict | None) -> str:
    """Why a planned run is, or is not, serving evidence: ``ok``, the probe's own error, the
    last line of a job that died before writing its raw, or the reason it was never emitted."""
    if "skip" in run:
        return f"not emitted: {run['skip'][:160]}"
    if record is None:
        tail = cdir / "raw" / f"{run['id']}.tail"
        if not tail.exists():
            return "not run"
        lines = [ln.strip() for ln in tail.read_text(errors="replace").splitlines() if ln.strip()]
        return f"no raw: {lines[-1][:200] if lines else 'empty log'}"
    outcome = record["outcome"]
    return "ok" if outcome.get("status") == "ok" else f"failed at {outcome['stage']}: {outcome['exc'][:200]}"


def build_records(cdir: Path) -> Path:
    out = cdir / "records.jsonl"
    n = 0
    with out.open("w") as fh:
        for run in load_plan(cdir):
            raw = cdir / "raw" / f"{run.get('id')}.json"
            record = None
            if "skip" not in run and raw.exists():
                record = build_record(run, json.loads(raw.read_text()), raw.stat().st_mtime)
                fh.write(json.dumps(record) + "\n")
                n += 1
            label = " ".join(str(run.get(k)) for k in ("backend", "version", "repo") if run.get(k))
            print(f"{run.get('id', '-')} {label} kv={run.get('kv_dtype') or 'auto'}: {run_status(run, cdir, record)}")
    print(f"wrote {out}: {n} records")
    return out


def load_records(cdir: Path) -> dict:
    path = cdir / "records.jsonl"
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", type=Path, default=HARNESS / "targets.yaml")
    ap.add_argument("--campaign", help="campaign name: <workspace>/campaigns/<name>/ (default: the --system)")
    ap.add_argument("--system", help="the system the campaign renders for and runs on, e.g. b200_sxm")
    ap.add_argument("--backends", default=None, help="comma list (default: sglang; with --gates every backend)")
    ap.add_argument("--only", default=None, help="comma list of repo substrings — only matching checkpoints")
    ap.add_argument("--full", action="store_true", help="all variants and topologies (default: representative tp1)")
    ap.add_argument("--isl", type=int, default=DEFAULT_ISL, help="the probes' prompt length (a gate may need another)")
    ap.add_argument("--version", default=None, help="the release probed (default: the framework default)")
    ap.add_argument("--gates", action="store_true", help="the runs the gates in scope read")
    ap.add_argument("--op", default=None, help="with --gates: only the gates of this collector op")
    ap.add_argument("--derive", action="store_true", help="with --gates: also the gates derived for unnamed paths")
    ap.add_argument("--plan", action="store_true", help="print the runs")
    ap.add_argument("--emit-shards", action="store_true", help="render, bundle and upload; write shard plans")
    ap.add_argument("--url-base", default=None, help="WebDAV directory the probe.serving shards fetch from")
    ap.add_argument("--no-upload", action="store_true", help="with --emit-shards: write bundles, do not upload")
    ap.add_argument("--import-raws", type=Path, default=None, metavar="DIR", help="file a job result's raws")
    ap.add_argument("--records", action="store_true", help="raw/ -> records.jsonl")
    args = ap.parse_args()

    targets = load_targets(args.targets)
    name = args.campaign or args.system
    if not name:
        ap.error("name the campaign (--campaign) or its system (--system)")
    cdir = campaign_dir(name)
    if args.import_raws:
        import_raws(args.import_raws, cdir)
    if args.records:
        build_records(cdir)
    if not (args.plan or args.emit_shards):
        return
    if not args.system:
        ap.error("--plan and --emit-shards need --system")
    backends = (args.backends or ("sglang,vllm,trtllm" if args.gates else "sglang")).split(",")
    if args.gates:
        runs = gate_runs(targets, args.system, backends, args.op, args.derive)
    else:
        pats = [p.strip() for p in (args.only or "").split(",") if p.strip()]
        # --only scopes the roster derivation too: a single checkpoint needs only its own fetched config
        runs = enumerate_runs(targets, args.system, args.full, backends, pats or None, args.isl, args.version)
        if pats:
            runs = [r for r in runs if any(p in (r.get("repo") or "") for p in pats)]
    if args.plan:
        for r in runs:
            print(json.dumps(r))
        return
    if not args.url_base:
        ap.error("--emit-shards needs --url-base (the share the job fetches the bundles from)")
    ROOT.mkdir(parents=True, exist_ok=True)
    emit_shards(runs, cdir, args.url_base.rstrip("/"), upload=not args.no_upload)


if __name__ == "__main__":
    main()
