#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a perf-data collection campaign as declared shards, one container per shard.

Why this exists (sm120 campaign, 2026-09-30): every box so far re-invented the
same launcher by hand — an image with the finalize deps added, ``--case-filter``
shards, one cwd per shard so staging CSVs never collide, a GPU pool, and the
finalize step. Each re-invention drifted somewhere (the sm120 run passed
``--keep-csv`` and silently produced no parquet/provenance for 13 shards; the
official image lacks pyarrow so the smoke run died at finalize). This tool fixes
the invariants once:

* the image is the framework_manifest pin for the (framework, op) being run —
  never a hand-typed tag — plus the finalize deps baked on top (``build-image``);
* one output/checkpoint namespace per shard (``<root>/<op>/<shard>/``), exactly
  as the collector-upgrade playbook §5 asks;
* ``collect.py`` is always invoked with ``--resume`` and never ``--keep-csv``,
  so a finished shard always ends with parquet + collection_meta.yaml;
* every container launch is appended to ``<root>/campaign.jsonl`` (image digest,
  argv, exit code, parquet outputs) — the work log the playbook §4 wants.

Shard plan (YAML)::

    backend: vllm
    sm: 120
    shards:
      - {op: gemm, name: bf16, case_filter: "['bfloat16', "}
      - {op: gemm, name: fp8,  case_filter: "['fp8', "}
      - {op: kda,  name: all}

Usage::

    python3 tools/perf_database/collect_campaign.py build-image --backend vllm --op gemm
    python3 tools/perf_database/collect_campaign.py run   --plan shards.yaml --root /ws/collect --gpus 0,1,2,3
    python3 tools/perf_database/collect_campaign.py status --plan shards.yaml --root /ws/collect
    python3 tools/perf_database/collect_campaign.py finalize --plan shards.yaml --root /ws/collect --gpus 0

``run`` is re-entrant: shards with a ``DONE`` marker are skipped, unfinished
shards resume from their checkpoint. ``finalize`` re-enters finished shards that
have no parquet (e.g. runs made by hand with ``--keep-csv``) — every task is
already checkpointed, so collect.py only finalizes. Merging the shards of one op
into one table is the delivery step (playbook §8) and is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]  # python/aisimulate
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

FINALIZE_DEPS = ("pyarrow", "pandas")  # collector/helper.py finalize_perf_files imports
CONTAINER_CHECKOUT = "/ais"
CONTAINER_OUT = "/out"


@dataclass(frozen=True)
class Shard:
    op: str
    name: str
    case_filter: str | None = None

    @property
    def key(self) -> str:
        return f"{self.op}/{self.name}"


@dataclass(frozen=True)
class Plan:
    backend: str
    sm: int
    shards: tuple[Shard, ...]


def load_plan(path: Path) -> Plan:
    doc = yaml.safe_load(path.read_text())
    shards = []
    seen: set[str] = set()
    for raw in doc["shards"]:
        s = Shard(op=str(raw["op"]), name=str(raw.get("name", "all")), case_filter=raw.get("case_filter") or None)
        if s.key in seen:
            raise ValueError(f"duplicate shard {s.key}")
        seen.add(s.key)
        shards.append(s)
    return Plan(backend=str(doc["backend"]), sm=int(doc["sm"]), shards=tuple(shards))


def pinned_image(backend: str, op: str) -> str:
    """The manifest-pinned image for (backend, op) — family pins included."""
    from collector.framework_manifest import resolve_op_runtime

    return resolve_op_runtime(backend, op).image()


def collect_image_tag(base_image: str) -> str:
    """Deterministic local tag for <base image> + finalize deps."""
    name = base_image.split("@", 1)[0].replace("/", "-").replace(":", "-")
    return f"aisim-collect:{name}"


def dockerfile(base_image: str) -> str:
    return (
        f"FROM {base_image}\n"
        "# collector finalize deps the framework image does not ship (collector/helper.py finalize_perf_files)\n"
        f"RUN pip install --no-cache-dir {' '.join(FINALIZE_DEPS)}\n"
    )


def build_image(base_image: str, *, network: str = "host", extra_build_args: list[str] | None = None) -> str:
    tag = collect_image_tag(base_image)
    cmd = ["docker", "build", "--network", network, "-t", tag, "-f", "-", "."] + (extra_build_args or [])
    subprocess.run(cmd, input=dockerfile(base_image), text=True, check=True)
    return tag


def image_digest(image: str) -> str | None:
    try:
        out = subprocess.run(
            # a locally built collect image has NO RepoDigests — `index .RepoDigests 0` fails and the launcher
            # refused its own build-image output (B200 2026-10-03); fall back to the image Id
            ["docker", "inspect", "--format", "{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}|{{.Id}}", image],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out
    except Exception:
        return None


def collector_argv(plan: Plan, shard: Shard) -> list[str]:
    """The exact collect.py argv for one shard. ``--resume`` always, ``--keep-csv`` never."""
    argv = [
        f"{CONTAINER_CHECKOUT}/collector/collect.py",
        "--backend",
        plan.backend,
        "--model-cases-full",
        "--sm",
        str(plan.sm),
        "--ops",
        shard.op,
        "--checkpoint-dir",
        f"{CONTAINER_OUT}/.ckpt",
        "--resume",
    ]
    if shard.case_filter:
        argv += ["--case-filter", shard.case_filter]
    assert "--keep-csv" not in argv
    return argv


def docker_argv(
    plan: Plan, shard: Shard, *, image: str, gpu: int, checkout: Path, shard_dir: Path, container_name: str
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        f'"device={gpu}"',
        "--shm-size",
        "16g",
        "-e",
        "CUDA_MPS_PIPE_DIRECTORY=/nonexistent-no-mps",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        f"PYTHONPATH={CONTAINER_CHECKOUT}",
        "-e",
        f"TRITON_CACHE_DIR={CONTAINER_OUT}/.cache/triton",
        "-e",
        f"DG_JIT_CACHE_DIR={CONTAINER_OUT}/.cache/deep_gemm",
        "-e",
        f"FLASHINFER_WORKSPACE_BASE={CONTAINER_OUT}/.cache",
        "-v",
        f"{checkout}:{CONTAINER_CHECKOUT}:ro",
        "-v",
        f"{shard_dir}:{CONTAINER_OUT}",
        "-w",
        CONTAINER_OUT,
        "--entrypoint",
        "python3",
        image,
        *collector_argv(plan, shard),
    ]


def shard_dir(root: Path, shard: Shard) -> Path:
    return root / shard.op / shard.name


def shard_state(root: Path, shard: Shard) -> str:
    d = shard_dir(root, shard)
    if not d.exists():
        return "pending"
    if list(d.glob("*.parquet")):
        return "finalized"
    if (d / "DONE").exists():
        return "done-no-parquet"
    if (d / ".ckpt").exists():
        return "started"
    return "pending"


def _log(root: Path, record: dict) -> None:
    with (root / "campaign.jsonl").open("a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n")


def run_shard(plan: Plan, shard: Shard, *, root: Path, gpu: int, checkout: Path, image: str, mode: str = "run") -> int:
    d = shard_dir(root, shard)
    (d / ".cache").mkdir(parents=True, exist_ok=True)
    name = f"{'f' if mode == 'finalize' else 'c'}_{plan.backend}_{shard.op}_{shard.name}"
    cmd = docker_argv(plan, shard, image=image, gpu=gpu, checkout=checkout, shard_dir=d, container_name=name)
    _log(
        root,
        {
            "event": "start",
            "mode": mode,
            "shard": shard.key,
            "gpu": gpu,
            "image": image,
            "image_digest": image_digest(image),
            "argv": collector_argv(plan, shard),
        },
    )
    with (d / "run.log").open("a") as log:
        log.write(f"### {time.strftime('%FT%T')} {mode} gpu{gpu} {shlex.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"### {time.strftime('%FT%T')} exit={rc}\n")
    parquets = sorted(p.name for p in d.glob("*.parquet"))
    if rc == 0:
        (d / "DONE").touch()
    _log(root, {"event": "end", "mode": mode, "shard": shard.key, "gpu": gpu, "exit": rc, "parquet": parquets})
    return rc


def _free_gpu(busy: dict[int, subprocess.Popen | None], gpus: list[int]) -> int | None:
    for g in gpus:
        p = busy.get(g)
        if p is None or p.poll() is not None:
            return g
    return None


def run_campaign(
    plan: Plan, *, plan_path: Path, root: Path, gpus: list[int], checkout: Path, image_override: str | None, mode: str
) -> int:
    """GPU pool over the plan. mode=run: skip DONE shards; mode=finalize: only done-no-parquet shards."""
    root.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in plan.shards:
        st = shard_state(root, s)
        if mode == "run" and st in ("finalized", "done-no-parquet") and (shard_dir(root, s) / "DONE").exists():
            continue
        if mode == "finalize" and st != "done-no-parquet":
            continue
        todo.append(s)
    print(f"{mode}: {len(todo)}/{len(plan.shards)} shards on gpus {gpus}")
    images = {s.op: (image_override or collect_image_tag(pinned_image(plan.backend, s.op))) for s in todo}
    for img in sorted(set(images.values())):
        if image_digest(img) is None:
            raise SystemExit(f"image {img} not present — run `build-image` for its base first")
    busy: dict[int, subprocess.Popen | None] = {}
    failures = 0
    while todo:
        g = _free_gpu(busy, gpus)
        if g is None:
            time.sleep(15)
            for p in busy.values():
                if p is not None and p.poll() not in (None, 0):
                    failures += 1
                    busy[[k for k, v in busy.items() if v is p][0]] = None
            continue
        s = todo.pop(0)
        print(f"{time.strftime('%T')} gpu{g} <- {s.key}")
        busy[g] = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "_shard",
                "--plan",
                str(plan_path),
                "--root",
                str(root),
                "--gpu",
                str(g),
                "--shard",
                s.key,
                "--checkout",
                str(checkout),
                "--image",
                images[s.op],
                "--mode",
                mode,
            ]
        )
    for p in busy.values():
        if p is not None and p.wait() != 0:
            failures += 1
    print(f"{mode} finished, {failures} shard(s) exited non-zero")
    return 1 if failures else 0


def status(plan: Plan, root: Path) -> None:
    for s in plan.shards:
        d = shard_dir(root, s)
        last = ""
        log = d / "run.log"
        if log.exists():
            tail = log.read_text(errors="replace").replace("\r", "\n").splitlines()
            last = next(
                (ln for ln in reversed(tail) if "it/s" in ln or "s/it" in ln or "Completed" in ln or "exit=" in ln), ""
            )
        print(f"{s.op:28s} {s.name:20s} {shard_state(root, s):16s} {last[:100]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-image", help="build <manifest image for (backend, op)> + finalize deps")
    b.add_argument("--backend", required=True)
    b.add_argument("--op", required=True, help="any op of the backend; family pins may select a different image")
    b.add_argument("--build-arg", action="append", default=[], help="extra docker build args, e.g. http_proxy=...")
    for name in ("run", "finalize", "status"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True, type=Path)
        p.add_argument("--root", required=True, type=Path, help="campaign root: one <op>/<shard>/ namespace per shard")
        if name != "status":
            p.add_argument("--gpus", required=True, help="comma list of GPU indices for the pool")
            p.add_argument("--checkout", type=Path, default=PKG_ROOT, help="python/aisimulate checkout to mount")
            p.add_argument("--image", default=None, help="override the manifest-derived collect image (A/B only)")
    h = sub.add_parser("_shard")  # internal: one shard in one container (spawned by run/finalize)
    for flag in ("--plan", "--root", "--checkout"):
        h.add_argument(flag, required=True, type=Path)
    h.add_argument("--gpu", required=True, type=int)
    h.add_argument("--shard", required=True)
    h.add_argument("--image", required=True)
    h.add_argument("--mode", required=True)
    args = ap.parse_args()
    if args.cmd == "_shard":
        plan = load_plan(args.plan)
        shard = next(s for s in plan.shards if s.key == args.shard)
        return run_shard(
            plan, shard, root=args.root, gpu=args.gpu, checkout=args.checkout, image=args.image, mode=args.mode
        )
    if args.cmd == "build-image":
        base = pinned_image(args.backend, args.op)
        tag = build_image(base, extra_build_args=[f"--build-arg={a}" for a in args.build_arg])
        print(f"built {tag} from {base}")
        return 0
    plan = load_plan(args.plan.resolve())
    if args.cmd == "status":
        status(plan, args.root.resolve())
        return 0
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    return run_campaign(
        plan,
        plan_path=args.plan.resolve(),
        root=args.root.resolve(),
        gpus=gpus,
        checkout=args.checkout.resolve(),
        image_override=args.image,
        mode=args.cmd,
    )


if __name__ == "__main__":
    raise SystemExit(main())
