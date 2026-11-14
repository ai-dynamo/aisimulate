# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe: what serving deploys for a generator-rendered config, on the collection pipeline.

The opharness serving probes (collector/opharness/components/probes/probe_<fw>.py) boot the
real framework on a dummy checkpoint with the engine arguments the generator rendered, and
record the resolved identity and the kernels each phase ran. ``probe_driver.py --emit-shards``
packs one run's inputs (its dummy model dir and rendered config) into a bundle and writes a
spec naming the probe script and its argv, both on the WebDAV share. This probe fetches them,
runs that script unchanged in the job's framework image (``{BUNDLE}`` is the unpacked bundle,
``{OUT}`` the shard dir), and leaves ``raw_<run id>.json`` for ``probe_driver.py --import-raws``.

Argument: ``spec=<url>`` (``--case-filter spec=http://.../<id>.json``).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_PROBES = Path(__file__).resolve().parents[1] / "opharness" / "components" / "probes"
# vLLM 0.32 runs FlashInfer's MoE autotune at startup over 21 token buckets; with an empty
# cache the Kimi-K3 dummy did not finish it within 1500 s on B200 (serving pays it once).
_TIMEOUT_S = 3600
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _fetch(url: str) -> bytes:
    with _OPENER.open(url, timeout=600) as resp:
        return resp.read()


def run(args: dict[str, str], out_dir: Path) -> dict:
    spec = json.loads(_fetch(args["spec"]))
    bundle = Path(tempfile.mkdtemp(prefix="serving_probe_"))
    with tarfile.open(fileobj=io.BytesIO(_fetch(spec["bundle"])), mode="r:gz") as tar:
        tar.extractall(bundle, filter="data")
    argv = [a.replace("{BUNDLE}", str(bundle)).replace("{OUT}", str(out_dir)) for a in spec["args"]]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **spec.get("env", {})}
    raw = Path(out_dir) / f"raw_{spec['run_id']}.json"
    try:
        proc = subprocess.run(
            [sys.executable, str(_PROBES / spec["script"]), *argv],
            env=env,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
        returncode, output = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, output = None, f"timeout after {_TIMEOUT_S}s: {exc.output or ''}"
    print(output[-4000:], flush=True)
    return {
        "run_id": spec["run_id"],
        "backend": spec["backend"],
        "script": spec["script"],
        "fingerprint": spec.get("fingerprint"),
        "returncode": returncode,
        "raw": raw.name if raw.exists() else None,
        "tail": output[-3000:],
        "failed": not raw.exists(),
    }
