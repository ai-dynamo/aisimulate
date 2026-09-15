# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-time release runner: pinned release wheel/inventory plus pinned qualification tooling."""

import hashlib
import importlib.metadata
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "1f728534910187ecb2f021ef5e4bd4949bd555d0"
TOOLING_SHA = "58256c63de898b57c207d8829e5bb0cb3a5c7c3c"
for filename, expected in [("source.tar", SOURCE_SHA), ("tooling.tar", TOOLING_SHA)]:
    with (ROOT / filename).open("rb") as source:
        actual = subprocess.check_output(["git", "get-tar-commit-id"], stdin=source, text=True).strip()
    if actual != expected:
        raise SystemExit(f"Wrong {filename} commit: {actual}")
wheels = list((ROOT / "wheels").glob("aisimulate-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected one release wheel")
wheel_sha = hashlib.sha256(wheels[0].read_bytes()).hexdigest()
dist = importlib.metadata.distribution("aisimulate")
if dist.version != "0.12.0":
    raise SystemExit(f"Wrong installed package version: {dist.version}")
with zipfile.ZipFile(wheels[0]) as wheel:
    for name in wheel.namelist():
        if (
            name.startswith(("aisimulate/", "aisimulate_core/", "aiconfigurator/", "aiconfigurator_core/"))
            and not name.endswith("/")
            and Path(dist.locate_file(name)).read_bytes() != wheel.read(name)
        ):
            raise SystemExit(f"Installed file differs from the qualified wheel: {name}")
from aisimulate import _runtime

runtime = Path(_runtime.__file__).resolve()
owned = {Path(dist.locate_file(p)).resolve() for p in dist.files or () if str(p).startswith("aisimulate/_runtime.")}
if runtime not in owned:
    raise SystemExit(f"Native runtime is not owned by the installed wheel: {runtime}")
os.environ["FPE_WHEEL_SHA256"] = wheel_sha
sys.path.insert(0, str(ROOT / "tooling/python/aisimulate"))
from tools.support_matrix import generate_fpe_support_matrix as generator

# These rows describe the verified installed release wheel, not the newer probe harness.
generator._source_sha = lambda: SOURCE_SHA
print(f"source_sha={SOURCE_SHA} tooling_sha={TOOLING_SHA} wheel_sha256={wheel_sha}", flush=True)
generator.main()
