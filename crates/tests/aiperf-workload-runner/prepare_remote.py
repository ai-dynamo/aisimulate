#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote-only path mapping, independent lock generation and normal-closure gate.

Invoke on the coordinator's remote compute target with AIPERF_SOURCE_ROOT set.
Never invoke on the local host. Does not compile or test.
"""
import json
import os
from pathlib import Path
import re
import subprocess

package = Path(__file__).resolve().parent
source = Path(os.environ["AIPERF_SOURCE_ROOT"]).resolve(strict=True)
manifest = package / "Cargo.toml"
text = manifest.read_text()
text = re.sub(r'path = "(?:\.\./)+aiperf-rs/\.worktrees/prove-simple-execution/(crates/[^\"]+)"', lambda match: 'path = "' + os.path.relpath(source / match[1], package) + '"', text)
manifest.write_text(text)
# A committed independent lock is evidence: never refresh it implicitly.
if not (package / "Cargo.lock").is_file():
    subprocess.run(["cargo", "generate-lockfile", "--manifest-path", str(manifest)], check=True)
result = subprocess.run(["cargo", "metadata", "--locked", "--format-version", "1", "--manifest-path", str(manifest)], check=True, capture_output=True, text=True)
metadata = json.loads(result.stdout)
packages = {item["id"]: item for item in metadata["packages"]}
nodes = {item["id"]: item for item in metadata["resolve"]["nodes"]}
root = metadata["resolve"]["root"]
seen = set()
pending = [root]
while pending:
    key = pending.pop()
    if key in seen:
        continue
    seen.add(key)
    for dependency in nodes[key]["deps"]:
        if any(kind["kind"] is None for kind in dependency["dep_kinds"]):
            pending.append(dependency["pkg"])
names = sorted(packages[key]["name"] for key in seen)
forbidden = {"aiperf-runtime", "aiperf-cli", "aiperf-simulate", "aiperf-app", "aiperf"}
assert not forbidden.intersection(names), "forbidden application dependency: " + repr(forbidden.intersection(names))
assert Path(metadata["workspace_root"]).resolve() == package, "proof must own its Cargo workspace"
# Root tests may consume owning libraries, never the application or simulate.
# Dependency dev-dependencies are not part of this package's test graph.
test_seen = set()
pending = [root]
while pending:
    key = pending.pop()
    if key in test_seen:
        continue
    test_seen.add(key)
    for dependency in nodes[key]["deps"]:
        kinds = {kind["kind"] for kind in dependency["dep_kinds"]}
        if kinds.intersection({None, "build"}) or (key == root and "dev" in kinds):
            pending.append(dependency["pkg"])
test_names = sorted(packages[key]["name"] for key in test_seen)
assert not forbidden.intersection(test_names), "forbidden test application dependency: " + repr(forbidden.intersection(test_names))
print(json.dumps({"workspace_root":str(package),"normal_closure":names,"test_closure":test_names,"forbidden_absent":sorted(forbidden)}, indent=2))
