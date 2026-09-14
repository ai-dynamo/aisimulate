# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare portable source-qualified inputs; never run a model or prediction.

Historical qualification modules come verbatim from this repository at immutable
5b6309570bb7e1ccf5fa461afa92392df11d9a50, retaining their original Apache-2.0
copyright/license headers. They qualify observations only; current SDK/native
imports remain separate. Provenance is recorded beside each generated packet.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

from refresh_reported_native import require, sha, write

REVISION = "5b6309570bb7e1ccf5fa461afa92392df11d9a50"
MODULES = {
    "collector/__init__.py": "f38debac7a7f711f1b87f9f65dd751bcd391166b619ce4dcc13f2fc12d63e84e",
    "collector/sglang/dsv41_forward_results.py": "7aa2ec62de645525dcc9a5a3e524c0be4982009459db310d0d95fa47823f0e50",
    "collector/sglang/dsv41_workloads.py": "8a29b4777dee2a82a59b5e921762aafea6c7a65181047428d9d036247c8bb3fe",
}
OBSERVATIONS = {
    "off": ("full", "d4bf262698790520abd704dae04bed269cad7cc8c10c2e5a7b90ac90a9ddc410"),
    "on": ("decoder_bounded", "a468a4e5364bebdfd28677a51fe489e4c17d5778ab11009b41dd69a7dc8a7568"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--silicon-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    provenance = {
        "repository": "https://github.com/ai-dynamo/aisimulate",
        "revision": REVISION,
        "license": "Apache-2.0",
        "modified": False,
        "files": {},
    }
    for relative, expected in MODULES.items():
        original = "python/aisimulate/" + relative
        raw = subprocess.check_output(["git", "show", f"{REVISION}:{original}"], cwd=args.repo)
        require(hashlib.sha256(raw).hexdigest() == expected, "qualification source differs")
        path = args.output / "qualifiers" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        provenance["files"][relative] = {"sha256": expected, "original_path": original}
    base = args.silicon_repo.resolve() / "data/experimental/deepseek-v41/gb300-silicon/prefix-refinement-v1"
    plan = base / "heldout-points.json"
    require(sha(plan) == "20fb257c97b8e87d4d1468a8ea9382de51a1584925a2c7e1e0af6a40708b4def", "plan differs")
    for profile, (directory, expected) in OBSERVATIONS.items():
        observed = base / directory / "heldout/forward-results.json"
        require(sha(observed) == expected, "observations differ")
        paths = {"heldout-plan": plan, "observations": observed}
        write(
            args.output / f"{profile}.json",
            {
                "inputs": {key: str(path) for key, path in paths.items()},
                "input_pins": {key: {"sha256": sha(path), "bytes": path.stat().st_size} for key, path in paths.items()},
            },
        )
    write(args.output / "source-provenance.json", provenance)


if __name__ == "__main__":
    main()
