# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Dynamo replay adapter against one exact AISimulate checkout."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

DYNAMO_REVISION = "def3b79b15c266805540a678dd400aeb6ccada1d"


# Compatibility adaptation of NVIDIA Dynamo's Apache-2.0 handoff adapter:
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/handoff.rs
# Modifies only the CI-built legacy token-KV DTO conversion, not engine scheduling.
def patch_handoff_timing(core: Path, dynamo: Path) -> dict:
    path = dynamo / "lib/mocker/src/common/handoff.rs"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != "99a2097d671f4a26dbd1761a44858a19f62ced37dd9209858627b2a289d7937d":
        raise ValueError("pinned Dynamo handoff adapter differs from reviewed source")
    api = (core / "src/engine/handoff.rs").read_text()
    match = re.search(r"pub struct HandoffTransferTiming\s*\{([^{}]*)\}", api)
    fields = dict(re.findall(r"^\s*pub (\w+): ([^\n,]+),\s*$", match[1], re.MULTILINE)) if match else {}
    legacy = {
        "mode": "TransferTimingMode",
        "full_prompt_tokens": "usize",
        "kv_bytes_per_token": "Option<usize>",
        "bandwidth_gb_s": "Option<f64>",
    }
    if fields not in (legacy, legacy | {"state_bytes": "usize"}):
        raise ValueError("unreviewed AISimulate handoff API; update the pinned adapter")
    patched = original
    if "state_bytes" in fields:
        anchor = b"        kv_bytes_per_token: timing.kv_bytes_per_token,\n"
        if original.count(anchor) != 1:
            raise ValueError("pinned handoff conversion anchor is not unique")
        patched = original.replace(
            anchor,
            anchor
            + (
                b"        // Modified by AISimulate CI: this legacy DTO models token KV only.\n"
                b"        state_bytes: 0,\n"
            ),
        )
        path.write_bytes(patched)
    return {
        "path": str(path.relative_to(dynamo)),
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(patched).hexdigest(),
        "token_only_state_bytes": 0,
        "api_has_state_bytes": "state_bytes" in fields,
    }


def build(source: Path, dynamo: Path, output: Path) -> None:
    source, dynamo, output = source.resolve(), dynamo.resolve(), output.resolve()
    revision = subprocess.check_output(["git", "-C", str(dynamo), "rev-parse", "HEAD"], text=True).strip()
    if revision != DYNAMO_REVISION:
        raise ValueError("unexpected Dynamo revision")
    subprocess.run(["git", "-C", str(dynamo), "diff", "--exit-code", "HEAD"], check=True)
    core = source / "crates/core"
    handoff_compatibility = patch_handoff_timing(core, dynamo)
    bindings = dynamo / "lib/bindings/python"
    for manifest, options in (
        (dynamo / "Cargo.toml", ""),
        (bindings / "Cargo.toml", ', optional = true, features = ["python"]'),
    ):
        text, count = re.subn(
            r"^aisimulate-core = .*",
            lambda _: f"aisimulate-core = {{ path = {json.dumps(str(core))}{options} }}",
            manifest.read_text(),
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected one AISimulate dependency in {manifest}")
        manifest.write_text(text)
    # Resolve the local core's dependency graph from Dynamo's existing lock.
    # Compilation stays locked to this resolved graph, which is saved below.
    lock = bindings / "Cargo.lock"
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--features", "ais-forward-pass"],
            cwd=bindings,
        )
    )
    linked = [p["manifest_path"] for p in metadata["packages"] if p["name"] == "aisimulate-core"]
    if linked != [str(core / "Cargo.toml")]:
        raise ValueError(f"Dynamo resolved a different AISimulate core: {linked}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--features",
            "ais-forward-pass",
            "--out",
            str(output),
        ],
        cwd=bindings,
        check=True,
    )
    subprocess.run(
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(output), str(dynamo)],
        check=True,
    )
    (output / "dynamo-Cargo.lock").write_bytes(lock.read_bytes())
    (output / "dynamo-build.json").write_text(
        json.dumps(
            {
                "source_sha": revision,
                "aisimulate_sha": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "features": ["ais-forward-pass"],
                "handoff_compatibility": handoff_compatibility,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "dynamo-build.patch").write_bytes(subprocess.check_output(["git", "-C", str(dynamo), "diff", "HEAD"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dynamo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.dynamo, args.output)
