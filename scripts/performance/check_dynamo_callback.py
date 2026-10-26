# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the complete patched Dynamo callback against the selected AIS core/SDK."""

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

from build_simulation_dynamo import (
    SPECULATION_DEPTH_NEW,
    SPECULATION_DEPTH_OLD,
    SPECULATION_DEPTH_PATH,
    SPECULATION_DEPTH_SHA256,
)

# Minimal interface scaffolding adapted from NVIDIA Dynamo, Apache-2.0:
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/perf_model.rs
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/kv-router/src/scheduling/prefill_load.rs
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified: retain only the callback traits; no Dynamo scheduling implementation.
SCAFFOLD = """// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted interfaces from NVIDIA Dynamo, modified to omit scheduling implementation:
// https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/perf_model.rs
// https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/kv-router/src/scheduling/prefill_load.rs
#![allow(dead_code)]
extern crate self as dynamo_mocker;
extern crate self as dynamo_kv_router;
pub trait PrefillLoadEstimator: Send + Sync {
    fn predict_prefill_duration(&self, batch_size: usize, effective_isl: usize,
        prefix: usize) -> anyhow::Result<std::time::Duration>;
}
pub mod common { pub mod perf_model {
    pub trait AisCallback: Send + Sync {
        fn predict_prefill(&self, batch_size: usize, effective_isl: usize,
            prefix: usize) -> anyhow::Result<f64>;
        fn predict_decode(&self, batch_size: usize, isl: usize,
            osl: usize) -> anyhow::Result<f64>;
    }
}}
mod callback;
"""


def check(source: Path, dynamo: Path) -> None:
    source, dynamo = source.resolve(), dynamo.resolve()
    callback = dynamo / SPECULATION_DEPTH_PATH
    patched = callback.read_bytes()
    if patched.count(SPECULATION_DEPTH_NEW.encode()) != 1:
        raise ValueError("callback must contain exactly the reviewed compatibility patch")
    original = patched.replace(SPECULATION_DEPTH_NEW.encode(), SPECULATION_DEPTH_OLD.encode())
    if hashlib.sha256(original).hexdigest() != SPECULATION_DEPTH_SHA256:
        raise ValueError("callback does not match the complete pinned Dynamo source")
    core = source / "crates/core"
    shape = re.search(
        r"pub (struct|enum) ForwardPassSpeculationConfig\b", (core / "src/perfmodel/fpm/config.rs").read_text()
    )
    if shape is None:
        raise ValueError("unrecognized speculation configuration API")
    # Validate every Python source file, so a different checkout's installed SDK
    # cannot silently supply the Python half of the selected Rust core.
    python_paths = site.getsitepackages()
    for package in ("aisimulate", "aisimulate_core"):
        installed = Path(importlib.import_module(package).__file__).resolve().parent
        expected = source / "python/aisimulate/src" / package
        for path in expected.rglob("*.py"):
            if (installed / path.relative_to(expected)).read_bytes() != path.read_bytes():
                raise ValueError(f"installed SDK differs from selected source: {path}")
        python_paths.append(str(installed.parent))
        print(f"SDK {package}: {installed}", flush=True)
    print(f"Callback: {callback}; sha256={hashlib.sha256(patched).hexdigest()}", flush=True)
    print(f"AIS core: {core}; speculation API={shape[1]}", flush=True)
    env = os.environ | {
        "PYO3_PYTHON": sys.executable,
        "PYTHONHOME": sys.base_prefix,
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
        "AIS_CALLBACK_GENERIC_CONFIG": str(int(shape[1] == "struct")),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="dynamo-callback-contract-") as directory:
        root = Path(directory)
        (root / "Cargo.toml").write_text(
            '[package]\nname = "dynamo-callback-contract"\nversion = "0.0.0"\nedition = "2024"\n'
            '[lib]\npath = "lib.rs"\n[features]\ndefault = ["ais-forward-pass"]\nais-forward-pass = []\n'
            "[dependencies]\n"
            f'aisimulate-core = {{ path = {json.dumps(str(core))}, features = ["embed-python"] }}\n'
            'pyo3 = { version = "0.23.4", features = ["auto-initialize", "abi3-py311"] }\n'
            'serde_json = "1"\nanyhow = "1"\n[profile.dev]\ndebug = 0\n[profile.test]\ndebug = 0\n'
        )
        shutil.copyfile(source / "Cargo.lock", root / "Cargo.lock")
        (root / "lib.rs").write_text(SCAFFOLD)
        contract = Path(__file__).resolve().with_name("dynamo_callback_contract.rs")
        (root / "callback.rs").write_bytes(
            patched + f"\n#[cfg(test)]\nmod consumer_contract {{ include!({json.dumps(str(contract))}); }}\n".encode()
        )
        manifest = ["--manifest-path", str(root / "Cargo.toml")]
        subprocess.run(
            ["cargo", "metadata", *manifest, "--offline", "--format-version", "1"],
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            ["cargo", "test", *manifest, "--offline", "--locked", "--lib", "--", "--test-threads=1"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dynamo", type=Path)
    args = parser.parse_args()
    check(args.source, args.dynamo)
