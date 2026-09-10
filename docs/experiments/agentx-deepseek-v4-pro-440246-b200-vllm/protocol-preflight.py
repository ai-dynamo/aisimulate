# SPDX-License-Identifier: Apache-2.0
"""Fail before model loading if plugins replace the native Rust IPC schema."""
import importlib.metadata
import json
import os
import sys

import vllm.entrypoints.cli.main  # Match the native benchmark entry point.
from vllm.plugins import load_general_plugins

load_general_plugins()

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs

fields = EngineCoreOutput.__struct_fields__
print(json.dumps({
    "plugins_allowlist": os.environ.get("VLLM_PLUGINS"),
    "installed_plugins": {
        ep.name: ep.value
        for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
    },
    "output_class": f"{EngineCoreOutput.__module__}.{EngineCoreOutput.__name__}",
    "output_fields": fields,
    "envelope_fields": EngineCoreOutputs.__struct_fields__,
    "omni_imported": "vllm_omni" in sys.modules,
}, indent=2), flush=True)
assert EngineCoreOutput.__module__ == "vllm.v1.engine"
assert len(fields) == 16 and fields[-1] == "new_sampling_mask", fields
assert len(EngineCoreOutputs.__struct_fields__) == 8
assert "vllm_omni" not in sys.modules
print("NATIVE_RUST_IPC_PREFLIGHT_PASS", flush=True)
