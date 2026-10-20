# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record which Triton fused-MoE kernel config a timed SGLang call used.

SGLang's ``try_get_optimal_moe_config`` (moe_runner/triton_utils/
fused_moe_triton_config.py @v0.5.20) first looks for a tuned JSON through
``get_moe_configs`` and falls back to ``get_default_config`` when none exists
for (E, N, device, dtype, block shape). The collector's direct path calls the
same two functions. This module observes those calls while a case is timed,
without changing which config is chosen, and appends one record per persisted
row to a JSONL sidecar next to the perf staging file. The perf row schema and
``kernel_source`` labels are unchanged.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

LOOKUP = "get_moe_configs"
DEFAULT = "get_default_config"


@contextmanager
def observe_triton_config_selection(namespaces):
    """Count tuned-config hits/misses and default-config calls.

    ``namespaces`` are modules or globals dicts that hold the two SGLang
    functions; each is wrapped for the duration of the block and restored.
    """
    observed = {"tuned_lookup_hits": 0, "tuned_lookup_misses": 0, "default_config_calls": 0}
    patched = []

    def _get(ns, name):
        return ns.get(name) if isinstance(ns, dict) else getattr(ns, name, None)

    def _set(ns, name, value):
        if isinstance(ns, dict):
            ns[name] = value
        else:
            setattr(ns, name, value)

    def wrap_lookup(original):
        def lookup(*args, **kwargs):
            result = original(*args, **kwargs)
            observed["tuned_lookup_hits" if result else "tuned_lookup_misses"] += 1
            return result

        return lookup

    def wrap_default(original):
        def default(*args, **kwargs):
            observed["default_config_calls"] += 1
            return original(*args, **kwargs)

        return default

    seen = set()
    for ns in namespaces:
        for name, wrap in ((LOOKUP, wrap_lookup), (DEFAULT, wrap_default)):
            original = _get(ns, name)
            if original is None or (id(ns), name) in seen:
                continue
            seen.add((id(ns), name))
            patched.append((ns, name, original))
            _set(ns, name, wrap(original))
    try:
        yield observed
    finally:
        for ns, name, original in reversed(patched):
            _set(ns, name, original)


def config_source(observed: dict) -> str:
    """``default``, ``tuned``, ``mixed`` or ``not_observed`` (no lookup ran)."""
    default = observed["default_config_calls"] > 0
    tuned = observed["tuned_lookup_hits"] > 0
    if default and tuned:
        return "mixed"
    if default:
        return "default"
    if tuned:
        return "tuned"
    return "not_observed"


def sidecar_path(perf_filename: str) -> Path:
    path = Path(perf_filename)
    return path.with_name(f"{path.stem}.triton_config.jsonl")


def append_record(perf_filename: str, row: dict, kernel_source: str, observed: dict) -> dict:
    record = {
        **row,
        "kernel_source": kernel_source,
        "triton_config_source": config_source(observed),
        **observed,
    }
    with sidecar_path(perf_filename).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record
