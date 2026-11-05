# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce a partial-data native scenario smoke; never publish qualification."""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hf-snapshot", type=Path, required=True)
parser.add_argument("--configuration", required=True)
parser.add_argument("--seconds", type=float, default=180)
parser.add_argument("--native-library", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.native_library:
    spec = importlib.util.spec_from_file_location("aisimulate._runtime", args.native_library)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

from scripts.fpm_accuracy.hf.dataset import HfDataset, _hub_cache_blob_roots
from scripts.fpm_accuracy.hf.overrides import load_overrides
from scripts.fpm_accuracy.scenario_contract import SCHEMA_VERSION, validate_scenario_row
from scripts.fpm_accuracy.scenarios import evaluate_scenario_case

root = args.hf_snapshot.resolve()
dataset = HfDataset(
    root,
    repo_id="nvidia/aisimulate-fpm-dataset",
    revision=root.name,
    allowed_file_roots=_hub_cache_blob_roots(root, root.name),
    overrides=load_overrides(None),
    overrides_sha256=None,
)
case = dataset.measurement_case(args.configuration)
anchor = case.observations[0].event_time_ms
if anchor is None or str(case.ordering) != "chronological":
    raise ValueError("This smoke requires source event timestamps.")
case = replace(
    case, observations=tuple(x for x in case.observations if x.event_time_ms <= anchor + args.seconds * 1000)
)
start = perf_counter()
row = evaluate_scenario_case(case)
elapsed = perf_counter() - start
counts = Counter(str(x.workload_kind) for x in case.observations)
# Validate exact scenario populations independently of any overview predictor.
source = {
    **row,
    "results": {"regression": {"metrics": {k: {"measured_count": counts[k]} for k in ("prefill", "decode", "mixed")}}},
}
validate_scenario_row(row, source, SCHEMA_VERSION)
serialized_bytes = len(json.dumps(row, allow_nan=False, sort_keys=True).encode())
for methods in [
    *row["results"].values(),
    *(methods for split in row["split_results"].values() for methods in split.values()),
]:
    for result in methods.values():
        if result:
            result.pop("offline_model", None)
            result.pop("online_model", None)
document = {
    "qualification": "SMOKE_ONLY: partial source population; not a publication campaign",
    "hf_repo": "nvidia/aisimulate-fpm-dataset",
    "hf_revision": root.name,
    "native_sha256": hashlib.sha256(args.native_library.read_bytes()).hexdigest() if args.native_library else None,
    "window_seconds": args.seconds,
    "elapsed_ms": case.observations[-1].event_time_ms - anchor,
    "evaluation_wall_seconds": elapsed,
    "complete_row_bytes_before_diagnostic_removal": serialized_bytes,
    "scenario_contract_validated": True,
    "row": row,
}
args.output.write_text(json.dumps(document, allow_nan=False, indent=2) + "\n")
print(f"Validated {len(case.observations)} observations in {elapsed:.2f}s; complete row {serialized_bytes} bytes")
