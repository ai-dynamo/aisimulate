# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind an excluded native NONE warmup to actual operation and setup calls.

Original read-only adapter over vllm-project/vllm
ced6857afa0ea7b2e3f0846a62e1394e90f15607, vllm/v1/worker/gpu/model_runner.py
(Apache-2.0). CPU activity categories follow Kineto
094d3c1d072362d0a919a77299459eee94f97931, libkineto/include/ActivityType.h
(BSD). No compute implementation is copied; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

from collector.glm53flash_graph_nodes import (
    EXECUTION_RANGE,
    _compose_execution,
    bind_native_eager_activity,
    trace_forward_identity,
)

NONE_EXECUTION_RANGE = "aisim.glm53/native_none_metadata_to_logits"
NONE_MODEL_RANGE = "aisim.glm53/native_none_raw_model"
NONE_SETUP_RANGES = {
    "prepared_inputs_to_raw_model_entry": "aisim.glm53/native_none_before_model",
    "raw_model_return_to_logits_entry": "aisim.glm53/native_none_before_logits",
}
OPERATION_RANGE_PREFIX = "aisim.glm53/"
NONE_MEASUREMENT_CONTRACT = "native_serving_none_events_v1"


def _interval(row, *, positive=True):
    start, duration = row.get("ts"), row.get("dur")
    if (
        row.get("ph") != "X"
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, duration))
        or duration < 0
        or (positive and duration == 0)
        or any(type(row.get(key)) is not int for key in ("pid", "tid"))
    ):
        raise ValueError("native NONE scope lacks a complete actual interval/thread")
    return start, start + duration


def _inside(row, scope):
    begin, end = _interval(row, positive=False)
    left, right = _interval(scope)
    return all(row[key] == scope[key] for key in ("pid", "tid")) and left <= begin <= end <= right


def _call_inventory(calls, expected_names):
    names = list(expected_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("native NONE operation manifest is empty or repeats a physical unit")
    if len(calls) != len(names) or {row.get("operation") for row in calls} != set(names):
        raise ValueError("native NONE lacks its complete unique actual operation calls")
    result = {}
    for row in calls:
        if (
            row.get("completed") is not True
            or not isinstance(row.get("source"), str)
            or not row["source"]
            or not isinstance(row.get("included_sources"), list)
            or not isinstance(row.get("excluded_collective_sources"), list)
            or any(
                not isinstance(value, str) or not value
                for value in row["included_sources"] + row["excluded_collective_sources"]
            )
            or row.get("parent_operation") not in (None, *names)
            or row.get("parent_operation") == row["operation"]
        ):
            raise ValueError("native NONE call lacks its original completed source witness")
        result[row["operation"]] = row
    for name, row in result.items():
        parent = row.get("parent_operation")
        if parent is not None:
            outer = result[parent]
            if (
                outer.get("parent_operation") is not None
                or outer["excluded_collective_sources"].count(row["source"]) != 1
            ):
                raise ValueError("native NONE nested call lacks an exact synchronous collective source")
        if sorted(row["excluded_collective_sources"]) != sorted(
            child["source"] for child in result.values() if child.get("parent_operation") == name
        ):
            raise ValueError("native NONE collective subtraction differs from actual nested source calls")
    return result


class NativeNoneExecution:
    """Persist the excluded warmup trace, stopping before native sampling.

    Only a calibration observer creates this helper. It uses that observer's
    existing profiler instance and never starts a second CUPTI subscriber.
    """

    def __init__(self, observer, output):
        if observer.profile_callback is not None:
            raise RuntimeError("native NONE observer already owns a profiler callback")
        self.observer, self.output = observer, Path(output)
        self.active = None
        observer.profile_callback = self.profile_finished

    def _open(self, name):
        if not self.active["profiled"]:
            return
        scope = self.observer.torch.profiler.record_function(name)
        scope.__enter__()
        self.active["scopes"].append(scope)

    def _close(self):
        if self.active["profiled"]:
            self.active["scopes"].pop().__exit__(None, None, None)

    def begin(self, record):
        if self.active is not None or self.observer.workload is None:
            raise RuntimeError("native NONE requires one fresh actual observer workload")
        profiled = record["repetition"] == 4
        if (
            self.observer.workload.sample != record["repetition"]
            or (self.observer.profiler is not None) != profiled
            or (record["sampling_role"] == "warmup") != (record["repetition"] < 5)
            or not 0 <= record["repetition"] < 15
        ):
            raise RuntimeError("native NONE profiler must observe only the fifth excluded warmup")
        self.active = {"record": record, "profiled": profiled, "phase": "before_model", "scopes": []}
        record["profiled"] = profiled
        self._open(NONE_EXECUTION_RANGE)
        self._open(NONE_SETUP_RANGES["prepared_inputs_to_raw_model_entry"])

    def boundary(self, next_phase):
        active = self.active
        expected = {"model": "before_model", "before_logits": "model", "logits": "before_logits"}
        if active is None or active["phase"] != expected.get(next_phase):
            raise RuntimeError("native NONE source boundary was omitted or repeated")
        self._close()
        if next_phase == "model":
            self._open(NONE_MODEL_RANGE)
        elif next_phase == "before_logits":
            self._open(NONE_SETUP_RANGES["raw_model_return_to_logits_entry"])
        active["phase"] = next_phase

    def end_logits(self):
        if self.active is None or self.active["phase"] != "logits":
            raise RuntimeError("native NONE cannot stop profiling before actual logits completion")
        self._close()
        self.active["phase"] = "complete"
        # Stop here, before sampling/readback can contribute unrelated GPU
        # activity. Observer.end later retains its original synchronization.
        self.observer._finish_profile()

    def _save_trace(self, profiler, calls, *, failed):
        record = self.active["record"]
        name = f"none-profile-rank-{record['tp_rank']}-forward-{record['invocation']}.json"
        if failed:
            name = "failed-" + name
        path = self.output / name
        if path.exists():
            raise RuntimeError("native NONE trace cannot overwrite an original forward")
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        trace["aisim_native_forward"] = trace_forward_identity(record)
        trace["aisim_native_none"] = {
            "measurement_contract": NONE_MEASUREMENT_CONTRACT,
            "model_identity_sha256": record["serving_none_model_sha256"],
            "source_boundary": "GPUModelRunner.prepare_inputs_return_to_compute_logits",
            "native_calls": calls,
            "failed": failed,
        }
        path.write_text(json.dumps(trace))
        return path, trace

    def profile_finished(self, profiler, workload, calls):
        if (
            self.active is None
            or not self.active["profiled"]
            or self.active["phase"] != "complete"
            or workload.invocation != self.active["record"]["invocation"]
        ):
            raise RuntimeError("native NONE warmup profile differs from its actual completed forward")
        # Durable original trace precedes strict derivation, including failures.
        path, trace = self._save_trace(profiler, calls, failed=False)
        binding = bind_none_execution(trace["traceEvents"], calls, self.observer._entries[workload.phase])
        self.active["record"]["native_none_profile"] = {
            "trace_file": path.name,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "binding": binding,
        }

    def complete(self, record):
        if (
            self.active is None
            or self.active["record"] is not record
            or self.active["phase"] != "complete"
            or self.observer.profiler is not None
            or self.active["profiled"] != ("native_none_profile" in record)
        ):
            raise RuntimeError("native NONE forward lacks its original completed observation")
        self.active = None

    def abort(self, error):
        if self.active is None:
            return
        active = self.active
        while active["scopes"]:
            active["scopes"].pop().__exit__(type(error), error, error.__traceback__)
        profiler, self.observer.profiler = self.observer.profiler, None
        if profiler is not None:
            profiler.stop()
            self._save_trace(profiler, self.observer.native_call_inventory(), failed=True)
        self.active = None


def bind_none_execution(events, calls, expected_names):
    """Recompute warmup ownership; never substitute activity time for events.

    Module timings and the two setup timings are separate directly observed
    CUDA-event intervals. This trace proves the actual calls and dispatch,
    but does not grant interpolation, runtime admission or timing acceptance.
    """
    inventory = _call_inventory(calls, expected_names)
    expected = {
        NONE_EXECUTION_RANGE,
        NONE_MODEL_RANGE,
        *NONE_SETUP_RANGES.values(),
        *(OPERATION_RANGE_PREFIX + name for name in inventory),
    }
    scopes = {}
    for row in events:
        name = row.get("name", "")
        if row.get("cat") != "user_annotation" or not name.startswith(OPERATION_RANGE_PREFIX):
            continue
        if name not in expected or name in scopes:
            raise ValueError("native NONE CPU ownership is unknown or duplicated")
        _interval(row)
        scopes[name] = row
    if set(scopes) != expected or "logits" not in inventory:
        raise ValueError("native NONE lacks its complete source-bound CPU ranges")
    region, model = scopes[NONE_EXECUTION_RANGE], scopes[NONE_MODEL_RANGE]
    logits = scopes[OPERATION_RANGE_PREFIX + "logits"]
    before_model, before_logits = (scopes[value] for value in NONE_SETUP_RANGES.values())
    ordered = [before_model, model, before_logits, logits]
    if any(not _inside(row, region) for row in ordered) or any(
        _interval(left)[1] > _interval(right)[0] for left, right in pairwise(ordered)
    ):
        raise ValueError("native NONE changed its prepared/model/logits boundary order")

    operation_scopes = {name: scopes[OPERATION_RANGE_PREFIX + name] for name in inventory}
    for name, row in operation_scopes.items():
        if name != "logits" and not _inside(row, model):
            raise ValueError("native NONE hidden-state operation is outside the actual model call")
        parent = inventory[name].get("parent_operation")
        if parent is not None and not _inside(row, operation_scopes[parent]):
            raise ValueError("native NONE collective is outside its witnessed enclosing module")
    for index, (name, row) in enumerate(operation_scopes.items()):
        left, right = _interval(row)
        for other, interval in list(operation_scopes.items())[index + 1 :]:
            start, end = _interval(interval)
            if (
                start < right
                and end > left
                and not (
                    inventory[name].get("parent_operation") == other or inventory[other].get("parent_operation") == name
                )
            ):
                raise ValueError("native NONE operation scopes overlap without a native collective relation")

    normalized = [dict(row, name=EXECUTION_RANGE) if row is region else row for row in events]
    checked = bind_native_eager_activity(normalized)
    candidates = list(operation_scopes.items()) + [(name, scopes[value]) for name, value in NONE_SETUP_RANGES.items()]
    ownership, controls = {}, []
    for row in events:
        if row.get("cat") not in ("cuda_runtime", "cuda_driver") or not _inside(row, region):
            continue
        start, end = _interval(row, positive=False)
        containing = []
        for name, scope in candidates:
            left, right = _interval(scope)
            if start == end and start in (left, right):
                raise ValueError("zero-duration CUDA call at native NONE boundary has ambiguous ownership")
            if start < right and end > left:
                if not _inside(row, scope):
                    raise ValueError("native NONE CUDA call straddles source ownership boundaries")
                containing.append(name)
        if len(containing) > 1:
            children = [name for name in containing if inventory.get(name, {}).get("parent_operation") in containing]
            if len(containing) != 2 or len(children) != 1:
                raise ValueError("native NONE CUDA call has ambiguous source ownership")
            containing = children
        correlation = row["args"]["correlation"]
        if containing:
            ownership[correlation] = containing[0]
        else:
            controls.append(correlation)

    activities = checked["activities"]
    for row in activities:
        owner = ownership.get(row["launch_correlation"])
        if owner is None:
            raise ValueError("native NONE device work is outside every original operation/setup scope")
        setup = owner in NONE_SETUP_RANGES
        row["operation"] = "native_graph_setup" if setup else owner
        row["source_boundary"] = owner if setup else inventory[owner]["source"]
        row["setup_boundary"] = owner if setup else None
    result = _compose_execution(None, region, activities)
    result.update(
        measurement_contract=NONE_MEASUREMENT_CONTRACT,
        native_calls=calls,
        source_scopes=scopes,
        api_ownership=[
            {"correlation": correlation, "owner": owner} for correlation, owner in sorted(ownership.items())
        ],
        unowned_control_correlations=sorted(controls),
        operation_activity_indices={
            name: [row["activity_index"] for row in activities if row["operation"] == name] for name in inventory
        },
        setup_activity_indices={
            name: [row["activity_index"] for row in activities if row["setup_boundary"] == name]
            for name in NONE_SETUP_RANGES
        },
        timing_method="separate_native_cuda_event_intervals_not_profiler_activity_time",
        dispatch_interpolation_admitted=False,
    )
    return result
