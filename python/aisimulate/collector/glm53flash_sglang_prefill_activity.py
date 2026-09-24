# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind original SGLang prefill calls to their CUDA API and GPU activities.

Original observer over sgl-project/sglang at
94602c9c2b7cbdb8efd5c52802dac6a1c180089e, srt/models/glm5_next.py,
srt/managers/mm_utils.py and srt/utils/common.py (Apache-2.0). No native
implementation is copied. CPU categories follow Kineto
094d3c1d072362d0a919a77299459eee94f97931, include/ActivityType.h (BSD).
See THIRD_PARTY_NOTICES.md for upstream attribution.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

from collector.glm53flash_contract import canonical_json, sha256_json
from collector.glm53flash_graph_nodes import (
    EXECUTION_RANGE,
    _compose_execution,
    bind_native_eager_activity,
    trace_forward_identity,
)
from collector.glm53flash_vllm_none_activity import _inside, _interval

PREFILL_RANGE = "aisim.glm53/sglang_prefill_embedding_to_logits"
SETUP_RANGE = "aisim.glm53/sglang_prefill_bump_allocator"
UNIT_PREFIX = "aisim.glm53/prefill_unit/"
METHOD = "native_sglang_prefill_events_v1"
SOURCE_PINS = {
    "srt/models/glm5_next.py": "12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2",
    "srt/utils/common.py": "52eedc9338c5d2565434d265858b5d47bcc67156d38353a5b218e91df7620e45",
    "srt/managers/mm_utils.py": "a5e34a1af72faadf610feb7bf20ae41f3a50b2bbb821e38e6b091e43e4ad46c2",
}
MODEL_CONTRACT = {
    "model_class": "sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration",
    "language_model_class": "sglang.srt.models.glm5_next.Glm5NextModel",
    "start_layer": 0,
    "end_layer": 45,
    "pp_size": 1,
    "dp_size": 1,
    "ep_size": 1,
    "text_only": True,
    "can_run_tbo": False,
    "dflash_capture": False,
    "layers_to_capture": [],
    "capture_aux_hidden_states": False,
    "input_embeds_buffer": False,
    "gemm_output_zero_allocator_size": 0,
    "bump_allocator_calls": 1,
    "bump_allocator_elements": 90,
    "bump_allocator_dtype": "torch.float32",
}


def _calls(calls, expected_names):
    names = set(expected_names)
    if not names or len(names) != len(expected_names):
        raise ValueError("native prefill requires a unique complete physical manifest")
    by_scope, by_name = {}, defaultdict(list)
    for call in calls:
        name, scope = call.get("operation"), call.get("scope_name")
        if (
            name not in names
            or not isinstance(scope, str)
            or not scope.startswith(UNIT_PREFIX)
            or scope in by_scope
            or call.get("completed") is not True
            or not isinstance(call.get("source"), str)
            or not call["source"]
            or any(
                not isinstance(call.get(key), list) or any(not isinstance(item, str) or not item for item in call[key])
                for key in ("included_sources", "excluded_collective_sources")
            )
        ):
            raise ValueError("native prefill call lacks unique completed source/scope ownership")
        by_scope[scope] = call
        by_name[name].append(call)
    if set(by_name) != names:
        raise ValueError("native prefill call inventory omits a physical operation")
    if set(by_scope) != {
        f"{UNIT_PREFIX}{index}/{call['operation']}"
        for index, call in enumerate(sorted(calls, key=lambda call: int(call["scope_name"].split("/")[-2])))
    }:
        raise ValueError("native prefill scope sequence is incomplete or changed")
    for name, parts in by_name.items():
        # Only the existing source-qualified sparse MLA latent projection can
        # be hoisted into a second disjoint part of one physical operation.
        if len(parts) > 1 and (
            not name.startswith("attention_")
            or not name.removeprefix("attention_").isdigit()
            or int(name.removeprefix("attention_")) % 4 != 3
            or len(parts) != 2
            or len({part["source"] for part in parts}) != 2
        ):
            raise ValueError("native prefill repeats an unqualified physical call")
    for scope, call in by_scope.items():
        parent = call.get("parent_scope_name")
        if parent is not None:
            outer = by_scope.get(parent)
            if (
                outer is None
                or outer.get("parent_scope_name") is not None
                or call.get("parent_operation") != outer["operation"]
                or call["operation"] == outer["operation"]
            ):
                raise ValueError("native prefill collective lacks its actual enclosing source part")
        elif call.get("parent_operation") is not None:
            raise ValueError("native prefill parent operation has no exact parent scope")
        children = [child["source"] for child in calls if child.get("parent_scope_name") == scope]
        if Counter(children) != Counter(call["excluded_collective_sources"]):
            raise ValueError("native prefill collective exclusions differ from actual nested calls")
    return by_scope


def bind_prefill_activity(events, calls, expected_names):
    """Strict ownership proof only; CUDA-event timings are supplied separately."""
    inventory = _calls(calls, expected_names)
    expected = {PREFILL_RANGE, SETUP_RANGE, *inventory}
    scopes = {}
    for row in events:
        name = row.get("name", "")
        if row.get("cat") != "user_annotation" or not name.startswith("aisim.glm53/"):
            continue
        if name not in expected or name in scopes:
            raise ValueError("native prefill CPU ownership range is unknown or duplicated")
        _interval(row)
        scopes[name] = row
    if set(scopes) != expected:
        raise ValueError("native prefill lacks complete source-bound CPU ranges")
    region = scopes[PREFILL_RANGE]
    parts = {scope: row for scope, row in scopes.items() if scope != PREFILL_RANGE}
    for scope, row in parts.items():
        if not _inside(row, region):
            raise ValueError("native prefill unit/setup is outside its exact model thread/range")
        parent = inventory.get(scope, {}).get("parent_scope_name")
        if parent is not None and not _inside(row, scopes[parent]):
            raise ValueError("native prefill collective is outside its actual parent call")
    for index, (scope, row) in enumerate(parts.items()):
        start, end = _interval(row)
        for other, value in list(parts.items())[index + 1 :]:
            left, right = _interval(value)
            if (
                start < right
                and end > left
                and not (
                    inventory.get(scope, {}).get("parent_scope_name") == other
                    or inventory.get(other, {}).get("parent_scope_name") == scope
                )
            ):
                raise ValueError("native prefill intervals overlap without an exact collective relation")
    normalized = [dict(row, name=EXECUTION_RANGE) if row is region else row for row in events]
    checked = bind_native_eager_activity(normalized)
    owners, controls = {}, []
    for row in events:
        if row.get("cat") not in ("cuda_runtime", "cuda_driver") or not _inside(row, region):
            continue
        start, end = _interval(row, positive=False)
        containing = []
        for scope, value in parts.items():
            left, right = _interval(value)
            if start == end and start in (left, right):
                raise ValueError("zero-duration CUDA API has ambiguous prefill boundary ownership")
            if start < right and end > left:
                if not _inside(row, value):
                    raise ValueError("CUDA API straddles original prefill source calls")
                containing.append(scope)
        if len(containing) > 1:
            child = [scope for scope in containing if inventory.get(scope, {}).get("parent_scope_name") in containing]
            if len(containing) != 2 or len(child) != 1:
                raise ValueError("CUDA API has ambiguous native prefill ownership")
            containing = child
        correlation = row["args"]["correlation"]
        if containing:
            owners[correlation] = containing[0]
        else:
            controls.append(correlation)
    activities = checked["activities"]
    unowned = [row for row in activities if row["launch_correlation"] not in owners]
    if unowned:
        error = ValueError("native prefill has unowned device activities; no accepted table")
        error.unowned_activities = unowned
        raise error
    for row in activities:
        scope = owners[row["launch_correlation"]]
        row["operation"] = "native_graph_setup" if scope == SETUP_RANGE else inventory[scope]["operation"]
        row["source_scope"] = scope
        row["source_boundary"] = (
            "sglang.srt.utils.common.BumpAllocator.__init__" if scope == SETUP_RANGE else inventory[scope]["source"]
        )
    result = _compose_execution(None, region, activities)
    result.update(
        measurement_contract=METHOD,
        native_calls=calls,
        source_scopes=scopes,
        api_ownership=[{"correlation": key, "scope": value} for key, value in sorted(owners.items())],
        unowned_control_correlations=sorted(controls),
        unowned_device_activities=[],
        operation_activity_indices={
            name: [row["activity_index"] for row in activities if row["operation"] == name]
            for name in [*expected_names, "native_graph_setup"]
        },
        contribution_counts={**dict(Counter(call["operation"] for call in calls)), "native_graph_setup": 1},
        timing_method="original_native_cuda_event_intervals_not_profiler_activity_time",
        dispatch_interpolation_admitted=False,
    )
    return result


def dispatch_signatures(binding):
    """Preserve full names and launch metadata; these are not interpolation keys."""
    return {
        name: sorted(
            canonical_json({"activity": row["activity"], **row["fingerprint"]})
            for row in binding["activities"]
            if row["operation"] == name
        )
        for name in binding["operation_activity_indices"]
    }


def _native_function(method, package, relative):
    function = inspect.unwrap(getattr(method, "__func__", method))
    if Path(inspect.getsourcefile(function) or "").resolve() != (package / relative).resolve():
        raise RuntimeError("native prefill callable does not belong to its exact loaded source")
    return function


def model_identity(model, runner):
    """Read actual loaded objects and source files before observing a request."""
    import sglang
    from sglang.srt.models import glm5_next

    package = Path(sglang.__file__).resolve().parent
    actual = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
    if actual != SOURCE_PINS:
        raise RuntimeError("native prefill loaded source differs from the reviewed model/allocator boundary")
    language = model.model
    if any(
        type(getattr(runner.server_args, key)) is not int or getattr(runner.server_args, key) != 1
        for key in ("pp_size", "dp_size", "ep_size")
    ):
        raise RuntimeError("native prefill requires actual pure tensor parallel topology")
    if type(model) is not glm5_next.Glm5NextForConditionalGeneration or type(language) is not glm5_next.Glm5NextModel:
        raise RuntimeError("native prefill requires exact uncompiled GLM native model classes")
    for owner in (model, language):
        _native_function(owner.forward, package, "srt/models/glm5_next.py")
    _native_function(glm5_next.general_mm_embed_routine, package, "srt/managers/mm_utils.py")
    _native_function(glm5_next.BumpAllocator.__init__, package, "srt/utils/common.py")
    if (
        language.start_layer != 0
        or language.end_layer != 45
        or not model.pp_group.is_first_rank
        or not model.pp_group.is_last_rank
        or not language.pp_group.is_first_rank
        or not language.pp_group.is_last_rank
        or language.gemm_output_zero_allocator_size != 0
        or language.dflash_capture is not False
        or language.layers_to_capture != []
        or model.capture_aux_hidden_states is not False
        or model.is_mrope_enabled is not False
    ):
        raise RuntimeError("native prefill model has unqualified conditional runtime work")
    return {"source_pins": actual, "native_model_contract": dict(MODEL_CONTRACT)}


def validate_forward_batch(batch):
    if (
        batch.can_run_tbo
        or batch.contains_mm_inputs()
        or batch.input_embeds is not None
        or not batch.forward_mode.is_extend()
        or batch.forward_mode.is_mixed()
    ):
        raise RuntimeError("native prefill has unqualified actual TBO/multimodal/input-copy/phase state")


class NativeSglangPrefillExecution:
    """Use the observer's fifth-warmup profiler and native setup events once."""

    def __init__(self, runner, observer, output):
        from sglang.srt.models import glm5_next

        if observer.profile_callback is not None or observer.profile_scope_ids:
            raise RuntimeError("native prefill observer already has a profile owner")
        model = runner.model
        self.identity = model_identity(model, runner)
        self.observer, self.output, self.active = observer, Path(output), None
        self.model, self.allocator = model, glm5_next.BumpAllocator
        self.original_model, self.original_allocator = model.forward, self.allocator.__init__
        self.model_signature, self.allocator_signature = (
            inspect.signature(self.original_model),
            inspect.signature(self.original_allocator),
        )
        observer.profile_scope_ids = True
        observer.defer_profile_start = True
        observer.profile_callback = self.profile_finished
        observer.enable_prefill_event_pool()
        self.restorations = []
        self._install()

    def _install(self):
        @functools.wraps(self.original_allocator)
        def allocate(instance, *args, **kwargs):
            active = self.active
            if active is None or not active["inside_model"]:
                return self.original_allocator(instance, *args, **kwargs)
            if (
                threading.get_ident() != active["thread"]
                or active["setup"] is not None
                or self.observer.active_interval is not None
            ):
                raise RuntimeError("native prefill allocator is repeated or belongs to another model thread")
            values = self.allocator_signature.bind(instance, *args, **kwargs).arguments
            if (
                values["buffer_size"] != 90
                or values["dtype"] is not self.observer.torch.float32
                or str(values["device"]).split(":")[0] != "cuda"
            ):
                raise RuntimeError("native prefill allocator arguments differ from the actual model contract")
            torch = self.observer.torch
            stream = torch.cuda.current_stream()
            start, end = self.observer.event_pool.pairs["setup"]
            proof = {
                "buffer_size": values["buffer_size"],
                "dtype": str(values["dtype"]),
                "device": str(values["device"]),
                "source": "sglang.srt.utils.common.BumpAllocator.__init__",
            }
            active["setup"] = {"start": start, "end": end, "proof": proof, "completed": False}
            start.record(stream)
            with self._scope(SETUP_RANGE):
                result = self.original_allocator(instance, *args, **kwargs)
            end.record(stream)
            if torch.cuda.current_stream() != stream:
                raise RuntimeError("native prefill allocator changed its current stream")
            active["setup"]["completed"] = True
            return result

        @functools.wraps(self.original_model)
        def forward(*args, **kwargs):
            active = self.active
            if active is None:
                return self.original_model(*args, **kwargs)
            if active["inside_model"] or active["completed"]:
                raise RuntimeError("native prefill original model must execute exactly once")
            values = self.model_signature.bind(*args, **kwargs).arguments
            batch = values["forward_batch"]
            validate_forward_batch(batch)
            active["inside_model"], active["thread"] = True, threading.get_ident()
            if self.observer.profiler is not None:
                self.observer.profiler.start()
                active["profile_started"] = True
            torch = self.observer.torch
            stream = torch.cuda.current_stream()
            whole_start, whole_end = self.observer.event_pool.pairs["whole"]
            active["whole"] = whole_start, whole_end
            whole_start.record(stream)
            try:
                with self._scope(PREFILL_RANGE):
                    result = self.original_model(*args, **kwargs)
                whole_end.record(stream)
                if torch.cuda.current_stream() != stream:
                    raise RuntimeError("native prefill model changed its current stream")
                if active["setup"] is None or active["setup"]["completed"] is not True:
                    raise RuntimeError("native prefill omitted its actual model allocator call")
                active["calls"] = self.observer.native_call_inventory()
                active["completed"] = True
                self.observer._finish_profile()
                return result
            finally:
                active["inside_model"] = False

        for owner, name, replacement in ((self.allocator, "__init__", allocate), (self.model, "forward", forward)):
            self.restorations.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

    def _scope(self, name):
        return (
            self.observer.torch.profiler.record_function(name) if self.observer.profiler is not None else nullcontext()
        )

    def begin(self, record):
        if self.active is not None or self.observer.workload is None or record["phase"] != "context":
            raise RuntimeError("native prefill requires one fresh actual context workload")
        sample = record["repetition"]
        if not 0 <= sample < 15 or (self.observer.profiler is not None) != (sample == 4):
            raise RuntimeError("native prefill profiler must be the fifth excluded warmup only")
        record["native_prefill_model_sha256"] = sha256_json(self.identity)
        self.active = {
            "record": record,
            "setup": None,
            "inside_model": False,
            "completed": False,
            "profile_started": False,
        }

    def _save(self, profiler, calls, *, failed):
        record = self.active["record"]
        name = f"prefill-profile-rank-{record['tp_rank']}-forward-{record['invocation']}.json"
        path = self.output / (("failed-" if failed else "") + name)
        if path.exists():
            raise RuntimeError("native prefill cannot overwrite original trace evidence")
        profiler.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        trace["aisim_native_forward"] = trace_forward_identity(record)
        trace["aisim_native_prefill"] = {
            "measurement_contract": METHOD,
            "model_identity_sha256": sha256_json(self.identity),
            "native_calls": calls,
            "setup": None if self.active["setup"] is None else self.active["setup"]["proof"],
            "failed": failed,
        }
        path.write_text(json.dumps(trace))
        return path, trace

    def profile_finished(self, profiler, workload, calls):
        if (
            self.active is None
            or not self.active["completed"]
            or workload.invocation != self.active["record"]["invocation"]
            or workload.sample != 4
        ):
            raise RuntimeError("native prefill profile is not its exact completed excluded warmup")
        path, trace = self._save(profiler, calls, failed=False)
        try:
            binding = bind_prefill_activity(trace["traceEvents"], calls, list(self.observer._entries["context"]))
        except BaseException as error:
            path.with_suffix(".failure.json").write_text(
                json.dumps(
                    {
                        "error": str(error),
                        "unowned_device_activities": getattr(error, "unowned_activities", None),
                        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            )
            raise
        self.active["record"]["native_prefill_profile"] = {
            "trace_file": path.name,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "binding": binding,
        }
        dispatch = dispatch_signatures(binding)
        dispatch.pop("native_graph_setup")
        return dispatch

    def complete(self, record):
        if (
            self.active is None
            or self.active["record"] is not record
            or not self.active["completed"]
            or self.observer.profiler is not None
            or not self.observer.event_pool.read_complete
        ):
            raise RuntimeError("native prefill has no complete original model/setup observation")
        setup = self.active["setup"]
        latency = setup["start"].elapsed_time(setup["end"])
        record["native_prefill_setup"] = {
            **setup["proof"],
            "completed": True,
            "latency": latency,
            "contribution_count": 1,
        }
        record["native_prefill_measurement_contract"] = METHOD
        record["native_prefill_calls"] = self.active["calls"]
        start, end = self.active["whole"]
        record.update(whole_forward_gpu_ms=start.elapsed_time(end), whole_forward_boundary="embedding_to_logits_gpu_v1")
        record["native_prefill_event_pool"] = self.observer.event_pool.finish()
        self.active = None

    def abort(self, error):
        if self.active is None:
            return
        profiler, self.observer.profiler = self.observer.profiler, None
        try:
            if profiler is not None and self.active["profile_started"]:
                profiler.stop()
                self._save(profiler, self.observer.native_call_inventory(), failed=True)
        finally:
            self.active = None
            self.close()
            self.observer.close()

    def close(self):
        self.observer.event_pool.close()
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)
        self.restorations.clear()
