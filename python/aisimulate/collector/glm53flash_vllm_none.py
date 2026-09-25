# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a serving NONE observation to the actual uncompiled native model.

Original read-only adapter over vllm-project/vllm at immutable revision
ced6857afa0ea7b2e3f0846a62e1394e90f15607 (Apache-2.0):
vllm/models/glm5next/nvidia/model.py and model_executor/models/glm4_1v.py.
No native compute implementation is copied. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

SOURCE_PINS = {
    "models/glm5next/nvidia/model.py": "d7353ea0c5708e40d65364b6a32ab63372aafbbebbf954722d635a125252a813",
    "model_executor/models/glm4_1v.py": "32bd289bf147daa27ad38e07306c30fd70707c51f3cea350f4a2154a66ea248a",
}


def validate_model_receipt(receipt):
    """Check persisted class/method proof before joining measured call evidence."""
    glm5 = "vllm.models.glm5next.nvidia.model."
    glm4 = "vllm.model_executor.models.glm4_1v.Glm4vForConditionalGeneration."
    expected = {
        "schema_name": "glm53flash_native_serving_none_model",
        "schema_version": 1,
        "source_pins": SOURCE_PINS,
        "classes": [
            glm5 + name for name in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM", "Glm5NextModel")
        ],
        "methods": [
            {"method": glm4 + "forward", "source": "model_executor/models/glm4_1v.py"},
            {"method": glm4 + "compute_logits", "source": "model_executor/models/glm4_1v.py"},
            {"method": glm5 + "Glm5NextForCausalLM.compute_logits", "source": "models/glm5next/nvidia/model.py"},
            {"method": glm5 + "Glm5NextModel.forward", "source": "models/glm5next/nvidia/model.py"},
        ],
        "compiled_model": False,
        "admission": "DIAGNOSTIC_ONLY_NATIVE_CALLS_STILL_REQUIRED",
    }
    if receipt != expected:
        raise ValueError("native NONE model proof differs from the actual uncompiled source contract")
    return receipt


class NativeNoneModelWitness:
    """Retain exact original objects before post-initialization observation.

    A descriptor named NONE alone does not establish Python operation calls.
    The runtime must also witness the original forward once and require the
    complete observed operation inventory. This class does not admit a table.
    """

    def __init__(self, model):
        import torch
        import vllm
        from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration
        from vllm.models.glm5next.nvidia.model import (
            Glm5NextForCausalLM,
            Glm5NextForConditionalGeneration,
            Glm5NextModel,
        )
        from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

        package = Path(vllm.__file__).resolve().parent
        actual_sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in SOURCE_PINS}
        if actual_sources != SOURCE_PINS:
            raise RuntimeError("native serving NONE model source differs from the frozen methods")
        self.model = model
        self.language = getattr(model, "language_model", None)
        self.text = getattr(self.language, "model", None)
        self.classes = (Glm5NextForConditionalGeneration, Glm5NextForCausalLM, Glm5NextModel)
        self.objects = (model, self.language, self.text)
        if tuple(type(item) for item in self.objects) != self.classes:
            raise RuntimeError("native serving NONE requires the actual unwrapped GLM model classes")
        self.module_call = torch.nn.Module.__call__
        self.has_compiled_submodule = has_compiled_submodule
        methods = (
            (model, "forward", Glm4vForConditionalGeneration.forward, "model_executor/models/glm4_1v.py"),
            (model, "compute_logits", Glm4vForConditionalGeneration.compute_logits, "model_executor/models/glm4_1v.py"),
            (self.language, "compute_logits", Glm5NextForCausalLM.compute_logits, "models/glm5next/nvidia/model.py"),
            (self.text, "forward", Glm5NextModel.forward, "models/glm5next/nvidia/model.py"),
        )
        self.methods, receipt_methods = [], []
        for owner, name, expected, source in methods:
            actual = getattr(owner, name)
            if (
                getattr(actual, "__self__", None) is not owner
                or getattr(actual, "__func__", None) is not expected
                or hasattr(expected, "__wrapped__")
                or Path(inspect.getfile(expected)).resolve() != (package / source).resolve()
            ):
                raise RuntimeError("native serving NONE method was compiled, decorated or replaced")
            self.methods.append((owner, name, actual))
            receipt_methods.append({"method": expected.__module__ + "." + expected.__qualname__, "source": source})
        self.receipt = {
            "schema_name": "glm53flash_native_serving_none_model",
            "schema_version": 1,
            "source_pins": actual_sources,
            "classes": [cls.__module__ + "." + cls.__name__ for cls in self.classes],
            "methods": receipt_methods,
            "compiled_model": False,
            "admission": "DIAGNOSTIC_ONLY_NATIVE_CALLS_STILL_REQUIRED",
        }
        self.bound = set()
        self.validate()

    def bind_observer_wrapper(self, name, wrapper):
        """Record only the two wrappers installed by the serving adapter itself."""
        if name not in ("forward", "compute_logits") or name in self.bound or not callable(wrapper):
            raise RuntimeError("native serving NONE observation wrapper is unknown or repeated")
        item = next(item for item in self.methods if item[0] is self.model and item[1] == name)
        if getattr(wrapper, "__wrapped__", None) != item[2] or getattr(self.model, name) is not wrapper:
            raise RuntimeError("native serving NONE wrapper does not retain its exact original callable")
        self.methods[self.methods.index(item)] = self.model, name, wrapper
        self.bound.add(name)
        self.validate()

    def validate(self):
        if (
            any(
                actual is not expected
                for actual, expected in zip(
                    (self.model, getattr(self.model, "language_model", None), getattr(self.language, "model", None)),
                    self.objects,
                    strict=True,
                )
            )
            or tuple(type(item) for item in self.objects) != self.classes
            or any(type(item).__call__ is not self.module_call for item in self.objects)
            or self.has_compiled_submodule(self.model)
            or any(getattr(item, "_compiled_call_impl", None) is not None for item in self.model.modules())
            or any(getattr(owner, name) != original for owner, name, original in self.methods)
        ):
            raise RuntimeError("native serving NONE original model/callable identity changed")


def validate_none_target(record):
    """Require the actual, unpadded native dispatch rather than a caller flag."""
    dispatch = record.get("native_dispatch", {})
    descriptor = dispatch.get("descriptor", {})
    if (
        record.get("runtime_mode") != "NONE"
        or descriptor.get("cg_mode") != "NONE"
        or record.get("phase") != "context"
        or record.get("used_cuda_graph") is not False
        or record.get("num_padded_tokens") != record.get("total_new_tokens")
        or dispatch.get("physical_tokens") != record.get("total_new_tokens")
        or dispatch.get("physical_requests") != record.get("batch_size")
        or descriptor.get("num_tokens") != record.get("total_new_tokens")
        or descriptor.get("num_reqs") != record.get("batch_size")
    ):
        raise RuntimeError("serving NONE diagnostic requires actual unpadded native prefill dispatch")


def _validate_operation_rows(record, rows):
    validate_none_target(record)
    if len(rows) != 277 or len({row["name"] for row in rows}) != 277:
        raise RuntimeError("serving NONE diagnostic requires all 277 physical native operations")
    if record.get("native_operation_calls") != {row["name"]: 1 for row in rows}:
        raise RuntimeError("serving NONE diagnostic requires exactly one actual call per physical operation")
    for row in rows:
        if (
            row.get("used_cuda_graph") is not False
            or row.get("invocation") != record["invocation"]
            or row.get("tp_rank") != record["tp_rank"]
            or row.get("phase") != record["phase"]
        ):
            raise RuntimeError("serving NONE operation was borrowed from a different execution")
    return rows


def diagnostic_operation_rows(record, rows):
    """Tag raw complete leaf-call evidence, never create accepted query rows."""
    for row in _validate_operation_rows(record, rows):
        row.update(
            serving_dispatch="NONE",
            native_dispatch=record["native_dispatch"],
            forward_id=record["forward_id"],
            serving_none_model_sha256=record["serving_none_model_sha256"],
            measurement_admission="DIAGNOSTIC_ONLY_NO_TABLE_EXPORT",
        )
    return rows


def measured_operation_rows(record, rows):
    """Write a distinct event contract; original diagnostic rows stay closed."""
    from collector.glm53flash_vllm_none_activity import NONE_MEASUREMENT_CONTRACT

    if (
        record.get("measurement_contract") != NONE_MEASUREMENT_CONTRACT
        or "measurement_admission" in record
        or any("measurement_admission" in row for row in rows)
    ):
        raise RuntimeError("native NONE measured rows require their new original writer contract")
    for row in _validate_operation_rows(record, rows):
        row.update(
            serving_dispatch="NONE",
            native_dispatch=record["native_dispatch"],
            forward_id=record["forward_id"],
            serving_none_model_sha256=record["serving_none_model_sha256"],
            measurement_contract=NONE_MEASUREMENT_CONTRACT,
            measurement_method="native_module_cuda_events_v1",
            profiled=record["profiled"],
            # Original kernel names remain evidence, but do not qualify a
            # specialization/interpolation rule by themselves.
            dispatch_fingerprint="",
        )
    return rows
