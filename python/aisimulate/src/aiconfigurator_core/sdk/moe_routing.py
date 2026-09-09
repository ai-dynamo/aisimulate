# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select measured marginals and install them in the DeepEP-LL prediction graph.

Measured marginals do not determine expert co-selection or token correlations.
The native consumer preserves expert IDs with contiguous placement, and models
the missing joint distribution using distinct Top-K quota routing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from functools import wraps
from pathlib import Path

from aiconfigurator_core.sdk import expert_popularity as database


class MoeRoutingError(ValueError):
    """Hard configuration/profile error; native best-available must not hide it."""


def _hard_routing_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise MoeRoutingError(str(error)) from error

    return checked


@_hard_routing_errors
def validate_routing_options(mode, alpha, legacy=None):
    if mode not in {"auto", "uniform", "random", "power-law"}:
        raise ValueError(f"Invalid moe_routing_mode: {mode!r}")
    if alpha is not None and (mode != "power-law" or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("moe_power_law_alpha must be finite and positive, and requires mode='power-law'")
    if legacy is not None and (mode != "auto" or alpha is not None):
        expected = "uniform" if mode == "uniform" else f"power_law_{alpha}" if alpha else "power_law"
        if legacy != expected:
            raise ValueError("workload_distribution conflicts with explicit moe_routing_mode/alpha")


@_hard_routing_errors
def select_profile(
    *,
    model_id,
    revision,
    num_layers,
    num_experts,
    top_k,
    phase,
    backend,
    mode="auto",
    alpha=None,
    legacy=None,
    enable_eplb=False,
    data_root=None,
    num_moe_layers=None,
):
    """Absence is fallback; a present incompatible/corrupt bundle is an error."""
    validate_routing_options(mode, alpha, legacy)
    provenance = {"requested_mode": mode, "prediction_phase": phase, "selected_mode": "power-law"}

    def fallback(reason):
        if mode == "random":
            raise ValueError(f"Random-input measured routing unavailable: {reason}")
        return {"provenance": {**provenance, "fallback_reason": reason}, "layers": []}

    if legacy is not None and mode == "auto":
        return {"provenance": {**provenance, "selected_mode": "legacy", "workload_distribution": legacy}, "layers": []}
    if mode in {"uniform", "power-law"}:
        return {"provenance": {**provenance, "selected_mode": mode, "alpha": alpha}, "layers": []}
    if backend != "deepep_ll":
        return fallback("unsupported_consumer_backend")
    if enable_eplb:
        raise ValueError("DeepEP-LL routing selection does not implement EPLB placement")
    # Resolve before loading: a missing file INSIDE a candidate is corruption,
    # not a missing model. Do not catch FileNotFoundError from the loader.
    root = Path(data_root) if data_root is not None else database._default_data_root()
    direct = root / database.model_id_to_bundle_name(model_id)
    if direct.exists() and not direct.is_dir():
        raise ValueError(f"Invalid expert popularity bundle: {direct}")
    try:
        bundle = database._resolve_bundle(model_id, data_root)
    except FileNotFoundError:
        return fallback("missing_bundle")
    metadata = database.validate_expert_popularity_bundle(bundle)
    identity = metadata["model"]
    if model_id not in [identity["id"], *identity.get("aliases", [])]:
        raise ValueError("Expert popularity model identity mismatch")
    if revision is not None and revision != identity["revision"]:
        raise ValueError("Expert popularity model revision mismatch")
    routing = metadata["routing"]
    if (routing["num_layers"], routing["num_routed_experts"], routing["top_k"]) != (num_layers, num_experts, top_k):
        raise ValueError("Expert popularity routing dimension mismatch")
    if num_moe_layers is not None and len(routing["moe_layer_ids"]) != num_moe_layers:
        raise ValueError("Expert popularity MoE layer coverage mismatch")
    measurement_phase = metadata["measurement"]["phase"]
    proxy = measurement_phase == "prefill" and phase == "decode" and mode == "auto"
    if measurement_phase != phase and not proxy:
        return fallback("phase_mismatch")
    table = database.load_expert_popularity(model_id, data_root)
    layers = []
    for layer in sorted(routing["moe_layer_ids"]):
        values = table.loc[table.layer_id == layer].sort_values("expert_id").assignment_share.tolist()
        if any(not math.isfinite(p) or p < 0 or p * top_k > 1 + 1e-10 for p in values):
            raise ValueError("Measured probabilities violate distinct Top-K inclusion bounds")
        layers.append({"layer_id": layer, "probabilities": values})
    digest = hashlib.sha256(json.dumps(layers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provenance.update(
        {
            "selected_mode": "measured_prefill_proxy" if proxy else "measured",
            "display_name": "Random-input measured",
            "model_id": identity["id"],
            "bundle_revision": identity["revision"],
            "revision_selection": "explicit" if revision else "bundle_pinned",
            "collection_checkpoint": metadata["provenance"]["collection_checkpoint"],
            "measurement_phase": measurement_phase,
            "profile_digest": digest,
            "placement": "contiguous_expert_id",
            "latency_source": "estimated",
            "compute_calibration": "balanced_exact_shape",
            "compute_calibration_policy": "moe_expert_compute_then_moe_perf; no distribution substitution",
            "moe_layer_ids": list(routing["moe_layer_ids"]),
            "fallback_reason": "prefill_proxy" if proxy else None,
        }
    )
    for layer in layers:
        layer.update({"profile_digest": digest, "seed": 0xA1C0DEE5EED00001, "trials": 4096})
    return {"provenance": provenance, "layers": layers}


@_hard_routing_errors
def apply_routing(model, model_info):
    """Expand aggregated LL operators by layer; keep other graph structure intact."""
    import aiconfigurator_core._aiconfigurator_core as native

    cfg = model.config
    if not hasattr(model, "_num_experts"):
        return
    model.moe_routing_provenance = {}
    for op_phase, phase in (("context", "prefill"), ("generation", "decode")):
        selection = select_profile(
            model_id=model.model_path,
            revision=cfg.moe_model_revision,
            num_layers=model_info["layers"],
            num_experts=model._num_experts,
            top_k=model._topk,
            phase=phase,
            backend=(cfg.moe_comm_backend or {}).get(op_phase) if cfg.forward_model == "op_level" else "fpm",
            mode=cfg.moe_routing_mode,
            alpha=cfg.moe_power_law_alpha,
            legacy=cfg.workload_distribution if cfg._legacy_workload_is_explicit else None,
            enable_eplb=cfg.enable_eplb,
            num_moe_layers=model_info.get("num_moe_layers"),
        )
        model.moe_routing_provenance[phase] = selection["provenance"]
        if not selection["layers"]:
            selection["provenance"]["workload_distribution"] = cfg.workload_distribution
            continue
        if cfg.wideep_num_slots not in (None, model._num_experts):
            raise ValueError("Measured routing does not support replicated expert slots")

        def count_compute(spec):
            kind, fields = next(iter(spec.items()))
            if kind == "Overlap":
                return sum(count_compute(child) for child in [*fields["group_a"], *fields["group_b"]])
            if kind == "Fallback" and "Moe" in json.dumps(fields):
                raise ValueError("Measured routing cannot be hidden behind a performance-data fallback")
            return int(kind == "MoeExpertCompute")

        if sum(count_compute(json.loads(op._spec_json())) for op in getattr(model, f"{op_phase}_ops")) != 1:
            raise ValueError(
                "Measured routing requires one aggregate MoE span; mixed spans need explicit layer mapping"
            )

        # TODO(EPLB): extend placement, replication and dynamic rebalance as
        # strategies separate from distribution selection. No rank permutation.
        def expand(spec):
            kind, fields = next(iter(spec.items()))
            if kind == "Overlap":
                fields["group_a"] = [new for child in fields["group_a"] for new in expand(child)]
                fields["group_b"] = [new for child in fields["group_b"] for new in expand(child)]
            if kind in {"MoeAllToAll", "MoeExpertCompute"}:
                if fields.get("enable_eplb"):
                    raise ValueError("Measured consumer requires fixed non-EPLB placement")
                # Whole-model legacy scales include dense layers. Divide by the
                # original scale's layer basis, then sum ONLY the measured MoE layers.
                result = []
                for layer in selection["layers"]:
                    item = copy.deepcopy(fields)
                    item["name"] += f"_layer_{layer['layer_id']}"
                    item["scale_factor"] /= model._num_layers
                    item["measured_routing"] = layer
                    if kind == "MoeExpertCompute":
                        item["routing_attention_tp_size"] = cfg.tp_size * cfg.cp_size if op_phase == "context" else 1
                    result.append({kind: item})
                return result
            return [spec]

        original = getattr(model, f"{op_phase}_ops")
        rewritten = []
        new_ops = []
        for op in original:
            before = json.loads(op._spec_json())
            after = expand(copy.deepcopy(before))
            rewritten.extend(after)
            if after == [before]:
                new_ops.append(op)
                continue
            for item in after:
                new_op = native.op_from_spec_json(json.dumps(item))
                if "measured_routing" not in new_op._spec_json():
                    raise RuntimeError("Native extension lacks measured routing support; rebuild AISimulate")
                args, kwargs = new_op.__getnewargs_ex__()
                new_op = type(op)(*args, **kwargs)
                new_op.__dict__.update(copy.deepcopy(getattr(op, "__dict__", {})))
                new_ops.append(new_op)
        if not any("measured_routing" in json.dumps(item) for item in rewritten):
            raise ValueError("Selected measured routing but no supported LL operators were built")
        setattr(model, f"{op_phase}_ops", new_ops)
