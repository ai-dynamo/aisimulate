# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared speculative replay input normalization and pre-execution provenance.

This module transports the canonical cost selection; it does not construct or
select a performance model. Acceptance and seed remain scheduler inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sweeper.replay import ReplaySpec

JSONValue = Any


def _pop_alias(
    config: dict[str, JSONValue],
    target: str,
    aliases: tuple[str, ...],
) -> JSONValue | None:
    configured = [alias for alias in aliases if alias in config]
    if len(configured) > 1:
        names = ", ".join(configured)
        raise ValueError(f"engine config duplicates {target}: {names}")
    if not configured:
        return None
    return config.pop(configured[0])


def _parse_speculation(raw):
    from pydantic import TypeAdapter

    from .config.engine import SpeculationConfig

    return TypeAdapter(SpeculationConfig).validate_python(raw)


def normalize_speculation_engine_args(raw: Mapping[str, JSONValue], *, role: str) -> dict[str, JSONValue]:
    """Copy and normalize SD inputs for both AISim and Dynamo execution.

    Acceptance and seed affect the native sampler, never the per-forward cost.
    Explicit method identity stays in the canonical timing configuration.
    """
    rank = dict(raw)
    speculation_raw = rank.pop("speculation", None)
    if speculation_raw is not None:
        speculation = _parse_speculation(speculation_raw)
        if any(rank.get(key) not in (None, 0) for key in ("aic_nextn", "nextn")) or any(
            rank.get(key) is not None
            for key in (
                "aic_nextn_accepted",
                "nextn_accepted",
                "aic_nextn_accept_rates",
                "nextn_accept_rates",
                "aic_mtp_seed",
                "mtp_seed",
            )
        ):
            raise ValueError("speculation cannot be combined with legacy speculative decoding fields")
        rank.pop("nextn", None)
        rank["aic_nextn"] = speculation.num_speculative_tokens
        if speculation.kind == "mtp":
            rank["aic_nextn_accepted"] = speculation.expected_accepted_tokens
        else:
            rank["aic_nextn_accept_rates"] = ",".join(str(rate) for rate in speculation.acceptance_rates)
        rank["aic_mtp_seed"] = speculation.seed
        timing = rank.get("timing_model")
        if isinstance(timing, Mapping) and timing.get("type") == "external" and timing.get("provider") == "aic":
            config = dict(timing.get("config", {}))
            cost = speculation.cost_config()
            if config.get("nextn") not in (None, 0) or config.get("speculation") not in (None, cost):
                raise ValueError("speculation conflicts with timing_model.config")
            config["speculation"] = cost
            rank["timing_model"] = {**timing, "config": config}
        elif not (
            speculation.kind == "ngram"
            and isinstance(timing, Mapping)
            and timing.get("type") in {"fixed", "polynomial"}
        ):
            raise ValueError(
                f"{speculation.kind} speculation requires an external AIC timing_model before normalization; "
                "materialize the target cost identity first"
            )
    nextn = _pop_alias(rank, "aic_nextn", ("aic_nextn", "nextn"))
    if nextn is not None:
        if not isinstance(nextn, int) or isinstance(nextn, bool) or not 0 <= nextn <= 5:
            raise ValueError(f"engine provider {role} aic_nextn must be an integer in 0..=5")
        nextn = nextn or None
    if nextn is not None:
        rank["aic_nextn"] = nextn

    accept_rates = _pop_alias(
        rank,
        "aic_nextn_accept_rates",
        ("aic_nextn_accept_rates", "nextn_accept_rates"),
    )
    if accept_rates is not None:
        if nextn is None:
            raise ValueError(f"engine provider {role} aic_nextn_accept_rates requires aic_nextn")
        if not isinstance(accept_rates, str):
            raise ValueError(f"engine provider {role} aic_nextn_accept_rates must be a string")
        rank["aic_nextn_accept_rates"] = accept_rates

    nextn_accepted = _pop_alias(
        rank,
        "aic_nextn_accepted",
        ("aic_nextn_accepted", "nextn_accepted"),
    )
    if nextn_accepted is not None:
        if nextn is None:
            raise ValueError(f"engine provider {role} aic_nextn_accepted requires aic_nextn")
        if accept_rates is not None:
            raise ValueError(f"engine provider {role} cannot set both aic_nextn_accepted and aic_nextn_accept_rates")
        rank["aic_nextn_accept_rates"] = _accept_rates_for_expected(nextn, nextn_accepted, role=role)

    mtp_seed = _pop_alias(rank, "aic_mtp_seed", ("aic_mtp_seed", "mtp_seed"))
    if mtp_seed is not None:
        if not isinstance(mtp_seed, int) or isinstance(mtp_seed, bool) or not 0 <= mtp_seed <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(f"engine provider {role} aic_mtp_seed must be an unsigned 64-bit integer")
        if nextn is None:
            # Older native descriptors serialize this disabled sampler default.
            # Drop it, but reject an active-looking seed rather than ignore it.
            if mtp_seed != 42:
                raise ValueError(f"engine provider {role} aic_mtp_seed requires aic_nextn")
        else:
            rank["aic_mtp_seed"] = mtp_seed

    return rank


def _accept_rates_for_expected(nextn: int, value: JSONValue, *, role: str) -> str:
    """Lower an explicit expected accepted-token count to conditional rates."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= nextn
    ):
        raise ValueError(f"engine provider {role} aic_nextn_accepted must be finite and within [0, {nextn}]")
    expected = float(value)
    whole = int(expected)
    fraction = expected - whole
    rates = [1.0] * whole
    if len(rates) < nextn:
        rates.append(fraction)
    rates.extend([0.0] * (nextn - len(rates)))
    return ",".join(format(rate, ".17g") for rate in rates)


def speculation_report_metadata(
    spec: ReplaySpec,
    *,
    resolved_role_args: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve speculative provenance before simulation, using the priced ranks.

    Callers with materialized engines pass role descriptors under ``aggregated``,
    ``prefill`` and ``decode``. The original spec retains user choices and whether
    KV capacity was fixed before materialization. Legacy architecture lookup can
    fail, so this function must be called before starting simulation.
    """
    deployment = spec.backend_deployment
    result = {}
    for role, authored_args in (
        ("aggregated", deployment.agg_engine_args),
        ("prefill", deployment.prefill_engine_args),
        ("decode", deployment.decode_engine_args),
    ):
        if not authored_args:
            continue
        authored = authored_args.get("rank", authored_args)
        resolved_args = (resolved_role_args or {}).get(role, authored_args)
        rank = resolved_args.get("rank", resolved_args)
        chosen = authored.get("speculation")
        depth = rank.get("aic_nextn", rank.get("nextn")) or (chosen or {}).get("num_speculative_tokens")
        if not depth:
            continue
        timing = rank.get("timing_model") or {}
        config = timing.get("config") or {}
        canonical = config.get("speculation") or {}
        authored_timing = authored.get("timing_model") or {}
        authored_canonical = (authored_timing.get("config") or {}).get("speculation")
        resolved = canonical.get("kind", chosen["kind"] if chosen else "mtp")
        target_model = config.get("model", config.get("model_path", authored.get("aic_model_path")))
        if chosen is None and not canonical and target_model and timing.get("type") not in {"fixed", "polynomial"}:
            from aisimulate_core.sdk.common import DSPARK_ARCHITECTURES
            from aisimulate_core.sdk.models.helpers import _get_model_info

            if _get_model_info(target_model)["architecture"] in DSPARK_ARCHITECTURES:
                resolved = "dspark"
        approximation = {
            "mtp": "op_level_nextn_target_layers_and_depth_plus_one_verification",
            "ngram": "op_level_prompt_lookup_verification",
            "dspark": "legacy_dspark_graph",
        }[resolved]
        if timing.get("type") in {"fixed", "polynomial"}:
            approximation = timing["type"] + "_timing"
        explicit_capacity = resolved_args.get(
            "num_gpu_blocks_is_explicit",
            authored_args.get("num_gpu_blocks_is_explicit", authored.get("num_gpu_blocks") is not None),
        )
        capacity_source = deployment.performance_model_metadata.get(role, {}).get(
            "capacity_source", "explicit_fixed" if explicit_capacity else "inferred"
        )
        result[role] = {
            "target_model": target_model,
            "requested": chosen
            or authored_canonical
            or canonical
            or {"kind": "legacy_nextn", "num_speculative_tokens": depth},
            "resolved_method": resolved,
            "num_speculative_tokens": depth,
            "expected_accepted_draft_tokens": chosen.get("expected_accepted_tokens")
            if chosen
            else authored.get("aic_nextn_accepted", authored.get("nextn_accepted")),
            "conditional_acceptance_rates": rank.get("aic_nextn_accept_rates", rank.get("nextn_accept_rates")),
            "seed": rank.get("aic_mtp_seed", rank.get("mtp_seed", (chosen or {}).get("seed", 42))),
            "cost_approximation": approximation,
            "capacity_source": capacity_source,
            "kv_transfer_approximation": "scalar_bytes_per_token" if deployment.deployment_mode == "disagg" else None,
            "qualification": "functional_only",
        }
    return result
