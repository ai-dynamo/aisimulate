# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM profile compatibility exports and runtime-facing validation bridges."""

from __future__ import annotations

import json

from aisimulate_core.fpm_profile import _MUTABLE_REFERENCES
from aisimulate_core.fpm_profile import FpmDeploymentProfile as FpmDeploymentProfile
from aisimulate_core.fpm_profile import FpmModelProfile as FpmModelProfile
from aisimulate_core.fpm_profile import FpmResourceProfile as FpmResourceProfile
from aisimulate_core.fpm_profile import load_fpm_profile as load_fpm_profile


def _quantization_from_engine_config(config_json: str) -> dict[str, str]:
    """Restore exact profile modes after checking the native wire dtypes.

    EngineConfig stores dtypes rather than quantization modes: ``sq`` and
    ``int8_wo`` both serialize as ``int8``. Both native timing and memory
    bridges use the profile to recover the mode, but only when every supplied
    wire dtype agrees with the selected deployment's canonical encoding.
    """
    from aisimulate_core.sdk.rust_engine_step import _moe_quant_to_dtype, _quant_to_dtype

    config = json.loads(config_json)
    profile = load_fpm_profile(config["extra"]["fpm_profile"])
    deployment = profile.select(
        model=config["model_name"],
        system=config["system_name"],
        backend=config["backend"],
        backend_version=config["backend_version"],
        tp_size=config["tp_size"],
        pp_size=config["pp_size"],
        attention_dp_size=1 if config.get("attention_dp_size") is None else config["attention_dp_size"],
        moe_tp_size=config.get("moe_tp_size"),
        moe_ep_size=config.get("moe_ep_size"),
        cp_size=1 if config.get("cp_size") is None else config["cp_size"],
    )
    modes = deployment.quantization_kwargs()
    for field, mode, encode in (
        ("weight_dtype", "gemm_quant_mode", _quant_to_dtype),
        ("moe_dtype", "moe_quant_mode", _moe_quant_to_dtype),
        ("activation_dtype", "fmha_quant_mode", _quant_to_dtype),
        ("kv_cache_dtype", "kvcache_quant_mode", _quant_to_dtype),
    ):
        supplied = config.get(field)
        expected = encode(modes[mode])
        if supplied is not None and supplied != expected:
            raise ValueError(
                f"FPM profile identity conflict: {field}={supplied!r}, "
                f"profile {mode}={modes[mode]!r} requires wire dtype {expected!r}"
            )
    modes.pop("attention_backend")
    return modes


def _validate_forward_pass_profile(config_json: str) -> str:
    """Validate schema/identity and return registration facts to the Rust owner.

    This path reads no timing data and does not construct an analytical model.
    Estimator defaults and interpolation selection belong to Rust.
    """
    from aisimulate_core.sdk import common
    from aisimulate_core.sdk.models.base import _MODEL_REGISTRY

    config = json.loads(config_json)
    profile = load_fpm_profile(config["fpm_profile"])
    version = config.get("backend_version")
    if not isinstance(version, str) or version.lower() in _MUTABLE_REFERENCES | {"previous", "next"}:
        raise ValueError("fpm_profile requires a literal backend_version")
    deployment = profile.select(
        model=config["model"],
        system=config["system"],
        backend=config["backend"],
        backend_version=version,
        tp_size=config["tp"],
        pp_size=config["pp"],
        attention_dp_size=config["attention_dp"],
        moe_tp_size=config.get("moe_tp_size"),
        moe_ep_size=config.get("moe_ep_size"),
    )
    deployment.validate_overrides(**{key: config.get(key) for key in deployment.quantization_kwargs()})
    family = common.ARCHITECTURE_TO_MODEL_FAMILY.get(profile.architecture, profile.architecture)
    return json.dumps(
        {
            "profile": profile.model_dump(mode="json"),
            "registered": family in _MODEL_REGISTRY,
            **deployment.quantization_kwargs(),
        }
    )
