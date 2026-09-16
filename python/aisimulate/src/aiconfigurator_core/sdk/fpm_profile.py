# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit identity and resource bounds for decoder-only FPM simulation.

Profiles describe every rank of a deployment without constructing operations.
Resource values are declarations supplied by the caller, not GPU measurements
or automatically inferred checkpoint metadata. Their provenance must say how
they were obtained; CUDA graph reservations are supplied separately at runtime.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aiconfigurator_core.sdk import common

_PositiveInt = Annotated[int, Field(gt=0)]
_Bytes = Annotated[int, Field(ge=0, le=2**53)]
_Nonempty = Annotated[str, Field(min_length=1, pattern=r"\S")]
_MUTABLE_REFERENCES = {"main", "master", "head", "latest", "current", "unknown", "tbd", "unpinned"}


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FpmResourceProfile(_ProfileModel):
    """Conservative rank-local bounds at a declared scheduler envelope.

    ``max_num_tokens`` and ``max_batch_size`` apply to each attention-DP rank,
    not the summed worker iteration used to query FPM timings. Overheads must
    exclude the separately configured CUDA graph reservation.
    """

    weights_bytes: _Bytes
    activations_bytes: _Bytes
    runtime_overhead_bytes: _Bytes
    comm_overhead_bytes: _Bytes
    kv_bytes_per_token: Annotated[int, Field(gt=0, le=2**53)]
    cache_layout: Literal["linear"]
    max_num_tokens: _PositiveInt
    max_batch_size: _PositiveInt
    provenance: _Nonempty

    @model_validator(mode="after")
    def _exact_total(self) -> FpmResourceProfile:
        if self.non_kv_bytes > 2**53:
            raise ValueError("total non-KV resource bytes must not exceed 2**53")
        return self

    @property
    def non_kv_bytes(self) -> int:
        return self.weights_bytes + self.activations_bytes + self.runtime_overhead_bytes + self.comm_overhead_bytes

    def validate_envelope(self, *, max_num_tokens: int, max_batch_size: int) -> None:
        for name, requested in (("max_num_tokens", max_num_tokens), ("max_batch_size", max_batch_size)):
            if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
                raise ValueError(f"{name} must be a positive integer, got {requested!r}")
            limit = getattr(self, name)
            if requested > limit:
                raise ValueError(
                    f"FPM resource envelope exceeded: requested rank-local {name}={requested}, "
                    f"profile bound={limit}; provide resource bounds for the larger scheduler envelope"
                )


class FpmDeploymentProfile(_ProfileModel):
    """One exact hardware, runtime, topology, precision and resource identity."""

    system: _Nonempty
    backend: Literal["vllm"]
    backend_version: _Nonempty
    tp: _PositiveInt
    pp: Literal[1] = 1
    dp: _PositiveInt
    moe_tp: _PositiveInt
    moe_ep: _PositiveInt
    cp: Literal[1] = 1
    gemm_quant_mode: _Nonempty
    moe_quant_mode: _Nonempty
    fmha_quant_mode: _Nonempty
    comm_quant_mode: _Nonempty
    kv_cache_dtype: _Nonempty
    moe_backend: _Nonempty = "auto"
    attention_backend: _Nonempty = "auto"
    enable_wideep: Literal[False] = False
    enable_eplb: Literal[False] = False
    resources: FpmResourceProfile

    @field_validator("pp", "cp", mode="before")
    @classmethod
    def _integer_dimensions(cls, value: Any) -> int:
        if type(value) is not int:
            raise ValueError("parallel dimensions must be integers, not booleans or floats")
        return value

    @field_validator("enable_wideep", "enable_eplb", mode="before")
    @classmethod
    def _boolean_flags(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("backend feature flags must be booleans")
        return value

    @field_validator("backend_version")
    @classmethod
    def _literal_version(cls, value: str) -> str:
        if value.lower() in _MUTABLE_REFERENCES | {"previous", "next"} or any(c.isspace() for c in value):
            raise ValueError("FPM profile backend_version must be a literal runtime version")
        return value

    @model_validator(mode="after")
    def _deployment_identity(self) -> FpmDeploymentProfile:
        modes = {
            "gemm_quant_mode": common.GEMMQuantMode,
            "moe_quant_mode": common.MoEQuantMode,
            "fmha_quant_mode": common.FMHAQuantMode,
            "comm_quant_mode": common.CommQuantMode,
            "kv_cache_dtype": common.KVCacheQuantMode,
        }
        for name, enum in modes.items():
            if getattr(self, name) not in enum.__members__:
                raise ValueError(f"unknown FPM {name}: {getattr(self, name)!r}")
        return self

    @property
    def parallel_tuple(self) -> tuple[int, int, int, int, int, int]:
        return self.tp, self.pp, self.dp, self.moe_tp, self.moe_ep, self.cp

    def match_identity(self) -> list[str]:
        """Return the exact 15-column native FPM identity, with no aliasing."""
        return [
            self.gemm_quant_mode,
            self.moe_quant_mode,
            self.fmha_quant_mode,
            self.comm_quant_mode,
            self.kv_cache_dtype,
            *(str(value) for value in self.parallel_tuple),
            self.moe_backend,
            self.attention_backend,
            str(self.enable_wideep),
            str(self.enable_eplb),
        ]

    def quantization_kwargs(self) -> dict[str, str]:
        return {
            "gemm_quant_mode": self.gemm_quant_mode,
            "moe_quant_mode": self.moe_quant_mode,
            "fmha_quant_mode": self.fmha_quant_mode,
            "comm_quant_mode": self.comm_quant_mode,
            "kvcache_quant_mode": self.kv_cache_dtype,
            "attention_backend": self.attention_backend,
        }

    def validate_overrides(self, **overrides: str | None) -> None:
        for name, expected in self.quantization_kwargs().items():
            supplied = overrides.get(name)
            if supplied is not None and supplied != expected:
                raise ValueError(
                    f"FPM profile identity conflict: {name}={supplied!r}, profile={expected!r}; "
                    "use matching measured precision and backend settings"
                )


class FpmModelProfile(_ProfileModel):
    """Serializable FPM metadata independent of an analytical model class."""

    schema_version: Literal[1]
    model: _Nonempty
    model_revision: _Nonempty
    architecture: _Nonempty
    context_length: _PositiveInt
    num_experts: Annotated[int, Field(ge=0)]
    provenance: _Nonempty
    deployments: Annotated[list[FpmDeploymentProfile], Field(min_length=1)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _integer_schema(cls, value: Any) -> int:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @field_validator("model_revision")
    @classmethod
    def _pinned_revision(cls, value: str) -> str:
        if value.strip().lower() in _MUTABLE_REFERENCES:
            raise ValueError("provide a pinned model_revision, not a mutable or unknown placeholder")
        return value

    @model_validator(mode="after")
    def _unique_deployments(self) -> FpmModelProfile:
        identities = set()
        for deployment in self.deployments:
            tp = deployment.dp == 1 and deployment.moe_ep == 1 and deployment.moe_tp == deployment.tp
            dep = deployment.tp == 1 and deployment.moe_tp == 1 and deployment.moe_ep == deployment.dp
            tep = deployment.dp == 1 and deployment.moe_tp == 1 and deployment.moe_ep == deployment.tp
            dense_tp = deployment.dp == 1 and deployment.moe_ep == 1 and deployment.moe_tp in (1, deployment.tp)
            if not (dense_tp if self.num_experts == 0 else tp or dep or tep):
                raise ValueError(
                    "FPM profile requires dense TP or an MoE TP, DEP, or TEP topology with matching world sizes"
                )
            identity = (
                deployment.system,
                deployment.backend,
                deployment.backend_version,
                deployment.parallel_tuple,
            )
            if identity in identities:
                raise ValueError(
                    "duplicate FPM deployment identity; specify one precision/resource profile per topology"
                )
            identities.add(identity)
            if deployment.moe_ep > 1 and (self.num_experts == 0 or self.num_experts % deployment.moe_ep):
                raise ValueError("num_experts must be positive and divisible by moe_ep for expert parallelism")
        return self

    def select(
        self,
        *,
        model: str,
        system: str,
        backend: str,
        backend_version: str,
        tp_size: int = 1,
        pp_size: int = 1,
        attention_dp_size: int = 1,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        cp_size: int = 1,
    ) -> FpmDeploymentProfile:
        if model != self.model:
            raise ValueError(f"FPM profile model identity mismatch: requested {model!r}, profile={self.model!r}")
        shape = (
            tp_size,
            pp_size,
            attention_dp_size,
            1 if moe_tp_size is None else moe_tp_size,
            1 if moe_ep_size is None else moe_ep_size,
            cp_size,
        )
        if any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("FPM deployment dimensions must be positive integers, not booleans")
        for deployment in self.deployments:
            if (
                deployment.system == system
                and deployment.backend == backend
                and deployment.backend_version == backend_version
                and deployment.parallel_tuple == shape
            ):
                return deployment
        raise ValueError(
            f"no matching FPM deployment profile for {model!r}, {system}/{backend}/{backend_version}, "
            f"(tp,pp,dp,moe_tp,moe_ep,cp)={shape}; provide explicit resources and precision for that deployment"
        )


def load_fpm_profile(value: dict[str, Any] | str | FpmModelProfile) -> FpmModelProfile:
    if isinstance(value, FpmModelProfile):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        value = json.loads(value)
    return FpmModelProfile.model_validate(value)


def _quantization_from_engine_config(config_json: str) -> dict[str, str]:
    """Restore exact profile modes after checking the native wire dtypes.

    EngineConfig stores dtypes rather than quantization modes: ``sq`` and
    ``int8_wo`` both serialize as ``int8``. Both native timing and memory
    bridges use the profile to recover the mode, but only when every supplied
    wire dtype agrees with the selected deployment's canonical encoding.
    """
    from aiconfigurator_core.sdk.rust_engine_step import _moe_quant_to_dtype, _quant_to_dtype

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


def resolve_fpm_interpolation(profile: FpmModelProfile | None, method: str = "auto") -> Literal["sol", "direct"]:
    """Choose from explicit registry presence, never from a caught build error."""
    if method not in {"auto", "sol", "direct"}:
        raise ValueError(f"unknown FPM interpolation {method!r}; choose auto, sol, or direct")
    if method == "direct":
        if profile is None:
            raise ValueError("direct FPM interpolation requires an fpm_profile with identity and resource metadata")
        return "direct"
    if profile is None:
        return "sol"
    from aiconfigurator_core.sdk.models.base import _MODEL_REGISTRY

    family = common.ARCHITECTURE_TO_MODEL_FAMILY.get(profile.architecture, profile.architecture)
    registered = family in _MODEL_REGISTRY
    if method == "sol" and not registered:
        raise ValueError(
            f"SOL interpolation requires a registered analytical model class for {profile.architecture!r}; "
            "choose fpm_interpolation='direct' to use the supplied profile"
        )
    return "sol" if registered else "direct"
