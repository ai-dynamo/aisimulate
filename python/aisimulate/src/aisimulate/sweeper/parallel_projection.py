# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured features and deterministic projection for parallel configs.

Vizier searches a compact, regular feature space.  The existing parallel
enumerator remains the source of truth: every suggestion is projected onto one
of the branch's backend-compatible, KV-feasible configs before replay.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .afd import AFDParallelConfig
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .search_space import BranchSpace

ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig | AFDParallelConfig


class InfeasibleParallelSelection(ValueError):
    """A sampled independent/custom parallel mapping has no legal pool member."""


USED_GPU_RATIO = "used_gpu_ratio"
PREFILL_GPU_SHARE = "prefill_gpu_share"
AGG_GPUS_PER_ENGINE = "agg_num_gpus_per_engine_target"
PREFILL_GPUS_PER_ENGINE = "prefill_num_gpus_per_engine_target"
DECODE_GPUS_PER_ENGINE = "decode_num_gpus_per_engine_target"
AGG_ATTENTION_MODE = "agg_attention_mode"
PREFILL_ATTENTION_MODE = "prefill_attention_mode"
DECODE_ATTENTION_MODE = "decode_attention_mode"
AGG_FFN_MODE = "agg_ffn_mode"
PREFILL_FFN_MODE = "prefill_ffn_mode"
DECODE_FFN_MODE = "decode_ffn_mode"
PARALLEL_CONFIG_CHOICE = "parallel_config_choice"

_ATTENTION_MODE_ORDER = ("tp", "dp")
_FFN_MODE_ORDER = ("ep", "tp")


@dataclass(frozen=True)
class ParallelParameter:
    """One Vizier-facing latent dimension."""

    name: str
    kind: Literal["float", "integer", "discrete", "categorical"]
    default: float | str
    values: tuple[float | str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    log_scale: bool = False

    @property
    def is_constant(self) -> bool:
        if self.kind in {"float", "integer"}:
            return self.minimum == self.maximum
        return len(self.values) == 1


@dataclass(frozen=True)
class ParallelProjection:
    """A latent request and the valid config selected for replay."""

    config: ParallelConfig
    requested_features: dict[str, float | str]
    actual_features: dict[str, float | str]
    distance: float
    mode_projected: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_features": self.requested_features,
            "actual_features": self.actual_features,
            "projection_distance": self.distance,
            "mode_projected": self.mode_projected,
            "actual_parallel_config": asdict(self.config),
        }


def _all_shapes(config: ParallelConfig) -> tuple[ParallelShape, ...]:
    if isinstance(config, ReplicaParallelConfig):
        return (config.shape,)
    if isinstance(config, AFDParallelConfig):
        return (config.companion.shape,) if config.companion is not None else ()
    return (config.prefill.shape, config.decode.shape)


def _parallel_role(config: ParallelConfig, role: str) -> ReplicaParallelConfig:
    if isinstance(config, ReplicaParallelConfig):
        if role != "agg":
            raise ValueError(f"aggregated parallel config has no {role!r} role")
        return config
    if isinstance(config, AFDParallelConfig):
        if config.companion is None or config.companion_role != role:
            raise ValueError(f"AFD parallel config has no {role!r} companion")
        return config.companion
    return config.prefill if role == "prefill" else config.decode


def _attention_mode(shape: ParallelShape) -> str:
    # The enumerator emits pure attention TP or DP.  G=1 is canonicalized as TP.
    return "dp" if shape.dp > 1 else "tp"


def _ffn_mode(shape: ParallelShape) -> str:
    # The enumerator emits pure MoE TP or EP.  G=1 is canonicalized as TP.
    return "ep" if shape.moe_ep > 1 else "tp"


def _config_key(config: ParallelConfig) -> tuple[int, ...]:
    def role_key(role: ReplicaParallelConfig) -> tuple[int, ...]:
        shape = role.shape
        return (shape.tp, shape.pp, shape.dp, shape.moe_tp, shape.moe_ep, role.replicas)

    if isinstance(config, ReplicaParallelConfig):
        return role_key(config)
    if isinstance(config, AFDParallelConfig):
        topology = config.topology
        topology_key = (
            topology.n_a_nodes,
            topology.n_f_nodes,
            topology.tp_a,
            topology.a_batch_size,
            topology.f_moe_ep_size,
            topology.num_microbatches,
        )
        return topology_key + (() if config.companion is None else role_key(config.companion))
    return (*role_key(config.prefill), *role_key(config.decode))


def _ordered_values(values: set[str], preferred: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in preferred if value in values)


def _geometric_default(values: tuple[float, ...]) -> float:
    target = math.sqrt(min(values) * max(values))
    return min(values, key=lambda value: (abs(math.log(value) - math.log(target)), value))


class ParallelConfigProjector:
    """Encode and project one deployment-mode branch's parallel config pool."""

    def __init__(self, branch: BranchSpace):
        if not branch.parallel_configs:
            raise ValueError("parallel projection requires at least one valid config")

        self.branch = branch
        self._flat = branch.flat_parallel_choices
        self._independent = bool(branch.parallel_independent_choices or branch.parallel_independent_log_ranges)
        self._hybrid = branch.deployment_mode == "disagg" and bool(branch.parallel_custom_choices or self._independent)
        self.gpu_budget = branch.gpu_budget or max(config.total_gpus for config in branch.parallel_configs)
        self.is_moe = any(
            shape.moe_tp > 1 or shape.moe_ep > 1 for config in branch.parallel_configs for shape in _all_shapes(config)
        )
        self._features = {config: self._encode(config) for config in branch.parallel_configs}
        self.parameters = self._build_parameters()
        self.constants = {parameter.name: parameter.default for parameter in self.parameters if parameter.is_constant}

    def _role_features(self, prefix: str, role: ReplicaParallelConfig) -> dict[str, float | str]:
        features: dict[str, float | str] = {
            f"{prefix}_num_gpus_per_engine_target": float(role.shape.gpus_per_worker),
            f"{prefix}_attention_mode": _attention_mode(role.shape),
        }
        if self.is_moe:
            features[f"{prefix}_ffn_mode"] = _ffn_mode(role.shape)
        return features

    def _encode(self, config: ParallelConfig) -> dict[str, float | str]:
        features: dict[str, float | str] = {USED_GPU_RATIO: config.total_gpus / self.gpu_budget}
        if isinstance(config, AFDParallelConfig):
            return features
        if isinstance(config, ReplicaParallelConfig):
            features.update(self._role_features("agg", config))
            return features

        features[PREFILL_GPU_SHARE] = config.prefill.total_gpus / config.total_gpus
        features.update(self._role_features("prefill", config.prefill))
        features.update(self._role_features("decode", config.decode))
        return features

    def _float_parameter(self, name: str, *, default: float) -> ParallelParameter:
        values = [float(features[name]) for features in self._features.values()]
        minimum, maximum = min(values), max(values)
        return ParallelParameter(
            name=name,
            kind="float",
            minimum=minimum,
            maximum=maximum,
            default=min(max(default, minimum), maximum),
        )

    def _discrete_parameter(self, name: str) -> ParallelParameter:
        values = tuple(sorted({float(features[name]) for features in self._features.values()}))
        return ParallelParameter(
            name=name,
            kind="discrete",
            values=values,
            default=_geometric_default(values),
            log_scale=True,
        )

    def _categorical_parameter(self, name: str, preferred: tuple[str, ...]) -> ParallelParameter:
        present = {str(features[name]) for features in self._features.values()}
        values = _ordered_values(present, preferred)
        return ParallelParameter(name=name, kind="categorical", values=values, default=values[0])

    def _independent_parameters(self) -> list[ParallelParameter]:
        parameters: list[ParallelParameter] = []
        for name in self.branch.parallel_independent_choices:
            bounds = self.branch.parallel_independent_log_ranges.get(name)
            if bounds is not None:
                parameters.append(
                    ParallelParameter(
                        name=name,
                        kind="integer",
                        minimum=float(bounds[0]),
                        maximum=float(bounds[1]),
                        default=float(bounds[0]),
                        log_scale=True,
                    )
                )
                continue
            values = self.branch.parallel_independent_choices[name]
            parameters.append(
                ParallelParameter(
                    name=name,
                    kind="discrete",
                    values=tuple(float(value) for value in values),
                    default=float(1 if 1 in values else values[0]),
                    # choices and linear ranges are linear unless the public
                    # domain explicitly requested scale: log.
                    log_scale=False,
                )
            )
        return parameters

    def _hybrid_parameters(self) -> tuple[ParallelParameter, ...]:
        parameters: list[ParallelParameter] = []
        for role in ("prefill", "decode"):
            choices = self.branch.parallel_custom_choices.get(role)
            if choices is not None:
                values = tuple(float(index) for index in range(len(choices)))
                parameters.append(
                    ParallelParameter(
                        name=f"{role}_{PARALLEL_CONFIG_CHOICE}",
                        kind="discrete",
                        values=values,
                        default=values[0],
                    )
                )
        parameters.extend(self._independent_parameters())

        independent_roles = {name.split("_", 1)[0] for name in self.branch.parallel_independent_choices}
        default_roles = (
            {
                "prefill",
                "decode",
            }
            - set(self.branch.parallel_custom_choices)
            - independent_roles
        )
        if not default_roles:
            return tuple(parameters)

        # A default role retains the built-in correlated projection. Shared
        # ratios remain the replica-footprint controls for the complete branch;
        # only role-specific worker-shape dimensions for default roles are added.
        parameters.extend(
            [
                self._float_parameter(USED_GPU_RATIO, default=1.0),
                self._float_parameter(PREFILL_GPU_SHARE, default=0.5),
            ]
        )
        for role in ("prefill", "decode"):
            if role not in default_roles:
                continue
            prefix = "prefill" if role == "prefill" else "decode"
            parameters.extend(
                [
                    self._discrete_parameter(PREFILL_GPUS_PER_ENGINE if role == "prefill" else DECODE_GPUS_PER_ENGINE),
                    self._categorical_parameter(
                        PREFILL_ATTENTION_MODE if role == "prefill" else DECODE_ATTENTION_MODE,
                        _ATTENTION_MODE_ORDER,
                    ),
                ]
            )
            if self.is_moe:
                parameters.append(
                    self._categorical_parameter(
                        PREFILL_FFN_MODE if prefix == "prefill" else DECODE_FFN_MODE,
                        _FFN_MODE_ORDER,
                    )
                )
        return tuple(parameters)

    def _build_parameters(self) -> tuple[ParallelParameter, ...]:
        if self._flat:
            values = tuple(float(index) for index in range(len(self.branch.parallel_configs)))
            return (
                ParallelParameter(
                    name=PARALLEL_CONFIG_CHOICE,
                    kind="discrete",
                    values=values,
                    default=values[0],
                ),
            )
        if self._hybrid:
            return self._hybrid_parameters()
        if self._independent:
            return tuple(self._independent_parameters())
        parameters = [self._float_parameter(USED_GPU_RATIO, default=1.0)]
        if self.branch.deployment_mode == "agg":
            parameters.extend(
                [
                    self._discrete_parameter(AGG_GPUS_PER_ENGINE),
                    self._categorical_parameter(AGG_ATTENTION_MODE, _ATTENTION_MODE_ORDER),
                ]
            )
            if self.is_moe:
                parameters.append(self._categorical_parameter(AGG_FFN_MODE, _FFN_MODE_ORDER))
            return tuple(parameters)

        parameters.extend(
            [
                self._float_parameter(PREFILL_GPU_SHARE, default=0.5),
                self._discrete_parameter(PREFILL_GPUS_PER_ENGINE),
                self._discrete_parameter(DECODE_GPUS_PER_ENGINE),
                self._categorical_parameter(PREFILL_ATTENTION_MODE, _ATTENTION_MODE_ORDER),
                self._categorical_parameter(DECODE_ATTENTION_MODE, _ATTENTION_MODE_ORDER),
            ]
        )
        if self.is_moe:
            parameters.extend(
                [
                    self._categorical_parameter(PREFILL_FFN_MODE, _FFN_MODE_ORDER),
                    self._categorical_parameter(DECODE_FFN_MODE, _FFN_MODE_ORDER),
                ]
            )
        return tuple(parameters)

    def requested_features(self, params: dict[str, Any]) -> dict[str, float | str]:
        requested: dict[str, float | str] = {}
        for parameter in self.parameters:
            value = params.get(parameter.name, parameter.default)
            requested[parameter.name] = str(value) if parameter.kind == "categorical" else float(value)
        return requested

    def project(self, params: dict[str, Any], backend: str) -> ParallelProjection:
        requested = self.requested_features(params)
        if self._flat:
            index = round(float(requested[PARALLEL_CONFIG_CHOICE]))
            selected = self.branch.parallel_configs[index]
            if backend not in self.branch.supported_backends.get(selected, frozenset()):
                raise InfeasibleParallelSelection(
                    f"backend {backend!r} does not support AFD parallel config choice {index}"
                )
            return ParallelProjection(
                config=selected,
                requested_features=requested,
                actual_features={PARALLEL_CONFIG_CHOICE: float(index)},
                distance=0.0,
                mode_projected=False,
            )
        if self._independent and not self._hybrid:

            def role(prefix: str) -> ReplicaParallelConfig:
                return ReplicaParallelConfig(
                    shape=ParallelShape(
                        tp=round(float(requested[f"{prefix}tp"])),
                        pp=round(float(requested[f"{prefix}pp"])),
                        dp=round(float(requested[f"{prefix}attention_dp"])),
                        moe_tp=round(float(requested[f"{prefix}moe_tp"])),
                        moe_ep=round(float(requested[f"{prefix}moe_ep"])),
                    ),
                    replicas=round(float(requested[f"{prefix}replicas"])),
                )

            selected: ParallelConfig = (
                role("")
                if self.branch.deployment_mode == "agg"
                else DisaggParallelConfig(
                    prefill=role("prefill_"),
                    decode=role("decode_"),
                )
            )
            return ParallelProjection(
                config=selected,
                requested_features=requested,
                actual_features=dict(requested),
                distance=0.0,
                mode_projected=False,
            )
        candidates = [
            config
            for config in self.branch.parallel_configs
            if backend in self.branch.supported_backends.get(config, frozenset())
        ]
        if not candidates:
            raise InfeasibleParallelSelection(f"backend {backend!r} has no valid parallel config in this branch")

        exact_features: dict[str, float | str] = {}
        if self._hybrid:
            for role, choices in self.branch.parallel_custom_choices.items():
                name = f"{role}_{PARALLEL_CONFIG_CHOICE}"
                index = round(float(requested[name]))
                target = choices[index]
                candidates = [config for config in candidates if _parallel_role(config, role) == target]
                exact_features[name] = float(index)

            independent_roles = {name.split("_", 1)[0] for name in self.branch.parallel_independent_choices}
            for role in independent_roles:
                prefix = f"{role}_"
                target = ReplicaParallelConfig(
                    shape=ParallelShape(
                        tp=round(float(requested[f"{prefix}tp"])),
                        pp=round(float(requested[f"{prefix}pp"])),
                        dp=round(float(requested[f"{prefix}attention_dp"])),
                        moe_tp=round(float(requested[f"{prefix}moe_tp"])),
                        moe_ep=round(float(requested[f"{prefix}moe_ep"])),
                    ),
                    replicas=round(float(requested[f"{prefix}replicas"])),
                )
                candidates = [config for config in candidates if _parallel_role(config, role) == target]
                exact_features.update(
                    {
                        name: requested[name]
                        for name in self.branch.parallel_independent_choices
                        if name.startswith(prefix)
                    }
                )
            if not candidates:
                raise InfeasibleParallelSelection(
                    f"custom/independent parallelism combination is infeasible for backend {backend!r}"
                )

        categorical_names = [
            parameter.name
            for parameter in self.parameters
            if parameter.kind == "categorical" and parameter.name in self._features[candidates[0]]
        ]

        def mismatches(config: ParallelConfig) -> int:
            actual = self._features[config]
            return sum(actual[name] != requested[name] for name in categorical_names)

        min_mismatches = min(mismatches(config) for config in candidates)
        candidates = [config for config in candidates if mismatches(config) == min_mismatches]

        numeric_parameters = [
            parameter
            for parameter in self.parameters
            if parameter.kind != "categorical" and parameter.name in self._features[candidates[0]]
        ]
        backend_features = [
            self._features[config]
            for config in self.branch.parallel_configs
            if backend in self.branch.supported_backends.get(config, frozenset())
        ]

        def transformed(name: str, value: float) -> float:
            return math.log2(value) if name.endswith("num_gpus_per_engine_target") else value

        ranges: dict[str, tuple[float, float]] = {}
        for parameter in numeric_parameters:
            values = [transformed(parameter.name, float(features[parameter.name])) for features in backend_features]
            ranges[parameter.name] = (min(values), max(values))

        def numeric_distance(config: ParallelConfig) -> float:
            actual = self._features[config]
            distance = 0.0
            for parameter in numeric_parameters:
                name = parameter.name
                lower, upper = ranges[name]
                if upper == lower:
                    continue
                delta = (transformed(name, float(actual[name])) - transformed(name, float(requested[name]))) / (
                    upper - lower
                )
                distance += delta * delta
            return distance

        selected = min(
            candidates,
            key=lambda config: (numeric_distance(config), _config_key(config)),
        )
        return ParallelProjection(
            config=selected,
            requested_features=requested,
            actual_features={**exact_features, **dict(self._features[selected])},
            distance=numeric_distance(selected),
            mode_projected=min_mismatches > 0,
        )
