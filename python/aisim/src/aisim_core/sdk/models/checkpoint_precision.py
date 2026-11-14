# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer checkpoint precision for model families with mixed-precision metadata.

A model family declares its decoder layer kinds, the checkpoint modules each op prices
and where its decoder layers live in the checkpoint. This module reads the precision of
every such module per layer from ``quantized_layers``, ``config_groups`` targets and
exclusions, and returns ``{op key: {mode: number of layers}}``. Ops that share a name fold
additively, so pricing each precision bucket for its own layer count keeps the per-layer
cost exact while the op list stays one op per precision.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from fnmatch import translate
from functools import cached_property

from aisim_core.sdk import common
from aisim_core.sdk.models.helpers import (
    _iter_mixed_precision_layer_algos,
    modelopt_mixed_precision_targets,
    quant_exclude_patterns,
)
from aisim_core.sdk.perf_database import is_blackwell_system

# Decoder-layer prefixes: HF ``model``, native ``backbone``, and their language_model wrappers.
LAYER_PREFIXES = (
    "model",
    "backbone",
    "language_model.model",
    "language_model.backbone",
    "model.language_model",
)

# ``(layer, module) -> candidate checkpoint paths`` of a module inside one decoder layer.
LayerPaths = Callable[[int, str], list[str]]


def decoder_paths(sublayer: str = "") -> LayerPaths:
    """Candidate checkpoint paths of ``module`` in decoder layer ``layer``, under every known prefix.

    ``sublayer`` is the block that holds the modules, such as Nemotron-H's ``mixer``.
    """
    inner = f".{sublayer}" if sublayer else ""
    return lambda layer, module: [f"{prefix}.layers.{layer}{inner}.{module}" for prefix in LAYER_PREFIXES]


def _target_regex(targets) -> re.Pattern | None:
    """One compiled matcher for ``Linear``, ``re:`` and glob targets."""
    # ``re:`` targets keep re.match prefix semantics; globs match the whole path.
    parts = [".*" if t == "Linear" else f"(?:{t[3:]}).*" if t.startswith("re:") else translate(t) for t in targets]
    try:
        return re.compile("|".join(f"(?:{part})" for part in parts)) if parts else None
    except re.error as error:
        raise ValueError(f"Invalid quantization target in {targets!r}: {error}") from error


def _is_pattern(target: str) -> bool:
    return target == "Linear" or target.startswith("re:") or any(char in target for char in "*?[")


class CheckpointPrecision:
    """Checkpoint quantization metadata parsed once for per-layer module lookups.

    Group targets apply only without a ``quantized_layers`` listing. Exclusions win.
    """

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.listed = False
        listed_entries, group_targets = [], []
        for target, algo, listed in _iter_mixed_precision_layer_algos(raw):
            self.listed |= listed
            (listed_entries if listed else group_targets).append((target, algo))
        # A layer listing is authoritative: group targets only apply without one.
        # One algorithm string per listed path keeps the cached form small; a path
        # listed with two algorithms is rejected only if a modeled module uses it.
        self.exact, self.conflicts, patterns = {}, {}, []
        for target, algo in listed_entries if self.listed else group_targets:
            if algo is None:
                continue
            if _is_pattern(target):
                patterns.append((target, algo))
            elif self.exact.setdefault(target, algo) != algo:
                self.conflicts.setdefault(target, {self.exact[target]}).add(algo)
        self.patterns = [(_target_regex([target]), algo) for target, algo in patterns]
        # ModelOpt also resolves a module from a listed descendant, such as ``<module>.weight``
        # or ``<experts>.0.up_proj`` (ModelOptMixedPrecisionConfig._resolve_quant_algo, prefix
        # lookup, vLLM v0.31.0). Index every ancestor of a literal listed key once so a module
        # lookup stays a dict hit even for large expert listings.
        self.descendants: dict[str, set[str]] = {}
        for target, algo in self.exact.items():
            parts = target.split(".")
            for end in range(1, len(parts)):
                self.descendants.setdefault(".".join(parts[:end]), set()).add(algo)
        for target, algos in self.conflicts.items():
            parts = target.split(".")
            for end in range(1, len(parts)):
                self.descendants.setdefault(".".join(parts[:end]), set()).update(algos)
        # Same coverage as helpers._modelopt_key_excluded, precompiled for per-layer lookups:
        # a bare module path also excludes its submodules.
        exclusions = quant_exclude_patterns(raw)
        self.excluded_exact = {t for t in exclusions if not _is_pattern(t)}
        self.excluded_regex = _target_regex([t for t in exclusions if _is_pattern(t)])

    def module(self, paths: Sequence[str]):
        """``(excluded, algos)`` of one module given its candidate checkpoint paths."""
        if any(self._excluded(path) for path in paths):
            return True, set()
        algos = {self.exact[path] for path in paths if path in self.exact}
        algos.update(algo for path in paths for algo in self.conflicts.get(path, ()))
        if not algos:
            # An exact listing wins; otherwise a listed descendant declares the module's precision.
            algos.update(algo for path in paths for algo in self.descendants.get(path, ()))
        algos.update(algo for regex, algo in self.patterns if any(regex.fullmatch(path) for path in paths))
        return False, algos

    def _excluded(self, path: str) -> bool:
        parts = path.split(".")
        if any(".".join(parts[:end]) in self.excluded_exact for end in range(1, len(parts) + 1)):
            return True
        return bool(self.excluded_regex and self.excluded_regex.fullmatch(path))

    def rows(
        self,
        layer_kinds: Sequence,
        kinds_to_modules: Mapping,
        layer_paths: LayerPaths,
        resolve: Callable[[str | None, bool], object],
    ) -> list:
        """``[(layer, kind, key, mode)]`` for every modeled op of every decoder layer.

        ``layer_kinds[i]`` is layer ``i``'s kind; ``kinds_to_modules[kind]`` maps an op key to
        the checkpoint modules it prices. ``resolve(algo or None, excluded)`` is a module's
        effective mode, so fused modules are compared after exclusion and unlisted rules
        apply. Conflicting algorithms for one module, or fused modules whose modes differ,
        are rejected.
        """
        rows = []
        for layer, kind in enumerate(layer_kinds):
            for key, modules in kinds_to_modules.get(kind, {}).items():
                modes = set()
                for module in modules:
                    excluded, algos = self.module(layer_paths(layer, module))
                    if len(algos) > 1:
                        raise ValueError(
                            f"Conflicting checkpoint precisions for module {module} in layer {layer}: {algos}"
                        )
                    modes.add(resolve(next(iter(algos), None), excluded))
                if len(modes) > 1:
                    raise ValueError(f"Fused modules {modules} in layer {layer} disagree on precision: {modes}")
                rows.append((layer, kind, key, modes.pop()))
        return rows

    @cached_property
    def algos(self) -> set[str]:
        """Every algorithm the checkpoint lists or targets."""
        return {*self.exact.values(), *(algo for _, algo in self.patterns)}

    @cached_property
    def targets_only(self) -> bool:
        return self.listed and modelopt_mixed_precision_targets(self.raw) is not None


_CHECKPOINT_CACHE: OrderedDict[int, tuple[dict, CheckpointPrecision]] = OrderedDict()


def checkpoint_precision(raw: dict) -> CheckpointPrecision:
    """Parse a checkpoint once per loaded config object; model construction repeats in sweeps."""
    entry = _CHECKPOINT_CACHE.get(id(raw))
    if entry is not None and entry[0] is raw:
        _CHECKPOINT_CACHE.move_to_end(id(raw))
        return entry[1]
    checkpoint = CheckpointPrecision(raw)
    _CHECKPOINT_CACHE[id(raw)] = (raw, checkpoint)
    if len(_CHECKPOINT_CACHE) > 8:
        _CHECKPOINT_CACHE.popitem(last=False)
    return checkpoint


def _nvfp4_lane(system: str | None, systems_path: str | None, native, weight_only):
    # Non-Blackwell systems have no native FP4 tensor cores: NVFP4 runs weight-only, as
    # resolve_nvfp4_for_system does for the global mode. Without a system, keep native NVFP4.
    return weight_only if system and not is_blackwell_system(system, systems_path) else native


def projection_modes(
    checkpoint: CheckpointPrecision,
    layer_kinds: Sequence,
    kinds_to_modules: Mapping,
    layer_paths: LayerPaths,
    default: common.GEMMQuantMode,
    system: str | None = None,
    systems_path: str | None = None,
) -> dict:
    """``{key: {mode: number of layers}}`` for every modeled GEMM projection of every decoder layer.

    Layers with different precisions get one bucket each. A projection that a listing leaves
    out is BF16 only for a targets-only ModelOpt MIXED_PRECISION export, the contract vLLM and
    SGLang implement (see ``modelopt_mixed_precision_targets``); other metadata keeps
    ``default``. An excluded module is always BF16. Unsupported algorithms are rejected.
    """
    gemm = common.GEMMQuantMode
    modes = {
        # shortcut: FP8 is dynamic only when the global mode already is; per-layer activation
        # scaling is not read from the checkpoint. Upgrade if an export mixes static and dynamic FP8.
        "fp8": gemm.fp8 if default == gemm.fp8 else gemm.fp8_static,
        # MXFP8 takes the fp8_block lane, as the global mode does (helpers._mixed_precision_algo_gemm_mode).
        "mxfp8": gemm.fp8_block,
        "fp8_block": gemm.fp8_block,
        "nvfp4": _nvfp4_lane(system, systems_path, gemm.nvfp4, gemm.nvfp4_wo),
        "w4a16_nvfp4": gemm.w4a16_nvfp4,
    }
    unlisted = gemm.bfloat16 if checkpoint.targets_only else default

    def resolve(algo, excluded):
        if excluded:
            return gemm.bfloat16
        if algo is None:
            return unlisted
        if algo not in modes:
            raise ValueError(f"Unsupported mixed precision for a projection: {algo}")
        return modes[algo]

    resolved = {}
    for _, _, key, mode in checkpoint.rows(layer_kinds, kinds_to_modules, layer_paths, resolve):
        resolved.setdefault(key, Counter())[mode] += 1
    return {key: dict(counts) for key, counts in resolved.items()}


def moe_modes(
    checkpoint: CheckpointPrecision,
    layer_kinds: Sequence,
    expert_modules: Mapping,
    layer_paths: LayerPaths,
    default: common.MoEQuantMode,
    system: str | None = None,
    execution_mode: Callable = lambda mode: mode,
    systems_path: str | None = None,
) -> dict:
    """``{mode: number of MoE layers}`` of the routed experts, or ``{}`` when every layer is ``default``.

    ``expert_modules[kind]`` lists the checkpoint modules that carry a MoE layer's routed-expert
    precision. ``execution_mode`` applies the same backend lane mapping as the global mode.
    NVFP4 layers then take the system's lane even when ``default`` is still the unresolved
    native mode, as on the canonical API path. Unlisted experts are BF16 only in a targets-only
    export; an excluded layer is always BF16.

    shortcut: a layer's precision is read from the declared expert modules (expert 0);
    per-expert precision within one layer is not modeled. Upgrade if an export quantizes
    experts of a layer differently.
    """
    moe = common.MoEQuantMode
    nvfp4 = _nvfp4_lane(system, systems_path, moe.nvfp4, moe.nvfp4_wo)
    modes = {
        "fp8": moe.fp8,
        "mxfp8": moe.fp8_block,
        "fp8_block": moe.fp8_block,
        "nvfp4": nvfp4,
        "w4a16_nvfp4": moe.w4a16_nvfp4,
    }
    unlisted = moe.bfloat16 if checkpoint.targets_only else default

    def resolve(algo, excluded):
        if excluded:
            return moe.bfloat16
        if algo is not None and algo not in modes:
            raise ValueError(f"Unsupported routed-expert precision: {algo}")
        mode = execution_mode(modes[algo]) if algo else unlisted
        return nvfp4 if mode in (moe.nvfp4, moe.nvfp4_wo) else mode

    experts = {kind: {"experts": modules} for kind, modules in expert_modules.items()}
    resolved = Counter(mode for *_, mode in checkpoint.rows(layer_kinds, experts, layer_paths, resolve))
    return {} if set(resolved) <= {default} else dict(resolved)
