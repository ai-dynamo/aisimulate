# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from fnmatch import translate
from functools import cached_property

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.models.base import BaseModel, register_model
from aisimulate_core.sdk.models.helpers import (
    _iter_mixed_precision_layer_algos,
    modelopt_mixed_precision_targets,
    mtp_scale_factor,
    quant_exclude_patterns,
    resolve_vllm_moe_execution_mode,
)
from aisimulate_core.sdk.perf_database import is_blackwell_system

# Decoder-layer prefixes: HF ``model``, native ``backbone``, and their language_model wrappers.
_LAYER_PREFIXES = (
    "model",
    "backbone",
    "language_model.model",
    "language_model.backbone",
    "model.language_model",
)


# Checkpoint module suffixes priced by each decoder GEMM, by hybrid layer kind. A fused
# GEMM (QKV) lists every module it fuses; they must share one precision.
_LAYER_PROJECTIONS = {
    "M": {"in_proj": ("in_proj",), "out_proj": ("out_proj",)},
    "*": {"qkv": ("q_proj", "k_proj", "v_proj"), "o_proj": ("o_proj",)},
    "E": {
        "shared_experts.up_proj": ("shared_experts.up_proj",),
        "shared_experts.down_proj": ("shared_experts.down_proj",),
        "fc1_latent_proj": ("fc1_latent_proj",),
        "fc2_latent_proj": ("fc2_latent_proj",),
    },
    "-": {"up_proj": ("up_proj",), "down_proj": ("down_proj",)},
}


def _target_regex(targets) -> re.Pattern | None:
    """One compiled matcher for ``Linear``, ``re:`` and glob targets."""
    # ``re:`` targets keep re.match prefix semantics; globs match the whole path.
    parts = [".*" if t == "Linear" else f"(?:{t[3:]}).*" if t.startswith("re:") else translate(t) for t in targets]
    try:
        return re.compile("|".join(f"(?:{part})" for part in parts)) if parts else None
    except re.error as error:
        raise ValueError(f"Invalid quantization target in {targets!r}: {error}") from error


class _CheckpointPrecision:
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
        # Same coverage as helpers._modelopt_key_excluded, precompiled for per-layer lookups:
        # a bare module path also excludes its submodules.
        exclusions = quant_exclude_patterns(raw)
        self.excluded_exact = {t for t in exclusions if not _is_pattern(t)}
        self.excluded_regex = _target_regex([t for t in exclusions if _is_pattern(t)])

    def module(self, layer: int, module: str):
        """``(excluded, algos)`` of one decoder module across the known layer prefixes."""
        paths = [f"{prefix}.layers.{layer}.mixer.{module}" for prefix in _LAYER_PREFIXES]
        if any(self._excluded(path) for path in paths):
            return True, set()
        algos = {self.exact[path] for path in paths if path in self.exact}
        algos.update(algo for path in paths for algo in self.conflicts.get(path, ()))
        algos.update(algo for regex, algo in self.patterns if any(regex.fullmatch(path) for path in paths))
        return False, algos

    def _excluded(self, path: str) -> bool:
        parts = path.split(".")
        if any(".".join(parts[:end]) in self.excluded_exact for end in range(1, len(parts) + 1)):
            return True
        return bool(self.excluded_regex and self.excluded_regex.fullmatch(path))

    def rows(self, pattern: str, kinds_to_modules: dict) -> list:
        """``[(projection, algo or None, excluded)]`` per decoder layer.

        Conflicting algorithms for one module, or fused modules that disagree, are rejected.
        """
        rows = []
        for layer, kind in enumerate(pattern):
            for projection, modules in kinds_to_modules.get(kind, {}).items():
                states = set()
                for module in modules:
                    excluded, algos = self.module(layer, module)
                    if len(algos) > 1:
                        raise ValueError(
                            f"Conflicting checkpoint precisions for Nemotron-H module {module} "
                            f"in layer {layer}: {algos}"
                        )
                    states.add("excluded" if excluded else next(iter(algos), None))
                if len(states) > 1:
                    raise ValueError(f"Fused Nemotron-H modules {modules} in layer {layer} disagree on precision")
                state = states.pop()
                rows.append((projection, None, True) if state == "excluded" else (projection, state, False))
        return rows

    @cached_property
    def targets_only(self) -> bool:
        return self.listed and modelopt_mixed_precision_targets(self.raw) is not None


_CHECKPOINT_CACHE: OrderedDict[int, tuple[dict, _CheckpointPrecision]] = OrderedDict()


def _checkpoint_precision(raw: dict) -> _CheckpointPrecision:
    """Parse a checkpoint once per loaded config object; model construction repeats in sweeps."""
    entry = _CHECKPOINT_CACHE.get(id(raw))
    if entry is not None and entry[0] is raw:
        _CHECKPOINT_CACHE.move_to_end(id(raw))
        return entry[1]
    checkpoint = _CheckpointPrecision(raw)
    _CHECKPOINT_CACHE[id(raw)] = (raw, checkpoint)
    if len(_CHECKPOINT_CACHE) > 8:
        _CHECKPOINT_CACHE.popitem(last=False)
    return checkpoint


def _is_pattern(target: str) -> bool:
    return target == "Linear" or target.startswith("re:") or any(char in target for char in "*?[")


def _projection_modes(
    raw: dict,
    pattern: str,
    default: common.GEMMQuantMode,
    system: str | None = None,
    checkpoint=None,
) -> dict:
    """Resolve every decoder GEMM projection from checkpoint metadata.

    Returns ``{projection: {mode: number of layers}}`` for Mamba in/out, attention QKV
    and output, shared-expert, latent and dense MLP projections. Layers with different
    precisions get one operation each, so per-layer precision is kept exactly.
    Unsupported algorithms are rejected.

    Metadata without per-layer information leaves every projection on ``default``.

    A projection that a listing leaves out is BF16 only for a targets-only
    ModelOpt MIXED_PRECISION export, the contract vLLM and SGLang implement
    (see ``modelopt_mixed_precision_targets``). Any other metadata keeps
    ``default`` for an unlisted projection instead of guessing a mode.
    """
    # Non-Blackwell systems have no native FP4 tensor cores: NVFP4 runs weight-only, as
    # resolve_nvfp4_for_system does for the global mode. Without a system, keep native NVFP4.
    nvfp4 = common.GEMMQuantMode.nvfp4_wo if system and not is_blackwell_system(system) else common.GEMMQuantMode.nvfp4
    # shortcut: FP8 is dynamic only when the global mode already is; per-layer activation
    # scaling is not read from the checkpoint. Upgrade if an export mixes static and dynamic FP8.
    modes = {
        "fp8": common.GEMMQuantMode.fp8 if default == common.GEMMQuantMode.fp8 else common.GEMMQuantMode.fp8_static,
        "mxfp8": common.GEMMQuantMode.fp8_block,
        "nvfp4": nvfp4,
        # vLLM runs W4A16_NVFP4 linears weight-only (FP4 Marlin, BF16 activations).
        "w4a16_nvfp4": common.GEMMQuantMode.w4a16_nvfp4,
    }
    checkpoint = checkpoint or _checkpoint_precision(raw)
    rows = checkpoint.rows(pattern, _LAYER_PROJECTIONS)
    unlisted = common.GEMMQuantMode.bfloat16 if checkpoint.targets_only else default
    resolved = {}
    for projection, algo, excluded in rows:
        if algo is not None and algo not in modes:
            raise ValueError(f"Unsupported mixed precision for Nemotron-H projection {projection}: {algo}")
        mode = common.GEMMQuantMode.bfloat16 if excluded else modes[algo] if algo else unlisted
        resolved.setdefault(projection, Counter())[mode] += 1
    return {projection: dict(counts) for projection, counts in resolved.items()}


def _moe_modes(
    raw: dict,
    pattern: str,
    default: common.MoEQuantMode,
    system: str | None = None,
    checkpoint=None,
    execution_mode=lambda mode: mode,
) -> dict:
    """Resolve the routed-expert precision of each MoE layer from checkpoint metadata.

    Returns ``{mode: number of MoE layers}`` unless every MoE layer resolves to ``default``,
    the system- and backend-resolved global mode.
    ``execution_mode`` applies the same backend lane mapping as the global mode.

    shortcut: a layer's precision is read from expert 0; per-expert precision within
    one layer is not modeled. Upgrade if an export quantizes experts of a layer differently.
    """
    nvfp4 = common.MoEQuantMode.nvfp4_wo if system and not is_blackwell_system(system) else common.MoEQuantMode.nvfp4
    modes = {
        "fp8": common.MoEQuantMode.fp8,
        "mxfp8": common.MoEQuantMode.fp8_block,
        "nvfp4": nvfp4,
        "w4a16_nvfp4": common.MoEQuantMode.w4a16_nvfp4,
    }
    checkpoint = checkpoint or _checkpoint_precision(raw)
    rows = checkpoint.rows(pattern, {"E": {"experts": ("experts.0.up_proj", "experts.0.down_proj")}})
    # Unlisted experts are BF16 only in a targets-only export; otherwise they keep the global mode.
    unlisted = common.MoEQuantMode.bfloat16 if checkpoint.targets_only else default
    # Keep the system-resolved global mode for the precision it already prices.
    family = {common.MoEQuantMode.nvfp4_wo: common.MoEQuantMode.nvfp4}
    resolved = Counter()
    for _, algo, excluded in rows:
        if algo is not None and algo not in modes:
            raise ValueError(f"Unsupported routed-expert precision for Nemotron-H: {algo}")
        # An explicit exclusion is BF16 in every metadata format.
        mode = common.MoEQuantMode.bfloat16 if excluded else execution_mode(modes[algo]) if algo else unlisted
        resolved[default if family.get(mode, mode) == family.get(default, default) else mode] += 1
    return {} if set(resolved) <= {default} else dict(resolved)


@register_model("NEMOTRONH")
class NemotronHModel(BaseModel):
    """
    NemotronH hybrid model implementation (Mamba + MoE + Transformer).

    This model supports the hybrid architecture where each layer can be one of:
    - 'M': Mamba2 layer (state-space model)
    - 'E': MoE layer (Mixture of Experts with shared expert)
    - '*': Transformer layer (standard attention)
    - '-': MLP layer (dense feed-forward)

    The layer sequence is defined by the `hybrid_override_pattern` string in NemotronHConfig.
    """

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        model = cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            backend_name=backend_name,
        )
        # extra_params is NemotronHConfig with hybrid layer configuration. Checkpoint metadata
        # prices each projection and routed-expert layer at its own precision; without
        # per-layer metadata every projection keeps the global mode.
        raw = model_info.get("raw_config", {})
        pattern = model_info["extra_params"].hybrid_override_pattern
        checkpoint = _checkpoint_precision(raw)
        if not model_info["gemm_quant_mode_is_explicit"]:
            model._projection_quant_modes = _projection_modes(
                raw, pattern, model_config.gemm_quant_mode, model_config.system, checkpoint
            )
        if not model_info["moe_quant_mode_is_explicit"]:
            model._moe_quant_modes = _moe_modes(
                raw,
                pattern,
                model_config.moe_quant_mode,
                model_config.system,
                checkpoint,
                lambda mode: resolve_vllm_moe_execution_mode(mode, backend_name, model_info["architecture"]),
            )
        model.set_hybrid_config(model_info["extra_params"])
        return model

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)
        # Framework backend, threaded to MoEDispatch (its comm flavor is a
        # constructor field since the pyo3 op unification).
        self._backend_name = backend_name
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._hybrid_config: common.NemotronHConfig | None = None
        self._projection_quant_modes: dict[str, dict[common.GEMMQuantMode, int]] = {}
        self._moe_quant_modes: dict[common.MoEQuantMode, int] = {}
        self._power_law_alpha = 1.01  # follow DeepSeek MoE
        # MTP (num_nextn_predict_layers > 0): scale generation iteration cost by
        # the small extra-layer factor (nextn+L)/L. Accepted-token progress is
        # applied by the upper prediction layer. nextn == 0 -> factor 1.0.
        #
        # APPROXIMATION (intentional, hybrid architecture). NemotronH is a mixed
        # Mamba / attention / MoE stack (e.g. Ultra-550B = 48 mamba + 48 moe + 12
        # attention over 108 layers) and its MTP block is attention+moe -- NOT the
        # dominant Mamba layer type. DeepSeek's uniform (nextn+L)/L is exact there only
        # because every layer is homogeneous (attn+MoE). Here we still spread that term
        # uniformly over all layer-type groups, so the extra MTP-block cost is slightly
        # mis-attributed: we add a little fake Mamba and under-count the real attn+moe
        # MTP block. This is bounded by ~1/L of one layer's cost (<1% of TPOT for these
        # deep models). Not worth modeling an explicit attn+moe MTP block for
        # sub-1% gain.
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)

    def set_hybrid_config(self, hybrid_config: common.NemotronHConfig) -> None:
        """
        Set the hybrid layer configuration and build operation pipelines.

        Args:
            hybrid_config: NemotronHConfig containing hybrid_override_pattern and layer parameters
        """
        self._hybrid_config = hybrid_config
        self._build_context_ops()
        self._build_generation_ops()

    def _projection_gemms(self, name: str, suffix: str, count, n: int, k: int, quant_mode, **kwargs) -> list:
        """One GEMM per checkpoint precision of a projection, each priced for its own layers.

        Ops that share a name fold additively, so the buckets sum to the per-layer cost.
        """
        buckets = self._projection_quant_modes.get(suffix)
        if not buckets:
            return [ops.GEMM(name, count, n, k, quant_mode, **kwargs)]
        total = sum(buckets.values())
        return [
            ops.GEMM(name, count if layers == total else count * layers / total, n, k, mode, **kwargs)
            for mode, layers in buckets.items()
        ]

    def _moe_buckets(self, count, moe_quant_mode) -> list:
        """``(mode, count)`` per routed-expert precision; one bucket keeps the global mode."""
        buckets = self._moe_quant_modes
        if not buckets:
            return [(moe_quant_mode, count)]
        total = sum(buckets.values())
        return [(mode, count * layers / total) for mode, layers in buckets.items()]

    def _count_layer_types(self) -> dict[str, int]:
        """Count occurrences of each layer type in the pattern."""
        pattern = self._hybrid_config.hybrid_override_pattern
        return {
            "M": pattern.count("M"),
            "E": pattern.count("E"),
            "*": pattern.count("*"),
            "-": pattern.count("-"),
        }

    def get_kvcache_elements_per_token(self) -> int:
        # Mamba state is fixed per request and must be budgeted separately.
        tp = self.config.tp_size
        kv_heads = (self._num_kv_heads + tp - 1) // tp
        return self._count_layer_types()["*"] * 2 * kv_heads * self._head_size

    def _build_context_ops(self) -> None:
        """Build the context (prefill) operations pipeline based on hybrid pattern."""
        if not self._hybrid_config:
            return

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        layer_counts = self._count_layer_types()
        cfg = self._hybrid_config

        # Use base model parameters for standard fields
        num_kv_heads_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size

        self.context_ops = []

        # Embedding
        self.context_ops.append(ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3))

        # Mamba layers (M): norm, in_proj GEMM, conv1d, ssm, out_proj GEMM, ar
        if layer_counts["M"] > 0:
            count = layer_counts["M"]
            nheads_per_gpu = cfg.mamba_num_heads // tp_size
            d_inner_per_gpu = nheads_per_gpu * cfg.mamba_head_dim
            n_groups_per_gpu = max(cfg.n_groups // tp_size, 1)
            in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * cfg.ssm_state_size + nheads_per_gpu
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mamba_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "context_mamba_in_proj_gemm",
                        "in_proj",
                        count,
                        in_proj_out_per_gpu,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.Mamba2Kernel(
                        "context_mamba_conv1d",
                        count,
                        "causal_conv1d_fn",
                        "context",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.Mamba2Kernel(
                        "context_mamba_ssm",
                        count,
                        "mamba_chunk_scan_combined",
                        "context",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    *self._projection_gemms(
                        "context_mamba_out_proj_gemm",
                        "out_proj",
                        count,
                        h,
                        d_inner_per_gpu,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("context_mamba_ar", count, h, tp_size),
                ]
            )

        # Transformer layers (*)
        if layer_counts["*"] > 0:
            count = layer_counts["*"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_attn_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "context_qkv_gemm",
                        "qkv",
                        count,
                        self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ContextAttention(
                        "context_attention",
                        count,
                        self._num_heads // tp_size,
                        num_kv_heads_per_gpu,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        head_size=self._head_size,
                    ),
                    *self._projection_gemms(
                        "context_proj_gemm",
                        "o_proj",
                        count,
                        h,
                        self._num_heads * self._head_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce("context_attn_ar", count, h, tp_size),
                ]
            )

        # MoE layers (E)
        if layer_counts["E"] > 0:
            count = layer_counts["E"]
            # Latent-MoE (e.g. Nemotron-3-Super): experts run on a compressed
            # `moe_latent_size` projection of hidden_states. Shared expert and the
            # router still operate on hidden_size.
            moe_h = cfg.moe_latent_size if cfg.moe_latent_size > 0 else h
            # Pre-norm for MoE
            self.context_ops.append(ops.ElementWise("context_moe_norm", count, 2 * h, 2 * h, 0.8))

            # Shared expert (always runs in parallel)
            # NemotronH uses simple MLP (not gated): up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.context_ops.extend(
                [
                    *self._projection_gemms(
                        "context_shared_up_gemm",
                        "shared_experts.up_proj",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "context_shared_relu2",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        0.8,
                    ),
                    *self._projection_gemms(
                        "context_shared_down_gemm",
                        "shared_experts.down_proj",
                        count,
                        h,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                ]
            )

            # Router GEMM
            self.context_ops.append(
                ops.GEMM(
                    "context_router_gemm",
                    count,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            # MoE dispatch and compute
            # When latent-MoE is active, fc1_latent_proj (h -> latent) runs before
            # dispatch, so dispatch/experts/combine all communicate in latent dim.
            if cfg.moe_latent_size > 0:
                self.context_ops.extend(
                    self._projection_gemms(
                        "context_fc1_latent_proj_gemm",
                        "fc1_latent_proj",
                        count,
                        cfg.moe_latent_size // tp_size,
                        h,
                        gemm_quant_mode,
                    )
                )
            for moe_mode, moe_count in self._moe_buckets(count, moe_quant_mode):
                self.context_ops.extend(
                    [
                        ops.MoEDispatch(
                            "context_moe_pre_dispatch",
                            moe_count,
                            moe_h,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            attention_dp_size,
                            True,
                            quant_mode=moe_mode,
                            backend=self._backend_name,
                        ),
                        ops.MoE(
                            "context_moe",
                            moe_count,
                            moe_h,
                            self._moe_inter_size,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            moe_mode,
                            workload_distribution,
                            attention_dp_size,
                            moe_kernel_source=self.config.moe_kernel_source,
                            is_gated=False,  # NemotronH uses Relu2 (non-gated)
                        ),
                        ops.MoEDispatch(
                            "context_moe_post_dispatch",
                            moe_count,
                            moe_h,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            attention_dp_size,
                            False,
                            quant_mode=moe_mode,
                            backend=self._backend_name,
                        ),
                    ]
                )
            if cfg.moe_latent_size > 0:
                # fc2_latent_proj (latent -> h) runs after expert combine,
                # before the post-MoE allreduce. Modeled as a row-parallel GEMM
                # whose partial sums are aggregated by the following AllReduce.
                self.context_ops.extend(
                    self._projection_gemms(
                        "context_fc2_latent_proj_gemm",
                        "fc2_latent_proj",
                        count,
                        h,
                        cfg.moe_latent_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    )
                )
            # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
            self.context_ops.append(ops.CustomAllReduce("context_moe_ar", count, h, tp_size))

        # MLP layers (-) - not present in Nemotron-3 Nano but in NemotronH model
        if layer_counts["-"] > 0:
            count = layer_counts["-"]
            # NemotronH MLP is non-gated: up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mlp_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "context_mlp_up_gemm",
                        "up_proj",
                        count,
                        self._inter_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "context_mlp_relu2",
                        count,
                        self._inter_size // tp_size,
                        self._inter_size // tp_size,
                        0.8,
                    ),
                    *self._projection_gemms(
                        "context_mlp_down_gemm",
                        "down_proj",
                        count,
                        h,
                        self._inter_size // tp_size,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("context_mlp_ar", count, h, tp_size),
                ]
            )

        # P2P communication for PP
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))

        # Logits GEMM
        self.context_ops.append(
            ops.GEMM(
                "context_logits_gemm",
                1,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

    def _build_generation_ops(self) -> None:
        """Build the generation (decoding) operations pipeline based on hybrid pattern."""
        if not self._hybrid_config:
            return

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        layer_counts = self._count_layer_types()
        cfg = self._hybrid_config

        # Use base model parameters for standard fields
        num_kv_heads_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size

        self.generation_ops = []

        # Embedding
        mtp = self._mtp_scale_factor
        self.generation_ops.append(ops.Embedding("generation_embedding", 1 * mtp, self._vocab_size, h, 0.3))

        # Mamba layers (M): norm, in_proj GEMM, conv1d, ssm, out_proj GEMM, ar
        if layer_counts["M"] > 0:
            count = layer_counts["M"] * mtp
            nheads_per_gpu = cfg.mamba_num_heads // tp_size
            d_inner_per_gpu = nheads_per_gpu * cfg.mamba_head_dim
            n_groups_per_gpu = max(cfg.n_groups // tp_size, 1)
            in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * cfg.ssm_state_size + nheads_per_gpu
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mamba_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "generation_mamba_in_proj_gemm",
                        "in_proj",
                        count,
                        in_proj_out_per_gpu,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.Mamba2Kernel(
                        "generation_mamba_conv1d",
                        count,
                        "causal_conv1d_update",
                        "generation",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.Mamba2Kernel(
                        "generation_mamba_ssm",
                        count,
                        "selective_state_update",
                        "generation",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    *self._projection_gemms(
                        "generation_mamba_out_proj_gemm",
                        "out_proj",
                        count,
                        h,
                        d_inner_per_gpu,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("generation_mamba_ar", count, h, tp_size),
                ]
            )

        # Transformer layers (*)
        if layer_counts["*"] > 0:
            count = layer_counts["*"] * mtp
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_attn_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "generation_qkv_gemm",
                        "qkv",
                        count,
                        self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.GenerationAttention(
                        "generation_attention",
                        count,
                        self._num_heads // tp_size,
                        num_kv_heads_per_gpu,
                        kvcache_quant_mode,
                        head_size=self._head_size,
                    ),
                    *self._projection_gemms(
                        "generation_proj_gemm",
                        "o_proj",
                        count,
                        h,
                        self._num_heads * self._head_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce("generation_attn_ar", count, h, tp_size),
                ]
            )

        # MoE layers (E)
        if layer_counts["E"] > 0:
            count = layer_counts["E"] * mtp
            # Latent-MoE (e.g. Nemotron-3-Super): experts run on a compressed
            # `moe_latent_size` projection of hidden_states. Shared expert and the
            # router still operate on hidden_size.
            moe_h = cfg.moe_latent_size if cfg.moe_latent_size > 0 else h
            # Pre-norm for MoE
            self.generation_ops.append(ops.ElementWise("generation_moe_norm", count, 2 * h, 2 * h, 0.8))

            # Shared expert (always runs in parallel)
            # NemotronH uses simple MLP (not gated): up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.generation_ops.extend(
                [
                    *self._projection_gemms(
                        "generation_shared_up_gemm",
                        "shared_experts.up_proj",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "generation_shared_relu2",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        0.8,
                    ),
                    *self._projection_gemms(
                        "generation_shared_down_gemm",
                        "shared_experts.down_proj",
                        count,
                        h,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                ]
            )

            # Router GEMM
            self.generation_ops.append(
                ops.GEMM(
                    "generation_router_gemm",
                    count,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            # MoE dispatch and compute
            # When latent-MoE is active, fc1_latent_proj (h -> latent) runs before
            # dispatch, so dispatch/experts/combine all communicate in latent dim.
            if cfg.moe_latent_size > 0:
                self.generation_ops.extend(
                    self._projection_gemms(
                        "generation_fc1_latent_proj_gemm",
                        "fc1_latent_proj",
                        count,
                        cfg.moe_latent_size // tp_size,
                        h,
                        gemm_quant_mode,
                    )
                )
            for moe_mode, moe_count in self._moe_buckets(count, moe_quant_mode):
                self.generation_ops.extend(
                    [
                        ops.MoEDispatch(
                            "generation_moe_pre_dispatch",
                            moe_count,
                            moe_h,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            attention_dp_size,
                            True,
                            quant_mode=moe_mode,
                            backend=self._backend_name,
                        ),
                        ops.MoE(
                            "generation_moe",
                            moe_count,
                            moe_h,
                            self._moe_inter_size,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            moe_mode,
                            workload_distribution,
                            attention_dp_size,
                            moe_kernel_source=self.config.moe_kernel_source,
                            is_gated=False,  # NemotronH uses Relu2 (non-gated)
                        ),
                        ops.MoEDispatch(
                            "generation_moe_post_dispatch",
                            moe_count,
                            moe_h,
                            self._topk,
                            self._num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            attention_dp_size,
                            False,
                            quant_mode=moe_mode,
                            backend=self._backend_name,
                        ),
                    ]
                )
            if cfg.moe_latent_size > 0:
                # fc2_latent_proj (latent -> h) runs after expert combine,
                # before the post-MoE allreduce. Modeled as a row-parallel GEMM
                # whose partial sums are aggregated by the following AllReduce.
                self.generation_ops.extend(
                    self._projection_gemms(
                        "generation_fc2_latent_proj_gemm",
                        "fc2_latent_proj",
                        count,
                        h,
                        cfg.moe_latent_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    )
                )
            # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
            self.generation_ops.append(ops.CustomAllReduce("generation_moe_ar", count, h, tp_size))

        # MLP layers (-)
        if layer_counts["-"] > 0:
            count = layer_counts["-"] * mtp
            # NemotronH MLP is non-gated: up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mlp_norm", count, 2 * h, 2 * h, 0.8),
                    *self._projection_gemms(
                        "generation_mlp_up_gemm",
                        "up_proj",
                        count,
                        self._inter_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "generation_mlp_relu2",
                        count,
                        self._inter_size // tp_size,
                        self._inter_size // tp_size,
                        0.8,
                    ),
                    *self._projection_gemms(
                        "generation_mlp_down_gemm",
                        "down_proj",
                        count,
                        h,
                        self._inter_size // tp_size,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("generation_mlp_ar", count, h, tp_size),
                ]
            )

        # P2P communication for PP
        pp_scale_factor = pp_size - 1
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * mtp, h, pp_size))

        # Logits GEMM
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * mtp,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )
