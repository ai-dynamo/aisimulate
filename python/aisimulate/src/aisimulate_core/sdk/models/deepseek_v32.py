# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Dense/decode composition and exclusion matching adapt (with modifications) SGLang:
# https://gitlab-master.nvidia.com/dl/sglang/sglang/-/tree/02c5a855aceb968c310e6fbc6632270e26edc84b/python/sglang/srt
# The vLLM copyright is retained from SGLang's deepseek_v2.py, which cites:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py

from __future__ import annotations

import logging
import re
from collections import Counter

import aisimulate_core.sdk.operations as ops
from aisimulate_core.sdk import common
from aisimulate_core.sdk.models.base import BaseModel, register_model
from aisimulate_core.sdk.models.blocks.moe import MoEBlockShape
from aisimulate_core.sdk.models.helpers import (
    attention_modules_excluded_from_quant,
    attention_projection_exclusions,
    build_large_ep_moe_ops,
    large_ep_gpus_per_node,
    mtp_scale_factor,
    quant_exclude_patterns,
    validate_trtllm_large_ep,
)

logger = logging.getLogger(__name__)


def _dsa_full_layer_fraction(raw_config: dict, num_layers: int) -> float:
    """Fraction of DSA layers that COMPUTE the indexer (full) vs reuse a shared
    topk index (skip). Replicates sglang ``dsa_layer_skips_topk``: a layer skips
    when ``index_topk_pattern[lid]=='S'``, else (with explicit offset)
    ``max(lid - offset + 1, 0) % freq != 0`` or (no offset) ``max(lid-1,0)%freq``.
    GLM-5.2:
    freq=4, offset=3, 78 layers -> 21 full / 57 skip = 0.2692 (NOT 1/freq=0.25 —
    layers 0..2 are full and the periodic pattern starts at the offset). Returns
    1.0 when freq<=1 / no skipping (DeepSeek-V3.2 / GLM-5)."""
    freq = int(raw_config.get("index_topk_freq", 1) or 1)
    pattern = raw_config.get("index_topk_pattern")
    offset = raw_config.get("index_skip_topk_offset")
    if freq <= 1 and not pattern:
        return 1.0

    def _skips(lid: int) -> bool:
        if pattern is not None:
            return lid < len(pattern) and pattern[lid] == "S"
        # Match sglang dsa_layer_skips_topk EXACTLY: with an explicit offset use
        # max(lid-offset+1,0)%freq; with no offset the default is max(lid-1,0)%freq
        # (NOT offset=1 — that would be max(lid,0)). GLM-5.2 sets offset=3.
        if offset is not None:
            return max(lid - offset + 1, 0) % freq != 0
        return max(lid - 1, 0) % freq != 0

    n_full = sum(1 for lid in range(int(num_layers)) if not _skips(lid))
    return n_full / int(num_layers) if num_layers else 1.0


def _quant_exclude_patterns(raw_config: dict) -> list:
    """All module-exclusion globs a ModelOpt/HF quant config can carry."""
    return quant_exclude_patterns(raw_config)


def _dsa_attention_modules_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps DSA attention projections unquantized."""
    return attention_modules_excluded_from_quant(raw_config)


def _shared_experts_excluded_from_quant(raw_config: dict) -> bool:
    """Return whether a GLM/DSA checkpoint keeps the MoE shared experts unquantized.

    nvidia/GLM-5.2-NVFP4 excludes every ``model.layers.N.mlp.shared_experts*`` from
    NVFP4 (shared experts stay bf16; only the routed experts are quantized), so the
    shared-expert GEMMs must be modeled at bf16, not the global gemm_quant_mode."""
    return any("shared_expert" in str(pattern) for pattern in _quant_exclude_patterns(raw_config))


def _dsa_gemm_quant_mode(extra_params: object, fallback: common.GEMMQuantMode) -> common.GEMMQuantMode:
    if isinstance(extra_params, dict):
        return extra_params.get("dsa_gemm_quant_mode", fallback)
    return fallback


def _dsa_attention_quant_modes(
    extra_params: object, fallback: common.GEMMQuantMode
) -> tuple[dict, common.GEMMQuantMode]:
    """Per-projection quant modes and the single module perf key.

    An explicit ``dsa_gemm_quant_mode`` override applies to every projection
    (back-compat). Otherwise groups named in ``dsa_attn_quant_exclusions``
    run BF16 and the rest keep the global mode. Module perf rows carry ONE
    gemm_type; for mixed checkpoints no row matches exactly, so the key
    follows o_proj — the largest projection by bytes and FLOPs.
    """
    explicit = None
    exclusions: frozenset = frozenset()
    if isinstance(extra_params, dict):
        explicit = extra_params.get("dsa_gemm_quant_mode")
        exclusions = extra_params.get("dsa_attn_quant_exclusions") or frozenset()
    if explicit is not None:
        modes = dict.fromkeys(("q", "kv", "o", "indexer"), explicit)
        return modes, explicit
    modes = {g: common.GEMMQuantMode.bfloat16 if g in exclusions else fallback for g in ("q", "kv", "o", "indexer")}
    distinct = set(modes.values())
    return modes, (distinct.pop() if len(distinct) == 1 else modes["o"])


def _dsa_shared_expert_quant_mode(extra_params: object, fallback: common.GEMMQuantMode) -> common.GEMMQuantMode:
    if isinstance(extra_params, dict):
        return extra_params.get("dsa_shared_expert_quant_mode", fallback)
    return fallback


def _dense_mlp_groups(raw_config: dict, num_layers: int, fallback: common.GEMMQuantMode) -> list:
    """Dense-prefix layer counts grouped by gate/up and down projection dtype."""
    count = raw_config.get("first_k_dense_replace")
    if count is None:
        count = 0
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= raw_config.get("num_hidden_layers", num_layers)
    ):
        raise ValueError("first_k_dense_replace must be an integer between zero and num_hidden_layers")
    count = min(count, num_layers)
    if not count:
        return []
    if raw_config.get("moe_layer_freq", 1) != 1:
        raise ValueError("DSA dense-prefix composition requires moe_layer_freq=1")
    if raw_config.get("hidden_act", "silu") != "silu":
        raise ValueError("DSA dense-prefix composition requires the silu activation")
    patterns = [str(pattern) for pattern in quant_exclude_patterns(raw_config)]

    def quant_mode(layer: int, projection: str) -> common.GEMMQuantMode:
        name = f"model.layers.{layer}.mlp.{projection}"
        shards = ("gate_proj", "up_proj") if projection == "gate_up_proj" else (projection,)
        # Native selection checks literal module paths on the unpacked shards
        # first. Wildcards are only applied to the packed runtime module after
        # the shard-consistency check, even if a packed wildcard would match.
        skipped = [
            any(f".{pattern.rstrip('.')}." in f".model.layers.{layer}.mlp.{shard}." for pattern in patterns)
            for shard in shards
        ]
        if any(skipped) != all(skipped):
            raise ValueError("DSA dense gate_proj and up_proj must use the same quantization for their fused GEMM")
        if all(skipped):
            return common.GEMMQuantMode.bfloat16
        for pattern in patterns:
            expression = pattern.replace(".", r"\.").replace("*", ".*")
            if any(re.fullmatch(expression, candidate) for candidate in (name, *name.split("."))):
                return common.GEMMQuantMode.bfloat16
        return fallback

    groups: Counter = Counter()
    for layer in range(count):
        groups[(quant_mode(layer, "gate_up_proj"), quant_mode(layer, "down_proj"))] += 1
    return [(count, gate, down) for (gate, down), count in groups.items()]


@register_model("DEEPSEEKV32")
class DeepSeekV32Model(BaseModel):
    """
    DeepSeek-V3.2 / GLM-5 style DeepSeekV32-family model.

    Attention is modeled with the full DSA module-level perf tables so we can
    distinguish architectures such as ``DeepseekV32ForCausalLM`` and
    ``GlmMoeDsaForCausalLM`` without reusing the old DeepSeek-V3 MLA model.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # GLM-5 DSA prefill CP: SGLang AllGather only. CP is modeled INSIDE
        # the engine's ContextDSAModule operator (operators/dsa.rs) +
        # DSA-specific MoE comm, NOT via the dense _cp_attn_comm_ops /
        # seq_split skeleton.
        return backend_name == "sglang"

    @classmethod
    def supports_dcp(cls, backend_name: str) -> bool:
        # Sparse-MLA (DSA) decode CP: vLLM `flashmla_sparse` / `flashinfer_mla_sparse`
        # + DCP-aware indexer. SGLang's NSA backend has no DCP path yet (the
        # DSA-family DCP PRs, #39330 / #36990 / #39117, are open), so it is not
        # claimed. The GenerationDSAModule op prices the gathered heads over
        # the per-rank KV stripe with the sparse attention over the rank's
        # ceil(topk / dcp) owned slots; BaseModel._apply_decode_context_parallel
        # adds the merge collectives and the indexer top-k gather.
        return backend_name == "vllm"

    def _validate_dcp_topology(self) -> None:
        super()._validate_dcp_topology()
        # vLLM's FlashMLA sparse backend (the default DSA decode kernel unless
        # FLASHINFER_MLA_SPARSE is selected, `attention_backend="flashinfer"`
        # here) only serves DCP with the `ag_rs` merge and the fp8_ds_mla
        # mixed-batch KV path (`flashmla_sparse.py`: NotImplementedError
        # otherwise). Do not price a configuration vLLM refuses to launch.
        backend = getattr(self, "_backend_name", None)
        attention_backend = str(getattr(self.config, "attention_backend", None) or "").lower()
        if backend != "vllm" or "flashinfer" in attention_backend:
            return
        if self._dcp_comm_style() != "ag_rs":
            raise ValueError(
                f"{self.architecture} decode context parallelism on vLLM's FlashMLA sparse backend only runs "
                f"with dcp_comm='ag_rs' (got {self.config.dcp_comm!r}); select attention_backend='flashinfer' "
                "(FLASHINFER_MLA_SPARSE) for the a2a merges."
            )
        if self.config.kvcache_quant_mode.value.memory != 1:
            raise ValueError(
                f"{self.architecture} decode context parallelism on vLLM's FlashMLA sparse backend requires an "
                f"fp8 KV cache (got kvcache_quant_mode={self.config.kvcache_quant_mode.name}); the bf16 sparse "
                "path returns no LSE for the DCP merge."
            )

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        moe_args = (model_info["topk"], model_info["num_experts"], model_info["moe_inter_size"])
        base_args = (
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
        )
        extra_params = dict(model_info["extra_params"])
        # Checkpoint-driven, not architecture-gated: vLLM honors ModelOpt
        # exclude_modules wildcards for any architecture (hf_quant_config.json
        # is read in transformers_utils/config.py:726; excluded prefixes fall
        # back to the unquantized path via ModelOptNvFp4Config.is_layer_excluded
        # -> is_layer_skipped, modelopt.py:150-161 @0.24.0). Exclusions are
        # PER-PROJECTION: DeepSeek-V3.2-NVFP4 keeps q/kv/indexer in BF16 but
        # quantizes o_proj; GLM-5 NVFP4 excludes the whole self_attn block.
        extra_params.setdefault(
            "dsa_attn_quant_exclusions",
            attention_projection_exclusions(
                model_info.get("raw_config", {}), precise_module_paths=backend_name == "vllm"
            ),
        )
        if _shared_experts_excluded_from_quant(model_info.get("raw_config", {})):
            extra_params.setdefault("dsa_shared_expert_quant_mode", common.GEMMQuantMode.bfloat16)
        # GLM-5.2 shares one DSA topk index across ``index_topk_freq`` layers
        # (GLM-5 / DeepSeek-V3.2 omit it => 1). The DSA modules amortize the
        # per-layer indexer cost over the group using the collected skip data.
        extra_params.setdefault("index_topk_freq", int(model_info.get("raw_config", {}).get("index_topk_freq", 1) or 1))
        # EXACT full-layer fraction (honors index_skip_topk_offset / pattern) so
        # the per-layer amortization weights real full vs skip counts, not the
        # 1/freq approximation (GLM-5.2: 21/78=0.2692, not 0.25 — under-counting
        # full made AIC predict too fast).
        # vLLM's layer rule matches this checkpoint pattern (deepseek_v2.py:
        # 1080-1106, vLLM v0.25.1 @752a3a504485790a). Pass the
        # physical layer fraction to the native operator; it only uses vLLM
        # skip measurements when the loaded tables have the matching key and
        # local head count. Older/full-only tables remain conservatively full.
        # TRT-LLM has no qualified skip producer and retains all-full pricing.
        extra_params.setdefault(
            "dsa_full_layer_fraction",
            _dsa_full_layer_fraction(model_info.get("raw_config", {}), model_info["layers"])
            if backend_name in {"sglang", "vllm"}
            else 1.0,
        )
        # Dense TP sharding is established for ordinary SGLang inference.
        # Other regimes retain their existing all-MoE approximation: dense
        # distribution under CP/ADP/large EP and draft-layer quantization need
        # separate modeling, as does stage ownership for pipeline parallelism.
        if (
            backend_name == "sglang"
            and model_config.pp_size == model_config.cp_size == model_config.attention_dp_size == 1
            and not model_config.moe_comm_backend
            and not model_config.nextn
        ):
            extra_params["dsa_dense_mlp_groups"] = _dense_mlp_groups(
                model_info.get("raw_config", {}), model_info["layers"], model_config.gemm_quant_mode
            )

        # One class for both regimes: ``__init__`` branches on
        # ``model_config.moe_comm_backend`` (set by the enumerator) for large EP.
        return cls(*moe_args, *base_args, extra_params, backend_name=backend_name)

    #: TRT-LLM large-EP decode PDL overlap discount, transcribed from the
    #: deleted ``TrtllmWideEPDeepSeekV32Model._pdl_factor`` (deepseek_v32.py:463
    #: at commit 8372e60). Scales every decode-layer op, attention included.
    _PDL_FACTOR = 0.9

    def _large_ep_moe_ops(self, phase: str, shape: MoEBlockShape, scale_factor: float) -> list:
        """MoE block for a large-EP config (``cfg.moe_comm_backend`` set).

        Body shared with the DeepSeek family in
        ``helpers.build_large_ep_moe_ops`` (distribution transcription notes
        live there). The shared-expert dtype is asymmetric in the legacy
        classes and is reproduced as such: trtllm sized its shared GEMMs with
        ``_dsa_shared_expert_quant_mode`` (deepseek_v32.py:536-555, 631-653 at
        commit 8372e60), i.e. bf16 for checkpoints like ``nvidia/GLM-5.2-NVFP4``
        that exclude ``mlp.shared_experts*`` from quantization, while sglang
        used the plain ``gemm_quant_mode`` (deepseek_v32.py:796-819) -- so the
        override is passed on trtllm only.
        """
        shared_gemm_quant_mode = (
            _dsa_shared_expert_quant_mode(self.extra_params, self.config.gemm_quant_mode)
            if self._backend_name == "trtllm"
            else None
        )
        return build_large_ep_moe_ops(
            phase,
            shape,
            self.config,
            scale_factor=scale_factor,
            backend_name=self._backend_name,
            model_family=self.model_family,
            power_law_alpha=self._power_law_alpha,
            gpus_per_node=self._gpus_per_node,
            shared_gemm_quant_mode=shared_gemm_quant_mode,
        )

    def _dense_mlp_ops(self, phase: str) -> list:
        groups = self.extra_params.get("dsa_dense_mlp_groups", [])
        result = []
        for index, (count, gate_quant, down_quant) in enumerate(groups):
            prefix = f"{phase}_dense" if len(groups) == 1 else f"{phase}_dense_{index}"
            inter = self._inter_size // self.config.tp_size
            result.extend(
                [
                    # DSA module timings stop before prepare_mlp's reduction;
                    # MoE pre-dispatch accounts for it only on the MoE layers.
                    ops.CustomAllReduce(f"{prefix}_attn_ar", count, self._hidden_size, self.config.tp_size),
                    ops.GEMM(f"{prefix}_gate_up_gemm", count, 2 * inter, self._hidden_size, gate_quant),
                    ops.ElementWise(f"{prefix}_act_gate", count, 2 * inter, inter, 0.8),
                    ops.GEMM(
                        f"{prefix}_down_gemm", count, self._hidden_size, inter, down_quant, low_precision_input=True
                    ),
                    ops.CustomAllReduce(f"{prefix}_ffn_ar", count, self._hidden_size, self.config.tp_size),
                ]
            )
        return result

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)

        self._backend_name = backend_name
        # Large EP: see ``ModelConfig.moe_comm_backend`` (enumerator-owned).
        self._is_large_ep = bool(self.config.moe_comm_backend)
        # Node width is a hardware fact with no default: an unset value would
        # silently mis-price cross-node all-to-all (see large_ep_gpus_per_node).
        self._gpus_per_node = large_ep_gpus_per_node(self.config) if self._is_large_ep else 0

        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size "
            f"({self.config.cp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._power_law_alpha = 1.01
        num_dense_layers = sum(count for count, _, _ in self.extra_params.get("dsa_dense_mlp_groups", []))
        self._num_moe_layers = self._num_layers - num_dense_layers
        if num_dense_layers and (self._inter_size <= 0 or self._inter_size % self.config.tp_size):
            raise ValueError("Dense intermediate_size must be positive and divisible by tp_size")

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        cp_size = self.config.cp_size  # context parallelism (token split, orthogonal to tp)
        pp_size = self.config.pp_size

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        dsa_attn_quant_modes, dsa_gemm_quant_mode = _dsa_attention_quant_modes(self.extra_params, gemm_quant_mode)
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        local_heads = self._num_heads // tp_size

        # MoE block shape (large-EP regime only; the fused spans below stay
        # hand-wired -- their generation dialect differs from the builder's).
        moe_shape = MoEBlockShape(
            hidden_size=h,
            moe_inter_size=self._moe_inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            # The legacy wideEP graphs model exactly one full-size shared expert
            # (WideEP ADP mode, shared_tp_size=1).
            num_shared_experts=1,
            # Descriptor-only: the builder scales by the model-owned scale_factor.
            num_moe_layers=self._num_layers,
        )
        if self._is_large_ep and backend_name == "trtllm":
            # ===== TRT-LLM large EP (wideEP) =====
            # Attention + non-MoE wiring transcribed verbatim from the deleted
            # TrtllmWideEPDeepSeekV32Model (deepseek_v32.py:488-710 at commit
            # 8372e60): the add_norms carry NO ``scale_num_tokens=cp_size`` (the
            # fused path's CP form) and the whole decode stack carries the PDL
            # discount.
            validate_trtllm_large_ep(
                attention_dp_size=attention_dp_size,
                moe_ep_size=moe_ep_size,
                topk=topk,
                num_experts=num_experts,
                wideep_num_slots=self.config.wideep_num_slots,
                enable_eplb=self.config.enable_eplb,
            )
            self.context_ops.extend(
                [
                    ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                    ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                    ops.ContextDSAModule(
                        "context_attention",
                        self._num_layers,
                        local_heads,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        dsa_gemm_quant_mode,
                        architecture=self.architecture,
                        cp_size=self.config.cp_size,
                        index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                        dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                        attn_projection_quant_modes=dsa_attn_quant_modes,
                    ),
                    ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ]
            )
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
            self.context_ops.append(
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            generation_scale = self._num_layers * self._mtp_scale_factor * self._PDL_FACTOR
            self.generation_ops.extend(
                [
                    ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                    ops.ElementWise("generation_add_norm_1", generation_scale, 2 * h, 2 * h, 0.8),
                    ops.GenerationDSAModule(
                        "generation_attention",
                        generation_scale,
                        local_heads,
                        kvcache_quant_mode,
                        dsa_gemm_quant_mode,
                        architecture=self.architecture,
                        index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                        dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                        attn_projection_quant_modes=dsa_attn_quant_modes,
                    ),
                    ops.ElementWise("generation_add_norm_2", generation_scale, 2 * h, 2 * h, 0.8),
                ]
            )
            self.generation_ops.extend(self._large_ep_moe_ops("generation", moe_shape, generation_scale))
            self.generation_ops.append(
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            )

            pp_scale_factor = pp_size - 1
            self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
            self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
            return

        if self._is_large_ep and backend_name == "sglang":
            # ===== sglang large-EP (deepep) =====
            # DSA attention + non-MoE wiring transcribed verbatim from the
            # deleted WideEPDeepSeekV32Model (deepseek_v32.py:754-868 at commit
            # 8372e60): TP all_gather/reduce_scatter around the DSA module and
            # NO embedding / add_norm / logits_gemm / P2P.
            self.context_ops.extend(
                [
                    *(
                        [
                            ops.NCCL(
                                "context_tp_all_gather",
                                self._num_layers,
                                "all_gather",
                                h,
                                tp_size,
                                common.CommQuantMode.half,
                            )
                        ]
                        if tp_size > 1
                        else []
                    ),
                    ops.ContextDSAModule(
                        "context_attention",
                        self._num_layers,
                        local_heads,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        dsa_gemm_quant_mode,
                        architecture=self.architecture,
                        cp_size=self.config.cp_size,
                        index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                        dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                        attn_projection_quant_modes=dsa_attn_quant_modes,
                    ),
                    *(
                        [
                            ops.NCCL(
                                "context_tp_reduce_scatter",
                                self._num_layers,
                                "reduce_scatter",
                                h,
                                tp_size,
                                common.CommQuantMode.half,
                            )
                        ]
                        if tp_size > 1
                        else []
                    ),
                ]
            )
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))

            generation_scale = self._num_layers * self._mtp_scale_factor
            self.generation_ops.append(
                ops.GenerationDSAModule(
                    "generation_attention",
                    generation_scale,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                    attn_projection_quant_modes=dsa_attn_quant_modes,
                )
            )
            self.generation_ops.extend(self._large_ep_moe_ops("generation", moe_shape, generation_scale))
            return

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, scale_num_tokens=cp_size),
                ops.ContextDSAModule(
                    "context_attention",
                    self._num_layers,
                    local_heads,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    cp_size=self.config.cp_size,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                    attn_projection_quant_modes=dsa_attn_quant_modes,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, scale_num_tokens=cp_size),
            ]
        )
        self.context_ops.extend(self._dense_mlp_ops("context"))

        fused_context_moe_ops = [
            ops.GEMM(
                "context_shared_gate_up_gemm",
                self._num_moe_layers,
                2 * self._moe_inter_size // moe_tp_size,
                h,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
            ),
            ops.ElementWise(
                "context_shared_act_gate",
                self._num_moe_layers,
                2 * self._moe_inter_size // moe_tp_size,
                self._moe_inter_size // moe_tp_size,
                0.8,
            ),
            ops.GEMM(
                "context_shared_ffn2_gemm",
                self._num_moe_layers,
                h,
                self._moe_inter_size // moe_tp_size,
                _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
            ),
            ops.GEMM(
                "context_router_gemm",
                self._num_moe_layers,
                self._num_experts,
                h,
                common.GEMMQuantMode.bfloat16,
            ),
            ops.MoEDispatch(
                "context_moe_pre_dispatch",
                self._num_moe_layers,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                True,
                quant_mode=moe_quant_mode,
                attn_cp_size=self.config.cp_size,
                backend=self._backend_name,
            ),
            ops.MoE(
                "context_moe",
                self._num_moe_layers,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                moe_quant_mode,
                workload_distribution,
                attention_dp_size,
                moe_kernel_source=self.config.moe_kernel_source,
            ),
            ops.MoEDispatch(
                "context_moe_post_dispatch",
                self._num_moe_layers,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                False,
                quant_mode=moe_quant_mode,
                attn_cp_size=self.config.cp_size,
                backend=self._backend_name,
            ),
        ]
        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (see the
            # generation site below).
            self.context_ops.extend(self._large_ep_moe_ops("context", moe_shape, self._num_layers))
        elif self._num_moe_layers:
            self.context_ops.extend(fused_context_moe_ops)
        self.context_ops.append(
            ops.GEMM(
                "context_logits_gemm",
                1,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise(
                    "generation_add_norm_1",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
                ops.GenerationDSAModule(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    local_heads,
                    kvcache_quant_mode,
                    dsa_gemm_quant_mode,
                    architecture=self.architecture,
                    index_topk_freq=self.extra_params.get("index_topk_freq", 1),
                    dsa_full_layer_fraction=self.extra_params.get("dsa_full_layer_fraction"),
                    attn_projection_quant_modes=dsa_attn_quant_modes,
                ),
                ops.ElementWise(
                    "generation_add_norm_2",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
            ]
        )
        self.generation_ops.extend(self._dense_mlp_ops("generation"))

        if self._is_large_ep:
            # Large EP on a framework without its own attention stack (the
            # sglang/trtllm branches returned above).
            self.generation_ops.extend(
                self._large_ep_moe_ops("generation", moe_shape, self._num_layers * self._mtp_scale_factor)
            )
        elif self._num_moe_layers:
            gen_shared_ops = [
                ops.GEMM(
                    "generation_shared_gate_up_gemm",
                    self._num_moe_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size // moe_tp_size,
                    h,
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
                ),
                ops.ElementWise(
                    "generation_shared_act_gate",
                    self._num_moe_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size // moe_tp_size,
                    self._moe_inter_size // moe_tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_shared_ffn2_gemm",
                    self._num_moe_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size // moe_tp_size,
                    _dsa_shared_expert_quant_mode(self.extra_params, gemm_quant_mode),
                ),
            ]

            gen_routed_ops = [
                ops.GEMM(
                    "generation_router_gemm",
                    self._num_moe_layers * self._mtp_scale_factor,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_moe_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    quant_mode=moe_quant_mode,
                    attn_cp_size=self.config.cp_size,
                    is_context=False,  # decode: MoEDispatch picks the decode-CP comm path
                    backend=self._backend_name,
                ),
                ops.MoE(
                    "generation_moe",
                    self._num_moe_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                    moe_kernel_source=self.config.moe_kernel_source,
                ),
                ops.MoEDispatch(
                    "generation_moe_post_dispatch",
                    self._num_moe_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                    quant_mode=moe_quant_mode,
                    attn_cp_size=self.config.cp_size,
                    is_context=False,  # decode: MoEDispatch picks the decode-CP comm path
                    backend=self._backend_name,
                ),
            ]
            self.generation_ops.append(
                ops.OverlapOp("generation_moe_overlap", group_a=gen_routed_ops, group_b=gen_shared_ops)
            )
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * self._mtp_scale_factor,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        seq_len = max(0, seq_len)
        extra = self.extra_params if isinstance(self.extra_params, dict) else {}
        kv_lora_rank = extra.get("kv_lora_rank", 512)
        qk_rope_head_dim = extra.get("qk_rope_head_dim", 64)
        index_head_dim = extra.get("index_head_dim", 128)
        return (
            self._num_layers
            * seq_len
            * (
                kv_lora_rank * self.config.kvcache_quant_mode.value.memory
                + qk_rope_head_dim * common.GEMMQuantMode.bfloat16.value.memory
                + common.indexer_cache_entry_bytes(index_head_dim)
            )
        )
