# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/models/base.py

"""
Base class and registry for the models package.

Each model family lives in its own module and registers itself via the
``@register_model("FAMILY")`` decorator. ``get_model()`` in the package's
``__init__.py`` does a registry lookup and dispatches to ``cls.create(...)``.

Adding a new model:
    1. Create ``models/<your_model>.py`` with::

        @register_model("YOUR_FAMILY")
        class YourModel(BaseModel):
            @classmethod
            def create(cls, model_info, model_config, backend_name):
                ...
            def __init__(self, ...):
                ...

    2. Register the architecture name(s) in
       ``aisimulate_core.sdk.common.ARCHITECTURE_TO_MODEL_FAMILY`` and add
       ``"YOUR_FAMILY"`` to ``ModelFamily``.

    No edits to ``models/__init__.py`` or ``get_model()`` are needed —
    auto-discovery imports every module in this package at import time.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from aisimulate_core.sdk import config
from aisimulate_core.sdk.config_builders import normalize_nextn
from aisimulate_core.sdk.speculation.base import NullScheme
from aisimulate_core.sdk.speculation.mtp import MTPScheme

logger = logging.getLogger(__name__)


_MODEL_REGISTRY: dict[str, type] = {}


def register_model(*families: str):
    """Decorator: register ``cls`` as the implementation of one or more families.

    Most classes register one family. Pass multiple when one model class
    handles several families with branching inside ``create()`` — e.g.
    ``DeepSeekModel`` is the entry point for both ``DEEPSEEK`` and
    ``KIMIK25``.

    Logs a warning if a family is already registered (catches typos where
    two files claim the same family).
    """
    if not families:
        raise ValueError("register_model requires at least one family name")

    def decorator(cls):
        for family in families:
            if family in _MODEL_REGISTRY:
                logger.warning(
                    "Overwriting model registration for family %r: %s -> %s",
                    family,
                    _MODEL_REGISTRY[family].__name__,
                    cls.__name__,
                )
            _MODEL_REGISTRY[family] = cls
        return cls

    return decorator


class BaseModel:
    """
    Base model class.
    """

    # Forward-pass modeling mode of this instance. ``get_model``'s fpm rewrite
    # flips it to "fpm"; consumers branch on it (base_backend mixed-step FPM
    # path, Rust engine-step gate).
    forward_model: str = "op_level"

    def __init__(
        self,
        model_path: str,
        model_family: str,
        architecture: str,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        inter_size: int,
        vocab_size: int,
        context_length: int,
        model_config: config.ModelConfig,
        extra_params=None,
    ) -> None:
        """Initialize base model metadata and derived runtime flags."""
        self.model_path = model_path
        self.model_family = model_family
        self.architecture = architecture
        self.config = model_config
        self.extra_params = extra_params
        self._use_qk_norm = bool(extra_params.get("use_qk_norm", False)) if isinstance(extra_params, dict) else False
        self.encoder_ops = []
        self.context_ops = []
        self.generation_ops = []

        # internal only
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._num_kv_heads_per_gpu = (self._num_kv_heads + model_config.tp_size - 1) // model_config.tp_size

        if self._num_layers % model_config.pp_size != 0:
            logger.warning(
                f"num_layers {self._num_layers} is not divisible by pp_size "
                f"{model_config.pp_size}. this will introduce additional rounding error. "
                f"Currently we're nothing to correct this."
            )

        assert self._num_heads % model_config.tp_size == 0, (
            f"num_heads {self._num_heads} should be divisible by tp_size {model_config.tp_size} "
        )

        # GENERAL CONTRACT of `_nextn`: "tokens verified per request per
        # decode step, minus one" — the engine's decode-batch multiplier is
        # `(_nextn + 1)`. MTP is the SPECIAL case that additionally bakes its
        # draft layers into the op lists at construction (via config.nextn /
        # mtp_scale_factor); scheme-based speculation (dspark/eagle3/...)
        # sets `_nextn = verify_width - 1` POST-construction instead
        # (speculation.materialize), so op counts carry no MTP layer scaling.
        # Consumers that mean "verify width" should read the property below;
        # consumers that mean "MTP depth" must gate on the scheme type, not
        # on `_nextn` alone.
        self._nextn = normalize_nextn(model_config.nextn)
        model_config.nextn = self._nextn

        # Speculative scheme (cost side). get_model() replaces this with the
        # resolved scheme after construction. The default must stay consistent
        # with _nextn for directly-constructed models (tests, tools): nextn>0
        # has always meant MTP semantics (verify width nextn+1).
        self.spec_scheme = NullScheme() if self._nextn == 0 else MTPScheme(depth=self._nextn)

    @property
    def verify_width(self) -> int:
        """Tokens verified per request per decode step (the engine's
        decode-batch multiplier). 1 = plain autoregressive decode."""
        return int(self._nextn) + 1

    @property
    def activation_hidden_size(self) -> int:
        return self._num_heads * self._head_size

    def get_additional_activation_bytes(self, num_tokens: int) -> float:
        """Architecture-specific buffers beyond the backend's generic workspace."""
        return 0.0

    def get_resident_weights_bytes(self) -> float:
        """Resident target weights per TP/EP rank, before PP division.

        Models with phase-dependent execution can override this inventory;
        skipping token work must never remove resident decoder weights.
        Scheme-owned draft weights are accounted for separately, including
        any draft weights absent from the materialized context-op subset.
        """
        return float(sum(op.get_weights() for op in self.context_ops if not op._name.startswith("draft_")))

    # ------------------------------------------------------------------
    # Context parallelism (CP) declaration + comm factory (1145-style).
    # GLM-5 DSA does NOT use these -- it handles CP inside ContextDSAModule.
    # Dense models opt in via supports_cp and splat _cp_attn_comm_ops at the
    # attention site; their FMHA cost is modeled by ContextAttention(cp_size=).
    # ------------------------------------------------------------------
    _BACKEND_CP_STYLE: ClassVar[dict] = {
        "sglang": "allgather",  # SGLang AllGather-of-KV variant
        "trtllm": "ring",  # Ring Attention (not yet wired)
    }

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        """Whether this (model, backend) combo supports context parallelism.

        Default False. CP-capable model classes override to declare which
        backends they support. ``get_model`` checks this BEFORE construction
        and raises a clear error rather than silently producing wrong numbers.
        """
        return False

    @classmethod
    def supports_dcp(cls, backend_name: str) -> bool:
        """Whether this (model, backend) combo models decode context parallelism.

        Default False. A DCP-capable model class overrides this once its
        generation pipeline prices the KV-sharded decode attention (per-rank
        KV / dcp, query gather, LSE merge collective). ``get_model`` checks this
        BEFORE construction so dcp_size>1 fails loud instead of silently
        pricing decode as if the KV were not sharded. This is a modeling
        capability check only; deployment policy (which roles may combine
        prefill CP with DCP) lives in the topology layer.
        """
        return False

    @classmethod
    def _resolve_cp_style(cls, backend_name: str) -> str:
        """Pick the CP variant for this (model, backend). Called only when cp_size>1."""
        return cls._BACKEND_CP_STYLE.get(backend_name, "none")

    def _cp_attn_comm_ops(self) -> list:
        """Per-layer CP cross-rank comm ops for this model's ``cp_style``.

        AllGather (sglang): one NCCL all-gather of the full KV, sized from
        ``get_kvcache_bytes_per_sequence(1) / num_layers`` (per-layer per-token
        KV bytes). Models splat this into ``context_ops`` adjacent to the
        attention op. Returns ``[]`` for cp_size<=1 or non-allgather styles.
        """
        import aisimulate_core.sdk.operations as ops

        cp_size = self.config.cp_size
        if cp_size <= 1:
            return []
        style = self.config.cp_style
        comm_bytes = self.config.comm_quant_mode.value.memory
        if style == "allgather":
            kv_bytes_per_token = self.get_kvcache_bytes_per_sequence(1) / self._num_layers
            return [
                ops.NCCL(
                    "context_cp_all_gather",
                    self._num_layers,
                    "all_gather",
                    num_elements_per_token=kv_bytes_per_token / comm_bytes,
                    num_gpus=cp_size,
                    comm_quant_mode=self.config.comm_quant_mode,
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Decode context parallelism (DCP): one rewrite for every model class.
    # The decode attention op prices the DCP group's gathered query heads
    # over this rank's 1/dcp KV stripe (Rust `dcp_size` on the op); the
    # collectives that make that possible are modeled here as NCCL ops:
    #   * query all-gather over the DCP group (skipped by the frameworks'
    #     replicate-q-proj option, which is NOT modeled: gathered Q is priced),
    #   * the partial-output merge -- "ag_rs" (LSE all-gather + fp32 output
    #     reduce-scatter, vLLM default) or "a2a" (one packed all-to-all,
    #     SGLang default on CUDA; the tiny LSE exchange rides inside it).
    # ------------------------------------------------------------------

    def _dcp_comm_style(self) -> str:
        style = getattr(self.config, "dcp_comm", None)
        if style is None:
            style = "a2a" if getattr(self, "_backend_name", None) == "sglang" else "ag_rs"
        if style not in ("ag_rs", "a2a"):
            raise ValueError(f"dcp_comm must be 'ag_rs' or 'a2a', got {style!r}")
        return style

    def _mla_latent_dims(self) -> tuple[int, int]:
        """(q dim gathered per head, output dim merged per head) for absorbed MLA decode."""
        kv_lora_rank, qk_rope_head_dim = 0, 0
        if isinstance(self.extra_params, dict):
            kv_lora_rank = int(self.extra_params.get("kv_lora_rank") or 0)
            qk_rope_head_dim = int(self.extra_params.get("qk_rope_head_dim") or 0)
        kv_lora_rank = kv_lora_rank or 512
        qk_rope_head_dim = qk_rope_head_dim or 64
        return kv_lora_rank + qk_rope_head_dim, kv_lora_rank

    def _decode_attention_dcp_dims(self, op) -> tuple[int, int, int] | None:
        """``(rank-local query heads, q dim per head, output dim per head)`` for a
        decode attention op, or ``None`` when ``op`` is not one.

        Matches on the engine's core classes: ops nested inside a FallbackOp
        come back as bare core instances, not the Python shell subclasses.
        """
        import aisimulate_core._native as _core

        if isinstance(op, _core.GenerationAttention):
            return int(op._n), int(op._head_size), int(op._head_size)
        if isinstance(op, _core.GenerationMLA):
            q_dim, v_dim = self._mla_latent_dims()
            return int(op._num_heads), q_dim, v_dim
        if isinstance(op, _core.WideEPGenerationMLA):
            q_dim, v_dim = self._mla_latent_dims()
            return 128 // int(op._tp_size), q_dim, v_dim
        if isinstance(op, _core.GenerationDSAModule):
            q_dim, v_dim = self._mla_latent_dims()
            return int(op._num_heads), q_dim, v_dim
        if isinstance(op, _core.MLAModule) and not op._is_context:
            q_dim, v_dim = self._mla_latent_dims()
            return int(op._num_heads), q_dim, v_dim
        return None

    @staticmethod
    def _through_fallback(op, leaf):
        """Apply ``leaf(op) -> (op, hit)`` to an op or to every interior of a ``FallbackOp``.

        ``FallbackOp`` exposes clones of its inner ops, so a block whose interior
        was touched is rebuilt around the new primary/fallback list. Returns the
        (possibly rebuilt) op and the first truthy ``hit``.
        """
        import aisimulate_core._native as _core
        import aisimulate_core.sdk.operations as ops

        if not isinstance(op, _core.FallbackOp):
            return leaf(op)
        primary, hit = BaseModel._through_fallback(op._primary, leaf)
        fallback = []
        for inner in op._fallback:
            inner, inner_hit = BaseModel._through_fallback(inner, leaf)
            fallback.append(inner)
            hit = hit or inner_hit
        if not hit:
            return op, hit
        return ops.FallbackOp(op._name, primary=primary, fallback=fallback), hit

    def _rewrite_op_for_dcp(self, op, dcp: int):
        """Return ``(op with dcp applied, (dims, scale) | None)``; the collectives
        are then priced once for the whole (possibly rebuilt) block."""

        def leaf(inner):
            dims = self._decode_attention_dcp_dims(inner)
            if dims is None:
                return inner, None
            inner._dcp_size = dcp
            return inner, (dims, float(inner._scale_factor))

        return self._through_fallback(op, leaf)

    def _dcp_attn_comm_ops(self, name: str, scale: float, *, n_local: int, q_dim: int, v_dim: int) -> list:
        """The per-layer DCP collectives that accompany one decode attention op."""
        import aisimulate_core.sdk.operations as ops

        dcp = int(self.config.dcp_size)
        comm_quant_mode = self.config.comm_quant_mode
        gathered_heads = n_local * dcp

        def nccl(suffix: str, kind: str, elements_per_token: int):
            # The NCCL table is keyed by the collective's whole buffer (nccl-tests
            # `size`: the all-gather receive buffer, the reduce-scatter input, the
            # all-to-all per-rank buffer), matching `context_cp_all_gather`.
            return ops.NCCL(
                f"{name}_dcp_{suffix}",
                scale,
                kind,
                num_elements_per_token=elements_per_token,
                num_gpus=dcp,
                comm_quant_mode=comm_quant_mode,
            )

        # Every rank contributes its n_local query heads; the gathered buffer
        # holds all of them.
        collectives = [nccl("q_all_gather", "all_gather", gathered_heads * q_dim)]
        # Partial-output merge (vllm/v1/attention/ops/dcp.py). Both styles pay
        # the collectives AND the elementwise passes around them; the latter
        # are launch/latency-bound at decode batch sizes but add up over the
        # layers, so they are priced explicitly.
        if self._dcp_comm_style() == "ag_rs":
            # `cp_lse_ag_out_rs`: all-gather the fp32 LSE (2 half-elements per
            # gathered head from each of the dcp ranks), `correct_attn_out`
            # rescales the partial outputs in place, then reduce-scatter the
            # corrected outputs by head.
            collectives.append(nccl("lse_all_gather", "all_gather", gathered_heads * 2 * dcp))
            collectives.append(
                ops.ElementWise(f"{name}_dcp_lse_correct", scale, gathered_heads * v_dim, gathered_heads * v_dim)
            )
            collectives.append(nccl("out_reduce_scatter", "reduce_scatter", gathered_heads * v_dim))
        else:
            # `dcp_a2a_lse_reduce`: pack output + LSE (2 half slots per head)
            # into one send buffer, one all-to-all, then unpack and combine
            # the dcp partials of this rank's heads.
            packed = gathered_heads * (v_dim + 2)
            collectives.append(ops.ElementWise(f"{name}_dcp_a2a_pack", scale, gathered_heads * v_dim, packed))
            collectives.append(nccl("out_all_to_all", "alltoall", packed))
            collectives.append(ops.ElementWise(f"{name}_dcp_a2a_combine", scale, packed, n_local * v_dim))
        return collectives

    def _dcp_kv_head_replication(self) -> int | None:
        """How many DCP ranks can share one KV copy, or ``None`` for "any that
        divide TP" (MLA: one latent KV, replicated on every TP rank). GQA
        classes return ``tp / kv_heads``: DCP only de-duplicates KV heads that
        TP already replicates (vLLM: ``dcp <= tp_size // total_kv_heads``)."""
        return None

    def _validate_dcp_topology(self) -> None:
        dcp = int(self.config.dcp_size)
        tp = int(self.config.tp_size)
        if tp % dcp != 0:
            raise ValueError(
                f"dcp_size={dcp} must divide the attention TP size ({tp}): DCP stripes the KV "
                "across ranks that already belong to the attention group."
            )
        limit = self._dcp_kv_head_replication()
        if limit is not None and dcp > limit:
            raise ValueError(
                f"dcp_size={dcp} exceeds the KV-head replication of this layout ({limit}): "
                "GQA decode CP only de-duplicates kv heads that TP replicates (dcp <= tp / kv_heads)."
            )

    def _stripe_context_for_dcp(self, op, dcp: int):
        """Mark the prefill-side attention ops of a DCP-striped engine.

        With the persistent KV striped, a prefill that reuses cached context
        (``prefix > 0``) first all-gathers the other ranks' stripes; the Rust
        context ops price that from ``dcp_size``. New-token attention is
        unchanged. FallbackOp interiors are clones, so the block is rebuilt.
        """
        import aisimulate_core._native as _core

        def leaf(inner):
            striped = isinstance(inner, (_core.ContextAttention, _core.ContextMLA, _core.ContextDSAModule)) or (
                isinstance(inner, _core.MLAModule) and inner._is_context
            )
            if striped:
                inner._dcp_size = dcp
            return inner, striped

        return self._through_fallback(op, leaf)

    def _apply_decode_context_parallel(self) -> None:
        """Rewrite ``generation_ops`` for ``config.dcp_size > 1``.

        Every top-level decode attention op gets ``_dcp_size`` and is followed
        by its merge collectives. Draft ops are left alone (the frameworks
        replicate the draft KV on every DCP rank). Called by ``get_model``
        after construction when ``dcp_size > 1``; a class that declares
        ``supports_dcp`` but exposes no rewritable decode attention op fails
        loud instead of silently pricing decode as unsharded.
        """
        dcp = int(self.config.dcp_size)
        self._validate_dcp_topology()
        rewritten: list = []
        touched = 0
        for op in self.generation_ops:
            if op._name.startswith("draft_"):
                rewritten.append(op)
                continue
            op, found = self._rewrite_op_for_dcp(op, dcp)
            rewritten.append(op)
            if found is None:
                continue
            (n_local, q_dim, v_dim), scale = found
            rewritten.extend(self._dcp_attn_comm_ops(op._name, scale, n_local=n_local, q_dim=q_dim, v_dim=v_dim))
            touched += 1
        if touched == 0:
            raise NotImplementedError(
                f"{type(self).__name__} declares supports_dcp but exposes no top-level decode attention op "
                "to shard; decode context parallelism cannot be priced for this graph."
            )
        self.generation_ops = rewritten
        # Prefill side of the same engine: cached-context gather (aggregated
        # serving). A decode-only worker never queries context_ops, so this is
        # inert there.
        self.context_ops = [
            op if op._name.startswith("draft_") else self._stripe_context_for_dcp(op, dcp)[0] for op in self.context_ops
        ]

    def _cp_kv_memory_divisor(self) -> int:
        """Per-rank persistent-KV divisor: 1 under prefill CP, ``dcp_size`` under DCP.

        Verified against sglang v0.5.13 that CP gives **no** per-rank KV-memory
        savings for any family -- each rank holds the FULL KV:

        - **Dense GQA**: prefill CP gathers + writes the full KV to every rank's
          pool (``cp_all_gather_rerange_kv_cache`` -> ``cp_allgather_and_save_kv_cache``,
          "write the full result into each rank's local memory pool").
        - **MLA / DSA** (DeepSeek V3/V3.2/V4, Kimi): the prefill gather is
          transient, but **decode does not run CP** (``*_use_prefill_cp`` require
          ``is_context_parallel_extend``) and reads the full KV resident in the
          local pool (decode page_table / ``cache_seqlens`` span the full seq_len
          with no gather) -- so the full KV must reside per rank.

        Prefill CP therefore saves prefill *compute*, not KV memory.

        Decode context parallelism is the knob that DOES shard the persistent
        KV: vLLM ``-dcp`` / SGLang ``--dcp-size`` stripe the KV by token
        position across the dcp ranks (rank ``r`` owns position ``p`` when
        ``p mod dcp == r``), so each rank stores about ``1/dcp`` of every
        sequence. Both frameworks widen the logical page by ``dcp`` so the
        per-rank stripe stays balanced to within one token.
        """
        return int(self.config.dcp_size)

    def get_kvcache_elements_per_token(self) -> int:
        """KV cache size per token (per GPU) summed over all layers, in elements.

        Multiply by ``kvcache_quant_mode.value.memory`` (bytes/elem) for byte size.

        - MLA models (DeepSeek V3/V3.2, Kimi K2/K2.5): the latent KV is shared
          across heads and not sharded by attention TP, so the per-GPU cost is
          ``num_layers * (kv_lora_rank + qk_rope_head_dim)``.
        - Otherwise (GQA/MHA): ``num_kv_heads_per_gpu * head_size * num_layers * 2``.
        """
        if self.model_family in ("DEEPSEEK", "DEEPSEEKV32", "KIMIK25"):
            kv_lora_rank, qk_rope_head_dim = 0, 0
            if isinstance(self.extra_params, dict):
                kv_lora_rank = self.extra_params.get("kv_lora_rank") or 0
                qk_rope_head_dim = self.extra_params.get("qk_rope_head_dim") or 0
            # Fallback to DeepSeek-V3 / Kimi K2 defaults if config didn't expose them.
            if kv_lora_rank == 0:
                kv_lora_rank = 512
            if qk_rope_head_dim == 0:
                qk_rope_head_dim = 64
            return self._num_layers * (kv_lora_rank + qk_rope_head_dim)

        num_kv_heads_per_gpu = (self._num_kv_heads + self.config.tp_size - 1) // self.config.tp_size
        return num_kv_heads_per_gpu * self._head_size * self._num_layers * 2

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        """KV cache bytes for one sequence on one GPU."""
        seq_len = max(0, seq_len)
        return seq_len * self.config.kvcache_quant_mode.value.memory * self.get_kvcache_elements_per_token()

    def get_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Largest single-sequence length whose KV cache fits in ``kv_budget_bytes``.

        The capacity-sizing inverse of :meth:`get_kvcache_bytes_per_sequence`. The
        base model's KV grows linearly -- a constant number of bytes per token --
        so the inverse is exact floor-division by that per-token size.

        Models whose KV growth is non-linear -- hybrid sliding-window attention
        (SWA layers cap at the window while global layers keep growing) or
        compressed / sparse attention (the per-token rate drops past a window, plus
        fixed decode-state buffers) -- override :meth:`get_kvcache_bytes_per_sequence`
        and also override this method (delegating to
        :meth:`_binary_search_kvcache_max_tokens`) so capacity follows their true piecewise
        curve instead of extrapolating the ``seq_len=1`` slope.
        """
        budget = float(kv_budget_bytes)
        per_token = self.get_kvcache_bytes_per_sequence(1)
        if budget <= 0.0 or per_token <= 0.0:
            return 0
        return int(budget // per_token)

    def get_kvcache_batch_capacity(self, kv_budget_bytes: float, max_batch_size: int) -> int:
        """Total-token capacity; models with per-request state may reserve it here."""
        return self.get_kvcache_max_tokens(kv_budget_bytes)

    def _binary_search_kvcache_max_tokens(self, kv_budget_bytes: float) -> int:
        """Monotonic-search inverse of :meth:`get_kvcache_bytes_per_sequence`.

        For non-linear-growth models, where a single per-token constant cannot
        describe the curve, so :meth:`get_kvcache_max_tokens` cannot floor-divide.
        ``get_kvcache_bytes_per_sequence`` is monotonic non-decreasing, so this
        doubles the trial length until its KV exceeds the budget, then binary
        searches for the largest length that still fits.
        """
        budget = float(kv_budget_bytes)
        if budget <= 0.0:
            return 0

        # The equal-size guard handles a model whose KV stops growing with length
        # (a cache fully capped by its window): once doubling the length leaves the
        # KV size unchanged it has saturated, every longer length fits too, and the
        # budget would never be exceeded -- so the loop would not otherwise stop.
        hi, hi_bytes = 1, self.get_kvcache_bytes_per_sequence(1)
        if hi_bytes <= 0.0:
            return 0
        while hi_bytes <= budget:
            nxt = hi * 2
            nxt_bytes = self.get_kvcache_bytes_per_sequence(nxt)
            if nxt_bytes == hi_bytes:
                # KV saturated (a fully window-capped cache): memory never binds, so
                # the real limit is the model's context length. Fall back to the
                # current step only when the context length is unknown.
                return int(self._context_length) if self._context_length > 0 else nxt
            hi, hi_bytes = nxt, nxt_bytes

        # bytes(hi // 2) <= budget < bytes(hi); binary search the boundary.
        lo = hi // 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.get_kvcache_bytes_per_sequence(mid) <= budget:
                lo = mid
            else:
                hi = mid
        return lo
