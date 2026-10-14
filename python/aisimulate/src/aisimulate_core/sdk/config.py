# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/config.py

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar, Union

from aisimulate_core.sdk import common
from aisimulate_core.sdk.fpm_config import FpmCompileConfig
from aisimulate_core.sdk.speculation.base import SpeculationConfig

KernelBackendT = TypeVar("KernelBackendT", bound=StrEnum)


def normalize_kernel_backend(
    value: str | KernelBackendT | None,
    enum_type: type[KernelBackendT],
    field_name: str,
) -> KernelBackendT | None:
    """Normalize a public string or enum value to its kernel-backend enum."""
    if value is None or isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of {choices}, got {value!r}.") from exc


def validate_parallel_size(name: str, value: object) -> int:
    """Return ``value`` when it is a positive ``int`` (not ``bool``); raise ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def normalize_kernel_source(value: str | None, field_name: str) -> str | None:
    """Validate an optional exact collected kernel-source label."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or None, got {value!r}.")
    return value


def has_video_input(
    *,
    num_videos: int = 0,
    video_height: int = 0,
    video_width: int = 0,
    video_frames: int = 0,
    num_video_tokens: int = 0,
) -> bool:
    """Return whether any video workload field was configured.

    Unsupported boundaries use this partial-input-aware predicate so an
    incomplete video request cannot silently degrade to a text-only request.
    """
    fields = {
        "num_videos": num_videos,
        "video_height": video_height,
        "video_width": video_width,
        "video_frames": video_frames,
        "num_video_tokens": num_video_tokens,
    }
    for name, value in fields.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative, got {value}.")
    return any((value or 0) > 0 for value in fields.values())


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    dcp_size: int | None = field(default=None, kw_only=True)
    fpm_config: FpmCompileConfig | None = field(default=None, kw_only=True)
    tp_size: int = 1
    pp_size: int = 1
    gemm_quant_mode: common.GEMMQuantMode | None = None
    # Internal provenance carried through dataclasses.replace() during sweeps.
    # None preserves the legacy direct-caller rule (non-None mode is explicit).
    _gemm_quant_mode_is_explicit: bool | None = field(default=None, repr=False, compare=False, kw_only=True)
    moe_quant_mode: common.MoEQuantMode | None = None
    kvcache_quant_mode: common.KVCacheQuantMode | None = None
    fmha_quant_mode: common.FMHAQuantMode | None = None
    comm_quant_mode: common.CommQuantMode | None = common.CommQuantMode.half
    moe_tp_size: int = None
    moe_ep_size: int = None
    attention_dp_size: int = 1
    # Vision encoder data parallelism: the ViT is replicated unsharded on every rank of
    # the TP group, images are sharded across the tp_size ranks
    enable_encoder_dp: bool = True
    cp_size: int = 1  # context parallelism: splits sequence tokens (not heads); folds into attn_width = tp*cp*dp
    # CP variant ("none" / "allgather" / "ulysses" / "ring"). Set by get_model
    # from backend_name when cp_size > 1; default "none". Dense models branch on
    # this in their op pipeline; GLM-5 DSA ignores it (handled in ContextDSAModule).
    cp_style: str = "none"
    # Decode context parallelism (vLLM ``-dcp`` / SGLang ``--dcp-size``) is the
    # keyword-only ``dcp_size`` above: it stripes the DECODE KV cache by token
    # position across ranks that already belong to the attention group, so it
    # does NOT fold into attn_width or total_gpus_per_worker. ``None`` means not
    # requested (priced as 1) and keeps the FPM cell identity on the unrecorded
    # profiles; an explicit 1 selects recorded-DCP1 cells. ``cp_size`` (prefill
    # CP) and ``dcp_size`` are orthogonal per-phase knobs; whether one deployment
    # may set both is a topology-layer decision (agg vs disagg), not a ModelConfig one.
    # DCP partial-output merge collective: "ag_rs" (query all-gather + LSE
    # all-gather + output reduce-scatter; vLLM default) or "a2a" (query
    # all-gather + one packed all-to-all; SGLang default on CUDA). None picks
    # the backend default in BaseModel._dcp_comm_style.
    dcp_comm: str | None = None
    # Replicated query projection under DCP (vLLM ``dcp_q_replicate`` / SGLang
    # ``--dcp-replicate-q-proj``): every rank runs the Q up-projection for the
    # whole DCP group's heads and skips the per-layer query all-gather. Only
    # meaningful with the a2a-style merges. None picks the backend/model default
    # in BaseModel._dcp_q_replicate.
    dcp_q_replicate: bool | None = None
    workload_distribution: str = "power_law"
    # EPD: this worker hosts only the language model -- the vision encoder
    # is served elsewhere (mirrors SGLang --language-only).  Like tp_size,
    # this describes the deployed worker, not the model: vision tokens still
    # extend the context; the ViT ops are simply not hosted here.
    language_only: bool = False
    # quantization options
    # MTP speculative decoding: draft length (compute/verification cost only).
    # Accepted-token progress belongs to the upper prediction/simulation layer.
    # Legacy sugar for speculation=SpeculationConfig("mtp", {"depth": nextn}).
    nextn: int = 0
    # Resolved by config_builders.resolve_speculation before model construction.
    speculation: SpeculationConfig | None = field(default=None, kw_only=True)
    overwrite_num_layers: int = 0
    # model builder falvors
    sms: int = 20
    moe_backend: common.MoEBackend | None = None
    # Attention kernel selection. Two consumers, one knob:
    #   * sglang WideEP MLA picks its kernel-source table ('flashinfer' | 'fa3';
    #     the ops apply the 'flashinfer' default themselves when unset).
    #   * dense attention (AIC-1715) uses it as the kernel-LANE override that
    #     heads the precedence order resolved by
    #     ``attention_lanes.resolve_attention_lane_order``.
    # ``None`` means "no override": the framework default for the database's
    # (backend, version, sm_version) wins. Do NOT default this to a lane name —
    # that would silently pin every model to that lane.
    attention_backend: common.AttentionBackend | None = None
    # Exact collected MoE kernel-source lane. This is intentionally separate
    # from ``moe_backend``, which describes topology/runtime behavior rather
    # than a single measured compute-kernel lane. ``None`` preserves the
    # existing framework/default lookup.
    moe_kernel_source: str | None = field(default=None, kw_only=True)
    # DEPRECATED and ignored (large-EP is selected per tuple via
    # moe_comm_backend); kept for a compatibility window because ModelConfig
    # is exported through the supported core SDK facade and removal breaks
    # ModelConfig(enable_wideep=...) callers AND silently shifts positional
    # arguments after this slot.
    enable_wideep: bool = False
    enable_eplb: bool = False  # Expert Parallel Load Balancing
    wideep_num_slots: int = None  # EPLB num_slots, defaults to num_experts if None
    # Forward-pass modeling switch. "op_level" (default) keeps the granular op
    # lists; "fpm" replaces each phase list with a single whole-model
    # fpm_forward op backed by collected forward-pass data (see
    # operations/fpm_forward.py). Validated in models.get_model.
    forward_model: str = "op_level"
    # internal — per-phase comm backend {"context": ..., "generation": ...}; set by the enumerator, never a user flag
    moe_comm_backend: dict | None = None
    # internal — set alongside moe_comm_backend by the enumerator (system topology).
    # No default: a wrong node width silently mis-prices cross-node all-to-all, so
    # large-EP construction raises when it is missing (models.helpers.large_ep_gpus_per_node).
    num_gpus_per_node: int | None = None
    # Internal system identity used by phase/quantization-specific communication
    # dtype selection.  It travels with ModelConfig through sweep replacements.
    system: str | None = None
    # DeepSeek-V4.1 text AR: use the backend-verified bounded decoder replay profile.
    decoder_replay: bool = False
    # Whole-forward database identity only; never changes arithmetic or memory.
    # Selection emits a warning with the original model mode and matched cell IDs.
    # Exact table-label matching is not independent runtime-precision proof.
    fpm_fmha_quant_mode: common.FMHAQuantMode | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        validate_parallel_size("cp_size", self.cp_size)
        if self.dcp_size is not None and (
            type(self.dcp_size) is not int or self.dcp_size <= 0 or self.tp_size % self.dcp_size
        ):
            raise ValueError("dcp_size must be positive and divide tp_size")
        self.moe_backend = normalize_kernel_backend(self.moe_backend, common.MoEBackend, "moe_backend")
        self.attention_backend = normalize_kernel_backend(
            self.attention_backend,
            common.AttentionBackend,
            "attention_backend",
        )
        self.moe_kernel_source = normalize_kernel_source(self.moe_kernel_source, "moe_kernel_source")

    def resolve_moe_parallelism(self) -> tuple[int, int]:
        """Resolve and validate MoE parallelism dimensions in-place.

        For MoE models, the attention width must match the expert width:
        ``tp_size * attention_dp_size == moe_tp_size * moe_ep_size``. If one
        MoE dimension is missing, infer it from the other. If both are missing,
        raise an error so callers do not silently get an MoE layout they did
        not request.
        """

        def _validate_positive(name: str, value: int) -> None:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        _validate_positive("tp_size", self.tp_size)
        _validate_positive("attention_dp_size", self.attention_dp_size)
        _validate_positive("cp_size", self.cp_size)
        # Context parallelism is an INDEPENDENT attention dimension: it splits
        # tokens (sequence), orthogonal to tp_size (which splits heads). It
        # contributes to the total width that the MoE must match:
        # tp_size * attention_dp_size * cp_size == moe_tp * moe_ep.

        attn_width = self.tp_size * self.attention_dp_size * self.cp_size
        moe_tp_size = self.moe_tp_size
        moe_ep_size = self.moe_ep_size
        if moe_tp_size is None and moe_ep_size is None:
            raise ValueError("At least one of moe_tp_size or moe_ep_size must be set for MoE models.")
        elif moe_tp_size is None:
            _validate_positive("moe_ep_size", moe_ep_size)
            if attn_width % moe_ep_size != 0:
                raise ValueError(
                    f"Cannot infer moe_tp_size: tp_size({self.tp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width} is not "
                    f"divisible by moe_ep_size({moe_ep_size})."
                )
            moe_tp_size = attn_width // moe_ep_size
        elif moe_ep_size is None:
            _validate_positive("moe_tp_size", moe_tp_size)
            if attn_width % moe_tp_size != 0:
                raise ValueError(
                    f"Cannot infer moe_ep_size: tp_size({self.tp_size}) * "
                    f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width} is not "
                    f"divisible by moe_tp_size({moe_tp_size})."
                )
            moe_ep_size = attn_width // moe_tp_size

        _validate_positive("moe_tp_size", moe_tp_size)
        _validate_positive("moe_ep_size", moe_ep_size)

        moe_width = moe_tp_size * moe_ep_size
        if attn_width != moe_width:
            raise ValueError(
                f"Parallelism width mismatch: tp_size({self.tp_size}) * "
                f"attention_dp_size({self.attention_dp_size}) * cp_size({self.cp_size}) = {attn_width}, but "
                f"moe_tp_size({moe_tp_size}) * moe_ep_size({moe_ep_size}) = "
                f"{moe_width}. These must be equal."
            )

        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        return self.moe_tp_size, self.moe_ep_size

    @property
    def total_gpus_per_worker(self) -> int:
        """GPUs occupied by a single worker = tp * pp * dp * cp.

        CP is an independent sequence-sharding dimension, so it multiplies the
        worker's GPU count just like tp/pp/dp. Used by the enumeration and the
        throughput-per-GPU normalization so cp configs are sized correctly.
        """
        return self.tp_size * self.pp_size * self.attention_dp_size * self.cp_size

    @property
    def attn_width(self) -> int:
        """Attention/context-parallel width = tp * cp * dp.

        Must equal moe_tp * moe_ep for MoE models (the width the __post_init__
        validation enforces) and is the attention-side throughput normalizer.
        """
        return self.tp_size * self.cp_size * self.attention_dp_size


@dataclass
class RuntimeConfig:
    """
    Runtime configuration.
    """

    batch_size: int = None
    beam_width: int = 1
    isl: int = None
    osl: int = None
    prefix: int = 0  # prefix len of isl
    ttft: float = None
    tpot: Union[float, list] = None
    request_latency: float = None  # it works together with ttft. 1. <= req_lat 2. <= req_lat and <= ttft
    seq_imbalance_correction_scale: float = 1.0
    # Separate correction scale for generation/decoding stage (do NOT reuse ctx scale).
    gen_seq_imbalance_correction_scale: float = 1.0
    # Engine-step executor selector. "rust" (the compiled engine) is the only
    # accepted value; None defaults to it. The deprecated "python" no-op was
    # removed after its one-release window (deprecation-cleanup PR).
    engine_step_backend: str | None = None
    image_height: int = 0
    image_width: int = 0
    num_images_per_request: int = 1
    num_image_tokens: int = 0  # override: ViT output tokens per image; ignored when image_height/width are set
    video_height: int = 0
    video_width: int = 0
    video_frames: int = 0
    num_videos_per_request: int = 0
    num_video_tokens: int = 0  # override: ViT output tokens per video; ignored when video dimensions are set
