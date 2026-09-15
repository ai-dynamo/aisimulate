# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Kimi processor/topology modeling is a modified adaptation (Apache-2.0),
# copyright 2026 the HuggingFace Inc. team and HuggingFace Team, and copyright
# contributors to the vLLM project:
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
# https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/models/kimi_k25_vit.py
# https://github.com/vllm-project/vllm/blob/d2906091bfc579cebefe3d8e8fb9077397ce9882/vllm/model_executor/layers/quantization/modelopt.py

import importlib.resources as pkg_resources
import json
import logging
import os
import re
import tempfile
import urllib.request
from functools import cache
from pathlib import Path

import yaml

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.common import (
    ARCHITECTURE_TO_MODEL_FAMILY,
    MULTIMODAL_TEXT_CONFIG_KEY,
    BlockConfig,
    DeepSeekV4Config,
    DefaultHFModels,
    HybridMoEConfig,
    KimiK3Config,
    Qwen35Config,
    VisionEncoderConfig,
)

logger = logging.getLogger(__name__)


_NEMOTRONH_LAYER_BLOCK_PATTERN = {
    "mamba": "M",
    "moe": "E",
    "attention": "*",
    "mlp": "-",
    "dense": "-",
    "ffn": "-",
    "M": "M",
    "E": "E",
    "*": "*",
    "-": "-",
}


def get_vision_encoder_config_from_model_info(model_info: dict) -> VisionEncoderConfig | None:
    """Return the vision tower config without discarding family-specific extras.

    Most VL families store :class:`VisionEncoderConfig` directly in
    ``extra_params``. Qwen3.5 must retain its language/GDN/MoE configuration
    there and therefore nests the vision contract under ``vision_config``.
    """
    direct_config = model_info.get("encoder_config")
    if isinstance(direct_config, VisionEncoderConfig):
        return direct_config
    extra_params = model_info.get("extra_params")
    if isinstance(extra_params, VisionEncoderConfig):
        return extra_params
    vision_config = getattr(extra_params, "vision_config", None)
    return vision_config if isinstance(vision_config, VisionEncoderConfig) else None


def _load_json_with_infinity(file_path) -> dict:
    """
    Load JSON file with support for JavaScript-style Infinity and NaN values.

    Standard JSON doesn't support Infinity/NaN, but HuggingFace configs may contain them (e.g., Nemotron-H-56B)
    This function pre-processes the file content to replace these values before parsing.
    """
    with open(file_path) as f:
        content = f.read()
    # Replace JavaScript-style Infinity/NaN with Python-compatible values
    # Use regex to match standalone Infinity/-Infinity/NaN (not part of a string)
    content = re.sub(r"\bInfinity\b", "null", content)
    content = re.sub(r"-Infinity\b", "null", content)
    content = re.sub(r"\bNaN\b", "null", content)
    return json.loads(content)


def _derive_nemotronh_hybrid_pattern(config: dict) -> str:
    """Return a NemotronH hybrid pattern from either legacy or layer-block config fields."""
    pattern = config.get("hybrid_override_pattern")
    if isinstance(pattern, str):
        return pattern

    layer_blocks = config.get("layers_block_type")
    if not isinstance(layer_blocks, list):
        raise TypeError("NemotronH config must define 'hybrid_override_pattern' or 'layers_block_type'.")

    try:
        return "".join(_NEMOTRONH_LAYER_BLOCK_PATTERN[str(block)] for block in layer_blocks)
    except KeyError as exc:
        supported = ", ".join(sorted(_NEMOTRONH_LAYER_BLOCK_PATTERN))
        raise ValueError(
            f"Unsupported NemotronH layer block type '{exc.args[0]}'. Supported values: {supported}"
        ) from exc


def filter_real_silicon_configs(
    parallel_config_list: list[list[int]],
    *,
    is_moe: bool = False,
    min_num_gpus: int | None = None,
    max_num_gpus: int | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[list[int]]:
    """Filter parallel configs for real-silicon sweep runs.

    Applies GPU count bounds and, for MoE models, restricts configs to pure
    TEP, pure DEP, and optionally pure TP patterns.

    Args:
        parallel_config_list: List of ``[tp, pp, dp, moe_tp, moe_ep, cp]`` configs.
        is_moe: Whether the model is MoE.
        min_num_gpus: Minimum total GPUs per config (inclusive).
        max_num_gpus: Maximum total GPUs per config (inclusive).
        allow_moe_pure_tp: When ``True`` (default, GQA+MoE models), pure TP
            configs are kept.  Set to ``False`` for MLA+MoE models (e.g.
            DeepSeek) to only allow TEP/DEP.

    Returns:
        Filtered list of parallel configurations.
    """
    filtered = []
    for cfg in parallel_config_list:
        tp, pp, dp, _moe_tp, _moe_ep, cp = cfg
        total_gpus = tp * pp * dp * cp

        # GPU count bounds
        if min_num_gpus is not None and total_gpus < min_num_gpus:
            continue
        if max_num_gpus is not None and total_gpus > max_num_gpus:
            continue

        # For MoE: only allow pure TEP, pure DEP, and optionally pure TP.
        # CP folds into the attention-side width (``tp * cp``), so "pure TEP"
        # means the attention width comes entirely from TP+CP, not DP.
        # - Pure TEP: tp * cp > 1, dp == 1, moe_tp == 1, moe_ep > 1
        # - Pure DEP: tp * cp == 1, dp > 1, moe_tp == 1, moe_ep > 1
        # - Pure TP:  tp * cp > 1, dp == 1, moe_tp > 1, moe_ep == 1
        #   (only for GQA+MoE; disabled for MLA+MoE via allow_moe_pure_tp=False)
        # Reject any config that doesn't match one of these patterns.
        if is_moe:
            attn_tp_width = tp * cp
            is_pure_tep = attn_tp_width > 1 and dp == 1 and _moe_tp == 1 and _moe_ep > 1
            is_pure_dep = attn_tp_width == 1 and dp > 1 and _moe_tp == 1 and _moe_ep > 1
            is_pure_tp = attn_tp_width > 1 and dp == 1 and _moe_tp > 1 and _moe_ep == 1
            if not allow_moe_pure_tp:
                is_pure_tp = False
            if not (is_pure_tep or is_pure_dep or is_pure_tp):
                continue

        filtered.append(cfg)
    return filtered


def enumerate_parallel_config(
    num_gpu_list: list[int],
    tp_list: list[int],
    pp_list: list[int],
    dp_list: list[int] = [1],
    moe_tp_list: list[int] = [1],
    moe_ep_list: list[int] = [1],
    cp_list: list[int] = [1],
    is_moe: bool = False,
    backend: common.BackendName = common.BackendName.trtllm,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    real_silicon_sweep: bool = False,
    min_num_gpus: int | None = None,
    max_num_gpus: int | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[list[int]]:
    """
    Enumerate parallel configurations based on parallel list.
    This is a helper function for agg_pareto and disagg_pareto to define search space.

    Args:
        num_gpu_list: list of number of gpus, this is used to filter out invalid parallel
            configurations
        tp_list: list of tensor parallel sizes
        pp_list: list of pipeline parallel sizes
        dp_list: list of data parallel sizes
        moe_tp_list: list of moe tensor parallel sizes
        moe_ep_list: list of moe expert parallel sizes
        is_moe: whether to use moe
        backend: backend name enum. Important for moe parallel enumeration as different backends
            have different moe parallel support.
        enable_wideep: DEPRECATED and ignored. Large-EP participation is decided per
            parallel config from perf-data coverage (``PerfDatabase.moe_a2a_coverage`` /
            ``moe_expert_compute_coverage``), not by a flag, so this no longer narrows the
            enumeration. Still accepted so existing callers keep working; restrict the
            search with ``moe_ep_list`` instead.
        real_silicon_sweep: when True, exclude PP (force pp_list=[1]) and filter by
            min_num_gpus/max_num_gpus bounds on total GPUs per config. For MoE models,
            only allows pure TEP, pure DEP, and (optionally) pure TP.
        min_num_gpus: minimum total GPUs per config (only applied when real_silicon_sweep=True).
        max_num_gpus: maximum total GPUs per config (only applied when real_silicon_sweep=True).
        allow_moe_pure_tp: when True (default, GQA+MoE models), pure TP configs are kept.
            Set to False for MLA+MoE models (e.g. DeepSeek) to only allow TEP/DEP.
            Only effective when real_silicon_sweep=True.
    Returns:
        parallel_config_list: list of parallel configurations
    """
    if real_silicon_sweep:
        pp_list = [1]

    # Only SGLang has CP-aware perf modeling today (DSA/dense prefill CP). If a
    # caller explicitly asks for cp > 1 on a non-SGLang backend, raise rather
    # than silently producing a cp=1 deployment plan while they think they ran
    # a CP sweep. Lift this guard when vLLM / TRT-LLM gain CP support.
    if backend != common.BackendName.sglang:
        unsupported_cp = sorted({cp for cp in cp_list if cp != 1})
        if unsupported_cp:
            raise ValueError(
                f"CP is only supported on sglang; got cp_list={unsupported_cp} for backend={backend.value}."
            )
        cp_list = [1]

    parallel_config_list = []
    for tp in tp_list:
        for pp in pp_list:
            if is_moe:
                for dp in dp_list:
                    for moe_tp in moe_tp_list:
                        for moe_ep in moe_ep_list:
                            for cp in cp_list:
                                # Total GPUs and MoE width must match (cp folds
                                # into the attention-side width tp*cp*dp).
                                if dp * tp * pp * cp not in num_gpu_list:
                                    continue
                                if dp * tp * cp != moe_tp * moe_ep:
                                    continue
                                # backend specific filters
                                # trtllm
                                if (
                                    backend == common.BackendName.trtllm and dp > 1 and tp > 1
                                ):  # trtllm as trtllm don't supports attn tp > 1
                                    continue
                                # sglang
                                elif backend == common.BackendName.sglang:
                                    if moe_backend == "megamoe" and moe_tp > 1:
                                        continue  # SGLang MegaMoE is EP-only (moe_tp=1).
                                elif backend == common.BackendName.vllm:  # noqa: SIM102
                                    if moe_tp > 1 and moe_ep > 1:
                                        continue  # vllm does not support MoE TP and MoE EP simultaneously
                                parallel_config_list.append([tp, pp, dp, moe_tp, moe_ep, cp])
            else:
                for cp in cp_list:
                    if tp * pp * cp in num_gpu_list:
                        parallel_config_list.append([tp, pp, 1, 1, 1, cp])

    # Apply real silicon sweep filters to reduce sweep time on real silicon
    if real_silicon_sweep:
        parallel_config_list = filter_real_silicon_configs(
            parallel_config_list,
            is_moe=is_moe,
            min_num_gpus=min_num_gpus,
            max_num_gpus=max_num_gpus,
            allow_moe_pure_tp=allow_moe_pure_tp,
        )

    return parallel_config_list


def enumerate_ttft_tpot_constraints(
    osl: int,
    request_latency: float,
    ttft: float | None = None,
) -> list[tuple[float, float]]:
    """
    Enumerate ttft and tpot constraints if given request latency.
    """
    assert osl > 1
    if ttft is None:
        ttft = request_latency * 0.95

    # typical values for ttft
    base_values = [300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 3000, 5000, 8000]
    base_min, base_max = base_values[0], base_values[-1]

    # values based on request_latency, only supplement values outside the base range
    interval_values = [request_latency * p for p in [0.1, 0.2, 0.3, 0.5, 0.7]]
    extra_values = [v for v in interval_values if v < base_min or v > base_max]

    ttft_set = set(base_values + extra_values)
    ttft_set.add(ttft)
    ttft_list = sorted([t for t in ttft_set if t < request_latency])
    return [(t, (request_latency - t) / (osl - 1)) for t in ttft_list]


def safe_mkdir(target_path: str, exist_ok: bool = True) -> Path:
    """
    Safely create a directory with path validation, sanitization, and security checks.

    This function validates the parent directory for security, sanitizes the target
    directory name, and creates the directory using pathlib.

    Args:
        target_path: The target directory path to create
        exist_ok: If True, don't raise an exception if the directory already exists

    Returns:
        Path: The resolved absolute path of the created directory

    Raises:
        ValueError: If the path is invalid or outside allowed directories
        OSError: If directory creation fails
    """

    def _sanitize_path_component(component: str) -> str:
        """
        Sanitize a path component (closure function).
        """
        if not component:
            return "unknown"

        # Replace dangerous characters with underscores
        sanitized = re.sub(r"[^\w\-_.]", "_", str(component))

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")

        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unknown"

        # Limit length to prevent extremely long filenames
        return sanitized[:100]

    if not target_path:
        raise ValueError("Target path cannot be empty")

    try:
        # Parse the target path
        target = Path(target_path)

        # Get parent directory and target directory name
        if target.is_absolute():
            # For absolute paths, validate the entire path
            parent_dir = target.parent
            dir_name = target.name
        else:
            # For relative paths, validate from current directory
            parent_dir = Path.cwd()
            # Split the relative path and sanitize each component
            parts = target.parts
            sanitized_parts = [_sanitize_path_component(part) for part in parts]

            # Build the final path
            final_target = parent_dir
            for part in sanitized_parts:
                final_target = final_target / part

            return safe_mkdir(str(final_target), exist_ok)

        # Validate parent directory security
        resolved_parent = parent_dir.resolve()

        # Security check: ensure no null bytes
        if "\x00" in str(resolved_parent):
            raise ValueError("Path contains null byte")

        # Check if the parent path is within allowed locations
        current_dir = Path.cwd().resolve()
        allowed_prefixes = [
            current_dir,
            Path.home().resolve(),
            Path("/tmp").resolve(),
            Path("/workspace").resolve(),
            Path("/var/tmp").resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ]

        # Verify the parent path is under an allowed prefix
        is_allowed = any(
            resolved_parent == prefix or resolved_parent.is_relative_to(prefix) for prefix in allowed_prefixes
        )

        if not is_allowed:
            raise ValueError(f"Path is outside allowed locations: {resolved_parent}")

        # Sanitize the target directory name and create final path
        sanitized_name = _sanitize_path_component(dir_name)
        final_path = resolved_parent / sanitized_name

        # Create the directory using pathlib
        final_path.mkdir(parents=True, exist_ok=exist_ok)

        return final_path

    except (OSError, ValueError) as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to create directory: {e}") from e


class HuggingFaceDownloadError(Exception):
    """
    Exception raised when a HuggingFace JSON file cannot be downloaded.
    """

    pass


def _get_hf_auth_headers() -> dict[str, str]:
    """Return HTTP auth headers using the cached HuggingFace token, if available.

    Token resolution order (first non-empty wins):
    1. ``HF_TOKEN`` environment variable
    2. ``HUGGING_FACE_HUB_TOKEN`` environment variable
    3. ``~/.cache/huggingface/token`` file
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # Fall back to the token file written by `huggingface-cli login`
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            with open(token_path) as f:
                hf_token = f.read().strip()
    headers: dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return headers


def _download_hf_json(hf_id: str, filename: str, *, raise_on_404: bool = True) -> dict | None:
    """Download and parse a JSON file from a HuggingFace model repo."""
    url = f"https://huggingface.co/{hf_id}/raw/main/{filename}"
    try:
        req = urllib.request.Request(url, headers=_get_hf_auth_headers())
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404 and not raise_on_404:
            return None
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        raise HuggingFaceDownloadError(
            f"Failed to download {hf_id}'s {filename} from HuggingFace: "
            f"HuggingFace returned HTTP error {e.code}: {e.reason}. "
            f"URL: {url}. If using a gated model, authenticate via one of the following "
            f"(in priority order): (1) set the HF_TOKEN environment variable, "
            f"(2) set the HUGGING_FACE_HUB_TOKEN environment variable, or "
            f"(3) run `huggingface-cli login` (token stored at {token_path})."
        ) from e
    except Exception as e:
        raise HuggingFaceDownloadError(f"Failed to download {hf_id}'s {filename} from HuggingFace: {e}") from e


def _download_hf_config(hf_id: str) -> dict:
    """
    Download a HuggingFace config.json file from the HuggingFace API.

    Args:
        hf_id: HuggingFace model ID

    Returns:
        dict: HuggingFace config.json dictionary

    Raises:
        HuggingFaceDownloadError: If the HuggingFace API returns an error
    """
    return _download_hf_json(hf_id, "config.json", raise_on_404=True) or {}


def _parse_nemotron_block_configs(block_configs: list[dict]) -> list[BlockConfig]:
    """
    Parse Nemotron's block_configs into a list of BlockConfig objects.
    Groups consecutive blocks with the same configuration together.

    Args:
        block_configs: List of block configuration dictionaries from HuggingFace config

    Returns:
        list[BlockConfig]: Grouped block configurations
    """
    if not block_configs:
        return None

    grouped_configs = []
    current_config = None
    current_count = 0

    for block in block_configs:
        attn = block.get("attention", {})
        ffn = block.get("ffn", {})

        n_heads_in_group = attn.get("n_heads_in_group")
        attn_no_op = attn.get("no_op", False)
        ffn_mult = ffn.get("ffn_mult", 3.5)
        ffn_no_op = ffn.get("no_op", False)

        # Create a tuple to compare configurations
        config_tuple = (n_heads_in_group, attn_no_op, ffn_mult, ffn_no_op)

        if current_config == config_tuple:
            current_count += 1
        else:
            if current_config is not None:
                grouped_configs.append(
                    BlockConfig(
                        attn_n_heads_in_group=current_config[0],
                        attn_no_op=current_config[1],
                        ffn_ffn_mult=current_config[2],
                        ffn_no_op=current_config[3],
                        num_inst=current_count,
                    )
                )
            current_config = config_tuple
            current_count = 1

    # Add the last group
    if current_config is not None:
        grouped_configs.append(
            BlockConfig(
                attn_n_heads_in_group=current_config[0],
                attn_no_op=current_config[1],
                ffn_ffn_mult=current_config[2],
                ffn_no_op=current_config[3],
                num_inst=current_count,
            )
        )

    return grouped_configs if grouped_configs else None


def _parse_qwen_vision_encoder_config(
    vision_cfg: dict | None,
    *,
    expected_out_hidden_size: int,
    supports_deepstack: bool,
    partial_rotary_factor: float,
) -> VisionEncoderConfig | None:
    """Parse the shared Qwen ViT and its architecture-specific merger count.

    Qwen3-VL may project intermediate deepstack features in addition to the
    final tower output. Qwen3.5 inherits that ViT implementation but deletes
    the deepstack mergers, so it always has exactly one PatchMerger instance.
    """
    if not vision_cfg:
        return None

    out_hidden_size = int(vision_cfg["out_hidden_size"])
    if out_hidden_size != expected_out_hidden_size:
        raise ValueError(
            "Qwen vision out_hidden_size must match the language hidden_size: "
            f"vision={out_hidden_size}, language={expected_out_hidden_size}"
        )

    deepstack_visual_indexes = tuple(vision_cfg.get("deepstack_visual_indexes", [])) if supports_deepstack else ()
    # PatchMerger pixel-shuffles spatial_merge_size² patches into one
    # visual token, then applies merger_dim -> merger_dim -> language hidden.
    merger_dim = int(vision_cfg["hidden_size"]) * int(vision_cfg["spatial_merge_size"]) ** 2
    return VisionEncoderConfig(
        depth=int(vision_cfg["depth"]),
        hidden_size=int(vision_cfg["hidden_size"]),
        num_heads=int(vision_cfg["num_heads"]),
        intermediate_size=int(vision_cfg["intermediate_size"]),
        patch_size=int(vision_cfg["patch_size"]),
        temporal_patch_size=int(vision_cfg["temporal_patch_size"]),
        spatial_merge_size=int(vision_cfg["spatial_merge_size"]),
        out_hidden_size=out_hidden_size,
        deepstack_visual_indexes=deepstack_visual_indexes,
        projector_dims=((merger_dim, merger_dim), (merger_dim, out_hidden_size)),
        projector_n_instances=1 + len(deepstack_visual_indexes),
        partial_rotary_factor=partial_rotary_factor,
        in_channels=int(vision_cfg.get("in_channels", 3)),
    )


def _kimi_processor_limits(processor_cfg: dict | None, vision_cfg: dict) -> dict:
    """Read legacy Moonshot or native Transformers processor geometry.

    Defaults are the image/video class attributes at Transformers commit
    cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55 (Apache-2.0), copyright 2026
    the HuggingFace Inc. team and HuggingFace Team. Modified adaptation:
    https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/image_processing_kimi_k25.py
    https://github.com/huggingface/transformers/blob/cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55/src/transformers/models/kimi_k25/video_processing_kimi_k25.py
    """
    if processor_cfg is None:
        processor_cfg = {}
    if not isinstance(processor_cfg, dict):
        raise ValueError("Kimi preprocessor_config must be an object")
    media = processor_cfg.get("media_proc_cfg", processor_cfg)
    if not isinstance(media, dict):
        raise ValueError("Kimi media_proc_cfg must be an object")
    video = processor_cfg.get("video_processor", {})
    if not isinstance(video, dict):
        raise ValueError("Kimi video processor config must be an object")

    def positive_int(value, name):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Kimi processor {name} must be a positive integer")
        return value

    def side_limit(settings, default):
        size = settings.get("size", {"max_height": default, "max_width": default})
        if not isinstance(size, dict) or size.get("max_height") != size.get("max_width"):
            raise ValueError("Kimi processor size requires identical max_height and max_width")
        return positive_int(size.get("max_height"), "size.max_height")

    side = positive_int(media.get("patch_limit_on_one_side", side_limit(media, 512)), "patch_limit_on_one_side")
    # Legacy Moonshot settings share one limit across image and video;
    # native Transformers processors each independently default to 512.
    video_default_side = side if "patch_limit_on_one_side" in media else 512
    if side_limit(video, video_default_side) != side:
        raise ValueError("Kimi image and video processor side limits must match")
    for settings in (media, video):
        if settings.get("do_resize", True) is not True:
            raise ValueError("Kimi processor do_resize=False is not modeled")
        if positive_int(settings.get("patch_size", vision_cfg["patch_size"]), "patch_size") != vision_cfg["patch_size"]:
            raise ValueError("Kimi processor patch_size must match vision_config")
        merge = settings.get("merge_kernel_size", settings.get("merge_size", 2))
        if positive_int(merge, "merge_size") != vision_cfg.get("merge_kernel_size", [2, 2])[0]:
            raise ValueError("Kimi processor merge_size must match vision_config")
    for name in ("fixed_output_tokens", "in_patch_limit_video", "max_num_frames_each_video"):
        if media.get(name) is not None:
            raise ValueError(f"Kimi processor {name} is not modeled")
    frames = positive_int(
        media.get("temporal_merge_kernel_size", video.get("temporal_patch_size", 4)), "temporal_merge_kernel_size"
    )
    if frames != 4:
        raise ValueError("Kimi processor temporal chunk size must be 4")
    return {
        "resize_mode": "kimi",
        "image_max_patches": positive_int(media.get("in_patch_limit", media.get("max_patches", 16384)), "max_patches"),
        "video_max_patches": positive_int(
            media.get("in_patch_limit_each_frame", video.get("max_patches", 4096)), "video max_patches"
        ),
        "max_patches_per_side": side,
        "max_video_frames": frames,
    }


def _parse_kimi_k25_vision_encoder_config(
    vision_cfg: dict | None,
    *,
    expected_out_hidden_size: int,
    root_quant_cfg: dict | None,
    processor_cfg: dict | None = None,
) -> VisionEncoderConfig | None:
    """Parse Kimi K2.5's spatial-temporal ViT and pooled PatchMerger.

    Sources: Transformers commit cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55
    and vLLM commit d2906091bfc579cebefe3d8e8fb9077397ce9882.
    """
    if not vision_cfg:
        return None

    merge_kernel = vision_cfg.get("merge_kernel_size", [2, 2])
    if not (isinstance(merge_kernel, list) and len(merge_kernel) == 2 and merge_kernel[0] == merge_kernel[1]):
        raise ValueError(f"Kimi K2.5 requires a square merge_kernel_size, got {merge_kernel!r}")
    if vision_cfg.get("mm_projector_type") != "patchmerger":
        raise ValueError(
            "Kimi K2.5 vision modeling requires mm_projector_type='patchmerger', "
            f"got {vision_cfg.get('mm_projector_type')!r}"
        )
    if vision_cfg.get("merge_type") != "sd2_tpool":
        raise ValueError(
            f"Kimi K2.5 vision modeling requires merge_type='sd2_tpool', got {vision_cfg.get('merge_type')!r}"
        )
    if vision_cfg.get("video_attn_type") != "spatial_temporal":
        raise ValueError(
            "Kimi K2.5 vision modeling requires video_attn_type='spatial_temporal', "
            f"got {vision_cfg.get('video_attn_type')!r}"
        )

    # The Moonshot checkpoint scopes quantization to text_config. NVIDIA's
    # model-level NVFP4 config explicitly excludes both vision components.
    if root_quant_cfg:
        exclusions = root_quant_cfg.get("ignore", [])
        if not isinstance(exclusions, (list, tuple)) or any(not isinstance(pattern, str) for pattern in exclusions):
            raise ValueError("Kimi K2.5 quantization_config.ignore must be a list or tuple of strings")
        # ModelOpt supports glob matching and legacy component substrings.
        # Accept only patterns that prove every descendant is excluded; a
        # child path, regex, different prefix or case cannot establish this.
        # vLLM d2906091, quantization/modelopt.py:is_layer_excluded (Apache-2.0).
        if root_quant_cfg.get("quant_method") != "modelopt":
            raise ValueError("Kimi model-level vision exclusions require supported modelopt matching semantics")
        vision_ignored = bool(set(exclusions) & {"*", "vision_tower", "vision_tower*", "vision_tower.*"})
        projector_ignored = bool(set(exclusions) & {"*", "mm_projector", "mm_projector*", "mm_projector.*"})
        if not (vision_ignored and projector_ignored):
            raise ValueError(
                "Kimi K2.5 has model-level quantization but does not explicitly exclude both "
                "vision_tower and mm_projector; refusing to infer encoder precision from the language model"
            )

    hidden_vit = int(vision_cfg["vt_hidden_size"])
    vision_heads = vision_cfg["vt_num_attention_heads"]
    if not isinstance(vision_heads, int) or isinstance(vision_heads, bool) or vision_heads <= 0:
        raise ValueError("Kimi K2.5 vision vt_num_attention_heads must be a positive integer")
    if hidden_vit % vision_heads:
        raise ValueError("Kimi K2.5 vision vt_hidden_size must be divisible by vt_num_attention_heads")
    spatial_merge_size = int(merge_kernel[0])
    merger_dim = hidden_vit * spatial_merge_size**2
    out_hidden_size = int(vision_cfg["text_hidden_size"])
    if out_hidden_size != expected_out_hidden_size:
        raise ValueError(
            f"Kimi K2.5 vision text_hidden_size ({out_hidden_size}) does not match "
            f"text_config.hidden_size ({expected_out_hidden_size})"
        )
    return VisionEncoderConfig(
        depth=int(vision_cfg["vt_num_hidden_layers"]),
        hidden_size=hidden_vit,
        num_heads=vision_heads,
        intermediate_size=int(vision_cfg["vt_intermediate_size"]),
        patch_size=int(vision_cfg["patch_size"]),
        temporal_patch_size=1,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        projector_dims=((merger_dim, merger_dim), (merger_dim, out_hidden_size)),
        partial_rotary_factor=1.0,
        in_channels=3,
        final_norm=True,
        pool_temporal=True,
        video_attention_type=str(vision_cfg["video_attn_type"]),
        projector_replicated=True,
        **_kimi_processor_limits(processor_cfg, vision_cfg),
    )


def _parse_llama4_vision_config(
    vision_cfg: dict,
    image_processor_cfg: dict | None,
    text_hidden_size: int,
) -> VisionEncoderConfig:
    """Translate the published Llama 4 tower and connector shapes."""
    required = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_channels",
        "intermediate_size",
        "image_size",
        "patch_size",
        "pixel_shuffle_ratio",
        "projector_input_dim",
        "projector_output_dim",
        "vision_output_dim",
    )
    missing = [key for key in required if key not in vision_cfg]
    if missing:
        raise ValueError(f"Llama 4 vision_config is missing required fields: {', '.join(missing)}")

    for key in required:
        if key == "pixel_shuffle_ratio":
            continue
        value = vision_cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Llama 4 vision_config.{key} must be a positive integer, got {value!r}")

    hidden = int(vision_cfg["hidden_size"])
    num_heads = int(vision_cfg["num_attention_heads"])
    if hidden % num_heads != 0:
        raise ValueError(
            "Llama 4 vision_config.hidden_size must be divisible by num_attention_heads: "
            f"hidden_size={hidden}, num_attention_heads={num_heads}"
        )

    processor_required = ("max_patches", "resize_to_max_canvas", "add_global_tile")
    if not isinstance(image_processor_cfg, dict):
        raise TypeError(
            "Llama 4 config must preserve image_processor_config metadata or provide preprocessor_config.json"
        )
    if "image_processor_type" in image_processor_cfg:
        if image_processor_cfg["image_processor_type"] not in ("Llama4ImageProcessor", "Llama4ImageProcessorFast"):
            raise ValueError(
                f"Unsupported Llama 4 image_processor_type: {image_processor_cfg['image_processor_type']!r}"
            )
        # Processor defaults and conditional global tile adapted from Transformers:
        # https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/models/llama4/image_processing_llama4_fast.py
        # Copyright 2025 HuggingFace Inc. team. All rights reserved.
        # Apache-2.0; modified to normalize metadata without loading image tensors.
        image_processor_cfg = {
            "max_patches": 16,
            "resize_to_max_canvas": False,
            "add_global_tile": True,
            "size": {"height": 336, "width": 336},
            **image_processor_cfg,
        }
    if "size" in image_processor_cfg:
        size = image_processor_cfg["size"]
        if (
            not isinstance(size, dict)
            or any(
                not isinstance(size.get(axis), int) or isinstance(size.get(axis), bool) for axis in ("height", "width")
            )
            or size != {"height": vision_cfg["image_size"], "width": vision_cfg["image_size"]}
        ):
            raise ValueError("Llama 4 image processor size must match the square vision_config.image_size")
    processor_missing = [key for key in processor_required if key not in image_processor_cfg]
    if processor_missing:
        raise ValueError("Llama 4 image_processor_config is missing required fields: " + ", ".join(processor_missing))
    max_patches = image_processor_cfg["max_patches"]
    if not isinstance(max_patches, int) or isinstance(max_patches, bool) or max_patches <= 0:
        raise ValueError(f"Llama 4 image_processor_config.max_patches must be a positive integer, got {max_patches!r}")
    for key in ("resize_to_max_canvas", "add_global_tile"):
        if not isinstance(image_processor_cfg[key], bool):
            raise TypeError(f"Llama 4 image_processor_config.{key} must be boolean")

    ratio = vision_cfg["pixel_shuffle_ratio"]
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 0:
        raise ValueError(f"Llama 4 pixel_shuffle_ratio must be positive, got {ratio!r}")
    spatial_merge_size = round(1.0 / float(ratio))
    if spatial_merge_size <= 0 or abs(float(ratio) * spatial_merge_size - 1.0) > 1e-9:
        raise ValueError(f"Llama 4 pixel_shuffle_ratio must have an integral reciprocal, got {ratio!r}")

    merge_dim = hidden * spatial_merge_size**2
    if merge_dim != vision_cfg["intermediate_size"]:
        raise ValueError(
            "Llama 4 pixel-shuffle width must equal vision intermediate_size: "
            f"{hidden} * {spatial_merge_size}^2 = {merge_dim}, config has {vision_cfg['intermediate_size']}"
        )
    if vision_cfg["projector_output_dim"] != vision_cfg["vision_output_dim"]:
        raise ValueError(
            "Llama 4 vision adaptor output must match multimodal projector input: "
            f"projector_output_dim={vision_cfg['projector_output_dim']} "
            f"vision_output_dim={vision_cfg['vision_output_dim']}"
        )

    adapter_hidden = int(vision_cfg["projector_input_dim"])
    adapter_out = int(vision_cfg["projector_output_dim"])
    return VisionEncoderConfig(
        depth=int(vision_cfg["num_hidden_layers"]),
        hidden_size=hidden,
        num_heads=num_heads,
        intermediate_size=int(vision_cfg["intermediate_size"]),
        patch_size=int(vision_cfg["patch_size"]),
        temporal_patch_size=1,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=text_hidden_size,
        projector_dims=(
            (merge_dim, adapter_hidden),
            (adapter_hidden, adapter_out),
            (int(vision_cfg["vision_output_dim"]), text_hidden_size),
        ),
        projector_n_instances=1,
        partial_rotary_factor=0.5,
        in_channels=int(vision_cfg["num_channels"]),
        image_size=int(vision_cfg["image_size"]),
        has_cls_token=True,
        max_num_tiles=max_patches,
        resize_to_max_canvas=image_processor_cfg["resize_to_max_canvas"],
        add_global_tile=image_processor_cfg["add_global_tile"],
        prompt_image_tokens=3,
        prompt_tokens_per_local_tile=1,
    )


def _parse_hf_config_json(config: dict) -> dict:
    """
    Convert a HuggingFace config.json dictionary into model configuration parameters.

    Args:
        config: HuggingFace config.json dictionary

    Returns:
        dict: Model configuration parameters

    Raises:
        ValueError: If a required field is missing from the config or the architecture is not supported
    """
    architecture = config["architectures"][0]
    vision_cfg = config.get("vision_config")
    vision_soft_tokens_per_image = config.get("vision_soft_tokens_per_image")
    processor_cfg = config.get("preprocessor_config")
    root_quant_cfg = config.get("quantization_config")
    encoder_config = None
    image_processor_cfg = config.get("image_processor_config")
    image_token_id = int(config.get("image_token_id") or 0)
    video_token_id = int(config.get("video_token_id") or 0)

    # For multimodal models, unwrap the nested text config so that all LLM
    # parameters (layers, hidden_size, MoE fields, etc.) are read from the
    # correct sub-dictionary while keeping the top-level architecture name.
    processor_cfg = config.get("preprocessor_config")
    text_key = MULTIMODAL_TEXT_CONFIG_KEY.get(architecture)
    if text_key and text_key in config:
        text_cfg = config[text_key]
        if not isinstance(text_cfg, dict):
            raise ValueError(
                f"Expected '{text_key}' to be a dict for architecture {architecture}, got {type(text_cfg).__name__}"
            )
        logger.info(
            "Multimodal model detected (%s). Reading LLM parameters from '%s'.",
            architecture,
            text_key,
        )
        config = {**text_cfg, **{"architectures": [architecture]}}

    if architecture not in ARCHITECTURE_TO_MODEL_FAMILY:
        raise ValueError(
            f"The model's architecture {architecture} is not supported. "
            f"Supported architectures: {', '.join(ARCHITECTURE_TO_MODEL_FAMILY.keys())}"
        )

    layers = config.get("num_hidden_layers")
    if layers is None:
        layer_blocks = config.get("layers_block_type")
        if isinstance(layer_blocks, list):
            layers = len(layer_blocks)
        else:
            pattern = config.get("hybrid_override_pattern")
            if isinstance(pattern, str):
                layers = len(pattern)
    if layers is None:
        raise ValueError("Model config must define 'num_hidden_layers' or a parseable layer pattern.")
    hidden_size = config["hidden_size"]
    n = config["num_attention_heads"]
    vocab = config["vocab_size"]
    context = config["max_position_embeddings"]

    # Handle nullable fields (e.g., Nemotron has null for these)
    n_kv = config.get("num_key_value_heads") or 0
    inter_size = config.get("intermediate_size") or 0
    d = config.get("head_dim") or config.get("attention_head_dim") or (hidden_size // n if n > 0 else 0)

    # MoE parameters
    # Explicit None checks so an explicit `num_experts_per_tok: 0` (dense model)
    # is preserved instead of falling through to the `top_k_experts` fallback.
    topk = config.get("num_experts_per_tok")
    if topk is None:
        topk = config.get("top_k_experts")
    if topk is None:
        # Step-3.7/3.5 spell the routing width moe_top_k.
        topk = config.get("moe_top_k")
    if topk is None:
        topk = 0
    num_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        # Step-3.7/3.5 spell the expert count moe_num_experts.
        or config.get("moe_num_experts")
        or config.get("num_experts", 0)
    )
    moe_inter_size = config.get("moe_intermediate_size", 0) or config.get("intermediate_size", 0)

    # Handle NemotronH-specific configuration (only fields unique to NemotronH)
    extra_params = None
    if architecture == "NemotronHForCausalLM":
        hybrid_override_pattern = _derive_nemotronh_hybrid_pattern(config)
        extra_params = common.NemotronHConfig(
            hybrid_override_pattern=hybrid_override_pattern,
            mamba_num_heads=config["mamba_num_heads"],
            mamba_head_dim=config["mamba_head_dim"],
            ssm_state_size=config["ssm_state_size"],
            conv_kernel=config["conv_kernel"],
            n_groups=config["n_groups"],
            chunk_size=config["chunk_size"],
            # Optional: 0 for non-MoE NemotronH models (e.g., Nemotron-H-56B)
            moe_shared_expert_intermediate_size=config.get("moe_shared_expert_intermediate_size", 0),
            # Optional: latent compression dim for routed experts (Nemotron-3-Super).
            # HF config uses None to mean "no compression"; map to 0 here.
            moe_latent_size=config.get("moe_latent_size") or 0,
            # Optional: Mamba2 SSM state dtype (vLLM reads the same key); "auto" = model dtype.
            mamba_ssm_cache_dtype=config.get("mamba_ssm_cache_dtype") or "auto",
        )
        logger.info(
            f"NemotronH hybrid config: pattern={extra_params.hybrid_override_pattern}, "
            f"mamba_heads={extra_params.mamba_num_heads}"
        )
    elif architecture == "DeciLMForCausalLM":
        if "block_configs" in config:
            extra_params = _parse_nemotron_block_configs(config["block_configs"])
    elif architecture == "MiMoV2FlashForCausalLM":
        # MiMo-V2-Flash: per-layer attention + FFN patterns; different dims for SWA vs global.
        moe_layer_freq_raw = config.get("moe_layer_freq", [])
        moe_layer_freq = (
            tuple(moe_layer_freq_raw) if isinstance(moe_layer_freq_raw, list) else tuple([moe_layer_freq_raw] * layers)
        )
        attn_pattern = tuple(config.get("hybrid_layer_pattern", []))
        if len(attn_pattern) != layers or len(moe_layer_freq) != layers:
            raise ValueError(
                f"Hybrid pattern length mismatch for {architecture}: "
                f"expected {layers} entries, got attn={len(attn_pattern)} moe={len(moe_layer_freq)}"
            )
        if any(v not in (0, 1) for v in (*attn_pattern, *moe_layer_freq)):
            raise ValueError(f"Hybrid patterns for {architecture} must contain only 0/1 values")
        extra_params = HybridMoEConfig(
            attn_layer_pattern=attn_pattern,
            moe_layer_freq=moe_layer_freq,
            swa_num_kv_heads=config.get("swa_num_key_value_heads", 0),
            swa_head_dim=config.get("swa_head_dim", 0),
            swa_v_head_dim=config.get("swa_v_head_dim", 0),
            global_v_head_dim=config.get("v_head_dim", 0),
            sliding_window_size=config.get("sliding_window_size", 0),
            dense_inter_size=0,  # dense layers use model-level inter_size
        )
        logger.info(
            f"MiMo-V2-Flash hybrid config: "
            f"global_attn_layers={sum(extra_params.attn_layer_pattern)}, "
            f"swa_layers={extra_params.attn_layer_pattern.count(0)}, "
            f"moe_layers={sum(extra_params.moe_layer_freq)}, "
            f"dense_layers={extra_params.moe_layer_freq.count(0)}"
        )
    elif architecture == "Llama4ForConditionalGeneration":
        # Llama 4: step-based patterns — generate normalized per-layer tuples.
        # Attention: even layers → local (0), odd layers → global (1).
        # FFN: layer i is MoE (1) if (i+1) % interleave_moe_layer_step == 0, else dense (0).
        step = config.get("interleave_moe_layer_step", 1)
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"interleave_moe_layer_step must be a positive integer, got {step}")
        attn_pattern = tuple(i % 2 for i in range(layers))
        moe_freq = tuple(1 if (i + 1) % step == 0 else 0 for i in range(layers))
        if not isinstance(vision_cfg, dict):
            raise TypeError("Llama 4 config must preserve vision_config metadata")
        llama4_vision_config = _parse_llama4_vision_config(vision_cfg, image_processor_cfg, hidden_size)
        extra_params = HybridMoEConfig(
            attn_layer_pattern=attn_pattern,
            moe_layer_freq=moe_freq,
            # All attention dims are uniform (0 → fall back to model-level defaults).
            sliding_window_size=config.get("attention_chunk_size", 0),
            dense_inter_size=config.get("intermediate_size_mlp", 0),
            vision_config=llama4_vision_config,
        )
        logger.info(
            f"Llama4 hybrid config: interleave_moe_layer_step={step}, "
            f"global_attn_layers={sum(attn_pattern)}, local_attn_layers={attn_pattern.count(0)}, "
            f"moe_layers={sum(moe_freq)}, dense_layers={moe_freq.count(0)}, "
            f"sliding_window_size={extra_params.sliding_window_size}, "
            f"vision_encoder={'enabled' if llama4_vision_config else 'absent'}"
        )
    elif architecture == "KimiK25ForConditionalGeneration":
        # KIMI K2.5 wraps a DeepSeek-V3-style MLA text model. Store v_head_dim so
        # DeepSeekModel can use the correct attention head size (128) for vLLM's
        # standard-attention path, instead of the generic hidden_size // n_heads = 112.
        # kv_lora_rank + qk_rope_head_dim drive the MLA latent KV cache size.
        extra_params = {
            "v_head_dim": config.get("v_head_dim", 0),
            "kv_lora_rank": config.get("kv_lora_rank", 0),
            "qk_rope_head_dim": config.get("qk_rope_head_dim", 0),
        }
        encoder_config = _parse_kimi_k25_vision_encoder_config(
            vision_cfg,
            expected_out_hidden_size=hidden_size,
            root_quant_cfg=root_quant_cfg,
            processor_cfg=processor_cfg,
        )
    elif architecture == "KimiK3ForConditionalGeneration":
        # Kimi-K3: hybrid KDA linear attention + MLA full attention with LatentMoE.
        # linear_attn_config.kda_layers / full_attn_layers are 1-based layer ids.
        linear_attn_cfg = config.get("linear_attn_config") or {}
        kda_layer_ids = set(linear_attn_cfg.get("kda_layers") or [])
        if not kda_layer_ids:
            raise ValueError("Kimi-K3 config must define linear_attn_config.kda_layers")
        out_of_range = sorted(i for i in kda_layer_ids if not 1 <= i <= layers)
        if out_of_range:
            raise ValueError(
                f"Kimi-K3 linear_attn_config.kda_layers contains out-of-range 1-based "
                f"layer ids {out_of_range} (num_hidden_layers={layers}); they would be "
                "silently dropped and produce a wrong hybrid layer plan."
            )
        layer_types = tuple("linear_attention" if (i + 1) in kda_layer_ids else "full_attention" for i in range(layers))
        kimi_vision_config = None
        if vision_cfg is not None and not isinstance(vision_cfg, dict):
            raise ValueError("Kimi K3 vision_config must be an object")
        if vision_cfg:
            # Kimi K3 reuses the MoonViT3D tower and PatchMergerV2 implemented
            # for Kimi K2.5. Sources: Transformers commit
            # cbc1651a032b923da7f4b44b3d0e6f68e6ba6b55 and vLLM commit
            # d2906091bfc579cebefe3d8e8fb9077397ce9882.
            def positive_int(value, name):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"Kimi K3 vision_config.{name} must be a positive integer")
                return value

            merge_kernel = vision_cfg.get("merge_kernel_size")
            if (
                not isinstance(merge_kernel, (list, tuple))
                or len(merge_kernel) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in merge_kernel)
            ):
                raise ValueError("Kimi K3 vision_config.merge_kernel_size must contain two positive integers")
            if merge_kernel[0] != merge_kernel[1]:
                raise ValueError(f"Kimi K3 requires a square merge_kernel_size, got {merge_kernel!r}")
            if vision_cfg.get("merge_type") != "sd2_tpool":
                raise ValueError(
                    f"Kimi K3 vision modeling requires merge_type='sd2_tpool', got {vision_cfg.get('merge_type')!r}"
                )
            if vision_cfg.get("mm_projector_type") != "patchmergerv2":
                raise ValueError(
                    "Kimi K3 vision modeling requires mm_projector_type='patchmergerv2', "
                    f"got {vision_cfg.get('mm_projector_type')!r}"
                )

            vision_hidden = positive_int(vision_cfg.get("vt_hidden_size"), "vt_hidden_size")
            # The patch embedder constructs a temporal sin/cos table even for images.
            if vision_hidden % 2 != 0:
                raise ValueError("Kimi K3 vision_config.vt_hidden_size must be even for temporal position embeddings")
            vision_heads = positive_int(vision_cfg.get("vt_num_attention_heads"), "vt_num_attention_heads")
            vision_depth = positive_int(vision_cfg.get("vt_num_hidden_layers"), "vt_num_hidden_layers")
            vision_intermediate = positive_int(vision_cfg.get("vt_intermediate_size"), "vt_intermediate_size")
            patch_size = positive_int(vision_cfg.get("patch_size"), "patch_size")
            # The upstream attention layer defaults only absent/null QKV width
            # to the tower width; zero and other falsy values are malformed.
            qkv_hidden = vision_cfg.get("qkv_hidden_size")
            qkv_hidden = vision_hidden if qkv_hidden is None else positive_int(qkv_hidden, "qkv_hidden_size")
            if qkv_hidden % vision_heads != 0:
                raise ValueError(
                    f"Kimi K3 qkv_hidden_size ({qkv_hidden}) must be divisible by "
                    f"vt_num_attention_heads ({vision_heads})"
                )
            if (qkv_hidden // vision_heads) % 4 != 0:
                raise ValueError("Kimi K3 attention head dimension must be divisible by 4 for 2D rotary embeddings")
            if positive_int(vision_cfg.get("mm_hidden_size"), "mm_hidden_size") != vision_hidden:
                raise ValueError("Kimi K3 PatchMergerV2 requires mm_hidden_size to match vt_hidden_size")
            out_hidden = positive_int(vision_cfg.get("text_hidden_size"), "text_hidden_size")
            if out_hidden != hidden_size:
                raise ValueError(
                    f"Kimi K3 vision text_hidden_size ({out_hidden}) must match "
                    f"the language hidden_size ({hidden_size})"
                )
            max_temporal_patches = positive_int(vision_cfg.get("init_pos_emb_time"), "init_pos_emb_time")
            merger_dim = vision_hidden * merge_kernel[0] * merge_kernel[1]
            kimi_vision_config = VisionEncoderConfig(
                depth=vision_depth,
                hidden_size=vision_hidden,
                num_heads=vision_heads,
                intermediate_size=vision_intermediate,
                patch_size=patch_size,
                temporal_patch_size=1,
                spatial_merge_size=merge_kernel[0],
                out_hidden_size=out_hidden,
                projector_dims=((merger_dim, merger_dim), (merger_dim, out_hidden)),
                partial_rotary_factor=1.0,
                in_channels=3,
                qkv_hidden_size=qkv_hidden,
                final_norm=True,
                pool_temporal=True,
                video_attention_type="spatial_temporal",
                max_temporal_patches=max_temporal_patches,
                projector_post_norm=True,
                projector_replicated=True,
                projector_pre_norm=False,
                **_kimi_processor_limits(processor_cfg, vision_cfg),
                encoder_type="kimi_k3_moonvit3d_patchmergerv2",
            )
        extra_params = KimiK3Config(
            layer_types=layer_types,
            kda_num_heads=linear_attn_cfg["num_heads"],
            kda_head_dim=linear_attn_cfg["head_dim"],
            kda_conv_kernel=linear_attn_cfg.get("short_conv_kernel_size", 4),
            q_lora_rank=config["q_lora_rank"],
            kv_lora_rank=config["kv_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            # KimiLinearConfig spells it num_experts_per_token (not _tok)
            topk=topk or config.get("num_experts_per_token", 0),
            num_experts=num_experts,
            moe_inter_size=config.get("moe_intermediate_size", 0),
            routed_expert_hidden_size=config.get("routed_expert_hidden_size", 0) or 0,
            num_shared_experts=config.get("num_shared_experts", 0),
            first_k_dense_replace=config.get("first_k_dense_replace", 0),
            dense_inter_size=config.get("intermediate_size", 0),
            attn_res_block_size=config.get("attn_res_block_size", 0) or 0,
            vision_config=kimi_vision_config,
        )
        logger.info(
            f"Kimi-K3 hybrid config: kda_layers={layer_types.count('linear_attention')}, "
            f"mla_layers={layer_types.count('full_attention')}, num_experts={num_experts}, "
            f"latent={extra_params.routed_expert_hidden_size}, shared={extra_params.num_shared_experts}, "
            f"vision={'enabled' if kimi_vision_config is not None else 'absent'}"
        )
    elif architecture in {"DeepSeekForCausalLM", "DeepseekV3ForCausalLM"}:
        # DeepSeek V3 / R1 / Kimi K2: MLA latent geometry from config so the KV
        # cache size is data-driven instead of hardcoded. v_head_dim feeds the
        # vLLM standard-attention path in DeepSeekModel (head_size=128 for the
        # MLA architecture, not the generic hidden_size // n_heads = 56).
        extra_params = {
            "v_head_dim": config.get("v_head_dim", 0),
            "kv_lora_rank": config.get("kv_lora_rank", 0),
            "qk_rope_head_dim": config.get("qk_rope_head_dim", 0),
        }
    elif architecture in {"DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"}:
        # DeepSeek-V3.2 / GLM-5 share the DSA attention pattern but have different
        # projection/indexer dimensions, so keep these structural fields attached
        # to the parsed config for model construction and perf-database lookup.
        extra_params = {
            "q_lora_rank": config["q_lora_rank"],
            "kv_lora_rank": config["kv_lora_rank"],
            "qk_nope_head_dim": config["qk_nope_head_dim"],
            "qk_rope_head_dim": config["qk_rope_head_dim"],
            "v_head_dim": config["v_head_dim"],
            "index_head_dim": config["index_head_dim"],
            "index_n_heads": config["index_n_heads"],
            "index_topk": config["index_topk"],
        }
    elif architecture == "DeepseekV4ForCausalLM":
        compress_ratios = tuple(config["compress_ratios"])
        if len(compress_ratios) < layers:
            raise ValueError(
                f"DeepSeek-V4 compress_ratios length {len(compress_ratios)} is smaller than num_hidden_layers {layers}"
            )
        extra_params = DeepSeekV4Config(
            q_lora_rank=config["q_lora_rank"],
            o_lora_rank=config["o_lora_rank"],
            o_groups=config["o_groups"],
            head_dim=config["head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            index_head_dim=config["index_head_dim"],
            index_n_heads=config["index_n_heads"],
            index_topk=config["index_topk"],
            sliding_window=config["sliding_window"],
            compress_ratios=compress_ratios[:layers],
            compress_rope_theta=config["compress_rope_theta"],
            num_hash_layers=config["num_hash_layers"],
            hc_mult=config["hc_mult"],
            hc_sinkhorn_iters=config["hc_sinkhorn_iters"],
            hc_eps=config["hc_eps"],
            n_shared_experts=config["n_shared_experts"],
        )
        logger.info(
            f"DeepSeek-V4 config: layers={layers}, "
            f"swa_layers={extra_params.compress_ratios.count(0)}, "
            f"csa_layers={extra_params.compress_ratios.count(4)}, "
            f"hca_layers={extra_params.compress_ratios.count(128)}, "
            f"hc_mult={extra_params.hc_mult}"
        )
    elif architecture in {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "MiniMaxM2ForCausalLM"}:
        # Qwen3-family and MiniMax-M2 attention include per-layer Q/K normalization.
        extra_params = {"architecture": architecture, "use_qk_norm": True}
    elif architecture == "Gemma4ForConditionalGeneration":
        # Gemma 4 hybrid attention + dense-MLP-plus-MoE FFN. Layer kind per `layer_types`.
        # Q/K/V head_dim and KV-head count differ between SWA and global layers; global
        # layers may set attention_k_eq_v (no v_proj, V reuses K projection output).
        layer_types_raw = config.get("layer_types", [])
        if len(layer_types_raw) != layers:
            raise ValueError(f"Gemma 4 layer_types length {len(layer_types_raw)} != num_hidden_layers {layers}")
        if any(lt not in ("sliding_attention", "full_attention") for lt in layer_types_raw):
            raise ValueError("Gemma 4 layer_types must contain only 'sliding_attention' or 'full_attention'")
        # Dense Gemma 4 variants (e.g. E2B/E4B/31B) leave the global-attention
        # head fields as null; fall back to the model-wide values in that case.
        swa_num_kv = config["num_key_value_heads"]
        swa_hd = config["head_dim"]
        global_num_kv = config.get("num_global_key_value_heads")
        if global_num_kv is None:
            global_num_kv = swa_num_kv
        global_hd = config.get("global_head_dim")
        if global_hd is None:
            global_hd = swa_hd

        gemma4_vision_config = None
        if vision_cfg is not None:
            if not isinstance(vision_cfg, dict):
                raise ValueError(
                    f"Expected 'vision_config' to be a dict for architecture {architecture}, "
                    f"got {type(vision_cfg).__name__}"
                )
            if vision_cfg.get("model_type") not in (None, "gemma4_vision"):
                raise ValueError(
                    f"Gemma 4 vision_config.model_type must be 'gemma4_vision', got {vision_cfg.get('model_type')!r}"
                )

            pooling_kernel_size = int(vision_cfg["pooling_kernel_size"])
            if pooling_kernel_size <= 0:
                raise ValueError("Gemma 4 vision pooling_kernel_size must be positive")
            soft_tokens_per_image = int(
                vision_soft_tokens_per_image
                if vision_soft_tokens_per_image is not None
                else vision_cfg.get("default_output_length", 0)
            )
            if soft_tokens_per_image <= 0:
                raise ValueError(
                    "Gemma 4 requires a positive vision_soft_tokens_per_image or vision_config.default_output_length"
                )
            supported_soft_token_budgets = (70, 140, 280, 560, 1120)
            if soft_tokens_per_image not in supported_soft_token_budgets:
                raise ValueError(
                    f"Gemma 4 vision soft-token budget must be one of {supported_soft_token_budgets}, "
                    f"got {soft_tokens_per_image}"
                )

            vision_hidden_size = int(vision_cfg["hidden_size"])
            vision_num_heads = vision_cfg["num_attention_heads"]
            if not isinstance(vision_num_heads, int) or isinstance(vision_num_heads, bool) or vision_num_heads <= 0:
                raise ValueError("Gemma 4 vision num_attention_heads must be a positive integer")
            vision_num_kv_heads = int(vision_cfg.get("num_key_value_heads", vision_num_heads))
            vision_head_dim = int(vision_cfg.get("head_dim", vision_hidden_size // vision_num_heads))
            if vision_num_heads * vision_head_dim != vision_hidden_size:
                raise ValueError(
                    "Gemma 4 vision attention geometry must satisfy num_attention_heads * head_dim == hidden_size"
                )
            if vision_num_kv_heads != vision_num_heads:
                raise ValueError(
                    "Gemma 4 vision modeling currently requires full MHA (num_key_value_heads == num_attention_heads)"
                )

            gemma4_vision_config = common.Gemma4VisionEncoderConfig(
                depth=int(vision_cfg["num_hidden_layers"]),
                hidden_size=vision_hidden_size,
                num_heads=vision_num_heads,
                intermediate_size=int(vision_cfg["intermediate_size"]),
                patch_size=int(vision_cfg["patch_size"]),
                temporal_patch_size=1,
                # This is average-pooling stride, not Qwen pixel shuffle. The
                # subclass keeps the distinction explicit for op construction.
                spatial_merge_size=pooling_kernel_size,
                out_hidden_size=hidden_size,
                projector_dims=((vision_hidden_size, hidden_size),),
                projector_n_instances=1,
                # Full 2-D RoPE, split independently across x/y head halves.
                partial_rotary_factor=1.0,
                in_channels=int(vision_cfg.get("num_channels", 3)),
                num_key_value_heads=vision_num_kv_heads,
                head_dim=vision_head_dim,
                pooling_kernel_size=pooling_kernel_size,
                position_embedding_size=int(vision_cfg["position_embedding_size"]),
                soft_tokens_per_image=soft_tokens_per_image,
                supported_soft_token_budgets=supported_soft_token_budgets,
                standardize=bool(vision_cfg.get("standardize", False)),
            )
        extra_params = common.Gemma4MixConfig(
            layer_types=tuple(layer_types_raw),
            swa_num_kv_heads=swa_num_kv,
            swa_head_dim=swa_hd,
            global_num_kv_heads=global_num_kv,
            global_head_dim=global_hd,
            sliding_window_size=config.get("sliding_window", 0),
            attention_k_eq_v=bool(config.get("attention_k_eq_v", False)),
            use_bidirectional_vision_attention=config.get("use_bidirectional_attention") == "vision",
            vision_config=gemma4_vision_config,
        )
        logger.info(
            f"Gemma 4 config: "
            f"swa_layers={extra_params.layer_types.count('sliding_attention')}, "
            f"global_layers={extra_params.layer_types.count('full_attention')}, "
            f"num_experts={num_experts}, top_k={topk}, "
            f"sw={extra_params.sliding_window_size}, k_eq_v_global={extra_params.attention_k_eq_v}, "
            f"bidir_vision={extra_params.use_bidirectional_vision_attention}, "
            f"vision={'enabled' if gemma4_vision_config is not None else 'disabled'}"
        )
    elif architecture == "MuseGlimmerForConditionalGeneration":
        # Muse Glimmer: dense hybrid SWA/global attention, uniform head geometry.
        # NoPE on global layers, logit softcapping, and qk_scale are shape-neutral
        # and deliberately not modeled.
        layer_types_raw = config.get("layer_types", [])
        if len(layer_types_raw) != layers:
            raise ValueError(f"Muse Glimmer layer_types length {len(layer_types_raw)} != num_hidden_layers {layers}")
        if any(lt not in ("sliding_attention", "full_attention") for lt in layer_types_raw):
            raise ValueError("Muse Glimmer layer_types must contain only 'sliding_attention' or 'full_attention'")
        sliding_window = config.get("sliding_window", 0)
        if not sliding_window or int(sliding_window) <= 0:
            raise ValueError("Muse Glimmer requires a positive sliding_window")
        extra_params = common.MuseGlimmerConfig(
            layer_types=tuple(layer_types_raw),
            sliding_window_size=int(sliding_window),
        )
        logger.info(
            f"Muse Glimmer config: "
            f"swa_layers={extra_params.layer_types.count('sliding_attention')}, "
            f"global_layers={extra_params.layer_types.count('full_attention')}, "
            f"sw={extra_params.sliding_window_size}"
        )
    elif architecture in {
        "Step3p7ForConditionalGeneration",
        "Step3p5ForCausalLM",
        "Step3p7FlashForCausalLM",
        "Step3p5FlashForCausalLM",
    }:
        # StepFun Step-3.7-Flash: hybrid SWA/global attention (Gemma-style
        # ``layer_types``) + dense-first-``first_k_dense_replace`` then MoE FFN,
        # with one shared expert on the MoE layers. attn_layer_pattern: 1=full,
        # 0=sliding.
        #
        # The authoritative HF config nests the decoder under ``text_config`` and
        # declares a SECOND attention geometry under
        # ``text_config.attention_other_setting`` — the sliding layers run 96 query
        # heads against the global layers' 64. Reading only the flat top level both
        # rejects real checkpoints and silently sizes every sliding layer with the
        # global head count.
        other = config.get("attention_other_setting") or {}
        swa_n_heads = int(other.get("num_attention_heads", 0) or 0)
        swa_hd_other = int(other.get("head_dim", 0) or 0)
        layer_types_raw = config.get("layer_types", [])
        # The published config sizes layer_types over the decoder PLUS the MTP
        # predict layers (45 + 3 = 48), so trim to the decoder's share before
        # building the per-layer pattern.
        mtp_layers = int(config.get("num_nextn_predict_layers", 0) or 0)
        if len(layer_types_raw) == layers + mtp_layers and mtp_layers:
            layer_types_raw = layer_types_raw[:layers]
        if len(layer_types_raw) != layers:
            raise ValueError(
                f"Step3p7 layer_types length {len(layer_types_raw)} != num_hidden_layers {layers} "
                f"(num_nextn_predict_layers={mtp_layers})"
            )
        if any(lt not in ("sliding_attention", "full_attention") for lt in layer_types_raw):
            raise ValueError("Step3p7 layer_types must contain only 'sliding_attention' or 'full_attention'")
        attn_pattern = tuple(1 if lt == "full_attention" else 0 for lt in layer_types_raw)
        # MoE placement: the published config enumerates the MoE layer indices in
        # ``moe_layers_enum`` (a comma-separated string); the curated fixtures use
        # the DeepSeek-style ``first_k_dense_replace`` prefix count. Honour both,
        # preferring the authoritative enumeration.
        moe_enum_raw = config.get("moe_layers_enum")
        moe_indices: set[int] | None = None
        if isinstance(moe_enum_raw, str) and moe_enum_raw.strip():
            moe_indices = {int(tok) for tok in moe_enum_raw.split(",") if tok.strip()}
        elif isinstance(moe_enum_raw, (list, tuple)) and moe_enum_raw:
            moe_indices = {int(tok) for tok in moe_enum_raw}
        if moe_indices is not None:
            moe_freq = tuple(1 if i in moe_indices else 0 for i in range(layers))
        else:
            first_k_dense = int(config.get("first_k_dense_replace", 0) or 0)
            moe_freq = tuple(0 if i < first_k_dense else 1 for i in range(layers))
        extra_params = HybridMoEConfig(
            attn_layer_pattern=attn_pattern,
            moe_layer_freq=moe_freq,
            # 0 on any field = fall back to the model-level default.
            swa_num_heads=swa_n_heads,
            swa_head_dim=swa_hd_other,
            sliding_window_size=config.get("sliding_window", 0) or config.get("sliding_window_size", 0),
            dense_inter_size=0,  # dense layers use model-level inter_size
            # Step3p7Attention builds q_norm/k_norm unconditionally, so there is
            # no config flag to read -- it is on for every layer of this family.
            use_qk_norm=True,
            use_head_wise_attn_gate=bool(config.get("use_head_wise_attn_gate", False)),
        )
        logger.info(
            f"Step3p7 hybrid config: "
            f"global_attn_layers={sum(attn_pattern)}, swa_layers={attn_pattern.count(0)}, "
            f"moe_layers={sum(moe_freq)}, dense_layers={moe_freq.count(0)}, "
            f"sliding_window_size={extra_params.sliding_window_size}, "
            f"swa_num_heads={extra_params.swa_num_heads or 'default'}, "
            f"head_wise_attn_gate={extra_params.use_head_wise_attn_gate}, "
            f"share_expert_dim={config.get('share_expert_dim', 0)}"
        )
    elif architecture in {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        # Qwen3.8-Max: same hybrid GDN + full-attention shape, but the released
        # checkpoint ships a FLAT config (no text_config nesting) under this
        # CausalLM architecture string instead of the VLM ConditionalGeneration
        # classes above.
        "Qwen3_5MoeForCausalLM",
    }:
        # Qwen3.5 hybrid GDN + full-attention model.
        layer_types_raw = config.get("layer_types", [])
        if len(layer_types_raw) != layers:
            raise ValueError(f"Qwen3.5 layer_types length {len(layer_types_raw)} != num_hidden_layers {layers}")
        vision_encoder_config = _parse_qwen_vision_encoder_config(
            vision_cfg,
            expected_out_hidden_size=hidden_size,
            supports_deepstack=False,
            partial_rotary_factor=1.0,
        )
        extra_params = Qwen35Config(
            layer_types=tuple(layer_types_raw),
            linear_num_key_heads=config["linear_num_key_heads"],
            linear_key_head_dim=config["linear_key_head_dim"],
            linear_num_value_heads=config["linear_num_value_heads"],
            linear_value_head_dim=config["linear_value_head_dim"],
            linear_conv_kernel_dim=config["linear_conv_kernel_dim"],
            topk=topk,
            num_experts=num_experts,
            moe_inter_size=moe_inter_size,
            shared_expert_inter_size=config.get("shared_expert_intermediate_size", 0),
            vision_config=vision_encoder_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        logger.info(
            f"Qwen3.5 hybrid config: architecture={architecture}, "
            f"linear_attn_layers={extra_params.layer_types.count('linear_attention')}, "
            f"full_attn_layers={extra_params.layer_types.count('full_attention')}, "
            f"num_experts={extra_params.num_experts}"
        )
    elif architecture in ("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"):
        extra_params = _parse_qwen_vision_encoder_config(
            vision_cfg,
            expected_out_hidden_size=hidden_size,
            supports_deepstack=True,
            # Preserve the existing Qwen3-VL rotary-table gate. The shared
            # builder treats any positive value as full-head vision RoPE.
            partial_rotary_factor=0.5,
        )
        if extra_params is not None:
            logger.info(
                "Qwen3VL vision encoder config: depth=%d, hidden=%d, patch=%d, spatial_merge=%d",
                extra_params.depth,
                extra_params.hidden_size,
                extra_params.patch_size,
                extra_params.spatial_merge_size,
            )
    return {
        "architecture": architecture,
        "layers": layers,
        "n": n,
        "n_kv": n_kv,
        "d": d,
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "vocab": vocab,
        "context": context,
        "topk": topk,
        "num_experts": num_experts,
        "moe_inter_size": moe_inter_size,
        "extra_params": extra_params,
        "encoder_config": encoder_config,
    }


def _get_model_config_path():
    """
    Get the model config path
    """
    if configured := os.environ.get("AICONFIGURATOR_MODEL_CONFIGS_PATH"):
        return Path(configured)
    return pkg_resources.files("aiconfigurator_core") / "model_configs"


def _load_pre_downloaded_hf_config(hf_id: str) -> dict:
    """Load a cached HuggingFace config.json from the model_configs package directory."""
    config_path = _get_model_config_path() / f"{hf_id.replace('/', '--')}_config.json"
    if not config_path.exists():
        raise ValueError(f"HuggingFace model {hf_id} is not cached in model_configs directory.")
    return _load_json_with_infinity(config_path)


def _load_pre_downloaded_hf_quant_config(hf_id: str) -> dict | None:
    """Load a cached hf_quant_config.json, returning None if not present."""
    config_path = _get_model_config_path() / f"{hf_id.replace('/', '--')}_hf_quant_config.json"
    if not config_path.exists():
        return None
    return _load_json_with_infinity(config_path)


def _load_local_config(path: str) -> dict:
    """Load config.json from a local directory path."""
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found at {config_path}")
    return _load_json_with_infinity(config_path)


def _load_local_quant_config(path: str) -> dict | None:
    """Load hf_quant_config.json from a local directory path if present."""
    config_path = Path(path) / "hf_quant_config.json"
    if not config_path.exists():
        return None
    return _load_json_with_infinity(config_path)


def _normalize_hf_quant_config(hf_quant_config: dict) -> dict:
    """Extract and normalize quant_method/kv_cache_quant_method from hf_quant_config."""
    quant_section = hf_quant_config.get("quantization")
    if not isinstance(quant_section, dict):
        return {}
    quant_algo = quant_section.get("quant_algo") or quant_section.get("quantization_algo")
    kv_algo = quant_section.get("kv_cache_quant_algo")
    normalized: dict[str, str] = {}
    if quant_algo:
        normalized["quant_method"] = str(quant_algo).lower()
    if kv_algo:
        normalized["kv_cache_quant_method"] = str(kv_algo).lower()
    return normalized


def _normalize_quant_algo(value: object) -> str | None:
    """Normalize a quantization algorithm string to a canonical form."""
    if value is None:
        return None
    algo = str(value).strip().lower()
    if not algo:
        return None
    aliases = {
        "fp8": "fp8",
        "fp8_block": "fp8_block",
        "nvfp4": "nvfp4",
        "mxfp4": "mxfp4",
        "w4a16_mxfp4": "mxfp4",
        "mixed_precision": "mixed_precision",
        "mixed-precision": "mixed_precision",
        "mixedprecision": "mixed_precision",
        # compressed-tensors: pass through as-is so models.py can handle it with
        # the correct partial-quantization semantics (MoE experts only, not all Linear layers).
        "compressed-tensors": "compressed-tensors",
    }
    return aliases.get(algo, algo)


def _categorize_ignore_pattern(pattern: str) -> set[str]:
    """Return the set of layer categories covered by a single compressed-tensors ignore pattern.

    Recognized categories:
        ``"attention"``       — self-attention projections (q/k/v/o)
        ``"routing_experts"`` — MoE routing-expert FFN weights
        ``"shared_experts"``  — shared-expert FFN weights
        ``"dense_mlp"``       — dense (non-MoE) FFN projections
        ``"lm_head"``         — vocabulary projection
    """
    p = str(pattern).lower()
    categories: set[str] = set()

    if any(kw in p for kw in ("self_attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        categories.add("attention")

    # shared_expert must be checked before the generic expert check
    if "shared_expert" in p:
        categories.add("shared_experts")
    elif "expert" in p:
        categories.add("routing_experts")

    if "lm_head" in p:
        categories.add("lm_head")

    # Dense MLP: pattern mentions "mlp" and a projection suffix, but not experts.
    # Matches both literal paths ("mlp.gate_proj") and regex groups ("mlp\.(gate|up|down)_proj").
    if "mlp" in p and "_proj" in p and "expert" not in p:
        categories.add("dense_mlp")

    return categories


def parse_compressed_tensors_quant(
    quantization_config: dict | None,
) -> tuple[str | None, frozenset[str]]:
    """Parse a compressed-tensors ``quantization_config`` dict.

    Returns ``(base_algo, ignored_categories)`` where:

    - ``base_algo``: weight quantization algorithm (``"int4_wo"``, ``"int8_wo"``,
      ``"fp8"``, ``"fp8_block"``), or ``None`` when the config carries no
      quantization information.
    - ``ignored_categories``: frozenset of layer-category names excluded from
      quantization (i.e. remaining in float16/bfloat16).  Empty when nothing is
      ignored or when ``base_algo`` is ``None``.

    The caller is responsible for mapping categories to SDK quant-mode fields.
    Recognized categories: ``"attention"``, ``"routing_experts"``,
    ``"shared_experts"``, ``"dense_mlp"``, ``"lm_head"``.
    """
    if not isinstance(quantization_config, dict):
        return None, frozenset()

    # Derive base algo from the first config_group's weight spec.
    base_algo: str | None = None
    for group in (quantization_config.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        num_bits = weights.get("num_bits")
        w_type = str(weights.get("type", "")).lower()
        if isinstance(num_bits, int):
            if num_bits == 4 and "int" in w_type:
                base_algo = "int4_wo"
            elif num_bits == 8 and "int" in w_type:
                base_algo = "int8_wo"
            elif num_bits == 8 and "float" in w_type:
                strategy = str(weights.get("strategy", "")).lower()
                block_structure = weights.get("block_structure")
                base_algo = "fp8_block" if strategy == "block" or block_structure else "fp8"
            elif num_bits == 4 and "float" in w_type:
                # MXFP4 packed weights (e.g. Kimi-K3 "mxfp4-pack-quantized"):
                # W4A16 base lane; per-SM MoE kernel routing may upgrade the
                # activation side (see operations/moe.py).
                base_algo = "w4a16_mxfp4"
        if base_algo:
            break

    if base_algo is None:
        return None, frozenset()

    ignored: set[str] = set()
    for pattern in quantization_config.get("ignore") or []:
        ignored |= _categorize_ignore_pattern(pattern)

    return base_algo, frozenset(ignored)


def _normalize_kv_cache_algo(value: object) -> str | None:
    """Normalize a KV cache quantization algorithm string."""
    if value is None:
        return None
    algo = str(value).strip().lower()
    if not algo:
        return None
    if algo in {"fp8", "e4m3", "e5m2"}:
        return "fp8"
    if algo in {"fp16", "float16", "bf16", "bfloat16"}:
        return "bfloat16"
    return algo


def _infer_quant_dynamic(quant_cfg: dict) -> bool | None:
    """
    Args:
        quant_cfg: The quantization configuration of config.json

    Returns:
        bool | None: The quantization dynamic
    """
    activation_scheme = str(quant_cfg.get("activation_scheme", "")).lower()
    if activation_scheme:
        if activation_scheme == "dynamic":
            return True
        if activation_scheme == "static":
            return False

    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        found_dynamic = []
        for group in config_groups.values():
            if not isinstance(group, dict):
                continue
            for key in ("input_activations", "weights"):
                item = group.get(key)
                if isinstance(item, dict) and "dynamic" in item:
                    found_dynamic.append(bool(item.get("dynamic")))
        if found_dynamic:
            return any(found_dynamic)

    return None


def _get_language_quantization_config(raw_config: dict) -> dict | None:
    """Read root-first language quantization without changing its declared scope."""
    if "quantization_config" in raw_config:
        return raw_config["quantization_config"]
    architecture = (raw_config.get("architectures") or [None])[0]
    text_key = MULTIMODAL_TEXT_CONFIG_KEY.get(architecture)
    nested = raw_config.get(text_key, {}) if text_key else {}
    return nested.get("quantization_config", {}) if isinstance(nested, dict) else {}


def _infer_quantization_fields(raw_config: dict) -> dict[str, object]:
    """Infer quant_method, kv_cache_quant_method, and quant_dynamic from config."""
    quant_cfg = _get_language_quantization_config(raw_config)
    quant_cfg = quant_cfg if isinstance(quant_cfg, dict) else {}

    hf_quant = raw_config.get("hf_quant_config")
    hf_quant = hf_quant if isinstance(hf_quant, dict) else {}
    hf_quant_section = hf_quant.get("quantization")
    hf_quant_section = hf_quant_section if isinstance(hf_quant_section, dict) else {}

    quant_algo = _normalize_quant_algo(
        hf_quant_section.get("quant_algo")
        or quant_cfg.get("quant_algo")
        or quant_cfg.get("quant_method")
        or quant_cfg.get("quantization_method")
    )

    kv_cache_algo = _normalize_kv_cache_algo(
        hf_quant_section.get("kv_cache_quant_algo")
        or quant_cfg.get("kv_cache_quant_algo")
        or quant_cfg.get("kv_cache_quant_method")
        or quant_cfg.get("kv_cache_dtype")
    )

    if kv_cache_algo is None:
        kv_cache_scheme = quant_cfg.get("kv_cache_scheme")
        if isinstance(kv_cache_scheme, dict):
            scheme_type = str(kv_cache_scheme.get("type", "")).lower()
            num_bits = kv_cache_scheme.get("num_bits")
            if "fp8" in scheme_type:
                kv_cache_algo = "fp8"
            elif scheme_type == "float" and isinstance(num_bits, int):
                # TODO: 4bit kv cache support
                if num_bits <= 8:
                    kv_cache_algo = "fp8"
                elif num_bits >= 16:
                    kv_cache_algo = "bfloat16"

    weight_block_size = quant_cfg.get("weight_block_size") or []
    if quant_algo == "fp8" and weight_block_size:
        quant_algo = "fp8_block"

    quant_dynamic = _infer_quant_dynamic(quant_cfg)

    logger.info(
        "Quant inference result: quant_algo=%s, kv_cache_quant_algo=%s, quant_dynamic=%s",
        quant_algo,
        kv_cache_algo,
        quant_dynamic,
    )

    inferred: dict[str, object] = {}
    if quant_algo:
        inferred["quant_algo"] = quant_algo
    if kv_cache_algo:
        inferred["kv_cache_quant_algo"] = kv_cache_algo
    if quant_dynamic is not None:
        inferred["quant_dynamic"] = quant_dynamic
    return inferred


def _attach_inferred_quant_fields(raw_config: dict) -> dict:
    """Attach inferred quantization fields to config, checking text_config for multimodal models."""
    # Keep text-only metadata nested: vision validation needs the original
    # model-level scope even after repeated loads of the cached raw config.
    inferred = _infer_quantization_fields(raw_config)
    for key, value in inferred.items():
        raw_config.setdefault(key, value)
    return raw_config


def _attach_hf_quant_config(raw_config: dict, hf_quant_config: dict | None) -> dict:
    """Merge normalized hf_quant_config fields into raw_config if present."""
    if not hf_quant_config:
        return raw_config
    raw_config["hf_quant_config"] = hf_quant_config
    if "quantization_config" not in raw_config:
        normalized = _normalize_hf_quant_config(hf_quant_config)
        if normalized:
            raw_config["quantization_config"] = normalized
    return raw_config


def _attach_llama4_processor_config(raw_config: dict, model_path: str) -> dict:
    """Load Llama 4's separate processor metadata, retaining bundled inline configs."""
    if (raw_config.get("architectures") or [None])[0] != "Llama4ForConditionalGeneration":
        return raw_config
    if "image_processor_config" in raw_config:
        return raw_config
    if os.path.isdir(model_path):
        processor_path = Path(model_path) / "preprocessor_config.json"
        processor_config = _load_json_with_infinity(processor_path) if processor_path.exists() else None
    elif model_path in DefaultHFModels:
        processor_path = _get_model_config_path() / f"{model_path.replace('/', '--')}_preprocessor_config.json"
        processor_config = _load_json_with_infinity(processor_path) if processor_path.exists() else None
    else:
        processor_config = _download_hf_json(model_path, "preprocessor_config.json", raise_on_404=False)
    if processor_config is not None:
        raw_config["image_processor_config"] = processor_config
    return raw_config


@cache
def _load_model_config_from_model_path(model_path: str) -> dict:
    """
    Get model configuration from model path.

    The model_path can be:
    1. A HuggingFace model path (e.g., "Qwen/Qwen3-32B")
    2. A local directory path containing config.json

    Args:
        model_path: HuggingFace model path or local directory path

    Returns:
        dict: Raw model configuration dictionary

    Raises:
        ValueError: If the model config cannot be found
        HuggingFaceDownloadError: If fetching from HuggingFace fails
    """
    # Check if it's a local path
    if os.path.isdir(model_path):
        config = _attach_llama4_processor_config(_load_local_config(model_path), model_path)
        return _attach_inferred_quant_fields(_attach_hf_quant_config(config, _load_local_quant_config(model_path)))

    # Otherwise treat as HuggingFace path
    if model_path in DefaultHFModels:
        config = _attach_llama4_processor_config(_load_pre_downloaded_hf_config(model_path), model_path)
        return _attach_inferred_quant_fields(
            _attach_hf_quant_config(config, _load_pre_downloaded_hf_quant_config(model_path))
        )

    config = _attach_llama4_processor_config(_download_hf_config(model_path), model_path)
    try:
        hf_quant_config = _download_hf_json(model_path, "hf_quant_config.json", raise_on_404=False)
    except Exception as exc:  # best-effort for optional quant config
        logger.debug("Failed to download hf_quant_config.json for %s: %s", model_path, exc)
        hf_quant_config = None
    return _attach_inferred_quant_fields(_attach_hf_quant_config(config, hf_quant_config))


@cache
def get_model_config_from_model_path(model_path: str) -> dict:
    """
    Get model configuration from model path and parse it into model configuration parameters.

    Args:
        model_path: HuggingFace model path or local directory path

    Returns:
        dict: Model configuration parameters and raw config under "raw_config".
        Quantization metadata retains its original root or text_config scope.
    """
    raw_config = _load_model_config_from_model_path(model_path)
    if (
        raw_config.get("architectures")
        in (
            ["KimiK25ForConditionalGeneration"],
            ["KimiK3ForConditionalGeneration"],
        )
        and model_path not in DefaultHFModels
    ):
        # Only Kimi consumes these processor files. Bundled checkpoints use
        # the pinned processor defaults; local/downloaded checkpoints may
        # override them with the original media_proc_cfg or native HF layout.
        processor = {}
        for filename, key in (
            ("preprocessor_config.json", None),
            ("video_preprocessor_config.json", "video_processor"),
        ):
            if os.path.isdir(model_path):
                path = Path(model_path) / filename
                data = _load_json_with_infinity(path) if path.is_file() else None
            else:
                data = _download_hf_json(model_path, filename, raise_on_404=False)
            if data is not None:
                if not isinstance(data, dict):
                    raise ValueError(f"Kimi {filename} must be an object")
                if key is None:
                    processor.update(data)
                else:
                    processor[key] = data
        raw_config = {**raw_config, "preprocessor_config": processor}
    parsed = _parse_hf_config_json(raw_config)
    if parsed["architecture"] == "DeepseekV4ForCausalLM" and model_path not in common.DEEPSEEK_V4_HF_MODELS:
        supported = ", ".join(sorted(common.DEEPSEEK_V4_HF_MODELS))
        logger.warning(
            "DeepSeek-V4 model path '%s' is not in the preview allowlist. "
            "Proceeding based on architecture; known cached configs: %s",
            model_path,
            supported,
        )
    logger.info(
        "Loaded model config for %s: %s",
        model_path,
        ", ".join(f"{k}={v}" for k, v in parsed.items()),
    )
    parsed["raw_config"] = raw_config
    return parsed


class ListFlowDumper(yaml.SafeDumper):
    """
    Dumper that will print dict items on new lines, but lists on one line.
    Example:
        decode_worker_config:
            backend_name: trtllm
            backend_version: 1.2.0rc5
            dp_list: [1]
            num_gpu_per_worker: [1, 2, 4, 8]
    """

    pass


def represent_list_flow(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        data,
        flow_style=True,  # force inline style
    )


ListFlowDumper.add_representer(list, represent_list_flow)


# ---------------------------------------------------------------------------
# Plain-text helpers (cat -v safe output)
# ---------------------------------------------------------------------------

_ANSI_ESCAPE_RE = re.compile(r"(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])")

# Compact mapping of Unicode characters emitted by plotext to ASCII.
# Only the characters actually produced by plotext's "clear" theme are
# included: box-drawing frame (U+2500 range), block/quadrant elements
# used for sub-cell plotting, the bullet marker, and braille dots.
_UNICODE_TO_ASCII = str.maketrans(
    {
        # Box-drawing (frame)
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u252c": "+",
        "\u2534": "+",
        "\u253c": "+",
        # Block elements
        "\u2580": "-",
        "\u2581": "_",
        "\u2584": "_",
        "\u2588": "#",
        "\u258c": "|",
        "\u2590": "|",
        # Quadrant block elements
        "\u2596": ".",
        "\u2597": ".",
        "\u2598": "'",
        "\u2599": "|",
        "\u259a": ":",
        "\u259b": "|",
        "\u259c": "|",
        "\u259d": "'",
        "\u259e": "/",
        "\u259f": "|",
        # Marker / bullet
        "\u2022": "*",
    }
)


def strip_unicode_to_ascii(text: str) -> str:
    """Strip ANSI escapes and replace Unicode graphics with ASCII.

    Intended for piped / redirected CLI output so that tools like
    ``cat -v`` render clean text instead of M-bM-^T... mojibake.
    """
    text = _ANSI_ESCAPE_RE.sub("", text)
    return text.translate(_UNICODE_TO_ASCII)
