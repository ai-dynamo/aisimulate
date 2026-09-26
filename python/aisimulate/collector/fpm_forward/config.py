# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed options for Dynamo-native FPM self-benchmark campaigns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FPM_FORWARD_OP = "fpm_forward"
FPM_WARMUP_ITERATIONS = 5
FPM_MEASUREMENT_REPEATS = 1
FPM_MAX_PREFILL_ISL = 8192
# Legacy standalone campaigns retain their extended-capture policy. Use
# --fpm-prefill-cudagraph-policy runtime to leave graph selection to vLLM.
FPM_MAX_PREFILL_CUDAGRAPH_SIZE = 2048
VLLM_AUTO_FIT_MAX_MODEL_LEN = -1
# Shared node-task pool, matching generator/config/deployment_config.yaml's
# SlurmConfig default. This is an editable starting point, not per-rank pinning.
FPM_SLURM_CPUS_PER_TASK = 16
FPM_SLURM_CPU_BIND = "cores"


def resolve_slurm_cpu_policy(
    cpus_per_task: int | None,
    cpu_bind: str | None,
    *,
    resume: bool = False,
    checkpoint_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
    smoke: bool = False,
) -> tuple[int | None, str | None]:
    """Freeze new defaults, or retain a verified archived policy on resume.

    An old plan with no CPU policy stays unverified. The execution transport
    rejects new workers for it, while compatible post-processing remains usable.
    """
    if (cpus_per_task is None or cpu_bind is None) and resume and checkpoint_dir and artifact_root:
        checkpoint_path = Path(checkpoint_dir).expanduser() / (
            "fpm_forward_smoke.json" if smoke else "fpm_forward.json"
        )
        if checkpoint_path.exists():
            from .runtime_memory import validate_saved_plan

            checkpoint = json.loads(checkpoint_path.read_text())
            digest = checkpoint.get("plan_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Slurm CPU policy requires a valid archived checkpoint plan hash")
            campaign = Path(artifact_root).expanduser() / digest[:16]
            if smoke:
                campaign /= "smoke"
            saved = json.loads((campaign / "collection-plan.json").read_text())
            validate_saved_plan(saved)
            if saved["sha256"] != digest:
                raise ValueError("archived Slurm CPU policy does not match the checkpoint")
            options = saved["options"]
            return (
                options.get("slurm_cpus_per_task") if cpus_per_task is None else cpus_per_task,
                options.get("slurm_cpu_bind") if cpu_bind is None else cpu_bind,
            )
    return (
        FPM_SLURM_CPUS_PER_TASK if cpus_per_task is None else cpus_per_task,
        FPM_SLURM_CPU_BIND if cpu_bind is None else cpu_bind,
    )


# Dynamo 1.5.0: components/src/dynamo/vllm/instrumented_scheduler.py,
# _kvwarm_flag_on / _bench_realseed_on at b83b1d9304ebfc624709ac46db32b1b6f1ff1615:
# https://github.com/ai-dynamo/dynamo/blob/b83b1d9304ebfc624709ac46db32b1b6f1ff1615/components/src/dynamo/vllm/instrumented_scheduler.py
# Decode uses the runtime's eligibility gate; cached prefill has a separate
# real-prefix seeding switch. Dynamo retains ownership of point generation
# and feasibility.
FPM_KV_WARMUP_DEFAULTS = {"DYN_BENCH_KV_WARMUP": "on", "DYN_BENCH_PREFILL_REAL_SEED": "on"}

PARALLEL_AXES = ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
PARALLEL_PRESETS = ("auto", "tp", "tep", "dep", "pure_tp")


def with_kv_warmup_defaults(overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve declared warm-up switches without replacing explicit choices."""

    resolved = copy.deepcopy(overrides)
    k8s = resolved.setdefault("K8sConfig", {})
    environment = k8s.setdefault("extra_env", [])
    if environment is None:
        environment = k8s["extra_env"] = []
    declared = set()
    for item in environment:
        name = item["name"]
        if name in FPM_KV_WARMUP_DEFAULTS:
            value = item.get("value")
            if not isinstance(value, str) or value.lower() not in {"on", "1", "true", "off", "0", "false"}:
                raise ValueError(f"{name} must be a literal on/off, 1/0 or true/false string")
            declared.add(name)
    environment.extend(
        {"name": name, "value": value} for name, value in FPM_KV_WARMUP_DEFAULTS.items() if name not in declared
    )
    return resolved


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_or_auto_int(value: str) -> int:
    parsed = int(value)
    if parsed != VLLM_AUTO_FIT_MAX_MODEL_LEN and parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive or -1 for runtime auto-fit")
    return parsed


def _at_least_two_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _memory_fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be finite and in (0, 1]")
    return parsed


def _optional_size_list(values: list[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(sorted(set(values)))


def _freeze_benchmark_points(path: str) -> tuple[str, str]:
    """Freeze validated coordinates; native Dynamo owns execution/row admission."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate benchmark-points field: {key}")
            result[key] = value
        return result

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "prefill", "decode"}:
        raise ValueError("benchmark-points manifest requires schema_version, prefill and decode")
    if type(payload["schema_version"]) is not int or payload["schema_version"] not in (1, 2, 3):
        raise ValueError("benchmark-points schema_version must be 1, 2 or 3")
    for phase in ("prefill", "decode"):
        if not isinstance(payload[phase], list) or any(not isinstance(point, dict) for point in payload[phase]):
            raise ValueError(f"benchmark-points {phase} must be a list of point objects")
        for point in payload[phase]:
            required = {"batch_size": 1, "total_kv_read_tokens": 0}
            if phase == "prefill":
                required["total_prefill_tokens"] = 1
            elif "total_prefill_tokens" in point:
                required["total_prefill_tokens"] = 0
            for field, minimum in required.items():
                value = point.get(field)
                if type(value) is not int or value < minimum:
                    raise ValueError(f"benchmark-points {phase} {field} must be an integer >= {minimum}")
            token_field = "total_prefill_tokens" if phase == "prefill" else "total_kv_read_tokens"
            if point[token_field] < point["batch_size"]:
                raise ValueError(f"benchmark-points {phase} {token_field} must be >= batch_size")
            if phase == "decode" and point.get("total_prefill_tokens", 0) != 0:
                raise ValueError("benchmark-points decode total_prefill_tokens must be zero")
    if not payload["prefill"] and not payload["decode"]:
        raise ValueError("benchmark-points manifest must contain at least one point")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _powers_of_two_up_to(limit: int) -> tuple[int, ...]:
    values = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    return tuple(values)


def _prefill_cudagraph_capture_sizes(max_isl: int, max_capture_size_limit: int) -> tuple[int, ...]:
    """Build the formal prefill capture axis for the FPM Scheduler sweep.

    Sizes through 512 mirror vLLM 0.24's balanced defaults. The 32-token
    stride above 512 is an explicit FPM collection policy that extends capture
    coverage to the configured limit (default 2048) without inheriting an
    unbounded runtime default.
    """

    max_capture_size = min(max_isl, max_capture_size_limit)
    sizes = [value for value in (1, 2, 4) if value <= max_capture_size]
    sizes.extend(range(8, min(max_capture_size + 1, 256), 8))
    if max_capture_size >= 256:
        sizes.extend(range(256, min(max_capture_size + 1, 513), 16))
    if max_capture_size > 512:
        sizes.extend(range(544, max_capture_size + 1, 32))
    # vLLM requires an explicitly supplied max capture size to equal the
    # largest explicit capture. Preserve an off-stride user endpoint exactly.
    sizes.append(max_capture_size)
    return tuple(sorted(set(sizes)))


def _dynamo_cudagraph_axis_points(capture_sizes: tuple[int, ...], limit: int) -> tuple[int, ...]:
    """Mirror PR11509's ``_cudagraph_axis_points`` candidate expansion."""

    configured = tuple(sorted({size for size in capture_sizes if size >= 1}))
    if not configured:
        return tuple(sorted({*_powers_of_two_up_to(limit), limit}))

    points: set[int] = set()
    for capture_size in configured:
        if capture_size > limit:
            continue
        points.add(capture_size)
        if capture_size < limit:
            points.add(capture_size + 1)

    if configured[-1] <= limit:
        value = configured[-1] * 2
        while value < limit:
            points.add(value)
            value *= 2
    points.add(limit)
    return tuple(sorted(points))


@dataclass(frozen=True, slots=True)
class PrefillSamplingProfile:
    """Collector-owned inputs to Dynamo's native prefill grid generator."""

    max_isl: int
    max_batch_size: int | None
    max_total_prefill_tokens: int
    cudagraph_capture_sizes: tuple[int, ...] | None
    max_cudagraph_capture_size: int | None
    new_token_axis_points: tuple[int, ...] | None
    max_new_token_samples: int | None
    max_kv_read_token_samples: int
    cudagraph_policy: str = "explicit"

    @classmethod
    def build(
        cls,
        *,
        max_isl: int,
        max_batch_size: int | None,
        max_cudagraph_capture_size: int | None = None,
        cudagraph_policy: str = "explicit",
    ) -> PrefillSamplingProfile:
        if max_isl < 2:
            raise ValueError("FPM max prefill ISL must be at least 2")
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("FPM max prefill batch size must be positive")
        if cudagraph_policy not in {"explicit", "runtime"}:
            raise ValueError("FPM prefill CUDA-graph policy must be runtime or explicit")
        if cudagraph_policy == "runtime" and max_cudagraph_capture_size is not None:
            raise ValueError("runtime prefill CUDA-graph policy cannot specify a capture size")
        capture_sizes = None
        new_token_points = None
        if cudagraph_policy == "explicit":
            if max_cudagraph_capture_size is None:
                max_cudagraph_capture_size = FPM_MAX_PREFILL_CUDAGRAPH_SIZE
            if type(max_cudagraph_capture_size) is not int or max_cudagraph_capture_size < 1:
                raise ValueError("FPM max prefill CUDA-graph capture size must be positive")
            capture_sizes = _prefill_cudagraph_capture_sizes(max_isl, max_cudagraph_capture_size)
            new_token_points = _dynamo_cudagraph_axis_points(capture_sizes, max_isl)
        batch_upper_bound = min(max_isl, max_batch_size or max_isl)
        # PR11509's KV axis contains zero, a batch-aligned minimum, powers of
        # two in block units, and an exact maximum. This upper bound prevents
        # its uniform limiter from deleting any generated candidate.
        max_total_kv_tokens = max_isl * batch_upper_bound
        max_kv_read_samples = max_total_kv_tokens.bit_length() + 2
        return cls(
            max_isl=max_isl,
            max_batch_size=max_batch_size,
            max_total_prefill_tokens=max_isl,
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=capture_sizes[-1] if capture_sizes is not None else None,
            new_token_axis_points=new_token_points,
            max_new_token_samples=max(2, len(new_token_points)) if new_token_points is not None else None,
            max_kv_read_token_samples=max_kv_read_samples,
            cudagraph_policy=cudagraph_policy,
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "max_isl": self.max_isl,
            "max_batch_size": self.max_batch_size,
            "max_total_prefill_tokens": self.max_total_prefill_tokens,
            "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes)
            if self.cudagraph_capture_sizes is not None
            else None,
            "cudagraph_capture_size_count": len(self.cudagraph_capture_sizes)
            if self.cudagraph_capture_sizes is not None
            else None,
            "max_cudagraph_capture_size": self.max_cudagraph_capture_size,
            "new_token_axis_points": list(self.new_token_axis_points)
            if self.new_token_axis_points is not None
            else None,
            "new_token_axis_point_count": len(self.new_token_axis_points)
            if self.new_token_axis_points is not None
            else None,
            "prefill_max_new_token_samples": self.max_new_token_samples,
            "prefill_max_kv_read_token_samples": self.max_kv_read_token_samples,
        }
        # Keep legacy explicit-capture plan identity unchanged. Runtime-selected
        # graphs and their candidate axes cannot be known before initialization.
        if self.cudagraph_policy == "runtime":
            payload["cudagraph_policy"] = "runtime"
        return payload


@dataclass(frozen=True, slots=True)
class FPMCollectionOptions:
    """Resolved FPM case-space controls.

    The options only narrow an AIC-declared/model-valid universe. They never
    grant support for a topology, quantization, or backend policy.
    """

    max_gpus: int
    gpu_counts: tuple[int, ...]
    parallel_presets: tuple[str, ...]
    parallel_axes: tuple[str, ...]
    # Explicit backend identity knobs (v6). The string backends use "auto"
    # (engine decides, row records "auto"); the two booleans have no auto -
    # they default to false and the row records a real boolean, so the
    # modeling-side str() normalization yields "False"/"True".
    moe_backend: str
    attention_backend: str
    enable_wideep: str
    enable_eplb: str
    weight_quantizations: tuple[str, ...]
    kv_cache_dtypes: tuple[str, ...]
    tp_sizes: tuple[int, ...] | None = None
    pp_sizes: tuple[int, ...] | None = None
    dp_sizes: tuple[int, ...] | None = None
    moe_tp_sizes: tuple[int, ...] | None = None
    moe_ep_sizes: tuple[int, ...] | None = None
    cp_sizes: tuple[int, ...] | None = None
    warmup_iterations: int = FPM_WARMUP_ITERATIONS
    # Keep the model context independent from the scheduled new-token budget.
    # vLLM 0.24 auto-fits this limit after model/CUDA-graph profiling, and
    # Dynamo records the resolved value in ``limits.max_model_len``.
    vllm_max_model_len: int = VLLM_AUTO_FIT_MAX_MODEL_LEN
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_prefill_isl: int | None = None
    max_prefill_batch_size: int | None = None
    max_prefill_cudagraph_size: int | None = None
    prefill_cudagraph_policy: str = "explicit"
    gpu_memory_utilization: float | None = None

    max_decode_batch_size: int | None = None
    decoder_replay: bool = False
    enforce_eager: bool = False
    benchmark_points_json: str | None = None
    benchmark_points_sha256: str | None = None
    executor: str = "kubernetes"
    slurm_container_image: str = ""
    slurm_container_mounts: tuple[str, ...] = ()
    # None preserves legacy saved-plan identity; fresh CLI plans resolve both.
    slurm_cpus_per_task: int | None = None
    slurm_cpu_bind: str | None = None

    def __post_init__(self) -> None:
        if (self.slurm_cpus_per_task is None) != (self.slurm_cpu_bind is None):
            raise ValueError("Slurm CPU policy requires both cpus-per-task and cpu-bind")
        if self.slurm_cpus_per_task is not None:
            if self.executor != "slurm":
                raise ValueError("Slurm CPU options require --fpm-executor slurm")
            if type(self.slurm_cpus_per_task) is not int or self.slurm_cpus_per_task < 1:
                raise ValueError("--fpm-slurm-cpus-per-task must be a positive integer")
            if self.slurm_cpu_bind not in {"cores", "none"}:
                raise ValueError("--fpm-slurm-cpu-bind must be cores or none")
        if self.prefill_cudagraph_policy not in {"runtime", "explicit"}:
            raise ValueError("--fpm-prefill-cudagraph-policy must be runtime or explicit")
        if self.prefill_cudagraph_policy == "runtime" and self.max_prefill_cudagraph_size is not None:
            raise ValueError("--fpm-prefill-cudagraph-policy runtime cannot use --fpm-max-prefill-cudagraph-size")
        if self.prefill_cudagraph_policy == "explicit" and self.max_prefill_cudagraph_size is None:
            object.__setattr__(self, "max_prefill_cudagraph_size", FPM_MAX_PREFILL_CUDAGRAPH_SIZE)
        if self.gpu_memory_utilization is not None and (
            type(self.gpu_memory_utilization) not in {int, float}
            or not math.isfinite(self.gpu_memory_utilization)
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("--fpm-gpu-memory-utilization must be finite and in (0, 1]")
        for name, value in (
            (
                "max-model-len",
                None if self.vllm_max_model_len == VLLM_AUTO_FIT_MAX_MODEL_LEN else self.vllm_max_model_len,
            ),
            ("max-num-batched-tokens", self.max_num_batched_tokens),
            ("max-num-seqs", self.max_num_seqs),
            ("max-decode-batch-size", self.max_decode_batch_size),
            ("max-prefill-cudagraph-size", self.max_prefill_cudagraph_size),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"--fpm-{name} must be a positive integer")
        if (
            self.max_prefill_isl is not None
            and self.max_num_batched_tokens is not None
            and self.max_prefill_isl > self.max_num_batched_tokens
        ):
            raise ValueError("--fpm-max-prefill-isl exceeds --fpm-max-num-batched-tokens")
        if (
            self.max_prefill_batch_size is not None
            and self.max_num_seqs is not None
            and self.max_prefill_batch_size > self.max_num_seqs
        ):
            raise ValueError("--fpm-max-prefill-batch-size exceeds --fpm-max-num-seqs")
        if (
            self.max_decode_batch_size is not None
            and self.max_num_seqs is not None
            and self.max_decode_batch_size > self.max_num_seqs
        ):
            raise ValueError("--fpm-max-decode-batch-size exceeds --fpm-max-num-seqs")
        self.validate_scheduler_limits()

    def validate_scheduler_limits(
        self, *, profile_max_num_tokens: int | None = None, profile_max_batch_size: int | None = None
    ) -> None:
        """Check known effective phase limits without inventing runtime defaults."""
        shared_tokens = self.max_num_batched_tokens or profile_max_num_tokens
        shared_sequences = self.max_num_seqs or profile_max_batch_size
        for phase, tokens, sequences in (
            ("decode", shared_tokens, self.max_decode_batch_size or shared_sequences),
            (
                "prefill",
                self.max_prefill_isl or shared_tokens or FPM_MAX_PREFILL_ISL,
                self.max_prefill_batch_size or shared_sequences,
            ),
        ):
            # vLLM v0.25.1 vllm/config/scheduler.py:286 rejects token budgets below
            # max_num_seqs: every scheduled sequence needs at least one token.
            if tokens is not None and sequences is not None and tokens < sequences:
                raise ValueError(
                    f"FPM {phase} max_num_batched_tokens ({tokens}) must be at least max_num_seqs ({sequences}); "
                    "edit the shared/prefill limits or profile bounds"
                )

    @property
    def prefill_sampling(self) -> PrefillSamplingProfile:
        return PrefillSamplingProfile.build(
            max_isl=self.max_prefill_isl
            if self.max_prefill_isl is not None
            else (self.max_num_batched_tokens or FPM_MAX_PREFILL_ISL),
            max_batch_size=self.max_prefill_batch_size
            if self.max_prefill_batch_size is not None
            else self.max_num_seqs,
            max_cudagraph_capture_size=self.max_prefill_cudagraph_size,
            cudagraph_policy=self.prefill_cudagraph_policy,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> FPMCollectionOptions:
        max_gpus = getattr(args, "fpm_max_gpus", None)
        if max_gpus is None:
            raise ValueError("--fpm-max-gpus is required with --ops fpm_forward")

        requested_counts = getattr(args, "fpm_gpu_counts", None)
        if requested_counts is None:
            counts = []
            value = 1
            while value <= max_gpus:
                counts.append(value)
                value *= 2
            if counts[-1] != max_gpus:
                counts.append(max_gpus)
        else:
            counts = sorted(set(requested_counts))
        over_limit = [count for count in counts if count > max_gpus]
        if over_limit:
            raise ValueError(f"--fpm-gpu-counts values exceed --fpm-max-gpus={max_gpus}: {over_limit}")

        explicit_presets = getattr(args, "fpm_parallel_presets", None)
        requested_axes = tuple(dict.fromkeys(getattr(args, "fpm_parallel_axes", None) or ()))
        requested_presets = tuple(dict.fromkeys(explicit_presets or (() if requested_axes else ("auto",))))
        if explicit_presets is not None and requested_axes:
            raise ValueError("--fpm-parallel-presets and legacy --fpm-parallel-axes cannot be combined")
        if "auto" in requested_presets and len(requested_presets) > 1:
            raise ValueError("parallel preset 'auto' cannot be combined with explicit presets")

        pp_sizes = _optional_size_list(getattr(args, "fpm_pp_sizes", None))
        cp_sizes = _optional_size_list(getattr(args, "fpm_cp_sizes", None))
        if pp_sizes not in {None, (1,)} or cp_sizes not in {None, (1,)}:
            raise ValueError("FPM typical-matrix V1 fixes PP=1 and CP=1")
        if {"pp", "cp"}.intersection(requested_axes):
            raise ValueError("FPM typical-matrix V1 does not vary PP or CP")

        executor = getattr(args, "fpm_executor", None) or "kubernetes"
        image = getattr(args, "fpm_slurm_container_image", None) or ""
        mounts = tuple(getattr(args, "fpm_slurm_container_mount", None) or ())
        cpus = getattr(args, "fpm_slurm_cpus_per_task", None)
        cpu_bind = getattr(args, "fpm_slurm_cpu_bind", None)
        if executor == "slurm" and not image:
            raise ValueError("--fpm-executor slurm requires --fpm-slurm-container-image")
        if executor != "slurm" and (image or mounts or cpus is not None or cpu_bind is not None):
            raise ValueError("Slurm container/CPU options require --fpm-executor slurm")
        if executor == "slurm":
            cpus, cpu_bind = resolve_slurm_cpu_policy(
                cpus,
                cpu_bind,
                resume=bool(getattr(args, "resume", False)),
                checkpoint_dir=getattr(args, "checkpoint_dir", None),
                artifact_root=getattr(args, "fpm_artifact_root", None) or "fpm_forward_artifacts",
                smoke=bool(getattr(args, "smoke", False)),
            )

        model_len = getattr(args, "fpm_max_model_len", None)
        if model_len is not None and model_len != -1 and model_len < 1:
            raise ValueError("--fpm-max-model-len must be positive or -1 for auto-fit")

        points_path = getattr(args, "fpm_benchmark_points_file", None)
        points_json = points_sha256 = None
        if points_path is not None:
            if getattr(args, "smoke", False):
                raise ValueError("--fpm-benchmark-points-file cannot be combined with --smoke")
            points_json, points_sha256 = _freeze_benchmark_points(points_path)

        return cls(
            benchmark_points_json=points_json,
            benchmark_points_sha256=points_sha256,
            max_gpus=max_gpus,
            gpu_counts=tuple(counts),
            parallel_presets=requested_presets,
            parallel_axes=requested_axes,
            moe_backend=(getattr(args, "fpm_moe_backend", None) or "auto").strip(),
            attention_backend=(getattr(args, "fpm_attention_backend", None) or "auto").strip(),
            enable_wideep=(getattr(args, "fpm_enable_wideep", None) or "false").strip(),
            enable_eplb=(getattr(args, "fpm_enable_eplb", None) or "false").strip(),
            weight_quantizations=tuple(
                dict.fromkeys(value.lower() for value in (getattr(args, "fpm_weight_quantizations", None) or ()))
            ),
            kv_cache_dtypes=tuple(dict.fromkeys(getattr(args, "fpm_kv_cache_dtypes", None) or ("auto",))),
            tp_sizes=_optional_size_list(getattr(args, "fpm_tp_sizes", None)),
            pp_sizes=pp_sizes,
            dp_sizes=_optional_size_list(getattr(args, "fpm_dp_sizes", None)),
            moe_tp_sizes=_optional_size_list(getattr(args, "fpm_moe_tp_sizes", None)),
            moe_ep_sizes=_optional_size_list(getattr(args, "fpm_moe_ep_sizes", None)),
            cp_sizes=cp_sizes,
            warmup_iterations=(
                FPM_WARMUP_ITERATIONS
                if getattr(args, "fpm_warmup_iterations", None) is None
                else args.fpm_warmup_iterations
            ),
            vllm_max_model_len=(
                VLLM_AUTO_FIT_MAX_MODEL_LEN
                if getattr(args, "fpm_max_model_len", None) is None
                else args.fpm_max_model_len
            ),
            max_num_batched_tokens=getattr(args, "fpm_max_num_batched_tokens", None),
            max_num_seqs=getattr(args, "fpm_max_num_seqs", None),
            max_decode_batch_size=getattr(args, "fpm_max_decode_batch_size", None),
            max_prefill_isl=getattr(args, "fpm_max_prefill_isl", None),
            max_prefill_batch_size=getattr(args, "fpm_max_prefill_batch_size", None),
            max_prefill_cudagraph_size=getattr(args, "fpm_max_prefill_cudagraph_size", None),
            prefill_cudagraph_policy=getattr(args, "fpm_prefill_cudagraph_policy", None) or "explicit",
            gpu_memory_utilization=getattr(args, "fpm_gpu_memory_utilization", None),
            decoder_replay=bool(getattr(args, "fpm_decoder_replay", False)),
            enforce_eager=bool(getattr(args, "fpm_enforce_eager", False)),
            executor=executor,
            slurm_container_image=image,
            slurm_container_mounts=mounts,
            slurm_cpus_per_task=cpus,
            slurm_cpu_bind=cpu_bind,
        )

    def to_dict(self) -> dict[str, object]:
        result = {
            "decoder_replay": self.decoder_replay,
            "executor": self.executor,
            "slurm_container_image": self.slurm_container_image,
            "slurm_container_mounts": list(self.slurm_container_mounts),
            "max_gpus": self.max_gpus,
            "gpu_counts": list(self.gpu_counts),
            "parallel_presets": list(self.parallel_presets),
            "parallel_axes": list(self.parallel_axes),
            "moe_backend": self.moe_backend,
            "attention_backend": self.attention_backend,
            "enable_wideep": self.enable_wideep,
            "enable_eplb": self.enable_eplb,
            "weight_quantizations": list(self.weight_quantizations),
            "kv_cache_dtypes": list(self.kv_cache_dtypes),
            "tp_sizes": list(self.tp_sizes) if self.tp_sizes is not None else None,
            "pp_sizes": list(self.pp_sizes) if self.pp_sizes is not None else None,
            "dp_sizes": list(self.dp_sizes) if self.dp_sizes is not None else None,
            "moe_tp_sizes": list(self.moe_tp_sizes) if self.moe_tp_sizes is not None else None,
            "moe_ep_sizes": list(self.moe_ep_sizes) if self.moe_ep_sizes is not None else None,
            "cp_sizes": list(self.cp_sizes) if self.cp_sizes is not None else None,
            "global_warmup_iterations": self.warmup_iterations,
            "vllm_max_model_len": self.vllm_max_model_len,
            "max_decode_batch_size": self.max_decode_batch_size,
            "warmup_repeats": 0,
            "measurement_repeats": FPM_MEASUREMENT_REPEATS,
            "point_source": "dynamo_native_self_benchmark",
            "prefill_sampling": self.prefill_sampling.to_dict(),
        }
        # Preserve the existing frozen-plan representation when the new shared
        # runtime limits are absent.
        if self.slurm_cpus_per_task is not None:
            result["slurm_cpus_per_task"] = self.slurm_cpus_per_task
            result["slurm_cpu_bind"] = self.slurm_cpu_bind
        if self.max_num_batched_tokens is not None:
            result["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.max_num_seqs is not None:
            result["max_num_seqs"] = self.max_num_seqs
        if self.gpu_memory_utilization is not None:
            result["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.enforce_eager:
            result["enforce_eager"] = True
        if self.benchmark_points_json is not None:
            result["benchmark_points"] = {
                "payload": json.loads(self.benchmark_points_json),
                "sha256": self.benchmark_points_sha256,
            }
        return result


def add_fpm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add FPM campaign controls to a collector or dedicated parser."""

    group = parser.add_argument_group(
        "FPM forward collection",
        "Whole-model forward-pass planning, execution, and publication.",
    )
    group.add_argument(
        "--fpm-model-profile",
        default=None,
        metavar="PATH",
        help="JSON or YAML FPM model profile with exact deployment identities and rank-local resource bounds.",
    )
    group.add_argument(
        "--fpm-runtime-instrumentation",
        default=None,
        metavar="PATH",
        help="Frozen instrumentation manifest for verifying collection against accepted runtime observations.",
    )
    group.add_argument(
        "--fpm-runtime-launch",
        default=None,
        metavar="PATH",
        help="JSON launch facts from the accepted runtime probe; requires runtime instrumentation and configuration.",
    )
    group.add_argument(
        "--fpm-runtime-configuration",
        default=None,
        metavar="KEY",
        help="Checkpoint configuration key for runtime observations; requires instrumentation and launch facts.",
    )
    group.add_argument(
        "--fpm-enforce-eager",
        action="store_true",
        default=None,
        help="Require native eager execution with CUDA graphs disabled; included in the frozen plan.",
    )
    group.add_argument(
        "--fpm-benchmark-points-file",
        metavar="PATH",
        default=None,
        help="Freeze a native Dynamo point manifest into the plan; cannot be combined with --smoke.",
    )
    group.add_argument(
        "--fpm-decoder-replay",
        action="store_true",
        default=None,
        help="Use true bounded decoder replay; requires verified runtime support.",
    )
    group.add_argument(
        "--fpm-max-model-len",
        type=_positive_or_auto_int,
        default=None,
        help="Maximum context length for both phases; defaults to the profile context or vLLM auto-fit (-1).",
    )
    group.add_argument(
        "--fpm-max-decode-batch-size",
        type=_positive_int,
        default=None,
        help="Optional native decode max-num-seqs bound.",
    )
    group.add_argument(
        "--fpm-max-gpus",
        type=_positive_int,
        default=None,
        help="Maximum GPUs used by one FPM deployment cell (required).",
    )
    group.add_argument(
        "--fpm-gpu-counts",
        nargs="+",
        type=_positive_int,
        default=None,
        help="Exact total-GPU counts to consider; defaults to powers of two up to the maximum.",
    )
    group.add_argument(
        "--fpm-parallel-presets",
        nargs="+",
        choices=PARALLEL_PRESETS,
        default=None,
        help="Typical deployment families: dense TP, MoE pure TP, TEP, or DEP.",
    )
    group.add_argument(
        "--fpm-parallel-axes",
        nargs="+",
        choices=PARALLEL_AXES,
        default=None,
        help="Deprecated compatibility filter; use --fpm-parallel-presets for new campaigns.",
    )
    group.add_argument(
        "--fpm-moe-backend",
        default=None,
        help="Engine MoE backend to pin (e.g. flashinfer_cutlass); default auto lets the engine decide.",
    )
    group.add_argument(
        "--fpm-attention-backend",
        default=None,
        help="Engine attention backend to pin; default auto lets the engine decide.",
    )
    group.add_argument(
        "--fpm-enable-wideep",
        choices=("true", "false"),
        default=None,
        help="Wide-EP on or off (boolean identity column; defaults to false).",
    )
    group.add_argument(
        "--fpm-enable-eplb",
        choices=("true", "false"),
        default=None,
        help="Expert-parallel load balancing on or off (boolean identity column; defaults to false).",
    )
    group.add_argument(
        "--fpm-weight-quantizations",
        nargs="+",
        default=None,
        help="Checkpoint-backed weight quantizations to retain; never changes checkpoint identity.",
    )
    group.add_argument(
        "--fpm-kv-cache-dtypes",
        nargs="+",
        default=None,
        help="Runtime KV-cache dtypes to collect; defaults to the generator/model setting.",
    )
    group.add_argument(
        "--fpm-model-config",
        default=None,
        metavar="PATH",
        help=(
            "Explicit local config.json (or directory containing it) for a model whose checkpoint is not "
            "locally accessible; the model remains identified by --model-path."
        ),
    )
    group.add_argument(
        "--fpm-warmup-iterations",
        type=_nonnegative_int,
        default=None,
        help=(
            "Dynamo scheduler global warmup decode iterations before the point sweep; "
            "does not repeat warmup for each point "
            f"(default: {FPM_WARMUP_ITERATIONS})."
        ),
    )
    group.add_argument(
        "--fpm-max-num-batched-tokens",
        type=_positive_int,
        default=None,
        help="Shared vLLM scheduled-token limit for prefill and decode; may be narrowed for prefill.",
    )
    group.add_argument(
        "--fpm-max-num-seqs",
        type=_positive_int,
        default=None,
        help="Shared vLLM sequence limit for prefill and decode; may be narrowed for prefill.",
    )
    group.add_argument(
        "--fpm-max-prefill-isl",
        dest="fpm_max_prefill_isl",
        type=_at_least_two_int,
        default=None,
        help=(
            "Maximum total scheduled prefill new-token axis; the batch=1 points also "
            "cover this per-request new-token length "
            "(defaults to the shared token limit, otherwise "
            f"{FPM_MAX_PREFILL_ISL} within supplied profile bounds)."
        ),
    )
    group.add_argument(
        "--fpm-max-prefill-batch-size",
        dest="fpm_max_prefill_batch_size",
        type=_positive_int,
        default=None,
        help="Optional narrower prefill max_num_seqs; defaults to shared/profile bounds or the Dynamo/vLLM value.",
    )
    group.add_argument(
        "--fpm-gpu-memory-utilization",
        type=_memory_fraction,
        default=None,
        help="vLLM fraction of total GPU memory for weights, runtime and KV cache; omitted uses the runtime default.",
    )
    group.add_argument(
        "--fpm-prefill-cudagraph-policy",
        choices=("runtime", "explicit"),
        default=None,
        help=(
            "Prefill graph policy: runtime leaves graph selection to vLLM; "
            "explicit retains the bounded capture axis (default)."
        ),
    )
    group.add_argument(
        "--fpm-max-prefill-cudagraph-size",
        dest="fpm_max_prefill_cudagraph_size",
        type=_positive_int,
        default=None,
        help=(
            "Largest prefill CUDA-graph capture size with the explicit policy; "
            f"defaults to {FPM_MAX_PREFILL_CUDAGRAPH_SIZE}. Cannot be combined with the runtime policy."
        ),
    )
    group.add_argument(
        "--fpm-artifact-root",
        default=None,
        help="Out-of-place root for generated artifacts, raw results, logs, and checkpoints.",
    )
    group.add_argument(
        "--fpm-database-root",
        default=None,
        help="Optional out-of-place systems/data root for formal database publication.",
    )
    group.add_argument(
        "--fpm-publish-partial",
        action="store_true",
        default=None,
        help="Publish rows from passed cells even when other cells failed (explicit opt-in; "
        "the checkpoint records exactly which cells are missing). Default keeps the "
        "all-cells-passed publication gate.",
    )
    for axis, option in (
        ("tp", "--fpm-tp-sizes"),
        ("pp", "--fpm-pp-sizes"),
        ("dp", "--fpm-dp-sizes"),
        ("moe_tp", "--fpm-moe-tp-sizes"),
        ("moe_ep", "--fpm-moe-ep-sizes"),
        ("cp", "--fpm-cp-sizes"),
    ):
        group.add_argument(
            option,
            nargs="+",
            type=_positive_int,
            default=None,
            help=f"Optional allowlist for the {axis} dimension.",
        )


def add_fpm_generator_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose deployment-only Generator inputs without importing its stack.

    Importing ``aisimulate.generator.api`` loads the rendering stack. That
    is appropriate during execution, but unnecessary for ``--plan-only`` and
    must not become a dependency of ordinary op-level collection.
    """

    group = parser.add_argument_group("FPM deployment inputs")
    group.add_argument(
        "--fpm-executor",
        choices=["kubernetes", "slurm"],
        default=None,
        help="Execution transport; Slurm runs inside an existing allocation.",
    )
    group.add_argument(
        "--fpm-slurm-container-image",
        default=None,
        help="Immutable Pyxis image reference or staged squashfs for Slurm.",
    )
    group.add_argument(
        "--fpm-slurm-container-mount",
        action="append",
        default=None,
        help="Pyxis SOURCE:TARGET mount; repeat for checkpoint/runtime/cache paths.",
    )
    group.add_argument(
        "--fpm-slurm-cpus-per-task",
        type=_positive_int,
        default=None,
        help="CPUs for each node's one-task Slurm step (new plans: 16); shared by local workers and schedulers.",
    )
    group.add_argument(
        "--fpm-slurm-cpu-bind",
        choices=["cores", "none"],
        default=None,
        help="Slurm node-task CPU binding (new plans: cores); no dedicated per-rank pinning.",
    )
    group.add_argument(
        "--generator-config",
        default=None,
        help="Deployment-only YAML containing supported K8sConfig fields.",
    )
    group.add_argument(
        "--generator-set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Inline deployment-only K8sConfig override.",
    )
    group.add_argument(
        "--config-template-version",
        "--generated-config-version",
        dest="generated_config_version",
        default=None,
        help="Rejected for FPM; the Collector resolves this from --dynamo-version.",
    )
    group.add_argument(
        "--dynamo-version",
        "--generator-dynamo-version",
        dest="generator_dynamo_version",
        default=None,
        help="Target Dynamo release used to resolve the Generator template.",
    )
    group.add_argument("--namespace", default=None)
    group.add_argument("--model-cache", default=None, metavar="NAME[:MOUNT[:SUBPATH]]")
    group.add_argument("--transport", choices=["nvlink", "ib", "efa"], default=None)
    group.add_argument(
        "--fpm-orchestrator",
        choices=["lws", "grove"],
        default=None,
        help="Multi-node FPM resource orchestrator; defaults to lws.",
    )
    group.add_argument("--image-pull-secret", dest="image_pull_secret", default=None)


def reject_fpm_arguments_without_fpm(args: argparse.Namespace) -> None:
    """Fail when a user supplies FPM-only controls without selecting the op."""

    if FPM_FORWARD_OP in (args.ops or ()):
        return
    explicitly_set = []
    for name in (
        "fpm_max_gpus",
        "fpm_gpu_counts",
        "fpm_weight_quantizations",
        "fpm_kv_cache_dtypes",
        "fpm_model_config",
        "fpm_model_profile",
        "fpm_runtime_instrumentation",
        "fpm_runtime_launch",
        "fpm_runtime_configuration",
        "fpm_tp_sizes",
        "fpm_pp_sizes",
        "fpm_dp_sizes",
        "fpm_moe_tp_sizes",
        "fpm_moe_ep_sizes",
        "fpm_cp_sizes",
        "fpm_parallel_axes",
        "fpm_parallel_presets",
        "fpm_moe_backend",
        "fpm_attention_backend",
        "fpm_enable_wideep",
        "fpm_enable_eplb",
        "fpm_warmup_iterations",
        "fpm_max_model_len",
        "fpm_max_num_batched_tokens",
        "fpm_max_num_seqs",
        "fpm_gpu_memory_utilization",
        "fpm_prefill_cudagraph_policy",
        "fpm_max_prefill_isl",
        "fpm_max_prefill_batch_size",
        "fpm_max_prefill_cudagraph_size",
        "fpm_artifact_root",
        "fpm_max_decode_batch_size",
        "fpm_decoder_replay",
        "fpm_enforce_eager",
        "fpm_benchmark_points_file",
        "fpm_executor",
        "fpm_slurm_container_image",
        "fpm_slurm_container_mount",
        "fpm_slurm_cpus_per_task",
        "fpm_slurm_cpu_bind",
        "fpm_database_root",
        "fpm_publish_partial",
        # Deployment-only Generator inputs are registered unconditionally on
        # the collector CLI (an argv pre-scan cannot track argparse's own
        # abbreviation semantics), so the explicit-only discipline is enforced
        # here for them too.
        "generator_config",
        "generator_set",
        "generated_config_version",
        "generator_dynamo_version",
        "namespace",
        "model_cache",
        "transport",
        "fpm_orchestrator",
        "image_pull_secret",
    ):
        if getattr(args, name, None) is not None:
            explicitly_set.append("--" + name.replace("_", "-"))
    if explicitly_set:
        raise ValueError("FPM-only arguments require --ops fpm_forward: " + ", ".join(explicitly_set))
