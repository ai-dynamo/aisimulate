# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place that keeps cases out of a collection.

An entry in ``SKIPS`` skips a case when every key matches:

- ``backend``, ``op`` and ``sm``: the run;
- ``model``: the ``architecture`` of the case file a model row comes from;
- ``dtype``: the case's dtype or quant mode;
- ``dispatch``: a kernel the op plan picks for the backend; the entry leaves
  the case out on that kernel only, and it still runs on any other;
- any other key: the case field (or property) of that name.

A list matches any of its values; a callable is a predicate on the value. An
entry without ``sm`` applies on every SM. A ``model`` entry drops that file's
rows before cases are merged, so a shape another model also needs is still
collected; besides ``model`` it may name only ``backend``, ``op`` and ``sm``.

Add an entry only when the cases would be mismeasured, would hang, would fail
as a whole block, or come from a model the backend cannot serve. Write the
reason as a comment above the entry, and edit or delete the entry when the
reason stops being true. Everything else runs, and its failures are recorded.

SM lists name the SMs the collector targets: 80, 86, 89, 90, 100, 103, 120.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

_DSA_AND_SPARSE_OPS = [
    "dsa_context_module",
    "dsa_generation_module",
    "dsa_context_module_skip_indexer",
    "dsa_generation_module_skip_indexer",
    "dsv4_paged_mqa_logits_module",
    "dsv4_csa_topk_calib",
    "glm5_mqa_logits_module",
    "glm5_topk_module",
    "glm5_dsa_attn_module",
]
_DSV4_ATTENTION_OPS = [
    "dsv4_csa_context_module",
    "dsv4_csa_generation_module",
    "dsv4_hca_context_module",
    "dsv4_hca_generation_module",
]
_ATTENTION_OPS = ["attention_context", "attention_generation"]

SKIPS: list[dict] = [
    # The sparse-attention (DSA) modules and the DeepSeek-V4 and GLM-5 sparse
    # kernels (DeepGEMM fp8_mqa_logits, FlashMLA sparse, JIT top-k) need Hopper
    # (SM90) or newer.
    {"op": _DSA_AND_SPARSE_OPS, "sm": [80, 86, 89]},
    # DeepSeek-V4 attention runs only on FlashMLA (SM90, SM10x) and SM100+ paths;
    # on SM89 serving fails the same way.
    {"op": _DSV4_ATTENTION_OPS, "sm": [80, 86, 89]},
    # compute_scale times FP8 quantization, which needs SM89 or newer.
    {"op": "compute_scale", "sm": [80, 86]},
    # FP8 tensor cores start at Ada (SM89).
    {"dtype": ["fp8", "fp8_static", "fp8_block"], "sm": [80, 86]},
    # MXFP4 weight-only MoE starts at SM89: SGLang maps no runner below it and
    # TRT-LLM runs it on SM90 only.
    {"dtype": "w4a16_mxfp4", "sm": [80, 86]},
    # FP4 tensor cores, which NVFP4 and MXFP8 x MXFP4 need, start at Blackwell (SM100).
    {"dtype": ["nvfp4", "w4a8_mxfp4_mxfp8"], "sm": [80, 86, 89, 90]},
    # SGLang: the collector times fp8_block GEMMs through DeepGEMM, which has no
    # SM89 or SM120 kernel, so every case would fail. Serving runs them on
    # Triton (SM89) and CUTLASS (SM120).
    {"backend": "sglang", "op": "gemm", "dtype": "fp8_block", "sm": [89, 120]},
    # SGLang: the MoE collector has no per-tensor fp8 path, so every case would
    # fail. Serving runs ModelOpt FP8 checkpoints through ModelOptFp8MoEMethod.
    {"backend": "sglang", "op": "moe", "dtype": "fp8", "sm": [89, 90, 100, 103, 120]},
    # SGLang: a MoE case runs on the runner serving picks, pinned per quant mode
    # and SM in cases/base_ops/moe.yaml (moe_sglang.backends). int4_wo has no pin
    # on SM80/86 and w4a8_mxfp4_mxfp8 none on SM120, so these cases would time a
    # guessed runner.
    {"backend": "sglang", "op": "moe", "dtype": "int4_wo", "sm": [80, 86]},
    {"backend": "sglang", "op": "moe", "dtype": "w4a8_mxfp4_mxfp8", "sm": [120]},
    # TRT-LLM has no MoE implementation for int4_wo (W4A16) on any SM: building
    # the layer rejects every case.
    {"backend": "trtllm", "op": "moe", "dtype": "int4_wo", "sm": [80, 86, 89, 90, 100, 103, 120]},
    # TRT-LLM: on SM89 and SM120 only CutlassFusedMoE serves these formats, and
    # it runs fp8_block on SM90/120 only and w4a16_mxfp4 on SM90 only, so
    # building the layer rejects every case.
    {"backend": "trtllm", "op": "moe", "dtype": "fp8_block", "sm": [89]},
    {"backend": "trtllm", "op": "moe", "dtype": "w4a16_mxfp4", "sm": [89, 120]},
    # vLLM never splits routed experts by TP and EP together: with EP on it
    # turns the whole TP x DP world into EP, so a case with tp > 1 and ep > 1
    # would time that EP layout under the wrong key.
    {"backend": "vllm", "op": "moe", "parallel": "tp_ep", "sm": [80, 86, 89, 90, 100, 103, 120]},
    # vLLM serves MXFP4 MoE on SM120 without MXFP8 activations (that backend
    # runs only on explicit request), so these cases would time a path serving
    # does not take.
    {"backend": "vllm", "op": "moe", "dtype": "w4a8_mxfp4_mxfp8", "sm": [120]},
    # vLLM runs both MLA generation BMMs as bf16 torch.bmm on NVIDIA GPUs (its
    # fp8 BMMs are ROCm only), so an fp8 row would label a kernel vLLM never runs.
    {"backend": "vllm", "op": ["mla_bmm_gen_pre", "mla_bmm_gen_post"], "dtype": "fp8", "sm": [89, 90, 100, 103, 120]},
    # SGLang: the MLA collector builds each case on the attention backend
    # serving picks, which it maps for SM89 and newer only.
    {"backend": "sglang", "op": ["mla_context", "mla_generation"], "dtype": "bfloat16", "sm": [80, 86]},
    # SGLang: the attention collector maps a serving backend for SM89 and newer
    # only (SGLang's default attention backend table), so every SM80/86 case
    # would fail.
    {"backend": "sglang", "op": _ATTENTION_OPS, "sm": [80, 86]},
    # SGLang: fa3 (head_dim <= 256) and trtllm_mha quantize Q to FP8 themselves
    # when the KV cache is FP8 (flashattention_backend.py, trtllm_mha_backend.py),
    # so a BF16-compute prefill on an FP8 KV cache does not exist there: the row
    # would time FP8 compute under a bfloat16 label. It is the FP8-compute case.
    {
        "backend": "sglang",
        "op": "attention_context",
        "dispatch": "trtllm_mha",
        "kv_cache_dtype": "fp8",
        "attn_dtype": "bfloat16",
    },
    {
        "backend": "sglang",
        "op": "attention_context",
        "dispatch": "fa3",
        "head_dim": lambda head_dim: head_dim <= 256,
        "kv_cache_dtype": "fp8",
        "attn_dtype": "bfloat16",
    },
    # SGLang: the flashinfer prefill plans BF16 Q against the FP8 KV cache
    # (flashinfer_backend.py) and has no FP8-compute path, so an FP8-compute row
    # would time BF16 compute under an fp8 label.
    {"backend": "sglang", "op": "attention_context", "dispatch": "flashinfer", "attn_dtype": "fp8"},
    # vLLM: the attention collector has no FP8-compute (FP8 FMHA) input, so an
    # FP8-compute row would re-time the FP8-KV case under an fp8 label.
    {"backend": "vllm", "op": "attention_context", "attn_dtype": "fp8"},
    # TRT-LLM on SM90: FP8-KV decode with 96 query heads over 4 KV heads aborts
    # the worker (launch failure, then SIGABRT) at every head_dim, batch and KV
    # length (h200 attention_generation shard, 2026-10-07: 512 cases, none passing).
    {
        "backend": "trtllm",
        "op": "attention_generation",
        "sm": [90],
        "kv_cache_dtype": "fp8",
        "num_heads": 96,
        "num_kv_heads": 4,
    },
    # TRT-LLM on SM90: FP8-KV full-attention decode at head_dim 256 with 32 query
    # heads per KV head aborts the worker once the KV length reaches 255 (same
    # shard: 169 cases); shorter KV lengths pass at every batch.
    {
        "backend": "trtllm",
        "op": "attention_generation",
        "sm": [90],
        "kv_cache_dtype": "fp8",
        "head_dim": 256,
        "window_size": 0,
        "q_per_kv": 32,
        "input_len": lambda input_len: input_len >= 255,
    },
    # SGLang: the mHC collector has no GLM-5.3-Flash route (the GLM call sites
    # were never ported to SGLang's current mHC module), so every case would fail.
    {"backend": "sglang", "op": "mhc_module", "model": "Glm5NextForConditionalGeneration"},
    # vLLM: mhc_module_perf has no architecture in its key, and DeepSeek-V4-Flash
    # shares (hc_mult, hidden_size) with GLM-5.3-Flash, whose rows vLLM
    # collects, so DeepSeek-V4 rows would merge with GLM's.
    {"backend": "vllm", "op": "mhc_module", "model": "DeepseekV4ForCausalLM"},
    # TRT-LLM cannot serve MiMo: it registers no model class for it.
    {"backend": "trtllm", "model": "MiMoForCausalLM"},
    # TRT-LLM cannot serve MiMo-V2-Flash: it registers no model class for it.
    # On SM100/103 its head_dim 192 attention also has no kernel.
    {"backend": "trtllm", "model": "MiMoV2FlashForCausalLM"},
    # TRT-LLM cannot serve Muse Glimmer: it registers no model class for it.
    {"backend": "trtllm", "model": "MuseGlimmerForConditionalGeneration"},
    # TRT-LLM cannot serve GLM-5.3-Flash: it registers no model class for it,
    # and its mHC collector drives only DeepSeek-V4's module API.
    {"backend": "trtllm", "model": "Glm5NextForConditionalGeneration"},
]

_RUN_KEYS = frozenset(("backend", "op", "sm"))
_MISSING = object()

# Case attribute names that carry a dtype or quant mode.
_DTYPE_FIELDS = ("moe_type", "gemm_type", "dtype", "kv_cache_dtype", "compute_dtype")
# Boolean case attributes that imply FP8.
_FP8_FLAG_FIELDS = ("use_fp8_kv_cache", "use_fp8_context_fmha")


def _check(skips: list[dict]) -> None:
    for index, entry in enumerate(skips):
        where = f"SKIPS[{index}]"
        if not isinstance(entry, dict) or not entry:
            raise ValueError(f"{where} must be a non-empty dict")
        for key, value in entry.items():
            if value in ([], (), "", None):
                raise ValueError(f"{where}.{key} must not be empty")
        if "model" in entry:
            extra = sorted(set(entry) - _RUN_KEYS - {"model"})
            if extra:
                raise ValueError(f"{where}: a model entry may name only backend, op and sm besides model, not {extra}")
            if not all(isinstance(model, str) for model in _values(entry["model"])):
                raise ValueError(f"{where}.model must name case-file architectures")
        if "sm" in entry and not all(type(sm) is int for sm in _values(entry["sm"])):
            raise ValueError(f"{where}.sm must be SM versions")


def _values(want) -> list:
    return list(want) if isinstance(want, (list, tuple, set, frozenset)) else [want]


def _matches(want, value) -> bool:
    if callable(want):
        return bool(want(value))
    return value in want if isinstance(want, (list, tuple, set, frozenset)) else value == want


def _applies(entry: dict, backend: str | None, op: str | None, sm: int | None) -> bool:
    """Whether an entry's run keys match this backend, op and SM."""
    for key, value in (("backend", backend), ("op", op), ("sm", sm)):
        if key in entry and (value is None or not _matches(entry[key], value)):
            return False
    return True


def model_entries(backend: str, op: str | None, sm: int | None) -> list[tuple[dict, frozenset[str]]]:
    """The ``model`` entries that apply to this backend, op and SM, each with the architectures it names.

    With ``op`` None, only the entries that leave a model out of every op.
    """
    return [
        (entry, frozenset(_values(entry["model"])))
        for entry in SKIPS
        if "model" in entry and _applies(entry, backend, op, sm)
    ]


def models(backend: str, op: str | None, sm: int | None) -> frozenset[str]:
    """The case-file architectures whose model rows this backend leaves out of an op on an SM."""
    return frozenset().union(*(architectures for _entry, architectures in model_entries(backend, op, sm)))


def split(
    cases: Iterable,
    backend: str,
    op: str,
    sm: int | None,
    dtypes_of: Callable[[object], Iterable[str]],
    *,
    dispatch_entries: bool | None = None,
) -> tuple[list, list[tuple[object, dict, str]]]:
    """Split cases into the kept ones and ``(case, entry, region)`` for each skipped one.

    ``dtypes_of`` gives a case's dtypes for the ``dtype`` key; ``region`` names
    what the entry matched, such as ``dtype=fp8_block`` or ``op=compute_scale``.
    ``dispatch_entries`` applies only the entries that name a ``dispatch``
    (True), only the others (False), or all of them (None).
    """
    entries = [
        entry
        for entry in SKIPS
        if "model" not in entry
        and _applies(entry, backend, op, sm)
        and (dispatch_entries is None or ("dispatch" in entry) == dispatch_entries)
    ]
    cases = list(cases)
    if not entries:
        return cases, []
    keys = sorted({key for entry in entries for key in entry} - _RUN_KEYS)
    found: dict[tuple, tuple[dict, str] | None] = {}
    kept, skipped = [], []
    for case in cases:
        values = tuple(tuple(dtypes_of(case)) if key == "dtype" else getattr(case, key, _MISSING) for key in keys)
        try:
            match = found[values]
        except KeyError:
            match = found[values] = _first_match(entries, keys, values, op)
        except TypeError:  # an unhashable field value: match without the cache
            match = _first_match(entries, keys, values, op)
        if match is None:
            kept.append(case)
        else:
            skipped.append((case, *match))
    return kept, skipped


def _first_match(entries: list[dict], keys: list[str], values: tuple, op: str) -> tuple[dict, str] | None:
    by_key = dict(zip(keys, values, strict=True))
    for entry in entries:
        region = []
        for key in entry:
            if key in _RUN_KEYS:
                continue
            want, value = entry[key], by_key[key]
            if key == "dtype":
                value = next((dtype for dtype in value if _matches(want, dtype)), _MISSING)
            elif value is not _MISSING and not _matches(want, value):
                value = _MISSING
            if value is _MISSING:
                break
            region.append(f"{key}={value}")
        else:
            return entry, ",".join(region) or f"op={op}"
    return None


def case_dtypes(test_case) -> list[str]:
    """Return the dtype/quant names a case requires (typed attribute access)."""
    dtypes = []
    for field in _DTYPE_FIELDS:
        value = getattr(test_case, field, None)
        if isinstance(value, str) and value:
            dtypes.append(value)
    for field in _FP8_FLAG_FIELDS:
        if getattr(test_case, field, False):
            dtypes.append("fp8")
    return dtypes


def filter_cases(test_cases, backend: str, op: str, sm_version: int | None):
    """Drop the cases an entry skips before queueing; return ``(kept, skipped)``.

    Cases are matched on their typed attributes, so a positional (list) case is
    skipped only by an entry that names no case field.
    """
    kept, skipped = split(test_cases, backend, op, sm_version, case_dtypes, dispatch_entries=False)
    if skipped:
        counts = Counter(repr(entry) for _case, entry, _region in skipped)
        summary = "; ".join(f"{count}x {entry}" for entry, count in sorted(counts.items()))
        logger.info(f"{op}: skipped {len(skipped)}/{len(kept) + len(skipped)} cases before queueing ({summary})")
    return kept, skipped


def detect_sm_version() -> int | None:
    """Detect the SM version of the local CUDA device, or None (e.g. XPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return major * 10 + minor
    except Exception:
        pass
    return None


_check(SKIPS)
