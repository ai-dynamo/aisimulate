# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallelism dimension registry and versioned validity rule set.

Parallelism is an open map of dimension name to degree (``{"tp": 4, "ep": 8}``).
The registry lists known dimensions; unknown names need an ``x-`` prefix and are
flagged. Validity rules live here, not in the dataset: each rule has a scope
(universal, framework, model), a severity (error, warn) and a verification
record. A rule whose inputs are unknown is skipped and reported as an evidence
gap, never guessed.

Convention. ``tp`` is the attention tensor-parallel degree inside one
attention-DP rank. The GPUs of one pipeline stage are therefore
``tp * attention_dp * cp``, and ``moe_tp * moe_ep * moe_cluster`` partition that
same group. Frameworks spell this differently (SGLang's ``tp_size`` and
TensorRT-LLM's ``tp_size`` already include attention DP); the framework rules
below state the mapping they checked.

Rule verification records name the source revision that was read. Rules marked
``verified=False`` are capped at ``warn`` until they are checked against each
framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

RULE_SET_VERSION = "parallelism-rules/v1"

Scope = Literal["universal", "framework", "model"]
Severity = Literal["error", "warn"]
# How a dimension combines into the GPU count of one pipeline stage group.
WorldRole = Literal["multiplies", "partitions"]


@dataclass(frozen=True)
class Dimension:
    name: str
    meaning: str
    applies_to: tuple[str, ...]  # attention | moe | prefill | decode | model
    world_role: WorldRole  # multiplies: factor of world size; partitions: divides an existing group
    partitions: str | None = None  # group that a "partitions" dimension subdivides


DIMENSIONS: dict[str, Dimension] = {
    d.name: d
    for d in (
        Dimension("tp", "Attention tensor parallel degree within one attention-DP rank", ("attention",), "multiplies"),
        Dimension("pp", "Pipeline parallel stages", ("model",), "multiplies"),
        Dimension(
            "attention_dp",
            "Data-parallel attention replicas that share MoE/expert layers",
            ("attention",),
            "multiplies",
        ),
        Dimension("cp", "Prefill/attention context parallel degree", ("attention", "prefill"), "multiplies"),
        Dimension(
            "moe_tp", "Tensor parallel degree inside expert layers", ("moe",), "partitions", "tp*attention_dp*cp"
        ),
        Dimension("ep", "Expert parallel degree", ("moe",), "partitions", "tp*attention_dp*cp"),
        Dimension("moe_cluster", "TensorRT-LLM MoE cluster degree", ("moe",), "partitions", "tp*attention_dp*cp"),
        Dimension("moe_dp", "SGLang MoE data-parallel degree", ("moe",), "partitions", "tp*attention_dp*cp"),
        Dimension("dcp", "Decode context parallel degree (reuses TP ranks)", ("decode",), "partitions", "tp*cp"),
    )
}
EXTENSION_PREFIX = "x-"
_WORLD_DIMS = tuple(n for n, d in DIMENSIONS.items() if d.world_role == "multiplies")


def normalize(parallelism: Mapping[str, int]) -> dict[str, int]:
    """Sorted keys with degree 1 removed, so IDs stay stable when a dimension is added at 1."""
    return {k: parallelism[k] for k in sorted(parallelism) if parallelism[k] != 1}


def world_size(parallelism: Mapping[str, int]) -> int:
    """GPUs implied by the multiplying dimensions (absent dimensions count as 1)."""
    size = 1
    for name in _WORLD_DIMS:
        size *= parallelism.get(name, 1)
    return size


@dataclass(frozen=True)
class Verification:
    verified: bool
    sources: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Context:
    """Everything a rule may read. ``None`` means unknown."""

    parallelism: Mapping[str, int]
    framework: str | None = None  # vllm | sglang | trtllm
    model: str | None = None  # model family key such as "minimax_m3"
    parallelism_complete: bool = False  # absent dimensions are 1 when true, unknown otherwise
    declared_world_size: int | None = None
    gpus_per_node: int | None = None
    num_experts: int | None = None
    num_attention_heads: int | None = None
    cp_type: str | None = None  # ULYSSES | STAR | HELIX (TensorRT-LLM)
    attention_dp_enabled: bool | None = None  # source boolean, kept apart from the degree


_MISSING = object()


def _get(ctx: Context, name: str, default=_MISSING):
    """Degree of ``name``; None when absent and the map is not complete."""
    if name in ctx.parallelism:
        return ctx.parallelism[name]
    if ctx.parallelism_complete:
        return 1
    return None if default is _MISSING else default


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    message: str
    rule_set_version: str = RULE_SET_VERSION


@dataclass(frozen=True)
class EvidenceGap:
    rule_id: str
    missing: tuple[str, ...]
    rule_set_version: str = RULE_SET_VERSION


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    gaps: list[EvidenceGap] = field(default_factory=list)
    rule_set_version: str = RULE_SET_VERSION

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    def blocks_import(self) -> bool:
        """Universal errors block import."""
        return any(f.severity == "error" and RULES_BY_ID[f.rule_id].scope == "universal" for f in self.findings)

    def blocks_qualification(self) -> bool:
        """Any error (universal, framework or model) blocks qualification; archival is still allowed."""
        return bool(self.errors)


class _Unknown(Exception):
    def __init__(self, *missing: str):
        self.missing = missing


def _need(**values):
    missing = tuple(k for k, v in values.items() if v is None)
    if missing:
        raise _Unknown(*missing)
    return values


@dataclass(frozen=True)
class Rule:
    rule_id: str
    scope: Scope
    severity: Severity
    description: str
    verification: Verification
    check: Callable[[Context], str | None]  # returns a message on violation
    frameworks: tuple[str, ...] = ()  # for scope == "framework"
    models: tuple[str, ...] = ()  # for scope == "model"

    def applies(self, ctx: Context) -> bool:
        if self.scope == "framework":
            return ctx.framework in self.frameworks
        if self.scope == "model":
            return ctx.model in self.models
        return True


# ---------------------------------------------------------------- rule bodies


def _positive(ctx: Context) -> str | None:
    bad = {k: v for k, v in ctx.parallelism.items() if isinstance(v, bool) or not isinstance(v, int) or v < 1}
    return f"degrees must be positive integers, got {bad}" if bad else None


def _registered(ctx: Context) -> str | None:
    unknown = [k for k in ctx.parallelism if k not in DIMENSIONS and not k.startswith(EXTENSION_PREFIX)]
    return (
        f"unregistered dimensions {unknown}; register them or use the '{EXTENSION_PREFIX}' prefix" if unknown else None
    )


def _world_consistent(ctx: Context) -> str | None:
    vals = _need(declared=ctx.declared_world_size)
    degs = {n: _get(ctx, n) for n in _WORLD_DIMS}
    if any(v is None for v in degs.values()):
        raise _Unknown(*(f"parallelism.{n}" for n, v in degs.items() if v is None))
    implied = world_size(degs)
    return (
        f"declared world size {vals['declared']} != tp*pp*attention_dp*cp = {implied}"
        if implied != vals["declared"]
        else None
    )


def _node_fit(ctx: Context) -> str | None:
    _need(gpus_per_node=ctx.gpus_per_node)
    degs = {n: _get(ctx, n) for n in _WORLD_DIMS}
    if any(v is None for v in degs.values()):
        raise _Unknown("parallelism (incomplete)")
    w = world_size(degs)
    g = ctx.gpus_per_node
    if w > g and w % g:
        return f"world size {w} is not a whole number of {g}-GPU nodes"
    return None


def _moe_world(ctx: Context) -> str | None:
    """moe_tp * moe_ep (* moe_cluster) must cover the attention group. Model-scope seed and TRT-LLM rule."""
    tp, dp, cp = _get(ctx, "tp"), _get(ctx, "attention_dp"), _get(ctx, "cp")
    mtp, ep, cl = _get(ctx, "moe_tp"), _get(ctx, "ep"), _get(ctx, "moe_cluster")
    for name, v in (("tp", tp), ("attention_dp", dp), ("cp", cp), ("moe_tp", mtp), ("ep", ep), ("moe_cluster", cl)):
        if v is None:
            raise _Unknown(f"parallelism.{name}")
    if mtp * ep * cl != tp * dp * cp:
        return f"moe_tp*ep*moe_cluster = {mtp * ep * cl} != tp*attention_dp*cp = {tp * dp * cp}"
    return None


def _minimax_moe(ctx: Context) -> str | None:
    # sdk/models/minimax_m3.py: tp * attention_dp == moe_tp * moe_ep
    tp, dp = _get(ctx, "tp"), _get(ctx, "attention_dp")
    mtp, ep = _get(ctx, "moe_tp"), _get(ctx, "ep")
    for name, v in (("tp", tp), ("attention_dp", dp), ("moe_tp", mtp), ("ep", ep)):
        if v is None:
            raise _Unknown(f"parallelism.{name}")
    if tp * dp != mtp * ep:
        return f"tp*attention_dp = {tp * dp} != moe_tp*ep = {mtp * ep}"
    if ctx.num_experts is not None and ep > ctx.num_experts:
        return f"ep ({ep}) exceeds num_experts ({ctx.num_experts})"
    return None


def _vllm_ep(ctx: Context) -> str | None:
    # vLLM expert parallelism spans TP x PCP x DP ranks; without it experts are TP-sharded (ep = 1).
    ep = ctx.parallelism.get("ep", 1)
    span = _need(tp=_get(ctx, "tp"), attention_dp=_get(ctx, "attention_dp"), cp=_get(ctx, "cp"))
    group = span["tp"] * span["attention_dp"] * span["cp"]
    if ep not in (1, group):
        return f"vLLM ep must be 1 or tp*attention_dp*cp = {group}, got {ep}"
    return None


def _vllm_dcp(ctx: Context) -> str | None:
    dcp = ctx.parallelism.get("dcp", 1)
    tp, pcp = _need(tp=_get(ctx, "tp"), cp=_get(ctx, "cp")).values()
    if pcp == 1:
        return f"tp ({tp}) must be divisible by dcp ({dcp})" if tp % dcp else None
    valid = sorted({1, pcp, tp * pcp})
    return f"with cp={pcp}, dcp must be one of {valid}, got {dcp}" if dcp not in valid else None


def _vllm_heads(ctx: Context) -> str | None:
    v = _need(heads=ctx.num_attention_heads, tp=_get(ctx, "tp"))
    return f"num_attention_heads ({v['heads']}) must be divisible by tp ({v['tp']})" if v["heads"] % v["tp"] else None


def _trtllm_cluster(ctx: Context) -> str | None:
    cluster, ep = _get(ctx, "moe_cluster"), _get(ctx, "ep")
    _need(moe_cluster=cluster, ep=ep)
    return f"moe_cluster ({cluster}) > 1 requires ep == 1, got {ep}" if cluster > 1 and ep != 1 else None


def _trtllm_ep_cp(ctx: Context) -> str | None:
    ep, cp = _get(ctx, "ep"), _get(ctx, "cp")
    _need(ep=ep, cp=cp, cp_type=ctx.cp_type)
    if ep != 1 and cp > 1 and ctx.cp_type.upper() != "HELIX":
        return f"ep ({ep}) with cp ({cp}) is only supported for cp_type HELIX, got {ctx.cp_type}"
    return None


def _sglang_moe_dp(ctx: Context) -> str | None:
    mdp = ctx.parallelism.get("moe_dp", 1)
    pp = _get(ctx, "pp")
    _need(pp=pp)
    return "SGLang moe_dp > 1 requires pp == 1" if mdp > 1 and pp != 1 else None


def _expert_divisible(ctx: Context) -> str | None:
    v = _need(num_experts=ctx.num_experts, ep=_get(ctx, "ep"))
    return (
        f"num_experts ({v['num_experts']}) is not divisible by ep ({v['ep']})" if v["num_experts"] % v["ep"] else None
    )


def _heads_divisible(ctx: Context) -> str | None:
    v = _need(heads=ctx.num_attention_heads, tp=_get(ctx, "tp"))
    return f"num_attention_heads ({v['heads']}) is not divisible by tp ({v['tp']})" if v["heads"] % v["tp"] else None


def _bool_agrees(ctx: Context) -> str | None:
    dp = _get(ctx, "attention_dp")
    _need(attention_dp_enabled=ctx.attention_dp_enabled, attention_dp=dp)
    if ctx.attention_dp_enabled != (dp > 1):
        return f"source boolean attention_dp_enabled={ctx.attention_dp_enabled} disagrees with attention_dp degree {dp}"
    return None


_VLLM = "vllm-project/vllm@80e0cd68f812bfef5e6e7cffef3350daa01ff222"
_TRT = "NVIDIA/TensorRT-LLM@ecf1fefa365fee24bea20d1236efc660975f3750"
_SGL = "sgl-project/sglang@ad73e4f5e48edebc397dafde256a2d3176e1202f"
_UNVERIFIED = Verification(False, note="not yet checked against vLLM, SGLang and TensorRT-LLM; capped at warn")

RULES: tuple[Rule, ...] = (
    Rule(
        "U001-positive-degrees",
        "universal",
        "error",
        "Every degree is a positive integer.",
        Verification(
            True,
            (f"{_VLLM}:vllm/config/parallel.py Field(ge=1)", f"{_TRT}:tensorrt_llm/mapping.py world_size check"),
        ),
        _positive,
    ),
    Rule(
        "U002-registered-dimensions",
        "universal",
        "warn",
        "Dimension names are registered or carry the x- prefix.",
        Verification(True, note="registry policy from DEP, not a framework constraint"),
        _registered,
    ),
    Rule(
        "U003-world-size-consistent",
        "universal",
        "error",
        "Declared world size equals pp * tp * attention_dp * cp.",
        Verification(
            True,
            (
                f"{_VLLM}:vllm/config/parallel.py world_size = pp*tp*pcp (*dp externally)",
                f"{_TRT}:tensorrt_llm/mapping.py world_size == tp*pp*cp (tp includes attention DP)",
                f"{_SGL}:python/sglang/srt/server_args.py tp_size is the stage world size, dp divides it",
            ),
            "Holds under the registry convention (tp = attention TP); framework spellings differ.",
        ),
        _world_consistent,
    ),
    Rule(
        "U004-node-layout",
        "universal",
        "warn",
        "World size above one node is a whole number of nodes of the system spec.",
        _UNVERIFIED,
        _node_fit,
    ),
    Rule(
        "U005-expert-divisibility",
        "universal",
        "warn",
        "num_experts divisible by ep.",
        Verification(
            False,
            (f"{_SGL}:layers/moe/token_dispatcher/deepep.py asserts divisibility for low-latency DeepEP only",),
            "vLLM and TensorRT-LLM were not found to require it; keep at warn.",
        ),
        _expert_divisible,
    ),
    Rule(
        "U006-head-divisibility",
        "universal",
        "warn",
        "num_attention_heads divisible by tp (vLLM enforces it; see F-vllm-heads).",
        _UNVERIFIED,
        _heads_divisible,
    ),
    Rule(
        "U007-boolean-numeric-agreement",
        "universal",
        "warn",
        "Source attention-DP boolean agrees with the attention_dp degree.",
        _UNVERIFIED,
        _bool_agrees,
    ),
    Rule(
        "M001-minimax-m3-moe-world",
        "model",
        "error",
        "tp * attention_dp == moe_tp * ep and ep <= num_experts.",
        Verification(True, ("aisim_core/sdk/models/minimax_m3.py:78-83",)),
        _minimax_moe,
        models=("minimax_m3",),
    ),
    Rule(
        "F-trtllm-moe-world",
        "framework",
        "error",
        "moe_tp * ep * moe_cluster == tp * attention_dp * cp.",
        Verification(
            True,
            (f"{_TRT}:tensorrt_llm/mapping.py:170-173 moe_world_size = tp (* cp unless ULYSSES)",),
            "TensorRT-LLM tp already includes attention DP, hence tp*attention_dp here. "
            "ULYSSES cp is not multiplied in upstream; this rule assumes non-ULYSSES.",
        ),
        _moe_world,
        frameworks=("trtllm",),
    ),
    Rule(
        "F-trtllm-cluster-ep",
        "framework",
        "error",
        "moe_cluster > 1 requires ep == 1.",
        Verification(True, (f"{_TRT}:tensorrt_llm/mapping.py:186-187",)),
        _trtllm_cluster,
        frameworks=("trtllm",),
    ),
    Rule(
        "F-trtllm-ep-cp",
        "framework",
        "error",
        "ep > 1 with cp > 1 is only supported for the HELIX cp type.",
        Verification(True, (f"{_TRT}:tensorrt_llm/mapping.py:182-185",)),
        _trtllm_ep_cp,
        frameworks=("trtllm",),
    ),
    Rule(
        "F-vllm-ep",
        "framework",
        "error",
        "ep is 1 or spans tp * attention_dp * cp (enable_expert_parallel).",
        Verification(
            True,
            (f"{_VLLM}:vllm/config/parallel.py:613-621,1393-1396 EP group spans TP x PCP x DP",),
            "Published FPM profiles tp1-dp4-ep4, tp4-ep4 and tp2 follow it.",
        ),
        _vllm_ep,
        frameworks=("vllm",),
    ),
    Rule(
        "F-vllm-dcp",
        "framework",
        "error",
        "dcp divides tp when cp == 1; with cp > 1 dcp is one of 1, cp, tp*cp.",
        Verification(True, (f"{_VLLM}:vllm/config/parallel.py:636-649",)),
        _vllm_dcp,
        frameworks=("vllm",),
    ),
    Rule(
        "F-vllm-heads",
        "framework",
        "error",
        "num_attention_heads divisible by tp.",
        Verification(True, (f"{_VLLM}:vllm/config/model.py:1575-1582",)),
        _vllm_heads,
        frameworks=("vllm",),
    ),
    Rule(
        "F-sglang-moe-dp-pp",
        "framework",
        "error",
        "moe_dp > 1 requires pp == 1.",
        Verification(True, (f"{_SGL}:python/sglang/srt/server_args.py:5308-5314",)),
        _sglang_moe_dp,
        frameworks=("sglang",),
    ),
)
RULES_BY_ID = {r.rule_id: r for r in RULES}
assert len(RULES_BY_ID) == len(RULES)


def evaluate(ctx: Context, rule_set_version: str = RULE_SET_VERSION) -> Report:
    """Run every applicable rule. Unverified rules never exceed ``warn``; unknown inputs become gaps."""
    if rule_set_version != RULE_SET_VERSION:
        raise ValueError(f"unsupported rule set {rule_set_version!r}; this library ships {RULE_SET_VERSION!r}")
    report = Report()
    # Later rules assume valid integers, so stop at the positive-degree check.
    first = RULES_BY_ID["U001-positive-degrees"]
    msg = first.check(ctx)
    if msg:
        report.findings.append(Finding(first.rule_id, first.severity, msg))
        return report
    for rule in RULES:
        if rule is first or not rule.applies(ctx):
            continue
        try:
            msg = rule.check(ctx)
        except _Unknown as exc:
            report.gaps.append(EvidenceGap(rule.rule_id, exc.missing))
            continue
        if msg:
            severity: Severity = rule.severity if rule.verification.verified else "warn"
            report.findings.append(Finding(rule.rule_id, severity, msg))
    return report
