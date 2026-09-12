# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Includes changes adapted from:
# https://github.com/ai-dynamo/aiconfigurator/blob/6290c161a354da5250c391bd43372b2e9c6f4a51/aic-core/src/aiconfigurator_core/sdk/speculation/materialize.py

"""Compile-time materialization of a speculative scheme into the model.

The compiled engine is the only step executor ("Python builds, Rust
executes"): per-op pricing happens in Rust from an ``EngineSpec`` built by
walking the model's op lists once. A scheme therefore cannot inject draft
ops at query time the way the retired Python step path allowed — instead,
``materialize_spec_scheme`` folds the scheme into the model right after
``get_model`` attaches it, using two engine channels:

* **verify width** rides the engine's ``nextn`` decode-batch multiplier:
  ``model._nextn = verify_width - 1`` is set POST-construction, so model
  families built their op counts with ``nextn=0`` (no MTP layer scaling is
  baked in) while the Rust runtime scales the generation batch by
  ``(_nextn + 1)`` — exactly the verify forward width. The agg scheduler's
  ``decode_query_tokens`` and ``max_decode_progress`` read the same
  attribute and stay consistent for free.
* **draft-op token widths** use a ``TokenScale`` engine wrapper. It maps
  both query tokens and batch size by ``tokens_per_request / verify_width``
  before dispatching to the original op. Every operation family retains its
  own nonlinear table lookup, token divisors, and repetition count.

**Dense decode attention** additionally carries the width channel end to
end: ``GenerationAttention`` (Python and Rust) folds the engine's width
multiplier out of its batch (``scale_num_tokens``, sequence basis — one KV
read per request) and prices a roofline guard from ``verify_query_tokens``
(lift to ideal-peak math of the full query block when compute, not the
shared KV read, would dominate — physics puts the crossover far above
practical widths, so the guard is a corner-case fence, not a calibration).
Exact wide-query pricing still requires collected data with a query_len
axis (delivery-package issue #7).

Measurements below are historical upstream reports, not rerun for this migration.

Transition-state limitations (tracked for follow-ups):

* MLA / DSA / linear-attention modules have no width channel yet: on those
  targets (e.g. DeepSeek-V4) verify attention keeps the conservative
  token-basis price.
* ``DraftOpSpec.query_overrides`` (e.g. a sliding-window KV cap) cannot
  ride the wire; schemes requiring them fail before materialization.
* Per-round host-side scheduling glue (draft assembly, acceptance
  processing, KV bookkeeping) is NOT modeled: it is stack-version software
  cost outside the op tables, and small (~1.7 ms chain / ~3.6 ms tree per
  round on the H200 v2 serving ledger).
* Predictions are graph-basis (the op tables are collected in-graph).
  Deployments must extend the engine's CUDA-graph capture sizes to cover
  ``max_concurrency x verify_width`` tokens (vLLM's default 512-token
  ceiling is insufficient for speculation); the measured off-graph cliff
  is ~5x round time and is a deployment configuration issue, not a
  modeling target.

Draft op names are prefixed ``draft_`` to keep the breakdown keys identical
to the retired runtime-injection path (and to keep a standalone draft
model's op names from colliding with the target's).
"""

from __future__ import annotations

import copy

from aiconfigurator_core.sdk.operations.attention import GenerationAttention
from aiconfigurator_core.sdk.speculation.base import NullScheme, SpecSchemeBase, positive_integer
from aiconfigurator_core.sdk.speculation.mtp import MTPScheme


def _fold_width(op, tokens_per_request: int, verify_width: int) -> None:
    tokens_per_request = positive_integer(tokens_per_request, "tokens_per_request")
    verify_width = positive_integer(verify_width, "verify_width")
    if isinstance(op, GenerationAttention):
        # Attention folds by the FULL width regardless of divisibility: a
        # block pass reads each request's KV once, so the correct price is
        # the request batch ("sequence basis"), with the real query width
        # carried separately for the engine's roofline guard.
        op._scale_num_tokens = (op._scale_num_tokens or 1) * verify_width
        op._verify_query_tokens = max(int(tokens_per_request), 1)
        return
    # The engine adapter serializes this marker as a typed query wrapper;
    # keeping the original Python op preserves its metadata and copy protocol.
    # Unsupported native graphs still fail at the standard conversion gate.
    op._draft_token_width = (tokens_per_request, verify_width)


def materialize_spec_scheme(model) -> None:
    """Fold ``model.spec_scheme`` into the op lists and width channel.

    No-op for the null scheme and for MTP (whose draft cost is already part
    of the model families' op construction under the legacy nextn contract).
    Idempotent per model instance (guarded by a marker attribute); model
    instances are created fresh per ``get_model`` call, so sweeps never see
    a double-materialized model.
    """
    scheme = getattr(model, "spec_scheme", None)
    # Exact-type Null check: a scheme SUBCLASSING NullScheme that overrides the
    # draft hooks is a draft scheme; MTP's cost is legacy-handled via nextn.
    if not isinstance(scheme, SpecSchemeBase) or isinstance(scheme, MTPScheme) or type(scheme) is NullScheme:
        return
    if getattr(model, "_spec_scheme_materialized", False):
        return

    verify_width = scheme.verify_width()

    generation_specs = scheme.build_draft_generation_ops(model)
    context_specs = scheme.build_draft_context_ops(model)
    # Validate both phases before modifying the model. Silently dropping these
    # overrides changes the requested workload (notably DSpark V4's KV cap).
    for spec in [*generation_specs, *context_specs]:
        if spec.query_overrides:
            raise ValueError("DraftOpSpec.query_overrides are unsupported by the compiled engine")

    # Shallow-copy before renaming/folding: the specs' op objects belong to
    # the scheme (some schemes cache a built draft model), and callers may
    # re-inspect build_draft_*_ops after materialization.
    for spec in generation_specs:
        op = copy.copy(spec.op)
        op._name = f"draft_{op._name}"
        _fold_width(op, spec.tokens_per_request, verify_width)
        model.generation_ops.append(op)

    for spec in context_specs:
        op = copy.copy(spec.op)
        op._name = f"draft_{op._name}"
        # Context-phase draft precompute runs over the same prompt tokens as
        # the target prefill: no width folding.
        model.context_ops.append(op)

    if verify_width > 1 and scheme.verify_attention_sequence_basis():
        # Target verify attention: the engine widens the decode batch by
        # verify_width, but the block-verify kernel reads each request's KV
        # once for all width query tokens — fold the batch back to the
        # request basis and carry the real query width for the roofline
        # guard. Dense decode attention only for now; MLA/DSA/linear
        # attention modules have no width channel yet and keep the
        # conservative token-basis price (documented transition limitation).
        for op in model.generation_ops:
            if isinstance(op, GenerationAttention) and not op._name.startswith("draft_"):
                op._scale_num_tokens = (op._scale_num_tokens or 1) * verify_width
                op._verify_query_tokens = verify_width

    # Set LAST: the draft-op builders above must see the model exactly as
    # constructed (some derive geometry from it), and the engine identity
    # memo keys off the final attribute state.
    model._nextn = verify_width - 1
    model._spec_scheme_materialized = True
