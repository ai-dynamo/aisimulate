# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail before model loading unless the image provides native FPM + KV warm-up."""

from __future__ import annotations

import json
import os
from pathlib import Path

_AUDIT_PATH = Path("/results/runtime-preflight.json")

GRAPH_AWARE_FIELDS = {
    "point_type",
    "benchmark_id",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "batch_size",
}
GRAPH_AWARE_METHODS = {
    "_bench_prefill_scheduled_tokens_per_req",
    "_bench_prefill_blocks_per_req",
    "_bench_blocks_per_req",
    "_bench_available_blocks",
    "_bench_usable_blocks",
    "_bench_prefill_point_feasible",
    "_bench_decode_point_feasible",
    "_bench_cudagraph_metadata",
    "_bench_seed_prompt_len",
    "_bench_cache_fake_prefixes",
    "_bench_save_current_point",
    "_bench_write_results",
    "_kvwarm_warm_eligible",
}


def _write_audit(audit: dict) -> None:
    _AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def main() -> None:
    phase = os.environ.get("FPM_BENCHMARK_MODE")
    real_prefill_requested = phase in {"prefill", "agg"} and os.environ.get(
        "DYN_BENCH_PREFILL_REAL_SEED", "off"
    ).lower() in {"on", "1", "true"}
    dsv41_adapter = os.environ.get("DYN_FPM_DSV41_REAL_KV") == "1"
    # The audit artifact exists precisely to document a rejected image, so an
    # image whose runtime module is missing entirely (pre-PR11509) must still
    # produce it before this process fails the pod.
    try:
        from dynamo.vllm.instrumented_scheduler import BenchmarkPoint, InstrumentedScheduler

        if dsv41_adapter:
            from dsv41_scheduler import DeepseekV41RealKVScheduler

            if InstrumentedScheduler is not DeepseekV41RealKVScheduler:
                raise RuntimeError("V4.1 source-checked scheduler activation did not occur")
            # Include the shared native SDK import in the rejected-image audit.
            from aisimulate_core.sdk.fpm_identity import execution_identity

            if not callable(execution_identity):
                raise RuntimeError("V4.1 runtime lacks the shared AISimulate identity helper")
    except Exception as error:
        _write_audit(
            {
                "schema_version": 1,
                "runtime_contract": "dynamo_pr11509_native_schema_v2_kvwarm_v1",
                "benchmark_point_fields": [],
                "missing_fields": [],
                "missing_methods": [],
                "import_error": str(error),
                "status": "failed",
            }
        )
        raise RuntimeError(
            "Dynamo runtime lacks the required native FPM/KV-warm contract; "
            f"runtime activation or identity preflight failed: {error}. "
            "Provide a compatible Dynamo image."
        ) from error

    fields = set(getattr(BenchmarkPoint, "__dataclass_fields__", {}))
    missing_fields = sorted(GRAPH_AWARE_FIELDS - fields)
    required_methods = set(GRAPH_AWARE_METHODS)
    if real_prefill_requested and not dsv41_adapter:
        # Native cached-prefill seeding is separate from decode warm-up:
        # https://github.com/ai-dynamo/dynamo/blob/b83b1d9304ebfc624709ac46db32b1b6f1ff1615/components/src/dynamo/vllm/instrumented_scheduler.py#L3326
        # The source-checked V4.1 adapter owns its own real-KV prefill path.
        required_methods.add("_bench_realseed_on")
    missing_methods = sorted(name for name in required_methods if not hasattr(InstrumentedScheduler, name))
    audit = {
        "schema_version": 1,
        "runtime_contract": "dynamo_pr11509_native_schema_v2_kvwarm_v1",
        "benchmark_point_fields": sorted(fields),
        "missing_fields": missing_fields,
        "missing_methods": missing_methods,
        "prefill_real_seed_requested": real_prefill_requested,
        "prefill_real_seed_implementation": "dsv41_adapter" if dsv41_adapter else "dynamo_native",
        "status": "passed" if not missing_fields and not missing_methods else "failed",
    }
    _write_audit(audit)
    if missing_fields or missing_methods:
        raise RuntimeError(
            "Dynamo runtime lacks the required native FPM/KV-warm contract; "
            f"missing_fields={missing_fields}, missing_methods={missing_methods}. "
            "Provide a compatible Dynamo image."
        )


if __name__ == "__main__":
    main()
