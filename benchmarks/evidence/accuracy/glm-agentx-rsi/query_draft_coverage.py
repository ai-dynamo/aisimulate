#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe BF16 MoE lookup coverage; this whole-target proxy is not a draft model."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Coverage output already exists")
    receipt = json.loads(args.canonical_receipt.read_bytes())
    config = ForwardPassPerfModelConfig(**receipt["canonical_config"])
    if (config.tp != 8 or config.attention_dp != 1 or config.backend != "sglang"
            or config.backend_version != "0.5.21" or config.nextn != 3
            or config.estimation_mode != "op_level" or config.database_mode != "SILICON"
            or config.fallback_policy != "deny"):
        raise ValueError("Requires the frozen canonical B300 TP8 op-data identity")
    result = {"schema": "aisimulate.glm-rsi.draft-coverage-proxy.v1",
              "scope": "Whole-target MoE-precision lookup probe only; not actual draft phase/shape geometry, "
                       "a supported modeling fix, GPU measurement or E2E accuracy result.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "canonical_receipt_sha256": hashlib.sha256(args.canonical_receipt.read_bytes()).hexdigest(),
              "planned_queries": 12, "attempted_queries": 0, "queries": []}
    for precision in (config.moe_quant_mode, "bfloat16"):
        model = None
        try:
            model = RustForwardPassPerfModel.best_available(replace(config, moe_quant_mode=precision))
            for lanes in (1, 8, 14):
                for context in (8192, 393216):
                    metrics = [{"scheduled_requests": {"num_decode_requests": lanes * 4,
                                "sum_decode_kv_tokens": lanes * 4 * context}}]
                    row = {"moe_quant_mode": precision, "lanes": lanes, "kv_tokens": context,
                           "metrics_by_attention_dp_rank": metrics}
                    result["attempted_queries"] += 1
                    try:
                        detail = model.estimate_forward_pass_detailed(metrics)
                        latency = detail["latency_ms"]
                        if not math.isfinite(latency) or latency <= 0:
                            raise ValueError("Invalid detailed-query latency")
                        row.update(status="priced", detail=detail)
                    except Exception as error:
                        row.update(status="unsupported_or_failed", error_type=type(error).__name__, error=str(error))
                    result["queries"].append(row)
                    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        except Exception as error:
            result.setdefault("construction_failures", []).append(
                {"moe_quant_mode": precision, "error_type": type(error).__name__, "error": str(error)})
        finally:
            if model is not None:
                model.close()
    result["unattempted_queries"] = result["planned_queries"] - result["attempted_queries"]
    result["priced_queries"] = sum(row["status"] == "priced" for row in result["queries"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("planned_queries", "attempted_queries",
                                                  "unattempted_queries", "priced_queries")}))


if __name__ == "__main__":
    main()
