# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record real Rust SDK results; no Python performance-estimation formulas.

Run once with the baseline wheel and once with the fixed wheel. A systems root
may contain either the complete collected rows or a training-only subset for
held-out validation. All original framework/version/schema columns must remain
unchanged. This script never rewrites or installs performance data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import aisimulate._runtime
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--systems-root", type=Path, required=True)
    parser.add_argument("--decode-history", type=int, nargs="+", default=[8192, 131072, 1048575])
    parser.add_argument("--context", nargs="+", default=["128:128", "1024:131072"])
    args = parser.parse_args()
    root = args.systems_root.resolve()
    database = PerfDatabase(
        system="b200_sxm",
        backend="sglang",
        version="0.5.14",
        systems_root=str(root),
        database_mode="SILICON",
        strict_provenance=False,
    )
    # Diagnostic op-only boundary: values still come exclusively from Rust.
    handle = EngineHandle.for_database(database, systems_path=str(root))
    rows = []
    context_shapes = [tuple(map(int, value.split(":"))) for value in args.context]
    shapes = [(True, q, p) for q, p in context_shapes]
    shapes += [(False, history + 1, 0) for history in args.decode_history]
    for prefill, sequence, prefix in shapes:
        for name, fraction in [("full", 1.0), ("skip", 0.0), ("glm52", 21 / 78)]:
            kwargs = dict(architecture="GlmMoeDsaForCausalLM", dsa_full_layer_fraction=fraction)
            scale = 78 if name == "glm52" else 1
            if prefill:
                op = ContextDSAModule(
                    name,
                    scale,
                    8,
                    common.KVCacheQuantMode.fp8,
                    common.FMHAQuantMode.bfloat16,
                    common.GEMMQuantMode.bfloat16,
                    **kwargs,
                )
            else:
                op = GenerationDSAModule(
                    name,
                    scale,
                    8,
                    common.KVCacheQuantMode.fp8,
                    common.GEMMQuantMode.bfloat16,
                    **kwargs,
                )
            op_json = "[" + op._spec_json() + "]"
            coordinates = dict(is_context=prefill, batch_size=1, s=sequence, prefix=prefix)
            measured_query = handle.evaluate_ops_json(op_json, **coordinates)[0]
            roofline = handle.evaluate_ops_sol_json(op_json, **coordinates)[0]
            rows.append(
                dict(
                    phase="context" if prefill else "generation",
                    variant=name,
                    batch_size=1,
                    query_sequence=sequence,
                    prefix=prefix,
                    history=None if prefill else sequence - 1,
                    op_spec=json.loads(op._spec_json()),
                    latency_ms=measured_query[1],
                    source=measured_query[3],
                    sol_ms=roofline[1],
                    sol_math_ms=roofline[2],
                    sol_memory_ms=roofline[3],
                )
            )
    # Whole-model composition uses the one canonical public constructor.
    request = ForwardPassPerfModelConfig(
        model="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        backend="sglang",
        backend_version="0.5.14",
        worker_type="aggregated",
        tp=8,
        moe_tp_size=1,
        moe_ep_size=8,
        kvcache_quant_mode="fp8",
        estimation_mode="op_level",
        systems_paths=(str(root),),
        strict_provenance=False,
    )
    model = RustForwardPassPerfModel.best_available(request)
    whole = []
    try:
        for history in args.decode_history:
            ops = model.static_phase_diagnostics(batch_size=1, context_length=history, prefill=False)
            whole.append(dict(history=history, operations=ops, predicted_total_ms=sum(o["latency_ms"] for o in ops)))
    finally:
        model.close()
    binary = Path(aisimulate._runtime.__file__)
    data_root = root / "data/b200_sxm/sparse_attention/sglang/0.5.14"
    receipt = dict(
        label=args.label,
        accuracy_acceptance="NOT_EVALUATED",
        runtime_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configuration={k: v for k, v in request.to_dict().items() if k != "systems_paths"},
        input_tables={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(data_root.glob("dsa_*module_perf.parquet"))
        },
        measurements=rows,
        whole_model_predictions=whole,
        limitations=[
            "Single-GPU module measurements use TP8 head geometry; not an eight-GPU whole forward.",
            "A collected coordinate tests table identity; held-out coordinates test extrapolation separately.",
            "No matching stock SGLang 0.5.14 whole-model reference: whole-model MAPE is not evaluated.",
            "Synthetic inputs do not establish checkpoint-value parity.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
