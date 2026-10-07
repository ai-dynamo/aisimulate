# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check measured FP8 cells through Rust and the canonical GLM model API.

The temporary systems copy has no other monolithic DSA donors. This makes
the model's chosen precision observable: a BF16 attention query cannot
silently succeed against an unrelated table. No production data is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

import aisimulate._runtime
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common
from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.errors import PerfDataNotAvailableError
from aisimulate_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems-root", required=True, type=Path)
    parser.add_argument("--context-parquet", required=True, type=Path)
    parser.add_argument("--generation-parquet", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    inputs = {"context": args.context_parquet, "generation": args.generation_parquet}
    rows = {phase: pq.read_table(path).to_pylist() for phase, path in inputs.items()}
    for phase_rows in rows.values():
        for row in phase_rows:
            assert row["model"] == "zai-org/GLM-5.2-FP8"
            assert row["architecture"] == "GlmMoeDsaForCausalLM"
            assert row["gemm_type"] == "fp8_block" and row["num_heads"] == 8
            assert row["mla_dtype"] == "bfloat16" and row["kv_cache_dtype"] == "fp8"
            assert row["framework"] == "SGLang" and row["version"] == "0.5.14"

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fp8-consumer-", dir=args.scratch_root) as tmp:
        systems = Path(tmp) / "systems"
        shutil.copytree(args.systems_root, systems)
        removed = []
        for phase in inputs:
            basename = f"dsa_{phase}_module_perf.parquet"
            for path in sorted(systems.rglob(basename)):
                removed.append(str(path.relative_to(systems)))
                path.unlink()
            target = systems / "data/b200_sxm/sparse_attention/sglang/0.5.14" / basename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inputs[phase], target)

        database = PerfDatabase(
            system="b200_sxm",
            backend="sglang",
            version="0.5.14",
            systems_root=str(systems),
            database_mode="SILICON",
            strict_provenance=False,
        )
        handle = EngineHandle.for_database(database, systems_path=str(systems))
        direct = []
        grouped = {}
        for phase, phase_rows in rows.items():
            for row in phase_rows:
                skip = "skip_indexer" in row["op_name"]
                kwargs = dict(architecture=row["architecture"], dsa_full_layer_fraction=0.0 if skip else 1.0)
                if phase == "context":
                    op = ContextDSAModule(
                        "measured_fp8",
                        1,
                        8,
                        common.KVCacheQuantMode.fp8,
                        common.FMHAQuantMode.bfloat16,
                        common.GEMMQuantMode.fp8_block,
                        **kwargs,
                    )
                else:
                    op = GenerationDSAModule(
                        "measured_fp8",
                        1,
                        8,
                        common.KVCacheQuantMode.fp8,
                        common.GEMMQuantMode.fp8_block,
                        **kwargs,
                    )
                result = handle.evaluate_ops_json(
                    "[" + op._spec_json() + "]",
                    is_context=phase == "context",
                    batch_size=row["batch_size"],
                    s=row["isl"] if phase == "context" else row["isl"] + row["step"],
                    prefix=row["step"] if phase == "context" else 0,
                )[0]
                assert math.isclose(result[1], row["latency"], rel_tol=1e-12, abs_tol=1e-12)
                direct.append(dict(row=row, sdk_latency_ms=result[1], source=result[3]))
                key = (phase, row["batch_size"], row["isl"], row["step"])
                grouped.setdefault(key, {})["skip" if skip else "full"] = row["latency"]

        request = ForwardPassPerfModelConfig(
            model="zai-org/GLM-5.2-FP8",
            system="b200_sxm",
            backend="sglang",
            backend_version="0.5.14",
            worker_type="aggregated",
            tp=8,
            moe_tp_size=1,
            moe_ep_size=8,
            kvcache_quant_mode="fp8",
            estimation_mode="op_level",
            systems_paths=(str(systems),),
            strict_provenance=False,
        )
        model = RustForwardPassPerfModel.best_available(request)
        canonical = []
        try:
            for (phase, batch, isl, step), pair in grouped.items():
                assert set(pair) == {"full", "skip"}
                # Measurement ledger, not a Python performance estimate.
                expected = 21 * pair["full"] + 57 * pair["skip"]
                point = dict(phase=phase, batch_size=batch, isl=isl, step=step, measured_attention_sum_ms=expected)
                try:
                    operations = model.static_phase_diagnostics(
                        batch_size=batch,
                        context_length=isl + step if phase == "context" else isl + step - 1,
                        prefix=step if phase == "context" else 0,
                        prefill=phase == "context",
                    )
                except PerfDataNotAvailableError as error:
                    point["lookup_error"] = str(error)
                else:
                    attention = next(op for op in operations if op["name"] == f"{phase}_attention")
                    point.update(
                        sdk_attention=attention,
                        exact=math.isclose(attention["latency_ms"], expected, rel_tol=1e-12, abs_tol=1e-12),
                    )
                canonical.append(point)
        finally:
            model.close()

    result = dict(
        label=args.label,
        accuracy_acceptance="NOT_EVALUATED",
        runtime_sha256=hashlib.sha256(Path(aisimulate._runtime.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={phase: hashlib.sha256(path.read_bytes()).hexdigest() for phase, path in inputs.items()},
        removed_dsa_sources=removed,
        direct_exact_cells=direct,
        canonical_model_cells=canonical,
        limitations=[
            "Private isolated systems root with strict_provenance=False; default packaged data unchanged.",
            "Exact cell consumption validates model precision and layer accounting, not prediction accuracy.",
            "Synthetic single-GPU module data uses TP8 head geometry; not a whole-model forward reference.",
            "Prefill includes the native single-module compiler entry cost; whole-model amortization unqualified.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
