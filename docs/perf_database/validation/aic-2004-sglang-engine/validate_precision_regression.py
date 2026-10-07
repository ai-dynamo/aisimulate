# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record real SDK precision keys and queries for the prediction-gate cases."""

import argparse
import hashlib
import json
from pathlib import Path

import aisimulate._runtime
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, common, config, engine, models
from aisimulate_core.sdk.errors import PerfDataNotAvailableError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--systems-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    cases = [
        (None, "b300_sxm", "sglang", "0.5.14"),
        ("fp8", "b200_sxm", "vllm", "0.24.0"),
        ("nvfp4", "b200_sxm", "sglang", "0.5.14"),
    ]
    for quant, system, backend, version in cases:
        request = ForwardPassPerfModelConfig(
            model="zai-org/GLM-5.2-FP8",
            system=system,
            backend=backend,
            backend_version=version,
            worker_type="aggregated",
            tp=8,
            moe_tp_size=1,
            moe_ep_size=8,
            gemm_quant_mode=quant,
            moe_quant_mode=quant,
            kvcache_quant_mode="fp8",
            estimation_mode="op_level",
            systems_paths=(str(args.systems_root.resolve()),),
            strict_provenance=False,
        )
        operation_model = models.get_model(
            request.model,
            config.ModelConfig(
                tp_size=8,
                moe_tp_size=1,
                moe_ep_size=8,
                gemm_quant_mode=common.GEMMQuantMode[quant] if quant else None,
                moe_quant_mode=common.MoEQuantMode[quant] if quant else None,
                kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            ),
            backend_name=backend,
        )
        model = RustForwardPassPerfModel.best_available(request)
        try:
            for phase in ["context", "generation"]:
                specs = json.loads(engine._ops_json(getattr(operation_model, f"{phase}_ops")))
                attention = next(
                    fields for spec in specs for tag, fields in spec.items() if tag == f"Dsa{phase.capitalize()}"
                )
                point = dict(
                    system=system,
                    backend=backend,
                    backend_version=version,
                    quant_override=quant,
                    phase=phase,
                    batch_size=1,
                    context_length=1024,
                    attention_spec=attention,
                )
                try:
                    operations = model.static_phase_diagnostics(
                        batch_size=1, context_length=1024, prefill=phase == "context"
                    )
                except PerfDataNotAvailableError as error:
                    point.update(status="DATA_MISS", error=str(error))
                else:
                    point.update(
                        status="OK", attention=next(op for op in operations if op["name"] == f"{phase}_attention")
                    )
                rows.append(point)
        finally:
            model.close()
    args.output.write_text(
        json.dumps(
            dict(
                label=args.label,
                accuracy_acceptance="NOT_EVALUATED",
                runtime_sha256=hashlib.sha256(Path(aisimulate._runtime.__file__).read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cases=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
