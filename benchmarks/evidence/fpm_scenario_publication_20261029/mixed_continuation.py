# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic adapter qualification; no silicon accuracy or performance claim.

Run with an installed AISim runtime, or supply --native-library to test a local
extension against this checkout's Python SDK. Requires the Gym dependencies and
pytest. Reuses the SDK's phase_roots and mixed_seed fixtures: prefill 2 ms,
decode 3 ms, decode-zero 1 ms, hence the native mixed prior is 4 ms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, help="Optional AISim _runtime extension to load explicitly.")
    parser.add_argument("--output", type=Path, required=True, help="Write compact synthetic evidence as JSON.")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(repository), str(repository / "python/aisimulate/tests")]
    if args.native_library is not None:
        sys.path.insert(0, str(repository / "python/aisimulate/src"))
        library = args.native_library.resolve(strict=True)
        spec = importlib.util.spec_from_file_location("aisimulate._runtime", library)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the supplied native extension.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    import pytest
    from unit.sdk.test_fpm_mixed_composition_integration import mixed_seed
    from unit.sdk.test_fpm_phase_readiness import phase_roots

    from aisimulate import _runtime
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned
    from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
    from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA
    from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics

    def row(index, kind, actual_ms):
        return ForwardPassIteration.single_rank(
            ForwardPassMetric(
                fpm_id=index,
                configuration_id="synthetic",
                version=1,
                worker_id="worker",
                dp_rank=0,
                counter_id=index,
                wall_time_s=actual_ms / 1000,
                scheduled=RequestMetrics(
                    num_prefill_requests=int(kind != "decode"),
                    sum_prefill_tokens=int(kind != "decode"),
                    num_decode_requests=int(kind != "prefill"),
                    sum_decode_kv_tokens=int(kind != "prefill"),
                ),
                queued=RequestMetrics(),
                capture_id="synthetic",
            )
        )

    results = []
    with tempfile.TemporaryDirectory(prefix="scenario-native-mixed-") as tmp, pytest.MonkeyPatch.context() as patch:
        directory = Path(tmp)
        config, _, source = mixed_seed.__wrapped__(directory, phase_roots.__wrapped__(directory, patch))
        request = ForwardPassPerfModelConfig(**config)
        original = [
            row(index, kind, 2 if kind == "prefill" else 3) for kind in ("prefill", "decode") for index in range(20)
        ]
        for method in ("gbdt", "hybrid", "selector"):
            # The SDK fixture has already built an explicit verified direct
            # profile. Exercise real manifest binding without staging an HF
            # source or rewriting that fixture's canonical identity.
            mixed = object.__new__(MixedOriginalA)
            mixed.scenario, mixed.manifests = "ac", {}
            mixed.provenance = json.loads(Path(source["manifest_path"]).read_text())["provenance"]
            mixed.prepared = SimpleNamespace(close=lambda: None)
            mixed.configure_request = lambda request: request
            started = time.perf_counter()
            model = LearnedScenarioPredictor(
                method, RustForwardPassPerfModel, request, fpm_learned, original, online=True, mixed=mixed
            )
            try:
                initial = model.predict(row(99, "mixed", 8).prediction_input())
                routes = []
                for index in range(30):
                    observation = row(100 + index, "mixed", 8)
                    prediction = model.predict(observation.prediction_input())
                    routes.append(prediction.metadata["temporary_mixed_composition"])
                    model.tune([observation])
                scope = model.diagnostics()["scopes"][0]
                final = model.predict(row(200, "mixed", 8).prediction_input())
                assert math.isclose(initial.value_ms, 4.0)
                assert math.isclose(final.value_ms, 8.0)
                assert scope["store_additions"] == (0 if method == "hybrid" else 1)
                assert ("contains_locally_mixed" in scope["stores"]) == (method != "hybrid")
                if method != "hybrid":
                    assert routes[20]["paired_comparisons"] == 0
                    assert routes[29]["paired_comparisons"] == 9
                if method == "gbdt":
                    assert routes[20]["quality_gate"] == {"alpha": 0.05, "min_comparisons": 5}
                results.append(
                    {
                        "method": method,
                        "initial_ms": initial.value_ms,
                        "final_ms": final.value_ms,
                        "store_additions": scope["store_additions"],
                        "first_ready_route": routes[20],
                        "final_route": routes[29],
                        "elapsed_s": time.perf_counter() - started,
                    }
                )
            finally:
                model.close()

    source_files = [
        "scripts/fpm_accuracy/models/learned_scenario.py",
        "scripts/fpm_accuracy/models/mixed_scenario.py",
        "scripts/fpm_accuracy/scenario_contract.py",
        "python/aisimulate/tests/unit/sdk/test_fpm_phase_readiness.py",
        "python/aisimulate/tests/unit/sdk/test_fpm_mixed_composition_integration.py",
    ]
    output = {
        "qualification": "Synthetic API and causality smoke only; no silicon accuracy or performance claim.",
        "repository_revision": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_sha256": _sha(Path(_runtime.__file__)),
        "script_sha256": _sha(Path(__file__)),
        "source_sha256": {path: _sha(repository / path) for path in source_files},
        "fixture": "SDK phase_roots + mixed_seed: prefill2/decode3/decode-zero1 ms; 40 A groups; 30 mixed C groups.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Synthetic native continuation passed for {len(results)} methods; evidence: {args.output}")


if __name__ == "__main__":
    main()
