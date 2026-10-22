# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from aisimulate_core.sdk import RustForwardPassPerfModel

r = Path("/home/yueqiz/work/fpm-model-publication-v1")
qs = json.loads((r / "fixture/queries.json").read_text())
source = (
    r / "fixture/seed" if sys.argv[1] == "write" else r / "cross-version-checkpoint"
)
config = {
    "model": "benchmark/private-model",
    "system": "benchmark_cpu",
    "backend": "vllm",
    "worker_type": "decode",
    "estimation_mode": "fpm_learned",
    "estimator_config": {
        "correction": {"enabled": False},
        "fpm_learned": {
            "checkpoint_path": str(source),
            "adaptation": "online",
            "update_mode": "leaf_only",
            "recent_window": 64,
            "update_every": 64,
        },
    },
}
m = RustForwardPassPerfModel.best_available(config)
if sys.argv[1] == "write":
    m.tune_with_fpms(qs[:3])
    m.save_online_checkpoint(r / "cross-version-checkpoint")
m.tune_with_fpms(qs[3:7])
m.flush_updates()
print(
    json.dumps(
        {
            "artifact": m.export_learned_artifact(),
            "predictions": [m.estimate_forward_pass_time_ms(q) for q in qs],
            "accepted": m.learned_online_diagnostics()["stores"]["pure_decode"][
                "accepted_observations"
            ],
        }
    )
)
