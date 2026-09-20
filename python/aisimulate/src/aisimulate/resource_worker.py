# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private subprocess entrypoint; imports simulator code only inside supervision."""

from __future__ import annotations

import json
import multiprocessing.spawn
import os
import pickle
import sys
from pathlib import Path

from .resources import ResourceLimitError
from .supervision import mark_shutdown, runtime_budget


def main() -> int:
    budget = runtime_budget()
    if budget is None:
        raise RuntimeError("resource worker requires a supervisor budget")
    if hasattr(os, "sched_getaffinity"):
        host_cpus = sorted(os.sched_getaffinity(0))
        # The host calibrator must see the serving host's CPUs, not this worker's slice.
        os.environ.setdefault("_AISIMULATE_HOST_CPUS", json.dumps(host_cpus))
        os.sched_setaffinity(0, host_cpus[: budget["cpu_limit"]])
    if hasattr(os, "nice"):
        os.nice(5)
    command = sys.argv[1]
    if command == "cli":
        from .main import main as _main

        return _main(sys.argv[2:])
    if command != "recommend":
        raise ValueError("invalid private execution command")
    root = Path(sys.argv[2])
    try:
        with (root / "input.pickle").open("rb") as source:
            preparation = pickle.load(source)
            process = multiprocessing.current_process()
            process._inheriting = True
            try:
                multiprocessing.spawn.prepare(preparation)
            finally:
                del process._inheriting
            raw_config, kwargs = pickle.load(source)
        from .config import CoreRecommendationConfig

        config = CoreRecommendationConfig.model_validate(raw_config)
        from .recommend import _run_recommendation

        result = _run_recommendation(config, **kwargs)
        (root / "result.json").write_text(result.to_json(), encoding="utf-8")
        return 0
    except Exception as exc:
        (root / "error.json").write_text(json.dumps({"message": f"{type(exc).__name__}: {exc}"}))
        return 3 if isinstance(exc, ResourceLimitError) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        mark_shutdown()
