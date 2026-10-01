# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External evidence: fit GP-UCB-PE on four synthetic points, then optimize acquisition."""

import datetime
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import jax
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_ucb_pe
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier.jax import optimizers


def main():
    started = time.monotonic()
    assert jax.config.jax_enable_x64 is True, "Frozen experiment requires JAX_ENABLE_X64=true"
    assert str(jax.numpy.asarray(1.0).dtype) == "float64"
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param("x", 0.0, 1.0)
    problem.search_space.root.add_float_param("y", 0.0, 1.0)
    problem.metric_information.append(vz.MetricInformation(name="objective", goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    designer = gp_ucb_pe.VizierGPUCBPEBandit(
        problem,
        rng=jax.random.PRNGKey(42),
        num_seed_trials=1,
        ard_random_restarts=1,
        ard_optimizer=optimizers.JaxoptScipyLbfgsB(
            options=optimizers.LbfgsBOptions(maxiter=20),
            max_duration=datetime.timedelta(seconds=30),
        ),
        acquisition_optimizer_factory=vb.VectorizedOptimizerFactory(
            strategy_factory=es.VectorizedEagleStrategyFactory(),
            max_evaluations=100,
            suggestion_batch_size=10,
        ),
    )
    completed = []
    for trial_id, (x, y) in enumerate([(0.1, 0.1), (0.2, 0.8), (0.7, 0.2), (0.9, 0.9)], 1):
        value = 1.0 - (x - 0.3) ** 2 - (y - 0.7) ** 2
        trial = vz.Trial(id=trial_id, parameters={"x": x, "y": y})
        trial.complete(vz.Measurement({"objective": value}))
        completed.append(trial)
    designer.update(vza.CompletedTrials(completed), vza.ActiveTrials([]))
    suggestion = designer.suggest(1)[0]
    prediction = suggestion.metadata.ns("google_gp_ucb_pe_bandit").ns("prediction_in_warped_y_space")
    acquisition = float(prediction["acquisition"])
    assert math.isfinite(acquisition), acquisition
    parameters = suggestion.parameters.as_dict()
    assert all(0 <= value <= 1 for value in parameters.values()), parameters
    assert all(device.platform == "cpu" for device in jax.devices())
    print(json.dumps({
        "status": "passed",
        "qualification": "external_readonly_mounted_supplemental_evidence",
        "designer": "VizierGPUCBPEBandit",
        "versions": {name: importlib.metadata.version(name) for name in ["google-vizier", "jax", "jaxlib"]},
        "seed": 42,
        "jax_enable_x64": jax.config.jax_enable_x64,
        "default_float_dtype": str(jax.numpy.asarray(1.0).dtype),
        "completed_observations": len(completed),
        "num_seed_trials": 1,
        "actual_gp_fit_and_acquisition": True,
        "ard_random_restarts": 1,
        "ard_max_iterations": 20,
        "acquisition_max_evaluations": 100,
        "parameters": parameters,
        "acquisition": acquisition,
        "use_ucb": prediction["use_ucb"],
        "seconds": round(time.monotonic() - started, 3),
        "dynamo_replay_calls": 0,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary": "Tiny synthetic 2D functional check with reduced optimization budgets; no search-quality claim",
    }, indent=2))


if __name__ == "__main__":
    main()
