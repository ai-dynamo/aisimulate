# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Sweeper with the transitional Dynamo replay composition."""

from __future__ import annotations

import argparse

import yaml
from aisimulate.sweeper import SmartSearchConfig, Sweeper
from dynamo.replay.simulation import DynamoReplayRunnerFactory
from pydantic import ValidationError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Sweeper sweep with Dynamo Replay"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        config = SmartSearchConfig.from_yaml(args.config)
        candidates = Sweeper(
            runner_factory=DynamoReplayRunnerFactory(),
        ).run(config)
    except OSError as exc:
        parser.error(f"could not read {args.config}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"malformed YAML in {args.config}: {exc}")
    except ValidationError as exc:
        parser.error(f"invalid config {args.config}: {exc}")
    if not candidates:
        parser.exit(1, "no feasible candidate found\n")
    for index, candidate in enumerate(candidates):
        print(
            f"{index}: score={candidate.score} used_gpus={candidate.used_gpus}"
        )


if __name__ == "__main__":
    main()
