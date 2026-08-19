# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Sweeper with the transitional Dynamo replay composition."""

from __future__ import annotations

import argparse

from dynamo.replay.simulation import DynamoReplayRunnerFactory
from pydantic import ValidationError

from aisimulate.sweeper import Sweeper
from aisimulate.sweeper.__main__ import (
    load_config_or_parser_error,
    print_candidates_or_exit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Sweeper sweep with Dynamo Replay"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config_or_parser_error(parser, args.config)
    try:
        result = Sweeper(
            runner_factory=DynamoReplayRunnerFactory(),
        ).run_result(config)
    except ValidationError as exc:
        parser.error(f"invalid adapter search space in {args.config}: {exc}")
    print_candidates_or_exit(config, result)


if __name__ == "__main__":
    main()
