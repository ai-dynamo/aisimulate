# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the pinned client's complete CLI envelope without running a benchmark."""
import json
from pathlib import Path
import sys

import yaml
from aiperf.cli_commands.profile import app
from aiperf.common.enums import ServerMetricsDiscoveryMode
from aiperf.config.config import AIPerfConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader import build_benchmark_plan

command = json.loads(Path(sys.argv[1]).read_text())
_, bound, _ = app.parse_args(command[2:])
config = resolve_config(bound.arguments['cli_config'], None)
config.benchmark.server_metrics.discovery.mode = ServerMetricsDiscoveryMode.DISABLED
payload = config.model_dump(mode='json')
build_benchmark_plan(AIPerfConfig.model_validate(payload))
Path(sys.argv[2]).write_text(yaml.safe_dump(payload, sort_keys=False))
print('Validated complete AgentX configuration envelope', flush=True)
