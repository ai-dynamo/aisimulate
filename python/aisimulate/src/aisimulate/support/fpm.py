# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter from the support request to the packaged whole-model FPM runner."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from .errors import SupportWorkflowError
from .plan import fpm_cli_args, prepare_systems_overlay
from .schema import SupportRequest


def run_fpm(
    request: SupportRequest,
    *,
    execute: bool,
    smoke: bool,
    limit: int | None,
    resume: bool,
    checkpoint_dir: str | None,
    output_dir: str,
) -> int:
    if request.identity.framework != request.fpm.backend:
        raise SupportWorkflowError(
            f"whole-model FPM MVP supports framework={request.fpm.backend!r}; "
            f"the requested cell uses {request.identity.framework!r}"
        )
    root = Path(output_dir)
    systems_root = prepare_systems_overlay(request, root)
    command = fpm_cli_args(
        request,
        plan_only=not execute,
        smoke=smoke,
        limit=limit,
        resume=resume,
        checkpoint_dir=checkpoint_dir or str(root / "fpm-checkpoint"),
        artifact_root=str(root / "fpm-artifacts"),
        database_root=str(systems_root / "data"),
    )
    if not execute:
        sys.stdout.write(shlex.join(command) + "\n")
        return 0

    from collector.fpm_forward.cli import main as fpm_main

    return fpm_main(command[3:])
