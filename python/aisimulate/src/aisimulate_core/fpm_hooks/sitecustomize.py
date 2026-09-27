# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-installer: put this directory on PYTHONPATH and every interpreter (including
spawned scheduler subprocesses) registers the FPM per-request hooks at startup."""

try:
    import aisimulate_core.fpm_hooks as _hooks

    _hooks.install()
except Exception:
    import logging

    logging.getLogger("aisimulate.fpm_hooks").exception("fpm_hooks: sitecustomize install failed")
