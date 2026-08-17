# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AISimulate command entry point with full AIC CLI compatibility."""

from aiconfigurator.main import main as _aiconfigurator_main


def main(argv: list[str] | None = None) -> None:
    """Run the proven AIC command surface under the AISimulate executable.

    The new ``predict``/``recommend`` command design is not allowed to replace
    this delegation until the AIC-to-AISimulate parity matrix has no
    unapproved gaps.
    """

    _aiconfigurator_main(argv)
