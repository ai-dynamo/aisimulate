# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the self-service support workflow."""


class SupportWorkflowError(ValueError):
    """The support request cannot advance without an explicit user action."""
