# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-preserving alias for the migrated engine module."""

import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("aiconfigurator_core.sdk.engine")
