# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the audited bundle; other pins use an explicit campaign-local manifest."""

from pathlib import Path

from .runtime_instrumentation import InstrumentationBundle, load_instrumentation


def bundled_instrumentation(version: str) -> InstrumentationBundle | None:
    if version != "0.27.0":
        return None
    return load_instrumentation(Path(__file__).parent / "runtime" / "vllm-0.27.0.json", version)
