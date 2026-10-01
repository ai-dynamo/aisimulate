# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal FPM compilation context; Rust owns the interpolation schema."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class FpmCompileConfig:
    """Carry a validated immutable options payload and its recorded backend label."""

    _options_json: str
    attention_backend: str | None

    def __init__(
        self,
        options: dict | None = None,
        *,
        attention_backend: str | None = None,
        fmha_quant_mode: str | None = None,
        comm_quant_mode: str | None = None,
        has_profile: bool = False,
    ) -> None:
        import aisimulate_core

        normalized = aisimulate_core.RustForwardPassPerfModel._normalize_fpm_options(
            json.dumps({} if options is None else options), fmha_quant_mode, comm_quant_mode, has_profile
        )
        object.__setattr__(self, "_options_json", normalized)
        object.__setattr__(self, "attention_backend", attention_backend)

    @property
    def options(self) -> dict:
        """Return a copy of the complete Rust-resolved options, including defaults."""
        return json.loads(self._options_json)

    def cache_identity(self) -> dict:
        return {"options": self.options, "attention_backend": self.attention_backend}


def resolve_fpm_config(model_config) -> FpmCompileConfig:
    """Resolve legacy direct model construction without Python-owned defaults."""
    if model_config.fpm_config is None:
        model_config.fpm_config = FpmCompileConfig()
    return model_config.fpm_config
