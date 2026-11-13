# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import compatibility for the former application and estimator namespaces."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import sys
from types import ModuleType

_ALIASES = {"aisimulate": "aisim", "aisimulate_core": "aisim_core"}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, canonical: str) -> None:
        self.fullname = fullname
        self.canonical = canonical

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        # Import once under its canonical name; never execute implementation
        # twice under a legacy name (which would split types and global state).
        sys.modules[self.fullname] = importlib.import_module(self.canonical)

    def get_code(self, fullname: str):
        # runpy uses get_code for `python -m aisimulate[.submodule]`.
        spec = importlib.util.find_spec(self.canonical)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "get_code"):
            raise ImportError(f"module {fullname!r} cannot be executed")
        return spec.loader.get_code(self.canonical)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        prefix, separator, suffix = fullname.partition(".")
        if not separator or prefix not in _ALIASES:
            return None
        canonical = f"{_ALIASES[prefix]}.{suffix}"
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(fullname, canonical), is_package=spec.submodule_search_locations is not None
        )


def alias_package(legacy: str) -> ModuleType:
    """Install on-demand aliases without eagerly importing optional dependencies."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
    canonical = importlib.import_module(_ALIASES[legacy])
    sys.modules[legacy] = canonical
    return canonical


def plugin_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Discover canonical and legacy plugin groups without importing plugins."""
    installed = importlib.metadata.entry_points()
    legacy = group.replace("aisim.", "aisimulate.", 1)
    canonical = list(installed.select(group=group))
    # Only collapse the same provider registered in both namespaces. Duplicate
    # providers inside one namespace still reach the resolver's conflict check.
    identities = {(entry.name, entry.value) for entry in canonical}
    return canonical + [
        entry for entry in installed.select(group=legacy) if (entry.name, entry.value) not in identities
    ]
