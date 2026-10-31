# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A standards exception scoped to ``versions: all`` is bounded only by ``verified_on``
(layer_permissions.md "Governed exceptions" item 3): it must name the collector pin of every
table it covers, so a pin bump that nobody re-verified fails here instead of being covered silently."""

from pathlib import Path

import pytest
import yaml
from collector.framework_manifest import MANIFEST_PATH, _registry_entries, _resolve_from, load_manifest
from collector.op_catalog import CATALOG_PATH, load_family_map

pytestmark = pytest.mark.unit
EXCEPTIONS = Path(__file__).resolve().parents[3] / "collector" / "evidence_exceptions.yaml"
ENTRIES = yaml.safe_load(EXCEPTIONS.read_text())["standards_exceptions"]


def _collector_pins(backend: str, table: str) -> set[str]:
    manifest, family_map = load_manifest(MANIFEST_PATH), load_family_map(CATALOG_PATH)
    return {
        _resolve_from(manifest, family_map, backend, entry).version
        for entry in _registry_entries(backend)
        if table in {Path(str(name)).stem for name in entry.perf_filenames}
    }


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["rule"].split("—")[-1].strip() for e in ENTRIES])
def test_verified_on_is_the_collector_pin_of_every_table_in_scope(entry):
    scope = entry["scope"]
    for table in scope["tables"]:
        pins = _collector_pins(scope["backend"], table)
        assert pins, f"{scope['backend']} has no collector for {table}"
        assert pins == {entry["verified_on"]}, (
            f"{table}: collected on {sorted(pins)}, verified on {entry['verified_on']}; re-verify the "
            "deviation on the new pin and move verified_on, or delete the entry"
        )
