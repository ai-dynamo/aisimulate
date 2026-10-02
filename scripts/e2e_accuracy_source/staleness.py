# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three-layer freshness pipeline.

Per docs/silicon_to_cli_estimate_mapping.md → "Data freshness and dedupe":

  1. Row-level dedupe        — one row per (config_id, isl, osl, conc),
                               error rows dropped, latest `date` chosen.
  2. Engine-version coherence — if a config_id mixes images, lock to the
                               most-recent image and drop earlier rows.
  3. Config-level staleness  — drop a config_id if its newest benchmark
                               is older than `max_age_days` from
                               `dump_max_date`.

`dump_max_date` defaults to the max date observed across the input rows
when not supplied; pass it explicitly when running against a release
where that detection might be unreliable (e.g. in tests).
"""

from __future__ import annotations

import collections
import datetime as _dt
from collections.abc import Iterable

from e2e_accuracy_source.schema import DropRecord, SiliconRow


def _parse_date(s: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


# 1. Row dedupe ------------------------------------------------------------


def dedupe_rows(rows: Iterable[SiliconRow]) -> tuple[list[SiliconRow], list[DropRecord]]:
    """Within each (config_id, isl, osl, conc) group: drop error rows,
    keep the latest by `date`. Other rows in the group are dropped as
    `dedupe:superseded`."""
    by_key: dict[tuple[int, int, int, int], list[SiliconRow]] = collections.defaultdict(list)
    drops: list[DropRecord] = []

    for r in rows:
        if not r.metrics:
            drops.append(
                DropRecord(
                    stage="dedupe",
                    reason="error or empty metrics",
                    config_id=r.config_id,
                    isl=r.isl,
                    osl=r.osl,
                    conc=r.conc,
                )
            )
            continue
        by_key[(r.config_id, r.isl, r.osl, r.conc)].append(r)

    kept: list[SiliconRow] = []
    for group in by_key.values():
        group_sorted = sorted(group, key=lambda x: x.date, reverse=True)
        kept.append(group_sorted[0])
        for older in group_sorted[1:]:
            drops.append(
                DropRecord(
                    stage="dedupe",
                    reason="superseded by later run in same (config,isl,osl,conc)",
                    config_id=older.config_id,
                    isl=older.isl,
                    osl=older.osl,
                    conc=older.conc,
                )
            )
    return kept, drops


# 2. Image coherence ------------------------------------------------------


def apply_image_coherence(rows: Iterable[SiliconRow]) -> tuple[list[SiliconRow], list[DropRecord]]:
    """Within each config_id, if multiple `image` values appear, lock to
    the most-recent image and drop rows from earlier images.

    Operates *after* row dedupe, so each (config_id, isl, osl, conc) is
    represented at most once.
    """
    rows = list(rows)
    by_cfg: dict[int, list[SiliconRow]] = collections.defaultdict(list)
    for r in rows:
        by_cfg[r.config_id].append(r)

    kept: list[SiliconRow] = []
    drops: list[DropRecord] = []
    for cfg_rows in by_cfg.values():
        latest_image = _select_latest_image(cfg_rows)
        for r in cfg_rows:
            if r.image == latest_image or r.image is None or latest_image is None:
                kept.append(r)
            else:
                drops.append(
                    DropRecord(
                        stage="image-coherence",
                        reason=f"image {r.image!r} superseded by {latest_image!r}",
                        config_id=r.config_id,
                        isl=r.isl,
                        osl=r.osl,
                        conc=r.conc,
                    )
                )
    return kept, drops


def _select_latest_image(cfg_rows: list[SiliconRow]) -> str | None:
    """Image whose row has the latest `date` within the config; ties
    broken by frequency."""
    candidates = [(r.date, r.image) for r in cfg_rows if r.image is not None]
    if not candidates:
        return None
    latest_date = max(date for date, _img in candidates)
    images_at_latest = [img for date, img in candidates if date == latest_date]
    counts = collections.Counter(images_at_latest)
    return counts.most_common(1)[0][0]


# 3. Config staleness ------------------------------------------------------


def apply_config_staleness(
    rows: Iterable[SiliconRow],
    max_age_days: int,
    dump_max_date: str | None = None,
) -> tuple[list[SiliconRow], list[DropRecord]]:
    """Drop a config_id whose newest benchmark date is older than
    `max_age_days` from `dump_max_date`. Operates after dedupe + coherence.
    """
    rows = list(rows)
    if not rows:
        return [], []

    if dump_max_date is None:
        dump_max_date = max(r.date for r in rows)
    ref = _parse_date(dump_max_date)

    latest_per_cfg: dict[int, str] = {}
    for r in rows:
        prev = latest_per_cfg.get(r.config_id)
        if prev is None or r.date > prev:
            latest_per_cfg[r.config_id] = r.date

    kept: list[SiliconRow] = []
    drops: list[DropRecord] = []
    cfg_age_cache: dict[int, int] = {}
    for r in rows:
        latest = latest_per_cfg[r.config_id]
        age = cfg_age_cache.get(r.config_id)
        if age is None:
            age = (ref - _parse_date(latest)).days
            cfg_age_cache[r.config_id] = age
        if age > max_age_days:
            drops.append(
                DropRecord(
                    stage="staleness",
                    reason=f"config newest benchmark {latest} is {age}d > {max_age_days}d threshold",
                    config_id=r.config_id,
                    isl=r.isl,
                    osl=r.osl,
                    conc=r.conc,
                )
            )
        else:
            kept.append(r)
    return kept, drops
