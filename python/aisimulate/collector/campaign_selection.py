# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit campaign scope selection; independent of observed failures."""


def select_campaign_cases(cases, include, exclude, *, op, logger):
    """Keep native case objects/order and report scope exclusions before sampling."""
    for fragments in (include, exclude):
        if fragments and any(not isinstance(value, str) or not value for value in fragments):
            raise ValueError("Campaign filters must be nonempty literal strings")
    if not include and not exclude:
        return cases
    planned = len(cases)
    selected = []
    outside_include = excluded = 0
    for case in cases:
        rendered = str(case)
        if include and not any(fragment in rendered for fragment in include):
            outside_include += 1
        elif exclude and any(fragment in rendered for fragment in exclude):
            excluded += 1
        else:
            selected.append(case)
    logger.info(
        f"{op}: campaign scope selected={len(selected)} generated={planned} "
        f"outside_include={outside_include} explicitly_excluded={excluded}; "
        "excluded cases are not collection successes or failures"
    )
    return selected
