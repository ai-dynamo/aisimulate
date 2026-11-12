# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public dataset repository names and the alias rule for the retired FPM repository (AISIM-2339).

``ai-dynamo/aisim-data`` is the public unified dataset. ``nvidia/aisimulate-fpm-dataset`` held the
FPM files before the move and is being retired; the move was byte-identical, so the same bytes live
at ``data/fpm/<rest>`` where the retired layout used ``data/<rest>``.

One rule, two cases, no silent fallback:

* **Historical pin** (retired repository + full commit SHA): resolved verbatim. A commit SHA names
  immutable content in the repository that holds it; rewriting it to another repository would turn
  a reproducible pin into a guess. The retired repository is retired, not deleted.
* **Symbolic reference** (retired repository + branch, tag or empty): a name, not content, so it
  follows the successor. The repository becomes ``ai-dynamo/aisim-data`` and ``data/<rest>`` becomes
  ``data/fpm/<rest>``. The result says ``via="successor-alias"`` so callers can report it.

Nothing here calls the network or writes to Hugging Face.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aisim_core.data.fpm_layout import physical_path

PUBLIC_REPO = "ai-dynamo/aisim-data"
RETIRED_FPM_REPO = "nvidia/aisimulate-fpm-dataset"
#: retired repository -> successor. The only entries are repositories whose files moved byte-identically.
SUCCESSORS: dict[str, str] = {RETIRED_FPM_REPO: PUBLIC_REPO}

_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ResolvedRef:
    repo: str
    revision: str
    #: ``verbatim`` (nothing rewritten) or ``successor-alias`` (retired name followed to its successor).
    via: str

    def path(self, path: str) -> str:
        """Repository path of ``path`` (as spelled for the *requested* repository) in the resolved one."""
        if self.via == "successor-alias":
            return physical_path(path)
        return path


def is_retired(repo: str) -> bool:
    return repo in SUCCESSORS


def resolve_ref(repo: str, revision: str | None) -> ResolvedRef:
    """Apply the alias rule to one ``repo`` + ``revision`` reference."""
    revision = revision or ""
    if repo in SUCCESSORS and not _SHA.fullmatch(revision):
        return ResolvedRef(SUCCESSORS[repo], revision or "main", "successor-alias")
    return ResolvedRef(repo, revision, "verbatim")


def retired_repo_hint(repo: str) -> str:
    """Actionable text for a retired repository that no longer serves a pinned commit."""
    successor = SUCCESSORS.get(repo)
    if successor is None:
        return ""
    return (
        f"{repo} is retired; its files are byte-identical in {successor} under data/fpm/ (the retired "
        f"data/<rest> is data/fpm/<rest>). Re-record the pin against a {successor} commit; the old pin is "
        f"never rewritten automatically."
    )
