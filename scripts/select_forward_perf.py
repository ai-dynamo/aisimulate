# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the advisory benchmark from the complete, current PR change set."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

# Keep these rules aligned with the measured Rust and Python call path.
PATH_PATTERNS = (
    ".github/workflows/performance.yml",
    "scripts/select_forward_perf.py",
    "Cargo.toml",
    "Cargo.lock",
    "crates/core/Cargo.toml",
    "crates/core/src/lib.rs",
    "crates/core/src/python.rs",
    "crates/core/src/engine/**",
    "crates/core/src/perfmodel/**",
    "python/aisimulate/pyproject.toml",
    "python/aisimulate/src/aisimulate/__init__.py",
    "python/aisimulate/src/aisimulate/sdk/**",
    "python/aisimulate/src/aisimulate_core/*.py",
    "python/aisimulate/src/aisimulate_core/sdk/**",
    "python/aisimulate/src/aisimulate_core/model_configs/Qwen--Qwen3-32B_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/Qwen--Qwen3-235B-A22B_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V3.2_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/deepseek-ai--DeepSeek-V4-Flash_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/Qwen--Qwen3.5-397B-A17B_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/openai--gpt-oss-120b_config.json",
    "python/aisimulate/src/aisimulate_core/model_configs/nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-FP8_config.json",
    "python/aisimulate/src/aisimulate_core/systems/*.yaml",
    "python/aisimulate/src/aisimulate_core/systems/support_matrix/**",
    "python/aisimulate/src/aisimulate_core/systems/fpe_support_matrix/**",
    "python/aisimulate/src/aisimulate_core/systems/data/b200_sxm/**",
    "python/aisimulate/src/aisimulate_core/systems/data/h100_sxm/**",
    "python/aisimulate/tools/forward_perf_gate/*.py",
    "python/aisimulate/tools/prediction_regression_gate/grid.py",
)


def matches_path(path: str) -> bool:
    """Match exact files, directory subtrees, and direct-child filename globs."""
    parent, _, name = path.rpartition("/")
    for pattern in PATH_PATTERNS:
        directory, _, filename = pattern.rpartition("/")
        if filename == "**":
            if path.startswith(directory + "/"):
                return True
        elif parent == directory and fnmatchcase(name, filename):
            return True
    return False


def github_api(endpoint: str) -> list:
    """Read all pages; an API error must not become a successful skip."""
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def _revision(pull: dict) -> tuple[str, str, str]:
    return pull["head"]["sha"], pull["base"]["sha"], pull["base"]["ref"]


def select_comparison(
    repository: str,
    event: str,
    ref: str,
    sha: str,
    pr_number: str = "",
    *,
    api=github_api,
) -> dict:
    if event == "push":
        match = re.fullmatch(r"refs/heads/pull-request/([1-9][0-9]*)", ref)
        if not match:
            raise ValueError(f"Invalid trusted PR ref: {ref}")
        pr_number = match[1]
    elif event != "workflow_dispatch" or not re.fullmatch(r"[1-9][0-9]*", pr_number):
        raise ValueError("Expected a trusted PR push or manual dispatch with a positive pr_number")

    endpoint = f"repos/{repository}/pulls/{pr_number}"
    pull = api(endpoint)[0]
    if event == "workflow_dispatch":
        sha = api(f"repos/{repository}/git/ref/heads/pull-request/{pr_number}")[0]["object"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or sha != pull["head"]["sha"]:
        raise ValueError(f"Trusted copy {sha} does not match PR #{pr_number} head {pull['head']['sha']}")

    run = True
    if event == "workflow_dispatch":
        reason = "Manual dispatch requests a comparison."
    elif pull["changed_files"] > 3000:
        reason = "PR exceeds the file API limit; run conservatively."
    else:
        pages = api(f"{endpoint}/files?per_page=100")
        if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
            raise ValueError("Invalid PR file-list response")
        files = [item for page in pages for item in page]
        if not files or len(files) != pull["changed_files"]:
            reason = "PR file list is empty or incomplete; run conservatively."
        else:
            run = any(
                matches_path(path) for item in files for path in (item["filename"], item.get("previous_filename", ""))
            )
            reason = "PR changes affect forward prediction." if run else "No PR files affect forward prediction."

    if _revision(api(endpoint)[0]) != _revision(pull):
        raise ValueError("PR head or base changed while selecting performance CI")
    return {
        "number": pr_number,
        "head_sha": sha,
        "base_ref": pull["base"]["ref"],
        "run_comparison": str(run).lower(),
        "reason": reason,
    }


def main() -> int:
    try:
        selection = select_comparison(
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_EVENT_NAME"],
            os.environ["GITHUB_REF"],
            os.environ["GITHUB_SHA"],
            os.environ.get("PR_NUMBER", ""),
        )
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Cannot select forward performance CI: {error}", file=sys.stderr)
        return 1
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in selection.items():
            output.write(f"{key}={value}\n")
    decision = "RUN" if selection["run_comparison"] == "true" else "SKIPPED"
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(
            "## Forward Prediction Performance selection\n\n"
            f"**{decision}** — {selection['reason']}\n\n"
            f"PR #{selection['number']}, head `{selection['head_sha']}`.\n"
        )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
