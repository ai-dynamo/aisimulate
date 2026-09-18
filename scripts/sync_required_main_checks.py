#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the CI ruleset with its source file; apply only with explicit admin authorization."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parents[1] / ".github/required-main-checks.json"
FIELDS = {"name", "target", "enforcement", "bypass_actors", "conditions", "rules"}
CONDITIONS = {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}}


def github_api(endpoint: str, *, payload: dict | None = None) -> dict:
    command = ["gh", "api", "--hostname", "github.com", "--method", "GET" if payload is None else "PUT", endpoint]
    if payload is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command,
        input=None if payload is None else json.dumps(payload),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    response = json.loads(result.stdout)
    if not isinstance(response, dict):
        raise ValueError(f"Expected an object from {endpoint}")
    return response


def validate_payload(payload: dict) -> None:
    """Keep this command limited to the additive, strict CI gate."""
    if (
        set(payload) != FIELDS
        or payload["name"] != "AISimulate required CI validation"
        or payload["target"] != "branch"
        or payload["enforcement"] != "active"
        or payload["bypass_actors"] != []
        or payload["conditions"] != CONDITIONS
        or len(payload["rules"]) != 1
        or set(payload["rules"][0]) != {"type", "parameters"}
        or payload["rules"][0]["type"] != "required_status_checks"
    ):
        raise ValueError("Expected the active, additive default-branch CI payload without bypasses")
    parameters = payload["rules"][0]["parameters"]
    if (
        set(parameters)
        != {"strict_required_status_checks_policy", "do_not_enforce_on_create", "required_status_checks"}
        or parameters["strict_required_status_checks_policy"] is not True
        or parameters["do_not_enforce_on_create"] is not False
    ):
        raise ValueError("The CI payload must require strict branch currency and enforce on creation")
    checks = parameters["required_status_checks"]
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            set(check) != {"context", "integration_id"}
            or not isinstance(check["context"], str)
            or not check["context"].strip()
            or type(check["integration_id"]) is not int
            or check["integration_id"] != 15368
            for check in checks
        )
        or len({check["context"] for check in checks}) != len(checks)
    ):
        raise ValueError("Expected unique, nonempty checks bound to GitHub Actions app 15368")


def normalized(payload: dict) -> dict:
    result = {field: payload[field] for field in sorted(FIELDS)}
    result = json.loads(json.dumps(result))
    result["rules"][0]["parameters"]["required_status_checks"].sort(key=lambda check: check["context"])
    return result


def inspect_target(current: dict, repository: str, ruleset_id: int, desired: dict) -> None:
    if (
        current["id"] != ruleset_id
        or current["source_type"] != "Repository"
        or current["source"].lower() != repository.lower()
        or current["name"] != desired["name"]
        or current["target"] != "branch"
        or current["conditions"] != CONDITIONS
        or len(current["rules"]) != 1
        or set(current["rules"][0]) != {"type", "parameters"}
        or current["rules"][0]["type"] != "required_status_checks"
        or set(current["rules"][0]["parameters"]) != set(desired["rules"][0]["parameters"])
    ):
        raise ValueError("Refusing to overwrite a different, inherited, or mixed-purpose ruleset")
    if "bypass_actors" not in current:
        raise ValueError(
            "GitHub hides bypass settings from this account. Rerun with Admin or edit-repository-rules access; "
            "visible settings alone cannot establish a complete match."
        )
    if current["bypass_actors"] != []:
        raise ValueError("The live ruleset has bypass actors; an administrator must reconcile them explicitly")


def synchronize(repository: str, ruleset_id: int, desired: dict, *, apply=False, backup=None, api=github_api) -> int:
    validate_payload(desired)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or ruleset_id <= 0:
        raise ValueError("Expected repository owner/name and a positive ruleset ID")
    if api(f"repos/{repository}")["default_branch"] != "main":
        raise ValueError("Expected main to remain the default branch")
    endpoint = f"repos/{repository}/rulesets/{ruleset_id}"
    current = api(endpoint)
    inspect_target(current, repository, ruleset_id, desired)
    before, after = normalized(current), normalized(desired)
    if before == after:
        print(f"Ruleset {ruleset_id} matches {PAYLOAD.name}; no update needed.")
        return 0
    print(
        "".join(
            difflib.unified_diff(
                (json.dumps(before, indent=2) + "\n").splitlines(keepends=True),
                (json.dumps(after, indent=2) + "\n").splitlines(keepends=True),
                fromfile=f"github.com/{repository}/rules/{ruleset_id}",
                tofile=str(PAYLOAD),
            )
        )
    )
    if not apply:
        print("Drift detected. Review the diff; an authorized administrator can rerun with --apply --backup PATH.")
        return 1
    if backup is None:
        raise ValueError("--apply requires --backup PATH to save the current complete ruleset")
    latest = api(endpoint)
    inspect_target(latest, repository, ruleset_id, desired)
    if normalized(latest) != before:
        raise ValueError("Ruleset changed during inspection; nothing applied. Inspect again before retrying.")
    if api(f"repos/{repository}")["default_branch"] != "main":
        raise ValueError("Default branch changed during inspection; nothing applied")
    # Create the evidence only after every pre-write check. Exclusive creation
    # prevents overwriting an administrator's earlier snapshot.
    with Path(backup).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(latest, indent=2) + "\n")
    try:
        api(endpoint, payload=desired)
        updated = api(endpoint)
        inspect_target(updated, repository, ruleset_id, desired)
        if normalized(updated) != after:
            raise ValueError("GitHub's read-back does not match the requested configuration")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Update or read-back failed; settings may already have changed. Inspect GitHub before retrying. {error}"
        ) from error
    print(f"Applied and verified ruleset {ruleset_id}. Previous definition saved to {backup}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="ai-dynamo/aisimulate")
    parser.add_argument(
        "--ruleset-id", type=int, required=True, help="Existing repository CI ruleset ID; never creates one"
    )
    parser.add_argument("--apply", action="store_true", help="Apply using the current gh account's ruleset permissions")
    parser.add_argument(
        "--backup", type=Path, help="New file for the complete pre-update definition (required to apply)"
    )
    args = parser.parse_args(argv)
    try:
        return synchronize(
            args.repository,
            args.ruleset_id,
            json.loads(PAYLOAD.read_text(encoding="utf-8")),
            apply=args.apply,
            backup=args.backup,
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Cannot synchronize CI rules: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
