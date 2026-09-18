#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and verify active main-branch CI rules; never change repository settings."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parents[1] / ".github/required-main-checks.json"


def github_api(endpoint: str) -> list:
    """Preserve page boundaries and fail on API, authentication, or decoding errors."""
    result = subprocess.run(
        ["gh", "api", "--method", "GET", "--paginate", "--slurp", endpoint],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    pages = json.loads(result.stdout)
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Expected response pages from {endpoint}")
    return pages


def expected_checks(payload: dict) -> list[dict]:
    if (
        payload["target"] != "branch"
        or payload["enforcement"] != "active"
        or payload["bypass_actors"] != []
        or payload["conditions"] != {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}}
        or len(payload["rules"]) != 1
        or payload["rules"][0]["type"] != "required_status_checks"
    ):
        raise ValueError("Expected the additive, active default-branch CI payload without bypass actors")
    parameters = payload["rules"][0]["parameters"]
    checks = parameters["required_status_checks"]
    if (
        parameters["strict_required_status_checks_policy"] is not True
        or parameters["do_not_enforce_on_create"] is not False
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check["context"], str)
            or not check["context"]
            or type(check["integration_id"]) is not int
            or check["integration_id"] <= 0
            for check in checks
        )
        or len({check["context"] for check in checks}) != len(checks)
    ):
        raise ValueError("Expected unique app-bound checks with strict branch currency")
    return checks


def verify_rules(rules: list[dict], checks: list[dict]) -> list[str]:
    """Report missing active requirements, including independent human protections."""
    errors = []
    for expected in checks:
        if not any(
            rule["type"] == "required_status_checks"
            and rule["parameters"]["strict_required_status_checks_policy"] is True
            and rule["parameters"]["do_not_enforce_on_create"] is False
            and any(
                check["context"] == expected["context"] and check.get("integration_id") == expected["integration_id"]
                for check in rule["parameters"]["required_status_checks"]
            )
            for rule in rules
        ):
            errors.append(
                f"Missing strict required check {expected['context']!r} from app {expected['integration_id']}"
            )

    review_rules = [rule["parameters"] for rule in rules if rule["type"] == "pull_request"]
    if not any(
        type(rule["required_approving_review_count"]) is int and rule["required_approving_review_count"] >= 1
        for rule in review_rules
    ):
        errors.append("Missing required human approval")
    for key in ("require_code_owner_review", "required_review_thread_resolution"):
        if not any(rule[key] is True for rule in review_rules):
            errors.append(f"Missing {key}")
    for kind in ("deletion", "non_fast_forward"):
        if not any(rule["type"] == kind for rule in rules):
            errors.append(f"Missing existing {kind} protection")
    return errors


def inspect_repository(repository: str, *, api=github_api) -> dict:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Expected repository owner/name")
    report = {
        "repository": repository,
        "branch": "main",
        "observed_at": datetime.now(UTC).isoformat(),
        "configuration_verified": False,
        "errors": [],
        "limitations": "Configuration snapshot only; controlled-PR evidence and human approval remain separate.",
    }
    try:
        (metadata,) = api(f"repos/{repository}")
        if metadata["default_branch"] != "main":
            raise ValueError("The payload targets the default branch, but main is not the default branch")
        report["expected_checks"] = expected_checks(json.loads(PAYLOAD.read_text(encoding="utf-8")))
        pages = api(f"repos/{repository}/rules/branches/main?per_page=100")
        if not all(isinstance(page, list) and all(isinstance(rule, dict) for rule in page) for page in pages):
            raise ValueError("Expected arrays of effective branch rules")
        rules = [rule for page in pages for rule in page]
        report["effective_rules"] = rules
        report["errors"].extend(verify_rules(rules, report["expected_checks"]))
        # Effective rules exclude disabled/evaluate rulesets. Read their sources
        # as well: a configured gate with bypass actors is not the intended gate.
        report["ci_rulesets"] = []
        for ruleset_id in sorted({rule["ruleset_id"] for rule in rules if rule["type"] == "required_status_checks"}):
            if type(ruleset_id) is not int or ruleset_id <= 0:
                raise ValueError("Invalid ruleset ID")
            (detail,) = api(f"repos/{repository}/rulesets/{ruleset_id}")
            report["ci_rulesets"].append(detail)
            if detail["id"] != ruleset_id or detail["enforcement"] != "active":
                report["errors"].append(f"CI ruleset {ruleset_id} changed during inspection")
            if detail.get("bypass_actors") != []:
                report["errors"].append(
                    f"CI ruleset {ruleset_id} has bypass actors or hides them; verify with administrator read access"
                )
        # Refuse a success assembled from different settings during a rollout.
        if pages != api(f"repos/{repository}/rules/branches/main?per_page=100"):
            report["errors"].append("Effective rules changed during inspection; rerun the verifier")
        # Bypass actors are only exposed in source details, not effective rules.
        # Their second read must follow the effective-rule recheck as well.
        for detail in report["ci_rulesets"]:
            (current,) = api(f"repos/{repository}/rulesets/{detail['id']}")
            if current != detail:
                report["errors"].append(f"CI ruleset {detail['id']} changed during inspection; rerun the verifier")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        report["errors"].append(f"Cannot verify configuration: {error}")
    report["configuration_verified"] = not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="ai-dynamo/aisimulate")
    parser.add_argument("--output", type=Path, help="Also save the JSON evidence snapshot to this file")
    args = parser.parse_args(argv)
    try:
        report = inspect_repository(args.repository)
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if report["configuration_verified"] else 1
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cannot verify configuration: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
