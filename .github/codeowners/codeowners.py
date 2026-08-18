# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate AISimulate's deliberately simple CODEOWNERS policy."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

OWNER_RE = re.compile(r"^@[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Rule:
    section: str
    pattern: str
    owners: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    teams: dict[str, str]
    catch_all: str
    rules: tuple[Rule, ...]


def _fail(message: str) -> None:
    raise SystemExit(message)


def load_policy(path: Path) -> Policy:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        _fail("areas.yaml must contain a mapping")

    teams = raw.get("teams")
    if not isinstance(teams, dict) or not teams:
        _fail("areas.yaml must declare a non-empty teams mapping")
    for label, owner in teams.items():
        if not isinstance(label, str) or not label.strip():
            _fail("team labels must be non-empty strings")
        if not isinstance(owner, str) or OWNER_RE.fullmatch(owner) is None:
            _fail(f"team {label!r} must resolve to one @org/team owner")

    meta = raw.get("meta")
    catch_all = meta.get("catch_all") if isinstance(meta, dict) else None
    if catch_all not in teams:
        _fail("meta.catch_all must name a declared team label")

    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        _fail("areas.yaml must declare a non-empty rules list")

    rules: list[Rule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, dict):
            _fail(f"rule {index} must be a mapping")
        section = raw_rule.get("section")
        pattern = raw_rule.get("pattern")
        labels = raw_rule.get("owners")
        if not isinstance(section, str) or not section.strip():
            _fail(f"rule {index} must declare a section")
        if not isinstance(pattern, str) or not pattern.strip():
            _fail(f"rule {index} must declare a pattern")
        if pattern != "*" and (
            not pattern.startswith("/")
            or any(character in pattern for character in " \\#!?[]")
            or "*" in pattern
        ):
            _fail(
                f"unsupported pattern {pattern!r}; use '*', an anchored file, "
                "or an anchored directory"
            )
        if not isinstance(labels, list) or not labels:
            _fail(f"rule {index} must declare at least one owner")
        unknown = [label for label in labels if label not in teams]
        if unknown:
            _fail(f"rule {index} references unknown team labels: {unknown}")
        rules.append(
            Rule(
                section=section,
                pattern=pattern,
                owners=tuple(teams[label] for label in labels),
            )
        )

    if rules[0].pattern != "*" or rules[0].owners != (teams[catch_all],):
        _fail("the first rule must be '*' owned only by meta.catch_all")
    return Policy(teams=dict(teams), catch_all=catch_all, rules=tuple(rules))


def matches(pattern: str, path: str) -> bool:
    if pattern == "*":
        return True
    anchored = pattern[1:]
    if anchored.endswith("/"):
        return path.startswith(anchored)
    return path == anchored


def owners_for(policy: Policy, path: str) -> tuple[str, ...]:
    owners: tuple[str, ...] = ()
    for rule in policy.rules:
        if matches(rule.pattern, path):
            owners = rule.owners
    return owners


def is_explicitly_owned(policy: Policy, path: str) -> bool:
    return any(
        rule.pattern != "*" and matches(rule.pattern, path) for rule in policy.rules
    )


def render(policy: Policy) -> str:
    lines = [
        "# CODEOWNERS -- generated from .github/codeowners/areas.yaml.",
        "# Do not hand-edit. Change areas.yaml and regenerate.",
        "#",
        "# Multiple owners on one line means any one owner can satisfy GitHub's",
        "# code-owner review requirement; the additional teams receive visibility.",
        "#",
        "# Team index:",
    ]
    for label, owner in sorted(policy.teams.items()):
        lines.append(f"#   {label:<20} {owner}")

    current_section = ""
    for rule in policy.rules:
        if rule.section != current_section:
            lines.extend(["", f"# === {rule.section} ==="])
            current_section = rule.section
        lines.append(f"{rule.pattern:<42} {' '.join(rule.owners)}")
    return "\n".join(lines) + "\n"


def tracked_files(repo: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files"],
        text=True,
    )
    return [line for line in output.splitlines() if line]


def validate(policy: Policy, repo: Path) -> None:
    paths = tracked_files(repo)
    shadow_files = [
        path for path in (".github/CODEOWNERS", "docs/CODEOWNERS") if path in paths
    ]
    if shadow_files:
        _fail(f"shadow CODEOWNERS files are not allowed: {shadow_files}")

    unowned = [path for path in paths if not is_explicitly_owned(policy, path)]
    if unowned:
        _fail(
            f"{len(unowned)} tracked path(s) use only the fallback owner; add "
            f"explicit rules: {unowned[:20]}"
        )
    print(
        f"CODEOWNERS coverage: {len(paths)}/{len(paths)} tracked paths explicitly owned"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", default=Path("."), type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.write == args.check:
        _fail("select exactly one of --write or --check")

    policy = load_policy(args.policy)
    expected = render(policy)
    if args.write:
        args.out.write_text(expected)
        print(f"wrote {args.out} ({len(policy.rules)} rules)")
    elif not args.out.exists() or args.out.read_text() != expected:
        _fail(f"{args.out} is out of date; regenerate it from {args.policy}")

    if args.validate:
        validate(policy, args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
