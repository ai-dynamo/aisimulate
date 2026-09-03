#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an AIConfigurator diff against AISimulate's stable mirror paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "scripts" / "aic_sync.toml"


class ManualChangesRequired(RuntimeError):
    """Raised when a sync range contains changes requiring manual adaptation."""


def _git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(source), *args))


def _require_source_path(source: Path, ref: str, path: str) -> None:
    result = subprocess.run(
        ("git", "-C", str(source), "cat-file", "-e", f"{ref}:{path}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"AIC sync source path does not exist at {ref}: {path}")


def _prefix_extended_header_paths(patch: bytes, target: str) -> bytes:
    """Map rename/copy metadata paths alongside the regular diff headers."""

    prefix = target.encode() + b"/"
    markers = (b"rename from ", b"rename to ", b"copy from ", b"copy to ")
    rewritten: list[bytes] = []
    for line in patch.splitlines(keepends=True):
        marker = next((candidate for candidate in markers if line.startswith(candidate)), None)
        if marker is None:
            rewritten.append(line)
            continue
        path = line[len(marker) :]
        if path.startswith(b'"'):
            path = b'"' + prefix + path[1:]
        else:
            path = prefix + path
        rewritten.append(marker + path)
    return b"".join(rewritten)


def _manual_changes(
    config: dict[str, object], source: Path, from_ref: str, to_ref: str
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for mapping in config.get("manual", []):
        upstream = str(mapping["source"]).strip("/")
        entries = _git(
            source,
            "diff",
            "--name-status",
            "--find-renames",
            from_ref,
            to_ref,
            "--",
            upstream,
        ).decode()
        lines = tuple(line for line in entries.splitlines() if line)
        if lines:
            changes.append(
                {
                    "source": upstream,
                    "reason": str(mapping["reason"]),
                    "entries": lines,
                }
            )
    return changes


def _manual_report(changes: list[dict[str, object]], from_ref: str, to_ref: str) -> str:
    lines = [
        "# AIConfigurator manual synchronization report",
        "",
        f"- From: `{from_ref}`",
        f"- To: `{to_ref}`",
        "",
    ]
    if not changes:
        lines.append("No configured manual paths changed in this range.")
        return "\n".join(lines) + "\n"
    for change in changes:
        lines.extend(
            (
                f"## `{change['source']}`",
                "",
                f"Reason: {change['reason']}",
                "",
            )
        )
        lines.extend(f"- `{entry}`" for entry in change["entries"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render(
    source: Path,
    from_ref: str,
    to_ref: str,
    *,
    manual_report: Path | None = None,
) -> bytes:
    config = tomllib.loads(LEDGER.read_text())
    _git(source, "rev-parse", "--verify", f"{from_ref}^{{commit}}")
    _git(source, "rev-parse", "--verify", f"{to_ref}^{{commit}}")

    manual_changes = _manual_changes(config, source, from_ref, to_ref)
    report = _manual_report(manual_changes, from_ref, to_ref)
    if manual_changes and manual_report is None:
        changed = ", ".join(str(change["source"]) for change in manual_changes)
        raise ManualChangesRequired(
            "manual AIC sync paths changed: "
            f"{changed}; rerun with --manual-report and review that report before advancing the ledger"
        )
    if manual_report is not None:
        manual_report.write_text(report)

    chunks: list[bytes] = []
    for mapping in config["mirror"]:
        upstream = str(mapping["source"]).strip("/")
        target = str(mapping["target"]).strip("/")
        _require_source_path(source, from_ref, upstream)
        patch = _git(
            source,
            "diff",
            "--binary",
            "--full-index",
            "--find-renames",
            f"--relative={upstream}",
            f"--src-prefix=a/{target}/",
            f"--dst-prefix=b/{target}/",
            from_ref,
            to_ref,
            "--",
            upstream,
        )
        if patch:
            patch = _prefix_extended_header_paths(patch, target)
            chunks.append(patch.rstrip(b"\n") + b"\n")
    return b"".join(chunks)


def main() -> None:
    config = tomllib.loads(LEDGER.read_text())
    parser = argparse.ArgumentParser(
        description="Create a binary-safe, path-rewritten AIC synchronization patch."
    )
    parser.add_argument("--source", type=Path, required=True, help="AIC git checkout")
    parser.add_argument(
        "--from-ref", default=config["upstream"]["last_synced"], help="old AIC ref"
    )
    parser.add_argument("--to-ref", required=True, help="new AIC ref")
    parser.add_argument("--output", type=Path, help="write patch here (default: stdout)")
    parser.add_argument(
        "--manual-report",
        type=Path,
        help="required output report when configured manual-adaptation paths changed",
    )
    args = parser.parse_args()

    try:
        patch = render(
            args.source.resolve(),
            args.from_ref,
            args.to_ref,
            manual_report=args.manual_report,
        )
    except ManualChangesRequired as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_bytes(patch)
    else:
        sys.stdout.buffer.write(patch)


if __name__ == "__main__":
    main()
