# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check local Markdown file/directory destinations in user-facing docs.

Uses markdown-it-py without network requests. Inline/image links and reference
definitions are checked; code examples and HTML comments are ignored. Diagnostic
lines identify the containing Markdown block. Anchors and HTML links are excluded.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "README.md",
    "DEVELOPMENT.md",
    "CONTRIBUTING.md",
    "python/aisimulate/README.md",
}
DOC_ROOTS = ("docs/", "python/aisimulate/docs/")


def local_links(text: str) -> list[tuple[int, str]]:
    """Return local destinations and their containing block's one-based line."""
    env = {}
    tokens = MarkdownIt("commonmark").parse(text, env)
    destinations = []
    for token in tokens:
        if token.type != "inline" or token.map is None:
            continue
        for child in token.children or []:
            attribute = {"link_open": "href", "image": "src"}.get(child.type)
            if attribute:
                destinations.append((token.map[0] + 1, child.attrGet(attribute)))
    # Also check definitions that are not currently used by a reference link.
    for reference in env.get("references", {}).values():
        destinations.append((reference["map"][0] + 1, reference["href"]))
    links = set()
    for line, value in destinations:
        if not value:
            continue
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        links.add((line, unquote(parsed.path)))
    return sorted(links)


def check_file(path: Path, root: Path) -> list[str]:
    failures = []
    for line, target in local_links(path.read_text(encoding="utf-8")):
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            failures.append(
                f"{path.relative_to(root)}:{line}: missing local destination: {target}"
            )
    return failures


def documentation_files(root: Path) -> list[Path]:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    return sorted(
        root / name
        for name in set(names)
        if name.endswith(".md") and (name in ENTRYPOINTS or name.startswith(DOC_ROOTS))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional Markdown files, relative to the current directory",
    )
    args = parser.parse_args()
    files = (
        [p.resolve() for p in args.paths] if args.paths else documentation_files(ROOT)
    )
    failures = []
    for path in files:
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(f"{path}: expected a Markdown file inside the repository")
            continue
        failures.extend(check_file(path, ROOT))
    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"Checked local destinations in {len(files)} Markdown files (anchors and remote URLs excluded)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
