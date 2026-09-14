# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check local Markdown file/directory destinations in user-facing docs.

No network requests or dependencies. Inline/image links and reference-link
definitions are checked; code examples and HTML comments are ignored. This is
a destination check, not an anchor, HTML, or full Markdown syntax validator.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {"README.md", "DEVELOPMENT.md", "CONTRIBUTING.md", "python/aisimulate/README.md"}
DOC_ROOTS = ("docs/", "python/aisimulate/docs/")


def prose(text: str) -> str:
    """Mask examples without changing offsets used for line diagnostics."""
    text = re.sub(r"<!--.*?-->", lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S)
    lines = text.splitlines(keepends=True)
    fence = None
    for i, line in enumerate(lines):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\n"))
        if fence:
            lines[i] = re.sub(r"[^\n]", " ", line)
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
        elif match:
            fence = match[1]
            lines[i] = re.sub(r"[^\n]", " ", line)
    text = "".join(lines)
    return re.sub(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)", lambda m: re.sub(r"[^\n]", " ", m[0]), text, flags=re.S)


def destination(text: str, start: int) -> str | None:
    """Read an angle-delimited or balanced bare Markdown destination."""
    start += len(text[start:]) - len(text[start:].lstrip())
    if start == len(text):
        return None
    if text[start] == "<":
        end = text.find(">", start + 1)
        return text[start + 1 : end] if end >= 0 else None
    depth = 0
    end = start
    while end < len(text):
        char = text[end]
        if char == "\\" and end + 1 < len(text):
            end += 2
            continue
        if char.isspace() or (char == ")" and depth == 0):
            break
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        end += 1
    return text[start:end] or None


def local_links(text: str) -> list[tuple[int, str]]:
    text = prose(text)
    starts = [m.end() for m in re.finditer(r"!?\[[^\]]*?\]\(", text)]
    starts += [m.end() for m in re.finditer(r"^ {0,3}\[[^\]\n]+\]:[ \t]*", text, flags=re.M)]
    links = []
    for start in sorted(starts):
        value = destination(text, start)
        if not value:
            continue
        value = html.unescape(re.sub(r"\\([\\`*_{}\[\]()#+.!<> -])", r"\1", value))
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        links.append((text.count("\n", 0, start) + 1, unquote(parsed.path)))
    return links


def check_file(path: Path, root: Path) -> list[str]:
    failures = []
    for line, target in local_links(path.read_text(encoding="utf-8")):
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            failures.append(f"{path.relative_to(root)}:{line}: missing local destination: {target}")
    return failures


def documentation_files(root: Path) -> list[Path]:
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
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
        "paths", nargs="*", type=Path, help="Optional Markdown files, relative to the current directory"
    )
    args = parser.parse_args()
    files = [p.resolve() for p in args.paths] if args.paths else documentation_files(ROOT)
    failures = []
    for path in files:
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(f"{path}: expected a Markdown file inside the repository")
            continue
        failures.extend(check_file(path, ROOT))
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked local destinations in {len(files)} Markdown files (anchors and remote URLs excluded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
