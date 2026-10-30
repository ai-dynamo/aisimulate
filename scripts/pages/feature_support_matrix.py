# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the Feature Support Matrix page from each branch's ``features.toml``.

The page is static HTML: every branch snapshot is rendered at build time and the
branch selector only toggles which one is visible.
"""

from __future__ import annotations

import html
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass

REPOSITORY_URL = "https://github.com/ai-dynamo/aisimulate"
DATA_PATH = "pages/feature-support-matrix/features.toml"
TEMPLATE_PATH = "pages/feature-support-matrix/template.html"

STATUS = {
    "supported": ("Supported", "bg-lime-100 text-lime-900 dark:bg-lime-900/40 dark:text-lime-200"),
    "limited": ("Limited", "bg-amber-50 text-amber-900 dark:bg-amber-900/30 dark:text-amber-200"),
    "not_supported": ("Not supported", "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400"),
    "na": ("N/A upstream", "bg-white text-slate-400 dark:bg-slate-900 dark:text-slate-500"),
}

_TOP_KEYS = {"schema_version", "engines", "sections"}
_SECTION_KEYS = {"id", "title", "intro", "row_header", "columns", "rows"}
_ROW_KEYS = {"name", "detail", "doc", "cells", "note"}
_DOC = re.compile(r"(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_.-]+\.md(?:#[a-z0-9_-]+)?")
_SECTION_ID = re.compile(r"[a-z][a-z0-9-]*")
_COLUMN_HEADER = "px-3 py-2 text-center font-semibold text-slate-700 dark:text-slate-200"
_ROW_HEADER = "px-3 py-2 text-left sticky left-0 bg-slate-50 dark:bg-slate-800"
_SECONDARY = "text-slate-500 dark:text-slate-400"


class FeatureMatrixError(ValueError):
    """Raised when a ``features.toml`` snapshot is invalid."""


@dataclass(frozen=True)
class Snapshot:
    branch: str
    commit: str | None
    data: dict


def _strings(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def load(text: str, *, source: str) -> dict:
    """Parse one branch's ``features.toml`` and reject anything the page cannot render faithfully."""

    def fail(message: str) -> None:
        raise FeatureMatrixError(f"{source}: {message}")

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise FeatureMatrixError(f"{source}: {exc}") from exc
    if set(data) - _TOP_KEYS:
        fail(f"unknown keys {sorted(set(data) - _TOP_KEYS)}")
    if data.get("schema_version") != 1:
        fail("schema_version must be 1")
    if not _strings(data.get("engines")):
        fail("engines must be a non-empty list of strings")
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        fail("sections must be a non-empty list")
    section_ids = set()
    for section in sections:
        section_id = section.get("id")
        if not isinstance(section_id, str) or not _SECTION_ID.fullmatch(section_id) or section_id in section_ids:
            fail(f"invalid or duplicate section id {section_id!r}")
        section_ids.add(section_id)
        if set(section) - _SECTION_KEYS:
            fail(f"section {section_id}: unknown keys {sorted(set(section) - _SECTION_KEYS)}")
        if not isinstance(section.get("title"), str) or not section["title"]:
            fail(f"section {section_id}: title is required")
        for key in ("intro", "row_header"):
            if key in section and not isinstance(section[key], str):
                fail(f"section {section_id}: {key} must be a string")
        columns = section.get("columns", data["engines"])
        if not _strings(columns):
            fail(f"section {section_id}: columns must be a non-empty list of strings")
        rows = section.get("rows")
        if not isinstance(rows, list) or not rows:
            fail(f"section {section_id}: rows must be a non-empty list")
        for row in rows:
            name = row.get("name")
            if not isinstance(name, str) or not name:
                fail(f"section {section_id}: every row needs a name")
            if set(row) - _ROW_KEYS:
                fail(f"row {name!r}: unknown keys {sorted(set(row) - _ROW_KEYS)}")
            doc = row.get("doc")
            if not isinstance(doc, str) or not _DOC.fullmatch(doc) or ".." in doc.split("#")[0].split("/"):
                fail(f"row {name!r}: doc must be a Markdown path under docs/")
            for key in ("note", "detail"):
                if key in row and not isinstance(row[key], str):
                    fail(f"row {name!r}: {key} must be a string")
            cells = row.get("cells")
            if not isinstance(cells, list) or len(cells) != len(columns):
                fail(f"row {name!r}: needs one cell per column ({len(columns)})")
            for cell in cells:
                status, separator, tag = cell.partition(":") if isinstance(cell, str) else ("", "", "")
                if status not in STATUS or (separator and not tag):
                    fail(f"row {name!r}: invalid cell {cell!r}")
    return data


def _heading_text(heading: str) -> str:
    """Reduce a Markdown heading to the text GitHub renders: no link, emphasis or HTML markup.

    Code spans keep their text verbatim, and an underscore inside a word is not emphasis.
    """
    codes: list[str] = []

    def protect(match: re.Match[str]) -> str:
        codes.append(match.group(1))
        return chr(0xE000 + len(codes) - 1)

    # Stand-ins keep code text out of the markup rules while links around it still match.
    text = re.sub(r"`([^`]*)`", protect, heading)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\*{1,3})(\S(?:.*?\S)?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)(_{1,3})(\S(?:.*?\S)?)\1(?!\w)", r"\2", text)
    return re.sub("[\ue000-\uf8ff]", lambda match: codes[ord(match.group()) - 0xE000], text)


def anchors(markdown: str) -> set[str]:
    """Return a page's explicit ``<a id>`` targets and GitHub heading slugs."""
    found = set(re.findall(r'<a id="([^"]+)"', markdown))
    for heading in re.findall(r"^#+ (.+)$", markdown, re.MULTILINE):
        found.add(re.sub(r"[^\w\- ]", "", _heading_text(heading).strip().lower()).replace(" ", "-"))
    return found


def check_docs(data: dict, read_doc: Callable[[str], str | None], *, source: str) -> None:
    """Reject rows whose page or anchor is missing from the tree the snapshot came from.

    ``read_doc`` takes a path under docs/ and returns its text, or None when absent.
    """
    targets: dict[str, set[str] | None] = {}
    for section in data["sections"]:
        for row in section["rows"]:
            path, _, anchor = row["doc"].partition("#")
            if path not in targets:
                text = read_doc(path)
                targets[path] = None if text is None else anchors(text)
            if targets[path] is None:
                raise FeatureMatrixError(f"{source}: row {row['name']!r}: docs/{path} does not exist")
            if anchor and anchor not in targets[path]:
                raise FeatureMatrixError(f"{source}: row {row['name']!r}: docs/{path} has no anchor #{anchor}")


def release_sort_key(branch: str) -> list[tuple[int, int, str]]:
    """Order release branches by version, numbers before suffixes."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"[./-]", branch)]


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def _cell(spec: str) -> str:
    status, _, tag = spec.partition(":")
    label, classes = STATUS[status]
    extra = f'<span class="block text-[11px] font-normal opacity-70">{_e(tag)}</span>' if tag else ""
    return f'<td class="px-3 py-2.5 text-center text-sm font-medium {classes}">{_e(label)}{extra}</td>'


def _section(section: dict, engines: list[str], prefix: str, docs_url: str) -> str:
    columns = section.get("columns", engines)
    has_notes = any("note" in row for row in section["rows"])
    wide = len(columns) > 3
    first = 18 if wide else 26 if has_notes else 22
    value_width = ((56 if wide else 45) if has_notes else 100 - first) / len(columns)
    colgroup = f'<col style="width:{first}%">' + "".join(f'<col style="width:{value_width:.2f}%">' for _ in columns)
    colgroup += "<col>" if has_notes else ""
    min_width = 200 + 110 * len(columns) + (230 if has_notes else 0)
    head = "".join(f'<th scope="col" class="{_COLUMN_HEADER}">{_e(name)}</th>' for name in columns)
    if has_notes:
        head += '<th scope="col" class="px-3 py-2 text-left">Notes</th>'
    body = []
    for row in section["rows"]:
        link = (
            f'<a href="{_e(docs_url + row["doc"])}" class="font-semibold text-slate-800 dark:text-slate-100 '
            f'no-underline hover:text-accent hover:underline">{_e(row["name"])}</a>'
        )
        if "detail" in row:
            link += f'<span class="block text-xs font-normal {_SECONDARY}">{_e(row["detail"])}</span>'
        notes = f'<td class="px-3 py-2.5 text-xs {_SECONDARY}">{_e(row.get("note", ""))}</td>' if has_notes else ""
        body.append(
            '<tr class="border-t border-slate-200 dark:border-slate-700">'
            '<th scope="row" class="px-3 py-2.5 text-left text-sm sticky left-0 bg-white dark:bg-slate-900">'
            f"{link}</th>{''.join(_cell(cell) for cell in row['cells'])}{notes}</tr>"
        )
    intro = f'<p class="text-sm {_SECONDARY} -mt-2 mb-3">{_e(section["intro"])}</p>' if "intro" in section else ""
    row_header = _e(section.get("row_header", "Technique"))
    return f"""
  <section id="{prefix}{section["id"]}" class="mb-10 scroll-mt-20">
    <h2 class="text-lg font-bold mb-3">{_e(section["title"])}</h2>
    {intro}
    <div class="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-700">
      <table class="w-full border-collapse table-fixed" style="min-width:{min_width}px"><colgroup>{colgroup}</colgroup>
        <thead class="bg-slate-50 dark:bg-slate-800 text-xs {_SECONDARY}">
          <tr><th scope="col" class="{_ROW_HEADER}">{row_header}</th>{head}</tr>
        </thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>
  </section>"""


def _view(snapshot: Snapshot) -> str:
    data = snapshot.data
    prefix = "" if snapshot.branch == "main" else re.sub(r"[^a-z0-9]+", "-", snapshot.branch.lower()) + "-"
    used = {cell.partition(":")[0] for section in data["sections"] for row in section["rows"] for cell in row["cells"]}
    legend = "".join(
        f'<span class="px-2.5 py-1 rounded-md text-xs font-medium border border-slate-200 dark:border-slate-700 '
        f'{classes}">{_e(label)}</span>'
        for status, (label, classes) in STATUS.items()
        if status in used
    )
    nav = "".join(
        f'<a href="#{prefix}{section["id"]}" class="px-3 py-1 rounded-full text-xs font-medium border border-slate-200 '
        "dark:border-slate-700 text-slate-600 dark:text-slate-300 no-underline hover:border-accent hover:text-accent "
        f'whitespace-nowrap">{_e(section["title"])}</a>'
        for section in data["sections"]
    )
    # Pin links to the rendered commit; a branch can move after the page is built.
    tree_url = f"{REPOSITORY_URL}/blob/{snapshot.commit or snapshot.branch}/"
    meta = [f"Branch <code>{_e(snapshot.branch)}</code>"]
    if snapshot.commit:
        meta.append(
            f'commit <a class="underline" href="{REPOSITORY_URL}/commit/{_e(snapshot.commit)}">'
            f"<code>{_e(snapshot.commit[:9])}</code></a>"
        )
    meta.append(f'source <a class="underline" href="{_e(tree_url + DATA_PATH)}"><code>features.toml</code></a>')
    sections = "".join(_section(section, data["engines"], prefix, tree_url + "docs/") for section in data["sections"])
    hidden = "" if snapshot.branch == "main" else " hidden"
    return f"""
  <div data-branch="{_e(snapshot.branch)}"{hidden}>
  <p class="text-xs text-slate-500 dark:text-slate-400 mb-6">{" · ".join(meta)}</p>
  <div class="flex flex-wrap items-center gap-2 mb-4">{legend}</div>
  <nav class="flex gap-2 overflow-x-auto pb-2 mb-8">{nav}</nav>
{sections}
  </div>"""


def render(template: str, snapshots: list[Snapshot]) -> str:
    """Fill the page template with one view per snapshot; ``main`` must come first."""
    if not snapshots or snapshots[0].branch != "main":
        raise FeatureMatrixError("the main snapshot must be rendered first")
    if sorted(re.findall(r"\{\{[^}]*\}\}", template)) != ["{{BRANCH_OPTIONS}}", "{{VIEWS}}"]:
        raise FeatureMatrixError("the page template needs {{BRANCH_OPTIONS}} and {{VIEWS}} once each")
    options = "".join(f'<option value="{_e(s.branch)}">{_e(s.branch)}</option>' for s in snapshots)
    return template.replace("{{BRANCH_OPTIONS}}", options).replace("{{VIEWS}}", "".join(_view(s) for s in snapshots))
