# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Dynamo replay adapter against one exact AISimulate checkout."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

DYNAMO_REVISION = "def3b79b15c266805540a678dd400aeb6ccada1d"


# Compatibility adaptation of NVIDIA Dynamo's Apache-2.0 handoff adapter:
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/common/handoff.rs
# Modifies only the CI-built legacy token-KV DTO conversion, not engine scheduling.
def patch_handoff_timing(core: Path, dynamo: Path) -> dict:
    path = dynamo / "lib/mocker/src/common/handoff.rs"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != "99a2097d671f4a26dbd1761a44858a19f62ced37dd9209858627b2a289d7937d":
        raise ValueError("pinned Dynamo handoff adapter differs from reviewed source")
    api = (core / "src/engine/handoff.rs").read_text()
    match = re.search(r"pub struct HandoffTransferTiming\s*\{([^{}]*)\}", api)
    fields = dict(re.findall(r"^\s*pub (\w+): ([^\n,]+),\s*$", match[1], re.MULTILINE)) if match else {}
    legacy = {
        "mode": "TransferTimingMode",
        "full_prompt_tokens": "usize",
        "kv_bytes_per_token": "Option<usize>",
        "bandwidth_gb_s": "Option<f64>",
    }
    if fields not in (legacy, legacy | {"state_bytes": "usize"}):
        raise ValueError("unreviewed AISimulate handoff API; update the pinned adapter")
    patched = original
    if "state_bytes" in fields:
        anchor = b"        kv_bytes_per_token: timing.kv_bytes_per_token,\n"
        if original.count(anchor) != 1:
            raise ValueError("pinned handoff conversion anchor is not unique")
        patched = original.replace(
            anchor,
            anchor
            + (
                b"        // Modified by AISimulate CI: this legacy DTO models token KV only.\n"
                b"        state_bytes: 0,\n"
            ),
        )
        path.write_bytes(patched)
    return {
        "path": str(path.relative_to(dynamo)),
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(patched).hexdigest(),
        "token_only_state_bytes": 0,
        "api_has_state_bytes": "state_bytes" in fields,
    }


class LiteralSite(NamedTuple):
    path: str  # Dynamo source file containing the literal.
    sha256: str  # Reviewed content of that file at DYNAMO_REVISION.
    anchor: str  # Text that occurs once in the file and precedes the literal.
    struct: str  # AISimulate struct built by the literal.
    api: str  # AISimulate core source file that defines the struct.


# Compatibility adaptation of NVIDIA Dynamo's Apache-2.0 sources at DYNAMO_REVISION:
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/grouped_scheduler/compatibility.rs
# https://github.com/ai-dynamo/dynamo/blob/def3b79b15c266805540a678dd400aeb6ccada1d/lib/mocker/src/replay/offline/extensions/kv_router/mod.rs
# Adds `field: None` for new optional AISimulate fields to these exhaustive
# struct literals only; Dynamo behavior is otherwise unchanged.
OPTIONAL_FIELD_LITERALS = (
    LiteralSite(
        "lib/mocker/src/grouped_scheduler/compatibility.rs",
        "93f9ca8a2faca81c96973b2f4b598d2ab16a33f166fdaa408fcf9b746f5ab369",
        "pub(super) fn native_request(&self, request: DirectRequest) -> Request {",
        "Request",
        "src/engine/protocol.rs",
    ),
    LiteralSite(
        "lib/mocker/src/replay/offline/extensions/kv_router/mod.rs",
        "fe45ce455bdab419f0d80edaf41e7b662bdb3bc993ed05293726ac765a05996f",
        "fn placement(&self, admission: WorkerAdmission) -> Placement {",
        "Placement",
        "src/replay/core/mod.rs",
    ),
)
_OPTION_TYPE = re.compile(r"(?:(?:std|core)::option::)?Option\s*<.*>", re.DOTALL)


_RUST_NON_CODE = re.compile(r"//[^\n]*|/\*.*?\*/|r(#*)\".*?\"\1|b?\"(?:\\.|[^\"\\])*\"|b?'(?:\\.|[^'\\\n])'", re.DOTALL)


def _mask(text: str) -> str:
    """Blank Rust comments and string/char literals, keeping offsets and newlines."""
    return _RUST_NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match[0]), text)


def _braced(text: str, start: int) -> tuple[int, int]:
    """Return the span inside the first brace at or after ``start``, outside comments and literals."""
    text = _mask(text)
    open_at = text.index("{", start)
    depth = 0
    for index in range(open_at, len(text)):
        depth += {"{": 1, "(": 1, "[": 1, "}": -1, ")": -1, "]": -1}.get(text[index], 0)
        if depth == 0:
            return open_at + 1, index
    raise ValueError("unbalanced braces")


def _top_level_items(body: str, angle: bool) -> list[str]:
    """Split a brace body at top-level commas, ignoring comments, literals and attributes."""
    body = _mask(body)
    # A `cfg` attribute can remove a field from the build; refuse rather than guess.
    if re.search(r"#\s*\[[^\]]*\bcfg\s*\(", body):
        raise ValueError(f"conditional (`cfg`) fields are not supported: {body.strip()!r}")
    body = re.sub(r"#\[(?:[^\[\]]|\[[^\[\]]*\])*\]", "", body)
    nesting = {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}
    if angle:  # Struct field types nest generics; literal expressions may compare.
        body = body.replace("->", "  ")
        nesting |= {"<": 1, ">": -1}
    items, depth, current = [], 0, []
    for char in body:
        depth += nesting.get(char, 0)
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [item.strip() for item in items if item.strip()]


def _struct_fields(api_source: str, struct: str) -> dict[str, tuple[bool, str]]:
    """Map each field of ``pub struct <struct>`` to (plain ``pub``, type)."""
    matches = list(re.finditer(rf"\bpub struct {struct}\b\s*(?:<[^>{{]*>)?\s*\{{", api_source))
    if len(matches) != 1:
        raise ValueError(f"expected one AISimulate `pub struct {struct}` with named fields, found {len(matches)}")
    start, end = _braced(api_source, matches[0].start())
    fields = {}
    for item in _top_level_items(api_source[start:end], angle=True):
        field = re.fullmatch(r"(pub(?:\s*\([^)]*\))?\s+)?(?:r#)?(\w+)\s*:\s*(.+)", item, re.DOTALL)
        if not field:
            raise ValueError(f"cannot parse AISimulate {struct} field {item!r}")
        fields[field[2]] = ((field[1] or "").strip() == "pub", re.sub(r"\s+", " ", field[3]))
    return fields


def _patch_literal(source: str, site: LiteralSite, struct_fields: dict) -> tuple[str, list[str]]:
    """Insert ``None`` for missing optional fields into the literal after ``site.anchor``."""
    if source.count(site.anchor) != 1:
        raise ValueError(f"{site.path}: literal anchor is not unique")
    literal = re.compile(rf"(?<![\w:]){site.struct}\s*\{{").search(source, source.index(site.anchor) + len(site.anchor))
    if not literal:
        raise ValueError(f"{site.path}: no {site.struct} literal after the anchor")
    start, close = _braced(source, literal.start())
    present = []
    for item in _top_level_items(source[start:close], angle=False):
        field = re.fullmatch(r"(?:r#)?(\w+)\s*(?::(?!:).*)?", item, re.DOTALL)
        if not field:
            raise ValueError(f"{site.path}: unsupported {site.struct} literal item {item!r}")
        present.append(field[1])
    stale = [name for name in present if name not in struct_fields]
    if stale:
        raise ValueError(f"{site.path}: {site.struct} literal sets fields AISimulate no longer has: {stale}")
    missing = [name for name in struct_fields if name not in present]
    required = [
        name for name in missing if not (struct_fields[name][0] and _OPTION_TYPE.fullmatch(struct_fields[name][1]))
    ]
    if required:
        raise ValueError(
            f"{site.path}: AISimulate {site.struct} adds fields Dynamo must set (not `pub` Option): {required}"
        )
    if not missing:
        return source, []
    line_start = source.rindex("\n", 0, close) + 1
    if source[line_start:close].strip() or not source[:line_start].rstrip().endswith(","):
        raise ValueError(f"{site.path}: {site.struct} literal is not multi-line with a trailing comma")
    indent = re.search(r"\n([ \t]*)\S", source[start - 1 : close])[1]
    inserted = f"{indent}// Modified by AISimulate CI: Dynamo does not set these optional fields yet.\n"
    inserted += "".join(f"{indent}{name}: None,\n" for name in missing)
    return source[:line_start] + inserted + source[line_start:], missing


# Fail closed: each file must match its reviewed sha256, every field the literal
# lacks must be a public `Option`, and the literal may not set a removed field.
def patch_optional_literal_fields(core: Path, dynamo: Path) -> list[dict]:
    originals, patched, records = {}, {}, []
    for site in OPTIONAL_FIELD_LITERALS:
        if site.path not in originals:
            originals[site.path] = (dynamo / site.path).read_bytes()
            patched[site.path] = originals[site.path].decode()
        if hashlib.sha256(originals[site.path]).hexdigest() != site.sha256:
            raise ValueError(f"pinned Dynamo {site.path} differs from reviewed source")
        struct_fields = _struct_fields((core / site.api).read_text(), site.struct)
        patched[site.path], inserted = _patch_literal(patched[site.path], site, struct_fields)
        records.append(
            {"path": site.path, "struct": site.struct, "before_sha256": site.sha256, "inserted_fields": inserted}
        )
    for record in records:
        record["after_sha256"] = hashlib.sha256(patched[record["path"]].encode()).hexdigest()
    for path, text in patched.items():
        if text.encode() != originals[path]:
            (dynamo / path).write_bytes(text.encode())
    return records


def build(source: Path, dynamo: Path, output: Path) -> None:
    source, dynamo, output = source.resolve(), dynamo.resolve(), output.resolve()
    revision = subprocess.check_output(["git", "-C", str(dynamo), "rev-parse", "HEAD"], text=True).strip()
    if revision != DYNAMO_REVISION:
        raise ValueError("unexpected Dynamo revision")
    subprocess.run(["git", "-C", str(dynamo), "diff", "--exit-code", "HEAD"], check=True)
    core = source / "crates/core"
    handoff_compatibility = patch_handoff_timing(core, dynamo)
    literal_compatibility = patch_optional_literal_fields(core, dynamo)
    bindings = dynamo / "lib/bindings/python"
    for manifest, options in (
        (dynamo / "Cargo.toml", ""),
        (bindings / "Cargo.toml", ', optional = true, features = ["python"]'),
    ):
        text, count = re.subn(
            r"^aisimulate-core = .*",
            lambda _: f"aisimulate-core = {{ path = {json.dumps(str(core))}{options} }}",
            manifest.read_text(),
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected one AISimulate dependency in {manifest}")
        manifest.write_text(text)
    # Resolve the local core's dependency graph from Dynamo's existing lock.
    # Compilation stays locked to this resolved graph, which is saved below.
    lock = bindings / "Cargo.lock"
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--features", "ais-forward-pass"],
            cwd=bindings,
        )
    )
    linked = [p["manifest_path"] for p in metadata["packages"] if p["name"] == "aisimulate-core"]
    if linked != [str(core / "Cargo.toml")]:
        raise ValueError(f"Dynamo resolved a different AISimulate core: {linked}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--features",
            "ais-forward-pass",
            "--out",
            str(output),
        ],
        cwd=bindings,
        check=True,
    )
    subprocess.run(
        ["uv", "build", "--wheel", "--no-build-isolation", "--out-dir", str(output), str(dynamo)],
        check=True,
    )
    (output / "dynamo-Cargo.lock").write_bytes(lock.read_bytes())
    (output / "dynamo-build.json").write_text(
        json.dumps(
            {
                "source_sha": revision,
                "aisimulate_sha": subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
                ).strip(),
                "features": ["ais-forward-pass"],
                "handoff_compatibility": handoff_compatibility,
                "literal_compatibility": literal_compatibility,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "dynamo-build.patch").write_bytes(subprocess.check_output(["git", "-C", str(dynamo), "diff", "HEAD"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dynamo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.dynamo, args.output)
