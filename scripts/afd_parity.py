#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare and refresh native AFD replay goldens after explained modeling changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "crates/core/parity_tests/perfmodel/goldens"

E2E_RTOL = 1e-6
# Absolute floor under a relative tolerance: golden fields that are pure
# float noise around zero (e.g. std_* of a constant frozen step, ~1e-13)
# have no meaningful relative error. 1e-9 is orders of magnitude tighter
# than the e2e rtol on any ms / tok-s quantity.
ABS_FLOOR = 1e-9

# summary fields that are counts/identifiers rather than deterministic floats
# of the measurement: compared exactly. Keys absent from the golden summary are
# not listed — a dead key here silently widens nothing but hides drift.
EXACT_KEYS = {
    "completed_requests",
    "num_requests",
    "num_e2e_latency_samples",
    "num_ttft_samples",
    "num_tpot_samples",
    "total_input_tokens",
    "total_output_tokens",
    "decode_gpus_per_worker",
    "prefill_gpus_per_worker",
}

# Wall-clock-derived fields: the replay's real execution speed on the golden
# machine, not simulated time. Excluded from the gate like every other
# wall-clock timestamp in the parity conventions.
WALL_CLOCK_KEYS = {
    "wall_time_ms",
    "processed_tokens_per_s",
    "processed_output_tokens_per_s",
}


class Divergence(Exception):
    pass


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def diff_maps(
    golden,
    actual,
    *,
    rtol: float,
    path: str = "",
    exact_keys: frozenset | set = frozenset(),
    require_keys: bool = False,
) -> list[str]:
    """Field-by-field recursive diff over golden keys; returns failure lines.

    Actual-only keys are allowed. Missing golden keys fail when
    ``require_keys`` is set, as required by the native replay gate.
    """
    failures: list[str] = []
    if isinstance(golden, dict) and isinstance(actual, dict):
        for key in sorted(golden):
            if key in WALL_CLOCK_KEYS:
                continue
            if key not in actual:
                if require_keys:
                    failures.append(f"{path}/{key}: golden key missing from actual")
                continue
            failures += diff_maps(
                golden[key],
                actual[key],
                rtol=rtol,
                path=f"{path}/{key}",
                exact_keys=exact_keys,
                require_keys=require_keys,
            )
        return failures
    if _is_number(golden) and _is_number(actual):
        leaf = path.rsplit("/", 1)[-1]
        if leaf in exact_keys:
            if golden != actual:
                failures.append(f"{path}: exact mismatch golden={golden!r} actual={actual!r}")
            return failures
        g, a = float(golden), float(actual)
        if not (math.isfinite(g) and math.isfinite(a)):
            if g != a:
                failures.append(f"{path}: non-finite mismatch golden={g!r} actual={a!r}")
            return failures
        delta = abs(g - a)
        limit = max(rtol * max(abs(g), 1e-300), ABS_FLOOR)
        if delta > limit:
            rel = delta / max(abs(g), 1e-300)
            failures.append(f"{path}: golden={g!r} actual={a!r} abs_delta={delta:.3e} rel_delta={rel:.3e} rtol={rtol}")
        return failures
    if golden != actual:
        failures.append(f"{path}: golden={golden!r} actual={actual!r}")
    return failures


def _load(path: Path):
    return json.loads(path.read_text())


def _check_e2e(case: str, actual_path: Path) -> list[str]:
    golden_case = _load(GOLDEN_DIR / "afd_e2e.json")["cases"][case]
    prediction = _load(actual_path)
    metrics = prediction.get("summary", prediction)
    return diff_maps(
        {"summary": golden_case["summary"]},
        {"summary": metrics},
        rtol=E2E_RTOL,
        exact_keys=EXACT_KEYS,
        require_keys=True,
    )


def _without_wall_clock(value):
    if isinstance(value, dict):
        return {key: _without_wall_clock(item) for key, item in value.items() if key not in WALL_CLOCK_KEYS}
    if isinstance(value, list):
        return [_without_wall_clock(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_build(manifest_path: Path) -> dict:
    """Refuse stale binaries or unrecorded source changes before generation."""
    from aisimulate import _runtime

    manifest = _load(manifest_path)
    for field in ("source_commit", "source_tree", "binary_sha256", "source_files", "data_tree"):
        if not manifest.get(field):
            raise ValueError(f"build manifest missing {field}")
    if len(manifest["source_commit"]) != 40 or len(manifest["source_tree"]) != 40:
        raise ValueError("build manifest requires immutable source commit/tree SHAs")
    for relative, expected in manifest["source_files"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or _sha256(path) != expected:
            raise ValueError(f"source changed after build: {relative}")
    if _sha256(Path(_runtime.__file__)) != manifest["binary_sha256"]:
        raise ValueError("loaded native binary differs from the build manifest")
    git = ["git", "--git-dir", str(manifest.get("source_git_dir", ROOT / ".git"))]
    tree = subprocess.check_output(git + ["rev-parse", f"{manifest['source_commit']}^{{tree}}"], text=True).strip()
    if tree != manifest["source_tree"]:
        raise ValueError("source commit and tree disagree")
    entries = subprocess.check_output(git + ["ls-tree", "-rz", manifest["source_commit"]]).split(b"\0")
    blobs = {}
    for entry in entries:
        if entry:
            metadata, name = entry.split(b"\t", 1)
            blobs[name.decode()] = metadata.split()[2].decode()
    for relative in manifest["source_files"]:
        source = ROOT / relative
        content = os.readlink(source).encode() if source.is_symlink() else source.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blobs.get(relative) != blob:
            raise ValueError(f"source differs from immutable commit: {relative}")
    tracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "crates/core/src",
                "python/aisimulate/src",
                "scripts/afd_parity.py",
                "Cargo.toml",
                "crates/core/Cargo.toml",
                "Cargo.lock",
                "python/aisimulate/pyproject.toml",
                "python/aisimulate/uv.lock",
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    required = {name for name in tracked if name and "/systems/data/" not in name}
    required.update(str(path.relative_to(ROOT)) for path in (ROOT / "crates/core/src").rglob("*.rs"))
    if required - manifest["source_files"].keys():
        raise ValueError("build manifest does not cover all runtime source inputs")
    data_path = "python/aisimulate/src/aisimulate_core/systems/data"
    if subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", data_path], cwd=ROOT):
        raise ValueError("performance data has unrecorded files")
    if subprocess.check_output(["git", "diff", "HEAD", "--", data_path], cwd=ROOT):
        raise ValueError("performance data has unrecorded changes")
    actual_tree = subprocess.check_output(["git", "rev-parse", f"HEAD:{data_path}"], cwd=ROOT, text=True).strip()
    if actual_tree != manifest["data_tree"]:
        raise ValueError("performance data tree differs from build manifest")
    return {key: value for key, value in manifest.items() if key not in {"source_files", "source_git_dir"}}


def _refresh_e2e(
    cases: list[str], reason: str, attributions: dict[str, str] | None = None, *, build_manifest: Path
) -> None:
    path = GOLDEN_DIR / "afd_e2e.json"
    fixture = _load(path)
    selected = (
        [name for name, record in fixture["cases"].items() if not record.get("historical", False)]
        if cases == ["all"]
        else cases
    )
    unknown = set(selected) - set(fixture["cases"])
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(f"unknown or duplicate cases: {cases}")
    if not reason.strip():
        raise ValueError("a modeling-change reason is required")
    attributions = attributions or {}
    if set(attributions) - set(selected) or any(not value.strip() for value in attributions.values()):
        raise ValueError("attribution overrides require a selected case and a nonempty description")
    provenance = _verified_build(build_manifest)
    env = dict(os.environ)
    cli = ROOT / "python/aisimulate/.venv/bin/aisimulate"
    for case in selected:
        record = fixture["cases"][case]
        config_digest = _sha256(ROOT / record["config"])
        with tempfile.TemporaryDirectory(prefix="afd-golden-refresh-") as directory:
            command = [
                str(cli),
                "predict",
                "--stack",
                "engine",
                "--config",
                str(ROOT / record["config"]),
                "--output-dir",
                directory,
                "--format",
                "json",
            ]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f"{case}: prediction failed:\n{result.stderr}")
            prediction = _load(Path(directory) / "prediction.json")
            summary = prediction.get("summary", prediction)
        if _sha256(ROOT / record["config"]) != config_digest:
            raise ValueError(f"{case}: configuration changed during prediction")
        for key in EXACT_KEYS:
            if key in record["summary"] and summary.get(key) != record["summary"][key]:
                raise ValueError(f"{case}: lifecycle/accounting field {key} changed")
        differences = diff_maps(record["summary"], summary, rtol=E2E_RTOL, exact_keys=EXACT_KEYS, require_keys=True)
        print(f"{case}: {len(differences)} changed fields", flush=True)
        # Keep the established portable summary surface. Runtime provenance
        # includes machine-local roots; generation identity is pinned below.
        record["summary"] = _without_wall_clock({key: summary[key] for key in record["summary"]})
        record["modeling_change"] = reason
        record["generation"] = {
            **provenance,
            "config_sha256": config_digest,
            "command": [
                "aisimulate",
                "predict",
                "--stack",
                "engine",
                "--config",
                record["config"],
                "--output-dir",
                "<temporary>",
                "--format",
                "json",
            ],
            "reason": reason,
        }
        if case in attributions:
            if record.get("attribution") != attributions[case]:
                record.setdefault("historical_attribution", []).append(record.get("attribution"))
            record["attribution"] = attributions[case]
    for case in selected:
        record = fixture["cases"][case]
        if _sha256(ROOT / record["config"]) != record["generation"]["config_sha256"]:
            raise ValueError(f"{case}: configuration changed before publication")
    # Recheck after every prediction; never publish mixed-source generations.
    _verified_build(build_manifest)
    if fixture.get("provenance", {}).get("git_head"):
        fixture.setdefault("historical_provenance", []).append(fixture.pop("provenance"))
    fixture["provenance"] = {
        "generator": "scripts/afd_parity.py refresh-e2e",
        "scope": "Generation identity is recorded per case; historical_attribution describes retired behavior only.",
        "tolerance": {"relative": E2E_RTOL, "absolute": ABS_FLOOR, "scope": "all golden keys except wall-clock fields"},
    }
    # All selected cases must succeed before the fixture is replaced.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="afd-golden-", delete=False) as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
        handle.write("\n")
        replacement = Path(handle.name)
    replacement.replace(path)


def layout_config(case: dict, directory: Path) -> dict:
    """Deterministic authored traffic shared by the layout gate and generator."""
    import copy

    raw = copy.deepcopy(case["config"])
    if case["trace"]:
        trace = directory / "requests.jsonl"
        trace.write_text(
            "".join(
                json.dumps(
                    {
                        "request_id": f"request-{i}",
                        "timestamp": i * 2,
                        "input_length": 128 + (i % 2) * 32,
                        "output_length": 8 + i % 2,
                        "hash_ids": [42, 43, 1000 + i],
                    }
                )
                + "\n"
                for i in range(4)
            )
        )
        raw["traffic"]["source"] = {"type": "trace", "format": "mooncake", "paths": [str(trace)]}
        raw["traffic"]["stop"] = {}
    return raw


def _refresh_layouts(cases: list[str], reason: str, build_manifest: Path) -> None:
    import yaml

    path = GOLDEN_DIR / "afd_execution_plan.json"
    original = path.read_bytes()
    fixture = json.loads(original)
    selected = list(fixture["cases"]) if cases == ["all"] else cases
    if not reason.strip() or len(selected) != len(set(selected)) or set(selected) - fixture["cases"].keys():
        raise ValueError("a reason and known, unique layout cases are required")
    provenance = _verified_build(build_manifest)
    for case in selected:
        record = fixture["cases"][case]
        with tempfile.TemporaryDirectory(prefix="afd-layout-refresh-") as directory:
            directory = Path(directory)
            raw = layout_config(record, directory)
            config = directory / "layout.yaml"
            config.write_text(yaml.safe_dump(raw))
            result = subprocess.run(
                [
                    str(ROOT / "python/aisimulate/.venv/bin/aisimulate"),
                    "predict",
                    "--stack",
                    "engine",
                    "--config",
                    str(config),
                    "--output-dir",
                    str(directory / "out"),
                    "--format",
                    "json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise ValueError(f"{case}: prediction failed:\n{result.stderr}")
            actual = _load(directory / "out/prediction.json")
            summary = actual.get("summary", actual)
            trace_hash = _sha256(directory / "requests.jsonl") if record["trace"] else None
        for field, expected in record["expected"].items():
            if field in EXACT_KEYS and summary.get(field) != expected:
                raise ValueError(f"{case}: lifecycle/accounting field {field} changed")
        failures = diff_maps(record["expected"], summary, rtol=E2E_RTOL, exact_keys=EXACT_KEYS, require_keys=True)
        print(f"{case}: {len(failures)} changed fields", flush=True)
        record["expected"] = {field: summary[field] for field in record["expected"]}
        record["modeling_change"] = reason
        record["generation"] = {
            **provenance,
            "reason": reason,
            "config_sha256": hashlib.sha256(json.dumps(record["config"], sort_keys=True).encode()).hexdigest(),
            "trace_sha256": trace_hash,
            "command": [
                "aisimulate",
                "predict",
                "--stack",
                "engine",
                "--config",
                "<generated-layout.yaml>",
                "--output-dir",
                "<temporary>",
                "--format",
                "json",
            ],
        }
    _verified_build(build_manifest)
    if path.read_bytes() != original:
        raise ValueError("layout fixture changed during generation")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="afd-layout-", delete=False) as handle:
        json.dump(fixture, handle, indent=2, sort_keys=True)
        handle.write("\n")
        replacement = Path(handle.name)
    replacement.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    e2e = sub.add_parser("e2e", help="diff an actual prediction.json against the e2e golden")
    e2e.add_argument("--case", required=True, help="case id in goldens/afd_e2e.json")
    e2e.add_argument("--actual", required=True, type=Path, help="actual prediction.json")

    refresh = sub.add_parser("refresh-e2e", help="regenerate selected current CLI cases after a modeling change")
    refresh.add_argument("--case", required=True, nargs="+", help="case IDs, or 'all'")
    refresh.add_argument("--reason", required=True, help="explain the intentional modeling change")
    refresh.add_argument("--build-manifest", required=True, type=Path, help="verified source/binary build identity")
    refresh.add_argument(
        "--attribution",
        nargs=2,
        action="append",
        metavar=("CASE", "DESCRIPTION"),
        help="Replace a selected case's description while retaining its historical attribution",
    )

    layouts = sub.add_parser("refresh-layouts", help="refresh real-source layout replay expectations")
    layouts.add_argument("--case", required=True, nargs="+")
    layouts.add_argument("--reason", required=True)
    layouts.add_argument("--build-manifest", required=True, type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "refresh-layouts":
            _refresh_layouts(args.case, args.reason, args.build_manifest)
            return 0
        if args.command == "refresh-e2e":
            _refresh_e2e(args.case, args.reason, dict(args.attribution or []), build_manifest=args.build_manifest)
            return 0
        failures = _check_e2e(args.case, args.actual)
    except (KeyError, OSError, ValueError) as exc:
        print(f"parity harness error: {exc}", file=sys.stderr)
        return 1

    if failures:
        print(f"PARITY DIVERGED ({len(failures)} fields):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
