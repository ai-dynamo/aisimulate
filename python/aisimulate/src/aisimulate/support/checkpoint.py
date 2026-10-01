# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable agent-supplied onboarding context, not a workflow executor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue

from aisimulate.config.common import StrictModel, load_yaml

from .schema import SupportRequest

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Progress(StrictModel):
    """Agent-reported progress; completion is not runtime verification."""

    stage: int = Field(default=1, strict=True, ge=1, le=6)
    status: Literal["in_progress", "blocked", "complete"] = "in_progress"
    blockers: list[str] = Field(default_factory=list)
    next_action: str | None = None


class Artifact(StrictModel):
    path: str = Field(min_length=1)
    kind: Literal["file", "request", "profile", "collector_checkpoint"] = "file"
    scope: Literal["input", "collection", "validation"] = "input"
    archived: bool = Field(default=False, strict=True)
    sha256: Digest | None = None
    context_sha256: Digest | None = None


class Acceptance(StrictModel):
    sha256: Digest
    revision: int = Field(strict=True, ge=1)


class Configuration(StrictModel):
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    validation_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    draft_request: dict[str, JsonValue] = Field(default_factory=dict)
    progress: Progress = Field(default_factory=lambda: Progress(stage=3))
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    acceptance: Acceptance | None = None
    history: list[dict[str, JsonValue]] = Field(default_factory=list)


class Checkpoint(StrictModel):
    schema_version: Literal["aisimulate-onboarding-checkpoint/v1"] = "aisimulate-onboarding-checkpoint/v1"
    revision: int = Field(default=0, strict=True, ge=0)
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    validation_inputs: dict[str, JsonValue] = Field(default_factory=dict)
    research: dict[str, JsonValue] = Field(default_factory=dict)
    decisions: dict[str, JsonValue] = Field(default_factory=dict)
    pending_questions: list[JsonValue] = Field(default_factory=list)
    progress: Progress = Field(default_factory=Progress)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    configurations: dict[str, Configuration] = Field(default_factory=dict)


def add_checkpoint_parsers(actions: Any) -> None:
    checkpoint = actions.add_parser("checkpoint", help="Save or inspect one resumable onboarding session.")
    checkpoint.add_argument("--file", required=True, help="Session JSON path; keep outside fresh init output roots.")
    checkpoint.add_argument("--update", help="JSON merge patch file, or - for stdin; supplied values are data only.")
    checkpoint.add_argument("--expect-revision", type=int, help="Required current revision for existing updates.")
    checkpoint.add_argument(
        "--accept-profile",
        action="append",
        default=[],
        metavar="CONFIG_ID",
        help="Record user approval of the exact complete draft; repeat to accept several configurations.",
    )
    resume = actions.add_parser("resume", help="Verify and report saved onboarding context without executing work.")
    resume.add_argument("--checkpoint", required=True)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _read_json(text: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value} is unsupported")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"duplicate JSON key {key!r}")
            values[key] = value
        return values

    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("checkpoint and update must be JSON objects")
    return value


def _load(path: Path) -> Checkpoint:
    payload = _read_json(path.read_text(encoding="utf-8"))
    if "schema_version" not in payload or "revision" not in payload:
        raise ValueError("saved checkpoint requires schema_version and revision")
    state = Checkpoint.model_validate(payload)
    if state.revision < 1:
        raise ValueError("saved checkpoint revision must be at least 1")
    groups = [(state.artifacts, None)] + [(config.artifacts, config) for config in state.configurations.values()]
    for refs, config in groups:
        for ref in refs.values():
            if ref.kind == "collector_checkpoint":
                if config is None or ref.scope != "collection" or ref.sha256 is not None:
                    raise ValueError(
                        "collector checkpoint references require configuration collection scope without SHA-256"
                    )
            elif ref.sha256 is None:
                raise ValueError("saved immutable artifact is missing its SHA-256 snapshot")
            if (ref.context_sha256 is None) != (ref.scope == "input"):
                raise ValueError("saved artifact context SHA-256 must exist only for collection/validation scope")
    for config in state.configurations.values():
        if config.acceptance is not None and config.acceptance.revision > state.revision:
            raise ValueError("profile acceptance revision cannot exceed checkpoint revision")
    for ref in state.artifacts.values():
        if ref.scope != "input":
            raise ValueError(
                "shared artifacts must use scope=input; put collection/validation files in a configuration"
            )
    return state


def _merge(original: Any, patch: Any) -> Any:
    if not isinstance(patch, dict):
        return deepcopy(patch)
    result = deepcopy(original) if isinstance(original, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge(result.get(key), value)
    return result


def _check_patch(patch: dict[str, Any]) -> None:
    if {"schema_version", "revision"} & patch.keys():
        raise ValueError("schema_version and revision are CLI-owned fields")
    groups = [patch.get("artifacts", {})]
    configurations = patch.get("configurations") or {}
    if not isinstance(configurations, dict):
        raise ValueError("configurations must be a JSON object keyed by configuration ID")
    for config in configurations.values():
        if not isinstance(config, dict):
            continue
        if "acceptance" in config:
            raise ValueError("acceptance is CLI-owned; use --accept-profile after explicit user approval")
        groups.append(config.get("artifacts", {}))
    for group in groups:
        if isinstance(group, dict):
            for ref in group.values():
                if isinstance(ref, dict) and {"sha256", "context_sha256"} & ref.keys():
                    raise ValueError("artifact sha256 and context_sha256 are captured by the CLI")


def _canonical_request(config: Configuration, *, require_complete: bool = False) -> dict[str, Any]:
    try:
        request = SupportRequest.model_validate(config.draft_request)
        if request.fpm_profile is None:
            raise ValueError("acceptance requires a complete draft_request with fpm_profile")
        request.scheduler_limits()
        if require_complete:
            from .runtime import verify_runtime_profile

            verify_runtime_profile(request)
        return request.model_dump(mode="json", exclude_none=True)
    except ValueError:
        if require_complete:
            raise
        return config.draft_request


def _source_refs(artifacts: dict[str, Artifact]) -> dict[str, Any]:
    return {
        name: {"path": ref.path, "sha256": ref.sha256, "kind": ref.kind}
        for name, ref in artifacts.items()
        if ref.scope == "input" and not ref.archived
    }


def _context(state: Checkpoint, config: Configuration, scope: str) -> str:
    request = deepcopy(_canonical_request(config))
    if scope == "collection":
        request.pop("workload", None)
        if isinstance(request.get("search"), dict):
            for key in ("objective", "seed"):
                request["search"].pop(key, None)
    value: dict[str, Any] = {
        "inputs": state.inputs,
        "configuration_inputs": config.inputs,
        "source_artifacts": _source_refs(state.artifacts),
        "configuration_source_artifacts": _source_refs(config.artifacts),
        "request": request,
    }
    if scope == "validation":
        value.update(
            validation_inputs=state.validation_inputs, configuration_validation_inputs=config.validation_inputs
        )
    return _digest(value)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if "\x00" in value:
        raise ValueError("artifact paths must contain no NUL characters")
    return (path if path.is_absolute() else base / path).resolve()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _capture_refs(state: Checkpoint, old: Checkpoint, target: Path) -> None:
    groups = [(state.artifacts, old.artifacts)] + [
        (config.artifacts, old.configurations[name].artifacts if name in old.configurations else {})
        for name, config in state.configurations.items()
    ]
    for refs, old_refs in groups:
        for name, ref in refs.items():
            path = _resolve(target.parent, ref.path)
            if path in {target, _lock_path(target)}:
                raise ValueError("artifact path collides with the session checkpoint or its lock")
            ref.path = os.path.relpath(path, target.parent)
            previous = old_refs.get(name)
            same_path = previous is not None and _resolve(target.parent, previous.path) == path
            if same_path:
                if (ref.kind, ref.scope) != (previous.kind, previous.scope):
                    raise ValueError("remove an artifact reference before changing its kind or scope")
                ref.sha256 = previous.sha256
                ref.context_sha256 = previous.context_sha256
                continue
            if not path.is_file():
                raise ValueError(f"new artifact must reference an existing file: {path}")
            ref.sha256 = None if ref.kind == "collector_checkpoint" else _file_digest(path)
            ref.context_sha256 = None
    for ref in state.artifacts.values():
        if ref.scope != "input":
            raise ValueError(
                "shared artifacts must use scope=input; put collection/validation files in a configuration"
            )
        if ref.kind == "collector_checkpoint":
            raise ValueError("mutable collector checkpoints belong to a configuration with scope=collection")
    for config in state.configurations.values():
        for ref in config.artifacts.values():
            if ref.kind == "collector_checkpoint" and ref.scope != "collection":
                raise ValueError("mutable collector checkpoints require scope=collection")
            if ref.scope != "input" and ref.context_sha256 is None:
                ref.context_sha256 = _context(state, config, ref.scope)


def _artifact_issues(ref: Artifact, state: Checkpoint, config: Configuration | None, target: Path) -> list[str]:
    if ref.archived:
        return []
    path = _resolve(target.parent, ref.path)
    issues = []
    if path in {target, _lock_path(target)}:
        return ["reference collides with checkpoint or lock"]
    if config is not None and ref.scope != "input" and ref.context_sha256 != _context(state, config, ref.scope):
        issues.append("stale: inputs or draft changed; archive this superseded reference and register current output")
    try:
        if not path.is_file():
            issues.append(f"missing file: {path}")
        elif ref.kind == "collector_checkpoint":
            _read_json(path.read_text(encoding="utf-8"))
        elif ref.sha256 is None or _file_digest(path) != ref.sha256:
            issues.append("file content differs from the saved SHA-256 snapshot")
        elif config is not None and ref.kind in {"request", "profile"}:
            expected = _canonical_request(config, require_complete=True)
            actual = load_yaml(path)
            if ref.kind == "request":
                actual = SupportRequest.model_validate(actual).model_dump(mode="json", exclude_none=True)
            else:
                from aisimulate.fpm_profile import FpmModelProfile

                actual = FpmModelProfile.model_validate(actual).model_dump(mode="json", exclude_none=True)
                expected = expected["fpm_profile"]
            if _digest(actual) != _digest(expected):
                issues.append(f"saved {ref.kind} does not match the current reviewed draft")
    except (OSError, ValueError) as exc:
        issues.append(f"cannot verify artifact: {exc}")
    return issues


def report(state: Checkpoint, target: Path) -> dict[str, Any]:
    issues = []
    shared_invalid = False
    for name, ref in state.artifacts.items():
        for detail in _artifact_issues(ref, state, None, target):
            issues.append({"configuration": None, "artifact": name, "detail": detail})
            shared_invalid = True
    configurations = {}
    for name, config in state.configurations.items():
        local_issues = []
        input_invalid = shared_invalid
        for artifact, ref in config.artifacts.items():
            for detail in _artifact_issues(ref, state, config, target):
                local_issues.append({"configuration": name, "artifact": artifact, "detail": detail})
                input_invalid |= ref.scope == "input"
        accepted = config.acceptance is not None
        if accepted:
            try:
                _canonical_request(config, require_complete=True)
                from .runtime import verify_checkpoint_runtime

                verify_checkpoint_runtime(state, name, target)
                if config.acceptance.sha256 != _context(state, config, "acceptance"):
                    raise ValueError("accepted inputs or draft no longer match; review and accept again")
            except (OSError, ValueError) as exc:
                accepted = False
                local_issues.append({"configuration": name, "artifact": None, "detail": str(exc)})
        accepted &= not input_invalid
        needs_attention = input_invalid or bool(local_issues)
        configurations[name] = {
            "profile_accepted": accepted,
            "effective_status": "needs_attention" if needs_attention else "accepted" if accepted else "draft",
            "reported_progress": config.progress.model_dump(mode="json"),
            "next_action": "Resolve the integrity issues and review affected inputs before continuing."
            if needs_attention
            else config.progress.next_action
            or (
                "Plan collection from the accepted draft." if accepted else "Continue investigation and profile review."
            ),
        }
        issues.extend(local_issues)
    return {
        "checkpoint": str(target),
        "state": state.model_dump(mode="json"),
        "configurations": configurations,
        "integrity_issues": issues,
        "archived_artifacts": [
            {"configuration": owner, "artifact": name, "path": ref.path}
            for owner, refs in [(None, state.artifacts)]
            + [(name, config.artifacts) for name, config in state.configurations.items()]
            for name, ref in refs.items()
            if ref.archived
        ],
        "verification": {
            "scope": (
                "Canonical profile acceptance and current file integrity only; stage completion is agent-reported."
            ),
            "archived_artifacts": "Historical references only; excluded from current file and semantic verification.",
            "collector_checkpoints": (
                "Existence and JSON readability only; the collector verifies frozen-plan identity on actual resume."
            ),
            "executed": False,
        },
    }


def _lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.lock")


def _target(value: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("checkpoint path must not be a symlink")
    path = path.parent.resolve() / path.name
    if path.exists() and not path.is_file():
        raise ValueError("checkpoint path must be a file")
    return path


def _atomic_save(state: Checkpoint, target: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(
    target: Path, *, patch: dict[str, Any] | None, expected_revision: int | None, accept: list[str]
) -> tuple[Checkpoint, bool]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(_lock_path(target), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "another checkpoint operation is writing this session; read the latest revision and retry"
            ) from exc
        _target(str(target))
        exists = target.exists()
        old = _load(target) if exists else Checkpoint()
        if exists and patch is None and not accept:
            return old, False
        if (exists and expected_revision is None) or (
            expected_revision is not None and expected_revision != old.revision
        ):
            raise ValueError(
                f"checkpoint revision is {old.revision}; retry after reading it with --expect-revision {old.revision}"
            )
        patch = patch or {}
        _check_patch(patch)
        state = Checkpoint.model_validate(_merge(old.model_dump(mode="json"), patch))
        state.revision = old.revision + 1
        for name, config in state.configurations.items():
            previous = old.configurations.get(name)
            if (
                previous is not None
                and previous.draft_request != config.draft_request
                and previous.draft_request.get("fpm_profile") is not None
            ):
                from .runtime import preserve_runtime_overrides

                config.draft_request = preserve_runtime_overrides(previous.draft_request, config.draft_request)
        _capture_refs(state, old, target)
        for name, config in state.configurations.items():
            previous = old.configurations.get(name)
            if previous is None:
                continue
            changed = _context(state, config, "acceptance") != _context(old, previous, "acceptance")
            collection_changed = _context(state, config, "collection") != _context(old, previous, "collection")
            validation_changed = _context(state, config, "validation") != _context(old, previous, "validation")
            draft_changed = previous.draft_request != config.draft_request
            if draft_changed or (changed and previous.acceptance is not None):
                config.history.append(
                    {
                        "event": "draft_replaced" if draft_changed else "acceptance_invalidated",
                        "revision": old.revision,
                        "draft_request": deepcopy(previous.draft_request),
                        "inputs": deepcopy(previous.inputs),
                        "shared_inputs": deepcopy(old.inputs),
                        "acceptance": previous.acceptance.model_dump(mode="json") if previous.acceptance else None,
                        "artifacts": {key: ref.model_dump(mode="json") for key, ref in previous.artifacts.items()},
                        "shared_artifacts": {key: ref.model_dump(mode="json") for key, ref in old.artifacts.items()},
                    }
                )
            if changed:
                config.acceptance = None
                progress_patch = patch.get("configurations", {}).get(name, {}).get("progress")
                explicit_blockers = isinstance(progress_patch, dict) and progress_patch.get("status") == "blocked"
                if config.progress.stage >= 3 and not explicit_blockers:
                    config.progress = Progress(stage=3, next_action="Review and accept the changed draft.")
            elif validation_changed and not collection_changed and config.progress.stage >= 6:
                config.progress = Progress(stage=6, next_action="Repeat validation with the changed validation inputs.")
        for name in accept:
            if name not in state.configurations:
                raise ValueError(f"unknown configuration {name!r}")
            config = state.configurations[name]
            _canonical_request(config, require_complete=True)
            from .runtime import verify_checkpoint_runtime

            verify_checkpoint_runtime(state, name, target)
            source_refs = [(ref, None) for ref in state.artifacts.values()] + [
                (ref, config) for ref in config.artifacts.values() if ref.scope == "input"
            ]
            if any(_artifact_issues(ref, state, owner, target) for ref, owner in source_refs):
                raise ValueError(f"cannot accept {name!r}: input artifacts failed integrity checks")
            config.acceptance = Acceptance(sha256=_context(state, config, "acceptance"), revision=state.revision)
        _atomic_save(state, target)
        return state, True


def run_checkpoint_command(args: argparse.Namespace) -> int:
    if args.support_action == "resume":
        target = _target(args.checkpoint)
        output = report(_load(target), target)
        output["saved"] = False
    else:
        import sys

        target = _target(args.file)
        if (
            args.update
            and args.update != "-"
            and Path(args.update).expanduser().resolve() in {target, _lock_path(target)}
        ):
            raise ValueError("update file must be separate from the checkpoint and its lock")
        patch = None
        if args.update:
            patch = _read_json(
                sys.stdin.read() if args.update == "-" else Path(args.update).expanduser().read_text(encoding="utf-8")
            )
        state, saved = save_checkpoint(
            target, patch=patch, expected_revision=args.expect_revision, accept=args.accept_profile
        )
        output = report(state, target)
        output["saved"] = saved
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 2 if output["integrity_issues"] else 0
