# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``perf-model`` command: check, convert and fetch performance model inputs.

``check`` builds the performance model of every entry of an ``ais_perf_model``
value and reports whether it builds and whether it can estimate yet.
``migrate`` converts the old estimator fields of a config file to
``ais_perf_model``. ``fetch`` downloads a published FPM profile and prints the
folder to name in ``systems_paths``.

Exit codes: 0 when everything is fine, 1 when the command found a problem with
the input (an entry that does not build, a field that cannot be converted, a
profile that cannot be fetched), 2 for a wrong command line, an input that
cannot be read or an output file that cannot be written.

The argument parser is built on every ``aisimulate`` call, so this module
imports only the standard library. Each action imports what it needs when it
runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

_FORMATS = ("table", "json")
_KINDS = ("predict", "recommend", "search")
_SECTION_BY_KIND = {"predict": "engine", "recommend": "engine", "search": "search_space"}
_NOT_SET = "(not set)"


# --- Parser ----------------------------------------------------------------------------------


def add_perf_model_parser(subparsers: Any) -> None:
    """Add the ``perf-model`` command and its actions to the ``aisimulate`` parser."""
    command = subparsers.add_parser(
        "perf-model",
        help="Check, convert and fetch performance model inputs.",
        description=(
            "Work with ais_perf_model, the one input that names a performance model. "
            "check builds the model of every entry, migrate converts the old estimator fields "
            "of a config file, and fetch downloads a published FPM profile."
        ),
    )
    actions = command.add_subparsers(dest="perf_model_action", required=True)
    _add_check_parser(actions)
    _add_migrate_parser(actions)
    _add_fetch_parser(actions)


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=_FORMATS, default="table", help="How to print the result (default: table).")


def _add_check_parser(actions: Any) -> None:
    check = actions.add_parser(
        "check",
        help="Build the performance model of every entry and report whether it builds.",
        description=(
            "Load an ais_perf_model value, resolve every entry and build its performance model. "
            "Each entry is reported with its worker type, its estimation_mode and ok or the error. "
            "A model that is built but cannot estimate yet is reported as 'ok, not ready' with the "
            "stores that still need observations. "
            "The exit code is 1 when an entry does not build."
        ),
    )
    check.add_argument("value", metavar="VALUE", help="An ais_perf_model value: a YAML or JSON file, or inline JSON.")
    check.add_argument(
        "--worker-type",
        dest="worker_types",
        action="append",
        default=[],
        metavar="TYPE",
        help=(
            "Check only this worker type (prefill, decode or aggregated); repeat the option for several. "
            "Required when VALUE is one config that does not write worker_type."
        ),
    )
    check.add_argument(
        "--fill",
        dest="fill",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "A value the serving component knows by itself, such as model=NAME; VALUE is read as YAML. "
            "A key the entry does not write is added, and a key it writes must be equal. "
            "Repeat the option for several keys."
        ),
    )
    check.add_argument(
        "--workspace",
        metavar="DIR",
        help="Folder in which an entry that names a dataset is prepared (default: a temporary folder).",
    )
    _add_format(check)


def _add_migrate_parser(actions: Any) -> None:
    migrate = actions.add_parser(
        "migrate",
        help="Convert the old estimator fields of a config file to ais_perf_model.",
        description=(
            "Read a predict, recommend or search config file and move its old estimator fields "
            "(estimation_mode, fallback_policy, estimator_config, systems_paths, database_mode, "
            "transfer_policy, fpm_profile and their legacy spellings) into engine.ais_perf_model or "
            "search_space.ais_perf_model. Without --output the plan is printed and nothing is written. "
            "The conversion is all or nothing: when one field cannot be converted, nothing is converted "
            "and the exit code is 1. "
            "The old fields start a relative path at the working directory and ais_perf_model starts it "
            "at the folder of the config file, so run this command from the directory the config is run "
            "from. A relative path is rewritten to start at the folder of FILE, or of CONFIG without "
            "--output. "
            "Do not convert a replay config that 'aisimulate onboard' wrote: onboard compares that file "
            "with the workload it reviewed and rejects the converted file."
        ),
    )
    migrate.add_argument("config", metavar="CONFIG", help="A predict, recommend or search config file.")
    migrate.add_argument(
        "--kind",
        choices=_KINDS,
        help=(
            "What kind of config the file is. By default it is read from the file: a top-level search_space "
            "is a search config, an engine section next to optimization or optimizer is a recommend config, "
            "and any other file with an engine section is a predict config."
        ),
    )
    migrate.add_argument(
        "--output",
        metavar="FILE",
        help=(
            "Write the converted config to FILE: as JSON when FILE ends with .json, as YAML otherwise. "
            "FILE may be CONFIG itself; another file that exists is replaced only with --overwrite. "
            "A write that fails leaves a file that exists as it was. "
            "Comments and formatting of the original file are not kept."
        ),
    )
    migrate.add_argument(
        "--overwrite",
        action="store_true",
        help="Let --output replace a file that exists and is not CONFIG itself.",
    )
    _add_format(migrate)


def _add_fetch_parser(actions: Any) -> None:
    fetch = actions.add_parser(
        "fetch",
        help="Download a published FPM profile and print the folder to name in systems_paths.",
        description=(
            "Download the files of one FPM profile that a dataset manifest pins, verify them and print "
            "the local folder together with the systems_paths line for an ais_perf_model entry. "
            "A profile that is already in the cache is not downloaded again. "
            "With --list, print the profiles of MANIFEST, or of every manifest shipped with this "
            "package when MANIFEST is left out; nothing is downloaded."
        ),
    )
    fetch.add_argument(
        "manifest",
        nargs="?",
        metavar="MANIFEST",
        help="A dataset manifest: a local file or the ID of a manifest shipped with this package.",
    )
    fetch.add_argument("profile", nargs="?", metavar="PROFILE", help="The name of the profile to fetch.")
    fetch.add_argument(
        "--list", dest="list_profiles", action="store_true", help="List profiles instead of fetching one."
    )
    fetch.add_argument(
        "--cache-dir", metavar="DIR", help="Folder that holds fetched profiles (default: the user cache folder)."
    )
    fetch.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download; fail when the profile is not in the cache yet.",
    )
    fetch.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="Also fetch a profile that is not admitted for serving, to reproduce an earlier result.",
    )
    _add_format(fetch)


# --- Output ----------------------------------------------------------------------------------


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _print_columns(rows: Sequence[Sequence[str]]) -> None:
    """Print rows as left-aligned columns; the last column is not padded."""
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]) - 1)]
    for row in rows:
        padded = [cell.ljust(width) for cell, width in zip(row, widths, strict=False)]
        print("  ".join([*padded, row[-1]]))


def _yaml_text(value: Any) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


# --- check -----------------------------------------------------------------------------------


def _fill_values(items: Sequence[str]) -> dict[str, Any]:
    """Read the ``--fill KEY=VALUE`` options into one mapping."""
    import yaml

    fill: dict[str, Any] = {}
    for item in items:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--fill takes KEY=VALUE, and {item!r} has no key or no '='")
        if key in fill:
            raise ValueError(f"--fill sets {key} more than once")
        try:
            fill[key] = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"--fill {key}: the value {text!r} is not valid YAML: {exc}") from exc
    return fill


def _entries_to_check(document: Any, requested: Sequence[str]) -> list[dict[str, Any]]:
    """Return the entries to build: the requested worker types, or every worker type the value names."""
    worker_types = list(dict.fromkeys(requested)) or list(document.worker_types)
    if not worker_types:
        raise ValueError(
            "the value is one config that does not write worker_type; "
            "name the worker types to check with --worker-type (prefill, decode or aggregated)"
        )
    return [document.select(worker_type) for worker_type in worker_types]


def _refuse_whole_config(document: Any) -> None:
    """Stop when the value is a predict, recommend or search config and not an ``ais_perf_model`` value."""
    if not document.bare:
        return
    [config] = document.entries
    # A search config is told by its search_space section, whatever else the file has.
    for section in ("search_space", "engine"):
        if section in config:
            raise ValueError(
                f"VALUE is a whole config: it has a top-level {section} section. "
                f"check takes the value of {section}.ais_perf_model, in a file of its own or as inline JSON"
            )


def _check_entry(entry: Mapping[str, Any], fill: Mapping[str, Any], workspace: str) -> dict[str, Any]:
    """Resolve one entry and build its performance model; report the outcome instead of raising."""
    from aisimulate_core.ais_perf_model import resolve

    result: dict[str, Any] = {
        "worker_type": entry["worker_type"],
        "estimation_mode": entry.get("estimation_mode"),
        "ok": False,
        "error": None,
        "config": None,
        "readiness": None,
    }
    try:
        config = resolve(entry, worker_type=entry["worker_type"], fill=fill, workspace=workspace)
        # The mode is reported as the config states it. Which estimator "auto" selects is not reported.
        result.update(config=config, estimation_mode=config.get("estimation_mode"))
        result["readiness"] = _readiness(config)
        result["ok"] = True
    except Exception as exc:
        # Whatever stops this entry is what the command reports; the other entries are still checked.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _readiness(config: Mapping[str, Any]) -> Any:
    """Build the performance model of a resolved config and return its readiness report."""
    from aisimulate_core.sdk import RustForwardPassPerfModel

    model = RustForwardPassPerfModel.best_available(dict(config))
    try:
        return model.readiness_report()
    finally:
        model.close()


def _run_check(args: argparse.Namespace) -> int:
    from aisimulate_core.ais_perf_model import load

    document = load(args.value)
    # Checked first: asking for --worker-type would send the user the wrong way.
    _refuse_whole_config(document)
    entries = _entries_to_check(document, args.worker_types)
    fill = _fill_values(args.fill)
    if args.workspace is not None:
        results = [_check_entry(entry, fill, args.workspace) for entry in entries]
    else:
        with tempfile.TemporaryDirectory(prefix="aisimulate-perf-model-") as workspace:
            results = [_check_entry(entry, fill, workspace) for entry in entries]
    ok = all(result["ok"] for result in results)
    if args.format == "json":
        _print_json({"ok": ok, "entries": results})
    else:
        rows = [("worker type", "estimation_mode", "result")]
        for result in results:
            mode = _NOT_SET if result["estimation_mode"] is None else str(result["estimation_mode"])
            rows.append((result["worker_type"], mode, _result_text(result)))
        _print_columns(rows)
    return 0 if ok else 1


def _result_text(result: Mapping[str, Any]) -> str:
    """The result cell of one entry: the error, ok, or ok with the stores that cannot estimate yet."""
    if not result["ok"]:
        return f"failed: {result['error']}"
    readiness = result["readiness"]
    if readiness["ready"]:
        return "ok"
    # A regression that has seen no data builds fine, so "ok" alone would hide that it cannot estimate.
    waiting = [
        f"{store['workload_kind']} ({store['not_ready_reason']})" for store in readiness["stores"] if not store["ready"]
    ]
    return f"ok, not ready: {', '.join(waiting)}"


# --- migrate ---------------------------------------------------------------------------------


def _config_kind(raw: Mapping[str, Any], path: str) -> str:
    """Tell a search, recommend or predict config by its top-level sections."""
    if "search_space" in raw:
        return "search"
    if "engine" in raw:
        # Only a recommend config has these two sections; predict rejects them.
        return "recommend" if "optimization" in raw or "optimizer" in raw else "predict"
    raise ValueError(
        f"cannot tell what kind of config {path} is: it has no top-level search_space or engine section; "
        "name the kind with --kind (predict, recommend or search)"
    )


def _load_config(path: str) -> dict[str, Any]:
    """Read the config file; a file that is not UTF-8 text is reported with its name."""
    from .config.common import load_yaml

    try:
        return load_yaml(path)
    except UnicodeDecodeError as exc:
        # load_yaml names the file for every other failure; this one would reach the user without it.
        raise ValueError(f"could not read configuration {path}: the file is not valid UTF-8 text ({exc})") from exc


def _refuse_to_replace_another_file(args: argparse.Namespace) -> None:
    """Stop before ``--output`` replaces a file that is not CONFIG itself, unless ``--overwrite`` allows it."""
    if args.output is None or args.overwrite:
        return
    # A name that is free is written; a folder or a device is reported by the write itself.
    if not os.path.isfile(args.output) or os.path.samefile(args.output, args.config):
        return
    raise ValueError(f"output file {args.output} exists; pass --overwrite to replace it")


def _config_text(path: str, config: Mapping[str, Any]) -> str:
    """The text to write to ``path``: JSON for a name that ends with ``.json``, YAML for any other."""
    return _json_text(path, config) if path.lower().endswith(".json") else _yaml_text(config)


def _json_text(path: str, config: Mapping[str, Any]) -> str:
    """The config as JSON text that reads back as the same config.

    A config file is read as YAML whatever its name is. YAML holds values that JSON does not, such
    as a date or a key that is not text, and it reads a number that JSON writes with an exponent
    as text. A config with such a value is refused: the file would mean something else.
    """
    import yaml

    advice = "name a file that does not end with .json to write YAML"
    try:
        text = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        read_back = yaml.safe_load(text)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot write {path} as JSON: {exc}; {advice}") from exc
    if read_back != config:
        raise ValueError(f"cannot write {path} as JSON: {_first_difference(config, read_back)}; {advice}")
    return text


def _first_difference(written: Any, read_back: Any, where: str = "") -> str:
    """Name the first value that is read back differently, by its keys, with both values."""
    if isinstance(written, dict) and isinstance(read_back, dict) and list(written) == list(read_back):
        pairs = [(f"{where}.{key}" if where else str(key), written[key], read_back[key]) for key in written]
    elif isinstance(written, list) and isinstance(read_back, list) and len(written) == len(read_back):
        pairs = [(f"{where}[{index}]", item, read_back[index]) for index, item in enumerate(written)]
    else:
        return f"{where or 'the config'} is {written!r} and would be read back from the file as {read_back!r}"
    return next(_first_difference(item, other, name) for name, item, other in pairs if item != other)


def _write_config(path: str, config: Mapping[str, Any]) -> None:
    """Write the converted config to ``path``; a write that fails leaves a file that exists as it was."""
    # The text is built before anything is touched, so a config that cannot be written as text changes nothing.
    text = _config_text(path, config)
    target = os.path.realpath(path)  # a config that is reached through a symlink stays a symlink
    if not _can_be_replaced(path, target):
        # A folder, a device, a read-only file or a folder that does not exist. Opening the path itself
        # reports the reason with the path as it was given, and it is the only way to write to a device.
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        return
    try:
        _replace_file(target, text)
    except OSError as exc:
        # The error of the system names the temporary file or no file at all.
        raise OSError(f"could not write {path}: {exc}") from exc


def _can_be_replaced(path: str, target: str) -> bool:
    """Whether ``path`` is written by putting a new file in the place of ``target``."""
    if os.path.exists(path):
        return os.path.isfile(path) and os.access(path, os.W_OK)
    return os.path.isdir(os.path.dirname(target))


def _replace_file(target: str, text: str) -> None:
    """Put ``text`` in ``target`` in one step: the file holds the old text or the new text, never a part."""
    # In the folder of the target, so that the move does not cross file systems. The name of the target
    # is not a part of the name: with it, a long name of the target is too long for a file name.
    temporary = os.path.join(os.path.dirname(target), f".{os.urandom(8).hex()}.tmp")
    created = False
    try:
        # Opening with "x" gives a new file the permissions of any new file (the umask applies); a file
        # that tempfile makes is readable by its owner only. It also never opens a file that exists.
        with open(temporary, "x", encoding="utf-8") as stream:
            created = True
            _keep_permissions(target, temporary)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if created:
            # After the move there is nothing to remove; before it the unfinished file must not stay behind.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def _keep_permissions(target: str, temporary: str) -> None:
    """Give the new file the permissions of the file it replaces, when there is one."""
    try:
        mode = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        return
    os.chmod(temporary, mode)


def _print_plan(report: Mapping[str, Any], key: str) -> None:
    print(f"config: {report['config']}")
    print(f"kind: {report['kind']}")
    print(f"working directory: {report['working_directory']} (relative paths in the old fields start here)")
    if report["status"] == "cannot_convert":
        for problem in report["problems"]:
            print(problem)
        print("Nothing was converted.")
        return
    if report["status"] == "nothing_to_convert":
        print("The config has no old estimator fields; there is nothing to convert.")
        return
    section, _, name = key.partition(".")
    print(f"\nnew key {key}:")
    print(_indented(_yaml_text({section: {name: report["ais_perf_model"]}})))
    print("\nremoved fields:")
    for path in report["removed"]:
        print(f"  {path}")
    if report["notes"]:
        print("\nnotes:")
        for note in report["notes"]:
            print(f"  {note}")
    if report["output"] is None:
        print("\nNothing was written. Pass --output FILE to write the converted config.")
    else:
        print(f"\nWrote {report['output']} (comments and formatting of the original file are not kept).")


def _indented(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.rstrip("\n").splitlines())


def _run_migrate(args: argparse.Namespace) -> int:
    from .config.ais_perf_model_migrate import CannotConvert, MigrationCheckError, migrate_mapping
    from .config.cli import config_folder

    raw = _load_config(args.config)
    _refuse_to_replace_another_file(args)
    kind = args.kind or _config_kind(raw, args.config)
    key = f"{_SECTION_BY_KIND[kind]}.ais_perf_model"
    report: dict[str, Any] = {
        "config": args.config,
        "kind": kind,
        "working_directory": os.getcwd(),
        "status": "converted",
        "ais_perf_model": None,
        "removed": [],
        "notes": [],
        "problems": [],
        "output": None,
    }
    # ais_perf_model starts a relative path at the folder of the file it is read from: FILE once it is written.
    read_from = args.config if args.output is None else args.output
    try:
        converted, removed, notes = migrate_mapping(
            raw, kind, config_dir=config_folder(read_from), cwd=report["working_directory"]
        )
    except CannotConvert as exc:
        report.update(status="cannot_convert", problems=list(exc.problems))
    except MigrationCheckError as exc:
        print(
            f"internal error: {exc}. Nothing was converted and nothing was written; "
            "please report this with the config file.",
            file=sys.stderr,
        )
        return 1
    else:
        if removed:
            report.update(
                ais_perf_model=converted[_SECTION_BY_KIND[kind]]["ais_perf_model"], removed=removed, notes=notes
            )
            if args.output is not None:
                _write_config(args.output, converted)
                report["output"] = args.output
        else:
            report["status"] = "nothing_to_convert"
    if args.format == "json":
        _print_json(report)
    else:
        _print_plan(report, key)
    return 1 if report["status"] == "cannot_convert" else 0


# --- fetch -----------------------------------------------------------------------------------


def _list_profiles(args: argparse.Namespace) -> int:
    from aisimulate_core.sdk.fpm_dataset import list_fpm_profiles

    if args.profile is not None:
        raise ValueError("--list takes at most MANIFEST; leave PROFILE out")
    profiles = list_fpm_profiles(args.manifest)
    if args.format == "json":
        _print_json(profiles)
    elif not profiles:
        print("No profiles.")
    else:
        rows = [("profile", "admission", "manifest")]
        rows.extend((record["profile"], record["admission"], record["manifest"]) for record in profiles)
        _print_columns(rows)
    return 0


def _known_profile(manifest: str, profile: str) -> Mapping[str, Any]:
    """Return what the manifest lists for the profile; a profile it does not have is a usage error."""
    from aisimulate_core.sdk.fpm_dataset import list_fpm_profiles

    records = {record["profile"]: record for record in list_fpm_profiles(manifest)}
    if profile not in records:
        raise ValueError(f"manifest {manifest} has no profile {profile!r}; it has: {', '.join(records) or 'none'}")
    return records[profile]


def _unqualified_hint(record: Mapping[str, Any], allow_unqualified: bool) -> str:
    """A line that names ``--allow-unqualified`` when the profile was refused as not admitted for serving."""
    # The profile store refuses such a profile before it looks at the cache or downloads, so without
    # the option this is the reason of the failure. The admission is read, not the text of the error.
    if allow_unqualified or record["admission"] == "serving":
        return ""
    # On a line of its own: the reason a manifest gives for the admission is one or more full sentences.
    return "\nPass --allow-unqualified to fetch the profile anyway."


def _run_fetch(args: argparse.Namespace) -> int:
    if args.list_profiles:
        return _list_profiles(args)
    if args.manifest is None or args.profile is None:
        raise ValueError("fetch takes MANIFEST and PROFILE; use --list to see the profiles")
    from aisimulate_core.sdk.fpm_dataset import materialize_fpm_profile

    record = _known_profile(args.manifest, args.profile)
    try:
        folder = materialize_fpm_profile(
            args.manifest,
            args.profile,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            allow_unqualified=args.allow_unqualified,
        )
    except (ValueError, OSError) as exc:
        # The manifest was read and the profile is known, so this is about the profile, not the command line.
        hint = _unqualified_hint(record, args.allow_unqualified)
        print(f"could not fetch profile {args.profile!r}: {exc}{hint}", file=sys.stderr)
        return 1
    path = os.fspath(folder)
    if args.format == "json":
        _print_json({"manifest": args.manifest, "profile": args.profile, "path": path, "systems_paths": [path]})
    else:
        print(f"profile {args.profile} is at {path}")
        print("Name the folder in the ais_perf_model entry that uses the profile:")
        print(f"  systems_paths: [{json.dumps(path)}]")
    return 0


# --- Entry point -----------------------------------------------------------------------------

_ACTIONS = {"check": _run_check, "migrate": _run_migrate, "fetch": _run_fetch}


def run_perf_model_command(args: argparse.Namespace) -> int:
    """Run one ``perf-model`` action and return the exit code.

    A wrong command line, an input that cannot be read or an output file that
    cannot be written is a ``ValueError`` or an ``OSError``; the caller reports
    it as a usage error.
    """
    return _ACTIONS[args.perf_model_action](args)
