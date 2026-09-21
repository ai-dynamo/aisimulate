# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the retained, qualified observed-MoE publisher without editing it.

The historical collection event identifies executed historical code. Current
replay orchestration and registry support are separately recorded in the result;
neither generated GPU timings. No collector import may leave the private package.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_QUALIFIED = {
    "v1": (
        "collector.sglang_rubin.publish_observed_moe",
        "sha256:067ad7797474518eab028911d4f0d6f314e1dd0456b08be5bae45de267f3e332",
    ),
    "v2": (
        "collector.sglang_rubin.publish_observed_moe_v2",
        "sha256:bbe3ebf2d450053f524d38b0a8ef97f0e55df000b6c6f61430b0207afc622eaf",
    ),
}
_V1_FILES = frozenset(
    {
        "collector/capabilities.py",
        "collector/case_generator.py",
        "collector/helper.py",
        "collector/model_cases.py",
        "collector/provenance.py",
        "collector/sglang_rubin/observed_moe_identity.json",
        "collector/sglang_rubin/publish_observed_moe.py",
        "collector/version_resolver.py",
    }
)
# These files support provenance.load_closures' registry enumeration. They do
# not replace any file in the approved historical publisher closure.
_SUPPORT_FILES = (
    "collector/hash_closures.yaml",
    "collector/framework_manifest.py",
    "collector/op_catalog.py",
    "collector/registry_types.py",
    "collector/sglang/registry.py",
    "collector/sglang_rubin/registry.py",
    "collector/trtllm/registry.py",
    "collector/vllm/registry.py",
    "collector/wideep/sglang/registry.py",
    "collector/wideep/trtllm/registry.py",
    "collector/wideep/vllm/registry.py",
)


def source_files(version):
    """Exact retained source set, including the identity files, for one profile."""
    if version not in _QUALIFIED:
        raise ValueError("Unknown frozen publisher profile")
    if version == "v1":
        return _V1_FILES
    return _V1_FILES | {
        "collector/sglang_rubin/publish_observed_moe_v2.py",
        "collector/sglang_rubin/observed_moe_v2_identity.json",
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record(path):
    _require(path.is_file() and not path.is_symlink(), f"Expected regular publisher source: {path}")
    content = path.read_bytes()
    return {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def _inventory(root):
    _require(root.is_dir() and not root.is_symlink(), "Unsafe publisher source directory")
    files = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"Symlink in publisher source: {path}")
        if not path.is_dir():
            files[str(path.relative_to(root))] = _record(path)
    return files


def _qualified_sources(root, version):
    files = _inventory(root)
    _require(files.keys() == source_files(version), "Incomplete or extra frozen publisher source files")
    digest = hashlib.sha256()
    for name in sorted(files):
        content = (root / name).read_bytes()
        _require(
            {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)} == files[name],
            "Frozen publisher source changed during validation",
        )
        digest.update(name.encode() + b"\0" + content + b"\0")
    _require(f"sha256:{digest.hexdigest()}" == _QUALIFIED[version][1], "Unapproved frozen publisher source")
    return files


class _CollectorImports(importlib.abc.MetaPathFinder):
    """Resolve collector modules only inside the explicit staged source set."""

    def __init__(self, package):
        self.package = package.resolve()
        self.files = _inventory(package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "collector" and not fullname.startswith("collector."):
            return None
        search = [str(self.package)] if fullname == "collector" else list(path or ())
        error = f"Collector import outside the staged publisher: {fullname}"
        if not search or any(not Path(p).resolve().is_relative_to(self.package) for p in search):
            raise ModuleNotFoundError(error)
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ModuleNotFoundError(error)
        if spec.origin is not None:
            origin = Path(spec.origin).resolve()
            if not origin.is_relative_to(self.package) or str(origin.relative_to(self.package)) not in self.files:
                raise ModuleNotFoundError(error)
        elif any(not Path(p).resolve().is_relative_to(self.package) for p in spec.submodule_search_locations or ()):
            raise ModuleNotFoundError(error)
        return spec


def _worker(request_path, result_path):
    request = json.loads(request_path.read_text())
    package = (request_path.parent / "package").resolve()
    expected = request["staged_sources"]
    _require(_inventory(package) == expected, "Staged publisher source changed")
    _require(not any(n == "collector" or n.startswith("collector.") for n in sys.modules), "Collector preloaded")
    sys.meta_path.insert(0, _CollectorImports(package))
    module, approved_hash = _QUALIFIED[request["version"]]
    provenance = importlib.import_module("collector.provenance")
    closures = provenance.load_closures(package / "collector/hash_closures.yaml")
    declared = {module.replace(".", "/") + ".py", *provenance.SHARED_CORE, *closures[module]}
    _require(declared == source_files(request["version"]), "Historical publisher closure declaration changed")
    _require(provenance.collector_hash(module, package, closures) == approved_hash, "Staged publisher hash changed")
    publisher = importlib.import_module(module)
    result = publisher.publish(**request["arguments"])
    _require(_inventory(package) == expected, "Executed publisher source changed")
    if not request["arguments"]["validate_only"]:
        _require(
            result["event"]["collector_hash"] == approved_hash, "Published event does not identify executed source"
        )
    imports = {
        name: str(Path(value.__file__).relative_to(package))
        for name, value in sys.modules.items()
        if (name == "collector" or name.startswith("collector.")) and getattr(value, "__file__", None)
    }
    _require({module, "collector.helper", "collector.provenance"} <= imports.keys(), "Publisher imports incomplete")
    result["replay"] = {
        "role": "CPU replay of qualified publication; no GPU measurement or provenance relabeling",
        "qualified_collector_hash": approved_hash,
        "qualified_sources": {name: expected[name] for name in sorted(source_files(request["version"]))},
        "current_support_sources": {name: expected[name] for name in _SUPPORT_FILES},
        "current_launcher": _record(Path(__file__)),
        "collector_imports": imports,
        "python_version": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "pyyaml")},
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def replay(*, version, publisher_source, **arguments):
    """Replay one fixed profile using its unmodified retained publisher sources."""
    _require(
        publisher_source is not None,
        "This fixed profile requires publisher_source (CLI: --publisher-source) pointing to its retained "
        "qualified evidence/publisher-source directory; current source cannot claim the historical collector hash",
    )
    source = Path(publisher_source)
    qualified = _qualified_sources(source, version)
    current_package = Path(__file__).resolve().parents[2]
    support = {name: _record(current_package / name) for name in _SUPPORT_FILES}
    launcher = _record(Path(__file__))
    entrypoint = current_package / (_QUALIFIED[version][0].replace(".", "/") + ".py")
    entrypoint_record = _record(entrypoint)
    # The worker changes cwd; preserve caller symlinks and lexical path components
    # so the historical publisher applies its original path checks.
    caller_cwd = os.getcwd()
    args = {key: os.path.join(caller_cwd, value) for key, value in arguments.items() if key != "validate_only"}
    args["validate_only"] = arguments.get("validate_only", False)
    _require(type(args["validate_only"]) is bool, "validate_only must be boolean")
    with tempfile.TemporaryDirectory(prefix="aisim-observed-moe-replay-") as temporary:
        work = Path(temporary)
        for root, records in ((source, qualified), (current_package, support)):
            for name, record in records.items():
                target = work / "package" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
                _require(_record(target) == record, "Publisher source changed during staging")
        runner = work / "replay.py"
        shutil.copyfile(__file__, runner)
        _require(_record(runner) == launcher, "Replay launcher changed during staging")
        request = work / "request.json"
        request.write_text(json.dumps({"version": version, "arguments": args, "staged_sources": qualified | support}))
        result_path = work / "result.json"
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(runner), str(request), str(result_path)],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError(f"Qualified publisher replay failed:\n{process.stderr}")
        result = json.loads(result_path.read_text())
        _require(_qualified_sources(source, version) == qualified, "Original publisher source changed during replay")
        _require(_record(Path(__file__)) == launcher == result["replay"]["current_launcher"], "Replay launcher changed")
        _require(
            {name: _record(current_package / name) for name in _SUPPORT_FILES} == support,
            "Current replay support changed",
        )
        _require(_record(entrypoint) == entrypoint_record, "Current replay entrypoint changed")
        result["replay"]["current_entrypoint"] = {str(entrypoint.relative_to(current_package)): entrypoint_record}
    return result


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
