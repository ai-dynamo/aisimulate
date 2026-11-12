# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import selected runtime adapters from verified frozen source, including on spawn.

Only manifest-declared modules use this loader. Runtime dependencies retain normal
Python imports; this verifies provenance, and is not a sandbox for custom code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path, PurePosixPath

BINDING_SCHEMA = "aisimulate-runtime-instrumentation-imports/v1"
_bundle = None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _Bundle(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self):
        context = json.loads(Path(os.environ["AISIMULATE_RUNTIME_CONTEXT"]).read_text())
        self.manifest_path = Path(os.environ["AISIMULATE_RUNTIME_INSTRUMENTATION"]).resolve()
        self.root = self.manifest_path.parent
        self.directory = Path(os.environ["AISIMULATE_RUNTIME_OBSERVATION_DIR"])
        self.path = self.directory / f"runtime-instrumentation-imports-{socket.gethostname()}-{os.getpid()}.json"
        self.record = {
            "schema_version": BINDING_SCHEMA,
            **{key: context[key] for key in ("attempt_id", "configuration", "phase")},
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "root": str(self.root),
            "status": "verifying",
            "expected_bundle_sha256": context["bundle_sha256"],
            "loader_sha256": _digest(Path(__file__).read_bytes()),
            "imports": {},
            "classes": {},
        }
        try:
            self.manifest = json.loads(self.manifest_path.read_text())
            self.files = self._files()
            self.record.update(files=self.files, bundle_sha256=self._identity(self.files))
            if self.record["bundle_sha256"] != context["bundle_sha256"]:
                raise ValueError("runtime instrumentation bundle hash mismatch")
            if context.get("instrumentation_binding") != {
                "schema_version": BINDING_SCHEMA,
                "loader_sha256": self.record["loader_sha256"],
            }:
                raise ValueError("runtime instrumentation loader identity mismatch")
            self.modules = {}
            for name in self.files:
                if name.endswith(".py"):
                    parts = list(PurePosixPath(name).with_suffix("").parts)
                    if parts[-1] == "__init__":
                        parts.pop()
                    self.modules[".".join(parts)] = name
            for name in self.modules:
                if name in sys.modules:
                    raise ValueError(f"instrumentation module already imported outside verified bundle loading: {name}")
            sys.meta_path.insert(0, self)
        except Exception as error:
            self.fail(error)
            raise

    def _files(self):
        files = {}
        for name in self.manifest["files"]:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
                raise ValueError("unsafe runtime instrumentation path")
            source = self.root / name
            if not source.resolve().is_relative_to(self.root) or source.is_symlink():
                raise ValueError("runtime instrumentation source escaped the frozen bundle")
            files[name] = _digest(source.read_bytes())
        return files

    def _identity(self, files):
        return _digest(_json({"manifest": self.manifest, "files": files}).encode())

    def write(self):
        if self.record["pid"] != os.getpid():
            # Forked workers inherit the verified modules, not the parent's
            # process identity or receipt file. Do not claim a fresh import.
            self.record["inherited_from_pid"] = self.record["pid"]
            self.record.update(pid=os.getpid(), hostname=socket.gethostname())
            self.path = self.directory / f"runtime-instrumentation-imports-{socket.gethostname()}-{os.getpid()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(_json(self.record) + "\n")
        temporary.replace(self.path)

    def fail(self, error):
        self.record.update(status="failed", error=f"{type(error).__name__}: {error}")
        self.write()

    def find_spec(self, fullname, path=None, target=None):
        name = self.modules.get(fullname)
        if name is None:
            return None
        source = self.root / name
        package = [] if source.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(fullname, source, loader=self, submodule_search_locations=package)

    def create_module(self, spec):
        return None

    def get_resource_reader(self, fullname):
        # Use Python's ordinary package-data interface; source execution still
        # passes through exec_module and all declared files remain hash-checked.
        source = self.root / self.modules[fullname]
        return importlib.machinery.SourceFileLoader(fullname, str(source)).get_resource_reader(fullname)

    def exec_module(self, module):
        name = self.modules[module.__name__]
        source = self.root / name
        try:
            raw = source.read_bytes()
            digest = _digest(raw)
            if digest != self.files[name]:
                raise ValueError(f"runtime instrumentation source changed: {name}")
            # Execute the bytes that were hashed, never an unrelated cached pyc.
            exec(compile(raw, str(source), "exec"), module.__dict__)
            self.record["imports"][module.__name__] = {"file": name, "path": str(source), "sha256": digest}
            self.write()
        except Exception as error:
            self.fail(error)
            raise

    def checked_record(self):
        try:
            if self.record["status"] == "failed":
                raise ValueError(self.record["error"])
            if json.loads(self.manifest_path.read_text()) != self.manifest or self._files() != self.files:
                raise ValueError("runtime instrumentation bundle changed after import")
            for name, evidence in self.record["imports"].items():
                module = sys.modules.get(name)
                if module is None or Path(module.__file__).resolve() != Path(evidence["path"]):
                    raise ValueError(f"runtime instrumentation imported module origin changed: {name}")
            self.write()
            return json.loads(_json(self.record))
        except Exception as error:
            self.fail(error)
            raise


def observed_binding():
    """Read actual imports after hooks initialize; declarations alone cannot pass."""
    if _bundle is None:
        raise ValueError("runtime instrumentation did not load through its verified entrypoint")
    return _bundle.checked_record()


def __getattr__(name):
    global _bundle
    field = {"ObservedWorker": "worker_class", "ObservedInstrumentedScheduler": "scheduler_class"}.get(name)
    if field is None:
        raise AttributeError(name)
    if _bundle is None:
        _bundle = _Bundle()
    try:
        module_name, _, class_name = _bundle.manifest[field].rpartition(".")
        module = importlib.import_module(module_name)
        actual = getattr(module, class_name)
        if not isinstance(actual, type) or actual.__module__ not in _bundle.record["imports"]:
            raise ValueError("instrumentation class must be defined by an imported frozen module")
        # An empty subclass preserves native methods and isinstance checks. Its
        # stable module/name also sends spawned-process unpickling through here.
        selected = type(name, (actual,), {"__module__": __name__})
        _bundle.record["classes"][field] = _bundle.manifest[field]
        _bundle.record["status"] = "verified"
        _bundle.checked_record()
        globals()[name] = selected
        return selected
    except Exception as error:
        _bundle.fail(error)
        raise
