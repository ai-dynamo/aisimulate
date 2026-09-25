# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read a shared graph receipt once within one complete ownership validation."""

import hashlib
import json
import os
import stat

from collector.glm53flash_graph_export import _local


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class ReceiptCache:
    """Scoped parsed receipts, with exact reference and final content checks.

    This cache never spans reader calls. Callers must verify it before returning
    accepted evidence, and must treat the parsed objects as read-only.
    """

    def __init__(self, root, files):
        self.root, self.files, self.entries = root, files, {}

    def read(self, reference):
        path = _local(self.root, reference["file"])
        digest = reference.get("sha256")
        if path.name in self.entries:
            previous, identity, value = self.entries[path.name]
            if digest != previous or _identity(path.stat()) != identity:
                raise ValueError("shared native receipt identity changed")
            return value
        with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("shared native receipt is not a regular file")
            raw = stream.read()
            identity = _identity(before)
            if identity != _identity(os.fstat(stream.fileno())) or identity != _identity(path.stat()):
                raise ValueError("shared native receipt changed during read")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("native graph evidence hash changed")
        value = json.loads(raw)
        self.entries[path.name] = digest, identity, value
        self.files.add(path.name)
        return value

    def verify(self):
        for name, (digest, identity, _) in self.entries.items():
            path = _local(self.root, name)
            with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as stream:
                if _identity(os.fstat(stream.fileno())) != identity:
                    raise ValueError("shared native receipt changed before final verification")
                current = hashlib.file_digest(stream, "sha256").hexdigest()
                if (
                    current != digest
                    or _identity(os.fstat(stream.fileno())) != identity
                    or _identity(path.stat()) != identity
                ):
                    raise ValueError("shared native receipt changed during validation")
