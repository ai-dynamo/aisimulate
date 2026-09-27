# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``python -m aisimulate_core.fpm_hooks <module> [-- args...]``: run ``<module>`` as ``__main__``
with the FPM per-request hooks installed here and in every subprocess (via PYTHONPATH)."""

from __future__ import annotations

import os
import runpy
import sys

from . import hook_path, install


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if args else 2
    module = args.pop(0)
    if args and args[0] == "--":
        args.pop(0)
    path = hook_path()
    existing = os.environ.get("PYTHONPATH", "")
    if path not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = path + (os.pathsep + existing if existing else "")
    if path not in sys.path:
        sys.path.insert(0, path)
    install()
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
