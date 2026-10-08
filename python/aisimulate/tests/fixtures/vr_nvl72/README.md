# Original Vera Rubin pilot publication

`original-pilot.tar.gz` contains the original pilot bundle from AISimulate commit
`d1e3cd84cb7f891046668ea5bd262009cac497d4`, under
`python/aisimulate/src/aisimulate_core/systems/`. It includes the fourteen files
listed in the original publication receipt, that receipt, and
`profile-evidence.json`. Paths and bytes are unchanged. The archive omits the
bundle README and unrelated systems; tar ownership and timestamps are normalized.

Source: https://github.com/ai-dynamo/aisimulate/tree/d1e3cd84cb7f891046668ea5bd262009cac497d4/python/aisimulate/src/aisimulate_core/systems

These NVIDIA-authored repository fixtures retain the project's Apache-2.0
license. They exercise the rename migration against its actual hash-pinned
input without requiring Git history, a network connection, or GPU collection.
The GEMM and MoE collection metadata include the source-attribution corrections
made after the historical publication receipt; the migration pins those bytes
explicitly.
