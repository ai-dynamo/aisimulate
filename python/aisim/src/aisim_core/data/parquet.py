# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet (zstd) read/write helpers that stamp and verify the schema ID.

All tabular dataset files are written through :func:`write_parquet`. The schema
ID lives in the Parquet key-value metadata so a reader can reject a file that
does not match the schema it expects. Offline only; no network access.
"""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_METADATA_KEY = b"aisim.schema"
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3


class ParquetSchemaError(ValueError):
    """Raised when a Parquet file lacks or mismatches the expected schema ID."""


def _stamp(table: pa.Table, schema_id: str, extra_metadata: Mapping[str, str] | None) -> pa.Table:
    if not schema_id:
        raise ValueError("schema_id must be non-empty")
    metadata = {SCHEMA_METADATA_KEY: schema_id.encode()}
    for key, value in (extra_metadata or {}).items():
        if key.encode() == SCHEMA_METADATA_KEY:
            raise ValueError("extra_metadata must not override the schema ID")
        metadata[key.encode()] = value.encode()
    # Replace rather than merge so pandas/foreign metadata never leaks in.
    return table.replace_schema_metadata(metadata)


def write_parquet(
    table: pa.Table,
    path: str | Path,
    schema_id: str,
    *,
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write ``table`` as zstd Parquet with ``schema_id`` in the file metadata.

    The write is atomic (temp file then rename) so an interrupted write never
    leaves a partial file at ``path``. Returns ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = _stamp(table, schema_id, extra_metadata)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(
            stamped,
            tmp_name,
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
        )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def read_parquet_schema_id(path: str | Path) -> str | None:
    """Return the schema ID stored in the file metadata, or None when absent."""
    metadata = pq.read_metadata(path).metadata or {}
    value = metadata.get(SCHEMA_METADATA_KEY)
    return None if value is None else value.decode()


def read_parquet(
    path: str | Path,
    expected_schema_id: str | tuple[str, ...] | None = None,
    *,
    columns: list[str] | None = None,
) -> pa.Table:
    """Read a Parquet file, rejecting a missing or unexpected schema ID.

    ``expected_schema_id`` may be one ID or several accepted versions. Pass
    ``None`` only for inspection; loaders must always name what they accept.
    """
    actual = read_parquet_schema_id(path)
    if expected_schema_id is not None:
        accepted = (expected_schema_id,) if isinstance(expected_schema_id, str) else tuple(expected_schema_id)
        if actual is None:
            raise ParquetSchemaError(f"{path}: Parquet file has no schema ID; expected one of {list(accepted)}")
        if actual not in accepted:
            raise ParquetSchemaError(f"{path}: schema ID {actual!r} not in accepted {list(accepted)}")
    return pq.read_table(path, columns=columns)
