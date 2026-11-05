# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize checked checkpoint metadata for both predictors, without network."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path


def canonical_json_bytes(value: Mapping) -> bytes:
    """Stable serialization used by snapshot hashes, distinct from HTTP hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _snapshot_files(snapshot: Mapping) -> dict[str, bytes]:
    files = {}
    for key, text_key, hash_key, filename in (
        ("config", "config_json", "config_sha256", "config.json"),
        ("hf_quant_config", "companion_json", "companion_sha256", "hf_quant_config.json"),
    ):
        value = snapshot.get(key)
        if value is None and key == "hf_quant_config":
            if snapshot.get(hash_key) is not None or snapshot.get(text_key) is not None:
                raise ValueError("checkpoint companion hash/text exists without companion metadata")
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint snapshot {key} must be a mapping")
        original = snapshot.get(text_key)
        if original is not None:
            if not isinstance(original, str) or json.loads(original) != value:
                raise ValueError(f"checkpoint snapshot {text_key} differs from parsed metadata")
            content = original.encode()
        else:
            content = canonical_json_bytes(value)
        expected = snapshot.get(hash_key)
        actual = hashlib.sha256(content).hexdigest()
        if expected != actual:
            raise ValueError(f"checkpoint snapshot {hash_key} mismatch: expected {expected!r}, actual {actual}")
        files[filename] = content
    return files


def snapshot_content_hash(snapshot: Mapping) -> str:
    files = _snapshot_files(snapshot)
    return hashlib.sha256(
        canonical_json_bytes({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    ).hexdigest()


def materialize_model_config(snapshot: Mapping, *, cache_root: str | Path | None = None) -> str:
    """Return an immutable directory containing only the validated model files.

    Publish the complete directory with one rename so concurrent predictor
    workers never observe a partially written checkpoint. Existing content is
    rechecked instead of trusting its directory name.
    """
    files = _snapshot_files(snapshot)
    digest = snapshot_content_hash(snapshot)
    root = (
        Path(cache_root)
        if cache_root is not None
        else Path(tempfile.gettempdir()) / f"aisim-e2e-gym-model-configs-{os.getuid()}"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / digest
    if not target.exists():
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            for name, content in files.items():
                (staging / name).write_bytes(content)
            try:
                staging.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if not target.is_dir() or {p.name for p in target.iterdir()} != set(files):
        raise ValueError("checkpoint snapshot cache has unexpected files")
    for name, content in files.items():
        if (target / name).read_bytes() != content:
            raise ValueError(f"checkpoint snapshot cache content mismatch: {name}")
    return str(target)


def normalize_trt_snapshot(snapshot: Mapping) -> dict:
    """Apply TRT's companion-file precedence without changing source evidence.

    Both cores otherwise combine inline metadata with the companion. Removing
    only the ignored inline declaration lets their existing companion loader
    receive TRT's selected declaration. The original config and hash remain
    available for auditing; the runtime config receives its own checked hash.
    """
    _snapshot_files(snapshot)
    result = deepcopy(dict(snapshot))
    if not result.get("hf_quant_config"):
        return result
    quant = result["hf_quant_config"].get("quantization")
    reviewed_keys = {"quant_algo", "kv_cache_quant_algo", "group_size", "exclude_modules"}
    if (
        not isinstance(quant, dict)
        or str(quant.get("quant_algo", "")).upper() != "NVFP4"
        or quant.get("group_size") != 16
        or not set(quant).issubset(reviewed_keys)
    ):
        return result
    config = result["config"]
    removed = []
    if "quantization_config" in config:
        del config["quantization_config"]
        removed.append("quantization_config")
    text = config.get("text_config")
    if isinstance(text, dict) and "quantization_config" in text:
        del text["quantization_config"]
        removed.append("text_config.quantization_config")
    if not removed:
        return result
    result.setdefault("source_config", deepcopy(snapshot["config"]))
    result.setdefault("source_config_sha256", snapshot["config_sha256"])
    if "config_json" in result:
        result.setdefault("source_config_json", result.pop("config_json"))
    result["config_sha256"] = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    result["snapshot_transformation"] = {
        "name": "trt_companion_quantization_precedence",
        "removed_paths": removed,
        "reason": "TRT-LLM reads hf_quant_config.json instead of inline quantization_config when present",
        "source_config_sha256": result["source_config_sha256"],
        "runtime_config_sha256": result["config_sha256"],
    }
    return result
