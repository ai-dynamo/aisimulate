# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encrypt private intermediate artifacts before uploading them to public CI."""

import argparse
import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"AISIM-HF-ARTIFACT-v1\0"


def transfer(source: Path, destination: Path, *, decrypt: bool):
    token = os.environ.get("HF_TOKEN")
    repository, run = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_RUN_ID")
    if not token or not repository or not run:
        raise ValueError("private artifacts require HF_TOKEN and GitHub workflow identity")
    context = f"{repository}:{run}".encode()
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=context, info=MAGIC).derive(token.encode())
    if not source.is_dir() or source.is_symlink():
        raise ValueError("private artifact source must be a directory")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("private artifact destination must be empty")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("private artifact destination must be outside the source")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("private artifact symlinks are forbidden")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if decrypt:
            if not relative.name.endswith(".enc"):
                raise ValueError("unexpected unencrypted CI artifact")
            relative = relative.with_name(relative.name[:-4])
        target = destination / (relative if decrypt else relative.with_name(relative.name + ".enc"))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with path.open("rb") as incoming, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as outgoing:
                temporary = Path(outgoing.name)
                if decrypt:
                    size = path.stat().st_size
                    if size < len(MAGIC) + 12 + 16 or incoming.read(len(MAGIC)) != MAGIC:
                        raise ValueError("invalid encrypted CI artifact")
                    nonce = incoming.read(12)
                    incoming.seek(-16, 2)
                    tag = incoming.read(16)
                    incoming.seek(len(MAGIC) + 12)
                    remaining = size - len(MAGIC) - 12 - 16
                    transform = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
                else:
                    nonce = os.urandom(12)
                    outgoing.write(MAGIC + nonce)
                    remaining = path.stat().st_size
                    transform = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
                transform.authenticate_additional_data(context + b"\0" + relative.as_posix().encode())
                while remaining:
                    block = incoming.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError("truncated private artifact")
                    remaining -= len(block)
                    outgoing.write(transform.update(block))
                outgoing.write(transform.finalize())
                if not decrypt:
                    outgoing.write(transform.tag)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "open"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    transfer(args.source, args.destination, decrypt=args.action == "open")
