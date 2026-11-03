"""Verify the additive native verifier delta against the qualified build receipt.

Run from the checkout root. This reads source files and writes only adjacent
evidence. It never rewrites production source or the original build receipt.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
RECEIPT = OUT / "candidate-build.json"
receipt = json.loads(RECEIPT.read_bytes())
spans = {
    "crates/core/src/perfmodel/fpm/model.rs": (
        "    /// Recheck immutable timing inputs",
        "    /// Whether some native estimate differs",
    ),
    "crates/core/src/perfmodel/py.rs": (
        "    fn verify_dependencies(&self, py:",
        "    fn dependency_info_json",
    ),
}
test_only = {
    "crates/core/src/perfmodel/fpm/snapshot/tests.rs",
    "crates/tests/public-api/src/lib.rs",
}
records = {}
patches = []
failures = []
for name, qualified_hash in receipt["source_files"].items():
    if not (
        name in {"Cargo.toml", "Cargo.lock"} or (name.startswith("crates/") and name.endswith((".rs", "Cargo.toml")))
    ):
        continue
    path = ROOT / name
    current = path.read_bytes() if path.is_file() else None
    current_hash = sha256(current) if current is not None else None
    record = {
        "qualified_sha256": qualified_hash,
        "current_sha256": current_hash,
        "classification": "unchanged",
    }
    if name in spans:
        if current is None:
            raise RuntimeError(f"Missing source: {name}")
        text = current.decode()
        start, end = spans[name]
        if text.count(start) != 1:
            raise RuntimeError(f"Ambiguous verifier start in {name}")
        a = text.index(start)
        b = text.index(end, a)
        reconstructed = text[:a] + text[b:]
        reconstructed_hash = sha256(reconstructed.encode())
        record.update(
            classification="additive_explicit_verifier_only",
            stripped_sha256=reconstructed_hash,
            stripped_matches_qualified=reconstructed_hash == qualified_hash,
            removed_start_line=text[:a].count("\n") + 1,
            removed_line_count=text[a:b].count("\n"),
        )
        if reconstructed_hash != qualified_hash:
            failures.append(name)
        patches.extend(
            difflib.unified_diff(
                reconstructed.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    elif current_hash != qualified_hash:
        if name in test_only:
            record["classification"] = "test_only"
        else:
            record["classification"] = "unexpected_difference"
            failures.append(name)
    records[name] = record

listed = subprocess.check_output(
    ["git", "ls-files", "-c", "-o", "--exclude-standard", "--", "crates"],
    cwd=ROOT,
    text=True,
).splitlines()
unrecorded = sorted(
    name
    for name in set(listed)
    if name.endswith((".rs", "Cargo.toml")) and name not in records and (ROOT / name).is_file()
)
failures.extend(unrecorded)
patch_path = OUT / "final-verifier-native-delta.patch"
patch_path.write_text("".join(patches))
result = {
    "created_utc": datetime.now(UTC).isoformat(),
    "checkout": str(ROOT),
    "observed_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "qualified_receipt": str(RECEIPT.relative_to(ROOT)),
    "qualified_receipt_sha256": sha256(RECEIPT.read_bytes()),
    "qualified_native_sha256": receipt["native_sha256"],
    "proof_script_sha256": sha256(Path(__file__).read_bytes()),
    "patch_sha256": sha256(patch_path.read_bytes()),
    "scope": (
        "Every Rust source and Cargo manifest/lock recorded by the qualified build; "
        "new tracked/untracked crate Rust sources and manifests are rejected. "
        "Only the two explicit verifier methods differ in native runtime sources. "
        "Snapshot unit tests and public API test crate changes are listed separately."
    ),
    "limitations": (
        "This is a source delta proof, not a new accuracy or host-cost measurement. "
        "The rebuilt extension has a different binary hash. The new verifier performs "
        "I/O only when explicitly called; its cost is not covered by prior measurements. "
        "Python caller, documentation and example changes are outside this native proof."
    ),
    "files": records,
    "unrecorded_native_inputs": unrecorded,
    "failures": failures,
    "passed": not failures,
}
(OUT / "final-verifier-native-delta.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": not failures, "native_inputs": len(records), "failures": failures}))
if failures:
    raise SystemExit(1)
