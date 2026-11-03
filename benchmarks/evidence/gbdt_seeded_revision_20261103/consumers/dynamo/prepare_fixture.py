"""Repackage the original self-authored two-store seed without changing its bytes.

The product example now demonstrates a decode-only worker. Consumer coverage
still needs aggregated pure-prefill/pure-decode with intentionally missing mixed
stores, so this separate fixture preserves that original functional population.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from aisimulate_core.sdk.fpm_learned import export_bundle


REVISION = "3cb6679d065f1ccfe3d4bbdd0e321a5a78238957"
PREFIX = "python/aisimulate/src/aisimulate_core/systems/fpm_learned/examples/synthetic-functional-v1"


def main():
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[4]

    def read(path):
        return subprocess.check_output(["git", "show", f"{REVISION}:{PREFIX}/{path}"], cwd=repo)

    original = json.loads(read("bundle.json"))
    with tempfile.TemporaryDirectory(prefix="gbdt-aggregated-consumer-seed-") as temporary:
        staging = Path(temporary)
        records = []
        for dependency in original["dependencies"]:
            contents = read(dependency["path"])
            digest = hashlib.sha256(contents).hexdigest()
            if digest != dependency["sha256"]:
                raise ValueError(f"Historical seed hash mismatch: {dependency['path']}")
            path = staging / dependency["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            records.append({**dependency, "git_revision": REVISION})
        destination = evidence / "aggregated-fixture"
        export_bundle(staging / "seed", destination, provenance=original["provenance"],
                      recipe_version=original["recipe_version"])
    current = json.loads((destination / "bundle.json").read_text())
    if current["identity"] != original["identity"] or current["seed"] != original["seed"]:
        raise ValueError("Repackaging changed the seed identity or payload")
    if current["provenance"] != original["provenance"]:
        raise ValueError("Repackaging changed source provenance")
    receipt = {"source_revision": REVISION, "source_directory": PREFIX,
               "source_bundle_sha256": hashlib.sha256(read("bundle.json")).hexdigest(),
               "bundle_sha256": hashlib.sha256((destination / "bundle.json").read_bytes()).hexdigest(),
               "seed_identity_and_provenance_unchanged": True,
               "dependencies": records,
               "scope": "Self-authored synthetic functional inputs; no hardware accuracy qualification."}
    (evidence / "fixture-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(destination / "bundle.json")


if __name__ == "__main__":
    main()
