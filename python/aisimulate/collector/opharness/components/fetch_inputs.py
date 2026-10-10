#!/usr/bin/env python3
"""onboard_model step 1: fetch a checkpoint's INPUT files from the Hub.

For every repo: config.json -> configs/<org>_<name>.json, hf_quant_config.json
-> configs/<org>_<name>_hfquant.json (when the artifact ships one), and every
other non-weight file (tokenizer, processor/preprocessor configs, chat
template, custom modeling/processing code, tiktoken/sentencepiece models)
-> configs/aux_files/<org>_<name>/ — the directory dummies.py provisions the
dummy dirs from. Weights are never downloaded. Missing config.json is a hard
stop (OWNER DECISION: fetch or exclude), never a silent skip.

Usage: fetch_model_inputs.py [--configs configs] REPO [REPO ...]
       fetch_model_inputs.py --from-roster           (every repo in configs/repos.txt)
Auth: HF token from $HF_TOKEN or ~/.cache/huggingface/token; proxies from the environment.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5", ".onnx", ".ckpt",
                   ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".pdf", ".parquet")
SKIP_NAMES = {".gitattributes", "README.md", "LICENSE", "LICENSE.txt", "NOTICE", "model.safetensors.index.json"}


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    p = Path.home() / ".cache" / "huggingface" / "token"
    return tok or (p.read_text().strip() if p.exists() else None)


def _get(url: str, token: str | None, binary: bool = False):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if binary else data.decode("utf-8")


_SDK_CONFIGS = Path(__file__).resolve().parents[3] / "src" / "aisimulate_core" / "model_configs"


def _bundled_fallback(repo: str, configs: Path, reason: str) -> dict | None:
    """Gated or vanished repos: this checkout ships the SDK's copy of the HF
    config (src/aisimulate_core/model_configs/<org>--<name>_config.json, plus
    the hf_quant config when the artifact has one). Use it, say so in
    configs/aux_files/<tag>/PROVENANCE.json, and leave the tokenizer to
    dummies.py (targets `tokenizer_from` / a sibling artifact) — every
    Blackwell box without an HF token staged meta-llama by hand from exactly
    these files (B200 / RTX 6000 / RTX 5000 handoffs, 2026-09-30)."""
    tag = repo.replace("/", "_")
    src = _SDK_CONFIGS / f"{repo.replace('/', '--')}_config.json"
    if not src.exists():
        return None
    (configs / f"{tag}.json").write_text(src.read_text())
    got = {"config": True, "hf_quant": False, "aux": [], "source": f"bundled SDK model_configs ({reason})"}
    q = _SDK_CONFIGS / f"{repo.replace('/', '--')}_hf_quant_config.json"
    if q.exists():
        (configs / f"{tag}_hfquant.json").write_text(q.read_text())
        got["hf_quant"] = True
    aux = configs / "aux_files" / tag
    aux.mkdir(parents=True, exist_ok=True)
    (aux / "PROVENANCE.json").write_text(json.dumps(
        {"repo": repo, "reason": reason, "config_source": str(src.relative_to(_SDK_CONFIGS.parents[2])),
         "hf_quant_source": str(q.relative_to(_SDK_CONFIGS.parents[2])) if q.exists() else None,
         "tokenizer": "not fetched — dummies.py provisions from targets tokenizer_from / a sibling artifact"}, indent=1))
    print(f"{repo}: {reason} — config taken from the bundled SDK copy (aux_files/{tag}/PROVENANCE.json); tokenizer NOT fetched")
    return got


def fetch(repo: str, configs: Path, token: str | None) -> dict:
    try:
        info = json.loads(_get(f"https://huggingface.co/api/models/{repo}", token))
    except Exception as e:  # 401/403/404/offline: fall back to the bundled config if there is one
        got = _bundled_fallback(repo, configs, f"hub api failed: {type(e).__name__}")
        if got:
            return got
        raise SystemExit(f"{repo}: Hub API failed ({e}) and no bundled SDK config — OWNER DECISION")
    if info.get("gated") and not token:
        got = _bundled_fallback(repo, configs, "gated, no token")
        if got:
            return got
        raise SystemExit(f"{repo}: gated — OWNER DECISION (accept the license with a token, or exclude in targets.yaml)")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    if "config.json" not in files:
        got = _bundled_fallback(repo, configs, "no config.json on the Hub")
        if got:
            return got
        raise SystemExit(f"{repo}: no config.json on the Hub — OWNER DECISION")
    tag = repo.replace("/", "_")
    base = f"https://huggingface.co/{repo}/resolve/main/"
    (configs / f"{tag}.json").write_text(_get(base + "config.json", token))
    got = {"config": True, "hf_quant": False, "aux": []}
    if "hf_quant_config.json" in files:
        (configs / f"{tag}_hfquant.json").write_text(_get(base + "hf_quant_config.json", token))
        got["hf_quant"] = True
    aux = configs / "aux_files" / tag
    aux.mkdir(parents=True, exist_ok=True)
    for f in files:
        if "/" in f or f in SKIP_NAMES or f in ("config.json", "hf_quant_config.json"):
            continue
        if f.lower().endswith(WEIGHT_SUFFIXES):
            continue
        (aux / f).write_bytes(_get(base + f, token, binary=True))
        got["aux"].append(f)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ws = os.environ.get("AIS_PROBE_WORKSPACE") or os.environ.get("AIC_PROBE_WORKSPACE") or os.getcwd()
    ap.add_argument("--configs", type=Path, default=Path(ws) / "configs",
                    help="workspace configs/ dir (default: $AIS_PROBE_WORKSPACE/configs, else ./configs)")
    ap.add_argument("--from-roster", action="store_true", help="every repo in <configs>/repos.txt")
    ap.add_argument("repos", nargs="*")
    args = ap.parse_args()
    repos = list(args.repos)
    if args.from_roster:
        for line in (args.configs / "repos.txt").read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                repos.append(line.split()[0])
    if not repos:
        ap.error("give repos or --from-roster")
    token = _token()
    failures = 0
    for repo in repos:
        try:
            got = fetch(repo, args.configs, token)
            print(f"fetched {repo}: hf_quant={got['hf_quant']} aux={len(got['aux'])} files "
                  f"({', '.join(x for x in got['aux'] if x.startswith('tok') or x.endswith('.py'))[:120]})")
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            failures += 1
        except Exception as e:  # network / http
            print(f"{repo}: fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
