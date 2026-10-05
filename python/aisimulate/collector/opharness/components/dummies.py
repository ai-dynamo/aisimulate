#!/usr/bin/env python3
"""Generate depth-shrunk dummy-model configs for backend-identity probing.

Rules (agreed 2026-08-16):
  * Cut depth only — every width dimension (hidden, heads, kv-heads, expert
    count, moe_intermediate) keeps its real value so TP/EP divisibility and
    quant shape checks behave exactly like the full checkpoint.
  * Interleaved models get one variant per layer kind (DSV4 csa/hca/full,
    GLM full/shared indexer, M3 dense/moe) so each kind is probed alone.
  * MoE models drop their leading dense layers except in an explicit
    head variant.
  * quantization_config is preserved verbatim except for per-layer entries
    (quantized_layers.layers.N, model.layers.N.* ignore/not-convert lists),
    which are filtered to the selected layers and renumbered.
  * MTP / next-N heads are zeroed for lean dummy loading.
  * Layers that PUBLISH state for later layers (DSV4.1 kv/index sources,
    the candidate source) are cut together with one consumer each — a
    consumer without its source is an invalid model, a source without a
    consumer never exercises the sharing path.
  * Memory-only tables are shrunk, never dropped (DSV4.1 engram hash tables:
    98 GB per layer at TP1; row count changes nothing about which kernels
    run — see ``_cut_engram``). Owner decision 2026-09-30.

Every variant records its provenance (source repo, original layer indices,
edits applied, caveats) in variants_manifest.yaml. A post-check scans the
final config for any surviving reference to a layer index outside the new
range and fails loudly — silent misalignment is the one failure mode this
tool must not have.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

# Collector ground truth (collect_dsv4_attn.py:313-316): csa=4, hca=128.
DSV4_RATIO_KIND = {4: "csa", 128: "hca", 0: "full"}

# Per-repo declarations live in targets.yaml (single per-model surface):
#   families.<fam>.checkpoints[].repo         -> special-family adapter routing
#   ...checkpoint_overrides.<repo>.dummy_overrides:
#       hfquant_exclude_from_sibling: <repo>  -> complete condensed hf_quant stubs
#       drop_auto_map: <reason>               -> declared auto_map removal
#   (both are RECORDED edits with facts in targets.yaml — never silent fallbacks)
# Loaded once at import; the roster itself lives in repos.txt (one repo per
# line, optional "<repo> <family>") so growing coverage never edits code.
def _load_targets_declarations() -> tuple[dict, dict, dict]:
    import yaml
    here = Path(__file__).resolve().parent
    for cand in (here / "targets.yaml", here.parent / "targets.yaml"):
        if cand.exists():
            t = yaml.safe_load(cand.read_text())
            break
    else:
        return {}, {}, {}, {}
    special, sib, drop, tok = {}, {}, {}, {}
    for fam, spec in (t.get("families") or {}).items():
        if fam != "roster":
            for ck in spec.get("checkpoints") or []:
                special[ck["repo"]] = fam
        for repo, o in (spec.get("checkpoint_overrides") or {}).items():
            dv = o.get("dummy_overrides") or {}
            if dv.get("family"):
                special[repo] = dv["family"]
            if dv.get("hfquant_exclude_from_sibling"):
                sib[repo] = dv["hfquant_exclude_from_sibling"]
            if dv.get("drop_auto_map"):
                drop[repo] = dv["drop_auto_map"]
            if dv.get("tokenizer_from"):
                tok[repo] = dv["tokenizer_from"]
    return special, sib, drop, tok


_SPECIAL, _HFQUANT_COMPLETE_FROM_SIBLING, _DROP_AUTO_MAP, _TOKENIZER_FROM = _load_targets_declarations()

# Frameworks load more than config.json from the model dir even for a
# dummy-weight probe: the tokenizer (vllm/sglang at engine init, trtllm at
# generate for end_id), custom modeling / processor code (Kimi: tiktoken.model
# + tokenization_kimi.py + modeling_*.py), preprocessor configs. None of it is
# derivable from config.json and NO earlier step provisioned it for new roster
# repos (found 2026-09-24: every GLM-5.3-BF16 run died on "Couldn't
# instantiate the backend tokenizer"). Everything in the source dir except the
# files this generator writes itself is an auxiliary artifact file.
_GENERATED_FILES = ("config.json", "hf_quant_config.json", "dtype_probe.safetensors")


def _provision_aux_files(out_dir: Path, repo: str, configs: Path, out_root: Path,
                         edits: list[str], caveats: list[str]) -> None:
    """Copy the auxiliary artifact files into a dummy dir. Sources, in order:
    the fetched aux dir (configs/aux_files/<org>_<name>/), an existing variant
    dir of the SAME repo (previous generation), the declared sibling
    (targets.yaml dummy_overrides.tokenizer_from — same tokenizer/code family,
    e.g. GLM-5.3-BF16 <- GLM-5.3). Anything else is a loud MISSING, never silent."""
    import shutil
    if (out_dir / "tokenizer_config.json").exists():
        return
    name = repo.split("/")[-1]
    candidates = [configs / "aux_files" / repo.replace("/", "_")]
    candidates += sorted(p for p in out_root.glob(f"*/{name}__*") if p.is_dir() and p != out_dir)
    sib = _TOKENIZER_FROM.get(repo)
    if sib:
        candidates += sorted(p for p in out_root.glob(f"*/{sib.split('/')[-1]}__*") if p.is_dir())
    for src in candidates:
        if (src / "tokenizer_config.json").exists():
            copied = []
            for f in sorted(src.iterdir()):
                if f.is_file() and f.name not in _GENERATED_FILES and not f.name.endswith(".safetensors") \
                        and not (out_dir / f.name).exists():
                    shutil.copy2(f, out_dir / f.name)
                    copied.append(f.name)
            where = src.relative_to(out_root) if src.is_relative_to(out_root) else src
            edits.append(f"aux files {len(copied)} from {where}"
                         + (f" (declared tokenizer_from {sib})" if sib and sib.split('/')[-1] in src.name else ""))
            return
    caveats.append("MISSING TOKENIZER/AUX FILES: no fetched aux dir, no earlier variant, no declared tokenizer_from")
    print(f"MISSING TOKENIZER {repo}: declare dummy_overrides.tokenizer_from in targets.yaml "
          f"or fetch configs/aux_files/{repo.replace('/', '_')}/", file=sys.stderr)


def load_repos(configs_dir: Path) -> dict[str, str]:
    roster = configs_dir / "repos.txt"
    repos: dict[str, str] = {}
    if roster.exists():
        for line in roster.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            repos[parts[0]] = parts[1] if len(parts) > 1 else _SPECIAL.get(parts[0], "generic")
    else:  # fall back to every fetched config
        for f in configs_dir.glob("*.json"):
            if f.name.endswith("_hfquant.json"):
                continue
            repo = f.stem.replace("_", "/", 1)
            repos[repo] = _SPECIAL.get(repo, "generic")
    return repos

# Rule-3 extension: when a framework probes WEIGHT-FILE properties (sglang
# reads the safetensors header dtype of one routed-expert tensor to pick the
# DSV4 expert layout — configs/deepseek_v4.py try_detect_fp4_experts), the
# dummy must carry that signal. We write a tiny single-tensor safetensors
# whose key+dtype mirror the REAL checkpoint header (fetched once into
# configs/dsv4_expert_dtypes.json). Without it the probe returns None and the
# env default (fp4=True) misclassifies converted-FP8 checkpoints -> the
# 'Hidden size mismatch' false positive this fixes.
def write_dtype_probe_safetensors(out_dir: Path, key: str, dtype: str, edits: list[str]) -> None:
    import struct
    nbytes = {"I8": 1, "U8": 1, "F8_E4M3": 1, "BF16": 2, "F16": 2, "F32": 4}[dtype] * 4
    header = {key: {"dtype": dtype, "shape": [4], "data_offsets": [0, nbytes]},
              "__metadata__": {"aic_probe": "dtype signal only; dummy weights are runtime-generated"}}
    hj = json.dumps(header).encode()
    pad = (8 - len(hj) % 8) % 8
    hj += b" " * pad
    with open(out_dir / "dtype_probe.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(b"\x00" * nbytes)
    edits.append(f"dtype_probe.safetensors written: {key}={dtype} (real header signal)")


_LAYER_REF_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|\*|$)")


def _slice_layer_lists(cfg: dict, n_layers: int, sel: list[int], edits: list[str], prefix: str = "") -> None:
    """Slice every list field whose length equals n_layers down to sel."""
    for key, val in cfg.items():
        if isinstance(val, list) and len(val) == n_layers:
            cfg[key] = [val[i] for i in sel]
            edits.append(f"sliced {prefix}{key}[{n_layers}] -> {len(sel)}")
        elif isinstance(val, dict):
            _slice_layer_lists(val, n_layers, sel, edits, prefix=f"{prefix}{key}.")


def _remap_layer_index_lists(cfg: dict, n_layers: int, sel: list[int], edits: list[str], prefix: str = "") -> None:
    """Renumber every list of LAYER INDICES to the selected layers.

    Some configs name layer kinds by index lists instead of a per-layer type
    list — Kimi-K3 ``linear_attn_config.kda_layers`` / ``full_attn_layers``.
    A depth cut that keeps them verbatim silently reassigns kinds (index 0
    of the cut model falls out of every list). A list qualifies when its key
    mentions ``layers`` and every element is an int in [0, n_layers).
    """
    new_index = {orig: new for new, orig in enumerate(sel)}

    def _is_layer_list(key, val):
        return (isinstance(val, list) and val
                and ("layers" in key.lower() or _LAYER_ID_KEY_RE.search(key.lower()))
                and all(isinstance(x, int) and not isinstance(x, bool) for x in val)
                and any(0 <= x <= n_layers for x in val))
    # Kimi-K3's lists are 1-BASED (configuration_kimi_k3.is_kda_layer:
    # `layer_idx + 1 in kda_layers`): they never contain 0 and run up to
    # n_layers. The base is a property of the CONFIG, so detect it once over
    # the union of the sibling lists (kda_layers alone tops out below
    # n_layers and would look 0-based) — never assume.
    union = {x for k, v in cfg.items() if _is_layer_list(k, v) for x in v}
    base = 1 if union and 0 not in union and max(union) >= n_layers else 0
    for key, val in cfg.items():
        if _is_layer_list(key, val):
            kept = [new_index[x - base] + base for x in val if (x - base) in new_index]
            cfg[key] = kept
            edits.append(f"remapped {prefix}{key} ({'1' if base else '0'}-based): {len(val)} layer indices -> {kept}")
        elif isinstance(val, dict):
            _remap_layer_index_lists(val, n_layers, sel, edits, prefix=f"{prefix}{key}.")


def _remap_quant_layer_entries(cfg: dict, sel: list[int], edits: list[str]) -> None:
    """Filter/renumber per-layer quantization entries to the selected layers.

    Layer references appear in several shapes across quantizers:
      * quantized_layers: {"layers.N.ffn.experts": {...}}          (modelopt dsv4)
      * ignore / modules_to_not_convert / ignored_layers / exclude_modules
      * config_groups.<g>.targets: ["backbone.layers.N.mixer...."] (nemotron)
    """
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        return
    new_index = {orig: new for new, orig in enumerate(sel)}

    ql = qc.get("quantized_layers")
    if isinstance(ql, dict):  # flat keys like "layers.5.ffn.experts"
        kept = {}
        for k, v in ql.items():
            m = re.search(r"^(.*?\blayers\.)(\d+)(.*)$", k)
            if m is None:
                kept[k] = v
            elif int(m.group(2)) in new_index:
                kept[f"{m.group(1)}{new_index[int(m.group(2))]}{m.group(3)}"] = v
        edits.append(f"quantized_layers: {len(ql)} -> {len(kept)} entries, renumbered")
        qc["quantized_layers"] = kept

    containers = [(qc, f) for f in
                  ("ignore", "modules_to_not_convert", "ignored_layers", "exclude_modules")]
    for grp in (qc.get("config_groups") or {}).values():
        if isinstance(grp, dict):
            containers.append((grp, "targets"))

    for container, field in containers:
        entries = container.get(field)
        if not isinstance(entries, list):
            continue
        kept, dropped = [], 0
        for e in entries:
            m = re.search(r"^(.*?\blayers\.)(\d+)(.*)$", e) if isinstance(e, str) else None
            if m is None:
                kept.append(e)  # no layer index (lm_head, wildcards, module classes)
            elif int(m.group(2)) in new_index:
                kept.append(f"{m.group(1)}{new_index[int(m.group(2))]}{m.group(3)}")
            else:
                dropped += 1
        if dropped or kept != entries:
            edits.append(f"{field}: renumbered, dropped {dropped} out-of-range entries")
        container[field] = kept


_LAYER_ID_KEY_RE = re.compile(r"(?:^|_)layer_ids?$")


def _check_no_stale_layer_refs(cfg: dict, max_layer: int) -> list[str]:
    """Scan the final config for layer-index references outside [0, max_layer).

    A nested sub-config that declares its own depth (vision_config
    num_hidden_layers, Inkling mtp_config num_nextn_predict_layers) is checked
    against THAT depth: its layer indices live on a different axis.
    """
    stale = []

    def walk(obj, path, max_layer):
        if isinstance(obj, dict):
            if path != "$" and isinstance(obj.get("num_hidden_layers"), int):
                max_layer = obj["num_hidden_layers"]
            elif path != "$" and isinstance(obj.get("num_nextn_predict_layers"), int) \
                    and obj["num_nextn_predict_layers"] > 0:
                max_layer = obj["num_nextn_predict_layers"]
            for k, v in obj.items():
                for m in _LAYER_REF_RE.finditer(str(k)):
                    if int(m.group(1)) >= max_layer:
                        stale.append(f"{path}.{k}")
                # scalar layer pointers (DSV4.1 candidate_source_layer_id; -1 = none)
                if _LAYER_ID_KEY_RE.search(str(k)) and isinstance(v, int) \
                        and not isinstance(v, bool) and v >= max_layer:
                    stale.append(f"{path}.{k} = {v}")
                walk(v, f"{path}.{k}", max_layer)
        elif isinstance(obj, list):
            # index lists under a *layers* / *_layer_ids key are layer references
            # too (Kimi-K3 kda_layers / full_attn_layers; DSV4.1
            # kv_source_layer_ids / engram_layer_ids — the latter slipped
            # through the generic adapter unremapped on 2026-09-27)
            leaf = path.rsplit(".", 1)[-1]
            if (("layers" in leaf.lower() or _LAYER_ID_KEY_RE.search(leaf))
                    and obj and all(isinstance(x, int) and not isinstance(x, bool) for x in obj)
                    # 0-based lists are stale at >= max_layer; 1-based ones (no 0) may reach max_layer
                    and (max(obj) >= max_layer if 0 in obj else max(obj) > max_layer)):
                stale.append(f"{path} = {obj[:6]}...")
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", max_layer)
        elif isinstance(obj, str):
            for m in _LAYER_REF_RE.finditer(obj):
                if int(m.group(1)) >= max_layer:
                    stale.append(f"{path} = {obj}")

    walk(cfg, "$", max_layer)
    return stale


# ---------------------------------------------------------------- adapters

def variants_dsv4(cfg: dict) -> list[dict]:
    n = cfg["num_hidden_layers"]
    n_hash = cfg.get("num_hash_layers", 0)
    ratios = cfg["compress_ratios"]
    # base checkpoints: len == n + n_hash; nvidia NVFP4 requants ship n + 1
    # with num_hash_layers still 3 — upstream inconsistency, tolerate it.
    assert len(ratios) >= n, f"compress_ratios len {len(ratios)} < num_hidden_layers {n}"
    tail = ratios[n:]
    main = ratios[:n]
    dspark = set(cfg.get("dspark_target_layer_ids", []))

    def pick(kind: str, count: int) -> list[int]:
        return [i for i, r in enumerate(main) if DSV4_RATIO_KIND[r] == kind and i not in dspark][:count]

    out = []
    for kind in ("csa", "hca", "full"):
        sel = pick(kind, 2)
        if sel:
            out.append({"name": kind, "sel": sel, "hash": False, "dspark": False})
    # one csa + one hca adjacent pair: the pool configurator sees both kinds
    csa1, hca1 = pick("csa", 1), pick("hca", 1)
    if csa1 and hca1:
        out.append({"name": "interleave_pair", "sel": sorted(csa1 + hca1), "hash": False, "dspark": False})
    # one layer of EVERY kv-spec kind: vllm's DSV4 kv grouping asserts the
    # full-MLA group exists and bounds SWA page sizes — variants missing a
    # kind violate that structural invariant (same lesson as gpt-oss SWA)
    full1 = pick("full", 1)
    if csa1 and hca1 and full1:
        out.append({"name": "rep_mix", "sel": sorted(full1 + csa1 + hca1), "hash": False, "dspark": False})
    if dspark:
        out.append({"name": "dspark", "sel": sorted(dspark), "hash": False, "dspark": True})
    if n_hash and len(tail) == n_hash:
        out.append({"name": "hash", "sel": pick("csa", 1) or [0], "hash": True, "dspark": False})
    return out


def apply_dsv4(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    n_hash = cfg.get("num_hash_layers", 0)
    sel = var["sel"]
    ratios = cfg["compress_ratios"]
    # compress_ratios is main+hash; handle it explicitly, then generic-slice the rest
    tail = ratios[n:] if (var["hash"] and n_hash) else []
    del cfg["compress_ratios"]  # main+hash length; sliced explicitly below
    _slice_layer_lists(cfg, n, sel, edits)
    cfg["compress_ratios"] = [ratios[i] for i in sel] + tail
    edits.append(f"compress_ratios -> {cfg['compress_ratios']}")
    cfg["num_hidden_layers"] = len(sel)
    if not var["hash"]:
        if n_hash:
            cfg["num_hash_layers"] = 0
            edits.append("num_hash_layers -> 0")
    if var["dspark"]:
        remap = {o: i for i, o in enumerate(sel)}
        cfg["dspark_target_layer_ids"] = [remap[i] for i in cfg["dspark_target_layer_ids"]]
        edits.append(f"dspark_target_layer_ids -> {cfg['dspark_target_layer_ids']}")
    else:
        dropped = [k for k in list(cfg) if k.startswith("dspark_")]
        for k in dropped:
            del cfg[k]
        if dropped:
            edits.append(f"dropped {dropped} (precedent: nvidia NVFP4 configs ship without dspark_*)")


# ---- DeepSeek V4.1 (text_config-nested; sparse-attention topology by SOURCE layers)
# compress_ratios per layer: 0 = sliding window, 2 = ratio-2 compressed,
# 1 = full-length compressed. Compressors + compressed KV live only on
# kv_source_layer_ids, indexers only on index_source_layer_ids; a consumer
# reuses the most recent source <= its own index. candidate_source_layer_id
# publishes candidate blocks that every LATER indexer masks with.
#   vllm 0.30.0  models/deepseek_v41/attention.py:256-299, :1035-1050
#   sglang main  models/deepseek_v4.py:1148-1165, layers/attention/dsv4/dsv41_sparse.py:189-191
DSV41_RATIO_KIND = {0: "swa", 2: "c2", 1: "c1"}

# Engram hash tables: rows = engram_num_embeddings[i], one row per hash
# bucket, buckets are (max_ngram-1) x n_heads PRIMES drawn in one ascending
# sequence above engram_vocab_size - 1 (vllm 0.30.0 models/deepseek_v41/common/
# engram.py:186-199; sglang main layers/engram.py:158-175). Reproduces the
# shipped config exactly: vocab 16000000 -> [384006168, 384016682]. A row is
# 256 fp8 + e8m0 scales, so the real table is 98 GB per layer at TP1: vllm
# allocates it on device, sglang main OOMs building a fp16 temp of it in
# initialize_dummy_weights (183 GiB, 2026-09-30). Row count is memory only —
# the hash/lookup/gate kernels and their per-token work are identical — so
# the dummy keeps every kernel and shrinks the bucket space. 100000 keeps the
# hashing regime (buckets per head > compressed vocab 99092).
ENGRAM_DUMMY_VOCAB_SIZE = 100000


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def engram_table_rows(vocab_size: int, n_layers: int, max_ngram_size: int, n_heads: int) -> list[int]:
    """Rows per engram layer = sum of its primes (framework derivation above)."""
    seen: set[int] = set()
    rows = []
    for _ in range(n_layers):
        total = 0
        for _ in range(max_ngram_size - 1):
            current = vocab_size - 1
            for _ in range(n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                total += current
        rows.append(total)
    return rows


def _cut_engram(tc: dict, remap: dict[int, int], edits: list[str]) -> None:
    """Keep the engram layers inside the cut, renumber them, shrink the tables.

    The shipped table sizes must match the prime-sum derivation first (a
    mismatch means the frameworks' rule changed and shrinking would be blind).
    Primes are drawn per kept layer IN ORDER, so a kept layer's rows are those
    of its new position, not its original one.
    """
    eng = list(tc.get("engram_layer_ids") or [])
    if not eng:
        return
    args = (tc["engram_max_ngram_size"], tc["engram_n_heads"])
    real = engram_table_rows(tc["engram_vocab_size"], len(eng), *args)
    if list(tc["engram_num_embeddings"]) != real:
        raise SystemExit(f"engram_num_embeddings {tc['engram_num_embeddings']} != prime-sum derivation "
                         f"{real}; the framework rule changed — re-verify before shrinking")
    kept = [i for i in eng if i in remap]
    tc["engram_layer_ids"] = [remap[i] for i in kept]
    tc["engram_vocab_size"] = ENGRAM_DUMMY_VOCAB_SIZE
    tc["engram_num_embeddings"] = engram_table_rows(ENGRAM_DUMMY_VOCAB_SIZE, len(kept), *args)
    edits.append(f"engram_layer_ids -> {tc['engram_layer_ids']}; tables shrunk to vocab "
                 f"{ENGRAM_DUMMY_VOCAB_SIZE} rows {tc['engram_num_embeddings']} "
                 f"(real {real[:len(kept)]}; memory-only, kernels unchanged)")


def variants_dsv41(cfg: dict) -> list[dict]:
    tc = cfg["text_config"]
    n = tc["num_hidden_layers"]
    main = [int(r) for r in tc["compress_ratios"][:n]]
    unknown = sorted(set(main) - set(DSV41_RATIO_KIND))
    assert not unknown, f"compress_ratios has unsupported values {unknown} (frameworks accept 0/1/2)"
    kv_src = set(tc.get("kv_source_layer_ids") or [])
    idx_src = set(tc.get("index_source_layer_ids") or [])
    cand = int(tc.get("candidate_source_layer_id", -1))
    engram = list(tc.get("engram_layer_ids") or [])
    sel: list[int] = []

    def take(i: int | None) -> None:
        if i is not None and 0 <= i < n and i not in sel:
            sel.append(i)

    def first(pred, start: int = 0) -> int | None:
        return next((i for i in range(start, n) if pred(i)), None)

    take(first(lambda i: main[i] == 0))                      # one sliding-window layer
    take(engram[0] if engram else None)                      # one engram layer (any kind)
    for r in (2, 1):                                         # each compressed kind: source + consumer
        s = first(lambda i: main[i] == r and i in kv_src)
        if s is None:
            continue
        take(s)
        take(first(lambda i: main[i] == r and i not in kv_src, s + 1))
    if cand >= 0:                                            # candidate source + a masked indexer after it
        take(cand)
        take(first(lambda i: main[i] == main[cand] and i not in kv_src, cand + 1))
        take(first(lambda i: i in idx_src, cand + 1))
    rep = sorted(sel)
    out = [{"name": "rep", "sel": rep}]
    # capacity fallback (observed by the driver, never predicted): drop the
    # plain-SWA layer and the candidate-masked indexer, keep every source/consumer pair
    keep = {engram[0] if engram else rep[0]} | kv_src | {cand}
    small = sorted(i for i in rep if i in keep or (i - 1) in keep)
    if len(small) < len(rep):
        out.append({"name": "rep_min", "sel": small})
    return out


def apply_dsv41(cfg: dict, var: dict, edits: list[str]) -> None:
    tc = cfg["text_config"]
    n = tc["num_hidden_layers"]
    sel = var["sel"]
    remap = {o: i for i, o in enumerate(sel)}
    ratios = [int(r) for r in tc["compress_ratios"]]
    del tc["compress_ratios"]  # main + MTP tail; sliced explicitly (tail goes with the MTP heads)
    _slice_layer_lists(tc, n, sel, edits)
    tc["compress_ratios"] = [ratios[i] for i in sel]
    edits.append(f"compress_ratios -> {tc['compress_ratios']} (MTP tail dropped with num_nextn_predict_layers)")
    for key in ("kv_source_layer_ids", "index_source_layer_ids"):
        kept = [remap[i] for i in tc.get(key) or [] if i in remap]
        edits.append(f"{key} -> {kept}")
        tc[key] = kept
    cand = int(tc.get("candidate_source_layer_id", -1))
    if cand >= 0:
        tc["candidate_source_layer_id"] = remap.get(cand, -1)
        edits.append(f"candidate_source_layer_id -> {tc['candidate_source_layer_id']}")
    _cut_engram(tc, remap, edits)
    tc["num_hidden_layers"] = len(sel)
    # structural invariants the frameworks assert at load (cited above): every
    # compressed layer has a kv source AND an index source at or below it, and
    # nothing after the last kv source compresses on its own (sglang
    # models/deepseek_v4.py:4316-4322 late_layer_start).
    for new, r in enumerate(tc["compress_ratios"]):
        if r > 0:
            if not any(s <= new for s in tc["kv_source_layer_ids"]) or \
                    not any(s <= new for s in tc["index_source_layer_ids"]):
                raise SystemExit(f"dsv41 cut {sel}: layer {sel[new]} (ratio {r}) has no source in the cut")
    dropped = [k for k in list(tc) if k.startswith("dspark_")]
    for k in dropped:
        del tc[k]
    if dropped:
        edits.append(f"dropped {dropped} (precedent: nvidia NVFP4 configs ship without dspark_*)")
    if tc.get("num_nextn_predict_layers"):
        tc["num_nextn_predict_layers"] = 0
        edits.append("num_nextn_predict_layers -> 0")
    if tc.get("first_k_dense_replace"):
        tc["first_k_dense_replace"] = sum(1 for i in sel if i < tc["first_k_dense_replace"])
        edits.append(f"first_k_dense_replace -> {tc['first_k_dense_replace']}")


def variants_glm(cfg: dict) -> list[dict]:
    idx_types = cfg["indexer_types"]
    mlp_types = cfg["mlp_layer_types"]
    out = []
    for indexer in ("full", "shared"):
        sel = [i for i, (a, b) in enumerate(zip(idx_types, mlp_types))
               if a == indexer and b == "sparse"][:2]
        if sel:
            out.append({"name": f"{indexer}_indexer_moe", "sel": sel})
    return out


def apply_glm(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    _slice_layer_lists(cfg, n, var["sel"], edits)
    cfg["num_hidden_layers"] = len(var["sel"])
    if cfg.get("first_k_dense_replace"):
        cfg["first_k_dense_replace"] = 0
        edits.append("first_k_dense_replace -> 0 (dense head dropped)")


def variants_m3(cfg: dict) -> list[dict]:
    tc = cfg["text_config"]
    moe = tc["moe_layer_freq"]
    out = []
    # The sparse variant keeps the checkpoint's dense head (layers 0..2) in front of the first sparse/MoE
    # layer. TRT-LLM's MiniMaxM3KVCacheManagerV2 does not read sparse_attention_freq: it allocates the
    # index-K side cache for `range(3, num_layers)` by checkpoint convention (sparse/minimax_m3/
    # cache_manager.py:275-303 @1.3.0rc29) while the model layer follows the config list, so a cut whose
    # sparse layers sit at 0..1 serves a sparse layer with no index cache -> "MiniMaxM3SparseRuntimeBackend
    # .forward requires ... idx_k_cache" on every SM (sm90 2026-10-01 and sm89 2026-10-04 both read it as a
    # framework gap). Same class as _ARCH_IMPLICIT_PERIODS: a framework constant the config cannot express.
    head = [i for i, f in enumerate(moe) if f == 0]
    sparse = [i for i, f in enumerate(moe) if f == 1]
    sel = (head[:3] if head[:3] == [0, 1, 2] else []) + sparse[:1]
    if sparse:
        out.append({"name": "moe_sparse_attn", "sel": sel})
    head = [i for i, f in enumerate(moe) if f == 0][:2]
    if head:
        out.append({"name": "dense_full_attn_head", "sel": head})
    return out


def apply_m3(cfg: dict, var: dict, edits: list[str]) -> None:
    tc = cfg["text_config"]
    n = tc["num_hidden_layers"]
    _slice_layer_lists(tc, n, var["sel"], edits, prefix="text_config.")
    tc["num_hidden_layers"] = len(var["sel"])
    for k in ("num_mtp_modules", "num_nextn_predict_layers"):
        if tc.get(k):
            tc[k] = 0
            edits.append(f"text_config.{k} -> 0")


def variants_gptoss(cfg: dict) -> list[dict]:
    lt = cfg["layer_types"]  # alternating sliding_attention / full_attention
    out = []
    for kind in ("sliding_attention", "full_attention"):
        sel = [i for i, t in enumerate(lt) if t == kind][:2]
        if sel:
            out.append({"name": kind, "sel": sel})
    if len(lt) > 1 and lt[0] != lt[1]:
        out.append({"name": "interleave_pair", "sel": [0, 1]})
    return out


def apply_gptoss(cfg: dict, var: dict, edits: list[str]) -> None:
    n = cfg["num_hidden_layers"]
    _slice_layer_lists(cfg, n, var["sel"], edits)
    cfg["num_hidden_layers"] = len(var["sel"])



_PERIOD_FIELDS = ("full_attention_interval", "attention_interval",
                  "linear_attention_interval", "moe_layer_interval")

# Some frameworks hardcode a layer-kind period by layer_id with NO config
# field to read it from. Declared here with the source citation; the dummy
# must cut to a whole period (rule 5) even though the config can't say so.
# sglang llama4.py:217 @0.5.16: use_rope = (layer_id + 1) % 4 != 0
_ARCH_IMPLICIT_PERIODS = {
    "Llama4ForConditionalGeneration": 4,
    "Llama4ForCausalLM": 4,
}


def _min_depth_for_periods(tc: dict) -> int:
    """Minimum layer count that keeps a period-derived architecture faithful.

    Some models derive layer kinds from a PERIOD rather than a per-layer list
    (Qwen3.5: full attention where (i+1) %% full_attention_interval == 0).
    Rescaling the period to fit a 2-layer cut produces a configuration that
    does not exist upstream (interval=1 means every layer is full attention)
    and still trips capacity asserts (mamba_cache_per_req > 0). Keep the real
    period and cut to a whole number of periods instead: a dummy must be a
    SHORTENED model, never a MODIFIED one.
    """
    periods = [int(tc[f]) for f in _PERIOD_FIELDS
               if isinstance(tc.get(f), int) and tc[f] > 1]
    for arch in (tc.get("architectures") or []):
        if arch in _ARCH_IMPLICIT_PERIODS:
            periods.append(_ARCH_IMPLICIT_PERIODS[arch])
    if not periods:
        return 0
    import math
    step = math.lcm(*periods) if len(periods) > 1 else periods[0]
    return step * 2  # two full periods: exercises both kinds with real spacing


def _layer_axis(cfg: dict) -> tuple[str | None, list]:
    """Find the per-layer type list (name, values) if the model interleaves."""
    tc = cfg.get("text_config", cfg)
    n = tc.get("num_hidden_layers")
    # Kimi-K3 / Kimi-Linear declare the interleave as INDEX LISTS
    # (linear_attn_config.kda_layers / full_attn_layers) rather than a
    # per-layer type list; synthesize the axis so the representative cut
    # holds one layer of every kind and the lists get remapped (found
    # 2026-09-24: the 2-layer K3 dummy kept the original lists, so index 0
    # silently became a full-attention layer and index 1 a KDA layer).
    la = tc.get("linear_attn_config")
    if isinstance(la, dict) and n and (la.get("kda_layers") or la.get("full_attn_layers")):
        kda, full = set(la.get("kda_layers") or []), set(la.get("full_attn_layers") or [])
        # the lists are 1-based (is_kda_layer checks layer_idx + 1); detect from data
        base = 1 if (0 not in (kda | full) and max(kda | full) >= n) else 0
        values = ["kda" if i + base in kda else "full" if i + base in full else "dense" for i in range(n)]
        if len(set(values)) > 1:
            return "linear_attn_config", values
    # Inkling declares its sliding-window layers as an index list
    # (local_layer_ids, 55 of 66) with the rest global. Found 2026-10-01 by the
    # extended stale-ref check: the 2-layer rep cut kept all 55 indices verbatim
    # and never held a global layer.
    for key, v in tc.items():
        if (_LAYER_ID_KEY_RE.search(key) and isinstance(v, list) and n and v
                and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
                and 0 <= min(v) and max(v) < n and 0 < len(set(v)) < n):
            tag = key[:-len("_layer_ids")] if key.endswith("_layer_ids") else key
            members = set(v)
            return key, [tag if i in members else "other" for i in range(n)]
    for key in ("layer_types", "attn_type_list", "hybrid_layer_pattern",
                "layers_block_type", "indexer_types", "mlp_layer_types", "moe_layer_freq"):
        v = tc.get(key)
        if isinstance(v, list) and n and len(v) == n and len(set(map(str, v))) > 1:
            return key, v
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and n and len(pat) == n:
        return "hybrid_override_pattern", list(pat)
    return None, []


def variants_generic(cfg: dict) -> list[dict]:
    """Depth-cut variants for any architecture: one per layer kind on the
    detected interleave axis (plus a mixed pair), or a single 2-layer variant
    for homogeneous models. MoE models skip the leading dense block."""
    tc = cfg.get("text_config", cfg)
    if "num_hidden_layers" not in tc and isinstance(tc.get("layers_block_type"), list):
        tc["num_hidden_layers"] = len(tc["layers_block_type"])  # Nemotron-Ultra schema
    n = tc["num_hidden_layers"]
    skip = int(tc.get("first_k_dense_replace") or 0)
    if "architectures" not in tc and cfg.get("architectures"):
        tc = {**tc, "architectures": cfg["architectures"]}
    min_depth = _min_depth_for_periods(tc)
    if min_depth and min_depth <= n:
        # period-derived architecture: keep the real period, take whole periods.
        # Also emit a one-period variant as the capacity fallback for very wide
        # models (Llama-4-Maverick's 8 MoE layers x 128 experts exceed one GPU).
        out = [{"name": f"depth{min_depth}", "sel": list(range(min_depth))}]
        if min_depth % 2 == 0 and min_depth // 2 >= 2:
            half = min_depth // 2
            out.append({"name": f"depth{half}", "sel": list(range(half))})
        return out
    axis, values = _layer_axis(cfg)
    if not axis:
        sel = list(range(skip, min(skip + 2, n))) or list(range(min(2, n)))
        return [{"name": "rep", "sel": sel}]
    kinds: list[str] = []
    for i, v in enumerate(values):
        if i >= skip and str(v) not in kinds:
            kinds.append(str(v))
    out = []
    for k in kinds[:4]:
        sel = [i for i, v in enumerate(values) if str(v) == k and i >= skip][:2]
        if sel:
            safe = "".join(ch if ch.isalnum() else "_" for ch in k)[:20]
            out.append({"name": f"{axis}_{safe}"[:40], "sel": sel})
    # A variant holding only ONE layer kind is often structurally illegal:
    # hybrid-SWA models assert "at least one SWA layer", mamba/GDN hybrids
    # divide by the attention-layer count, vllm's kv grouping needs every
    # kv-spec kind. So the FULL-COVERAGE variant (one layer of every kind)
    # comes first and single-kind variants are kept only as extras.
    cover: list[int] = []
    for k in kinds:
        idx = [i for i, v in enumerate(values) if str(v) == k and i >= skip]
        if idx:
            cover.append(idx[0])
    if len(cover) > 1:
        out.insert(0, {"name": "all_kinds", "sel": sorted(cover)})
    return out


def apply_generic(cfg: dict, var: dict, edits: list[str]) -> None:
    tc = cfg.get("text_config", cfg)
    n = tc["num_hidden_layers"]
    sel = var["sel"]
    pat = tc.get("hybrid_override_pattern")
    if isinstance(pat, str) and len(pat) == n:
        tc["hybrid_override_pattern"] = "".join(pat[i] for i in sel)
        edits.append(f"hybrid_override_pattern -> {tc['hybrid_override_pattern']}")
    _slice_layer_lists(tc, n, sel, edits)
    _remap_layer_index_lists(tc, n, sel, edits)
    tc["num_hidden_layers"] = len(sel)
    if tc.get("first_k_dense_replace"):
        tc["first_k_dense_replace"] = 0
        edits.append("first_k_dense_replace -> 0")
    for k in ("num_nextn_predict_layers", "num_mtp_modules"):
        if tc.get(k):
            tc[k] = 0
            edits.append(f"{k} -> 0")


ADAPTERS = {
    "dsv4": (variants_dsv4, apply_dsv4),
    "dsv41": (variants_dsv41, apply_dsv41),
    "glm": (variants_glm, apply_glm),
    "m3": (variants_m3, apply_m3),
    "gptoss": (variants_gptoss, apply_gptoss),
    "generic": (variants_generic, apply_generic),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, required=True, help="dir of <org>_<repo>.json originals")
    ap.add_argument("--out", type=Path, required=True, help="output root for dummy model dirs")
    args = ap.parse_args()

    manifest = {"generator": Path(__file__).name, "rule": "depth-only cut, width preserved", "variants": []}
    failures = 0
    for repo, family in load_repos(args.configs).items():
        src = args.configs / (repo.replace("/", "_") + ".json")
        if not src.exists():
            print(f"MISSING {src}", file=sys.stderr)
            failures += 1
            continue
        base = json.loads(src.read_text())
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        make_variants, apply = ADAPTERS[family]
        for var in make_variants(base):
            cfg = copy.deepcopy(base)
            edits: list[str] = []
            apply(cfg, var, edits)
            for k in ("num_nextn_predict_layers",):
                if cfg.get(k):
                    cfg[k] = 0
                    edits.append(f"{k} -> 0")
            _remap_quant_layer_entries(cfg, var["sel"], edits)
            if repo in _DROP_AUTO_MAP and cfg.pop("auto_map", None) is not None:
                edits.append(f"auto_map removed: {_DROP_AUTO_MAP[repo]}")
            new_n = cfg.get("num_hidden_layers") or cfg["text_config"]["num_hidden_layers"]
            stale = _check_no_stale_layer_refs(cfg, new_n)
            tag = f"{repo.split('/')[-1]}__{var['name']}"
            out_dir = args.out / family / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
            caveats: list[str] = []
            _provision_aux_files(out_dir, repo, args.configs, args.out, edits, caveats)
            _dtp = args.configs / "dsv4_expert_dtypes.json"
            if _dtp.exists():
                _dt = json.loads(_dtp.read_text()).get(repo)
                if _dt and _dt.get("dtype"):
                    write_dtype_probe_safetensors(out_dir, _dt["key"], _dt["dtype"], edits)
            # modelopt/NVFP4 repos carry the authoritative quant description in
            # a SEPARATE hf_quant_config.json; without it the framework loads
            # the checkpoint as unquantized (looked like a silent-downgrade bug
            # until the missing file was found). Remap its layer refs too.
            hq_src = args.configs / (repo.replace("/", "_") + "_hfquant.json")
            if hq_src.exists():
                hq = json.loads(hq_src.read_text())
                sib = _HFQUANT_COMPLETE_FROM_SIBLING.get(repo)
                if sib and "exclude_modules" not in (hq.get("quantization") or {}):
                    sib_hq = json.loads((args.configs / (sib.replace("/", "_") + "_hfquant.json")).read_text())
                    hq.setdefault("quantization", {})["exclude_modules"] = \
                        sib_hq["quantization"]["exclude_modules"]
                    edits.append(f"hf_quant exclude_modules completed from sibling {sib}")
                _remap_quant_layer_entries({"quantization_config": hq.get("quantization", hq)},
                                           var["sel"], edits)
                (out_dir / "hf_quant_config.json").write_text(json.dumps(hq, indent=2))
                edits.append("hf_quant_config.json copied + layer refs remapped")
            entry = {
                "variant": tag,
                "repo": repo,
                "family": family,
                "layer_kind": var["name"],
                "original_layer_indices": var["sel"],
                "num_layers": new_n,
                "source_config_sha256_16": src_sha,
                "edits": edits,
                "stale_layer_refs": stale,
                "caveats": caveats,
            }
            if family == "glm":
                freq, off = base.get("index_topk_freq"), base.get("index_skip_topk_offset")
                entry["caveats"].append(
                    f"index_topk_freq={freq}/offset={off} are phase-based on absolute layer index; "
                    f"original indices {var['sel']} remap to 0..{new_n - 1}, so topk phase may differ "
                    "from the full model — cross-check the indexer topk path on the real depth once."
                )
            if stale:
                print(f"STALE LAYER REFS in {tag}: {stale}", file=sys.stderr)
                failures += 1
            manifest["variants"].append(entry)
            print(f"wrote {out_dir}  layers={new_n}  orig={var['sel']}")

    mpath = args.out / "variants_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {mpath}  ({len(manifest['variants'])} variants, {failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
