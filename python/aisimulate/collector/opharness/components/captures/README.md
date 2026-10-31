# captures/ — how every verdict under results/pathdiff/ was produced

Each `*.py` here is ONE collector capture: it imports the collector's
in-process entry (or drives `op_smoke.py`) for a single op cell and is run
inside the framework image under `path_diff.py --capture`:

```
python3 components/path_diff.py --capture --out /work/facts/pathdiff/opcov_<name>.json \
    [--env AIC_DSA_CONTEXT_SEQ_LENS=4096] -- /work/.../captures/<name>.py
```

`verdicts_<fw>.sh` re-derives the gate files for one framework from the
captures and the serving records (`archive/records.jsonl`, or a raw probe
JSON via `--serving-raw` for evidence outside the plan). They need
`AIS_PROBE_WORKSPACE` (the probe workspace root: `facts/`, `archive/`) and
`AIS_SM` (default sm90).

Rules learned the hard way (each has a finding in `results/findings.yaml`):

- Capture ONE cell at the serving record's isl / kv dtype / prefix. A whole
  sweep unions length-conditional paths (sglang DSA: dense FA3 below the
  dense threshold, indexer + sparse above) and never matches one record.
  Use the collectors' cell filters (`--env AIC_DSA_CONTEXT_*`, op_smoke
  `--case-index` / `--case-prefix`); the capture and the verdict record them.
- Capture precision must match serving (fp8_block vs bfloat16 GEMM changes
  the fused-a projection kernel).
- Explained divergences (bf16-weight MLA cells with no serving artifact) are
  written next to the facts, never into the gate directory.
- The sglang module collectors run each case in a subprocess; capture their
  in-process entry, or the parent profiler sees nothing.
