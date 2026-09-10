# GB300 TP4 whole-forward FPM

Decoder replay: **OFF**. [Independent accuracy report](report/README.md); [calibration-only provenance](calibration/README.md).

The three verification scopes remain separate; every error table includes MAPE and WAPE. Original segmented core-ON CIs and all failure/coverage data remain in reports/.

The configuration uses a repository-relative systems_path; run the predictor from the AISimulate checkout root.

To reproduce the figures and CSV without predictions or private files:

```bash
python render_report.py --input-root . --output report-reproduced
```

The renderer validates all frozen public input hashes and recomputes errors from the published comparison pairs. It requires a new output directory. derivation-receipt.json lists every approved path transformation and original source hash.
