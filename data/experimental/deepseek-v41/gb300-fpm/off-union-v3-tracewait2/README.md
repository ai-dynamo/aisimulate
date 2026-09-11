# GB300 TP4 whole-forward FPM union verification

Decoder replay **OFF**. [Seven-metric MAPE/WAPE report](report/README.md); [dual-source calibration](calibration/README.md).

Original heldout observations are reused; this is not a new test population. Core/field/service remain separate. Calibration-coordinate, observed and prediction coverage are distinct. Balanced aggregate interpolation does not establish equivalent heterogeneous-request geometry.

Native errors use individual recorded DeviceTimer intervals; HTTP errors use individual cohorts. The retained legacy region label does not mean another median reduction. All original failures, precision misses, output disagreements, surplus and scenario CIs remain in reports. Core ON keeps separate 73/26 complete-trial conditional lifecycle CIs and a descriptive split trial, with no pooled CI.

Tables and CSV contain seven metrics; the figure is a four-metric overview. MAPE and WAPE use the same supported pairs. Calibration self-queries never enter the accuracy tables or figures.

Run the predictor from the repository root using the relative systems_path. Reproduce presentation without private files or predictions: `python render_report.py --input-root . --output report-reproduced`. A fresh output is required.
