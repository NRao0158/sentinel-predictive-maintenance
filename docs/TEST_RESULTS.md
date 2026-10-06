# Verification results

Validated on Windows / Python 3.12 with pinned dependencies.

Six tests passed: causal history invariance, per-engine feature isolation, invalid schema rejection, target and alert lifecycle behavior, machine split/artifact integrity, and Streamlit startup plus alert/replay interaction. Live local browser reviewed: title, navigation, replay controls, risk and RUL cards, sensitivity and alert action render without errors.

Experiment results: 100 official test engines, 25 positive within 30 cycles. Selected logistic classifier PR-AUC=0.948946, ROC-AUC=0.9792, precision=0.875, recall=0.84, F1=0.857143 at validation-selected threshold=0.13. Three false positives and four missed positives. Selected boosted RUL regressor MAE=12.887907 and RMSE=18.005104 cycles with targets capped at 125. Baseline median MAE=38.01.

Public hosting: deployed to https://nihal-sentinel.streamlit.app/ on October 6, 2026. Model loading and prediction studio verified in the browser with Python 3.12. Public deployment does not require the laptop to remain running.

Known issues: sandboxed Streamlit initialization stalled; verified outside sandbox on local loopback. Docker container runtime remains unverified. Engine benchmark does not validate real logistics device behavior. No open functional failures found in the executed tests.
