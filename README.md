# Sentinel — Equipment Failure Prediction

[Live prediction demo](https://nihal-sentinel.streamlit.app/) · [GitHub repository](https://github.com/NRao0158/sentinel-predictive-maintenance)

An independent machine-learning portfolio project: predict failure within **30 operating cycles** and estimate remaining useful life from sensor history. Includes a Streamlit prediction studio, historical replay, batch inference, validation evidence, and a simulated alert lifecycle.

**Scope:** NASA C-MAPSS FD001 simulated turbofan engines. Inspired by logistics equipment maintenance, but not trained or validated on printers, scanners, or facility telemetry. No production integrations, live alerts, measured cost savings, or claimed internship experience.

## Run the demo

Python 3.12 recommended. From this repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The pretrained artifact and compact demo history are included. Demo startup needs no dataset download or model training. On WSL/Linux, activate with `source .venv/bin/activate`.

## Reproduce the experiment

```powershell
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python -m sentinel.train
python -m pytest -q
```

Training uses only 60 of the 100 official training engines; validation uses 20; an internal audit uses 20. The official test comprises 100 different engines, scored at each engine's final available cycle. Unit numbers overlap across official files but represent different engines. No random row split, future-looking rolling windows, failure indicators, or unit identity features. Actual failure-cycle rows are excluded during training and internal evaluation.

Classification compares a prevalence baseline, scaled logistic regression, and histogram gradient boosting. Selection uses validation PR-AUC. Alert threshold minimizes a stated validation cost assumption: missed positive = 5, false positive = 1. These are per-observation costs, not economic estimates. RUL regression compares a median baseline with gradient boosting; targets are capped at 125 cycles. Results and provenance are in `artifacts/metrics.json`.

## Repository map

- `sentinel/data.py`: schema validation, strictly causal rolling features, target construction.
- `sentinel/train.py`: deterministic machine split, baselines, model selection, threshold policy, metrics, artifacts.
- `app.py`: replay, scoring, model evidence, sensitivity, session-local alerts.
- `tests/`: causality, machine boundaries, bad input, alert deduplication, artifact integrity.
- `scripts/download_data.py`: NASA-linked PHM dataset retrieval.
- `docs/`: model card, architecture, deployment, learning guide, resume wording.
- `.github/workflows/ci.yml`: dataset download and tests on push/PR.

## Deploy

Use free Streamlit Community Cloud from a GitHub repository; see [deployment instructions](docs/DEPLOYMENT.md). Docker is also included. GPU hardware is unnecessary for these tabular models.

## Sources

A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Center of Excellence. [NASA repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). Source data is downloaded separately; model artifacts and a small simulation sample are included for the demo. NASA does not endorse this project. See [model card](docs/MODEL_CARD.md).
