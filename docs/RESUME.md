# Resume entry

**Sentinel — Equipment Failure Prediction** | Python, scikit-learn, pandas, Streamlit, Docker, GitHub Actions

- Built a predictive maintenance pipeline on NASA C-MAPSS FD001 simulated engine telemetry, using causal rolling features and machine-disjoint training, validation and audit splits.
- Achieved 0.949 PR-AUC, 84% recall and 87.5% precision for failure within 30 operating cycles on 100 official test engines; estimated capped remaining useful life with 12.9-cycle MAE versus a 38.0-cycle median baseline.
- Implemented a Streamlit inference demo with historical replay, batch scoring, model sensitivity and simulated alerts with duplicate suppression and acknowledgement; added automated leakage and inference tests.

Use only after running and understanding the project. Add GitHub link after publishing. Add “deployed” and a public demo link only after the deployment is verified. Do not claim real-world downtime reduction, company use, printer/scanner validation, or internship employment.

## Short project description

An independent predictive-maintenance project inspired by logistics equipment support. Trained and evaluated failure-risk and remaining-life models on NASA's simulated engine benchmark, packaged in a reproducible inference demo.

## Metrics context

30-cycle positive = true RUL <=30. The selected threshold is 0.13 from validation-only cost tuning. Official test contains 25 positives and 75 negatives: 21 true positives, 3 false positives, 4 false negatives. PR-AUC prevalence baseline =0.25. RUL error uses targets capped at 125 cycles. Report results as simulation benchmark performance.
