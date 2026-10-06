# Understand and present the project

1. Read `sentinel/data.py`; explain why random row splitting leaks machine-specific patterns and why trailing windows must never see future telemetry.
2. Read `sentinel/train.py`; understand the label construction, RUL cap, PR-AUC and cost-based threshold. Explain how test offsets are used for evaluation but never features.
3. Reproduce metrics; compare to both baselines. Discuss why PR-AUC is useful and why accuracy alone can conceal missed failures.
4. Run the studio and replay an engine. Explain risk versus current fault status and cycles versus time.
5. Inspect sensitivity and permutation importance; explain why correlated features and unrealistic perturbations limit explanations.
6. Test duplicate alerts and acknowledgement. Identify what would change for persistent multi-user production operation.

## Two-minute demo

Start with the maintenance decision: which machine should be inspected before failure? Show 30-cycle risk and RUL on an unseen simulated engine. Rewind history and show predictions changing. Create and acknowledge an alert; show deduplication. Open Model evidence and compare baselines on 100 official test engines. Close by explaining the domain gap and what approved printer/scanner telemetry would be needed.

## Interview questions

- Why machine-disjoint evaluation? Nearby cycles from the same engine are highly dependent.
- Why no GPU or neural network? The tabular baseline is fast and reproducible; complexity requires measured benefit.
- What does a 90% risk score mean? An uncalibrated classifier score; calibrated likelihood requires a separate validation procedure.
- Can this prevent printer failures? Not as trained. It demonstrates a transferable pipeline, but a new domain needs new features and labels.
- What business impact was achieved? No real operational impact was measured; only benchmark prediction quality and simulated workflow are reported.
- What is the biggest next improvement? Domain data and robust generalization evaluation, followed by calibration and incident-level lead-time measurement.

Before listing the project, run it yourself and be able to explain these choices. Describe it as an independent project, not employment or internship experience.
