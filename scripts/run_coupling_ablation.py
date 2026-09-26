"""Run the coupling ablation on the real featured dataset (SIH26082).

Read-only with respect to shipped artefacts: it trains two throwaway models in
memory and writes a JSON report under reports/. No model file is overwritten.
"""

import json
import sys

from ml.evaluation.ablation import run_coupling_ablation

result = run_coupling_ablation(
    data_path="data/processed/featured_dataset.csv",
    target="pm25",
    horizons=[6, 24],
    model_type="random_forest",
    output_path="reports/coupling_ablation_pm25.json",
)

print()
print("=" * 72)
print(result["summary"])
print("=" * 72)
print("group present:", result["group_present"])
print("group absent :", result["group_absent"])
print(json.dumps(result["delta"], indent=2))

with open("reports/coupling_ablation_pm25.json", encoding="utf-8") as fh:
    saved = json.load(fh)
print("saved keys:", sorted(saved.keys()))
sys.exit(0)
