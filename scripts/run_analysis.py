from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_geo_dataset
from src.model import run_nested_cv


def main():
    parser = argparse.ArgumentParser(
        description="Run the GSE279435 nested-CV reanalysis."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer bootstrap resamples for CI checks.",
    )
    args = parser.parse_args()

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_geo_dataset(
        ROOT / "data" / "GSE279435_series_matrix.txt.gz"
    )
    results = run_nested_cv(
        X,
        y,
        meta,
        outer_splits=5,
        inner_splits=4,
        random_state=42,
        n_boot=300 if args.quick else 2000,
    )

    (artifacts / "analysis_summary.json").write_text(
        json.dumps(results.summary, indent=2)
    )
    results.oof.to_csv(artifacts / "oof_predictions.csv", index=False)
    results.fold_metrics.to_csv(
        artifacts / "fold_metrics.csv", index=False
    )
    results.feature_stability.to_csv(
        artifacts / "feature_stability.csv", index=False
    )
    results.roc_points.to_csv(artifacts / "roc_points.csv", index=False)
    results.pr_points.to_csv(artifacts / "pr_points.csv", index=False)
    results.calibration_points.to_csv(
        artifacts / "calibration_points.csv", index=False
    )

    print(json.dumps(results.summary, indent=2))


if __name__ == "__main__":
    main()
