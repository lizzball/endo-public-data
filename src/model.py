from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DetectionFilter(BaseEstimator, TransformerMixin):
    """Keep features detected in at least min_fraction of training samples.

    Because this transformer lives inside the sklearn Pipeline, it is re-fit only
    on training data in every inner and outer CV split.
    """

    def __init__(self, min_fraction: float = 0.75):
        self.min_fraction = min_fraction

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        fractions = X.notna().mean(axis=0)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.features_ = np.asarray(
            fractions[fractions >= self.min_fraction].index, dtype=object
        )
        if len(self.features_) == 0:
            raise ValueError("DetectionFilter removed all features.")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.loc[:, self.features_]

    def get_feature_names_out(self, input_features=None):
        return self.features_


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("detect", DetectionFilter(min_fraction=0.75)),
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=10000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def parameter_grid() -> Dict[str, Iterable[float]]:
    return {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }


def threshold_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_ci(y_true, y_prob, metric="roc_auc", n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "roc_auc":
            score = roc_auc_score(yt, yp)
        elif metric == "average_precision":
            score = average_precision_score(yt, yp)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores.append(score)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


@dataclass
class NestedCVResults:
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    feature_stability: pd.DataFrame
    summary: Dict[str, object]
    roc_points: pd.DataFrame
    pr_points: pd.DataFrame
    calibration_points: pd.DataFrame


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
    n_boot: int = 2000,
) -> NestedCVResults:
    y = pd.Series(y, index=X.index).astype(int)
    outer = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1
    )

    oof_prob = pd.Series(index=X.index, dtype=float)
    fold_rows = []
    coefficient_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search = GridSearchCV(
            estimator=build_pipeline(random_state=random_state + fold),
            param_grid=parameter_grid(),
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        prob = search.predict_proba(X_test)[:, 1]
        oof_prob.iloc[test_idx] = prob

        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "roc_auc": roc_auc_score(y_test, prob),
                "average_precision": average_precision_score(y_test, prob),
                "best_C": search.best_params_["model__C"],
                "best_l1_ratio": search.best_params_["model__l1_ratio"],
                "eligible_features": len(
                    search.best_estimator_.named_steps["detect"].features_
                ),
            }
        )

        model = search.best_estimator_.named_steps["model"]
        features = search.best_estimator_.named_steps["detect"].features_
        for feature, coef in zip(features, model.coef_.ravel()):
            coefficient_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "coefficient": float(coef),
                    "selected": int(abs(coef) > 1e-8),
                }
            )

    if oof_prob.isna().any():
        raise RuntimeError("Some samples did not receive an out-of-fold prediction.")

    oof = meta.loc[X.index, ["title", "group", "disease"]].copy()
    oof["y_true"] = y
    oof["oof_probability"] = oof_prob
    oof = oof.reset_index()

    coef_df = pd.DataFrame(coefficient_rows)
    if coef_df.empty:
        feature_stability = pd.DataFrame(
            columns=[
                "feature",
                "selection_frequency",
                "median_coefficient",
                "mean_abs_coefficient",
            ]
        )
    else:
        feature_stability = (
            coef_df.groupby("feature")
            .agg(
                selected_folds=("selected", "sum"),
                median_coefficient=("coefficient", "median"),
                mean_abs_coefficient=(
                    "coefficient",
                    lambda s: np.mean(np.abs(s)),
                ),
                folds_eligible=("fold", "nunique"),
            )
            .reset_index()
        )
        feature_stability["selection_frequency"] = (
            feature_stability["selected_folds"] / outer_splits
        )
        feature_stability["eligibility_frequency"] = (
            feature_stability["folds_eligible"] / outer_splits
        )
        feature_stability = feature_stability.sort_values(
            ["selection_frequency", "mean_abs_coefficient"],
            ascending=False,
        )

    auc = roc_auc_score(y, oof_prob)
    ap = average_precision_score(y, oof_prob)
    auc_ci = bootstrap_metric_ci(
        y, oof_prob, "roc_auc", n_boot=n_boot, seed=random_state
    )
    ap_ci = bootstrap_metric_ci(
        y,
        oof_prob,
        "average_precision",
        n_boot=n_boot,
        seed=random_state + 11,
    )
    base_metrics = threshold_metrics(y, oof_prob, threshold=0.5)

    fpr, tpr, roc_thr = roc_curve(y, oof_prob)
    precision, recall, _ = precision_recall_curve(y, oof_prob)
    frac_pos, mean_pred = calibration_curve(
        y, oof_prob, n_bins=6, strategy="quantile"
    )

    summary = {
        "n_subjects": int(len(y)),
        "n_endometriosis": int(y.sum()),
        "n_control": int((1 - y).sum()),
        "n_input_features": int(X.shape[1]),
        "oof_roc_auc": float(auc),
        "oof_roc_auc_ci_95": [float(auc_ci[0]), float(auc_ci[1])],
        "oof_average_precision": float(ap),
        "oof_average_precision_ci_95": [float(ap_ci[0]), float(ap_ci[1])],
        "brier_score": float(brier_score_loss(y, oof_prob)),
        "threshold_0_5": base_metrics,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_state": random_state,
        "feature_qc": "Detection >=75% within each training fold only",
        "model": "Elastic-net logistic regression",
    }

    roc_points = pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": roc_thr}
    )
    pr_points = pd.DataFrame(
        {"recall": recall, "precision": precision}
    )
    calibration_points = pd.DataFrame(
        {
            "mean_predicted_probability": mean_pred,
            "observed_fraction_positive": frac_pos,
        }
    )

    return NestedCVResults(
        oof=oof,
        fold_metrics=pd.DataFrame(fold_rows),
        feature_stability=feature_stability,
        summary=summary,
        roc_points=roc_points,
        pr_points=pr_points,
        calibration_points=calibration_points,
    )
