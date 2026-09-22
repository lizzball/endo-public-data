from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Endometriosis Biomarker Signal Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {max-width: 1220px; padding-top: 2.2rem; padding-bottom: 4rem;}
.hero {background:#211a20; padding:38px 42px 40px; border-radius:0 0 26px 26px; color:#fff; margin-bottom:16px;}
.hero-kicker {font-size:.78rem; font-weight:700; letter-spacing:.02em; opacity:.95;}
.hero-title {font-family: Georgia, 'Times New Roman', serif; font-size:3.55rem; line-height:1.02; font-weight:700; margin:24px 0 22px;}
.hero-sub {font-size:1rem; line-height:1.55; max-width:980px;}
.summary-box {border-left:4px solid #8f2336; padding:14px 18px; background:rgba(143,35,54,.045); margin:10px 0 16px;}
.small-note {font-size:.86rem; opacity:.75; line-height:1.55;}
[data-testid="stMetric"] {border:1px solid rgba(120,120,120,.22); padding:15px 16px; border-radius:16px;}
[data-baseweb="tab-list"] {gap:1.15rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">PUBLIC ENDOMETRIOSIS SERUM miRNA MODEL · GSE279435</div>
  <div class="hero-title">Endometriosis Biomarker Signal Lab</div>
  <div class="hero-sub">Nested-CV serum miRNA model with train-fold feature QC, threshold sensitivity, biomarker stability and subtype-level error review.</div>
</div>
""",
    unsafe_allow_html=True,
)

required = [
    "analysis_summary.json",
    "oof_predictions.csv",
    "fold_metrics.csv",
    "feature_stability.csv",
    "roc_points.csv",
    "pr_points.csv",
    "calibration_points.csv",
]
missing = [name for name in required if not (ART / name).exists()]
if missing:
    st.error(
        "Model artifacts have not been generated yet. Run "
        "`python scripts/run_analysis.py` or let the repository's GitHub "
        "Action build them. Missing: " + ", ".join(missing)
    )
    st.stop()

summary = json.loads((ART / "analysis_summary.json").read_text())
oof = pd.read_csv(ART / "oof_predictions.csv")
folds = pd.read_csv(ART / "fold_metrics.csv")
stability = pd.read_csv(ART / "feature_stability.csv")
roc_points = pd.read_csv(ART / "roc_points.csv")
pr_points = pd.read_csv(ART / "pr_points.csv")
calibration = pd.read_csv(ART / "calibration_points.csv")

st.markdown(
    f"""
<div class="summary-box"><b>Analysis summary.</b> This reproducible reanalysis uses {summary['n_subjects']} serum samples
({summary['n_endometriosis']} endometriosis, {summary['n_control']} controls) from GEO GSE279435. The public values are normalized −ΔCt.
Feature detection filtering, median imputation, scaling and model tuning occur inside the training data of each cross-validation split.
This is a methods case study, not a validated clinical diagnostic.</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Biological subjects", summary["n_subjects"])
c2.metric("OOF AUROC", f"{summary['oof_roc_auc']:.3f}")
auc_ci = summary["oof_roc_auc_ci_95"]
c3.metric("Bootstrap 95% CI", f"{auc_ci[0]:.2f}–{auc_ci[1]:.2f}")
c4.metric("Avg precision", f"{summary['oof_average_precision']:.3f}")

validation_tab, decision_tab, stability_tab, subtype_tab, data_tab = st.tabs(
    [
        "Model validation",
        "Decision layer",
        "Biomarker stability",
        "Subtype review",
        "Data & limitations",
    ]
)

with validation_tab:
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=roc_points["fpr"],
                y=roc_points["tpr"],
                mode="lines",
                name="OOF ROC",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title="Out-of-fold ROC",
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
            height=410,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pr_points["recall"],
                y=pr_points["precision"],
                mode="lines",
                name="OOF PR",
            )
        )
        prevalence = summary["n_endometriosis"] / summary["n_subjects"]
        fig.add_hline(
            y=prevalence,
            line_dash="dash",
            annotation_text="Prevalence baseline",
        )
        fig.update_layout(
            title="Precision–recall curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=410,
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Outer-fold performance")
        st.dataframe(
            folds[
                [
                    "fold",
                    "n_test",
                    "roc_auc",
                    "average_precision",
                    "eligible_features",
                    "best_C",
                    "best_l1_ratio",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=calibration["mean_predicted_probability"],
                y=calibration["observed_fraction_positive"],
                mode="lines+markers",
                name="OOF calibration",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Ideal",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Calibration · Brier score {summary['brier_score']:.3f}",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed endometriosis fraction",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

with decision_tab:
    st.subheader("Threshold sensitivity")
    threshold = st.slider(
        "Intervention / review threshold", 0.05, 0.95, 0.50, 0.01
    )
    y_true = oof["y_true"].to_numpy(dtype=int)
    y_prob = oof["oof_probability"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sensitivity", f"{sensitivity:.1%}")
    m2.metric("Specificity", f"{specificity:.1%}")
    m3.metric("PPV", f"{ppv:.1%}")
    m4.metric("NPV", f"{npv:.1%}")

    left, right = st.columns([0.8, 1.2])
    with left:
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual control", "Actual END"],
            columns=["Pred control", "Pred END"],
        )
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"Confusion matrix · threshold {threshold:.2f}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        hist = oof.copy()
        hist["Outcome"] = hist["group"]
        fig = px.histogram(
            hist,
            x="oof_probability",
            color="Outcome",
            barmode="overlay",
            opacity=0.65,
            nbins=20,
            title="OOF probability distribution",
        )
        fig.add_vline(x=threshold, line_dash="dash")
        fig.update_xaxes(title="Predicted probability of endometriosis")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "PPV and NPV here reflect this study's roughly balanced case/control "
        "sample, not population prevalence."
    )

with stability_tab:
    st.subheader("Features retained by elastic-net across outer folds")
    show = stability.query("selection_frequency > 0").head(20).copy()
    if show.empty:
        st.info(
            "No features had non-zero coefficients across the fitted "
            "outer-fold models."
        )
    else:
        show["direction"] = np.where(
            show["median_coefficient"] >= 0,
            "Higher END probability",
            "Lower END probability",
        )
        fig = px.bar(
            show.sort_values("selection_frequency"),
            x="selection_frequency",
            y="feature",
            orientation="h",
            hover_data=[
                "median_coefficient",
                "mean_abs_coefficient",
                "folds_eligible",
                "direction",
            ],
            title="Selection frequency across outer folds",
        )
        fig.update_xaxes(tickformat=".0%", title="Selection frequency")
        fig.update_yaxes(title=None)
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "A non-zero elastic-net coefficient is counted as selected. "
            "Stability here measures reproducibility across the five outer "
            "training folds; it is not evidence that a marker is clinically validated."
        )

with subtype_tab:
    st.subheader("Where does the model struggle?")
    review = oof.copy()
    review["display_group"] = np.where(
        review["group"].eq("Control"),
        "Control",
        review["disease"].replace("", "Endometriosis"),
    )
    counts = review["display_group"].value_counts()
    keep_groups = counts[counts >= 3].index
    plot_df = review[review["display_group"].isin(keep_groups)].copy()
    order = (
        plot_df.groupby("display_group")["oof_probability"]
        .median()
        .sort_values()
        .index.tolist()
    )
    fig = px.box(
        plot_df,
        x="oof_probability",
        y="display_group",
        points="all",
        category_orders={"display_group": order},
        title="Out-of-fold probabilities by disease subtype / control",
    )
    fig.update_xaxes(title="Predicted probability of endometriosis")
    fig.update_yaxes(title=None)
    fig.update_layout(height=max(420, 44 * len(order)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Subtype views are exploratory and can be very small; they were not "
        "used as the primary modelling target."
    )

with data_tab:
    st.subheader("Dataset provenance")
    st.markdown(
        """
- **GEO:** [GSE279435](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279435), public since October 2024 and last updated January 23, 2026.
- **Cohort:** 67 endometriosis and 60 control serum samples collected prior to surgery.
- **Assay:** TaqMan OpenArray Human Advanced miRNA Panel; the public series matrix contains normalized **−ΔCt** values.
- **Primary source paper:** Ravaggi et al., *Biomedicines* (2024), "Circulating Serum Micro-RNA as Non-Invasive Diagnostic Biomarkers of Endometriosis."
- **Follow-up validation paper:** Ravaggi et al., *Human Reproduction* (2026), "Serum miRNA-based diagnostic models for endometriosis: from discovery to validation."
"""
    )

    st.subheader("What this model does differently")
    st.markdown(
        """
The portfolio model uses an **elastic-net logistic regression** rather than trying to reproduce the authors' random-forest/RFE pipeline. Detection filtering (≥75%), median imputation, scaling and hyperparameter selection are all fitted **inside training folds**. The displayed ROC/AUPRC use held-out out-of-fold predictions from a 5-fold outer CV with a 4-fold inner tuning loop.
"""
    )

    st.subheader("Limitations")
    st.markdown(
        """
This is a small retrospective cohort, and the controls are women with benign gynecologic conditions rather than a population-screening sample. Internal cross-validation does not substitute for independent external validation. Threshold-dependent PPV/NPV are prevalence-sensitive. The source study performed hemolysis QC before inclusion, but this reanalysis does not have an independent pre-analytic cohort in which to test robustness to collection, site or assay shifts. The model is for reproducible methods demonstration only and should not be used for patient-level diagnosis.
"""
    )

st.divider()
st.markdown(
    "<div class='small-note'>Built as a public-data translational modelling "
    "case study. Source data: NCBI GEO GSE279435.</div>",
    unsafe_allow_html=True,
)
