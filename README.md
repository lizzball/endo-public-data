# Endometriosis Biomarker Signal Lab

A reproducible public-data case study using **GEO GSE279435**, a serum miRNA RT-qPCR cohort of 67 patients with endometriosis and 60 controls.

The goal is not to claim a clinical biomarker. The project demonstrates a leakage-aware workflow for small-n, high-dimensional translational datasets:

- public GEO ingestion and provenance tracking
- normalized −ΔCt matrix parsing
- feature detection QC fit within training folds only
- median imputation and scaling inside the modelling pipeline
- nested cross-validation for elastic-net logistic regression
- out-of-fold ROC/AUPRC and bootstrap confidence intervals
- threshold sensitivity / confusion-matrix exploration
- biomarker selection stability across outer folds
- exploratory subtype-level error review

## Data

NCBI GEO: **GSE279435**  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279435

The GEO record was public on 2024-10-30 and lists a last update of 2026-01-23. The dataset contains 127 serum samples: 67 endometriosis and 60 controls. The source study profiled a 754-miRNA TaqMan OpenArray panel and deposited normalized −ΔCt values in GEO.

Primary paper: Ravaggi A et al. *Circulating Serum Micro-RNA as Non-Invasive Diagnostic Biomarkers of Endometriosis.* Biomedicines. 2024.

Follow-up: Ravaggi A et al. *Serum miRNA-based diagnostic models for endometriosis: from discovery to validation.* Human Reproduction. 2026.

## Modelling strategy

The primary model is an **elastic-net logistic regression**. The workflow intentionally differs from the publication's random-forest/RFE model so this repository is an independent methods reanalysis rather than a reconstruction of the paper.

For each outer CV fold:

1. retain miRNAs detected in at least 75% of the **training fold**;
2. median-impute missing values using the training fold;
3. standardize using the training fold;
4. tune elastic-net `C` and `l1_ratio` using inner stratified CV;
5. generate probabilities only for the held-out outer fold.

The displayed AUROC and average precision are calculated from pooled **out-of-fold predictions**. Bootstrap intervals are descriptive internal-validation intervals, not external-validation uncertainty.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_analysis.py
streamlit run app.py
```

The analysis script downloads the public GEO series matrix on first run and writes derived model artifacts to `artifacts/`. The raw GEO file is intentionally not committed.

## Important limitations

This is a small retrospective dataset. Controls have benign gynecologic conditions, so performance does not directly represent population screening. Internal nested CV does not establish external clinical validity, and PPV/NPV depend on disease prevalence. This repository is a reproducible portfolio and methods demonstration only, **not a diagnostic tool**.
