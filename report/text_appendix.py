"""Appendices A-C."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _extract(path, start_marker, end_marker=None, max_lines=60):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if start_marker in ln)
    out = []
    for ln in lines[i:]:
        if end_marker and out and end_marker in ln:
            break
        out.append(ln.rstrip())
        if len(out) >= max_lines:
            break
    while out and not out[-1].strip():
        out.pop()
    return out


def appendices(R, pct, fmt_ms_rows, shap_rows):
    tree = [
        "ML_Diploma/",
        "  data/                     raw Parquet partitions and train_target.csv",
        "  pd_scoring/               shared Python package",
        "    config.py               paths, partition split, random seed",
        "    features.py             loan-level records -> application features",
        "    data.py                 memory-aware loading of feature partitions",
        "    metrics.py              AUC, Gini, KS, Hosmer-Lemeshow, PSI, binomial test",
        "    risk.py                 master scale, Vasicek stress, IRB capital, EL",
        "    labels.py               readable feature descriptions",
        "  scripts/",
        "    01_build_features.py    build 515 features for 3M applications",
        "    02_baseline_lgbm.py     Experiment 1: baseline reproduction",
        "    03_feature_selection.py Experiment 2: feature value and top-K selection",
        "    04_train_models.py      Experiment 3: LogReg, LightGBM, CatBoost (GPU)",
        "    05_validate.py          Experiment 4: OOT validation, calibration, PSI",
        "    06_shap.py              Experiment 5: SHAP analysis",
        "    07_stress_test.py       Experiment 6: stress testing",
        "    08_prepare_app_assets.py, 09_eda_figures.py, 10_result_figures.py",
        "  app/",
        "    app.py                  Streamlit user interface",
        "    scoring.py              model loading, scoring and explanation",
        "  artifacts/                features, models, results (JSON/CSV), figures",
        "  report/                   scripts that build this report",
    ]
    blocks = [
        ("h1", "APPENDIX A. PROJECT STRUCTURE AND SOURCE CODE"),
        ("p", "The project is organised as a small Python package with numbered scripts, so that every result in "
              "Chapter 4 can be reproduced by running the scripts in order. The structure is shown below."),
        ("code", tree),
        ("h2", "A.1 Feature builder (pd_scoring/features.py, extract)"),
        ("p", "The extract shows how the count features and the payment-history features are calculated with "
              "NumPy without creating a large one-hot matrix."),
        ("code", _extract("pd_scoring/features.py", "def build_features", "# attributes of the latest", 58)),
        ("h2", "A.2 Stress and capital formulas (pd_scoring/risk.py, extract)"),
        ("code", _extract("pd_scoring/risk.py", "def retail_correlation", "def lifetime_pd", 40)),
        ("h2", "A.3 CatBoost training (scripts/04_train_models.py, extract)"),
        ("code", _extract("scripts/04_train_models.py", "def run_catboost", "best_cb =", 30)),
        ("h2", "A.4 Population stability index and binomial test (pd_scoring/metrics.py, extract)"),
        ("code", _extract("pd_scoring/metrics.py", "def psi(", None, 30)),
        ("h1", "APPENDIX B. APPLICATION USER GUIDE"),
        ("p", "The application is started from the project folder with the command "
              "*streamlit run app/app.py* and opens in a web browser. It has four sections, selected in the "
              "sidebar."),
        ("p", "**Applicant assessment.** The user chooses a starting template and edits the table of credit "
              "products. For each product, the credit type and status codes, the 25-month payment history (one "
              "digit per month, latest first) and the overdue flags can be changed; rows can be added or "
              "deleted. The binned bureau fields are available in the expander \"Advanced bureau fields\". The "
              "result is updated immediately."),
        ("fig", "app_applicant_high.png", "Figure B.1. Applicant assessment for a high-risk applicant from the OOT sample", 6.0),
        ("p", "**Batch scoring.** A CSV or Parquet file with loan-level records can be uploaded. The application "
              "scores every applicant, shows the grade distribution and, if the file contains the target, the "
              "ROC AUC and the comparison of predicted and observed default rates by grade. The scores can be "
              "downloaded as a CSV file."),
        ("fig", "app_batch.png", "Figure B.2. Batch scoring of the bundled sample of 3,000 applicants", 6.0),
        ("p", "**Stress lab.** Sliders set the severity of the macro scenario, the LGD, the correlation and the "
              "size of a bottom-up payment shock. The page shows the stressed mean PD, expected loss, approval "
              "rate and grade migration, and solves the reverse stress test for a chosen loss limit."),
        ("fig", "app_stress.png", "Figure B.3. Stress lab", 6.0),
        ("p", "**Model card.** This page summarises the model, the validation results and the main limitations, "
              "so that a user can see how far the output can be trusted."),
        ("h1", "APPENDIX C. ADDITIONAL RESULTS"),
        ("table", "Table C.1. Master scale on the OOT sample with the binomial test",
         ["Grade", "PD range", "Applicants", "Share", "Defaults", "Mean PD", "ODR", "Binomial 99% limit", "Result"],
         fmt_ms_rows, [0.55, 1.0, 0.85, 0.6, 0.7, 0.65, 0.65, 0.8, 0.6], 8.5),
        ("table", "Table C.2. Twenty most important features of the final model (mean |SHAP|, OOT subsample)",
         ["#", "Feature", "Description", "Mean |SHAP|"],
         shap_rows, [0.3, 2.0, 2.9, 0.8], 8.5, ["center", "left", "left", "center"]),
    ]
    return blocks
