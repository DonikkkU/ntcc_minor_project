# Credit Risk Assessment for Loan Default Prediction

Term paper (NTCC), B.Sc. Information Technology, Amity University in Tashkent.
Author: Doniyor Umarov. Faculty guide: Mr Rajabov Javokhirbek.

The project builds a probability-of-default (PD) model for retail loan applicants from their credit bureau
histories, validates it out of time, explains it with SHAP, stress-tests the portfolio, and serves the
model in a small web application.

- Report: [`report/NTCC_report_final.pdf`](report/NTCC_report_final.pdf) (Word version: `report/NTCC_report_final.docx`)
- Data: [Google Drive folder](https://drive.google.com/drive/folders/1t55_IpGozKeYocJcxBtJ2eLFY62-PHJN?usp=sharing),
  field descriptions in [`data_desc.md`](data_desc.md)

## Results in brief

The data hold 3 million applications, each with the list of loan products in the applicant's credit history; the default rate is 3.5%. Applications are
numbered in time order, so the models are trained on the oldest applications and tested on the newest ones.

| Split | Applications | Use |
|---|---|---|
| Train | 0 – 1,999,999 | fitting |
| Validation | 2,000,000 – 2,499,999 | early stopping, tuning |
| Out-of-time (OOT) test | 2,500,000 – 2,999,999 | final evaluation |

| Model (OOT test) | ROC AUC | Gini |
|---|---|---|
| Baseline LightGBM (original notebook features) | 0.7356 | 0.471 |
| Logistic regression | 0.7213 | 0.443 |
| LightGBM, new features | 0.7433 | 0.487 |
| **CatBoost `cb_d8_lr06_bern` (final)** | **0.7433** | **0.487** |

- The baseline reaches AUC 0.7586 on a random split but 0.7356 out of time: a random split overstates quality.
- Final model: KS 0.357, mean PD 3.37% against an observed default rate of 3.28%, PSI 0.0005.
  The gain over the baseline is +0.0077 AUC (bootstrap 95% CI 0.0061–0.0091).
- Master scale: 10 grades (R1–R10). R1–R5 are approved, R6–R7 go to manual review, R8–R10 are declined.
- Stress tests (Vasicek one-factor model): mean PD rises to 6.7% in a 1-in-20 year scenario and to 13.3% in
  a 1-in-1000 year scenario. A reverse stress test shows that the expected loss rate reaches the 5% limit in
  about a 1-in-26 year scenario.

## Repository layout

```
notebooks/
  PD_research.ipynb      the research: data analysis, feature building, Experiments 1-6
                         (baseline, feature selection, model comparison, validation, SHAP,
                         stress tests), every figure of the report and the files for the app
  Baseline.ipynb, Data_Loading_and_Analyse.ipynb, baseline_notebook.py, data_partition_loader.py
                         the original baseline notebooks and data loader the project started from
pd_scoring/      shared code used by the research notebook and the app: feature building, data loading,
                 metrics, risk formulas (master scale, Vasicek stress, IRB capital)
app/             Streamlit application (applicant scoring, batch scoring, stress lab, model card)
artifacts/       trained models, feature list, app files, result tables (results/) and figures (figures/)
report/          the final report (DOCX and PDF)
```

## How to run

Python 3.12. Training CatBoost needs a GPU (CUDA); everything else runs on CPU.

```powershell
pip install -r requirements.txt
```

**Application only.** The trained model and the app files are in the repository, so the app runs without
the raw data:

```powershell
streamlit run app/app.py
```

**Research notebook.** Put the data from the Google Drive folder in `data/` (`data/train_data/*.pq`,
`data/train_target.csv`) and run `notebooks/PD_research.ipynb` from top to bottom (all notebooks read the data from `../data/`). The notebook is saved with
all outputs, so it can also be read on GitHub without running it.

- With `RETRAIN = False` (default) it loads the trained models from `artifacts/models/` and recomputes every
  result and figure from them (about 20 minutes; building the features the first time adds about 15).
- With `RETRAIN = True` it trains all models from scratch (several hours).

Building the features needs about 14 GB of RAM; the data is loaded one partition at a time.

## Use of AI tools

An AI assistant (Claude, Anthropic) was used in this project, as disclosed in Section 3.10 of the report.
