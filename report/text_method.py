"""Chapter 3: methodology. ``R`` is the dictionary of experiment results."""


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def ch3(R):
    e = R["eda"]
    lp = e["loans_per_app"]
    d6 = e["dr_by_worst6"]
    dr = R["dr_parts"]
    train_dr = sum(dr[i]["mean"] * dr[i]["size"] for i in range(8)) / sum(dr[i]["size"] for i in range(8))
    valid_dr = (dr[8]["mean"] + dr[9]["mean"]) / 2
    oot_dr = (dr[10]["mean"] + dr[11]["mean"]) / 2
    nf_all = R["n_features_all"]
    n_cnt = R["n_count_features"]
    K = R["K"]
    return [
        ("h1", "3. METHODOLOGY"),
        ("p", "This chapter explains how the work was carried out: the data, the preparation of features, the "
              "sampling design, the models, the metrics and the stress testing method. The description is "
              "detailed enough for the experiments to be repeated with the scripts listed in Appendix A."),
        ("h2", "3.1 Research Design"),
        ("p", "The project follows the usual sequence of a supervised learning study, adapted to the needs of "
              "credit risk. The loan-level records are first turned into one row per application. The "
              "applications are then split by time into training, validation and out-of-time samples. Several "
              "models are trained and compared, the best one is validated with the standard PD tests and "
              "explained with SHAP, and the resulting PDs are used for stress testing. The last step is a "
              "software application that applies the same feature builder and model to new input. The whole "
              "process is shown in Figure 3.1."),
        ("fig", "pipeline.png", "Figure 3.1. Overall pipeline of the project", 6.2),
        ("h2", "3.2 Dataset Description"),
        ("p", f"The data consists of anonymised credit bureau histories of {R['n_apps']:,} loan applications. "
              f"The features are stored in twelve Parquet files with {R['n_rows']:,} rows in total, where each "
              "row describes one credit product of one applicant. The target file contains one row per "
              "application with a binary flag that equals one if the applicant later defaulted. The applications "
              "are numbered so that a larger id means a later application date, and the products inside an "
              "application are numbered by opening date (field rn). The main facts about the dataset are given "
              "in Table 3.1."),
        ("table", "Table 3.1. Summary of the dataset",
         ["Parameter", "Value"],
         [["Applications (ids)", f"{R['n_apps']:,}"],
          ["Loan-level records", f"{R['n_rows']:,}"],
          ["Raw fields per record (excluding id, rn)", "59"],
          ["Credit products per applicant (partition 5)", f"mean {lp['mean']:.1f}, median {lp['50%']:.0f}, max {lp['max']:.0f}"],
          ["Defaults (flag = 1)", f"{R['n_defaults']:,} ({pct(R['default_rate'])})"],
          ["Applications per partition", "250,000 (partitions 0-11)"],
          ["Default definition", "more than 90 days past due"]],
         [3.2, 2.8]),
        ("p", "The 59 raw fields fall into five groups, summarised in Table 3.2. All amounts, terms and ratios "
              "were converted by the data provider into bin codes from 0 to 19, and the categorical fields were "
              "replaced by integer codes. This protects the privacy of the clients but also removes the original "
              "meaning of the values, which has consequences that are discussed in Section 3.3."),
        ("table", "Table 3.2. Groups of raw fields in the loan-level records",
         ["Group", "Fields", "Content"],
         [["Dates and terms", "pre_since_opened, pre_since_confirmed, pre_pterm, pre_fterm, pre_till_pclose, pre_till_fclose, pclose_flag, fclose_flag", "Age of the product, planned and actual terms (binned)"],
          ["Amounts", "pre_loans_credit_limit, pre_loans_next_pay_summ, pre_loans_outstanding, pre_loans_total_overdue, pre_loans_max_overdue_sum, pre_loans_credit_cost_rate", "Limit, outstanding and overdue amounts, cost of credit (binned)"],
          ["Overdue counts", "pre_loans5 ... pre_loans90, is_zero_loans5 ... is_zero_loans90", "Number of late payments by delay bucket and flags of no overdues"],
          ["Ratios", "pre_util, pre_over2limit, pre_maxover2limit, is_zero_util, is_zero_over2limit, is_zero_maxover2limit", "Utilisation and overdue-to-limit ratios (binned) and zero flags"],
          ["Payment history and type", "enc_paym_0 ... enc_paym_24, enc_loans_account_holder_type, enc_loans_credit_status, enc_loans_credit_type, enc_loans_account_cur", "Monthly payment statuses for 25 months; product type, status, currency"]],
         [1.3, 2.9, 1.8], 9),
        ("h2", "3.3 Exploratory Analysis"),
        ("p", f"In partition 5, which was used for the exploratory figures, the number of credit products per "
              f"applicant varies from one to {lp['max']:.0f}, with a median of {lp['50%']:.0f} (Figure 3.2). "
              "Applicants with very short histories default most often, at almost 6% for a single product. The "
              "default rate falls to about 3% for applicants with eight to twelve products, which is the expected "
              "pattern, because a longer record of products that were serviced without problems is itself "
              "evidence of reliability. For very long histories the rate rises again, possibly because such "
              "applicants carry many loans at the same time, but these groups are small and their rates are noisy."),
        ("fig", "eda_products.png", "Figure 3.2. Distribution of credit products per applicant and the default rate by number of products (partition 5)", 6.2),
        ("p", "The monthly payment statuses turned out to be the most informative raw fields. After checking "
              "their distributions, four codes were identified: 0 for a payment on time, 1 and 2 for late "
              "payments of increasing severity, and 3 for months with no information, usually because the "
              "product was not yet open or already closed. An important detail was found here: in months 11, 20 "
              "and 24 the same statuses are stored with an offset of one (1 to 4 instead of 0 to 3). This was "
              "confirmed by cross-tabulating neighbouring months, and the offset was removed before the payment "
              "features were calculated. Figure 3.3 shows that the share of months without information grows "
              f"with the distance from the application date. Applicants with a seriously late payment in the last "
              f"six months default at a rate of {pct(d6['2'][0], 1)}, against {pct(d6['0'][0], 1)} for applicants "
              "with no late payments in that period."),
        ("fig", "eda_payments.png", "Figure 3.3. Payment status by month and the default rate by the worst status in the last six months", 6.2),
        ("p", "The binned fields behave differently. Figure 3.4 shows the default rate for each code of the "
              "utilisation field pre_util. The rate does not grow or fall steadily with the code; it jumps between "
              "neighbouring codes. The same was seen for the credit limit and the overdue-to-limit ratios. This "
              "means that the bin codes cannot be treated as ordered values, and any analysis that assumes "
              "\"higher code means more risk\" would be wrong. For this reason the count features of the baseline, "
              "which treat every code as a separate category, were kept, and the stress scenarios in Section 3.8 "
              "were built only on fields with a clear meaning."),
        ("fig", "eda_bins.png", "Figure 3.4. Default rate by bin code of the utilisation field", 5.2),
        ("h2", "3.4 Feature Engineering"),
        ("p", "A PD model needs one row per application, so the loan-level records have to be aggregated. The "
              f"baseline representation was reproduced first: every raw field and every value it takes was turned "
              f"into a count of the applicant's products with that value. This gives {n_cnt} count features, for "
              "example the number of products whose latest payment status is 0. The feature builder was "
              "rewritten with vectorised NumPy operations so that one partition of about two million records is "
              "processed in roughly six seconds."),
        ("p", f"On top of the counts, {nf_all - n_cnt} new features were designed. They are grouped in Table 3.3. "
              "The most important group describes the payment history over recent windows: for each status and "
              "for the last 3, 6, 12 and 25 months, the number of product-months with that status and its share "
              "of all product-months. A feature for the number of months since the latest late payment on any "
              "product was also added. The other groups contain the share of products with each zero-overdue "
              "flag, the mean and maximum of the bin codes, and the attributes of the most recently opened "
              "product. Although the bin codes are not ordered, their means and maxima were kept as compact "
              "summaries and the selection step was left to decide whether they are useful."),
        ("table", "Table 3.3. Engineered features",
         ["Group", "Number", "Example"],
         [["Payment status counts and shares by window", "32", "Share of on-time product-months in the last 6 months"],
          ["Months since latest late payment", "1", "0 = late in the latest month, 99 = never late"],
          ["Bin code means and maxima", "38", "Mean code of pre_loans_credit_limit"],
          ["Shares of flags", "10", "Share of products with no 5-30 day overdues"],
          ["Latest product attributes", "14", "Credit type of the most recent product"],
          ["Number of products", "1", "Count of records for the application"]],
         [2.6, 0.8, 2.6]),
        ("p", f"All {nf_all} features were first evaluated with LightGBM on a 25% sample of the training data. "
              f"The features were ranked by total gain, and models with the top 25, 50, 100, 150, 200 and 300 "
              f"features were compared on the validation sample. The final models use the top {K} features, "
              "which keeps almost all of the predictive power and reduces memory use, an important practical "
              "point because the training matrix has two million rows."),
        ("h2", "3.5 Sampling and Validation Scheme"),
        ("p", "Since the ids follow the application date, a time-based split could be made without calendar "
              "dates. Partitions 0 to 7 (applications 0 to 1,999,999) form the training sample, partitions 8 "
              "and 9 the validation sample, which is used for early stopping and for fitting calibration "
              "functions, and partitions 10 and 11 (applications 2,500,000 to 2,999,999) the out-of-time (OOT) "
              "sample. The OOT sample is used only once, to report final results. Figure 3.5 shows that the "
              "default rate is not constant across partitions: it rises from about 3.1% to over 4% in the middle "
              f"of the period and falls again. The average rate is {pct(train_dr)} in the training sample, "
              f"{pct(valid_dr)} in validation and {pct(oot_dr)} in the OOT sample, so the OOT test also checks "
              "how the model reacts to a shift in the overall default level."),
        ("fig", "default_rate_by_partition.png", "Figure 3.5. Default rate by partition and the time-based split", 6.0),
        ("h2", "3.6 Models and Training Settings"),
        ("p", "Four models were compared on the same split. The first is the baseline LightGBM with its original "
              "hyperparameters and class weight, trained on the count features only. The second is a logistic "
              "regression on the selected features, standardised and with L2 regularisation, trained on a 25% "
              "sample of the training data. It plays the role of the traditional scorecard benchmark. The third "
              "is LightGBM with settings chosen for the larger feature set and without class weights. The last "
              "is CatBoost, trained on the GPU in three configurations that differ in tree depth, regularisation "
              "and row sampling. All boosting models "
              "used early stopping on the validation AUC. The settings are listed in Table 3.4."),
        ("table", "Table 3.4. Model settings",
         ["Model", "Features", "Main settings"],
         [["LightGBM baseline", f"{n_cnt} counts", f"learning rate 0.05, max depth 5, 64 leaves, L2 = 1, scale_pos_weight = {(1 - train_dr) / train_dr:.1f}, early stop 50"],
          ["Logistic regression", f"top {K}", "standardised inputs, L2, C = 0.1, 25% training sample"],
          ["LightGBM (new features)", f"top {K}", "learning rate 0.03, 64 leaves, min child 300, feature fraction 0.5, bagging 0.8, L2 = 5, early stop 200"],
          ["CatBoost (3 configurations)", f"top {K}", "GPU, Logloss, 254 borders, early stop 300; (a) depth 6, lr 0.05, L2 = 3; (b) depth 8, lr 0.05, L2 = 5; (c) depth 8, lr 0.06, L2 = 10, Bernoulli subsample 0.8, up to 12,000 trees"]],
         [1.6, 1.0, 3.4], 9.5),
        ("p", "Class weights were not used in the new models. For boosted trees, weighting usually brings little "
              "or no gain in ranking quality, while it always moves the predicted probabilities away from the true "
              "default rate. Without weights, a model trained with log-loss already estimates the probability of default "
              "directly, which is what a PD model has to do."),
        ("h2", "3.7 Evaluation Metrics"),
        ("p", "Discrimination was measured with the ROC AUC, the Gini coefficient and the KS statistic:"),
        ("eq", "Gini = 2 × AUC − 1,    KS = max_{t} | F_{bad}(t) − F_{good}(t) |", "(3.1)"),
        ("p", "where F_{bad} and F_{good} are the cumulative distributions of the score among defaulters and "
              "non-defaulters. The precision-recall AUC was also recorded, because with a default rate of about "
              "3.5% it shows the quality of the riskiest part of the ranking more clearly. Bootstrap resampling "
              "with 200 replications was used to obtain 95% confidence intervals for the AUC and for the "
              "difference between two models (Efron & Tibshirani, 1993)."),
        ("p", "Calibration was measured with the Brier score, the log-loss, the comparison of mean PD with the "
              "observed default rate (ODR), and the Hosmer-Lemeshow statistic on ten groups of equal size:"),
        ("eq", "HL = Σ_{g=1..10} (O_{g} − E_{g})^{2} / [E_{g} (1 − p̄_{g})]", "(3.2)"),
        ("p", "where O_{g} and E_{g} are the observed and expected numbers of defaults in group g and p̄_{g} is the "
              "mean PD of the group. In addition, a one-sided binomial test was applied to each rating grade of the "
              "master scale: a grade fails (\"red\") if its number of defaults is above the 99th percentile of the "
              "binomial distribution with the grade's mean PD (Tasche, 2008)."),
        ("p", "Stability was measured with the population stability index between a reference and a current "
              "sample:"),
        ("eq", "PSI = Σ_{i} (a_{i} − e_{i}) · ln(a_{i} / e_{i})", "(3.3)"),
        ("p", "where e_{i} and a_{i} are the shares of the reference and current samples in bin i. Values below 0.1 "
              "are usually read as no significant shift, 0.1 to 0.25 as a moderate shift and above 0.25 as a "
              "major shift (Siddiqi, 2017). When applied to a single feature, the same index is called the "
              "characteristic stability index (CSI). The AUC was also calculated for each block of 50,000 "
              "applications in the validation and OOT samples."),
        ("h2", "3.8 Stress Testing Framework"),
        ("p", "Because the dataset has no calendar dates, it was not possible to fit a regression of the default "
              "rate on real macroeconomic series. Three complementary methods were used instead."),
        ("p", "**Top-down macro scenarios.** Each scenario is defined as a quantile q of the systematic risk factor "
              "of the Vasicek model. The stressed PD of every applicant is"),
        ("eq", "PD_{stress} = Φ( [Φ^{−1}(PD) + √ρ · Φ^{−1}(q)] / √(1 − ρ) )", "(3.4)"),
        ("p", "where Φ is the standard normal distribution function and ρ is the asset correlation. The Basel "
              "formula for other retail exposures was used for ρ, and as a check the correlation was also "
              "estimated from the variation of default rates across the 60 blocks of 50,000 applications, after "
              "removing the part of the variance explained by binomial sampling. The scenarios are listed in "
              "Table 3.5. Their severity can be connected to the narratives in Bhalla (2023), such as a rise in "
              "unemployment of several points, but the mapping is judgemental, because no macro data was "
              "available to estimate it."),
        ("table", "Table 3.5. Macro stress scenarios",
         ["Scenario", "Factor quantile q", "LGD", "Interpretation"],
         [["Baseline", "-", "60%", "Model PDs as estimated"],
          ["Mild downturn", "0.80", "60%", "A bad year that happens about once in five years"],
          ["Adverse", "0.95", "60%", "Once in twenty years, e.g. a sharp rise in unemployment"],
          ["Severe", "0.99", "70%", "Once in a hundred years, with downturn LGD"],
          ["Extreme (Basel)", "0.999", "70%", "Regulatory capital confidence level"]],
         [1.4, 1.2, 0.6, 2.8]),
        ("p", "**Bottom-up sensitivity.** In the second method the borrower records themselves were changed and "
              "the applications were scored again by the model. A random 5% to 50% of the products whose latest "
              "payment was on time were set to late (status 1) with the matching overdue flag; in another "
              "scenario existing recent lateness was made more severe. This shows how the model reacts to a wave "
              "of new delinquencies, independently of the Vasicek assumptions."),
        ("p", "**Reverse stress test.** The third method searched for the factor quantile at which the expected "
              "loss of the OOT portfolio, with downturn LGD, reaches a loss limit of 5% of exposure, and for the "
              "share of newly late loans that doubles the average PD."),
        ("p", "Expected loss and capital were calculated per unit of exposure, since the amounts in the data are "
              "binned. LGD was set to 60% for unsecured retail loans and 70% in downturn, as an assumption. The "
              "IRB capital requirement per unit of exposure is"),
        ("eq", "K = LGD · [ PD_{stress}(q = 0.999) − PD ],    EL = PD · LGD · EAD", "(3.5)"),
        ("h2", "3.9 Tools and Technologies"),
        ("p", "The project was written in Python 3.12. The main libraries and their roles are listed in Table 3.6; "
              "metrics and the logistic regression come from scikit-learn (Pedregosa et al., 2011). "
              "Training was done on a laptop with 16 logical CPU cores, 14 GB of RAM and an NVIDIA GeForce RTX "
              "4060 GPU with 8 GB of memory. The limited RAM shaped some technical choices: the features are "
              "stored per partition in compact integer types, and LightGBM reads the baseline matrix through its "
              "Sequence interface in batches instead of one large array."),
        ("table", "Table 3.6. Software used in the project",
         ["Tool", "Version", "Purpose"],
         [["pandas, NumPy, PyArrow", R["ver"]["pandas"] + ", " + R["ver"]["numpy"] + ", " + R["ver"]["pyarrow"], "Reading Parquet files, feature engineering"],
          ["scikit-learn", R["ver"]["sklearn"], "Logistic regression, metrics, isotonic calibration"],
          ["LightGBM", R["ver"]["lightgbm"], "Baseline and benchmark boosting models"],
          ["CatBoost", R["ver"]["catboost"], "Final model, GPU training, SHAP values"],
          ["SHAP", R["ver"]["shap"], "Plots of SHAP values"],
          ["SciPy", R["ver"]["scipy"], "Binomial and chi-square tests, normal distribution"],
          ["Matplotlib", R["ver"]["matplotlib"], "Figures for the report"],
          ["Streamlit, Plotly", R["ver"]["streamlit"] + ", " + R["ver"]["plotly"], "Software application"]],
         [1.8, 1.3, 2.9]),
    ]
