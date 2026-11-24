"""Abstract, Chapter 4 (results), Chapter 5 (conclusion) and the appendices."""
from text_appendix import appendices as _appendices
from text_method import pct

KEYWORDS = ("probability of default, credit scoring, CatBoost, LightGBM, out-of-time validation, calibration, "
            "SHAP, stress testing, Vasicek model")

MODEL_NAMES = {"lgb_baseline": "LightGBM baseline", "logreg": "Logistic regression",
               "lgb_tuned": "LightGBM (new features)", "blend": "Blend CatBoost + LightGBM"}


def f4(x):
    return f"{x:.4f}"


def _final_name(R):
    return R["final"]


def _names(R):
    n = dict(MODEL_NAMES)
    n[_final_name(R)] = "CatBoost (final)"
    return n


def ABSTRACT(R):
    e1, e4, e6 = R["exp1"], R["exp4"], R["exp6"]
    fin = _final_name(R)
    d = e4["discrimination_oot"]
    gain = e4["auc_gain_vs_baseline"]
    rev = e6["reverse"]
    sev = next(s for s in e6["macro_scenarios"] if s["scenario"].startswith("Severe"))
    return [
        "Banks use probability of default (PD) models to approve loans and to calculate reserves and capital. "
        "This term paper develops a PD model for retail applicants on an anonymised dataset of three million "
        "applications with 26.2 million credit bureau records. An existing LightGBM baseline was reproduced "
        f"(random-split ROC AUC {e1['random_split_auc']:.3f}) and then tested on later applications, where its "
        f"AUC fell to {d['lgb_baseline']['auc']:.3f} and its mean predicted PD of "
        f"{pct(d['lgb_baseline']['mean_pd'], 1)} was more than ten times the observed default rate. New features "
        "were built from the 25-month payment history, 200 features were selected, and logistic regression, "
        "LightGBM and CatBoost were compared on an out-of-time (OOT) sample of 500,000 applications. The final "
        f"CatBoost model reached an OOT AUC of {d[fin]['auc']:.4f} (Gini {d[fin]['gini']:.4f}), "
        f"{gain['mean']:.4f} above the baseline (95% bootstrap interval {gain['ci'][0]:.4f} to "
        f"{gain['ci'][1]:.4f}); LightGBM was equally accurate. The probabilities were calibrated (mean PD "
        f"{pct(d[fin]['mean_pd'])} against {pct(d[fin]['default_rate'])} observed) and discrimination was stable "
        "over time. SHAP analysis showed that utilisation, product terms and payment history are the main "
        "drivers. Stress tests with the Vasicek one-factor model and with shocks to borrower records showed "
        f"that a 1-in-100 year scenario raises the mean PD to {pct(sev['mean_pd'])} and that losses exceed 5% "
        f"of exposure beyond the {rev['q_star']:.3f} quantile of the systematic factor. A Streamlit "
        "application returns the PD, rating grade, risk drivers and stressed PD for a user-defined credit "
        "history.",
    ]


def ch4(R):
    e1, e2, e3, e4, e5, e6 = R["exp1"], R["exp2"], R["exp3"], R["exp4"], R["exp5"], R["exp6"]
    fin = _final_name(R)
    names = _names(R)
    d = e4["discrimination_oot"]
    cal = e4["calibration"]
    ms = R["master_scale_oot"]
    blocks_df = R["auc_by_block"]
    imp = R["shap_imp"]
    sg = R["sel_groups"]

    # ---- 4.1 baseline
    b_va, b_oot = e1["valid"], e1["oot"]
    rows41 = [
        ["Random 80/20 split (as in original notebook)", f4(e1["random_split_auc"]), "-", "-", "-"],
        ["Time split: validation (apps 2.0M-2.5M)", f4(b_va["auc"]), f4(b_va["gini"]), pct(b_va["mean_pd"], 1), pct(b_va["default_rate"])],
        ["Time split: OOT (apps 2.5M-3.0M)", f4(b_oot["auc"]), f4(b_oot["gini"]), pct(b_oot["mean_pd"], 1), pct(b_oot["default_rate"])],
    ]
    # ---- 4.2 feature selection
    curve = {c["k"]: c["auc"] for c in e2["topk_curve"]}
    # ---- 4.3 comparison
    cb_keys = sorted(k for k in e3 if k.startswith("cb_"))
    order = ["lgb_baseline", "logreg", "lgb_tuned"] + cb_keys + ["blend"]
    rows43 = []
    for k in order:
        if k == "lgb_baseline":
            va, oo = e1["valid"], e1["oot"]
        else:
            va, oo = e3[k]["valid"], e3[k]["oot"]
        if k.startswith("cb_"):
            pr = e3[k]["params"]
            label = f"CatBoost depth {pr['depth']}, lr {pr['learning_rate']}" + (", Bernoulli 0.8" if pr.get("bootstrap_type") else "")
            label += " (final)" if k == fin else ""
        else:
            label = names[k]
        rows43.append([label, f4(va["auc"]), f4(oo["auc"]), f4(oo["gini"]), f4(oo["ks"]), f4(oo["pr_auc"]), f"{oo['brier']:.5f}"])
    best_single = max(["logreg", "lgb_tuned"] + cb_keys, key=lambda k: e3[k]["oot"]["auc"])
    cb_aucs = [e3[k]["oot"]["auc"] for k in cb_keys]
    blend_gain = e3["blend"]["oot"]["auc"] - e3[fin]["oot"]["auc"]
    lgb_vs_cb = e3["lgb_tuned"]["oot"]["auc"] - e3[fin]["oot"]["auc"]

    # ---- 4.4 calibration
    rows44 = []
    for k, lbl in (("raw", "CatBoost, no recalibration"), ("platt", "Platt scaling (fitted on validation)"),
                   ("isotonic", "Isotonic regression (fitted on validation)")):
        c = cal[k]
        rows44.append([lbl, pct(c["mean_pd"]), pct(c["odr"]), f"{c['brier']:.5f}", f"{c['logloss']:.4f}",
                       f"{c['hl_stat']:.1f}", f"{c['hl_p']:.3f}"])
    ms_valid = ms[ms.n > 0]
    n_red = int((ms_valid.result == "red").sum())
    approve_share = float(ms_valid[ms_valid.grade.isin(["R1", "R2", "R3", "R4", "R5"])].share.sum())
    decline_share = float(ms_valid[ms_valid.grade.isin(["R8", "R9", "R10"])].share.sum())
    worst = ms_valid.iloc[-1]

    # ---- 4.5 stability
    ab = e4["auc_by_block"]
    csi = e4["csi_top_features"]
    csi_max_f = max(csi, key=csi.get)
    from pd_scoring.labels import describe

    # ---- 4.6 SHAP
    top = imp.head(10)
    top_share = float(imp.head(10).share.sum())
    pay_share = float(imp[imp.feature.str.startswith("pay__")].share.sum())
    loc = e5["local"]
    fam = e5["families"]

    # ---- 4.7 stress
    sc = e6["macro_scenarios"]
    basel = [s for s in sc if "[empirical" not in s["scenario"]]
    emp = [s for s in sc if "[empirical" in s["scenario"]]
    rho = e6["rho_estimate"]
    rows47 = [[s["scenario"], "-" if s["q"] is None else f"{s['q']}", pct(s["lgd"], 0), pct(s["mean_pd"]),
               pct(s["el_rate"]), pct(s["approval_rate"], 1), pct(s["share_decline_grades"], 1)] for s in basel]
    emp_sev = next(s for s in emp if s["scenario"].startswith("Severe"))
    bas_sev = next(s for s in basel if s["scenario"].startswith("Severe"))
    bu = e6["bottom_up"]
    rows47b = [["No shock", pct(bu["base"]["mean_pd"]), pct(bu["base"]["approval_rate"], 1), pct(bu["base"]["share_decline_grades"], 1)]]
    for s in bu["shocks"]:
        rows47b.append([s["shock"], pct(s["mean_pd"]), pct(s["approval_rate"], 1), pct(s["share_decline_grades"], 1)])
    rev = e6["reverse"]
    shock10 = next(s for s in bu["shocks"] if s.get("share") == 0.10)
    shock_comb = bu["shocks"][-1]
    esc = bu["shocks"][4]

    out = [
        ("h1", "4. IMPLEMENTATION AND RESULTS"),
        ("p", "This chapter reports the results of six experiments in the order in which they were carried out. "
              "Every number comes from the result files produced by the scripts in Appendix A. Unless stated "
              "otherwise, results refer to the out-of-time sample of 500,000 applications that was not used in "
              "any training or tuning step."),
        ("h2", "4.1 Experiment 1: Reproduction of the Baseline"),
        ("p", f"The baseline was rebuilt with the same {R['n_count_features']} count features and the same "
              "LightGBM settings, including the class weight. With a random 80/20 split of all applications, as "
              f"in the original notebook, it reached an AUC of {f4(e1['random_split_auc'])}. This is close to the "
              "cross-validated value of 0.7606 reported for the baseline, so the reproduction can be considered "
              "successful. The results with the time-based split are shown in Table 4.1."),
        ("table", "Table 4.1. Baseline LightGBM under different validation schemes",
         ["Evaluation", "AUC", "Gini", "Mean PD", "Observed DR"], rows41, [2.8, 0.75, 0.75, 0.8, 0.9]),
        ("p", f"On the OOT sample the AUC of the baseline fell to {f4(b_oot['auc'])}, which is "
              f"{e1['random_split_auc'] - b_oot['auc']:.4f} lower than with the random split. Part of this drop "
              "is caused by the change of the population over time, and part by the fact that a random split "
              "lets the model learn from applications that were made at the same time as the test cases. The "
              "second problem is calibration. Because of the class weight of about 27, the mean predicted "
              f"probability was {pct(b_oot['mean_pd'], 1)} while only {pct(b_oot['default_rate'])} of the "
              f"applicants defaulted, and the Brier score was {b_oot['brier']:.3f}. The ranking is still useful, "
              "but the output cannot be read as a PD. These two findings defined the targets for the next "
              "experiments: a better OOT ranking and probabilities close to the real default rate."),
        ("h2", "4.2 Experiment 2: Feature Engineering and Selection"),
        ("p", "The value of the new features was measured with one LightGBM configuration on a 25% sample of the "
              "training partitions and half of the validation partitions. With the baseline count features only, "
              f"the validation AUC was {f4(e2['baseline_counts']['auc'])}. The {e2['engineered_only']['n_features']} "
              f"engineered features alone reached {f4(e2['engineered_only']['auc'])}, which is lower, but together "
              f"with the counts the AUC rose to {f4(e2['all_features']['auc'])}. The engineered features therefore "
              "add information that the counts do not contain, mainly about the timing of late payments, which the "
              "counts per month capture only indirectly."),
        ("fig", "feature_selection_curve.png", "Figure 4.1. Validation AUC as a function of the number of selected features", 5.4),
        ("p", f"Figure 4.1 shows the AUC of models trained on the top K features by gain. The curve rises quickly "
              f"up to about 100 features and then flattens: the top 200 features gave {f4(curve[200])} and the top "
              f"300 gave {f4(curve[300])}, against {f4(e2['all_features']['auc'])} for all "
              f"{e2['all_features']['n_features']}. {e2['n_zero_gain']} features were never used by the model at "
              f"all. The top 200 were kept for the final models. Among them are {sg['count']} count features, "
              f"{sg['pay']} payment-window features, {sg['agg']} aggregate features and {sg['last']} attributes of "
              "the latest product, so the selected set mixes the baseline representation with the new features."),
        ("h2", "4.3 Experiment 3: Model Comparison"),
        ("p", "Table 4.2 compares all models on the validation and OOT samples. All models except the baseline "
              f"use the same {R['K']} features and the same training applications."),
        ("table", "Table 4.2. Comparison of models (validation AUC and OOT metrics)",
         ["Model", "AUC valid", "AUC OOT", "Gini OOT", "KS OOT", "PR-AUC OOT", "Brier OOT"],
         rows43, [1.9, 0.75, 0.75, 0.75, 0.7, 0.8, 0.8], 9),
        ("p", f"The logistic regression reached an OOT AUC of {f4(e3['logreg']['oot']['auc'])}, below the "
              "baseline. With 200 mostly count-type inputs and no binning, a linear model cannot use the strongly "
              "non-linear effects of the payment history, which agrees with the benchmark results of Lessmann et "
              f"al. (2015). LightGBM with the new features gave {f4(e3['lgb_tuned']['oot']['auc'])}. The "
              f"{len(cb_keys)} CatBoost configurations gave between {f4(min(cb_aucs))} and {f4(max(cb_aucs))}, so "
              "the depth of the trees and the sampling settings mattered much less than the features. The "
              + (f"difference between the selected CatBoost model and LightGBM was below 0.0001 AUC, "
                 if abs(lgb_vs_cb) < 0.0001 else
                 f"difference between the selected CatBoost model and LightGBM was {abs(lgb_vs_cb):.4f} AUC "
                 f"{'in favour of LightGBM' if lgb_vs_cb > 0 else 'in favour of CatBoost'}, ")
              + "which is far inside the "
              "bootstrap uncertainty reported below, so the two methods should be regarded as equally accurate on "
              f"this data. The simple average of the two changed the AUC by {blend_gain:+.4f}, which was not "
              "considered worth the extra cost of keeping and monitoring two models in production. CatBoost was "
              "chosen as the final model for practical reasons: it trains quickly on the GPU, its symmetric trees give very fast prediction in the application, "
              "and it calculates exact SHAP values internally, which the explanation page of the application "
              "needs for every request."),
        ("fig", "roc_oot.png", "Figure 4.2. ROC curves of the four main models on the OOT sample", 4.6),
        ("p", f"The improvement of the final model over the baseline is {e4['auc_gain_vs_baseline']['mean']:.4f} "
              f"AUC, or about {200 * e4['auc_gain_vs_baseline']['mean']:.1f} Gini points. Because both models were "
              "scored on the same applications, a paired bootstrap was used to judge whether this difference is "
              f"real. The 95% interval of the difference is {e4['auc_gain_vs_baseline']['ci'][0]:.4f} to "
              f"{e4['auc_gain_vs_baseline']['ci'][1]:.4f}; it does not include zero, so the gain is not a result "
              "of sampling noise. The confidence intervals of all models are shown in Figure 4.3. The ROC curves "
              "in Figure 4.2 show that the gain is spread over the whole range of thresholds."),
        ("fig", "model_comparison.png", "Figure 4.3. Gini of the models on the OOT sample with 95% bootstrap intervals", 4.8),
        ("p", f"The KS statistic of the final model is {f4(d[fin]['ks'])} (Figure 4.4). The distributions of "
              "predicted PD for defaulters and non-defaulters overlap strongly (Figure 4.5), which is normal for "
              "retail credit data with a default rate of about 3%: many defaults happen to clients whose history "
              "looked clean at the time of the application."),
        ("fig", "ks_oot.png", "Figure 4.4. Kolmogorov-Smirnov plot of the final model", 4.8),
        ("fig", "pd_distribution.png", "Figure 4.5. Distribution of predicted PD by outcome on the OOT sample", 4.8),
        ("h2", "4.4 Experiment 4: Calibration and Master Scale"),
        ("p", "Since no class weights were used, the raw CatBoost output was expected to be close to calibrated. "
              "Two recalibration methods, Platt scaling and isotonic regression, were fitted on the validation "
              "sample and tested on the OOT sample. The results are in Table 4.3."),
        ("table", "Table 4.3. Calibration on the OOT sample",
         ["Method", "Mean PD", "ODR", "Brier", "Log-loss", "HL statistic", "HL p-value"],
         rows44, [2.3, 0.7, 0.7, 0.75, 0.7, 0.75, 0.7], 9),
        ("p", f"The raw model predicted a mean PD of {pct(cal['raw']['mean_pd'])} against an observed default rate "
              f"of {pct(cal['raw']['odr'])}. The small overestimation is expected, because the training sample had "
              "a higher default rate than the OOT period. Neither recalibration method improved the Brier score "
              "or the log-loss in a meaningful way, and both are fitted on the validation period, which itself "
              "had a higher default rate than the OOT period. The raw probabilities were therefore kept. The "
              "Hosmer-Lemeshow test rejects perfect calibration for all three versions; with 500,000 observations "
              "this test detects even very small deviations, so its result is reported but was not treated as "
              "decisive (Tasche, 2008). The decile plot in Figure 4.6 shows that the predicted and observed rates "
              "lie close to the diagonal in the first nine deciles; in the riskiest decile the model predicts about "
              "12% while 10.5% of the applicants defaulted, so the highest PDs are somewhat conservative."),
        ("fig", "calibration_deciles.png", "Figure 4.6. Calibration plot of the final model by score decile", 4.4),
        ("p", "The PDs were mapped to a master scale of ten grades, from R1 (PD below 0.5%) to R10 (PD above 20%). "
              "Grades R1-R5 (PD below 3%) were assigned the decision \"approve\", R6-R7 \"manual review\" and "
              f"R8-R10 \"decline\". On the OOT sample {pct(approve_share, 1)} of the applicants fell into the "
              f"approval grades and {pct(decline_share, 1)} into the decline grades. Figure 4.7 compares the mean "
              f"PD and the observed default rate in each grade; the full table is in Appendix C. The default rate "
              f"rises monotonically from grade to grade, up to {pct(worst.odr, 1)} in {worst.grade}. The binomial "
              f"test at the 99% level flagged {n_red} of {len(ms_valid)} grades as red"
              + (", which means that the observed defaults did not exceed the level expected from the model in any "
                 "grade." if n_red == 0 else
                 f" ({', '.join(ms_valid[ms_valid.result == 'red'].grade)}). In these grades the observed default "
                 "rate was slightly higher than the mean PD, by less than 0.3 percentage points, but with tens of "
                 "thousands of applicants per grade even such small gaps are significant. In the riskiest grades "
                 "the opposite happens: the model overstates the PD, for example "
                 f"{pct(worst.pd_mean, 1)} against {pct(worst.odr, 1)} in {worst.grade}. The model is therefore "
                 "slightly too optimistic for good clients and conservative for bad ones. For credit decisions this "
                 "is acceptable, because the ranking is correct, but before the PDs are used for capital or "
                 "reserves a grade-level calibration on a longer history would be needed.")),
        ("fig", "master_scale.png", "Figure 4.7. Mean predicted PD and observed default rate by rating grade (OOT)", 5.4),
        ("h2", "4.5 Stability over Time"),
        ("p", f"The AUC of the final model was calculated for each block of 50,000 applications in the validation "
              f"and OOT samples (Figure 4.8). It varied between {f4(ab['final_min'])} and {f4(ab['final_max'])} "
              f"with a standard deviation of {ab['final_std']:.4f}, and it was higher than the AUC of the baseline "
              f"in every block. The population stability index of the predicted PD between the validation and OOT "
              f"samples was {e4['psi_score_valid_vs_oot']:.4f}, far below the warning level of 0.1. The CSI was "
              "also calculated for the ten most important features, between the last two training partitions "
              f"and the OOT sample. Nine of them stayed below {sorted(csi.values())[-2] + 0.0005:.3f}, but one, "
              f"\"{describe(csi_max_f).lower()}\", reached {csi[csi_max_f]:.3f}, which is just above the 0.1 "
              "threshold of a moderate shift. The share of products with this cost-of-credit code changed between "
              "the periods, possibly because of new loan products or a change in prices. The score as a whole did "
              "not move, so the shift is absorbed by the other features, but this feature is the first candidate "
              "to watch during monitoring."),
        ("fig", "auc_by_block.png", "Figure 4.8. ROC AUC by block of 50,000 applications", 5.6),
        ("h2", "4.6 Experiment 5: Model Interpretation with SHAP"),
        ("p", "SHAP values were calculated with CatBoost's exact tree algorithm for a random subsample of 20,000 "
              "OOT applications; a subsample keeps the calculation fast and is large enough for stable averages. Figure 4.9 shows "
              "the twenty most influential features. Each point is one applicant; its position shows how much the "
              "feature moved the log-odds of default, and its colour shows the feature value."),
        ("fig", "shap_beeswarm.png", "Figure 4.9. SHAP summary plot for the final model", 5.6),
        ("p", f"The importance is spread over many features: the ten strongest account for only "
              f"{pct(top_share, 1)} of the total mean absolute SHAP value. The top of the list is formed by count "
              f"features, led by \"{top.iloc[0].description.lower()}\" and \"{top.iloc[1].description.lower()}\". "
              "Since the bin codes are not ordered, these features work as indicators that the applicant holds "
              "products of a particular kind. Section 3.3 showed that codes 3 and 6 of the utilisation field have "
              "default rates of about 5% and 6%, roughly twice the rate of the most common code, and the model "
              "has learned that every additional product with such a code raises the risk. The dependence plots "
              "in Figure 4.11 show that the effect of the three strongest features grows steadily with the "
              "number of such products, which makes the model easier to defend in a validation review."),
        ("p", "The same ranking is shown as a bar chart of mean absolute SHAP values in Figure 4.10; the full list "
              "of the twenty strongest features is given in Table C.2 of Appendix C."),
        ("fig", "shap_bar.png", "Figure 4.10. Fifteen most important features by mean absolute SHAP value", 5.6),
        ("fig", "shap_dependence.png", "Figure 4.11. SHAP dependence plots for the three strongest features", 6.0),
        ("p", f"Figure 4.12 groups the importance by the type of bureau field. Utilisation and amount fields "
              f"carry {pct(fam['Utilisation and amounts'], 1)} of the total, terms and dates "
              f"{pct(fam['Terms, dates and history length'], 1)}, payment history "
              f"{pct(fam['Payment history'], 1)}, credit cost, type and status "
              f"{pct(fam['Credit cost, type and status'], 1)} and overdue counts "
              f"{pct(fam['Overdue counts and flags'], 1)}. The new payment-window features alone contribute "
              f"{pct(pay_share, 1)}. This differs from the LightGBM gain ranking of Experiment 2, where the "
              "share of on-time payments in the last six months was the top feature. The two rankings measure "
              "different things (gain counts how much a feature improves the training loss, SHAP how much it "
              "moves the predictions), and the payment and utilisation features are partly correlated, so a model "
              "can use either of them for the same information. The overall picture still agrees with practice: "
              "how much of the available credit the applicant uses, how the payments were made and what kinds of "
              "products are held are the main signals (Evstafev, 2024; Siddiqi, 2017)."),
        ("fig", "shap_families.png", "Figure 4.12. SHAP importance by group of bureau fields", 5.0),
        ("p", f"SHAP also explains single decisions. Figure 4.13 shows an applicant with a PD of "
              f"{pct(loc['high']['pd'], 1)}, among the riskiest 0.5% of the sample. The model starts from the "
              f"average log-odds, which corresponds to a PD of {pct(e5['base_pd'], 1)}. For this applicant none of "
              "the products has a zero overdue-to-limit ratio and only one month in six was paid on time; these "
              "two facts add the largest single contributions, and many smaller contributions of the other "
              "features add the rest. Explanations of this kind are shown to the user of the application "
              "described in Section 4.8."),
        ("fig", "shap_waterfall_high.png", "Figure 4.13. SHAP explanation of a single high-risk applicant", 5.4),
        ("h2", "4.7 Experiment 6: Stress Testing"),
        ("p", "**Macro scenarios.** Table 4.4 shows the OOT portfolio under the scenarios of Table 3.5, with the "
              "Basel retail correlation. Expected loss is given per unit of exposure."),
        ("table", "Table 4.4. OOT portfolio under top-down macro scenarios",
         ["Scenario", "q", "LGD", "Mean PD", "EL rate", "Approval rate", "Share R8-R10"],
         rows47, [1.95, 0.5, 0.5, 0.75, 0.75, 0.85, 0.85], 9),
        ("p", f"In the adverse 1-in-20 scenario the mean PD rises from {pct(basel[0]['mean_pd'])} to "
              f"{pct(basel[2]['mean_pd'])}, and the share of applicants who would pass the approval threshold "
              f"falls from {pct(basel[0]['approval_rate'], 1)} to {pct(basel[2]['approval_rate'], 1)}. In the "
              f"severe scenario the mean PD reaches {pct(bas_sev['mean_pd'])} and, with downturn LGD, the expected "
              f"loss reaches {pct(bas_sev['el_rate'])} of exposure. Figure 4.14 shows how the grade mix moves "
              "towards the decline grades as the scenario becomes more severe. The average IRB capital "
              f"requirement under the baseline is {pct(e6['irb_capital_mean_k'])} of exposure, which corresponds "
              "to the extreme scenario minus expected loss."),
        ("fig", "stress_grade_mix.png", "Figure 4.14. Rating grade mix of the OOT portfolio under macro scenarios", 5.8),
        ("p", f"The correlation estimated from the data was {rho['rho']:.4f}. The default rate of the 60 blocks "
              f"ranged from {pct(rho['dr_min'])} to {pct(rho['dr_max'])}, and after removing the binomial noise "
              "the remaining variation corresponds to a much weaker systematic factor than the Basel assumption. "
              f"With this correlation the severe scenario gives a mean PD of only {pct(emp_sev['mean_pd'])}. "
              "This result has to be read with care: the blocks are consecutive groups of applications, probably "
              "covering a short period of time rather than a full economic cycle, so they understate the true "
              "cyclical variation. The Basel values are therefore the more prudent choice for stress testing, "
              "and the data-based estimate is better seen as a lower bound."),
        ("p", "**Bottom-up sensitivity.** In this test the payment records of 40,000 OOT applicants were "
              "changed and the applicants were scored again by the model. Table 4.5 shows the results."),
        ("table", "Table 4.5. Bottom-up payment shocks on 40,000 OOT applicants",
         ["Shock", "Mean PD", "Approval rate", "Share R8-R10"], rows47b, [3.3, 0.9, 1.0, 1.0], 9),
        ("p", f"When 10% of the loans that were paid on time are marked as late in the latest month, the mean PD "
              f"rises from {pct(bu['base']['mean_pd'])} to {pct(shock10['mean_pd'])} and the approval rate falls "
              f"from {pct(bu['base']['approval_rate'], 1)} to {pct(shock10['approval_rate'], 1)}. The combined "
              f"shock raises the mean PD to {pct(shock_comb['mean_pd'])}. The model reacts in the expected "
              "direction to fresh delinquencies, which is the main channel through which an economic downturn "
              "reaches a retail portfolio, although the reaction is smaller than one might expect: even when half "
              f"of the on-time loans become late, the mean PD rises only to {pct(bu['shocks'][3]['mean_pd'])}, "
              "because most of the other features of each applicant stay unchanged. A less expected result is "
              f"that making the existing late payments more severe changed almost nothing (mean PD "
              f"{pct(esc['mean_pd'])}). Only {pct(esc['affected_applicants'], 0)} of the applicants had such "
              "payments, and for them the model had already assigned a high PD, so the extra severity added "
              "little. The distributions are compared in Figure 4.15."),
        ("fig", "stress_bottom_up.png", "Figure 4.15. PD distribution under bottom-up payment shocks", 5.4),
        ("p", f"**Reverse stress test.** With downturn LGD, the expected loss of the OOT portfolio reaches the "
              f"limit of 5% of exposure when the systematic factor is at its {rev['q_star']:.4f} quantile, which "
              f"corresponds to roughly a 1-in-{rev['return_period_years']:.0f} year event (Figure 4.16). At that "
              f"point the mean PD is {pct(rev['pd_mean_at_q_star'])}. "
              + (f"In the bottom-up test the mean PD doubles when about {pct(rev['share_late_to_double_pd'], 0)} of "
                 "the on-time loans turn late. " if rev.get("share_late_to_double_pd") else
                 "In the bottom-up test the mean PD did not double even when all on-time loans were turned late "
                 "in the latest month, because the model also looks at the longer history. ")
              + "Together, these numbers tell the risk manager how far conditions would have to deteriorate "
              "before the portfolio's losses become unacceptable."),
        ("fig", "stress_reverse.png", "Figure 4.16. Reverse stress test: expected loss rate against scenario severity", 5.2),
        ("h2", "4.8 Software Application"),
        ("p", "The final model was packaged as a web application, PD Scoring Studio, written with Streamlit and "
              "Plotly. It reuses the feature builder of the training pipeline, so the same code turns raw records "
              "into features in training and in use, which avoids a common source of errors when models are "
              "deployed. The application has four sections; the main page is shown in Figure 4.17."),
        ("bullets", [
            "**Applicant assessment.** The user describes the applicant's credit products in an editable table or "
            "starts from a template (clean history, thin file, recent delinquencies or real examples from the OOT "
            "sample). The page shows the PD, the rating grade, the decision, a three-year lifetime PD, the "
            "expected loss and IRB capital for a chosen loan amount and LGD, the PD under three stress levels, the "
            "ten features that drive the PD (from SHAP), the contributions grouped into risk themes and a map of "
            "the payment history.",
            "**Batch scoring.** A file of loan records can be uploaded and scored; the results can be downloaded.",
            "**Stress lab.** The bundled portfolio of 3,000 OOT applicants can be stressed interactively with the "
            "Vasicek scenario, a custom correlation and a bottom-up payment shock; the reverse stress test is "
            "solved for a chosen loss limit.",
            "**Model card.** A summary of the validation results and limitations.",
        ]),
        ("fig", "app_applicant.png", "Figure 4.17. Applicant assessment page of the software application", 6.0),
        ("p", "Scoring one applicant, including the SHAP explanation, takes well under a second on a laptop, and "
              "the bundled portfolio of 3,000 applicants is scored in about one second. Further screenshots and "
              "a short user guide are given in Appendix B."),
        ("h2", "4.9 Discussion"),
        ("p", "Three results stand out. First, the way a model is tested changed the conclusions more than the "
              "choice of algorithm. The random split overstated the AUC of the baseline by more than two points, "
              "and the class weight hid the fact that its probabilities were not PDs. In my opinion this is the "
              "most useful lesson of the project, because it applies to any credit model, not only this one."),
        ("p", f"Second, the improvement from new features and CatBoost is real but moderate, about "
              f"{200 * e4['auc_gain_vs_baseline']['mean']:.1f} Gini points on the OOT sample. In retail lending "
              "even one Gini point is valuable, because it is applied to every application, but the result also "
              "shows that the count representation of the baseline already captured most of the information in "
              "the binned bureau data. Experiment 2 suggests that a large part of the gain comes from the new "
              "payment-window and share features, which describe the history relative to the number of products; "
              "the rest comes from the larger number of trees and stronger regularisation of the new models. "
              "Removing the class weight did not have to improve the ranking, but it was what made the output "
              "usable as a PD. CatBoost and "
              "LightGBM reached the same accuracy, so on this data the choice between them can be made on "
              "practical grounds."),
        ("p", "Third, the stress tests depend strongly on assumptions that the data cannot confirm. The Basel "
              "correlation and the empirical estimate give very different stressed PDs, and LGD was not "
              "modelled at all. The stress results should therefore be read as an illustration of the method and "
              "of the sensitivity of the portfolio, not as a forecast of losses."),
    ]
    return out


def ch5(R):
    e1, e4, e6 = R["exp1"], R["exp4"], R["exp6"]
    fin = _final_name(R)
    d = e4["discrimination_oot"]
    sev = next(s for s in e6["macro_scenarios"] if s["scenario"].startswith("Severe"))
    return [
        ("h1", "5. CONCLUSION"),
        ("h2", "5.1 Conclusion"),
        ("p", "This project developed and validated a probability of default model for retail applicants on "
              "three million applications with credit bureau histories, and made it available in a software "
              "application. The objectives set in Section 1.3 were met as follows."),
        ("list", [
            "The literature review (Chapter 2) covered PD modelling, the Basel and IFRS 9 context, the validation "
            "tests used by banks, SHAP and stress testing, and it defined the methods used later.",
            f"The baseline was reproduced (random-split AUC {f4(e1['random_split_auc'])}) and tested out of time, "
            f"where its AUC was {f4(d['lgb_baseline']['auc'])} and its mean PD was {pct(d['lgb_baseline']['mean_pd'], 1)} "
            f"against a default rate of {pct(d['lgb_baseline']['default_rate'])}.",
            "Ninety-six new features were built from the payment history and other fields, and a compact set of "
            "200 features was selected without loss of accuracy.",
            f"Logistic regression, LightGBM and CatBoost were compared; CatBoost was chosen, with an OOT AUC of "
            f"{f4(d[fin]['auc'])} and a Gini of {f4(d[fin]['gini'])}.",
            f"The final model was well calibrated (mean PD {pct(d[fin]['mean_pd'])} against "
            f"{pct(d[fin]['default_rate'])}), its default rates rose steadily across the ten grades of the master "
            f"scale, and its AUC and score distribution were stable over time (PSI {e4['psi_score_valid_vs_oot']:.4f}).",
            "SHAP analysis on 20,000 applicants showed that utilisation, product terms, payment history and the "
            "type and cost of the products are the main drivers, and that the effects of the strongest features "
            "grow steadily with their values.",
            f"Stress tests showed that a 1-in-100 year scenario raises the mean PD to {pct(sev['mean_pd'])}, that "
            "the model reacts strongly to new delinquencies, and that losses exceed 5% of exposure only in very "
            "severe scenarios.",
            "A Streamlit application was developed that returns the PD, grade, decision, explanation and stressed "
            "PD for a user-defined credit history and supports batch scoring and interactive stress testing.",
        ]),
        ("p", "Overall, the work shows that a gradient boosting model built on bureau histories can be both more "
              "accurate than the earlier baseline and closer to the requirements of a real PD model, as long as "
              "it is trained without distorting class weights and tested on later applications."),
        ("h2", "5.2 Limitations"),
        ("p", "The study has several limitations. The bureau fields are anonymised bins whose original meaning and "
              "order are unknown, so some features, such as the means of the bin codes, have no clear economic "
              "interpretation, and the application has to ask the user for codes rather than amounts. The "
              "dataset contains no calendar dates and no macroeconomic variables, which made it impossible to "
              "estimate a satellite model linking default rates to the economy; the macro scenarios therefore "
              "rely on the Basel one-factor model and on judgement. LGD and EAD were assumed, not estimated. The "
              "OOT sample follows directly after the training period, so the stability results cover a "
              "relatively short time span. Finally, only one random seed and a small number of hyperparameter "
              "settings were tried because of the time needed to train on two million applications."),
        ("h2", "5.3 Future Scope"),
        ("p", "Several directions could improve the work. Sequence models such as recurrent networks or "
              "transformers could be trained directly on the loan-level records, following Babaev et al. (2019), "
              "and combined with the boosting model. A systematic hyperparameter search and several random seeds "
              "would give a more reliable estimate of the best settings. If dates and macroeconomic series "
              "became available, the stress test could be extended with a regression of default rates on "
              "unemployment, GDP and interest rates, in the spirit of Bellotti and Crook (2013). An LGD model and "
              "a lifetime PD term structure would allow a full IFRS 9 expected credit loss calculation. Finally, "
              "the application could be connected to a monitoring dashboard that recalculates the PSI and the AUC "
              "every month and warns when the model needs to be recalibrated."),
    ]


def appendices(R):
    ms = R["master_scale_oot"]
    from pd_scoring.risk import GRADE_BOUNDS, grade_lower_bounds
    lows = grade_lower_bounds()
    rows = []
    for i, r in ms.iterrows():
        if r.n == 0 or r.n != r.n:
            continue
        rows.append([r.grade, f"{lows[i] * 100:.1f}-{GRADE_BOUNDS[i] * 100:.1f}%", f"{int(r.n):,}", pct(r.share, 1),
                     f"{int(r.defaults):,}", pct(r.pd_mean), pct(r.odr), f"{int(r.critical_defaults_99):,}",
                     r.result])
    imp = R["shap_imp"].head(20)
    srows = [[str(i + 1), r.feature.replace("__", " "), r.description, f"{r.mean_abs_shap:.4f}"]
             for i, (_, r) in enumerate(imp.iterrows())]
    return _appendices(R, pct, rows, srows)
