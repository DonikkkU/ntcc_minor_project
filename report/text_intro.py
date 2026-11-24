"""Chapters 1 and 2 of the report (do not depend on experiment results)."""

TITLE = "Probability of Default Modelling for Retail Credit Scoring Using Gradient Boosting on Credit Bureau Histories"

CH1 = [
    ("h1", "1. INTRODUCTION"),
    ("p", "Lending is the core business of most commercial banks, and every loan carries the chance that the "
          "borrower will not pay it back. A bank that issues thousands of consumer loans a day cannot study each "
          "applicant by hand, so it relies on statistical models that turn the information about a person into a "
          "single number: the probability that this person will default within a given period. This number is "
          "called the probability of default (PD). It is used to accept or reject applications, to set the "
          "interest rate, to calculate loan loss reserves and to decide how much capital the bank must hold."),
    ("p", "This project builds such a model on a large anonymised set of credit bureau histories. Every "
          "application in the data is linked to the list of the applicant's previous and current credit "
          "products, and the target shows whether the client later failed to repay for more than 90 days. The "
          "work starts from an existing baseline, repeats it under stricter testing conditions, and then tries to "
          "improve it with new features and a CatBoost model. Besides accuracy, the report looks at the things a "
          "bank would ask about before using a PD model: whether the probabilities are calibrated, whether the "
          "model is stable over time, which factors drive its decisions and how the portfolio behaves under "
          "stress. A small software application was also developed so that a user can enter a credit history and "
          "see the resulting PD."),
    ("h2", "1.1 Background"),
    ("p", "Credit risk is the risk of loss when a borrower does not meet the obligations of a loan contract. In "
          "the Basel framework the expected loss on a loan is split into three parts: the probability of default "
          "(PD), the loss given default (LGD), which is the share of the exposure that is lost if default happens, "
          "and the exposure at default (EAD) (Basel Committee on Banking Supervision [BCBS], 2006). Expected loss "
          "is simply their product. Of the three, PD is the component that depends most on the borrower's "
          "behaviour, and it is usually the first model a bank develops. Practitioners estimate that PD models "
          "make up the large majority of all models in a typical retail bank (Evstafev, 2024)."),
    ("p", "Two regulatory regimes make PD models necessary. Under the internal ratings-based (IRB) approach of "
          "Basel II and Basel III, a bank may use its own PD estimates to calculate risk-weighted assets and "
          "capital requirements, provided the models pass supervisory validation (BCBS, 2006; European Banking "
          "Authority [EBA], 2017). Under the IFRS 9 accounting standard, banks must hold reserves equal to "
          "expected credit losses, which requires 12-month PD for performing loans and lifetime PD for loans "
          "whose risk has increased significantly (International Accounting Standards Board [IASB], 2014). In "
          "both cases the model must not only rank clients correctly but also give probabilities that match the "
          "default rates actually observed."),
    ("p", "For a long time the standard tool for PD was logistic regression on a small number of carefully "
          "binned variables, presented as a scorecard (Siddiqi, 2017). Over the last decade, gradient boosting "
          "methods such as XGBoost, LightGBM and CatBoost have become common in credit scoring competitions and "
          "in many banks, because they can use hundreds of raw features and capture non-linear effects without "
          "manual binning (Lessmann et al., 2015). The price of this flexibility is a model that is harder to "
          "explain, which is why explainability tools such as SHAP are now often part of model documentation "
          "(Bussmann et al., 2021)."),
    ("h2", "1.2 Problem Statement"),
    ("p", "The starting point of this project was a baseline built earlier: the loan-level records were "
          "one-hot encoded, counted per application and passed to a LightGBM classifier. The baseline reported "
          "a cross-validated ROC AUC of 0.7606 and a best validation AUC of 0.7924. These numbers, however, "
          "were obtained with random cross-validation, in which applications from the same period appear both "
          "in training and in testing. A bank always applies a model to future applications, so a random split "
          "may give an optimistic picture. The baseline also used a class weight that pushes the predicted "
          "probabilities far above the real default rate, which makes its output unusable as a PD without "
          "further correction. Finally, the baseline did not include any analysis of stability, explanation or "
          "stress behaviour."),
    ("p", "The problem addressed in this report is therefore the following: to develop a PD model on credit "
          "bureau histories that ranks applicants at least as well as the baseline when tested on later "
          "applications, produces calibrated probabilities, can be explained, and can be used to estimate how "
          "portfolio risk changes under adverse conditions."),
    ("h2", "1.3 Aim and Objectives"),
    ("p", "The aim of the project is to build, validate and deploy a probability of default model for retail "
          "applicants using gradient boosting on credit bureau data. The following objectives were set:"),
    ("list", [
        "To review the literature on PD modelling, its regulatory background, validation and stress testing.",
        "To reproduce the existing LightGBM baseline and evaluate it with a time-based out-of-time (OOT) test.",
        "To design additional features from the payment history and to select a compact feature set.",
        "To train and compare logistic regression, LightGBM and CatBoost models on the same split.",
        "To validate the final model: discrimination, calibration, rating-grade tests and stability over time.",
        "To explain the model with SHAP values on a subsample of applicants.",
        "To stress test the portfolio with top-down, bottom-up and reverse scenarios.",
        "To develop a software application that returns the PD, rating grade and main risk drivers for a user's input.",
    ]),
    ("h2", "1.4 Scope of the Study"),
    ("p", "The study covers application-level PD with a fixed performance window, as defined in the dataset: "
          "the target equals one if the client became more than 90 days past due. The data contains only "
          "credit bureau information; there are no demographic, income or application form variables, and all "
          "numeric fields were binned and coded by the data provider. LGD and EAD are not modelled; where they "
          "are needed for expected loss or capital, standard assumptions are used and stated. The dataset has no "
          "calendar dates, so the stress tests rely on the Basel one-factor model and on shocks to borrower "
          "records rather than on a regression of default rates on real macroeconomic series. Deep learning on "
          "the raw sequences was outside the scope because of the available hardware."),
    ("h2", "1.5 Structure of the Report"),
    ("p", "Chapter 2 reviews the literature on credit scoring, PD validation, explainability and stress "
          "testing. Chapter 3 describes the data, the features, the validation design, the models and the stress "
          "testing method. Chapter 4 presents the results of all experiments and the software application. "
          "Chapter 5 gives the conclusions, the limitations and the directions for future work. The appendices "
          "contain the project structure, key parts of the source code, a short user guide for the application "
          "and additional result tables."),
]

CH2 = [
    ("h1", "2. LITERATURE REVIEW"),
    ("p", "This chapter summarises the research and guidance that shaped the choices made in the project. It "
          "moves from the general idea of credit scoring to the specific methods used later: gradient boosting, "
          "PD validation, SHAP and stress testing."),
    ("h2", "2.1 Credit Scoring and the Probability of Default"),
    ("p", "Credit scoring is the use of statistical models to estimate the risk that a borrower will not repay. "
          "Thomas (2000) describes two main types. Application scoring is used when a new client asks for a "
          "loan, while behavioural scoring is applied to existing clients using their account history. Hand and "
          "Henley (1997) review the early statistical methods and point out that the definition of a \"bad\" "
          "client, the choice of the observation and performance windows and the sample design matter as much as "
          "the algorithm itself. The most common definition of default, which is also used in this project, is "
          "a payment more than 90 days past due, in line with the Basel definition (BCBS, 2006)."),
    ("p", "A PD model gives a number between zero and one, but in practice banks rarely use it directly. The "
          "probabilities are grouped into rating grades on a master scale, each grade with a PD range, and "
          "decisions such as approval or pricing are set per grade (Siddiqi, 2017). Bhalla (2019) also "
          "distinguishes point-in-time (PIT) PD, which reflects current economic conditions, from "
          "through-the-cycle (TTC) PD, which reflects the long-run average. Accounting provisions under IFRS 9 "
          "are based on PIT estimates, while regulatory capital is closer to TTC. For loans with a longer life, "
          "the 12-month PD can be extended to a lifetime PD by chaining conditional default probabilities over "
          "the years (Bhalla, 2019)."),
    ("p", "Industry accounts give a practical view of how such models are built. Evstafev (2024) stresses that "
          "the strongest predictors in retail PD models are past delinquency and the length of the credit "
          "history, while demographic variables are weak, and that ROC AUC and the Gini coefficient are "
          "preferred to threshold-based metrics such as precision and recall. A large PD model for small "
          "business clients described by Tochka Bank (2022) used overlapping quarterly cohorts, an out-of-time "
          "validation sample and a selection step that compared the SHAP importance of each factor with that of "
          "random noise variables. Both sources show that time-based validation and feature stability are "
          "treated as essential in banking practice."),
    ("h2", "2.2 Regulatory Context: Basel IRB and IFRS 9"),
    ("p", "The Basel II framework allows banks to calculate capital for credit risk with the IRB approach. The "
          "capital requirement for each exposure is derived from the asymptotic single risk factor model, in "
          "which the default of every borrower depends on one common systematic factor and on an individual "
          "factor (Gordy, 2003). The formula is based on the work of Vasicek (2002) and gives the PD conditional "
          "on a very bad realisation of the systematic factor, at the 99.9th percentile. For retail exposures the "
          "correlation with the systematic factor is fixed by the regulator: 0.15 for mortgages, 0.04 for "
          "qualifying revolving exposures, and a value between 0.03 and 0.16 that decreases with PD for other "
          "retail loans (BCBS, 2005a). The same formula is often used outside capital calculation as a simple "
          "way to stress PDs."),
    ("p", "The EBA (2017) guidelines on PD estimation require banks to use a long-run average default rate for "
          "calibration, to group clients into grades with homogeneous risk and to test the model regularly for "
          "discrimination, calibration and stability. The Basel Committee's working paper on validation (BCBS, "
          "2005b) lists the tests commonly used for this purpose, including the accuracy ratio (Gini), the "
          "binomial test per grade and the Hosmer-Lemeshow test. These tests were adopted in Chapter 4."),
    ("h2", "2.3 Statistical and Machine Learning Methods for PD"),
    ("p", "Logistic regression remains the reference method in credit scoring. It is easy to explain and to "
          "document, and the coefficients can be turned into scorecard points (Siddiqi, 2017). Its weakness is "
          "that non-linear relationships and interactions have to be built in by hand, usually through binning "
          "and weight-of-evidence transformations."),
    ("p", "Large benchmark studies have compared many classifiers on credit data. Baesens et al. (2003) found "
          "that several methods, including logistic regression and neural networks, performed similarly on the "
          "datasets of that time. The update by Lessmann et al. (2015), which used 41 classifiers on eight "
          "datasets, showed that ensembles, especially heterogeneous ensembles and boosted trees, were usually "
          "better than single models. Gunnarsson et al. (2021) tested deep learning for credit scoring and "
          "concluded that it did not outperform gradient boosting on tabular credit data. This agrees with "
          "general studies of tabular data, where tree-based models are still a strong default choice "
          "(Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022). Babaev et al. (2019) showed, on the other hand, "
          "that recurrent networks applied directly to sequences of banking transactions can beat models built "
          "on hand-made aggregates, which suggests that the sequence structure of credit histories carries extra "
          "information."),
    ("h2", "2.4 Gradient Boosting: LightGBM and CatBoost"),
    ("p", "Gradient boosting builds an additive model of many small decision trees, where each new tree is fitted "
          "to the gradient of the loss function of the current ensemble (Friedman, 2001). LightGBM speeds up "
          "this process with histogram-based splits, leaf-wise tree growth and sampling techniques that reduce "
          "the number of examples and features examined at each step (Ke et al., 2017). It was the model used in "
          "the baseline of this project."),
    ("p", "CatBoost, developed by Yandex, differs from other implementations in two ways (Prokhorenkova et al., "
          "2018). It uses ordered boosting, which reduces the prediction shift caused by fitting each tree on the "
          "same examples whose residuals it uses, and it builds symmetric (oblivious) trees in which the same "
          "split is applied at every node of a level. Symmetric trees act as a form of regularisation and make "
          "prediction very fast. CatBoost also provides efficient GPU training (Dorogush et al., 2018), which "
          "was important here because the training set contains two million applications."),
    ("h2", "2.5 Validation of PD Models"),
    ("p", "PD validation is usually divided into three areas. Discrimination measures how well the model ranks "
          "defaulters above non-defaulters. The ROC curve and its area (AUC) are the standard tools (Fawcett, "
          "2006); in banking the Gini coefficient, equal to 2·AUC − 1, and the Kolmogorov-Smirnov (KS) statistic "
          "are reported as well (Siddiqi, 2017). Calibration measures whether the predicted probabilities agree "
          "with the observed default rates. Tasche (2008) reviews the binomial test, which checks each rating "
          "grade separately, and the Hosmer-Lemeshow test, which compares observed and expected defaults across "
          "groups (Hosmer et al., 2013). Stability measures whether the population and the model behaviour "
          "change over time; the population stability index (PSI) is the most widely used measure, and its "
          "statistical properties were studied by Yurdakul and Naranjo (2020)."),
    ("p", "Machine learning models often need a separate calibration step. Niculescu-Mizil and Caruana (2005) "
          "showed that boosted trees tend to give distorted probabilities and that Platt scaling (Platt, 1999) "
          "or isotonic regression (Zadrozny & Elkan, 2002) can correct them. Reweighting the classes during "
          "training, which is a common way to deal with rare defaults, changes the predicted probabilities as "
          "well and must be undone before the output can be read as a PD."),
    ("h2", "2.6 Explainability in Credit Models"),
    ("p", "Lenders must be able to explain their decisions to regulators and, in many countries, to customers. "
          "SHAP (SHapley Additive exPlanations) assigns to each feature a contribution to an individual "
          "prediction, based on Shapley values from cooperative game theory, so that the contributions add up "
          "to the difference between the prediction and the average prediction (Lundberg & Lee, 2017). For tree "
          "ensembles the values can be calculated exactly and quickly (Lundberg et al., 2020). Bussmann et al. "
          "(2021) applied SHAP to a credit risk model of small companies and showed that it can group borrowers "
          "with similar risk profiles and make a boosted model acceptable from a supervisory point of view."),
    ("h2", "2.7 Stress Testing of Credit Portfolios"),
    ("p", "Stress testing asks how a portfolio would behave under extreme but plausible events. Bhalla (2023) "
          "lists three types: scenario analysis, in which a set of adverse economic conditions is applied; "
          "sensitivity analysis, in which one risk factor is changed at a time; and reverse stress testing, which "
          "starts from an unacceptable outcome and works back to the conditions that would cause it. Typical "
          "scenario variables include GDP, unemployment, inflation, interest rates and house prices, for example "
          "an increase in unemployment by several percentage points (Bhalla, 2019, 2023)."),
    ("p", "In a full stress testing framework, the default rate of each segment is linked to macroeconomic "
          "variables through a regression model, and the macro scenarios are then translated into stressed PDs. "
          "Bellotti and Crook (2009, 2013) included macroeconomic variables in survival and dynamic models of "
          "consumer default and used them to forecast default rates under stress. When such time series are not "
          "available, the Vasicek one-factor model provides a consistent shortcut: a scenario is expressed as a "
          "quantile of the systematic factor, and every PD is moved to its conditional value at that quantile "
          "(Vasicek, 2002; BCBS, 2005a). This approach was used in the present project, together with "
          "borrower-level shocks that are passed through the model itself."),
    ("h2", "2.8 Summary and Research Gap"),
    ("p", "The literature suggests that gradient boosting is a strong choice for PD on large tabular data, but "
          "that its results must be checked with time-based validation, calibrated, and explained. Most published "
          "studies report only discrimination on a random split. This project combines, on one dataset of three "
          "million applications, a reproduction of a random-split baseline, an out-of-time comparison of three "
          "model families, the validation tests recommended by the regulators, a SHAP analysis and a stress "
          "testing exercise, and makes the final model available in a working application."),
]
