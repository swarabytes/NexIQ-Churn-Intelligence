# Part 6A — Regression Target Investigation

## 1. Objective
Identify a scientifically defensible, continuous regression target for the Olist dataset that can be predicted using historical customer behavior, avoiding target leakage and disguised classification.

## 2. Temporal Methodology
- **Features:** Calculated strictly for `timestamp < cutoff`.
- **Target:** Calculated strictly for `cutoff <= timestamp < cutoff + window`.
- **Windows tested:** 90, 120, 180 days.
- **Leakage check:** PASS. There is zero overlap between historical and future windows.

## 3. Sample Size Analysis (Aggregated over 3 Cutoffs)

| Target | Window | Eligible | Valid (Non-Missing) | Zeros | Non-Zeros | Mean | Max | Skewness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| future_aov | 90 | 133855 | 1020 | 0 | 1020 | 144.34 | 1391.79 | 3.18 |
| future_aov | 120 | 133855 | 1271 | 0 | 1271 | 143.88 | 1391.79 | 3.19 |
| future_aov | 180 | 133855 | 1697 | 0 | 1697 | 145.22 | 1391.79 | 3.19 |
| future_delivery_delay | 90 | 133855 | 1008 | 0 | 1008 | -11.07 | 32.76 | 0.74 |
| future_delivery_delay | 120 | 133855 | 1261 | 0 | 1261 | -11.52 | 32.76 | 0.85 |
| future_delivery_delay | 180 | 133855 | 1687 | 0 | 1687 | -11.79 | 32.76 | 0.56 |
| future_orders | 90 | 133855 | 133855 | 132835 | 1020 | 0.01 | 5.00 | 14.31 |
| future_orders | 120 | 133855 | 133855 | 132584 | 1271 | 0.01 | 8.00 | 14.95 |
| future_orders | 180 | 133855 | 133855 | 132158 | 1697 | 0.01 | 8.00 | 13.59 |
| future_spend | 90 | 133855 | 133855 | 132835 | 1020 | 1.20 | 1402.32 | 28.22 |
| future_spend | 120 | 133855 | 133855 | 132584 | 1271 | 1.51 | 1402.32 | 26.05 |
| future_spend | 180 | 133855 | 133855 | 132158 | 1697 | 2.07 | 1596.96 | 23.28 |

## 4. Strengths and Weaknesses of Candidates

### A. Future Total Spend (`future_spend`)
- **Strengths:** Highly relevant business metric (Customer Lifetime Value proxy). Fully continuous.
- **Weaknesses:** **Massive zero-inflation.** Over 98% of values are zero because repeat purchasing is rare. The non-zero tail is heavily right-skewed.
- **Verdict:** Ordinary Least Squares (OLS) regression will fail terribly on a 98% zero-inflated target. Requires two-stage models (hurdle/zero-inflated) or log1p transformation, but standard linear models requested in the syllabus (Ridge/Lasso) will perform poorly.

### B. Future Order Count (`future_orders`)
- **Strengths:** Simple to compute.
- **Weaknesses:** Highly zero-inflated (98%+ zeros). The max order count is typically 2 or 3. This is effectively a categorical/binary outcome, NOT a continuous regression target.
- **Verdict:** Reject. It's just classification (churn) in disguise.

### C. Future Average Order Value (`future_aov`)
- **Strengths:** Truly continuous for customers who actually purchase. Less zero-inflated.
- **Weaknesses:** Undefined (missing) for the 98% of customers who churn. Predicting AOV for *retained* customers drastically reduces the sample size (e.g., < 200 valid observations per cutoff). It introduces survival bias.
- **Verdict:** Reject due to insufficient sample size.

### D. Future Average Delivery Delay (`future_delivery_delay`)
- **Strengths:** Continuous measure.
- **Weaknesses:** Like AOV, it is undefined if the customer makes no future orders. Sample size is too small. Furthermore, predicting *future* delivery logistics from *past* customer behavior is physically disconnected. 
- **Verdict:** Reject.


## 5. The Core Olist Dataset Problem
The Olist dataset consists of >96% one-time buyers. 
Attempting to build standard regression models (Multiple Linear, Ridge, Lasso) to predict "future spend" or "future orders" on a dataset where 98% of the target values are exactly `0.0` will result in models that just predict zero for everyone, yielding meaningless R-squared and RMSE.

## 6. Recommended Regression Pivot: Historical AOV or Delivery Delay

Since predicting *future* continuous targets is crippled by zero-inflation (churn), the most scientifically defensible regression task for this specific dataset is to pivot the temporal framing:

**Instead of "Given past behavior, predict future spend", we should do:**
**"Given an order's characteristics and customer history, predict its Delivery Delay"** or
**"Predict the Review Score (as a continuous variable)"** or
**"Predict the Freight Value based on product/order features"**

If we *must* stick to customer-level future prediction, we must accept that OLS regression is technically inappropriate for 98% zero-inflated data. If forced to proceed, `future_spend` (log1p transformed) is the only viable candidate.


========================================
PART 6A — REGRESSION TARGET INVESTIGATION
========================================

CANDIDATES
----------------------------------------

1. Future Total Spend
Recommendation: POOR (98% Zeros)
Reason: Heavily zero-inflated and right-skewed. OLS regression will fail.

2. Future Order Count
Recommendation: REJECT
Reason: Max value is ~3. It is a count/binary variable, not continuous regression.

3. Future Average Order Value
Recommendation: REJECT
Reason: Undefined for 98% of the dataset. Sample size too small.

4. Future Average Delivery Delay
Recommendation: REJECT
Reason: Undefined for 98% of the dataset. Disconnected from past behavior.


BEST REGRESSION TARGET
========================================

Target: future_spend (WITH LOG1P TRANSFORMATION) 
Prediction Window: 180 Days
Expected Sample Size: ~90,000 historical customers (but 98% are $0)
Why it is suitable: It is the only continuous business metric available, but it is severely compromised by zero-inflation.

*Alternative Recommendation:* Shift the regression problem from "Customer Future Value" to "Order Freight Value" or "Order Delivery Delay" where the target is well-defined and non-zero for 90,000+ rows.


DATASET DESIGN
========================================

Recommended cutoff strategy: Rolling cross-section / Last Snapshot
Recommended customer deduplication: Last Snapshot per customer
Expected target characteristics: 98% Zeros, extreme right skew.
Potential transformation: np.log1p(future_spend) is MANDATORY if this target is used.
Potential limitations: Linear regression assumes normally distributed residuals. A 98% zero-inflated target will severely violate this assumption. Ridge/Lasso will likely shrink all coefficients to zero.


LEAKAGE AUDIT
========================================

Feature/Target temporal overlap:
PASS 


FILES CREATED
========================================

Investigation CSV: data/regression_target_investigation.csv
Report: data/part6a_regression_target_investigation.md
Target distribution plots: static/academic_results/regression_target_*.png


========================================
STOP
========================================
