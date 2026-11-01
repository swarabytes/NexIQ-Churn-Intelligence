# Academic Classification Report — Olist Customer Churn Prediction

---

## A. Dataset

| Property | Value |
|---|---|
| Source | Brazilian E-Commerce (Olist) — Kaggle |
| Dataset file | data/olist_churn_dataset.csv |
| Total rows | 1,683 |
| Features used | 13 (all behavioural) |
| Target variable | churn |
| Class 0 (retained) | 67 (3.98%) |
| Class 1 (churned)  | 1,616 (96.02%) |
| Majority-class baseline | 96.02% |

**Churn definition:**
> Churn = 1 → the customer made NO purchase during the 180-day observation window after the prediction cutoff.
> Churn = 0 → the customer made AT LEAST ONE purchase during the 180-day window.

### Train / Test Split

| Split | Rows | Class 0 | Class 1 |
|---|---|---|---|
| Training (80%) | 1346 | 54 | 1292 |
| Testing (20%)  | 337  | 13  | 324  |

Split parameters: `test_size=0.20, stratify=y, random_state=42` — identical for all models.

---

## B. Preprocessing

All preprocessing steps were fitted **exclusively on training data** and then applied to the test set.

| Step | Detail |
|---|---|
| has_review | Binary flag derived from `avg_review_score` BEFORE any imputation. 1 = review existed; 0 = no review before cutoff. |
| Missing values | `SimpleImputer(strategy='median')` — training-set median applied to test set |
| Scaling | `StandardScaler()` — zero mean, unit variance fitted on training set only |
| Dropped | `customer_unique_id`, `cutoff_date`, `customer_state` |
| Categorical | None remaining after dropping `customer_state` |

---

## C. Models

### 1. Logistic Regression
A linear probabilistic classifier that models the log-odds of churn as a linear function of the features.
`class_weight='balanced'` adjusts internal sample weights inversely proportional to class frequencies,
allowing the model to learn from the minority class (retained = 67 samples) without oversampling.

### 2. Decision Tree
A non-parametric classifier that recursively partitions the feature space into rectangular regions.
`max_depth=5` prevents overfitting. `class_weight='balanced'` adjusts split criteria for imbalance.
Provides interpretable rules and native feature importance via Gini impurity reduction.

### 3. Random Forest
An ensemble of 100 decision trees trained on bootstrapped samples with random feature subsets.
Reduces variance compared to a single tree. `max_depth=10`, `class_weight='balanced'`.
Feature importance is averaged across all trees (mean decrease in impurity).

### 4. Gaussian Naive Bayes
A probabilistic classifier assuming conditional independence of features given the class,
with Gaussian-distributed feature likelihoods. No `class_weight` parameter available.
Balanced sample weights applied to GaussianNB.fit() using sklearn.utils.class_weight.compute_sample_weight (computed from y_train only).
Standard feature importance is not directly available for Gaussian Naive Bayes.

---

## D. Results

| Model | Accuracy | Precision | Recall | F1 | Specificity | Bal. Acc | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9614 | 0.9642 | 0.9969 | 0.9803 | 0.0769 | 0.5369 | 0.6472 | 0.9728 |
| Naive Bayes | 0.8249 | 0.9783 | 0.8364 | 0.9018 | 0.5385 | 0.6874 | 0.6702 | 0.9747 |
| Logistic Regression | 0.7211 | 0.9752 | 0.7284 | 0.8339 | 0.5385 | 0.6334 | 0.6377 | 0.9694 |
| Decision Tree | 0.7033 | 0.9746 | 0.7099 | 0.8214 | 0.5385 | 0.6242 | 0.6090 | 0.9688 |

**Majority-class baseline (predict always Churn=1):** 0.9614

---

## E. Confusion Matrices

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 7 | 6 | 88 | 236 |
| Decision Tree | 7 | 6 | 94 | 230 |
| Random Forest | 1 | 12 | 1 | 323 |
| Naive Bayes | 7 | 6 | 53 | 271 |

Specificity = TN / (TN + FP) — correctly identified retained customers.

---

## F. Interpretation

**Best model (composite score): Random Forest**

Model selection criteria (in priority order):
1. F1-score (harmonic mean of Precision and Recall)
2. Recall (minimise False Negatives — missed churners)
3. Precision (avoid excessive false alarms)
4. Balanced Accuracy
5. ROC-AUC
6. PR-AUC

> **Accuracy is NOT the primary metric.** A dummy model that predicts Churn=1 for every customer
> achieves 96.02% accuracy. Therefore accuracy alone is meaningless for this dataset.

---

## G. Limitations

> **Minority class size:** The dataset contains only 67 retained customers out of 1,683 observations.
> Therefore minority-class metrics — particularly recall, precision, and F1-score — may have high variance.

> **Majority-class baseline:** The 96.02% majority-class baseline makes accuracy an unreliable primary metric.

> **SMOTE:** No oversampling was performed in this experiment. If minority-class recall is unacceptably low,
> SMOTE applied strictly after the train/test split may be considered in a subsequent experiment.

> **Dataset scope:** The Olist dataset covers 2016–2018 Brazilian e-commerce. Results may not generalise
> to other markets or time periods.