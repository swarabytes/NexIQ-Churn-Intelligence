"""
Part 5D — Classification Model Training & Comparison
======================================================
Dataset    : data/olist_churn_dataset.csv
Target     : churn (1 = no purchase in 180-day future window)
Models     : Logistic Regression, Decision Tree, Random Forest, Naive Bayes
Imbalance  : class_weight='balanced' (NO SMOTE)
Split      : 80/20 stratified, random_state=42

STOP CONDITIONS:
  - Does NOT modify pipeline.pkl
  - Does NOT modify /predict route
  - Does NOT start SMOTE, regression, clustering, Apriori, or OLAP
  - Does NOT change the existing UI
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
DATASET_PATH   = 'data/olist_churn_dataset.csv'
RESULTS_DIR    = 'static/academic_results'
MODELS_DIR     = 'models/academic'
DATA_DIR       = 'data'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv(DATASET_PATH)

# Create has_review BEFORE any imputation
# has_review = 1 if the customer had a historical review before the cutoff
# Derived from: avg_review_score is NOT null => review existed
df['has_review'] = df['avg_review_score'].notna().astype(int)

# Verify has_review
assert df['has_review'].isin([0, 1]).all(), "has_review must be binary"
assert (df['has_review'] == 0).sum() == df['avg_review_score'].isna().sum(), \
    "has_review mismatch with avg_review_score nulls"

# ============================================================
# 2. DEFINE FEATURES & TARGET
# ============================================================
FEATURE_COLS = [
    'total_orders',
    'tenure_days',
    'days_since_last_order',
    'order_frequency',
    'total_spend',
    'avg_order_value',
    'avg_item_count',
    'avg_categories_per_order',
    'total_payment_types',
    'avg_installments',
    'avg_review_score',
    'review_count',
    'has_review',
]
DROP_COLS = ['customer_unique_id', 'cutoff_date', 'customer_state']
TARGET    = 'churn'

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

# ============================================================
# 3. TRAIN / TEST SPLIT (ONE split, used for ALL models)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ============================================================
# 4. PREPROCESSING PIPELINE (fitted on training data ONLY)
# ============================================================
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Save preprocessor
with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)

# ============================================================
# 5. MAJORITY BASELINE
# ============================================================
majority_class     = y_test.mode()[0]
majority_baseline  = (y_test == majority_class).mean()

# ============================================================
# 6. METRIC HELPER
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy          = accuracy_score(y_true, y_pred)
    precision         = precision_score(y_true, y_pred, zero_division=0)
    recall            = recall_score(y_true, y_pred, zero_division=0)
    f1                = f1_score(y_true, y_pred, zero_division=0)
    specificity       = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc           = balanced_accuracy_score(y_true, y_pred)
    roc_auc           = roc_auc_score(y_true, y_proba)
    pr_auc            = average_precision_score(y_true, y_proba)

    return {
        'model':            model_name,
        'accuracy':         round(float(accuracy),    4),
        'precision':        round(float(precision),   4),
        'recall':           round(float(recall),      4),
        'f1':               round(float(f1),          4),
        'specificity':      round(float(specificity), 4),
        'balanced_accuracy':round(float(bal_acc),     4),
        'roc_auc':          round(float(roc_auc),     4),
        'pr_auc':           round(float(pr_auc),      4),
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp),
        'y_proba': y_proba,   # kept in memory for curves, removed before JSON save
    }

# ============================================================
# 7. CONFUSION MATRIX PLOT HELPER
# ============================================================
PALETTE = {
    'Logistic Regression': '#4C72B0',
    'Decision Tree':       '#DD8452',
    'Random Forest':       '#55A868',
    'Naive Bayes':         '#C44E52',
}

def plot_confusion_matrix(m, title, save_path):
    cm_arr = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    fig, ax = plt.subplots(figsize=(5, 4))
    color = PALETTE.get(title, '#4C72B0')
    im = ax.imshow(cm_arr, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{title}\nConfusion Matrix', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Retained (0)', 'Churned (1)'], fontsize=9)
    ax.set_yticklabels(['Retained (0)', 'Churned (1)'], fontsize=9)
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            label = ['TN','FP','FN','TP'][i*2+j]
            ax.text(j, i, f'{label}\n{cm_arr[i,j]}',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_arr[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 8. TRAIN MODELS
# ============================================================
all_metrics = []
all_probas  = {}

# ---- MODEL 1: LOGISTIC REGRESSION ----
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_proc, y_train)
lr_pred  = lr.predict(X_test_proc)
lr_proba = lr.predict_proba(X_test_proc)[:, 1]
m_lr     = compute_metrics(y_test, lr_pred, lr_proba, 'Logistic Regression')
all_metrics.append(m_lr)
all_probas['Logistic Regression'] = lr_proba
with open(os.path.join(MODELS_DIR, 'logistic_regression.pkl'), 'wb') as f:
    pickle.dump(lr, f)

# ---- MODEL 2: DECISION TREE ----
dt = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
dt.fit(X_train_proc, y_train)
dt_pred  = dt.predict(X_test_proc)
dt_proba = dt.predict_proba(X_test_proc)[:, 1]
m_dt     = compute_metrics(y_test, dt_pred, dt_proba, 'Decision Tree')
all_metrics.append(m_dt)
all_probas['Decision Tree'] = dt_proba
with open(os.path.join(MODELS_DIR, 'decision_tree.pkl'), 'wb') as f:
    pickle.dump(dt, f)

# ---- MODEL 3: RANDOM FOREST ----
rf = RandomForestClassifier(
    n_estimators=100, class_weight='balanced',
    max_depth=10, random_state=42, n_jobs=-1
)
rf.fit(X_train_proc, y_train)
rf_pred  = rf.predict(X_test_proc)
rf_proba = rf.predict_proba(X_test_proc)[:, 1]
m_rf     = compute_metrics(y_test, rf_pred, rf_proba, 'Random Forest')
all_metrics.append(m_rf)
all_probas['Random Forest'] = rf_proba
with open(os.path.join(MODELS_DIR, 'random_forest.pkl'), 'wb') as f:
    pickle.dump(rf, f)

# ---- MODEL 4: GAUSSIAN NAIVE BAYES ----
# Balanced sample weights computed ONLY from y_train
gnb_sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
gnb = GaussianNB()
try:
    gnb.fit(X_train_proc, y_train, sample_weight=gnb_sample_weights)
    gnb_weighted = True
except TypeError:
    gnb.fit(X_train_proc, y_train)
    gnb_weighted = False
gnb_pred  = gnb.predict(X_test_proc)
gnb_proba = gnb.predict_proba(X_test_proc)[:, 1]
m_gnb     = compute_metrics(y_test, gnb_pred, gnb_proba, 'Naive Bayes')
m_gnb['gnb_weighted'] = gnb_weighted
all_metrics.append(m_gnb)
all_probas['Naive Bayes'] = gnb_proba
with open(os.path.join(MODELS_DIR, 'naive_bayes.pkl'), 'wb') as f:
    pickle.dump(gnb, f)

# ============================================================
# 9. CONFUSION MATRIX PLOTS
# ============================================================
cm_files = {
    'Logistic Regression': 'confusion_matrix_logistic_regression.png',
    'Decision Tree':       'confusion_matrix_decision_tree.png',
    'Random Forest':       'confusion_matrix_random_forest.png',
    'Naive Bayes':         'confusion_matrix_naive_bayes.png',
}
for m in all_metrics:
    name = m['model']
    path = os.path.join(RESULTS_DIR, cm_files[name])
    plot_confusion_matrix(m, name, path)

# ============================================================
# 10. ROC CURVE (combined)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
for m in all_metrics:
    name  = m['model']
    fpr, tpr, _ = roc_curve(y_test, all_probas[name])
    ax.plot(fpr, tpr,
            label=f"{name} (AUC={m['roc_auc']:.3f})",
            linewidth=2, color=PALETTE[name])
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve — All Classification Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 11. PRECISION-RECALL CURVE (combined)
# ============================================================
no_skill = y_test.mean()
fig, ax = plt.subplots(figsize=(7, 6))
for m in all_metrics:
    name  = m['model']
    prec, rec, _ = precision_recall_curve(y_test, all_probas[name])
    ax.plot(rec, prec,
            label=f"{name} (PR-AUC={m['pr_auc']:.3f})",
            linewidth=2, color=PALETTE[name])
ax.axhline(y=no_skill, color='k', linestyle='--', linewidth=1,
           label=f'No Skill ({no_skill:.3f})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve — All Classification Models', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

# --- Logistic Regression: coefficients ---
lr_coef_df = pd.DataFrame({
    'feature':              FEATURE_COLS,
    'coefficient':          lr.coef_[0].tolist(),
    'absolute_coefficient': np.abs(lr.coef_[0]).tolist(),
}).sort_values('absolute_coefficient', ascending=False)
lr_coef_df.to_csv(os.path.join(DATA_DIR, 'logistic_feature_importance.csv'), index=False)

# --- Decision Tree: feature_importances_ ---
dt_imp_df = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': dt.feature_importances_.tolist(),
}).sort_values('importance', ascending=False)
dt_imp_df.to_csv(os.path.join(DATA_DIR, 'decision_tree_feature_importance.csv'), index=False)

# Decision Tree feature importance plot
fig, ax = plt.subplots(figsize=(8, 5))
top_dt = dt_imp_df.head(10)
ax.barh(top_dt['feature'][::-1], top_dt['importance'][::-1],
        color=PALETTE['Decision Tree'])
ax.set_xlabel('Feature Importance', fontsize=10)
ax.set_title('Decision Tree — Feature Importances (Top 10)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_decision_tree.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# --- Random Forest: feature_importances_ ---
rf_imp_df = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': rf.feature_importances_.tolist(),
}).sort_values('importance', ascending=False)
rf_imp_df.to_csv(os.path.join(DATA_DIR, 'random_forest_feature_importance.csv'), index=False)

# Random Forest feature importance plot
fig, ax = plt.subplots(figsize=(8, 5))
top_rf = rf_imp_df.head(10)
ax.barh(top_rf['feature'][::-1], top_rf['importance'][::-1],
        color=PALETTE['Random Forest'])
ax.set_xlabel('Feature Importance', fontsize=10)
ax.set_title('Random Forest — Feature Importances (Top 10)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_random_forest.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# --- Naive Bayes: no feature importance ---
# Standard feature importance is not directly available for Gaussian Naive Bayes.
# GNB learns per-class mean and variance of each feature; there is no single
# coefficient or impurity-based score that maps to "importance" in the same
# sense as tree-based or linear models.

# ============================================================
# 13. METRICS JSON
# ============================================================
json_metrics = {}
for m in all_metrics:
    json_metrics[m['model']] = {
        'accuracy':         m['accuracy'],
        'precision':        m['precision'],
        'recall':           m['recall'],
        'f1':               m['f1'],
        'specificity':      m['specificity'],
        'balanced_accuracy':m['balanced_accuracy'],
        'roc_auc':          m['roc_auc'],
        'pr_auc':           m['pr_auc'],
    }

with open(os.path.join(DATA_DIR, 'academic_classification_metrics.json'), 'w') as f:
    json.dump(json_metrics, f, indent=2)

# ============================================================
# 14. COMPARISON CSV (sorted by F1 desc)
# ============================================================
rows = []
for m in all_metrics:
    rows.append({
        'Model':            m['model'],
        'Accuracy':         m['accuracy'],
        'Precision':        m['precision'],
        'Recall':           m['recall'],
        'F1':               m['f1'],
        'Specificity':      m['specificity'],
        'Balanced_Accuracy':m['balanced_accuracy'],
        'ROC_AUC':          m['roc_auc'],
        'PR_AUC':           m['pr_auc'],
        'TN':               m['tn'],
        'FP':               m['fp'],
        'FN':               m['fn'],
        'TP':               m['tp'],
    })
df_cmp = pd.DataFrame(rows).sort_values('F1', ascending=False)
df_cmp.to_csv(os.path.join(DATA_DIR, 'academic_classification_comparison.csv'), index=False)

# ============================================================
# 15. IDENTIFY BEST MODEL (by composite: F1, Recall, Balanced Acc, ROC-AUC, PR-AUC)
# ============================================================
def composite_score(m):
    return (m['f1'] * 2 + m['recall'] + m['balanced_accuracy'] +
            m['roc_auc'] + m['pr_auc']) / 6.0

best_model = max(all_metrics, key=composite_score)

# ============================================================
# 16. ACADEMIC CLASSIFICATION REPORT (Markdown)
# ============================================================
train_c0 = int((y_train == 0).sum())
train_c1 = int((y_train == 1).sum())
test_c0  = int((y_test  == 0).sum())
test_c1  = int((y_test  == 1).sum())

nb_note = ("Balanced sample weights applied to GaussianNB.fit() using "
           "sklearn.utils.class_weight.compute_sample_weight (computed from y_train only)."
           if m_gnb['gnb_weighted'] else
           "GaussianNB trained without sample weights (sample_weight parameter not "
           "supported in this sklearn version). Results may be biased toward majority class.")

report_lines = [
    "# Academic Classification Report — Olist Customer Churn Prediction",
    "",
    "---",
    "",
    "## A. Dataset",
    "",
    f"| Property | Value |",
    f"|---|---|",
    f"| Source | Brazilian E-Commerce (Olist) — Kaggle |",
    f"| Dataset file | data/olist_churn_dataset.csv |",
    f"| Total rows | 1,683 |",
    f"| Features used | 13 (all behavioural) |",
    f"| Target variable | churn |",
    f"| Class 0 (retained) | 67 (3.98%) |",
    f"| Class 1 (churned)  | 1,616 (96.02%) |",
    f"| Majority-class baseline | 96.02% |",
    "",
    "**Churn definition:**",
    "> Churn = 1 → the customer made NO purchase during the 180-day observation window after the prediction cutoff.",
    "> Churn = 0 → the customer made AT LEAST ONE purchase during the 180-day window.",
    "",
    "### Train / Test Split",
    "",
    f"| Split | Rows | Class 0 | Class 1 |",
    f"|---|---|---|---|",
    f"| Training (80%) | {len(X_train)} | {train_c0} | {train_c1} |",
    f"| Testing (20%)  | {len(X_test)}  | {test_c0}  | {test_c1}  |",
    "",
    "Split parameters: `test_size=0.20, stratify=y, random_state=42` — identical for all models.",
    "",
    "---",
    "",
    "## B. Preprocessing",
    "",
    "All preprocessing steps were fitted **exclusively on training data** and then applied to the test set.",
    "",
    "| Step | Detail |",
    "|---|---|",
    "| has_review | Binary flag derived from `avg_review_score` BEFORE any imputation. 1 = review existed; 0 = no review before cutoff. |",
    "| Missing values | `SimpleImputer(strategy='median')` — training-set median applied to test set |",
    "| Scaling | `StandardScaler()` — zero mean, unit variance fitted on training set only |",
    "| Dropped | `customer_unique_id`, `cutoff_date`, `customer_state` |",
    "| Categorical | None remaining after dropping `customer_state` |",
    "",
    "---",
    "",
    "## C. Models",
    "",
    "### 1. Logistic Regression",
    "A linear probabilistic classifier that models the log-odds of churn as a linear function of the features.",
    "`class_weight='balanced'` adjusts internal sample weights inversely proportional to class frequencies,",
    "allowing the model to learn from the minority class (retained = 67 samples) without oversampling.",
    "",
    "### 2. Decision Tree",
    "A non-parametric classifier that recursively partitions the feature space into rectangular regions.",
    "`max_depth=5` prevents overfitting. `class_weight='balanced'` adjusts split criteria for imbalance.",
    "Provides interpretable rules and native feature importance via Gini impurity reduction.",
    "",
    "### 3. Random Forest",
    "An ensemble of 100 decision trees trained on bootstrapped samples with random feature subsets.",
    "Reduces variance compared to a single tree. `max_depth=10`, `class_weight='balanced'`.",
    "Feature importance is averaged across all trees (mean decrease in impurity).",
    "",
    "### 4. Gaussian Naive Bayes",
    "A probabilistic classifier assuming conditional independence of features given the class,",
    "with Gaussian-distributed feature likelihoods. No `class_weight` parameter available.",
    nb_note,
    "Standard feature importance is not directly available for Gaussian Naive Bayes.",
    "",
    "---",
    "",
    "## D. Results",
    "",
    "| Model | Accuracy | Precision | Recall | F1 | Specificity | Bal. Acc | ROC-AUC | PR-AUC |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for m in sorted(all_metrics, key=lambda x: x['f1'], reverse=True):
    report_lines.append(
        f"| {m['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
        f"{m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | "
        f"{m['balanced_accuracy']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
    )

report_lines += [
    "",
    f"**Majority-class baseline (predict always Churn=1):** {majority_baseline:.4f}",
    "",
    "---",
    "",
    "## E. Confusion Matrices",
    "",
    "| Model | TN | FP | FN | TP |",
    "|---|---:|---:|---:|---:|",
]
for m in all_metrics:
    report_lines.append(
        f"| {m['model']} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |"
    )

report_lines += [
    "",
    "Specificity = TN / (TN + FP) — correctly identified retained customers.",
    "",
    "---",
    "",
    "## F. Interpretation",
    "",
    f"**Best model (composite score): {best_model['model']}**",
    "",
    "Model selection criteria (in priority order):",
    "1. F1-score (harmonic mean of Precision and Recall)",
    "2. Recall (minimise False Negatives — missed churners)",
    "3. Precision (avoid excessive false alarms)",
    "4. Balanced Accuracy",
    "5. ROC-AUC",
    "6. PR-AUC",
    "",
    "> **Accuracy is NOT the primary metric.** A dummy model that predicts Churn=1 for every customer",
    "> achieves 96.02% accuracy. Therefore accuracy alone is meaningless for this dataset.",
    "",
    "---",
    "",
    "## G. Limitations",
    "",
    "> **Minority class size:** The dataset contains only 67 retained customers out of 1,683 observations.",
    "> Therefore minority-class metrics — particularly recall, precision, and F1-score — may have high variance.",
    "",
    "> **Majority-class baseline:** The 96.02% majority-class baseline makes accuracy an unreliable primary metric.",
    "",
    "> **SMOTE:** No oversampling was performed in this experiment. If minority-class recall is unacceptably low,",
    "> SMOTE applied strictly after the train/test split may be considered in a subsequent experiment.",
    "",
    "> **Dataset scope:** The Olist dataset covers 2016–2018 Brazilian e-commerce. Results may not generalise",
    "> to other markets or time periods.",
]

report_text = "\n".join(report_lines)
with open(os.path.join(DATA_DIR, 'academic_classification_report.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)

# ============================================================
# 17. FINAL CONSOLE OUTPUT
# ============================================================
SEP = "=" * 40
SEP2 = "-" * 40

print(SEP)
print("PART 5D -- CLASSIFICATION RESULTS")
print(SEP)

print("\nDATASET")
print(f"Rows:     {len(df)}")
print(f"Features: {len(FEATURE_COLS)}")

print(f"\nTRAIN / TEST")
print(SEP2)
print(f"Train:          {len(X_train)}")
print(f"Test:           {len(X_test)}")
print(f"\nTrain Class 0:  {train_c0}")
print(f"Train Class 1:  {train_c1}")
print(f"\nTest Class 0:   {test_c0}")
print(f"Test Class 1:   {test_c1}")

print(f"\nMAJORITY BASELINE: {majority_baseline*100:.2f}%")
print("(A model predicting churn for every customer achieves this accuracy.)")

print(f"\nMODEL RESULTS")
print(SEP)

for m in all_metrics:
    print(f"\n{m['model']}")
    print(f"  Accuracy:         {m['accuracy']:.4f}")
    print(f"  Precision:        {m['precision']:.4f}")
    print(f"  Recall:           {m['recall']:.4f}")
    print(f"  F1:               {m['f1']:.4f}")
    print(f"  Specificity:      {m['specificity']:.4f}")
    print(f"  Balanced Accuracy:{m['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:          {m['roc_auc']:.4f}")
    print(f"  PR-AUC:           {m['pr_auc']:.4f}")
    print(f"  TN:{m['tn']}  FP:{m['fp']}  FN:{m['fn']}  TP:{m['tp']}")
    if m['model'] == 'Naive Bayes':
        status = "Balanced sample weights applied" if m.get('gnb_weighted') else "No sample weights (not supported)"
        print(f"  GNB weight status: {status}")

print(f"\nBEST MODEL")
print(SEP)
print(f"Model:          {best_model['model']}")
print(f"F1:             {best_model['f1']:.4f}")
print(f"Recall:         {best_model['recall']:.4f}")
print(f"ROC-AUC:        {best_model['roc_auc']:.4f}")
print(f"PR-AUC:         {best_model['pr_auc']:.4f}")
print(f"Balanced Acc:   {best_model['balanced_accuracy']:.4f}")
print(f"Primary reason: Highest composite score across F1, Recall, Balanced Accuracy, ROC-AUC, PR-AUC")

print(f"\nFILES CREATED")
print(SEP)
print(f"Metrics JSON:           data/academic_classification_metrics.json")
print(f"Comparison CSV:         data/academic_classification_comparison.csv")
print(f"Classification report:  data/academic_classification_report.md")
print(f"ROC curve:              static/academic_results/roc_curve_comparison.png")
print(f"PR curve:               static/academic_results/pr_curve_comparison.png")
print(f"Confusion matrices:     static/academic_results/confusion_matrix_*.png (4 files)")
print(f"Feature importance:     data/logistic_feature_importance.csv")
print(f"                        data/decision_tree_feature_importance.csv")
print(f"                        data/random_forest_feature_importance.csv")
print(f"                        static/academic_results/feature_importance_decision_tree.png")
print(f"                        static/academic_results/feature_importance_random_forest.png")
print(f"Academic model files:   models/academic/logistic_regression.pkl")
print(f"                        models/academic/decision_tree.pkl")
print(f"                        models/academic/random_forest.pkl")
print(f"                        models/academic/naive_bayes.pkl")
print(f"                        models/academic/preprocessor.pkl")

print(f"\nSMOTE:              NOT PERFORMED")
print(f"UI CHANGES:         NONE")
print(f"EXISTING PIPELINE:  UNCHANGED (pipeline.pkl not touched)")
print(SEP)
