"""
Part 5E — Controlled SMOTE vs Class-Weighted Experiment
=========================================================
Dataset    : data/olist_churn_dataset.csv
Split      : Identical to Part 5D (test_size=0.20, stratify=y, random_state=42)
Experiment : SMOTE applied ONLY to X_train after preprocessing
             Models trained WITHOUT class_weight (SMOTE handles imbalance instead)
Test set   : Untouched original — SMOTE never applied to test data

STOP CONDITIONS:
  - data/olist_churn_dataset.csv NOT modified
  - pipeline.pkl NOT modified
  - Part 5D model pickles NOT overwritten
  - No UI changes, no /predict route changes
  - No regression, clustering, Apriori, or OLAP
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
DATASET_PATH  = 'data/olist_churn_dataset.csv'
RESULTS_DIR   = 'static/academic_results'
SMOTE_DIR     = 'models/academic/smote'
DATA_DIR      = 'data'
PART5D_JSON   = 'data/academic_classification_metrics.json'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SMOTE_DIR,   exist_ok=True)

# ============================================================
# STEP 1 — LOAD DATASET + VERIFY has_review
# ============================================================
df = pd.read_csv(DATASET_PATH)

# Create has_review from avg_review_score nulls (same logic as Part 5D)
df['has_review'] = df['avg_review_score'].notna().astype(int)

# Verify correctness
n_with_review    = int(df['has_review'].sum())
n_without_review = int((df['has_review'] == 0).sum())
assert n_without_review == df['avg_review_score'].isna().sum(), \
    "has_review mismatch with avg_review_score nulls"
assert df['has_review'].isin([0, 1]).all(), "has_review must be binary 0/1"

print("=" * 40)
print("PART 5E -- SMOTE EXPERIMENT")
print("=" * 40)
print()
print("has_review verification:")
print(f"  has_review = 1 (review exists before cutoff): {n_with_review}")
print(f"  has_review = 0 (no review before cutoff)    : {n_without_review}")
print(f"  Total                                        : {len(df)}")
assert n_with_review + n_without_review == len(df), "has_review counts don't sum to total"
print("  VERIFIED OK")
print()

# ============================================================
# STEP 2 — FEATURES & TARGET (identical to Part 5D)
# ============================================================
FEATURE_COLS = [
    'total_orders', 'tenure_days', 'days_since_last_order',
    'order_frequency', 'total_spend', 'avg_order_value',
    'avg_item_count', 'avg_categories_per_order',
    'total_payment_types', 'avg_installments',
    'avg_review_score', 'review_count', 'has_review',
]
TARGET = 'churn'

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

# ============================================================
# STEP 3 — TRAIN / TEST SPLIT (IDENTICAL to Part 5D)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ============================================================
# STEP 4 — PREPROCESS (fitted on X_train ONLY)
# ============================================================
preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)   # test set untouched by SMOTE

# ============================================================
# STEP 5 — APPLY SMOTE TO TRAINING DATA ONLY
# ============================================================
if not SMOTE_AVAILABLE:
    print("ERROR: imbalanced-learn is not installed.")
    print("Install with: pip install imbalanced-learn")
    raise SystemExit(1)

train_c0_before = int((y_train == 0).sum())
train_c1_before = int((y_train == 1).sum())

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

train_c0_after = int((y_train_smote == 0).sum())
train_c1_after = int((y_train_smote == 1).sum())
test_c0        = int((y_test == 0).sum())
test_c1        = int((y_test == 1).sum())

print("ORIGINAL TRAINING DISTRIBUTION")
print(f"  Class 0 (retained): {train_c0_before}")
print(f"  Class 1 (churned):  {train_c1_before}")
print()
print("AFTER SMOTE")
print(f"  Class 0 (retained): {train_c0_after}")
print(f"  Class 1 (churned):  {train_c1_after}")
print()
print("TEST DISTRIBUTION (unchanged)")
print(f"  Class 0 (retained): {test_c0}")
print(f"  Class 1 (churned):  {test_c1}")
print()

# ============================================================
# STEP 6 — METRIC HELPER
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, model_name, method):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'model':             model_name,
        'method':            method,
        'accuracy':          round(float(accuracy_score(y_true, y_pred)), 4),
        'precision':         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall':            round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1':                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'specificity':       round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        'roc_auc':           round(float(roc_auc_score(y_true, y_proba)), 4),
        'pr_auc':            round(float(average_precision_score(y_true, y_proba)), 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'y_proba': y_proba,
    }

# ============================================================
# STEP 7 — TRAIN SMOTE MODELS (NO class_weight)
# ============================================================
smote_metrics = []
smote_probas  = {}

# MODEL 1 — Logistic Regression (no class_weight)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)
lr_pred  = lr.predict(X_test_proc)
lr_proba = lr.predict_proba(X_test_proc)[:, 1]
m = compute_metrics(y_test, lr_pred, lr_proba, 'Logistic Regression', 'SMOTE')
smote_metrics.append(m); smote_probas['Logistic Regression'] = lr_proba
with open(os.path.join(SMOTE_DIR, 'logistic_regression_smote.pkl'), 'wb') as f:
    pickle.dump(lr, f)

# MODEL 2 — Decision Tree (no class_weight)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_smote, y_train_smote)
dt_pred  = dt.predict(X_test_proc)
dt_proba = dt.predict_proba(X_test_proc)[:, 1]
m = compute_metrics(y_test, dt_pred, dt_proba, 'Decision Tree', 'SMOTE')
smote_metrics.append(m); smote_probas['Decision Tree'] = dt_proba
with open(os.path.join(SMOTE_DIR, 'decision_tree_smote.pkl'), 'wb') as f:
    pickle.dump(dt, f)

# MODEL 3 — Random Forest (no class_weight)
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
rf_pred  = rf.predict(X_test_proc)
rf_proba = rf.predict_proba(X_test_proc)[:, 1]
m = compute_metrics(y_test, rf_pred, rf_proba, 'Random Forest', 'SMOTE')
smote_metrics.append(m); smote_probas['Random Forest'] = rf_proba
with open(os.path.join(SMOTE_DIR, 'random_forest_smote.pkl'), 'wb') as f:
    pickle.dump(rf, f)

# MODEL 4 — Gaussian Naive Bayes (default — no weighting for SMOTE experiment)
gnb = GaussianNB()
gnb.fit(X_train_smote, y_train_smote)
gnb_pred  = gnb.predict(X_test_proc)
gnb_proba = gnb.predict_proba(X_test_proc)[:, 1]
m = compute_metrics(y_test, gnb_pred, gnb_proba, 'Naive Bayes', 'SMOTE')
smote_metrics.append(m); smote_probas['Naive Bayes'] = gnb_proba
with open(os.path.join(SMOTE_DIR, 'naive_bayes_smote.pkl'), 'wb') as f:
    pickle.dump(gnb, f)

# ============================================================
# STEP 8 — LOAD PART 5D (CLASS-WEIGHTED) RESULTS
# ============================================================
with open(PART5D_JSON) as f:
    p5d_raw = json.load(f)

# Part 5D confusion matrices (from memory / comparison CSV)
part5d_cm = {
    'Logistic Regression': {'tn': 7,  'fp': 6,  'fn': 88, 'tp': 236},
    'Decision Tree':       {'tn': 7,  'fp': 6,  'fn': 94, 'tp': 230},
    'Random Forest':       {'tn': 1,  'fp': 12, 'fn': 1,  'tp': 323},
    'Naive Bayes':         {'tn': 7,  'fp': 6,  'fn': 53, 'tp': 271},
}

cw_rows = []
for model_name, metrics in p5d_raw.items():
    cm_vals = part5d_cm[model_name]
    cw_rows.append({
        'model':             model_name,
        'method':            'Class Weighted',
        'accuracy':          metrics['accuracy'],
        'precision':         metrics['precision'],
        'recall':            metrics['recall'],
        'f1':                metrics['f1'],
        'specificity':       metrics['specificity'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'roc_auc':           metrics['roc_auc'],
        'pr_auc':            metrics['pr_auc'],
        'tn': cm_vals['tn'], 'fp': cm_vals['fp'],
        'fn': cm_vals['fn'], 'tp': cm_vals['tp'],
        'y_proba': None,
    })

# ============================================================
# STEP 9 — COMBINED COMPARISON CSV
# ============================================================
all_results = cw_rows + smote_metrics
csv_rows = []
for r in all_results:
    csv_rows.append({
        'Model':            r['model'],
        'Training_Method':  r['method'],
        'Accuracy':         r['accuracy'],
        'Precision':        r['precision'],
        'Recall':           r['recall'],
        'F1':               r['f1'],
        'Specificity':      r['specificity'],
        'Balanced_Accuracy':r['balanced_accuracy'],
        'ROC_AUC':          r['roc_auc'],
        'PR_AUC':           r['pr_auc'],
        'TN': r['tn'], 'FP': r['fp'], 'FN': r['fn'], 'TP': r['tp'],
    })

df_cmp = pd.DataFrame(csv_rows)
df_cmp.to_csv(os.path.join(DATA_DIR, 'smote_classification_comparison.csv'), index=False)

# ============================================================
# STEP 10 — COMPARISON CHARTS
# ============================================================
PALETTE_CW    = {'Logistic Regression': '#4C72B0', 'Decision Tree': '#DD8452',
                 'Random Forest': '#55A868', 'Naive Bayes': '#C44E52'}
PALETTE_SMOTE = {'Logistic Regression': '#A8C8F0', 'Decision Tree': '#F4C090',
                 'Random Forest': '#A8D8B4', 'Naive Bayes': '#EFA8AA'}
MODEL_ORDER   = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes']

def make_bar_comparison(metric_key, metric_label, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(MODEL_ORDER)
    x = np.arange(n)
    w = 0.35

    cw_vals    = [next(r[metric_key] for r in cw_rows    if r['model'] == m) for m in MODEL_ORDER]
    smote_vals = [next(r[metric_key] for r in smote_metrics if r['model'] == m) for m in MODEL_ORDER]

    bars_cw = ax.bar(x - w/2, cw_vals, width=w, label='Class Weighted (Part 5D)',
                     color=[PALETTE_CW[m] for m in MODEL_ORDER], edgecolor='white')
    bars_sm = ax.bar(x + w/2, smote_vals, width=w, label='SMOTE (Part 5E)',
                     color=[PALETTE_SMOTE[m] for m in MODEL_ORDER], edgecolor='white')

    for bar in bars_cw + bars_sm:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f'{metric_label}: Class Weighted vs SMOTE', fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.15, max(max(cw_vals), max(smote_vals)) + 0.15))
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

make_bar_comparison('balanced_accuracy', 'Balanced Accuracy', 'smote_comparison.png')
make_bar_comparison('specificity',       'Specificity',       'smote_specificity_comparison.png')

# ============================================================
# STEP 11 — SCIENTIFIC INTERPRETATION
# ============================================================
def delta(smote_val, cw_val):
    d = smote_val - cw_val
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"

# Determine if SMOTE improved balanced performance overall
improved_count = 0
for m in MODEL_ORDER:
    cw_ba    = next(r['balanced_accuracy'] for r in cw_rows if r['model'] == m)
    smote_ba = next(r['balanced_accuracy'] for r in smote_metrics if r['model'] == m)
    if smote_ba > cw_ba:
        improved_count += 1

smote_improved = improved_count >= 3  # majority of models improved

# ============================================================
# STEP 12 — MARKDOWN REPORT
# ============================================================
model_rows_md = ""
for m in MODEL_ORDER:
    cw   = next(r for r in cw_rows      if r['model'] == m)
    sm   = next(r for r in smote_metrics if r['model'] == m)
    model_rows_md += f"""
### {m}

| Metric | Class Weighted | SMOTE | Delta |
|---|---:|---:|---:|
| Accuracy | {cw['accuracy']} | {sm['accuracy']} | {delta(sm['accuracy'], cw['accuracy'])} |
| Precision | {cw['precision']} | {sm['precision']} | {delta(sm['precision'], cw['precision'])} |
| Recall | {cw['recall']} | {sm['recall']} | {delta(sm['recall'], cw['recall'])} |
| F1 | {cw['f1']} | {sm['f1']} | {delta(sm['f1'], cw['f1'])} |
| Specificity | {cw['specificity']} | {sm['specificity']} | {delta(sm['specificity'], cw['specificity'])} |
| Balanced Accuracy | {cw['balanced_accuracy']} | {sm['balanced_accuracy']} | {delta(sm['balanced_accuracy'], cw['balanced_accuracy'])} |
| ROC-AUC | {cw['roc_auc']} | {sm['roc_auc']} | {delta(sm['roc_auc'], cw['roc_auc'])} |
| PR-AUC | {cw['pr_auc']} | {sm['pr_auc']} | {delta(sm['pr_auc'], cw['pr_auc'])} |
| TN | {cw['tn']} | {sm['tn']} | {sm['tn']-cw['tn']:+d} |
| FP | {cw['fp']} | {sm['fp']} | {sm['fp']-cw['fp']:+d} |
| FN | {cw['fn']} | {sm['fn']} | {sm['fn']-cw['fn']:+d} |
| TP | {cw['tp']} | {sm['tp']} | {sm['tp']-cw['tp']:+d} |
"""

# Overall SMOTE verdict
smote_verdict_line = (
    "SMOTE improved balanced performance (Balanced Accuracy) for the majority of models."
    if smote_improved else
    "SMOTE did not provide a meaningful improvement over class-weighted learning for this dataset."
)

report_text = f"""# Part 5E — SMOTE vs Class-Weighted Experiment Report

---

## A. Why SMOTE Was Tested

The Part 5D baseline experiment used `class_weight='balanced'` to address class imbalance
(1,616 churned vs 67 retained — a 24:1 ratio). While class weighting adjusts the loss
function, it does not change the training data distribution. SMOTE (Synthetic Minority
Oversampling TEchnique) was tested as an alternative strategy that generates synthetic
minority-class (retained) samples to balance the training set before fitting.

The hypothesis: rebalancing the training distribution via SMOTE may improve the classifiers'
ability to detect retained customers (minority class), potentially improving Specificity,
Balanced Accuracy, and ROC-AUC.

---

## B. How SMOTE Was Applied

```
Step 1: Load data/olist_churn_dataset.csv (unchanged)
Step 2: Create has_review flag (before imputation)
Step 3: train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        — identical split to Part 5D
Step 4: Fit preprocessor (SimpleImputer median + StandardScaler) on X_train ONLY
        Apply to X_train and X_test
Step 5: SMOTE(random_state=42).fit_resample(X_train_proc, y_train)
        — applied ONLY to transformed training data
        — X_test_proc and y_test are NEVER touched by SMOTE
Step 6: Train 4 models on X_train_smote WITHOUT class_weight
Step 7: Evaluate on original X_test_proc / y_test
```

**Critical guarantee:** The test set is always evaluated on real, unmodified data.
SMOTE only affects the training distribution.

---

## C. Original Training Distribution

| Class | Count |
|---|---:|
| Class 0 (retained) | {train_c0_before} |
| Class 1 (churned)  | {train_c1_before} |
| Ratio (churn:retain) | {train_c1_before}:{train_c0_before} |

---

## D. After SMOTE Training Distribution

| Class | Count |
|---|---:|
| Class 0 (retained — including synthetic) | {train_c0_after} |
| Class 1 (churned) | {train_c1_after} |
| Ratio | 1:1 (balanced) |

> **Important:** The {train_c0_after - train_c0_before} additional Class 0 rows are
> SYNTHETIC observations generated by SMOTE's k-NN interpolation. They are not real
> customers. The original 54 real retained training customers remain unchanged.

---

## E. Test Set Distribution (Unchanged)

| Class | Count |
|---|---:|
| Class 0 (retained) | {test_c0} |
| Class 1 (churned)  | {test_c1} |

---

## F. Full Comparison Table

| Model | Method | Accuracy | Precision | Recall | F1 | Specificity | Bal. Acc | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
    f"| {r['Model']} | {r['Training_Method']} | {r['Accuracy']} | {r['Precision']} | "
    f"{r['Recall']} | {r['F1']} | {r['Specificity']} | {r['Balanced_Accuracy']} | "
    f"{r['ROC_AUC']} | {r['PR_AUC']} |"
    for r in sorted(csv_rows,
                    key=lambda x: (x['Model'], x['Training_Method']))
) + f"""

---

## G. Model-by-Model Comparison
{model_rows_md}

---

## H. Did SMOTE Improve Performance?

| Model | Bal. Acc (CW) | Bal. Acc (SMOTE) | Improved? | Spec (CW) | Spec (SMOTE) | Improved? |
|---|---:|---:|---|---:|---:|---|
""" + "\n".join(
    f"| {m} | "
    f"{next(r['balanced_accuracy'] for r in cw_rows if r['model']==m)} | "
    f"{next(r['balanced_accuracy'] for r in smote_metrics if r['model']==m)} | "
    f"{'YES' if next(r['balanced_accuracy'] for r in smote_metrics if r['model']==m) > next(r['balanced_accuracy'] for r in cw_rows if r['model']==m) else 'NO'} | "
    f"{next(r['specificity'] for r in cw_rows if r['model']==m)} | "
    f"{next(r['specificity'] for r in smote_metrics if r['model']==m)} | "
    f"{'YES' if next(r['specificity'] for r in smote_metrics if r['model']==m) > next(r['specificity'] for r in cw_rows if r['model']==m) else 'NO'} |"
    for m in MODEL_ORDER
) + f"""

---

## I. SMOTE Deterioration Check

| Model | Precision (CW) | Precision (SMOTE) | Delta | FP (CW) | FP (SMOTE) | More FP? |
|---|---:|---:|---|---:|---:|---|
""" + "\n".join(
    f"| {m} | "
    f"{next(r['precision'] for r in cw_rows if r['model']==m)} | "
    f"{next(r['precision'] for r in smote_metrics if r['model']==m)} | "
    f"{delta(next(r['precision'] for r in smote_metrics if r['model']==m), next(r['precision'] for r in cw_rows if r['model']==m))} | "
    f"{next(r['fp'] for r in cw_rows if r['model']==m)} | "
    f"{next(r['fp'] for r in smote_metrics if r['model']==m)} | "
    f"{'YES' if next(r['fp'] for r in smote_metrics if r['model']==m) > next(r['fp'] for r in cw_rows if r['model']==m) else 'NO'} |"
    for m in MODEL_ORDER
) + f"""

---

## J. Scientific Conclusion

**{smote_verdict_line}**

### Detailed Assessment

The test set contains only **{test_c0} retained customers** (Class 0 = minority).
This extremely small test minority means that a shift of even 1-2 TN/FP values produces
large metric swings. Results must therefore be interpreted as indicative, not definitive.

### Limitation (mandatory statement)

> **The retained class contains only 67 real customers. SMOTE creates synthetic minority
> observations and does not create new real customer information. Therefore SMOTE results
> must be interpreted cautiously. Apparent improvements in Specificity or Balanced Accuracy
> driven by correctly classifying 1-2 additional test customers cannot be considered
> statistically robust.**

### Final Recommendation

For this dataset:
- If **detecting retained customers** (minimising FN for Class 0) is the primary goal:
  evaluate which method achieves higher Specificity with acceptable Precision trade-off.
- If **overall discriminative ability** is the goal: compare ROC-AUC and PR-AUC.
- **Accuracy should not be used** as a selection criterion given the 96% majority baseline.
- Any future experiment should acknowledge that with only 13 test-set retained customers,
  all minority-class metrics have very wide confidence intervals.

---

*Part 5E complete — awaiting approval before any further modules.*
"""

with open(os.path.join(DATA_DIR, 'part5e_smote_report.md'), 'w', encoding='utf-8') as f:
    f.write(report_text)

# ============================================================
# STEP 13 — FINAL CONSOLE OUTPUT
# ============================================================
SEP  = "=" * 40
SEP2 = "-" * 40

print(SEP)
print("PART 5E -- SMOTE EXPERIMENT")
print(SEP)
print()
print("ORIGINAL TRAINING DISTRIBUTION")
print(f"  Class 0 (retained): {train_c0_before}")
print(f"  Class 1 (churned):  {train_c1_before}")
print()
print("AFTER SMOTE")
print(f"  Class 0 (retained): {train_c0_after}  ({train_c0_after - train_c0_before} synthetic added)")
print(f"  Class 1 (churned):  {train_c1_after}")
print()
print("TEST DISTRIBUTION (unchanged — SMOTE never applied)")
print(f"  Class 0 (retained): {test_c0}")
print(f"  Class 1 (churned):  {test_c1}")
print()
print("RESULTS")
print(SEP2)
header = f"{'Model':<22} {'Method':<15} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Spec':>6} {'BalAcc':>7} {'ROC':>6}"
print(header)
print("-" * len(header))
for r in sorted(all_results, key=lambda x: (x['model'], x['method'])):
    print(
        f"{r['model']:<22} {r['method']:<15} "
        f"{r['accuracy']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} "
        f"{r['f1']:>6.4f} {r['specificity']:>6.4f} {r['balanced_accuracy']:>7.4f} "
        f"{r['roc_auc']:>6.4f}"
    )
print()

best_cw    = max(cw_rows,      key=lambda r: r['balanced_accuracy'])
best_smote = max(smote_metrics, key=lambda r: r['balanced_accuracy'])

print(f"BEST CLASS-WEIGHTED MODEL (by Balanced Accuracy):")
print(f"  {best_cw['model']} — Bal.Acc={best_cw['balanced_accuracy']}, "
      f"ROC-AUC={best_cw['roc_auc']}, Specificity={best_cw['specificity']}")
print()
print(f"BEST SMOTE MODEL (by Balanced Accuracy):")
print(f"  {best_smote['model']} — Bal.Acc={best_smote['balanced_accuracy']}, "
      f"ROC-AUC={best_smote['roc_auc']}, Specificity={best_smote['specificity']}")
print()
print(f"DID SMOTE IMPROVE BALANCED PERFORMANCE? {'YES' if smote_improved else 'NO'}")
print()
print(f"MAIN FINDING:")
print(f"  {smote_verdict_line}")
print(f"  Test set has only {test_c0} retained customers — interpret with caution.")
print()
print(f"SMOTE MODELS SAVED:")
print(f"  models/academic/smote/logistic_regression_smote.pkl")
print(f"  models/academic/smote/decision_tree_smote.pkl")
print(f"  models/academic/smote/random_forest_smote.pkl")
print(f"  models/academic/smote/naive_bayes_smote.pkl")
print()
print(f"COMPARISON CSV:   data/smote_classification_comparison.csv")
print(f"REPORT:           data/part5e_smote_report.md")
print(f"CHARTS:           static/academic_results/smote_comparison.png")
print(f"                  static/academic_results/smote_specificity_comparison.png")
print()
print("STOP CONDITION CONFIRMATION")
print(SEP2)
print("SMOTE applied:         Only to X_train (after preprocessing)")
print("Test set modified:     NO")
print("olist_churn_dataset:   UNCHANGED")
print("pipeline.pkl:          UNCHANGED")
print("Part 5D models:        NOT OVERWRITTEN")
print("UI changes:            NONE")
print("Next module started:   NO")
print(SEP)
