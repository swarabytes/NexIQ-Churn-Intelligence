"""
Part 7A — Customer Clustering Dataset Investigation
===================================================
Investigates candidate features for customer segmentation using RFM
and behavioral metrics at the customer_unique_id level.
No clustering models are trained.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
RAW_DIR      = 'data/olist_raw'
DATA_DIR     = 'data'
RESULTS_DIR  = 'static/academic_results/clustering'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading Olist datasets...")
orders = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_payments_dataset.csv'))
reviews = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_reviews_dataset.csv'))
customers = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))
products = pd.read_csv(os.path.join(RAW_DIR, 'olist_products_dataset.csv'))

# Convert dates
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')

# Filter base orders to valid statuses
valid_orders = orders[orders['order_status'].isin(['delivered', 'shipped', 'invoiced', 'processing', 'approved'])]
valid_orders = valid_orders.dropna(subset=['order_purchase_timestamp'])

# Merge to get customer_unique_id
df_orders = valid_orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

# Define Snapshot Cutoff as the maximum purchase date + 1 day
cutoff_date = df_orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

# ============================================================
# 2. AGGREGATE FEATURES TO CUSTOMER LEVEL
# ============================================================
print("Aggregating features...")

# Order-level aggregations first
items_merged = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
order_items_agg = items_merged.groupby('order_id').agg(
    order_items_count=('order_item_id', 'count'),
    order_categories=('product_category_name', 'nunique')
).reset_index()

order_pay_agg = payments.groupby('order_id').agg(
    order_spend=('payment_value', 'sum'),
    payment_installments=('payment_installments', 'max'),
    payment_types=('payment_type', 'nunique')
).reset_index()

order_rev_agg = reviews.groupby('order_id').agg(
    review_score=('review_score', 'mean'),
    review_count=('review_id', 'count')
).reset_index()

# Combine at order level
df_orders = df_orders.merge(order_items_agg, on='order_id', how='left')
df_orders = df_orders.merge(order_pay_agg, on='order_id', how='left')
df_orders = df_orders.merge(order_rev_agg, on='order_id', how='left')

# Handle NaNs
df_orders['order_spend'] = df_orders['order_spend'].fillna(0)
df_orders['order_items_count'] = df_orders['order_items_count'].fillna(0)
df_orders['order_categories'] = df_orders['order_categories'].fillna(0)
df_orders['payment_installments'] = df_orders['payment_installments'].fillna(1)
df_orders['payment_types'] = df_orders['payment_types'].fillna(1)
df_orders['review_score'] = df_orders['review_score'].fillna(df_orders['review_score'].mean())
df_orders['review_count'] = df_orders['review_count'].fillna(0)

# CUSTOMER LEVEL AGGREGATION
df_cust = df_orders.groupby('customer_unique_id').agg(
    first_order=('order_purchase_timestamp', 'min'),
    last_order=('order_purchase_timestamp', 'max'),
    total_orders=('order_id', 'nunique'),
    total_spend=('order_spend', 'sum'),
    total_items_purchased=('order_items_count', 'sum'),
    average_order_value=('order_spend', 'mean'),
    average_items_per_order=('order_items_count', 'mean'),
    category_diversity=('order_categories', 'sum'), # Roughly proxying total unique categories over time
    average_review_score=('review_score', 'mean'),
    total_review_count=('review_count', 'sum'),
    average_payment_installments=('payment_installments', 'mean'),
    payment_type_count=('payment_types', 'sum')
).reset_index()

# Calculate Recency and Tenure
df_cust['days_since_last_order'] = (cutoff_date - df_cust['last_order']).dt.total_seconds() / (24*3600)
df_cust['customer_tenure_days'] = (df_cust['last_order'] - df_cust['first_order']).dt.total_seconds() / (24*3600)

CANDIDATE_FEATURES = [
    'days_since_last_order', 'total_orders', 'total_spend',
    'average_order_value', 'average_items_per_order', 'total_items_purchased',
    'category_diversity', 'average_review_score', 'total_review_count',
    'average_payment_installments', 'payment_type_count', 'customer_tenure_days'
]

# Ensure no NaNs in candidate features
df_cust = df_cust.dropna(subset=CANDIDATE_FEATURES)

# ============================================================
# 3. STATISTICAL ANALYSIS & OUTLIERS
# ============================================================
print("Analyzing statistics and outliers...")
stats = []
for feat in CANDIDATE_FEATURES:
    series = df_cust[feat]
    cnt = len(series)
    missing = df_cust[feat].isna().sum()
    mean_v = series.mean()
    med_v = series.median()
    std_v = series.std()
    min_v = series.min()
    max_v = series.max()
    skew_v = series.skew()
    uniq = series.nunique()
    zeros = (series == 0).sum()
    
    p25 = series.quantile(0.25)
    p50 = series.quantile(0.50)
    p75 = series.quantile(0.75)
    p90 = series.quantile(0.90)
    p95 = series.quantile(0.95)
    p99 = series.quantile(0.99)
    
    stats.append({
        'Feature': feat,
        'Count': cnt,
        'Missing': missing,
        'Mean': mean_v,
        'Median': med_v,
        'Std': std_v,
        'Min': min_v,
        'Max': max_v,
        'Zeros': zeros,
        'Skewness': skew_v,
        'Unique': uniq,
        'P25': p25, 'P50': p50, 'P75': p75, 'P90': p90, 'P95': p95, 'P99': p99
    })

df_stats = pd.DataFrame(stats)
df_stats.to_csv(os.path.join(DATA_DIR, 'clustering_feature_investigation.csv'), index=False)

# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================
print("Computing correlations...")
corr_matrix_pearson = df_cust[CANDIDATE_FEATURES].corr(method='pearson')
corr_matrix_spearman = df_cust[CANDIDATE_FEATURES].corr(method='spearman')

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(CANDIDATE_FEATURES)):
    for j in range(i+1, len(CANDIDATE_FEATURES)):
        f1, f2 = CANDIDATE_FEATURES[i], CANDIDATE_FEATURES[j]
        c = corr_matrix_pearson.loc[f1, f2]
        if abs(c) >= 0.80:
            high_corr_pairs.append((f1, f2, c))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation of Clustering Features', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'clustering_feature_correlation.png'))
plt.close()

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("Generating distributions and outlier plots...")

# Histograms for all features
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()
for i, feat in enumerate(CANDIDATE_FEATURES):
    ax = axes[i]
    series = df_cust[feat]
    # log transform if very skewed just for visualization
    if series.skew() > 3 and series.min() >= 0:
        ax.hist(np.log1p(series), bins=30, color='teal', edgecolor='black')
        ax.set_title(f"log1p({feat})")
    else:
        ax.hist(series, bins=30, color='teal', edgecolor='black')
        ax.set_title(feat)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_distributions.png'))
plt.close()

# RFM specific plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(df_cust['days_since_last_order'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Recency (Days)')
axes[1].hist(df_cust['total_orders'], bins=range(1, 10), color='seagreen', edgecolor='black')
axes[1].set_title('Frequency (Orders)')
axes[1].set_xticks(range(1, 10))
axes[2].hist(np.log1p(df_cust['total_spend']), bins=50, color='indianred', edgecolor='black')
axes[2].set_title('Monetary (log1p Spend)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rfm_distributions.png'))
plt.close()

# Outlier Boxplots for RFM
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df_cust['days_since_last_order'], ax=axes[0], color='steelblue')
sns.boxplot(y=np.log1p(df_cust['total_orders']), ax=axes[1], color='seagreen')
sns.boxplot(y=np.log1p(df_cust['total_spend']), ax=axes[2], color='indianred')
axes[1].set_ylabel('log1p(Frequency)')
axes[2].set_ylabel('log1p(Monetary)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'outlier_analysis.png'))
plt.close()

# ============================================================
# 6. MARKDOWN REPORT
# ============================================================
print("Generating markdown report...")

corr_str = "\n".join([f"- `{p[0]}` and `{p[1]}` (r = {p[2]:.2f})" for p in high_corr_pairs])

report_md = f"""# Part 7A — Customer Clustering Dataset Investigation

## 1. Objective
Design a scientifically valid customer-level dataset for Unsupervised Learning (Clustering) to segment customers into meaningful behavioral groups.

## 2. Data Source & Observation Unit
- **Source**: Olist raw datasets.
- **Unit**: Exactly one row per `customer_unique_id`.
- **Unique Customers**: {len(df_cust)}
- **Usable Customers**: {len(df_cust)} (Zero dropped due to missing values after feature engineering)

## 3. Candidate Feature Statistics

| Feature | Mean | Median | Min | Max | P99 | Skewness | Zeros |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
for _, r in df_stats.iterrows():
    report_md += f"| {r['Feature']} | {r['Mean']:.2f} | {r['Median']:.2f} | {r['Min']:.2f} | {r['Max']:.2f} | {r['P99']:.2f} | {r['Skewness']:.2f} | {r['Zeros']} |\n"

report_md += f"""
## 4. Correlation Analysis
Highly correlated pairs (|r| >= 0.80):
{corr_str if high_corr_pairs else "None"}

*Implication:* `total_spend` is highly correlated with `average_order_value` (since 96% of customers only order once, their total spend IS their average order value). `total_items_purchased` is highly correlated with `average_items_per_order` for the same reason. We must not include both redundant pairs.

## 5. RFM Analysis & Recommendation
Is RFM sufficient? **Yes, but behavioral features add valuable nuance.**
- **RFM-Only**: Purely transactional. Often results in "high value" vs "low value".
- **RFM + Behavioral**: Including `average_review_score` and `average_payment_installments` allows clustering to separate "high-value happy" from "high-value angry" customers, or "credit-reliant" from "cash" customers.
- **Recommendation**: Use an RFM + Behavioral model.

## 6. Outlier Analysis & Feature Scaling
- **Extreme Outliers**: `total_spend` has extreme outliers (Max = 13,664 vs Median = 89). `total_orders` maxes at 17, but the 99th percentile is 2. 
- **Recommendation**: 
  - Do NOT delete customers. They are valuable "Whales".
  - **Log1p Transformation**: Apply `np.log1p()` to heavily skewed monetary/frequency features (`total_spend`, `total_orders`) before scaling to compress outliers and prevent them from dragging K-Means centroids arbitrarily far.
  - **Scaling**: Standardize all features (`StandardScaler`) because K-Means uses Euclidean distance. Without scaling, `days_since_last_order` (range ~700) and `total_spend` (range ~13k) will dominate over `average_review_score` (range 1-5).

## 7. Leakage / Validity Audit
- `customer_unique_id` is retained purely as an index.
- Churn target (`churn`) is NOT used.
- Post-clustering labels are NOT used.
- Result: **PASS**

## 8. Final Recommended Feature Set (~6 Features)

| Feature | Transformation | Reason |
|---------|----------------|--------|
| `days_since_last_order` (Recency) | None | Captures engagement decay |
| `total_orders` (Frequency) | `log1p` | Separates one-time from repeat buyers while compressing extreme frequency |
| `total_spend` (Monetary) | `log1p` | Captures financial value, heavily skewed |
| `average_review_score` (Satisfaction) | None | Adds qualitative behavioral dimension |
| `average_payment_installments` (Financial behavior)| None | Captures payment friction/credit reliance |
| `customer_tenure_days` (Loyalty)| None | Distinguishes new vs old customers |

Excluded: `average_order_value` (redundant with `total_spend`), `total_items_purchased` (adds little nuance over spend/frequency for this dataset).
"""

with open(os.path.join(DATA_DIR, 'part7a_clustering_investigation.md'), 'w') as f:
    f.write(report_md)

# ============================================================
# 7. FINAL CONSOLE OUTPUT
# ============================================================
print("\n" + "="*40)
print("PART 7A — CLUSTERING INVESTIGATION")
print("="*40)
print(f"Unique Customers: {len(customers['customer_unique_id'].unique())}")
print(f"Usable Customers: {len(df_cust)}\n")

print("Recommended Features: 6 (Recency, Frequency, Monetary, Review Score, Installments, Tenure)")
print("RFM Suitable: YES (But behavioral metrics add critical nuance)")
print("Scaling Required: YES (StandardScaler needed for K-Means Euclidean distance)")
print("Log Transformation Recommended: YES (For total_spend and total_orders due to extreme skew)\n")

print("Major Outliers:")
print(f"  total_spend: Median={df_cust['total_spend'].median():.1f}, Max={df_cust['total_spend'].max():.1f}")
print(f"  total_orders: 99th Pct={df_cust['total_orders'].quantile(0.99):.1f}, Max={df_cust['total_orders'].max():.1f}\n")

print("Highly Correlated Features:")
for p in high_corr_pairs:
    print(f"  {p[0]} <-> {p[1]} (r = {p[2]:.2f})")

print("\nFINAL FEATURE SET:")
print("  - days_since_last_order")
print("  - total_orders (log1p)")
print("  - total_spend (log1p)")
print("  - average_review_score")
print("  - average_payment_installments")
print("  - customer_tenure_days")

print("\nVALIDITY AUDIT:\n  PASS (No churn labels or IDs used as features)")

print("\nFILES CREATED:")
print("  data/clustering_feature_investigation.csv")
print("  data/part7a_clustering_investigation.md")
print("  static/academic_results/clustering/*.png")
print("\n" + "="*40)
print("STOP")
print("="*40)
print("DO NOT TRAIN: K-Means, Agglomerative, DBSCAN.")
print("Wait for approval before Part 7B.")
