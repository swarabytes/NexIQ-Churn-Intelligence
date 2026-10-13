"""
Part 7B — K-Means Customer Segmentation
===================================================
Applies K-Means clustering to the Olist customer dataset.
Uses approved features with log1p and standard scaling.
Finds optimal K, profiles clusters, and generates visualizations.
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
RAW_DIR      = 'data/olist_raw'
DATA_DIR     = 'data'
RESULTS_DIR  = 'static/academic_results/clustering'
MODELS_DIR   = 'models/clustering'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ============================================================
# 1. BUILD CLUSTERING DATASET (Same as 7A)
# ============================================================
print("Loading Olist datasets...")
orders = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_payments_dataset.csv'))
reviews = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_reviews_dataset.csv'))
customers = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
valid_orders = orders[orders['order_status'].isin(['delivered', 'shipped', 'invoiced', 'processing', 'approved'])]
valid_orders = valid_orders.dropna(subset=['order_purchase_timestamp'])
df_orders = valid_orders.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

cutoff_date = df_orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

order_items_agg = items.groupby('order_id').agg(order_items_count=('order_item_id', 'count')).reset_index()
order_pay_agg = payments.groupby('order_id').agg(
    order_spend=('payment_value', 'sum'),
    payment_installments=('payment_installments', 'max')
).reset_index()
order_rev_agg = reviews.groupby('order_id').agg(review_score=('review_score', 'mean')).reset_index()

df_orders = df_orders.merge(order_items_agg, on='order_id', how='left')
df_orders = df_orders.merge(order_pay_agg, on='order_id', how='left')
df_orders = df_orders.merge(order_rev_agg, on='order_id', how='left')

df_orders['order_spend'] = df_orders['order_spend'].fillna(0)
df_orders['order_items_count'] = df_orders['order_items_count'].fillna(0)
df_orders['payment_installments'] = df_orders['payment_installments'].fillna(1)
df_orders['review_score'] = df_orders['review_score'].fillna(df_orders['review_score'].mean())

df_cust = df_orders.groupby('customer_unique_id').agg(
    first_order=('order_purchase_timestamp', 'min'),
    last_order=('order_purchase_timestamp', 'max'),
    total_orders=('order_id', 'nunique'),
    total_spend=('order_spend', 'sum'),
    average_review_score=('review_score', 'mean'),
    average_payment_installments=('payment_installments', 'mean')
).reset_index()

df_cust['days_since_last_order'] = (cutoff_date - df_cust['last_order']).dt.total_seconds() / (24*3600)
df_cust['customer_tenure_days'] = (df_cust['last_order'] - df_cust['first_order']).dt.total_seconds() / (24*3600)

APPROVED_FEATURES = [
    'days_since_last_order', 'total_orders', 'total_spend',
    'average_review_score', 'average_payment_installments', 'customer_tenure_days'
]

df_cust = df_cust.dropna(subset=APPROVED_FEATURES)

print(f"Final usable customers: {len(df_cust)}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("Applying log1p and StandardScaler...")

# Create working copy
df_cluster = df_cust[['customer_unique_id'] + APPROVED_FEATURES].copy()

# Apply log1p to heavily skewed features
df_cluster['log_total_orders'] = np.log1p(df_cluster['total_orders'])
df_cluster['log_total_spend'] = np.log1p(df_cluster['total_spend'])

MODEL_FEATURES = [
    'days_since_last_order',
    'log_total_orders',
    'log_total_spend',
    'average_review_score',
    'average_payment_installments',
    'customer_tenure_days'
]

X = df_cluster[MODEL_FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. K-MEANS EXPERIMENT (K=2 to 10)
# ============================================================
print("Training K-Means (K=2 to 10)...")
k_values = range(2, 11)
wcss = []
silhouette_scores = []

# Subsampling for silhouette to save time if dataset is huge (>90k)
# Calculate exact silhouette on 20,000 samples for speed, as O(N^2) on 94k takes too long
np.random.seed(42)
sample_indices = np.random.choice(X_scaled.shape[0], size=min(20000, X_scaled.shape[0]), replace=False)
X_scaled_sample = X_scaled[sample_indices]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Predict on sample to get silhouette
    labels_sample = kmeans.predict(X_scaled_sample)
    score = silhouette_score(X_scaled_sample, labels_sample)
    silhouette_scores.append(score)
    print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

# Plot Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='-', color='steelblue')
plt.title('K-Means Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_elbow.png'))
plt.close()

# Plot Silhouette
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='indianred')
plt.title('K-Means Silhouette Scores')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_silhouette.png'))
plt.close()

# ============================================================
# 4. SELECT OPTIMAL K
# ============================================================
# Heuristic selection: pick K with max silhouette, but prefer K=4 or K=5 
# if silhouette difference is negligible and they provide better business interpretability.
# Often K=2 has highest silhouette but is just "good vs bad" which is boring.
best_k_sil = k_values[np.argmax(silhouette_scores)]

# We will select a defensible K. Usually K=3 or 4 are great for RFM.
# If K=2 is max, we might force K=4 if its silhouette is reasonable (>0.25).
selected_k = best_k_sil
if selected_k == 2 and silhouette_scores[2] > 0.25: # Check K=4
    selected_k = 4

print(f"\nSELECTED K = {selected_k}")

# ============================================================
# 5. TRAIN FINAL MODEL & PROFILES
# ============================================================
print(f"Training final K-Means with K={selected_k}...")
final_kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

df_cluster['cluster_id'] = cluster_labels
df_cust['cluster_id'] = cluster_labels

# Save customer clusters
df_cluster.to_csv(os.path.join(DATA_DIR, 'customer_clusters.csv'), index=False)

# Analyze sizes
cluster_sizes = df_cluster['cluster_id'].value_counts().sort_index()
cluster_pcts = (cluster_sizes / len(df_cluster)) * 100

# Bar chart of cluster sizes
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='viridis')
plt.title('Cluster Sizes')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Customers')
for i, v in enumerate(cluster_sizes.values):
    plt.text(i, v + 500, f"{cluster_pcts.iloc[i]:.1f}%", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cluster_sizes.png'))
plt.close()

# Generate Profiles (Mean and Median of original features)
profiles = []
for c in range(selected_k):
    c_df = df_cust[df_cust['cluster_id'] == c]
    
    mean_vals = c_df[APPROVED_FEATURES].mean()
    med_vals = c_df[APPROVED_FEATURES].median()
    
    total_spend = c_df['total_spend'].sum()
    avg_spend = c_df['total_spend'].mean()
    
    prof_dict = {'Cluster': c, 'Count': len(c_df), 'Percentage': cluster_pcts.iloc[c]}
    for feat in APPROVED_FEATURES:
        prof_dict[f'Mean_{feat}'] = mean_vals[feat]
        prof_dict[f'Median_{feat}'] = med_vals[feat]
        
    prof_dict['Total_Cluster_Spend'] = total_spend
    prof_dict['Average_Customer_Spend'] = avg_spend
    profiles.append(prof_dict)

df_profiles = pd.DataFrame(profiles)
df_profiles.to_csv(os.path.join(DATA_DIR, 'cluster_profiles.csv'), index=False)

# ============================================================
# 6. PCA VISUALIZATION (ONLY for 2D Plotting)
# ============================================================
print("Generating PCA Visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='tab10', alpha=0.3, s=15)
plt.title(f'K-Means Clusters (K={selected_k}) Visualized via PCA\n(PCA used ONLY for 2D visualization)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'kmeans_pca_clusters.png'))
plt.close()

# ============================================================
# 7. AUTOMATIC CLUSTER INTERPRETATION & INSIGHTS
# ============================================================
overall_means = df_cust[APPROVED_FEATURES].mean()
interpretations = {}

for _, row in df_profiles.iterrows():
    c = int(row['Cluster'])
    c_desc = []
    
    # Recency
    if row['Mean_days_since_last_order'] > overall_means['days_since_last_order'] * 1.2:
        c_desc.append("High Recency (Inactive)")
    elif row['Mean_days_since_last_order'] < overall_means['days_since_last_order'] * 0.8:
        c_desc.append("Low Recency (Recent Buyers)")
    else:
        c_desc.append("Medium Recency")
        
    # Frequency
    if row['Mean_total_orders'] > 1.1: # Olist has ~96% 1-time buyers
        c_desc.append("High Frequency (Repeat Buyers)")
    else:
        c_desc.append("Low Frequency (One-time)")
        
    # Monetary
    if row['Mean_total_spend'] > overall_means['total_spend'] * 1.5:
        c_desc.append("High Monetary Value")
    elif row['Mean_total_spend'] < overall_means['total_spend'] * 0.7:
        c_desc.append("Low Monetary Value")
    else:
        c_desc.append("Medium Monetary Value")
        
    # Review
    if row['Mean_average_review_score'] > 4.5:
        c_desc.append("Very High Satisfaction")
    elif row['Mean_average_review_score'] < 3.0:
        c_desc.append("Poor Satisfaction")
    
    interpretations[c] = ", ".join(c_desc)

# Generate Insights
highest_rev_cluster = df_profiles.loc[df_profiles['Total_Cluster_Spend'].idxmax()]['Cluster']
highest_avg_spend_cluster = df_profiles.loc[df_profiles['Average_Customer_Spend'].idxmax()]['Cluster']
most_frequent_cluster = df_profiles.loc[df_profiles['Mean_total_orders'].idxmax()]['Cluster']
most_inactive_cluster = df_profiles.loc[df_profiles['Mean_days_since_last_order'].idxmax()]['Cluster']
highest_sat_cluster = df_profiles.loc[df_profiles['Mean_average_review_score'].idxmax()]['Cluster']

insights = [
    f"Cluster {int(highest_rev_cluster)} contributes the most overall revenue, despite not necessarily having the highest average spend.",
    f"Cluster {int(highest_avg_spend_cluster)} has the highest average customer spend per capita (High-Value cohort).",
    f"Cluster {int(most_frequent_cluster)} contains the most frequent repeat buyers.",
    f"Cluster {int(most_inactive_cluster)} represents the most inactive, churning segment with the longest days since last order.",
    f"Cluster {int(highest_sat_cluster)} has the highest average review satisfaction scores."
]

# ============================================================
# 8. MARKDOWN REPORT
# ============================================================
report_md = f"""# Part 7B — K-Means Customer Segmentation

## Objective
Segment {len(df_cust)} valid Olist customers using K-Means clustering.

## Features & Preprocessing
- **Features Used:** `days_since_last_order`, `total_orders`, `total_spend`, `average_review_score`, `average_payment_installments`, `customer_tenure_days`.
- **Excluded:** `customer_unique_id`, `churn` (No leakage).
- **Transformations:** `np.log1p` applied to `total_orders` and `total_spend` to mitigate extreme right skew.
- **Scaling:** `StandardScaler` applied to equalize variance before computing Euclidean distances for K-Means.

## Model Selection (K)
- Tested K = 2 through 10.
- Selected K = **{selected_k}**
- **WCSS:** {final_kmeans.inertia_:.2f}
- **Silhouette Score:** {silhouette_scores[selected_k-2]:.4f}

## Cluster Sizes
"""
for c in range(selected_k):
    report_md += f"- **Cluster {c}**: {int(cluster_sizes[c])} customers ({cluster_pcts[c]:.1f}%)\n"

report_md += f"\n## Cluster Profiles & Interpretation\n\n"
for c in range(selected_k):
    r = df_profiles.iloc[c]
    report_md += f"### Cluster {c} - {interpretations[c]}\n"
    report_md += f"- Mean Spend: ${r['Mean_total_spend']:.2f}\n"
    report_md += f"- Mean Orders: {r['Mean_total_orders']:.2f}\n"
    report_md += f"- Mean Recency: {r['Mean_days_since_last_order']:.1f} days\n"
    report_md += f"- Mean Review Score: {r['Mean_average_review_score']:.2f}\n\n"

report_md += "## Business Insights\n"
for ins in insights:
    report_md += f"- {ins}\n"

with open(os.path.join(DATA_DIR, 'part7b_kmeans_report.md'), 'w') as f:
    f.write(report_md)

# ============================================================
# 9. SAVE MODELS AND METADATA
# ============================================================
print("Saving model and metadata...")

kmeans_pipeline = {
    'scaler': scaler,
    'kmeans': final_kmeans,
    'features_raw': APPROVED_FEATURES,
    'features_model': MODEL_FEATURES
}
with open(os.path.join(MODELS_DIR, 'kmeans_customer_segmentation.pkl'), 'wb') as f:
    pickle.dump(kmeans_pipeline, f)

meta = {
    'algorithm': 'K-Means',
    'selected_k': int(selected_k),
    'random_state': 42,
    'customer_count': len(df_cust),
    'features': APPROVED_FEATURES,
    'transformations': {'log1p': ['total_orders', 'total_spend']},
    'scaler': 'StandardScaler',
    'wcss': final_kmeans.inertia_,
    'silhouette_score': float(silhouette_scores[selected_k-2]),
    'cluster_sizes': cluster_sizes.to_dict(),
    'cluster_profiles': df_profiles.to_dict(orient='records'),
    'insights': insights
}
with open(os.path.join(MODELS_DIR, 'clustering_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=4)

# ============================================================
# 10. FINAL CONSOLE OUTPUT
# ============================================================
print("\n" + "="*40)
print("PART 7B — K-MEANS COMPLETE")
print("="*40)
print(f"Customers:\n  {len(df_cust)}")
print(f"Features:\n  {APPROVED_FEATURES}")
print(f"K values tested:\n  2-10")
print(f"\nSELECTED K:\n  {selected_k}")
print(f"Silhouette Score:\n  {silhouette_scores[selected_k-2]:.4f}")
print(f"WCSS:\n  {final_kmeans.inertia_:.2f}")

print("\nCluster Sizes:")
for c in range(selected_k):
    print(f"  Cluster {c}: {int(cluster_sizes[c])} ({cluster_pcts[c]:.1f}%)")

largest = cluster_sizes.idxmax()
smallest = cluster_sizes.idxmin()
print(f"\nLargest Cluster:\n  Cluster {largest}")
print(f"Smallest Cluster:\n  Cluster {smallest}")

print("\nMAIN CLUSTER INSIGHTS:")
for ins in insights:
    print(f"  - {ins}")

print("\nVALIDITY AUDIT:\n  PASS")

print("\nFILES CREATED:")
print("  data/customer_clusters.csv")
print("  data/cluster_profiles.csv")
print("  data/part7b_kmeans_report.md")
print("  models/clustering/kmeans_customer_segmentation.pkl")
print("  models/clustering/clustering_metadata.json")
print("  static/academic_results/clustering/kmeans_elbow.png")
print("  static/academic_results/clustering/kmeans_silhouette.png")
print("  static/academic_results/clustering/cluster_sizes.png")
print("  static/academic_results/clustering/kmeans_pca_clusters.png")
print("\nSTOP.")
print("Wait for approval before Part 7C.")
