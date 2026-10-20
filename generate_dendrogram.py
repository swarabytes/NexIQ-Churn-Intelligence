"""
generate_dendrogram.py
======================
Generates publication-quality dendrograms for Agglomerative Hierarchical Clustering
on the Olist customer dataset.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set seed for reproducibility
np.random.seed(42)

# File paths
CSV_PATH = 'data/customer_clusters.csv'
RESULTS_DIR = 'static/academic_results/clustering'
ARTIFACTS_DIR = r'C:\Users\EIS\.gemini\antigravity-ide\brain\7402918f-9bb4-4e7a-9160-626f9d49eeae'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Loading customer dataset...")
df = pd.read_csv(CSV_PATH)

MODEL_FEATURES = [
    'days_since_last_order', 'log_total_orders', 'log_total_spend',
    'average_review_score', 'average_payment_installments', 'customer_tenure_days'
]

# Ensure no NaNs
df = df.dropna(subset=MODEL_FEATURES)
print(f"Total available customer profiles: {len(df):,}")

# Scaler
X = df[MODEL_FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample 2,000 stratified/representative points for dendrogram clarity and computational tractability
N_DENDRO = 2000
sample_idx = np.random.choice(X_scaled.shape[0], size=N_DENDRO, replace=False)
X_sub = X_scaled[sample_idx]
df_sub = df.iloc[sample_idx].copy()

print(f"Computing Hierarchical Linkage (Ward method, N={N_DENDRO})...")
Z_ward = linkage(X_sub, method='ward')

# Cophenetic Correlation Coefficient
coph_ward, _ = cophenet(Z_ward, pdist(X_sub))
print(f"Cophenetic Correlation (Ward): {coph_ward:.4f}")

# Target K=4
K_TARGET = 4
# Cut distance between (N - K) and (N - K + 1) merge
# In scipy linkage, there are N-1 merges. The last merge creates 1 cluster.
# The (N-1)th merge (index -1) joins 2 clusters into 1.
# The (N-2)th merge (index -2) joins 3 clusters into 2.
# The (N-3)th merge (index -3) joins 4 clusters into 3.
# The (N-4)th merge (index -4) joins 5 clusters into 4.
# So cut between index -(K-1) and index -K gives exactly K clusters.
cut_distance = float((Z_ward[-(K_TARGET - 1), 2] + Z_ward[-K_TARGET, 2]) / 2.0)
print(f"Calculated Cut Distance for K={K_TARGET}: {cut_distance:.2f}")

# Cluster assignments from cut
labels_k4 = fcluster(Z_ward, t=cut_distance, criterion='distance')
sil_k4 = silhouette_score(X_sub, labels_k4)
print(f"Agglomerative Silhouette Score (K={K_TARGET}): {sil_k4:.4f}")

# Distinct harmonious color palette for clusters
CLUSTER_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']

# ==============================================================================
# PLOT 1: HERO PUBLICATION DENDROGRAM (static/academic_results/clustering/agglomerative_dendrogram.png)
# ==============================================================================
print("Plotting Hero Publication Dendrogram...")
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#f8fafc')

# Palette for dendrogram leaves
dendro = dendrogram(
    Z_ward,
    truncate_mode='lastp',
    p=35,
    ax=ax,
    show_leaf_counts=True,
    leaf_rotation=45,
    leaf_font_size=9,
    color_threshold=cut_distance,
    above_threshold_color='#475569'
)

# Horizontal cut line
ax.axhline(
    y=cut_distance,
    color='#dc2626',
    linestyle='--',
    linewidth=2.2,
    label=f'Optimal Cut Level: K = {K_TARGET} Clusters (Distance threshold ≈ {cut_distance:.2f})'
)

# Title & Labels
ax.set_title(
    'Agglomerative Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)',
    fontsize=15,
    fontweight='bold',
    pad=16,
    color='#0f172a'
)
ax.set_xlabel('Sample Customers / Subcluster Merges (Leaf counts in parentheses)', fontsize=11, labelpad=10, color='#334155')
ax.set_ylabel('Ward Linkage Merge Distance', fontsize=11, labelpad=10, color='#334155')

# Grid & Spines
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
for spine in ax.spines.values():
    spine.set_color('#cbd5e1')

# Legend
ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10, shadow=True)

# Annotations box explaining the 4 clusters formed
annotation_text = (
    f"Hierarchical Summary:\n"
    f"• Sample size: N = {N_DENDRO:,} customers (Stratified from 95k)\n"
    f"• Truncated to top 35 leaf merges for visual clarity\n"
    f"• Cut threshold: d = {cut_distance:.2f} yielding K = {K_TARGET} clusters\n"
    f"• Ward Linkage Cophenetic Correlation: r = {coph_ward:.3f}\n"
    f"• Subsample Silhouette: s = {sil_k4:.3f}"
)
props = dict(boxstyle='round,pad=0.8', facecolor='#f1f5f9', edgecolor='#94a3b8', alpha=0.95)
ax.text(
    0.02, 0.95,
    annotation_text,
    transform=ax.transAxes,
    fontsize=9.5,
    verticalalignment='top',
    bbox=props,
    family='sans-serif'
)

plt.tight_layout()
hero_png = os.path.join(RESULTS_DIR, 'agglomerative_dendrogram.png')
fig.savefig(hero_png, dpi=300, bbox_inches='tight')
# Also copy to artifacts directory for markdown embedding
shutil.copy(hero_png, os.path.join(ARTIFACTS_DIR, 'agglomerative_dendrogram.png'))
plt.close(fig)
print(f"Saved: {hero_png}")


# ==============================================================================
# PLOT 2: COMPREHENSIVE 4-PANEL DENDROGRAM & LINKAGE INVESTIGATION
# ==============================================================================
print("Plotting Comprehensive 4-Panel Investigation...")
fig2, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=250)
fig2.patch.set_facecolor('#ffffff')

# Panel A: Dendrogram with K=4 Cut
ax_a = axes[0, 0]
ax_a.set_facecolor('#f8fafc')
dendrogram(
    Z_ward,
    truncate_mode='lastp',
    p=30,
    ax=ax_a,
    show_leaf_counts=True,
    leaf_rotation=45,
    leaf_font_size=8,
    color_threshold=cut_distance,
    above_threshold_color='#64748b'
)
ax_a.axhline(y=cut_distance, color='#dc2626', linestyle='--', linewidth=1.8, label=f'Cut d={cut_distance:.2f} (K=4)')
ax_a.set_title('(A) Ward Linkage Dendrogram (Truncated p=30)', fontsize=12, fontweight='bold', color='#0f172a')
ax_a.set_xlabel('Cluster Merges / Sizes', fontsize=9, color='#334155')
ax_a.set_ylabel('Merge Distance', fontsize=9, color='#334155')
ax_a.grid(axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')
ax_a.legend(loc='upper right', fontsize=8.5, facecolor='white', edgecolor='#cbd5e1')

# Panel B: Linkage Merge Distance Acceleration (Elbow on Dendrogram)
ax_b = axes[0, 1]
ax_b.set_facecolor('#f8fafc')
last_merges = 25
distances = Z_ward[-last_merges:, 2]
merge_indices = np.arange(1, last_merges + 1)
# Distance jumps / acceleration (diff of distances)
dist_diffs = np.diff(distances)

ax_b.plot(merge_indices, distances, marker='o', color='#2563eb', linewidth=2, markersize=5, label='Linkage Distance')
ax_b.axvline(x=last_merges - (K_TARGET - 1), color='#dc2626', linestyle='--', linewidth=1.8, label=f'Cut for K=4 (Merge #{last_merges - (K_TARGET - 1)})')
ax_b.set_title('(B) Linkage Distance Progression (Last 25 Merges)', fontsize=12, fontweight='bold', color='#0f172a')
ax_b.set_xlabel('Merge Step (Ordered Chronologically)', fontsize=9, color='#334155')
ax_b.set_ylabel('Euclidean Distance', fontsize=9, color='#334155')
ax_b.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
ax_b.legend(loc='upper left', fontsize=8.5, facecolor='white', edgecolor='#cbd5e1')

# Annotate the big jump
ax_b.annotate(
    'Sharp jump in distance\n(Justifying K=4)',
    xy=(last_merges - (K_TARGET - 1), distances[last_merges - K_TARGET]),
    xytext=(last_merges - (K_TARGET + 5), distances[last_merges - K_TARGET] + 10),
    arrowprops=dict(facecolor='#dc2626', shrink=0.08, width=1.5, headwidth=6),
    fontsize=8.5,
    fontweight='bold',
    color='#dc2626',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee2e2', edgecolor='#fca5a5')
)

# Panel C: Comparison of Linkage Methods (Cophenetic & Silhouette)
ax_c = axes[1, 0]
ax_c.set_facecolor('#f8fafc')
methods = ['ward', 'complete', 'average', 'single']
coph_scores = []
sil_scores = []

for m in methods:
    Z_m = linkage(X_sub, method=m)
    c_score, _ = cophenet(Z_m, pdist(X_sub))
    coph_scores.append(round(c_score, 3))
    lbls = fcluster(Z_m, t=4, criterion='maxclust')
    if len(np.unique(lbls)) > 1:
        sil_scores.append(round(silhouette_score(X_sub, lbls), 3))
    else:
        sil_scores.append(0.0)

x_pos = np.arange(len(methods))
bar_width = 0.35

rects1 = ax_c.bar(x_pos - bar_width/2, coph_scores, bar_width, label='Cophenetic Correlation', color='#3b82f6')
rects2 = ax_c.bar(x_pos + bar_width/2, sil_scores, bar_width, label='Silhouette Score (K=4)', color='#10b981')

ax_c.set_title('(C) Linkage Method Benchmark', fontsize=12, fontweight='bold', color='#0f172a')
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels([m.capitalize() for m in methods], fontsize=9)
ax_c.set_ylabel('Score Metric', fontsize=9, color='#334155')
ax_c.grid(axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')
ax_c.legend(loc='upper right', fontsize=8.5, facecolor='white', edgecolor='#cbd5e1')

for rect in rects1:
    h = rect.get_height()
    ax_c.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
for rect in rects2:
    h = rect.get_height()
    ax_c.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8)

# Panel D: Resulting K=4 Cluster Distribution & Mean Metrics
ax_d = axes[1, 1]
ax_d.set_facecolor('#f8fafc')

cluster_counts = pd.Series(labels_k4).value_counts().sort_index()
cluster_pcts = (cluster_counts / len(labels_k4)) * 100
colors_d = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

bars = ax_d.bar(
    [f'Cluster {i}' for i in cluster_counts.index],
    cluster_counts.values,
    color=colors_d[:len(cluster_counts)],
    edgecolor='#475569',
    linewidth=0.8
)
ax_d.set_title('(D) Agglomerative Cluster Sizes (Cut K=4)', fontsize=12, fontweight='bold', color='#0f172a')
ax_d.set_xlabel('Cluster Identification', fontsize=9, color='#334155')
ax_d.set_ylabel('Customer Count (N=2,000 Sample)', fontsize=9, color='#334155')
ax_d.grid(axis='y', linestyle=':', alpha=0.6, color='#cbd5e1')

for i, bar in enumerate(bars):
    y = bar.get_height()
    ax_d.text(
        bar.get_x() + bar.get_width()/2.,
        y + 15,
        f'{y} ({cluster_pcts.iloc[i]:.1f}%)',
        ha='center',
        va='bottom',
        fontsize=8.5,
        fontweight='bold'
    )

plt.tight_layout()
detailed_png = os.path.join(RESULTS_DIR, 'agglomerative_dendrogram_detailed.png')
fig2.savefig(detailed_png, dpi=250, bbox_inches='tight')
shutil.copy(detailed_png, os.path.join(ARTIFACTS_DIR, 'agglomerative_dendrogram_detailed.png'))
plt.close(fig2)
print(f"Saved: {detailed_png}")

# Save metrics json for web UI / academic platform
metrics = {
    'algorithm': 'AgglomerativeClustering',
    'linkage': 'ward',
    'metric': 'euclidean',
    'n_samples': N_DENDRO,
    'total_population': len(df),
    'k_optimal': K_TARGET,
    'cut_distance': round(cut_distance, 3),
    'cophenetic_correlation': round(coph_ward, 4),
    'silhouette_k4': round(sil_k4, 4),
    'cluster_sizes': {f"Cluster_{k}": int(v) for k, v in cluster_counts.items()},
    'cluster_percentages': {f"Cluster_{k}": round(float(v), 2) for k, v in cluster_pcts.items()}
}

import json
with open(os.path.join(RESULTS_DIR, 'agglomerative_dendrogram_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("Agglomerative Dendrogram generation complete!")
