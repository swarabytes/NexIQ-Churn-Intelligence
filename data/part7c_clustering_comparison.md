# Part 7C — Agglomerative Clustering & Final Comparison

## 1. Agglomerative Methodology
- **Algorithm**: Agglomerative Clustering
- **Linkage**: Ward linkage (minimizes variance within clusters, ideal for standardized numerical features, highly compatible with Euclidean distance).
- **Subsampling**: Full agglomerative on 95k rows requires a huge distance matrix. Used a defensible sample of N=20,000 for clustering and N=2,000 for the Dendrogram.

## 2. Clustering Algorithm Comparison
| Algorithm | K | Silhouette | Largest % | Smallest % |
|---|---|---|---|---|
| K-Means | 4 | 0.3071 | 61.3% | 3.1% |
| Agglomerative | 4 | 0.2648 | 66.2% | 3.1% |

## 3. Final Algorithm Selection
**Selected Algorithm: K-Means**
- **Reasoning**: Both methods yielded highly comparable Silhouette scores. However, K-Means is vastly superior in scalability (O(N) vs O(N³)), making it the only viable algorithm for the full 95k production dataset. Agglomerative is retained as a validation check.

## 4. Final Cluster Names (K-Means)
- **Cluster 0: Mainstream Recent Buyers** - Recent, standard spend, high satisfaction.
- **Cluster 1: One-Time High-Spenders** - Low frequency but massive single-purchase monetary value.
- **Cluster 2: Frequent High-Value Buyers** - Highest repeat purchase rate and monetary value.
- **Cluster 3: Unsatisfied Buyers** - Significantly lower review scores indicating churn due to poor experience.

## 5. Main Business Insights
1. Retain the "Frequent High-Value Buyers" via loyalty programs. They are rare but generate massive recurring revenue.
2. The "Unsatisfied Buyers" cluster requires immediate customer-support outreach.
3. The "Mainstream Recent Buyers" should be targeted with follow-up campaigns to convert them into repeat buyers before they slip into the "Churned Occasional" segment.
