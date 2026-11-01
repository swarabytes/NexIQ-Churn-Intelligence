# Part 7B — K-Means Customer Segmentation

## Objective
Segment 94986 valid Olist customers using K-Means clustering.

## Features & Preprocessing
- **Features Used:** `days_since_last_order`, `total_orders`, `total_spend`, `average_review_score`, `average_payment_installments`, `customer_tenure_days`.
- **Excluded:** `customer_unique_id`, `churn` (No leakage).
- **Transformations:** `np.log1p` applied to `total_orders` and `total_spend` to mitigate extreme right skew.
- **Scaling:** `StandardScaler` applied to equalize variance before computing Euclidean distances for K-Means.

## Model Selection (K)
- Tested K = 2 through 10.
- Selected K = **4**
- **WCSS:** 295860.19
- **Silhouette Score:** 0.3071

## Cluster Sizes
- **Cluster 0**: 57681 customers (60.7%)
- **Cluster 1**: 16875 customers (17.8%)
- **Cluster 2**: 2887 customers (3.0%)
- **Cluster 3**: 17543 customers (18.5%)

## Cluster Profiles & Interpretation

### Cluster 0 - Medium Recency, Low Frequency (One-time), Low Monetary Value, Very High Satisfaction
- Mean Spend: $110.34
- Mean Orders: 1.00
- Mean Recency: 238.9 days
- Mean Review Score: 4.74

### Cluster 1 - Medium Recency, Low Frequency (One-time), High Monetary Value
- Mean Spend: $358.18
- Mean Orders: 1.00
- Mean Recency: 261.5 days
- Mean Review Score: 4.30

### Cluster 2 - Medium Recency, High Frequency (Repeat Buyers), High Monetary Value
- Mean Spend: $308.41
- Mean Orders: 2.11
- Mean Recency: 227.2 days
- Mean Review Score: 4.17

### Cluster 3 - Medium Recency, Low Frequency (One-time), Medium Monetary Value, Poor Satisfaction
- Mean Spend: $139.05
- Mean Orders: 1.00
- Mean Recency: 247.3 days
- Mean Review Score: 1.85

## Business Insights
- Cluster 0 contributes the most overall revenue, despite not necessarily having the highest average spend.
- Cluster 1 has the highest average customer spend per capita (High-Value cohort).
- Cluster 2 contains the most frequent repeat buyers.
- Cluster 1 represents the most inactive, churning segment with the longest days since last order.
- Cluster 0 has the highest average review satisfaction scores.
