# Part 7A — Customer Clustering Dataset Investigation

## 1. Objective
Design a scientifically valid customer-level dataset for Unsupervised Learning (Clustering) to segment customers into meaningful behavioral groups.

## 2. Data Source & Observation Unit
- **Source**: Olist raw datasets.
- **Unit**: Exactly one row per `customer_unique_id`.
- **Unique Customers**: 94986
- **Usable Customers**: 94986 (Zero dropped due to missing values after feature engineering)

## 3. Candidate Feature Statistics

| Feature | Mean | Median | Min | Max | P99 | Skewness | Zeros |
|---|---:|---:|---:|---:|---:|---:|---:|
| days_since_last_order | 244.09 | 224.79 | 1.00 | 729.49 | 580.99 | 0.45 | 0 |
| total_orders | 1.03 | 1.00 | 1.00 | 16.00 | 2.00 | 11.51 | 0 |
| total_spend | 165.69 | 107.90 | 0.00 | 13664.08 | 1098.68 | 9.11 | 1 |
| average_order_value | 160.75 | 105.74 | 0.00 | 13664.08 | 1058.85 | 9.31 | 1 |
| average_items_per_order | 1.14 | 1.00 | 0.00 | 21.00 | 3.00 | 7.62 | 3 |
| total_items_purchased | 1.18 | 1.00 | 0.00 | 24.00 | 4.00 | 7.44 | 3 |
| category_diversity | 1.03 | 1.00 | 0.00 | 15.00 | 2.00 | 5.93 | 1302 |
| average_review_score | 4.11 | 5.00 | 1.00 | 5.00 | 5.00 | -1.42 | 0 |
| total_review_count | 1.03 | 1.00 | 0.00 | 16.00 | 2.00 | 8.54 | 684 |
| average_payment_installments | 2.92 | 2.00 | 0.00 | 24.00 | 10.00 | 1.60 | 2 |
| payment_type_count | 1.06 | 1.00 | 1.00 | 16.00 | 2.00 | 7.67 | 0 |
| customer_tenure_days | 2.67 | 0.00 | 0.00 | 633.08 | 94.75 | 12.26 | 92349 |

## 4. Correlation Analysis
Highly correlated pairs (|r| >= 0.80):
- `total_orders` and `category_diversity` (r = 0.82)
- `total_orders` and `total_review_count` (r = 0.89)
- `total_orders` and `payment_type_count` (r = 0.82)
- `total_spend` and `average_order_value` (r = 0.98)
- `average_items_per_order` and `total_items_purchased` (r = 0.89)

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
