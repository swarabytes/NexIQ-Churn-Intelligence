# Part 8 — Association Rule Mining

## Dataset Overview
- **Total Orders:** 99441
- **Valid Transactions:** 96824
- **Multi-Category Transactions:** 726 (0.75%)
- **Transaction Setup:** Order-level, duplicate categories within the same order were removed.

## Methodology
- **Algorithm:** Apriori
- **Min Support:** 5e-05 (Chosen because multi-category orders are extremely rare in Olist. Higher support yields zero rules).
- **Min Confidence:** 0.01
- **Condition:** Lift > 1.0

## Results
- **Frequent Itemsets:** 108
- **Valid Association Rules:** 1

## Top 5 Rules by Lift
1. `home_confort` → `bed_bath_table` (Support: 0.00044, Conf: 0.11, Lift: 1.12)

## Business Insights
- Cross-selling opportunity: Customers buying [home_confort] are strongly associated with [bed_bath_table] (Lift: 1.12).
- Product bundling: [home_confort] almost reliably leads to [bed_bath_table] with 10.8% confidence.
- Due to the extremely low rate of multi-category orders (most customers buy 1 item), overall support for rules is very low. These rules represent niche, specialized shopper behaviors rather than broad trends.
