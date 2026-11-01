# Part 6B — Order-Level Regression Target Investigation

## 1. Objective
Investigate legitimate, scientifically defensible regression targets at the **ORDER LEVEL** using the Olist dataset. The goal is to find a continuous numerical target that can be predicted from information available at the time of purchase without suffering from extreme zero-inflation, categorical constraints, or target leakage.

## 2. Order-Level Methodology
- **Observation Unit**: One unique `order_id`.
- **Pre-filtering**: Canceled and unavailable orders are removed.
- **Feature Constraint**: All predictor information must be known *at the time the order is placed*. Information generated post-purchase (e.g., delivery date, review scores) cannot be used to predict targets occurring at or before that future event.

## 3. Sample-Size Analysis
Total Valid Orders in Base Dataset: 98202

| Target | Valid Obs | Missing | Mean | Median | Std | Min | Max | Skewness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Delivery Delay | 96470 | 1732 | -11.18 | -11.95 | 10.18 | -146.02 | 188.98 | 2.01 |
| Freight Value | 98199 | 3 | 22.82 | 17.19 | 21.65 | 0.00 | 1794.96 | 12.10 |
| Order Total Value | 98201 | 1 | 160.27 | 105.28 | 219.13 | 9.59 | 13664.08 | 9.26 |
| Order Item Count | 98199 | 3 | 1.14 | 1.00 | 0.54 | 1 | 21 | 7.55 |

## 4. Leakage Analysis

### Delivery Delay
- **Safe Predictors**: product weight, product dimensions, order purchase time/day, seller state, customer state, freight value.
- **Unsafe (Leakage)**: actual delivery date, review scores (occur after delivery).

### Freight Value
- **Safe Predictors**: product weight, product volume, customer state, seller state, product category, order total item count.
- **Unsafe (Leakage)**: freight value components, delivery delay (occurs after).

### Order Total Value
- **Unsafe (Leakage)**: order total value is a direct mathematical sum of product prices and freight. Trying to predict order total using product prices is circular accounting, not predictive machine learning.

### Order Item Count
- **Unsafe (Leakage)**: using item identifiers or sum of items. 


## 5. Regression Suitability

**Delivery Delay:** 
Very suitable. It is a true continuous variable (can be positive or negative) measuring physical logistics. Predictors (distance/state, weight, volume) are known at purchase time. Missingness (~2.9%) is acceptable (just drop orders that were never delivered).

**Freight Value:** 
Very suitable. Strictly continuous numerical (monetary). Fully known at checkout. Predictors (weight, volume, origin/destination states) determine the cost without causing target leakage.

**Order Total Value:**
Unsuitable. Predicting the total cart value from the items in the cart is circular math, not machine learning. Predicting it from customer demographics yields extremely weak models.

**Order Item Count:**
Unsuitable. It is highly discrete (Poisson distribution). ~88% of orders have exactly 1 item. This violates linear regression assumptions and is better suited for Poisson regression or classification.

## 6. Target Recommendations

### Primary Target: Freight Value (`order_freight_value`)
- **Why suitable:** Continuous, real-world monetary interpretation, zero missing data, no zero-inflation (min > 0), strong physical predictors available at purchase (weight, volume, geo coordinates).
- **Prediction Time:** At checkout.
- **Target Time:** At checkout.

### Backup Target: Delivery Delay (`delivery_delay_days`)
- **Why suitable:** True continuous variable spanning positive (late) and negative (early) values. Meaningful logistics application. 
- **Prediction Time:** At checkout.
- **Target Time:** Upon delivery (future).

## 7. Recommended Dataset Design
- **Splitting:** A standard random 80/20 split is acceptable if we treat orders as independent transactions. However, if multiple orders come from the same seller, a **Grouped Split** (by `seller_id` or `customer_unique_id`) is technically superior to prevent data leakage across the train/test boundary.
- **Transformation:** Freight Value is right-skewed. A logarithmic transformation `np.log1p()` is highly recommended for linear models.

## 8. Proposed Part 6C Methodology
Create a robust order-level regression dataset using `order_freight_value` as the target.
Merge in product weight/volume and customer/seller geographical features.
Ensure no target leakage occurs.
