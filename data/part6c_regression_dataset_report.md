# Part 6C — Regression Dataset Report

## Dataset Construction
- **Observation Unit**: One unique order.
- **Aggregation**: Order items are summed per `order_id` to prevent duplicates.
- **Total Rows**: 98199
- **Duplicates**: 0

## Target Definition
- **Target**: `order_freight_value`
- **Definition**: Sum of freight values for all items in the order.

## Features
**Numerical:**
- `product_weight_g` (Sum of all item weights in order)
- `product_volume_cm3` (Sum of all item volumes in order)
- `order_item_count` (Number of items in the order)

**Categorical:**
- `customer_state` (State code of the customer)
- `seller_state` (State code of the primary seller)
- `product_category` (Mode category among items in the order)
- `order_month` (Month of order purchase)
- `order_weekday` (Day of week of order purchase)

## Missing-Value Handling
- Numerical: Median Imputation
- Categorical: Most Frequent Imputation + OneHotEncoding

## Leakage Exclusions
The following were strictly excluded:
- Individual item `freight_value`
- Payment totals (`payment_value`)
- Actual/Estimated delivery dates
- Post-purchase review data
- `customer_unique_id` (not used as predictor)
- `seller_id` (kept ONLY for grouping, not passed to models)

## Split Strategy
- GroupShuffleSplit (test_size=0.20, random_state=42) grouped by `seller_id`.
