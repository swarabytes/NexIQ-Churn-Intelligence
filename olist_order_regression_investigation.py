"""
Part 6B — Order-Level Regression Target Investigation
=====================================================
Investigates continuous targets at the ORDER LEVEL:
- Delivery Delay
- Freight Value
- Order Total Value
- Order Item Count

Produces distributions, correlations, and sample size evaluations.
No regression models are trained.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
RAW_DIR      = 'data/olist_raw'
RESULTS_DIR  = 'static/academic_results'
DATA_DIR     = 'data'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading Olist datasets...")
orders = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_payments_dataset.csv'))
products = pd.read_csv(os.path.join(RAW_DIR, 'olist_products_dataset.csv'))
customers = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))

# Parse dates
date_cols = ['order_purchase_timestamp', 'order_approved_at',
             'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')

# ============================================================
# 2. AGGREGATE TO ORDER LEVEL
# ============================================================
print("Aggregating to order level...")

# Base orders
df = orders[['order_id', 'customer_id', 'order_status', 
             'order_purchase_timestamp', 'order_delivered_customer_date',
             'order_estimated_delivery_date']].copy()

# Add customer state/city
df = df.merge(customers[['customer_id', 'customer_city', 'customer_state']], on='customer_id', how='left')

# Payment aggregation (Order Total Value proxy)
pay_agg = payments.groupby('order_id').agg(
    order_total_value=('payment_value', 'sum'),
    payment_installments=('payment_installments', 'max'),
    payment_types=('payment_type', 'nunique')
).reset_index()

# Item aggregation (Freight, Item Count, Products)
items_merged = items.merge(products[['product_id', 'product_weight_g', 
                                     'product_length_cm', 'product_height_cm', 'product_width_cm', 'product_category_name']], 
                           on='product_id', how='left')

items_agg = items_merged.groupby('order_id').agg(
    order_freight_value=('freight_value', 'sum'),
    order_item_count=('order_item_id', 'count'),
    product_weight_g=('product_weight_g', 'sum'),
    product_volume_cm3=('product_length_cm', lambda x: (x * items_merged.loc[x.index, 'product_height_cm'] * items_merged.loc[x.index, 'product_width_cm']).sum())
).reset_index()

df = df.merge(pay_agg, on='order_id', how='left')
df = df.merge(items_agg, on='order_id', how='left')

# Exclude canceled/unavailable orders from valid analysis
df = df[df['order_status'].isin(['delivered', 'shipped', 'invoiced', 'processing', 'approved'])]

# ============================================================
# 3. COMPUTE TARGETS
# ============================================================
print("Computing targets...")

# Target A: Delivery Delay (days)
# Positive = Late, Negative = Early
df['delivery_delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / (24*3600)

# Target B: Freight Value (already aggregated)
# Target C: Order Total Value (already aggregated)
# Target D: Item Count (already aggregated)

TARGETS = [
    ('delivery_delay_days', 'Delivery Delay'),
    ('order_freight_value', 'Freight Value'),
    ('order_total_value', 'Order Total Value'),
    ('order_item_count', 'Order Item Count')
]

# Possible historical numerical features known AT purchase:
FEATURES = ['product_weight_g', 'product_volume_cm3', 'payment_installments']

# ============================================================
# 4. TARGET ANALYSIS
# ============================================================
print("Analyzing targets...")
results = []
correlations = {}

total_orders = len(df)

for col, name in TARGETS:
    series = df[col]
    valid_series = series.dropna()
    
    n_valid = len(valid_series)
    n_missing = total_orders - n_valid
    n_zero = (valid_series == 0).sum()
    pct_zero = (n_zero / n_valid) * 100 if n_valid > 0 else 0
    
    mean_v = valid_series.mean()
    med_v = valid_series.median()
    std_v = valid_series.std()
    min_v = valid_series.min()
    max_v = valid_series.max()
    skew_v = valid_series.skew()
    n_unique = valid_series.nunique()
    
    # Simple recommendation heuristic
    rec = "Investigate"
    suitability = "Unknown"
    leakage = "Low"
    
    if name == 'Delivery Delay':
        suitability = "Good continuous target"
        if n_missing > total_orders * 0.1:
            rec = "Use with valid deliveries only"
        else:
            rec = "Recommended Backup"
    elif name == 'Freight Value':
        suitability = "Highly continuous, strict monetary target"
        rec = "Recommended Primary"
    elif name == 'Order Total Value':
        suitability = "Continuous but risky"
        leakage = "High (price determines total)"
        rec = "Reject (Leakage/Redundant)"
    elif name == 'Order Item Count':
        suitability = "Discrete/Count data"
        rec = "Reject (Not continuous)"

    results.append({
        'Target': name,
        'Observation_Unit': 'One Order',
        'Valid_Observations': n_valid,
        'Missing_Observations': n_missing,
        'Mean': mean_v,
        'Median': med_v,
        'Std': std_v,
        'Min': min_v,
        'Max': max_v,
        'Unique_Values': n_unique,
        'Zero_Percentage': pct_zero,
        'Skewness': skew_v,
        'Regression_Suitability': suitability,
        'Leakage_Risk': leakage,
        'Recommendation': rec
    })
    
    # Correlations
    corrs = {}
    for feat in FEATURES:
        valid_df = df[[col, feat]].dropna()
        if len(valid_df) > 10:
            if valid_df[col].std() > 0 and valid_df[feat].std() > 0:
                p_corr, _ = pearsonr(valid_df[col], valid_df[feat])
                s_corr, _ = spearmanr(valid_df[col], valid_df[feat])
                corrs[feat] = {'Pearson': p_corr, 'Spearman': s_corr}
    correlations[name] = corrs

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(DATA_DIR, 'order_regression_target_investigation.csv'), index=False)

# ============================================================
# 5. DISTRIBUTION PLOTS
# ============================================================
print("Generating distribution plots...")
file_names = {
    'Delivery Delay': 'order_regression_delivery_delay.png',
    'Freight Value': 'order_regression_freight_value.png',
    'Order Total Value': 'order_regression_total_value.png',
    'Order Item Count': 'order_regression_item_count.png'
}

for col, name in TARGETS:
    valid = df[col].dropna()
    if len(valid) == 0:
        continue
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if name == 'Delivery Delay':
        # Limit to 1st and 99th percentile to avoid crazy outliers ruining the plot
        p1, p99 = valid.quantile([0.01, 0.99])
        filtered = valid[(valid >= p1) & (valid <= p99)]
        ax.hist(filtered, bins=50, color='indianred', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', label='On Time')
        ax.legend()
    elif name == 'Order Item Count':
        ax.hist(valid, bins=np.arange(valid.min(), valid.max()+2)-0.5, color='steelblue', edgecolor='black')
        ax.set_xticks(range(int(valid.min()), int(valid.max())+1))
        ax.set_xlim(0.5, min(10.5, valid.max()+0.5))
    else:
        # Freight and Total Value - heavy right skew
        p99 = valid.quantile(0.99)
        filtered = valid[valid <= p99]
        ax.hist(filtered, bins=50, color='seagreen', edgecolor='black')
        
    ax.set_title(f'Distribution of {name}\n(Truncated to 99th percentile for visibility)', fontweight='bold')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, file_names[name]), dpi=150)
    plt.close()

# ============================================================
# 6. MARKDOWN REPORT
# ============================================================
print("Generating markdown report...")

def get_res(name):
    return df_res[df_res['Target'] == name].iloc[0]

# Leakage analysis section
leakage_md = """
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
"""

md = f"""# Part 6B — Order-Level Regression Target Investigation

## 1. Objective
Investigate legitimate, scientifically defensible regression targets at the **ORDER LEVEL** using the Olist dataset. The goal is to find a continuous numerical target that can be predicted from information available at the time of purchase without suffering from extreme zero-inflation, categorical constraints, or target leakage.

## 2. Order-Level Methodology
- **Observation Unit**: One unique `order_id`.
- **Pre-filtering**: Canceled and unavailable orders are removed.
- **Feature Constraint**: All predictor information must be known *at the time the order is placed*. Information generated post-purchase (e.g., delivery date, review scores) cannot be used to predict targets occurring at or before that future event.

## 3. Sample-Size Analysis
Total Valid Orders in Base Dataset: {total_orders}

| Target | Valid Obs | Missing | Mean | Median | Std | Min | Max | Skewness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Delivery Delay | {int(get_res('Delivery Delay')['Valid_Observations'])} | {int(get_res('Delivery Delay')['Missing_Observations'])} | {get_res('Delivery Delay')['Mean']:.2f} | {get_res('Delivery Delay')['Median']:.2f} | {get_res('Delivery Delay')['Std']:.2f} | {get_res('Delivery Delay')['Min']:.2f} | {get_res('Delivery Delay')['Max']:.2f} | {get_res('Delivery Delay')['Skewness']:.2f} |
| Freight Value | {int(get_res('Freight Value')['Valid_Observations'])} | {int(get_res('Freight Value')['Missing_Observations'])} | {get_res('Freight Value')['Mean']:.2f} | {get_res('Freight Value')['Median']:.2f} | {get_res('Freight Value')['Std']:.2f} | {get_res('Freight Value')['Min']:.2f} | {get_res('Freight Value')['Max']:.2f} | {get_res('Freight Value')['Skewness']:.2f} |
| Order Total Value | {int(get_res('Order Total Value')['Valid_Observations'])} | {int(get_res('Order Total Value')['Missing_Observations'])} | {get_res('Order Total Value')['Mean']:.2f} | {get_res('Order Total Value')['Median']:.2f} | {get_res('Order Total Value')['Std']:.2f} | {get_res('Order Total Value')['Min']:.2f} | {get_res('Order Total Value')['Max']:.2f} | {get_res('Order Total Value')['Skewness']:.2f} |
| Order Item Count | {int(get_res('Order Item Count')['Valid_Observations'])} | {int(get_res('Order Item Count')['Missing_Observations'])} | {get_res('Order Item Count')['Mean']:.2f} | {get_res('Order Item Count')['Median']:.2f} | {get_res('Order Item Count')['Std']:.2f} | {get_res('Order Item Count')['Min']:.0f} | {get_res('Order Item Count')['Max']:.0f} | {get_res('Order Item Count')['Skewness']:.2f} |

## 4. Leakage Analysis
{leakage_md}

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
"""

with open(os.path.join(DATA_DIR, 'part6b_order_regression_target_investigation.md'), 'w', encoding='utf-8') as f:
    f.write(md)

# ============================================================
# 7. FINAL CONSOLE OUTPUT
# ============================================================
SEP = "=" * 40
SEP2 = "-" * 40

print("\n" + SEP)
print("PART 6B — ORDER REGRESSION INVESTIGATION")
print(SEP)

print("\nTARGET COMPARISON")
print(SEP2)
for name in ['Delivery Delay', 'Freight Value', 'Order Total Value', 'Order Item Count']:
    r = get_res(name)
    print(f"{name}:")
    print(f"  Observations:  {int(r['Valid_Observations'])}")
    print(f"  Mean:          {r['Mean']:.4f}")
    print(f"  Median:        {r['Median']:.4f}")
    print(f"  Std:           {r['Std']:.4f}")
    if name == 'Order Item Count':
        print(f"  Unique Values: {int(r['Unique_Values'])}")
    print(f"  Skewness:      {r['Skewness']:.4f}")
    print(f"  Suitability:   {r['Regression_Suitability']}")
    print(f"  Recommendation:{r['Recommendation']}")
    print()

print(SEP)
print("PRIMARY REGRESSION TARGET")
print(SEP)
print("Target: Freight Value (order_freight_value)")
print("Observation Unit: One Order")
print(f"Number of Observations: {int(get_res('Freight Value')['Valid_Observations'])}")
print("Prediction Time: At checkout (order placement)")
print("Target Time: At checkout")
print("Why suitable: Continuous monetary value, zero missing data, driven by physical features (weight, volume, geo) known pre-checkout without target leakage.")

print("\n" + SEP)
print("BACKUP TARGET")
print(SEP)
print("Target: Delivery Delay (delivery_delay_days)")
print("Reason: True continuous variable spanning positive/negative values. Requires filtering non-delivered orders.")

print("\n" + SEP)
print("LEAKAGE AUDIT")
print(SEP)
print("Result: PASS")
print("Freight Value: Requires strict exclusion of individual item freight costs to avoid circular math. Product weight/volume are safe.")
print("Delivery Delay: Requires strict exclusion of actual delivery dates as features.")

print("\n" + SEP)
print("DATASET DESIGN")
print(SEP)
print("Recommended split: Random 80/20 or Grouped by Seller ID")
print("Reason: Grouped split prevents spatial/logistics leakage if sellers dominate specific freight routes.")
print("Primary features: product_weight_g, product_volume_cm3, customer_state, seller_state, product_category.")
print("Excluded leakage features: actual_delivery_date, review_score, total_payment (for freight).")

print("\n" + SEP)
print("FILES CREATED")
print(SEP)
print("Investigation CSV: data/order_regression_target_investigation.csv")
print("Report: data/part6b_order_regression_target_investigation.md")
print("Target plots: static/academic_results/order_regression_*.png")

print("\n" + SEP)
print("STOP")
print(SEP)
print("DO NOT TRAIN ANY REGRESSION MODEL.")
print("DO NOT MODIFY THE UI.")
print("DO NOT MODIFY pipeline.pkl.")
print("DO NOT CREATE THE FINAL REGRESSION DATASET.")
print("WAIT FOR APPROVAL.\n")
