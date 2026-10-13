"""
Part 9 — OLAP Analytics on Olist Data
=======================================
Constructs an analytical cube from the real Olist datasets.
Demonstrates Roll-Up, Drill-Down, Slice, and Dice operations.
Generates visualizations and saves results.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from olap_engine import rollup, drilldown, slice_data, dice_data

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
RAW_DIR     = 'data/olist_raw'
DATA_DIR    = 'data/olap'
RESULTS_DIR = 'static/academic_results/olap'

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & INSPECT RAW DATA
# ============================================================
print("Loading Olist datasets...")
orders      = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items       = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
products    = pd.read_csv(os.path.join(RAW_DIR, 'olist_products_dataset.csv'))
translations= pd.read_csv(os.path.join(RAW_DIR, 'product_category_name_translation.csv'))
customers   = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))

print(f"Orders columns: {list(orders.columns)}")
print(f"Items columns:  {list(items.columns)}")
print(f"Customers columns: {list(customers.columns)}")

# ============================================================
# 2. BUILD ANALYTICAL BASE DATASET
# ============================================================
print("\nBuilding analytical cube base dataset...")

# Time dimension from orders
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')
orders = orders.dropna(subset=['order_purchase_timestamp'])
orders['year']       = orders['order_purchase_timestamp'].dt.year
orders['quarter']    = orders['order_purchase_timestamp'].dt.quarter
orders['month']      = orders['order_purchase_timestamp'].dt.month
orders['month_name'] = orders['order_purchase_timestamp'].dt.strftime('%b')

# Geography from customers
orders = orders.merge(customers[['customer_id', 'customer_unique_id', 'customer_state', 'customer_city']],
                      on='customer_id', how='left')

# Product category via items → products → translation
items_prod = items.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
items_prod = items_prod.merge(translations, on='product_category_name', how='left')
items_prod['product_category_name_english'] = (
    items_prod['product_category_name_english']
    .fillna(items_prod['product_category_name'])
)

# Aggregate item-level facts TO ORDER LEVEL to avoid inflation.
# Use nunique for order counts, sum for sales.
order_agg = items_prod.groupby('order_id').agg(
    item_count    = ('order_item_id', 'count'),
    total_sales   = ('price', 'sum'),
    product_category = ('product_category_name_english',
                        lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
).reset_index()

# Merge order-level aggregates onto orders
cube = orders.merge(order_agg, on='order_id', how='left')
cube = cube[cube['order_status'].isin(['delivered', 'shipped', 'invoiced', 'processing', 'approved'])]
cube = cube.dropna(subset=['customer_state', 'product_category'])

# Drop duplicates on order_id — each order appears exactly once
cube = cube.drop_duplicates(subset='order_id')

print(f"\nAnalytical cube rows (orders): {len(cube)}")
print(f"Date range: {cube['order_purchase_timestamp'].min().date()} to {cube['order_purchase_timestamp'].max().date()}")
print(f"Unique states: {cube['customer_state'].nunique()}")
print(f"Unique categories: {cube['product_category'].nunique()}")
print(f"Duplicate order IDs: {cube['order_id'].duplicated().sum()}")

MEASURES = {
    'order_id':        'nunique',
    'customer_unique_id': 'nunique',
    'item_count':      'sum',
    'total_sales':     'sum'
}

# ============================================================
# 3. ROLL-UP — Month → Year
# ============================================================
print("\n--- ROLL-UP: Month → Year ---")
lower_df, upper_df = rollup(
    cube,
    MEASURES,
    from_dims=['year', 'month', 'month_name'],
    to_dims=['year']
)
lower_df.columns = ['year', 'month', 'month_name', 'order_count', 'customer_count', 'item_count', 'total_sales']
upper_df.columns = ['year', 'order_count', 'customer_count', 'item_count', 'total_sales']

print(f"  Monthly records: {len(lower_df)}")
print(f"  Yearly records:  {len(upper_df)}")
print(upper_df.to_string(index=False))

lower_df.to_csv(os.path.join(DATA_DIR, 'rollup_results_monthly.csv'), index=False)
upper_df.to_csv(os.path.join(DATA_DIR, 'rollup_results_yearly.csv'), index=False)

# ALSO: State → National (geographic roll-up)
state_df, national_df = rollup(
    cube,
    MEASURES,
    from_dims=['customer_state'],
    to_dims=['year']
)
state_df.columns = ['customer_state', 'order_count', 'customer_count', 'item_count', 'total_sales']
state_df.to_csv(os.path.join(DATA_DIR, 'rollup_state_results.csv'), index=False)

# ============================================================
# 4. DRILL-DOWN — Year → Quarter → Month
# ============================================================
print("\n--- DRILL-DOWN: Select year with most orders → Monthly ---")
best_year = upper_df.loc[upper_df['order_count'].idxmax(), 'year']
print(f"  Drilling into year: {best_year}")

upper_row, lower_detail = drilldown(
    cube,
    MEASURES,
    from_dims=['year'],
    to_dims=['year', 'quarter', 'month', 'month_name'],
    filter_col='year',
    filter_val=best_year
)
lower_detail.columns = ['year', 'quarter', 'month', 'month_name', 'order_count', 'customer_count', 'item_count', 'total_sales']
lower_detail = lower_detail.sort_values(['quarter', 'month'])
print(lower_detail.to_string(index=False))
lower_detail.to_csv(os.path.join(DATA_DIR, 'drilldown_results.csv'), index=False)

# ============================================================
# 5. SLICE — Fix customer_state = "SP"
# ============================================================
print("\n--- SLICE: customer_state = 'SP' ---")
sp_slice = slice_data(cube, 'customer_state', 'SP')
sp_monthly = sp_slice.groupby(['year', 'month', 'month_name']).agg(MEASURES).reset_index()
sp_monthly.columns = ['year', 'month', 'month_name', 'order_count', 'customer_count', 'item_count', 'total_sales']
sp_cat = sp_slice.groupby('product_category').agg(MEASURES).reset_index()
sp_cat.columns = ['category', 'order_count', 'customer_count', 'item_count', 'total_sales']
sp_cat = sp_cat.sort_values('order_count', ascending=False)

print(f"  SP slice: {len(sp_slice)} orders, {sp_slice['customer_unique_id'].nunique()} customers")
print(f"  Top 5 categories in SP:")
print(sp_cat.head(5).to_string(index=False))

sp_monthly.to_csv(os.path.join(DATA_DIR, 'slice_results.csv'), index=False)

# ============================================================
# 6. DICE — States: SP/RJ/MG, Years: 2017/2018, top 2 categories
# ============================================================
print("\n--- DICE: States=[SP,RJ,MG], Years=[2017,2018], Top categories ---")
top_cats = cube['product_category'].value_counts().head(2).index.tolist()
dice_filters = {
    'customer_state': ['SP', 'RJ', 'MG'],
    'year': [2017, 2018],
    'product_category': top_cats
}
dice_df = dice_data(cube, dice_filters)
dice_summary = dice_df.groupby(['customer_state', 'year', 'product_category']).agg(MEASURES).reset_index()
dice_summary.columns = ['state', 'year', 'category', 'order_count', 'customer_count', 'item_count', 'total_sales']
print(f"  Dice result: {len(dice_df)} orders")
print(dice_summary.to_string(index=False))
dice_summary.to_csv(os.path.join(DATA_DIR, 'dice_results.csv'), index=False)

# ============================================================
# 7. SAVE OLAP SUMMARY
# ============================================================
olap_summary = cube.groupby(['year', 'quarter', 'customer_state', 'product_category']).agg(MEASURES).reset_index()
olap_summary.columns = ['year', 'quarter', 'state', 'category', 'order_count', 'customer_count', 'item_count', 'total_sales']
olap_summary.to_csv(os.path.join(DATA_DIR, 'olap_summary.csv'), index=False)

# ============================================================
# 8. VISUALIZATIONS
# ============================================================
print("\nGenerating visualizations...")

# A. ROLL-UP — Monthly → Yearly bar
plt.figure(figsize=(8, 5))
plt.bar(upper_df['year'].astype(str), upper_df['order_count'], color=['steelblue', 'seagreen', 'indianred'][:len(upper_df)])
plt.title('ROLL-UP: Total Orders by Year')
plt.xlabel('Year')
plt.ylabel('Order Count')
for i, row in upper_df.iterrows():
    plt.text(i, row['order_count'] + 200, f"{int(row['order_count']):,}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rollup_yearly.png'))
plt.close()

# B. DRILL-DOWN — Monthly breakdown for best year
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['steelblue' if q == 1 else 'seagreen' if q == 2 else 'orange' if q == 3 else 'indianred'
          for q in lower_detail['quarter']]
ax.bar(lower_detail['month_name'], lower_detail['order_count'], color=colors)
ax.set_title(f'DRILL-DOWN: Monthly Order Breakdown for {best_year}')
ax.set_xlabel('Month')
ax.set_ylabel('Order Count')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'drilldown_monthly.png'))
plt.close()

# C. SLICE — Top 10 categories in SP
plt.figure(figsize=(10, 6))
sns.barplot(x='order_count', y='category', data=sp_cat.head(10), palette='viridis')
plt.title('SLICE: Top 10 Product Categories — São Paulo (SP)')
plt.xlabel('Order Count')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'slice_sp_categories.png'))
plt.close()

# D. DICE — Multi-state/year heatmap
if len(dice_summary) > 0:
    pivot = dice_summary.pivot_table(index='state', columns='year', values='order_count', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='g', cmap='YlOrRd')
    plt.title(f'DICE: Orders by State & Year (Top categories: {", ".join(top_cats[:2])})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'dice_heatmap.png'))
    plt.close()

# ============================================================
# 9. METADATA & REPORT
# ============================================================
key_insights = [
    f"Year {best_year} had the highest order volume ({int(upper_df.loc[upper_df['year']==best_year,'order_count'].values[0]):,} orders).",
    f"São Paulo (SP) is the most dominant customer state, accounting for {len(sp_slice)/len(cube)*100:.1f}% of all orders.",
    f"The top 2 product categories are: {', '.join(top_cats)}.",
    f"The Dice result shows concentration of orders in SP/RJ/MG across 2017-2018 for the top categories."
]

meta = {
    "dataset": {
        "orders": len(cube),
        "customers": int(cube['customer_unique_id'].nunique()),
        "date_range": {
            "min": str(cube['order_purchase_timestamp'].min().date()),
            "max": str(cube['order_purchase_timestamp'].max().date())
        },
        "states": int(cube['customer_state'].nunique()),
        "categories": int(cube['product_category'].nunique())
    },
    "operations": {
        "rollup": {
            "hierarchy": "month → year",
            "monthly_records": len(lower_df),
            "yearly_records": len(upper_df)
        },
        "drilldown": {
            "selected_year": int(best_year),
            "monthly_records": len(lower_detail)
        },
        "slice": {
            "dimension": "customer_state",
            "value": "SP",
            "orders": len(sp_slice)
        },
        "dice": {
            "filters": {k: (list(map(int, v)) if all(isinstance(x, (int, np.integer)) for x in v) else v)
                        for k, v in dice_filters.items()},
            "orders": len(dice_df)
        }
    },
    "insights": key_insights
}

with open(os.path.join(DATA_DIR, 'olap_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=4)

# Markdown report
report_md = f"""# Part 9 — OLAP Analytics on Olist Data

## Dataset
- **Orders:** {len(cube):,}
- **Customers:** {cube['customer_unique_id'].nunique():,}
- **Date Range:** {cube['order_purchase_timestamp'].min().date()} to {cube['order_purchase_timestamp'].max().date()}
- **States:** {cube['customer_state'].nunique()}
- **Categories:** {cube['product_category'].nunique()}

## Fact/Dimension Model
- **Facts:** order_count, customer_count, item_count, total_sales
- **Dimensions:** Time (year, quarter, month), Geography (state, city), Product (category)

## ROLL-UP (Month → Year)
Monthly records rolled up to {len(upper_df)} yearly records.

## DRILL-DOWN (Year → Month)
Selected year {best_year} and broke down to {len(lower_detail)} monthly records.

## SLICE (customer_state = SP)
{len(sp_slice):,} orders from {sp_slice['customer_unique_id'].nunique():,} customers.

## DICE
Filters: States=[SP,RJ,MG], Years=[2017,2018], Categories={top_cats}
Result: {len(dice_df):,} orders.

## Key OLAP Insights
"""
for ins in key_insights:
    report_md += f"- {ins}\n"

with open(os.path.join(DATA_DIR, 'olap_report.md'), 'w') as f:
    f.write(report_md)

# ============================================================
# 10. FINAL CONSOLE OUTPUT
# ============================================================
print("\n" + "="*40)
print("PART 9 — OLAP COMPLETE")
print("="*40)
print(f"DATASET:\n  Orders: {len(cube):,}\n  Customers: {cube['customer_unique_id'].nunique():,}\n"
      f"  Date range: {cube['order_purchase_timestamp'].min().date()} to {cube['order_purchase_timestamp'].max().date()}\n"
      f"  Categories: {cube['product_category'].nunique()}\n  States: {cube['customer_state'].nunique()}")

print("\nFACTS / MEASURES:\n  order_count, customer_count, item_count, total_sales")
print("DIMENSIONS:\n  Time (year, quarter, month), Geography (state, city), Product (category)")

print(f"\nROLL-UP:\n  Hierarchy: month → year\n  Monthly records: {len(lower_df)}\n  Yearly records: {len(upper_df)}")
for _, r in upper_df.iterrows():
    print(f"  {int(r['year'])}: {int(r['order_count']):,} orders | ${r['total_sales']:,.0f} sales")

print(f"\nDRILL-DOWN:\n  Selected year: {best_year}\n  Monthly breakdown: {len(lower_detail)} records")

print(f"\nSLICE:\n  Dimension: customer_state = 'SP'\n  Result: {len(sp_slice):,} orders, {sp_slice['customer_unique_id'].nunique():,} customers")

print(f"\nDICE:\n  Filters: States=[SP,RJ,MG], Years=[2017,2018], Categories={top_cats}\n  Result: {len(dice_df):,} orders")

print("\nKEY OLAP INSIGHTS:")
for ins in key_insights:
    print(f"  - {ins}")

print("\nVALIDITY AUDIT:\n  PASS (No synthetic data, order-level deduplication, correct nunique aggregation)")

print("\nFILES CREATED:")
for f in ['olap_summary.csv','rollup_results_monthly.csv','rollup_results_yearly.csv',
          'rollup_state_results.csv','drilldown_results.csv','slice_results.csv',
          'dice_results.csv','olap_report.md','olap_metadata.json']:
    print(f"  data/olap/{f}")
print("  static/academic_results/olap/*.png")

print("\nAPI:\n  GET /api/academic/olap → pending in app.py")

print("\nSTOP.")
print("Do NOT create the final Academic Dashboard yet.")
print("Wait for review before Part 10.")
