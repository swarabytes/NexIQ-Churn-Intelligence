"""
Part 6C — Train Regression Models for Order Freight Value
=========================================================
Target     : order_freight_value
Observation: One Olist Order
Split      : GroupShuffleSplit (test_size=0.20) grouped by seller_id
Models     : Simple LR, Multiple LR, Lasso, Ridge, Elastic Net
Metrics    : R2, Adjusted R2, MAE, MSE, RMSE
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
RAW_DIR      = 'data/olist_raw'
DATA_DIR     = 'data'
RESULTS_DIR  = 'static/academic_results/regression'
MODELS_DIR   = 'models/regression'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ============================================================
# 1. BUILD REGRESSION DATASET
# ============================================================
print("Loading raw Olist datasets...")
orders = pd.read_csv(os.path.join(RAW_DIR, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(RAW_DIR, 'olist_order_items_dataset.csv'))
products = pd.read_csv(os.path.join(RAW_DIR, 'olist_products_dataset.csv'))
customers = pd.read_csv(os.path.join(RAW_DIR, 'olist_customers_dataset.csv'))
sellers = pd.read_csv(os.path.join(RAW_DIR, 'olist_sellers_dataset.csv'))

# Convert dates
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'], errors='coerce')

# Exclude unavailable / canceled
orders = orders[orders['order_status'].isin(['delivered', 'shipped', 'invoiced', 'processing', 'approved'])]

# Filter out orders with missing purchase timestamps
orders = orders.dropna(subset=['order_purchase_timestamp'])

# Create time-based features known at checkout
orders['order_month'] = orders['order_purchase_timestamp'].dt.month
orders['order_weekday'] = orders['order_purchase_timestamp'].dt.weekday

# Merge customers
df = orders[['order_id', 'customer_id', 'order_month', 'order_weekday']].copy()
df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

# Prepare items features
items_prod = items.merge(products[['product_id', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'product_category_name']], on='product_id', how='left')
items_prod = items_prod.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

items_prod['product_volume_cm3'] = items_prod['product_length_cm'] * items_prod['product_height_cm'] * items_prod['product_width_cm']

# Aggregate items to Order level
items_agg = items_prod.groupby('order_id').agg(
    order_freight_value=('freight_value', 'sum'),
    product_weight_g=('product_weight_g', 'sum'),
    product_volume_cm3=('product_volume_cm3', 'sum'),
    order_item_count=('order_item_id', 'count'),
    seller_id=('seller_id', 'first'), # Taking first seller. Assumption: mostly 1 seller per order.
    seller_state=('seller_state', 'first'),
    product_category=('product_category_name', lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
).reset_index()

# Combine base orders with item aggregations
df = df.merge(items_agg, on='order_id', how='inner')

# Ensure exactly one row per order
assert df['order_id'].is_unique, "Duplicate order_id detected!"

# Final Features
TARGET = 'order_freight_value'
NUM_FEATURES = ['product_weight_g', 'product_volume_cm3', 'order_item_count']
CAT_FEATURES = ['customer_state', 'seller_state', 'product_category', 'order_month', 'order_weekday']
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Keep seller_id ONLY for grouping, customer_id and order_id for dropping later
df = df[['order_id', 'customer_id', 'seller_id', TARGET] + ALL_FEATURES]

# Drop rows with missing TARGET
df = df.dropna(subset=[TARGET])

# Save Dataset
df.to_csv(os.path.join(DATA_DIR, 'olist_freight_regression_dataset.csv'), index=False)

print("\n--- TARGET DATA QUALITY ---")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Number of duplicate order_ids: {df.duplicated(subset=['order_id']).sum()}")
print("Missing values per feature:")
print(df[ALL_FEATURES].isna().sum())
print(f"\nTarget mean: {df[TARGET].mean():.2f}")
print(f"Target median: {df[TARGET].median():.2f}")
print(f"Target std: {df[TARGET].std():.2f}")
print(f"Target min: {df[TARGET].min():.2f}")
print(f"Target max: {df[TARGET].max():.2f}")
print(f"Target skewness: {df[TARGET].skew():.2f}")

# ============================================================
# 2. DATASET REPORT
# ============================================================
report_md = f"""# Part 6C — Regression Dataset Report

## Dataset Construction
- **Observation Unit**: One unique order.
- **Aggregation**: Order items are summed per `order_id` to prevent duplicates.
- **Total Rows**: {len(df)}
- **Duplicates**: {df.duplicated(subset=['order_id']).sum()}

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
"""
with open(os.path.join(DATA_DIR, 'part6c_regression_dataset_report.md'), 'w') as f:
    f.write(report_md)

# ============================================================
# 3. TRAIN / TEST SPLIT (GROUPED BY SELLER)
# ============================================================
X = df[ALL_FEATURES]
y = df[TARGET]
groups = df['seller_id']

# We must ensure `order_month` and `order_weekday` are treated as strings/categories for OneHotEncoder
X['order_month'] = X['order_month'].astype(str)
X['order_weekday'] = X['order_weekday'].astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

print("\n--- TRAIN / TEST SPLIT ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Unique sellers in train: {groups_train.nunique()}")
print(f"Unique sellers in test: {groups_test.nunique()}")

overlap = set(groups_train).intersection(set(groups_test))
print(f"Seller overlap between train/test: {len(overlap)}")
assert len(overlap) == 0, "Leakage! Sellers overlap between train and test."

# ============================================================
# 4. PREPROCESSING
# ============================================================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, NUM_FEATURES),
        ('cat', cat_transformer, CAT_FEATURES)
    ])

# ============================================================
# 5. TRAIN MODELS
# ============================================================
print("\nTraining models...")
metrics_results = []
fitted_models = {}

def get_adj_r2(r2, n, p):
    if p >= n - 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def evaluate_model(name, y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = get_adj_r2(r2, n, p)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'Model': name,
        'R2': r2,
        'Adjusted_R2': adj_r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'p': p,
        'y_pred': y_pred
    }

# A. Simple Linear Regression (Only product_weight_g)
slr_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['product_weight_g'])
    ])
slr_pipeline = Pipeline(steps=[('preprocessor', slr_preprocessor), ('model', LinearRegression())])
slr_pipeline.fit(X_train, y_train)
y_pred_slr = slr_pipeline.predict(X_test)
m_slr = evaluate_model('Simple Linear Regression', y_test, y_pred_slr, len(y_test), 1)
metrics_results.append(m_slr)
fitted_models['Simple Linear Regression'] = slr_pipeline

# Base pipeline for multiple models
def build_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Get number of features after one-hot encoding for Adjusted R2
# We fit the preprocessor to count the output columns
X_train_proc = preprocessor.fit_transform(X_train)
p_multiple = X_train_proc.shape[1]

# B. Multiple Linear Regression
mlr = build_pipeline(LinearRegression())
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
m_mlr = evaluate_model('Multiple Linear Regression', y_test, y_pred_mlr, len(y_test), p_multiple)
metrics_results.append(m_mlr)
fitted_models['Multiple Linear Regression'] = mlr

# C. Lasso Regression (alpha=0.01)
las = build_pipeline(Lasso(alpha=0.01, random_state=42))
las.fit(X_train, y_train)
y_pred_las = las.predict(X_test)
m_las = evaluate_model('Lasso', y_test, y_pred_las, len(y_test), p_multiple)
metrics_results.append(m_las)
fitted_models['Lasso'] = las

# D. Ridge Regression (alpha=1.0)
rid = build_pipeline(Ridge(alpha=1.0, random_state=42))
rid.fit(X_train, y_train)
y_pred_rid = rid.predict(X_test)
m_rid = evaluate_model('Ridge', y_test, y_pred_rid, len(y_test), p_multiple)
metrics_results.append(m_rid)
fitted_models['Ridge'] = rid

# E. Elastic Net (alpha=0.01, l1_ratio=0.5)
enet = build_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))
enet.fit(X_train, y_train)
y_pred_enet = enet.predict(X_test)
m_enet = evaluate_model('Elastic Net', y_test, y_pred_enet, len(y_test), p_multiple)
metrics_results.append(m_enet)
fitted_models['Elastic Net'] = enet

# ============================================================
# 6. MODEL COMPARISON
# ============================================================
df_metrics = pd.DataFrame(metrics_results).sort_values(by='RMSE', ascending=True)

# Determine best model
best_model_row = df_metrics.iloc[0]
best_model_name = best_model_row['Model']
best_model = fitted_models[best_model_name]

print("\n--- MODEL COMPARISON ---")
print(f"{'Model':<30} | {'R²':<8} | {'Adj R²':<8} | {'MAE':<8} | {'MSE':<10} | {'RMSE':<8}")
print("-" * 80)
for idx, row in df_metrics.iterrows():
    print(f"{row['Model']:<30} | {row['R2']:.4f}   | {row['Adjusted_R2']:.4f}   | {row['MAE']:.4f}   | {row['MSE']:.4f}     | {row['RMSE']:.4f}")

df_metrics_save = df_metrics.drop(columns=['p', 'y_pred'])
df_metrics_save.to_csv(os.path.join(DATA_DIR, 'part6c_regression_results.csv'), index=False)

# ============================================================
# 7. RESIDUAL ANALYSIS
# ============================================================
best_y_pred = best_model_row['y_pred']
residuals = y_test - best_y_pred

res_mean = residuals.mean()
res_median = residuals.median()
res_std = residuals.std()

print(f"\n--- RESIDUAL ANALYSIS (Best Model: {best_model_name}) ---")
print(f"Mean residual: {res_mean:.4f}")
print(f"Median residual: {res_median:.4f}")
print(f"Residual Std Dev: {res_std:.4f}")
print(f"MAE: {best_model_row['MAE']:.4f}")
print(f"RMSE: {best_model_row['RMSE']:.4f}")

if best_model_row['R2'] > 0.6 and res_std < y_test.std():
    res_status = "Good"
elif best_model_row['R2'] > 0.3:
    res_status = "Moderate"
else:
    res_status = "Poor"
print(f"Automatic interpretation: {res_status}")

# ============================================================
# 8. MODEL INTERPRETATION (Best Model)
# ============================================================
print(f"\n--- MODEL INTERPRETATION ({best_model_name}) ---")

def get_feature_names(pipeline):
    try:
        num_cols = NUM_FEATURES
        cat_enc = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        cat_cols = cat_enc.get_feature_names_out(CAT_FEATURES)
        return list(num_cols) + list(cat_cols)
    except Exception as e:
        return []

feature_names = get_feature_names(best_model)
if feature_names and hasattr(best_model.named_steps['model'], 'coef_'):
    coefs = best_model.named_steps['model'].coef_
    df_coef = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    
    top_pos = df_coef.sort_values(by='Coefficient', ascending=False).head(10)
    top_neg = df_coef.sort_values(by='Coefficient', ascending=True).head(10)
    
    print("Top 10 POSITIVE coefficients (Increase Freight Value):")
    for _, r in top_pos.iterrows():
        print(f"  {r['Feature']:<40} : {r['Coefficient']:.4f}")
        
    print("\nTop 10 NEGATIVE coefficients (Decrease Freight Value):")
    for _, r in top_neg.iterrows():
        print(f"  {r['Feature']:<40} : {r['Coefficient']:.4f}")
else:
    print("Feature importance not directly extracted for this model/setup.")

# ============================================================
# 9. VISUALIZATIONS
# ============================================================
print("\nGenerating visualizations...")

# Target Distribution
plt.figure(figsize=(8,5))
plt.hist(df[TARGET], bins=100, color='seagreen', edgecolor='black')
plt.title('Target Distribution: Order Freight Value')
plt.xlabel('Freight Value')
plt.ylabel('Frequency')
plt.xlim(0, df[TARGET].quantile(0.99))
plt.savefig(os.path.join(RESULTS_DIR, 'target_distribution.png'))
plt.close()

# Actual vs Predicted & Residuals for each model
for _, row in df_metrics.iterrows():
    m_name = row['Model']
    clean_name = m_name.replace(" ", "_").lower()
    preds = row['y_pred']
    
    # Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, preds, alpha=0.3, color='blue')
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2) # y=x line
    plt.title(f'Actual vs Predicted: {m_name}')
    plt.xlabel('Actual Freight Value')
    plt.ylabel('Predicted Freight Value')
    plt.xlim(0, y_test.quantile(0.99))
    plt.ylim(0, y_test.quantile(0.99))
    plt.savefig(os.path.join(RESULTS_DIR, f'actual_vs_predicted_{clean_name}.png'))
    plt.close()
    
    # Residual Plot
    plt.figure(figsize=(8,5))
    res = y_test - preds
    plt.scatter(preds, res, alpha=0.3, color='purple')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(f'Residual Plot: {m_name}')
    plt.xlabel('Predicted Freight Value')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.xlim(0, pd.Series(preds).quantile(0.99))
    plt.ylim(res.quantile(0.01), res.quantile(0.99))
    plt.savefig(os.path.join(RESULTS_DIR, f'residual_plot_{clean_name}.png'))
    plt.close()

# Model Comparison Bar Chart
metrics_to_plot = ['R2', 'RMSE', 'MAE']
df_plot = df_metrics_save.set_index('Model')[metrics_to_plot]
df_plot.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('Model Comparison: R², RMSE, MAE')
plt.ylabel('Score / Error Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
plt.close()

# Feature Coefficients Plot (for best model if linear)
if feature_names and hasattr(best_model.named_steps['model'], 'coef_'):
    combined = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient')
    plt.figure(figsize=(10, 8))
    plt.barh(combined['Feature'], combined['Coefficient'], color=['red' if x < 0 else 'green' for x in combined['Coefficient']])
    plt.title(f'Top Features: {best_model_name}')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_coefficients_best.png'))
    plt.close()

# ============================================================
# 10. SAVE ARTIFACTS
# ============================================================
print("\nSaving models and metadata...")

for m_name, pipeline in fitted_models.items():
    clean_name = m_name.replace(" ", "_").lower()
    with open(os.path.join(MODELS_DIR, f'{clean_name}_regression.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)

meta = {
    'target': TARGET,
    'features': ALL_FEATURES,
    'excluded_features': ['seller_id', 'customer_unique_id', 'payment_value', 'actual_delivery_date', 'review_score'],
    'split_strategy': 'GroupShuffleSplit by seller_id (test_size=0.20)',
    'random_state': 42,
    'models_trained': list(fitted_models.keys()),
    'best_model': best_model_name,
    'metrics': df_metrics_save.to_dict(orient='records')
}
with open(os.path.join(MODELS_DIR, 'regression_metadata.json'), 'w') as f:
    json.dump(meta, f, indent=4)

# ============================================================
# 11. FINAL OUTPUT
# ============================================================
print("\n" + "="*40)
print("PART 6C — REGRESSION COMPLETE")
print("="*40)
print(f"\nDataset: data/olist_freight_regression_dataset.csv")
print(f"Target: {TARGET}")
print(f"Observations: {len(df)}")
print(f"Train: {len(X_train)}")
print(f"Test: {len(X_test)}")
print(f"Split: Seller-grouped 80/20")
print(f"\nModels:")
for m in fitted_models.keys():
    print(f"  {m}")

print(f"\nBEST MODEL:\n  {best_model_name}")
print(f"R²:          {best_model_row['R2']:.4f}")
print(f"Adjusted R²: {best_model_row['Adjusted_R2']:.4f}")
print(f"MAE:         {best_model_row['MAE']:.4f}")
print(f"MSE:         {best_model_row['MSE']:.4f}")
print(f"RMSE:        {best_model_row['RMSE']:.4f}")

print("\nTOP FEATURES:")
if feature_names and hasattr(best_model.named_steps['model'], 'coef_'):
    for _, r in top_pos.head(3).iterrows():
        print(f"  + {r['Feature']}")
    for _, r in top_neg.head(3).iterrows():
        print(f"  - {r['Feature']}")
else:
    print("  (Check part6c output plots for feature info)")

print("\nLEAKAGE AUDIT:\n  PASS")
print("\nPIPELINE STATUS:\n  Existing churn pipeline untouched")

print("\nFILES CREATED:")
print("  data/olist_freight_regression_dataset.csv")
print("  data/part6c_regression_dataset_report.md")
print("  data/part6c_regression_results.csv")
print("  static/academic_results/regression/*.png")
print("  models/regression/*.pkl")
print("  models/regression/regression_metadata.json")
