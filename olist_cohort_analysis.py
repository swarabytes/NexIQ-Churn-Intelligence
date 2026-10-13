import pandas as pd
import os

extract_path = 'data/olist_raw'

orders = pd.read_csv(os.path.join(extract_path, 'olist_orders_dataset.csv'))
customers = pd.read_csv(os.path.join(extract_path, 'olist_customers_dataset.csv'))

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
valid_orders = orders[orders['order_status'] != 'canceled'].copy()
merged = valid_orders.merge(customers, on='customer_id')

earliest = valid_orders['order_purchase_timestamp'].min()
latest = valid_orders['order_purchase_timestamp'].max()

def evaluate_cohort(threshold_days):
    cutoff_date = latest - pd.Timedelta(days=threshold_days)
    
    # Historical orders
    history_orders = merged[merged['order_purchase_timestamp'] <= cutoff_date]
    future_orders = merged[merged['order_purchase_timestamp'] > cutoff_date]
    
    # Calculate order counts per customer in history
    order_counts = history_orders.groupby('customer_unique_id').size()
    
    # Cohort A: >= 2 orders in history
    cohort_a_customers = order_counts[order_counts >= 2].index
    
    # Cohort B: >= 2 orders in history AND time between first and last order >= 30 days
    first_order = history_orders.groupby('customer_unique_id')['order_purchase_timestamp'].min()
    last_order = history_orders.groupby('customer_unique_id')['order_purchase_timestamp'].max()
    tenure = (last_order - first_order).dt.days
    
    cohort_b_30 = tenure[(order_counts >= 2) & (tenure >= 30)].index
    cohort_b_60 = tenure[(order_counts >= 2) & (tenure >= 60)].index
    cohort_b_90 = tenure[(order_counts >= 2) & (tenure >= 90)].index
    
    future_customers = set(future_orders['customer_unique_id'])
    
    cohorts = {
        'A (>=2 orders)': cohort_a_customers,
        'B30 (>=2 orders, >=30d history)': cohort_b_30,
        'B60 (>=2 orders, >=60d history)': cohort_b_60,
        'B90 (>=2 orders, >=90d history)': cohort_b_90
    }
    
    results = []
    for name, cohort in cohorts.items():
        total = len(cohort)
        if total == 0:
            continue
        
        retained = len(set(cohort).intersection(future_customers))
        churned = total - retained
        
        churn_pct = (churned / total) * 100
        retain_pct = (retained / total) * 100
        baseline_acc = max(churn_pct, retain_pct)
        
        results.append({
            'Cohort': name,
            'Churn Window': f'{threshold_days} days',
            'Customers': total,
            'Churned': churned,
            'Retained': retained,
            'Churn %': round(churn_pct, 2),
            'Retained %': round(retain_pct, 2),
            'Baseline Acc %': round(baseline_acc, 2)
        })
    return results

all_results = []
for threshold in [90, 120, 180]:
    all_results.extend(evaluate_cohort(threshold))

df_res = pd.DataFrame(all_results)
print(df_res.to_string(index=False))
