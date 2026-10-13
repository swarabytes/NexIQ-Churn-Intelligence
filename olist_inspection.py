import pandas as pd
import os

extract_path = 'data/olist_raw'

# Load Orders
orders = pd.read_csv(os.path.join(extract_path, 'olist_orders_dataset.csv'))
customers = pd.read_csv(os.path.join(extract_path, 'olist_customers_dataset.csv'))

# Dates
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
valid_orders = orders[orders['order_status'] != 'canceled'].copy()

earliest = valid_orders['order_purchase_timestamp'].min()
latest = valid_orders['order_purchase_timestamp'].max()

print("--- ORDER TIMELINE ---")
print(f"Earliest: {earliest}")
print(f"Latest: {latest}")
print(f"Total valid orders: {valid_orders.shape[0]}")

# Join to get unique customers (customer_unique_id)
merged = valid_orders.merge(customers, on='customer_id')

unique_customers = merged['customer_unique_id'].nunique()
orders_per_customer = merged.groupby('customer_unique_id').size()

print("\n--- CUSTOMER STATS ---")
print(f"Unique customers: {unique_customers}")
print(f"Customers with 1 order: {(orders_per_customer == 1).sum()}")
print(f"Customers with >1 orders: {(orders_per_customer > 1).sum()}")

# Churn thresholds
# To prevent target leakage, we split the timeline.
# Let's say observation period is earliest -> (latest - X days).
# Churn prediction window is (latest - X days) -> latest.
print("\n--- CHURN THRESHOLDS ---")
for threshold in [90, 120, 180]:
    cutoff_date = latest - pd.Timedelta(days=threshold)
    
    # Active before cutoff
    history_orders = merged[merged['order_purchase_timestamp'] <= cutoff_date]
    customers_in_history = set(history_orders['customer_unique_id'])
    
    # Orders in the churn window
    future_orders = merged[merged['order_purchase_timestamp'] > cutoff_date]
    customers_in_future = set(future_orders['customer_unique_id'])
    
    # A customer is churned if they were in history but NOT in future
    churned = customers_in_history - customers_in_future
    retained = customers_in_history.intersection(customers_in_future)
    
    total = len(customers_in_history)
    print(f"Threshold: {threshold} days")
    print(f"  Observation period: {earliest.date()} to {cutoff_date.date()}")
    print(f"  Churn window: {cutoff_date.date()} to {latest.date()}")
    print(f"  Customers existing before cutoff: {total}")
    if total > 0:
        print(f"  Churned: {len(churned)} ({len(churned)/total*100:.2f}%)")
        print(f"  Active/Retained: {len(retained)} ({len(retained)/total*100:.2f}%)")
