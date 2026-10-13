import os
import pandas as pd
import numpy as np

# Configuration
RAW_DATA_DIR = 'data/olist_raw'
PROCESSED_DATA_PATH = 'data/processed_ecommerce_data.csv'
CHURN_THRESHOLD_DAYS = 90  # Any customer inactive for more than 90 days is considered At Risk / Churned

def load_data():
    """Loads all necessary Olist CSV files into Pandas DataFrames."""
    print("Loading raw data files...")
    
    customers = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_customers_dataset.csv'))
    orders = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_items_dataset.csv'))
    payments = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_order_reviews_dataset.csv'))
    products = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_products_dataset.csv'))
    
    return customers, orders, order_items, payments, reviews, products

def clean_and_merge(customers, orders, order_items, payments, reviews, products):
    """Merges relational tables and processes date formats."""
    print("Cleaning and merging datasets...")
    
    # 1. Convert dates to datetime
    date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_columns:
        orders[col] = pd.to_datetime(orders[col])
        
    # Merge orders with customers
    # Note: 'customer_id' in orders links to a specific order, while 'customer_unique_id' in customers identifies the actual person
    orders_customers = pd.merge(orders, customers, on='customer_id', how='inner')
    
    # Focus on delivered orders only to ensure valid purchases
    delivered_orders = orders_customers[orders_customers['order_status'] == 'delivered'].copy()
    
    # 2. Aggregate Payments to order level
    order_payments = payments.groupby('order_id').agg(
        total_payment=('payment_value', 'sum'),
        payment_count=('payment_sequential', 'count')
    ).reset_index()
    
    # 3. Aggregate Reviews to order level
    order_reviews = reviews.groupby('order_id').agg(
        review_score=('review_score', 'mean')
    ).reset_index()
    
    # 4. Prepare Category diversity per customer
    # Merge items with products to get categories
    items_products = pd.merge(order_items, products[['product_id', 'product_category_name']], on='product_id', how='left')
    
    # Map each item back to the unique customer
    customer_items = pd.merge(delivered_orders[['order_id', 'customer_unique_id']], items_products, on='order_id', how='inner')
    
    # Count unique categories purchased by each customer
    customer_categories = customer_items.groupby('customer_unique_id').agg(
        category_diversity=('product_category_name', 'nunique')
    ).reset_index()
    
    # 5. Bring everything together at the Order level
    df_merged = pd.merge(delivered_orders, order_payments, on='order_id', how='left')
    df_merged = pd.merge(df_merged, order_reviews, on='order_id', how='left')
    
    # Calculate delivery delay in days (actual - estimated)
    # Positive means delayed, negative means early
    df_merged['delivery_delay'] = (df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']).dt.total_seconds() / (24 * 3600)
    
    return df_merged, customer_categories

def create_customer_features(df_merged, customer_categories):
    """Aggregates order-level data into a single customer-level dataset."""
    print("Building customer-level features...")
    
    customer_df = df_merged.groupby('customer_unique_id').agg(
        first_purchase_date=('order_purchase_timestamp', 'min'),
        last_purchase_date=('order_purchase_timestamp', 'max'),
        total_orders=('order_id', 'nunique'),
        total_spend=('total_payment', 'sum'),
        average_review_score=('review_score', 'mean'),
        payment_count=('payment_count', 'sum'),
        delivery_delay=('delivery_delay', 'mean')
    ).reset_index()
    
    # Add category diversity
    customer_df = pd.merge(customer_df, customer_categories, on='customer_unique_id', how='left')
    
    # RFM metrics calculation
    # Simulate "today" as the most recent purchase date in the entire dataset
    dataset_max_date = customer_df['last_purchase_date'].max()
    
    customer_df['inactivity_days'] = (dataset_max_date - customer_df['last_purchase_date']).dt.total_seconds() / (24 * 3600)
    customer_df['recency'] = customer_df['inactivity_days']  # recency is the exact same as inactivity days here
    customer_df['frequency'] = customer_df['total_orders']
    customer_df['monetary'] = customer_df['total_spend']
    
    # Safe division for average order value
    customer_df['average_order_value'] = np.where(
        customer_df['total_orders'] > 0, 
        customer_df['total_spend'] / customer_df['total_orders'], 
        0
    )
    
    # Label Creation: Churn based on inactivity rule
    customer_df['churn'] = (customer_df['inactivity_days'] > CHURN_THRESHOLD_DAYS).astype(int)
    
    # Data Cleaning / Missing value imputation
    # Impute missing average review score with the median across all customers
    customer_df['average_review_score'] = customer_df['average_review_score'].fillna(customer_df['average_review_score'].median())
    customer_df['delivery_delay'] = customer_df['delivery_delay'].fillna(0)
    customer_df['category_diversity'] = customer_df['category_diversity'].fillna(0)
    customer_df['payment_count'] = customer_df['payment_count'].fillna(1)
    
    # Rename id to customer_id to standardise
    customer_df.rename(columns={'customer_unique_id': 'customer_id'}, inplace=True)
    
    # Format output fields
    customer_df['first_purchase_date'] = customer_df['first_purchase_date'].dt.date
    customer_df['last_purchase_date'] = customer_df['last_purchase_date'].dt.date
    
    # Reorder columns logically
    cols = [
        'customer_id', 'first_purchase_date', 'last_purchase_date', 'inactivity_days',
        'recency', 'frequency', 'monetary', 'total_orders', 'total_spend', 'average_order_value',
        'category_diversity', 'average_review_score', 'payment_count', 'delivery_delay',
        'churn'
    ]
    
    return customer_df[cols]

def main():
    print("--- Starting Olist Data Pipeline ---")
    
    # Ensure data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found.")
        print("Please ensure the extracted Olist CSVs are placed there.")
        return
        
    try:
        # Pipeline Steps
        customers, orders, order_items, payments, reviews, products = load_data()
        
        df_merged, customer_categories = clean_and_merge(
            customers, orders, order_items, payments, reviews, products
        )
        
        final_customer_df = create_customer_features(df_merged, customer_categories)
        
        # Output
        final_customer_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"\nData successfully processed and saved to: {PROCESSED_DATA_PATH}")
        
        # Summary printing
        total_customers = len(final_customer_df)
        total_orders = final_customer_df['total_orders'].sum()
        at_risk = final_customer_df['churn'].sum()
        active = total_customers - at_risk
        
        min_date = final_customer_df['first_purchase_date'].min()
        max_date = final_customer_df['last_purchase_date'].max()
        
        print("\n--- Data Summary ---")
        print(f"Dataset Date Range: {min_date} to {max_date}")
        print(f"Total Unique Customers: {total_customers:,}")
        print(f"Total Orders: {total_orders:,}")
        print(f"Active Customers ({CHURN_THRESHOLD_DAYS} days or less): {active:,} ({active/total_customers*100:.1f}%)")
        print(f"At-Risk ('Churned') Customers (>{CHURN_THRESHOLD_DAYS} days): {at_risk:,} ({at_risk/total_customers*100:.1f}%)")
        print("------------------------------------")
        
    except Exception as e:
        print(f"\nPipeline failed due to an error: {str(e)}")

if __name__ == "__main__":
    main()
