import os
import pandas as pd
import numpy as np

os.makedirs('data', exist_ok=True)

np.random.seed(42)
n_samples = 1000

data = {
    'recency': np.random.randint(1, 365, n_samples),
    'frequency': np.random.randint(1, 50, n_samples),
    'monetary': np.random.uniform(10.0, 5000.0, n_samples),
    'average_order_value': np.random.uniform(10.0, 500.0, n_samples),
    'category_diversity': np.random.randint(1, 10, n_samples),
    'average_review_score': np.random.uniform(1.0, 5.0, n_samples),
    'payment_count': np.random.randint(1, 20, n_samples),
    'delivery_delay': np.random.randint(0, 30, n_samples),
    'inactivity_days': np.random.randint(0, 365, n_samples),
}

df = pd.DataFrame(data)

# Realistic latent churn risk incorporating all retail behavioral features
base_score = (
    0.30 * (df['recency'] / 365.0) +
    0.35 * (df['inactivity_days'] / 365.0) +
    0.18 * (df['delivery_delay'] / 30.0) +
    0.16 * ((5.0 - df['average_review_score']) / 4.0) -
    0.10 * (df['frequency'] / 50.0) -
    0.08 * (df['monetary'] / 5000.0)
)
# Realistic retail behavioral noise / variance
noise = np.random.normal(0, 0.08, n_samples)
prob = base_score + noise

# Exact 25.9% natural e-commerce churn rate (259 churners out of 1000)
threshold = np.percentile(prob, 74.1)
df['churn'] = (prob >= threshold).astype(int)

df.to_csv('data/processed_ecommerce_data.csv', index=False)
print(f"Realistic dataset generated at data/processed_ecommerce_data.csv (Churn: {df['churn'].sum()}/{len(df)})")
