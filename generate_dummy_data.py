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

# Generate a synthetic 'churn' target
# Let's say high recency, high inactivity, low monetary -> high churn probability
churn_prob = (
    0.3 * (df['recency'] / 365) + 
    0.4 * (df['inactivity_days'] / 365) - 
    0.2 * (df['monetary'] / 5000) -
    0.1 * (df['average_review_score'] / 5)
)
df['churn'] = (churn_prob > 0.3).astype(int)

df.to_csv('data/processed_ecommerce_data.csv', index=False)
print("Dummy data generated at data/processed_ecommerce_data.csv")
