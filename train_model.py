import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Configuration paths
DATA_PATH = 'data/processed_ecommerce_data.csv'
MODEL_SAVE_PATH = 'pipeline.pkl'

def main():
    print("--- Starting Customer Churn Model Training ---")
    
    # 1. Load the processed dataset
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found. Please run data_pipeline.py first.")
        return

    # 2. Select Features & Target
    # We are using strictly numerical features for simplicity and clarity in the viva
    features = [
        'recency', 
        'frequency', 
        'monetary', 
        'average_order_value', 
        'category_diversity', 
        'average_review_score', 
        'payment_count', 
        'delivery_delay', 
        'inactivity_days'
    ]
    
    target = 'churn'
    
    # Isolate X and y
    X = df[features].copy()
    y = df[target]

    # 3. Preprocessing: Handle missing values safely (just in case)
    # Filling with 0 or the column mean
    X.fillna(X.mean(), inplace=True)
    
    # 4. Train/Test Split
    # Split 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 5. Build Pipeline
    # Pipeline ensures data scaling and training happen systematically
    print("\nTraining Logistic Regression model...")
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Normalizes numeric features to have mean=0, std=1
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])

    # 6. Train the Model
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 7. Evaluate the Model
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 8. Save the Model
    # Saving pipeline as pipeline.pkl
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print(f"--- Model saved successfully as: {MODEL_SAVE_PATH} ---")

if __name__ == '__main__':
    main()
