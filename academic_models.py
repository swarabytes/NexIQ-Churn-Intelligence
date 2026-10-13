import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ==================================================
# CONFIGURATION
# ==================================================
DATASET_PATH = 'data/real_ecommerce_data.csv'
TARGET_COLUMN = 'churn'
NUMERICAL_COLUMNS = [
    'recency', 'frequency', 'monetary', 'average_order_value', 
    'average_review_score', 'payment_count', 'delivery_delay', 'inactivity_days'
]
CATEGORICAL_COLUMNS = [
    'category_diversity'  # Example: Treating this as categorical if applicable
]

MODELS_DIR = 'models'
METRICS_FILE = os.path.join(MODELS_DIR, 'classification_metrics.json')


def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
            
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp)
        }
    }


def inspect_dataset(df):
    print("--- DATASET INSPECTION ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    if TARGET_COLUMN in df.columns:
        print(f"\nTarget Column ('{TARGET_COLUMN}') Distribution:")
        print(df[TARGET_COLUMN].value_counts(normalize=True))
        print(f"Total Churned (1): {df[TARGET_COLUMN].sum()}")
        print(f"Total Not Churned (0): {df.shape[0] - df[TARGET_COLUMN].sum()}")
    else:
        print(f"\nERROR: Target column '{TARGET_COLUMN}' not found!")
    print("--------------------------\n")


def train_academic_models():
    print("--- Starting Academic Classification Training ---")
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please provide the REAL dataset before proceeding.")
        return False

    df = pd.read_csv(DATASET_PATH)
    
    # 1. Dataset Inspection & Leakage Check Prep
    inspect_dataset(df)

    if TARGET_COLUMN not in df.columns:
        return False
        
    X = df[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # 2. Train/Test Split (Strictly before any imputation/scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLUMNS),
            ('cat', categorical_transformer, CATEGORICAL_COLUMNS)
        ])

    results = []
    
    # Helper to extract feature names from pipeline
    def get_feature_names(preprocessor):
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                try:
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                    feature_names.extend(cat_features)
                except:
                    feature_names.extend(columns)
        return feature_names

    # 4. Logistic Regression
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    
    lr_eval = evaluate_model(y_test, lr_pred, lr_proba)
    lr_eval["model"] = "Logistic Regression"
    
    feature_names = get_feature_names(lr_pipeline.named_steps['preprocessor'])
    coefs = lr_pipeline.named_steps['classifier'].coef_[0]
    lr_eval["Logistic Regression Coefficients"] = {feature_names[i]: float(coefs[i]) for i in range(len(feature_names))}
    results.append(lr_eval)
    
    with open(os.path.join(MODELS_DIR, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr_pipeline, f)

    # 5. Decision Tree
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
    ])
    dt_pipeline.fit(X_train, y_train)
    dt_pred = dt_pipeline.predict(X_test)
    dt_proba = dt_pipeline.predict_proba(X_test)[:, 1]
    
    dt_eval = evaluate_model(y_test, dt_pred, dt_proba)
    dt_eval["model"] = "Decision Tree"
    importances = dt_pipeline.named_steps['classifier'].feature_importances_
    dt_eval["feature_importance"] = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
    results.append(dt_eval)
    
    with open(os.path.join(MODELS_DIR, 'decision_tree.pkl'), 'wb') as f:
        pickle.dump(dt_pipeline, f)

    # 6. Random Forest
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    
    rf_eval = evaluate_model(y_test, rf_pred, rf_proba)
    rf_eval["model"] = "Random Forest"
    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    rf_eval["feature_importance"] = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
    results.append(rf_eval)
    
    with open(os.path.join(MODELS_DIR, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rf_pipeline, f)

    # 7. Naive Bayes
    nb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    # GaussianNB expects dense arrays, and OneHotEncoder might produce sparse arrays
    # If using sparse matrices, convert them to dense before GaussianNB.
    # We can handle this by using a custom dense transformer or changing OHE to sparse=False
    categorical_transformer.set_params(onehot__sparse_output=False)
    
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    nb_proba = nb_pipeline.predict_proba(X_test)[:, 1]
    
    nb_eval = evaluate_model(y_test, nb_pred, nb_proba)
    nb_eval["model"] = "Naive Bayes"
    nb_eval["feature_importance"] = None
    nb_eval["interpretation"] = "Gaussian Naive Bayes does not provide direct feature importance."
    results.append(nb_eval)
    
    with open(os.path.join(MODELS_DIR, 'naive_bayes.pkl'), 'wb') as f:
        pickle.dump(nb_pipeline, f)
        
    # Save metrics
    with open(METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("--- Academic Classification Training Complete ---")
    print(f"Results saved to {METRICS_FILE}")
    return True

if __name__ == '__main__':
    train_academic_models()
