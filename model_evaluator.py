import os
import json
import time
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

DATA_PATH = os.path.join('data', 'processed_ecommerce_data.csv')
MODELS_DIR = 'models'
HISTORY_FILE = os.path.join(MODELS_DIR, 'experiment_history.json')
LATEST_COMPARISON_FILE = os.path.join(MODELS_DIR, 'latest_comparison.json')
PROD_META_FILE = os.path.join(MODELS_DIR, 'production_model_meta.json')
PROD_PIPELINE_PATH = 'pipeline.pkl'
BACKUP_PIPELINE_PATH = os.path.join(MODELS_DIR, 'backup_pipeline.pkl')

FEATURES = [
    'recency', 'frequency', 'monetary', 'average_order_value',
    'category_diversity', 'average_review_score', 'payment_count',
    'delivery_delay', 'inactivity_days'
]
TARGET = 'churn'

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    avail_features = [f for f in FEATURES if f in df.columns]
    X = df[avail_features].copy()
    X.fillna(X.mean(), inplace=True)
    y = df[TARGET].astype(int)
    return X, y, avail_features

def evaluate_predictions(y_true, y_pred, y_prob, train_time, pred_time):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None else 0.0
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC curve points (sampled to 35 points for compact JSON and fast UI rendering)
    roc_points = []
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        step = max(1, len(fpr) // 35)
        for idx in range(0, len(fpr), step):
            roc_points.append({'fpr': round(float(fpr[idx]), 4), 'tpr': round(float(tpr[idx]), 4)})
        if roc_points[-1]['fpr'] != 1.0:
            roc_points.append({'fpr': 1.0, 'tpr': 1.0})

    # PR curve points
    pr_points = []
    if y_prob is not None:
        p_vals, r_vals, _ = precision_recall_curve(y_true, y_prob)
        step = max(1, len(p_vals) // 35)
        for idx in range(0, len(p_vals), step):
            pr_points.append({'recall': round(float(r_vals[idx]), 4), 'precision': round(float(p_vals[idx]), 4)})

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'specificity': round(spec, 4),
        'roc_auc': round(auc, 4),
        'pr_auc': round(pr_auc, 4),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        },
        'train_time_s': round(train_time, 4),
        'pred_time_ms': round(pred_time * 1000, 2),
        'roc_curve': roc_points,
        'pr_curve': pr_points
    }

def rank_models(models_list, criterion='f1_score'):
    """
    Ranks models based on configurable criteria.
    Primary: 'f1_score' or 'pr_auc'
    Secondary: 'recall'
    Tertiary: 'roc_auc'
    Quaternary: 'accuracy'
    """
    primary_key = 'pr_auc' if criterion == 'pr_auc' else 'f1_score'

    def sort_key(m):
        mets = m['metrics']
        return (
            mets.get(primary_key, 0),
            mets.get('recall', 0),
            mets.get('roc_auc', 0),
            mets.get('accuracy', 0)
        )

    sorted_models = sorted(models_list, key=sort_key, reverse=True)
    
    for idx, m in enumerate(sorted_models):
        m['rank'] = idx + 1
        m['is_best'] = (idx == 0)

    return sorted_models

def train_and_evaluate_all_models(criterion='f1_score', imbalance_method='balanced', test_size=0.2, random_state=42):
    """
    Trains and evaluates:
    1. Logistic Regression
    2. Random Forest Classifier
    3. XGBoost Classifier
    under identical 80:20 stratified split and preprocessing.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_config = [
        {
            'key': 'logistic_regression',
            'name': 'Logistic Regression',
            'family': 'Linear Probabilistic Classifier',
            'desc': 'L2-regularized logistic regression with balanced class weighting and standardized features.',
            'create': lambda cw: LogisticRegression(
                class_weight='balanced' if cw == 'balanced' else None,
                max_iter=1000,
                C=1.0,
                random_state=random_state
            )
        },
        {
            'key': 'random_forest',
            'name': 'Random Forest',
            'family': 'Ensemble of Decision Trees (Bagging)',
            'desc': '100 decision trees with bootstrap aggregation, balanced sub-sampling, and feature subsampling.',
            'create': lambda cw: RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight='balanced' if cw == 'balanced' else None,
                random_state=random_state
            )
        },
        {
            'key': 'xgboost',
            'name': 'XGBoost',
            'family': 'Gradient Boosted Decision Trees (Boosting)',
            'desc': 'Sequential gradient-boosted trees with log-loss objective and class-ratio positive weighting.',
            'create': lambda cw: xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=(len(y_train) - sum(y_train)) / max(1, sum(y_train)) if cw == 'balanced' else 1.0,
                random_state=random_state,
                eval_metric='logloss'
            )
        }
    ]

    evaluated_models = []
    trained_pipelines = {}

    for cfg in models_config:
        clf = cfg['create'](imbalance_method)
        steps = [('scaler', StandardScaler())]
        
        if imbalance_method == 'smote':
            steps.append(('smote', SMOTE(random_state=random_state)))
            
        steps.append(('classifier', clf))
        pipeline = ImbPipeline(steps)

        # 1. Fit with timing
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        # 2. Predict with timing
        t1 = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time = time.time() - t1
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # 3. Compute all metrics
        metrics = evaluate_predictions(y_test, y_pred, y_prob, train_time, pred_time)

        # 4. Feature importance if available
        feat_imp = {}
        fitted_clf = pipeline.named_steps['classifier']
        if hasattr(fitted_clf, 'feature_importances_'):
            imp_vals = fitted_clf.feature_importances_
            feat_imp = {feature_names[i]: round(float(imp_vals[i]), 4) for i in range(len(feature_names))}
        elif hasattr(fitted_clf, 'coef_'):
            coef_vals = fitted_clf.coef_[0]
            feat_imp = {feature_names[i]: round(float(coef_vals[i]), 4) for i in range(len(feature_names))}

        model_record = {
            'key': cfg['key'],
            'name': cfg['name'],
            'family': cfg['family'],
            'desc': cfg['desc'],
            'metrics': metrics,
            'feature_importance': feat_imp,
            'imbalance_method': imbalance_method
        }
        evaluated_models.append(model_record)
        trained_pipelines[cfg['key']] = pipeline

        # Save individual candidate model pickle
        candidate_path = os.path.join(MODELS_DIR, f"candidate_{cfg['key']}.pkl")
        joblib.dump(pipeline, candidate_path)

    # Rank the models based on configured criterion
    ranked_models = rank_models(evaluated_models, criterion=criterion)
    best_model = ranked_models[0]

    # Current production model metadata
    prod_meta = get_production_model_meta()

    run_payload = {
        'run_id': f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_version': 'Olist Brazilian E-Commerce (processed_ecommerce_data.csv)',
        'dataset_shape': {'total_samples': len(X), 'train_samples': len(X_train), 'test_samples': len(X_test), 'features': len(feature_names)},
        'class_distribution': {
            'train_class_0': int((y_train == 0).sum()),
            'train_class_1': int((y_train == 1).sum()),
            'test_class_0': int((y_test == 0).sum()),
            'test_class_1': int((y_test == 1).sum()),
            'churn_ratio': round(float(y.mean() * 100), 2)
        },
        'imbalance_method': imbalance_method,
        'selection_criterion': criterion,
        'models': ranked_models,
        'best_model': {
            'key': best_model['key'],
            'name': best_model['name'],
            'f1_score': best_model['metrics']['f1_score'],
            'recall': best_model['metrics']['recall'],
            'roc_auc': best_model['metrics']['roc_auc'],
            'accuracy': best_model['metrics']['accuracy']
        },
        'active_production_model': prod_meta
    }

    # Save to latest_comparison.json
    with open(LATEST_COMPARISON_FILE, 'w', encoding='utf-8') as f:
        json.dump(run_payload, f, indent=2)

    # Append to experiment_history.json
    history = get_experiment_history()
    # Keep up to 20 most recent runs
    history_entry = {
        'run_id': run_payload['run_id'],
        'timestamp': run_payload['timestamp'],
        'imbalance_method': imbalance_method,
        'criterion': criterion,
        'best_model': best_model['name'],
        'best_f1': best_model['metrics']['f1_score'],
        'best_accuracy': best_model['metrics']['accuracy'],
        'best_recall': best_model['metrics']['recall'],
        'models_summary': [
            {'key': m['key'], 'name': m['name'], 'rank': m['rank'], 'f1': m['metrics']['f1_score'], 'acc': m['metrics']['accuracy'], 'rec': m['metrics']['recall'], 'auc': m['metrics']['roc_auc']}
            for m in ranked_models
        ]
    }
    history.insert(0, history_entry)
    history = history[:25]
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return run_payload

def get_latest_comparison():
    if os.path.exists(LATEST_COMPARISON_FILE):
        try:
            with open(LATEST_COMPARISON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['active_production_model'] = get_production_model_meta()
                return data
        except Exception:
            pass
    # If not yet trained, run it automatically once!
    return train_and_evaluate_all_models()

def get_experiment_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def get_production_model_meta():
    if os.path.exists(PROD_META_FILE):
        try:
            with open(PROD_META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    
    # Infer from pipeline.pkl if exists
    name = "Logistic Regression"
    key = "logistic_regression"
    if os.path.exists(PROD_PIPELINE_PATH):
        try:
            p = joblib.load(PROD_PIPELINE_PATH)
            clf = p.named_steps.get('classifier')
            if isinstance(clf, RandomForestClassifier):
                name = "Random Forest"
                key = "random_forest"
            elif isinstance(clf, xgb.XGBClassifier):
                name = "XGBoost"
                key = "xgboost"
        except Exception:
            pass

    return {
        'model_key': key,
        'model_name': name,
        'updated_at': 'Initial System Setup',
        'status': 'Active Production',
        'selection_reason': 'Default Champion Pipeline'
    }

def set_production_model(model_key, reason=None):
    """
    Sets the specified model as the active production pipeline (pipeline.pkl).
    Supports: 'logistic_regression', 'random_forest', 'xgboost', or 'best'.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    latest = get_latest_comparison()
    
    if model_key in ('best', 'auto'):
        model_key = latest['best_model']['key']

    valid_keys = {'logistic_regression', 'random_forest', 'xgboost'}
    if model_key not in valid_keys:
        raise ValueError(f"Invalid model key '{model_key}'. Must be one of {valid_keys} or 'best'")

    candidate_path = os.path.join(MODELS_DIR, f"candidate_{model_key}.pkl")
    if not os.path.exists(candidate_path):
        # Trigger training if candidate does not exist
        train_and_evaluate_all_models()

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate model file {candidate_path} not found.")

    # 1. Back up existing pipeline.pkl if it exists
    if os.path.exists(PROD_PIPELINE_PATH):
        try:
            shutil.copy2(PROD_PIPELINE_PATH, BACKUP_PIPELINE_PATH)
        except Exception as e:
            print(f"Warning backing up pipeline: {e}")

    # 2. Overwrite pipeline.pkl with the candidate model
    shutil.copy2(candidate_path, PROD_PIPELINE_PATH)

    # 3. Find metrics of chosen model
    chosen_record = None
    for m in latest.get('models', []):
        if m['key'] == model_key:
            chosen_record = m
            break

    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }

    meta = {
        'model_key': model_key,
        'model_name': model_display_names.get(model_key, model_key),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'selection_reason': reason or f"User explicitly selected {model_display_names.get(model_key, model_key)} as Production Model",
        'status': 'Active Production',
        'metrics': chosen_record['metrics'] if chosen_record else {},
        'features': FEATURES
    }

    with open(PROD_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Also update latest_comparison active production model
    latest['active_production_model'] = meta
    with open(LATEST_COMPARISON_FILE, 'w', encoding='utf-8') as f:
        json.dump(latest, f, indent=2)

    return meta
