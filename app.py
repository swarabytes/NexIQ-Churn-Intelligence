from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import uuid
import os
import json
from functools import wraps
import db

app = Flask(__name__)
# Secret key for secure session management
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key_change_in_prod")

# ---------------------------------------------------------
# Load Model at Startup
# ---------------------------------------------------------
try:
    pipeline = joblib.load('pipeline.pkl')
    print("Successfully loaded model pipeline (pipeline.pkl)")
except Exception as e:
    print(f"Error loading model pipeline: {e}")
    pipeline = None

try:
    db.init_db()
    print("Database initialized successfully.")
except Exception as e:
    print(f"Error initializing database: {e}")


# ---------------------------------------------------------
# Authentication Decorator
# ---------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'company_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        company = db.authenticate_company(email, password)
        if company:
            session['company_id'] = company['id']
            session['company_name'] = company['company_name']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid email or password. Please try again.")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        success, msg = db.create_company(company_name, email, password)
        if success:
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error=msg)
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------------------------------------------------
# Application Routes
# ---------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """Landing page."""
    return render_template('index.html')

@app.route('/predict_ui', methods=['GET'])
@login_required
def predict_ui():
    """Prediction form page."""
    return render_template('predict.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """Analytics dashboard page."""
    return render_template('dashboard.html')

@app.route('/bulk-upload', methods=['GET', 'POST'])
@login_required
def bulk_upload():
    """Bulk upload page and file handler scoped to company."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        allowed_extensions = {'.csv', '.xlsx'}
        ext = os.path.splitext(file.filename)[1].lower()
        
        if ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type. Allowed: CSV, XLSX"}), 400
            
        try:
            if ext == '.csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine="openpyxl")
                
            MAX_ROWS = 5000
            if len(df) > MAX_ROWS:
                return jsonify({
                    "error": "File too large",
                    "details": f"Maximum allowed is {MAX_ROWS} rows per batch."
                }), 400
                
            required_cols = [
                'recency', 'frequency', 'monetary', 'average_order_value',
                'category_diversity', 'average_review_score', 'payment_count',
                'delivery_delay', 'inactivity_days'
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return jsonify({"error": "Missing required columns", "details": f"Missing: {', '.join(missing_cols)}"}), 400
                
            batch_id = f"BATCH_{str(uuid.uuid4())[:8].upper()}"
            
            total_rows = len(df)
            valid_rows = 0
            invalid_rows = 0
            high_risk = 0
            medium_risk = 0
            low_risk = 0
            total_prob = 0.0
            rev_at_risk = 0.0
            
            if pipeline is None:
                return jsonify({"error": "Model not loaded. Please ensure pipeline.pkl exists."}), 500
            
            company_id = session['company_id']
            
            for index, row in df.iterrows():
                try:
                    features = {col: float(row[col]) for col in required_cols}
                    df_input = pd.DataFrame([features])
                    prediction_array = pipeline.predict(df_input)
                    prediction = int(prediction_array[0])
                    probability = float(pipeline.predict_proba(df_input)[0][1])
                    
                    if probability < 0.3:
                        risk_level = "Low"
                        low_risk += 1
                    elif 0.3 <= probability <= 0.7:
                        risk_level = "Medium"
                        medium_risk += 1
                    else:
                        risk_level = "High"
                        high_risk += 1
                        rev_at_risk += features['monetary']
                        
                    total_prob += probability
                    
                    cust_id = str(row.get('customer_id', ''))
                    if not cust_id or pd.isna(row.get('customer_id')) or cust_id.strip() == 'nan':
                        cust_id = f"CUST_{str(uuid.uuid4())[:8].upper()}"
                        
                    health_score = round((1.0 - probability) * 100, 1)
                    
                    action = "Standard engagement"
                    if features['inactivity_days'] > 90:
                        action = "Send re-engagement offer"
                    elif features['monetary'] > 500 and probability > 0.7:
                        action = "Direct VIP outreach"
                    elif features['average_review_score'] < 3.0:
                        action = "Customer support follow-up"
                        
                    db.insert_prediction(
                        company_id=company_id,
                        customer_id=cust_id,
                        data=features,
                        prediction=prediction,
                        probability=probability,
                        risk_level=risk_level,
                        batch_id=batch_id,
                        source_file=file.filename,
                        upload_mode="bulk",
                        health_score=health_score,
                        recommendation=action
                    )
                    valid_rows += 1
                except Exception as row_error:
                    invalid_rows += 1
                    
            avg_prob = (total_prob / valid_rows) if valid_rows > 0 else 0
            
            return jsonify({
                "message": "Batch processed successfully",
                "batch_id": batch_id,
                "filename": file.filename,
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "invalid_rows": invalid_rows,
                "high_risk_count": high_risk,
                "medium_risk_count": medium_risk,
                "low_risk_count": low_risk,
                "average_probability": round(avg_prob, 3),
                "revenue_at_risk": round(rev_at_risk, 2)
            }), 200
            
        except Exception as e:
            return jsonify({"error": "Failed to process bulk upload.", "details": str(e)}), 400
            
    return render_template('bulk_upload.html')

@app.route('/export-batch/<batch_id>', methods=['GET'])
@login_required
def export_batch(batch_id):
    """Export predictions for a specific batch, scoped to company."""
    try:
        from flask import make_response
        import io
        company_id = session['company_id']
        predictions = db.get_predictions_by_batch(company_id, batch_id)
        if not predictions:
            return jsonify({"error": "No data found for this batch"}), 404

        df = pd.DataFrame(predictions)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        output = make_response(csv_buffer.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=batch_{batch_id}_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        return jsonify({"error": "Failed to export batch CSV", "details": str(e)}), 500

@app.route('/api/dashboard-data', methods=['GET'])
@login_required
def dashboard_data():
    """Return JSON metrics scoped to the logged-in company."""
    try:
        company_id = session['company_id']
        stats = db.get_advanced_stats(company_id)
        predictions = db.get_all_predictions_with_priority(company_id)
        alerts = db.get_alerts(company_id)
        return jsonify({
            "stats": stats,
            "recent_predictions": predictions,
            "alerts": alerts
        })
    except Exception as e:
        print(f"Dashboard data error: {e}")
        return jsonify({"error": "Failed to load dashboard data"}), 500

@app.route('/export', methods=['GET'])
@login_required
def export_csv():
    """Export all company predictions as CSV."""
    try:
        from flask import make_response
        import io
        company_id = session['company_id']
        predictions = db.get_all_predictions_with_priority(company_id)
        if not predictions:
            return jsonify({"error": "No data to export yet"}), 404

        df = pd.DataFrame(predictions)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        output = make_response(csv_buffer.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=predictions_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({"error": "CSV export failed", "details": str(e)}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict churn risk based on incoming e-commerce features."""
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Please ensure pipeline.pkl exists."}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        features = {
            'recency': data.get('recency', 0.0),
            'frequency': data.get('frequency', 0.0),
            'monetary': data.get('monetary', 0.0),
            'average_order_value': data.get('average_order_value', 0.0),
            'category_diversity': data.get('category_diversity', 0.0),
            'average_review_score': data.get('average_review_score', 0.0),
            'payment_count': data.get('payment_count', 0.0),
            'delivery_delay': data.get('delivery_delay', 0.0),
            'inactivity_days': data.get('inactivity_days', 0.0)
        }
        
        df_input = pd.DataFrame([features])
        prediction_array = pipeline.predict(df_input)
        prediction = int(prediction_array[0])
        
        probability_array = pipeline.predict_proba(df_input)
        probability = float(probability_array[0][1])  
        
        if probability < 0.3:
            risk_level = "Low"
        elif 0.3 <= probability <= 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
            
        customer_id = f"CUST_{str(uuid.uuid4())[:8].upper()}"
        company_id = session['company_id']
        
        db.insert_prediction(
            company_id=company_id,
            customer_id=customer_id,
            data=features,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
        
        insight = "Stable behavior"
        action = "Standard engagement"
        
        if features['inactivity_days'] > 90:
            insight = "Customer inactive for long period"
            action = "Send re-engagement offer"
        elif features['monetary'] > 500 and probability > 0.7:
            insight = "High value customer at risk"
            action = "Direct VIP outreach"
        elif features['average_review_score'] < 3.0:
            insight = "Bad experience risk"
            action = "Customer support follow-up"
        
        return jsonify({
            "customer_id": customer_id,
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "insight": insight,
            "recommended_action": action
        }), 200

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process prediction.", "details": str(e)}), 400

@app.route('/api/academic/classification', methods=['GET'])
def academic_classification():
    """Return academic classification metrics for comparison."""
    metrics_path = os.path.join('models', 'classification_metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics), 200
    else:
        return jsonify({"error": "Metrics not found. Please train academic models first."}), 404

# ---------------------------------------------------------
# Deletion & Reset Routes
# ---------------------------------------------------------

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
@login_required
def delete_prediction(prediction_id):
    """Delete a single prediction, accessible only by owning company."""
    company_id = session['company_id']
    success = db.delete_prediction(company_id, prediction_id)
    if success:
        return jsonify({"status": "success", "message": "Prediction deleted."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to delete or not authorized."}), 400

@app.route('/api/batch/<batch_id>', methods=['DELETE'])
@login_required
def delete_batch(batch_id):
    """Delete a whole batch, accessible only by owning company."""
    company_id = session['company_id']
    success = db.delete_batch(company_id, batch_id)
    if success:
        return jsonify({"status": "success", "message": "Batch deleted."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to delete batch or not authorized."}), 400

@app.route('/api/reset', methods=['DELETE'])
@login_required
def reset_data():
    """Reset all predictions for the logged-in company."""
    company_id = session['company_id']
    success = db.reset_company_data(company_id)
    if success:
        return jsonify({"status": "success", "message": "All company data reset successfully."}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to reset data."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/api/academic/regression')
def academic_regression():
    metrics_file = os.path.join('models', 'regression', 'regression_metadata.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Regression metrics not found. Run regression script first."}), 404

@app.route('/api/academic/clustering')
def academic_clustering():
    metrics_file = os.path.join('models', 'clustering', 'clustering_metadata.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Clustering metadata not found. Run clustering script first."}), 404

@app.route('/api/academic/association-rules')
def academic_association_rules():
    meta_file = os.path.join('data', 'association_rules', 'association_rules_metadata.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Association rules metadata not found. Run association rules script first."}), 404

@app.route('/api/academic/overview')
def academic_overview():
    summary_file = os.path.join('data', 'olist_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({
        "orders": 99441, "customers": 99441, "products": 32951,
        "items": 112650, "reviews": 99224, "payments": 103886,
        "date_min": "2016-09-04", "date_max": "2018-10-17"
    })

@app.route('/academic')
@app.route('/academic/<section>')
def academic_dashboard(section='overview'):
    """Academic Analytics Dashboard page for ML, DM, and OLAP review."""
    return render_template('academic_dashboard.html', initial_section=section)

@app.route('/api/academic/olap')
def academic_olap():
    meta_file = os.path.join('data', 'olap', 'olap_metadata.json')
    if not os.path.exists(meta_file):
        return jsonify({"error": "OLAP metadata not found. Run OLAP script first."}), 404

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    operation = request.args.get('operation')
    state_param = request.args.get('state')
    year_param = request.args.get('year')
    category_param = request.args.get('category')

    # Handle explicit operations or queries
    if operation == 'rollup':
        yearly_csv = os.path.join('data', 'olap', 'rollup_results_yearly.csv')
        records = []
        if os.path.exists(yearly_csv):
            records = pd.read_csv(yearly_csv).to_dict(orient='records')
        return jsonify({
            "operation": "rollup",
            "rollup": {
                **meta.get('operations', {}).get('rollup', {}),
                "data": records
            }
        })

    elif operation == 'drilldown':
        drill_csv = os.path.join('data', 'olap', 'drilldown_results.csv')
        records = []
        if os.path.exists(drill_csv):
            df_drill = pd.read_csv(drill_csv)
            if year_param:
                try:
                    df_drill = df_drill[df_drill['year'] == int(year_param)]
                except ValueError:
                    pass
            records = df_drill.to_dict(orient='records')
        return jsonify({
            "operation": "drilldown",
            "drilldown": {
                **meta.get('operations', {}).get('drilldown', {}),
                "data": records
            }
        })

    elif operation == 'slice':
        target_state = state_param.strip().upper() if state_param else 'SP'
        summary_csv = os.path.join('data', 'olap', 'olap_summary.csv')
        slice_records = []
        total_orders = 0
        total_sales = 0.0
        if os.path.exists(summary_csv):
            try:
                from olap_engine import slice_data
                df_summary = pd.read_csv(summary_csv)
                df_sliced = slice_data(df_summary, 'state', target_state)
                # Aggregate by year and category or return top records
                agg = df_sliced.groupby(['year', 'state']).agg({
                    'order_count': 'sum',
                    'customer_count': 'sum',
                    'total_sales': 'sum'
                }).reset_index()
                slice_records = agg.to_dict(orient='records')
                total_orders = int(df_sliced['order_count'].sum())
                total_sales = float(df_sliced['total_sales'].sum())
            except Exception as e:
                pass

        if not slice_records and os.path.exists(os.path.join('data', 'olap', 'slice_results.csv')):
            slice_records = pd.read_csv(os.path.join('data', 'olap', 'slice_results.csv')).to_dict(orient='records')
            total_orders = meta.get('operations', {}).get('slice', {}).get('orders', 41125)

        return jsonify({
            "operation": "slice",
            "slice": {
                "dimension": "customer_state",
                "value": target_state,
                "orders": total_orders,
                "total_sales": total_sales,
                "data": slice_records
            }
        })

    elif operation == 'dice':
        summary_csv = os.path.join('data', 'olap', 'olap_summary.csv')
        dice_records = []
        total_orders = 0
        if os.path.exists(summary_csv):
            try:
                from olap_engine import dice_data
                df_summary = pd.read_csv(summary_csv)
                states = [s.strip().upper() for s in state_param.split(',')] if state_param else ['SP', 'RJ', 'MG']
                years = [int(y.strip()) for y in year_param.split(',')] if year_param else [2017, 2018]
                cats = [c.strip() for c in category_param.split(',')] if category_param else ['bed_bath_table', 'health_beauty']
                filters = {'state': states, 'year': years, 'category': cats}
                df_diced = dice_data(df_summary, filters)
                dice_records = df_diced.to_dict(orient='records')
                total_orders = int(df_diced['order_count'].sum())
            except Exception as e:
                pass

        if not dice_records and os.path.exists(os.path.join('data', 'olap', 'dice_results.csv')):
            dice_records = pd.read_csv(os.path.join('data', 'olap', 'dice_results.csv')).to_dict(orient='records')
            total_orders = meta.get('operations', {}).get('dice', {}).get('orders', 12597)

        return jsonify({
            "operation": "dice",
            "dice": {
                "filters": {
                    "customer_state": states if state_param else ['SP', 'RJ', 'MG'],
                    "year": years if year_param else [2017, 2018],
                    "product_category": cats if category_param else ['bed_bath_table', 'health_beauty']
                },
                "orders": total_orders,
                "data": dice_records
            }
        })

    elif operation == 'pivot':
        row_dim = request.args.get('row_dim', 'state')
        col_dim = request.args.get('col_dim', 'year')
        measure = request.args.get('measure', 'total_sales')
        summary_csv = os.path.join('data', 'olap', 'olap_summary.csv')
        pivot_records = []
        col_names = []
        if os.path.exists(summary_csv):
            try:
                from olap_engine import pivot_data
                df_summary = pd.read_csv(summary_csv)
                pt = pivot_data(df_summary, row_dim=row_dim, col_dim=col_dim, measure=measure)
                pt['Total'] = pt.sum(axis=1)
                pt = pt.sort_values(by='Total', ascending=False)
                col_names = [str(c) for c in pt.columns if c != 'Total']
                pt_reset = pt.reset_index()
                pt_reset.columns = [str(c) for c in pt_reset.columns]
                pivot_records = pt_reset.to_dict(orient='records')
            except Exception as e:
                pass

        return jsonify({
            "operation": "pivot",
            "pivot": {
                "row_dim": row_dim,
                "col_dim": col_dim,
                "measure": measure,
                "columns": col_names,
                "data": pivot_records
            }
        })

    if operation and operation in meta.get('operations', {}):
        return jsonify({operation: meta['operations'][operation]})
    return jsonify(meta)

# ---------------------------------------------------------
# Schema Explorer API — Live Dataset Introspection
# ---------------------------------------------------------
@app.route('/api/academic/schema', methods=['GET'])
def academic_schema():
    """Introspect actual Olist CSV files and return live schema metadata."""
    RAW_PATH = os.path.join('data', 'olist_raw')

    # Map of logical table names to CSV files
    TABLE_MAP = {
        'olist_orders':       'olist_orders_dataset.csv',
        'olist_customers':    'olist_customers_dataset.csv',
        'olist_order_items':  'olist_order_items_dataset.csv',
        'olist_products':     'olist_products_dataset.csv',
        'olist_sellers':      'olist_sellers_dataset.csv',
        'olist_payments':     'olist_order_payments_dataset.csv',
        'olist_reviews':      'olist_order_reviews_dataset.csv',
        'olist_geolocation':  'olist_geolocation_dataset.csv',
        'category_translation': 'product_category_name_translation.csv',
    }

    # Define real foreign-key relationships between tables
    RELATIONSHIPS = [
        {'from': 'olist_orders',      'to': 'olist_customers',    'key': 'customer_id',  'type': 'many-to-one'},
        {'from': 'olist_order_items',  'to': 'olist_orders',       'key': 'order_id',     'type': 'many-to-one'},
        {'from': 'olist_order_items',  'to': 'olist_products',     'key': 'product_id',   'type': 'many-to-one'},
        {'from': 'olist_order_items',  'to': 'olist_sellers',      'key': 'seller_id',    'type': 'many-to-one'},
        {'from': 'olist_payments',     'to': 'olist_orders',       'key': 'order_id',     'type': 'many-to-one'},
        {'from': 'olist_reviews',      'to': 'olist_orders',       'key': 'order_id',     'type': 'many-to-one'},
        {'from': 'olist_products',     'to': 'category_translation', 'key': 'product_category_name', 'type': 'many-to-one'},
    ]

    # DW schema mappings — which raw tables map to which DW role
    DW_ROLES = {
        'olist_orders':       {'role': 'fact', 'dw_name': 'FACT_ORDERS'},
        'olist_order_items':  {'role': 'fact', 'dw_name': 'FACT_ORDER_ITEMS'},
        'olist_payments':     {'role': 'fact', 'dw_name': 'FACT_PAYMENTS'},
        'olist_reviews':      {'role': 'fact', 'dw_name': 'FACT_REVIEWS'},
        'olist_customers':    {'role': 'dimension', 'dw_name': 'DIM_CUSTOMER'},
        'olist_products':     {'role': 'dimension', 'dw_name': 'DIM_PRODUCT'},
        'olist_sellers':      {'role': 'dimension', 'dw_name': 'DIM_SELLER'},
        'olist_geolocation':  {'role': 'dimension', 'dw_name': 'DIM_GEOLOCATION'},
        'category_translation': {'role': 'dimension', 'dw_name': 'DIM_CATEGORY'},
    }

    tables = {}
    for table_name, csv_file in TABLE_MAP.items():
        filepath = os.path.join(RAW_PATH, csv_file)
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, nrows=5)
            df_full_info = pd.read_csv(filepath)
            total_rows = len(df_full_info)

            columns = []
            for col in df_full_info.columns:
                col_dtype = str(df_full_info[col].dtype)
                non_null = int(df_full_info[col].notna().sum())
                n_unique = int(df_full_info[col].nunique())
                sample_vals = df[col].dropna().head(3).astype(str).tolist()

                # Detect key type
                key_type = None
                if col.endswith('_id') or col == 'order_id':
                    key_type = 'PK' if col == df_full_info.columns[0] or n_unique == total_rows else 'FK'
                if col == 'product_category_name':
                    key_type = 'FK'

                columns.append({
                    'name': col,
                    'dtype': col_dtype,
                    'non_null': non_null,
                    'null_count': total_rows - non_null,
                    'null_pct': round(100 * (total_rows - non_null) / total_rows, 2) if total_rows > 0 else 0,
                    'unique': n_unique,
                    'sample': sample_vals,
                    'key_type': key_type,
                })

            dw_info = DW_ROLES.get(table_name, {})

            tables[table_name] = {
                'csv_file': csv_file,
                'total_rows': total_rows,
                'total_columns': len(df_full_info.columns),
                'columns': columns,
                'sample_data': df.head(3).fillna('').astype(str).to_dict(orient='records'),
                'dw_role': dw_info.get('role', 'unknown'),
                'dw_name': dw_info.get('dw_name', table_name.upper()),
            }
        except Exception as e:
            tables[table_name] = {'error': str(e), 'csv_file': csv_file}

    return jsonify({
        'tables': tables,
        'relationships': RELATIONSHIPS,
        'total_tables': len(tables),
        'source': 'LIVE — Introspected from actual Olist CSV files',
    })

@app.route('/api/academic/exploration', methods=['GET'])
def academic_exploration():
    """Real Data Exploration endpoint with pagination, descriptive stats, and null counts."""
    dataset_name = request.args.get('dataset', 'customers').lower()
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 15))

    file_map = {
        'customers': os.path.join('data', 'olist_raw', 'olist_customers_dataset.csv'),
        'orders': os.path.join('data', 'olist_raw', 'olist_orders_dataset.csv'),
        'items': os.path.join('data', 'olist_raw', 'olist_order_items_dataset.csv'),
        'payments': os.path.join('data', 'olist_raw', 'olist_order_payments_dataset.csv'),
        'reviews': os.path.join('data', 'olist_raw', 'olist_order_reviews_dataset.csv'),
        'products': os.path.join('data', 'olist_raw', 'olist_products_dataset.csv'),
        'churn': os.path.join('data', 'processed_ecommerce_data.csv'),
        'regression': os.path.join('data', 'olist_freight_regression_dataset.csv')
    }

    target_file = file_map.get(dataset_name)
    if not target_file or not os.path.exists(target_file):
        return jsonify({"error": f"Dataset '{dataset_name}' not found."}), 404

    try:
        # Load sample/data efficiently
        # Read total rows count fast
        with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
            total_rows = sum(1 for _ in f) - 1

        skip = (page - 1) * page_size
        df = pd.read_csv(target_file, skiprows=range(1, skip + 1) if skip > 0 else None, nrows=page_size)

        # Get summary info from a 2000-row sample for stats & missing counts
        df_sample = pd.read_csv(target_file, nrows=2000)

        # Missing values
        missing_stats = []
        for col in df_sample.columns:
            m_count = int(df_sample[col].isna().sum())
            missing_stats.append({
                "column": col,
                "missing_count": m_count,
                "missing_pct": round((m_count / len(df_sample)) * 100, 2),
                "dtype": str(df_sample[col].dtype)
            })

        # Numerical descriptive stats
        numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
        desc_stats = []
        for col in numeric_cols:
            s = df_sample[col].dropna()
            if len(s) > 0:
                desc_stats.append({
                    "column": col,
                    "count": int(s.count()),
                    "mean": round(float(s.mean()), 3),
                    "median": round(float(s.median()), 3),
                    "std": round(float(s.std()), 3),
                    "min": round(float(s.min()), 3),
                    "max": round(float(s.max()), 3)
                })

        # Feature-specific distribution if requested
        feature_name = request.args.get('feature')
        feature_dist = None
        if feature_name and feature_name in df_sample.columns:
            f_series = df_sample[feature_name].dropna()
            is_num = pd.api.types.is_numeric_dtype(f_series)
            if is_num and len(f_series) > 0:
                counts, bin_edges = np.histogram(f_series, bins=min(15, max(5, f_series.nunique())))
                labels = [f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}" for i in range(len(counts))]
                q1, med, q3 = float(np.percentile(f_series, 25)), float(np.percentile(f_series, 50)), float(np.percentile(f_series, 75))
                feature_dist = {
                    "type": "numeric",
                    "column": feature_name,
                    "histogram": {"labels": labels, "counts": counts.tolist()},
                    "box": {
                        "min": round(float(f_series.min()), 2),
                        "q1": round(q1, 2),
                        "median": round(med, 2),
                        "q3": round(q3, 2),
                        "max": round(float(f_series.max()), 2)
                    }
                }
            else:
                top_v = f_series.value_counts().head(12)
                feature_dist = {
                    "type": "categorical",
                    "column": feature_name,
                    "categories": top_v.index.astype(str).tolist(),
                    "counts": top_v.values.tolist()
                }

        # Correlation Matrix for top numeric columns
        corr_data = None
        if len(numeric_cols) >= 2:
            num_sub = numeric_cols[:8]
            corr_df = df_sample[num_sub].corr().round(2).fillna(0)
            corr_data = {
                "columns": num_sub,
                "values": corr_df.values.tolist()
            }

        return jsonify({
            "dataset": dataset_name,
            "filename": os.path.basename(target_file),
            "total_rows": total_rows,
            "columns": list(df.columns),
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total_rows + page_size - 1) // page_size),
            "data": df.fillna("").to_dict(orient='records'),
            "missing_stats": missing_stats,
            "desc_stats": desc_stats,
            "feature_distribution": feature_dist,
            "correlation_matrix": corr_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Dynamic Academic Analysis API
# ---------------------------------------------------------

@app.route('/api/academic/run', methods=['POST'])
def academic_run():
    """
    Unified dynamic analysis endpoint.
    Accepts: { analysis_type, model, parameters }
    Returns: computed metrics, charts data, interpretation, query_repr
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No JSON payload provided'}), 400

        analysis_type = str(data.get('analysis_type', '')).strip().lower()
        model         = str(data.get('model', '')).strip().lower()
        params        = data.get('parameters', {})

        if not analysis_type or not model:
            return jsonify({'success': False, 'error': 'analysis_type and model are required'}), 400

        from dynamic_runner import run_analysis
        result = run_analysis(analysis_type, model, params)

        if not result.get('success'):
            return jsonify(result), 422

        return jsonify(result), 200

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/academic/states', methods=['GET'])
def academic_states():
    """Return list of unique states from OLAP cube."""
    try:
        olap_csv = os.path.join('data', 'olap', 'olap_summary.csv')
        if not os.path.exists(olap_csv):
            return jsonify([])
        df = pd.read_csv(olap_csv, usecols=['state'])
        states = sorted(df['state'].dropna().unique().tolist())
        return jsonify(states)
    except Exception as e:
        return jsonify([])


@app.route('/api/academic/categories', methods=['GET'])
def academic_categories():
    """Return list of unique product categories from OLAP cube."""
    try:
        olap_csv = os.path.join('data', 'olap', 'olap_summary.csv')
        if not os.path.exists(olap_csv):
            return jsonify([])
        df = pd.read_csv(olap_csv, usecols=['category'])
        cats = sorted(df['category'].dropna().unique().tolist())
        return jsonify(cats)
    except Exception as e:
        return jsonify([])


if __name__ == '__main__':
    app.run(debug=True, port=5000)




