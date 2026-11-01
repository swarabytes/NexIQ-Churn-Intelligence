# NexIQ — Tricky & High-Pressure Examiner Questions Master Guide
*Strategic defenses and technical proofs for college viva evaluations.*

---

### Question 1: "Your model achieves 98.0% accuracy and 100% recall. Isn't that clearly an overfitted model on a small dataset?"
**Examiner's Intent:** Testing whether you understand the difference between memorization (overfitting) and true generalization, and whether you evaluated on an independent holdout.

**Bulletproof Spoken Answer:**
> "No, professor. A high accuracy alone can be suspicious, but our evaluation proves genuine generalization for three mathematical reasons:
> 1. **Independent Holdout Validation**: The 98% accuracy and 100% recall were computed on an untouched holdout test set of 200 samples that was never used during training.
> 2. **Multi-Metric Consistency**: We evaluated Precision (92.9%), F1-Score (0.963), ROC-AUC (0.9992), and PR-AUC (0.9978)—all metrics align consistently.
> 3. **Physical Monotonicity**: In e-commerce, features like `inactivity_days` and `recency` have near-deterministic separation with churn. A customer with 300 days of dormancy and a 1-star review has an almost 100% probability of having churned. Logistic Regression with L2 regularization penalizes extreme weights, preventing boundary over-partitioning."

---

### Question 2: "Random Forest and XGBoost are state-of-the-art non-linear algorithms. Why did simple Logistic Regression beat them in your benchmark?"
**Examiner's Intent:** Checking if you blindly believe 'complex algorithms are always better' or if you understand bias-variance tradeoffs.

**Bulletproof Spoken Answer:**
> "This outcome is mathematically sound and rooted in two fundamental principles:
> 1. **Linear Log-Odds Monotonicity**: The relationship between normalized RFM features and churn probability is smoothly monotonic. As inactivity and recency rise, churn odds increase smoothly. A logistic sigmoid boundary fits this perfectly.
> 2. **Sample Size vs Partition Variance**: Deep tree ensembles (100 trees in Random Forest, gradient-boosted stumps in XGBoost) partition continuous feature spaces into rigid step-functions. With 800 training rows, this creates small partition pockets that caused 5 to 9 False Negatives. Logistic Regression achieved 0 False Negatives (100% Recall), making it the superior production classifier."

---

### Question 3: "Why didn't you use Deep Learning or an Artificial Neural Network?"
**Examiner's Intent:** Testing your engineering pragmatism versus buzzword chasing.

**Bulletproof Spoken Answer:**
> "Using deep learning on a 1,000-row tabular dataset with 9 engineered features would be poor engineering. Neural networks require tens of thousands of rows to generalize without severe overfitting and require extensive hyperparameter tuning. 
> Furthermore, deep learning functions as a black box, destroying regulatory explainability. Logistic Regression trains in 9.6 milliseconds, runs inference in 1 millisecond, and provides transparent mathematical feature weights that executives can audit."

---

### Question 4: "Why did you use domain-driven feature engineering instead of Principal Component Analysis (PCA)?"
**Examiner's Intent:** Testing your understanding of unsupervised dimensionality reduction versus domain interpretability.

**Bulletproof Spoken Answer:**
> "PCA projects variables onto orthogonal eigenvectors to maximize variance, but it completely destroys business interpretability. An e-commerce retention director cannot act on 'Principal Component 1 has a value of +2.4'. 
> With domain-driven features, we know the exact physical root cause: a customer has a 14-day delivery delay and a 1-star review score. This allows NexIQ to trigger specific prescriptive actions (such as a delivery refund voucher or direct VIP outreach) which would be impossible with PCA."

---

### Question 5: "How did you prevent Data Leakage? Prove to me that your pipeline is clean."
**Examiner's Intent:** Trying to catch you on a common flaw where `StandardScaler` or `SMOTE` is fitted on the entire dataset before splitting.

**Bulletproof Spoken Answer:**
> "We strictly encapsulated preprocessing and modeling inside Scikit-Learn and Imbalanced-Learn Pipelines (`imblearn.pipeline.Pipeline`).
> When `pipeline.fit(X_train, y_train)` is executed:
> 1. `StandardScaler` calculates mean ($\mu$) and standard deviation ($\sigma$) solely on the 800 training rows.
> 2. `SMOTE` (when enabled) synthesizes minority points strictly within the 800 training rows.
> 3. During `pipeline.predict(X_test)`, the 200 holdout test samples are transformed using the training parameters. Zero test set statistics ever touch the training phase."

---

### Question 6: "Why is accuracy a dangerous metric for customer churn?"
**Examiner's Intent:** Testing your understanding of class imbalance.

**Bulletproof Spoken Answer:**
> "In our dataset, 74.1% of customers are non-churners. If an algorithm simply predicted 'Retained' (0) for every single customer without looking at any features, it would achieve 74.1% accuracy while catching zero churners. 
> Accuracy treats False Positives and False Negatives equally. In churn prediction, a False Negative (failing to detect a churning customer) causes permanent revenue loss. We therefore prioritize Recall, F1-Score, and PR-AUC over raw accuracy."

---

### Question 7: "What is the difference between ROC-AUC and PR-AUC, and which is more reliable here?"
**Examiner's Intent:** Advanced metric evaluation under class imbalance.

**Bulletproof Spoken Answer:**
> "ROC-AUC plots True Positive Rate versus False Positive Rate. Because False Positive Rate has True Negatives in the denominator ($FPR = rac{FP}{TN + FP}$), a large negative class (148 non-churners) keeps the denominator large, which can make the ROC curve look deceptively optimistic.
> PR-AUC (Precision-Recall Area under Curve) plots Precision versus Recall, focusing strictly on the minority class. On imbalanced datasets, PR-AUC provides a much stricter, higher-fidelity measure of model quality. Logistic Regression achieved an exceptional 0.9978 PR-AUC."

---

### Question 8: "What happens if a brand-new customer makes their first order? How does NexIQ predict their churn?"
**Examiner's Intent:** Testing your knowledge of the 'Cold-Start Problem'.

**Bulletproof Spoken Answer:**
> "For a customer with one completed order, their attributes are $Recency = 0$, $Frequency = 1$, and $Inactivity = 0$. When passed through `pipeline.pkl`, the model evaluates these low values and predicts a very low churn probability ($pprox 0.02\%$), categorizing them as Low Risk.
> However, for a user who has just registered an account but placed zero orders, predictive churn modeling is technically not applicable because RFM requires transaction history. Such users enter an onboarding marketing funnel rather than a churn triage queue."

---

### Question 9: "Why did you use SQLite instead of a production database like PostgreSQL?"
**Examiner's Intent:** Architecture justification and trade-offs.

**Bulletproof Spoken Answer:**
> "SQLite is a zero-configuration, serverless, self-contained ACID-compliant SQL database engine. For our prototype and viva demonstration, SQLite eliminates external daemon dependencies while delivering sub-millisecond query execution directly in memory and on disk.
> Because our SQL queries in `db.py` use clean parameterized syntax and strict foreign keys, migrating to PostgreSQL for enterprise cloud deployment only requires updating the connection URI in SQLAlchemy or psycopg2 without rewriting table schemas or route handlers."

---

### Question 10: "How does your multi-tenant architecture prevent Company A from seeing Company B's data?"
**Examiner's Intent:** Information security and tenant isolation verification.

**Bulletproof Spoken Answer:**
> "Multi-tenant isolation is enforced at two distinct security layers:
> 1. **Cryptographic Session Scoping**: When a company logs in, their validated primary key `company_id` is stored in Flask's cryptographically signed HTTP-only session cookie.
> 2. **Mandatory Query Parameterization**: Every database function in `db.py` enforces `WHERE company_id = ?` parameter binding. Even if an attacker manually modifies a customer ID or prediction ID in an HTTP request, the backend SQL query checks `AND company_id = session['company_id']`, returning an authorization failure."

---

### Question 11: "Explain how 1-click model switching works without restarting the Flask server."
**Examiner's Intent:** Software engineering and operational persistence understanding.

**Bulletproof Spoken Answer:**
> "When an administrator selects a model in the Academic Lab:
> 1. `model_evaluator.set_production_model(model_key)` backs up the active model to `models/backup_pipeline.pkl`.
> 2. It copies the chosen candidate model (`models/candidate_<key>.pkl`) into the production root file `pipeline.pkl` and updates `production_model_meta.json`.
> 3. It immediately invokes `reload_production_pipeline()` in `app.py`. This function executes `joblib.load('pipeline.pkl')` and re-assigns the in-memory global `pipeline` variable. All subsequent `/predict` and `/bulk-upload` requests instantly use the new model without dropping HTTP connections."

---

### Question 12: "What is the difference between an OLTP database and an OLAP data warehouse?"
**Examiner's Intent:** DWDM fundamental concepts.

**Bulletproof Spoken Answer:**
> "OLTP (Online Transaction Processing) systems are normalized (3NF) to support fast, continuous, concurrent ACID writes, inserts, and updates with zero data redundancy (like our SQLite `predictions` table).
> OLAP (Online Analytical Processing) systems use denormalized multidimensional Star or Snowflake schemas optimized for complex aggregate read queries (SUM, AVG, GROUP BY) across millions of historical transactions across time, geography, and product hierarchies."

---

### Question 13: "What does the horizontal cut line on a Hierarchical Clustering Dendrogram represent?"
**Examiner's Intent:** Unsupervised learning and dendrogram interpretation.

**Bulletproof Spoken Answer:**
> "The horizontal cut line represents a cophenetic distance threshold. In agglomerative hierarchical clustering, clusters are merged bottom-up based on minimum variance (Ward's linkage).
> The height of each vertical node represents the dissimilarity between merged clusters. Drawing a horizontal line across the dendrogram at a chosen distance slices the tree; the number of vertical lines intersected by the cut line equals the exact number of formed customer clusters."

---

### Question 14: "Why was the Apriori association rule mining output sparse on the Olist dataset?"
**Examiner's Intent:** Did you run real data or mock data?

**Bulletproof Spoken Answer:**
> "This proves that our analysis was executed against actual physical transactions rather than synthetic mock baskets.
> In real-world Brazilian Olist retail, only 726 out of 96,824 valid orders (0.75%) contained products across multiple distinct product categories—most consumers purchased a single specific product per order. We had to calibrate minimum support to appropriate empirical thresholds to discover genuine co-purchases (e.g., `furniture_decor` $ightarrow$ `bed_bath_table` with a Lift of > 2.1)."

---

### Question 15: "What are the primary limitations of your project, and what would you build next?"
**Examiner's Intent:** Testing engineering self-awareness and maturity.

**Bulletproof Spoken Answer:**
> "Our primary limitations are:
> 1. **Temporal Static Snapshot**: Churn is currently evaluated at a static point in time rather than via continuous survival analysis (Cox Proportional Hazards).
> 2. **Write Concurrency**: SQLite utilizes database-level locking during write transactions, which is unsuitable for massive concurrent write streams.
> In our next phase, we would migrate to PostgreSQL with Amazon Aurora, implement continuous feature store streaming using Apache Kafka, and integrate SHAP (Shapley Additive Explanations) waterfall plots for individual prediction interpretability."
