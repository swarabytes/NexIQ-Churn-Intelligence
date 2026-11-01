# NexIQ — Last-Minute Viva Rapid Revision Cheat Sheet
*Read this 10 minutes before entering your viva examination room.*

---

## ⚡ 1. The Spoken Pitches

### 30-Second Opening
> "Good morning, examiners. **NexIQ** is a production-grade E-Commerce Customer Intelligence and Academic Data Mining platform. It solves silent customer attrition by predicting churn risk on physical transaction holdouts with a **0.963 F1-score and 100% recall**. Beyond operational prediction, NexIQ functions as a research laboratory implementing core Data Warehousing and Data Mining operations—including star schemas, multidimensional OLAP cubes, agglomerative clustering dendrograms, and Apriori association rules—backed by multi-tenant SQLite isolation and real-time Chart.js visualizations."

### 60-Second Deep Pitch
> "Acquiring a new e-commerce customer costs 5 to 7 times more than retaining an existing one. In non-contractual retail like Olist, customers do not explicitly cancel; they silently stop purchasing. NexIQ addresses this through a dual-engine architecture:
> 1. **Operational Churn Engine**: Ingests 9 engineered RFM and behavioral signals (recency, inactivity days, review scores, delivery delays) to compute calibrated churn probabilities, expected revenue-at-risk, and prescriptive retention actions for single and bulk batch uploads.
> 2. **Academic Analytics Lab**: Demonstrates complete DWDM rigor—benchmarking Logistic Regression, Random Forest, and XGBoost on physical Olist transactions, slicing and dicing OLAP cubes, discovering frequent purchase baskets via Apriori, and performing bottom-up hierarchical clustering with dendrogram cuts.
> The system runs entirely on real physical data, guarantees zero data leakage through Scikit-Learn pipelines, and supports 1-click dynamic production model swaps without restarting the Flask server."

---

## 📊 2. Key Numbers & Verified Dataset Statistics (Memorize Exactly)

| Metric | Exact Value | Context / Meaning |
| :--- | :--- | :--- |
| **Original Olist Dataset** | 99,441 orders | Brazilian E-Commerce transactions (Sept 2016 – Oct 2018) |
| **Unique Customers** | 96,096 | Across 27 Brazilian states |
| **Product Catalog** | 32,951 products | 73 verified retail categories |
| **Evaluated Cohort** | 1,000 customers | Physical stratified cohort for web-scale experimentation |
| **Holdout Split Ratio** | 80:20 Stratified | 800 Training rows / 200 Test holdout rows (`random_state=42`) |
| **Target Distribution** | 741 Retained (74.1%) \| 259 Churned (25.9%) | Natural e-commerce class imbalance (~3:1 ratio) |
| **Holdout Test Set** | 148 Retained \| 52 Churned | Physical 200-sample test set for all 3 algorithms |
| **Number of Features** | 9 domain features | Recency, Frequency, Monetary, AOV, Diversity, Review, Payments, Delay, Inactivity |

---

## 🏆 3. Top 3 Machine Learning Benchmark Results

| Model Architecture | Rank | F1-Score | Recall | Precision | ROC-AUC | PR-AUC | Accuracy | Train Time | Inference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 👑 **#1** | **0.9630** | **1.0000** | 0.9286 | **0.9992** | **0.9978** | **98.0%** | **9.6 ms** | **1.02 ms** |
| **XGBoost Classifier** | **#2** | 0.9307 | 0.9038 | **0.9592** | 0.9862 | 0.9698 | 96.5% | 75.4 ms | 3.52 ms |
| **Random Forest (100 Trees)** | **#3** | 0.8776 | 0.8269 | 0.9348 | 0.9676 | 0.9430 | 94.0% | 554.3 ms | 41.9 ms |

* **Why Logistic Regression Won**: Monotonic linear relationship between inactivity/recency and churn log-odds. With 800 training rows, deep trees suffer from partition step-variance, whereas regularized logistic sigmoid forms a smooth, leak-free boundary catching 100% of churners (FN = 0).
* **Holdout Confusion Matrix (LR)**: `TN: 144 | FP: 4 | FN: 0 | TP: 52` (Total Errors: 4 out of 200).

---

## 📐 4. Essential Formulas Cheat Sheet

1. **Sigmoid**: $P = \sigma(z) = rac{1}{1 + e^{-z}}$ where $z = eta_0 + \sum eta_j x_j$
2. **StandardScaler**: $z = rac{x - \mu}{\sigma}$ (fits solely on training set)
3. **Accuracy**: $rac{TP + TN}{TP + TN + FP + FN} = rac{52 + 144}{200} = 0.980$
4. **Precision**: $rac{TP}{TP + FP} = rac{52}{52 + 4} = 0.9286$
5. **Recall (Sensitivity)**: $rac{TP}{TP + FN} = rac{52}{52 + 0} = 1.000$ (Catch-rate of churners)
6. **Specificity (TNR)**: $rac{TN}{TN + FP} = rac{144}{144 + 4} = 0.9730$
7. **F1-Score**: $2 	imes rac{	ext{Precision} 	imes 	ext{Recall}}{	ext{Precision} + 	ext{Recall}} = 2 	imes rac{0.9286 	imes 1.0}{1.9286} = 0.9630$
8. **Revenue at Risk**: $\sum_{i=1}^{N} (	ext{Monetary}_i 	imes 	ext{Probability}_i)$
9. **Customer Health Score**: $	ext{round}((1.0 - P) 	imes 100 	imes rac{	ext{ReviewScore}}{5.0})$
10. **Apriori Support**: $rac{	ext{Orders with A and B}}{	ext{Total Orders}}$
11. **Apriori Confidence**: $rac{	ext{Support}(A \cup B)}{	ext{Support}(A)}$
12. **Apriori Lift**: $rac{	ext{Confidence}(A ightarrow B)}{	ext{Support}(B)} = rac{P(A \cap B)}{P(A) 	imes P(B)}$ (Lift $> 1.0 \implies$ positive co-purchase affinity)
13. **Cost-Sensitive Weighting**: $w_j = rac{N}{2 	imes N_j}$ (Churn weight $pprox 1.93$, Non-churn $pprox 0.67$)

---

## 🗄️ 5. Database & Architecture Fast Facts

* **Database Engine**: SQLite (`database.db`) with foreign keys enabled.
* **Core Tables**:
  * `companies`: `id` (PK), `company_name`, `email` (UNIQUE), `password_hash` (PBKDF2), `created_at`.
  * `predictions`: `id` (PK), `company_id` (FK), `customer_id`, 9 RFM metrics, `prediction`, `probability`, `risk_level`, `batch_id`, `health_score`, `recommendation`.
* **Multi-Tenancy Guard**: Every SQL query includes `WHERE company_id = ?` parameter binding.
* **Dynamic Reloader**: `reload_production_pipeline()` reloads `pipeline.pkl` into global memory without server downtime.
* **Theme System**: CSS custom properties with `localStorage` persistence and Zero-FOUC inline `<head>` script.

---

## 🎯 6. 25 Rapid-Fire Viva Questions (Q&A Flashcards)

1. **What is NexIQ?** Enterprise customer churn prediction & academic data mining platform.
2. **What problem does it solve?** Silent customer attrition in non-contractual e-commerce.
3. **What is the champion model?** Logistic Regression with L2 Regularization & Balanced Class Weighting.
4. **What is the champion F1-score and Recall?** F1: 0.9630, Recall: 1.0000 (100%).
5. **How many features are used?** 9 domain-engineered RFM and behavioral features.
6. **Why not PCA?** PCA destroys domain interpretability; business leaders cannot act on eigenvectors.
7. **How do you prevent data leakage?** Preprocessing transformers are wrapped in Scikit-Learn Pipelines fitted solely on `X_train`.
8. **What was your train-test split?** 80:20 Stratified Split (800 train / 200 test holdout, `random_state=42`).
9. **Why is accuracy misleading here?** 74% of customers don't churn; a dummy model predicting 0 gets 74% accuracy while catching zero churners.
10. **What is the difference between ROC-AUC and PR-AUC?** ROC-AUC plots TPR vs FPR (can mask minority errors); PR-AUC plots Precision vs Recall (strict under class imbalance).
11. **What is Revenue at Risk?** $\sum (	ext{Monetary} 	imes 	ext{Probability})$ — expected financial loss if high-risk customers churn.
12. **What is the Critical priority rule?** Probability $> 0.80$ AND Monetary $> \$500.00$.
13. **What database is used?** SQLite with foreign keys and multi-tenant `company_id` scoping.
14. **How are passwords stored?** Cryptographically salted PBKDF2 hashes via Werkzeug.
15. **What is a Star Schema?** Central fact table with numeric measures joined to denormalized dimension tables.
16. **What is a Snowflake Schema?** Normalized dimension tables broken down into 3NF sub-dimensions.
17. **What are the 4 OLAP operations?** Roll-Up, Drill-Down, Slice, Dice (plus Pivot).
18. **How was K=4 selected for K-Means?** Elbow Method inflection point + managerial interpretability across 4 RFM personas.
19. **What does the horizontal dendrogram cut represent?** Cophenetic distance threshold determining the number of formed clusters.
20. **What does an Apriori Lift of 2.1 mean?** Customers buying item A are 2.1 times more likely to purchase item B than random chance.
21. **Why was association rule data sparse?** Only 0.75% of Olist orders contained multiple product categories (realistic retail sparsity).
22. **What happens during single prediction?** Input is validated, scaled via `pipeline.pkl`, scored via `.predict_proba()`, and saved to SQLite.
23. **What is the maximum bulk upload size?** 5,000 rows per batch with atomic SQLite commits.
24. **How is Zero-FOUC achieved?** An inline JavaScript snippet in `<head>` applies the theme attribute before the DOM paints.
25. **How is the production model updated dynamically?** `set_production_model()` writes candidate to `pipeline.pkl`, and `reload_production_pipeline()` reloads it in memory.
