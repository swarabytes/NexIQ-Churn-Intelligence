"""
Part 5B — Olist Churn Dataset Design Investigation
===================================================
Investigates:
  1. Multiple time-based prediction cutoffs (Rolling / Historical)
  2. Multiple cohort definitions (A, B30, B60, B90)
  3. Multiple future windows (90, 120, 180 days)
  4. Both churn and return framing
  5. Class balance and minority-class sufficiency

STOP CONDITIONS (enforced):
  - Does NOT create olist_customer_churn.csv
  - Does NOT train models
  - Does NOT apply SMOTE / oversampling / undersampling
  - Does NOT modify the existing application
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# 0. LOAD RAW DATA
# ---------------------------------------------------------------------------
RAW_PATH = 'data/olist_raw'

print("=" * 80)
print("PART 5B — OLIST CHURN DATASET DESIGN INVESTIGATION")
print("=" * 80)
print()

orders    = pd.read_csv(os.path.join(RAW_PATH, 'olist_orders_dataset.csv'))
customers = pd.read_csv(os.path.join(RAW_PATH, 'olist_customers_dataset.csv'))

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Remove cancelled orders only
valid_orders = orders[orders['order_status'] != 'canceled'].copy()
merged = valid_orders.merge(customers, on='customer_id')

earliest_date = merged['order_purchase_timestamp'].min()
latest_date   = merged['order_purchase_timestamp'].max()
total_span    = (latest_date - earliest_date).days

print(f"Dataset timeline")
print(f"  Earliest order : {earliest_date.date()}")
print(f"  Latest order   : {latest_date.date()}")
print(f"  Total span     : {total_span} days")
print(f"  Total orders   : {len(merged):,}")
print(f"  Unique customers (customer_unique_id): "
      f"{merged['customer_unique_id'].nunique():,}")
print()

# ---------------------------------------------------------------------------
# 1. CUTOFF STRATEGY EXPLANATION
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 1 — CUTOFF STRATEGIES")
print("=" * 80)
print("""
STRATEGY A: SINGLE LATE CUTOFF (previous approach)
   One cutoff placed 180 days before the last recorded order.
   Cohort = customers with >=2 orders before that single date.
   Future window = that single date onward.
   Limitation: Only 421 customers qualify; very small dataset.

STRATEGY B: ROLLING QUARTERLY CUTOFFS (proposed here)
   Multiple cutoffs spaced quarterly across the dataset timeline.
   Each cutoff defines an independent observation snapshot.
   For each cutoff:
     - Historical window = all data BEFORE the cutoff
     - Feature computation uses ONLY historical window
     - Future window = [cutoff, cutoff + W] days
     - Label: did the customer purchase in the future window?
   Cutoff dates are chosen so that the full future window fits within
   the dataset timeline (i.e., cutoff + W <= latest_date).

REPEATED-CUSTOMER HANDLING:
   A customer appearing at multiple cutoffs creates SEPARATE OBSERVATIONS,
   each with independently computed features and labels.
   This is methodologically valid because:
     (a) Each observation reflects a genuinely different behavioral state
         at a different point in time.
     (b) Historical features are recomputed fresh from scratch for each
         cutoff with no forward-looking information.
     (c) This approach is standard in survival analysis and customer
         lifetime value research ("panel data" structure).
   HOWEVER: the resulting dataset is NOT i.i.d. (observations from the
   same customer are correlated). We will flag this clearly and recommend
   either (i) using the LAST eligible snapshot per customer, or
   (ii) using a grouped train/test split that prevents a customer from
   appearing in both sets simultaneously.
   For this investigation we report both "total observations" and
   "unique customers" so the user can decide.
""")

# ---------------------------------------------------------------------------
# 2. GENERATE ROLLING CUTOFFS
# ---------------------------------------------------------------------------
MAX_FUTURE_WINDOW   = 180  # days (maximum window we test)
MIN_HISTORY_FOR_B90 = 90   # minimum history days for B90 cohort to form

cutoffs = []
candidate = earliest_date + pd.Timedelta(days=MIN_HISTORY_FOR_B90)

while candidate + pd.Timedelta(days=MAX_FUTURE_WINDOW) <= latest_date:
    cutoffs.append(candidate)
    candidate += pd.Timedelta(days=90)

print("=" * 80)
print("SECTION 2 — GENERATED ROLLING CUTOFF DATES")
print("=" * 80)
print(f"  Number of cutoffs generated: {len(cutoffs)}")
for i, c in enumerate(cutoffs):
    remaining = (latest_date - c).days
    hist_days = int((c - earliest_date).days)
    print(f"  Cutoff {i+1:2d}: {c.date()}  "
          f"(history={hist_days}d, remaining={remaining}d)")

single_cutoff = latest_date - pd.Timedelta(days=180)
print(f"\n  [Reference] Single late cutoff (previous approach): "
      f"{single_cutoff.date()}")
print()

# ---------------------------------------------------------------------------
# 3. CORE EVALUATION FUNCTION
# ---------------------------------------------------------------------------
FUTURE_WINDOWS   = [90, 120, 180]
COHORT_SPAN_DAYS = {
    'A (>=2 orders)':          0,
    'B30 (>=2 orders, >=30d)': 30,
    'B60 (>=2 orders, >=60d)': 60,
    'B90 (>=2 orders, >=90d)': 90,
}
MIN_MINORITY_CLASS = 50


def build_observation(cutoff, future_window_days, cohort_min_span):
    window_end = cutoff + pd.Timedelta(days=future_window_days)
    if window_end > latest_date + pd.Timedelta(days=1):
        return None

    hist = merged[merged['order_purchase_timestamp'] < cutoff]
    if hist.empty:
        return None

    grp = hist.groupby('customer_unique_id')['order_purchase_timestamp']
    first_order = grp.min()
    last_order  = grp.max()
    order_count = grp.count()
    tenure_days = (last_order - first_order).dt.days

    mask = order_count >= 2
    if cohort_min_span > 0:
        mask = mask & (tenure_days >= cohort_min_span)

    qualifying = order_count[mask].index
    if len(qualifying) == 0:
        return None

    future = merged[
        (merged['order_purchase_timestamp'] >= cutoff) &
        (merged['order_purchase_timestamp'] <  window_end)
    ]
    future_customers = set(future['customer_unique_id'].unique())

    obs = pd.DataFrame({'customer_unique_id': qualifying})
    obs['returned']    = obs['customer_unique_id'].isin(future_customers).astype(int)
    obs['churn']       = 1 - obs['returned']
    obs['cutoff']      = cutoff
    obs['window_days'] = future_window_days
    return obs


def summarise(obs, cohort_name, cutoff_label, future_window_days):
    total      = len(obs)
    churned    = int(obs['churn'].sum())
    retained   = int(obs['returned'].sum())
    churn_pct  = 100 * churned  / total
    retain_pct = 100 * retained / total
    baseline   = max(churn_pct, retain_pct)
    minority   = min(churned, retained)
    weak_flag  = 'WEAK' if minority < MIN_MINORITY_CLASS else ''

    return {
        'Cohort':             cohort_name,
        'Future Window':      future_window_days,
        'Cutoff Strategy':    cutoff_label,
        'Observations':       total,
        'Unique Customers':   obs['customer_unique_id'].nunique(),
        'Class 0 (retained)': retained,
        'Class 1 (churn)':    churned,
        'Minority %':         round(min(churn_pct, retain_pct), 2),
        'Baseline %':         round(baseline, 2),
        'Flag':               weak_flag,
    }


# ---------------------------------------------------------------------------
# 4. SINGLE LATE CUTOFF RESULTS (reference)
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 3 — SINGLE LATE CUTOFF RESULTS (Reference / Previous Approach)")
print("=" * 80)

single_results = []
for cohort_name, min_span in COHORT_SPAN_DAYS.items():
    for fw in FUTURE_WINDOWS:
        obs = build_observation(single_cutoff, fw, min_span)
        if obs is not None:
            single_results.append(
                summarise(obs, cohort_name,
                          f'Single ({single_cutoff.date()})', fw))

df_single = pd.DataFrame(single_results)
print(df_single.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 5. ROLLING CUTOFFS — POOLED
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 4 — ROLLING CUTOFFS (POOLED — all cutoffs combined)")
print("=" * 80)
print("NOTE: Panel data. A customer may appear multiple times.")
print("      Unique Customers column counts deduplicated IDs.")
print()

pooled_results = []
for cohort_name, min_span in COHORT_SPAN_DAYS.items():
    for fw in FUTURE_WINDOWS:
        frames = []
        for cutoff in cutoffs:
            if cutoff + pd.Timedelta(days=fw) > latest_date + pd.Timedelta(days=1):
                continue
            obs = build_observation(cutoff, fw, min_span)
            if obs is not None:
                frames.append(obs)
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        pooled_results.append(
            summarise(pooled, cohort_name, 'Rolling (pooled)', fw))

df_pooled = pd.DataFrame(pooled_results)
print(df_pooled.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 6. ROLLING CUTOFFS — LAST SNAPSHOT PER CUSTOMER (i.i.d. safe)
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 5 — ROLLING CUTOFFS (LAST SNAPSHOT PER CUSTOMER — i.i.d. safe)")
print("=" * 80)
print("NOTE: For each customer, only the most recent qualifying cutoff")
print("      observation is retained — one row per customer.")
print("      This is safe for standard train/test splitting.")
print()

last_snap_results = []
for cohort_name, min_span in COHORT_SPAN_DAYS.items():
    for fw in FUTURE_WINDOWS:
        frames = []
        for cutoff in cutoffs:
            if cutoff + pd.Timedelta(days=fw) > latest_date + pd.Timedelta(days=1):
                continue
            obs = build_observation(cutoff, fw, min_span)
            if obs is not None:
                frames.append(obs)
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        pooled_sorted = pooled.sort_values('cutoff')
        last_snap = pooled_sorted.drop_duplicates(
            subset='customer_unique_id', keep='last')
        last_snap_results.append(
            summarise(last_snap, cohort_name, 'Rolling (last-snapshot)', fw))

df_last = pd.DataFrame(last_snap_results)
print(df_last.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 7. LABEL FRAMING COMPARISON
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 6 — LABEL FRAMING COMPARISON")
print("=" * 80)
print("""
CHURN FRAMING (recommended for academia):
   Churn = 1  ->  customer did NOT purchase in the future window
   Churn = 0  ->  customer DID purchase in the future window
   Majority class = churned (no purchase) => high baseline.
   Aligns with standard churn prediction literature.

RETURN FRAMING (equivalent — different label names only):
   Return = 1  ->  customer DID purchase in the future window
   Return = 0  ->  customer did NOT purchase in the future window
   This is the mathematical INVERSE of churn framing.
   Class distributions are IDENTICAL. No data changes.
   In return framing the minority class = returned customers.

RECOMMENDATION:
   Use CHURN framing for academic paper consistency.
   Document clearly in the paper:
     Churn = 1 means the customer made no purchase in the future window.
     Churn = 0 means the customer made at least one purchase.
   Framing as "return prediction" does NOT improve model performance.
""")

# ---------------------------------------------------------------------------
# 8. SCORING & RECOMMENDATION
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 7 — SCORING & RECOMMENDATION")
print("=" * 80)

scored = []
for row in last_snap_results:
    obs_count = row['Observations']
    minority  = min(row['Class 0 (retained)'], row['Class 1 (churn)'])
    baseline  = row['Baseline %']

    score = 0
    if obs_count >= 200:  score += 2
    elif obs_count >= 100: score += 1
    if minority >= 100:   score += 3
    elif minority >= 50:  score += 2
    elif minority >= 30:  score += 1
    if baseline < 90:     score += 2
    if baseline < 80:     score += 2

    scored.append({**row, 'Score': score})

df_scored = pd.DataFrame(scored).sort_values(
    ['Score', 'Observations', 'Minority %'],
    ascending=[False, False, False]
).reset_index(drop=True)

print("All last-snapshot configurations ranked by composite score:")
print(df_scored.to_string(index=False))
print()

best = df_scored.iloc[0]
print("-" * 80)
print("RECOMMENDED DESIGN")
print("-" * 80)
print(f"  Cohort              : {best['Cohort']}")
print(f"  Future Window       : {best['Future Window']} days")
print(f"  Cutoff Strategy     : {best['Cutoff Strategy']}")
print(f"  Total Observations  : {best['Observations']}")
print(f"  Unique Customers    : {best['Unique Customers']}")
print(f"  Class 0 (retained)  : {best['Class 0 (retained)']}")
print(f"  Class 1 (churn)     : {best['Class 1 (churn)']}")
print(f"  Minority %          : {best['Minority %']}%")
print(f"  Majority Baseline   : {best['Baseline %']}%")
if best['Flag'] == 'WEAK':
    print(f"  WARNING: Minority class below 50 — statistically weak")
print()

# ---------------------------------------------------------------------------
# 9. LEAKAGE PIPELINE
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 8 — LEAKAGE-FREE PIPELINE FOR RECOMMENDED DESIGN")
print("=" * 80)
print(f"""
Recommended: {best['Cohort']} | {best['Future Window']}-day window | {best['Cutoff Strategy']}

  Historical Data
    All orders with purchase_timestamp < cutoff_date
    |
    v
  Feature Calculation (from historical data ONLY)
    - total_orders          (count of orders in history)
    - days_since_last_order (cutoff - max historical order date)
    - avg_order_value       (from order_items, history only)
    - order_frequency       (orders / tenure days)
    - tenure_days           (last_hist_order - first_hist_order)
    - review_score_avg      (reviews received before cutoff)
    - payment_types_used    (distinct methods in history)
    |
    v
  Prediction Cutoff Date
    (HARD BOUNDARY — no future data crosses this line)
    |
    v
  Future Window: [cutoff, cutoff + {best['Future Window']} days)
    Used ONLY to compute the label.
    |
    v
  Label Assignment
    Churn = 1  ->  no purchase detected in future window
    Churn = 0  ->  at least one purchase in future window

  GUARANTEE: No transaction in [cutoff, cutoff+window) is used
  in any feature. No transaction after (cutoff+window) is used.
""")

# ---------------------------------------------------------------------------
# 10. FULL COMPARISON TABLE
# ---------------------------------------------------------------------------
print("=" * 80)
print("SECTION 9 — FULL COMPARISON TABLE")
print("=" * 80)

all_combined = (
    [dict(r, **{'Strategy Type': 'Single cutoff'})      for r in single_results] +
    [dict(r, **{'Strategy Type': 'Rolling last-snap'})  for r in last_snap_results]
)
df_all = pd.DataFrame(all_combined)[[
    'Cohort', 'Future Window', 'Strategy Type',
    'Observations', 'Unique Customers',
    'Class 0 (retained)', 'Class 1 (churn)',
    'Minority %', 'Baseline %', 'Flag'
]]
print(df_all.to_string(index=False))
print()

print("=" * 80)
print("INVESTIGATION COMPLETE — Awaiting approval before dataset generation.")
print("=" * 80)
