import time, requests, json

time.sleep(3)

BASE = 'http://127.0.0.1:5000'
print('=== Dynamic Analytics API Verification ===')

# 1. Core routes
routes = ['/', '/login', '/register', '/academic']
for r in routes:
    try:
        resp = requests.get(BASE + r, timeout=8, allow_redirects=False)
        print(f'  {resp.status_code}  GET {r}')
    except Exception as e:
        print(f'  ERR  GET {r}: {e}')

# 2. Dynamic /api/academic/run
tests = [
    {'analysis_type':'classification','model':'logistic_regression','parameters':{'test_size':0.2,'random_state':42}},
    {'analysis_type':'classification','model':'random_forest',      'parameters':{'test_size':0.2,'n_estimators':50,'random_state':42}},
    {'analysis_type':'classification','model':'decision_tree',      'parameters':{'test_size':0.2,'max_depth':5}},
    {'analysis_type':'classification','model':'naive_bayes',        'parameters':{'test_size':0.2}},
    {'analysis_type':'regression',    'model':'simple_linear',      'parameters':{'test_size':0.2}},
    {'analysis_type':'regression',    'model':'ridge',              'parameters':{'alpha':1.0}},
    {'analysis_type':'clustering',    'model':'kmeans',             'parameters':{'n_clusters':3,'sample_size':2000}},
    {'analysis_type':'clustering',    'model':'kmeans',             'parameters':{'n_clusters':5,'sample_size':2000}},
    {'analysis_type':'association_rules','model':'apriori',         'parameters':{'min_support':0.00005,'min_confidence':0.01}},
    {'analysis_type':'olap',          'model':'rollup',             'parameters':{'dimension':'time','level':'year'}},
    {'analysis_type':'olap',          'model':'drilldown',          'parameters':{'year':'2018'}},
    {'analysis_type':'olap',          'model':'slice',              'parameters':{'dimension':'geography','state':'SP'}},
    {'analysis_type':'olap',          'model':'dice',               'parameters':{'states':['SP','RJ'],'years':[2017,2018],'categories':['bed_bath_table']}},
]

print('\n--- /api/academic/run ---')
for t in tests:
    try:
        r = requests.post(BASE + '/api/academic/run', json=t, timeout=120)
        d = r.json()
        ok = 'OK' if d.get('success') else 'FAIL'
        cached_str = '[C]' if d.get('_cached') else '[N]'
        m = d.get('metrics', {})
        if 'accuracy' in m:          metric = f"acc={m['accuracy']:.3f}"
        elif 'r2' in m:              metric = f"r2={m['r2']:.3f}"
        elif 'silhouette_score' in m:metric = f"sil={m['silhouette_score']:.3f}"
        elif 'n_rules' in m:         metric = f"rules={m['n_rules']}"
        elif 'total_orders' in m:    metric = f"orders={m['total_orders']}"
        else:                        metric = d.get('error','')[:40]
        print(f'  {ok} {cached_str} {r.status_code} {t["analysis_type"]:18s} {t["model"]:20s} {metric}')
    except Exception as e:
        print(f'  ERR {t["analysis_type"]:18s} {t["model"]}: {e}')

# 3. Validation
print('\n--- Validation ---')
for t in [
    {'analysis_type':'classification','model':'bad_model','parameters':{}},
    {'analysis_type':'bad_type','model':'logistic_regression','parameters':{}},
]:
    try:
        r = requests.post(BASE + '/api/academic/run', json=t, timeout=10)
        d = r.json()
        status = 'CORRECTLY REJECTED' if (not d.get('success') and r.status_code in (400,422)) else 'WRONG'
        print(f'  {status}: {t["analysis_type"]}/{t["model"]} -> {d.get("error","")}')
    except Exception as e:
        print(f'  ERR: {e}')

# 4. Caching
print('\n--- Cache test (2nd call should be cached) ---')
payload = {'analysis_type':'classification','model':'logistic_regression','parameters':{'test_size':0.2,'random_state':42}}
try:
    r = requests.post(BASE + '/api/academic/run', json=payload, timeout=60)
    d = r.json()
    print(f'  _cached={d.get("_cached")} (expected True)')
except Exception as e:
    print(f'  ERR: {e}')

# 5. OLAP filters
print('\n--- OLAP filter endpoints ---')
for ep in ['/api/academic/states', '/api/academic/categories']:
    try:
        r = requests.get(BASE + ep, timeout=5)
        d = r.json()
        print(f'  {r.status_code} {ep} -> {len(d)} items, first: {d[:3]}')
    except Exception as e:
        print(f'  ERR {ep}: {e}')

print('\n=== DONE ===')
