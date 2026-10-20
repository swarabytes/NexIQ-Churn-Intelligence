/* ═══════════════════════════════════════════════════════════════
   academic.js — Dynamic Analytics Laboratory
   All analyses computed by the real backend API.
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── STATE ────────────────────────────────────────────────────
const STATE = {
  currentResult: null,
  currentExecutedConfig: null, // { type, model, paramsStr }
  analysisHistory: [],      // [{type, model, params, result, ts}]
  tableData: [],            // raw rows for current table
  tableFiltered: [],        // filtered + sorted rows
  tablePage: 1,
  tablePageSize: 15,
  tableSortCol: null,
  tableSortAsc: true,
  charts: [],               // Chart.js instances (to destroy on update)
  olap: { states: [], categories: [] },
  currentExploration: null  // Holds exploration data for feature switching
};

// ─── MODEL DEFINITIONS ────────────────────────────────────────
const MODEL_MAP = {
  classification: [
    { value: 'logistic_regression', label: '🔢 Logistic Regression' },
    { value: 'decision_tree',       label: '🌳 Decision Tree' },
    { value: 'random_forest',       label: '🌲 Random Forest' },
    { value: 'naive_bayes',         label: '📊 Gaussian Naive Bayes' },
    { value: 'compare_all',         label: '📊 Compare All Models (Benchmarking)' }
  ],
  regression: [
    { value: 'simple_linear',   label: '📏 Simple Linear Regression' },
    { value: 'multiple_linear', label: '📐 Multiple Linear Regression' },
    { value: 'lasso',           label: '🎯 Lasso Regression' },
    { value: 'ridge',           label: '🔺 Ridge Regression' },
    { value: 'elastic_net',     label: '⚡ Elastic Net' },
    { value: 'compare_all',     label: '📊 Compare All Models (Benchmarking)' }
  ],
  clustering: [
    { value: 'kmeans',         label: '🔵 K-Means Clustering' },
    { value: 'agglomerative',  label: '🌿 Agglomerative Clustering' }
  ],
  association_rules: [
    { value: 'apriori', label: '🔗 Apriori' }
  ],
  olap: [
    { value: 'rollup',    label: '⬆️ Roll-Up' },
    { value: 'drilldown', label: '⬇️ Drill-Down' },
    { value: 'slice',     label: '🔪 Slice' },
    { value: 'dice',      label: '🎲 Dice' },
    { value: 'pivot',     label: '🔄 Pivot (Rotate)' }
  ]
};

const DATASET_LABELS = {
  classification:   'Olist Churn Dataset (1,684 customers)',
  regression:       'Olist Freight Regression Dataset (98,200 orders, sampled)',
  clustering:       'Olist Churn Dataset (sampled for scalability)',
  association_rules:'Olist Order Items (real transactions)',
  olap:             'Olist OLAP Summary Cube (year/quarter/state/category)'
};

// ─── PARAMETER DEFINITIONS ────────────────────────────────────
const PARAMS_DEF = {
  // Classification
  logistic_regression: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05, hint: '0.10–0.40' },
    { id: 'C', label: 'C (Regularization)', type: 'number', default: 1.0, min: 0.01, step: 0.1, hint: 'Inverse of regularization strength' },
    { id: 'class_weight', label: 'Class Weight', type: 'select', options: ['balanced', 'none'], default: 'balanced' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  decision_tree: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'max_depth', label: 'Max Depth', type: 'number', default: 5, min: 1, max: 30, step: 1, hint: 'Higher = more complex tree' },
    { id: 'min_samples_split', label: 'Min Samples Split', type: 'number', default: 2, min: 2, step: 1 },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  random_forest: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'n_estimators', label: 'Number of Trees', type: 'number', default: 100, min: 10, max: 500, step: 10 },
    { id: 'max_depth', label: 'Max Depth', type: 'number', default: 10, min: 1, max: 30, step: 1 },
    { id: 'class_weight', label: 'Class Weight', type: 'select', options: ['balanced', 'none'], default: 'balanced' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  naive_bayes: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'var_smoothing', label: 'Var Smoothing', type: 'number', default: 1e-9, min: 1e-12, step: 1e-9, hint: 'Numerical stability term' }
  ],
  // Regression
  simple_linear: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  multiple_linear: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  lasso: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'alpha', label: 'Alpha (Regularization)', type: 'number', default: 1.0, min: 0.001, step: 0.1, hint: 'Higher = more regularization' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  ridge: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'alpha', label: 'Alpha (Regularization)', type: 'number', default: 1.0, min: 0.001, step: 0.1 },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  elastic_net: [
    { id: 'test_size', label: 'Test Size', type: 'number', default: 0.20, min: 0.1, max: 0.4, step: 0.05 },
    { id: 'alpha', label: 'Alpha', type: 'number', default: 1.0, min: 0.001, step: 0.1 },
    { id: 'l1_ratio', label: 'L1 Ratio', type: 'number', default: 0.5, min: 0, max: 1, step: 0.1, hint: '0=Ridge, 1=Lasso' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  // Clustering
  kmeans: [
    { id: 'n_clusters', label: 'K (Clusters)', type: 'number', default: 4, min: 2, max: 10, step: 1, hint: '2–10' },
    { id: 'init', label: 'Initialization', type: 'select', options: ['k-means++', 'random'], default: 'k-means++' },
    { id: 'sample_size', label: 'Sample Size', type: 'number', default: 5000, min: 500, max: 10000, step: 500, hint: 'Rows used (performance)' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  agglomerative: [
    { id: 'n_clusters', label: 'Clusters', type: 'number', default: 4, min: 2, max: 10, step: 1 },
    { id: 'linkage', label: 'Linkage Method', type: 'select', options: ['ward', 'complete', 'average', 'single'], default: 'ward', hint: 'Ward minimizes within-cluster variance' },
    { id: 'sample_size', label: 'Sample Size', type: 'number', default: 2000, min: 200, max: 3000, step: 200, hint: 'Capped at 3000 for speed' },
    { id: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 }
  ],
  // Association
  apriori: [
    { id: 'min_support', label: 'Min Support', type: 'number', default: 0.00005, min: 0.00001, step: 0.00005, hint: 'Very low because Olist is sparse' },
    { id: 'min_confidence', label: 'Min Confidence', type: 'number', default: 0.01, min: 0.001, step: 0.005 },
    { id: 'min_lift', label: 'Min Lift', type: 'number', default: 1.01, min: 0.5, step: 0.1 },
    { id: 'max_len', label: 'Max Itemset Length', type: 'number', default: 3, min: 2, max: 5, step: 1 }
  ],
  // OLAP
  rollup: [
    { id: 'dimension', label: 'Dimension', type: 'select', options: ['time', 'geography', 'product'], default: 'time' },
    { id: 'level', label: 'Level', type: 'select', options: ['year', 'quarter'], default: 'year' },
    { id: 'measure', label: 'Measure', type: 'select', options: ['order_count', 'total_sales', 'customer_count', 'item_count'], default: 'order_count' }
  ],
  drilldown: [
    { id: 'year', label: 'Year to Drill Into', type: 'select', options: ['All', '2016', '2017', '2018'], default: '2018' },
    { id: 'dimension', label: 'Dimension', type: 'select', options: ['time'], default: 'time' },
    { id: 'measure', label: 'Measure', type: 'select', options: ['order_count', 'total_sales', 'customer_count'], default: 'order_count' }
  ],
  slice: [
    { id: 'dimension', label: 'Slice Dimension', type: 'select', options: ['geography', 'time', 'product'], default: 'geography' },
    { id: 'state', label: 'State (if Geography)', type: 'text', default: 'SP', hint: 'e.g. SP, RJ, MG' },
    { id: 'year', label: 'Year (if Time)', type: 'text', default: '', hint: 'e.g. 2018' },
    { id: 'category', label: 'Category (if Product)', type: 'text', default: '', hint: 'e.g. bed_bath_table' },
    { id: 'measure', label: 'Measure', type: 'select', options: ['order_count', 'total_sales', 'customer_count'], default: 'order_count' }
  ],
  dice: [],  // Dice has dynamic multi-select — handled separately
  pivot: [
    { id: 'row_dim', label: 'Row Dimension', type: 'select', options: ['state', 'category', 'year', 'quarter'], default: 'state' },
    { id: 'col_dim', label: 'Column Dimension (Rotate)', type: 'select', options: ['year', 'quarter', 'state', 'category'], default: 'year' },
    { id: 'measure', label: 'Measure', type: 'select', options: ['total_sales', 'order_count', 'customer_count', 'item_count'], default: 'total_sales' }
  ]
};

// ─── INIT ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchOlapFilters();
});

async function fetchOlapFilters() {
  try {
    const [sr, cr] = await Promise.all([
      fetch('/api/academic/states').then(r => r.json()),
      fetch('/api/academic/categories').then(r => r.json())
    ]);
    STATE.olap.states = Array.isArray(sr) ? sr : [];
    STATE.olap.categories = Array.isArray(cr) ? cr : [];
  } catch(e) {
    console.warn('Could not load OLAP filters', e);
  }
}

// ─── VIEW SWITCHING & SIDEBAR ROUTING ────────────────────────
function updateBreadcrumb(title) {
  const el = document.getElementById('currentSectionBreadcrumb');
  if (el) el.textContent = title;
}

function switchLabTab(view, btn) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active-view'));
  document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));

  const el = document.getElementById(`view-${view}`);
  if (el) {
    el.classList.add('active-view');
  }

  const targetBtn = btn || document.querySelector(`.sidebar-btn[data-view="${view}"]`);
  if (targetBtn) {
    targetBtn.classList.add('active');
  }

  const titleMap = {
    overview: 'Overview',
    lab: 'Analysis Laboratory',
    exploration: 'Data Exploration',
    schema: 'Data Warehouse Schemas',
    methodology: 'Methodology'
  };
  updateBreadcrumb(titleMap[view] || view);

  if (view === 'exploration') loadExploration(1);
  if (view === 'schema') loadLiveSchema();
}

// Backwards compatibility alias for existing callers
function showView(view, btn) {
  switchLabTab(view, btn);
}

function selectAnalysisFromSidebar(analysisType, btn) {
  // Switch to the laboratory view
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active-view'));
  document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));

  const labView = document.getElementById('view-lab');
  if (labView) labView.classList.add('active-view');

  const targetBtn = btn || document.querySelector(`.sidebar-btn[data-type="${analysisType}"]`);
  if (targetBtn) targetBtn.classList.add('active');

  const typeNames = {
    classification: 'Classification',
    regression: 'Regression',
    clustering: 'Clustering',
    association_rules: 'Association Rules',
    olap: 'OLAP Operations'
  };
  updateBreadcrumb(`Analysis / ${typeNames[analysisType] || analysisType}`);

  // Set the analysis type in the dropdown
  const typeSelect = document.getElementById('analysisType');
  if (typeSelect) {
    typeSelect.value = analysisType;
    onTypeChange();

    // Automatically select the first model for convenience
    const mSel = document.getElementById('modelSelect');
    if (mSel && mSel.options.length > 1) {
      mSel.selectedIndex = 1;
      onModelChange();
    }
  }

  const labTitle = document.getElementById('labControlTitle');
  if (labTitle) labTitle.textContent = `${typeNames[analysisType] || 'Analysis'} Configuration`;
}

// ─── CASCADING DROPDOWNS ──────────────────────────────────────
function onTypeChange() {
  const type  = document.getElementById('analysisType').value;
  const mSel  = document.getElementById('modelSelect');
  const badge = document.getElementById('datasetBadge');

  // Reset model dropdown
  mSel.innerHTML = '<option value="">— Select Algorithm —</option>';
  mSel.disabled = !type;

  if (type && MODEL_MAP[type]) {
    MODEL_MAP[type].forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.value;
      opt.textContent = m.label;
      mSel.appendChild(opt);
    });
  }

  badge.textContent = DATASET_LABELS[type] || 'Select an analysis type';
  document.getElementById('paramsPanel').classList.add('hidden');
  document.getElementById('olapFiltersPanel').classList.add('hidden');
  document.getElementById('runBtn').disabled = true;
}

function onModelChange() {
  const model = document.getElementById('modelSelect').value;
  const type  = document.getElementById('analysisType').value;
  const runBtn = document.getElementById('runBtn');

  runBtn.disabled = !model;
  renderParamsPanel(type, model);
}

// ─── PARAMETERS PANEL ─────────────────────────────────────────
function renderParamsPanel(type, model) {
  const panel  = document.getElementById('paramsPanel');
  const body   = document.getElementById('paramsBody');
  const olapFP = document.getElementById('olapFiltersPanel');
  const olapFB = document.getElementById('olapFiltersBody');

  checkStaleParameters();

  // Dice needs custom multi-select panel
  if (model === 'dice') {
    panel.classList.add('hidden');
    olapFP.classList.remove('hidden');
    renderDiceFilters(olapFB);
    attachParamChangeListeners();
    return;
  }

  olapFP.classList.add('hidden');

  const defs = PARAMS_DEF[model];
  if (!defs || defs.length === 0) { 
    panel.classList.add('hidden'); 
    return; 
  }

  body.innerHTML = defs.map(p => renderParam(p)).join('');
  panel.classList.remove('hidden');
  attachParamChangeListeners();
}

function attachParamChangeListeners() {
  const inputs = document.querySelectorAll('#paramsPanel input, #paramsPanel select, #olapFiltersPanel input, #modelSelect');
  inputs.forEach(el => {
    el.removeEventListener('change', checkStaleParameters);
    el.removeEventListener('input', checkStaleParameters);
    el.addEventListener('change', checkStaleParameters);
    el.addEventListener('input', checkStaleParameters);
  });
}

function checkStaleParameters() {
  if (!STATE.currentExecutedConfig) return;
  const currentType = document.getElementById('analysisType')?.value;
  const currentModel = document.getElementById('modelSelect')?.value;
  if (!currentType || !currentModel) return;

  const currentParams = readParams(currentModel);
  const currentStr = JSON.stringify(currentParams);

  const banner = document.getElementById('staleWarningBanner');
  const modelNameEl = document.getElementById('staleModelName');

  const isStale = (
    currentType !== STATE.currentExecutedConfig.type ||
    currentModel !== STATE.currentExecutedConfig.model ||
    currentStr !== STATE.currentExecutedConfig.paramsStr
  );

  if (isStale && STATE.currentResult) {
    if (modelNameEl) modelNameEl.textContent = `${STATE.currentExecutedConfig.model.replace('_',' ')} (${STATE.currentExecutedConfig.type})`;
    if (banner) banner.classList.remove('hidden');
  } else {
    if (banner) banner.classList.add('hidden');
  }
}

// ─── CHART EXPORT ─────────────────────────────────────────────
function downloadChartPNG(canvasId, title = 'chart') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const imageURI = canvas.toDataURL('image/png');
  const link = document.createElement('a');
  link.download = `${title}_${new Date().toISOString().slice(0,10)}.png`;
  link.href = imageURI;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function renderParam(p) {
  if (p.type === 'select') {
    const opts = p.options.map(o =>
      `<option value="${o}" ${o === p.default ? 'selected' : ''}>${o}</option>`
    ).join('');
    return `<div class="param-group">
      <label class="param-label">${p.label}</label>
      <select class="param-select" id="param_${p.id}">${opts}</select>
      ${p.hint ? `<span class="param-hint">${p.hint}</span>` : ''}
    </div>`;
  } else {
    const attrs = [
      `type="${p.type}"`,
      `value="${p.default}"`,
      p.min !== undefined ? `min="${p.min}"` : '',
      p.max !== undefined ? `max="${p.max}"` : '',
      p.step !== undefined ? `step="${p.step}"` : ''
    ].filter(Boolean).join(' ');
    return `<div class="param-group">
      <label class="param-label">${p.label}</label>
      <input class="param-input" id="param_${p.id}" ${attrs}/>
      ${p.hint ? `<span class="param-hint">${p.hint}</span>` : ''}
    </div>`;
  }
}

function renderDiceFilters(container) {
  const stateOpts = STATE.olap.states.slice(0, 50).map(s =>
    `<label class="olap-checkbox-item"><input type="checkbox" value="${s}" name="d_state" ${['SP','RJ','MG'].includes(s)?'checked':''}/> ${s}</label>`
  ).join('');
  const yearOpts = ['2016','2017','2018'].map(y =>
    `<label class="olap-checkbox-item"><input type="checkbox" value="${y}" name="d_year" ${y!=='2016'?'checked':''}/> ${y}</label>`
  ).join('');
  const catOpts = STATE.olap.categories.slice(0, 60).map(c =>
    `<label class="olap-checkbox-item"><input type="checkbox" value="${c}" name="d_cat" ${ ['bed_bath_table','health_beauty'].includes(c)?'checked':''}/> ${c}</label>`
  ).join('');

  container.innerHTML = `
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
    <div class="olap-multi-group">
      <div class="olap-multi-label">States</div>
      <div class="olap-checkbox-list">${stateOpts || '<em style="font-size:0.75rem;color:#64748b">Loading...</em>'}</div>
    </div>
    <div class="olap-multi-group">
      <div class="olap-multi-label">Years</div>
      <div class="olap-checkbox-list">${yearOpts}</div>
    </div>
    <div class="olap-multi-group">
      <div class="olap-multi-label">Categories</div>
      <div class="olap-checkbox-list">${catOpts || '<em style="font-size:0.75rem;color:#64748b">Loading...</em>'}</div>
    </div>
  </div>`;
}

function toggleParams() {
  const body = document.getElementById('paramsBody');
  const icon = document.getElementById('paramsToggleIcon');
  body.classList.toggle('hidden');
  icon.textContent = body.classList.contains('hidden') ? '▶' : '▼';
}

// ─── READ PARAMS ──────────────────────────────────────────────
function readParams(model) {
  const params = {};
  const defs = PARAMS_DEF[model] || [];

  if (model === 'dice') {
    params.states     = [...document.querySelectorAll('input[name="d_state"]:checked')].map(i => i.value);
    params.years      = [...document.querySelectorAll('input[name="d_year"]:checked')].map(i => parseInt(i.value));
    params.categories = [...document.querySelectorAll('input[name="d_cat"]:checked')].map(i => i.value);
    return params;
  }

  defs.forEach(p => {
    const el = document.getElementById(`param_${p.id}`);
    if (!el) return;
    if (p.type === 'number') {
      params[p.id] = parseFloat(el.value);
    } else {
      params[p.id] = el.value;
    }
  });
  return params;
}

// ─── RUN ANALYSIS ─────────────────────────────────────────────
async function runAnalysis() {
  const type  = document.getElementById('analysisType').value;
  const model = document.getElementById('modelSelect').value;
  if (!type || !model) return;

  const params = readParams(model);

  // Show loading
  showLoading(type, model);
  animateLoadingSteps(type, model);

  try {
    const res = await fetch('/api/academic/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis_type: type, model, parameters: params })
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      showError(data.error || 'Analysis failed. Please check the backend logs.');
      return;
    }

    STATE.currentResult = data;
    STATE.currentExecutedConfig = {
      type,
      model,
      paramsStr: JSON.stringify(params)
    };
    const banner = document.getElementById('staleWarningBanner');
    if (banner) banner.classList.add('hidden');

    addToHistory(type, model, params, data);
    renderResult(data);

  } catch(e) {
    showError(`Network error: ${e.message}`);
  }
}

// ─── LOADING ANIMATION ────────────────────────────────────────
let loadStepTimer = null;
function animateLoadingSteps(type, model) {
  const steps = ['ls1','ls2','ls3','ls4','ls5'];
  steps.forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
    if (i === 0) el.classList.add('active');
  });

  let idx = 0;
  clearInterval(loadStepTimer);
  loadStepTimer = setInterval(() => {
    if (idx < steps.length - 1) {
      document.getElementById(steps[idx]).classList.remove('active');
      document.getElementById(steps[idx]).classList.add('done');
      idx++;
      document.getElementById(steps[idx]).classList.add('active');
    }
  }, 600);
}

function showLoading(type, model) {
  const modelLabel = MODEL_MAP[type]?.find(m => m.value === model)?.label || model;
  document.getElementById('loadingTitle').textContent = `Running ${modelLabel}...`;
  hide('idleState'); hide('resultArea'); hide('errorState');
  show('loadingState');
}

// ─── ERROR STATE ──────────────────────────────────────────────
function showError(msg) {
  clearInterval(loadStepTimer);
  document.getElementById('errorMsg').textContent = msg;
  hide('loadingState'); hide('idleState'); hide('resultArea');
  show('errorState');
}

function resetToIdle() {
  hide('loadingState'); hide('resultArea'); hide('errorState');
  show('idleState');
}

// ─── RESULT RENDERING ─────────────────────────────────────────
function renderResult(data) {
  clearInterval(loadStepTimer);
  hide('loadingState'); hide('idleState'); hide('errorState');

  // Header
  document.getElementById('resTitle').textContent = data.model || data.operation || data.analysis_type;
  document.getElementById('resType').textContent   = (data.analysis_type || '').toUpperCase();
  document.getElementById('resDataset').textContent = data.dataset || 'Olist';
  const statusEl = document.getElementById('resStatus');
  if (data._cached) {
    statusEl.textContent = 'Cached';
    statusEl.className = 'tag tag-cached';
  } else {
    statusEl.textContent = 'Computed';
    statusEl.className = 'tag tag-status';
  }
  const timing = data.train_time_s || data.compute_time_s;
  document.getElementById('resTiming').textContent = timing ? `${timing}s compute time` : '';

  // Destroy old charts
  STATE.charts.forEach(c => { try { c.destroy(); } catch(e){} });
  STATE.charts = [];

  // Render by type
  switch(data.analysis_type) {
    case 'classification':   renderClassification(data); break;
    case 'regression':       renderRegression(data); break;
    case 'clustering':       renderClustering(data); break;
    case 'association_rules':renderAssociation(data); break;
    case 'olap':             renderOlap(data); break;
  }

  // Interpretation
  if (data.interpretation) {
    document.getElementById('interpretationBox').innerHTML =
      data.interpretation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  }

  // Accordion: Dataset
  renderDatasetAccordion(data);
  renderParamsAccordion(data);
  renderQueryAccordion(data);
  renderFeaturesAccordion(data);

  show('resultArea');
}

// ─── METRICS RENDERER ─────────────────────────────────────────
function renderMetrics(metricsObj, formatters) {
  const grid = document.getElementById('metricsCards');
  grid.innerHTML = '';
  const entries = Object.entries(metricsObj).filter(([k,v]) => v !== null && v !== undefined);
  entries.forEach(([key, val], i) => {
    const fmt = formatters && formatters[key] ? formatters[key] : v => v;
    const highlight = i === 0 ? 'highlight' : '';
    grid.innerHTML += `<div class="metric-card ${highlight}">
      <div class="metric-value">${fmt(val)}</div>
      <div class="metric-label">${key.replace(/_/g,' ')}</div>
    </div>`;
  });
}

const pct  = v => `${(v*100).toFixed(1)}%`;
const num2 = v => {
  if (typeof v !== 'number') return v;
  if (v >= 1000000) return (v / 1000000).toFixed(2) + 'M';
  if (v >= 10000) return (v / 1000).toFixed(1) + 'K';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};
const int0 = v => {
  if (typeof v !== 'number') return v;
  if (v >= 1000000) return (v / 1000000).toFixed(2) + 'M';
  if (v >= 10000) return (v / 1000).toFixed(1) + 'K';
  return Math.round(v).toLocaleString();
};

// ─── CLASSIFICATION ───────────────────────────────────────────
function renderClassification(data) {
  const m = data.metrics;
  const isComparison = data.is_comparison || data.model === 'compare_all';

  if (isComparison && data.comparison) {
    // ── Compare All Models mode ──
    const compRaw = data.comparison;
    const compList = Array.isArray(compRaw) ? compRaw : Object.entries(compRaw).map(([k, v]) => ({ model: k, ...v }));
    const metrics = ['accuracy','precision','recall','f1_score','roc_auc','balanced_accuracy'];

    renderMetrics({
      models_compared: compList.length,
      best_f1: Math.max(...compList.map(item => item.f1_score || 0)),
      best_roc_auc: Math.max(...compList.map(item => item.roc_auc || 0))
    }, {
      models_compared: v => v,
      best_f1: pct,
      best_roc_auc: v => v ? v.toFixed(4) : 'N/A'
    });

    const grid = document.getElementById('chartsGrid');
    grid.innerHTML = '';

    // Multi-metric comparison bar chart (grouped)
    const canvasIdComp = 'chart_compare';
    grid.innerHTML += chartCardHTMLWithAction('Model Comparison — All Metrics', canvasIdComp, 'chart_compare', 'Model_Comparison');
    setTimeout(() => {
      const datasets = metrics.map((met, mi) => ({
        label: met.replace('_',' ').replace(/\b\w/g, l => l.toUpperCase()),
        data: compList.map(item => +(((item[met] || 0) * 100).toFixed(2))),
        backgroundColor: PALETTE[mi] + 'cc',
        borderRadius: 4
      }));
      const ctx = document.getElementById(canvasIdComp);
      if (!ctx) return;
      const c = new Chart(ctx, {
        type: 'bar',
        data: { labels: compList.map(item => item.model || item.model_key), datasets },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } },
            tooltip: { callbacks: { label: ctx2 => ` ${ctx2.dataset.label}: ${ctx2.parsed.y.toFixed(1)}%` } }
          },
          scales: {
            y: { beginAtZero: true, max: 105, title: { display: true, text: 'Score (%)' } },
            x: { ticks: { font: { size: 10 } } }
          }
        }
      });
      STATE.charts.push(c);
    }, 50);

    // Summary table
    setTable(compList.map(item => ({
      Model: item.model || item.model_key,
      Accuracy: pct(item.accuracy || 0),
      Precision: pct(item.precision || 0),
      Recall: pct(item.recall || 0),
      F1: pct(item.f1_score || 0),
      'ROC-AUC': item.roc_auc !== null && item.roc_auc !== undefined ? Number(item.roc_auc).toFixed(4) : 'N/A',
      'Bal. Accuracy': pct(item.balanced_accuracy || 0),
      Specificity: pct(item.specificity || 0)
    })));

  } else {
    // ── Single model mode ──
    renderMetrics({
      accuracy: m.accuracy, precision: m.precision, recall: m.recall,
      f1_score: m.f1_score, specificity: m.specificity, roc_auc: m.roc_auc,
      balanced_accuracy: m.balanced_accuracy
    }, {
      accuracy: pct, precision: pct, recall: pct, f1_score: pct,
      specificity: pct, roc_auc: v => v ? v.toFixed(4) : 'N/A',
      balanced_accuracy: pct
    });

    const grid = document.getElementById('chartsGrid');
    grid.innerHTML = '';

    // Class distribution info card
    if (data.class_distribution) {
      const cd = data.class_distribution;
      grid.innerHTML += `<div class="chart-card">
        <div class="chart-card-title">Class Distribution — Imbalance Context</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;">
          <div style="background:#f0fdf4;padding:0.7rem;border-radius:8px;text-align:center;">
            <div style="font-size:1.3rem;font-weight:800;color:#059669;">${(cd.non_churn_count ?? cd.non_churn ?? 0).toLocaleString()}</div>
            <div style="font-size:0.72rem;color:#64748b;">Non-Churn (0)</div>
          </div>
          <div style="background:#fff1f2;padding:0.7rem;border-radius:8px;text-align:center;">
            <div style="font-size:1.3rem;font-weight:800;color:#dc2626;">${(cd.churn_count ?? cd.churn ?? 0).toLocaleString()}</div>
            <div style="font-size:0.72rem;color:#64748b;">Churn (1) — ${cd.churn_pct || '?'}%</div>
          </div>
        </div>
        <div style="margin-top:0.7rem;padding:0.5rem;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;font-size:0.75rem;color:#92400e;">
          ⚠️ Severe class imbalance (ratio ${cd.imbalance_ratio || 'N/A'}). Small minority-class test sample limits recall reliability.
        </div>
      </div>`;
    }

    // Confusion Matrix
    const cm = data.confusion_matrix || {};
    grid.innerHTML += `<div class="chart-card">
      <div class="chart-card-title">Confusion Matrix</div>
      <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.5rem;">Rows = Actual class, Columns = Predicted class</div>
      <div class="cm-grid">
        <div class="cm-cell cm-tn">${cm.TN||0}<small>TN — Correct non-churn</small></div>
        <div class="cm-cell cm-fp">${cm.FP||0}<small>FP — False alarm</small></div>
        <div class="cm-cell cm-fn">${cm.FN||0}<small>FN — Missed churn</small></div>
        <div class="cm-cell cm-tp">${cm.TP||0}<small>TP — Correct churn</small></div>
      </div>
    </div>`;

    // Metrics bar chart
    const canvasId1 = 'chart_clf_metrics';
    grid.innerHTML += chartCardHTMLWithAction('Metric Comparison (This Run)', canvasId1, 'chart_clf_metrics', 'Classification_Metrics');
    const labels1 = ['Accuracy','Precision','Recall','F1','Specificity','Balanced Acc'];
    const vals1 = [m.accuracy, m.precision, m.recall, m.f1_score, m.specificity, m.balanced_accuracy].map(v => v ? v*100 : 0);
    setTimeout(() => {
      const c = makeBarChart(canvasId1, labels1, vals1, 'Score (%)', ['#2563eb','#7c3aed','#0891b2','#059669','#d97706','#dc2626']);
      STATE.charts.push(c);
    }, 50);

    // ROC Curve with diagonal reference line
    if (data.roc_curve) {
      const canvasId2 = 'chart_roc';
      const aucLabel = m.roc_auc ? ` (AUC = ${m.roc_auc.toFixed(3)})` : '';
      grid.innerHTML += chartCardHTMLWithAction(`ROC Curve${aucLabel}`, canvasId2, 'chart_roc', 'ROC_Curve');
      setTimeout(() => {
        const fpr = data.roc_curve.fpr, tpr = data.roc_curve.tpr;
        const diag = fpr.map((f, i) => ({ x: f, y: f })); // diagonal random classifier
        const ctx = document.getElementById(canvasId2);
        if (!ctx) return;
        const c = new Chart(ctx, {
          type: 'line',
          data: {
            labels: fpr,
            datasets: [
              { label: `ROC Curve (AUC=${m.roc_auc ? m.roc_auc.toFixed(3) : 'N/A'})`, data: tpr, borderColor: '#2563eb', backgroundColor: '#2563eb22', fill: true, pointRadius: 0, tension: 0.3, borderWidth: 2 },
              { label: 'Random Classifier', data: fpr, borderColor: '#94a3b8', backgroundColor: 'transparent', borderDash: [5, 5], pointRadius: 0, tension: 0, borderWidth: 1 }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } }, tooltip: { mode: 'index' } },
            scales: {
              x: { title: { display: true, text: 'False Positive Rate (FPR)' }, min: 0, max: 1 },
              y: { title: { display: true, text: 'True Positive Rate (TPR)' }, min: 0, max: 1 }
            }
          }
        });
        STATE.charts.push(c);
      }, 50);
    }

    // PR Curve with baseline precision line
    if (data.pr_curve) {
      const canvasId3 = 'chart_pr';
      const apLabel = data.pr_curve.avg_precision ? ` (AP=${data.pr_curve.avg_precision.toFixed(3)})` : '';
      grid.innerHTML += chartCardHTMLWithAction(`Precision–Recall Curve${apLabel}`, canvasId3, 'chart_pr', 'PR_Curve');
      setTimeout(() => {
        const recalls = data.pr_curve.recall, precisions = data.pr_curve.precision;
        const baseline = data.pr_curve.baseline_precision || null;
        const ctx = document.getElementById(canvasId3);
        if (!ctx) return;
        const datasets = [{
          label: `PR Curve (AP=${data.pr_curve.avg_precision ? data.pr_curve.avg_precision.toFixed(3) : 'N/A'})`,
          data: precisions, borderColor: '#7c3aed', backgroundColor: '#7c3aed22', fill: true,
          pointRadius: 0, tension: 0.3, borderWidth: 2
        }];
        if (baseline) {
          datasets.push({
            label: `Baseline (no-skill): ${(baseline*100).toFixed(1)}%`,
            data: new Array(recalls.length).fill(baseline),
            borderColor: '#94a3b8', borderDash: [5, 5], backgroundColor: 'transparent',
            pointRadius: 0, tension: 0, borderWidth: 1
          });
        }
        const c = new Chart(ctx, {
          type: 'line',
          data: { labels: recalls, datasets },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
            scales: {
              x: { title: { display: true, text: 'Recall' }, min: 0, max: 1 },
              y: { title: { display: true, text: 'Precision' }, min: 0, max: 1 }
            }
          }
        });
        STATE.charts.push(c);
      }, 50);
    }

    // Feature importance with positive/negative coloring
    if (data.feature_importance) {
      const canvasId4 = 'chart_feat';
      grid.innerHTML += chartCardHTMLWithAction('Feature Importance / Coefficients', canvasId4, 'chart_feat', 'Feature_Importance');
      const fi = data.feature_importance;
      const sorted = Object.entries(fi).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12);
      setTimeout(() => {
        const ctx = document.getElementById(canvasId4);
        if (!ctx) return;
        const colors = sorted.map(e => e[1] >= 0 ? '#2563ebcc' : '#dc2626cc');
        const c = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: sorted.map(e => e[0].replace('num__','').replace('cat__','').substring(0, 22)),
            datasets: [{ data: sorted.map(e => e[1]), backgroundColor: colors, borderRadius: 3, barThickness: 14 }]
          },
          options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: { callbacks: { label: ctx2 => ` Value: ${ctx2.parsed.x.toFixed(4)} (${ctx2.parsed.x >= 0 ? 'positive' : 'negative'} effect)` } }
            },
            scales: {
              x: { title: { display: true, text: 'Coefficient / Importance' } },
              y: { ticks: { font: { size: 10 } } }
            }
          }
        });
        STATE.charts.push(c);
      }, 50);
    }

    // Table: CM breakdown
    setTable([
      { Metric: 'True Negatives (TN)', Count: cm.TN||0, Meaning: 'Correctly predicted non-churn' },
      { Metric: 'False Positives (FP)', Count: cm.FP||0, Meaning: 'Non-churn predicted as churn' },
      { Metric: 'False Negatives (FN)', Count: cm.FN||0, Meaning: 'Churn missed by model' },
      { Metric: 'True Positives (TP)', Count: cm.TP||0, Meaning: 'Correctly predicted churn' }
    ]);
  }
}

// ─── REGRESSION ───────────────────────────────────────────────
function renderRegression(data) {
  const m = data.metrics;
  const isComparison = data.is_comparison || data.model === 'compare_all';

  if (isComparison && data.comparison) {
    const compRaw = data.comparison;
    const compList = Array.isArray(compRaw) ? compRaw : Object.entries(compRaw).map(([k, v]) => ({ model: k, ...v }));

    renderMetrics({
      models_compared: compList.length,
      best_r2: Math.max(...compList.map(item => item.r2 || 0))
    }, {
      models_compared: v => v,
      best_r2: v => Number(v).toFixed(4)
    });
    const grid = document.getElementById('chartsGrid');
    grid.innerHTML = '';
    const canvasIdR2 = 'chart_reg_compare';
    grid.innerHTML += chartCardHTMLWithAction('R² Comparison Across Models', canvasIdR2, 'chart_reg_compare', 'Regression_R2');
    setTimeout(() => {
      const datasets = [
        { label: 'R²', data: compList.map(item => +(Number(item.r2 || 0)).toFixed(4)), backgroundColor: '#2563ebcc', borderRadius: 4 },
        { label: 'Adj R²', data: compList.map(item => +(Number(item.adj_r2 || 0)).toFixed(4)), backgroundColor: '#7c3aedcc', borderRadius: 4 }
      ];
      const ctx = document.getElementById(canvasIdR2);
      if (!ctx) return;
      const c = new Chart(ctx, { type: 'bar', data: { labels: compList.map(item => item.model || item.model_key), datasets },
        options: { responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
          scales: { y: { beginAtZero: true, title: { display: true, text: 'R² Score' } } } } });
      STATE.charts.push(c);
    }, 50);
    const canvasIdErr = 'chart_reg_errors';
    grid.innerHTML += chartCardHTMLWithAction('MAE & RMSE — Error Comparison (lower = better)', canvasIdErr, 'chart_reg_errors', 'Regression_Errors');
    setTimeout(() => {
      const datasets = [
        { label: 'MAE', data: compList.map(item => +(Number(item.mae || 0)).toFixed(3)), backgroundColor: '#0891b2cc', borderRadius: 4 },
        { label: 'RMSE', data: compList.map(item => +(Number(item.rmse || 0)).toFixed(3)), backgroundColor: '#dc2626cc', borderRadius: 4 }
      ];
      const ctx = document.getElementById(canvasIdErr);
      if (!ctx) return;
      const c = new Chart(ctx, { type: 'bar', data: { labels: compList.map(item => item.model || item.model_key), datasets },
        options: { responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
          scales: { y: { beginAtZero: true, title: { display: true, text: 'Error (BRL)' } } } } });
      STATE.charts.push(c);
    }, 100);
    setTable(compList.map(item => ({
      Model: item.model || item.model_key,
      'R²': Number(item.r2 || 0).toFixed(4),
      'Adj R²': Number(item.adj_r2 || 0).toFixed(4),
      MAE: Number(item.mae || 0).toFixed(3),
      RMSE: Number(item.rmse || 0).toFixed(3)
    })));
  } else {
    renderMetrics(m, { r2: v => v.toFixed(4), adj_r2: v => v.toFixed(4), mae: num2, mse: num2, rmse: num2 });
    const grid = document.getElementById('chartsGrid');
    grid.innerHTML = '';

    // Actual vs Predicted with identity line y=x
    if (data.actual_vs_predicted) {
      const canvasId = 'chart_avp';
      grid.innerHTML += chartCardHTMLWithAction('Actual vs Predicted (dashed = perfect fit y=x)', canvasId, 'chart_avp', 'Actual_vs_Predicted');
      setTimeout(() => {
        const avp = data.actual_vs_predicted;
        const pts = avp.actual.map((x, i) => ({ x, y: avp.predicted[i] }));
        const minV = Math.min(...avp.actual, ...avp.predicted);
        const maxV = Math.max(...avp.actual, ...avp.predicted);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        const c = new Chart(ctx, { type: 'scatter',
          data: { datasets: [
            { label: 'Predictions', data: pts, backgroundColor: '#2563eb55', pointRadius: 2 },
            { label: 'y=x (perfect)', data: [{ x: minV, y: minV }, { x: maxV, y: maxV }], type: 'line', borderColor: '#dc2626', borderDash: [5, 4], pointRadius: 0, fill: false, backgroundColor: 'transparent', borderWidth: 1.5 }
          ] },
          options: { responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
            scales: { x: { title: { display: true, text: 'Actual (BRL)' } }, y: { title: { display: true, text: 'Predicted (BRL)' } } } } });
        STATE.charts.push(c);
      }, 50);
    }

    // Residuals histogram
    if (data.residuals) {
      const canvasId = 'chart_res_hist';
      grid.innerHTML += chartCardHTMLWithAction('Residual Distribution', canvasId, 'chart_res_hist', 'Residuals');
      setTimeout(() => {
        const bins = histogram(data.residuals, 20);
        const c = makeBarChart(canvasId, bins.labels, bins.counts, 'Frequency', Array(bins.labels.length).fill('#0891b2aa'));
        STATE.charts.push(c);
      }, 50);
    }

    // Residual Scatter vs Fitted with y=0 line
    if (data.actual_vs_predicted && data.residuals) {
      const canvasId = 'chart_res_scatter';
      grid.innerHTML += chartCardHTMLWithAction('Residuals vs Fitted (dashed = zero line)', canvasId, 'chart_res_scatter', 'Residual_Scatter');
      setTimeout(() => {
        const avp = data.actual_vs_predicted;
        const res = data.residuals;
        const pts = avp.predicted.slice(0, 500).map((p, i) => ({ x: p, y: res[i] || 0 }));
        const minF = Math.min(...avp.predicted), maxF = Math.max(...avp.predicted);
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        const c = new Chart(ctx, { type: 'scatter',
          data: { datasets: [
            { label: 'Residuals', data: pts, backgroundColor: '#059669aa', pointRadius: 2 },
            { label: 'y=0', data: [{ x: minF, y: 0 }, { x: maxF, y: 0 }], type: 'line', borderColor: '#dc2626', borderDash: [5, 4], pointRadius: 0, fill: false, backgroundColor: 'transparent', borderWidth: 1.5 }
          ] },
          options: { responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } } },
            scales: { x: { title: { display: true, text: 'Fitted Value (BRL)' } }, y: { title: { display: true, text: 'Residual' } } } } });
        STATE.charts.push(c);
      }, 100);
    }

    // Coefficients with polarity coloring
    if (data.coefficients) {
      const canvasId = 'chart_coef';
      grid.innerHTML += chartCardHTMLWithAction('Coefficients (blue=+, red=−)', canvasId, 'chart_coef', 'Coefficients');
      const coef = data.coefficients;
      const sorted = Object.entries(coef).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 15);
      setTimeout(() => {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        const colors = sorted.map(e => parseFloat(e[1]) >= 0 ? '#2563ebcc' : '#dc2626cc');
        const c = new Chart(ctx, { type: 'bar',
          data: { labels: sorted.map(e => e[0].replace('num__','').replace('cat__','').substring(0, 22)),
            datasets: [{ data: sorted.map(e => parseFloat(e[1].toFixed(4))), backgroundColor: colors, borderRadius: 3, barThickness: 14 }] },
          options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false },
              tooltip: { callbacks: { label: ctx2 => ` ${ctx2.parsed.x.toFixed(4)} (${ctx2.parsed.x >= 0 ? '↑ increases' : '↓ decreases'} freight)` } } },
            scales: { x: { title: { display: true, text: 'Coefficient' } } } } });
        STATE.charts.push(c);
      }, 150);
    }
    setTable(Object.entries(m).map(([k, v]) => ({ Metric: k.toUpperCase(), Value: typeof v === 'number' ? v.toFixed(5) : v })));
  }
}

// ─── CLUSTERING ───────────────────────────────────────────────
function renderClustering(data) {
  const m = data.metrics;
  renderMetrics({
    silhouette_score: m.silhouette_score, n_clusters: m.n_clusters, wcss: m.wcss, sample_size: data.sample_size
  }, {
    silhouette_score: v => v.toFixed(4), n_clusters: v => v,
    wcss: v => v ? Math.round(v).toLocaleString() : 'N/A', sample_size: v => v ? v.toLocaleString() : 'N/A'
  });
  const grid = document.getElementById('chartsGrid');
  grid.innerHTML = '';

  // WCSS Elbow
  if (data.extra && data.extra.elbow) {
    const elbowCanvasId = 'chart_elbow';
    grid.innerHTML += chartCardHTMLWithAction('WCSS Elbow Curve — Optimal K Selection', elbowCanvasId, 'chart_elbow', 'WCSS_Elbow');
    const elbow = data.extra.elbow;
    setTimeout(() => {
      const ctx = document.getElementById(elbowCanvasId);
      if (!ctx) return;
      const c = new Chart(ctx, { type: 'line', data: { labels: Object.keys(elbow),
        datasets: [{ label: 'WCSS', data: Object.values(elbow), borderColor: '#dc2626', backgroundColor: '#dc262622',
          pointRadius: 6, pointBackgroundColor: '#dc2626', fill: true, tension: 0.3, borderWidth: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
          tooltip: { callbacks: { label: ctx2 => ` WCSS: ${Math.round(ctx2.parsed.y).toLocaleString()}` } } },
          scales: { x: { title: { display: true, text: 'K (Number of Clusters)' } }, y: { title: { display: true, text: 'WCSS' }, beginAtZero: false } } } });
      STATE.charts.push(c);
    }, 50);
  }

  // Silhouette across K
  if (data.extra && data.extra.silhouette_by_k) {
    const silCanvasId = 'chart_silhouette_k';
    grid.innerHTML += chartCardHTMLWithAction('Silhouette Score across K=2..7 (green=best)', silCanvasId, 'chart_silhouette_k', 'Silhouette_K');
    const silK = data.extra.silhouette_by_k;
    setTimeout(() => {
      const vals = Object.values(silK);
      const maxSil = Math.max(...vals);
      const ctx = document.getElementById(silCanvasId);
      if (!ctx) return;
      const c = new Chart(ctx, { type: 'bar', data: { labels: Object.keys(silK).map(k => `K=${k}`),
        datasets: [{ label: 'Silhouette', data: vals, backgroundColor: vals.map(v => v === maxSil ? '#059669cc' : '#2563eb88'), borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
          tooltip: { callbacks: { label: ctx2 => ` ${ctx2.parsed.y.toFixed(4)} (${ctx2.parsed.y === maxSil ? '★ best' : 'candidate'})` } } },
          scales: { y: { beginAtZero: true, max: 1, title: { display: true, text: 'Silhouette Score' } } } } });
      STATE.charts.push(c);
    }, 100);
  }

  // Cluster sizes
  const sizes = data.cluster_sizes || {};
  const sizeCanvasId = 'chart_csizes';
  grid.innerHTML += chartCardHTMLWithAction('Cluster Size Distribution', sizeCanvasId, 'chart_csizes', 'Cluster_Sizes');
  setTimeout(() => {
    const labels = Object.keys(sizes).map(k => `Cluster ${k}`);
    const vals = Object.values(sizes);
    const c = makeBarChart(sizeCanvasId, labels, vals, 'Customers', PALETTE.slice(0, labels.length));
    STATE.charts.push(c);
  }, 150);

  // PCA 2D Scatter
  if (data.scatter_data) {
    const sc = data.scatter_data;
    const pc1 = sc.pca_var ? (sc.pca_var[0]*100).toFixed(1) : '?';
    const pc2 = sc.pca_var ? (sc.pca_var[1]*100).toFixed(1) : '?';
    const scCanvasId = 'chart_scatter';
    grid.innerHTML += chartCardHTMLWithAction(`PCA Scatter — PC1 ${pc1}% + PC2 ${pc2}% variance`, scCanvasId, 'chart_scatter', 'PCA_Scatter');
    setTimeout(() => {
      const nCl = [...new Set(sc.cluster)].length;
      const datasets = [];
      for (let ci = 0; ci < nCl; ci++) {
        const pts = sc.x.map((x, i) => sc.cluster[i] === ci ? { x, y: sc.y[i] } : null).filter(Boolean);
        datasets.push({ label: `Cluster ${ci}`, data: pts, backgroundColor: PALETTE[ci] + 'bb', pointRadius: 2.5 });
      }
      const c = makeScatterMulti(scCanvasId, datasets);
      STATE.charts.push(c);
    }, 200);
  }

  // Agglomerative Hierarchical Dendrogram
  if (data.extra && (data.extra.dendrogram || data.extra.dendrogram_sample)) {
    const dendro = data.extra.dendrogram || {};
    const linkageMethod = (data.extra.linkage || 'Ward').toUpperCase();
    const nClusters = data.parameters?.n_clusters || dendro.n_clusters || 4;
    const cutDist = (dendro.cut_distance !== undefined && dendro.cut_distance !== null) ? dendro.cut_distance : 'N/A';
    window._lastDendroData = dendro;
    window._lastDendroK = nClusters;

    grid.innerHTML += `
      <div class="chart-card" id="dendrogram_card" style="grid-column: 1 / -1;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.85rem;flex-wrap:wrap;gap:0.5rem;">
          <div>
            <div class="chart-card-title">🌳 Hierarchical Clustering Dendrogram (${linkageMethod} Linkage)</div>
            <p style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">
              Bottom-up cluster merges. Red dashed line indicates the <strong>K = ${nClusters}</strong> cut threshold at merge distance <strong>d ≈ ${cutDist}</strong>.
            </p>
          </div>
          <div class="olap-view-toggle">
            <button class="olap-toggle-btn active" onclick="switchDendroView('Image', this)">🖼️ Publication Plot</button>
            <button class="olap-toggle-btn" onclick="switchDendroView('Canvas', this)">✨ Interactive Canvas</button>
            <button class="olap-toggle-btn" onclick="switchDendroView('MergeDist', this)">📊 Merge Distances</button>
          </div>
        </div>

        <!-- 1. Publication Plot View -->
        <div id="dendroImageView" style="text-align:center;padding:0.5rem 0;">
          <div style="background:#ffffff;border:1px solid var(--border);border-radius:var(--radius-md);padding:0.75rem;display:inline-block;max-width:100%;box-shadow:var(--shadow-sm);">
            <img id="dendroImgElement" src="${dendro.image || '/static/academic_results/clustering/agglomerative_dendrogram.png'}" 
                 alt="Hierarchical Clustering Dendrogram" 
                 style="max-width:100%;height:auto;border-radius:var(--radius-sm);display:block;margin:0 auto;" />
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.6rem;padding:0 0.5rem;font-size:0.75rem;color:var(--text-muted);">
              <span>${dendro.note || 'Hierarchical clustering tree with cluster cut-off line'}</span>
              <a href="${dendro.image || '/static/academic_results/clustering/agglomerative_dendrogram.png'}" download="hierarchical_dendrogram.png" class="chart-action-btn" style="text-decoration:none;">💾 Save Plot</a>
            </div>
          </div>
        </div>

        <!-- 2. Interactive Canvas View -->
        <div id="dendroCanvasView" class="hidden">
          <div style="background:#ffffff;border:1px solid var(--border);border-radius:var(--radius-md);padding:1rem;position:relative;">
            <div style="height:380px;position:relative;">
              <canvas id="chart_dendrogram_tree" style="width:100%;height:100%;display:block;"></canvas>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.6rem;font-size:0.75rem;color:var(--text-muted);">
              <span>Interactive vector tree (truncated to top ${dendro.p || 30} leaf nodes)</span>
              <span style="color:#ef4444;font-weight:600;">── Cut Line: K=${nClusters} (d=${cutDist})</span>
            </div>
          </div>
        </div>

        <!-- 3. Scree / Merge Distances View -->
        <div id="dendroMergeDistView" class="hidden">
          <div style="height:300px;">
            <canvas id="chart_dendrogram_distances"></canvas>
          </div>
        </div>
      </div>
    `;

    setTimeout(() => {
      if (dendro.icoord && dendro.dcoord) {
        drawInteractiveDendrogram('chart_dendrogram_tree', dendro, nClusters);
      }

      const dists = dendro.top_distances || data.extra.dendrogram_sample || [];
      const distCtx = document.getElementById('chart_dendrogram_distances');
      if (distCtx && dists.length) {
        const c = new Chart(distCtx, {
          type: 'bar',
          data: {
            labels: dists.map((_, i) => `Merge ${i + 1}`),
            datasets: [{
              label: 'Linkage Merge Distance',
              data: dists,
              backgroundColor: dists.map(d => (cutDist !== 'N/A' && d >= cutDist) ? '#ef4444cc' : '#7c3aedaa'),
              borderColor: dists.map(d => (cutDist !== 'N/A' && d >= cutDist) ? '#ef4444' : '#7c3aed'),
              borderWidth: 1.5,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx2 => ` Distance: ${ctx2.parsed.y} ${cutDist !== 'N/A' && ctx2.parsed.y >= cutDist ? '(Above K Cutoff)' : ''}`
                }
              }
            },
            scales: {
              y: { beginAtZero: true, title: { display: true, text: 'Linkage Distance' } },
              x: { title: { display: true, text: 'Consecutive Merges (Scree Step)' }, ticks: { font: { size: 10 } } }
            }
          }
        });
        STATE.charts.push(c);
      }
    }, 120);
  }

  setTable(data.cluster_profiles || []);
}

// ─── ASSOCIATION RULES ────────────────────────────────────────
function renderAssociation(data) {
  renderMetrics({
    total_rules: data.metrics?.n_rules, total_itemsets: data.metrics?.n_itemsets, transactions: data.metrics?.transactions
  }, { total_rules: v => v, total_itemsets: v => v, transactions: v => v?.toLocaleString() });

  const grid = document.getElementById('chartsGrid');
  grid.innerHTML = '';

  if (data.no_rules || !data.rules?.length) {
    grid.innerHTML = `<div class="chart-card" style="grid-column:1/-1;"><div style="text-align:center;padding:2rem;color:#64748b;"><div style="font-size:2rem;">🔍</div>
      <p><strong>No association rules found</strong> with the current thresholds.</p>
      <p style="margin-top:0.5rem;font-size:0.82rem;">Try lowering minimum support or confidence. Olist dataset is inherently sparse (~0.75% multi-category orders).</p>
      ${data.frequent_itemsets?.length ? `<p style="margin-top:0.5rem;font-size:0.82rem;color:#2563eb;">✅ ${data.frequent_itemsets.length} frequent itemsets found — lower min_confidence to generate rules.</p>` : ''}</div></div>`;
  } else {
    // Support vs Confidence bubble chart (size/color = lift)
    const canvasId1 = 'chart_sup_conf';
    const maxLift = Math.max(...data.rules.map(r => r.lift));
    grid.innerHTML += chartCardHTMLWithAction('Support vs Confidence (bubble size = lift)', canvasId1, 'chart_sup_conf', 'Support_Confidence');
    setTimeout(() => {
      const ctx = document.getElementById(canvasId1);
      if (!ctx) return;
      const pts = data.rules.map(r => ({ x: r.support, y: r.confidence, r: Math.max(4, Math.min(18, (r.lift / maxLift) * 18)) }));
      const cols = data.rules.map(r => `hsl(${Math.round(250 - (r.lift / maxLift) * 200)},70%,55%)`);
      const c = new Chart(ctx, { type: 'bubble',
        data: { datasets: [{ label: 'Rules (bigger=higher lift)', data: pts, backgroundColor: cols.map(c2 => c2 + 'aa'), borderColor: cols }] },
        options: { responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 10 } } },
            tooltip: { callbacks: { label: ctx2 => { const idx = ctx2.dataIndex; const r = data.rules[idx]; return r ? [`Support: ${r.support.toFixed(5)}`, `Confidence: ${r.confidence.toFixed(3)}`, `Lift: ${r.lift.toFixed(3)}`] : []; } } } },
          scales: { x: { title: { display: true, text: 'Support' } }, y: { title: { display: true, text: 'Confidence' } } } } });
      STATE.charts.push(c);
    }, 50);

    // Top rules lift bar
    const canvasId2 = 'chart_lift';
    const top = data.rules.slice(0, 12);
    grid.innerHTML += chartCardHTMLWithAction('Top Rules by Lift', canvasId2, 'chart_lift', 'Top_Rules_Lift');
    setTimeout(() => {
      const labels = top.map(r => `${r.antecedents.join('+')}→${r.consequents.join('+')}`).map(l => l.length > 30 ? l.substring(0, 28) + '…' : l);
      const c = makeBarChart(canvasId2, labels, top.map(r => r.lift), 'Lift', top.map((_, i) => `hsl(${250-i*15},70%,55%)`));
      STATE.charts.push(c);
    }, 100);

    // Rule network visual
    const networkEdges = Array.isArray(data.rule_network) ? data.rule_network : (data.rule_network?.edges || []);
    if (networkEdges.length > 0) {
      const networkTop = networkEdges.slice(0, 12);
      grid.innerHTML += `<div class="chart-card" style="grid-column:1/-1;">
        <div class="chart-card-title" style="margin-bottom:0.7rem;">Rule Network — Antecedent → Consequent (top ${networkTop.length} by lift)</div>
        <div class="rule-network-container">${networkTop.map(edge => `
          <div class="rule-network-card">
            <span style="color:#059669;font-weight:700;">${edge.source || edge.antecedent || '?'}</span>
            <span class="rule-network-arrow">→</span>
            <span style="color:#2563eb;font-weight:700;">${edge.target || edge.consequent || '?'}</span>
            <div style="font-size:0.68rem;color:#64748b;margin-top:3px;">Lift: <strong>${(edge.lift||0).toFixed(2)}</strong> · Conf: ${(edge.confidence||0).toFixed(2)}</div>
          </div>`).join('')}
        </div>
      </div>`;
    }
  }

  // Frequent itemsets bar
  if (data.frequent_itemsets && data.frequent_itemsets.length) {
    const canvasId3 = 'chart_itemsets';
    grid.innerHTML += chartCardHTMLWithAction('Top Frequent Itemsets by Support', canvasId3, 'chart_itemsets', 'Frequent_Itemsets');
    const topSets = data.frequent_itemsets.slice(0, 15);
    setTimeout(() => {
      const labels = topSets.map(it => (Array.isArray(it.itemsets) ? it.itemsets.join(' + ') : String(it.itemsets))).map(l => l.length > 28 ? l.substring(0, 26) + '…' : l);
      const c = makeBarChart(canvasId3, labels, topSets.map(it => it.support), 'Support', PALETTE.slice(0, labels.length));
      STATE.charts.push(c);
    }, 150);
  }

  setTable((data.rules || []).map(r => ({ antecedents: r.antecedents.join(' + '), consequents: r.consequents.join(' + '), support: r.support.toFixed(6), confidence: r.confidence.toFixed(4), lift: r.lift.toFixed(4) })));
}

// ─── OLAP ─────────────────────────────────────────────────────
function renderOlap(data) {
  const m = data.metrics;
  renderMetrics({
    total_orders: m.total_orders, total_sales: m.total_sales, total_customers: m.total_customers, records: m.records_returned
  }, { total_orders: v => int0(v), total_sales: v => `R$ ${num2(v)}`, total_customers: v => int0(v), records: v => int0(v) });

  const rawData = data.data || [];
  const grid = document.getElementById('chartsGrid');
  grid.innerHTML = '';

  if (!rawData.length) {
    grid.innerHTML = `<div class="chart-card" style="grid-column:1/-1;text-align:center;padding:2rem;color:#64748b;">No data returned for these filters.</div>`;
  } else if (data.model_key === 'pivot' || (data.operation && data.operation.toLowerCase().includes('pivot'))) {
    renderPivotOlap(data, rawData, grid);
  } else {
    const allKeys = Object.keys(rawData[0]);
    const dimKey = allKeys.find(k => !['order_count','customer_count','item_count','total_sales','avg_order_value'].includes(k)) || allKeys[0];
    const mKeys = ['order_count','total_sales','customer_count'].filter(k => allKeys.includes(k));
    const dimVals = rawData.map(r => r[dimKey]);
    const isNumericDim = dimVals.every(v => !isNaN(v));

    grid.innerHTML += `<div class="chart-card" style="grid-column:1/-1;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;flex-wrap:wrap;gap:0.5rem;">
        <div class="chart-card-title">${(data.operation||'OLAP').toUpperCase()} — ${dimKey} Dimension</div>
        <div class="olap-view-toggle">
          <button class="olap-toggle-btn active" onclick="switchOlapView('Chart', this)">📊 Chart</button>
          <button class="olap-toggle-btn" onclick="switchOlapView('Table', this)">📋 Table</button>
          <button class="olap-toggle-btn" onclick="switchOlapView('Query', this)">🔤 Query Logic</button>
        </div>
      </div>
      <div id="olapChartView">${mKeys[0] ? '<div style="height:280px;"><canvas id="chart_olap_main"></canvas></div>' : ''}
        ${allKeys.includes('total_sales') && isNumericDim ? '<div style="height:240px;margin-top:1rem;"><canvas id="chart_olap_trend"></canvas></div>' : ''}
      </div>
      <div id="olapTableView" class="hidden"><div class="table-wrapper" style="max-height:400px;overflow:auto;"><table class="data-table">
        <thead><tr>${allKeys.map(k => `<th>${k}</th>`).join('')}</tr></thead>
        <tbody>${rawData.slice(0,50).map(r => `<tr>${allKeys.map(k => `<td>${r[k]??''}</td>`).join('')}</tr>`).join('')}</tbody>
      </table></div></div>
      <div id="olapQueryView" class="hidden">${data.query_representation ? `<pre class="query-code">${escHtml(data.query_representation)}</pre>` : '<em style="color:#64748b;">No query representation available.</em>'}</div>
    </div>`;

    if (mKeys[0]) {
      setTimeout(() => {
        const ctx = document.getElementById('chart_olap_main');
        if (!ctx) return;
        const labels = rawData.slice(0, 25).map(r => String(r[dimKey]));
        const vals = rawData.slice(0, 25).map(r => r[mKeys[0]] || 0);
        const c = new Chart(ctx, { type: 'bar', data: { labels, datasets: [{ label: mKeys[0].replace('_',' '), data: vals, backgroundColor: PALETTE[0]+'cc', borderRadius: 4 }] },
          options: { responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx2 => ` ${mKeys[0].replace('_',' ')}: ${ctx2.parsed.y.toLocaleString()}` } } },
            scales: { y: { beginAtZero: true, title: { display: true, text: mKeys[0].replace('_',' ') } }, x: { ticks: { maxRotation: 40, font: { size: 10 } } } } } });
        STATE.charts.push(c);
      }, 50);
    }
    if (allKeys.includes('total_sales') && isNumericDim) {
      setTimeout(() => {
        const c2 = makeLineChartXY('chart_olap_trend',
          rawData.slice(0, 20).map(r => String(r[dimKey])), rawData.slice(0, 20).map(r => r.total_sales || 0),
          'Total Sales (BRL)', '#059669', dimKey, 'Sales (BRL)');
        if (c2) STATE.charts.push(c2);
      }, 100);
    }
  }
  setTable(rawData);
}

function renderPivotOlap(data, rawData, grid) {
  const allKeys = Object.keys(rawData[0]);
  const rowDimKey = allKeys[0];
  const colKeys = allKeys.filter(k => k !== rowDimKey && k !== 'Total');
  const hasTotal = allKeys.includes('Total');
  const measure = data.measure || (data.parameters && data.parameters.measure) || 'Measure';
  const isCurrency = measure === 'total_sales' || measure.includes('sales');

  const formatVal = (v) => {
    if (v === null || v === undefined) return '0';
    if (typeof v === 'number') {
      return isCurrency ? `R$ ${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : v.toLocaleString();
    }
    return String(v);
  };

  const topRows = rawData.slice(0, 15);

  grid.innerHTML = `
    <div class="chart-card" style="grid-column:1/-1;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;flex-wrap:wrap;gap:0.5rem;">
        <div>
          <div class="chart-card-title">🔄 PIVOT / ROTATE — ${rowDimKey.toUpperCase()} (Rows) × ${(data.parameters?.col_dim || 'COLUMNS').toUpperCase()}</div>
          <p style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">Metric: <strong>${measure.replace('_', ' ')}</strong> aggregated across rotated dimensions</p>
        </div>
        <div class="olap-view-toggle">
          <button class="olap-toggle-btn active" onclick="switchOlapView('Chart', this)">📊 Grouped Chart</button>
          <button class="olap-toggle-btn" onclick="switchOlapView('Table', this)">📋 Cross-Tab Pivot Table</button>
          <button class="olap-toggle-btn" onclick="switchOlapView('Query', this)">🔤 SQL / CrossTab Query</button>
        </div>
      </div>

      <!-- Chart View -->
      <div id="olapChartView">
        <div style="height:320px;">
          <canvas id="chart_olap_pivot_main"></canvas>
        </div>
        ${hasTotal ? `
        <div style="height:220px;margin-top:1.25rem;">
          <canvas id="chart_olap_pivot_total"></canvas>
        </div>` : ''}
      </div>

      <!-- Pivot Table View -->
      <div id="olapTableView" class="hidden">
        <div class="table-wrapper" style="max-height:480px;overflow:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th style="background:var(--bg-subtle);font-weight:700;position:sticky;left:0;z-index:2;">${rowDimKey.toUpperCase()}</th>
                ${colKeys.map(c => `<th>${c}</th>`).join('')}
                ${hasTotal ? `<th style="background:rgba(5,150,105,0.1);color:#059669;font-weight:700;">TOTAL</th>` : ''}
              </tr>
            </thead>
            <tbody>
              ${rawData.map(r => `
                <tr>
                  <td style="font-weight:600;background:var(--bg-subtle);position:sticky;left:0;z-index:1;">${r[rowDimKey]}</td>
                  ${colKeys.map(c => `<td>${formatVal(r[c])}</td>`).join('')}
                  ${hasTotal ? `<td style="font-weight:700;background:rgba(5,150,105,0.06);color:#059669;">${formatVal(r['Total'])}</td>` : ''}
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Query View -->
      <div id="olapQueryView" class="hidden">
        ${data.query_representation ? `<pre class="query-code">${escHtml(data.query_representation)}</pre>` : '<em style="color:#64748b;">No query representation available.</em>'}
      </div>
    </div>
  `;

  setTimeout(() => {
    const ctx = document.getElementById('chart_olap_pivot_main');
    if (!ctx) return;

    const labels = topRows.map(r => String(r[rowDimKey]));
    const datasets = colKeys.map((cKey, idx) => ({
      label: String(cKey),
      data: topRows.map(r => Number(r[cKey]) || 0),
      backgroundColor: PALETTE[idx % PALETTE.length] + 'cc',
      borderColor: PALETTE[idx % PALETTE.length],
      borderWidth: 1.5,
      borderRadius: 4
    }));

    const chart = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: (ctx2) => ` ${ctx2.dataset.label}: ${isCurrency ? 'R$ ' + ctx2.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2}) : ctx2.parsed.y.toLocaleString()}`
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: measure.replace('_', ' ') }
          },
          x: {
            ticks: { maxRotation: 40, font: { size: 10 } }
          }
        }
      }
    });
    STATE.charts.push(chart);

    if (hasTotal) {
      const ctxTotal = document.getElementById('chart_olap_pivot_total');
      if (ctxTotal) {
        const totalChart = new Chart(ctxTotal, {
          type: 'bar',
          data: {
            labels,
            datasets: [{
              label: `Total ${measure.replace('_', ' ')}`,
              data: topRows.map(r => Number(r['Total']) || 0),
              backgroundColor: '#059669cc',
              borderColor: '#059669',
              borderWidth: 1.5,
              borderRadius: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx2) => ` Total: ${isCurrency ? 'R$ ' + ctx2.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2}) : ctx2.parsed.y.toLocaleString()}`
                }
              }
            },
            scales: {
              y: { beginAtZero: true, title: { display: true, text: `Total ${measure.replace('_', ' ')}` } },
              x: { ticks: { maxRotation: 40, font: { size: 10 } } }
            }
          }
        });
        STATE.charts.push(totalChart);
      }
    }
  }, 60);
}

function switchOlapView(view, btn) {
  ['olapChartView','olapTableView','olapQueryView'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });
  const el = document.getElementById(`olap${view}View`);
  if (el) el.classList.remove('hidden');
  const btns = btn?.closest('.olap-view-toggle')?.querySelectorAll('.olap-toggle-btn');
  if (btns) btns.forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

function switchDendroView(view, btn) {
  ['dendroImageView', 'dendroCanvasView', 'dendroMergeDistView'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });
  const target = document.getElementById(`dendro${view}View`);
  if (target) target.classList.remove('hidden');

  const btns = btn?.closest('.olap-view-toggle')?.querySelectorAll('.olap-toggle-btn');
  if (btns) btns.forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  if (view === 'Canvas' && window._lastDendroData) {
    setTimeout(() => {
      drawInteractiveDendrogram('chart_dendrogram_tree', window._lastDendroData, window._lastDendroK || 4);
    }, 40);
  }
}

function drawInteractiveDendrogram(canvasId, dendro, nClusters) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !dendro || !dendro.icoord || !dendro.dcoord) return;

  const parent = canvas.parentElement;
  const width = Math.floor(parent.clientWidth) || 750;
  const height = Math.floor(parent.clientHeight) || 380;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const padL = 60, padR = 30, padT = 30, padB = 65;
  const plotW = Math.max(10, width - padL - padR);
  const plotH = Math.max(10, height - padT - padB);

  const allX = dendro.icoord.flat();
  const allY = dendro.dcoord.flat();
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const maxY = (Math.max(...allY) || 1) * 1.08;

  const toX = x => padL + ((x - minX) / (maxX - minX || 1)) * plotW;
  const toY = y => padT + (1 - (y / maxY)) * plotH;

  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = '#fafafa';
  ctx.fillRect(padL, padT, plotW, plotH);

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  const numYTicks = 6;
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#64748b';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (let i = 0; i <= numYTicks; i++) {
    const val = (maxY / numYTicks) * i;
    const yPos = toY(val);
    ctx.beginPath();
    ctx.setLineDash([2, 3]);
    ctx.moveTo(padL, yPos);
    ctx.lineTo(padL + plotW, yPos);
    ctx.stroke();

    ctx.fillText(val.toFixed(1), padL - 8, yPos);
  }
  ctx.setLineDash([]);

  ctx.save();
  ctx.translate(16, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = '600 11px sans-serif';
  ctx.fillStyle = '#475569';
  ctx.textAlign = 'center';
  ctx.fillText('Linkage Distance', 0, 0);
  ctx.restore();

  const colorMap = {
    'C0': '#2563eb', 'C1': '#ea580c', 'C2': '#16a34a', 'C3': '#dc2626',
    'C4': '#9333ea', 'C5': '#854d0e', 'C6': '#db2777', 'C7': '#475569',
    'b': '#2563eb',  'g': '#16a34a',  'r': '#dc2626',  'c': '#0891b2',
    'm': '#c026d3',  'y': '#ca8a04',  'k': '#1e293b'
  };

  dendro.icoord.forEach((branchX, idx) => {
    const branchY = dendro.dcoord[idx];
    const rawColor = (dendro.color_list && dendro.color_list[idx]) || '#475569';
    const strokeColor = colorMap[rawColor] || (rawColor.startsWith('#') ? rawColor : '#475569');

    ctx.beginPath();
    ctx.moveTo(toX(branchX[0]), toY(branchY[0]));
    ctx.lineTo(toX(branchX[1]), toY(branchY[1]));
    ctx.lineTo(toX(branchX[2]), toY(branchY[2]));
    ctx.lineTo(toX(branchX[3]), toY(branchY[3]));
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.8;
    ctx.stroke();
  });

  if (dendro.cut_distance !== undefined && dendro.cut_distance !== null) {
    const cutY = toY(dendro.cut_distance);
    if (cutY >= padT && cutY <= padT + plotH) {
      ctx.beginPath();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.moveTo(padL, cutY);
      ctx.lineTo(padL + plotW, cutY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`K=${nClusters} Cut (d ≈ ${dendro.cut_distance})`, padL + plotW - 6, cutY - 6);
    }
  }

  if (dendro.ivl && dendro.ivl.length) {
    const leafXPositions = [];
    dendro.icoord.forEach((branchX, idx) => {
      if (dendro.dcoord[idx][0] === 0) leafXPositions.push(branchX[0]);
      if (dendro.dcoord[idx][3] === 0) leafXPositions.push(branchX[3]);
    });
    const sortedLeafX = [...new Set(leafXPositions)].sort((a, b) => a - b);

    ctx.font = '9px sans-serif';
    ctx.fillStyle = '#334155';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    dendro.ivl.forEach((lbl, i) => {
      const xVal = sortedLeafX[i] !== undefined ? sortedLeafX[i] : (minX + (i + 0.5) * ((maxX - minX) / dendro.ivl.length));
      const screenX = toX(xVal);
      const screenY = padT + plotH + 8;

      ctx.save();
      ctx.translate(screenX, screenY);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(String(lbl), 0, 0);
      ctx.restore();
    });
  }

  ctx.strokeStyle = '#cbd5e1';
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, plotW, plotH);
}

// ─── ACCORDION HELPERS ────────────────────────────────────────
function renderDatasetAccordion(data) {
  const el = document.getElementById('accDataset');
  const fmt = v => typeof v === 'number' ? v.toLocaleString() : (v || 'N/A');
  el.innerHTML = `<div class="detail-grid">
    <div class="detail-item"><div class="detail-item-label">Dataset</div><div class="detail-item-value">${data.dataset||'Olist'}</div></div>
    <div class="detail-item"><div class="detail-item-label">Total Rows</div><div class="detail-item-value">${fmt(data.dataset_rows || data.transactions)}</div></div>
    <div class="detail-item"><div class="detail-item-label">Training Samples</div><div class="detail-item-value">${fmt(data.train_samples || data.sample_size)}</div></div>
    <div class="detail-item"><div class="detail-item-label">Test Samples</div><div class="detail-item-value">${fmt(data.test_samples)}</div></div>
    ${data.features ? `<div class="detail-item" style="grid-column:1/-1;"><div class="detail-item-label">Features</div><div class="detail-item-value" style="font-size:0.75rem;word-break:break-all;">${(data.features||[]).join(', ')}</div></div>` : ''}
    ${data.used_sampling ? `<div class="detail-item" style="grid-column:1/-1;background:#fffbeb;border:1px solid #fde68a;"><div class="detail-item-label">⚠ Sampling</div><div class="detail-item-value" style="font-size:0.8rem;">Sampling was used for scalability. Full dataset has more rows.</div></div>` : ''}
  </div>`;
}

function renderParamsAccordion(data) {
  const el = document.getElementById('accParams');
  const params = data.parameters || {};
  if (!Object.keys(params).length) { el.innerHTML = '<em>Default parameters used.</em>'; return; }
  el.innerHTML = `<div class="detail-grid">` +
    Object.entries(params).map(([k,v]) =>
      `<div class="detail-item"><div class="detail-item-label">${k}</div><div class="detail-item-value">${Array.isArray(v) ? v.join(', ') : v}</div></div>`
    ).join('') + `</div>`;
}

function renderQueryAccordion(data) {
  const el = document.getElementById('accQuery');
  if (data.query_representation) {
    el.innerHTML = `<p style="font-size:0.72rem;color:#64748b;margin-bottom:0.5rem;font-style:italic;">Analytical Query Representation — for educational purposes</p>
      <pre class="query-code">${escHtml(data.query_representation)}</pre>`;
  } else {
    el.innerHTML = '<em>No query representation available.</em>';
  }
}

function renderFeaturesAccordion(data) {
  const el = document.getElementById('accFeatures');
  const fi = data.feature_importance || data.coefficients;
  if (!fi) { el.innerHTML = '<em>Not applicable for this model.</em>'; return; }
  const sorted = Object.entries(fi).sort((a,b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0,15);
  const maxVal = Math.max(...sorted.map(e => Math.abs(e[1])));
  el.innerHTML = `<div class="feat-bar-list">` +
    sorted.map(([name, val]) => {
      const pct = maxVal > 0 ? (Math.abs(val) / maxVal * 100).toFixed(1) : 0;
      return `<div class="feat-bar-item">
        <div class="feat-bar-name">${name}</div>
        <div class="feat-bar-track"><div class="feat-bar-fill" style="width:${pct}%"></div></div>
        <div class="feat-bar-val">${parseFloat(val).toFixed(4)}</div>
      </div>`;
    }).join('') + `</div>`;
}

function toggleAccordion(btn) {
  const body = btn.nextElementSibling;
  const icon = btn.querySelector('.acc-icon');
  body.classList.toggle('hidden');
  icon.textContent = body.classList.contains('hidden') ? '▼' : '▲';
}

// ─── TABLE ────────────────────────────────────────────────────
function setTable(rows) {
  STATE.tableData     = rows || [];
  STATE.tableFiltered = [...STATE.tableData];
  STATE.tablePage     = 1;
  STATE.tableSortCol  = null;
  document.getElementById('tableSearch').value = '';
  renderTable();
}

function filterTable() {
  const q = document.getElementById('tableSearch').value.toLowerCase();
  if (!q) {
    STATE.tableFiltered = [...STATE.tableData];
  } else {
    STATE.tableFiltered = STATE.tableData.filter(row =>
      Object.values(row).some(v => String(v).toLowerCase().includes(q))
    );
  }
  STATE.tablePage = 1;
  renderTable();
}

function sortTable(col) {
  if (STATE.tableSortCol === col) {
    STATE.tableSortAsc = !STATE.tableSortAsc;
  } else {
    STATE.tableSortCol = col;
    STATE.tableSortAsc = true;
  }
  STATE.tableFiltered.sort((a, b) => {
    const av = a[col], bv = b[col];
    if (typeof av === 'number') return STATE.tableSortAsc ? av - bv : bv - av;
    return STATE.tableSortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });
  renderTable();
}

function renderTable() {
  const head = document.getElementById('tableHead');
  const body = document.getElementById('tableBody');
  const pag  = document.getElementById('tablePagination');
  const rows  = STATE.tableFiltered;

  if (!rows.length) {
    head.innerHTML = '';
    body.innerHTML = '<tr><td colspan="99" style="text-align:center;color:#64748b;padding:1rem;">No data to display.</td></tr>';
    pag.innerHTML  = '';
    return;
  }

  const cols = Object.keys(rows[0]);
  const start = (STATE.tablePage - 1) * STATE.tablePageSize;
  const pageRows = rows.slice(start, start + STATE.tablePageSize);
  const totalPages = Math.ceil(rows.length / STATE.tablePageSize);

  head.innerHTML = `<tr>${cols.map(c =>
    `<th onclick="sortTable('${c}')">${c.replace(/_/g,' ')} ${STATE.tableSortCol===c ? (STATE.tableSortAsc?'↑':'↓') : ''}</th>`
  ).join('')}</tr>`;

  body.innerHTML = pageRows.map(row =>
    `<tr>${cols.map(c => {
      const v = row[c];
      const fmt = typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : parseFloat(v.toFixed(5))) : (v ?? '');
      return `<td>${fmt}</td>`;
    }).join('')}</tr>`
  ).join('');

  // Pagination
  let pagHtml = '';
  for (let p = 1; p <= Math.min(totalPages, 10); p++) {
    pagHtml += `<button class="page-btn ${p===STATE.tablePage?'active':''}" onclick="goPage(${p})">${p}</button>`;
  }
  if (totalPages > 10) pagHtml += `<span class="page-info">… of ${totalPages} pages</span>`;
  pagHtml += `<span class="page-info">${rows.length} rows</span>`;
  pag.innerHTML = pagHtml;
}

function goPage(p) {
  STATE.tablePage = p;
  renderTable();
}

function exportCSV() {
  if (!STATE.tableFiltered.length) return;
  const cols = Object.keys(STATE.tableFiltered[0]);
  const csv  = [cols.join(','), ...STATE.tableFiltered.map(r =>
    cols.map(c => JSON.stringify(r[c] ?? '')).join(',')
  )].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = `academic_result_${Date.now()}.csv`; a.click();
  URL.revokeObjectURL(url);
}

// ─── HISTORY ──────────────────────────────────────────────────
function addToHistory(type, model, params, result) {
  const entry = { type, model, params, result, ts: new Date() };
  STATE.analysisHistory.unshift(entry);
  if (STATE.analysisHistory.length > 10) STATE.analysisHistory.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('historyList');
  if (!STATE.analysisHistory.length) {
    list.innerHTML = '<div class="history-empty">No analyses yet</div>';
    return;
  }
  list.innerHTML = STATE.analysisHistory.map((e, i) => {
    const metric = getKeyMetric(e.result);
    return `<div class="history-item" onclick="reloadHistory(${i})">
      <div class="history-item-title">${e.result.model || e.model}</div>
      <div class="history-item-meta">${e.type} · ${metric} · ${e.ts.toLocaleTimeString()}</div>
    </div>`;
  }).join('');
}

function getKeyMetric(result) {
  const m = result.metrics;
  if (!m) return '';
  if (m.accuracy !== undefined) return `Acc ${(m.accuracy*100).toFixed(1)}%`;
  if (m.r2 !== undefined) return `R² ${m.r2.toFixed(3)}`;
  if (m.silhouette_score !== undefined) return `Sil ${m.silhouette_score.toFixed(3)}`;
  if (m.n_rules !== undefined) return `${m.n_rules} rules`;
  if (m.total_orders !== undefined) return `${m.total_orders?.toLocaleString()} orders`;
  return '';
}

function reloadHistory(i) {
  const e = STATE.analysisHistory[i];
  if (e) {
    STATE.currentResult = e.result;
    renderResult(e.result);
    switchLabTab('lab');
    updateBreadcrumb(`History / ${e.result.model || e.model}`);
  }
}

// ─── RESET ────────────────────────────────────────────────────
function resetLab() {
  const aType = document.getElementById('analysisType');
  if (aType) aType.value = '';
  const mSel = document.getElementById('modelSelect');
  if (mSel) {
    mSel.innerHTML = '<option value="">— Select Algorithm —</option>';
    mSel.disabled = true;
  }
  const badge = document.getElementById('datasetBadge');
  if (badge) badge.textContent = 'Select an analysis module';
  const pPanel = document.getElementById('paramsPanel');
  if (pPanel) pPanel.classList.add('hidden');
  const olapPanel = document.getElementById('olapFiltersPanel');
  if (olapPanel) olapPanel.classList.add('hidden');
  const rBtn = document.getElementById('runBtn');
  if (rBtn) rBtn.disabled = true;

  STATE.charts.forEach(c => { try { c.destroy(); } catch(e){} });
  STATE.charts = [];
  hide('resultArea'); hide('loadingState'); hide('errorState');
  show('idleState');
  updateBreadcrumb('Analysis Laboratory');
}

// ─── LIVE SCHEMA EXPLORER (Redesigned Dimensional DW & ERD Engine) ───────
STATE.schemaData = null;
STATE.currentSchemaView = 'star';
STATE.schemaZoom = {
  star: { scale: 1.0, panX: 0, panY: 0 },
  snowflake: { scale: 0.90, panX: 0, panY: 0 },
  constellation: { scale: 0.92, panX: 0, panY: 0 }
};
STATE.schemaTableCollapsed = {
  star: {},
  snowflake: {},
  constellation: {}
};

async function loadLiveSchema(targetSchema = 'star') {
  if (STATE.schemaData) {
    if (targetSchema) showSchema(targetSchema);
    return;
  }

  const starEl = document.getElementById('starSchemaContent');
  const snowEl = document.getElementById('snowflakeSchemaContent');
  const constEl = document.getElementById('constellationSchemaContent');
  const compEl = document.getElementById('comparisonSchemaContent');

  [starEl, snowEl, constEl, compEl].forEach(el => {
    if (el) el.innerHTML = `<div class="state-panel" style="padding:2.5rem;"><div class="loading-spinner"></div><p>Introspecting physical tables &amp; relational keys from live Olist dataset...</p></div>`;
  });

  try {
    const res = await fetch('/api/academic/schema');
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    STATE.schemaData = data;
    renderLiveStarSchema(data);
    renderLiveSnowflakeSchema(data);
    renderLiveConstellationSchema(data);
    renderLiveComparisonSchema(data);

    if (targetSchema) showSchema(targetSchema);
  } catch (err) {
    console.error('Failed to load live schema:', err);
    [starEl, snowEl, constEl, compEl].forEach(el => {
      if (el) el.innerHTML = `<div class="state-panel error-state"><div class="state-icon">⚠️</div><p>Failed to load live schema: ${err.message}</p></div>`;
    });
  }
}

function showSchema(schema, btn) {
  STATE.currentSchemaView = schema;
  document.querySelectorAll('.schema-view').forEach(v => { v.classList.add('hidden'); v.classList.remove('active-schema'); });
  document.querySelectorAll('.schema-tab').forEach(b => b.classList.remove('active'));
  
  const el = document.getElementById(`schema-${schema}`);
  if (el) { el.classList.remove('hidden'); el.classList.add('active-schema'); }
  if (btn) btn.classList.add('active');

  // Control toolbar & legend visibility
  const tb = document.getElementById('schemaToolbar');
  const lb = document.getElementById('schemaLegendBar');
  const subdimItem = document.getElementById('legendSubdimItem');

  if (schema === 'comparison') {
    if (tb) tb.style.display = 'none';
    if (lb) lb.style.display = 'none';
  } else {
    if (tb) tb.style.display = 'flex';
    if (lb) lb.style.display = 'flex';
    if (subdimItem) subdimItem.style.display = (schema === 'snowflake') ? 'inline-flex' : 'none';
    updateSchemaZoomBadge(schema);
  }

  if (!STATE.schemaData) {
    loadLiveSchema(schema);
  } else {
    // Redraw arrows after view switch paint
    requestAnimationFrame(() => {
      redrawActiveSchemaArrows();
    });
  }
}

function updateSchemaZoomBadge(schema) {
  const badge = document.getElementById('schemaZoomVal');
  if (!badge) return;
  const zoom = STATE.schemaZoom[schema] || { scale: 1.0 };
  badge.textContent = `${Math.round(zoom.scale * 100)}%`;
}

// ── ERD HELPER: build a dtype short label ──
function dtypeShort(d = '') {
  if (!d) return 'Varchar';
  if (d.startsWith('int'))   return 'Integer';
  if (d.startsWith('float')) return 'Float';
  if (d === 'object')        return 'Varchar(50)';
  if (d.startsWith('date') || d.includes('datetime')) return 'Timestamp';
  if (d === 'bool')          return 'Boolean';
  return d.substring(0, 10);
}

// Check if a column in a specific table represents an additive measure
function isMeasureCol(tableName, colName) {
  const c = (colName || '').toLowerCase();
  if (c === 'price' || c === 'freight_value' || c === 'payment_value' || c === 'review_score') return true;
  if (c === 'payment_installments' || c === 'payment_sequential') return true;
  if (tableName === 'olist_order_items' && c === 'order_item_id') return true;
  return false;
}

// Format column key/measure badge
function getColBadge(tableName, col) {
  const isMeasure = isMeasureCol(tableName, col.name);
  if (isMeasure) return { label: 'MEASURE', cls: 'badge-measure' };
  const kt = (col.key_type || '').toUpperCase().trim();
  if (kt === 'PK FK' || kt === 'PKFK') return { label: 'PK FK', cls: 'badge-pkfk' };
  if (kt === 'PK') return { label: 'PK', cls: 'badge-pk' };
  if (kt === 'FK') return { label: 'FK', cls: 'badge-fk' };
  return { label: '', cls: 'badge-none' };
}

// ── ERD HELPER: build one Expandable ERD table box (returns HTML string) ──
function erdTableHTML(schemaId, key, table, role = 'dimension', left = 0, top = 0) {
  if (!table) return '';
  const domId = `erd-${schemaId}-${key}`;
  const isCollapsed = STATE.schemaTableCollapsed[schemaId]?.[key] === true;

  let roleClass = 'role-dim';
  let roleLabel = 'DIMENSION TABLE';
  if (role === 'fact') {
    roleClass = 'role-fact';
    roleLabel = 'FACT TABLE';
  } else if (role === 'sub-dim') {
    roleClass = 'role-subdim';
    roleLabel = 'SUB-DIMENSION';
  } else if (role === 'shared-dim') {
    roleClass = 'role-dim';
    roleLabel = 'SHARED DIMENSION';
  }

  // Ensure title strictly matches role (e.g. DIM_ORDERS when acting as a dimension)
  let title = table.dw_name || key.toUpperCase();
  if (role === 'fact') {
    title = title.replace(/^DIM_/, 'FACT_');
  } else {
    title = title.replace(/^FACT_/, 'DIM_');
  }

  const subInfo = `${table.csv_file || key} · ${(table.total_rows || 0).toLocaleString()} rows`;
  const cols = table.columns || [];

  // Determine which columns to show in collapsed vs expanded view
  let displayedCols = cols;
  if (isCollapsed) {
    // Collapsed: show PKs, FKs, and primary measures (or first 4 if none flagged)
    displayedCols = cols.filter(c => c.key_type || isMeasureCol(key, c.name));
    if (displayedCols.length === 0) displayedCols = cols.slice(0, 4);
  }

  const colRows = displayedCols.map(c => {
    const badge = getColBadge(key, c);
    return `<div class="erd-col-row" data-col="${c.name}">
      <span class="erd-col-name" title="${c.name}">${c.name}</span>
      <span class="erd-col-type">${dtypeShort(c.dtype)}</span>
      <span class="erd-badge-key ${badge.cls}">${badge.label || '&nbsp;'}</span>
    </div>`;
  }).join('');

  const hiddenCount = cols.length - displayedCols.length;
  const toggleBtnText = isCollapsed ? '＋' : '－';
  const toggleBtnTitle = isCollapsed ? 'Expand all columns' : 'Collapse to key columns';

  const footerAction = isCollapsed
    ? `<button class="erd-footer-action" onclick="event.stopPropagation();toggleTableExpand('${schemaId}','${key}')">+ Expand ${hiddenCount > 0 ? hiddenCount + ' more' : 'all'} columns</button>`
    : `<button class="erd-footer-action" onclick="event.stopPropagation();toggleTableExpand('${schemaId}','${key}')">－ Collapse view</button>`;

  return `<div class="erd-table" id="${domId}" style="left:${left}px;top:${top}px;" onclick="showLiveTableDetail('${key}')" title="Click to view full dataset stats and preview">
    <div class="erd-table-header">
      <div class="erd-header-top">
        <span class="erd-role-badge ${roleClass}">${roleLabel}</span>
        <button class="erd-toggle-btn" onclick="event.stopPropagation();toggleTableExpand('${schemaId}','${key}')" title="${toggleBtnTitle}">${toggleBtnText}</button>
      </div>
      <div class="erd-table-name" title="${title}">${title}</div>
      <div class="erd-table-sub" title="${subInfo}">${subInfo}</div>
    </div>
    <div class="erd-cols-list">
      ${colRows}
    </div>
    <div class="erd-table-footer">
      ${footerAction}
      <button class="erd-footer-inspect" onclick="event.stopPropagation();showLiveTableDetail('${key}')" title="Inspect table data">🔍 Preview</button>
    </div>
  </div>`;
}

// Toggle individual table expand/collapse
function toggleTableExpand(schemaId, key) {
  if (!STATE.schemaTableCollapsed[schemaId]) STATE.schemaTableCollapsed[schemaId] = {};
  STATE.schemaTableCollapsed[schemaId][key] = !STATE.schemaTableCollapsed[schemaId][key];

  // Re-render current schema DOM & redraw connectors
  if (schemaId === 'star') renderLiveStarSchema(STATE.schemaData);
  else if (schemaId === 'snowflake') renderLiveSnowflakeSchema(STATE.schemaData);
  else if (schemaId === 'constellation') renderLiveConstellationSchema(STATE.schemaData);
}

// Global Toolbar: Expand All tables in current schema view
function expandAllTables() {
  const schemaId = STATE.currentSchemaView;
  if (!STATE.schemaData || schemaId === 'comparison') return;
  STATE.schemaTableCollapsed[schemaId] = {};
  Object.keys(STATE.schemaData.tables).forEach(k => {
    STATE.schemaTableCollapsed[schemaId][k] = false;
  });
  if (schemaId === 'star') renderLiveStarSchema(STATE.schemaData);
  else if (schemaId === 'snowflake') renderLiveSnowflakeSchema(STATE.schemaData);
  else if (schemaId === 'constellation') renderLiveConstellationSchema(STATE.schemaData);
}

// Global Toolbar: Collapse All tables in current schema view
function collapseAllTables() {
  const schemaId = STATE.currentSchemaView;
  if (!STATE.schemaData || schemaId === 'comparison') return;
  STATE.schemaTableCollapsed[schemaId] = {};
  Object.keys(STATE.schemaData.tables).forEach(k => {
    STATE.schemaTableCollapsed[schemaId][k] = true;
  });
  if (schemaId === 'star') renderLiveStarSchema(STATE.schemaData);
  else if (schemaId === 'snowflake') renderLiveSnowflakeSchema(STATE.schemaData);
  else if (schemaId === 'constellation') renderLiveConstellationSchema(STATE.schemaData);
}

// Global Toolbar: Zoom In / Out
function zoomSchema(delta) {
  const schemaId = STATE.currentSchemaView;
  if (schemaId === 'comparison') return;
  const zoom = STATE.schemaZoom[schemaId] || { scale: 1.0, panX: 0, panY: 0 };
  zoom.scale = Math.min(2.2, Math.max(0.4, Number((zoom.scale + delta).toFixed(2))));
  applyErdTransform(schemaId);
}

// Global Toolbar: Reset Layout
function resetSchemaLayout() {
  const schemaId = STATE.currentSchemaView;
  if (schemaId === 'comparison') return;
  const defaultScales = { star: 0.95, snowflake: 0.85, constellation: 0.90 };
  STATE.schemaZoom[schemaId] = { scale: defaultScales[schemaId] || 1.0, panX: 0, panY: 0 };
  applyErdTransform(schemaId);
  redrawActiveSchemaArrows();
}

// Global Toolbar: Fit to Screen
function fitSchemaToScreen() {
  const schemaId = STATE.currentSchemaView;
  if (schemaId === 'comparison') return;
  const viewport = document.getElementById(`${schemaId}-viewport`);
  const content = document.getElementById(`${schemaId}-content`);
  if (!viewport || !content) return;

  const tables = content.querySelectorAll('.erd-table');
  if (tables.length === 0) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  tables.forEach(t => {
    const l = parseFloat(t.style.left) || 0;
    const top = parseFloat(t.style.top) || 0;
    const w = t.offsetWidth || 280;
    const h = t.offsetHeight || 180;
    minX = Math.min(minX, l);
    minY = Math.min(minY, top);
    maxX = Math.max(maxX, l + w);
    maxY = Math.max(maxY, top + h);
  });

  const boundW = maxX - minX + 60;
  const boundH = maxY - minY + 60;
  const vpW = viewport.clientWidth;
  const vpH = viewport.clientHeight;

  const scaleX = vpW / boundW;
  const scaleY = vpH / boundH;
  let scale = Math.min(scaleX, scaleY, 1.05);
  scale = Math.max(0.4, Number(scale.toFixed(2)));

  const panX = Math.round((vpW - boundW * scale) / 2 - minX * scale + 30 * scale);
  const panY = Math.round((vpH - boundH * scale) / 2 - minY * scale + 30 * scale);

  STATE.schemaZoom[schemaId] = { scale, panX, panY };
  applyErdTransform(schemaId);
}

function applyErdTransform(schemaId) {
  const content = document.getElementById(`${schemaId}-content`);
  if (!content) return;
  const zoom = STATE.schemaZoom[schemaId] || { scale: 1.0, panX: 0, panY: 0 };
  content.style.transform = `translate(${zoom.panX}px, ${zoom.panY}px) scale(${zoom.scale})`;
  updateSchemaZoomBadge(schemaId);
}

// Interactive Smooth Pan and Zoom drag engine
function initErdPanZoom(viewportId, contentId, schemaId) {
  const viewport = document.getElementById(viewportId);
  const content = document.getElementById(contentId);
  if (!viewport || !content) return;

  let isDragging = false;
  let startX = 0, startY = 0;
  let initialPanX = 0, initialPanY = 0;

  viewport.onmousedown = (e) => {
    if (e.target.closest('.erd-table') || e.target.closest('button')) return;
    isDragging = true;
    viewport.classList.add('is-panning');
    startX = e.clientX;
    startY = e.clientY;
    const zoom = STATE.schemaZoom[schemaId] || { scale: 1.0, panX: 0, panY: 0 };
    initialPanX = zoom.panX;
    initialPanY = zoom.panY;
    e.preventDefault();
  };

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    const zoom = STATE.schemaZoom[schemaId] || { scale: 1.0, panX: 0, panY: 0 };
    zoom.panX = initialPanX + dx;
    zoom.panY = initialPanY + dy;
    applyErdTransform(schemaId);
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      viewport.classList.remove('is-panning');
    }
  });

  viewport.onwheel = (e) => {
    e.preventDefault();
    const zoom = STATE.schemaZoom[schemaId] || { scale: 1.0, panX: 0, panY: 0 };
    const delta = e.deltaY < 0 ? 0.08 : -0.08;
    const newScale = Math.min(2.2, Math.max(0.4, Number((zoom.scale + delta).toFixed(2))));
    
    const rect = viewport.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    zoom.panX -= (mouseX - zoom.panX) * (newScale / zoom.scale - 1);
    zoom.panY -= (mouseY - zoom.panY) * (newScale / zoom.scale - 1);
    zoom.scale = newScale;
    
    applyErdTransform(schemaId);
  };
}

// ── SVG DEFS (Markers) ──
const SVG_DEFS = `<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="#64748b"/>
  </marker>
  <marker id="arrow-indigo" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="#4f46e5"/>
  </marker>
  <marker id="arrow-amber" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="#d97706"/>
  </marker>
  <marker id="arrow-sky" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="#0284c7"/>
  </marker>
  <marker id="arrow-purple" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
    <path d="M 0 0 L 8 4 L 0 8 Z" fill="#7c3aed"/>
  </marker>
</defs>`;

// Compute precise anchor port in untransformed canvas coordinates
function getAnchorPortUntransformed(tableEl, colName, side) {
  const tableLeft = parseFloat(tableEl.style.left) || 0;
  const tableTop = parseFloat(tableEl.style.top) || 0;
  const tableWidth = tableEl.offsetWidth || 280;

  // Try finding specific column row offsetTop
  let rowY = null;
  if (colName) {
    const rowEl = tableEl.querySelector(`[data-col="${colName}"]`);
    if (rowEl) {
      rowY = tableTop + rowEl.offsetTop + (rowEl.offsetHeight || 22) / 2;
    }
  }

  // Fallback to table header center
  if (rowY === null) {
    rowY = tableTop + 32;
  }

  const x = (side === 'right') ? (tableLeft + tableWidth) : tableLeft;
  return { x, y: rowY };
}

// Draw orthogonal / smooth bezier connector between specific table keys
function drawErdLine(svgEl, contentEl, fromId, fromCol, toId, toCol, label = '', color = '#64748b') {
  const fromEl = document.getElementById(fromId);
  const toEl = document.getElementById(toId);
  if (!fromEl || !toEl || !svgEl || !contentEl) return;

  const fLeft = parseFloat(fromEl.style.left) || 0;
  const fWidth = fromEl.offsetWidth || 280;
  const tLeft = parseFloat(toEl.style.left) || 0;
  const tWidth = toEl.offsetWidth || 280;

  // Determine cleanest exit/entry edges based on column position
  let exitSide = 'right';
  let enterSide = 'left';

  if (fLeft + fWidth <= tLeft) {
    // From is to the left of To
    exitSide = 'right';
    enterSide = 'left';
  } else if (fLeft >= tLeft + tWidth) {
    // From is to the right of To
    exitSide = 'left';
    enterSide = 'right';
  } else {
    // Same column / vertically aligned
    exitSide = 'right';
    enterSide = 'right';
  }

  const p1 = getAnchorPortUntransformed(fromEl, fromCol, exitSide);
  const p2 = getAnchorPortUntransformed(toEl, toCol, enterSide);

  const dx = Math.abs(p2.x - p1.x);
  const curvature = Math.max(50, dx * 0.45);
  const cp1x = exitSide === 'right' ? p1.x + curvature : p1.x - curvature;
  const cp2x = enterSide === 'left' ? p2.x - curvature : p2.x + curvature;

  const d = `M ${p1.x} ${p1.y} C ${cp1x} ${p1.y}, ${cp2x} ${p2.y}, ${p2.x} ${p2.y}`;

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  path.setAttribute('stroke', color);
  path.setAttribute('class', 'erd-connector');
  path.setAttribute('marker-end', 'url(#arrow)');
  svgEl.appendChild(path);

  // Optional key pill label
  if (label) {
    const midX = 0.125 * p1.x + 0.375 * cp1x + 0.375 * cp2x + 0.125 * p2.x;
    const midY = 0.125 * p1.y + 0.375 * p1.y + 0.375 * p2.y + 0.125 * p2.y;

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    const textLen = label.length * 6.2 + 12;

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', midX - textLen / 2);
    rect.setAttribute('y', midY - 9);
    rect.setAttribute('width', textLen);
    rect.setAttribute('height', 18);
    rect.setAttribute('rx', 4);
    rect.setAttribute('fill', '#ffffff');
    rect.setAttribute('stroke', '#cbd5e1');
    rect.setAttribute('stroke-width', '1');

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', midX);
    text.setAttribute('y', midY);
    text.setAttribute('class', 'erd-connector-label');
    text.textContent = label;

    g.appendChild(rect);
    g.appendChild(text);
    svgEl.appendChild(g);
  }
}

// ─── 1. STAR SCHEMA (SINGLE CENTRAL FACT TABLE + DENORMALIZED DIMENSIONS) ───
function renderLiveStarSchema(data) {
  const container = document.getElementById('starSchemaContent');
  if (!container || !data?.tables) return;
  const t = data.tables;

  // Single central fact table in Center Column (X=540).
  // Left Column (X=60): Customers, Products.
  // Right Column (X=1000): Orders (dimension), Sellers, Payments (dimension).
  // NO stacked fact tables, 180-200px open corridors, zero overlapping!
  const POS = {
    // Solitary Single Central Fact Table
    fact:      { left: 540,  top: 220 },
    // Surrounding Denormalized Dimensions
    customers: { left: 60,   top: 60  },
    products:  { left: 60,   top: 460 },
    orders:    { left: 1000, top: 40  },
    sellers:   { left: 1000, top: 340 },
    payments:  { left: 1000, top: 640 },
  };

  container.innerHTML = `
    <div class="erd-viewport" id="star-viewport">
      <div class="erd-viewport-content" id="star-content" style="width:1380px;height:900px;">
        <svg class="erd-svg" id="star-svg">${SVG_DEFS}</svg>
        ${erdTableHTML('star', 'olist_customers', t['olist_customers'], 'dimension', POS.customers.left, POS.customers.top)}
        ${erdTableHTML('star', 'olist_products', t['olist_products'], 'dimension', POS.products.left, POS.products.top)}
        ${erdTableHTML('star', 'olist_order_items', t['olist_order_items'], 'fact', POS.fact.left, POS.fact.top)}
        ${erdTableHTML('star', 'olist_orders', t['olist_orders'], 'dimension', POS.orders.left, POS.orders.top)}
        ${erdTableHTML('star', 'olist_sellers', t['olist_sellers'], 'dimension', POS.sellers.left, POS.sellers.top)}
        ${erdTableHTML('star', 'olist_payments', t['olist_payments'], 'dimension', POS.payments.left, POS.payments.top)}
      </div>
    </div>
    <div class="schema-note">
      <strong>⭐ Star Schema Architecture:</strong> Uses a <strong>single central fact table</strong> (<code>FACT_ORDER_ITEMS</code>) containing additive quantitative measures (<code>price</code>, <code>freight_value</code>) connected directly to surrounding <strong>de-normalized dimension tables</strong> (<code>DIM_CUSTOMER</code>, <code>DIM_PRODUCT</code>, <code>DIM_ORDERS</code>, <code>DIM_SELLER</code>, <code>DIM_PAYMENTS</code>) via 1-hop foreign keys. Designed for rapid aggregations and minimal join overhead.
    </div>
  `;

  initErdPanZoom('star-viewport', 'star-content', 'star');
  applyErdTransform('star');

  requestAnimationFrame(() => {
    drawStarArrows();
  });
}

function drawStarArrows() {
  const svg = document.getElementById('star-svg');
  const content = document.getElementById('star-content');
  if (!svg || !content) return;
  svg.innerHTML = SVG_DEFS;

  // Every single dimension connects directly into the ONE central fact table
  // Left Column Dimensions -> Central Fact (exit right, enter left)
  drawErdLine(svg, content, 'erd-star-olist_customers', 'customer_id', 'erd-star-olist_order_items', 'order_id', 'customer_id', '#0284c7');
  drawErdLine(svg, content, 'erd-star-olist_products', 'product_id', 'erd-star-olist_order_items', 'product_id', 'product_id', '#d97706');

  // Right Column Dimensions -> Central Fact (exit left, enter right)
  drawErdLine(svg, content, 'erd-star-olist_orders', 'order_id', 'erd-star-olist_order_items', 'order_id', 'order_id', '#4f46e5');
  drawErdLine(svg, content, 'erd-star-olist_sellers', 'seller_id', 'erd-star-olist_order_items', 'seller_id', 'seller_id', '#059669');
  drawErdLine(svg, content, 'erd-star-olist_payments', 'order_id', 'erd-star-olist_order_items', 'order_id', 'order_id', '#7c3aed');
}

// ─── 2. SNOWFLAKE SCHEMA (SINGLE CENTRAL FACT + NORMALIZED SUB-TABLES) ───
function renderLiveSnowflakeSchema(data) {
  const container = document.getElementById('snowflakeSchemaContent');
  if (!container || !data?.tables) return;
  const t = data.tables;

  // Single central fact table in Center Column (X=780).
  // Normalized Sub-tables (Level 2) on outer flanks (X=50, X=1500).
  // Primary Dimensions (Level 1) in middle columns (X=420, X=1140).
  const POS = {
    // Level 2 Sub-dimensions (Normalized 3NF Sub-tables)
    cat_trans: { left: 50,   top: 60  },
    geo:       { left: 50,   top: 460 },
    // Level 1 Dimensions (Middle Left)
    products:  { left: 420,  top: 60  },
    customers: { left: 420,  top: 460 },
    // Solitary Single Central Fact Table (Center)
    fact:      { left: 780,  top: 240 },
    // Level 1 Dimensions (Middle Right)
    orders:    { left: 1140, top: 60  },
    sellers:   { left: 1140, top: 340 },
    payments:  { left: 1140, top: 620 },
    // Level 2 Normalized Feedback Sub-table (Far Right)
    reviews:   { left: 1500, top: 60  },
  };

  container.innerHTML = `
    <div class="erd-viewport" id="snowflake-viewport">
      <div class="erd-viewport-content" id="snowflake-content" style="width:1860px;height:900px;">
        <svg class="erd-svg" id="snowflake-svg">${SVG_DEFS}</svg>
        ${erdTableHTML('snowflake', 'category_translation', t['category_translation'], 'sub-dim', POS.cat_trans.left, POS.cat_trans.top)}
        ${erdTableHTML('snowflake', 'olist_geolocation', t['olist_geolocation'], 'sub-dim', POS.geo.left, POS.geo.top)}
        ${erdTableHTML('snowflake', 'olist_products', t['olist_products'], 'dimension', POS.products.left, POS.products.top)}
        ${erdTableHTML('snowflake', 'olist_customers', t['olist_customers'], 'dimension', POS.customers.left, POS.customers.top)}
        ${erdTableHTML('snowflake', 'olist_order_items', t['olist_order_items'], 'fact', POS.fact.left, POS.fact.top)}
        ${erdTableHTML('snowflake', 'olist_orders', t['olist_orders'], 'dimension', POS.orders.left, POS.orders.top)}
        ${erdTableHTML('snowflake', 'olist_sellers', t['olist_sellers'], 'dimension', POS.sellers.left, POS.sellers.top)}
        ${erdTableHTML('snowflake', 'olist_payments', t['olist_payments'], 'dimension', POS.payments.left, POS.payments.top)}
        ${erdTableHTML('snowflake', 'olist_reviews', t['olist_reviews'], 'sub-dim', POS.reviews.left, POS.reviews.top)}
      </div>
    </div>
    <div class="schema-note">
      <strong>❄️ Snowflake Schema Architecture:</strong> Uses a <strong>single central fact table</strong> (<code>FACT_ORDER_ITEMS</code>), while surrounding dimension tables are <strong>normalized and split into sub-tables</strong>. <code>DIM_PRODUCT</code> normalizes category codes into sub-table <code>DIM_CATEGORY</code>; <code>DIM_CUSTOMER</code> normalizes postal prefixes into sub-table <code>DIM_GEOLOCATION</code>; <code>DIM_ORDERS</code> links feedback to <code>DIM_REVIEWS</code>. Eliminates attribute redundancy through 3NF decomposition.
    </div>
  `;

  initErdPanZoom('snowflake-viewport', 'snowflake-content', 'snowflake');
  applyErdTransform('snowflake');

  requestAnimationFrame(() => {
    drawSnowflakeArrows();
  });
}

function drawSnowflakeArrows() {
  const svg = document.getElementById('snowflake-svg');
  const content = document.getElementById('snowflake-content');
  if (!svg || !content) return;
  svg.innerHTML = SVG_DEFS;

  // Level 2 Sub-dimensions -> Level 1 Dimensions
  drawErdLine(svg, content, 'erd-snowflake-category_translation', 'product_category_name', 'erd-snowflake-olist_products', 'product_category_name', 'category_name', '#7c3aed');
  drawErdLine(svg, content, 'erd-snowflake-olist_geolocation', 'geolocation_zip_code_prefix', 'erd-snowflake-olist_customers', 'customer_zip_code_prefix', 'zip_prefix', '#7c3aed');
  drawErdLine(svg, content, 'erd-snowflake-olist_reviews', 'order_id', 'erd-snowflake-olist_orders', 'order_id', 'order_id', '#7c3aed');

  // Level 1 Dimensions -> Single Central Fact Table
  drawErdLine(svg, content, 'erd-snowflake-olist_products', 'product_id', 'erd-snowflake-olist_order_items', 'product_id', 'product_id', '#d97706');
  drawErdLine(svg, content, 'erd-snowflake-olist_customers', 'customer_id', 'erd-snowflake-olist_order_items', 'order_id', 'customer_id', '#0284c7');
  drawErdLine(svg, content, 'erd-snowflake-olist_orders', 'order_id', 'erd-snowflake-olist_order_items', 'order_id', 'order_id', '#4f46e5');
  drawErdLine(svg, content, 'erd-snowflake-olist_sellers', 'seller_id', 'erd-snowflake-olist_order_items', 'seller_id', 'seller_id', '#059669');
  drawErdLine(svg, content, 'erd-snowflake-olist_payments', 'order_id', 'erd-snowflake-olist_order_items', 'order_id', 'order_id', '#7c3aed');
}

// ─── 3. GALAXY SCHEMA (FACT CONSTELLATION: MULTIPLE CENTRAL FACTS + SHARED DIMS) ───
function renderLiveConstellationSchema(data) {
  const container = document.getElementById('constellationSchemaContent');
  if (!container || !data?.tables) return;
  const t = data.tables;

  // Features multiple central fact tables sharing common dimension tables
  // Left: Fact 1 (Orders) & Fact 2 (Payments)
  // Center: Shared Common Dimensions (Customers, Products, Sellers)
  // Right: Fact 3 (Order Items) & Fact 4 (Reviews)
  const POS = {
    // Multiple Central Fact Tables (Flanks)
    f_orders:    { left: 60,  top: 80  },
    f_payments:  { left: 60,  top: 480 },
    // Shared Common Dimension Tables (Center Hub)
    d_customers: { left: 510, top: 40  },
    d_products:  { left: 510, top: 360 },
    d_sellers:   { left: 510, top: 660 },
    // Multiple Central Fact Tables (Flanks)
    f_items:     { left: 960, top: 80  },
    f_reviews:   { left: 960, top: 480 },
  };

  container.innerHTML = `
    <div class="erd-viewport" id="constellation-viewport">
      <div class="erd-viewport-content" id="constellation-content" style="width:1380px;height:900px;">
        <svg class="erd-svg" id="constellation-svg">${SVG_DEFS}</svg>
        ${erdTableHTML('constellation', 'olist_orders', t['olist_orders'], 'fact', POS.f_orders.left, POS.f_orders.top)}
        ${erdTableHTML('constellation', 'olist_payments', t['olist_payments'], 'fact', POS.f_payments.left, POS.f_payments.top)}
        <div style="cursor:pointer;" onclick="highlightLiveConstellation('customer')">
          ${erdTableHTML('constellation', 'olist_customers', t['olist_customers'], 'shared-dim', POS.d_customers.left, POS.d_customers.top)}
        </div>
        <div style="cursor:pointer;" onclick="highlightLiveConstellation('product')">
          ${erdTableHTML('constellation', 'olist_products', t['olist_products'], 'shared-dim', POS.d_products.left, POS.d_products.top)}
        </div>
        <div style="cursor:pointer;" onclick="highlightLiveConstellation('seller')">
          ${erdTableHTML('constellation', 'olist_sellers', t['olist_sellers'], 'shared-dim', POS.d_sellers.left, POS.d_sellers.top)}
        </div>
        ${erdTableHTML('constellation', 'olist_order_items', t['olist_order_items'], 'fact', POS.f_items.left, POS.f_items.top)}
        ${erdTableHTML('constellation', 'olist_reviews', t['olist_reviews'], 'fact', POS.f_reviews.left, POS.f_reviews.top)}
      </div>
    </div>
    <div class="schema-note">
      <strong>✨ Galaxy Schema (Fact Constellation) Architecture:</strong> Features <strong>multiple central fact tables</strong> (<code>FACT_ORDERS</code>, <code>FACT_PAYMENTS</code>, <code>FACT_ORDER_ITEMS</code>, <code>FACT_REVIEWS</code>) that share <strong>common conformed dimension tables</strong> (<code>DIM_CUSTOMER</code>, <code>DIM_PRODUCT</code>, <code>DIM_SELLER</code>). Enables cross-process drill-across analysis across orders, transactions, fulfillment, and customer satisfaction.
    </div>
  `;

  initErdPanZoom('constellation-viewport', 'constellation-content', 'constellation');
  applyErdTransform('constellation');

  requestAnimationFrame(() => {
    drawConstellationArrows();
  });
}

function drawConstellationArrows() {
  const svg = document.getElementById('constellation-svg');
  const content = document.getElementById('content');
  const constContent = document.getElementById('constellation-content');
  if (!svg || !constContent) return;
  svg.innerHTML = SVG_DEFS;

  // Shared Customer Dimension connects to Orders (left), Payments (left), Items (right)
  drawErdLine(svg, constContent, 'erd-constellation-olist_customers', 'customer_id', 'erd-constellation-olist_orders', 'customer_id', 'customer_id', '#4f46e5');
  drawErdLine(svg, constContent, 'erd-constellation-olist_customers', 'customer_id', 'erd-constellation-olist_payments', 'order_id', 'customer_id', '#0284c7');
  drawErdLine(svg, constContent, 'erd-constellation-olist_customers', 'customer_id', 'erd-constellation-olist_order_items', 'order_id', 'customer_id', '#4f46e5');

  // Shared Product Dimension connects to Items (right) and Reviews (right)
  drawErdLine(svg, constContent, 'erd-constellation-olist_products', 'product_id', 'erd-constellation-olist_order_items', 'product_id', 'product_id', '#d97706');
  drawErdLine(svg, constContent, 'erd-constellation-olist_products', 'product_id', 'erd-constellation-olist_reviews', 'review_id', 'product_id', '#d97706');

  // Shared Seller Dimension connects to Items (right)
  drawErdLine(svg, constContent, 'erd-constellation-olist_sellers', 'seller_id', 'erd-constellation-olist_order_items', 'seller_id', 'seller_id', '#059669');

  // Orders connects to Reviews
  drawErdLine(svg, constContent, 'erd-constellation-olist_orders', 'order_id', 'erd-constellation-olist_reviews', 'order_id', 'order_id', '#7c3aed');
}

function highlightLiveConstellation(dim) {
  document.querySelectorAll('#constellation-content .erd-table').forEach(n => n.classList.remove('highlighted'));
  const map = {
    customer: ['erd-constellation-olist_orders', 'erd-constellation-olist_payments', 'erd-constellation-olist_customers', 'erd-constellation-olist_order_items'],
    product:  ['erd-constellation-olist_order_items', 'erd-constellation-olist_reviews', 'erd-constellation-olist_products'],
    seller:   ['erd-constellation-olist_order_items', 'erd-constellation-olist_sellers']
  };
  (map[dim] || []).forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('highlighted');
  });
}

function redrawActiveSchemaArrows() {
  const schema = STATE.currentSchemaView;
  if (schema === 'star') drawStarArrows();
  else if (schema === 'snowflake') drawSnowflakeArrows();
  else if (schema === 'constellation') drawConstellationArrows();
}

// ─── 4. SCHEMA COMPARISON (COMPREHENSIVE PROPER TABLE) ────────
function renderLiveComparisonSchema(data) {
  const container = document.getElementById('comparisonSchemaContent');
  if (!container) return;
  const t = data?.tables || {};

  const orderRows = (t['olist_orders']?.total_rows || 99441).toLocaleString();
  const itemRows  = (t['olist_order_items']?.total_rows || 112650).toLocaleString();
  const geoRows   = (t['olist_geolocation']?.total_rows || 1000163).toLocaleString();
  const prodRows  = (t['olist_products']?.total_rows || 32951).toLocaleString();

  container.innerHTML = `
    <div class="comparison-container">
      <div class="comparison-table-wrapper">
        <table class="comparison-table">
          <thead>
            <tr>
              <th style="min-width:180px;">Evaluation Criteria</th>
              <th style="min-width:240px;"><span class="comparison-badge badge-star">⭐ Star Schema</span></th>
              <th style="min-width:260px;"><span class="comparison-badge badge-snowflake">❄️ Snowflake Schema</span></th>
              <th style="min-width:280px;"><span class="comparison-badge badge-galaxy">✨ Fact Constellation</span></th>
              <th style="min-width:220px;">Olist Physical Implementation</th>
            </tr>
          </thead>
          <tbody>
            <tr onclick="explainLiveRow(this, 'structure')" class="active-row">
              <td><div class="comparison-aspect-header"><span>🏛️</span> <strong>Structure &amp; Topology</strong></div></td>
              <td>Single central fact table surrounded radially by flat dimension tables. Simple hub-and-spoke star shape.</td>
              <td>Central fact table connected to hierarchical, multi-tiered dimension sub-tables branching like a snowflake.</td>
              <td>Multiple distinct fact tables interconnected through shared conformed dimensions in an enterprise constellation matrix.</td>
              <td>Logical dimensional views mapped over raw relational CSV files.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'facts')">
              <td><div class="comparison-aspect-header"><span>📊</span> <strong>Fact Tables</strong></div></td>
              <td><strong>1 Central Fact:</strong> Line items (<code>FACT_ORDER_ITEMS</code>). Granular transaction rows with numeric measures.</td>
              <td><strong>1 Central Fact:</strong> Line items (<code>FACT_ORDER_ITEMS</code>) joined to root-level dimensions.</td>
              <td><strong>4 Fact Tables:</strong> <code>FACT_ORDERS</code>, <code>FACT_ORDER_ITEMS</code>, <code>FACT_PAYMENTS</code>, <code>FACT_REVIEWS</code>.</td>
              <td>Directly backed by ${itemRows} order items and ${orderRows} parent orders.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'dims')">
              <td><div class="comparison-aspect-header"><span>📐</span> <strong>Dimension Tables</strong></div></td>
              <td>Wide, de-normalized dimension tables with redundant descriptive attributes for maximum query convenience.</td>
              <td>Normalized dimensions decomposed into parent and child lookup sub-dimensions (e.g. Geolocation and Categories).</td>
              <td>Conformed shared dimensions consistent across multiple business processes (Customers, Products, Sellers).</td>
              <td>${prodRows} products, 96,096 unique customer entities.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'normalization')">
              <td><div class="comparison-aspect-header"><span>⚖️</span> <strong>Normalization Level</strong></div></td>
              <td><strong>De-normalized</strong> (1NF / 2NF). Attributes intentionally repeated to eliminate multi-table joins.</td>
              <td><strong>Normalized (3NF)</strong>. Hierarchies isolated into separate tables to eliminate data redundancy.</td>
              <td><strong>Hybrid / Conformed 3NF</strong>. Dimensions normalized to ensure singular definitions across all business marts.</td>
              <td>Raw CSVs exhibit natural 3NF normalization.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'relationships')">
              <td><div class="comparison-aspect-header"><span>🔗</span> <strong>Relationships &amp; Joins</strong></div></td>
              <td>1-Hop Foreign Key joins (Fact &rarr; Dimension). Lowest join depth and lowest SQL join overhead.</td>
              <td>Multi-hop cascading joins (Fact &rarr; Level 1 Dimension &rarr; Level 2 Sub-dimension). Higher join complexity.</td>
              <td>Cross-process Drill-Across joins sharing conformed dimensions. Inter-fact queries mediated by shared keys.</td>
              <td>Verified Foreign Keys (<code>order_id</code>, <code>product_id</code>, <code>customer_id</code>).</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'advantages')">
              <td><div class="comparison-aspect-header"><span>✨</span> <strong>Key Advantages</strong></div></td>
              <td>Simplest SQL queries, fastest aggregation performance, high BI tool compatibility, easy for business analysts.</td>
              <td>Minimal storage consumption, zero attribute redundancy, clean single-point update integrity for master data.</td>
              <td>Enterprise-wide analytical coverage, avoids departmental silos, enables cross-departmental correlation analysis.</td>
              <td>Enables fast customer churn RFM scoring in memory.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'limitations')">
              <td><div class="comparison-aspect-header"><span>⚠️</span> <strong>Limitations</strong></div></td>
              <td>Storage redundancy across wide dimensions; potential update anomalies if historical attributes mutate.</td>
              <td>More complex SQL queries, higher join latencies for large scans, harder for ad-hoc business users to write.</td>
              <td>High modeling complexity, requires strict governance and conformed dimension master data management.</td>
              <td>Requires unified ETL pipelines to synchronize 9 raw CSVs.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'use_cases')">
              <td><div class="comparison-aspect-header"><span>🎯</span> <strong>Target Use Cases</strong></div></td>
              <td>Departmental data marts, executive KPI reporting, high-concurrency dashboards, simple slice-and-dice OLAP.</td>
              <td>Deep taxonomic catalogs (e.g., product classification, state-city-zip postal boundaries), storage-constrained systems.</td>
              <td>Enterprise Data Warehouses (EDW), unified customer intelligence covering billing, delivery, and sentiment.</td>
              <td>Production academic laboratory benchmarking Olist churn.</td>
            </tr>
            <tr onclick="explainLiveRow(this, 'live_volume')">
              <td><div class="comparison-aspect-header"><span>💾</span> <strong>Physical Olist Volume</strong></div></td>
              <td>${itemRows} line items + ${orderRows} orders parsed directly from disk.</td>
              <td>${geoRows} normalized geolocation coordinates + 73 translated product categories.</td>
              <td>4 complete operational workflows: sales, items, payments, customer reviews.</td>
              <td>Live file introspection via <code>/api/academic/schema</code>.</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div id="comparisonExplain" class="comparison-explanation-panel">
        <div class="comparison-explanation-title">
          <span>🏛️</span> <strong>Structure &amp; Topology Architectural Context</strong>
        </div>
        <div>
          Star schemas favor query simplicity and speed via denormalization. Snowflake schemas prioritize zero attribute redundancy and 3NF normalization. Fact Constellation schemas provide true enterprise coverage by uniting multiple transaction marts around conformed dimensions.
        </div>
      </div>
    </div>
  `;
}

const LIVE_ROW_EXPLANATIONS = {
  structure:     'Star schemas favor query simplicity and speed via denormalization. Snowflake schemas prioritize zero attribute redundancy and 3NF normalization. Fact Constellation schemas provide true enterprise coverage by uniting multiple transaction marts around conformed dimensions.',
  facts:         'Star and Snowflake schemas revolve around a single fact table (Order Items). In contrast, the Fact Constellation models 4 distinct business processes simultaneously: Orders, Order Items, Payments, and Reviews.',
  dims:          'Star stores wide denormalized dimension rows for fast queries. Snowflake separates Geolocation and Product Categories into normalized sub-dimensions. Constellation shares dimensions like Customers across all facts.',
  normalization: 'Snowflake eliminates repeated string values by decomposing hierarchies to 3NF. Star accepts storage duplication to avoid expensive multi-table joins during OLAP aggregations.',
  relationships: 'Star requires only 1 JOIN to reach customer or product details. Snowflake requires 2 or more JOINs to reach translated categories or zip coordinates. Constellation queries only the relevant fact table.',
  advantages:    'Star provides the highest read speeds for BI dashboards. Snowflake preserves disk space and ensures atomic master data updates. Constellation empowers cross-process analysis like customer lifetime value across orders and payments.',
  limitations:   'Star trades storage space for query speed. Snowflake adds join latency and SQL cognitive load. Constellation demands strict data governance to ensure dimension keys remain strictly conformed across all marts.',
  use_cases:     'Star is ideal for single-department slice-and-dice. Snowflake fits complex classification hierarchies like logistics postcodes. Constellation is the backbone of unified enterprise customer intelligence.',
  live_volume:   'All schemas shown are introspected directly from the physical CSV files in data/olist_raw/. Nothing is mock or static.'
};

function explainLiveRow(row, key) {
  document.querySelectorAll('.comparison-table tbody tr').forEach(r => r.classList.remove('active-row'));
  row.classList.add('active-row');
  const box = document.getElementById('comparisonExplain');
  if (box) {
    const title = row.cells[0]?.innerText?.trim() || 'Architectural Context';
    box.innerHTML = `
      <div class="comparison-explanation-title">
        <span>📋</span> <strong>${title} Analysis</strong>
      </div>
      <div>${LIVE_ROW_EXPLANATIONS[key] || ''}</div>
    `;
  }
}

function showLiveTableDetail(tableKey) {
  if (!STATE.schemaData || !STATE.schemaData.tables[tableKey]) return;
  const table = STATE.schemaData.tables[tableKey];

  document.getElementById('tableDetailTitle').textContent = `📂 ${table.dw_name || tableKey.toUpperCase()} (${table.csv_file})`;

  const colsHTML = (table.columns || []).map(c => {
    const badge = getColBadge(tableKey, c);
    return `
      <div class="col-item ${c.key_type === 'PK' ? 'col-pk' : (c.key_type === 'FK' ? 'col-fk' : '')}">
        <div>
          <span class="erd-badge-key ${badge.cls}" style="margin-right:6px;">${badge.label || 'ATTR'}</span>
          <span class="col-name">${c.name}</span>
          <div style="font-size:0.68rem;color:#64748b;">${c.non_null.toLocaleString()} non-null (${c.null_pct}% null)</div>
        </div>
        <span class="col-type">${dtypeShort(c.dtype)}</span>
      </div>
    `;
  }).join('');

  // Sample data preview rows
  let sampleHTML = '';
  if (table.sample_data && table.sample_data.length > 0) {
    const sampleCols = Object.keys(table.sample_data[0]).slice(0, 4);
    sampleHTML = `
      <div style="font-size:0.75rem;font-weight:700;color:#1e293b;margin:0.9rem 0 0.4rem;">Live Physical Sample Records</div>
      <div class="table-wrapper" style="max-height:160px;overflow:auto;">
        <table class="data-table" style="font-size:0.72rem;">
          <thead><tr>${sampleCols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody>
            ${table.sample_data.map(r => `<tr>${sampleCols.map(c=>`<td>${r[c]}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  document.getElementById('tableDetailBody').innerHTML = `
    <div class="detail-item" style="margin-bottom:0.8rem;padding:0.7rem;background:#f0f4ff;border-radius:8px;">
      <div class="detail-item-label">Data Warehouse Role &amp; Physical File</div>
      <div class="detail-item-value">${(table.dw_role || 'Dimension').toUpperCase()} · ${(table.total_rows || 0).toLocaleString()} rows · ${table.total_columns} columns</div>
    </div>
    
    <div style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:0.4rem;">
      Physical Column Definitions (${(table.columns || []).length})
    </div>
    <div class="col-list" style="max-height:240px;overflow-y:auto;">
      ${colsHTML}
    </div>

    ${sampleHTML}
  `;

  show('tableDetailOverlay');
  show('tableDetailPanel');
}

function closeTableDetail() {
  hide('tableDetailOverlay');
  hide('tableDetailPanel');
}

// ─── DATA EXPLORATION ─────────────────────────────────────────
let explPage = 1;
async function loadExploration(page) {
  explPage = page;
  const dataset = document.getElementById('explorationDataset').value;
  const container = document.getElementById('explorationContent');
  container.innerHTML = `<div class="state-panel" style="padding:2rem;"><div class="loading-spinner"></div><p>Loading ${dataset}...</p></div>`;
  try {
    const res = await fetch(`/api/academic/exploration?dataset=${dataset}&page=${page}&page_size=20`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderExploration(data);
  } catch(e) {
    container.innerHTML = `<div class="state-panel error-state"><div class="state-icon">⚠️</div><p>${e.message}</p></div>`;
  }
}

function renderExploration(data) {
  STATE.currentExploration = data;
  const container = document.getElementById('explorationContent');
  const chartsArea = document.getElementById('explorationChartsArea');

  // Populate feature selector and show live charts container
  if (chartsArea) {
    chartsArea.classList.remove('hidden');
    const featSelect = document.getElementById('explFeatureSelect');
    if (featSelect && data.columns) {
      featSelect.innerHTML = data.columns.map(c => `<option value="${c}">${c}</option>`).join('');
    }

    // Render Missing Values Chart
    if (data.missing_stats && data.missing_stats.length) {
      if (STATE.chartExplMissing) {
        try { STATE.chartExplMissing.destroy(); } catch(e){}
      }
      const ctxMissing = document.getElementById('chart_expl_missing');
      if (ctxMissing) {
        const topMissing = [...data.missing_stats].sort((a,b) => b.missing_pct - a.missing_pct).slice(0, 15);
        STATE.chartExplMissing = new Chart(ctxMissing, {
          type: 'bar',
          data: {
            labels: topMissing.map(s => s.column),
            datasets: [{
              label: 'Missing %',
              data: topMissing.map(s => s.missing_pct),
              backgroundColor: topMissing.map(s => s.missing_pct > 0 ? '#dc2626aa' : '#059669aa'),
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)}% missing` } }
            },
            scales: {
              y: { beginAtZero: true, max: 100, title: { display: true, text: 'Missing Rate (%)' } },
              x: { ticks: { maxRotation: 45, font: { size: 9 } } }
            }
          }
        });
      }
    }

    // Load initial feature distribution for first column
    if (data.columns && data.columns.length) {
      loadFeatureDistribution(data.dataset, data.columns[0]);
    }

    // Render Correlation Matrix Heatmap if available
    const corrContainer = document.getElementById('explCorrContainer');
    const corrDiv = document.getElementById('explCorrHeatmap');
    if (corrContainer && corrDiv) {
      if (data.correlation_matrix && data.correlation_matrix.columns?.length) {
        corrContainer.classList.remove('hidden');
        const cols = data.correlation_matrix.columns;
        const vals = data.correlation_matrix.values;
        let heatmapHTML = `<table class="data-table" style="font-size:0.72rem;text-align:center;"><thead><tr><th></th>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
        for (let i = 0; i < cols.length; i++) {
          heatmapHTML += `<tr><th style="text-align:left;">${cols[i]}</th>`;
          for (let j = 0; j < cols.length; j++) {
            const val = vals[i][j];
            let bg = '#ffffff';
            let tc = '#1e293b';
            if (val > 0) {
              const alpha = Math.min(0.85, Math.abs(val) * 0.85 + 0.1);
              bg = `rgba(37, 99, 235, ${alpha.toFixed(2)})`;
              if (alpha > 0.45) tc = '#ffffff';
            } else if (val < 0) {
              const alpha = Math.min(0.85, Math.abs(val) * 0.85 + 0.1);
              bg = `rgba(220, 38, 38, ${alpha.toFixed(2)})`;
              if (alpha > 0.45) tc = '#ffffff';
            }
            heatmapHTML += `<td style="background:${bg};color:${tc};font-weight:600;padding:6px 4px;">${val.toFixed(2)}</td>`;
          }
          heatmapHTML += `</tr>`;
        }
        heatmapHTML += `</tbody></table>`;
        corrDiv.innerHTML = heatmapHTML;
      } else {
        corrContainer.classList.add('hidden');
      }
    }
  }

  let html = `<div class="exploration-card">
    <h3 class="section-heading">Dataset Overview: ${data.filename}</h3>
    <div class="exploration-meta-grid">
      <div class="expl-meta-item"><div class="expl-meta-label">Total Rows</div><div class="expl-meta-value">${(data.total_rows||0).toLocaleString()}</div></div>
      <div class="expl-meta-item"><div class="expl-meta-label">Columns</div><div class="expl-meta-value">${(data.columns||[]).length}</div></div>
      <div class="expl-meta-item"><div class="expl-meta-label">Page</div><div class="expl-meta-value">${data.page} / ${data.total_pages}</div></div>
    </div>

    <h4 style="font-size:0.85rem;font-weight:700;margin:1rem 0 0.5rem;color:#1e293b;">Missing Values Analysis</h4>
    <div class="table-wrapper"><table class="data-table">
      <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing %</th></tr></thead>
      <tbody>${(data.missing_stats||[]).map(s =>
        `<tr><td>${s.column}</td><td>${s.dtype}</td><td>${s.missing_count}</td><td>${s.missing_pct}%</td></tr>`
      ).join('')}</tbody>
    </table></div>

    ${data.desc_stats?.length ? `
    <h4 style="font-size:0.85rem;font-weight:700;margin:1rem 0 0.5rem;color:#1e293b;">Descriptive Statistics (Numeric Features)</h4>
    <div class="table-wrapper"><table class="data-table">
      <thead><tr><th>Column</th><th>Count</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>${data.desc_stats.map(s =>
        `<tr><td>${s.column}</td><td>${s.count.toLocaleString()}</td><td>${s.mean}</td><td>${s.median}</td><td>${s.std}</td><td>${s.min}</td><td>${s.max}</td></tr>`
      ).join('')}</tbody>
    </table></div>` : ''}

    <h4 style="font-size:0.85rem;font-weight:700;margin:1rem 0 0.5rem;color:#1e293b;">Data Records Preview (rows ${(data.page-1)*20+1}–${Math.min(data.page*20, data.total_rows)})</h4>
    <div class="table-wrapper"><table class="data-table">
      <thead><tr>${(data.columns||[]).map(c=>`<th>${c}</th>`).join('')}</tr></thead>
      <tbody>${(data.data||[]).map(row =>
        `<tr>${(data.columns||[]).map(c=>`<td>${row[c]??''}</td>`).join('')}</tr>`
      ).join('')}</tbody>
    </table></div>

    <div class="table-pagination">
      ${data.page > 1 ? `<button class="page-btn" onclick="loadExploration(${data.page-1})">← Prev</button>` : ''}
      <span class="page-info">Page ${data.page} of ${data.total_pages}</span>
      ${data.page < data.total_pages ? `<button class="page-btn" onclick="loadExploration(${data.page+1})">Next →</button>` : ''}
    </div>
  </div>`;

  container.innerHTML = html;
}

// ─── FEATURE DISTRIBUTION HANDLER ─────────────────────────────
async function onExplorationFeatureChange() {
  const featSelect = document.getElementById('explFeatureSelect');
  const datasetSelect = document.getElementById('explorationDataset');
  if (!featSelect || !datasetSelect) return;
  const feature = featSelect.value;
  const dataset = datasetSelect.value;
  loadFeatureDistribution(dataset, feature);
}

async function loadFeatureDistribution(dataset, feature) {
  if (!feature) return;
  const summaryEl = document.getElementById('explFeatureSummary');
  if (summaryEl) summaryEl.textContent = `Loading distribution for ${feature}...`;

  try {
    const res = await fetch(`/api/academic/exploration?dataset=${dataset}&feature=${encodeURIComponent(feature)}&page_size=1`);
    const data = await res.json();
    const fDist = data.feature_distribution;
    if (!fDist) {
      if (summaryEl) summaryEl.textContent = `No distribution available for ${feature}`;
      return;
    }

    if (STATE.chartExplFeat) {
      try { STATE.chartExplFeat.destroy(); } catch(e){}
    }
    const ctxFeat = document.getElementById('chart_expl_feat');
    if (!ctxFeat) return;

    if (fDist.type === 'numeric') {
      const b = fDist.box || {};
      if (summaryEl) {
        summaryEl.innerHTML = `<strong>Numeric Feature:</strong> Min: ${b.min} | Q1: ${b.q1} | Median: ${b.median} | Q3: ${b.q3} | Max: ${b.max}`;
      }
      STATE.chartExplFeat = new Chart(ctxFeat, {
        type: 'bar',
        data: {
          labels: fDist.histogram.labels,
          datasets: [{
            label: `Frequency (${feature})`,
            data: fDist.histogram.counts,
            backgroundColor: '#2563ebaa',
            borderRadius: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => ` Frequency: ${ctx.parsed.y}` } }
          },
          scales: {
            y: { beginAtZero: true, title: { display: true, text: 'Frequency' } },
            x: { ticks: { maxRotation: 45, font: { size: 9 } } }
          }
        }
      });
    } else {
      if (summaryEl) {
        summaryEl.innerHTML = `<strong>Categorical Feature:</strong> Top categories by frequency`;
      }
      STATE.chartExplFeat = new Chart(ctxFeat, {
        type: 'bar',
        data: {
          labels: fDist.categories,
          datasets: [{
            label: `Count (${feature})`,
            data: fDist.counts,
            backgroundColor: '#7c3aedaa',
            borderRadius: 3
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => ` Count: ${ctx.parsed.x}` } }
          },
          scales: {
            x: { beginAtZero: true, title: { display: true, text: 'Count' } },
            y: { ticks: { font: { size: 9 } } }
          }
        }
      });
    }
  } catch (err) {
    if (summaryEl) summaryEl.textContent = `Failed to load distribution: ${err.message}`;
  }
}

// ─── CHART HELPERS ────────────────────────────────────────────
function chartCardHTML(title, canvasId) {
  return `<div class="chart-card">
    <div class="chart-card-title">${title}</div>
    <div class="chart-canvas-wrap"><canvas id="${canvasId}"></canvas></div>
  </div>`;
}

function chartCardHTMLWithAction(title, canvasId, downloadId, downloadFilename = 'chart') {
  return `<div class="chart-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.7rem;gap:0.5rem;flex-wrap:wrap;">
      <div class="chart-card-title" style="margin-bottom:0;">${title}</div>
      <button class="btn-chart-action" onclick="downloadChartPNG('${canvasId}', '${downloadFilename}')" title="Download chart as PNG">📷 PNG</button>
    </div>
    <div class="chart-canvas-wrap"><canvas id="${canvasId}"></canvas></div>
  </div>`;
}

const PALETTE = ['#2563eb','#7c3aed','#0891b2','#059669','#d97706','#dc2626','#f59e0b','#84cc16','#06b6d4','#ec4899'];

function makeBarChart(canvasId, labels, data, ylabel, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data, backgroundColor: colors || PALETTE, borderRadius: 4 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, title: { display: !!ylabel, text: ylabel } },
        x: { ticks: { maxRotation: 40, font: { size: 10 } } }
      }
    }
  });
}

function makeHorizBarChart(canvasId, labels, data, xlabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data, backgroundColor: PALETTE[0], borderRadius: 3, barThickness: 14 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, title: { display: true, text: xlabel } } }
    }
  });
}

function makeLineChart(canvasId, xData, yData, label, color, xlabel, ylabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: xData,
      datasets: [{
        label, data: yData, borderColor: color, backgroundColor: color+'22',
        pointRadius: 0, fill: true, tension: 0.3
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: !!xlabel, text: xlabel }, ticks: { maxTicksLimit: 8 } },
        y: { title: { display: !!ylabel, text: ylabel }, beginAtZero: false }
      }
    }
  });
}

function makeLineChartXY(canvasId, labels, data, label, color, xlabel, ylabel) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{ label, data, borderColor: color, backgroundColor: color+'22', pointRadius: 4, fill: true, tension: 0.2 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: !!xlabel, text: xlabel } },
        y: { title: { display: !!ylabel, text: ylabel }, beginAtZero: false }
      }
    }
  });
}

function makeScatterChart(canvasId, xData, yData, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  const data = xData.map((x,i) => ({ x, y: yData[i] }));
  return new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label:'', data, backgroundColor: color+'88', pointRadius: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Actual' } },
        y: { title: { display: true, text: 'Predicted' } }
      }
    }
  });
}

function makeScatterSingle(canvasId, pts, color, label) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [{ label, data: pts, backgroundColor: color+'88', pointRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Support' } },
        y: { title: { display: true, text: 'Confidence' } }
      }
    }
  });
}

function makeScatterMulti(canvasId, datasets) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'PCA Component 1' } },
        y: { title: { display: true, text: 'PCA Component 2' } }
      }
    }
  });
}

// ─── HISTOGRAM HELPER ─────────────────────────────────────────
function histogram(arr, bins) {
  const min = Math.min(...arr), max = Math.max(...arr);
  const step = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  const labels = [];
  for (let i = 0; i < bins; i++) {
    labels.push(`${(min + i*step).toFixed(1)}`);
  }
  arr.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), bins - 1);
    counts[idx]++;
  });
  return { labels, counts };
}

// ─── DOM HELPERS ──────────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if (el) el.classList.remove('hidden'); }
function hide(id) { const el = document.getElementById(id); if (el) el.classList.add('hidden'); }
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
