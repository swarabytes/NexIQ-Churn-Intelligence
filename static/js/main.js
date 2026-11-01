document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // GLOBAL UI UTILITIES
    // ============================================================

    // ---- Light & Dark Theme Toggle ----
    function initThemeToggle() {
        const toggleBtn = document.getElementById('theme-toggle');
        const root = document.documentElement;

        function updateToggleState(theme) {
            if (!toggleBtn) return;
            const iconEl = toggleBtn.querySelector('.theme-toggle-icon');
            const labelEl = toggleBtn.querySelector('.theme-toggle-label');
            if (theme === 'light') {
                if (iconEl) iconEl.textContent = '☀️';
                if (labelEl) labelEl.textContent = 'Light';
                toggleBtn.setAttribute('title', 'Switch to Dark Mode');
                toggleBtn.setAttribute('aria-label', 'Current theme is Light. Click to switch to Dark Mode.');
            } else {
                if (iconEl) iconEl.textContent = '🌙';
                if (labelEl) labelEl.textContent = 'Dark';
                toggleBtn.setAttribute('title', 'Switch to Light Mode');
                toggleBtn.setAttribute('aria-label', 'Current theme is Dark. Click to switch to Light Mode.');
            }
        }

        const initialTheme = root.getAttribute('data-theme') || localStorage.getItem('nexiq_theme') || 'dark';
        root.setAttribute('data-theme', initialTheme);
        updateToggleState(initialTheme);

        if (toggleBtn) {
            toggleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                const currentTheme = root.getAttribute('data-theme') || 'dark';
                const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
                root.setAttribute('data-theme', nextTheme);
                try {
                    localStorage.setItem('nexiq_theme', nextTheme);
                } catch (err) {}
                updateToggleState(nextTheme);
                window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: nextTheme } }));
            });
        }
    }
    initThemeToggle();

    // ---- Toast Notification System ----
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }

    function showToast(message, type = 'success', duration = 3500) {
        const icons = { success: '✅', error: '❌', warning: '⚠️' };
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || '💬'}</span>
            <span class="toast-msg">${message}</span>
            <button class="toast-close" aria-label="Close">×</button>
        `;
        toastContainer.appendChild(toast);

        const removeToast = () => {
            toast.classList.add('removing');
            toast.addEventListener('animationend', () => toast.remove(), { once: true });
        };

        toast.querySelector('.toast-close').addEventListener('click', removeToast);
        setTimeout(removeToast, duration);
    }

    // ---- Loading Spinner Overlay ----
    let loaderOverlay = document.querySelector('.loader-overlay');
    if (!loaderOverlay) {
        loaderOverlay = document.createElement('div');
        loaderOverlay.className = 'loader-overlay';
        loaderOverlay.innerHTML = `
            <div class="loader-spinner"></div>
            <span class="loader-text">Analyzing customer data...</span>
        `;
        document.body.appendChild(loaderOverlay);
    }

    function showLoader(text = 'Analyzing customer data...') {
        loaderOverlay.querySelector('.loader-text').textContent = text;
        loaderOverlay.classList.add('active');
    }

    function hideLoader() {
        loaderOverlay.classList.remove('active');
    }


    // ============================================================
    // CONSTANTS
    // ============================================================
    const SAMPLE_DATA = {
        "recency": 15,
        "frequency": 12,
        "monetary": 450,
        "average_order_value": 37.5,
        "category_diversity": 4,
        "average_review_score": 4.2,
        "payment_count": 15,
        "delivery_delay": 2,
        "inactivity_days": 15
    };


    // ============================================================
    // PREDICT PAGE
    // ============================================================
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        setupPredictPage();
        setupChurnTabsAndComparison();
    }

    function setupPredictPage() {
        // Keep Recency and Inactivity Days synchronized for seamless UX
        const recencyInput = document.getElementById('recency');
        const inactInput = document.getElementById('inactivity_days');
        if (recencyInput && inactInput) {
            recencyInput.addEventListener('input', () => {
                inactInput.value = recencyInput.value;
                const err = document.getElementById('err-inactivity_days');
                if (err) err.classList.remove('visible');
            });
            inactInput.addEventListener('input', () => {
                recencyInput.value = inactInput.value;
                const err = document.getElementById('err-recency');
                if (err) err.classList.remove('visible');
            });
        }
        const btnAutofill = document.getElementById('btn-autofill');
        const btnPasteJson = document.getElementById('btn-paste-json');
        const jsonInput = document.getElementById('json-input');
        const btnParseJson = document.getElementById('btn-parse-json');

        // Add error message spans under each input
        predictForm.querySelectorAll('.form-group input').forEach(input => {
            const errMsg = document.createElement('span');
            errMsg.className = 'field-error-msg';
            errMsg.id = `err-${input.name}`;
            input.parentElement.appendChild(errMsg);

            // Live validation on blur
            input.addEventListener('blur', () => validateField(input));
            input.addEventListener('input', () => {
                if (input.classList.contains('input-error')) validateField(input);
            });
        });

        // Autofill
        btnAutofill.addEventListener('click', () => {
            for (const key in SAMPLE_DATA) {
                const input = document.getElementById(key);
                if (input) {
                    input.value = SAMPLE_DATA[key];
                    input.classList.remove('input-error');
                    input.classList.add('input-valid');
                    const errEl = document.getElementById(`err-${key}`);
                    if (errEl) errEl.classList.remove('visible');
                }
            }
            showToast('Sample data applied!', 'success', 2000);
        });

        // Toggle JSON Textarea
        btnPasteJson.addEventListener('click', () => {
            const isHidden = jsonInput.style.display === 'none';
            jsonInput.style.display = isHidden ? 'block' : 'none';
            btnParseJson.style.display = isHidden ? 'inline-block' : 'none';
        });

        // Parse JSON
        btnParseJson.addEventListener('click', () => {
            try {
                const data = JSON.parse(jsonInput.value);
                for (const key in data) {
                    const input = document.getElementById(key);
                    if (input) input.value = data[key];
                }
                jsonInput.style.display = 'none';
                btnParseJson.style.display = 'none';
                showToast('JSON applied successfully!', 'success', 2000);
            } catch (e) {
                showToast('Invalid JSON format. Please check your input.', 'error');
            }
        });

        // Form Submit
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Validate all fields first
            let isValid = true;
            predictForm.querySelectorAll('.form-group input').forEach(input => {
                if (!validateField(input)) isValid = false;
            });

            if (!isValid) {
                showToast('Please fix the validation errors before submitting.', 'error');
                return;
            }

            // Build payload
            const formData = new FormData(predictForm);
            const payload = {};
            formData.forEach((value, key) => { payload[key] = parseFloat(value); });

            const btnPredict = document.getElementById('btn-predict');
            btnPredict.disabled = true;
            showLoader('Analyzing customer risk profile...');

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await res.json();
                if (result.error) throw new Error(result.error);

                showResult(result);
                showToast(`Analysis complete — Risk Level: ${result.risk_level}`, 'success');
                document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
            } catch (err) {
                showToast('Analysis failed: ' + err.message, 'error');
            } finally {
                hideLoader();
                btnPredict.disabled = false;
            }
        });
    }

    function validateField(input) {
        const errEl = document.getElementById(`err-${input.name}`);
        const val = input.value.trim();
        let error = '';

        if (val === '' || val === null) {
            error = 'This field is required.';
        } else if (isNaN(parseFloat(val))) {
            error = 'Must be a valid number.';
        } else if (parseFloat(val) < 0) {
            error = 'Value cannot be negative.';
        }

        if (error) {
            input.classList.add('input-error');
            input.classList.remove('input-valid');
            if (errEl) { errEl.textContent = error; errEl.classList.add('visible'); }
            return false;
        } else {
            input.classList.remove('input-error');
            input.classList.add('input-valid');
            if (errEl) errEl.classList.remove('visible');
            return true;
        }
    }



    // ============================================================
    // CUSTOMER CHURN — TABS & MODEL COMPARISON
    // ============================================================
    function setupChurnTabsAndComparison() {
        const tabBtns = document.querySelectorAll('.churn-tab-btn');
        if (!tabBtns.length) return;

        let comparisonDataLoaded = false;
        let metricsChartInstance = null;
        let featureChartInstance = null;

        function switchChurnTab(targetTabId) {
            tabBtns.forEach(btn => {
                const isActive = btn.getAttribute('data-tab') === targetTabId;
                btn.classList.toggle('active', isActive);
                btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
            });

            document.querySelectorAll('.churn-tab-pane').forEach(pane => {
                pane.style.display = (pane.id === targetTabId) ? 'block' : 'none';
            });

            if (targetTabId === 'tab-model-comparison' && !comparisonDataLoaded) {
                loadModelComparisonData();
            }
        }

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTabId = btn.getAttribute('data-tab');
                switchChurnTab(targetTabId);
            });
        });

        // Jump button from Single Result to Model Comparison
        const btnJump = document.getElementById('btn-jump-to-compare');
        if (btnJump) {
            btnJump.addEventListener('click', () => {
                switchChurnTab('tab-model-comparison');
                document.getElementById('tab-model-comparison').scrollIntoView({ behavior: 'smooth' });
                runLiveComparison();
            });
        }

        // Live simulator buttons
        const btnSync = document.getElementById('btn-sync-prediction-data');
        if (btnSync) {
            btnSync.addEventListener('click', () => {
                showToast('Synchronized form parameters!', 'success', 2000);
                runLiveComparison();
            });
        }

        const btnAtRisk = document.getElementById('btn-sample-at-risk');
        if (btnAtRisk) {
            btnAtRisk.addEventListener('click', () => {
                const atRiskSample = {
                    recency: 110,
                    frequency: 1,
                    monetary: 75,
                    average_order_value: 75,
                    category_diversity: 1,
                    average_review_score: 1.5,
                    payment_count: 1,
                    delivery_delay: 8,
                    inactivity_days: 120
                };
                for (const k in atRiskSample) {
                    const el = document.getElementById(k);
                    if (el) el.value = atRiskSample[k];
                }
                showToast('Loaded dormant at-risk profile!', 'warning', 2500);
                runLiveComparison(atRiskSample);
            });
        }

        const btnRunAll = document.getElementById('btn-run-all-models');
        if (btnRunAll) {
            btnRunAll.addEventListener('click', () => {
                runLiveComparison();
            });
        }

        // Fetch and load benchmark data
        async function loadModelComparisonData() {
            try {
                const res = await fetch('/api/academic/classification');
                const data = await res.json();
                if (!Array.isArray(data)) throw new Error('Invalid format');
                renderComparisonTable(data);
                renderComparisonCharts(data);
                comparisonDataLoaded = true;
            } catch (err) {
                console.warn('Could not load comparison data', err);
            }
        }

        function renderComparisonTable(models) {
            const tbody = document.getElementById('model-comparison-table-body');
            if (!tbody) return;

            const roles = {
                'Logistic Regression': { role: '👑 Production Champion', badge: 'low' },
                'Random Forest':       { role: 'Ensemble Benchmark',    badge: 'medium' },
                'XGBoost':             { role: 'Gradient Boosting Benchmark', badge: 'high' }
            };

            const modelIcons = {
                'Logistic Regression': '📈',
                'Random Forest':       '🌲',
                'XGBoost':             '⚡'
            };

            tbody.innerHTML = models.map(m => {
                const name = m.model || 'Unknown';
                const icon = modelIcons[name] || '📊';
                const acc = (m.accuracy * 100).toFixed(1) + '%';
                const prec = (m.precision * 100).toFixed(1) + '%';
                const rec = (m.recall * 100).toFixed(1) + '%';
                const f1 = m.f1_score ? m.f1_score.toFixed(3) : 'N/A';
                const spec = m.specificity ? (m.specificity * 100).toFixed(1) + '%' : 'N/A';
                const roc = m.roc_auc ? m.roc_auc.toFixed(3) : 'N/A';
                const cm = m.confusion_matrix || {};
                const roleInfo = roles[name] || { role: 'Evaluated Model', badge: 'low' };

                return `
                    <tr>
                        <td style="white-space: nowrap;">
                            <div style="display: flex; align-items: center; gap: 0.6rem;">
                                <span style="font-size: 1.15rem; line-height: 1;">${icon}</span>
                                <strong style="font-size: 0.92rem; color: var(--text-primary);">${name}</strong>
                            </div>
                        </td>
                        <td style="color: var(--success); font-weight: 700; text-align: center; white-space: nowrap; font-family: var(--font-mono);">${acc}</td>
                        <td style="text-align: center; white-space: nowrap; font-family: var(--font-mono);">${prec}</td>
                        <td style="text-align: center; white-space: nowrap; font-family: var(--font-mono);">${rec}</td>
                        <td style="color: #60a5fa; font-weight: 700; text-align: center; white-space: nowrap; font-family: var(--font-mono);">${f1}</td>
                        <td style="text-align: center; white-space: nowrap; font-family: var(--font-mono);">${spec}</td>
                        <td style="color: #a78bfa; font-weight: 700; text-align: center; white-space: nowrap; font-family: var(--font-mono);">${roc}</td>
                        <td style="text-align: center; white-space: nowrap;">
                            <div class="cm-table-inline">
                                <span class="cm-tag cm-tag-tn" title="True Negatives (Retained)">TN: ${cm.TN ?? 0}</span>
                                <span class="cm-tag cm-tag-fp" title="False Positives (Type I Error)">FP: ${cm.FP ?? 0}</span>
                                <span class="cm-tag cm-tag-fn" title="False Negatives (Missed Churn)">FN: ${cm.FN ?? 0}</span>
                                <span class="cm-tag cm-tag-tp" title="True Positives (Caught Churn)">TP: ${cm.TP ?? 0}</span>
                            </div>
                        </td>
                        <td style="text-align: center; white-space: nowrap;">
                            <span class="matrix-status-pill ${roleInfo.badge}">${roleInfo.role}</span>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function renderComparisonCharts(models) {
            const ctxBar = document.getElementById('modelMetricsBarChart');
            if (ctxBar) {
                if (metricsChartInstance) metricsChartInstance.destroy();
                const labels = models.map(m => m.model);
                metricsChartInstance = new Chart(ctxBar, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Accuracy (%)',
                                data: models.map(m => (m.accuracy * 100).toFixed(1)),
                                backgroundColor: 'rgba(59, 130, 246, 0.85)'
                            },
                            {
                                label: 'Precision (%)',
                                data: models.map(m => (m.precision * 100).toFixed(1)),
                                backgroundColor: 'rgba(16, 185, 129, 0.85)'
                            },
                            {
                                label: 'Recall (%)',
                                data: models.map(m => (m.recall * 100).toFixed(1)),
                                backgroundColor: 'rgba(245, 158, 11, 0.85)'
                            },
                            {
                                label: 'F1-Score (×100)',
                                data: models.map(m => (m.f1_score * 100).toFixed(1)),
                                backgroundColor: 'rgba(168, 85, 247, 0.85)'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: { callback: v => v + '%' }
                            }
                        }
                    }
                });
            }

            const ctxFeat = document.getElementById('featureImportanceCompareChart');
            if (ctxFeat) {
                if (featureChartInstance) featureChartInstance.destroy();

                const rfModel  = models.find(m => m.model === 'Random Forest') || {};
                const xgbModel = models.find(m => m.model === 'XGBoost') || {};
                const fi = rfModel.feature_importance || xgbModel.feature_importance || {};
                const featLabels = Object.keys(fi).map(k => k.replace(/_/g, ' '));
                const rfValues  = Object.values(rfModel.feature_importance || {}).map(v => (v * 100).toFixed(1));
                const xgbValues = Object.values(xgbModel.feature_importance || {}).map(v => (v * 100).toFixed(1));

                featureChartInstance = new Chart(ctxFeat, {
                    type: 'bar',
                    data: {
                        labels: featLabels.length ? featLabels : ['inactivity days', 'recency', 'monetary', 'average review score', 'frequency', 'delivery delay'],
                        datasets: [
                            {
                                label: 'Random Forest Importance (%)',
                                data: rfValues.length ? rfValues : [43.4, 23.7, 11.2, 6.2, 3.5, 2.8],
                                backgroundColor: 'rgba(168, 85, 247, 0.85)'
                            },
                            {
                                label: 'XGBoost Importance (%)',
                                data: xgbValues.length ? xgbValues : [38.2, 25.1, 14.8, 8.3, 7.6, 6.0],
                                backgroundColor: 'rgba(16, 185, 129, 0.85)'
                            }
                        ]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'bottom' }
                        },
                        scales: {
                            x: {
                                beginAtZero: true,
                                ticks: { callback: v => v + '%' }
                            }
                        }
                    }
                });
            }
        }

        async function runLiveComparison(customPayload = null) {
            let payload = customPayload;
            if (!payload) {
                payload = {};
                const form = document.getElementById('predict-form');
                if (form) {
                    const fd = new FormData(form);
                    fd.forEach((v, k) => { payload[k] = parseFloat(v) || 0; });
                }
            }

            if (!payload.recency && !payload.monetary) {
                payload = { ...SAMPLE_DATA };
            }

            const grid = document.getElementById('live-models-output-grid');
            if (!grid) return;
            grid.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--text-secondary);">Calculating live inference across all models...</div>';

            try {
                const res = await fetch('/api/compare-predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.error);

                const banner = document.getElementById('compare-consensus-banner');
                if (banner) {
                    banner.style.display = 'flex';
                    document.getElementById('consensus-heading').textContent = 'Model Agreement Consensus';
                    document.getElementById('consensus-sub').textContent = data.consensus;
                    document.getElementById('consensus-percentage').textContent = data.average_probability + '%';
                }

                const icons = {
                    'logistic_regression': '📈',
                    'random_forest':       '🌲',
                    'xgboost':             '⚡'
                };

                grid.innerHTML = data.models.map(m => {
                    const riskCls = m.risk_level.toLowerCase() + '-risk';
                    const icon = icons[m.model_key] || '📊';
                    return `
                        <div class="live-model-result-card ${riskCls}">
                            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">${icon}</div>
                            <div class="live-model-name">${m.model_name}</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">${m.model_type}</div>
                            
                            <div class="live-model-prob ${riskCls}">${m.probability}%</div>
                            
                            <div style="margin: 0.5rem 0;">
                                <span class="badge ${m.risk_level.toLowerCase()}">${m.risk_level} Churn Risk</span>
                            </div>
                            
                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.5rem;">
                                Prediction: <strong style="color: var(--text-primary);">${m.prediction === 1 ? 'High Risk' : 'Retained'}</strong>
                            </div>
                        </div>
                    `;
                }).join('');

            } catch (err) {
                grid.innerHTML = `<div style="grid-column: 1 / -1; color: var(--danger); text-align: center; padding: 1rem;">Simulation error: ${err.message}</div>`;
            }
        }
    }


    function showResult(result) {
        const section = document.getElementById('result-section');
        section.style.display = 'block';

        document.getElementById('res-cust-id').innerText = result.customer_id;

        const badge = document.getElementById('res-badge');
        badge.className = `badge ${result.risk_level.toLowerCase()}`;
        badge.innerText = `${result.risk_level} Risk`;

        if (window._gaugeInterval) clearInterval(window._gaugeInterval);

        const targetPercent = Math.max(0, Math.min(100, Math.round(result.probability * 100)));
        const probAnim = document.getElementById('res-prob-anim');
        const circProg = document.getElementById('circular-progress');

        let color = '#10b981';
        if (result.risk_level === 'Medium') color = '#f59e0b';
        if (result.risk_level === 'High') color = '#ef4444';

        if (targetPercent === 0) {
            probAnim.innerText = '0%';
            circProg.style.background = `conic-gradient(${color} 0deg, rgba(255,255,255,0.07) 0deg)`;
        } else {
            let cur = 0;
            const step = Math.max(1, Math.ceil(targetPercent / 25));
            window._gaugeInterval = setInterval(() => {
                cur += step;
                if (cur >= targetPercent) {
                    cur = targetPercent;
                    clearInterval(window._gaugeInterval);
                }
                probAnim.innerText = `${cur}%`;
                circProg.style.background = `conic-gradient(${color} ${cur * 3.6}deg, rgba(255,255,255,0.07) 0deg)`;
            }, 18);
        }

        const healthScore = result.health_score !== undefined
            ? Math.round(result.health_score)
            : Math.max(0, Math.min(100, Math.round((1.0 - result.probability) * 100)));

        const healthVal = document.getElementById('res-health-val');
        if (healthVal) healthVal.innerText = `${healthScore}%`;

        const healthFill = document.getElementById('res-health-fill');
        if (healthFill) {
            setTimeout(() => {
                healthFill.style.width = `${healthScore}%`;
            }, 150);
        }

        document.getElementById('res-insight').innerText =
            result.insight || 'Customer shows stable purchasing behavior.';

        const actions = Array.isArray(result.recommended_action)
            ? result.recommended_action
            : [result.recommended_action || 'Continue standard engagement.'];
        document.getElementById('res-actions').innerHTML =
            actions.map(a => `<li>${a}</li>`).join('');
    }


    // ============================================================
    // DASHBOARD PAGE
    // ============================================================
    if (document.getElementById('kpi-total')) {
        setupDashboardPage();
    }

    // ---- Pagination State ----
    let allPredictions = [];
    let filteredData = [];
    let currentPage = 1;
    const PAGE_SIZE = 10;

    let chartState = {
        distMetric: 'count',
        spendView: 'total',
        trendMetric: 'probability',
        trendRange: '20',
        activeRiskFilter: null
    };

    async function setupDashboardPage() {
        try {
            showLoader('Loading intelligence data...');
            const res = await fetch('/api/dashboard-data');
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            allPredictions = data.recent_predictions || [];

            // KPIs
            animateValue('kpi-total', 0, data.stats.total_predictions, 1000);
            animateValue('kpi-high-risk', 0, data.stats.high_risk_count, 1000);
            const avgPct = Math.round((data.stats.avg_risk_probability || 0) * 100);
            document.getElementById('kpi-avg-risk').innerText = `${avgPct}%`;
            document.getElementById('kpi-rev-risk').innerText = `$${Number(data.stats.revenue_at_risk).toLocaleString()}`;

            // Executive Summary
            const total = data.stats.total_predictions;
            const pctHigh = total > 0 ? Math.round((data.stats.high_risk_count / total) * 100) : 0;
            const execEl = document.getElementById('exec-summary-text');
            if (execEl) execEl.innerText =
                `${pctHigh}% of your customers are identified as High Risk. ` +
                `Revenue loss potential stands at $${Number(data.stats.revenue_at_risk).toLocaleString()}. ` +
                `${data.alerts.length} active alert${data.alerts.length !== 1 ? 's' : ''} require attention.`;

            // Alerts
            renderAlerts(data.alerts);

            // Charts
            buildCharts(allPredictions);

            // Table with search/filter/pagination
            filteredData = [...allPredictions];
            renderTable();
            setupTableControls();

            hideLoader();
        } catch (err) {
            hideLoader();
            showToast('Failed to load dashboard data: ' + err.message, 'error');
            console.error('Dashboard load error:', err);
        }

        // Side panel close
        document.getElementById('close-panel').addEventListener('click', closeSidePanel);
        document.getElementById('panel-overlay').addEventListener('click', closeSidePanel);
    }

    function renderAlerts(alerts) {
        const list = document.getElementById('alerts-list');
        const countBadge = document.getElementById('alerts-count');
        if (!list) return;

        if (countBadge) countBadge.textContent = alerts.length;

        if (!alerts || alerts.length === 0) {
            list.innerHTML = '<p style="color:var(--text-secondary); font-size:0.9rem;">✅ No active alerts — all systems normal.</p>';
            return;
        }

        list.innerHTML = '';
        alerts.forEach(alert => {
            const div = document.createElement('div');
            div.className = `alert-item ${alert.type}`;
            div.innerHTML = `
                <div class="alert-item-cust">${alert.customer_id}</div>
                <div class="alert-item-msg">${alert.type === 'danger' ? '🔴' : '🟡'} ${alert.message}</div>
            `;
            list.appendChild(div);
        });
    }

    // ---- Search, Filter, Pagination ----
    function setupTableControls() {
        const searchInput = document.getElementById('searchInput');
        const filterRisk = document.getElementById('filterRisk');
        const filterPrio = document.getElementById('filterPriority');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');

        [searchInput, filterRisk, filterPrio].forEach(el => {
            if (el) el.addEventListener('input', () => { currentPage = 1; applyFilters(); });
        });

        if (btnPrev) btnPrev.addEventListener('click', () => { if (currentPage > 1) { currentPage--; renderTable(); } });
        if (btnNext) btnNext.addEventListener('click', () => {
            const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
            if (currentPage < totalPages) { currentPage++; renderTable(); }
        });
    }

    function applyFilters() {
        const query = (document.getElementById('searchInput')?.value || '').toLowerCase();
        const risk = (document.getElementById('filterRisk')?.value || '');
        const priority = (document.getElementById('filterPriority')?.value || '');

        chartState.activeRiskFilter = risk || null;
        if (typeof updateChartFilterUI === 'function') {
            updateChartFilterUI();
        }

        filteredData = allPredictions.filter(p => {
            const matchSearch = !query || p.customer_id.toLowerCase().includes(query);
            const matchRisk = !risk || p.risk_level === risk;
            const matchPrio = !priority || p.priority === priority;
            return matchSearch && matchRisk && matchPrio;
        });

        renderTable();
    }

    function renderTable() {
        const tbody = document.getElementById('customer-table-body');
        const noMsg = document.getElementById('no-data-msg');
        if (!tbody) return;

        tbody.innerHTML = '';

        if (filteredData.length === 0) {
            if (noMsg) noMsg.style.display = 'block';
            updatePagination(0);
            return;
        }
        if (noMsg) noMsg.style.display = 'none';

        const totalPages = Math.ceil(filteredData.length / PAGE_SIZE);
        if (currentPage > totalPages) currentPage = totalPages;

        const start = (currentPage - 1) * PAGE_SIZE;
        const pageRows = filteredData.slice(start, start + PAGE_SIZE);

        pageRows.forEach((p, i) => {
            const tr = document.createElement('tr');
            tr.style.animationDelay = `${i * 40}ms`;
            tr.classList.add('anim-fade-in');

            const riskClass = p.risk_level.toLowerCase();
            const probPct = Math.round(p.probability * 100);

            let prioClass = 'low';
            if (p.priority === 'Critical') prioClass = 'high';
            if (p.priority === 'Important') prioClass = 'medium';

            tr.innerHTML = `
                <td><strong>${p.customer_id}</strong></td>
                <td>${p.recency} days</td>
                <td>$${Number(p.monetary).toLocaleString()}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <div style="width:50px; height:6px; background:var(--border-light); border-radius:3px; overflow:hidden;">
                            <div style="width:${probPct}%; height:100%; background:${riskClass === 'high' ? 'var(--danger)' : riskClass === 'medium' ? 'var(--warning)' : 'var(--success)'}; border-radius:3px;"></div>
                        </div>
                        ${probPct}%
                    </div>
                </td>
                <td><span class="badge ${riskClass}">${p.risk_level}</span></td>
                <td><span class="badge ${prioClass}">${p.priority || 'Monitor'}</span></td>
                <td>
                    <button class="btn-expand" data-id="${p.id}">View →</button>
                    <button class="btn-delete" data-id="${p.id}" style="color: #ff4757; background: transparent; border: 1px solid rgba(255,71,87,0.5); border-radius: 4px; padding: 2px 6px; cursor: pointer; margin-left: 5px; font-size: 0.8rem;">Delete</button>
                </td>
            `;
            tbody.appendChild(tr);
        });

        // Re-bind expand & delete buttons
        document.querySelectorAll('.btn-expand').forEach(btn => {
            btn.addEventListener('click', e => {
                const id = e.currentTarget.getAttribute('data-id');
                const rowData = filteredData.find(p => p.id == id);
                if (rowData) openSidePanel(rowData);
            });
        });

        document.querySelectorAll('.btn-delete').forEach(btn => {
            btn.addEventListener('click', e => {
                const id = e.currentTarget.getAttribute('data-id');
                deletePrediction(id);
            });
        });

        updatePagination(filteredData.length);
    }

    function updatePagination(total) {
        const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
        const start = Math.min((currentPage - 1) * PAGE_SIZE + 1, total);
        const end = Math.min(currentPage * PAGE_SIZE, total);

        const infoEl = document.getElementById('pagination-info');
        if (infoEl) infoEl.textContent = total > 0 ? `Showing ${start}–${end} of ${total}` : 'No results';

        const indEl = document.getElementById('page-indicator');
        if (indEl) indEl.textContent = `Page ${currentPage} of ${totalPages}`;

        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');
        if (btnPrev) btnPrev.disabled = currentPage <= 1;
        if (btnNext) btnNext.disabled = currentPage >= totalPages;
    }


    // ---- Side Panel ----
    function openSidePanel(data) {
        const panel = document.getElementById('side-panel');
        const overlay = document.getElementById('panel-overlay');
        const content = document.getElementById('panel-content');

        let insight = 'Customer shows stable purchasing behavior.';
        let action = 'Continue standard engagement campaigns.';

        if (data.inactivity_days > 90) {
            insight = 'Customer inactive for long period → High churn risk.';
            action = 'Send win-back email campaign with exclusive discount.';
        } else if (data.monetary > 500 && data.probability > 0.7) {
            insight = 'High value customer is at significant risk of leaving.';
            action = 'Direct VIP outreach via dedicated account manager.';
        } else if (data.average_review_score < 3.0) {
            insight = 'Consistently poor reviews → Bad experience risk.';
            action = 'Escalate to customer support for immediate resolution.';
        } else if (data.risk_level === 'Medium') {
            insight = 'Moderate drop in engagement detected.';
            action = 'Send personalized recommendations & loyalty reward.';
        }

        const healthScore = Math.round((1 - data.probability) * 100);
        const riskColor = data.risk_level === 'High' ? 'var(--danger)' : data.risk_level === 'Medium' ? 'var(--warning)' : 'var(--success)';

        content.innerHTML = `
            <div style="margin-top:1.2rem;">
                <p style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.2rem;">CUSTOMER ID</p>
                <h4 style="font-size:1.1rem; margin-bottom:1.5rem; letter-spacing:0.02em;">${data.customer_id}</h4>
                
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-bottom:1.5rem;">
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Risk Level</span>
                        <span class="badge ${data.risk_level.toLowerCase()}" style="display:inline-block;">${data.risk_level}</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Probability</span>
                        <span style="font-weight:700; font-size:1.4rem; color:${riskColor};">${Math.round(data.probability * 100)}%</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Health Score</span>
                        <span style="font-weight:700; font-size:1.4rem; color:var(--success);">${healthScore}/100</span>
                    </div>
                    <div style="background:rgba(0,0,0,0.25); padding:1rem; border-radius:10px; text-align:center;">
                        <span style="font-size:0.75rem; color:var(--text-secondary); text-transform:uppercase; display:block; margin-bottom:0.3rem;">Priority</span>
                        <span class="badge ${data.priority === 'Critical' ? 'high' : data.priority === 'Important' ? 'medium' : 'low'}" style="display:inline-block;">${data.priority || 'Monitor'}</span>
                    </div>
                </div>

                <div style="background:rgba(239,68,68,0.1); padding:1rem; border-radius:10px; border-left:3px solid var(--danger); margin-bottom:1rem;">
                    <h5 style="color:var(--danger); margin-bottom:0.4rem; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">⚡ Risk Explanation</h5>
                    <p style="font-size:0.9rem; line-height:1.5;">${insight}</p>
                </div>
                <div style="background:rgba(16,185,129,0.1); padding:1rem; border-radius:10px; border-left:3px solid var(--success); margin-bottom:1.5rem;">
                    <h5 style="color:var(--success); margin-bottom:0.4rem; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">✅ Suggested Action</h5>
                    <p style="font-size:0.9rem; line-height:1.5;">${action}</p>
                </div>

                <h5 style="border-bottom:1px solid var(--border-light); padding-bottom:0.6rem; margin-bottom:1rem; font-size:0.82rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.05em;">Behavioral Metrics</h5>
                <ul style="list-style:none; display:flex; flex-direction:column; gap:0.6rem;">
                    ${[
                ['Recency', `${data.recency} days`],
                ['Frequency', `${data.frequency} orders`],
                ['Monetary Value', `$${Number(data.monetary).toLocaleString()}`],
                ['Avg Order Value', `$${data.average_order_value}`],
                ['Category Diversity', data.category_diversity],
                ['Avg Review Score', `${data.average_review_score} / 5`],
                ['Payment Count', data.payment_count],
                ['Delivery Delay', `${data.delivery_delay} days`],
                ['Inactivity Days', `${data.inactivity_days} days`],
            ].map(([label, val]) => `
                        <li style="display:flex; justify-content:space-between; align-items:center; font-size:0.88rem;">
                            <span style="color:var(--text-secondary);">${label}</span>
                            <strong>${val}</strong>
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;

        panel.classList.add('open');
        overlay.classList.add('active');
    }

    function closeSidePanel() {
        document.getElementById('side-panel')?.classList.remove('open');
        document.getElementById('panel-overlay')?.classList.remove('active');
    }


    // ---- Animate Counter ----
    function animateValue(id, start, end, duration) {
        const obj = document.getElementById(id);
        if (!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }

    // ============================================================
    // HIGHLY VISUALIZED & INTERACTIVE CHARTS ENGINE
    // ============================================================


    let chartInstances = {
        dist: null,
        spend: null,
        trend: null
    };

    let currentTrendPredictions = [];

    // Filter Priority Queue table when a chart segment is clicked
    function filterTableByRisk(riskTier) {
        const filterEl = document.getElementById('filterRisk');
        if (!filterEl) return;

        if (chartState.activeRiskFilter === riskTier) {
            chartState.activeRiskFilter = null;
            filterEl.value = '';
        } else {
            chartState.activeRiskFilter = riskTier;
            filterEl.value = riskTier;
        }

        currentPage = 1;
        applyFilters();
        updateChartFilterUI();

        const tableEl = document.getElementById('customer-table');
        if (tableEl) {
            tableEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        showToast(
            chartState.activeRiskFilter 
                ? `Filtered priority queue to ${chartState.activeRiskFilter} Risk customers.` 
                : 'Showing all customer risk levels.',
            'success',
            2200
        );
    }

    function updateChartFilterUI() {
        const resetBtn = document.getElementById('resetDistFilter');
        if (resetBtn) {
            resetBtn.style.display = chartState.activeRiskFilter ? 'inline-block' : 'none';
            if (chartState.activeRiskFilter) {
                resetBtn.textContent = `✕ Clear ${chartState.activeRiskFilter} Filter`;
            }
        }
        document.querySelectorAll('.stat-pill').forEach(pill => {
            const risk = pill.getAttribute('data-risk');
            if (risk === chartState.activeRiskFilter) {
                pill.classList.add('active');
            } else {
                pill.classList.remove('active');
            }
        });
    }

    // Chart.js Donut Center Text Plugin
    const donutCenterPlugin = {
        id: 'donutCenterHUD',
        beforeDraw(chart) {
            if (chart.config.type !== 'doughnut') return;
            const { ctx, chartArea } = chart;
            if (!chartArea) return;

            const centerX = (chartArea.left + chartArea.right) / 2;
            const centerY = (chartArea.top + chartArea.bottom) / 2;

            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            const primaryColor = isLight ? '#0f172a' : '#f8fafc';
            const secondaryColor = isLight ? '#64748b' : '#94a3b8';

            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            if (chart.activeHoverInfo) {
                const h = chart.activeHoverInfo;
                // Hover title
                ctx.font = '700 11px Inter, sans-serif';
                ctx.fillStyle = h.color || primaryColor;
                ctx.fillText(h.title.toUpperCase(), centerX, centerY - 16);

                // Hover value
                ctx.font = '800 22px Inter, sans-serif';
                ctx.fillStyle = primaryColor;
                ctx.fillText(h.value, centerX, centerY + 6);

                // Hover subtext
                ctx.font = '600 11px Inter, sans-serif';
                ctx.fillStyle = secondaryColor;
                ctx.fillText(h.sub, centerX, centerY + 25);
            } else {
                // Default center HUD
                const total = chart.centerMainText || '0';
                const sub = chart.centerSubText || 'CUSTOMERS';

                ctx.font = '800 24px Inter, sans-serif';
                ctx.fillStyle = primaryColor;
                ctx.fillText(total, centerX, centerY - 4);

                ctx.font = '600 10px Inter, sans-serif';
                ctx.fillStyle = secondaryColor;
                ctx.fillText(sub, centerX, centerY + 18);
            }
            ctx.restore();
        }
    };

    // Chart.js Bar Value Labels Plugin
    const barValueLabelsPlugin = {
        id: 'barValueLabels',
        afterDatasetsDraw(chart) {
            if (chart.config.type !== 'bar') return;
            const { ctx } = chart;
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            const labelColor = isLight ? '#0f172a' : '#f8fafc';

            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.font = '700 11px Inter, sans-serif';
            ctx.fillStyle = labelColor;

            chart.data.datasets.forEach((dataset, i) => {
                const meta = chart.getDatasetMeta(i);
                meta.data.forEach((bar, index) => {
                    const rawVal = dataset.data[index];
                    if (rawVal === undefined || rawVal === null) return;
                    let displayVal = chartState.spendView === 'avg'
                        ? `$${Math.round(rawVal).toLocaleString()}/avg`
                        : `$${Math.round(rawVal).toLocaleString()}`;
                    if (rawVal === 0) displayVal = '$0';
                    ctx.fillText(displayVal, bar.x, bar.y - 6);
                });
            });
            ctx.restore();
        }
    };

    // Chart.js Trend Reference Line & Threshold Plugin
    const trendThresholdPlugin = {
        id: 'trendThresholdAnnotations',
        beforeDraw(chart) {
            if (chart.config.type !== 'line') return;
            if (chartState.trendMetric !== 'probability') return;

            const { ctx, chartArea, scales } = chart;
            if (!chartArea || !scales.y) return;

            const yScale = scales.y;
            const y70 = yScale.getPixelForValue(70);
            const y40 = yScale.getPixelForValue(40);

            ctx.save();

            // Shaded danger zone (> 70%)
            if (y70 >= chartArea.top && y70 <= chartArea.bottom) {
                ctx.fillStyle = 'rgba(239, 68, 68, 0.06)';
                ctx.fillRect(chartArea.left, chartArea.top, chartArea.width, y70 - chartArea.top);

                // 70% Danger Line
                ctx.beginPath();
                ctx.setLineDash([5, 4]);
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.55)';
                ctx.lineWidth = 1.5;
                ctx.moveTo(chartArea.left, y70);
                ctx.lineTo(chartArea.right, y70);
                ctx.stroke();

                ctx.setLineDash([]);
                ctx.fillStyle = '#ef4444';
                ctx.font = '600 10px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('⚡ 70% HIGH RISK THRESHOLD', chartArea.right - 8, y70 - 4);
            }

            // 40% Caution Line
            if (y40 >= chartArea.top && y40 <= chartArea.bottom) {
                ctx.beginPath();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
                ctx.lineWidth = 1.2;
                ctx.moveTo(chartArea.left, y40);
                ctx.lineTo(chartArea.right, y40);
                ctx.stroke();

                ctx.setLineDash([]);
                ctx.fillStyle = '#f59e0b';
                ctx.font = '600 10px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('40% CAUTION LINE', chartArea.right - 8, y40 - 4);
            }

            ctx.restore();
        }
    };

    function setupChartInteractivity() {
        // Doughnut metric pills
        document.querySelectorAll('#distMetricPills .chart-pill').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#distMetricPills .chart-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                chartState.distMetric = btn.getAttribute('data-metric');
                renderRiskDistChart(allPredictions);
            };
        });

        // Clear filter button
        const resetBtn = document.getElementById('resetDistFilter');
        if (resetBtn) {
            resetBtn.onclick = () => {
                chartState.activeRiskFilter = null;
                const filterEl = document.getElementById('filterRisk');
                if (filterEl) filterEl.value = '';
                currentPage = 1;
                applyFilters();
                updateChartFilterUI();
                renderRiskDistChart(allPredictions);
                renderSpendChurnChart(allPredictions);
                showToast('Cleared risk tier filter.', 'info', 1800);
            };
        }

        // Spend view pills
        document.querySelectorAll('#spendMetricPills .chart-pill').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#spendMetricPills .chart-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                chartState.spendView = btn.getAttribute('data-view');
                renderSpendChurnChart(allPredictions);
            };
        });

        // Trend metric pills
        document.querySelectorAll('#trendMetricPills .chart-pill').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#trendMetricPills .chart-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                chartState.trendMetric = btn.getAttribute('data-metric');
                renderActivityTrendChart(allPredictions);
            };
        });

        // Trend range pills
        document.querySelectorAll('#trendRangePills .chart-pill').forEach(btn => {
            btn.onclick = () => {
                document.querySelectorAll('#trendRangePills .chart-pill').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                chartState.trendRange = btn.getAttribute('data-range');
                renderActivityTrendChart(allPredictions);
            };
        });
    }

    function buildCharts(predictions) {
        setupChartInteractivity();
        renderRiskDistChart(predictions);
        renderSpendChurnChart(predictions);
        renderActivityTrendChart(predictions);
    }

    // ─────────────────────────────────────────────────────────────
    // 1. RISK DISTRIBUTION (DOUGHNUT)
    // ─────────────────────────────────────────────────────────────
    function renderRiskDistChart(predictions) {
        const ctxEl = document.getElementById('riskDistChart');
        if (!ctxEl) return;

        if (chartInstances.dist) {
            chartInstances.dist.destroy();
            chartInstances.dist = null;
        }

        const counts = { Low: 0, Medium: 0, High: 0 };
        const spends = { Low: 0, Medium: 0, High: 0 };

        predictions.forEach(p => {
            if (counts[p.risk_level] !== undefined) {
                counts[p.risk_level]++;
                spends[p.risk_level] += (p.monetary || 0);
            }
        });

        const totalCount = predictions.length;
        const totalRevenue = Object.values(spends).reduce((a, b) => a + b, 0);

        const isRevenue = chartState.distMetric === 'revenue';
        const chartData = isRevenue 
            ? [spends.Low, spends.Medium, spends.High]
            : [counts.Low, counts.Medium, counts.High];

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const cardBg = isLight ? '#ffffff' : '#141b2d';

        const chart = new Chart(ctxEl, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: chartData,
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    hoverBackgroundColor: ['#34d399', '#fbbf24', '#f87171'],
                    borderColor: cardBg,
                    borderWidth: 3,
                    hoverBorderWidth: 4,
                    hoverOffset: 12,
                    spacing: 3
                }]
            },
            options: {
                cutout: '72%',
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: 8 },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: isLight ? '#475569' : '#94a3b8',
                            font: { family: "'Inter', sans-serif", weight: '600', size: 11 },
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: isLight ? 'rgba(255,255,255,0.95)' : 'rgba(11,15,25,0.95)',
                        titleColor: isLight ? '#0f172a' : '#f8fafc',
                        bodyColor: isLight ? '#475569' : '#cbd5e1',
                        borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.15)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        boxPadding: 6,
                        callbacks: {
                            label(context) {
                                const val = context.parsed;
                                const tier = context.label.replace(' Risk', '');
                                const tierCount = counts[tier];
                                const tierSpend = spends[tier];
                                const pct = totalCount > 0 ? Math.round((tierCount / totalCount) * 100) : 0;
                                if (isRevenue) {
                                    return ` Revenue: $${Math.round(val).toLocaleString()} (${pct}%) • ${tierCount} Customers`;
                                }
                                return ` Customers: ${val} (${pct}%) • Revenue: $${Math.round(tierSpend).toLocaleString()}`;
                            },
                            afterLabel() {
                                return '💡 Click slice to filter Priority Queue';
                            }
                        }
                    }
                },
                onClick(event, elements) {
                    if (elements && elements.length > 0) {
                        const idx = elements[0].index;
                        const riskTiers = ['Low', 'Medium', 'High'];
                        filterTableByRisk(riskTiers[idx]);
                    }
                },
                onHover(event, elements) {
                    if (elements && elements.length > 0) {
                        const idx = elements[0].index;
                        const riskTiers = ['Low', 'Medium', 'High'];
                        const colors = ['#10b981', '#f59e0b', '#ef4444'];
                        const tier = riskTiers[idx];
                        const tierCount = counts[tier];
                        const tierSpend = spends[tier];
                        const pct = totalCount > 0 ? Math.round((tierCount / totalCount) * 100) : 0;

                        chart.activeHoverInfo = {
                            title: `${tier} Risk`,
                            color: colors[idx],
                            value: isRevenue ? `$${Math.round(tierSpend).toLocaleString()}` : `${tierCount} (${pct}%)`,
                            sub: isRevenue ? `${tierCount} customers` : `$${Math.round(tierSpend).toLocaleString()} spend`
                        };
                    } else {
                        chart.activeHoverInfo = null;
                    }
                    chart.draw();
                }
            },
            plugins: [donutCenterPlugin]
        });

        // Set default center HUD metrics
        chart.centerMainText = isRevenue 
            ? `$${Math.round(totalRevenue).toLocaleString()}` 
            : String(totalCount);
        chart.centerSubText = isRevenue 
            ? 'TOTAL REVENUE' 
            : 'CUSTOMERS';

        chartInstances.dist = chart;

        // Render interactive footer stats
        const footerEl = document.getElementById('distFooterStats');
        if (footerEl) {
            const tiers = [
                { key: 'Low', color: '#10b981', label: 'Low Risk' },
                { key: 'Medium', color: '#f59e0b', label: 'Medium Risk' },
                { key: 'High', color: '#ef4444', label: 'High Risk' }
            ];
            footerEl.innerHTML = tiers.map(t => {
                const c = counts[t.key];
                const s = spends[t.key];
                const pct = totalCount > 0 ? Math.round((c / totalCount) * 100) : 0;
                const isSelected = chartState.activeRiskFilter === t.key;
                return `
                    <div class="stat-pill ${isSelected ? 'active' : ''}" data-risk="${t.key}" title="Click to filter table by ${t.label}">
                        <span class="stat-dot" style="background:${t.color};"></span>
                        <span><strong>${t.key}:</strong> ${c} (${pct}%) • $${Math.round(s).toLocaleString()}</span>
                    </div>
                `;
            }).join('');

            footerEl.querySelectorAll('.stat-pill').forEach(pill => {
                pill.onclick = () => {
                    const r = pill.getAttribute('data-risk');
                    filterTableByRisk(r);
                };
            });
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 2. SPEND VS CHURN RISK (GRADIENT BARS)
    // ─────────────────────────────────────────────────────────────
    function renderSpendChurnChart(predictions) {
        const ctxEl = document.getElementById('spendChurnChart');
        if (!ctxEl) return;

        if (chartInstances.spend) {
            chartInstances.spend.destroy();
            chartInstances.spend = null;
        }

        const counts = { Low: 0, Medium: 0, High: 0 };
        const spends = { Low: 0, Medium: 0, High: 0 };

        predictions.forEach(p => {
            if (counts[p.risk_level] !== undefined) {
                counts[p.risk_level]++;
                spends[p.risk_level] += (p.monetary || 0);
            }
        });

        const isAvg = chartState.spendView === 'avg';
        const displayData = isAvg
            ? [
                counts.Low > 0 ? Math.round(spends.Low / counts.Low) : 0,
                counts.Medium > 0 ? Math.round(spends.Medium / counts.Medium) : 0,
                counts.High > 0 ? Math.round(spends.High / counts.High) : 0
            ]
            : [spends.Low, spends.Medium, spends.High];

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.07)';
        const textColor = isLight ? '#475569' : '#94a3b8';

        // Create Canvas Linear Gradients for each bar
        const ctx = ctxEl.getContext('2d');
        const gradLow = ctx.createLinearGradient(0, 0, 0, 240);
        gradLow.addColorStop(0, 'rgba(16, 185, 129, 0.95)');
        gradLow.addColorStop(1, 'rgba(16, 185, 129, 0.25)');

        const gradMed = ctx.createLinearGradient(0, 0, 0, 240);
        gradMed.addColorStop(0, 'rgba(245, 158, 11, 0.95)');
        gradMed.addColorStop(1, 'rgba(245, 158, 11, 0.25)');

        const gradHigh = ctx.createLinearGradient(0, 0, 0, 240);
        gradHigh.addColorStop(0, 'rgba(239, 68, 68, 0.95)');
        gradHigh.addColorStop(1, 'rgba(239, 68, 68, 0.25)');

        const chart = new Chart(ctxEl, {
            type: 'bar',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    label: isAvg ? 'Average Spend ($)' : 'Total Monetary ($)',
                    data: displayData,
                    backgroundColor: [gradLow, gradMed, gradHigh],
                    borderColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 2,
                    borderRadius: 10,
                    borderSkipped: false,
                    barPercentage: 0.55
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { top: 24, bottom: 4, left: 8, right: 8 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isLight ? 'rgba(255,255,255,0.95)' : 'rgba(11,15,25,0.95)',
                        titleColor: isLight ? '#0f172a' : '#f8fafc',
                        bodyColor: isLight ? '#475569' : '#cbd5e1',
                        borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.15)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label(context) {
                                const idx = context.dataIndex;
                                const tier = ['Low', 'Medium', 'High'][idx];
                                const c = counts[tier];
                                const s = spends[tier];
                                const avg = c > 0 ? Math.round(s / c) : 0;
                                return [
                                    ` Total Spend: $${Math.round(s).toLocaleString()}`,
                                    ` Customers: ${c}`,
                                    ` Average / Cust: $${avg.toLocaleString()}`
                                ];
                            },
                            afterLabel() {
                                return '💡 Click bar to filter Priority Queue';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: gridColor },
                        ticks: {
                            color: textColor,
                            callback(v) { return '$' + Number(v).toLocaleString(); },
                            font: { size: 10, family: "'Inter', sans-serif" }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: textColor,
                            font: { size: 11, weight: '600', family: "'Inter', sans-serif" }
                        }
                    }
                },
                onClick(event, elements) {
                    if (elements && elements.length > 0) {
                        const idx = elements[0].index;
                        const riskTiers = ['Low', 'Medium', 'High'];
                        filterTableByRisk(riskTiers[idx]);
                    }
                }
            },
            plugins: [barValueLabelsPlugin]
        });

        chartInstances.spend = chart;

        // Render summary footer stats
        const footerEl = document.getElementById('spendFooterStats');
        if (footerEl) {
            const totalRev = Object.values(spends).reduce((a, b) => a + b, 0);
            const highRiskSpend = spends.High;
            const highRiskPct = totalRev > 0 ? Math.round((highRiskSpend / totalRev) * 100) : 0;
            footerEl.innerHTML = `
                <div style="font-size: 0.8rem; color: var(--text-secondary);">
                    Total Revenue: <strong style="color:var(--text-primary);">$${Math.round(totalRev).toLocaleString()}</strong>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">
                    Revenue at Risk: <strong style="color:var(--danger);">$${Math.round(highRiskSpend).toLocaleString()} (${highRiskPct}%)</strong>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary);">
                    Mode: <strong style="color:var(--primary);">${isAvg ? 'Per Customer Average' : 'Cumulative Revenue'}</strong>
                </div>
            `;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 3. PREDICTION ACTIVITY TREND (INTERACTIVE LINE CHART)
    // ─────────────────────────────────────────────────────────────
    function renderActivityTrendChart(predictions) {
        const ctxEl = document.getElementById('activityTrendChart');
        if (!ctxEl) return;

        if (chartInstances.trend) {
            chartInstances.trend.destroy();
            chartInstances.trend = null;
        }

        // Determine range
        let subset = [...predictions];
        if (chartState.trendRange === '10') {
            subset = subset.slice(0, 10);
        } else if (chartState.trendRange === '20') {
            subset = subset.slice(0, 20);
        }
        // Chronological order: oldest to newest
        subset.reverse();
        currentTrendPredictions = subset;

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.07)';
        const textColor = isLight ? '#475569' : '#94a3b8';

        // Metric values
        let values = [];
        let yAxisLabel = 'Churn Probability %';
        let yMax = 100;
        let yMin = 0;

        if (chartState.trendMetric === 'probability') {
            values = subset.map(p => Math.round((p.probability || 0) * 100));
            yAxisLabel = 'Churn Probability (%)';
            yMax = 100;
            yMin = 0;
        } else if (chartState.trendMetric === 'monetary') {
            values = subset.map(p => Math.round(p.monetary || 0));
            yAxisLabel = 'Monetary Spend ($)';
            const maxVal = Math.max(...values, 100);
            yMax = Math.ceil(maxVal * 1.15);
            yMin = 0;
        } else if (chartState.trendMetric === 'recency') {
            values = subset.map(p => Math.round(p.recency || 0));
            yAxisLabel = 'Recency (Days Inactive)';
            const maxVal = Math.max(...values, 30);
            yMax = Math.ceil(maxVal * 1.15);
            yMin = 0;
        }

        // Dynamic point colors based on individual customer risk tier
        const pointColors = subset.map(p => {
            if (p.probability >= 0.7 || p.risk_level === 'High') return '#ef4444';
            if (p.probability >= 0.4 || p.risk_level === 'Medium') return '#f59e0b';
            return '#10b981';
        });

        // Clean sequence labels: #1, #2...
        const labels = subset.map((p, i) => `#${i + 1}`);

        // Area linear gradient
        const ctx = ctxEl.getContext('2d');
        const areaGrad = ctx.createLinearGradient(0, 0, 0, 300);
        areaGrad.addColorStop(0, 'rgba(59, 130, 246, 0.45)');
        areaGrad.addColorStop(0.5, 'rgba(99, 102, 241, 0.15)');
        areaGrad.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        const chart = new Chart(ctxEl, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: yAxisLabel,
                    data: values,
                    borderColor: '#3b82f6',
                    borderWidth: 3,
                    backgroundColor: areaGrad,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointHoverRadius: 10,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: isLight ? '#ffffff' : '#0b0f19',
                    pointBorderWidth: 2,
                    pointHoverBorderWidth: 3,
                    pointHoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: { padding: { top: 20, right: 16, bottom: 4, left: 4 } },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isLight ? 'rgba(255,255,255,0.97)' : 'rgba(11,15,25,0.97)',
                        titleColor: isLight ? '#0f172a' : '#f8fafc',
                        bodyColor: isLight ? '#334155' : '#cbd5e1',
                        borderColor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(59,130,246,0.3)',
                        borderWidth: 1.5,
                        padding: 12,
                        cornerRadius: 10,
                        displayColors: false,
                        callbacks: {
                            title(items) {
                                const idx = items[0].dataIndex;
                                const p = currentTrendPredictions[idx];
                                return `Customer: ${p.customer_id} (Seq #${idx + 1})`;
                            },
                            label(context) {
                                const idx = context.dataIndex;
                                const p = currentTrendPredictions[idx];
                                const prob = Math.round((p.probability || 0) * 100);
                                const riskEmoji = p.risk_level === 'High' ? '🔴' : p.risk_level === 'Medium' ? '🟡' : '🟢';
                                return [
                                    ` Risk Tier: ${p.risk_level} ${riskEmoji}`,
                                    ` Churn Probability: ${prob}%`,
                                    ` Lifetime Spend: $${Number(p.monetary).toLocaleString()}`,
                                    ` Recency: ${p.recency} days`
                                ];
                            },
                            afterLabel() {
                                return '👉 Click node to view customer deep-dive';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        min: yMin,
                        max: yMax,
                        grid: { color: gridColor },
                        ticks: {
                            color: textColor,
                            font: { size: 10, family: "'Inter', sans-serif" },
                            callback(v) {
                                if (chartState.trendMetric === 'probability') return v + '%';
                                if (chartState.trendMetric === 'monetary') return '$' + Number(v).toLocaleString();
                                return v;
                            }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: textColor,
                            font: { size: 11, family: "'Inter', sans-serif" }
                        }
                    }
                },
                onClick(event, elements) {
                    if (elements && elements.length > 0) {
                        const idx = elements[0].index;
                        const pred = currentTrendPredictions[idx];
                        if (pred) {
                            openSidePanel(pred);
                            showToast(`Opened inspection dossier for ${pred.customer_id}`, 'info', 2000);
                        }
                    }
                }
            },
            plugins: [trendThresholdPlugin]
        });

        chartInstances.trend = chart;

        // Update summary chip
        const chip = document.getElementById('trendSummaryChip');
        if (chip && values.length > 0) {
            const latest = values[values.length - 1];
            const peak = Math.max(...values);
            const unit = chartState.trendMetric === 'probability' ? '%' : chartState.trendMetric === 'monetary' ? '$' : 'd';
            chip.innerHTML = chartState.trendMetric === 'monetary'
                ? `Latest: <strong>$${latest.toLocaleString()}</strong> | Peak: <strong>$${peak.toLocaleString()}</strong>`
                : `Latest: <strong>${latest}${unit}</strong> | Peak: <strong>${peak}${unit}</strong>`;
        }
    }

    // Re-render charts when light/dark theme changes
    window.addEventListener('themechange', () => {
        if (allPredictions && allPredictions.length > 0) {
            buildCharts(allPredictions);
        }
    });

    // ---- Deletion & Reset Handlers ----
    window.deletePrediction = async function(id) {
        if (!confirm("Are you sure you want to delete this prediction?")) return;
        
        try {
            const res = await fetch(`/api/prediction/${id}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('Prediction deleted successfully.', 'success');
                allPredictions = allPredictions.filter(p => String(p.id) !== String(id));
                applyFilters();
            } else {
                throw new Error(data.message);
            }
        } catch (err) {
            showToast('Error deleting: ' + err.message, 'error');
        }
    };

    window.resetCompanyData = async function() {
        if (!confirm("⚠️ WARNING: This will permanently delete ALL prediction and batch data for your company. This action cannot be undone. Are you sure?")) return;
        
        try {
            const res = await fetch('/api/reset', { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('All data reset successfully.', 'success');
                setTimeout(() => window.location.reload(), 1500);
            } else {
                throw new Error(data.message);
            }
        } catch (err) {
            showToast('Error resetting data: ' + err.message, 'error');
        }
    };

});
