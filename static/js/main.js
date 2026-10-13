document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // GLOBAL UI UTILITIES
    // ============================================================

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
        "inactivity_days": 20
    };


    // ============================================================
    // PREDICT PAGE
    // ============================================================
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        setupPredictPage();
    }

    function setupPredictPage() {
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

    function showResult(result) {
        const section = document.getElementById('result-section');
        section.style.display = 'block';

        document.getElementById('res-cust-id').innerText = result.customer_id;

        const badge = document.getElementById('res-badge');
        badge.className = `badge ${result.risk_level.toLowerCase()}`;
        badge.innerText = `${result.risk_level} Risk`;

        const probPercent = Math.round(result.probability * 100);
        const probAnim = document.getElementById('res-prob-anim');
        const circProg = document.getElementById('circular-progress');

        let color = '#10b981';
        if (result.risk_level === 'Medium') color = '#f59e0b';
        if (result.risk_level === 'High') color = '#ef4444';

        let cur = 0;
        const interval = setInterval(() => {
            cur++;
            probAnim.innerText = `${cur}%`;
            circProg.style.background = `conic-gradient(${color} ${cur * 3.6}deg, rgba(255,255,255,0.07) 0deg)`;
            if (cur >= probPercent) clearInterval(interval);
        }, 15);

        const healthScore = 100 - probPercent;
        setTimeout(() => {
            document.getElementById('res-health-fill').style.width = `${healthScore}%`;
        }, 300);

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

    // ---- Build Charts ----
    function buildCharts(predictions) {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";

        // Risk Distribution Doughnut
        const riskCounts = { Low: 0, Medium: 0, High: 0 };
        predictions.forEach(p => { if (riskCounts[p.risk_level] !== undefined) riskCounts[p.risk_level]++; });

        const ctxDist = document.getElementById('riskDistChart');
        if (ctxDist) {
            new Chart(ctxDist, {
                type: 'doughnut',
                data: {
                    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                    datasets: [{
                        data: [riskCounts.Low, riskCounts.Medium, riskCounts.High],
                        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                        borderWidth: 0,
                        hoverOffset: 6
                    }]
                },
                options: {
                    cutout: '72%',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        }

        // Spend vs Churn Bar
        const spendSum = { Low: 0, Medium: 0, High: 0 };
        predictions.forEach(p => { if (spendSum[p.risk_level] !== undefined) spendSum[p.risk_level] += (p.monetary || 0); });

        const ctxSpend = document.getElementById('spendChurnChart');
        if (ctxSpend) {
            new Chart(ctxSpend, {
                type: 'bar',
                data: {
                    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                    datasets: [{
                        label: 'Total Monetary ($)',
                        data: [spendSum.Low, spendSum.Medium, spendSum.High],
                        backgroundColor: ['rgba(16,185,129,0.6)', 'rgba(245,158,11,0.6)', 'rgba(239,68,68,0.6)'],
                        borderRadius: 8
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }

        // Activity Trend Line
        const ctxTrend = document.getElementById('activityTrendChart');
        if (ctxTrend) {
            const trendData = predictions.slice(0, 20).reverse().map(p => Math.round(p.probability * 100));
            const labels = trendData.map((_, i) => `#${i + 1}`);
            new Chart(ctxTrend, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Churn Probability %',
                        data: trendData,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59,130,246,0.08)',
                        tension: 0.45,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        pointBackgroundColor: '#3b82f6'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
    }

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
