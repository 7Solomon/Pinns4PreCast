const RunManager = {
    activeRunId: null,

    init: function () {
        document.addEventListener('DOMContentLoaded', () => this.loadRunsList());
    },

    loadRunsList: function () {
        fetch('/info/runs/data')
            .then(r => r.json())
            .then(runs => {
                const container = document.getElementById('runs-container');
                const template = document.getElementById('run-card-template');

                container.innerHTML = ''; // Clear loading spinner

                if (runs.length === 0) {
                    container.innerHTML = '<div class="alert alert-info text-center">No training runs found. Start one!</div>';
                    return;
                }

                runs.forEach(run => {
                    console.log("Run Data:", run)
                    const clone = template.content.cloneNode(true);

                    // 1. Set ID
                    const card = clone.querySelector('.run-card');
                    card.setAttribute('data-id', run.id);

                    const titleEl = clone.querySelector('.run-title');
                    const dateEl = clone.querySelector('.run-date');
                    const badge = clone.querySelector('.run-status');
                    const dot = clone.querySelector('.status-dot');

                    if (run.start_time && run.start_time !== "Unknown") {
                        const dateObj = new Date(run.start_time);

                        // Main Title: "Fri, Dec 12"
                        titleEl.textContent = dateObj.toLocaleDateString(undefined, {
                            weekday: 'short', month: 'short', day: 'numeric'
                        });

                        // Subtitle: "14:30 · Ep 5 · L 0.0452"
                        let infoText = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

                        if (run.epoch !== undefined && run.epoch !== null) {
                            infoText += ` · Ep ${run.epoch}`;
                        }

                        if (run.loss !== undefined && run.loss !== null) {
                            const lossVal = typeof run.loss === 'number' ? run.loss.toFixed(4) : run.loss;
                            infoText += ` · L ${lossVal}`;
                        }

                        if (dateEl) dateEl.textContent = infoText;

                    } else {
                        // Fallback for Legacy Runs
                        titleEl.textContent = `Run #${run.id}`;
                        if (dateEl) dateEl.textContent = "Legacy Run";
                    }

                    // 3. Set Status Colors
                    // Reset classes first (safest approach)
                    badge.className = 'badge rounded-pill run-status';
                    dot.className = 'status-dot';

                    const status = run.status.toLowerCase();
                    badge.textContent = status.toUpperCase();

                    switch (status) {
                        case 'running':
                            badge.classList.add('bg-primary'); // Blue
                            dot.classList.add('bg-running');   // Pulsing Blue
                            break;

                        case 'finished':
                        case 'success':
                            badge.classList.add('bg-success'); // Green
                            dot.classList.add('bg-finished');
                            break;

                        case 'aborted':
                        case 'killed':
                        case 'stopped':
                            badge.classList.add('bg-danger'); // Red
                            dot.classList.add('bg-aborted');
                            break;

                        case 'failed':
                        case 'error':
                            badge.classList.add('bg-warning', 'text-dark'); // Orange
                            dot.classList.add('bg-error');
                            break;

                        default:
                            badge.classList.add('bg-secondary'); // Grey
                            dot.classList.add('bg-secondary');
                    }

                    // 4. Event Listeners
                    const header = clone.querySelector('.run-header');
                    header.addEventListener('click', () => this.toggleRun(run.id));

                    const refreshBtn = clone.querySelector('.btn-refresh');
                    refreshBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.refreshRun(run.id, refreshBtn);
                    });

                    // Unique IDs for Radio Buttons
                    const radioTemp = clone.querySelector('.vis-type-temp');
                    const radioAlpha = clone.querySelector('.vis-type-alpha');
                    const labelTemp = clone.querySelector('.vis-label-temp');
                    const labelAlpha = clone.querySelector('.vis-label-alpha');
                    const epochSel = clone.querySelector('.epoch-selector');

                    const tempId = `temp-${run.id}`;
                    const alphaId = `alpha-${run.id}`;
                    const groupName = `visType-${run.id}`;

                    radioTemp.id = tempId; radioTemp.name = groupName; labelTemp.setAttribute('for', tempId);
                    radioAlpha.id = alphaId; radioAlpha.name = groupName; labelAlpha.setAttribute('for', alphaId);

                    // Chart Triggers
                    radioTemp.addEventListener('change', () => RunCharts.renderSensorChart(run.id));
                    radioAlpha.addEventListener('change', () => RunCharts.renderSensorChart(run.id));
                    epochSel.addEventListener('change', () => RunCharts.renderSensorChart(run.id));

                    container.appendChild(clone);
                });
            })
            .catch(err => {
                document.getElementById('runs-container').innerHTML = `<div class="alert alert-danger">Error loading runs: ${err}</div>`;
            });
    },

    toggleRun: function (runId) {
        const card = document.querySelector(`.run-card[data-id="${runId}"]`);
        const body = card.querySelector('.run-body');
        const isOpening = body.style.display === 'none';

        // Close all others
        document.querySelectorAll('.run-body').forEach(el => el.style.display = 'none');

        if (isOpening) {
            body.style.display = 'block';
            this.activeRunId = runId;
            RunCharts.fetchAndPlotLoss(runId);
            RunCharts.loadSensorVis(runId);
        }
    },

    refreshRun: function (runId, btn) {
        const originalHTML = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Loading...';

        Promise.all([
            RunCharts.fetchAndPlotLoss(runId),
            RunCharts.loadSensorVis(runId)
        ]).finally(() => {
            btn.disabled = false;
            btn.innerHTML = originalHTML;
        });
    },

    startNewRun: function () {
        const btn = document.querySelector('.btn-primary');
        if (!btn) return; // Safety check

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Initializing...';

        fetch('/api/define_model', { method: 'POST' })
            .then(() => fetch('/api/define_training_pipeline', { method: 'POST' }))
            .then(() => fetch('/api/train', { method: 'POST' }))
            .then(() => {
                setTimeout(() => location.reload(), 1000);
            })
            .catch(err => {
                alert("Error starting run: " + err);
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-fill"></i> Start New Run';
            });
    },

    stopRun: function () {
        if (!confirm("Are you sure you want to stop the current training?")) return;

        fetch('/api/stop_training', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                alert(data.message);
                setTimeout(() => this.loadRunsList(), 1000);
            })
            .catch(err => alert("Error: " + err));
    },

    killRun: function () {
        if (!confirm("Are you sure you want to Kill the current training?")) return;
        fetch('/api/kill_training', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                alert(data.message);
                setTimeout(() => this.loadRunsList(), 1000);
            })
            .catch(err => alert("Error: " + err));
    }
};

RunManager.init();
