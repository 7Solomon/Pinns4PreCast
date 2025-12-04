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
                    const clone = template.content.cloneNode(true);

                    // 1. Set ID
                    const card = clone.querySelector('.run-card');
                    card.setAttribute('data-id', run.id);

                    // 2. Status & Title
                    const dot = clone.querySelector('.status-dot');
                    const badge = clone.querySelector('.run-status');

                    clone.querySelector('.run-title').textContent = `Run #${run.id}`;
                    badge.textContent = run.status.toUpperCase();

                    if (run.status === 'running') {
                        dot.classList.add('bg-running');
                        badge.classList.add('bg-success'); // Bootstrap green badge
                    } else {
                        dot.classList.add('bg-finished');
                        badge.classList.add('bg-secondary'); // Bootstrap grey badge
                    }

                    // 3. Event Listeners

                    // Toggle Accordion
                    const header = clone.querySelector('.run-header');
                    header.addEventListener('click', () => this.toggleRun(run.id));

                    // Refresh Button
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

                    const tempId = `temp-${run.id}`;
                    const alphaId = `alpha-${run.id}`;
                    const groupName = `visType-${run.id}`;

                    radioTemp.id = tempId; radioTemp.name = groupName; labelTemp.setAttribute('for', tempId);
                    radioAlpha.id = alphaId; radioAlpha.name = groupName; labelAlpha.setAttribute('for', alphaId);

                    // Chart Triggers
                    radioTemp.addEventListener('change', () => RunCharts.renderSensorChart(run.id));
                    radioAlpha.addEventListener('change', () => RunCharts.renderSensorChart(run.id));
                    clone.querySelector('.epoch-selector').addEventListener('change', () => RunCharts.renderSensorChart(run.id));

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
    }
};

RunManager.init();
