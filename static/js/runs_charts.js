const RunCharts = {
    // Cache data here to avoid re-fetching when switching tabs/epochs
    cache: {},

    /**
     * Fetches log data and plots the Loss History
     */
    fetchAndPlotLoss: function (runId) {
        const card = document.querySelector(`.run-card[data-id="${runId}"]`);
        if (!card) return;

        fetch(`/info/run/${runId}/log`)
            .then(r => r.json())
            .then(data => {
                const getTraceData = (key) => {
                    return {
                        x: data.history.map(h => h[key] !== null ? h.step : null),
                        y: data.history.map(h => h[key])
                    };
                };

                const traces = [];

                // --- GROUP 1: MAIN LOSSES ---
                const tLoss = getTraceData('loss');
                traces.push({
                    x: tLoss.x, y: tLoss.y,
                    name: 'Total Loss', mode: 'lines',
                    line: { color: 'black', width: 3 },
                    legendgroup: 'total'
                });

                // --- GROUP 2: PHYSICS ---
                const pLoss = getTraceData('loss_physics');
                traces.push({
                    x: pLoss.x, y: pLoss.y,
                    name: 'Phys Total', mode: 'lines',
                    line: { color: '#1f77b4', width: 2 },
                    legendgroup: 'physics'
                });

                const pLossTemp = getTraceData('loss_phys_temperature');
                traces.push({
                    x: pLossTemp.x, y: pLossTemp.y,
                    name: 'Phys (Temp)', mode: 'lines',
                    line: { color: '#aec7e8', dash: 'dot' },
                    legendgroup: 'physics'
                });

                const pLossAlpha = getTraceData('loss_phys_alpha');
                traces.push({
                    x: pLossAlpha.x, y: pLossAlpha.y,
                    name: 'Phys (Alpha)', mode: 'lines',
                    line: { color: '#1f77b4', dash: 'dot' },
                    legendgroup: 'physics'
                });

                // --- GROUP 3: BC ---
                const bLoss = getTraceData('loss_bc');
                traces.push({
                    x: bLoss.x, y: bLoss.y,
                    name: 'BC Total', mode: 'lines',
                    line: { color: '#ff7f0e', width: 2 },
                    legendgroup: 'bc'
                });

                // --- GROUP 4: IC ---
                const iLoss = getTraceData('loss_ic');
                traces.push({
                    x: iLoss.x, y: iLoss.y,
                    name: 'IC Total', mode: 'lines',
                    line: { color: '#2ca02c', width: 2 },
                    legendgroup: 'ic'
                });

                // --- VALIDATION ---
                const valTotal = getTraceData('val_loss');
                if (valTotal.y.some(y => y !== null)) {
                    traces.push({
                        x: valTotal.x.filter(x => x !== null),
                        y: valTotal.y.filter(y => y !== null),
                        name: 'Val Total', mode: 'markers',
                        marker: { symbol: 'diamond', color: 'red', size: 8 }
                    });
                }

                const layout = {
                    margin: { t: 30, l: 60, r: 20, b: 40 },
                    showlegend: true,
                    legend: { orientation: 'h', y: 1.1, tracegroupgap: 5 },
                    yaxis: {
                        type: 'log',
                        title: 'Loss (Log Scale)',
                        autorange: true
                    },
                    xaxis: { title: 'Steps' },
                    dragmode: 'pan'
                };

                const config = {
                    responsive: true,
                    scrollZoom: true,
                    displayModeBar: true,
                };

                // Find the chart container inside the specific card
                const chartDiv = card.querySelector('.chart-loss');
                Plotly.react(chartDiv, traces, layout, config);
            })
            .catch(err => console.error("Loss plot error:", err));
    },

    /**
     * Fetches sensor CSV data and populates the dropdown
     */
    loadSensorVis: function (runId) {
        const card = document.querySelector(`.run-card[data-id="${runId}"]`);
        const container = card.querySelector('.vis-container');

        container.innerHTML = '<div class="d-flex justify-content-center align-items-center h-100"><div class="spinner-border text-primary"></div></div>';

        fetch(`/info/run/${runId}/vis/sensor`)
            .then(r => r.json())
            .then(data => {
                // Cache data
                this.cache[runId] = data;

                const tempKeys = Object.keys(data.temperature);

                // Handle No Data Case
                if (tempKeys.length === 0) {
                    container.innerHTML = `
                        <div class="d-flex flex-column justify-content-center align-items-center h-100 text-muted">
                            <p>No visualization data found.</p>
                        </div>`;
                    return;
                }

                // Update Dropdown
                const sel = card.querySelector('.epoch-selector');
                sel.innerHTML = '';
                tempKeys.forEach(fname => {
                    const opt = document.createElement('option');
                    opt.value = fname;
                    opt.text = fname.replace('.csv', '').replace('_', ' ');
                    sel.appendChild(opt);
                });
                // Select last epoch by default
                sel.value = tempKeys[tempKeys.length - 1];

                // Render
                this.renderSensorChart(runId);
            })
            .catch(err => {
                container.innerHTML = `<div class="alert alert-danger m-3">Error: ${err}</div>`;
            });
    },

    /**
     * Draws the Sensor Chart based on current Radio/Dropdown selection
     */
    renderSensorChart: function (runId) {
        const card = document.querySelector(`.run-card[data-id="${runId}"]`);
        if (!card) return;

        // Get UI states relative to this card
        const isTemp = card.querySelector('.vis-type-temp').checked;
        const selectedEpoch = card.querySelector('.epoch-selector').value;
        const container = card.querySelector('.vis-container');

        if (!this.cache[runId]) return;

        const dataset = isTemp ? this.cache[runId].temperature : this.cache[runId].alpha;

        if (!selectedEpoch || !dataset[selectedEpoch]) return;

        // Parse CSV Data
        const parsed = this.parseCSV(dataset[selectedEpoch]);

        // Build Traces
        // Format: Time_s, Time_h, T1, T2 ...
        const traces = [];
        // Loop through columns starting at index 2 (T1)
        for (let i = 2; i < parsed.headers.length; i++) {
            traces.push({
                x: parsed.columns[1], // Index 1 is Time_h
                y: parsed.columns[i], // Sensor Value
                mode: 'lines',
                name: parsed.headers[i]
            });
        }

        const layout = {
            margin: { t: 30, l: 50, r: 20, b: 40 },
            xaxis: { title: 'Time (hours)' },
            yaxis: { title: isTemp ? 'Temperature (°C)' : 'Degree of Hydration' },
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 },
            dragmode: 'pan'
        };

        const config = {
            responsive: true,
            scrollZoom: true,
            displayModeBar: true,
        };

        Plotly.newPlot(container, traces, layout, config);
    },

    parseCSV: function (csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        const columns = headers.map(() => []);

        for (let i = 1; i < lines.length; i++) {
            const row = lines[i].split(',');
            if (row.length === headers.length) {
                row.forEach((val, colIndex) => {
                    columns[colIndex].push(parseFloat(val));
                });
            }
        }
        return { headers, columns };
    }
};
