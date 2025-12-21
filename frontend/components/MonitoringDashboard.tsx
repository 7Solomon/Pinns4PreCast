import React, { useState, useMemo } from 'react';
import { Activity, Database, TrendingUp, Clock, Zap, Eye } from 'lucide-react';


// Mock hook - replace with actual implementation
const useMonitoringWebSocket = ({ runId, onMetricsUpdate }: any) => {
    const [isConnected, setIsConnected] = React.useState(false);
    return { isConnected };
};

interface RunInfo {
    run_id: string;
    status: string;
    start_time?: string;
    epoch?: number;
    loss?: number;
}

export default function MonitoringDashboard() {
    const [selectedRun, setSelectedRun] = useState<string | null>(null);
    const [runs, setRuns] = useState<RunInfo[]>([]);
    const [metricsData, setMetricsData] = useState<any[]>([]);

    // WebSocket connection for real-time updates
    const { isConnected } = useMonitoringWebSocket({
        runId: selectedRun,
        onMetricsUpdate: (metrics: any) => {
            setMetricsData(prev => [...prev, metrics]);
        }
    });

    // Fetch available runs on mount
    React.useEffect(() => {
        fetch('http://localhost:8000/monitor/runs')
            .then(res => res.json())
            .then(data => setRuns(data.runs || []));
    }, []);

    const activeRuns = useMemo(() =>
        runs.filter(r => r.status === 'running'),
        [runs]
    );

    const completedRuns = useMemo(() =>
        runs.filter(r => r.status === 'completed'),
        [runs]
    );

    return (
        <div className="w-full h-screen bg-slate-950 flex overflow-hidden">

            {/* LEFT SIDEBAR: Run List */}
            <div className="w-80 border-r border-slate-800 bg-slate-900 flex flex-col">
                <div className="p-4 border-b border-slate-800">
                    <div className="flex items-center gap-3 mb-2">
                        <Activity className="text-emerald-400" size={24} />
                        <h2 className="text-xl font-bold text-white">Monitoring</h2>
                    </div>
                    <p className="text-xs text-slate-500">View any training run</p>
                </div>

                {/* Active Runs */}
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {activeRuns.length > 0 && (
                        <div className="p-3">
                            <h3 className="text-xs font-bold text-emerald-400 uppercase mb-2 flex items-center gap-2">
                                <Zap size={12} />
                                Active ({activeRuns.length})
                            </h3>
                            {activeRuns.map(run => (
                                <RunCard
                                    key={run.run_id}
                                    run={run}
                                    isSelected={selectedRun === run.run_id}
                                    onClick={() => setSelectedRun(run.run_id)}
                                />
                            ))}
                        </div>
                    )}

                    {/* Completed Runs */}
                    {completedRuns.length > 0 && (
                        <div className="p-3">
                            <h3 className="text-xs font-bold text-slate-500 uppercase mb-2 flex items-center gap-2">
                                <Database size={12} />
                                History ({completedRuns.length})
                            </h3>
                            {completedRuns.map(run => (
                                <RunCard
                                    key={run.run_id}
                                    run={run}
                                    isSelected={selectedRun === run.run_id}
                                    onClick={() => setSelectedRun(run.run_id)}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* MAIN CONTENT: Visualizations */}
            <div className="flex-1 flex flex-col">
                {selectedRun ? (
                    <>
                        {/* Header */}
                        <div className="p-4 border-b border-slate-800 bg-slate-900">
                            <div className="flex items-center justify-between">
                                <div>
                                    <h3 className="text-lg font-bold text-white">{selectedRun}</h3>
                                    <div className="flex items-center gap-4 mt-1">
                                        <span className={`text-xs px-2 py-1 rounded ${isConnected ? 'bg-green-500/20 text-green-400' : 'bg-slate-800 text-slate-500'
                                            }`}>
                                            {isConnected ? '● Live' : '○ Historical'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Visualization Grid */}
                        <div className="flex-1 p-4 overflow-y-auto grid grid-cols-1 lg:grid-cols-2 gap-4">

                            {/* Loss Curve */}
                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                                <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                                    <TrendingUp size={16} className="text-blue-400" />
                                    Training Loss
                                </h4>
                                <div className="h-64 flex items-center justify-center text-slate-600">
                                    {metricsData.length > 0 ? (
                                        <div>Chart with {metricsData.length} points</div>
                                    ) : (
                                        <div>No metrics data yet</div>
                                    )}
                                </div>
                            </div>

                            {/* Sensor Data */}
                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                                <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                                    <Eye size={16} className="text-cyan-400" />
                                    Sensor Data
                                </h4>
                                <div className="h-64 flex items-center justify-center text-slate-600">
                                    Sensor visualization
                                </div>
                            </div>

                            {/* Metrics Table */}
                            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 lg:col-span-2">
                                <h4 className="text-sm font-bold text-white mb-3">Metrics History</h4>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs">
                                        <thead>
                                            <tr className="text-slate-500 border-b border-slate-800">
                                                <th className="text-left p-2">Epoch</th>
                                                <th className="text-left p-2">Loss</th>
                                                <th className="text-left p-2">Physics</th>
                                                <th className="text-left p-2">BC</th>
                                                <th className="text-left p-2">IC</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {metricsData.slice(-10).map((m, i) => (
                                                <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                                                    <td className="p-2 text-slate-300">{m.epoch}</td>
                                                    <td className="p-2 text-slate-300">{m.loss?.toFixed(4)}</td>
                                                    <td className="p-2 text-slate-400">{m.loss_physics?.toFixed(4)}</td>
                                                    <td className="p-2 text-slate-400">{m.loss_bc?.toFixed(4)}</td>
                                                    <td className="p-2 text-slate-400">{m.loss_ic?.toFixed(4)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="flex-1 flex items-center justify-center text-slate-600">
                        <div className="text-center">
                            <Database size={48} className="mx-auto mb-3 opacity-20" />
                            <p>Select a run to view monitoring data</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function RunCard({ run, isSelected, onClick }: any) {
    return (
        <button
            onClick={onClick}
            className={`w-full text-left p-3 rounded-lg mb-2 transition-all ${isSelected
                    ? 'bg-blue-500/20 border border-blue-500/50'
                    : 'bg-slate-800 border border-slate-700 hover:border-slate-600'
                }`}
        >
            <div className="flex items-start justify-between mb-1">
                <span className="text-xs font-mono text-white truncate">
                    {run.run_id.split('_').pop()}
                </span>
                <span className={`text-[10px] px-2 py-0.5 rounded ${run.status === 'running'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-slate-700 text-slate-400'
                    }`}>
                    {run.status}
                </span>
            </div>

            <div className="flex items-center gap-3 text-[10px] text-slate-500">
                {run.epoch !== undefined && (
                    <span className="flex items-center gap-1">
                        <Clock size={10} />
                        Epoch {run.epoch}
                    </span>
                )}
                {run.loss !== undefined && (
                    <span>Loss: {run.loss.toFixed(4)}</span>
                )}
            </div>
        </button>
    );
}