import React, { useState, useEffect } from 'react';
import { Handle, Position } from 'reactflow';
import { Activity, TrendingUp, Clock, Zap, AlertCircle } from 'lucide-react';

interface MonitorNodeData {
    label: string;
    category: string;
    run_id?: string;
    run_path?: string;
    update_interval?: number;
}

interface TrainingStatus {
    id: string;
    status: string;
    epoch: number;
    loss: number | null;
    last_update: number;
}

interface MetricData {
    epoch: number;
    loss: number;
    loss_physics?: number;
    loss_bc?: number;
    loss_ic?: number;
    [key: string]: number | undefined;
}

export default function MonitorNode({ data }: { data: MonitorNodeData }) {
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [metrics, setMetrics] = useState<MetricData[]>([]);
    const [isLive, setIsLive] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        // Extract run_id from data (could be passed via config or connection)
        const runId = data.run_id;

        if (!runId) {
            setError("No run_id connected");
            return;
        }

        setIsLive(true);
        setError(null);

        const fetchStatus = async () => {
            try {
                const res = await fetch(`http://localhost:8000/monitor/status/${runId}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const statusData = await res.json();
                setStatus(statusData);

                // Stop polling if training finished
                if (statusData.status === 'completed' || statusData.status === 'failed') {
                    setIsLive(false);
                }
            } catch (err) {
                console.error('Failed to fetch status:', err);
                setError('Connection lost');
            }
        };

        const fetchMetrics = async () => {
            try {
                const res = await fetch(`http://localhost:8000/monitor/metrics/${runId}?limit=50`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const data = await res.json();
                setMetrics(data.metrics || []);
            } catch (err) {
                console.error('Failed to fetch metrics:', err);
            }
        };

        // Initial fetch
        fetchStatus();
        fetchMetrics();

        // Set up polling intervals
        const statusInterval = setInterval(fetchStatus, (data.update_interval || 1) * 1000);
        const metricsInterval = setInterval(fetchMetrics, 5000);

        return () => {
            clearInterval(statusInterval);
            clearInterval(metricsInterval);
        };
    }, [data.run_id, data.update_interval]);

    const getStatusColor = () => {
        if (!status) return 'bg-slate-700';
        switch (status.status) {
            case 'running': return 'bg-emerald-600';
            case 'completed': return 'bg-blue-600';
            case 'failed': return 'bg-red-600';
            default: return 'bg-yellow-600';
        }
    };

    const latestMetric = metrics[metrics.length - 1];

    return (
        <div className="min-w-[320px] max-w-[380px] rounded-xl border-2 border-slate-700 bg-slate-900 shadow-2xl transition-all hover:shadow-emerald-500/10">
            {/* Header */}
            <div className={`${getStatusColor()} px-4 py-3 rounded-t-xl flex justify-between items-center`}>
                <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-white" />
                    <span className="text-sm font-bold text-white">{data.label}</span>
                </div>
                {isLive && !error && (
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                        <span className="text-xs text-white/80 font-semibold">LIVE</span>
                    </div>
                )}
                {error && (
                    <div className="flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4 text-white/70" />
                        <span className="text-xs text-white/70">{error}</span>
                    </div>
                )}
            </div>

            {/* Body */}
            <div className="p-4 space-y-3">
                {status ? (
                    <>
                        {/* Primary Metrics */}
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-gradient-to-br from-slate-800/80 to-slate-800/40 p-3 rounded-lg border border-slate-700/50">
                                <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1.5">
                                    <Clock size={12} />
                                    Epoch
                                </div>
                                <div className="text-xl font-bold text-white">
                                    {status.epoch || 0}
                                </div>
                            </div>

                            <div className="bg-gradient-to-br from-emerald-900/30 to-slate-800/40 p-3 rounded-lg border border-emerald-700/30">
                                <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1.5">
                                    <TrendingUp size={12} />
                                    Total Loss
                                </div>
                                <div className="text-xl font-bold text-emerald-400">
                                    {status.loss ? status.loss.toFixed(4) : 'N/A'}
                                </div>
                            </div>
                        </div>

                        {/* Detailed Metrics Breakdown */}
                        {latestMetric && (
                            <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700/30 space-y-2">
                                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">
                                    Loss Components
                                </div>
                                {[
                                    { key: 'loss_physics', label: 'Physics', color: 'text-blue-400' },
                                    { key: 'loss_bc', label: 'Boundary', color: 'text-purple-400' },
                                    { key: 'loss_ic', label: 'Initial', color: 'text-cyan-400' }
                                ].map(({ key, label, color }) => (
                                    latestMetric[key] !== undefined && (
                                        <div key={key} className="flex justify-between items-center text-xs">
                                            <span className="text-slate-400">
                                                {label}:
                                            </span>
                                            <span className={`font-mono font-semibold ${color}`}>
                                                {latestMetric[key].toFixed(6)}
                                            </span>
                                        </div>
                                    )
                                ))}
                            </div>
                        )}

                        {/* Mini Loss Chart */}
                        {metrics.length > 1 && (
                            <div className="bg-slate-800/30 p-3 rounded-lg border border-slate-700/30">
                                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">
                                    Loss History
                                </div>
                                <svg className="w-full h-20" viewBox="0 0 200 60" preserveAspectRatio="none">
                                    {/* Grid lines */}
                                    <line x1="0" y1="30" x2="200" y2="30" stroke="#334155" strokeWidth="1" strokeDasharray="2,2" />

                                    {/* Loss curve */}
                                    <polyline
                                        fill="none"
                                        stroke="url(#lossGradient)"
                                        strokeWidth="2.5"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        points={metrics
                                            .slice(-30)
                                            .map((m, i, arr) => {
                                                const x = (i / Math.max(arr.length - 1, 1)) * 200;
                                                const allLosses = arr.map(a => a.loss || 0).filter(l => l > 0);
                                                const max = Math.max(...allLosses, 1);
                                                const min = Math.min(...allLosses, 0);
                                                const range = max - min || 1;
                                                const y = 55 - ((m.loss - min) / range) * 50;
                                                return `${x},${Math.max(5, Math.min(55, y))}`;
                                            })
                                            .join(' ')}
                                    />

                                    {/* Gradient definition */}
                                    <defs>
                                        <linearGradient id="lossGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stopColor="#3b82f6" />
                                            <stop offset="100%" stopColor="#10b981" />
                                        </linearGradient>
                                    </defs>
                                </svg>
                                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                                    <span>-{metrics.slice(-30).length} epochs</span>
                                    <span>now</span>
                                </div>
                            </div>
                        )}

                        {/* Status Footer */}
                        <div className="text-xs text-slate-500 text-center pt-2 border-t border-slate-800">
                            {status.status === 'running' && '⚡ Training in progress...'}
                            {status.status === 'completed' && '✓ Training completed successfully'}
                            {status.status === 'failed' && '✗ Training failed - check logs'}
                            {status.status === 'initializing' && '🔄 Initializing training...'}
                        </div>
                    </>
                ) : (
                    <div className="text-center py-8 text-slate-500">
                        <Zap size={40} className="mx-auto mb-3 opacity-20" />
                        <div className="text-sm font-medium">Waiting for training...</div>
                        <div className="text-xs mt-1 opacity-70">Connect a logger node to monitor</div>
                    </div>
                )}
            </div>

            {/* Input Handle */}
            <Handle
                type="target"
                position={Position.Left}
                id="run_path"
                className="!w-3 !h-3 !bg-emerald-400 !border-2 !border-slate-900"
                style={{ left: -6 }}
            />
        </div>
    );
}