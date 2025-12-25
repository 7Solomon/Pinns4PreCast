import React, { useState, useMemo, useRef } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { RefreshCcw, Maximize2, Activity } from 'lucide-react';


// Color palette for different loss components (PINN specific)
const LOSS_COLORS: Record<string, string> = {
    loss_step: '#f472b6',           // Pink (Total/Step Loss)
    loss_epoch: '#f472b6',          // Pink (Epoch Loss)  
    loss_phys_temperature: '#60a5fa', // Blue (Physics)
    loss_phys_alpha: '#fbbf24',     // Amber
    loss_bc_temperature: '#34d399', // Emerald (Boundary)
    loss_ic_temperature: '#a78bfa', // Purple (Initial)
    loss_ic_alpha: '#f87171',       // Red
    // Fallback for any other loss_ keys
};


interface LossChartProps {
    data: any[];            // The full history array from parent
    runId: string | null;
    isConnected: boolean;
}


export default function LossChart({ data, runId, isConnected }: LossChartProps) {
    const [useLogScale, setUseLogScale] = useState(true);


    const chartData = useMemo(() => {
        if (!data || data.length === 0) return [];


        return data.map((d, index) => {
            const base = {
                step: Number(d.step || index),
                epoch: Number(d.epoch || 0),
            };


            const metrics: Record<string, any> = {};


            // Case 1: Nested metrics (old format)
            if (d.metrics && typeof d.metrics === 'object') {
                Object.assign(metrics, d.metrics);
            }


            // Case 2: FLAT loss_* keys directly on root (your NEW format)
            Object.keys(d).forEach((key) => {
                if (key.startsWith('loss_')) {
                    metrics[key] = d[key];
                }
            });


            // Clean ALL loss_ metrics
            Object.keys(metrics).forEach((key) => {
                if (!key.startsWith('loss_')) return;


                let value = metrics[key];
                const numValue = typeof value === 'number' ? value : Number(value);


                if (useLogScale && (!Number.isFinite(numValue) || numValue <= 0)) {
                    metrics[key] = null;
                } else {
                    metrics[key] = Number.isFinite(numValue) ? numValue : null;
                }
            });


            return { ...base, ...metrics };
        });
    }, [data, useLogScale]);



    const lossKeys = useMemo(() => {
        if (chartData.length === 0) return [];


        const keySet = new Set<string>();
        chartData.forEach(row => {
            Object.keys(row)
                .filter(key =>
                    key.startsWith('loss_') &&
                    key !== 'loss_step' &&      // Exclude total step loss if you want
                    key !== 'loss_epoch'    // Exclude total epoch loss
                )
                .forEach(key => keySet.add(key));
        });


        return Array.from(keySet).sort();
    }, [chartData]);



    console.log('🔍 LossChart Debug:', {
        dataLength: chartData.length,
        rawFirstPoint: data[0],           // ← SHOWS WHAT WE ACTUALLY GET
        rawFirstKeys: Object.keys(data[0] || {}),
        processedFirst: chartData[0],     // ← AFTER PROCESSING
        foundLossKeys: lossKeys,
        sampleRaw: data.slice(0, 2)
    });



    // Formatter for scientific notation in tooltips/axes
    const formatScientific = (value: number | null) => {
        if (value === null || value === undefined || !Number.isFinite(value)) return "N/A";
        if (value === 0) return "0";
        if (Math.abs(value) < 0.001 || Math.abs(value) > 1000) {
            return value.toExponential(2);
        }
        return value.toFixed(4);
    };


    return (
        <div className="flex flex-col h-[400px] bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            {/* HEADER & CONTROLS */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/50">
                <div className="flex items-center gap-3">
                    {/* Status Indicator */}
                    <div className={`flex items-center gap-2 px-2 py-1 rounded-full border ${isConnected
                        ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                        : 'bg-slate-800 border-slate-700 text-slate-400'
                        }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
                            }`} />
                        <span className="text-[10px] font-bold uppercase tracking-wider">
                            {isConnected ? 'Live Training' : 'Offline'}
                        </span>
                    </div>


                    <span className="text-xs text-slate-500 font-mono">
                        Run: {runId?.split('_').pop() || 'N/A'} | {chartData.length} points
                    </span>
                </div>


                {/* Chart Controls */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => setUseLogScale(!useLogScale)}
                        className={`px-3 py-1 text-[10px] font-bold rounded border transition-colors ${useLogScale
                            ? 'bg-blue-500/20 text-blue-400 border-blue-500/30'
                            : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700'
                            }`}
                    >
                        {useLogScale ? 'LOG SCALE' : 'LINEAR'}
                    </button>
                </div>
            </div>


            {/* CHART AREA */}
            <div className="flex-1 w-full bg-slate-950/30 relative">
                {chartData.length === 0 ? (
                    // Empty State
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                        <Activity className="w-8 h-8 mb-2 opacity-20" />
                        <p className="text-xs">Waiting for metrics...</p>
                    </div>
                ) : lossKeys.length === 0 ? (
                    // No loss keys found
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-amber-400">
                        <Activity className="w-8 h-8 mb-2 opacity-50" />
                        <p className="text-xs">No loss metrics found in data</p>
                        <p className="text-[10px] mt-1">Check console for debug info</p>
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />


                            <XAxis
                                dataKey="step"
                                stroke="#475569"
                                fontSize={10}
                                tickFormatter={(val) => `${Math.round(Number(val))}`}
                                label={{
                                    value: 'Step',
                                    position: 'insideBottomRight',
                                    offset: -5,
                                    fill: '#475569',
                                    fontSize: 10
                                }}
                            />


                            <YAxis
                                stroke="#475569"
                                fontSize={10}
                                scale={useLogScale ? 'log' : 'linear'}
                                domain={['auto', 'auto']}
                                tickFormatter={formatScientific}
                                width={50}
                            />


                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#0f172a',
                                    borderColor: '#1e293b',
                                    fontSize: '12px'
                                }}
                                formatter={(value: any, name: any) => {
                                    const formattedVal = formatScientific(Number(value));
                                    const formattedName = typeof name === 'string'
                                        ? name.replace('loss_', '').replace(/_/g, ' ').toUpperCase()
                                        : name;
                                    return [formattedVal, formattedName];
                                }}
                                labelFormatter={(label) => `Step: ${Math.round(Number(label))}`}
                            />


                            <Legend
                                verticalAlign="top"
                                height={36}
                                iconType="circle"
                                wrapperStyle={{ fontSize: '10px', opacity: 0.8 }}
                                formatter={(value) =>
                                    value.replace('loss_', '').replace(/_/g, ' ').toUpperCase()
                                }
                            />


                            {/* Dynamic Lines based on data keys */}
                            {lossKeys.map((key) => (
                                <Line
                                    key={key}
                                    type="monotone"
                                    dataKey={key}
                                    stroke={LOSS_COLORS[key] || '#94a3b8'}
                                    strokeWidth={key.includes('step') || key.includes('epoch') ? 2.5 : 1.5}
                                    dot={false}
                                    connectNulls={true}
                                    activeDot={{ r: 4, strokeWidth: 2 }}
                                    isAnimationActive={false}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
}