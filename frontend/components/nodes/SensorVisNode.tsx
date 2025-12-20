import React, { memo, useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { clsx } from 'clsx';
import {
    Activity, RefreshCcw, Settings, Database, Box, Play, Layers,
    Cpu, Wind, Flame, Hexagon, Anchor, FileJson,
    Minimize2, Maximize2, Thermometer, Target, Radar
} from 'lucide-react';

// --- CONFIGURATION ---

const IconMap: Record<string, any> = {
    "sensor": Target, "thermometer": Thermometer, "radar": Radar,
    "database": Database, "function": Activity, "box": Box, "play": Play,
    "settings": Settings, "layers": Layers, "network-wired": Cpu,
    "wind": Wind, "fire": Flame, "hexagon": Hexagon, "anchor": Anchor,
    "default": FileJson
};

const PortColors: Record<string, string> = {
    "field": "bg-indigo-500 border-indigo-500",
    "spatial": "bg-teal-500 border-teal-500",
    "time": "bg-amber-500 border-amber-500",
    "model": "bg-purple-500 border-purple-500",
    "dataset": "bg-emerald-500 border-emerald-500",
    "default": "bg-slate-400 border-slate-400"
};

const SENSOR_STYLES: Record<string, { color: string; width: number; dash?: string; name: string }> = {
    'temperature': { color: '#ef4444', width: 2, name: 'Temp (°C)' },
    'T': { color: '#ef4444', width: 2, name: 'Temp' },
    'alpha': { color: '#3b82f6', width: 2, name: 'Hydration (α)' },
    'u': { color: '#10b981', width: 2, name: 'Disp X' },
    'v': { color: '#84cc16', width: 2, name: 'Disp Y' },
    'value': { color: '#e2e8f0', width: 2, name: 'Value' },
};

const formatDecimal = (value: any) => {
    if (value === null || value === undefined || value === "") return "";
    const num = Number(value);
    if (num === 0) return "0";
    if (Math.abs(num) < 0.001 || Math.abs(num) > 10000) return num.toExponential(2);
    return num.toFixed(3);
};

// --- MAIN COMPONENT ---

const SensorVisNode = ({ id, data, selected }: NodeProps) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [plotData, setPlotData] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [useLogScale, setUseLogScale] = useState(false);
    const [xDomain, setXDomain] = useState<[number, number] | null>(null);

    const chartRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const lastMouseX = useRef<number>(0);

    const runId = data.config?.run_id;
    const hasActiveRun = runId && runId !== "unknown";

    // --- DATA FETCHING ---
    useEffect(() => {
        if (!hasActiveRun) return;
        let isMounted = true;

        const fetchData = async (isPolling = false) => {
            try {
                if (!isPolling) setLoading(true);
                const response = await fetch(`http://localhost:8000/monitor/sensor/${runId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const result = await response.json();

                if (isMounted && result.data && Array.isArray(result.data)) {
                    const cleanData = result.data.map((row: any) => {
                        const newRow: any = { ...row };
                        newRow.step = Number(newRow.step);
                        Object.keys(newRow).forEach(k => {
                            if (['step', 't', 'time', 'timestamp'].includes(k)) return;
                            const val = Number(newRow[k]);
                            if (isNaN(val) || (useLogScale && val <= 0)) newRow[k] = null;
                            else newRow[k] = val;
                        });
                        return newRow;
                    });
                    setPlotData(cleanData);
                    setError(null);
                }
            } catch (e: any) {
                if (isMounted && plotData.length === 0) setError("Waiting for stream...");
            } finally {
                if (isMounted && !isPolling) setLoading(false);
            }
        };

        fetchData(false);
        const interval = setInterval(() => fetchData(true), 2000);
        return () => { isMounted = false; clearInterval(interval); };
    }, [runId, id, hasActiveRun, useLogScale]);

    // --- CHART INTERACTIONS ---
    const visibleKeys = useMemo(() => {
        if (plotData.length === 0) return [];
        return Object.keys(plotData[0]).filter(k =>
            !['step', 'epoch', 'time', 't', 'timestamp'].includes(k) && plotData.some(row => row[k] !== null)
        );
    }, [plotData]);

    const handleWheel = useCallback((e: WheelEvent) => {
        e.preventDefault(); e.stopPropagation();
        if (plotData.length === 0) return;
        const allSteps = plotData.map(d => d.step);
        const min = Math.min(...allSteps);
        const max = Math.max(...allSteps);
        const [currMin, currMax] = xDomain || [min, max];
        const delta = (currMax - currMin) * 0.1;
        const newMin = e.deltaY < 0 ? currMin + delta : currMin - delta;
        const newMax = e.deltaY < 0 ? currMax - delta : currMax + delta;
        if (newMax > newMin) setXDomain([newMin, newMax]);
    }, [xDomain, plotData]);

    useEffect(() => {
        const node = chartRef.current;
        if (node) {
            node.addEventListener('wheel', handleWheel, { passive: false });
            return () => node.removeEventListener('wheel', handleWheel);
        }
    }, [handleWheel, isExpanded]);

    const handleMouseDown = (e: React.MouseEvent) => {
        e.stopPropagation(); e.preventDefault();
        setIsDragging(true);
        lastMouseX.current = e.clientX;
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging || !chartRef.current || plotData.length === 0) return;
        e.stopPropagation(); e.preventDefault();
        const currentX = e.clientX;
        const { width } = chartRef.current.getBoundingClientRect();
        const allSteps = plotData.map(d => d.step);
        const min = Math.min(...allSteps);
        const max = Math.max(...allSteps);
        const [currMin, currMax] = xDomain || [min, max];
        const shift = ((lastMouseX.current - currentX) / width) * (currMax - currMin);
        setXDomain([currMin + shift, currMax + shift]);
        lastMouseX.current = currentX;
    };

    // --- RENDERING ---
    const IconComponent = data.icon && IconMap[data.icon] ? IconMap[data.icon] : Target;
    const inputKeys = Object.keys(data.inputs || {});
    const outputKeys = Object.keys(data.outputs || {});

    return (
        <div className={clsx(
            "rounded-xl border bg-slate-900 shadow-2xl transition-all duration-300 flex flex-col relative",
            selected ? "border-cyan-400 ring-2 ring-cyan-500/20" : "border-slate-800",
            isExpanded ? "w-[500px]" : "min-w-[200px]"
        )}>

            {/* HEADER */}
            <div className="bg-slate-950/80 rounded-t-xl pb-2">
                <div className="h-1 w-full bg-cyan-500 opacity-70 rounded-t-xl mb-3" />

                <div className="px-3">
                    <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-slate-800 text-cyan-400 border border-slate-700/50">
                                <IconComponent size={18} />
                            </div>
                            <div>
                                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-0.5">Sensor</span>
                                <h3 className="text-sm font-bold text-slate-200 leading-tight">{data.label || "Sensor"}</h3>
                            </div>
                        </div>
                        <button onClick={() => setIsExpanded(!isExpanded)} className="text-slate-500 hover:text-white p-1 hover:bg-slate-800 rounded bg-slate-900 border border-slate-800">
                            {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                        </button>
                    </div>

                    <div className="flex flex-row gap-8">
                        {/* INPUTS (LEFT) */}
                        <div className="flex flex-col gap-2 flex-1">
                            {inputKeys.map((key) => {
                                const portData = data.inputs[key];
                                const type = (typeof portData === 'object') ? portData.type : portData;
                                const colorClass = PortColors[type] || PortColors.default;

                                return (
                                    <div key={key} className="relative flex items-center h-5 w-full">
                                        {/* STRICTLY LEFT POSITIONED HANDLE */}
                                        <Handle
                                            type="target"
                                            position={Position.Left}
                                            id={key}
                                            className={clsx(
                                                "absolute !top-1/2 !-translate-y-1/2 !-left-[16px] z-50",
                                                "!w-2.5 !h-2.5 !border-2 transition-all hover:scale-125",
                                                colorClass
                                            )}
                                        />
                                        <span className="text-[10px] font-mono text-slate-400 ml-1">{key}</span>
                                    </div>
                                );
                            })}
                            {inputKeys.length === 0 && <span className="text-[9px] text-slate-700 italic">No Inputs</span>}
                        </div>

                        {/* OUTPUTS (RIGHT) */}
                        <div className="flex flex-col gap-2 items-end flex-1">
                            {outputKeys.map((key) => {
                                const portData = data.outputs[key];
                                const type = (typeof portData === 'object') ? portData.type : portData;
                                const colorClass = PortColors[type] || PortColors.default;

                                return (
                                    <div key={key} className="relative flex items-center h-5 w-full justify-end">
                                        <span className="text-[10px] font-mono text-slate-400 mr-1">{key}</span>
                                        {/* STRICTLY RIGHT POSITIONED HANDLE */}
                                        <Handle
                                            type="source"
                                            position={Position.Right}
                                            id={key}
                                            className={clsx(
                                                "absolute !top-1/2 !-translate-y-1/2 !-right-[16px] z-50",
                                                "!w-2.5 !h-2.5 !border-2 transition-all hover:scale-125",
                                                colorClass
                                            )}
                                        />
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>

            {/* CHART */}
            {isExpanded && (
                <div className="border-t border-slate-800 h-[300px] bg-slate-900/50 rounded-b-xl flex flex-col">
                    {!hasActiveRun ? (
                        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-2">
                            <Thermometer size={24} className="opacity-20" />
                            <p className="text-xs">Waiting for run...</p>
                        </div>
                    ) : (
                        <>
                            <div className="flex items-center justify-between px-3 py-1.5 bg-slate-900/80 border-b border-slate-800">
                                <div className="text-[10px] text-slate-500 font-mono flex items-center gap-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span> LIVE
                                </div>
                                <div className="flex gap-2">
                                    {xDomain && (
                                        <button onClick={(e) => { e.stopPropagation(); setXDomain(null); }} className="flex items-center gap-1 px-1.5 py-0.5 text-[9px] font-bold rounded border bg-yellow-500/10 text-yellow-500 border-yellow-500/30 hover:bg-yellow-500/20">
                                            <RefreshCcw size={8} /> RESET
                                        </button>
                                    )}
                                    <button onClick={() => setUseLogScale(!useLogScale)} className="px-1.5 py-0.5 text-[9px] font-bold rounded border bg-slate-800 text-slate-500 border-slate-700">
                                        {useLogScale ? "LOG" : "LIN"}
                                    </button>
                                </div>
                            </div>
                            <div ref={chartRef} className="flex-1 relative nodrag nowheel bg-slate-900/30 rounded-b-xl overflow-hidden"
                                onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={() => setIsDragging(false)} onMouseLeave={() => setIsDragging(false)}>
                                {!loading && !error && plotData.length > 0 && (
                                    <div className="w-full h-full p-2 select-none" onMouseDown={e => e.stopPropagation()}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={plotData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} vertical={false} />
                                                <XAxis dataKey="step" type="number" stroke="#475569" fontSize={9} domain={xDomain || ['dataMin', 'dataMax']} tickCount={5} />
                                                <YAxis stroke="#475569" fontSize={9} scale={useLogScale ? 'log' : 'linear'} domain={['auto', 'auto']} tickFormatter={formatDecimal} width={35} />
                                                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '11px', padding: '8px' }} formatter={(val: any) => [formatDecimal(val), '']} labelStyle={{ color: '#64748b' }} />
                                                <Legend wrapperStyle={{ fontSize: '10px' }} iconSize={8} formatter={(val) => SENSOR_STYLES[val]?.name || val} />
                                                {visibleKeys.map((key) => {
                                                    const style = SENSOR_STYLES[key] || { color: '#cbd5e1', width: 1, name: 'Val' };
                                                    return <Line key={key} type="monotone" dataKey={key} stroke={style.color} strokeWidth={style.width} dot={false} isAnimationActive={false} connectNulls />;
                                                })}
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

export default memo(SensorVisNode);
