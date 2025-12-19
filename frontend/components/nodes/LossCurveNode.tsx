import React, { memo, useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { clsx } from 'clsx';
import {
  Activity, RefreshCcw, Settings, Database, Box, Play, Layers,
  Cpu, Wind, Flame, Hexagon, Anchor, FileJson, HelpCircle, Minimize2, Maximize2,
  LineChart as LineChartIcon
} from 'lucide-react';

// --- CONFIGURATION ---

const IconMap: Record<string, any> = {
  "database": Database, "function": Activity, "box": Box, "play": Play,
  "settings": Settings, "layers": Layers, "network-wired": Cpu,
  "wind": Wind, "fire": Flame, "hexagon": Hexagon, "anchor": Anchor,
  "default": FileJson
};

const PortColors: Record<string, string> = {
  "model": "bg-purple-500 border-purple-500",
  "dataset": "bg-emerald-500 border-emerald-500",
  "solver": "bg-orange-500 border-orange-500",
  "problem": "bg-blue-500 border-blue-500",
  "logger": "bg-pink-500 border-pink-500",
  "default": "bg-slate-400 border-slate-400"
};

const METRIC_STYLES: Record<string, { color: string; width: number; dash?: string; name: string }> = {
  'loss_step': { color: '#f97316', width: 3, name: 'Total Loss' },
  'loss_phys_temperature': { color: '#3b82f6', width: 2, name: 'Phys (Temp)' },
  'loss_phys_alpha': { color: '#22d3ee', width: 2, dash: '5 5', name: 'Phys (Alpha)' },
  'loss_bc_temperature': { color: '#10b981', width: 2, name: 'BC (Temp)' },
  'loss_ic_temperature': { color: '#a855f7', width: 2, name: 'IC (Temp)' },
  'loss_ic_alpha': { color: '#ff00ff', width: 2, dash: '3 3', name: 'IC (Alpha)' },
};
const DEFAULT_STYLE = { color: '#94a3b8', width: 1, dash: undefined, name: 'Metric' };

const formatScientific = (value: any) => {
  if (value === null || value === undefined || value === "") return "";
  const num = Number(value);
  if (num === 0) return "0";
  if (Math.abs(num) < 0.01 || Math.abs(num) > 1000) return num.toExponential(1);
  return num.toFixed(2);
};

// --- MAIN COMPONENT ---

const LossCurveNode = ({ data, selected }: NodeProps) => {
  // --- VISUAL STATE ---
  const [isExpanded, setIsExpanded] = useState(false);

  // --- DATA STATE ---
  const [plotData, setPlotData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useLogScale, setUseLogScale] = useState(true);
  const [xDomain, setXDomain] = useState<[number, number] | null>(null);

  // --- INTERACTION ---
  const chartRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const lastMouseX = useRef<number>(0);

  const runId = data.config?.run_id;
  const hasActiveRun = runId && runId !== "unknown";

  // --- FETCHING LOGIC ---
  useEffect(() => {
    if (!hasActiveRun) return;

    let isMounted = true; // Safety flag

    const fetchData = async (isPolling = false) => {
      try {
        // Only show loading spinner on the very first fetch, not polling
        if (!isPolling) setLoading(true);

        const response = await fetch(`http://localhost:8000/monitor/metrics/${runId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const result = await response.json();

        if (isMounted && result.metrics && Array.isArray(result.metrics)) {
          const cleanData = result.metrics.map((row: any) => {
            const newRow: any = { ...row };
            newRow.step = Number(newRow.step);
            Object.keys(newRow).forEach(k => {
              if (['step', 'epoch', 'timestamp'].includes(k)) return;
              const val = Number(newRow[k]);
              // Handle Log Scale Zeros/Negatives
              if (isNaN(val) || (useLogScale && val <= 0)) newRow[k] = null;
              else newRow[k] = val;
            });
            return newRow;
          });

          setPlotData(cleanData);
          setError(null);
        }
      } catch (e: any) {
        if (isMounted && plotData.length === 0) setError("Connection lost");
      } finally {
        if (isMounted && !isPolling) setLoading(false);
      }
    };

    // Initial Fetch
    fetchData(false);

    // Polling Interval
    const interval = setInterval(() => fetchData(true), 2000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [runId, hasActiveRun, useLogScale]); // Removed plotData.length to prevent loops


  // --- CHART LOGIC ---
  const visibleKeys = useMemo(() => {
    if (plotData.length === 0) return [];
    return Object.keys(plotData[0]).filter(k =>
      !['step', 'epoch', 'timestamp'].includes(k) && plotData.some(row => row[k] !== null)
    ).filter(k => METRIC_STYLES[k]);
  }, [plotData]);

  const getDataBounds = () => {
    if (plotData.length === 0) return { min: 0, max: 0 };
    const allSteps = plotData.map(d => d.step);
    return { min: Math.min(...allSteps), max: Math.max(...allSteps) };
  };

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (plotData.length === 0) return;
    const { min, max } = getDataBounds();
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
  }, [handleWheel, isExpanded, hasActiveRun]);

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
    const { min, max } = getDataBounds();
    const [currMin, currMax] = xDomain || [min, max];
    const shift = ((lastMouseX.current - currentX) / width) * (currMax - currMin);
    setXDomain([currMin + shift, currMax + shift]);
    lastMouseX.current = currentX;
  };

  // --- RENDER PREP ---
  const IconComponent = data.icon && IconMap[data.icon] ? IconMap[data.icon] : Activity;
  const inputKeys = Object.keys(data.inputs || {});
  const outputKeys = Object.keys(data.outputs || {});

  return (
    <div className={clsx(
      "rounded-xl border bg-slate-900 shadow-2xl transition-all duration-300 flex flex-col isolate",
      // ^^^ REMOVED 'overflow-hidden' from here so handles can stick out
      selected ? "border-pink-400 ring-2 ring-pink-500/20 shadow-pink-900/20" : "border-slate-800",
      isExpanded ? "w-[600px]" : "min-w-[240px]"
    )}>

      {/* --- STANDARD NODE HEADER & PORTS --- */}
      <div className="bg-slate-950/80 rounded-t-xl"> {/* Added rounded-t-xl here */}
        <div className="h-1 w-full bg-pink-500 opacity-70 rounded-t-xl" />

        <div className="p-4">
          <div className="flex justify-between items-start mb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-slate-800 text-pink-400 border border-slate-700/50">
                <IconComponent size={18} />
              </div>
              <div>
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-0.5">
                  Visualization
                </span>
                <h3 className="text-sm font-bold text-slate-200 leading-tight">
                  {data.label}
                </h3>
              </div>
            </div>

            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-slate-500 hover:text-white transition-colors p-1 hover:bg-slate-800 rounded bg-slate-900 border border-slate-800"
              title={isExpanded ? "Collapse View" : "Expand View"}
            >
              {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
          </div>

          <div className="flex flex-row gap-8">
            {/* INPUTS */}
            <div className="flex flex-col gap-3 flex-1">
              {inputKeys.map((key) => {
                const portData = data.inputs[key];
                const type = (typeof portData === 'object') ? portData.type : portData;
                const isRequired = (typeof portData === 'object') ? (portData.required !== false) : true;
                const colorClass = PortColors[type] || PortColors.default;

                return (
                  <div key={key} className="relative flex items-center h-5 group">
                    <Handle
                      type="target"
                      position={Position.Left}
                      id={key}
                      className={clsx(
                        "!w-3 !h-3 !border-2 !-left-[18px] transition-all hover:scale-125",
                        isRequired ? colorClass : `${colorClass.replace('bg-', 'text-')} !bg-slate-900`
                      )}
                    />
                    <span className={clsx(
                      "text-xs font-medium capitalize transition-colors",
                      isRequired ? "text-slate-300" : "text-slate-500 italic"
                    )}>
                      {key}
                    </span>
                  </div>
                );
              })}
              {inputKeys.length === 0 && <div className="text-[10px] text-slate-700 italic py-1">No Inputs</div>}
            </div>

            {/* OUTPUTS */}
            <div className="flex flex-col gap-3 items-end flex-1">
              {outputKeys.map((key) => {
                const portData = data.outputs[key];
                const type = (typeof portData === 'object') ? portData.type : portData;
                const colorClass = PortColors[type] || PortColors.default;

                return (
                  <div key={key} className="relative flex items-center h-5 justify-end group">
                    <span className="text-xs font-medium text-slate-300 capitalize mr-2">{key}</span>
                    <Handle
                      type="source"
                      position={Position.Right}
                      id={key}
                      className={clsx("!w-3 !h-3 !border-2 !-right-[18px] transition-all hover:scale-125", colorClass)}
                    />
                  </div>
                );
              })}
              {outputKeys.length === 0 && <div className="text-[10px] text-slate-700 italic py-1">No Outputs</div>}
            </div>
          </div>
        </div>
      </div>

      {/* --- CHART / PLACEHOLDER SECTION --- */}
      {isExpanded && (
        <div className="border-t border-slate-800 flex flex-col h-[400px] bg-slate-900/50 animate-in slide-in-from-top-2 duration-200 rounded-b-xl">

          {/* 1. STATE: NOT RUNNING */}
          {!hasActiveRun && (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
              <div className="p-4 bg-slate-900 rounded-full mb-3 border border-slate-800">
                <LineChartIcon size={32} className="opacity-50" />
              </div>
              <p className="text-sm font-medium">Not currently running</p>
              <p className="text-xs text-slate-600 mt-1">Start training to see metrics</p>
            </div>
          )}

          {/* 2. STATE: RUNNING (Show Chart) */}
          {hasActiveRun && (
            <>
              <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800">
                <div className="text-xs text-slate-500 font-mono">Run: {runId.split('_').pop()}</div>
                <div className="flex gap-2">
                  {xDomain && (
                    <button onClick={(e) => { e.stopPropagation(); setXDomain(null); }} className="flex items-center gap-1 px-2 py-1 text-[10px] font-bold rounded border bg-yellow-500/10 text-yellow-500 border-yellow-500/30">
                      <RefreshCcw size={10} /> RESET
                    </button>
                  )}
                  <button onClick={() => setUseLogScale(!useLogScale)} className="px-2 py-1 text-[10px] font-bold rounded border bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700">
                    {useLogScale ? "LOG" : "LIN"}
                  </button>
                </div>
              </div>

              <div
                ref={chartRef}
                className={clsx(
                  "flex-1 relative nodrag nowheel bg-slate-900/30 rounded-b-xl", // Added rounded-b-xl
                  isDragging ? "cursor-grabbing" : "cursor-default"
                )}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={() => setIsDragging(false)}
                onMouseLeave={() => setIsDragging(false)}
              >
                {loading && plotData.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs animate-pulse">Loading Metrics...</div>
                )}

                {error && (
                  <div className="absolute inset-0 flex items-center justify-center text-red-400 text-xs">{error}</div>
                )}

                {!loading && !error && plotData.length > 0 && (
                  <div className="w-full h-full p-2 select-none" onMouseDown={e => e.stopPropagation()}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={plotData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                        <XAxis dataKey="step" type="number" stroke="#64748b" fontSize={10} domain={xDomain || ['dataMin', 'dataMax']} allowDataOverflow tickCount={6} />
                        <YAxis stroke="#64748b" fontSize={10} scale={useLogScale ? 'log' : 'linear'} domain={['auto', 'auto']} allowDataOverflow tickFormatter={formatScientific} width={45} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }} formatter={(val: any) => [formatScientific(val), '']} labelStyle={{ color: '#94a3b8' }} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} formatter={(val) => METRIC_STYLES[val]?.name || val} />
                        {visibleKeys.map((key) => {
                          const style = METRIC_STYLES[key] || DEFAULT_STYLE;
                          return <Line key={key} type="monotone" dataKey={key} stroke={style.color} strokeWidth={style.width} strokeDasharray={style.dash} dot={false} isAnimationActive={false} connectNulls />;
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

export default memo(LossCurveNode);
