import React, { memo, useEffect, useState, useMemo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { clsx } from 'clsx';
import { Activity, Loader2, Maximize2, Minimize2, AlertCircle } from 'lucide-react';

// --- CONFIGURATION ---
const METRIC_STYLES: Record<string, { color: string; width: number; dash?: string; name: string }> = {
  'loss': { color: '#f97316', width: 2.5, name: 'Total Loss' }, // Orange
  'loss_physics': { color: '#3b82f6', width: 2, name: 'Phys Total' }, // Blue
  'loss_phys_temperature': { color: '#60a5fa', width: 1.5, dash: '5 5', name: 'Phys (Temp)' },
  'loss_phys_alpha': { color: '#3b82f6', width: 1.5, dash: '3 3', name: 'Phys (Alpha)' },
  'loss_bc': { color: '#10b981', width: 2, name: 'BC Total' },   // Green
  'loss_ic': { color: '#8b5cf6', width: 2, name: 'IC Total' },   // Purple
  'val_loss': { color: '#ef4444', width: 2, name: 'Val Total' },  // Red
};

const DEFAULT_STYLE = { color: '#94a3b8', width: 1, dash: undefined, name: 'Metric' };

const formatScientific = (value: any) => {
  if (value === null || value === undefined || value === "") return "";
  const num = Number(value);
  if (num === 0) return "0";
  if (Math.abs(num) < 0.01 || Math.abs(num) > 1000) return num.toExponential(1);
  return num.toFixed(2);
};

const LossCurveNode = ({ data, selected }: NodeProps) => {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // States
  const [useLogScale, setUseLogScale] = useState(true);
  const [showAllMetrics, setShowAllMetrics] = useState(false);

  const runId = data.config?.run_id || "unknown";
  const refreshRate = data.config?.refresh_rate || 2000;

  useEffect(() => {
    if (!runId || runId === "unknown") return;

    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/monitor/metrics/${runId}`);
        if (!response.ok) {
          if (response.status === 404) return;
          throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.metrics && Array.isArray(result.metrics)) {
          // --- CRITICAL FIX START ---
          const cleanData = result.metrics.map((row: any) => {
            const newRow: any = { ...row };

            // 1. Ensure 'step' is a number
            newRow.step = Number(newRow.step);

            Object.keys(newRow).forEach(k => {
              if (k === 'step' || k === 'epoch' || k === 'timestamp') return;

              // 2. Handle Empty Strings from Backend
              if (newRow[k] === "" || newRow[k] === null || newRow[k] === undefined) {
                newRow[k] = null;
              } else {
                // 3. Force Convert to Number
                const val = Number(newRow[k]);

                // 4. Handle Zero for Log Scale (Log(0) = Crash)
                // If value is 0 or NaN, treat as null for chart
                if (isNaN(val) || (useLogScale && val <= 0)) {
                  newRow[k] = null;
                } else {
                  newRow[k] = val;
                }
              }
            });
            return newRow;
          });
          // --- CRITICAL FIX END ---

          if (cleanData.length > 0) {
            setPlotData(cleanData);
            setError(null);
          }
        }
      } catch (e: any) {
        console.error("Poll failed:", e);
        setError("Connection lost");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, refreshRate);
    return () => clearInterval(interval);
  }, [runId, refreshRate, useLogScale]); // Added useLogScale dependency to re-clean 0s

  // Determine lines to draw
  const availableKeys = useMemo(() => {
    if (plotData.length === 0) return [];
    return Object.keys(plotData[0]).filter(k =>
      k !== 'step' && k !== 'epoch' && k !== 'timestamp'
    );
  }, [plotData]);

  const visibleKeys = showAllMetrics
    ? availableKeys
    : availableKeys.filter(k =>
      k === 'loss' || k === 'val_loss' || k.startsWith('loss_physics')
    );

  return (
    <div className={clsx(
      "min-w-[600px] h-[450px] rounded-xl border bg-slate-900/95 backdrop-blur-md shadow-2xl flex flex-col transition-all duration-200",
      selected ? "border-blue-400 ring-2 ring-blue-500/20" : "border-slate-800"
    )}>

      <Handle type="target" position={Position.Left} id="run_id"
        className="!w-3 !h-3 !border-2 !bg-slate-900 border-blue-500" />

      {/* HEADER: Draggable area */}
      <div className="h-1 w-full bg-blue-500 rounded-t-xl" />
      <div className="p-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 rounded-t-xl cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-blue-500" />
          <div className="flex flex-col">
            <span className="font-bold text-slate-200 text-sm">Loss Metrics</span>
            <span className="text-[10px] text-slate-500 font-mono">
              {runId !== "unknown" ? runId.slice(0, 15) : "Waiting..."}
            </span>
          </div>
        </div>

        {/* CONTROLS: Non-Draggable area */}
        <div className="flex items-center gap-2 nodrag cursor-auto">
          <button
            onClick={() => setUseLogScale(!useLogScale)}
            className={clsx(
              "px-2 py-1 text-[10px] font-bold rounded border transition-colors",
              useLogScale
                ? "bg-blue-500/20 text-blue-400 border-blue-500/50"
                : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700"
            )}
          >
            {useLogScale ? "LOG" : "LIN"}
          </button>

          <button
            onClick={() => setShowAllMetrics(!showAllMetrics)}
            className="p-1 hover:bg-slate-800 rounded text-slate-400 transition-colors"
          >
            {showAllMetrics ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* CHART: Non-Draggable area */}
      <div className="flex-1 p-2 min-h-0 relative nodrag cursor-auto bg-slate-950/30">

        {error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 text-sm gap-2">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        ) : plotData.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm animate-pulse">
            Waiting for training to start...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={plotData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />

              <XAxis
                dataKey="step"
                type="number"
                stroke="#64748b"
                fontSize={10}
                domain={['dataMin', 'dataMax']} // Tight fit to data
                tickCount={8}
              />

              <YAxis
                stroke="#64748b"
                fontSize={10}
                scale={useLogScale ? 'log' : 'linear'}
                domain={['auto', 'auto']}
                allowDataOverflow={true}
                tickFormatter={formatScientific}
                width={45}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  borderColor: '#334155',
                  borderRadius: '6px',
                  fontSize: '11px'
                }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number | string | undefined) => [formatScientific(Number(value ?? 0)), '']}
              />

              <Legend
                verticalAlign="top"
                height={30}
                iconSize={10}
                wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }}
                formatter={(value) => METRIC_STYLES[value]?.name || value}
              />

              {visibleKeys.map((key) => {
                const style = METRIC_STYLES[key] || DEFAULT_STYLE;

                return (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={style.color}
                    strokeWidth={style.width}
                    strokeDasharray={style.dash}
                    dot={false}
                    connectNulls={true}
                    isAnimationActive={false}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

export default memo(LossCurveNode);
