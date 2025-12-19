import React, { memo, useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { clsx } from 'clsx';
import { Activity, RefreshCcw } from 'lucide-react';

// --- COLOR CONFIGURATION ---
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

const LossCurveNode = ({ data, selected }: NodeProps) => {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // States
  const [useLogScale, setUseLogScale] = useState(true);
  const [showAllMetrics, setShowAllMetrics] = useState(false);

  // --- ZOOM & PAN STATE ---
  const [xDomain, setXDomain] = useState<[number, number] | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);

  // DRAG STATE
  const [isDragging, setIsDragging] = useState(false);
  const lastMouseX = useRef<number>(0);

  const runId = data.config?.run_id || "unknown";

  useEffect(() => {
    if (!runId || runId === "unknown") return;

    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/monitor/metrics/${runId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const result = await response.json();
        if (result.metrics && Array.isArray(result.metrics)) {
          const cleanData = result.metrics.map((row: any) => {
            const newRow: any = { ...row };
            newRow.step = Number(newRow.step);
            Object.keys(newRow).forEach(k => {
              if (['step', 'epoch', 'timestamp'].includes(k)) return;
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
        if (plotData.length === 0) setError("Connection lost");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [runId, useLogScale]);


  const visibleKeys = useMemo(() => {
    if (plotData.length === 0) return [];
    const allKeys = Object.keys(plotData[0]).filter(k =>
      !['step', 'epoch', 'timestamp'].includes(k) &&
      plotData.some(row => row[k] !== null)
    );
    return showAllMetrics ? allKeys : allKeys.filter(k => METRIC_STYLES[k]);
  }, [plotData, showAllMetrics]);

  // --- HELPERS ---
  const getDataBounds = () => {
    if (plotData.length === 0) return { min: 0, max: 0 };
    const allSteps = plotData.map(d => d.step);
    return { min: Math.min(...allSteps), max: Math.max(...allSteps) };
  };

  // --- ZOOM HANDLER (WHEEL) ---
  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();

    if (plotData.length === 0) return;

    const { min: dataMin, max: dataMax } = getDataBounds();
    const [currentMin, currentMax] = xDomain || [dataMin, dataMax];
    const range = currentMax - currentMin;
    const ZOOM_SPEED = 0.1;
    const delta = range * ZOOM_SPEED;


    let newMin, newMax;

    if (e.deltaY < 0) { // Zoom In
      newMin = currentMin + delta;
      newMax = currentMax - delta;
    } else { // Zoom Out
      newMin = currentMin - delta;
      newMax = currentMax + delta;
    }

    if (newMax <= newMin) return;
    setXDomain([newMin, newMax]);
  }, [xDomain, plotData]);

  useEffect(() => {
    const node = chartRef.current;
    if (!node) return;

    node.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      node.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);

  // --- PAN HANDLERS (DRAG) ---
  const handleMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation(); // Stop React Flow from dragging the node
    e.preventDefault();
    setIsDragging(true);
    lastMouseX.current = e.clientX;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    // Only run if we are actively dragging
    if (!isDragging || !chartRef.current || plotData.length === 0) return;

    e.stopPropagation();
    e.preventDefault();

    const currentX = e.clientX;
    const pixelDelta = lastMouseX.current - currentX; // Positive = Dragged Left

    // 1. Calculate how much to shift the domain
    const { width } = chartRef.current.getBoundingClientRect();
    const { min: dataMin, max: dataMax } = getDataBounds();
    const [currentMin, currentMax] = xDomain || [dataMin, dataMax];
    const domainRange = currentMax - currentMin;

    // Convert pixels to domain units
    const domainShift = (pixelDelta / width) * domainRange;

    // 2. Apply shift
    setXDomain([currentMin + domainShift, currentMax + domainShift]);

    // 3. Update reference for next frame
    lastMouseX.current = currentX;
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDragging(false);
  };

  const resetZoom = (e: React.MouseEvent) => {
    e.stopPropagation();
    setXDomain(null);
  };

  return (
    <div className={clsx(
      "min-w-[600px] h-[450px] rounded-xl border bg-slate-900/95 backdrop-blur-md shadow-2xl flex flex-col transition-all duration-200",
      selected ? "border-blue-400 ring-2 ring-blue-500/20" : "border-slate-800"
    )}>
      <Handle type="target" position={Position.Left} id="run_id" className="!bg-slate-900 border-blue-500" />

      {/* HEADER */}
      <div className="p-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 rounded-t-xl cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-blue-500" />
          <span className="font-bold text-slate-200 text-sm">Training Loss</span>
        </div>

        <div className="flex items-center gap-2 nodrag cursor-auto">
          {xDomain && (
            <button onClick={resetZoom} className="flex items-center gap-1 px-2 py-1 text-[10px] font-bold rounded border bg-yellow-500/10 text-yellow-500 border-yellow-500/30 hover:bg-yellow-500/20">
              <RefreshCcw size={10} /> RESET
            </button>
          )}
          <button onClick={() => setUseLogScale(!useLogScale)} className="px-2 py-1 text-[10px] font-bold rounded border bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700">
            {useLogScale ? "LOG" : "LIN"}
          </button>
        </div>
      </div>

      {/* CHART AREA */}
      <div
        ref={chartRef}
        className={clsx(
          "flex-1 p-2 min-h-0 relative nodrag nowheel bg-slate-900/50 pointer-events-auto rounded-b-xl",
          isDragging ? "cursor-grabbing" : "cursor-default" // Change cursor on drag
        )}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}  // SAFETY END
      >
        {error || plotData.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
            {error ? error : "Waiting for data..."}
          </div>
        ) : (
          /* "select-none" prevents text highlighting while dragging */
          <div className="w-full h-full select-none">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={plotData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />

                <XAxis
                  dataKey="step"
                  type="number"
                  stroke="#64748b"
                  fontSize={10}
                  domain={xDomain || ['dataMin', 'dataMax']}
                  allowDataOverflow={true}
                  tickCount={6}
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
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(val: any) => [formatScientific(val), '']}
                />

                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} formatter={(val) => METRIC_STYLES[val]?.name || val} />

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
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(LossCurveNode);
