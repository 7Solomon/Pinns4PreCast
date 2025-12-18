
import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Papa from 'papaparse';
import { clsx } from 'clsx';
import { Activity, Loader2 } from 'lucide-react';

const LossCurveNode = ({ data, selected }: NodeProps) => {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The run_id comes from the node's data after execution
  const runId = data.config?.run_id || "unknown";
  const metrics = data.config?.metrics || ["loss"];
  const refreshRate = data.config?.refresh_rate || 2000;

  useEffect(() => {
    if (!runId || runId === "unknown") {
      setError("No run ID available yet");
      return;
    }

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`http://localhost:8000/monitor/metrics/${runId}`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.metrics && result.metrics.length > 0) {
          // Keep last 50 points for smooth visualization
          setPlotData(result.metrics.slice(-50));
        }
      } catch (e) {
        console.error("Polling failed", e);
        setError("Failed to fetch metrics");
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchData();

    // Set up polling interval
    const interval = setInterval(fetchData, refreshRate);

    return () => clearInterval(interval);
  }, [runId, refreshRate]);

  return (
    <div className={clsx(
      "min-w-[500px] h-[350px] rounded-xl border bg-slate-900/95 backdrop-blur-md shadow-2xl flex flex-col",
      selected ? "border-orange-400 ring-2 ring-orange-500/20" : "border-slate-800"
    )}>
      {/* Input Handle for run_id */}
      <Handle
        type="target"
        position={Position.Left}
        id="run_id"
        className="!w-3 !h-3 !border-2 !bg-slate-900 border-blue-500"
      />

      {/* Header */}
      <div className="h-1 w-full bg-orange-500 rounded-t-xl" />
      <div className="p-3 border-b border-slate-800 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-orange-500" />
          <span className="font-bold text-slate-200">{data.label}</span>
          {runId !== "unknown" && (
            <span className="text-xs text-slate-500">Run: {runId.slice(0, 8)}</span>
          )}
        </div>
        {loading && <Loader2 size={14} className="animate-spin text-slate-500" />}
      </div>

      {/* Chart Area */}
      <div className="flex-1 p-3 min-h-0">
        {error ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">
            {error}
          </div>
        ) : plotData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">
            Waiting for training data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={plotData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="step"
                stroke="#94a3b8"
                fontSize={10}
                label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                stroke="#94a3b8"
                fontSize={10}
                domain={['auto', 'auto']}
                label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  borderColor: '#334155',
                  color: '#f1f5f9',
                  fontSize: '12px'
                }}
              />
              {metrics.map((metric: string, idx: number) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={getColorForMetric(metric, idx)}
                  strokeWidth={2}
                  dot={false}
                  connectNulls={true}
                  name={metric}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};

// Helper function for metric colors
function getColorForMetric(metric: string, fallbackIdx: number): string {
  const colorMap: Record<string, string> = {
    'loss': '#f97316',           // orange
    'loss_physics': '#3b82f6',   // blue
    'loss_bc': '#10b981',        // green
    'loss_ic': '#8b5cf6',        // purple
    'loss_epoch': '#f59e0b',     // amber
  };

  return colorMap[metric] || `hsl(${fallbackIdx * 60}, 70%, 60%)`;
}

export default memo(LossCurveNode);
