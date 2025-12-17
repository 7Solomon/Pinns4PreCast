// src/components/nodes/WidgetNode.tsx
import React, { memo, useEffect, useState } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Papa from 'papaparse';
import { clsx } from 'clsx';
import { Activity, Loader2 } from 'lucide-react';

// Common handle styles
const HandleStyle = "!w-3 !h-3 !border-2 !bg-slate-900";

const LossCurveNode = ({ data, selected }: NodeProps) => {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // 1. Get Widget Spec from inputs or config
  // In your python node, you output 'widget_spec'. 
  // For now, we assume the CONFIG contains the initial setup, 
  // or we read the 'run_id' from the inputs if you are propagating data live.

  // HOWEVER, a cleaner pattern for the Editor:
  // The Node Config itself usually holds the settings.
  // The 'run_id' often comes from a prop or context if connected live.

  // Let's assume for this specific node, the user enters 'run_id' in config 
  // OR it's passed via connection (which is harder to read in ReactFlow without edge traversal).

  // SIMPLIFICATION: We will poll based on a config field for now, 
  // or you can set this state via a WebSocket/API event.

  const runId = data.config?.run_id || "latest";
  const metrics = data.config?.metrics || ["loss_physics", "loss_bc", "loss_ic", "loss", "loss_step", "loss_epoch", "loss_phys_temperature", "loss_phys_alpha", "loss_bc_temperature", "loss_ic_temperature", "loss_ic_alpha"];
  const pollInterval = data.config?.refresh_rate || 2000;

  // 2. Polling Effect
  useEffect(() => {
    if (!runId) return;

    const fetchData = async () => {
      try {
        // Adjust URL to your actual API
        const response = await fetch(`http://localhost:8000/runs/${runId}/metrics.csv`);
        const text = await response.text();

        // Parse CSV
        const result = Papa.parse(text, { header: true, dynamicTyping: true });
        const rows = result.data;

        // Basic filtering/cleaning if needed
        setPlotData(rows.slice(-50)); // Keep last 50 points for "Live" feel
      } catch (e) {
        console.error("Polling failed", e);
      }
    };

    const interval = setInterval(fetchData, pollInterval);
    fetchData(); // Initial load

    return () => clearInterval(interval);
  }, [runId, pollInterval]);

  return (
    <div className={clsx(
      "min-w-[400px] h-[300px] rounded-xl border bg-slate-900/95 backdrop-blur-md shadow-2xl flex flex-col",
      selected ? "border-orange-400 ring-2 ring-orange-500/20" : "border-slate-800"
    )}>
      {/* Handles (Still need to connect to things!) */}
      <Handle type="target" position={Position.Left} id="run_id" className={clsx(HandleStyle, "border-blue-500")} />

      {/* Header */}
      <div className="h-1 w-full bg-orange-500 rounded-t-xl" />
      <div className="p-3 border-b border-slate-800 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Activity size={16} className="text-orange-500" />
          <span className="font-bold text-slate-200">{data.label}</span>
        </div>
        {loading && <Loader2 size={14} className="animate-spin text-slate-500" />}
      </div>

      {/* Chart Area */}
      <div className="flex-1 p-2 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={plotData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="step" stroke="#94a3b8" fontSize={10} />
            <YAxis stroke="#94a3b8" fontSize={10} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }}
            />
            {metrics.map((metric: string, idx: number) => (
              <Line
                key={metric}
                type="monotone"
                dataKey={metric}
                stroke={idx === 0 ? "#f97316" : "#3b82f6"} // Orange then Blue
                strokeWidth={2}
                dot={false}
                connectNulls={true} // <--- CRITICAL for your sparse CSV
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default memo(LossCurveNode);
