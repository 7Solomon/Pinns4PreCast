import React, { useState, useEffect, useCallback } from 'react';
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';
import { HardDrive } from 'lucide-react';

type RunInfo = {
    run_id: string;
    status?: string;
    start_time?: string;
    has_checkpoints?: boolean;
    checkpoints?: { name: string; path: string }[];
    has_sensors?: boolean;
    has_vtk?: boolean;
};

type RunSelectorData = {
    label: string;
    config?: { run_id?: string };  // this mirrors backend RunIdChooserConfig
};




export default function RunSelectorNode({ id, data }: NodeProps<RunSelectorData>) {
    const [runs, setRuns] = useState<RunInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { setNodes } = useReactFlow();
    useEffect(() => {
        let isMounted = true;
        setLoading(true);
        setError(null);

        fetch('http://localhost:8000/monitor/runs')
            .then(async (res) => {
                if (!res.ok) throw new Error(`Failed to load runs: ${res.status}`);
                return res.json();
            })
            .then((data) => {
                if (!isMounted) return;
                const runs: RunInfo[] = data?.runs || [];
                setRuns(runs);
            })
            .catch((err) => {
                if (!isMounted) return;
                console.error(err);
                setError(err.message ?? 'Failed to load runs');
            })
            .finally(() => {
                if (isMounted) setLoading(false);
            });

        return () => {
            isMounted = false;
        };
    }, []);

    const updateSelectedRun = useCallback(
        (newRunId: string) => {
            setNodes((nodes) =>
                nodes.map((node) => {
                    if (node.id !== id) return node;
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            config: {
                                ...(node.data.config || {}),
                                run_id: newRunId,
                            },
                        },
                    };
                })
            );
        },
        [id, setNodes]
    );
    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        updateSelectedRun(e.target.value || '');
    };

    const selectedRunId = data.config?.run_id || '';

    return (
        <div className="relative min-w-[240px] bg-slate-950 border border-slate-800 rounded-xl shadow-xl p-3">
            {/* Header */}
            <div className="flex items-center justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 flex items-center justify-center bg-slate-900 border border-slate-800 rounded-lg text-emerald-400">
                        <HardDrive size={18} />
                    </div>
                    <div className="min-w-0">
                        <div
                            className="text-sm font-bold text-slate-200 truncate"
                            title={data.label || 'Run Selector'}
                        >
                            {data.label || 'Run Selector'}
                        </div>
                        <div className="text-[10px] uppercase tracking-wider font-medium text-slate-500">
                            Select Training Run
                        </div>
                    </div>
                </div>
            </div>

            {/* Body */}
            <div className="text-xs text-slate-300 space-y-2">
                {loading && <div className="text-slate-500 text-[11px] italic">Loading runs…</div>}
                {error && (
                    <div className="text-red-400 text-[11px]">
                        Error: {error}
                    </div>
                )}

                {!loading && !error && (
                    <div className="flex flex-col gap-1">
                        <label className="text-[11px] text-slate-400">Run ID</label>
                        <select
                            className="text-xs bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-100 focus:outline-none focus:border-emerald-500"
                            value={selectedRunId}
                            onChange={handleChange}
                        >
                            <option value="">-- Select Run --</option>
                            {runs.map((run) => (
                                <option key={run.run_id} value={run.run_id}>
                                    {run.run_id}
                                    {run.status ? ` [${run.status}]` : ''}
                                    {run.has_checkpoints ? ' (ckpt)' : ''}
                                </option>
                            ))}
                        </select>

                        {selectedRunId ? (
                            <div className="mt-1 text-[10px] text-emerald-400 font-mono">
                                Active: {selectedRunId}
                            </div>
                        ) : (
                            <div className="mt-1 text-[10px] text-slate-500 italic">
                                No run selected
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Output handle: exposes run_id */}
            <Handle
                type="source"
                position={Position.Right}
                id="run_id"
                className="!bg-emerald-500 !w-3 !h-3 !border-2 !border-slate-950 hover:scale-125 transition-transform"
            />
        </div>
    );
}
