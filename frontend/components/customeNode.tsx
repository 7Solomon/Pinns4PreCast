import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { clsx } from 'clsx';
import { Settings } from 'lucide-react';

// DEFINITION: This matches what your Python backend sends
export type NodeData = {
    label: string;
    category: string;
    inputs: Record<string, any>;
    outputs: Record<string, any>;
    config: any;
};

export default function CustomNode({ data, selected }: NodeProps<NodeData>) {

    // Color coding based on category
    const headerColor = data.category === "Physics" ? "bg-blue-600" :
        data.category === "Model" ? "bg-purple-600" :
            data.category === "Solver" ? "bg-emerald-600" :
                "bg-slate-700";

    // Safely get keys (default to empty array if undefined)
    const inputKeys = Object.keys(data.inputs || {});
    const outputKeys = Object.keys(data.outputs || {});

    return (
        <div className={clsx(
            "min-w-[200px] rounded-lg border-2 bg-slate-900 shadow-xl transition-all",
            selected ? "border-blue-400 shadow-blue-500/20" : "border-slate-700"
        )}>

            {/* Header */}
            <div className={`${headerColor} px-3 py-2 rounded-t-md flex justify-between items-center`}>
                <span className="text-sm font-bold text-white tracking-wide">{data.label}</span>
                <Settings className="w-4 h-4 text-white/50 cursor-pointer hover:text-white" />
            </div>

            {/* Body */}
            <div className="p-3 flex flex-row gap-4 min-h-[50px]">

                {/* Input Ports (Left) */}
                <div className="flex flex-col gap-3">
                    {inputKeys.map((key) => (
                        <div key={key} className="relative flex items-center h-4">
                            <Handle
                                type="target"
                                position={Position.Left}
                                id={key}
                                className="!w-3 !h-3 !bg-slate-400 !-left-4 hover:!bg-blue-400 transition-colors"
                            />
                            <span className="text-xs text-slate-300 ml-1 capitalize">{key}</span>
                        </div>
                    ))}
                </div>

                {/* Spacer */}
                <div className="flex-1"></div>

                {/* Output Ports (Right) */}
                <div className="flex flex-col gap-3 items-end">
                    {outputKeys.map((key) => (
                        <div key={key} className="relative flex items-center h-4 justify-end">
                            <span className="text-xs text-slate-300 mr-1 capitalize">{key}</span>
                            <Handle
                                type="source"
                                position={Position.Right}
                                id={key}
                                className="!w-3 !h-3 !bg-slate-400 !-right-4 hover:!bg-green-400 transition-colors"
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
