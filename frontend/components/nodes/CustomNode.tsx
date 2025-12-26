import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { clsx } from 'clsx';
import {
  Settings, Database, Activity, Box, Play, Layers,
  Cpu, Wind, Flame, Hexagon, Anchor, FileJson, HelpCircle
} from 'lucide-react';

// 1. Icon Mapping
const IconMap: Record<string, any> = {
  "database": Database, "function": Activity, "box": Box, "play": Play,
  "settings": Settings, "layers": Layers, "network-wired": Cpu,
  "wind": Wind, "fire": Flame, "hexagon": Hexagon, "anchor": Anchor,
  "default": FileJson
};

// 2. Port Colors Mapping
const PortColors: Record<string, string> = {
  "model": "bg-purple-500 border-purple-500",
  "dataset": "bg-emerald-500 border-emerald-500",
  "solver": "bg-orange-500 border-orange-500",
  "problem": "bg-blue-500 border-blue-500",
  "default": "bg-slate-400 border-slate-400"
};

export interface NodeData {
  label: string;
  category: string;
  icon?: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  config?: any;
}

const CustomNode = ({ data, selected }: NodeProps<NodeData>) => {
  const IconComponent = data.icon && IconMap[data.icon]
    ? IconMap[data.icon]
    : HelpCircle;

  const inputKeys = Object.keys(data.inputs || {});
  const outputKeys = Object.keys(data.outputs || {});

  return (
    <div className={clsx(
      "min-w-[240px] rounded-xl border bg-slate-900 shadow-2xl transition-all duration-200",
      selected
        ? "border-blue-400 ring-4 ring-blue-500/30 shadow-blue-500/50"
        : "border-slate-700 hover:border-slate-600",
    )}>

      {/* Header Line */}
      <div className={clsx(
        "h-1.5 w-full rounded-t-xl",
        data.category === "Physics" ? "bg-blue-500" :
          data.category === "Model" ? "bg-purple-500" :
            data.category === "Training" ? "bg-orange-500" : "bg-slate-600"
      )} />

      <div className="p-4 bg-slate-900">
        {/* Title Area */}
        <div className="flex justify-between items-start mb-5">
          <div className="flex items-center gap-3">
            {/* Icon Box */}
            <div className="p-2.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 shadow-lg">
              <IconComponent size={20} />
            </div>
            <div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-1">
                {data.category}
              </span>
              <h3 className="text-sm font-bold text-slate-100 leading-tight">
                {data.label}
              </h3>
            </div>
          </div>
          {/* Settings Button */}
          {data.config && Object.keys(data.config).length > 0 && (
            <button className="text-slate-500 hover:text-slate-300 transition-colors p-1.5 hover:bg-slate-800 rounded border border-slate-700">
              <Settings size={14} />
            </button>
          )}
        </div>

        <div className="flex flex-row gap-8">
          {/* Inputs Column */}
          <div className="flex flex-col gap-3 flex-1">
            {inputKeys.map((key) => {
              const portData = data.inputs[key];
              const isObject = typeof portData === 'object' && portData !== null;
              const type = isObject ? portData.type : portData;
              const isRequired = isObject ? (portData.required !== false) : true;
              const colorClass = PortColors[type] || PortColors.default;

              return (
                <div key={key} className="relative flex items-center h-5 group">
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={key}
                    className={clsx(
                      "!w-3 !h-3 !border-2 !-left-[18px] transition-all hover:scale-125 shadow-lg",
                      isRequired
                        ? colorClass
                        : `${colorClass.replace('bg-', 'text-')} !bg-slate-900`
                    )}
                  />
                  <span className={clsx(
                    "text-xs font-medium capitalize transition-colors",
                    isRequired ? "text-slate-200" : "text-slate-500 italic"
                  )}>
                    {key}
                  </span>
                </div>
              );
            })}
            {inputKeys.length === 0 && (
              <div className="text-[10px] text-slate-600 italic py-1">No Inputs</div>
            )}
          </div>

          {/* Outputs Column */}
          <div className="flex flex-col gap-3 items-end flex-1">
            {outputKeys.map((key) => {
              const portData = data.outputs[key];
              const type = (typeof portData === 'object') ? portData.type : portData;
              const colorClass = PortColors[type] || PortColors.default;

              return (
                <div key={key} className="relative flex items-center h-5 justify-end group">
                  <span className="text-xs font-medium text-slate-200 capitalize mr-2">
                    {key}
                  </span>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={key}
                    className={clsx(
                      "!w-3 !h-3 !border-2 !-right-[18px] transition-all hover:scale-125 shadow-lg",
                      colorClass
                    )}
                  />
                </div>
              );
            })}
            {outputKeys.length === 0 && (
              <div className="text-[10px] text-slate-600 italic py-1">No Outputs</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(CustomNode);