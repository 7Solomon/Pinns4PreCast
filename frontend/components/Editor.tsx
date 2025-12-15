"use client";

import React, { useCallback, useEffect, useState, useMemo } from 'react';
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  Connection,
  Node,
  Panel
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';
import {
  Database,
  Activity,
  Box,
  Play,
  Settings,
  Layers,
  FileJson,
  Cpu,
  Wind,
  Plus
} from 'lucide-react';
import CustomNode, { NodeData } from './customeNode';

const nodeTypes = { custom: CustomNode };

// Map Python string IDs to React Icons
const IconMap: Record<string, any> = {
  "database": Database,
  "function": Activity,
  "box": Box,
  "play": Play,
  "settings": Settings,
  "layers": Layers,
  "network-wired": Cpu,
  "truck": Database, // Loader
  "wind": Wind,     // PDE/Nature
  "default": FileJson
};

export default function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [registry, setRegistry] = useState<Record<string, NodeData>>({});

  // Fetch Registry
  useEffect(() => {
    axios.get<NodeData[]>('http://localhost:8000/registry')
      .then(res => {
        const reg: Record<string, NodeData> = {};
        res.data.forEach((n: any) => {
          reg[n.type] = n;
        });
        setRegistry(reg);
      })
      .catch(console.error);
  }, []);

  // Helper: Group nodes by Category
  const categories = useMemo(() => {
    const groups: Record<string, string[]> = {};
    Object.values(registry).forEach((node: any) => {
      const cat = node.category || "Uncategorized";
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(node.type);
    });
    return groups;
  }, [registry]);

  const addNode = (type: string) => {
    const nodeDef = registry[type];
    if (!nodeDef) return;

    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      type: 'custom',
      position: {
        x: Math.random() * 400 + 200,
        y: Math.random() * 400 + 100
      },
      data: { ...nodeDef }
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const runSimulation = async () => {
    const payload = {
      nodes: nodes.map(n => ({
        id: n.id,
        type: (n.data as any).type,
        config: n.data.config || {},
        position: n.position
      })),
      connections: edges.map(e => ({
        source_node: e.source,
        source_port: e.sourceHandle,
        target_node: e.target,
        target_port: e.targetHandle
      })),
      target_node_id: nodes.find(n => (n.data as any).category === 'Training')?.id || "unknown"
    };

    try {
      const res = await axios.post('http://localhost:8000/execute', payload);
      alert(`Success: ${res.data.message}`);
    } catch (e: any) {
      alert(`Error: ${e.response?.data?.detail || e.message}`);
    }
  };

  return (
    <div className="w-full h-full flex bg-[#0f172a] text-slate-200 font-sans">

      {/* --- SIDEBAR --- */}
      <div className="w-80 flex-shrink-0 border-r border-slate-800 bg-slate-900/50 backdrop-blur-sm flex flex-col">

        {/* Header */}
        <div className="p-6 border-b border-slate-800">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            PINN Studio
          </h1>
          <p className="text-xs text-slate-500 mt-1">PreCast Neural Solver</p>
        </div>

        {/* Scrollable Node List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {Object.entries(categories).map(([category, nodeTypes]) => (
            <div key={category}>
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 ml-1">
                {category}
              </h3>
              <div className="grid grid-cols-1 gap-2">
                {nodeTypes.map((type) => {
                  const node = registry[type];
                  const Icon = IconMap[node.icon as string] || IconMap["default"];

                  return (
                    <button
                      key={type}
                      onClick={() => addNode(type)}
                      className="group flex items-center gap-3 p-3 rounded-lg bg-slate-800 border border-slate-700/50 hover:border-blue-500/50 hover:bg-slate-800/80 hover:shadow-lg transition-all text-left"
                    >
                      <div className="p-2 rounded-md bg-slate-900 text-blue-400 group-hover:text-blue-300 group-hover:scale-110 transition-transform">
                        <Icon size={18} />
                      </div>
                      <div>
                        <div className="text-sm font-medium text-slate-200 group-hover:text-white">
                          {node.label}
                        </div>
                        <div className="text-[10px] text-slate-500 line-clamp-1">
                          {/* Use description if available, else trunc */}
                          {(node as any).description || "Add to graph"}
                        </div>
                      </div>
                      <Plus size={14} className="ml-auto opacity-0 group-hover:opacity-100 text-blue-400 transition-opacity" />
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* --- GRAPH CANVAS --- */}
      <div className="flex-1 relative bg-slate-950">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          className="bg-slate-950"
        >
          {/* Dots Pattern instead of plain black */}
          <Background color="#334155" variant={BackgroundVariant.Dots} gap={20} size={1} />

          <Controls className="!bg-slate-800 !border-slate-700 !fill-slate-400" />

          <Panel position="top-right">
            <button
              onClick={runSimulation}
              className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2.5 rounded-lg shadow-xl shadow-emerald-900/20 font-bold transition-all hover:scale-105 active:scale-95"
            >
              <Play size={18} fill="currentColor" />
              Run Training
            </button>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  );
}
