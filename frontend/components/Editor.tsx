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
  Panel,
  MiniMap
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';
import {
  Database, Activity, Box, Play, Settings, Layers, FileJson, Cpu, Wind, Plus,
  Search, Save, FolderOpen, Zap, ChevronRight, ChevronDown, X
} from 'lucide-react';
import CustomNode, { NodeData } from './customeNode';
import MonitorNode from './MonitorNode';
import { SaveDialog, LoadDialog } from './SaveLoadDialogs';

const nodeTypes = {
  custom: CustomNode,
  monitor: MonitorNode
};

const IconMap: Record<string, any> = {
  "database": Database, "function": Activity, "box": Box, "play": Play,
  "settings": Settings, "layers": Layers, "network-wired": Cpu,
  "truck": Database, "wind": Wind, "activity": Activity, "default": FileJson
};

export default function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [registry, setRegistry] = useState<Record<string, NodeData>>({});
  const [searchTerm, setSearchTerm] = useState("");
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // Dialog states
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showLoadDialog, setShowLoadDialog] = useState(false);

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

  const categories = useMemo(() => {
    const groups: Record<string, string[]> = {};
    Object.values(registry).forEach((node: any) => {
      const cat = node.category || "Uncategorized";
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(node.type);
    });
    return groups;
  }, [registry]);

  const filteredCategories = useMemo(() => {
    if (!searchTerm && !selectedCategory) return categories;

    const filtered: Record<string, string[]> = {};
    Object.entries(categories).forEach(([category, nodeTypes]) => {
      if (selectedCategory && category !== selectedCategory) return;

      const matchingNodes = nodeTypes.filter(type => {
        const node = registry[type];
        const searchLower = searchTerm.toLowerCase();
        return (
          node.label.toLowerCase().includes(searchLower) ||
          (node as any).description?.toLowerCase().includes(searchLower) ||
          type.toLowerCase().includes(searchLower)
        );
      });

      if (matchingNodes.length > 0) {
        filtered[category] = matchingNodes;
      }
    });

    return filtered;
  }, [categories, searchTerm, selectedCategory, registry]);

  const toggleCategory = (category: string) => {
    const newCollapsed = new Set(collapsedCategories);
    if (newCollapsed.has(category)) {
      newCollapsed.delete(category);
    } else {
      newCollapsed.add(category);
    }
    setCollapsedCategories(newCollapsed);
  };

  const addNode = (type: string) => {
    const nodeDef = registry[type];
    if (!nodeDef) return;

    let nodeType = 'custom';
    if (type === 'live_training_monitor') {
      nodeType = 'monitor';
    }

    const newNode: Node = {
      id: `${type}-${Date.now()}`,
      type: nodeType,
      position: {
        x: Math.random() * 400 + 200,
        y: Math.random() * 400 + 100
      },
      data: { ...nodeDef, type }
    };
    setNodes((nds) => nds.concat(newNode));
  };

  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) => addEdge(params, eds));
  }, [setEdges]);

  const runSimulation = async () => {
    setIsRunning(true);
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
    } finally {
      setIsRunning(false);
    }
  };

  const clearGraph = () => {
    if (confirm("Clear all nodes and connections?")) {
      setNodes([]);
      setEdges([]);
    }
  };

  // ============================================================================
  // SAVE/LOAD FUNCTIONS
  // ============================================================================

  const handleSave = async (name: string, description: string, tags: string[], overwrite: boolean) => {
    const payload = {
      name,
      description,
      tags,
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
      overwrite
    };

    try {
      const res = await axios.post('http://localhost:8000/graphs/save', payload);
      alert(`Graph saved as "${name}"`);
      setShowSaveDialog(false);
    } catch (e: any) {
      alert(`Failed to save: ${e.response?.data?.detail || e.message}`);
    }
  };

  const handleLoad = async (filename: string) => {
    try {
      const res = await axios.get(`http://localhost:8000/graphs/load/${filename}`);
      const graphData = res.data.graph;

      // Clear current graph
      setNodes([]);
      setEdges([]);

      // Load nodes
      const loadedNodes = graphData.nodes.map((n: any) => {
        const nodeDef = registry[n.type];

        let nodeType = 'custom';
        if (n.type === 'live_training_monitor') {
          nodeType = 'monitor';
        }

        return {
          id: n.id,
          type: nodeType,
          position: n.position,
          data: { ...nodeDef, type: n.type, config: n.config }
        };
      });

      // Load edges
      const loadedEdges = graphData.connections.map((c: any, idx: number) => ({
        id: `e${idx}`,
        source: c.source_node,
        sourceHandle: c.source_port,
        target: c.target_node,
        targetHandle: c.target_port
      }));

      setNodes(loadedNodes);
      setEdges(loadedEdges);
      setShowLoadDialog(false);

      alert(`Loaded "${graphData.name}"`);
    } catch (e: any) {
      alert(`Failed to load: ${e.response?.data?.detail || e.message}`);
    }
  };

  return (
    <div className="w-full h-full flex bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-200 font-sans">

      {/* Sidebar */}
      <div className="w-96 flex-shrink-0 border-r border-slate-800/50 bg-slate-900/30 backdrop-blur-xl flex flex-col shadow-2xl">
        <div className="p-6 border-b border-slate-800/50 bg-gradient-to-br from-blue-600/10 to-emerald-600/10">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500">
              <Zap size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                PINN Studio
              </h1>
              <p className="text-xs text-slate-500">Physics-Informed Neural Networks</p>
            </div>
          </div>

          <div className="relative mt-4">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
            />
            {searchTerm && (
              <button onClick={() => setSearchTerm("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                <X size={14} />
              </button>
            )}
          </div>

          <div className="flex flex-wrap gap-2 mt-3">
            <button onClick={() => setSelectedCategory(null)} className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${!selectedCategory ? 'bg-blue-500 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}>
              All
            </button>
            {Object.keys(categories).slice(0, 3).map(cat => (
              <button key={cat} onClick={() => setSelectedCategory(cat === selectedCategory ? null : cat)} className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${selectedCategory === cat ? 'bg-emerald-500 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}>
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {Object.entries(filteredCategories).map(([category, nodeTypes]) => (
            <div key={category}>
              <button onClick={() => toggleCategory(category)} className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-slate-800/30 transition-colors group mb-2">
                <div className="flex items-center gap-2">
                  {collapsedCategories.has(category) ? <ChevronRight size={16} className="text-slate-500" /> : <ChevronDown size={16} className="text-slate-500" />}
                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">{category}</h3>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-500">{nodeTypes.length}</span>
                </div>
              </button>

              {!collapsedCategories.has(category) && (
                <div className="grid grid-cols-1 gap-2 ml-2">
                  {nodeTypes.map((type) => {
                    const node = registry[type];
                    const Icon = IconMap[node.icon as string] || IconMap["default"];

                    return (
                      <button key={type} onClick={() => addNode(type)} className="group flex items-center gap-3 p-3 rounded-xl bg-gradient-to-br from-slate-800/40 to-slate-800/20 border border-slate-700/30 hover:border-blue-500/50 hover:from-slate-800/60 hover:to-slate-800/40 hover:shadow-lg hover:shadow-blue-500/10 transition-all text-left transform hover:scale-[1.02] active:scale-[0.98]">
                        <div className="p-2.5 rounded-lg bg-gradient-to-br from-slate-900 to-slate-800 text-blue-400 group-hover:text-blue-300 group-hover:scale-110 group-hover:rotate-3 transition-all shadow-lg">
                          <Icon size={18} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-semibold text-slate-200 group-hover:text-white transition-colors">{node.label}</div>
                          <div className="text-[10px] text-slate-500 line-clamp-1 mt-0.5">{(node as any).description || "Add to graph"}</div>
                        </div>
                        <Plus size={14} className="opacity-0 group-hover:opacity-100 text-blue-400 transition-all group-hover:rotate-90" />
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-slate-800/50 bg-slate-900/50">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <div className="flex items-center gap-4">
              <span>{nodes.length} Nodes</span>
              <span>{edges.length} Connections</span>
            </div>
            <button onClick={clearGraph} className="text-red-400 hover:text-red-300 transition-colors">Clear</button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          className="bg-slate-950"
          defaultEdgeOptions={{ style: { stroke: '#475569', strokeWidth: 2 }, animated: true }}
        >
          <Background color="#334155" variant={BackgroundVariant.Dots} gap={24} size={1.5} className="opacity-40" />
          <Controls className="!bg-slate-800/90 !backdrop-blur-sm !border-slate-700 !shadow-xl" />
          <MiniMap className="!bg-slate-800/90 !backdrop-blur-sm !border-slate-700" />

          <Panel position="top-right" className="flex gap-3">
            <button
              onClick={() => setShowSaveDialog(true)}
              className="flex items-center gap-2 bg-slate-800/90 backdrop-blur-sm hover:bg-slate-700/90 text-slate-200 px-4 py-2.5 rounded-lg shadow-xl border border-slate-700/50 transition-all hover:scale-105 active:scale-95"
            >
              <Save size={16} />Save
            </button>

            <button
              onClick={() => setShowLoadDialog(true)}
              className="flex items-center gap-2 bg-slate-800/90 backdrop-blur-sm hover:bg-slate-700/90 text-slate-200 px-4 py-2.5 rounded-lg shadow-xl border border-slate-700/50 transition-all hover:scale-105 active:scale-95"
            >
              <FolderOpen size={16} />Load
            </button>

            <button
              onClick={runSimulation}
              disabled={isRunning}
              className="flex items-center gap-2 bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 disabled:from-slate-700 disabled:to-slate-700 text-white px-6 py-2.5 rounded-lg shadow-xl shadow-emerald-900/30 font-bold transition-all hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isRunning ? (
                <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Running...</>
              ) : (
                <><Play size={18} fill="currentColor" />Run Training</>
              )}
            </button>
          </Panel>
        </ReactFlow>
      </div>

      {/* Dialogs */}
      <SaveDialog
        isOpen={showSaveDialog}
        onClose={() => setShowSaveDialog(false)}
        onSave={handleSave}
        nodes={nodes}
        edges={edges}
      />

      <LoadDialog
        isOpen={showLoadDialog}
        onClose={() => setShowLoadDialog(false)}
        onLoad={handleLoad}
      />

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.3); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(71, 85, 105, 0.5); border-radius: 3px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(71, 85, 105, 0.8); }
      `}</style>
    </div>
  );
}