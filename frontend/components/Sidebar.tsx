import React, { useMemo, useState } from 'react';
import {
    Database, Activity, Box, Play, Settings, Layers, FileJson,
    Cpu, Wind, Plus, Search, X, ChevronRight, ChevronDown, Zap
} from 'lucide-react';
import { NodeData } from './nodes/CustomNode';

const IconMap: Record<string, any> = {
    "database": Database, "function": Activity, "box": Box, "play": Play,
    "settings": Settings, "layers": Layers, "network-wired": Cpu,
    "truck": Database, "wind": Wind, "activity": Activity, "default": FileJson
};

interface SidebarProps {
    registry: Record<string, NodeData>;
    onAddNode: (type: string) => void;
    onClear: () => void;
    stats: { nodes: number; edges: number };
}

export default function Sidebar({ registry, onAddNode, onClear, stats }: SidebarProps) {
    const [searchTerm, setSearchTerm] = useState("");
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
    const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());

    // --- Filtering Logic ---
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

    return (
        <div className="w-96 flex-shrink-0 border-r border-slate-800/50 bg-slate-900/30 backdrop-blur-xl flex flex-col shadow-2xl">
            {/* Header */}
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

                {/* Search Input */}
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

                {/* Quick Category Filters */}
                <div className="flex flex-wrap gap-2 mt-3">
                    <button
                        onClick={() => setSelectedCategory(null)}
                        className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${!selectedCategory ? 'bg-blue-500 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}
                    >
                        All
                    </button>
                    {Object.keys(categories).slice(0, 3).map(cat => (
                        <button
                            key={cat}
                            onClick={() => setSelectedCategory(cat === selectedCategory ? null : cat)}
                            className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${selectedCategory === cat ? 'bg-emerald-500 text-white' : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'}`}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            </div>

            {/* Node List */}
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
                                        <button
                                            key={type}
                                            onClick={() => onAddNode(type)}
                                            className="group flex items-center gap-3 p-3 rounded-xl bg-gradient-to-br from-slate-800/40 to-slate-800/20 border border-slate-700/30 hover:border-blue-500/50 hover:from-slate-800/60 hover:to-slate-800/40 hover:shadow-lg hover:shadow-blue-500/10 transition-all text-left transform hover:scale-[1.02] active:scale-[0.98]"
                                        >
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

            {/* Footer Stats */}
            <div className="p-4 border-t border-slate-800/50 bg-slate-900/50">
                <div className="flex items-center justify-between text-xs text-slate-500">
                    <div className="flex items-center gap-4">
                        <span>{stats.nodes} Nodes</span>
                        <span>{stats.edges} Connections</span>
                    </div>
                    <button onClick={onClear} className="text-red-400 hover:text-red-300 transition-colors">Clear</button>
                </div>
            </div>
        </div>
    );
}
