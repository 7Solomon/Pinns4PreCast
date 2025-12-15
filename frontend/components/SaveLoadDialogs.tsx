import React, { useState, useEffect } from 'react';
import { X, Save, FolderOpen, Trash2, Copy, Calendar, Hash, Tag } from 'lucide-react';

interface GraphMetadata {
    name: string;
    filename: string;
    description: string;
    tags: string[];
    created_at: string;
    updated_at: string;
    node_count: number;
    connection_count: number;
}

interface SaveDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, description: string, tags: string[], overwrite: boolean) => void;
    nodes: any[];
    edges: any[];
}

interface LoadDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onLoad: (filename: string) => void;
}

// ============================================================================
// SAVE DIALOG
// ============================================================================

export function SaveDialog({ isOpen, onClose, onSave, nodes, edges }: SaveDialogProps) {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [tags, setTags] = useState<string[]>([]);
    const [tagInput, setTagInput] = useState('');
    const [overwrite, setOverwrite] = useState(false);
    const [existingGraphs, setExistingGraphs] = useState<GraphMetadata[]>([]);

    useEffect(() => {
        if (isOpen) {
            // Fetch existing graphs to check for duplicates
            fetch('http://localhost:8000/graphs/list')
                .then(res => res.json())
                .then(data => setExistingGraphs(data.graphs))
                .catch(console.error);
        }
    }, [isOpen]);

    const handleAddTag = () => {
        if (tagInput.trim() && !tags.includes(tagInput.trim())) {
            setTags([...tags, tagInput.trim()]);
            setTagInput('');
        }
    };

    const handleRemoveTag = (tag: string) => {
        setTags(tags.filter(t => t !== tag));
    };

    const handleSave = () => {
        if (!name.trim()) {
            alert('Please enter a graph name');
            return;
        }

        // Check if name exists
        const exists = existingGraphs.some(g =>
            g.name.toLowerCase() === name.trim().toLowerCase()
        );

        if (exists && !overwrite) {
            if (!confirm(`A graph named "${name}" already exists. Overwrite it?`)) {
                return;
            }
            setOverwrite(true);
        }

        onSave(name.trim(), description.trim(), tags, overwrite || exists);

        // Reset form
        setName('');
        setDescription('');
        setTags([]);
        setTagInput('');
        setOverwrite(false);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-lg mx-4">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500">
                            <Save size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">Save Graph Template</h2>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {nodes.length} nodes, {edges.length} connections
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                        <X size={20} className="text-slate-400" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-4">
                    {/* Name Input */}
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                            Graph Name *
                        </label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="e.g., Concrete Heat Transfer Model"
                            className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                            autoFocus
                        />
                    </div>

                    {/* Description Input */}
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                            Description
                        </label>
                        <textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="What does this graph do?"
                            rows={3}
                            className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all resize-none"
                        />
                    </div>

                    {/* Tags Input */}
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                            Tags
                        </label>
                        <div className="flex gap-2 mb-2">
                            <input
                                type="text"
                                value={tagInput}
                                onChange={(e) => setTagInput(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                                placeholder="Add a tag..."
                                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                            />
                            <button
                                onClick={handleAddTag}
                                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                            >
                                Add
                            </button>
                        </div>
                        {tags.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                                {tags.map(tag => (
                                    <span
                                        key={tag}
                                        className="inline-flex items-center gap-1.5 px-3 py-1 bg-blue-500/20 border border-blue-500/30 text-blue-300 text-sm rounded-full"
                                    >
                                        <Tag size={12} />
                                        {tag}
                                        <button
                                            onClick={() => handleRemoveTag(tag)}
                                            className="hover:text-blue-100"
                                        >
                                            <X size={12} />
                                        </button>
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-slate-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-semibold rounded-lg transition-all hover:scale-105 active:scale-95"
                    >
                        Save Graph
                    </button>
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// LOAD DIALOG
// ============================================================================

export function LoadDialog({ isOpen, onClose, onLoad }: LoadDialogProps) {
    const [graphs, setGraphs] = useState<GraphMetadata[]>([]);
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedTag, setSelectedTag] = useState<string | null>(null);

    useEffect(() => {
        if (isOpen) {
            fetchGraphs();
        }
    }, [isOpen]);

    const fetchGraphs = async () => {
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/graphs/list');
            const data = await res.json();
            setGraphs(data.graphs);
        } catch (err) {
            console.error('Failed to fetch graphs:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (filename: string, graphName: string) => {
        if (!confirm(`Delete "${graphName}"? This cannot be undone.`)) return;

        try {
            await fetch(`http://localhost:8000/graphs/delete/${filename}`, {
                method: 'DELETE'
            });
            fetchGraphs();
        } catch (err) {
            alert('Failed to delete graph');
            console.error(err);
        }
    };

    const handleDuplicate = async (filename: string, originalName: string) => {
        const newName = prompt(`Enter name for duplicate of "${originalName}":`, `${originalName} (Copy)`);
        if (!newName) return;

        try {
            const res = await fetch(`http://localhost:8000/graphs/duplicate/${filename}?new_name=${encodeURIComponent(newName)}`, {
                method: 'POST'
            });

            if (!res.ok) {
                const error = await res.json();
                alert(error.detail || 'Failed to duplicate graph');
                return;
            }

            fetchGraphs();
        } catch (err) {
            alert('Failed to duplicate graph');
            console.error(err);
        }
    };

    // Filter graphs
    const filteredGraphs = graphs.filter(g => {
        const matchesSearch = searchTerm === '' ||
            g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            g.description.toLowerCase().includes(searchTerm.toLowerCase());

        const matchesTag = !selectedTag || g.tags.includes(selectedTag);

        return matchesSearch && matchesTag;
    });

    // Get all unique tags
    const allTags = [...new Set(graphs.flatMap(g => g.tags))];

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-3xl mx-4 max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500">
                            <FolderOpen size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">Load Graph Template</h2>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {graphs.length} saved {graphs.length === 1 ? 'graph' : 'graphs'}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                        <X size={20} className="text-slate-400" />
                    </button>
                </div>

                {/* Search and Filter */}
                <div className="p-4 border-b border-slate-800 space-y-3">
                    <input
                        type="text"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        placeholder="Search graphs..."
                        className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                    />

                    {allTags.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            <button
                                onClick={() => setSelectedTag(null)}
                                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${!selectedTag
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                                    }`}
                            >
                                All
                            </button>
                            {allTags.map(tag => (
                                <button
                                    key={tag}
                                    onClick={() => setSelectedTag(tag === selectedTag ? null : tag)}
                                    className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${selectedTag === tag
                                        ? 'bg-emerald-500 text-white'
                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                                        }`}
                                >
                                    {tag}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* Graph List */}
                <div className="flex-1 overflow-y-auto p-4 space-y-3">
                    {loading ? (
                        <div className="text-center py-12 text-slate-500">
                            <div className="w-8 h-8 border-2 border-slate-700 border-t-blue-500 rounded-full animate-spin mx-auto mb-3" />
                            Loading graphs...
                        </div>
                    ) : filteredGraphs.length === 0 ? (
                        <div className="text-center py-12 text-slate-500">
                            <FolderOpen size={48} className="mx-auto mb-3 opacity-20" />
                            <p className="text-sm">
                                {searchTerm || selectedTag ? 'No matching graphs found' : 'No saved graphs yet'}
                            </p>
                        </div>
                    ) : (
                        filteredGraphs.map(graph => (
                            <div
                                key={graph.filename}
                                className="group bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 hover:border-blue-500/50 hover:bg-slate-800/60 transition-all"
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1 min-w-0">
                                        <h3 className="text-lg font-semibold text-white mb-1">
                                            {graph.name}
                                        </h3>
                                        {graph.description && (
                                            <p className="text-sm text-slate-400 mb-2 line-clamp-2">
                                                {graph.description}
                                            </p>
                                        )}
                                        <div className="flex items-center gap-4 text-xs text-slate-500">
                                            <span className="flex items-center gap-1">
                                                <Hash size={12} />
                                                {graph.node_count} nodes
                                            </span>
                                            <span className="flex items-center gap-1">
                                                <Calendar size={12} />
                                                {new Date(graph.updated_at).toLocaleDateString()}
                                            </span>
                                        </div>
                                        {graph.tags.length > 0 && (
                                            <div className="flex flex-wrap gap-1.5 mt-2">
                                                {graph.tags.map(tag => (
                                                    <span
                                                        key={tag}
                                                        className="px-2 py-0.5 bg-blue-500/20 border border-blue-500/30 text-blue-300 text-xs rounded"
                                                    >
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {/* Actions */}
                                    <div className="flex items-center gap-2 ml-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button
                                            onClick={() => onLoad(graph.filename)}
                                            className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
                                            title="Load this graph"
                                        >
                                            <FolderOpen size={16} />
                                        </button>
                                        <button
                                            onClick={() => handleDuplicate(graph.filename, graph.name)}
                                            className="p-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                                            title="Duplicate"
                                        >
                                            <Copy size={16} />
                                        </button>
                                        <button
                                            onClick={() => handleDelete(graph.filename, graph.name)}
                                            className="p-2 bg-red-600/80 hover:bg-red-600 text-white rounded-lg transition-colors"
                                            title="Delete"
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-slate-800">
                    <button
                        onClick={onClose}
                        className="w-full px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
                    >
                        Cancel
                    </button>
                </div>
            </div>
        </div>
    );
}