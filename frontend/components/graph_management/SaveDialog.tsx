import React, { useState, useEffect } from 'react';
import { X, Save, FolderOpen, Trash2, Copy, Calendar, Hash, Tag } from 'lucide-react';
import { GraphMetadata } from './GraphMetadata';

interface SaveDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, description: string, tags: string[], overwrite: boolean) => void;
    nodes: any[];
    edges: any[];
}

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
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999]">
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
