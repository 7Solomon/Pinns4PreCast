"use client";

import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { X, Save, Tag, Hash, Link2, RefreshCw, AlertTriangle, Copy } from "lucide-react";
import { GraphMetadata } from "./GraphMetadata";

interface SaveDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, description: string, tags: string[], overwrite: boolean) => void;
    nodes: any[];
    edges: any[];
    currentGraphName?: string | null; // The file currently loaded
}

export function SaveDialog({ isOpen, onClose, onSave, nodes, edges, currentGraphName }: SaveDialogProps) {
    // Form State
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [tags, setTags] = useState<string[]>([]);
    const [tagInput, setTagInput] = useState("");

    // Safety State
    const [confirmOverwrite, setConfirmOverwrite] = useState(false);

    // Data State
    const [existingGraphs, setExistingGraphs] = useState<GraphMetadata[]>([]);
    const [checking, setChecking] = useState(false);

    const portalTarget = typeof window !== "undefined" ? document.getElementById("modal-root") : null;

    // 1. Fetch existing graphs on open
    useEffect(() => {
        if (!isOpen) return;
        setChecking(true);
        fetch("http://localhost:8000/graphs/list")
            .then((res) => res.json())
            .then((data) => setExistingGraphs(data.graphs ?? []))
            .catch(console.error)
            .finally(() => setChecking(false));

        // Reset form
        setName("");
        setDescription("");
        setConfirmOverwrite(false);
    }, [isOpen]);

    // 2. Check for Name Collision
    const nameCollision = useMemo(() => {
        const trimmed = name.trim().toLowerCase();
        if (!trimmed) return false;
        // Check if name exists AND it is not the current file (if we want to allow saving current as new under same name)
        return existingGraphs.some((g) => g.name.toLowerCase() === trimmed);
    }, [existingGraphs, name]);

    // 3. Tag Logic
    const handleAddTag = () => {
        const t = tagInput.trim();
        if (!t || tags.includes(t)) return;
        setTags([...tags, t]);
        setTagInput("");
    };
    const handleRemoveTag = (tag: string) => setTags(tags.filter((t) => t !== tag));

    // 4. ACTION: Quick Update (Save Current)
    const handleQuickUpdate = () => {
        if (!currentGraphName) return;

        // Find metadata of current graph to preserve description/tags if user hasn't typed anything
        const currentMeta = existingGraphs.find(g => g.name === currentGraphName);

        onSave(
            currentGraphName,
            description || currentMeta?.description || "", // Use new desc or fallback to old
            tags.length > 0 ? tags : (currentMeta?.tags || []),
            true // Force Overwrite
        );
        onClose();
    };

    // 5. ACTION: Save New (or Overwrite selected)
    const handleSaveAs = () => {
        const trimmedName = name.trim();
        if (!trimmedName) return alert("Please enter a name");

        // If collision exists, user MUST have checked the box
        if (nameCollision && !confirmOverwrite) {
            return alert("This name already exists. Please check 'Overwrite existing file' to proceed.");
        }

        onSave(trimmedName, description, tags, true); // true because if collision+checked, we overwrite. If no collision, overwrite=true is harmless.
        onClose();
    };

    if (!isOpen || !portalTarget) return null;

    return createPortal(
        <div className="bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]" style={{ position: "fixed", inset: 0 }}>
            <div className="bg-slate-900/95 border border-slate-700 rounded-2xl shadow-2xl w-[min(92vw,36rem)] overflow-hidden flex flex-col max-h-[90vh]">

                {/* Header */}
                <div className="flex items-center justify-between px-6 py-5 border-b border-slate-800 bg-slate-950/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-blue-600 text-white">
                            <Save size={20} />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-white">Save Graph</h2>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {nodes.length} nodes • {edges.length} connections
                            </p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors">
                        <X size={18} />
                    </button>
                </div>

                <div className="p-6 overflow-y-auto custom-scrollbar space-y-8">

                    {/* SECTION 1: QUICK UPDATE (Only if file loaded) */}
                    {currentGraphName && (
                        <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 flex flex-col gap-3">
                            <div className="flex items-start justify-between">
                                <div>
                                    <h3 className="text-sm font-bold text-blue-200">Update Current File</h3>
                                    <p className="text-xs text-blue-300/70 mt-1">
                                        Overwrites <span className="font-mono bg-blue-500/20 px-1 rounded">{currentGraphName}</span> with current changes.
                                    </p>
                                </div>
                                <button
                                    onClick={handleQuickUpdate}
                                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold rounded-lg transition-colors shadow-lg shadow-blue-900/20"
                                >
                                    <RefreshCw size={14} /> Update
                                </button>
                            </div>
                        </div>
                    )}

                    {/* SECTION 2: SAVE AS NEW */}
                    <div className={currentGraphName ? "pt-6 border-t border-slate-800" : ""}>
                        <h3 className="text-sm font-bold text-slate-200 mb-4 flex items-center gap-2">
                            {currentGraphName ? <Copy size={14} className="text-slate-500" /> : null}
                            Save as New
                        </h3>

                        <div className="space-y-4">
                            {/* Name Input */}
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase">Filename</label>
                                <input
                                    type="text"
                                    value={name}
                                    onChange={(e) => {
                                        setName(e.target.value);
                                        setConfirmOverwrite(false); // Reset confirmation if name changes
                                    }}
                                    placeholder="e.g. Experiment_Variation_A"
                                    className={`w-full px-4 py-2.5 bg-slate-800 border rounded-lg text-white placeholder-slate-600 focus:outline-none focus:ring-2 transition-all
                                        ${nameCollision
                                            ? "border-amber-500/50 focus:ring-amber-500/30"
                                            : "border-slate-700 focus:ring-blue-500/50"
                                        }
                                    `}
                                />
                                {/* Collision Warning */}
                                {nameCollision && (
                                    <div className="mt-2 p-2 bg-amber-500/10 border border-amber-500/20 rounded-lg flex items-start gap-2">
                                        <AlertTriangle size={14} className="text-amber-500 mt-0.5 shrink-0" />
                                        <div className="flex-1">
                                            <p className="text-xs text-amber-200">
                                                File with this name already exists.
                                            </p>
                                            <label className="flex items-center gap-2 mt-2 cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={confirmOverwrite}
                                                    onChange={(e) => setConfirmOverwrite(e.target.checked)}
                                                    className="w-4 h-4 rounded border-amber-500/50 bg-amber-500/10 text-amber-500 focus:ring-0"
                                                />
                                                <span className="text-xs font-bold text-amber-500 select-none">Overwrite existing file</span>
                                            </label>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Description */}
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase">Description</label>
                                <textarea
                                    value={description}
                                    onChange={(e) => setDescription(e.target.value)}
                                    placeholder="Optional notes..."
                                    rows={2}
                                    className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none text-sm"
                                />
                            </div>

                            {/* Tags */}
                            <div>
                                <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase">Tags</label>
                                <div className="flex gap-2 mb-2">
                                    <input
                                        value={tagInput}
                                        onChange={(e) => setTagInput(e.target.value)}
                                        onKeyDown={(e) => e.key === "Enter" && handleAddTag()}
                                        placeholder="Add tag..."
                                        className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                                    />
                                    <button onClick={handleAddTag} className="px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors">
                                        Add
                                    </button>
                                </div>
                                <div className="flex flex-wrap gap-2 min-h-[24px]">
                                    {tags.map((tag) => (
                                        <span key={tag} className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 border border-slate-700 text-slate-300 text-xs rounded-full">
                                            {tag}
                                            <button onClick={() => handleRemoveTag(tag)} className="hover:text-white"><X size={12} /></button>
                                        </span>
                                    ))}
                                </div>
                            </div>

                            {/* Save As Button */}
                            <button
                                onClick={handleSaveAs}
                                disabled={!name || (nameCollision && !confirmOverwrite)}
                                className={`w-full py-3 rounded-lg font-bold shadow-lg transition-all flex items-center justify-center gap-2
                                    ${!name || (nameCollision && !confirmOverwrite)
                                        ? "bg-slate-800 text-slate-500 cursor-not-allowed shadow-none"
                                        : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/20 active:scale-[0.98]"
                                    }
                                `}
                            >
                                <Save size={16} />
                                {nameCollision ? "Overwrite & Save" : "Save as New File"}
                            </button>
                        </div>
                    </div>

                </div>
            </div>
        </div>,
        portalTarget
    );
}
