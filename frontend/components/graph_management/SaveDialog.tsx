"use client";

import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { X, Save, Tag, Hash, Link2 } from "lucide-react";
import { GraphMetadata } from "./GraphMetadata";

interface SaveDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (name: string, description: string, tags: string[], overwrite: boolean) => void;
    nodes: any[];
    edges: any[];
}

export function SaveDialog({ isOpen, onClose, onSave, nodes, edges }: SaveDialogProps) {
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");
    const [tags, setTags] = useState<string[]>([]);
    const [tagInput, setTagInput] = useState("");
    const [overwrite, setOverwrite] = useState(false);
    const [existingGraphs, setExistingGraphs] = useState<GraphMetadata[]>([]);
    const [checking, setChecking] = useState(false);

    const portalTarget =
        typeof window !== "undefined" ? document.getElementById("modal-root") : null;

    useEffect(() => {
        if (!isOpen) return;

        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };

        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [isOpen, onClose]);

    useEffect(() => {
        if (!isOpen) return;

        setChecking(true);
        fetch("http://localhost:8000/graphs/list")
            .then((res) => res.json())
            .then((data) => setExistingGraphs(data.graphs ?? []))
            .catch(console.error)
            .finally(() => setChecking(false));
    }, [isOpen]);

    const nameExists = useMemo(() => {
        const trimmed = name.trim().toLowerCase();
        if (!trimmed) return false;
        return existingGraphs.some((g) => g.name.toLowerCase() === trimmed);
    }, [existingGraphs, name]);

    const handleAddTag = () => {
        const t = tagInput.trim();
        if (!t) return;
        if (tags.includes(t)) return;
        setTags([...tags, t]);
        setTagInput("");
    };

    const handleRemoveTag = (tag: string) => {
        setTags(tags.filter((t) => t !== tag));
    };

    const resetForm = () => {
        setName("");
        setDescription("");
        setTags([]);
        setTagInput("");
        setOverwrite(false);
    };

    const handleSave = () => {
        const trimmedName = name.trim();
        if (!trimmedName) {
            alert("Please enter a graph name");
            return;
        }

        const exists = existingGraphs.some(
            (g) => g.name.toLowerCase() === trimmedName.toLowerCase()
        );

        if (exists && !overwrite) {
            const ok = confirm(`A graph named "${trimmedName}" already exists. Overwrite it?`);
            if (!ok) return;
            setOverwrite(true);
            onSave(trimmedName, description.trim(), tags, true);
            resetForm();
            return;
        }

        onSave(trimmedName, description.trim(), tags, overwrite || exists);
        resetForm();
    };

    if (!isOpen || !portalTarget) return null;

    return createPortal(
        <div
            className="bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]"
            style={{ position: "fixed", inset: 0 }}
            role="dialog"
            aria-modal="true"
            onMouseDown={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div className="bg-slate-900/95 border border-slate-700 rounded-2xl shadow-2xl w-[min(92vw,34rem)] overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-5 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500">
                            <Save size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-white">Save template</h2>
                            <p className="text-xs text-slate-400 mt-0.5 flex items-center gap-3">
                                <span className="inline-flex items-center gap-1">
                                    <Hash size={12} /> {nodes.length} nodes
                                </span>
                                <span className="inline-flex items-center gap-1">
                                    <Link2 size={12} /> {edges.length} connections
                                </span>
                            </p>
                        </div>
                    </div>

                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                        aria-label="Close"
                    >
                        <X size={18} className="text-slate-400" />
                    </button>
                </div>

                {/* Body */}
                <div className="px-6 py-5 space-y-4">
                    {/* Name */}
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                            Graph name <span className="text-slate-500">*</span>
                        </label>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => {
                                setName(e.target.value);
                                setOverwrite(false);
                            }}
                            placeholder="e.g., Concrete Heat Transfer Model"
                            className={`w-full px-4 py-2.5 bg-slate-800 border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 transition-all
                ${nameExists ? "border-amber-500/60 focus:ring-amber-500/30" : "border-slate-700 focus:ring-blue-500/50"}
              `}
                            autoFocus
                        />

                        <div className="mt-2 text-xs">
                            {checking ? (
                                <span className="text-slate-500">Checking existing names…</span>
                            ) : nameExists ? (
                                <span className="text-amber-300">
                                    Name already exists. Saving will overwrite if confirmed.
                                </span>
                            ) : (
                                <span className="text-slate-500">Pick a unique name to avoid overwriting.</span>
                            )}
                        </div>
                    </div>

                    {/* Description */}
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

                    {/* Tags */}
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">Tags</label>

                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={tagInput}
                                onChange={(e) => setTagInput(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        e.preventDefault();
                                        handleAddTag();
                                    }
                                }}
                                placeholder="Add a tag…"
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
                            <div className="flex flex-wrap gap-2 mt-3">
                                {tags.map((tag) => (
                                    <span
                                        key={tag}
                                        className="inline-flex items-center gap-1.5 px-3 py-1 bg-blue-500/20 border border-blue-500/30 text-blue-300 text-sm rounded-full"
                                    >
                                        <Tag size={12} />
                                        {tag}
                                        <button
                                            onClick={() => handleRemoveTag(tag)}
                                            className="hover:text-blue-100"
                                            aria-label={`Remove tag ${tag}`}
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
                <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
                    >
                        Cancel
                    </button>

                    <button
                        onClick={handleSave}
                        className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-semibold rounded-lg transition-all hover:scale-[1.02] active:scale-[0.99]"
                    >
                        Save
                    </button>
                </div>
            </div>
        </div>,
        portalTarget
    );
}
