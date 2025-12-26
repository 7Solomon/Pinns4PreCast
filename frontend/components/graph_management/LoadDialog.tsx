"use client";

import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { X, FolderOpen, Trash2, Copy, Calendar, Hash, Search } from "lucide-react";
import { GraphMetadata } from "./GraphMetadata";

interface LoadDialogProps {
    isOpen: boolean;
    onClose: () => void;
    onLoad: (filename: string) => void;
}

export function LoadDialog({ isOpen, onClose, onLoad }: LoadDialogProps) {
    const [graphs, setGraphs] = useState<GraphMetadata[]>([]);
    const [loading, setLoading] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [selectedTag, setSelectedTag] = useState<string | null>(null);

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
        if (isOpen) fetchGraphs();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    const fetchGraphs = async () => {
        setLoading(true);
        try {
            const res = await fetch("http://localhost:8000/graphs/list");
            const data = await res.json();
            setGraphs(data.graphs ?? []);
        } catch (err) {
            console.error("Failed to fetch graphs:", err);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (filename: string, graphName: string) => {
        if (!confirm(`Delete "${graphName}"? This cannot be undone.`)) return;

        try {
            await fetch(`http://localhost:8000/graphs/delete/${filename}`, {
                method: "DELETE",
            });
            fetchGraphs();
        } catch (err) {
            alert("Failed to delete graph");
            console.error(err);
        }
    };

    const handleDuplicate = async (filename: string, originalName: string) => {
        const newName = prompt(
            `Enter name for duplicate of "${originalName}":`,
            `${originalName} (Copy)`
        );
        if (!newName) return;

        try {
            const res = await fetch(
                `http://localhost:8000/graphs/duplicate/${filename}?new_name=${encodeURIComponent(
                    newName
                )}`,
                { method: "POST" }
            );

            if (!res.ok) {
                const error = await res.json().catch(() => ({}));
                alert(error.detail || "Failed to duplicate graph");
                return;
            }

            fetchGraphs();
        } catch (err) {
            alert("Failed to duplicate graph");
            console.error(err);
        }
    };

    const filteredGraphs = useMemo(() => {
        return graphs.filter((g) => {
            const matchesSearch =
                searchTerm === "" ||
                g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                g.description.toLowerCase().includes(searchTerm.toLowerCase());

            const matchesTag = !selectedTag || g.tags.includes(selectedTag);

            return matchesSearch && matchesTag;
        });
    }, [graphs, searchTerm, selectedTag]);

    const allTags = useMemo(() => [...new Set(graphs.flatMap((g) => g.tags))], [graphs]);

    if (!isOpen || !portalTarget) return null;

    return createPortal(
        <div
            // Backdrop: blocks canvas/sidebar visually and captures clicks.
            className="bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]"
            style={{ position: "fixed", inset: 0 }}
            role="dialog"
            aria-modal="true"
            onMouseDown={(e) => {
                // click backdrop closes, click inside panel doesn't
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div
                // Panel: narrower + responsive.
                className="bg-slate-900/95 border border-slate-700 rounded-2xl shadow-2xl w-[min(92vw,48rem)] max-h-[80vh] flex flex-col overflow-hidden"
            >
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-5 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-emerald-500">
                            <FolderOpen size={20} className="text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-white">Load template</h2>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {loading ? "Fetching saved graphs…" : `${graphs.length} saved ${graphs.length === 1 ? "graph" : "graphs"}`}
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

                {/* Search + tags */}
                <div className="px-6 py-4 border-b border-slate-800 space-y-3">
                    <div className="relative">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            placeholder="Search graphs…"
                            className="w-full pl-9 pr-3 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all"
                        />
                    </div>

                    {allTags.length > 0 && (
                        <div className="flex flex-wrap gap-2">
                            <button
                                onClick={() => setSelectedTag(null)}
                                className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${!selectedTag
                                    ? "bg-blue-500 text-white"
                                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                                    }`}
                            >
                                All
                            </button>
                            {allTags.map((tag) => (
                                <button
                                    key={tag}
                                    onClick={() => setSelectedTag(tag === selectedTag ? null : tag)}
                                    className={`px-3 py-1 rounded-full text-xs font-medium transition-all ${selectedTag === tag
                                        ? "bg-emerald-500 text-white"
                                        : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                                        }`}
                                >
                                    {tag}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* List */}
                <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
                    {loading ? (
                        <div className="space-y-3">
                            {Array.from({ length: 5 }).map((_, i) => (
                                <div
                                    key={i}
                                    className="bg-slate-800/40 border border-slate-700/50 rounded-xl p-4"
                                >
                                    <div className="h-4 w-1/2 bg-slate-700/60 rounded animate-pulse mb-3" />
                                    <div className="h-3 w-3/4 bg-slate-700/40 rounded animate-pulse mb-2" />
                                    <div className="h-3 w-2/3 bg-slate-700/30 rounded animate-pulse" />
                                </div>
                            ))}
                        </div>
                    ) : filteredGraphs.length === 0 ? (
                        <div className="text-center py-12 text-slate-500">
                            <FolderOpen size={44} className="mx-auto mb-3 opacity-20" />
                            <p className="text-sm">
                                {searchTerm || selectedTag ? "No matching graphs found" : "No saved graphs yet"}
                            </p>
                        </div>
                    ) : (
                        filteredGraphs.map((graph) => (
                            <div
                                key={graph.filename}
                                className="group bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 hover:border-blue-500/50 hover:bg-slate-800/60 transition-all"
                            >
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex-1 min-w-0">
                                        <h3 className="text-base font-semibold text-white mb-1 truncate">
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
                                                {graph.tags.map((tag) => (
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
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => onLoad(graph.filename)}
                                            className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
                                            title="Load"
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
                <div className="px-6 py-4 border-t border-slate-800">
                    <button
                        onClick={onClose}
                        className="w-full px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>,
        portalTarget
    );
}
