"use client";

import React, { useState } from 'react';
import axios from 'axios';
import { Play, Save, FolderOpen } from 'lucide-react';
import { SaveDialog } from './graph_management/SaveDialog';
import { LoadDialog } from './graph_management/LoadDialog';
import Sidebar from './Sidebar';
import FlowCanvas from './FlowCanvas';
import { useFlowEditor } from '@/hooks/useFlowEditor';
import CustomNode from './nodes/CustomNode';
import LossCurveNode from './nodes/LossCurveNode';

const nodeTypes = {
    custom: CustomNode,
    loss_curve: LossCurveNode
};

export default function FlowEditor() {
    const {
        nodes, edges, registry, isRunning,
        onNodesChange, onEdgesChange, onConnect,
        addNode, clearGraph, runSimulation, stopSimulation, saveGraph, loadGraph
    } = useFlowEditor();

    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [showLoadDialog, setShowLoadDialog] = useState(false);
    const [isStopping, setIsStopping] = useState(false)

    const handleStop = async () => {
        setIsStopping(true);
        await stopSimulation();
        setIsStopping(false);
    }


    const handleSaveSubmit = async (name: string, description: string, tags: string[], overwrite: boolean) => {
        try {
            await saveGraph(name, description, tags, overwrite);
            alert(`Graph saved as "${name}"`);
            setShowSaveDialog(false);
        } catch (e: any) {
            alert(`Failed to save: ${e.response?.data?.detail || e.message}`);
        }
    };

    const handleLoadSubmit = async (filename: string) => {
        try {
            if (Object.keys(registry).length === 0) {
                alert('Registry not loaded yet.');
                return;
            }
            const res = await axios.get(`http://localhost:8000/graphs/load/${filename}`);
            loadGraph(res.data.graph);
            alert(`Loaded "${res.data.graph.name}"`);
            setShowLoadDialog(false);
        } catch (e: any) {
            alert(`Failed to load: ${e.response?.data?.detail || e.message}`);
        }
    };

    return (
        <>
            {/* Main Application UI */}
            <div className="w-full h-full flex bg-slate-950 relative isolate overflow-hidden">

                <Sidebar
                    registry={registry}
                    onAddNode={addNode}
                    onClear={clearGraph}
                    stats={{ nodes: nodes.length, edges: edges.length }}
                />

                <FlowCanvas
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                />

                {/* --- CONTROL PANEL OVERLAY --- */}
                <div className="absolute top-4 right-4 z-50 flex gap-3 pointer-events-auto">
                    <button
                        onClick={() => { setShowLoadDialog(false); setShowSaveDialog(true) }}
                        className="flex gap-2 items-center bg-slate-800 text-slate-200 px-4 py-2 rounded-lg border border-slate-700 shadow-xl hover:bg-slate-700 transition-all hover:scale-105"
                    >
                        <Save size={16} /> Save
                    </button>

                    <button
                        onClick={() => { setShowSaveDialog(false); setShowLoadDialog(true) }}
                        className="flex gap-2 items-center bg-slate-800 text-slate-200 px-4 py-2 rounded-lg border border-slate-700 shadow-xl hover:bg-slate-700 transition-all hover:scale-105"
                    >
                        <FolderOpen size={16} /> Load
                    </button>

                    {isRunning ? (
                        <button
                            onClick={handleStop}
                            disabled={isStopping} // Disable while stopping
                            className={`btn-red ${isStopping ? 'opacity-50' : 'animate-pulse'}`}
                        >
                            {isStopping ? "Stopping..." : "Stop Training"}
                        </button>
                    ) : (<button onClick={runSimulation} className="btn-green">
                        Run Training
                    </button>)}


                </div>
            </div>
            {showSaveDialog && (
                <SaveDialog
                    isOpen={showSaveDialog}
                    onClose={() => setShowSaveDialog(false)}
                    onSave={handleSaveSubmit}
                    nodes={nodes}
                    edges={edges}
                />
            )}

            {showLoadDialog && (
                <LoadDialog
                    isOpen={showLoadDialog}
                    onClose={() => setShowLoadDialog(false)}
                    onLoad={handleLoadSubmit}
                />
            )}
            <style jsx global>{`
                .custom-scrollbar::-webkit-scrollbar { width: 6px; }
                .custom-scrollbar::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.3); }
                .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(71, 85, 105, 0.5); border-radius: 3px; }
            `}</style>
        </>
    );
}
