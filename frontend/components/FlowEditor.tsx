"use client";

import React, { useState } from 'react';
import axios from 'axios';
import { Play, Save, FolderOpen, Square } from 'lucide-react';
import { SaveDialog } from './graph_management/SaveDialog';
import { LoadDialog } from './graph_management/LoadDialog';
import Sidebar from './Sidebar';
import FlowCanvas from './FlowCanvas';
import { useFlowEditor } from '@/hooks/useFlowEditor';
import CustomNode from './nodes/CustomNode';
import LossCurveNode from './nodes/LossCurveNode';
import ConfigNode from './nodes/config/ConfigNode';
import SensorVisNode from './nodes/SensorVisNode';
import RunSelectorNode from './nodes/RunChooseNode';
import MonitoringDashboard from './MonitoringDashboard';

const nodeTypes = {
    default: CustomNode,
    loss_curve: LossCurveNode,
    config: ConfigNode,
    sensor_vis: SensorVisNode,
    run_id_selector: RunSelectorNode
};

export default function FlowEditor() {
    const {
        nodes, edges, registry, isRunning,
        onNodesChange, onEdgesChange, onConnect,
        addNode, clearGraph, runSimulation, stopSimulation, saveGraph, loadGraph
    } = useFlowEditor();

    const [showSaveDialog, setShowSaveDialog] = useState(false);
    const [showLoadDialog, setShowLoadDialog] = useState(false);
    const [isStopping, setIsStopping] = useState(false);
    const [currentFileName, setCurrentFileName] = useState('')

    const [activeTab, setActiveTab] = useState<'editor' | 'monitor'>('editor');


    const handleStop = async () => {
        setIsStopping(true);
        await stopSimulation();
        setIsStopping(false);
    };

    const handleSaveSubmit = async (name: string, description: string, tags: string[], overwrite: boolean) => {
        try {
            await saveGraph(name, description, tags, overwrite);
            setCurrentFileName(name)
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
            setCurrentFileName(res.data.graph.name)
            alert(`Loaded "${res.data.graph.name}"`);
            setShowLoadDialog(false);
        } catch (e: any) {
            alert(`Failed to load: ${e.response?.data?.detail || e.message}`);
        }
    };

    return (
        <>
            <div className="w-full h-full flex bg-slate-950 relative overflow-hidden">
                {/* TABS */}
                <div className="absolute top-6 left-6 z-50 flex bg-slate-900/90 backdrop-blur border border-slate-700 rounded-xl p-1">
                    <button
                        onClick={() => setActiveTab('editor')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'editor'
                            ? 'bg-slate-800 text-white shadow-lg'
                            : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                            }`}
                    >
                        Editor
                    </button>
                    <button
                        onClick={() => setActiveTab('monitor')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'monitor'
                            ? 'bg-emerald-500/90 text-white shadow-lg'
                            : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                            }`}
                    >
                        Monitor
                    </button>
                </div>

                {/* EDITOR TAB */}
                {activeTab === 'editor' && (
                    <>
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

                        {/* Control Panel - Improved styling */}
                        <div className="absolute top-6 right-6 z-50 flex gap-3">
                            <button
                                onClick={() => { setShowLoadDialog(false); setShowSaveDialog(true); }}
                                className="flex gap-2 items-center bg-slate-800 text-slate-100 px-5 py-2.5 rounded-lg border border-slate-600 shadow-2xl hover:bg-slate-700 hover:border-slate-500 transition-all hover:scale-105 active:scale-95 font-medium"
                            >
                                <Save size={18} /> Save
                            </button>

                            <button
                                onClick={() => { setShowSaveDialog(false); setShowLoadDialog(true); }}
                                className="flex gap-2 items-center bg-slate-800 text-slate-100 px-5 py-2.5 rounded-lg border border-slate-600 shadow-2xl hover:bg-slate-700 hover:border-slate-500 transition-all hover:scale-105 active:scale-95 font-medium"
                            >
                                <FolderOpen size={18} /> Load
                            </button>

                            {isRunning ? (
                                <button
                                    onClick={handleStop}
                                    disabled={isStopping}
                                    className={`flex gap-2 items-center px-6 py-2.5 rounded-lg shadow-2xl transition-all font-semibold ${isStopping
                                        ? 'bg-red-800 text-red-200 border border-red-700 cursor-not-allowed'
                                        : 'bg-red-600 text-white border border-red-500 hover:bg-red-500 hover:scale-105 active:scale-95 animate-pulse'
                                        }`}
                                >
                                    <Square size={18} fill="currentColor" />
                                    {isStopping ? "Stopping..." : "Stop"}
                                </button>
                            ) : (
                                <button
                                    onClick={runSimulation}
                                    className="flex gap-2 items-center bg-emerald-600 text-white px-6 py-2.5 rounded-lg border border-emerald-500 shadow-2xl shadow-emerald-900/30 hover:bg-emerald-500 hover:scale-105 active:scale-95 transition-all font-semibold"
                                >
                                    <Play size={18} fill="currentColor" />
                                    Run
                                </button>
                            )}
                        </div>
                    </>
                )}

                {/* MONITORING TAB */}
                {activeTab === 'monitor' && (
                    <div className="w-full h-full">
                        <MonitoringDashboard />
                    </div>
                )}
            </div>

            {/* Dialogs stay outside tabs */}
            {showSaveDialog && (
                <SaveDialog
                    isOpen={showSaveDialog}
                    currentGraphName={currentFileName}
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
        </>
    );
}