import React, { useState, useCallback } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Settings, X } from 'lucide-react';
import { ConfigForm } from './ConfigForm';

const Modal = ({ isOpen, onClose, title, children }: any) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={onClose}>
            <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-[600px] max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()}>
                <div className="flex justify-between items-center p-4 border-b border-slate-700">
                    <h3 className="text-lg font-semibold text-white">{title}</h3>
                    <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
                </div>
                <div className="p-6 overflow-y-auto custom-scrollbar flex-1">
                    {children}
                </div>
            </div>
        </div>
    );
};

export default function ConfigNode({ data, id }: NodeProps) {
    const [isModalOpen, setIsModalOpen] = useState(false);

    // Local state for config, synced with data.config
    // In ReactFlow, to update data persistently, you usually use a hook from the parent,
    // but for this example, we'll mutate the data object directly or expect an update callback.
    const [config, setConfig] = useState(data.config || {});

    const handleSave = useCallback((newConfig: any) => {
        data.config = newConfig; // Direct mutation for ReactFlow state
        setConfig(newConfig);
        setIsModalOpen(false);
    }, [data]);

    return (
        <div className="relative group min-w-[180px] bg-slate-900 border border-blue-500/50 rounded-lg shadow-lg transition-all hover:border-blue-400">
            {/* Header */}
            <div className="px-4 py-2 bg-slate-800 rounded-t-lg border-b border-slate-700 flex items-center gap-2">
                <div className="p-1 bg-blue-500/20 rounded">
                    <Settings size={14} className="text-blue-400" />
                </div>
                <span className="text-sm font-medium text-slate-200">{data.label}</span>
            </div>

            {/* Body */}
            <div className="p-3">
                <div className="text-xs text-slate-400 mb-3">
                    {data.description || "Configuration Node"}
                </div>

                <button
                    onClick={() => setIsModalOpen(true)}
                    className="w-full py-1.5 px-3 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-xs text-blue-300 transition-colors flex items-center justify-center gap-2"
                >
                    <Settings size={12} /> Edit Configuration
                </button>
            </div>

            {/* Output Port */}
            <Handle type="source" position={Position.Right} id="config" className="!bg-blue-500" />

            {/* Modal */}
            <Modal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                title={`Edit ${data.label}`}
            >
                {data.default_config ? (
                    <ConfigForm
                        schema={data.default_config}
                        initialData={config}
                        onSave={handleSave}
                    />
                ) : (
                    <div className="text-red-400">No Schema definition found for this node.</div>
                )}
            </Modal>
        </div>
    );
}
