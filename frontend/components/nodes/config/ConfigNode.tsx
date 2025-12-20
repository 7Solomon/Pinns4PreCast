import React, { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom'; // <--- CRITICAL IMPORT
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow';
import { Settings, X, Maximize2 } from 'lucide-react';
import { ConfigForm } from './ConfigForm';

// --- PORTAL MODAL COMPONENT ---
const Modal = ({ isOpen, onClose, title, children }: any) => {
    // Prevent scrolling body when modal is open
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => { document.body.style.overflow = 'unset'; };
    }, [isOpen]);

    if (!isOpen) return null;

    // We render this directly into the document.body to escape the Zoom/Pan transform
    return createPortal(
        <div
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
            onClick={onClose}
        >
            <div
                className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200"
                onClick={e => e.stopPropagation()} // Stop click from closing modal
            >
                {/* Header */}
                <div className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-950">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
                            <Settings size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-white leading-none">{title}</h3>
                            <p className="text-xs text-slate-400 mt-1">Update configuration parameters</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Body - using 'nodrag' is good practice even in portals just in case */}
                <div className="p-6 overflow-y-auto custom-scrollbar flex-1 bg-slate-900 nodrag cursor-default">
                    {children}
                </div>
            </div>
        </div>,
        document.body // Target the body
    );
};

// --- MAIN NODE COMPONENT ---
export default function ConfigNode({ data, id }: NodeProps) {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const { setNodes } = useReactFlow();

    const handleSave = useCallback((newConfig: any) => {
        setNodes((nodes) => nodes.map((node) => {
            if (node.id === id) {
                return {
                    ...node,
                    data: { ...node.data, config: newConfig }
                };
            }
            return node;
        }));
        setIsModalOpen(false);
    }, [id, setNodes]);

    return (
        // This container stays SMALL and CLEAN in your graph
        <div className="relative min-w-[240px] bg-slate-950 border border-slate-800 rounded-xl shadow-xl transition-all hover:border-blue-500/50 group isolate z-10">

            {/* Header Area */}
            <div className="p-3 flex items-center justify-between gap-3 bg-slate-950 rounded-t-xl">
                <div className="flex items-center gap-3 flex-1">
                    {/* Icon Container */}
                    <div className="w-10 h-10 flex-none flex items-center justify-center bg-slate-900 border border-slate-800 rounded-lg text-blue-400 group-hover:text-blue-300 transition-colors">
                        <Settings size={20} />
                    </div>

                    {/* Title & Type */}
                    <div className="min-w-0 flex-1">
                        <div className="text-sm font-bold text-slate-200 leading-snug truncate" title={data.label}>
                            {data.label}
                        </div>
                        <div className="text-[10px] uppercase tracking-wider font-medium text-slate-500 mt-0.5">
                            Configuration
                        </div>
                    </div>
                </div>

                {/* Edit Button - Opens Portal Modal */}
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        setIsModalOpen(true);
                    }}
                    className="flex-none p-2 bg-slate-900 border border-slate-800 rounded-lg text-slate-400 hover:text-white hover:border-slate-600 hover:bg-slate-800 transition-all"
                    title="Edit Settings"
                >
                    <Maximize2 size={16} />
                </button>
            </div>

            {/* Config Summary */}
            <div className="px-3 pb-3 pt-0">
                {data.config && Object.keys(data.config).length > 0 ? (
                    <div className="text-[10px] text-slate-500 font-mono mt-2 px-2 py-1 bg-slate-900 rounded border border-slate-800 truncate">
                        {JSON.stringify(data.config).slice(0, 30)}...
                    </div>
                ) : (
                    <div className="text-[10px] text-slate-600 italic mt-2 px-2">Default Config</div>
                )}
            </div>

            {/* Port */}
            <Handle
                type="source"
                position={Position.Right}
                id="config"
                className="!bg-blue-500 !w-3 !h-3 !border-2 !border-slate-950 transition-transform hover:scale-125 z-50"
            />

            {/* Modal - Now rendered via Portal */}
            <Modal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                title={data.label}
            >
                {data.default_config ? (
                    <ConfigForm
                        schema={data.default_config}
                        initialData={data.config || {}}
                        onSave={handleSave}
                    />
                ) : (
                    <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                        <p>No configuration schema available.</p>
                    </div>
                )}
            </Modal>
        </div>
    );
}
