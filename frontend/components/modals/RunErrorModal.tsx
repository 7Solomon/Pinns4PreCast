import React from 'react';

interface ErrorModalProps {
    isOpen: boolean;
    onClose: () => void;
    errorTitle: string;
    errorMessage: string;
    nodeId?: string;
}

export const GlobalErrorModal: React.FC<ErrorModalProps> = ({
    isOpen, onClose, errorTitle, errorMessage, nodeId
}) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md border-l-4 border-red-500">

                {/* Header */}
                <div className="flex justify-between items-start mb-4">
                    <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                        ⚠️ {errorTitle || "Execution Failed"}
                    </h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">
                        &times;
                    </button>
                </div>

                {/* Content */}
                <div className="mb-6">
                    <p className="text-gray-600 mb-2">
                        An error stopped the training pipeline:
                    </p>
                    <div className="bg-red-50 p-3 rounded text-sm font-mono text-red-800 break-words whitespace-pre-wrap max-h-40 overflow-y-auto">
                        {errorMessage}
                    </div>
                    {nodeId && (
                        <p className="text-xs text-gray-400 mt-2">
                            Source Node ID: <span className="font-mono">{nodeId}</span>
                        </p>
                    )}
                </div>

                {/* Footer */}
                <div className="flex justify-end gap-2">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors"
                    >
                        Close
                    </button>
                    <button
                        onClick={() => {
                            onClose();
                        }}
                        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                    >
                        Inspect Node
                    </button>
                </div>
            </div>
        </div>
    );
};
