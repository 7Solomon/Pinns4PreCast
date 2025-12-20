import React from 'react';
import ReactFlow, { Background, Controls, MiniMap, BackgroundVariant, Node, Edge, Connection, NodeTypes } from 'reactflow';
import 'reactflow/dist/style.css';

interface FlowCanvasProps {
    nodes: Node[];
    edges: Edge[];
    onNodesChange: any;
    onEdgesChange: any;
    onConnect: (connection: Connection) => void;
    nodeTypes: NodeTypes;
}

export default function FlowCanvas({
    nodes, edges, onNodesChange, onEdgesChange, onConnect, nodeTypes
}: FlowCanvasProps) {

    // Hide the React Flow attribution watermark
    const proOptions = { hideAttribution: true };

    return (
        <div className="flex-1 relative h-full bg-slate-950">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                deleteKeyCode={['Backspace', 'Delete']}
                nodeTypes={nodeTypes}
                proOptions={proOptions}
                fitView
                defaultEdgeOptions={{
                    style: { stroke: '#60a5fa', strokeWidth: 2 },
                    animated: true
                }}
                minZoom={0.1}
                maxZoom={2}
            >
                <Background
                    color="#334155"
                    variant={BackgroundVariant.Dots}
                    gap={24}
                    size={1.5}
                    className="opacity-30"
                />
                <Controls
                    className="!bg-slate-800 !border-slate-600 !fill-slate-200 shadow-xl"
                    style={{ border: '1px solid rgb(71, 85, 105)' }}
                />
                <MiniMap
                    className="!bg-slate-800 !border-slate-600 shadow-xl"
                    style={{ border: '1px solid rgb(71, 85, 105)' }}
                    nodeColor={() => '#475569'}
                    maskColor="rgba(2, 6, 23, 0.8)"
                />
            </ReactFlow>
        </div>
    );
}