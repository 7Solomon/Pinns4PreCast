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
                defaultEdgeOptions={{ style: { stroke: '#475569', strokeWidth: 2 }, animated: true }}
                minZoom={0.1}
                maxZoom={2}
            >
                <Background color="#334155" variant={BackgroundVariant.Dots} gap={24} size={1.5} className="opacity-40" />
                <Controls className="!bg-slate-800/90 !border-slate-700 !fill-slate-200" />
                <MiniMap
                    className="!bg-slate-800/90 !border-slate-700"
                    nodeColor={() => '#475569'}
                    maskColor="rgba(15, 23, 42, 0.6)"
                />
            </ReactFlow>
        </div>
    );
}
