import { useState, useEffect, useCallback } from 'react';
import {
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    Node,
    Edge
} from 'reactflow';
import axios from 'axios';
import { NodeData } from '@/components/nodes/CustomNode';

export const useFlowEditor = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [registry, setRegistry] = useState<Record<string, NodeData>>({});
    const [isRunning, setIsRunning] = useState(false);

    // Load Registry
    useEffect(() => {
        axios.get<NodeData[]>('http://localhost:8000/registry')
            .then(res => {
                const reg: Record<string, NodeData> = {};
                res.data.forEach((n: any) => {
                    reg[n.type] = n;
                });
                console.log('Registry loaded with types:', Object.keys(reg));
                setRegistry(reg);
            })
            .catch(err => {
                console.error('Failed to load registry:', err);
            });
    }, []);

    const addNode = useCallback((type: string) => {
        const nodeDef = registry[type];
        if (!nodeDef) return;

        let nodeType = 'custom';
        if (type === 'live_training_monitor') {
            nodeType = 'monitor';
        }

        const newNode: Node = {
            id: `${type}-${Date.now()}`,
            type: nodeType,
            position: {
                x: Math.random() * 400 + 200,
                y: Math.random() * 400 + 100
            },
            data: { ...nodeDef, type }
        };
        setNodes((nds) => nds.concat(newNode));
    }, [registry, setNodes]);

    const onConnect = useCallback((params: Connection) => {
        setEdges((eds) => addEdge(params, eds));
    }, [setEdges]);

    const clearGraph = useCallback(() => {
        if (confirm("Clear all nodes and connections?")) {
            setNodes([]);
            setEdges([]);
        }
    }, [setNodes, setEdges]);

    // --- API ACTIONS ---


    const runSimulation = async () => {
        setIsRunning(true);
        const payload = {
            nodes: nodes.map(n => ({
                id: n.id,
                type: (n.data as any).type,
                config: n.data.config || {},
                position: n.position
            })),
            connections: edges.map(e => ({
                source_node: e.source,
                source_port: e.sourceHandle,
                target_node: e.target,
                target_port: e.targetHandle
            })),
            target_node_id: nodes.find(n => (n.data as any).category === 'Training')?.id || "unknown"
        };
        console.log("payload")
        console.log(payload)

        try {
            const res = await axios.post('http://localhost:8000/execute', payload);

            // Extract run_id from response
            const runId = res.data.run_id;
            const widgets = res.data.widgets || [];

            // Update visualization nodes with the run_id
            if (runId) {
                setNodes(nds => nds.map(node => {
                    // Find loss_curve nodes and inject the run_id
                    if ((node.data as any).type === 'loss_curve') {
                        return {
                            ...node,
                            data: {
                                ...node.data,
                                config: {
                                    ...(node.data.config || {}),
                                    run_id: runId
                                }
                            }
                        };
                    }
                    return node;
                }));
            }

            alert(`Success: ${res.data.message}\nRun ID: ${runId}`);
        } catch (e: any) {
            alert(`Error: ${e.response?.data?.detail || e.message}`);
        } finally {
            setIsRunning(false);
        }
    };

    const saveGraph = async (name: string, description: string, tags: string[], overwrite: boolean) => {
        const payload = {
            name,
            description,
            tags,
            nodes: nodes.map(n => ({
                id: n.id,
                type: (n.data as any).type,
                config: n.data.config || {},
                position: n.position
            })),
            connections: edges.map(e => ({
                source_node: e.source,
                source_port: e.sourceHandle,
                target_node: e.target,
                target_port: e.targetHandle
            })),
            overwrite
        };

        const res = await axios.post('http://localhost:8000/graphs/save', payload);
        return res.data;
    };

    const loadGraph = (graphData: any) => {
        // Clear current graph
        setNodes([]);
        setEdges([]);
        console.log(graphData)
        // Load nodes with validation against registry
        const loadedNodes = graphData.nodes.map((n: any) => {
            const nodeDef = registry[n.type];

            if (!nodeDef) {
                console.error(`Missing node type "${n.type}" in registry. Available:`, Object.keys(registry));
                // Return a placeholder node
                return {
                    id: n.id,
                    type: 'custom',
                    position: n.position,
                    data: {
                        label: `[Missing: ${n.type}]`,
                        category: 'Unknown',
                        inputs: {},
                        outputs: {},
                        type: n.type,
                        config: n.config
                    }
                };
            }

            let nodeType = 'custom';
            if (n.type === 'loss_curve') {
                nodeType = 'loss_curve';
            }

            return {
                id: n.id,
                type: nodeType,
                position: n.position,
                data: { ...nodeDef, type: n.type, config: n.config }
            };
        });

        const loadedEdges = graphData.connections.map((c: any, idx: number) => ({
            id: `e${idx}`,
            source: c.source_node,
            sourceHandle: c.source_port,
            target: c.target_node,
            targetHandle: c.target_port
        }));

        setNodes(loadedNodes);
        setEdges(loadedEdges);
    };

    return {
        nodes,
        edges,
        registry,
        isRunning,
        onNodesChange,
        onEdgesChange,
        onConnect,
        addNode,
        clearGraph,
        runSimulation,
        saveGraph,
        loadGraph,
        setNodes, // exposed in case you need direct access
        setEdges
    };
};
