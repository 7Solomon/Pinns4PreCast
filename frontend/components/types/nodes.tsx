interface NodeStatus {
    status: 'pending' | 'running' | 'completed' | 'error';
    error?: string;
}