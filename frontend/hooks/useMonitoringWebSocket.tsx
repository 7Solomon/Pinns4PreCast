import { useEffect, useRef, useState, useCallback } from 'react';

type EventType =
  | 'connection_established'
  | 'training_started'
  | 'training_epoch_end'
  | 'training_completed'
  | 'training_stopped'
  | 'metrics_updated'
  | 'sensor_data_updated'
  | 'checkpoint_saved'
  | 'run_status_changed';

interface MonitoringEvent {
  type: EventType;
  run_id: string;
  data: any;
  timestamp: number;
}

interface UseMonitoringWebSocketOptions {
  runId: string | null;
  onMetricsUpdate?: (metrics: any) => void;
  onSensorUpdate?: (data: any) => void;
  onStatusChange?: (status: string) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

export function useMonitoringWebSocket({
  runId,
  onMetricsUpdate,
  onSensorUpdate,
  onStatusChange,
  autoReconnect = true,
  reconnectInterval = 3000
}: UseMonitoringWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<MonitoringEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!runId || !mountedRef.current) return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    console.log(`[WebSocket] Connecting to run: ${runId}`);

    const ws = new WebSocket(`ws://localhost:8000/ws/monitor/${runId}`);

    ws.onopen = () => {
      console.log(`[WebSocket] Connected to run: ${runId}`);
      setIsConnected(true);

      // Start heartbeat to keep connection alive
      const heartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000); // Ping every 30 seconds

      (ws as any)._heartbeat = heartbeat;
    };

    ws.onmessage = (event) => {
      try {
        const data: MonitoringEvent = JSON.parse(event.data);
        setLastEvent(data);

        // Route to appropriate callback
        switch (data.type) {
          case 'metrics_updated':
            onMetricsUpdate?.(data.data);
            break;

          case 'sensor_data_updated':
            onSensorUpdate?.(data.data);
            break;

          case 'training_completed':
          case 'training_stopped':
            onStatusChange?.(data.data.status);
            break;
        }
      } catch (error) {
        console.error('[WebSocket] Error parsing message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    ws.onclose = () => {
      console.log(`[WebSocket] Disconnected from run: ${runId}`);
      setIsConnected(false);

      // Clear heartbeat
      if ((ws as any)._heartbeat) {
        clearInterval((ws as any)._heartbeat);
      }

      // Auto-reconnect if enabled
      if (autoReconnect && mountedRef.current) {
        console.log(`[WebSocket] Reconnecting in ${reconnectInterval}ms...`);
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectInterval);
      }
    };

    wsRef.current = ws;
  }, [runId, onMetricsUpdate, onSensorUpdate, onStatusChange, autoReconnect, reconnectInterval]);

  useEffect(() => {
    mountedRef.current = true;

    if (runId) {
      connect();
    }

    return () => {
      mountedRef.current = false;

      // Cleanup
      if (wsRef.current) {
        wsRef.current.close();
      }

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [runId, connect]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  return {
    isConnected,
    lastEvent,
    disconnect,
    reconnect: connect
  };
}


