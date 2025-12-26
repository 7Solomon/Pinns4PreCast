import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

type EventType =
  | 'connection_established'
  | 'training_started'
  | 'training_epoch_end'
  | 'training_completed'
  | 'training_stopped'
  | 'metrics_history'
  | 'metrics_updates_since'
  | 'metrics_updated'
  | 'sensor_data_history'
  | 'sensor_data_since'
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
  reconnectInterval = 5000
}: UseMonitoringWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<MonitoringEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const metricLocalStorageKey = useMemo(() => `metrics_${runId}`, [runId]);
  const sensorLocalStorageKey = useMemo(() => `sensors_${runId}`, [runId]);

  const getCachedMetrics = (key: string): any[] => {
    try {
      const cached = localStorage.getItem(key);
      return cached ? JSON.parse(cached) : [];
    } catch {
      return [];
    }
  };

  const cacheMetrics = (metrics: any[], key: string) => {
    try {
      localStorage.setItem(key, JSON.stringify(metrics));
    } catch { }
  };

  const connect = useCallback(() => {
    if (!runId || !mountedRef.current) return;

    if (wsRef.current && [0, 1].includes(wsRef.current.readyState)) {
      console.log(`[WebSocket] Already connected/connecting, skipping`);
      return;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'Reconnecting');
      wsRef.current = null;
    }

    console.log(`[WebSocket] Connecting to run: ${runId}`);
    const ws = new WebSocket(`ws://localhost:8000/ws/monitor/${runId}`);

    ws.onopen = () => {
      console.log(`[WebSocket] Connected to run: ${runId}`);
      setIsConnected(true);

      const cachedMetrics = getCachedMetrics(metricLocalStorageKey);
      if (cachedMetrics.length > 0) {
        console.log(`[Cache] Restored ${cachedMetrics.length} metrics instantly`);
        onMetricsUpdate?.(cachedMetrics);
      }

      const lastStep = cachedMetrics[cachedMetrics.length - 1]?.step || 0;
      ws.send(JSON.stringify({
        type: 'request_history_since',
        run_id: runId,
        last_step: lastStep
      }));

      // Heartbeat
      const heartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
      (ws as any)._heartbeat = heartbeat;
    };

    ws.onmessage = (event) => {
      try {
        const data: MonitoringEvent = JSON.parse(event.data);
        setLastEvent(data);

        switch (data.type) {
          case 'metrics_history':
            console.log(`[WS] History: ${data.data.length} points`);
            cacheMetrics(data.data, metricLocalStorageKey);
            onMetricsUpdate?.(data.data);
            break;

          case 'metrics_updates_since':
            console.log(`[WS] Updates: ${data.data.length} new points`);
            onMetricsUpdate?.(data.data);
            break;

          case 'metrics_updated':
            if (data.data && data.data.metrics) {
              console.log(`[WS] Live Update (Step ${data.data.step})`);
              onMetricsUpdate?.(data.data.metrics);
            }
            break;

          case 'sensor_data_history':
            console.log(`[WS] Sensor History: ${data.data.length} points`);
            cacheMetrics(data.data, sensorLocalStorageKey);
            onSensorUpdate?.(data.data);
            break;

          case 'sensor_data_since':
            console.log(`[WS] Sensor Updates: ${data.data.length} points`);
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
      console.error('[WebSocket] Error details:', {
        url: ws.url,
        readyState: ws.readyState,
        error
      });
    };

    ws.onclose = (event) => {
      console.log(`[WebSocket] Disconnected from run: ${runId}`, event.code, event.reason);
      setIsConnected(false);

      if ((ws as any)._heartbeat) {
        clearInterval((ws as any)._heartbeat);
      }

      if (autoReconnect && mountedRef.current &&
        event.code !== 1000 &&
        wsRef.current === ws &&
        runId === ws.url.split('/').pop()) {
        console.log(`[WebSocket] Scheduling reconnect...`);
        reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
      }
    };

    wsRef.current = ws;
  }, [runId, onMetricsUpdate, onSensorUpdate, onStatusChange, autoReconnect, reconnectInterval, metricLocalStorageKey, sensorLocalStorageKey]);

  useEffect(() => {
    if (!runId) return;
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Unmounting');
        wsRef.current = null;
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
