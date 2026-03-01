import { useState, useRef, useCallback, useEffect } from 'react';

const RECONNECT_DELAY = 2000;

export function useWebSocket(url) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const urlRef = useRef(url);
  useEffect(() => { urlRef.current = url; }, [url]);

  const connectRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return;
    try {
      const ws = new WebSocket(urlRef.current);
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        setConnected(true);
        ws.send(JSON.stringify({ type: 'log_subscribe' }));
      };
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer.current = setTimeout(() => {
          if (wsRef.current !== ws && connectRef.current) connectRef.current();
        }, RECONNECT_DELAY);
      };
      ws.onerror = () => setConnected(false);
      ws.onmessage = (ev) => {
        try {
          setLastMessage(JSON.parse(ev.data));
        } catch { /* ignore parse errors */ }
      };
      wsRef.current = ws;
    } catch {
      reconnectTimer.current = setTimeout(() => {
        if (connectRef.current) connectRef.current();
      }, RECONNECT_DELAY);
    }
  }, []);

  useEffect(() => { connectRef.current = connect; }, [connect]);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimer.current);
    if (wsRef.current) {
      try { wsRef.current.send(JSON.stringify({ type: 'stop' })); } catch (err) { void err; }
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const send = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  useEffect(() => () => {
    clearTimeout(reconnectTimer.current);
    if (wsRef.current) wsRef.current.close();
  }, []);

  return { connected, lastMessage, connect, disconnect, send };
}
