'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Event } from '@/lib/types';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export function useWebSocket(channel: string = 'events') {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<Event | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    const ws = new WebSocket(`${WS_URL}/ws/${channel}`);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log(`WebSocket connected to ${channel}`);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') return;
        
        setLastEvent(data);
        setEvents((prev) => [data, ...prev].slice(0, 100));
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log(`WebSocket disconnected from ${channel}`);
      
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    wsRef.current = ws;
  }, [channel]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const pause = useCallback((sessionId: string) => {
    send({ type: 'pause', session_id: sessionId });
  }, [send]);

  const resume = useCallback((sessionId: string, instruction?: string) => {
    send({ type: 'resume', session_id: sessionId, instruction });
  }, [send]);

  const veto = useCallback((sessionId: string) => {
    send({ type: 'veto', session_id: sessionId });
  }, [send]);

  useEffect(() => {
    connect();
    
    // Keep alive
    const interval = setInterval(() => {
      send({ type: 'ping' });
    }, 30000);
    
    return () => {
      clearInterval(interval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect, send]);

  return {
    isConnected,
    lastEvent,
    events,
    send,
    pause,
    resume,
    veto,
  };
}
