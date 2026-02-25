'use client';

import { useWebSocket } from './useWebSocket';

export function useEvents() {
  const { isConnected, lastEvent, events } = useWebSocket('events');
  
  return {
    connectionStatus: isConnected ? 'connected' : 'disconnected',
    lastEvent,
    events,
  };
}
