'use client';

import { useEvents } from '@/hooks/useEvents';
import { formatDistanceToNow } from 'date-fns';
import { Zap } from 'lucide-react';

const eventTypeColors: Record<string, string> = {
  CONFIG_CHANGED: 'border-layer1',
  OODA_PHASE: 'border-layer2',
  HEALING_EVENT: 'border-healing',
  RESEARCH_PROGRESS: 'border-layer3',
  BUDGET_ALERT: 'border-warning',
  PROVIDER_STATUS: 'border-layer2',
  SESSION_UPDATE: 'border-layer1',
};

export function RecentEvents() {
  const { events, connectionStatus } = useEvents();

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-400">Recent Events</h3>
        <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-success' : 'bg-warning'}`} />
      </div>
      
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {events.slice(0, 10).map((event) => (
          <div
            key={event.event_id}
            className={`p-2 bg-cyber-dark/50 rounded border-l-2 ${eventTypeColors[event.type] || 'border-gray-500'}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="w-3 h-3 text-cyber-primary" />
                <span className="text-xs font-medium">{event.type}</span>
              </div>
              <span className="text-xs text-gray-500">
                {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
              </span>
            </div>
            {event.source && (
              <div className="text-xs text-gray-500 mt-1">
                Source: {event.source}
              </div>
            )}
          </div>
        ))}
        
        {events.length === 0 && (
          <div className="text-center text-gray-500 py-4">
            Waiting for events...
          </div>
        )}
      </div>
    </div>
  );
}
