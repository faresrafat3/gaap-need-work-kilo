'use client';

import { useQuery } from '@tanstack/react-query';
import { sessionsApi } from '@/lib/api';
import { Play, Pause, CheckCircle, XCircle, Clock } from 'lucide-react';

const statusIcons: Record<string, any> = {
  running: Play,
  paused: Pause,
  completed: CheckCircle,
  failed: XCircle,
};

const statusColors: Record<string, string> = {
  running: 'text-layer3',
  paused: 'text-warning',
  completed: 'text-success',
  failed: 'text-error',
};

export function ActiveSessions() {
  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => sessionsApi.list(),
    refetchInterval: 10000,
  });

  if (isLoading) {
    return (
      <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4 animate-pulse">
        <div className="h-6 bg-layer1/20 rounded w-1/3 mb-4" />
        <div className="h-20 bg-layer1/10 rounded" />
      </div>
    );
  }

  const sessionList = sessions?.data || [];

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-400">Active Sessions</h3>
        <span className="text-xs text-gray-500">
          {sessionList.filter((s: any) => s.status === 'running').length} running
        </span>
      </div>
      
      <div className="space-y-2">
        {sessionList.slice(0, 5).map((session: any) => {
          const StatusIcon = statusIcons[session.status] || Clock;
          const statusColor = statusColors[session.status] || 'text-gray-400';
          
          return (
            <div
              key={session.id}
              className="flex items-center justify-between p-2 bg-cyber-dark/50 rounded"
            >
              <div className="flex items-center gap-2">
                <StatusIcon className={`w-4 h-4 ${statusColor}`} />
                <div>
                  <div className="text-sm font-medium capitalize">{session.type}</div>
                  <div className="text-xs text-gray-500">
                    {session.id.slice(0, 8)}...
                  </div>
                </div>
              </div>
              <div className="text-xs text-gray-500">
                {session.status}
              </div>
            </div>
          );
        })}
        
        {sessionList.length === 0 && (
          <div className="text-center text-gray-500 py-4">
            No active sessions
          </div>
        )}
      </div>
    </div>
  );
}
