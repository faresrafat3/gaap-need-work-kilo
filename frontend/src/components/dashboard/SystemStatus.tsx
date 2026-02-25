'use client';

import { useQuery } from '@tanstack/react-query';
import { systemApi } from '@/lib/api';
import { Cpu, HardDrive, Activity, Clock } from 'lucide-react';

export function SystemStatus() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => systemApi.getHealth(),
    refetchInterval: 10000,
  });

  const { data: metrics } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: () => systemApi.getMetrics(),
    refetchInterval: 5000,
  });

  if (isLoading) {
    return (
      <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4 animate-pulse">
        <div className="h-6 bg-layer1/20 rounded w-1/3 mb-4" />
        <div className="h-20 bg-layer1/10 rounded" />
      </div>
    );
  }

  const statusColor = health?.data?.status === 'healthy' 
    ? 'text-success' 
    : health?.data?.status === 'degraded' 
      ? 'text-warning' 
      : 'text-error';

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">System Status</h3>
      
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-3 h-3 rounded-full ${statusColor.replace('text-', 'bg-')} animate-pulse`} />
        <span className={`font-medium capitalize ${statusColor}`}>
          {health?.data?.status || 'Unknown'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center gap-2 text-sm">
          <Cpu className="w-4 h-4 text-layer2" />
          <span className="text-gray-400">Requests:</span>
          <span className="font-mono">{metrics?.data?.requests_total || 0}</span>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <Activity className="w-4 h-4 text-layer3" />
          <span className="text-gray-400">Success:</span>
          <span className="font-mono text-success">{metrics?.data?.requests_success || 0}</span>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <HardDrive className="w-4 h-4 text-layer1" />
          <span className="text-gray-400">Memory:</span>
          <span className="font-mono">{metrics?.data?.memory_usage_mb || 0} MB</span>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-healing" />
          <span className="text-gray-400">Uptime:</span>
          <span className="font-mono">
            {Math.floor((metrics?.data?.uptime_seconds || 0) / 3600)}h
          </span>
        </div>
      </div>
    </div>
  );
}
