'use client';

import { useQuery } from '@tanstack/react-query';
import { providersApi } from '@/lib/api';
import { Server, CheckCircle, AlertCircle, XCircle } from 'lucide-react';

export function ProviderHealth() {
  const { data: providers, isLoading } = useQuery({
    queryKey: ['providers'],
    queryFn: () => providersApi.list(),
    refetchInterval: 15000,
  });

  if (isLoading) {
    return (
      <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4 animate-pulse">
        <div className="h-6 bg-layer1/20 rounded w-1/3 mb-4" />
        <div className="h-20 bg-layer1/10 rounded" />
      </div>
    );
  }

  const providerList = providers?.data || [];

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'degraded':
        return <AlertCircle className="w-4 h-4 text-warning" />;
      default:
        return <XCircle className="w-4 h-4 text-error" />;
    }
  };

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-400">Providers</h3>
        <span className="text-xs text-gray-500">{providerList.length} active</span>
      </div>
      
      <div className="space-y-2">
        {providerList.slice(0, 5).map((provider: any) => (
          <div 
            key={provider.name}
            className="flex items-center justify-between p-2 bg-cyber-dark/50 rounded"
          >
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-layer2" />
              <span className="text-sm font-medium">{provider.name}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">
                {provider.stats?.requests || 0} req
              </span>
              {getHealthIcon(provider.health)}
            </div>
          </div>
        ))}
        
        {providerList.length === 0 && (
          <div className="text-center text-gray-500 py-4">
            No providers configured
          </div>
        )}
      </div>
    </div>
  );
}
