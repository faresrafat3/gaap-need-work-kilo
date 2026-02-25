'use client';

import { useQuery } from '@tanstack/react-query';
import { budgetApi } from '@/lib/api';
import { DollarSign, TrendingDown, AlertTriangle } from 'lucide-react';

export function BudgetGauge() {
  const { data: budget, isLoading } = useQuery({
    queryKey: ['budget'],
    queryFn: () => budgetApi.get(),
    refetchInterval: 30000,
  });

  const { data: alerts } = useQuery({
    queryKey: ['budget-alerts'],
    queryFn: () => budgetApi.getAlerts(),
  });

  if (isLoading) {
    return (
      <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4 animate-pulse">
        <div className="h-6 bg-layer1/20 rounded w-1/3 mb-4" />
        <div className="h-20 bg-layer1/10 rounded" />
      </div>
    );
  }

  const percentage = budget?.data?.percentage_used || 0;
  const isWarning = percentage > 70;
  const isCritical = percentage > 90;

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-400">Budget Usage</h3>
        {alerts?.data?.length > 0 && (
          <AlertTriangle className="w-4 h-4 text-warning" />
        )}
      </div>
      
      <div className="flex items-center gap-4">
        <div className="relative w-20 h-20">
          <svg className="w-20 h-20 transform -rotate-90">
            <circle
              cx="40"
              cy="40"
              r="35"
              stroke="currentColor"
              strokeWidth="6"
              fill="none"
              className="text-layer1/30"
            />
            <circle
              cx="40"
              cy="40"
              r="35"
              stroke="currentColor"
              strokeWidth="6"
              fill="none"
              strokeDasharray={`${percentage * 2.2} 220`}
              className={isCritical ? 'text-error' : isWarning ? 'text-warning' : 'text-success'}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-lg font-bold font-mono">{percentage.toFixed(0)}%</span>
          </div>
        </div>
        
        <div className="flex-1 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Monthly</span>
            <span className="font-mono">
              ${budget?.data?.monthly_used?.toFixed(2) || 0} / ${budget?.data?.monthly_limit || 0}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Daily</span>
            <span className="font-mono">
              ${budget?.data?.daily_used?.toFixed(2) || 0} / ${budget?.data?.daily_limit || 0}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Remaining</span>
            <span className={`font-mono ${isCritical ? 'text-error' : 'text-success'}`}>
              ${budget?.data?.remaining?.toFixed(2) || 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
