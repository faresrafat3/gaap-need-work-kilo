'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { budgetApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { Progress } from '@/components/common/Progress';
import { EmptyState } from '@/components/common/EmptyState';
import { DollarSign, TrendingUp, TrendingDown, AlertTriangle, Download, Settings, RefreshCw } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';

export default function BudgetPage() {
  const queryClient = useQueryClient();
  const [selectedPeriod, setSelectedPeriod] = useState('month');

  const { data: budget, isLoading: budgetLoading } = useQuery({
    queryKey: ['budget'],
    queryFn: () => budgetApi.get(),
  });

  const { data: usage, isLoading: usageLoading } = useQuery({
    queryKey: ['budget-usage', selectedPeriod],
    queryFn: () => budgetApi.getUsage(selectedPeriod),
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['budget-alerts'],
    queryFn: () => budgetApi.getAlerts(),
    refetchInterval: 30000,
  });

  const updateLimitsMutation = useMutation({
    mutationFn: (limits: any) => budgetApi.updateLimits(limits),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['budget'] });
    },
  });

  const getUsageColor = (percentage: number) => {
    if (percentage < 50) return 'text-success';
    if (percentage < 80) return 'text-warning';
    return 'text-error';
  };

  const getProgressColor = (percentage: number) => {
    if (percentage < 50) return 'bg-success';
    if (percentage < 80) return 'bg-warning';
    return 'bg-error';
  };

  if (budgetLoading || usageLoading || alertsLoading) {
    return (
      <div className="flex h-screen bg-cyber-dark">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin w-8 h-8 border-2 border-layer1 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }

  const monthlyUsage = budget?.data?.monthly_used || 0;
  const monthlyLimit = budget?.data?.monthly_limit || 1;
  const monthlyPercentage = (monthlyUsage / monthlyLimit) * 100;

  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Budget" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-success">
                  ${monthlyLimit.toFixed(2)}
                </div>
                <div className="text-sm text-gray-400 mt-1">Monthly Limit</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className={`text-3xl font-bold ${getUsageColor(monthlyPercentage)}`}>
                  ${monthlyUsage.toFixed(2)}
                </div>
                <div className="text-sm text-gray-400 mt-1">Used This Month</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold">
                  ${(monthlyLimit - monthlyUsage).toFixed(2)}
                </div>
                <div className="text-sm text-gray-400 mt-1">Remaining</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className={`text-3xl font-bold ${getUsageColor(monthlyPercentage)}`}>
                  {monthlyPercentage.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-400 mt-1">Budget Used</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Budget Overview</h3>
                  <div className="flex gap-2">
                    <select
                      value={selectedPeriod}
                      onChange={(e) => setSelectedPeriod(e.target.value)}
                      className="bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-1 text-sm"
                    >
                      <option value="day">Today</option>
                      <option value="week">This Week</option>
                      <option value="month">This Month</option>
                    </select>
                  </div>
                </div>
              </CardHeader>
              <CardBody>
                <div className="space-y-6">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Monthly Budget</span>
                      <span className={`text-sm font-medium ${getUsageColor(monthlyPercentage)}`}>
                        {monthlyPercentage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 bg-cyber-dark rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getProgressColor(monthlyPercentage)} transition-all`}
                        style={{ width: `${Math.min(monthlyPercentage, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between mt-1 text-xs text-gray-500">
                      <span>${monthlyUsage.toFixed(2)} used</span>
                      <span>${monthlyLimit.toFixed(2)} limit</span>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Daily Budget</span>
                      <span className="text-sm font-medium">
                        {((budget?.data?.daily_used / budget?.data?.daily_limit) * 100 || 0).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={(budget?.data?.daily_used / budget?.data?.daily_limit) * 100 || 0} size="sm" />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-400">Per Task Limit</span>
                      <span className="text-sm font-medium">
                        {((budget?.data?.per_task_used / budget?.data?.per_task_limit) * 100 || 0).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={(budget?.data?.per_task_used / budget?.data?.per_task_limit) * 100 || 0} size="sm" />
                  </div>
                </div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Active Alerts</h3>
              </CardHeader>
              <CardBody>
                {alerts?.data?.length > 0 ? (
                  <div className="space-y-3">
                    {alerts.data.map((alert: any, i: number) => (
                      <div
                        key={i}
                        className={`p-3 rounded-lg border ${
                          alert.level === 'critical'
                            ? 'bg-error/10 border-error/30'
                            : alert.level === 'warning'
                            ? 'bg-warning/10 border-warning/30'
                            : 'bg-layer3/10 border-layer3/30'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                            alert.level === 'critical' ? 'text-error' : 'text-warning'
                          }`} />
                          <div>
                            <div className="text-sm font-medium">{alert.message}</div>
                            <div className="text-xs text-gray-500 mt-1">
                              {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No active alerts"
                    description="You're within budget limits"
                  />
                )}
              </CardBody>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Usage by Provider</h3>
                <Button variant="ghost" size="sm" leftIcon={<Download className="w-4 h-4" />}>
                  Export
                </Button>
              </div>
            </CardHeader>
            <CardBody>
              {usage?.data?.providers?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Provider</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Requests</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Tokens</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Cost</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">% of Total</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Trend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {usage.data.providers.map((provider: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm font-medium">{provider.name}</td>
                          <td className="py-3 text-sm">{provider.requests.toLocaleString()}</td>
                          <td className="py-3 text-sm">{provider.tokens.toLocaleString()}</td>
                          <td className="py-3 text-sm font-mono">${provider.cost.toFixed(4)}</td>
                          <td className="py-3">
                            <Badge variant="default">{provider.percentage.toFixed(1)}%</Badge>
                          </td>
                          <td className="py-3">
                            {provider.trend > 0 ? (
                              <div className="flex items-center gap-1 text-error">
                                <TrendingUp className="w-4 h-4" />
                                <span className="text-sm">+{provider.trend.toFixed(1)}%</span>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1 text-success">
                                <TrendingDown className="w-4 h-4" />
                                <span className="text-sm">{provider.trend.toFixed(1)}%</span>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No usage data"
                  description="Usage statistics will appear here after API calls"
                />
              )}
            </CardBody>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <h3 className="font-semibold">Recent Transactions</h3>
            </CardHeader>
            <CardBody>
              {usage?.data?.transactions?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Time</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Provider</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Model</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Tokens</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {usage.data.transactions.map((tx: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm text-gray-400">
                            {formatDistanceToNow(new Date(tx.timestamp), { addSuffix: true })}
                          </td>
                          <td className="py-3 text-sm">{tx.provider}</td>
                          <td className="py-3 text-sm font-mono">{tx.model}</td>
                          <td className="py-3 text-sm">{tx.tokens.toLocaleString()}</td>
                          <td className="py-3 text-sm font-mono">${tx.cost.toFixed(6)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No transactions"
                  description="Recent API transactions will appear here"
                />
              )}
            </CardBody>
          </Card>
        </main>
      </div>
    </div>
  );
}
