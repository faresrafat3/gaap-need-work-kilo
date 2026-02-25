'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { healingApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { Switch } from '@/components/common/Input';
import { Progress } from '@/components/common/Progress';
import { EmptyState } from '@/components/common/EmptyState';
import { Activity, RefreshCw, Settings, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';

export default function HealingPage() {
  const queryClient = useQueryClient();
  const [showConfig, setShowConfig] = useState(false);

  const { data: config, isLoading: configLoading } = useQuery({
    queryKey: ['healing-config'],
    queryFn: () => healingApi.getConfig(),
  });

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['healing-history'],
    queryFn: () => healingApi.getHistory(50),
    refetchInterval: 10000,
  });

  const { data: patterns } = useQuery({
    queryKey: ['healing-patterns'],
    queryFn: () => healingApi.getPatterns(),
  });

  const resetMutation = useMutation({
    mutationFn: () => healingApi.reset(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['healing'] });
    },
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-error" />;
      default:
        return <Activity className="w-4 h-4 text-warning" />;
    }
  };

  const getLevelColor = (level: number) => {
    const colors = ['text-success', 'text-layer3', 'text-warning', 'text-error', 'text-error'];
    return colors[level - 1] || 'text-gray-400';
  };

  if (configLoading || historyLoading) {
    return (
      <div className="flex h-screen bg-cyber-dark">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin w-8 h-8 border-2 border-layer1 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Healing" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-success">
                  {config?.data?.stats?.success_count || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Successful Heals</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-error">
                  {config?.data?.stats?.failed_count || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Failed Attempts</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold">
                  Level {config?.data?.max_healing_level || 5}
                </div>
                <div className="text-sm text-gray-400 mt-1">Max Healing Level</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold">
                  {config?.data?.stats?.patterns_detected || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Patterns Detected</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Configuration</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowConfig(!showConfig)}
                  >
                    <Settings className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardBody className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Enable Learning</span>
                  <Switch checked={config?.data?.enable_learning} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Max Healing Level</span>
                  <span className="font-mono">{config?.data?.max_healing_level}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Max Retries/Level</span>
                  <span className="font-mono">{config?.data?.max_retries_per_level}</span>
                </div>
                <div>
                  <span className="text-sm text-gray-400">Exponential Backoff</span>
                  <Progress
                    value={config?.data?.exponential_backoff ? 100 : 0}
                    size="sm"
                    className="mt-2"
                  />
                </div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Detected Patterns</h3>
              </CardHeader>
              <CardBody>
                {patterns?.data?.length > 0 ? (
                  <div className="space-y-3">
                    {patterns.data.slice(0, 5).map((pattern: any, i: number) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3 bg-cyber-dark rounded-lg"
                      >
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-warning" />
                          <span className="text-sm">{pattern.type}</span>
                        </div>
                        <Badge variant="warning">{pattern.count}</Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No patterns detected"
                    description="The system will detect recurring failure patterns"
                  />
                )}
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">Quick Actions</h3>
              </CardHeader>
              <CardBody className="space-y-3">
                <Button
                  variant="secondary"
                  isFullWidth
                  onClick={() => resetMutation.mutate()}
                  isLoading={resetMutation.isPending}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Reset Statistics
                </Button>
                <Button variant="secondary" isFullWidth>
                  Export History
                </Button>
                <Button variant="secondary" isFullWidth>
                  Configure Patterns
                </Button>
              </CardBody>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader>
              <h3 className="font-semibold">Healing History</h3>
            </CardHeader>
            <CardBody>
              {history?.data?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Time</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Error Type</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Level</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Status</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.data.map((event: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm">
                            {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
                          </td>
                          <td className="py-3 text-sm font-mono">{event.error_type}</td>
                          <td className="py-3">
                            <Badge variant="default" className={getLevelColor(event.level)}>
                              L{event.level}
                            </Badge>
                          </td>
                          <td className="py-3">
                            <div className="flex items-center gap-2">
                              {getStatusIcon(event.status)}
                              <span className="text-sm capitalize">{event.status}</span>
                            </div>
                          </td>
                          <td className="py-3 text-sm font-mono">
                            {event.duration_ms}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No healing events"
                  description="Events will appear here when self-healing is triggered"
                />
              )}
            </CardBody>
          </Card>
        </main>
      </div>
    </div>
  );
}
