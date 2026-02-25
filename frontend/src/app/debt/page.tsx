'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { debtApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { Progress } from '@/components/common/Progress';
import { EmptyState } from '@/components/common/EmptyState';
import { Code, AlertTriangle, Clock, CheckCircle, TrendingUp, Filter, RefreshCw, Wrench } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';

export default function DebtPage() {
  const queryClient = useQueryClient();
  const [selectedPriority, setSelectedPriority] = useState<string>('all');

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['debt-summary'],
    queryFn: () => debtApi.getSummary(),
  });

  const { data: items, isLoading: itemsLoading } = useQuery({
    queryKey: ['debt-items', selectedPriority],
    queryFn: () => debtApi.getItems(selectedPriority),
  });

  const { data: trends } = useQuery({
    queryKey: ['debt-trends'],
    queryFn: () => debtApi.getTrends(),
  });

  const resolveMutation = useMutation({
    mutationFn: (id: string) => debtApi.resolve(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['debt'] });
    },
  });

  const scanMutation = useMutation({
    mutationFn: () => debtApi.scan(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['debt'] });
    },
  });

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'text-error bg-error/20';
      case 'high':
        return 'text-error bg-error/10';
      case 'medium':
        return 'text-warning bg-warning/20';
      case 'low':
        return 'text-layer3 bg-layer3/20';
      default:
        return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'code_smell':
        return <Code className="w-4 h-4 text-warning" />;
      case 'bug':
        return <AlertTriangle className="w-4 h-4 text-error" />;
      case 'vulnerability':
        return <AlertTriangle className="w-4 h-4 text-error" />;
      case 'documentation':
        return <Code className="w-4 h-4 text-layer3" />;
      default:
        return <Code className="w-4 h-4 text-gray-400" />;
    }
  };

  if (summaryLoading || itemsLoading) {
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
        <Header title="Technical Debt" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-error">
                  {summary?.data?.total_items || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Total Items</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-warning">
                  {summary?.data?.estimated_hours?.toFixed(1) || 0}h
                </div>
                <div className="text-sm text-gray-400 mt-1">Est. Time to Fix</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-layer3">
                  {summary?.data?.debt_score || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Debt Score</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-success">
                  {summary?.data?.resolved_this_month || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Resolved This Month</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <h3 className="font-semibold">By Priority</h3>
              </CardHeader>
              <CardBody className="space-y-4">
                {summary?.data?.by_priority ? (
                  Object.entries(summary.data.by_priority).map(([priority, count]: [string, any]) => (
                    <div key={priority} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge className={getPriorityColor(priority)}>{priority}</Badge>
                      </div>
                      <span className="font-mono">{count}</span>
                    </div>
                  ))
                ) : (
                  <EmptyState title="No data" description="Priority breakdown will appear here" />
                )}
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">By Category</h3>
              </CardHeader>
              <CardBody className="space-y-4">
                {summary?.data?.by_category ? (
                  Object.entries(summary.data.by_category).map(([category, data]: [string, any]) => (
                    <div key={category} className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400 capitalize">{category.replace('_', ' ')}</span>
                        <span>{data.count}</span>
                      </div>
                      <Progress value={data.percentage || 0} size="sm" />
                    </div>
                  ))
                ) : (
                  <EmptyState title="No data" description="Category breakdown will appear here" />
                )}
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Quick Actions</h3>
                </div>
              </CardHeader>
              <CardBody className="space-y-3">
                <Button
                  variant="secondary"
                  isFullWidth
                  onClick={() => scanMutation.mutate()}
                  isLoading={scanMutation.isPending}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Run Debt Scan
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<Wrench className="w-4 h-4" />}>
                  Auto-Fix Safe Items
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<TrendingUp className="w-4 h-4" />}>
                  View Trends
                </Button>
              </CardBody>
            </Card>
          </div>

          {trends?.data && (
            <Card className="mt-6">
              <CardHeader>
                <h3 className="font-semibold">Debt Trend (Last 30 Days)</h3>
              </CardHeader>
              <CardBody>
                <div className="h-40 flex items-end gap-1">
                  {trends.data.map((day: any, i: number) => {
                    const height = day.count > 0 ? Math.max((day.count / (summary?.data?.max_daily || 10)) * 100, 5) : 5;
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-layer3/30 rounded-t hover:bg-layer3/50 transition-all cursor-pointer group relative"
                        style={{ height: `${height}%` }}
                        title={`${day.date}: ${day.count} items`}
                      >
                        <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 bg-cyber-darker px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                          {day.count} items
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-500">
                  <span>30 days ago</span>
                  <span>Today</span>
                </div>
              </CardBody>
            </Card>
          )}

          <Card className="mt-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Debt Items</h3>
                <div className="flex items-center gap-2">
                  <Filter className="w-4 h-4 text-gray-400" />
                  <select
                    value={selectedPriority}
                    onChange={(e) => setSelectedPriority(e.target.value)}
                    className="bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-1 text-sm"
                  >
                    <option value="all">All Priorities</option>
                    <option value="critical">Critical</option>
                    <option value="high">High</option>
                    <option value="medium">Medium</option>
                    <option value="low">Low</option>
                  </select>
                </div>
              </div>
            </CardHeader>
            <CardBody>
              {items?.data?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Type</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Description</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Location</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Priority</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Est. Time</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.data.map((item: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3">
                            <div className="flex items-center gap-2">
                              {getTypeIcon(item.type)}
                              <span className="text-sm capitalize">{item.type.replace('_', ' ')}</span>
                            </div>
                          </td>
                          <td className="py-3 text-sm max-w-sm truncate">{item.description}</td>
                          <td className="py-3 text-sm font-mono text-gray-400">
                            {item.file}:{item.line}
                          </td>
                          <td className="py-3">
                            <Badge className={getPriorityColor(item.priority)}>
                              {item.priority}
                            </Badge>
                          </td>
                          <td className="py-3 text-sm">{item.estimated_minutes}m</td>
                          <td className="py-3">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => resolveMutation.mutate(item.id)}
                              isLoading={resolveMutation.isPending}
                              leftIcon={<CheckCircle className="w-4 h-4" />}
                            >
                              Resolve
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No debt items"
                  description="Great! Your codebase has no tracked technical debt"
                />
              )}
            </CardBody>
          </Card>

          {summary?.data?.recently_resolved?.length > 0 && (
            <Card className="mt-6">
              <CardHeader>
                <h3 className="font-semibold">Recently Resolved</h3>
              </CardHeader>
              <CardBody>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Description</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Type</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Resolved</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Time Saved</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.data.recently_resolved.map((item: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm max-w-sm truncate">{item.description}</td>
                          <td className="py-3">
                            <Badge variant="success">{item.type}</Badge>
                          </td>
                          <td className="py-3 text-sm text-gray-400">
                            {formatDistanceToNow(new Date(item.resolved_at), { addSuffix: true })}
                          </td>
                          <td className="py-3 text-sm text-success">+{item.time_saved}m</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardBody>
            </Card>
          )}
        </main>
      </div>
    </div>
  );
}
