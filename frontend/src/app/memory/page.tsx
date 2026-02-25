'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { memoryApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { Progress } from '@/components/common/Progress';
import { EmptyState } from '@/components/common/EmptyState';
import { Database, Brain, Archive, Trash2, Search, RefreshCw, Layers } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';

export default function MemoryPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['memory-stats'],
    queryFn: () => memoryApi.getStats(),
    refetchInterval: 15000,
  });

  const { data: tiers, isLoading: tiersLoading } = useQuery({
    queryKey: ['memory-tiers'],
    queryFn: () => memoryApi.getTiers(),
  });

  const { data: searchResults } = useQuery({
    queryKey: ['memory-search', searchQuery],
    queryFn: () => memoryApi.search(searchQuery),
    enabled: searchQuery.length > 2,
  });

  const consolidateMutation = useMutation({
    mutationFn: () => memoryApi.consolidate(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory'] });
    },
  });

  const clearMutation = useMutation({
    mutationFn: (tier: string) => memoryApi.clear(tier),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory'] });
    },
  });

  const getTierIcon = (tier: string) => {
    switch (tier) {
      case 'working':
        return <Brain className="w-5 h-5 text-success" />;
      case 'short_term':
        return <Database className="w-5 h-5 text-layer3" />;
      case 'long_term':
        return <Archive className="w-5 h-5 text-warning" />;
      default:
        return <Layers className="w-5 h-5 text-gray-400" />;
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'working':
        return 'border-success/30';
      case 'short_term':
        return 'border-layer3/30';
      case 'long_term':
        return 'border-warning/30';
      default:
        return 'border-layer1/30';
    }
  };

  if (statsLoading || tiersLoading) {
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
        <Header title="Memory" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-success">
                  {stats?.data?.working_memory_count || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Working Memory</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-layer3">
                  {stats?.data?.short_term_count || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Short Term</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-warning">
                  {stats?.data?.long_term_count || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Long Term</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold">
                  {stats?.data?.total_size_mb?.toFixed(1) || 0} MB
                </div>
                <div className="text-sm text-gray-400 mt-1">Total Size</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Memory Tiers</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => consolidateMutation.mutate()}
                    isLoading={consolidateMutation.isPending}
                    leftIcon={<RefreshCw className="w-4 h-4" />}
                  >
                    Consolidate
                  </Button>
                </div>
              </CardHeader>
              <CardBody>
                {tiers?.data?.length > 0 ? (
                  <div className="space-y-4">
                    {tiers.data.map((tier: any, i: number) => (
                      <div
                        key={i}
                        className={`p-4 bg-cyber-darker rounded-lg border ${getTierColor(tier.name)}`}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-3">
                            {getTierIcon(tier.name)}
                            <div>
                              <div className="font-medium capitalize">
                                {tier.name.replace('_', ' ')}
                              </div>
                              <div className="text-xs text-gray-500">{tier.description}</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            <Badge variant="default">{tier.count} items</Badge>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => clearMutation.mutate(tier.name)}
                              className="text-error hover:text-error"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-xs text-gray-400">
                            <span>Capacity</span>
                            <span>{tier.utilization?.toFixed(1)}%</span>
                          </div>
                          <Progress value={tier.utilization || 0} size="sm" />
                        </div>
                        <div className="mt-3 text-xs text-gray-500">
                          TTL: {tier.ttl_seconds ? formatDistanceToNow(Date.now() + tier.ttl_seconds * 1000) : 'Permanent'}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No memory tiers"
                    description="Memory tiers will appear here when initialized"
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
                  onClick={() => consolidateMutation.mutate()}
                  isLoading={consolidateMutation.isPending}
                  leftIcon={<RefreshCw className="w-4 h-4" />}
                >
                  Consolidate All
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<Database className="w-4 h-4" />}>
                  Export Memory
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<Archive className="w-4 h-4" />}>
                  Archive Old
                </Button>
              </CardBody>
            </Card>
          </div>

          <Card className="mt-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Search Memory</h3>
              </div>
            </CardHeader>
            <CardBody>
              <div className="mb-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search memories..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-cyber-dark border border-layer1/30 rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-layer1"
                  />
                </div>
              </div>
              {searchQuery.length > 2 ? (
                searchResults?.data?.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-layer1/30">
                          <th className="text-left py-3 text-sm font-medium text-gray-400">ID</th>
                          <th className="text-left py-3 text-sm font-medium text-gray-400">Content</th>
                          <th className="text-left py-3 text-sm font-medium text-gray-400">Tier</th>
                          <th className="text-left py-3 text-sm font-medium text-gray-400">Created</th>
                        </tr>
                      </thead>
                      <tbody>
                        {searchResults.data.map((memory: any, i: number) => (
                          <tr key={i} className="border-b border-layer1/20">
                            <td className="py-3 text-sm font-mono">{memory.id?.slice(0, 8)}...</td>
                            <td className="py-3 text-sm max-w-md truncate">{memory.content}</td>
                            <td className="py-3">
                              <Badge variant="default">{memory.tier}</Badge>
                            </td>
                            <td className="py-3 text-sm text-gray-400">
                              {formatDistanceToNow(new Date(memory.created_at), { addSuffix: true })}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <EmptyState
                    title="No results found"
                    description="Try a different search query"
                  />
                )
              ) : (
                <EmptyState
                  title="Search memory"
                  description="Enter at least 3 characters to search"
                />
              )}
            </CardBody>
          </Card>

          {stats?.data?.recently_consolidated && (
            <Card className="mt-6">
              <CardHeader>
                <h3 className="font-semibold">Recent Consolidations</h3>
              </CardHeader>
              <CardBody>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Time</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">From</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">To</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Items</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Freed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.data.recently_consolidated.map((event: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm">
                            {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
                          </td>
                          <td className="py-3">
                            <Badge variant="default">{event.from_tier}</Badge>
                          </td>
                          <td className="py-3">
                            <Badge variant="default">{event.to_tier}</Badge>
                          </td>
                          <td className="py-3 text-sm">{event.items_moved}</td>
                          <td className="py-3 text-sm font-mono">{event.space_freed_mb} MB</td>
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
