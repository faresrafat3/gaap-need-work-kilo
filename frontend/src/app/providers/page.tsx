'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { providersApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Server, Plus, Trash2, TestTube, Power } from 'lucide-react';
import { motion } from 'framer-motion';
import { useState } from 'react';

export default function ProvidersPage() {
  const queryClient = useQueryClient();
  const [showAddModal, setShowAddModal] = useState(false);

  const { data: providers, isLoading } = useQuery({
    queryKey: ['providers'],
    queryFn: () => providersApi.list(),
  });

  const testMutation = useMutation({
    mutationFn: (name: string) => providersApi.test(name),
  });

  const toggleMutation = useMutation({
    mutationFn: ({ name, enable }: { name: string; enable: boolean }) =>
      enable ? providersApi.enable(name) : providersApi.disable(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: (name: string) => providersApi.remove(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['providers'] }),
  });

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'text-success';
      case 'degraded':
        return 'text-warning';
      default:
        return 'text-error';
    }
  };

  if (isLoading) {
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
        <Header title="Providers" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold">LLM Providers</h2>
              <p className="text-sm text-gray-500 mt-1">
                Manage and configure your LLM providers
              </p>
            </div>
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-layer1 text-white hover:bg-layer1/80 transition-all"
            >
              <Plus className="w-4 h-4" />
              Add Provider
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {providers?.data?.map((provider: any) => (
              <motion.div
                key={provider.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-cyber-darker border border-layer1/30 rounded-lg p-4"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <Server className="w-5 h-5 text-layer2" />
                    <div>
                      <h3 className="font-medium">{provider.name}</h3>
                      <span className="text-xs text-gray-500">{provider.type}</span>
                    </div>
                  </div>
                  <span className={`text-xs font-medium ${getHealthColor(provider.health)}`}>
                    {provider.health}
                  </span>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Priority</span>
                    <span>{provider.priority}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Requests</span>
                    <span>{provider.stats?.requests || 0}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Avg Latency</span>
                    <span>{provider.stats?.avgLatency?.toFixed(0) || 0}ms</span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => testMutation.mutate(provider.name)}
                    disabled={testMutation.isPending}
                    className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded bg-cyber-dark text-gray-400 hover:text-white transition-all text-sm"
                  >
                    <TestTube className="w-3 h-3" />
                    Test
                  </button>
                  <button
                    onClick={() =>
                      toggleMutation.mutate({
                        name: provider.name,
                        enable: !provider.enabled,
                      })
                    }
                    className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 rounded bg-cyber-dark text-gray-400 hover:text-white transition-all text-sm"
                  >
                    <Power className="w-3 h-3" />
                    {provider.enabled ? 'Disable' : 'Enable'}
                  </button>
                  <button
                    onClick={() => deleteMutation.mutate(provider.name)}
                    className="flex items-center justify-center p-1.5 rounded bg-cyber-dark text-error hover:bg-error/20 transition-all"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>

          {providers?.data?.length === 0 && (
            <div className="text-center text-gray-500 py-12">
              No providers configured. Add one to get started.
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
