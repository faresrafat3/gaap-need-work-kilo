'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { sessionsApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Play, Pause, Download, Trash2, Clock, Activity } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { motion } from 'framer-motion';

export default function SessionsPage() {
  const queryClient = useQueryClient();

  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => sessionsApi.list(),
    refetchInterval: 5000,
  });

  const pauseMutation = useMutation({
    mutationFn: (id: string) => sessionsApi.pause(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  });

  const resumeMutation = useMutation({
    mutationFn: (id: string) => sessionsApi.resume(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  });

  const exportMutation = useMutation({
    mutationFn: ({ id, format }: { id: string; format: string }) =>
      sessionsApi.export(id, format),
    onSuccess: (data) => {
      // Download the exported file
      const blob = new Blob([JSON.stringify(data.data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session-${data.data.id}.json`;
      a.click();
      URL.revokeObjectURL(url);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => sessionsApi.remove(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sessions'] }),
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-success bg-success/20';
      case 'paused':
        return 'text-warning bg-warning/20';
      case 'completed':
        return 'text-blue-400 bg-blue-400/20';
      case 'failed':
        return 'text-error bg-error/20';
      default:
        return 'text-gray-400 bg-gray-400/20';
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
        <Header title="Sessions" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold">Sessions</h2>
              <p className="text-sm text-gray-500 mt-1">
                Manage research and work sessions
              </p>
            </div>
          </div>

          <div className="bg-cyber-darker border border-layer1/30 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-cyber-dark">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                    ID
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                    Created
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-400">
                    Updated
                  </th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-400">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-layer1/20">
                {sessions?.data?.map((session: any) => (
                  <motion.tr
                    key={session.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-cyber-dark/50 transition-colors"
                  >
                    <td className="px-4 py-3 font-mono text-sm">
                      {session.id.slice(0, 8)}...
                    </td>
                    <td className="px-4 py-3">
                      <span className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-layer2" />
                        <span className="capitalize">{session.type}</span>
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium capitalize ${getStatusColor(
                          session.status
                        )}`}
                      >
                        {session.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatDistanceToNow(new Date(session.created), {
                        addSuffix: true,
                      })}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatDistanceToNow(new Date(session.updated), {
                        addSuffix: true,
                      })}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-2">
                        {session.status === 'running' && (
                          <button
                            onClick={() => pauseMutation.mutate(session.id)}
                            className="p-1.5 rounded bg-cyber-dark text-warning hover:bg-warning/20 transition-all"
                          >
                            <Pause className="w-4 h-4" />
                          </button>
                        )}
                        {session.status === 'paused' && (
                          <button
                            onClick={() => resumeMutation.mutate(session.id)}
                            className="p-1.5 rounded bg-cyber-dark text-success hover:bg-success/20 transition-all"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        )}
                        <button
                          onClick={() =>
                            exportMutation.mutate({ id: session.id, format: 'json' })
                          }
                          className="p-1.5 rounded bg-cyber-dark text-gray-400 hover:text-white transition-all"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => deleteMutation.mutate(session.id)}
                          className="p-1.5 rounded bg-cyber-dark text-error hover:bg-error/20 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>

            {sessions?.data?.length === 0 && (
              <div className="text-center text-gray-500 py-12">
                No sessions found
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
