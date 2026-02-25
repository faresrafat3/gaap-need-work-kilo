'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { securityApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Card, CardHeader, CardBody } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { Badge } from '@/components/common/Badge';
import { Progress } from '@/components/common/Progress';
import { EmptyState } from '@/components/common/EmptyState';
import { Shield, AlertTriangle, CheckCircle, XCircle, Key, Eye, Lock, RefreshCw, Download } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';

export default function SecurityPage() {
  const queryClient = useQueryClient();

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['security-status'],
    queryFn: () => securityApi.getStatus(),
    refetchInterval: 30000,
  });

  const { data: events, isLoading: eventsLoading } = useQuery({
    queryKey: ['security-events'],
    queryFn: () => securityApi.getEvents(50),
  });

  const { data: keys } = useQuery({
    queryKey: ['security-keys'],
    queryFn: () => securityApi.getKeys(),
  });

  const scanMutation = useMutation({
    mutationFn: () => securityApi.scan(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security'] });
    },
  });

  const rotateKeyMutation = useMutation({
    mutationFn: (keyId: string) => securityApi.rotateKey(keyId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security-keys'] });
    },
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'secure':
        return <CheckCircle className="w-5 h-5 text-success" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-warning" />;
      case 'critical':
        return <XCircle className="w-5 h-5 text-error" />;
      default:
        return <Shield className="w-5 h-5 text-gray-400" />;
    }
  };

  if (statusLoading || eventsLoading) {
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
        <Header title="Security" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardBody className="text-center">
                <div className="flex justify-center mb-2">
                  {getStatusIcon(status?.data?.overall_status || 'unknown')}
                </div>
                <div className="text-lg font-semibold capitalize">
                  {status?.data?.overall_status || 'Unknown'}
                </div>
                <div className="text-sm text-gray-400 mt-1">Overall Status</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-error">
                  {status?.data?.critical_issues || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Critical Issues</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-warning">
                  {status?.data?.warnings || 0}
                </div>
                <div className="text-sm text-gray-400 mt-1">Warnings</div>
              </CardBody>
            </Card>
            <Card>
              <CardBody className="text-center">
                <div className="text-3xl font-bold text-success">
                  {status?.data?.last_scan ? formatDistanceToNow(new Date(status.data.last_scan), { addSuffix: true }) : 'Never'}
                </div>
                <div className="text-sm text-gray-400 mt-1">Last Scan</div>
              </CardBody>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">Security Score</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => scanMutation.mutate()}
                    isLoading={scanMutation.isPending}
                    leftIcon={<RefreshCw className="w-4 h-4" />}
                  >
                    Scan
                  </Button>
                </div>
              </CardHeader>
              <CardBody className="space-y-4">
                <div className="text-center">
                  <div className="text-5xl font-bold text-success mb-2">
                    {status?.data?.score || 0}
                  </div>
                  <div className="text-sm text-gray-400">out of 100</div>
                </div>
                <Progress value={status?.data?.score || 0} size="lg" />
                <div className="text-xs text-gray-500 text-center">
                  Based on {status?.data?.checks_performed || 0} security checks
                </div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="font-semibold">API Keys</h3>
              </CardHeader>
              <CardBody>
                {keys?.data?.length > 0 ? (
                  <div className="space-y-3">
                    {keys.data.map((key: any, i: number) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-3 bg-cyber-dark rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <Key className="w-4 h-4 text-layer3" />
                          <div>
                            <div className="text-sm font-medium">{key.name}</div>
                            <div className="text-xs text-gray-500 font-mono">
                              {key.prefix}...{key.suffix}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={key.status === 'valid' ? 'success' : 'error'}>
                            {key.status}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => rotateKeyMutation.mutate(key.id)}
                            isLoading={rotateKeyMutation.isPending}
                          >
                            <RefreshCw className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <EmptyState
                    title="No API keys"
                    description="API keys will appear here when configured"
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
                  onClick={() => scanMutation.mutate()}
                  isLoading={scanMutation.isPending}
                  leftIcon={<Shield className="w-4 h-4" />}
                >
                  Run Security Scan
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<Lock className="w-4 h-4" />}>
                  Audit Permissions
                </Button>
                <Button variant="secondary" isFullWidth leftIcon={<Download className="w-4 h-4" />}>
                  Export Report
                </Button>
              </CardBody>
            </Card>
          </div>

          {status?.data?.issues?.length > 0 && (
            <Card className="mt-6">
              <CardHeader>
                <h3 className="font-semibold">Active Issues</h3>
              </CardHeader>
              <CardBody>
                <div className="space-y-3">
                  {status.data.issues.map((issue: any, i: number) => (
                    <div
                      key={i}
                      className={`p-4 rounded-lg border ${
                        issue.severity === 'critical'
                          ? 'bg-error/5 border-error/30'
                          : issue.severity === 'high'
                          ? 'bg-error/5 border-error/20'
                          : 'bg-warning/5 border-warning/30'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={`w-5 h-5 mt-0.5 ${
                            issue.severity === 'critical' || issue.severity === 'high'
                              ? 'text-error'
                              : 'text-warning'
                          }`} />
                          <div>
                            <div className="font-medium">{issue.title}</div>
                            <div className="text-sm text-gray-400 mt-1">{issue.description}</div>
                            {issue.recommendation && (
                              <div className="text-xs text-gray-500 mt-2 p-2 bg-cyber-dark rounded">
                                {issue.recommendation}
                              </div>
                            )}
                          </div>
                        </div>
                        <Badge className={getSeverityColor(issue.severity)}>
                          {issue.severity}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardBody>
            </Card>
          )}

          <Card className="mt-6">
            <CardHeader>
              <h3 className="font-semibold">Security Events</h3>
            </CardHeader>
            <CardBody>
              {events?.data?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-layer1/30">
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Time</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Type</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Description</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Severity</th>
                        <th className="text-left py-3 text-sm font-medium text-gray-400">Source</th>
                      </tr>
                    </thead>
                    <tbody>
                      {events.data.map((event: any, i: number) => (
                        <tr key={i} className="border-b border-layer1/20">
                          <td className="py-3 text-sm text-gray-400">
                            {formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}
                          </td>
                          <td className="py-3">
                            <Badge variant="default">{event.type}</Badge>
                          </td>
                          <td className="py-3 text-sm max-w-md truncate">{event.description}</td>
                          <td className="py-3">
                            <Badge className={getSeverityColor(event.severity)}>
                              {event.severity}
                            </Badge>
                          </td>
                          <td className="py-3 text-sm font-mono">{event.source}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="No security events"
                  description="Security events will appear here when detected"
                />
              )}
            </CardBody>
          </Card>
        </main>
      </div>
    </div>
  );
}
