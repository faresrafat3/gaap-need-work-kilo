import React, { useState, useEffect } from 'react';
import { Wrench, AlertTriangle, ShieldCheck, Activity, TerminalSquare } from 'lucide-react';
import { api, HealingStatus } from '../api';

interface HealingHistoryItem {
  task_id: string;
  level: string;
  action: string;
  success: boolean;
  timestamp: string;
  error_category: string;
  details: string;
}

export default function SelfHealing() {
  const [status, setStatus] = useState<HealingStatus | null>(null);
  const [history, setHistory] = useState<HealingHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchHealing() {
      try {
        const [statusData, historyData] = await Promise.all([
          api.healing.status(),
          api.healing.history(10),
        ]);
        setStatus(statusData);
        setHistory((historyData as { items: HealingHistoryItem[] }).items || []);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch healing status:', err);
        setError(err instanceof Error ? err.message : 'Failed to load healing status');
      } finally {
        setLoading(false);
      }
    }

    fetchHealing();
    const interval = setInterval(fetchHealing, 15000);
    return () => clearInterval(interval);
  }, []);

  const isHealthy = !status || (status.total_attempts === 0 || status.recovery_rate > 0.5);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading healing status...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <Wrench className="w-6 h-6 mr-3 text-amber-500" />
            Self-Healing System
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Automatic error detection, diagnosis, and recovery.</p>
        </div>
        <div className={`flex items-center px-4 py-2 rounded-lg border ${
          isHealthy 
            ? 'bg-emerald-500/10 border-emerald-500/20' 
            : 'bg-amber-500/10 border-amber-500/20'
        }`}>
          <ShieldCheck className={`w-5 h-5 mr-2 ${isHealthy ? 'text-emerald-400' : 'text-amber-400'}`} />
          <span className={`text-sm font-medium ${isHealthy ? 'text-emerald-400' : 'text-amber-400'}`}>
            {isHealthy ? 'System Stable' : 'Issues Detected'}
          </span>
        </div>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          Failed to load healing status: {error}
        </div>
      )}

      {/* Stats */}
      {status && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4">
            <p className="text-xs text-zinc-500 uppercase">Total Attempts</p>
            <p className="text-xl font-semibold text-zinc-200">{status.total_attempts}</p>
          </div>
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4">
            <p className="text-xs text-zinc-500 uppercase">Successful</p>
            <p className="text-xl font-semibold text-emerald-400">{status.successful_recoveries}</p>
          </div>
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4">
            <p className="text-xs text-zinc-500 uppercase">Escalations</p>
            <p className="text-xl font-semibold text-amber-400">{status.escalations}</p>
          </div>
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4">
            <p className="text-xs text-zinc-500 uppercase">Recovery Rate</p>
            <p className="text-xl font-semibold text-zinc-200">{(status.recovery_rate * 100).toFixed(1)}%</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Issues Column */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center">
            <Activity className="w-4 h-4 mr-2 text-amber-500" />
            Active Diagnosis Queue
          </h3>
          
          {history.filter(h => !h.success).length > 0 ? (
            history.filter(h => !h.success).slice(0, 5).map((issue, idx) => (
              <div key={idx} className="bg-zinc-900/40 border border-amber-500/30 rounded-xl p-5 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>
                
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    <span className="text-sm font-mono text-zinc-300">Level: {issue.level}</span>
                  </div>
                  <span className="text-xs font-medium px-2 py-1 rounded bg-zinc-800 text-zinc-400 border border-zinc-700 uppercase">
                    {issue.action}
                  </span>
                </div>
                
                <p className="text-sm font-mono text-rose-400 bg-rose-400/10 p-2 rounded border border-rose-400/20 mb-4">
                  {issue.details || issue.error_category}
                </p>
                
                <div className="flex items-center justify-between text-xs text-zinc-500">
                  <span>Task: {issue.task_id}</span>
                  <button className="text-amber-400 hover:text-amber-300 font-medium transition-colors">
                    View Stack Trace &rarr;
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center">
              <ShieldCheck className="w-8 h-8 text-emerald-500 mx-auto mb-3" />
              <p className="text-zinc-400 text-sm">No active issues detected.</p>
            </div>
          )}
        </div>

        {/* Resolved Issues Column */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center">
            <TerminalSquare className="w-4 h-4 mr-2 text-zinc-500" />
            Recent Resolutions
          </h3>
          
          <div className="space-y-3">
            {history.filter(h => h.success).slice(0, 5).length > 0 ? (
              history.filter(h => h.success).slice(0, 5).map((issue, idx) => (
                <div key={idx} className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4 hover:bg-zinc-900/60 transition-colors">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-mono text-emerald-400">Level: {issue.level}</span>
                    <span className="text-xs text-zinc-500">{new Date(issue.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <p className="text-sm text-zinc-300 mb-2 line-clamp-1" title={issue.details}>
                    <span className="text-zinc-500 mr-2">Action:</span>
                    {issue.action}
                  </p>
                  <p className="text-sm text-zinc-400 bg-zinc-950 p-2 rounded border border-zinc-800/50">
                    <span className="text-emerald-500 mr-2">Category:</span>
                    {issue.error_category}
                  </p>
                </div>
              ))
            ) : (
              <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center text-zinc-500">
                No resolved issues yet
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
