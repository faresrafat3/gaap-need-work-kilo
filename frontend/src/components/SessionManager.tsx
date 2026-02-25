import React, { useState, useEffect } from 'react';
import { Layers, Play, Square, Clock, ArrowRight, Save } from 'lucide-react';
import { api, Session } from '../api';

export default function SessionManager() {
  const [activeSessions, setActiveSessions] = useState<Session[]>([]);
  const [archivedSessions, setArchivedSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchSessions() {
      try {
        const data = await api.sessions.list(20, 0);
        
        const active = data.sessions.filter(s => 
          s.status === 'running' || s.status === 'paused' || s.status === 'pending'
        );
        const archived = data.sessions.filter(s => 
          s.status === 'completed' || s.status === 'failed' || s.status === 'cancelled'
        );
        
        setActiveSessions(active);
        setArchivedSessions(archived);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch sessions:', err);
        setError(err instanceof Error ? err.message : 'Failed to load sessions');
      } finally {
        setLoading(false);
      }
    }

    fetchSessions();
    const interval = setInterval(fetchSessions, 15000);
    return () => clearInterval(interval);
  }, []);

  const formatTimeAgo = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffMins > 0) return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
    return 'Just now';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading sessions...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <Layers className="w-6 h-6 mr-3 text-indigo-500" />
            Session Management
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Manage active agentic sessions, view history, and resume past contexts.</p>
        </div>
        <button className="flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition-colors">
          <Play className="w-4 h-4 mr-2" />
          New Session
        </button>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          Failed to load sessions: {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Sessions */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Active & Paused Sessions</h3>
          {activeSessions.length === 0 ? (
            <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center text-zinc-500">
              No active sessions
            </div>
          ) : (
            activeSessions.map(session => (
              <div key={session.id} className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 hover:border-indigo-500/30 transition-colors group">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h4 className="text-zinc-100 font-medium">{session.name}</h4>
                    <p className="text-xs font-mono text-zinc-500 mt-1">{session.id}</p>
                  </div>
                  <span className={`px-2.5 py-1 rounded text-xs font-medium uppercase border ${
                    session.status === 'running' 
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                      : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                  }`}>
                    {session.status}
                  </span>
                </div>
                
                <div className="mb-4">
                  <div className="flex justify-between text-xs text-zinc-400 mb-1.5">
                    <span>Progress</span>
                    <span>{session.tasks_completed} / {session.tasks_total} Tasks</span>
                  </div>
                  <div className="w-full h-1.5 bg-zinc-950 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-indigo-500 rounded-full" 
                      style={{ width: `${session.progress * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex justify-between items-center pt-4 border-t border-zinc-800/50">
                  <span className="text-xs text-zinc-500 flex items-center">
                    <Clock className="w-3.5 h-3.5 mr-1.5" />
                    Started {formatTimeAgo(session.created_at)}
                  </span>
                  <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="p-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded transition-colors" title="Save State">
                      <Save className="w-4 h-4" />
                    </button>
                    {session.status === 'running' ? (
                      <button className="p-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 rounded transition-colors" title="Pause">
                        <Square className="w-4 h-4" />
                      </button>
                    ) : (
                      <button className="p-1.5 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 rounded transition-colors" title="Resume">
                        <Play className="w-4 h-4" />
                      </button>
                    )}
                    <button className="p-1.5 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 rounded transition-colors flex items-center text-xs font-medium px-3">
                      Enter <ArrowRight className="w-3.5 h-3.5 ml-1" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Archived Sessions */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Session History</h3>
          {archivedSessions.length === 0 ? (
            <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center text-zinc-500">
              No archived sessions
            </div>
          ) : (
            archivedSessions.map(session => (
              <div key={session.id} className="bg-zinc-950/50 border border-zinc-800/30 rounded-xl p-4">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-zinc-300 text-sm font-medium">{session.name}</h4>
                  <span className={`text-xs font-medium ${session.status === 'completed' ? 'text-emerald-500' : 'text-rose-500'}`}>
                    {session.status}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs text-zinc-500">
                  <span className="font-mono">{session.id}</span>
                  <span>{session.tasks_completed}/{session.tasks_total} tasks</span>
                </div>
                <div className="mt-3 pt-3 border-t border-zinc-800/30 text-right">
                  <button className="text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
                    View Logs &rarr;
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
