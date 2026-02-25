import React, { useState } from 'react';
import { AgentStatus } from '../types';
import { Network, Plus, Play, Square, RefreshCw, ShieldAlert, Cpu } from 'lucide-react';

export default function SwarmControl() {
  // Mock data for agents
  const [agents, setAgents] = useState<AgentStatus[]>([
    { id: 'agt-01', role: 'Architect', status: 'working', currentTask: 'Designing system schema for new feature' },
    { id: 'agt-02', role: 'Coder', status: 'working', currentTask: 'Implementing AST parser in Python' },
    { id: 'agt-03', role: 'Critic', status: 'idle', currentTask: 'Awaiting code for review' },
    { id: 'agt-04', role: 'Researcher', status: 'idle', currentTask: 'Standby' },
    { id: 'agt-05', role: 'DebtScanner', status: 'error', currentTask: 'Failed to access SQLite DB' },
  ]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header Section */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <Network className="w-6 h-6 mr-3 text-emerald-500" />
            Swarm Intelligence
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Manage and monitor autonomous agents in the swarm.</p>
        </div>
        <div className="flex space-x-3">
          <button className="flex items-center px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-sm font-medium transition-colors border border-zinc-700">
            <RefreshCw className="w-4 h-4 mr-2" />
            Sync State
          </button>
          <button className="flex items-center px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 rounded-lg text-sm font-medium transition-colors border border-emerald-500/20">
            <Plus className="w-4 h-4 mr-2" />
            Spawn Agent
          </button>
        </div>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
}

function AgentCard({ agent, key }: { agent: AgentStatus, key?: string | number }) {
  const statusConfig = {
    working: { color: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/20', icon: Cpu, label: 'Working' },
    idle: { color: 'text-zinc-400', bg: 'bg-zinc-800/50', border: 'border-zinc-700', icon: Square, label: 'Idle' },
    error: { color: 'text-rose-400', bg: 'bg-rose-400/10', border: 'border-rose-400/20', icon: ShieldAlert, label: 'Error' }
  };

  const config = statusConfig[agent.status];
  const StatusIcon = config.icon;

  return (
    <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 flex flex-col relative overflow-hidden group">
      {/* Background Glow for working agents */}
      {agent.status === 'working' && (
        <div className="absolute -top-10 -right-10 w-32 h-32 bg-emerald-500/5 rounded-full blur-2xl"></div>
      )}

      <div className="flex justify-between items-start mb-4 relative z-10">
        <div>
          <div className="flex items-center space-x-2 mb-1">
            <h3 className="text-lg font-semibold text-zinc-200">{agent.role}</h3>
            <span className="text-xs font-mono text-zinc-500 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
              {agent.id}
            </span>
          </div>
          <div className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${config.bg} ${config.color} ${config.border} border`}>
            <StatusIcon className="w-3 h-3 mr-1.5" />
            {config.label}
          </div>
        </div>
        
        {/* Action Menu (Dots) could go here */}
        <button className="text-zinc-500 hover:text-zinc-300 transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"></path></svg>
        </button>
      </div>

      <div className="flex-1 relative z-10">
        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Current Task</p>
        <p className="text-sm text-zinc-300 line-clamp-2 h-10">
          {agent.currentTask || 'No active task'}
        </p>
      </div>

      <div className="mt-5 pt-4 border-t border-zinc-800/50 flex justify-between items-center relative z-10">
        <div className="flex space-x-2">
          <button 
            className={`p-1.5 rounded-md transition-colors ${agent.status === 'working' ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed' : 'bg-zinc-800 hover:bg-zinc-700 text-emerald-400'}`}
            disabled={agent.status === 'working'}
            title="Start Agent"
          >
            <Play className="w-4 h-4" />
          </button>
          <button 
            className={`p-1.5 rounded-md transition-colors ${agent.status === 'idle' ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed' : 'bg-zinc-800 hover:bg-zinc-700 text-rose-400'}`}
            disabled={agent.status === 'idle'}
            title="Stop Agent"
          >
            <Square className="w-4 h-4" />
          </button>
        </div>
        
        <button className="text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors">
          View Logs &rarr;
        </button>
      </div>
    </div>
  );
}
