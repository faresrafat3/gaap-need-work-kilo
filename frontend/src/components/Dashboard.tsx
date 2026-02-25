import React, { useState, useEffect } from 'react';
import OODADisplay from './OODADisplay';
import Terminal from './Terminal';
import SteeringControl from './SteeringControl';
import { SystemEvent, SystemMetrics, OODAStage } from '../types';
import { Cpu, Database, Users, AlertTriangle } from 'lucide-react';
import { api, SystemHealth } from '../api';

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

export default function Dashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<OODAStage>('ORIENT');
  
  const [metrics, setMetrics] = useState<SystemMetrics>({
    status: 'online',
    activeAgents: 4,
    memoryUsage: 42,
    uptime: '14h 22m',
    currentStage: 'ORIENT'
  });

  const [events, setEvents] = useState<SystemEvent[]>([
    { id: '1', timestamp: new Date(Date.now() - 10000).toISOString(), level: 'info', source: 'Layer0_Observe', message: 'Ingesting new user prompt: "Analyze the latest PRs"' },
    { id: '2', timestamp: new Date(Date.now() - 8000).toISOString(), level: 'info', source: 'Memory_VectorDB', message: 'Querying ChromaDB for related context...' },
    { id: '3', timestamp: new Date(Date.now() - 5000).toISOString(), level: 'success', source: 'Memory_VectorDB', message: 'Retrieved 3 relevant episodic memories.' },
    { id: '4', timestamp: new Date(Date.now() - 2000).toISOString(), level: 'warn', source: 'Layer1_Orient', message: 'High ambiguity detected in prompt. Synthesizing clarification strategy.' },
  ]);

  useEffect(() => {
    async function fetchHealth() {
      try {
        const data = await api.system.health();
        setHealth(data);
        
        const memComponent = data.components.find(c => c.name === 'memory');
        const memUsage = memComponent?.details 
          ? Math.round(((memComponent.details['working'] as Record<string, number>)?.size || 0) / 
             ((memComponent.details['working'] as Record<string, number>)?.max_size || 1) * 100)
          : 42;
        
        setMetrics(prev => ({
          ...prev,
          status: data.status as 'online' | 'degraded' | 'offline',
          uptime: formatUptime(data.uptime_seconds),
          memoryUsage: memUsage,
        }));
        setError(null);
      } catch (err) {
        console.error('Failed to fetch health:', err);
        setError(err instanceof Error ? err.message : 'Failed to connect to backend');
      } finally {
        setLoading(false);
      }
    }

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const stages: OODAStage[] = ['OBSERVE', 'ORIENT', 'DECIDE', 'ACT', 'LEARN'];
    let currentIndex = 1;

    const interval = setInterval(() => {
      currentIndex = (currentIndex + 1) % stages.length;
      const newStage = stages[currentIndex];
      setCurrentStage(newStage);
      
      setMetrics(prev => ({ ...prev, currentStage: newStage }));
      
      setEvents(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          level: 'info',
          source: `Layer${currentIndex}_${newStage}`,
          message: `Transitioned to ${newStage} phase. Executing sub-routines...`
        }
      ].slice(-50));

    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const activeAgents = health?.components.filter(c => c.status === 'healthy').length || 0;
  const healingQueue = health?.components.filter(c => c.status !== 'healthy').length || 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading system status...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          Failed to connect to backend: {error}
        </div>
      )}

      {/* Top Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard icon={Cpu} label="System Status" value={health?.status?.toUpperCase() || 'UNKNOWN'} valueColor={metrics.status === 'online' ? 'text-emerald-400' : 'text-amber-400'} />
        <StatCard icon={Users} label="Healthy Components" value={activeAgents.toString()} />
        <StatCard icon={Database} label="Vector DB Load" value={`${metrics.memoryUsage}%`} />
        <StatCard icon={AlertTriangle} label="Issues Detected" value={healingQueue === 0 ? 'None' : `${healingQueue} Issues`} valueColor={healingQueue === 0 ? 'text-zinc-400' : 'text-amber-400'} />
      </div>

      {/* OODA Loop Visualization */}
      <OODADisplay currentStage={currentStage} />

      {/* Steering Control (New Feature based on Backend WebSockets) */}
      <SteeringControl />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Terminal */}
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Live Event Stream</h3>
          <Terminal events={events} />
        </div>

        {/* Right Column: Active Agents / Quick Actions */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">Active Swarm</h3>
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-4 space-y-3">
            <AgentRow name="Architect" status="working" task="Structuring response" />
            <AgentRow name="Researcher" status="idle" task="Waiting for queries" />
            <AgentRow name="Coder" status="working" task="Generating AST" />
            <AgentRow name="Critic" status="idle" task="Standby for review" />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, valueColor = "text-zinc-100" }: { icon: any, label: string, value: string, valueColor?: string }) {
  return (
    <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 flex items-center">
      <div className="p-3 bg-zinc-800/50 rounded-lg mr-4">
        <Icon className="w-5 h-5 text-zinc-400" />
      </div>
      <div>
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1">{label}</p>
        <p className={`text-xl font-semibold font-mono ${valueColor}`}>{value}</p>
      </div>
    </div>
  );
}

function AgentRow({ name, status, task }: { name: string, status: 'working'|'idle'|'error', task: string }) {
  const statusColors = {
    working: 'bg-emerald-500',
    idle: 'bg-zinc-500',
    error: 'bg-rose-500'
  };

  return (
    <div className="flex items-center justify-between p-3 bg-zinc-950/50 rounded-lg border border-zinc-800/50">
      <div className="flex items-center">
        <div className="relative flex h-3 w-3 mr-3">
          {status === 'working' && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>}
          <span className={`relative inline-flex rounded-full h-3 w-3 ${statusColors[status]}`}></span>
        </div>
        <div>
          <p className="text-sm font-medium text-zinc-200">{name}</p>
          <p className="text-xs text-zinc-500 truncate max-w-[120px]">{task}</p>
        </div>
      </div>
      <span className="text-xs font-mono text-zinc-500 uppercase">{status}</span>
    </div>
  );
}
