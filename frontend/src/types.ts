export type OODAStage = 'IDLE' | 'OBSERVE' | 'ORIENT' | 'DECIDE' | 'ACT' | 'LEARN';

export interface SystemEvent {
  id: string;
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'success';
  source: string; // e.g., 'Layer0_Observe', 'Swarm_Orchestrator', 'Memory_VectorDB'
  message: string;
  metadata?: Record<string, any>;
}

export interface AgentStatus {
  id: string;
  role: string;
  status: 'idle' | 'working' | 'error';
  currentTask?: string;
}

export interface SystemMetrics {
  status: 'online' | 'degraded' | 'offline';
  activeAgents: number;
  memoryUsage: number; // percentage
  uptime: string;
  currentStage: OODAStage;
}
