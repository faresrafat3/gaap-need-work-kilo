const API_BASE = '/api';

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

export interface SystemHealth {
  status: string;
  version: string;
  uptime_seconds: number;
  timestamp: string;
  components: Array<{
    name: string;
    status: string;
    message: string;
    latency_ms?: number;
    details?: Record<string, unknown>;
  }>;
}

export interface Session {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'normal' | 'high' | 'critical';
  tags: string[];
  config: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  tasks_total: number;
  tasks_completed: number;
  tasks_failed: number;
  cost_usd: number;
  tokens_used: number;
}

export interface SessionList {
  sessions: Session[];
  total: number;
}

export interface MemoryStats {
  working: Record<string, unknown>;
  episodic: Record<string, unknown>;
  semantic: Record<string, unknown>;
  procedural: Record<string, unknown>;
}

export interface HealingStatus {
  total_attempts: number;
  successful_recoveries: number;
  escalations: number;
  recovery_rate: number;
  errors_by_category: Record<string, number>;
  healing_by_level: Record<string, Record<string, number>>;
}

export interface HealingConfig {
  success: boolean;
  config?: Record<string, unknown>;
  error?: string;
}

export interface BudgetStatus {
  monthly_limit: number;
  daily_limit: number;
  per_task_limit: number;
  monthly_spent: number;
  daily_spent: number;
  monthly_remaining: number;
  daily_remaining: number;
  monthly_percentage: number;
  daily_percentage: number;
  throttling: boolean;
  hard_stop: boolean;
}

export interface ConfigResponse {
  success: boolean;
  config?: Record<string, unknown>;
  error?: string;
}

export const api = {
  system: {
    health: () => fetchApi<SystemHealth>('/system/health'),
  },
  
  sessions: {
    list: (limit = 50, offset = 0) => 
      fetchApi<SessionList>(`/sessions?limit=${limit}&offset=${offset}`),
    get: (id: string) => fetchApi<Session>(`/sessions/${id}`),
    create: (data: { name: string; description?: string; priority?: string }) =>
      fetchApi<Session>('/sessions', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (id: string, data: Partial<Session>) =>
      fetchApi<Session>(`/sessions/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    delete: (id: string) =>
      fetchApi<{ success: boolean; message: string }>(`/sessions/${id}`, {
        method: 'DELETE',
      }),
    pause: (id: string) => fetchApi<Session>(`/sessions/${id}/pause`, { method: 'POST' }),
    resume: (id: string) => fetchApi<Session>(`/sessions/${id}/resume`, { method: 'POST' }),
  },
  
  memory: {
    stats: () => fetchApi<MemoryStats>('/memory/stats'),
    tiers: () => fetchApi<{ tiers: Array<{ name: string; level: number; size: number; max_size?: number; description: string }> }>('/memory/tiers'),
    consolidate: (source: string, target: string) =>
      fetchApi<{ success: boolean; items_consolidated: number; message: string }>('/memory/consolidate', {
        method: 'POST',
        body: JSON.stringify({ source_tier: source, target_tier: target }),
      }),
    clear: (tier: string) =>
      fetchApi<{ success: boolean; tier: string; items_cleared: number; message: string }>(`/memory/clear/${tier}`, {
        method: 'POST',
      }),
    search: (query: string, limit = 10) =>
      fetchApi<{ query: string; results: unknown[]; total: number }>(`/memory/search?query=${encodeURIComponent(query)}&limit=${limit}`),
  },
  
  healing: {
    status: () => fetchApi<HealingStatus>('/healing/stats'),
    config: () => fetchApi<HealingConfig>('/healing/config'),
    updateConfig: (data: Record<string, unknown>) =>
      fetchApi<HealingConfig>('/healing/config', {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    history: (limit = 100) =>
      fetchApi<{ items: unknown[]; total: number }>(`/healing/history?limit=${limit}`),
    patterns: () => fetchApi<unknown[]>('/healing/patterns'),
    reset: () => fetchApi<{ success: boolean; message: string }>('/healing/reset', { method: 'POST' }),
  },
  
  budget: {
    status: () => fetchApi<BudgetStatus>('/budget'),
    usage: (period = 'daily', limit = 100) =>
      fetchApi<unknown>(`/budget/usage?period=${period}&limit=${limit}`),
    alerts: () => fetchApi<{ alerts: unknown[]; total: number }>('/budget/alerts'),
    updateLimits: (data: Record<string, unknown>) =>
      fetchApi<{ success: boolean; limits?: Record<string, unknown>; error?: string }>('/budget/limits', {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
  },
  
  config: {
    get: () => fetchApi<ConfigResponse>('/config'),
    update: (config: Record<string, unknown>) =>
      fetchApi<ConfigResponse>('/config', {
        method: 'PUT',
        body: JSON.stringify({ config, validate: true }),
      }),
    getModule: (module: string) => fetchApi<ConfigResponse>(`/config/${module}`),
    updateModule: (module: string, config: Record<string, unknown>) =>
      fetchApi<ConfigResponse>(`/config/${module}`, {
        method: 'PUT',
        body: JSON.stringify({ config, validate: true }),
      }),
    reload: () => fetchApi<ConfigResponse>('/config/reload', { method: 'POST' }),
    presets: () => fetchApi<unknown[]>('/config/presets/list'),
    schema: () => fetchApi<Record<string, unknown>>('/config/schema/all'),
  },
};
