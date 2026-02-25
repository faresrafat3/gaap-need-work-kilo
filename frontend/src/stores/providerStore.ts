import { create } from 'zustand';

interface Provider {
  name: string;
  type: string;
  enabled: boolean;
  priority: number;
  models: string[];
  health: 'healthy' | 'degraded' | 'unhealthy';
  stats: {
    requests: number;
    tokens: number;
    cost: number;
    avgLatency: number;
  };
}

interface ProviderState {
  providers: Provider[];
  selectedProvider: string | null;
  isLoading: boolean;
  error: string | null;

  setProviders: (providers: Provider[]) => void;
  addProvider: (provider: Provider) => void;
  updateProvider: (name: string, updates: Partial<Provider>) => void;
  removeProvider: (name: string) => void;
  setSelectedProvider: (name: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  getProvider: (name: string) => Provider | undefined;
  getEnabledProviders: () => Provider[];
  getHealthyProviders: () => Provider[];
}

export const useProviderStore = create<ProviderState>((set, get) => ({
  providers: [],
  selectedProvider: null,
  isLoading: false,
  error: null,

  setProviders: (providers) => set({ providers }),

  addProvider: (provider) =>
    set((state) => ({
      providers: [...state.providers, provider].sort((a, b) => a.priority - b.priority),
    })),

  updateProvider: (name, updates) =>
    set((state) => ({
      providers: state.providers.map((p) =>
        p.name === name ? { ...p, ...updates } : p
      ),
    })),

  removeProvider: (name) =>
    set((state) => ({
      providers: state.providers.filter((p) => p.name !== name),
      selectedProvider: state.selectedProvider === name ? null : state.selectedProvider,
    })),

  setSelectedProvider: (name) => set({ selectedProvider: name }),

  setLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  getProvider: (name) => get().providers.find((p) => p.name === name),

  getEnabledProviders: () => get().providers.filter((p) => p.enabled),

  getHealthyProviders: () =>
    get().providers.filter((p) => p.enabled && p.health === 'healthy'),
}));
