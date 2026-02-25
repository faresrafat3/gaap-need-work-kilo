import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface ConfigState {
  config: Record<string, any> | null;
  selectedModule: string | null;
  hasUnsavedChanges: boolean;
  isLoading: boolean;
  error: string | null;

  setConfig: (config: Record<string, any>) => void;
  updateModule: (module: string, values: Record<string, any>) => void;
  setSelectedModule: (module: string | null) => void;
  setHasUnsavedChanges: (hasChanges: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState = {
  config: null,
  selectedModule: null,
  hasUnsavedChanges: false,
  isLoading: false,
  error: null,
};

export const useConfigStore = create<ConfigState>()(
  persist(
    (set, get) => ({
      ...initialState,

      setConfig: (config) => set({ config, hasUnsavedChanges: false }),

      updateModule: (module, values) =>
        set((state) => ({
          config: state.config
            ? { ...state.config, [module]: { ...state.config[module], ...values } }
            : { [module]: values },
          hasUnsavedChanges: true,
        })),

      setSelectedModule: (module) => set({ selectedModule: module }),

      setHasUnsavedChanges: (hasChanges) => set({ hasUnsavedChanges: hasChanges }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      reset: () => set(initialState),
    }),
    {
      name: 'gaap-config-store',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        selectedModule: state.selectedModule,
      }),
    }
  )
);
