import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface Session {
  id: string;
  type: 'research' | 'chat' | 'task';
  status: 'running' | 'paused' | 'completed' | 'failed';
  created: string;
  updated: string;
  data: Record<string, any>;
}

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  isLoading: boolean;
  error: string | null;

  setSessions: (sessions: Session[]) => void;
  addSession: (session: Session) => void;
  updateSession: (id: string, updates: Partial<Session>) => void;
  removeSession: (id: string) => void;
  setActiveSession: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  getActiveSession: () => Session | undefined;
  getRunningSessions: () => Session[];
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      isLoading: false,
      error: null,

      setSessions: (sessions) => set({ sessions }),

      addSession: (session) =>
        set((state) => ({
          sessions: [session, ...state.sessions],
        })),

      updateSession: (id, updates) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, ...updates, updated: new Date().toISOString() } : s
          ),
        })),

      removeSession: (id) =>
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== id),
          activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
        })),

      setActiveSession: (id) => set({ activeSessionId: id }),

      setLoading: (loading) => set({ isLoading: loading }),

      setError: (error) => set({ error }),

      getActiveSession: () =>
        get().sessions.find((s) => s.id === get().activeSessionId),

      getRunningSessions: () =>
        get().sessions.filter((s) => s.status === 'running'),
    }),
    {
      name: 'gaap-session-store',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);
