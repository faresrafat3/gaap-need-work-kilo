import { create } from 'zustand';

interface Event {
  event_id: string;
  type: string;
  timestamp: string;
  source: string;
  data: Record<string, any>;
}

interface EventState {
  events: Event[];
  maxEvents: number;
  isConnected: boolean;
  connectionStatus: 'connected' | 'disconnected' | 'reconnecting';

  eventTypeFilter: string | null;
  sourceFilter: string | null;

  addEvent: (event: Event) => void;
  clearEvents: () => void;
  setConnected: (connected: boolean) => void;
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'reconnecting') => void;
  setEventTypeFilter: (type: string | null) => void;
  setSourceFilter: (source: string | null) => void;

  getFilteredEvents: () => Event[];
  getEventsByType: (type: string) => Event[];
}

export const useEventStore = create<EventState>((set, get) => ({
  events: [],
  maxEvents: 100,
  isConnected: false,
  connectionStatus: 'disconnected',
  eventTypeFilter: null,
  sourceFilter: null,

  addEvent: (event) =>
    set((state) => {
      const newEvents = [event, ...state.events];
      return {
        events: newEvents.slice(0, state.maxEvents),
      };
    }),

  clearEvents: () => set({ events: [] }),

  setConnected: (connected) =>
    set({
      isConnected: connected,
      connectionStatus: connected ? 'connected' : 'disconnected',
    }),

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  setEventTypeFilter: (type) => set({ eventTypeFilter: type }),

  setSourceFilter: (source) => set({ sourceFilter: source }),

  getFilteredEvents: () => {
    const { events, eventTypeFilter, sourceFilter } = get();
    return events.filter((event) => {
      if (eventTypeFilter && event.type !== eventTypeFilter) return false;
      if (sourceFilter && event.source !== sourceFilter) return false;
      return true;
    });
  },

  getEventsByType: (type) => get().events.filter((e) => e.type === type),
}));
