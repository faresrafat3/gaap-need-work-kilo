import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

type Theme = 'dark' | 'light' | 'system';
type SidebarState = 'expanded' | 'collapsed';

interface UIState {
  theme: Theme;
  setTheme: (theme: Theme) => void;

  sidebarState: SidebarState;
  toggleSidebar: () => void;
  setSidebarState: (state: SidebarState) => void;

  activeModal: string | null;
  modalData: Record<string, any> | null;
  openModal: (modalId: string, data?: Record<string, any>) => void;
  closeModal: () => void;

  unreadCount: number;
  incrementUnread: () => void;
  clearUnread: () => void;

  searchQuery: string;
  searchOpen: boolean;
  setSearchQuery: (query: string) => void;
  setSearchOpen: (open: boolean) => void;

  shortcutsEnabled: boolean;
  toggleShortcuts: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'dark',
      setTheme: (theme) => set({ theme }),

      sidebarState: 'expanded',
      toggleSidebar: () =>
        set((state) => ({
          sidebarState: state.sidebarState === 'expanded' ? 'collapsed' : 'expanded',
        })),
      setSidebarState: (state) => set({ sidebarState: state }),

      activeModal: null,
      modalData: null,
      openModal: (modalId, data) => set({ activeModal: modalId, modalData: data || null }),
      closeModal: () => set({ activeModal: null, modalData: null }),

      unreadCount: 0,
      incrementUnread: () => set((state) => ({ unreadCount: state.unreadCount + 1 })),
      clearUnread: () => set({ unreadCount: 0 }),

      searchQuery: '',
      searchOpen: false,
      setSearchQuery: (query) => set({ searchQuery: query }),
      setSearchOpen: (open) => set({ searchOpen: open }),

      shortcutsEnabled: true,
      toggleShortcuts: () =>
        set((state) => ({ shortcutsEnabled: !state.shortcutsEnabled })),
    }),
    {
      name: 'gaap-ui-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        theme: state.theme,
        sidebarState: state.sidebarState,
        shortcutsEnabled: state.shortcutsEnabled,
      }),
    }
  )
);
