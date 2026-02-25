'use client';

import { useEffect, useCallback, useRef } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  action: () => void;
  description: string;
  preventDefault?: boolean;
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], enabled = true) {
  const shortcutsRef = useRef(shortcuts);

  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        if (event.key !== 'Escape') {
          return;
        }
      }

      for (const shortcut of shortcutsRef.current) {
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey || event.metaKey : true;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;
        const metaMatch = shortcut.meta ? event.metaKey : true;
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();

        if (ctrlMatch && shiftMatch && altMatch && metaMatch && keyMatch) {
          if (shortcut.preventDefault !== false) {
            event.preventDefault();
          }
          shortcut.action();
          break;
        }
      }
    },
    [enabled]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);
}

export function useAppShortcuts() {
  const actions = {
    search: () => {
      const searchInput = document.querySelector('[data-shortcut="search"]') as HTMLInputElement;
      searchInput?.focus();
    },
    toggleSidebar: () => {
      const toggleBtn = document.querySelector('[data-shortcut="toggle-sidebar"]') as HTMLButtonElement;
      toggleBtn?.click();
    },
    refresh: () => {
      window.location.reload();
    },
    openHelp: () => {
      const helpBtn = document.querySelector('[data-shortcut="help"]') as HTMLButtonElement;
      helpBtn?.click();
    },
    closeModal: () => {
      const closeBtn = document.querySelector('[data-shortcut="close-modal"]') as HTMLButtonElement;
      closeBtn?.click();
    },
    navigateDashboard: () => {
      window.location.href = '/';
    },
    navigateConfig: () => {
      window.location.href = '/config';
    },
    navigateProviders: () => {
      window.location.href = '/providers';
    },
    navigateResearch: () => {
      window.location.href = '/research';
    },
    navigateSessions: () => {
      window.location.href = '/sessions';
    },
  };

  const shortcuts: KeyboardShortcut[] = [
    { key: '1', ctrl: true, action: actions.navigateDashboard, description: 'Go to Dashboard' },
    { key: '2', ctrl: true, action: actions.navigateConfig, description: 'Go to Config' },
    { key: '3', ctrl: true, action: actions.navigateProviders, description: 'Go to Providers' },
    { key: '4', ctrl: true, action: actions.navigateResearch, description: 'Go to Research' },
    { key: '5', ctrl: true, action: actions.navigateSessions, description: 'Go to Sessions' },
    { key: 'k', ctrl: true, action: actions.search, description: 'Open search' },
    { key: 'b', ctrl: true, action: actions.toggleSidebar, description: 'Toggle sidebar' },
    { key: 'r', ctrl: true, action: actions.refresh, description: 'Refresh page' },
    { key: '/', action: actions.search, description: 'Open search' },
    { key: '?', shift: true, action: actions.openHelp, description: 'Open help' },
    { key: 'Escape', action: actions.closeModal, description: 'Close modal' },
  ];

  useKeyboardShortcuts(shortcuts);

  return { shortcuts };
}

export function useShortcutsHelp() {
  const { shortcuts } = useAppShortcuts();

  const formatShortcut = (shortcut: KeyboardShortcut) => {
    const parts: string[] = [];
    if (shortcut.ctrl) parts.push('Ctrl');
    if (shortcut.shift) parts.push('Shift');
    if (shortcut.alt) parts.push('Alt');
    parts.push(shortcut.key.toUpperCase());
    return parts.join(' + ');
  };

  return {
    shortcuts: shortcuts.map((s) => ({
      keys: formatShortcut(s),
      description: s.description,
    })),
  };
}
