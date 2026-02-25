'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, X, Command, ArrowRight } from 'lucide-react';
import { clsx } from 'clsx';
import { useRouter } from 'next/navigation';

interface SearchItem {
  id: string;
  title: string;
  description?: string;
  href: string;
  category?: string;
  icon?: React.ReactNode;
}

interface SearchProps {
  items?: SearchItem[];
  placeholder?: string;
  isOpen?: boolean;
  onClose?: () => void;
}

const defaultItems: SearchItem[] = [
  { id: '1', title: 'Dashboard', href: '/', category: 'Navigation', icon: <Command className="w-4 h-4" /> },
  { id: '2', title: 'Configuration', href: '/config', category: 'Navigation' },
  { id: '3', title: 'Providers', href: '/providers', category: 'Navigation' },
  { id: '4', title: 'Research', href: '/research', category: 'Navigation' },
  { id: '5', title: 'Sessions', href: '/sessions', category: 'Navigation' },
  { id: '6', title: 'Healing Monitor', href: '/healing', category: 'Navigation' },
  { id: '7', title: 'Memory Dashboard', href: '/memory', category: 'Navigation' },
  { id: '8', title: 'Budget', href: '/budget', category: 'Navigation' },
  { id: '9', title: 'Security', href: '/security', category: 'Navigation' },
  { id: '10', title: 'Technical Debt', href: '/debt', category: 'Navigation' },
];

export function SearchCommand({ items = defaultItems, placeholder = 'Search...', isOpen: externalIsOpen, onClose }: SearchProps) {
  const router = useRouter();
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const isOpen = externalIsOpen ?? internalIsOpen;

  const filteredItems = items.filter(
    (item) =>
      item.title.toLowerCase().includes(query.toLowerCase()) ||
      item.category?.toLowerCase().includes(query.toLowerCase())
  );

  const openSearch = useCallback(() => {
    setInternalIsOpen(true);
    setTimeout(() => inputRef.current?.focus(), 0);
  }, []);

  const closeSearch = useCallback(() => {
    setInternalIsOpen(false);
    setQuery('');
    setSelectedIndex(0);
    onClose?.();
  }, [onClose]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (isOpen) {
          closeSearch();
        } else {
          openSearch();
        }
      }
      if (e.key === '/' && !isOpen) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault();
          openSearch();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, openSearch, closeSearch]);

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((prev) => (prev + 1) % filteredItems.length);
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) => (prev - 1 + filteredItems.length) % filteredItems.length);
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredItems[selectedIndex]) {
            router.push(filteredItems[selectedIndex].href);
            closeSearch();
          }
          break;
        case 'Escape':
          closeSearch();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredItems, selectedIndex, router, closeSearch]);

  return (
    <>
      <button
        onClick={openSearch}
        className="flex items-center gap-2 px-3 py-1.5 bg-cyber-dark border border-layer1/30 rounded-lg text-gray-400 hover:text-white hover:border-layer1/50 transition-all text-sm"
        data-shortcut="search"
      >
        <Search className="w-4 h-4" />
        <span>{placeholder}</span>
        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 bg-cyber-darker rounded text-xs">
          <Command className="w-3 h-3" />K
        </kbd>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-start justify-center pt-[20vh]"
            onClick={closeSearch}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -20 }}
              className="w-full max-w-lg bg-cyber-darker border border-layer1/30 rounded-lg shadow-xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 p-4 border-b border-layer1/20">
                <Search className="w-5 h-5 text-gray-500" />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setSelectedIndex(0);
                  }}
                  placeholder={placeholder}
                  className="flex-1 bg-transparent text-white placeholder-gray-500 focus:outline-none"
                />
                {query && (
                  <button onClick={() => setQuery('')} className="text-gray-500 hover:text-white">
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              <div className="max-h-80 overflow-y-auto">
                {filteredItems.length > 0 ? (
                  <div className="py-2">
                    {filteredItems.map((item, index) => (
                      <button
                        key={item.id}
                        onClick={() => {
                          router.push(item.href);
                          closeSearch();
                        }}
                        onMouseEnter={() => setSelectedIndex(index)}
                        className={clsx(
                          'w-full flex items-center justify-between px-4 py-2.5 text-left transition-colors',
                          index === selectedIndex
                            ? 'bg-layer1/20 text-white'
                            : 'text-gray-400 hover:bg-cyber-dark hover:text-white'
                        )}
                      >
                        <div className="flex items-center gap-3">
                          {item.icon && <span className="text-gray-500">{item.icon}</span>}
                          <div>
                            <div className="text-sm font-medium">{item.title}</div>
                            {item.description && (
                              <div className="text-xs text-gray-500">{item.description}</div>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {item.category && (
                            <span className="text-xs text-gray-500">{item.category}</span>
                          )}
                          <ArrowRight className="w-4 h-4 opacity-50" />
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="py-8 text-center text-gray-500">
                    No results found for "{query}"
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between px-4 py-2 border-t border-layer1/20 text-xs text-gray-500">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1">
                    <kbd className="px-1 py-0.5 bg-cyber-dark rounded">↑↓</kbd> Navigate
                  </span>
                  <span className="flex items-center gap-1">
                    <kbd className="px-1 py-0.5 bg-cyber-dark rounded">Enter</kbd> Select
                  </span>
                  <span className="flex items-center gap-1">
                    <kbd className="px-1 py-0.5 bg-cyber-dark rounded">Esc</kbd> Close
                  </span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}