'use client';

import { Bell, RefreshCw } from 'lucide-react';
import { useEvents } from '@/hooks/useEvents';

interface HeaderProps {
  title: string;
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

export function Header({ title }: HeaderProps) {
  const { connectionStatus } = useEvents();

  return (
    <header className="h-16 bg-cyber-darker border-b border-layer1/30 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-semibold">{title}</h2>
        <div className={cn(
          'flex items-center gap-1 px-2 py-1 rounded text-xs',
          connectionStatus === 'connected' 
            ? 'bg-success/20 text-success' 
            : 'bg-warning/20 text-warning'
        )}>
          <div className={cn(
            'w-1.5 h-1.5 rounded-full',
            connectionStatus === 'connected' ? 'bg-success' : 'bg-warning'
          )} />
          {connectionStatus}
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        <button className="p-2 hover:bg-layer1/10 rounded-lg transition-colors">
          <RefreshCw className="w-5 h-5 text-gray-400" />
        </button>
        <button className="p-2 hover:bg-layer1/10 rounded-lg transition-colors relative">
          <Bell className="w-5 h-5 text-gray-400" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full" />
        </button>
      </div>
    </header>
  );
}
