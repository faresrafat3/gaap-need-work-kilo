import React, { useEffect, useRef } from 'react';
import { SystemEvent } from '../types';
import { Terminal as TerminalIcon } from 'lucide-react';

interface TerminalProps {
  events: SystemEvent[];
}

export default function Terminal({ events }: TerminalProps) {
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  const getColorForLevel = (level: string) => {
    switch (level) {
      case 'info': return 'text-blue-400';
      case 'warn': return 'text-amber-400';
      case 'error': return 'text-rose-400';
      case 'success': return 'text-emerald-400';
      default: return 'text-zinc-400';
    }
  };

  return (
    <div className="bg-[#0a0a0a] border border-zinc-800/50 rounded-xl flex flex-col h-[400px] overflow-hidden shadow-2xl">
      <div className="h-10 bg-zinc-900/80 border-b border-zinc-800/50 flex items-center px-4 justify-between">
        <div className="flex items-center text-zinc-400 text-xs font-mono">
          <TerminalIcon className="w-4 h-4 mr-2" />
          system_event_stream.log
        </div>
        <div className="flex space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed space-y-2">
        {events.map((event) => (
          <div key={event.id} className="flex items-start hover:bg-zinc-900/30 px-2 py-1 rounded transition-colors">
            <span className="text-zinc-600 min-w-[80px] shrink-0">
              {new Date(event.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit', second:'2-digit' })}
            </span>
            <span className={`min-w-[120px] shrink-0 font-semibold ${getColorForLevel(event.level)}`}>
              [{event.source}]
            </span>
            <span className="text-zinc-300 break-words">
              {event.message}
            </span>
          </div>
        ))}
        <div ref={endOfMessagesRef} />
      </div>
    </div>
  );
}
