import React, { useState } from 'react';
import { Pause, Play, XOctagon, Activity, MessageSquare } from 'lucide-react';

export default function SteeringControl() {
  const [instruction, setInstruction] = useState('');
  const [status, setStatus] = useState<'running' | 'paused'>('running');

  return (
    <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center">
          <Activity className="w-4 h-4 mr-2 text-emerald-500" />
          Active Steering
        </h3>
        <div className="flex space-x-2">
          {status === 'running' ? (
            <button 
              onClick={() => setStatus('paused')}
              className="flex items-center px-3 py-1.5 bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 rounded-lg text-xs font-medium transition-colors border border-amber-500/20"
            >
              <Pause className="w-3.5 h-3.5 mr-1.5" /> Pause Execution
            </button>
          ) : (
            <button 
              onClick={() => setStatus('running')}
              className="flex items-center px-3 py-1.5 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors border border-emerald-500/20"
            >
              <Play className="w-3.5 h-3.5 mr-1.5" /> Resume
            </button>
          )}
          <button className="flex items-center px-3 py-1.5 bg-rose-500/10 text-rose-500 hover:bg-rose-500/20 rounded-lg text-xs font-medium transition-colors border border-rose-500/20">
            <XOctagon className="w-3.5 h-3.5 mr-1.5" /> Veto Action
          </button>
        </div>
      </div>

      <div className="relative">
        <MessageSquare className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
        <input 
          type="text"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          placeholder="Inject new instruction into current loop (e.g., 'Focus on error handling first')..."
          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2.5 pl-10 pr-4 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50"
        />
        <button 
          className="absolute right-2 top-1/2 -translate-y-1/2 px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded text-xs font-medium transition-colors"
          disabled={!instruction.trim()}
        >
          Inject
        </button>
      </div>
    </div>
  );
}

