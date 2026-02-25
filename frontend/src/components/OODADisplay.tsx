import React from 'react';
import { OODAStage } from '../types';
import { Activity, Eye, Compass, Brain, Zap, BookOpen } from 'lucide-react';

interface OODADisplayProps {
  currentStage: OODAStage;
}

export default function OODADisplay({ currentStage }: OODADisplayProps) {
  const stages = [
    { id: 'OBSERVE', label: 'Observe', icon: Eye, color: 'text-blue-400', bg: 'bg-blue-400/10', border: 'border-blue-400/20' },
    { id: 'ORIENT', label: 'Orient', icon: Compass, color: 'text-amber-400', bg: 'bg-amber-400/10', border: 'border-amber-400/20' },
    { id: 'DECIDE', label: 'Decide', icon: Brain, color: 'text-purple-400', bg: 'bg-purple-400/10', border: 'border-purple-400/20' },
    { id: 'ACT', label: 'Act', icon: Zap, color: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/20' },
    { id: 'LEARN', label: 'Learn', icon: BookOpen, color: 'text-indigo-400', bg: 'bg-indigo-400/10', border: 'border-indigo-400/20' },
  ];

  return (
    <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center">
          <Activity className="w-4 h-4 mr-2 text-zinc-500" />
          Cognitive Loop Status
        </h3>
      </div>

      <div className="relative flex justify-between items-center">
        {/* Connecting Line */}
        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-0.5 bg-zinc-800 -z-10"></div>

        {stages.map((stage, index) => {
          const Icon = stage.icon;
          const isActive = currentStage === stage.id;
          const isPast = stages.findIndex(s => s.id === currentStage) > index;

          return (
            <div key={stage.id} className="flex flex-col items-center relative z-10">
              <div 
                className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-500 ${
                  isActive 
                    ? `${stage.bg} ${stage.border} ${stage.color} scale-110 shadow-[0_0_20px_rgba(0,0,0,0.2)] shadow-${stage.color.split('-')[1]}-500/20` 
                    : isPast
                      ? 'bg-zinc-800 border-zinc-700 text-zinc-400'
                      : 'bg-zinc-950 border-zinc-800 text-zinc-600'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'animate-pulse' : ''}`} />
              </div>
              <span className={`mt-3 text-xs font-mono font-medium tracking-wide ${isActive ? stage.color : 'text-zinc-500'}`}>
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
