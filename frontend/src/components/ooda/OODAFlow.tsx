'use client';

import { useWebSocket } from '@/hooks/useWebSocket';
import { motion } from 'framer-motion';
import { Eye, Compass, Brain, Zap, BookOpen } from 'lucide-react';
import { cn } from '@/lib/utils';

const phases = [
  { name: 'OBSERVE', icon: Eye, color: 'text-cyan-400', bg: 'bg-cyan-400/20' },
  { name: 'ORIENT', icon: Compass, color: 'text-yellow-400', bg: 'bg-yellow-400/20' },
  { name: 'DECIDE', icon: Brain, color: 'text-purple-400', bg: 'bg-purple-400/20' },
  { name: 'ACT', icon: Zap, color: 'text-green-400', bg: 'bg-green-400/20' },
  { name: 'LEARN', icon: BookOpen, color: 'text-blue-400', bg: 'bg-blue-400/20' },
];

export function OODAFlow() {
  const { lastEvent, isConnected } = useWebSocket('ooda');

  const currentPhase = lastEvent?.data?.phase || 'OBSERVE';
  const task = lastEvent?.data?.task || 'Idle';
  const iteration = lastEvent?.data?.iteration || 0;

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">OODA Loop</h3>
        <div className={cn(
          'flex items-center gap-2 text-xs',
          isConnected ? 'text-success' : 'text-warning'
        )}>
          <div className={cn('w-2 h-2 rounded-full', isConnected ? 'bg-success' : 'bg-warning')} />
          {isConnected ? 'Live' : 'Disconnected'}
        </div>
      </div>

      <div className="flex items-center justify-between mb-6">
        {phases.map((phase, index) => {
          const Icon = phase.icon;
          const isActive = currentPhase === phase.name;
          
          return (
            <div key={phase.name} className="flex items-center">
              <motion.div
                className={cn(
                  'relative flex flex-col items-center p-4 rounded-lg transition-all',
                  isActive ? phase.bg : 'bg-cyber-dark/50',
                  isActive && 'ring-2 ring-offset-2 ring-offset-cyber-darker'
                )}
                animate={isActive ? { scale: [1, 1.05, 1] } : {}}
                transition={{ duration: 0.5, repeat: isActive ? Infinity : 0 }}
              >
                <Icon className={cn('w-6 h-6 mb-2', isActive ? phase.color : 'text-gray-500')} />
                <span className={cn(
                  'text-xs font-medium',
                  isActive ? 'text-white' : 'text-gray-500'
                )}>
                  {phase.name}
                </span>
                
                {isActive && (
                  <motion.div
                    className={cn('absolute inset-0 rounded-lg', phase.bg)}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 0.5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                )}
              </motion.div>
              
              {index < phases.length - 1 && (
                <div className="w-8 h-0.5 bg-layer1/30 mx-2" />
              )}
            </div>
          );
        })}
      </div>

      <div className="bg-cyber-dark/50 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Current Task</span>
          <span className="text-xs text-gray-500">Iteration: {iteration}</span>
        </div>
        <p className="text-sm font-mono">{task}</p>
      </div>
    </div>
  );
}
