'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface StageIndicatorProps {
  phase: string;
  task: string;
  tokens: number;
  progress?: number;
}

export function StageIndicator({ phase, task, tokens, progress }: StageIndicatorProps) {
  const phaseColors: Record<string, string> = {
    OBSERVE: 'from-cyan-500 to-cyan-600',
    ORIENT: 'from-yellow-500 to-yellow-600',
    DECIDE: 'from-purple-500 to-purple-600',
    ACT: 'from-green-500 to-green-600',
    LEARN: 'from-blue-500 to-blue-600',
  };

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <div className="flex items-center gap-4">
        <motion.div
          className={cn(
            'w-12 h-12 rounded-full bg-gradient-to-br flex items-center justify-center',
            phaseColors[phase] || 'from-gray-500 to-gray-600'
          )}
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <span className="text-xl font-bold">{phase[0]}</span>
        </motion.div>

        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className="font-medium">{phase}</span>
            <span className="text-xs text-gray-500">{tokens} tokens</span>
          </div>
          
          <p className="text-sm text-gray-400 truncate">{task}</p>
          
          {progress !== undefined && (
            <div className="mt-2 h-1 bg-cyber-dark rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-layer1 to-layer3"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
