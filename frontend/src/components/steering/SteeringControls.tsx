'use client';

import { useState } from 'react';
import { Pause, Play, X, Send, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface SteeringControlsProps {
  sessionId?: string;
  isPaused: boolean;
  onPause: () => void;
  onResume: (instruction?: string) => void;
  onVeto: () => void;
}

export function SteeringControls({
  sessionId,
  isPaused,
  onPause,
  onResume,
  onVeto,
}: SteeringControlsProps) {
  const [showInput, setShowInput] = useState(false);
  const [instruction, setInstruction] = useState('');

  const handleResume = () => {
    onResume(instruction || undefined);
    setInstruction('');
    setShowInput(false);
  };

  return (
    <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-4">Steering Controls</h3>
      
      <div className="flex items-center gap-3">
        <button
          onClick={isPaused ? handleResume : onPause}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
            isPaused
              ? 'bg-success/20 text-success hover:bg-success/30'
              : 'bg-warning/20 text-warning hover:bg-warning/30'
          )}
        >
          {isPaused ? (
            <>
              <Play className="w-4 h-4" />
              Resume
            </>
          ) : (
            <>
              <Pause className="w-4 h-4" />
              Pause
            </>
          )}
        </button>

        <button
          onClick={onVeto}
          className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-error/20 text-error hover:bg-error/30 transition-all"
        >
          <X className="w-4 h-4" />
          Veto
        </button>

        <button
          onClick={() => setShowInput(!showInput)}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
            showInput
              ? 'bg-layer1/40 text-white'
              : 'bg-layer1/20 text-gray-300 hover:bg-layer1/30'
          )}
        >
          <Send className="w-4 h-4" />
          Steer
        </button>
      </div>

      <AnimatePresence>
        {showInput && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-4 p-4 bg-cyber-dark rounded-lg border border-layer1/20">
              <label className="text-sm text-gray-400 block mb-2">
                Steering Instruction
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={instruction}
                  onChange={(e) => setInstruction(e.target.value)}
                  placeholder="e.g., 'Use PostgreSQL instead of MongoDB'"
                  className="flex-1 bg-cyber-darker border border-layer1/30 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-layer1"
                />
                <button
                  onClick={handleResume}
                  disabled={!instruction.trim()}
                  className="px-4 py-2 bg-layer1 text-white rounded-lg font-medium hover:bg-layer1/80 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Apply
                </button>
              </div>
              
              <div className="flex items-center gap-2 mt-3 text-xs text-gray-500">
                <AlertTriangle className="w-3 h-3" />
                <span>Steering will restart the OODA loop with the new instruction</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
