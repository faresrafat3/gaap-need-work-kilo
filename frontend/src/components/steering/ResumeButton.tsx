'use client';

import { Play } from 'lucide-react';
import { motion } from 'framer-motion';

interface ResumeButtonProps {
  onClick: () => void;
  disabled?: boolean;
  hasInstruction?: boolean;
}

export function ResumeButton({ onClick, disabled, hasInstruction }: ResumeButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all disabled:opacity-50 ${
        hasInstruction
          ? 'bg-layer1 text-white hover:bg-layer1/80'
          : 'bg-success/20 text-success hover:bg-success/30'
      }`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Play className="w-4 h-4" />
      Resume
    </motion.button>
  );
}
