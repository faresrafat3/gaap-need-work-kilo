'use client';

import { Pause } from 'lucide-react';
import { motion } from 'framer-motion';

interface PauseButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function PauseButton({ onClick, disabled }: PauseButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-warning/20 text-warning hover:bg-warning/30 transition-all disabled:opacity-50"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Pause className="w-4 h-4" />
      Pause
    </motion.button>
  );
}
