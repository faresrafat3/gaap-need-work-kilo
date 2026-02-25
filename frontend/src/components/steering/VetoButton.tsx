'use client';

import { X, AlertTriangle } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface VetoButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export function VetoButton({ onClick, disabled }: VetoButtonProps) {
  const [confirmOpen, setConfirmOpen] = useState(false);

  const handleVeto = () => {
    onClick();
    setConfirmOpen(false);
  };

  return (
    <>
      <motion.button
        onClick={() => setConfirmOpen(true)}
        disabled={disabled}
        className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-error/20 text-error hover:bg-error/30 transition-all disabled:opacity-50"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <X className="w-4 h-4" />
        Veto
      </motion.button>

      <AnimatePresence>
        {confirmOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
            onClick={() => setConfirmOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-cyber-darker border border-error/30 rounded-lg p-6 max-w-md mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-error/20 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-error" />
                </div>
                <div>
                  <h3 className="font-semibold">Confirm Veto</h3>
                  <p className="text-sm text-gray-400">This action cannot be undone</p>
                </div>
              </div>

              <p className="text-sm text-gray-300 mb-6">
                Are you sure you want to veto the current action? This will cancel the ongoing process and may require manual intervention to recover.
              </p>

              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setConfirmOpen(false)}
                  className="px-4 py-2 rounded-lg bg-cyber-dark text-gray-300 hover:bg-cyber-dark/80 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleVeto}
                  className="px-4 py-2 rounded-lg bg-error text-white hover:bg-error/80 transition-all"
                >
                  Confirm Veto
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
