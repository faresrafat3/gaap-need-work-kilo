'use client';

import { useState } from 'react';
import { HelpCircle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '@/lib/api';
import { ExplanationSidebar } from './ExplanationSidebar';

interface WhyButtonProps {
  decisionId: string;
  decisionType?: 'tool' | 'provider' | 'strategy' | 'action';
  onExplain?: (explanation: Explanation) => void;
}

export interface Explanation {
  decision_id: string;
  semantic_rule: SemanticRule | null;
  episodic_memories: EpisodicMemory[];
  llm_reasoning: string;
  confidence: number;
  timestamp: string;
}

export interface SemanticRule {
  id: string;
  name: string;
  description: string;
  weight: number;
  matched_conditions: string[];
}

export interface EpisodicMemory {
  id: string;
  summary: string;
  relevance_score: number;
  timestamp: string;
  outcome: 'success' | 'failure' | 'neutral';
  context: string;
}

export function WhyButton({ decisionId, decisionType = 'action', onExplain }: WhyButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExplanation = async () => {
    if (explanation) {
      setIsOpen(true);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.get(`/api/explain/${decisionId}`, {
        params: { type: decisionType },
      });
      const data = response.data as Explanation;
      setExplanation(data);
      setIsOpen(true);
      onExplain?.(data);
    } catch (err) {
      setError('Failed to fetch explanation');
      console.error('Failed to fetch explanation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <>
      <motion.button
        onClick={fetchExplanation}
        disabled={isLoading}
        className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-layer1/20 text-gray-400 hover:bg-layer1/40 hover:text-white transition-all"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        title="Why this decision?"
      >
        {isLoading ? (
          <Loader2 className="w-3 h-3 animate-spin" />
        ) : (
          <HelpCircle className="w-3 h-3" />
        )}
      </motion.button>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed top-4 right-4 bg-error/20 text-error px-4 py-2 rounded-lg z-50"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      <ExplanationSidebar
        isOpen={isOpen}
        onClose={handleClose}
        explanation={explanation}
        isLoading={isLoading}
      />
    </>
  );
}