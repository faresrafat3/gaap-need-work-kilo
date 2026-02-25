'use client';

import { useState } from 'react';
import { X, ChevronDown, ChevronRight, Brain, BookOpen, Lightbulb, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import type { Explanation, SemanticRule, EpisodicMemory } from './WhyButton';

interface ExplanationSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: Explanation | null;
  isLoading: boolean;
}

interface ExpandableSectionProps {
  title: string;
  icon: React.ReactNode;
  defaultExpanded?: boolean;
  children: React.ReactNode;
}

function ExpandableSection({ title, icon, defaultExpanded = false, children }: ExpandableSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border-b border-layer1/20 last:border-b-0">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 p-4 hover:bg-cyber-dark/50 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
        {icon}
        <span className="font-medium">{title}</span>
      </button>
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pl-11">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SemanticRuleSection({ rule }: { rule: SemanticRule | null }) {
  if (!rule) {
    return (
      <div className="text-gray-500 italic">No semantic rule applied</div>
    );
  }

  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm text-gray-400">Rule</div>
        <div className="font-medium text-layer1">{rule.name}</div>
      </div>
      <div>
        <div className="text-sm text-gray-400">Description</div>
        <div className="text-sm">{rule.description}</div>
      </div>
      <div>
        <div className="text-sm text-gray-400">Weight</div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-cyber-dark rounded-full overflow-hidden">
            <div
              className="h-full bg-layer1 rounded-full"
              style={{ width: `${rule.weight * 100}%` }}
            />
          </div>
          <span className="text-sm font-mono">{(rule.weight * 100).toFixed(0)}%</span>
        </div>
      </div>
      {rule.matched_conditions.length > 0 && (
        <div>
          <div className="text-sm text-gray-400 mb-1">Matched Conditions</div>
          <ul className="space-y-1">
            {rule.matched_conditions.map((condition, i) => (
              <li key={i} className="text-sm flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-success" />
                {condition}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function EpisodicMemoryItem({ memory, rank }: { memory: EpisodicMemory; rank: number }) {
  const outcomeColors = {
    success: 'text-success',
    failure: 'text-error',
    neutral: 'text-gray-400',
  };

  return (
    <div className="p-3 bg-cyber-dark rounded-lg space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">#{rank}</span>
        <span className={`text-xs ${outcomeColors[memory.outcome]}`}>
          {memory.outcome}
        </span>
      </div>
      <p className="text-sm">{memory.summary}</p>
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>Relevance: {(memory.relevance_score * 100).toFixed(0)}%</span>
        <span>{new Date(memory.timestamp).toLocaleDateString()}</span>
      </div>
    </div>
  );
}

function LLMReasoningSection({ reasoning, confidence }: { reasoning: string; confidence: number }) {
  return (
    <div className="space-y-3">
      <div className="p-3 bg-cyber-dark rounded-lg">
        <p className="text-sm whitespace-pre-wrap">{reasoning}</p>
      </div>
      <div>
        <div className="text-sm text-gray-400 mb-1">Confidence</div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-cyber-dark rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                confidence > 0.7 ? 'bg-success' : confidence > 0.4 ? 'bg-warning' : 'bg-error'
              }`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
          <span className="text-sm font-mono">{(confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}

export function ExplanationSidebar({ isOpen, onClose, explanation, isLoading }: ExplanationSidebarProps) {
  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="absolute right-0 top-0 bottom-0 w-96 bg-cyber-darker border-l border-layer1/30 shadow-xl overflow-hidden flex flex-col"
      >
        <div className="flex items-center justify-between p-4 border-b border-layer1/30">
          <h2 className="text-lg font-semibold">Why This Decision?</h2>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="w-8 h-8 animate-spin text-layer1" />
            </div>
          ) : explanation ? (
            <div className="divide-y divide-layer1/20">
              <ExpandableSection
                title="Semantic Rule"
                icon={<Brain className="w-4 h-4 text-layer1" />}
                defaultExpanded
              >
                <SemanticRuleSection rule={explanation.semantic_rule} />
              </ExpandableSection>

              <ExpandableSection
                title={`Episodic Memories (${explanation.episodic_memories.length})`}
                icon={<BookOpen className="w-4 h-4 text-layer3" />}
                defaultExpanded
              >
                <div className="space-y-2">
                  {explanation.episodic_memories.length > 0 ? (
                    explanation.episodic_memories.slice(0, 3).map((memory, i) => (
                      <EpisodicMemoryItem key={memory.id} memory={memory} rank={i + 1} />
                    ))
                  ) : (
                    <div className="text-gray-500 italic">No relevant memories found</div>
                  )}
                </div>
              </ExpandableSection>

              <ExpandableSection
                title="LLM Reasoning"
                icon={<Lightbulb className="w-4 h-4 text-warning" />}
                defaultExpanded
              >
                <LLMReasoningSection
                  reasoning={explanation.llm_reasoning}
                  confidence={explanation.confidence}
                />
              </ExpandableSection>
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 text-gray-500">
              No explanation available
            </div>
          )}
        </div>

        {explanation && (
          <div className="p-4 border-t border-layer1/30 text-xs text-gray-500">
            Decision ID: {explanation.decision_id}
            <br />
            Generated: {new Date(explanation.timestamp).toLocaleString()}
          </div>
        )}
      </motion.div>
    </div>,
    document.body
  );
}