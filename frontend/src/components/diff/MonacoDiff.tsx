'use client';

import { useEffect, useRef, useState } from 'react';
import { Loader2, Check, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/common/Button';
import { motion, AnimatePresence } from 'framer-motion';

interface MonacoDiffProps {
  original: string;
  modified: string;
  originalTitle?: string;
  modifiedTitle?: string;
  language?: string;
  onAccept?: () => void;
  onReject?: () => void;
  readOnly?: boolean;
  showLineNumbers?: boolean;
  minHeight?: number;
}

interface DiffChange {
  type: 'added' | 'removed' | 'unchanged';
  lineNumber: number;
  content: string;
  originalLineNumber?: number;
  modifiedLineNumber?: number;
}

declare global {
  interface Window {
    monaco: typeof import('monaco-editor');
  }
}

function computeDiff(original: string, modified: string): { original: DiffChange[]; modified: DiffChange[] } {
  const originalLines = original.split('\n');
  const modifiedLines = modified.split('\n');

  const originalChanges: DiffChange[] = [];
  const modifiedChanges: DiffChange[] = [];

  const maxLen = Math.max(originalLines.length, modifiedLines.length);

  let origIdx = 0;
  let modIdx = 0;

  const diff = require('diff');
  const diffResult = diff.diffLines(original, modified);

  diffResult.forEach((part: any) => {
    const lines = part.value.split('\n');
    if (lines[lines.length - 1] === '') lines.pop();

    lines.forEach((line: string) => {
      if (part.added) {
        modifiedChanges.push({
          type: 'added',
          lineNumber: modIdx + 1,
          content: line,
          modifiedLineNumber: modIdx + 1,
        });
        modIdx++;
      } else if (part.removed) {
        originalChanges.push({
          type: 'removed',
          lineNumber: origIdx + 1,
          content: line,
          originalLineNumber: origIdx + 1,
        });
        origIdx++;
      } else {
        originalChanges.push({
          type: 'unchanged',
          lineNumber: origIdx + 1,
          content: line,
          originalLineNumber: origIdx + 1,
          modifiedLineNumber: modIdx + 1,
        });
        modifiedChanges.push({
          type: 'unchanged',
          lineNumber: modIdx + 1,
          content: line,
          originalLineNumber: origIdx + 1,
          modifiedLineNumber: modIdx + 1,
        });
        origIdx++;
        modIdx++;
      }
    });
  });

  return { original: originalChanges, modified: modifiedChanges };
}

function DiffLine({
  change,
  showLineNumbers,
}: {
  change: DiffChange;
  showLineNumbers: boolean;
}) {
  const bgColors = {
    added: 'bg-success/10',
    removed: 'bg-error/10',
    unchanged: '',
  };

  const lineColors = {
    added: 'border-l-2 border-success',
    removed: 'border-l-2 border-error',
    unchanged: '',
  };

  const textColors = {
    added: 'text-success',
    removed: 'text-error line-through',
    unchanged: 'text-gray-300',
  };

  const prefix = change.type === 'added' ? '+' : change.type === 'removed' ? '-' : ' ';

  return (
    <div
      className={`flex font-mono text-sm ${bgColors[change.type]} ${lineColors[change.type]}`}
    >
      {showLineNumbers && (
        <div className="w-12 px-2 text-right text-gray-600 select-none border-r border-layer1/20">
          {change.type === 'added' ? change.modifiedLineNumber : change.originalLineNumber || ''}
        </div>
      )}
      <div className={`w-6 text-center ${textColors[change.type]} select-none`}>
        {prefix}
      </div>
      <div className={`flex-1 px-2 whitespace-pre ${textColors[change.type]}`}>
        {change.content || ' '}
      </div>
    </div>
  );
}

function SimpleDiffView({
  original,
  modified,
  showLineNumbers,
  viewMode,
}: {
  original: DiffChange[];
  modified: DiffChange[];
  showLineNumbers: boolean;
  viewMode: 'side-by-side' | 'unified';
}) {
  if (viewMode === 'unified') {
    const unified: DiffChange[] = [];
    let origIdx = 0;
    let modIdx = 0;

    while (origIdx < original.length || modIdx < modified.length) {
      const origChange = original[origIdx];
      const modChange = modified[modIdx];

      if (origChange?.type === 'removed') {
        unified.push(origChange);
        origIdx++;
      } else if (modChange?.type === 'added') {
        unified.push(modChange);
        modIdx++;
      } else if (origChange?.type === 'unchanged' && modChange?.type === 'unchanged') {
        unified.push(origChange);
        origIdx++;
        modIdx++;
      } else {
        if (origIdx < original.length) {
          unified.push(origChange);
          origIdx++;
        }
        if (modIdx < modified.length) {
          unified.push(modChange);
          modIdx++;
        }
      }
    }

    return (
      <div className="bg-cyber-darker rounded-lg overflow-hidden border border-layer1/20">
        <div className="overflow-auto">
          {unified.map((change, i) => (
            <DiffLine key={i} change={change} showLineNumbers={showLineNumbers} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-1">
      <div className="bg-cyber-darker rounded-lg overflow-hidden border border-layer1/20">
        <div className="overflow-auto">
          {original.map((change, i) => (
            <DiffLine key={i} change={change} showLineNumbers={showLineNumbers} />
          ))}
        </div>
      </div>
      <div className="bg-cyber-darker rounded-lg overflow-hidden border border-layer1/20">
        <div className="overflow-auto">
          {modified.map((change, i) => (
            <DiffLine key={i} change={change} showLineNumbers={showLineNumbers} />
          ))}
        </div>
      </div>
    </div>
  );
}

export function MonacoDiff({
  original,
  modified,
  originalTitle = 'Original',
  modifiedTitle = 'Modified',
  language = 'typescript',
  onAccept,
  onReject,
  readOnly = false,
  showLineNumbers = true,
  minHeight = 300,
}: MonacoDiffProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [viewMode, setViewMode] = useState<'side-by-side' | 'unified'>('side-by-side');
  const [diffData, setDiffData] = useState<{ original: DiffChange[]; modified: DiffChange[] } | null>(null);

  useEffect(() => {
    try {
      const result = computeDiff(original, modified);
      setDiffData(result);
    } catch {
      setDiffData({
        original: original.split('\n').map((line, i) => ({
          type: 'unchanged' as const,
          lineNumber: i + 1,
          content: line,
          originalLineNumber: i + 1,
        })),
        modified: modified.split('\n').map((line, i) => ({
          type: 'unchanged' as const,
          lineNumber: i + 1,
          content: line,
          modifiedLineNumber: i + 1,
        })),
      });
    }
  }, [original, modified]);

  useEffect(() => {
    const loadMonaco = async () => {
      if (window.monaco) {
        setIsLoaded(true);
        return;
      }

      try {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js';
        script.onload = () => {
          (window as any).require.config({
            paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' },
          });
          (window as any).require(['vs/editor/editor.main'], () => {
            setIsLoaded(true);
          });
        };
        document.body.appendChild(script);
      } catch {
        setIsLoaded(false);
      }
    };

    loadMonaco();
  }, []);

  useEffect(() => {
    if (!isLoaded || !containerRef.current || !window.monaco) return;

    const editor = window.monaco.editor.createDiffEditor(containerRef.current, {
      theme: 'vs-dark',
      readOnly,
      minimap: { enabled: false },
      lineNumbers: showLineNumbers ? 'on' : 'off',
      scrollBeyondLastLine: false,
      renderSideBySide: viewMode === 'side-by-side',
      automaticLayout: true,
    });

    const originalModel = window.monaco.editor.createModel(original, language);
    const modifiedModel = window.monaco.editor.createModel(modified, language);

    editor.setModel({
      original: originalModel,
      modified: modifiedModel,
    });

    return () => {
      editor.dispose();
      originalModel.dispose();
      modifiedModel.dispose();
    };
  }, [isLoaded, original, modified, language, readOnly, showLineNumbers, viewMode]);

  const stats = diffData
    ? {
        added: diffData.modified.filter((c) => c.type === 'added').length,
        removed: diffData.original.filter((c) => c.type === 'removed').length,
        unchanged: diffData.original.filter((c) => c.type === 'unchanged').length,
      }
    : null;

  return (
    <div className="flex flex-col rounded-lg overflow-hidden border border-layer1/30">
      <div className="flex items-center justify-between px-4 py-2 bg-cyber-dark border-b border-layer1/30">
        <div className="flex items-center gap-4">
          <span className="font-medium">Diff View</span>
          {stats && (
            <div className="flex items-center gap-3 text-sm">
              <span className="text-success">+{stats.added}</span>
              <span className="text-error">-{stats.removed}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setViewMode(viewMode === 'side-by-side' ? 'unified' : 'side-by-side')}
            leftIcon={viewMode === 'side-by-side' ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          >
            {viewMode === 'side-by-side' ? 'Unified' : 'Side by Side'}
          </Button>
          {onAccept && (
            <Button variant="success" size="sm" onClick={onAccept} leftIcon={<Check className="w-4 h-4" />}>
              Accept
            </Button>
          )}
          {onReject && (
            <Button variant="error" size="sm" onClick={onReject} leftIcon={<X className="w-4 h-4" />}>
              Reject
            </Button>
          )}
        </div>
      </div>

      <div className="relative">
        {!isLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-cyber-darker/80 z-10">
            <div className="flex items-center gap-2 text-gray-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Loading editor...</span>
            </div>
          </div>
        )}
        {diffData && !isLoaded && (
          <div style={{ minHeight }} className="p-2">
            <SimpleDiffView
              original={diffData.original}
              modified={diffData.modified}
              showLineNumbers={showLineNumbers}
              viewMode={viewMode}
            />
          </div>
        )}
        <div
          ref={containerRef}
          style={{ minHeight, display: isLoaded ? 'block' : 'none' }}
          className="monaco-container"
        />
      </div>
    </div>
  );
}