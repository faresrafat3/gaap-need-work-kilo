'use client';

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { researchApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Search, Loader2, ExternalLink, CheckCircle, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ResearchPage() {
  const [query, setQuery] = useState('');
  const [depth, setDepth] = useState(2);
  const [results, setResults] = useState<any>(null);

  const searchMutation = useMutation({
    mutationFn: (params: { query: string; depth: number }) =>
      researchApi.search(params.query, params.depth),
    onSuccess: (data) => {
      setResults(data.data);
    },
  });

  const handleSearch = () => {
    if (query.trim()) {
      searchMutation.mutate({ query, depth });
    }
  };

  const getETSColor = (score: number) => {
    if (score >= 0.9) return 'text-success';
    if (score >= 0.7) return 'text-layer3';
    if (score >= 0.5) return 'text-warning';
    return 'text-error';
  };

  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Research" />
        <main className="flex-1 overflow-y-auto p-6">
          {/* Search Form */}
          <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Deep Discovery Engine</h2>
            
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter your research query..."
                  className="w-full bg-cyber-dark border border-layer1/30 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-layer1"
                />
              </div>
              
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Depth:</label>
                <select
                  value={depth}
                  onChange={(e) => setDepth(parseInt(e.target.value))}
                  className="bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-layer1"
                >
                  {[1, 2, 3, 4, 5].map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>
              
              <button
                onClick={handleSearch}
                disabled={searchMutation.isPending || !query.trim()}
                className="flex items-center gap-2 px-6 py-3 rounded-lg bg-layer1 text-white hover:bg-layer1/80 transition-all disabled:opacity-50"
              >
                {searchMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                Research
              </button>
            </div>
          </div>

          {/* Results */}
          <AnimatePresence mode="wait">
            {results && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* Metrics */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Sources</div>
                    <div className="text-2xl font-bold mt-1">
                      {results.finding?.sources?.length || 0}
                    </div>
                  </div>
                  <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Hypotheses</div>
                    <div className="text-2xl font-bold mt-1">
                      {results.finding?.hypotheses?.length || 0}
                    </div>
                  </div>
                  <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Time</div>
                    <div className="text-2xl font-bold mt-1">
                      {results.metrics?.total_time_ms
                        ? `${(results.metrics.total_time_ms / 1000).toFixed(1)}s`
                        : '-'}
                    </div>
                  </div>
                  <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Status</div>
                    <div className="text-2xl font-bold mt-1">
                      {results.success ? (
                        <CheckCircle className="w-6 h-6 text-success" />
                      ) : (
                        <AlertCircle className="w-6 h-6 text-error" />
                      )}
                    </div>
                  </div>
                </div>

                {/* Sources */}
                <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-6">
                  <h3 className="text-lg font-semibold mb-4">Sources</h3>
                  <div className="space-y-3">
                    {results.finding?.sources?.map((source: any, index: number) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-cyber-dark rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className={`px-2 py-1 rounded text-xs font-medium ${getETSColor(
                              source.ets_score || 0.5
                            )} bg-opacity-20`}
                          >
                            ETS: {((source.ets_score || 0.5) * 100).toFixed(0)}%
                          </div>
                          <div>
                            <div className="font-medium">{source.title || source.url}</div>
                            <div className="text-xs text-gray-500">{source.domain}</div>
                          </div>
                        </div>
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 hover:bg-layer1/10 rounded transition-all"
                        >
                          <ExternalLink className="w-4 h-4 text-gray-400" />
                        </a>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Hypotheses */}
                {results.finding?.hypotheses?.length > 0 && (
                  <div className="bg-cyber-darker border border-layer1/30 rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-4">Hypotheses</h3>
                    <div className="space-y-3">
                      {results.finding.hypotheses.map((hyp: any, index: number) => (
                        <div
                          key={index}
                          className="p-4 bg-cyber-dark rounded-lg border-l-4 border-layer2"
                        >
                          <div className="font-medium">{hyp.claim}</div>
                          <div className="flex items-center gap-4 mt-2 text-sm text-gray-400">
                            <span>Confidence: {(hyp.confidence * 100).toFixed(0)}%</span>
                            <span>Status: {hyp.status}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {!results && !searchMutation.isPending && (
            <div className="text-center text-gray-500 py-12">
              Enter a query to start your research
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
