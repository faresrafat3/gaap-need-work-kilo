import React, { useState, useEffect } from 'react';
import { BrainCircuit, Database, Search, GitMerge, Clock } from 'lucide-react';
import { api, MemoryStats } from '../api';

export default function CognitiveMemory() {
  const [activeTab, setActiveTab] = useState<'episodic' | 'semantic'>('episodic');
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchMemory() {
      try {
        const data = await api.memory.stats();
        setStats(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch memory stats:', err);
        setError(err instanceof Error ? err.message : 'Failed to load memory');
      } finally {
        setLoading(false);
      }
    }

    fetchMemory();
    const interval = setInterval(fetchMemory, 30000);
    return () => clearInterval(interval);
  }, []);

  const episodicMemories = stats?.episodic 
    ? [
        { id: 'ep-001', time: 'Recent', type: 'Task Execution', content: `Total episodes: ${(stats.episodic as Record<string, number>).total_episodes || 0}`, confidence: 0.92 },
        { id: 'ep-002', time: 'Active', type: 'Memory Usage', content: `Task index size: ${(stats.episodic as Record<string, number>).task_index_size || 0}`, confidence: 0.85 },
      ]
    : [];

  const semanticNodes = stats?.semantic
    ? [
        { id: 'sem-001', concept: 'Working Memory', connections: (stats.working as Record<string, number>).size || 0, strength: 'High', lastUpdated: 'Live' },
        { id: 'sem-002', concept: 'Semantic Rules', connections: (stats.semantic as Record<string, number>).total_rules || 0, strength: 'Medium', lastUpdated: 'Live' },
        { id: 'sem-003', concept: 'Procedures', connections: (stats.procedural as Record<string, number>).total_procedures || 0, strength: 'Variable', lastUpdated: 'Live' },
      ]
    : [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading memory stats...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <BrainCircuit className="w-6 h-6 mr-3 text-purple-500" />
            Cognitive Memory
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Explore Episodic (short-term) and Semantic (long-term) memory stores.</p>
        </div>
        <div className="flex space-x-2 bg-zinc-950 p-1 rounded-lg border border-zinc-800">
          <button 
            onClick={() => setActiveTab('episodic')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'episodic' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            Episodic
          </button>
          <button 
            onClick={() => setActiveTab('semantic')}
            className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'semantic' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            Semantic
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          Failed to load memory: {error}
        </div>
      )}

      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
        <input 
          type="text" 
          placeholder={`Search ${activeTab} memory...`}
          className="w-full bg-zinc-900/40 border border-zinc-800/50 rounded-xl py-3 pl-12 pr-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
        />
      </div>

      {/* Content Area */}
      {activeTab === 'episodic' ? (
        <div className="space-y-4">
          {episodicMemories.length === 0 ? (
            <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center text-zinc-500">
              No episodic memories found
            </div>
          ) : (
            episodicMemories.map(memory => (
              <div key={memory.id} className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 hover:border-zinc-700 transition-colors">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center space-x-3">
                    <span className="px-2.5 py-1 bg-zinc-800 text-zinc-300 text-xs font-medium rounded-md border border-zinc-700">
                      {memory.type}
                    </span>
                    <span className="text-xs font-mono text-zinc-500 flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {memory.time}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-purple-400 bg-purple-400/10 px-2 py-1 rounded border border-purple-400/20">
                    Confidence: {memory.confidence}
                  </div>
                </div>
                <p className="text-sm text-zinc-300 leading-relaxed">
                  {memory.content}
                </p>
              </div>
            ))
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {semanticNodes.length === 0 ? (
            <div className="col-span-3 bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-8 text-center text-zinc-500">
              No semantic nodes found
            </div>
          ) : (
            semanticNodes.map(node => (
              <div key={node.id} className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 flex flex-col items-center text-center hover:border-purple-500/30 transition-colors group">
                <div className="w-16 h-16 rounded-full bg-zinc-950 border-2 border-zinc-800 flex items-center justify-center mb-4 group-hover:border-purple-500/50 transition-colors">
                  <Database className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-lg font-semibold text-zinc-200 mb-1">{node.concept}</h3>
                <p className="text-xs text-zinc-500 mb-4">Last updated: {node.lastUpdated}</p>
                
                <div className="w-full flex justify-between items-center pt-4 border-t border-zinc-800/50">
                  <div className="flex items-center text-xs text-zinc-400">
                    <GitMerge className="w-4 h-4 mr-1.5 text-zinc-500" />
                    {node.connections} items
                  </div>
                  <span className="text-xs font-medium text-purple-400">
                    {node.strength} Strength
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
