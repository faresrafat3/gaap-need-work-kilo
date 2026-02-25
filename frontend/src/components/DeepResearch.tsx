import React, { useState } from 'react';
import { Search, Globe, FileText, CheckCircle2, Loader2, ArrowRight } from 'lucide-react';

export default function DeepResearch() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  // Mock Data
  const recentSearches = [
    { id: 'rs-1', topic: 'Latest React 19 features and useActionState', status: 'completed', sources: 12, confidence: 0.94 },
    { id: 'rs-2', topic: 'Python AST parsing best practices', status: 'completed', sources: 8, confidence: 0.88 },
    { id: 'rs-3', topic: 'SQLite concurrent write locks mitigation', status: 'completed', sources: 15, confidence: 0.91 },
  ];

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setIsSearching(true);
    // Simulate API call
    setTimeout(() => {
      setIsSearching(false);
      setQuery('');
    }, 2000);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <Search className="w-6 h-6 mr-3 text-blue-500" />
            Deep Research Engine
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Web-augmented research with ETS scoring and hypothesis building.</p>
        </div>
      </div>

      {/* Search Input Area */}
      <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <form onSubmit={handleSearch} className="relative">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Globe className="h-5 w-5 text-zinc-500" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a topic for deep autonomous research..."
            className="block w-full pl-12 pr-32 py-4 bg-zinc-950 border border-zinc-800 rounded-xl text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
          />
          <div className="absolute inset-y-0 right-2 flex items-center">
            <button
              type="submit"
              disabled={isSearching || !query.trim()}
              className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {isSearching ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Researching
                </>
              ) : (
                <>
                  Start Research
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </button>
          </div>
        </form>
        
        {/* Research Settings/Toggles could go here */}
        <div className="mt-4 flex items-center space-x-6 text-sm text-zinc-400">
          <label className="flex items-center cursor-pointer hover:text-zinc-300">
            <input type="checkbox" className="mr-2 rounded border-zinc-700 bg-zinc-900 text-blue-500 focus:ring-blue-500/50" defaultChecked />
            Build Hypothesis
          </label>
          <label className="flex items-center cursor-pointer hover:text-zinc-300">
            <input type="checkbox" className="mr-2 rounded border-zinc-700 bg-zinc-900 text-blue-500 focus:ring-blue-500/50" defaultChecked />
            Calculate ETS Score
          </label>
          <label className="flex items-center cursor-pointer hover:text-zinc-300">
            <input type="checkbox" className="mr-2 rounded border-zinc-700 bg-zinc-900 text-blue-500 focus:ring-blue-500/50" />
            Deep Web Crawl
          </label>
        </div>
      </div>

      {/* Recent Research Sessions */}
      <div>
        <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider mb-4">Recent Research Sessions</h3>
        <div className="grid grid-cols-1 gap-4">
          {recentSearches.map((session) => (
            <div key={session.id} className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-5 flex items-center justify-between hover:bg-zinc-900/60 transition-colors cursor-pointer group">
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-blue-500/10 rounded-lg border border-blue-500/20 text-blue-400 group-hover:bg-blue-500/20 transition-colors">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="text-zinc-200 font-medium">{session.topic}</h4>
                  <div className="flex items-center mt-1 space-x-4 text-xs text-zinc-500">
                    <span className="flex items-center">
                      <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-500" />
                      {session.status}
                    </span>
                    <span>{session.sources} Sources Analyzed</span>
                  </div>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-xs font-mono text-zinc-500 mb-1">ETS Confidence</div>
                <div className="text-lg font-semibold text-blue-400">{(session.confidence * 100).toFixed(0)}%</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
