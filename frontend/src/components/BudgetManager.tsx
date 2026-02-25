import React, { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, AlertCircle, BarChart3, PieChart } from 'lucide-react';
import { api, BudgetStatus } from '../api';

export default function BudgetManager() {
  const [budget, setBudget] = useState<BudgetStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchBudget() {
      try {
        const data = await api.budget.status();
        setBudget(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch budget:', err);
        setError(err instanceof Error ? err.message : 'Failed to load budget');
      } finally {
        setLoading(false);
      }
    }

    fetchBudget();
    const interval = setInterval(fetchBudget, 30000);
    return () => clearInterval(interval);
  }, []);

  const budgetState = budget ? {
    totalLimit: budget.monthly_limit,
    currentSpend: budget.monthly_spent,
    currency: 'USD',
    status: budget.hard_stop ? 'critical' : budget.throttling ? 'warning' : 'healthy',
  } : {
    totalLimit: 50.00,
    currentSpend: 12.45,
    currency: 'USD',
    status: 'healthy' as const,
  };

  const spendPercentage = budget ? budget.monthly_percentage : (budgetState.currentSpend / budgetState.totalLimit) * 100;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading budget status...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <DollarSign className="w-6 h-6 mr-3 text-emerald-500" />
            Budget & Token Management
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Monitor API costs, token usage, and set spending limits.</p>
        </div>
        <button className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-sm font-medium transition-colors border border-zinc-700">
          Export Report
        </button>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          Failed to load budget: {error}
        </div>
      )}

      {/* Main Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Total Spend Card */}
        <div className="lg:col-span-2 bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
          <div className="flex justify-between items-start mb-8">
            <div>
              <p className="text-sm font-medium text-zinc-400 uppercase tracking-wider mb-2">Total Spend</p>
              <div className="flex items-baseline">
                <span className="text-4xl font-bold text-zinc-100">${budgetState.currentSpend.toFixed(2)}</span>
                <span className="text-sm text-zinc-500 ml-2">/ ${budgetState.totalLimit.toFixed(2)}</span>
              </div>
            </div>
            <div className={`flex items-center px-3 py-1 rounded-full ${
              budgetState.status === 'healthy' 
                ? 'bg-emerald-500/10 border border-emerald-500/20'
                : budgetState.status === 'warning'
                ? 'bg-amber-500/10 border border-amber-500/20'
                : 'bg-rose-500/10 border border-rose-500/20'
            }`}>
              <TrendingUp className={`w-4 h-4 mr-2 ${
                budgetState.status === 'healthy' 
                  ? 'text-emerald-400'
                  : budgetState.status === 'warning'
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`} />
              <span className={`text-xs font-medium ${
                budgetState.status === 'healthy' 
                  ? 'text-emerald-400'
                  : budgetState.status === 'warning'
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`}>
                {budgetState.status === 'healthy' ? 'Healthy' : budgetState.status === 'warning' ? 'Warning' : 'Critical'}
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-zinc-500 font-mono">
              <span>0%</span>
              <span>{spendPercentage.toFixed(1)}% Used</span>
              <span>100%</span>
            </div>
            <div className="w-full h-3 bg-zinc-950 rounded-full overflow-hidden border border-zinc-800">
              <div 
                className={`h-full rounded-full transition-all duration-500 ${
                  spendPercentage > 90 ? 'bg-rose-500' : spendPercentage > 75 ? 'bg-amber-500' : 'bg-emerald-500'
                }`}
                style={{ width: `${Math.min(spendPercentage, 100)}%` }}
              ></div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-zinc-950/50 rounded-lg p-3">
              <p className="text-xs text-zinc-500 uppercase">Daily Spend</p>
              <p className="text-lg font-semibold text-zinc-200">${budget?.daily_spent.toFixed(2) || '0.00'}</p>
            </div>
            <div className="bg-zinc-950/50 rounded-lg p-3">
              <p className="text-xs text-zinc-500 uppercase">Daily Limit</p>
              <p className="text-lg font-semibold text-zinc-200">${budget?.daily_limit.toFixed(2) || '0.00'}</p>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-8 flex space-x-4">
            <button className="flex-1 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-sm font-medium transition-colors">
              Increase Limit
            </button>
            <button className="flex-1 py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 rounded-lg text-sm font-medium transition-colors border border-rose-500/20">
              Emergency Stop
            </button>
          </div>
        </div>

        {/* Provider Breakdown - Placeholder since usage API is optional */}
        <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
          <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center mb-6">
            <PieChart className="w-4 h-4 mr-2 text-blue-400" />
            Monthly Breakdown
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-zinc-400">Monthly Limit</span>
              <span className="text-zinc-200 font-mono">${budget?.monthly_limit.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Spent</span>
              <span className="text-emerald-400 font-mono">${budget?.monthly_spent.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Remaining</span>
              <span className="text-zinc-200 font-mono">${budget?.monthly_remaining.toFixed(2)}</span>
            </div>
            <div className="pt-4 border-t border-zinc-800">
              <div className="flex justify-between">
                <span className="text-zinc-400">Per Task Limit</span>
                <span className="text-zinc-200 font-mono">${budget?.per_task_limit.toFixed(2)}</span>
              </div>
            </div>
            {budget?.throttling && (
              <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <p className="text-xs text-amber-400">Throttling active - requests may be delayed</p>
              </div>
            )}
            {budget?.hard_stop && (
              <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg">
                <p className="text-xs text-rose-400">Hard stop reached - non-critical requests blocked</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Transactions - Placeholder */}
      <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center mb-6">
          <BarChart3 className="w-4 h-4 mr-2 text-purple-400" />
          Recent Token Usage
        </h3>
        <div className="text-center text-zinc-500 py-8">
          Transaction history available via /api/budget/usage endpoint
        </div>
      </div>
    </div>
  );
}
