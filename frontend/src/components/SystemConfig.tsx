import React, { useState, useEffect } from 'react';
import { Settings, Key, Server, Shield, Save, RefreshCw } from 'lucide-react';
import { api, ConfigResponse } from '../api';

export default function SystemConfig() {
  const [isSaving, setIsSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    async function fetchConfig() {
      try {
        const data = await api.config.get();
        if (data.success && data.config) {
          setConfig(data.config);
        }
        setError(null);
      } catch (err) {
        console.error('Failed to fetch config:', err);
        setError(err instanceof Error ? err.message : 'Failed to load config');
      } finally {
        setLoading(false);
      }
    }

    fetchConfig();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.config.reload();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save config');
    } finally {
      setTimeout(() => setIsSaving(false), 1000);
    }
  };

  const handleChange = (module: string, key: string, value: unknown) => {
    if (!config) return;
    const moduleConfig = (config[module] as Record<string, unknown>) || {};
    moduleConfig[key] = value;
    setConfig({ ...config, [module]: moduleConfig });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading configuration...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-10">
      {/* Header */}
      <div className="flex items-center justify-between bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center">
            <Settings className="w-6 h-6 mr-3 text-zinc-400" />
            System Configuration
          </h2>
          <p className="text-sm text-zinc-400 mt-1">Manage API keys, model selection, and core engine settings.</p>
        </div>
        <button 
          onClick={handleSave}
          disabled={isSaving}
          className="flex items-center px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors"
        >
          {isSaving ? <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
          {isSaving ? 'Saving...' : 'Save Configuration'}
        </button>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 text-rose-400 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: API Keys & Providers */}
        <div className="lg:col-span-2 space-y-6">
          {/* Model Providers */}
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center mb-6">
              <Server className="w-4 h-4 mr-2 text-blue-400" />
              Model Providers
            </h3>
            
            <div className="space-y-4">
              <ConfigInput 
                label="OpenAI API Key" 
                type="password" 
                placeholder="sk-..." 
                defaultValue="sk-proj-********************************"
              />
              <ConfigInput 
                label="Anthropic API Key" 
                type="password" 
                placeholder="sk-ant-..." 
                defaultValue=""
              />
              <ConfigInput 
                label="Google Gemini API Key" 
                type="password" 
                placeholder="AIza..." 
                defaultValue="AIzaSy********************************"
              />
              
              <div className="pt-4 border-t border-zinc-800/50">
                <label className="block text-sm font-medium text-zinc-300 mb-2">Primary Reasoning Model</label>
                <select className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2.5 px-3 text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50">
                  <option value="gpt-4-turbo">GPT-4 Turbo (OpenAI)</option>
                  <option value="claude-3-opus">Claude 3 Opus (Anthropic)</option>
                  <option value="gemini-1.5-pro">Gemini 1.5 Pro (Google)</option>
                </select>
              </div>
            </div>
          </div>

          {/* External Services */}
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center mb-6">
              <Key className="w-4 h-4 mr-2 text-amber-400" />
              External Services
            </h3>
            
            <div className="space-y-4">
              <ConfigInput 
                label="Tavily API Key (Search)" 
                type="password" 
                placeholder="tvly-..." 
                defaultValue="tvly-********************************"
              />
              <ConfigInput 
                label="GitHub Personal Access Token" 
                type="password" 
                placeholder="ghp_..." 
                defaultValue=""
              />
            </div>
          </div>
        </div>

        {/* Right Column: Engine Settings */}
        <div className="space-y-6">
          <div className="bg-zinc-900/40 border border-zinc-800/50 rounded-xl p-6">
            <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center mb-6">
              <Shield className="w-4 h-4 mr-2 text-purple-400" />
              Engine Parameters
            </h3>
            
            <div className="space-y-5">
              <ToggleSetting 
                label="Auto-Healing" 
                description="Allow system to automatically recover from errors without user intervention."
                defaultChecked={(config?.execution as Record<string, unknown>)?.self_healing_enabled !== false}
              />
              <ToggleSetting 
                label="Deep Web Crawl" 
                description="Enable recursive scraping for research tasks (consumes more tokens)."
                defaultChecked={false}
              />
              <ToggleSetting 
                label="Memory Consolidation" 
                description="Periodically compress episodic memory into semantic concepts."
                defaultChecked={true}
              />
              
              <div className="pt-4 border-t border-zinc-800/50">
                <label className="block text-sm font-medium text-zinc-300 mb-2">
                  Max Swarm Agents
                  <span className="float-right text-zinc-500">4</span>
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  defaultValue="4"
                  className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-zinc-300 mb-2">
                  Confidence Threshold
                  <span className="float-right text-zinc-500">0.85</span>
                </label>
                <input 
                  type="range" 
                  min="0.5" 
                  max="0.99" 
                  step="0.01"
                  defaultValue="0.85"
                  className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
                <p className="text-xs text-zinc-500 mt-2">Minimum ETS score required before acting autonomously.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfigInput({ label, type, placeholder, defaultValue }: { label: string, type: string, placeholder: string, defaultValue: string }) {
  return (
    <div>
      <label className="block text-sm font-medium text-zinc-300 mb-1.5">{label}</label>
      <input 
        type={type} 
        placeholder={placeholder}
        defaultValue={defaultValue}
        className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2 px-3 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 font-mono text-sm"
      />
    </div>
  );
}

function ToggleSetting({ label, description, defaultChecked }: { label: string, description: string, defaultChecked: boolean }) {
  const [checked, setChecked] = useState(defaultChecked);
  
  return (
    <div className="flex items-start justify-between">
      <div className="pr-4">
        <label className="text-sm font-medium text-zinc-200 cursor-pointer" onClick={() => setChecked(!checked)}>
          {label}
        </label>
        <p className="text-xs text-zinc-500 mt-0.5 leading-relaxed">{description}</p>
      </div>
      <button 
        onClick={() => setChecked(!checked)}
        className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${checked ? 'bg-purple-500' : 'bg-zinc-700'}`}
      >
        <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${checked ? 'translate-x-4' : 'translate-x-0'}`} />
      </button>
    </div>
  );
}
