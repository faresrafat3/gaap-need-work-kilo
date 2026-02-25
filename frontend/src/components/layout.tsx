import React from 'react';
import { 
  LayoutDashboard, 
  Network, 
  BrainCircuit, 
  Search, 
  Wrench, 
  Settings,
  Activity,
  DollarSign,
  Layers
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export default function Layout({ children, activeTab, setActiveTab }: LayoutProps) {
  const navItems = [
    { id: 'dashboard', label: 'Mission Control', icon: LayoutDashboard },
    { id: 'sessions', label: 'Sessions', icon: Layers },
    { id: 'swarm', label: 'Swarm Intelligence', icon: Network },
    { id: 'memory', label: 'Cognitive Memory', icon: BrainCircuit },
    { id: 'research', label: 'Deep Research', icon: Search },
    { id: 'healing', label: 'Self-Healing', icon: Wrench },
    { id: 'budget', label: 'Budget & Tokens', icon: DollarSign },
    { id: 'settings', label: 'System Config', icon: Settings },
  ];

  return (
    <div className="flex h-screen w-full overflow-hidden bg-zinc-950 text-zinc-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-zinc-800/50 bg-zinc-900/20 flex flex-col">
        <div className="h-16 flex items-center px-6 border-b border-zinc-800/50">
          <Activity className="w-6 h-6 text-emerald-500 mr-3" />
          <h1 className="font-bold text-lg tracking-tight">GAAP <span className="text-zinc-500 font-normal text-sm">v0.9</span></h1>
        </div>
        
        <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive 
                    ? 'bg-emerald-500/10 text-emerald-400' 
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
                }`}
              >
                <Icon className={`w-5 h-5 mr-3 ${isActive ? 'text-emerald-400' : 'text-zinc-500'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-zinc-800/50">
          <div className="flex items-center justify-between text-xs font-mono text-zinc-500">
            <span>API Connection</span>
            <span className="flex items-center text-emerald-500">
              <span className="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse"></span>
              Connected
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <header className="h-16 flex items-center justify-between px-8 border-b border-zinc-800/50 bg-zinc-950/50 backdrop-blur-sm z-10">
          <h2 className="text-lg font-semibold text-zinc-100 capitalize">
            {navItems.find(i => i.id === activeTab)?.label || activeTab}
          </h2>
          <div className="flex items-center space-x-4">
            <div className="px-3 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-xs font-mono text-zinc-400">
              OODA Loop: <span className="text-emerald-400">ACTIVE</span>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto p-8">
          {children}
        </div>
      </main>
    </div>
  );
}

