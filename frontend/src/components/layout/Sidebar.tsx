'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Settings, 
  Server, 
  Search, 
  Heart, 
  Database, 
  DollarSign, 
  Shield, 
  Activity,
  Bug,
  Moon
} from 'lucide-react';
import clsx from 'clsx';
import { LayerNavigation } from '@/components/layer-nav/LayerNavigation';
import { useState } from 'react';

const navItems = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Config', href: '/config', icon: Settings },
  { name: 'Providers', href: '/providers', icon: Server },
  { name: 'Research', href: '/research', icon: Search },
  { name: 'Sessions', href: '/sessions', icon: Activity },
  { name: 'Healing', href: '/healing', icon: Heart },
  { name: 'Memory', href: '/memory', icon: Database },
  { name: 'Dream', href: '/dream', icon: Moon },
  { name: 'Budget', href: '/budget', icon: DollarSign },
  { name: 'Security', href: '/security', icon: Shield },
  { name: 'Debt', href: '/debt', icon: Bug },
];

export function Sidebar() {
  const pathname = usePathname();
  const [currentLayer, setCurrentLayer] = useState<'strategy' | 'tactics' | 'execution'>('strategy');

  return (
    <aside className="w-64 bg-cyber-darker border-r border-layer1/30 flex flex-col">
      <div className="p-4 border-b border-layer1/30">
        <h1 className="text-2xl font-bold text-cyber-primary font-mono">
          GAAP
        </h1>
        <p className="text-xs text-gray-500 mt-1">Control Interface</p>
      </div>
      
      <div className="p-3">
        <LayerNavigation
          currentLayer={currentLayer}
          onLayerSelect={setCurrentLayer}
          compact
        />
      </div>
      
      <nav className="flex-1 p-2">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-lg transition-all',
                    isActive
                      ? 'bg-layer1/20 text-cyber-primary glow-layer1'
                      : 'text-gray-400 hover:text-white hover:bg-layer1/10'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-layer1/30">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          <span className="text-xs text-gray-500">System Online</span>
        </div>
      </div>
    </aside>
  );
}