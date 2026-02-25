'use client';

import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { SystemStatus } from '@/components/dashboard/SystemStatus';
import { BudgetGauge } from '@/components/dashboard/BudgetGauge';
import { ProviderHealth } from '@/components/dashboard/ProviderHealth';
import { RecentEvents } from '@/components/dashboard/RecentEvents';
import { ActiveSessions } from '@/components/dashboard/ActiveSessions';

export default function DashboardPage() {
  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Dashboard" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <SystemStatus />
            <BudgetGauge />
            <ProviderHealth />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <RecentEvents />
            <ActiveSessions />
          </div>
        </main>
      </div>
    </div>
  );
}
