import React, { useState } from 'react';
import Layout from './components/layout';
import Dashboard from './components/Dashboard';
import SwarmControl from './components/SwarmControl';
import CognitiveMemory from './components/CognitiveMemory';
import DeepResearch from './components/DeepResearch';
import SelfHealing from './components/SelfHealing';
import SystemConfig from './components/SystemConfig';
import BudgetManager from './components/BudgetManager';
import SessionManager from './components/SessionManager';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'sessions':
        return <SessionManager />;
      case 'swarm':
        return <SwarmControl />;
      case 'memory':
        return <CognitiveMemory />;
      case 'research':
        return <DeepResearch />;
      case 'healing':
        return <SelfHealing />;
      case 'budget':
        return <BudgetManager />;
      case 'settings':
        return <SystemConfig />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
      {renderContent()}
    </Layout>
  );
}
