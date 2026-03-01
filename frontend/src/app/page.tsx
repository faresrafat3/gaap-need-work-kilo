'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  Settings,
  History,
  Menu,
  X,
  Sparkles,
  ChevronRight,
  Shield,
  Network,
  BookOpen,
  Wrench,
  Bot
} from 'lucide-react'
import { Dashboard } from '@/components/gaap/Dashboard'
import { ChatInterface } from '@/components/gaap/ChatInterface'
import { ResearchModule } from '@/components/gaap/ResearchModule'
import { ConfigurationPanel } from '@/components/gaap/ConfigurationPanel'
import { SessionsManagement } from '@/components/gaap/SessionsManagement'
import { ValidatorsModule } from '@/components/gaap/ValidatorsModule'
import { ContextModule } from '@/components/gaap/ContextModule'
import { KnowledgeModule } from '@/components/gaap/KnowledgeModule'
import { MaintenanceModule } from '@/components/gaap/MaintenanceModule'
import { SwarmModule } from '@/components/gaap/SwarmModule'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/gaap/ThemeToggle'
import { GlobalSearch } from '@/components/gaap/GlobalSearch'
import { cn } from '@/lib/utils'

type Tab = 'dashboard' | 'chat' | 'research' | 'validators' | 'context' | 'knowledge' | 'maintenance' | 'swarm' | 'sessions' | 'config'

const tabs: { id: Tab; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'لوحة التحكم', icon: LayoutDashboard },
  { id: 'chat', label: 'المحادثة', icon: MessageSquare },
  { id: 'research', label: 'البحث', icon: Search },
  { id: 'validators', label: 'الفحص', icon: Shield },
  { id: 'context', label: 'السياق', icon: Network },
  { id: 'knowledge', label: 'المعرفة', icon: BookOpen },
  { id: 'maintenance', label: 'الصيانة', icon: Wrench },
  { id: 'swarm', label: 'السرب', icon: Bot },
  { id: 'sessions', label: 'الجلسات', icon: History },
  { id: 'config', label: 'الإعدادات', icon: Settings },
]

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />
      case 'chat':
        return <ChatInterface />
      case 'research':
        return <ResearchModule />
      case 'validators':
        return <ValidatorsModule />
      case 'context':
        return <ContextModule />
      case 'knowledge':
        return <KnowledgeModule />
      case 'maintenance':
        return <MaintenanceModule />
      case 'swarm':
        return <SwarmModule />
      case 'sessions':
        return <SessionsManagement />
      case 'config':
        return <ConfigurationPanel />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-card/80 backdrop-blur-xl border-b border-primary/10">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold gradient-text">GAAP</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="lg:hidden fixed top-16 left-0 right-0 z-40 bg-card/95 backdrop-blur-xl border-b border-primary/10 p-4"
          >
            <div className="grid grid-cols-5 gap-2">
              {tabs.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => {
                      setActiveTab(tab.id)
                      setMobileMenuOpen(false)
                    }}
                    className={cn(
                      "flex flex-col items-center gap-1 p-2 rounded-lg transition-all",
                      activeTab === tab.id
                        ? "bg-primary/20 text-primary"
                        : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                    )}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="text-xs">{tab.label}</span>
                  </button>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex min-h-screen">
        {/* Desktop Sidebar */}
        <motion.aside
          initial={false}
          animate={{ width: sidebarOpen ? 240 : 72 }}
          className="hidden lg:flex flex-col fixed right-0 top-0 h-screen bg-card border-l border-border z-40 shrink-0"
        >
          {/* Logo */}
          <div className="p-4 border-b border-border h-16">
            <div className="flex items-center gap-3">
              <motion.div
                className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center flex-shrink-0"
                whileHover={{ scale: 1.05 }}
              >
                <Sparkles className="w-5 h-5 text-white" />
              </motion.div>
              <AnimatePresence>
                {sidebarOpen && (
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    className="flex flex-col"
                  >
                    <span className="font-bold text-lg gradient-text">GAAP</span>
                    <span className="text-xs text-muted-foreground">v0.9.0</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all relative",
                    activeTab === tab.id
                      ? "bg-primary/20 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                  )}
                  whileHover={{ x: -2 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  <AnimatePresence>
                    {sidebarOpen && (
                      <motion.span
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -10 }}
                        className="text-sm font-medium whitespace-nowrap"
                      >
                        {tab.label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                  {activeTab === tab.id && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-primary rounded-r-full"
                    />
                  )}
                </motion.button>
              )
            })}
          </nav>

          {/* Bottom Section */}
          <div className="p-3 border-t border-border space-y-2">
            <GlobalSearch />
            <ThemeToggle />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="w-full justify-center"
            >
              <motion.div
                animate={{ rotate: sidebarOpen ? 0 : 180 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            </Button>
          </div>
        </motion.aside>

        {/* Spacer for Sidebar */}
        <div className={cn("hidden lg:block shrink-0 transition-all duration-300", sidebarOpen ? "w-[240px]" : "w-[72px]")} />

        {/* Main Content */}
        <main className="flex-1 min-h-screen p-4 lg:p-6 pt-20 lg:pt-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-br from-primary/3 to-accent/3 rounded-full blur-3xl" />
      </div>
    </div>
  )
}
