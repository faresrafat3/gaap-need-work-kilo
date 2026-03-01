'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChevronDown,
  Bot,
  Zap,
  Globe,
  RefreshCw,
  AlertCircle,
  Loader2,
  WifiOff,
  CheckCircle2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useLiveProviders, ProviderInfo } from '@/hooks/useLiveProviders'

const getProviderIcon = (name: string) => {
  switch (name.toLowerCase()) {
    case 'kimi':
      return <Bot className="w-4 h-4" />
    case 'deepseek':
      return <Zap className="w-4 h-4" />
    case 'glm':
      return <Globe className="w-4 h-4" />
    default:
      return <Bot className="w-4 h-4" />
  }
}

const getStatusIcon = (status: ProviderInfo['status']) => {
  switch (status) {
    case 'active':
      return (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
        </span>
      )
    case 'error':
      return <AlertCircle className="w-3.5 h-3.5 text-red-500" />
    case 'offline':
      return <WifiOff className="w-3.5 h-3.5 text-gray-500" />
  }
}

const getStatusColor = (status: ProviderInfo['status']) => {
  switch (status) {
    case 'active':
      return 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
    case 'error':
      return 'bg-red-500/10 text-red-500 border-red-500/20'
    case 'offline':
      return 'bg-gray-500/10 text-gray-500 border-gray-500/20'
  }
}

export function ProviderSelector() {
  const { providers, loading, error, refresh, isRefreshing } = useLiveProviders()
  const [selected, setSelected] = useState<string>('')
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Load saved preference and set initial selection
  useEffect(() => {
    const saved = localStorage.getItem('gaap_provider')
    if (saved) {
      setSelected(saved)
    } else if (providers.length > 0) {
      // Select first active provider by default
      const firstActive = providers.find((p) => p.status === 'active')
      setSelected(firstActive?.name || providers[0].name)
    }
  }, [providers])

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleSelect = (providerName: string) => {
    setSelected(providerName)
    localStorage.setItem('gaap_provider', providerName)
    setIsOpen(false)
  }

  const selectedProvider = providers.find((p) => p.name === selected)

  // Loading state
  if (loading && providers.length === 0) {
    return (
      <div className="flex items-center gap-2">
        <Skeleton className="h-9 w-32 rounded-md" />
      </div>
    )
  }

  // Error state
  if (error && providers.length === 0) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              className="gap-2 border-red-500/30 bg-red-500/10 text-red-500 hover:bg-red-500/20"
            >
              <AlertCircle className="w-4 h-4" />
              <span className="text-xs">خطأ في الاتصال</span>
              <RefreshCw
                className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
              />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p className="text-xs">انقر لإعادة المحاولة</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  const activeProviders = providers.filter((p) => p.status === 'active')

  // Type assertion to access isStale property
  const providerWithStale = selectedProvider as ProviderInfo & { isStale?: boolean }

  return (
    <div className="relative" ref={dropdownRef}>
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsOpen(!isOpen)}
          className={`
            gap-2 min-w-[140px] justify-between
            ${selectedProvider?.status === 'error' ? 'border-red-500/30' : ''}
            ${selectedProvider?.status === 'offline' ? 'border-gray-500/30' : ''}
          `}
        >
          <div className="flex items-center gap-2">
            {selectedProvider ? (
              <>
                {getProviderIcon(selectedProvider.name)}
                <span className="capitalize">{selectedProvider.name}</span>
                <Badge
                  variant="outline"
                  className={`text-[10px] px-1.5 py-0 ${getStatusColor(
                    selectedProvider.status
                  )}`}
                >
                  {providerWithStale?.isStale ? (
                    <span className="flex items-center gap-1">
                      <WifiOff className="w-3 h-3" />
                      قديم
                    </span>
                  ) : (
                    (selectedProvider.actual_model || 'unknown').slice(0, 8) + '...'
                  )}
                </Badge>
              </>
            ) : (
              <span className="text-muted-foreground">اختر المزود</span>
            )}
          </div>
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-4 h-4 opacity-50" />
          </motion.div>
        </Button>
      </motion.div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="absolute top-full mt-2 right-0 z-50 w-72 bg-popover border rounded-lg shadow-2xl overflow-hidden"
          >
            {/* Header */}
            <div className="px-3 py-2 border-b bg-muted/50 flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                المزودين المتاحين ({activeProviders.length})
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={(e) => {
                  e.stopPropagation()
                  refresh()
                }}
                disabled={isRefreshing}
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
                />
              </Button>
            </div>

            {/* Provider List */}
            <div className="max-h-[300px] overflow-y-auto py-1">
              {providers.length === 0 ? (
                <div className="px-3 py-4 text-center text-sm text-muted-foreground">
                  لا يوجد مزودين متاحين
                </div>
              ) : (
                providers.map((provider) => (
                  <motion.button
                    key={provider.name}
                    onClick={() => handleSelect(provider.name)}
                    whileHover={{ backgroundColor: 'hsl(var(--accent))' }}
                    className={`
                      w-full flex items-center gap-3 px-3 py-2.5 text-sm
                      transition-colors border-l-2
                      ${selected === provider.name ? 'bg-accent border-l-primary' : 'border-l-transparent'}
                      ${provider.status !== 'active' ? 'opacity-60' : ''}
                    `}
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {getProviderIcon(provider.name)}
                      <div className="flex flex-col items-start min-w-0">
                        <span className="capitalize font-medium truncate">
                          {provider.name}
                        </span>
                        <span className="text-[10px] text-muted-foreground truncate max-w-[120px]">
                          {provider.actual_model || 'unknown'}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Status indicator */}
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-center w-5">
                              {getStatusIcon(provider.status)}
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="left">
                            <p className="text-xs">
                              {provider.status === 'active'
                                ? 'نشط'
                                : provider.status === 'error'
                                ? 'خطأ'
                                : 'غير متصل'}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>

                      {/* Latency */}
                      {provider.status === 'active' && (
                        <Badge
                          variant="outline"
                          className={`
                            text-[10px] px-1.5 py-0 font-mono
                            ${
                              (provider.latency_ms || 0) < 200
                                ? 'bg-emerald-500/10 text-emerald-500'
                                : (provider.latency_ms || 0) < 500
                                ? 'bg-amber-500/10 text-amber-500'
                                : 'bg-red-500/10 text-red-500'
                            }
                          `}
                        >
                          {provider.latency_ms}ms
                        </Badge>
                      )}
                    </div>
                  </motion.button>
                ))
              )}
            </div>

            {/* Footer with last updated */}
            <div className="px-3 py-2 border-t bg-muted/30 text-[10px] text-muted-foreground flex items-center justify-between">
              <span>
                {isRefreshing ? (
                  <span className="flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    جاري التحديث...
                  </span>
                ) : (
                  `تم التحديث: ${new Date().toLocaleTimeString('ar-SA', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}`
                )}
              </span>
              {error && (
                <span className="text-red-500 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  خطأ
                </span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
