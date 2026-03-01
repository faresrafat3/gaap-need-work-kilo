'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, CheckCircle2, AlertCircle, WifiOff, Clock } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { Skeleton } from '@/components/ui/skeleton'
import { ProviderInfo } from '@/hooks/useLiveProviders'

interface ProviderStatusBadgeProps {
  provider: ProviderInfo
  isRefreshing?: boolean
  size?: 'sm' | 'md' | 'lg'
  showTooltip?: boolean
}

const statusConfig = {
  active: {
    color: 'bg-emerald-500',
    borderColor: 'border-emerald-500/30',
    textColor: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10',
    icon: CheckCircle2,
    label: 'نشط',
    pulse: true,
  },
  error: {
    color: 'bg-red-500',
    borderColor: 'border-red-500/30',
    textColor: 'text-red-500',
    bgColor: 'bg-red-500/10',
    icon: AlertCircle,
    label: 'خطأ',
    pulse: false,
  },
  offline: {
    color: 'bg-gray-500',
    borderColor: 'border-gray-500/30',
    textColor: 'text-gray-500',
    bgColor: 'bg-gray-500/10',
    icon: WifiOff,
    label: 'غير متصل',
    pulse: false,
  },
  unknown: {
    color: 'bg-gray-500',
    borderColor: 'border-gray-500/30',
    textColor: 'text-gray-500',
    bgColor: 'bg-gray-500/10',
    icon: WifiOff,
    label: 'غير معروف',
    pulse: false,
  },
}

function formatTimeAgo(dateString: string | null): string {
  if (!dateString) return 'غير معروف'
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'الآن'
  if (diffMins < 60) return `منذ ${diffMins} دقيقة`
  if (diffHours < 24) return `منذ ${diffHours} ساعة`
  return `منذ ${diffDays} يوم`
}

function formatLatency(ms: number | null): string {
  if (ms === null) return 'N/A'
  if (ms < 100) return `${ms}ms`
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function ProviderStatusBadge({
  provider,
  isRefreshing = false,
  size = 'md',
  showTooltip = true,
}: ProviderStatusBadgeProps) {
  const config = statusConfig[provider.status] || statusConfig.unknown
  const StatusIcon = config.icon

  const sizeClasses = {
    sm: 'h-5 text-[10px] px-1.5 gap-1',
    md: 'h-6 text-xs px-2 gap-1.5',
    lg: 'h-7 text-sm px-2.5 gap-2',
  }

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-3.5 h-3.5',
    lg: 'w-4 h-4',
  }

  const pulseSizes = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-2.5 h-2.5',
  }

  const badge = (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.2 }}
    >
      <Badge
        variant="outline"
        className={`
          ${sizeClasses[size]}
          ${config.bgColor} ${config.borderColor} ${config.textColor}
          font-medium cursor-help transition-all duration-200
          hover:shadow-md hover:scale-105
        `}
      >
        <AnimatePresence mode="wait">
          {isRefreshing ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0, rotate: -180 }}
              animate={{ opacity: 1, rotate: 0 }}
              exit={{ opacity: 0, rotate: 180 }}
              transition={{ duration: 0.3 }}
            >
              <Loader2 className={`${iconSizes[size]} animate-spin`} />
            </motion.div>
          ) : (
            <motion.div
              key="status"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5"
            >
              {config.pulse && (
                <span className={`relative flex ${pulseSizes[size]}`}>
                  <span
                    className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.color} opacity-75`}
                  />
                  <span
                    className={`relative inline-flex rounded-full ${pulseSizes[size]} ${config.color}`}
                  />
                </span>
              )}
              {!config.pulse && (
                <StatusIcon className={iconSizes[size]} />
              )}
              <span className="truncate max-w-[120px]">{provider.actual_model || 'unknown'}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </Badge>
    </motion.div>
  )

  if (!showTooltip) {
    return badge
  }

  // Check if provider is stale
  const providerWithStale = provider as ProviderInfo & { isStale?: boolean }

  return (
    <TooltipProvider delayDuration={100}>
      <Tooltip>
        <TooltipTrigger asChild>{badge}</TooltipTrigger>
        <TooltipContent
          side="bottom"
          className="p-0 bg-popover border shadow-xl rounded-lg overflow-hidden max-w-[280px]"
        >
          <div className="p-3 space-y-3">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`p-1.5 rounded-md ${config.bgColor}`}>
                  <StatusIcon className={`w-4 h-4 ${config.textColor}`} />
                </div>
                <div>
                  <p className="font-semibold text-sm">{provider.display_name}</p>
                  <p className="text-xs text-muted-foreground capitalize">
                    {provider.name}
                  </p>
                </div>
              </div>
              <Badge
                variant="outline"
                className={`${config.bgColor} ${config.borderColor} ${config.textColor} text-[10px]`}
              >
                {config.label}
              </Badge>
            </div>

            {/* Divider */}
            <div className="h-px bg-border" />

            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="space-y-1">
                <p className="text-muted-foreground flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  التأخير
                </p>
                <p className="font-medium font-mono">
                  {formatLatency(provider.latency_ms)}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-muted-foreground flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" />
                  معدل النجاح
                </p>
                <p className="font-medium font-mono">
                  {provider.success_rate.toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Model Info */}
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">النموذج الحالي</p>
              <p className="text-xs font-medium bg-muted px-2 py-1 rounded">
                {provider.actual_model || 'unknown'}
              </p>
            </div>

            {/* Last Seen */}
            <div className="flex items-center justify-between text-[10px] text-muted-foreground">
              <span>آخر تحديث:</span>
              <span className={providerWithStale.isStale ? 'text-amber-500 font-medium' : ''}>
                {formatTimeAgo(provider.last_seen)}
                {providerWithStale.isStale && ' ⚠'}
              </span>
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export function ProviderStatusBadgeSkeleton({
  size = 'md',
}: {
  size?: 'sm' | 'md' | 'lg'
}) {
  const sizeClasses = {
    sm: 'h-5 w-20',
    md: 'h-6 w-24',
    lg: 'h-7 w-28',
  }

  return <Skeleton className={`${sizeClasses[size]} rounded-full`} />
}
