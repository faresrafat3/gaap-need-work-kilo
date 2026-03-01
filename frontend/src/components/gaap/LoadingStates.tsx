'use client'

import { motion } from 'framer-motion'
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'

// Skeleton for cards
export function CardSkeleton({ className = '' }: { className?: string }) {
  return (
    <div className={`animate-pulse space-y-3 ${className}`}>
      <div className="h-4 bg-muted rounded w-1/4" />
      <div className="h-20 bg-muted rounded" />
      <div className="h-4 bg-muted rounded w-3/4" />
    </div>
  )
}

// Skeleton for messages
export function MessageSkeleton() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex gap-3"
    >
      <div className="w-8 h-8 rounded-full bg-muted animate-pulse" />
      <div className="flex-1 space-y-2">
        <div className="h-16 bg-muted rounded-lg" />
        <div className="h-4 bg-muted rounded w-1/2" />
      </div>
    </motion.div>
  )
}

// Loading spinner
export function LoadingSpinner({ 
  size = 'md', 
  text = 'جاري التحميل...',
  fullScreen = false 
}: { 
  size?: 'sm' | 'md' | 'lg'
  text?: string
  fullScreen?: boolean
}) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8', 
    lg: 'w-12 h-12'
  }
  
  const content = (
    <div className="flex flex-col items-center justify-center gap-3">
      <Loader2 className={`${sizeClasses[size]} animate-spin text-primary`} />
      {text && <p className="text-sm text-muted-foreground">{text}</p>}
    </div>
  )
  
  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-background/80 flex items-center justify-center z-50">
        {content}
      </div>
    )
  }
  
  return content
}

// Error state with retry
export function ErrorState({ 
  title = 'حدث خطأ',
  message,
  onRetry,
  className = '' 
}: { 
  title?: string
  message?: string
  onRetry?: () => void
  className?: string
}) {
  return (
    <Card className={className}>
      <CardContent className="flex flex-col items-center justify-center py-8 text-center">
        <AlertCircle className="w-12 h-12 text-destructive mb-4" />
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        {message && (
          <p className="text-sm text-muted-foreground mb-4 max-w-md">
            {message}
          </p>
        )}
        {onRetry && (
          <Button onClick={onRetry} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            إعادة المحاولة
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

// Empty state
export function EmptyState({ 
  icon,
  title,
  description,
  action,
  className = '' 
}: { 
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 text-center ${className}`}>
      {icon && <div className="mb-4 text-muted-foreground">{icon}</div>}
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      {description && (
        <p className="text-sm text-muted-foreground mb-4 max-w-md">
          {description}
        </p>
      )}
      {action}
    </div>
  )
}

// Skeleton for table rows
export function TableRowSkeleton({ columns = 4 }: { columns?: number }) {
  return (
    <div className="flex items-center gap-4 p-4 border-b">
      {Array.from({ length: columns }).map((_, i) => (
        <div key={i} className="h-4 bg-muted animate-pulse rounded" style={{ width: `${20 + Math.random() * 60}%` }} />
      ))}
    </div>
  )
}

// Skeleton for stats card
export function StatCardSkeleton() {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <div className="h-3 w-16 bg-muted animate-pulse rounded" />
          <div className="h-8 w-24 bg-muted animate-pulse rounded" />
        </div>
        <div className="h-10 w-10 bg-muted animate-pulse rounded-full" />
      </div>
    </Card>
  )
}

// Skeleton for code block
export function CodeBlockSkeleton() {
  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <div className="h-6 w-16 bg-muted animate-pulse rounded" />
        <div className="h-6 w-20 bg-muted animate-pulse rounded" />
      </div>
      <div className="space-y-1">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-4 bg-muted animate-pulse rounded" style={{ width: `${60 + Math.random() * 40}%` }} />
        ))}
      </div>
    </div>
  )
}
