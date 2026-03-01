'use client'

import { useEffect, useState } from 'react'
import { AlertCircle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ErrorBoundaryProps {
  error: Error & { digest?: string }
  reset: () => void
}

export default function ErrorBoundary({ error, reset }: ErrorBoundaryProps) {
  const [isRetrying, setIsRetrying] = useState(false)

  useEffect(() => {
    // Log error to monitoring service
    console.error('Application error:', error)
  }, [error])

  const handleReset = () => {
    setIsRetrying(true)
    reset()
    setTimeout(() => setIsRetrying(false), 1000)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4" dir="rtl">
      <div className="max-w-md w-full space-y-6 text-center">
        <div className="flex justify-center">
          <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-500" />
          </div>
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">
            حدث خطأ غير متوقع
          </h1>
          <p className="text-muted-foreground">
            عذراً، حدث خطأ في التطبيق. يرجى المحاولة مرة أخرى.
          </p>
        </div>

        {process.env.NODE_ENV === 'development' && (
          <div className="bg-muted rounded-lg p-4 text-left overflow-auto max-h-48">
            <p className="text-xs font-mono text-red-500">
              {error.message}
            </p>
            {error.digest && (
              <p className="text-xs font-mono text-muted-foreground mt-2">
                Digest: {error.digest}
              </p>
            )}
          </div>
        )}

        <div className="flex gap-3 justify-center">
          <Button
            onClick={handleReset}
            disabled={isRetrying}
            className="gap-2"
          >
            <RefreshCw className={`w-4 h-4 ${isRetrying ? 'animate-spin' : ''}`} />
            {isRetrying ? 'جاري المحاولة...' : 'إعادة المحاولة'}
          </Button>
          
          <Button
            variant="outline"
            onClick={() => window.location.href = '/'}
          >
            الصفحة الرئيسية
          </Button>
        </div>

        <p className="text-xs text-muted-foreground">
          Error Boundary • {new Date().toLocaleString('ar-SA')}
        </p>
      </div>
    </div>
  )
}
