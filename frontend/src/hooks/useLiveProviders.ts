'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { ProviderInfoSchema, safeJsonParse } from '@/lib/validation'
import { z } from 'zod'

export type ProviderInfo = z.infer<typeof ProviderInfoSchema>

interface UseLiveProvidersReturn {
  providers: ProviderInfo[]
  loading: boolean
  error: string | null
  lastUpdated: Date | null
  refresh: () => Promise<void>
  isRefreshing: boolean
}

const REFRESH_INTERVAL = 30000 // 30 seconds
const STALE_THRESHOLD = 5 * 60 * 1000 // 5 minutes
const MAX_RETRY_ATTEMPTS = 5

const getRetryDelay = (attempt: number): number => {
  return Math.min(1000 * Math.pow(2, attempt), 30000) // Exponential backoff, max 30s
}

const transformApiResponse = (data: unknown[]): ProviderInfo[] => {
  const now = new Date()

  return data.map((provider: any) => {
    const lastSeen = new Date(provider.last_seen || provider.lastSeen || provider.updated_at || now)
    const isStale = now.getTime() - lastSeen.getTime() > STALE_THRESHOLD

    return {
      name: provider.name || provider.id || 'unknown',
      display_name: provider.display_name || provider.displayName || provider.name || provider.id || 'Unknown',
      actual_model: provider.actual_model || provider.actualModel || provider.model || null,
      default_model: provider.default_model || provider.defaultModel || provider.model || 'unknown',
      status: mapStatus(provider.status),
      last_seen: lastSeen.toISOString(),
      latency_ms: provider.latency_ms || provider.latencyMs || provider.latency || 0,
      success_rate: provider.success_rate || provider.successRate || provider.success_rate || 0,
      accounts_count: provider.accounts_count || provider.accountsCount || 0,
      healthy_accounts: provider.healthy_accounts || provider.healthyAccounts || 0,
      models_available: provider.models_available || provider.modelsAvailable || [],
      provider_type: provider.provider_type || provider.providerType || 'unknown',
      error_message: provider.error_message || provider.errorMessage || null,
      cached: provider.cached || false,
      cache_age_seconds: provider.cache_age_seconds || provider.cacheAgeSeconds || null,
      isStale,
    } as ProviderInfo
  })
}

const mapStatus = (status: string): 'active' | 'error' | 'offline' => {
  if (status === 'active' || status === 'online' || status === 'healthy') {
    return 'active'
  }
  if (status === 'error' || status === 'unhealthy' || status === 'down' || status === 'failed') {
    return 'error'
  }
  return 'offline'
}

export function useLiveProviders(): UseLiveProvidersReturn {
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const abortControllerRef = useRef<AbortController | null>(null)
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const retryAttemptRef = useRef(0)
  const isMountedRef = useRef(true)

  const fetchProviders = useCallback(async (isManualRefresh = false) => {
    // Cancel any in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }

    abortControllerRef.current = new AbortController()

    if (isManualRefresh) {
      setIsRefreshing(true)
    }

    try {
      const response = await fetch('/api/providers/live', {
        signal: abortControllerRef.current.signal,
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Handle different API response structures
      const providersData = data.providers || data.data || data || []

      if (!Array.isArray(providersData)) {
        throw new Error('Invalid response format: expected array of providers')
      }

      const transformedProviders = transformApiResponse(providersData)

      if (isMountedRef.current) {
        setProviders(transformedProviders)
        setLastUpdated(new Date())
        setError(null)
        retryAttemptRef.current = 0
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return // Request was cancelled, don't update state
      }

      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch providers'

      if (isMountedRef.current) {
        setError(errorMessage)

        // Retry with exponential backoff
        if (retryAttemptRef.current < MAX_RETRY_ATTEMPTS) {
          const delay = getRetryDelay(retryAttemptRef.current)
          retryTimeoutRef.current = setTimeout(() => {
            if (isMountedRef.current) {
              retryAttemptRef.current += 1
              fetchProviders()
            }
          }, delay)
        }
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
        setIsRefreshing(false)
      }
    }
  }, [])

  const refresh = useCallback(async () => {
    retryAttemptRef.current = 0
    await fetchProviders(true)
  }, [fetchProviders])

  useEffect(() => {
    isMountedRef.current = true

    // Initial fetch
    fetchProviders()

    // Set up auto-refresh interval
    intervalRef.current = setInterval(() => {
      fetchProviders()
    }, REFRESH_INTERVAL)

    return () => {
      isMountedRef.current = false

      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }

      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current)
      }

      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [fetchProviders])

  // Update stale status periodically without refetching
  useEffect(() => {
    const staleInterval = setInterval(() => {
      if (providers.length > 0) {
        setProviders((current) =>
          current.map((provider) => {
            const lastSeen = new Date(provider.last_seen || Date.now())
            const isStale = Date.now() - lastSeen.getTime() > STALE_THRESHOLD
            return { ...provider, isStale }
          })
        )
      }
    }, 60000) // Check every minute

    return () => clearInterval(staleInterval)
  }, [providers.length])

  return {
    providers,
    loading,
    error,
    lastUpdated,
    refresh,
    isRefreshing,
  }
}
