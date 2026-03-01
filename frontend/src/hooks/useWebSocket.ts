'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

interface UseWebSocketOptions {
  url: string
  onMessage?: (data: any) => void
  onConnect?: () => void
  onDisconnect?: () => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

interface UseWebSocketReturn {
  sendMessage: (data: any) => void
  status: 'connecting' | 'connected' | 'disconnected'
  reconnect: () => void
}

const DEFAULT_RECONNECT_ATTEMPTS = 5
const DEFAULT_RECONNECT_INTERVAL = 1000
const MAX_RECONNECT_INTERVAL = 30000

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    reconnectAttempts = DEFAULT_RECONNECT_ATTEMPTS,
    reconnectInterval = DEFAULT_RECONNECT_INTERVAL,
  } = options

  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting')
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isUnmountingRef = useRef(false)

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const getBackoffDelay = useCallback(
    (attempt: number): number => {
      const delay = reconnectInterval * Math.pow(2, attempt)
      return Math.min(delay, MAX_RECONNECT_INTERVAL)
    },
    [reconnectInterval]
  )

  const connect = useCallback(() => {
    if (isUnmountingRef.current) return

    clearReconnectTimer()

    try {
      setStatus('connecting')
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (isUnmountingRef.current) {
          ws.close()
          return
        }
        reconnectCountRef.current = 0
        setStatus('connected')
        onConnect?.()
      }

      ws.onmessage = (event) => {
        if (isUnmountingRef.current) return
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch {
          onMessage?.(event.data)
        }
      }

      ws.onclose = () => {
        if (isUnmountingRef.current) return
        setStatus('disconnected')
        onDisconnect?.()

        if (reconnectCountRef.current < reconnectAttempts) {
          const delay = getBackoffDelay(reconnectCountRef.current)
          reconnectCountRef.current++
          reconnectTimerRef.current = setTimeout(() => {
            connect()
          }, delay)
        }
      }

      ws.onerror = () => {
        if (isUnmountingRef.current) return
      }
    } catch {
      setStatus('disconnected')
      if (reconnectCountRef.current < reconnectAttempts) {
        const delay = getBackoffDelay(reconnectCountRef.current)
        reconnectCountRef.current++
        reconnectTimerRef.current = setTimeout(() => {
          connect()
        }, delay)
      }
    }
  }, [url, onMessage, onConnect, onDisconnect, reconnectAttempts, getBackoffDelay, clearReconnectTimer])

  const sendMessage = useCallback((data: any) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data)
      ws.send(message)
    }
  }, [])

  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    reconnectCountRef.current = 0
    clearReconnectTimer()
    connect()
  }, [connect, clearReconnectTimer])

  useEffect(() => {
    isUnmountingRef.current = false
    connect()

    return () => {
      isUnmountingRef.current = true
      clearReconnectTimer()
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect, clearReconnectTimer])

  return {
    sendMessage,
    status,
    reconnect,
  }
}
