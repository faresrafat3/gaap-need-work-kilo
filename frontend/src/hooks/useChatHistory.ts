'use client'

import { useEffect, useState } from 'react'
import { Message } from '@/lib/store'
import { safeJsonParse } from '@/lib/validation'

const STORAGE_KEY = 'gaap_chat_history'
const MAX_MESSAGES = 100
const STORAGE_VERSION = '1'

interface StorageData {
  version: string
  messages: Message[]
  updatedAt: string
}

function isStorageAvailable(): boolean {
  try {
    const storage = window.localStorage
    const test = '__storage_test__'
    storage.setItem(test, test)
    storage.removeItem(test)
    return true
  } catch {
    return false
  }
}

function migrateData(data: unknown): Message[] {
  // Handle legacy format (just array of messages)
  if (Array.isArray(data)) {
    return data.map((m: any) => ({
      ...m,
      timestamp: new Date(m.timestamp),
    }))
  }

  // Handle versioned format
  const versioned = data as StorageData | null
  if (versioned?.version === STORAGE_VERSION && Array.isArray(versioned.messages)) {
    return versioned.messages.map((m: any) => ({
      ...m,
      timestamp: new Date(m.timestamp),
    }))
  }

  return []
}

export function useChatHistory() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoaded, setIsLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!isStorageAvailable()) {
      setError('localStorage not available')
      setIsLoaded(true)
      return
    }

    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const parsed = safeJsonParse<unknown>(saved, null)
        if (parsed) {
          const migrated = migrateData(parsed)
          setMessages(migrated)
        }
      }
    } catch (e) {
      console.error('Failed to load chat history:', e)
      setError('Failed to load chat history')
    } finally {
      setIsLoaded(true)
    }
  }, [])

  const saveMessages = (msgs: Message[]) => {
    // Limit messages to prevent storage overflow
    const trimmed = msgs.slice(-MAX_MESSAGES)
    setMessages(trimmed)

    if (!isStorageAvailable()) {
      setError('localStorage not available')
      return
    }

    try {
      const data: StorageData = {
        version: STORAGE_VERSION,
        messages: trimmed,
        updatedAt: new Date().toISOString(),
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
      setError(null)
    } catch (e) {
      console.error('Failed to save chat history:', e)
      setError('Failed to save chat history')
      
      // If quota exceeded, try to clear some old data
      if (e instanceof Error && e.name === 'QuotaExceededError') {
        try {
          localStorage.clear()
          localStorage.setItem(STORAGE_KEY, JSON.stringify({
            version: STORAGE_VERSION,
            messages: trimmed.slice(-20), // Keep only last 20
            updatedAt: new Date().toISOString(),
          }))
        } catch {
          // Last resort: disable local storage
          console.error('Could not recover from quota exceeded')
        }
      }
    }
  }

  const clearHistory = () => {
    setMessages([])
    if (isStorageAvailable()) {
      try {
        localStorage.removeItem(STORAGE_KEY)
        setError(null)
      } catch (e) {
        console.error('Failed to clear chat history:', e)
      }
    }
  }

  const exportHistory = (): string => {
    const data: StorageData = {
      version: STORAGE_VERSION,
      messages,
      updatedAt: new Date().toISOString(),
    }
    return JSON.stringify(data, null, 2)
  }

  return { 
    messages, 
    saveMessages, 
    clearHistory,
    isLoaded,
    error,
    exportHistory,
  }
}
