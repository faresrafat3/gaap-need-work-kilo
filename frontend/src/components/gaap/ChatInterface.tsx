'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Bot, User, Copy, Check, Code, Sparkles, Loader2, RefreshCw, AlertCircle, Clock, MessageSquare } from 'lucide-react'
import { useGAAPStore, Message } from '@/lib/store'
import { useChatHistory } from '@/hooks/useChatHistory'
import { useToast } from '@/hooks/use-toast'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { ProviderSelector } from './ProviderSelector'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const sanitizeInput = (input: string): string => {
  return input.trim().slice(0, 10000)
}

const REQUEST_TIMEOUT = 30000

export function ChatInterface() {
  const { messages: storedMessages, saveMessages, clearHistory } = useChatHistory()
  const { isStreaming, addMessage, setStreaming } = useGAAPStore()
  const { toast } = useToast()
  const messages = storedMessages
  const [input, setInput] = useState('')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isRetrying, setIsRetrying] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    if (input) setError(null)
  }, [input])

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setStreaming(false)
    }
  }

  const handleSend = async () => {
    const sanitized = sanitizeInput(input)
    if (!sanitized || isStreaming) return

    abortControllerRef.current = new AbortController()
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: sanitized,
      timestamp: new Date(),
    }
    addMessage(userMessage)
    setInput('')
    setError(null)
    setStreaming(true)

    const updatedMessages = [...messages, userMessage]
    saveMessages(updatedMessages)

    const selectedProvider = localStorage.getItem('gaap_provider') || 'kimi'

    try {
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('timeout')), REQUEST_TIMEOUT)
      })

      const fetchPromise = fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: updatedMessages,
          provider: selectedProvider
        }),
        signal: abortControllerRef.current.signal
      })

      const response = await Promise.race([fetchPromise, timeoutPromise]) as Response

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let assistantContent = ''
      const codeBlocks: { language: string; code: string }[] = []

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        codeBlocks: [],
      }
      addMessage(assistantMessage)

      if (!reader) {
        throw new Error('No response stream')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n').filter(Boolean)

        for (const line of lines) {
          try {
            const data = JSON.parse(line)
            if (data.type === 'content') {
              assistantContent += data.content
            } else if (data.type === 'code') {
              codeBlocks.push({ language: data.language, code: data.code })
            } else if (data.type === 'done') {
              break
            }
          } catch {
          }
        }

        const updatedMessage: Message = {
          ...assistantMessage,
          content: assistantContent,
          codeBlocks,
        }
        addMessage(updatedMessage)
      }

      const finalMessages = [...updatedMessages, {
        ...assistantMessage,
        content: assistantContent,
        codeBlocks,
      }]
      saveMessages(finalMessages)

    } catch (err: any) {
      let errorMessage = 'حدث خطأ غير متوقع'
      
      if (err.name === 'AbortError') {
        errorMessage = 'تم إلغاء الطلب'
      } else if (err.message === 'timeout') {
        errorMessage = 'انتهت مهلة الطلب. يرجى المحاولة مرة أخرى.'
      } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        errorMessage = 'فشل في الاتصال بالخادم. تأكد من تشغيل الخادم.'
      } else {
        errorMessage = err.message || 'حدث خطأ في المعالجة'
      }

      toast({
        title: 'خطأ في المحادثة',
        description: errorMessage,
        variant: 'destructive',
      })
      
      setError(errorMessage)
      
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ ${errorMessage}`,
        timestamp: new Date(),
      }
      addMessage(errorMsg)
    } finally {
      setStreaming(false)
      setIsRetrying(false)
      abortControllerRef.current = null
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const copyToClipboard = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString('ar-SA', { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  return (
    <Card className="w-full h-[calc(100vh-8rem)] flex flex-col overflow-hidden">
      <CardHeader className="pb-3 border-b flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <MessageSquare className="w-5 h-5 text-primary" />
            المحادثة
          </CardTitle>
          <ProviderSelector />
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center h-full text-center space-y-4 py-8"
          >
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold gradient-text mb-2">مرحباً بك في GAAP</h3>
              <p className="text-muted-foreground max-w-md">
                يمكنني مساعدتك في تحليل الكود، إجراء الأبحاث، وحل المشكلات التقنية.
                <br />
                <span className="text-sm">اكتب رسالتك أدناه وابدأ المحادثة!</span>
              </p>
            </div>
            <div className="flex gap-2 flex-wrap justify-center">
              <Badge variant="secondary">تحليل الكود</Badge>
              <Badge variant="secondary">البحث</Badge>
              <Badge variant="secondary">الصيانة</Badge>
            </div>
          </motion.div>
        ) : (
          <AnimatePresence>
            {messages.map((msg, index) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  msg.role === 'user' 
                    ? 'bg-primary/20' 
                    : 'bg-accent/20'
                }`}>
                  {msg.role === 'user' ? (
                    <User className="w-4 h-4 text-primary" />
                  ) : (
                    <Bot className="w-4 h-4 text-accent" />
                  )}
                </div>

                <div className={`flex-1 max-w-[80%] ${msg.role === 'user' ? 'text-left' : ''}`}>
                  <div className={`rounded-lg p-3 ${
                    msg.role === 'user' 
                      ? 'bg-primary/10 border border-primary/20' 
                      : 'bg-secondary/50 border border-secondary'
                  }`}>
                    {msg.content.startsWith('❌') ? (
                      <div className="flex items-start gap-2 text-red-500">
                        <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>{msg.content.slice(2)}</span>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap prose prose-sm dark:prose-invert max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    )}
                    
                    {msg.codeBlocks && msg.codeBlocks.length > 0 && (
                      <div className="mt-3 space-y-2">
                        {msg.codeBlocks.map((block, i) => (
                          <div key={i} className="relative rounded-md overflow-hidden">
                            <div className="absolute top-2 right-2 flex gap-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs"
                                onClick={() => copyToClipboard(block.code, `${msg.id}-${i}`)}
                              >
                                {copiedId === `${msg.id}-${i}` ? (
                                  <Check className="w-3 h-3" />
                                ) : (
                                  <Copy className="w-3 h-3" />
                                )}
                              </Button>
                            </div>
                            <SyntaxHighlighter
                              language={block.language}
                              style={vscDarkPlus}
                              className="text-sm rounded-md"
                            >
                              {block.code}
                            </SyntaxHighlighter>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className={`flex items-center gap-1 mt-1 text-xs text-muted-foreground ${
                    msg.role === 'user' ? 'justify-end' : ''
                  }`}>
                    <Clock className="w-3 h-3" />
                    {formatTime(msg.timestamp)}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
        
        <AnimatePresence>
          {isStreaming && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center">
                <Bot className="w-4 h-4 text-accent" />
              </div>
              <div className="bg-secondary/50 rounded-lg px-4 py-3 flex items-center gap-2">
                <span className="flex gap-1">
                  <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </span>
                <span className="text-sm text-muted-foreground">جاري الكتابة...</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        <div ref={messagesEndRef} />
      </CardContent>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="px-4 py-2 bg-red-500/10 border-t border-red-500/20"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-red-500 text-sm">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
              <div className="flex gap-2">
                {isStreaming ? (
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={handleCancel}
                    className="text-red-500 border-red-500/20 hover:bg-red-500/10"
                  >
                    إلغاء
                  </Button>
                ) : (
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => {
                      setIsRetrying(true)
                      handleSend()
                    }}
                    className="text-red-500 border-red-500/20 hover:bg-red-500/10"
                  >
                    <RefreshCw className="w-3 h-3 mr-1" />
                    إعادة المحاولة
                  </Button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-4 border-t bg-card/50 flex-shrink-0">
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="اكتب رسالتك هنا... (Enter للإرسال، Shift+Enter لسطر جديد)"
            className="min-h-[60px] max-h-[200px] resize-none"
            disabled={isStreaming}
          />
          <Button 
            onClick={handleSend} 
            disabled={!input.trim() || isStreaming}
            className="h-auto"
          >
            {isStreaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        <div className="flex justify-between mt-2 text-xs text-muted-foreground">
          <span>Enter للإرسال • Shift+Enter لسطر جديد</span>
          <span>{input.length}/10000</span>
        </div>
      </div>
    </Card>
  )
}
