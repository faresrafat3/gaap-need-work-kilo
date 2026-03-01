import { NextRequest, NextResponse } from 'next/server'
import { rateLimit, getClientIP, createRateLimitResponse } from '@/lib/rate-limit'
import { logger, logRequest } from '@/lib/logger'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatRequestBody {
  messages: ChatMessage[]
  provider?: string
  stream?: boolean
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  const clientIP = getClientIP(request)
  
  // Rate limiting
  const rateLimitResult = rateLimit(clientIP, {
    interval: 60000, // 1 minute
    maxRequests: parseInt(process.env.RATE_LIMIT_REQUESTS_PER_MINUTE || '60'),
  })

  if (!rateLimitResult.success) {
    logger.warn('Rate limit exceeded', { clientIP, path: '/api/chat' })
    return createRateLimitResponse(rateLimitResult.reset)
  }

  try {
    const body: ChatRequestBody = await request.json()
    const { messages, provider = 'kimi', stream = false } = body

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      const response = NextResponse.json(
        { error: 'No messages provided' },
        { status: 400 }
      )
      logRequest(request, response, Date.now() - startTime)
      return response
    }

    const lastMessage = messages.filter(m => m.role === 'user').pop()
    
    if (!lastMessage) {
      const response = NextResponse.json(
        { error: 'No user message found' },
        { status: 400 }
      )
      logRequest(request, response, Date.now() - startTime)
      return response
    }

    logger.info('Chat request', { provider, messageCount: messages.length, clientIP })

    // Try backend with retry logic
    const result = await callBackendWithRetry(messages, provider, stream)
    
    logRequest(request, result, Date.now() - startTime)
    return result

  } catch (error) {
    logger.error('Chat error', { error: error instanceof Error ? error.message : 'Unknown', clientIP })
    
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { error: 'Request was cancelled' },
        { status: 499 }
      )
    }

    const response = NextResponse.json(
      { 
        error: 'حدث خطأ في المعالجة',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
    logRequest(request, response, Date.now() - startTime)
    return response
  }
}

async function callBackendWithRetry(
  messages: ChatMessage[],
  provider: string,
  stream: boolean,
  maxRetries: number = 3
): Promise<Response> {
  const lastMessage = messages.filter(m => m.role === 'user').pop()
  
  if (!lastMessage) {
    return NextResponse.json(
      { error: 'No user message found' },
      { status: 400 }
    )
  }

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(`${PYTHON_API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': stream ? 'text/event-stream' : 'application/json',
        },
        body: JSON.stringify({
          message: lastMessage.content,
          provider: provider,
          context: {
            history: messages.slice(0, -1).map(m => ({
              role: m.role,
              content: m.content,
            })),
          },
        }),
        signal: AbortSignal.timeout(60000),
      })

      if (!response.ok) {
        // If not found, try fallback to session endpoint
        if (response.status === 404 && attempt === maxRetries - 1) {
          return await fallbackToSessionEndpoint(lastMessage.content)
        }
        
        if (response.status >= 500) {
          // Retry on server errors
          if (attempt < maxRetries - 1) {
            await sleep(1000 * Math.pow(2, attempt))
            continue
          }
        }
        
        throw new Error(`Backend error: ${response.status}`)
      }

      // Handle streaming response
      if (stream) {
        return handleStreamingResponse(response)
      }

      // Handle JSON response
      const data = await response.json()
      return NextResponse.json(data)

    } catch (error) {
      if (attempt === maxRetries - 1) {
        logger.error('All retries failed', { error: error instanceof Error ? error.message : 'Unknown' })
        return mockResponse()
      }
      
      // Exponential backoff
      await sleep(1000 * Math.pow(2, attempt))
    }
  }

  return mockResponse()
}

async function fallbackToSessionEndpoint(message: string): Promise<Response> {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: 'Chat Session',
        description: message.slice(0, 100),
      }),
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      throw new Error(`Session endpoint error: ${response.status}`)
    }

    return mockResponse()
  } catch (error) {
    logger.error('Fallback failed', { error: error instanceof Error ? error.message : 'Unknown' })
    return mockResponse()
  }
}

function handleStreamingResponse(backendResponse: Response): Response {
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()
  
  const stream = new ReadableStream({
    async start(controller) {
      const reader = backendResponse.body?.getReader()
      
      if (!reader) {
        controller.close()
        return
      }

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n').filter(line => line.trim())

          for (const line of lines) {
            try {
              // Try to parse as JSON
              const data = JSON.parse(line)
              controller.enqueue(encoder.encode(JSON.stringify(data) + '\n'))
            } catch {
              // If not JSON, treat as content
              controller.enqueue(encoder.encode(JSON.stringify({
                type: 'content',
                content: line,
              }) + '\n'))
            }
          }
        }
        
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'done' }) + '\n'))
      } catch (error) {
        logger.error('Stream error', { error: error instanceof Error ? error.message : 'Unknown' })
        controller.enqueue(encoder.encode(JSON.stringify({
          type: 'error',
          error: 'Stream interrupted',
        }) + '\n'))
      } finally {
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  })
}

function mockResponse(): Response {
  const encoder = new TextEncoder()
  
  const stream = new ReadableStream({
    async start(controller) {
      const content = `أهلاً! أنا GAAP، منصة الذكاء الاصطناعي المستقلة.

يمكنني مساعدتك في:
- 🎯 تحليل وكتابة التعليمات البرمجية
- 🔍 إجراء أبحاث معمقة
- 🛠️ حل المشكلات التقنية
- 📊 تفسير البيانات والنتائج

⚠️ ملاحظة: الـ Python Backend مش شغال حالياً. شغّله بأمر:
\`\`\`bash
cd /path/to/gaap
uvicorn gaap.api.main:app --reload --port 8000
\`\`\``

      const words = content.split(' ')
      
      for (let i = 0; i < words.length; i++) {
        const chunk = JSON.stringify({
          type: 'content',
          content: words[i] + (i < words.length - 1 ? ' ' : ''),
        })
        controller.enqueue(encoder.encode(chunk + '\n'))
        await sleep(30 + Math.random() * 20)
      }

      controller.enqueue(encoder.encode(JSON.stringify({ type: 'done' }) + '\n'))
      controller.close()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  })
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
