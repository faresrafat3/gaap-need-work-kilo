import { NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

/**
 * GET /api/health
 * Check system health - points to /health endpoint (not /api/system/health)
 */
export async function GET() {
  const startTime = Date.now()
  
  // Check Next.js health
  const nextjsHealth = {
    status: 'healthy' as const,
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    timestamp: new Date().toISOString(),
  }

  // Check Python backend health
  let pythonHealth: {
    status: 'healthy' | 'unhealthy' | 'disconnected' | 'unreachable'
    url: string
    latency: number | null
    error: string | null
    version?: string
  } = {
    status: 'disconnected',
    url: PYTHON_API_URL,
    latency: null,
    error: null,
  }

  try {
    const pyStart = Date.now()
    const response = await fetch(`${PYTHON_API_URL}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    })
    const pyLatency = Date.now() - pyStart

    if (response.ok) {
      const data = await response.json()
      pythonHealth = {
        status: 'healthy',
        url: PYTHON_API_URL,
        latency: pyLatency,
        error: null,
        version: data.version,
      }
    } else {
      pythonHealth = {
        status: 'unhealthy',
        url: PYTHON_API_URL,
        latency: pyLatency,
        error: `HTTP ${response.status}`,
      }
    }
  } catch (error) {
    pythonHealth = {
      status: 'unreachable',
      url: PYTHON_API_URL,
      latency: null,
      error: error instanceof Error ? error.message : 'Unknown error',
    }
  }

  const totalLatency = Date.now() - startTime
  const overallStatus = pythonHealth.status === 'healthy' 
    ? 'healthy' 
    : 'degraded'

  return NextResponse.json({
    status: overallStatus,
    timestamp: new Date().toISOString(),
    latency: totalLatency,
    services: {
      nextjs: nextjsHealth,
      python: pythonHealth,
    },
    config: {
      pythonApiUrl: PYTHON_API_URL,
    },
    message: getStatusMessage(overallStatus, pythonHealth),
  })
}

function getStatusMessage(status: string, pythonHealth: { status: string; error: string | null }) {
  if (status === 'healthy') {
    return '✅ النظام يعمل بكامل طاقته - Python Backend متصل'
  }
  if (status === 'degraded') {
    return `⚠️ النظام يعمل بشكل جزئي - Python Backend: ${pythonHealth.error || 'غير متصل'}`
  }
  return '❌ النظام في حالة خطأ'
}
