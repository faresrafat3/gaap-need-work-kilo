import { NextRequest, NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

/**
 * GET /api/providers
 * List all providers from backend
 */
export async function GET() {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/providers`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(10000),
    })

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`)
    }

    const providers = await response.json()
    return NextResponse.json({ providers })
  } catch (error) {
    console.error('Failed to fetch providers:', error)
    
    // Return fallback providers
    return NextResponse.json({
      providers: [
        {
          name: 'kimi',
          type: 'webchat',
          enabled: true,
          priority: 1,
          models: ['kimi-k2.5-thinking'],
          health: 'unknown',
          stats: {},
        },
        {
          name: 'deepseek',
          type: 'webchat',
          enabled: true,
          priority: 2,
          models: ['deepseek-chat'],
          health: 'unknown',
          stats: {},
        },
        {
          name: 'glm',
          type: 'webchat',
          enabled: true,
          priority: 3,
          models: ['GLM-5'],
          health: 'unknown',
          stats: {},
        },
      ],
      error: error instanceof Error ? error.message : 'Failed to fetch providers',
    }, { status: 200 })
  }
}

/**
 * POST /api/providers
 * Create a new provider
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    const response = await fetch(`${PYTHON_API_URL}/api/providers`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Backend error: ${response.status}`)
    }

    const provider = await response.json()
    return NextResponse.json(provider, { status: 201 })
  } catch (error) {
    console.error('Failed to create provider:', error)
    return NextResponse.json(
      { 
        error: 'فشل في إنشاء المزود',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
