import { NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

/**
 * GET /api/providers/live
 * Proxies to Python backend /api/providers/status
 * Returns provider data with actual models
 */
export async function GET() {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/providers/status`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Cache-Control': 'no-cache',
      },
      signal: AbortSignal.timeout(10000),
    })

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Failed to fetch live providers:', error)
    
    // Return fallback data when backend is unavailable
    return NextResponse.json({
      providers: [
        {
          name: 'kimi',
          display_name: 'Kimi',
          actual_model: 'kimi-k2.5-thinking',
          default_model: 'kimi-k2.5-thinking',
          status: 'unknown',
          last_seen: new Date().toISOString(),
          latency_ms: 0,
          success_rate: 0,
          accounts_count: 0,
          healthy_accounts: 0,
          models_available: ['kimi-k2.5-thinking'],
          provider_type: 'webchat',
          error_message: error instanceof Error ? error.message : 'Backend unavailable',
          cached: false,
          cache_age_seconds: null,
        },
        {
          name: 'deepseek',
          display_name: 'DeepSeek',
          actual_model: 'deepseek-chat',
          default_model: 'deepseek-chat',
          status: 'unknown',
          last_seen: new Date().toISOString(),
          latency_ms: 0,
          success_rate: 0,
          accounts_count: 0,
          healthy_accounts: 0,
          models_available: ['deepseek-chat'],
          provider_type: 'webchat',
          error_message: error instanceof Error ? error.message : 'Backend unavailable',
          cached: false,
          cache_age_seconds: null,
        },
        {
          name: 'glm',
          display_name: 'GLM',
          actual_model: 'GLM-5',
          default_model: 'GLM-5',
          status: 'unknown',
          last_seen: new Date().toISOString(),
          latency_ms: 0,
          success_rate: 0,
          accounts_count: 0,
          healthy_accounts: 0,
          models_available: ['GLM-5'],
          provider_type: 'webchat',
          error_message: error instanceof Error ? error.message : 'Backend unavailable',
          cached: false,
          cache_age_seconds: null,
        },
      ],
      last_updated: new Date().toISOString(),
      total_providers: 3,
      active_providers: 0,
      failed_providers: 3,
      cache_hit: false,
      refresh_in_progress: false,
    }, { status: 200 })
  }
}
