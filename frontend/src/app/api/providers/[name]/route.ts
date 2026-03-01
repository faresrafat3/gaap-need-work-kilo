import { NextRequest, NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

/**
 * GET /api/providers/[name]
 * Get a specific provider
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params
    
    const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(10000),
    })

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'المزود غير موجود' },
          { status: 404 }
        )
      }
      throw new Error(`Backend error: ${response.status}`)
    }

    const provider = await response.json()
    return NextResponse.json(provider)
  } catch (error) {
    console.error('Failed to fetch provider:', error)
    return NextResponse.json(
      { error: 'فشل في جلب المزود' },
      { status: 500 }
    )
  }
}

/**
 * PUT /api/providers/[name]
 * Update a provider
 */
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params
    const body = await request.json()
    
    const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}`, {
      method: 'PUT',
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
    return NextResponse.json(provider)
  } catch (error) {
    console.error('Failed to update provider:', error)
    return NextResponse.json(
      { 
        error: 'فشل في تحديث المزود',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * DELETE /api/providers/[name]
 * Delete a provider
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params
    
    const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'المزود غير موجود' },
          { status: 404 }
        )
      }
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Backend error: ${response.status}`)
    }

    return NextResponse.json({ 
      success: true,
      message: 'تم حذف المزود بنجاح',
      name 
    })
  } catch (error) {
    console.error('Failed to delete provider:', error)
    return NextResponse.json(
      { 
        error: 'فشل في حذف المزود',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * POST /api/providers/[name]/test
 * Test a provider connection
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params
    const { action } = await request.json().catch(() => ({ action: 'test' }))
    
    if (action === 'test') {
      const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}/test`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        signal: AbortSignal.timeout(30000),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Backend error: ${response.status}`)
      }

      const result = await response.json()
      return NextResponse.json(result)
    }
    
    if (action === 'enable') {
      const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}/enable`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        signal: AbortSignal.timeout(30000),
      })

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`)
      }

      return NextResponse.json({ success: true, message: 'تم تفعيل المزود' })
    }
    
    if (action === 'disable') {
      const response = await fetch(`${PYTHON_API_URL}/api/providers/${name}/disable`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        signal: AbortSignal.timeout(30000),
      })

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`)
      }

      return NextResponse.json({ success: true, message: 'تم تعطيل المزود' })
    }

    return NextResponse.json({ error: 'إجراء غير معروف' }, { status: 400 })
  } catch (error) {
    console.error('Provider action failed:', error)
    return NextResponse.json(
      { 
        error: 'فشل في تنفيذ الإجراء',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
