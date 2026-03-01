import { NextRequest, NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'

/**
 * GET /api/sessions/[id]
 * Get a specific session
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    
    const response = await fetch(`${PYTHON_API_URL}/api/sessions/${id}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(10000),
    })

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'الجلسة غير موجودة' },
          { status: 404 }
        )
      }
      throw new Error(`Backend error: ${response.status}`)
    }

    const session = await response.json()
    return NextResponse.json(session)
  } catch (error) {
    console.error('Failed to fetch session:', error)
    return NextResponse.json(
      { error: 'فشل في جلب الجلسة' },
      { status: 500 }
    )
  }
}

/**
 * PUT /api/sessions/[id]
 * Update a session
 */
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    const body = await request.json()
    
    const response = await fetch(`${PYTHON_API_URL}/api/sessions/${id}`, {
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

    const session = await response.json()
    return NextResponse.json(session)
  } catch (error) {
    console.error('Failed to update session:', error)
    return NextResponse.json(
      { 
        error: 'فشل في تحديث الجلسة',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}

/**
 * DELETE /api/sessions/[id]
 * Delete a session
 */
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    
    const response = await fetch(`${PYTHON_API_URL}/api/sessions/${id}`, {
      method: 'DELETE',
      headers: {
        'Accept': 'application/json',
      },
      signal: AbortSignal.timeout(30000),
    })

    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: 'الجلسة غير موجودة' },
          { status: 404 }
        )
      }
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `Backend error: ${response.status}`)
    }

    return NextResponse.json({ 
      success: true,
      message: 'تم حذف الجلسة بنجاح',
      id 
    })
  } catch (error) {
    console.error('Failed to delete session:', error)
    return NextResponse.json(
      { 
        error: 'فشل في حذف الجلسة',
        detail: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
