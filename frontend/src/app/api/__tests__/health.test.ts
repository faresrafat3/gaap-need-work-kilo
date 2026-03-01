import { describe, it, expect, vi, beforeEach } from 'vitest'
import { GET } from '../health/route'

describe('Health API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should return healthy status when backend is available', async () => {
    // Mock successful backend response
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ version: '1.0.0', status: 'healthy' }),
    } as Response)

    const response = await GET()
    const data = await response.json()

    expect(response.status).toBe(200)
    expect(data.status).toBe('healthy')
    expect(data.services.python.status).toBe('healthy')
    expect(data.services.nextjs.status).toBe('healthy')
  })

  it('should return degraded status when backend is unavailable', async () => {
    // Mock failed backend response
    vi.mocked(fetch).mockRejectedValueOnce(new Error('Connection refused'))

    const response = await GET()
    const data = await response.json()

    expect(response.status).toBe(200)
    expect(data.status).toBe('degraded')
    expect(data.services.python.status).toBe('unreachable')
  })

  it('should return degraded status when backend returns error', async () => {
    // Mock backend error response
    vi.mocked(fetch).mockResolvedValueOnce({
      ok: false,
      status: 500,
    } as Response)

    const response = await GET()
    const data = await response.json()

    expect(response.status).toBe(200)
    expect(data.status).toBe('degraded')
    expect(data.services.python.status).toBe('unhealthy')
  })
})
