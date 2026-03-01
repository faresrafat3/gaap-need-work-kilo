import { describe, it, expect, vi, beforeEach } from 'vitest'
import { GET, POST } from '../providers/route'

describe('Providers API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('GET /api/providers', () => {
    it('should return providers list from backend', async () => {
      const mockProviders = [
        { name: 'kimi', type: 'webchat', enabled: true, models: ['kimi-k2.5'] },
        { name: 'deepseek', type: 'webchat', enabled: true, models: ['deepseek-chat'] },
      ]

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProviders,
      } as Response)

      const response = await GET()
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.providers).toBeDefined()
      expect(Array.isArray(data.providers)).toBe(true)
    })

    it('should return fallback providers when backend fails', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Backend unavailable'))

      const response = await GET()
      const data = await response.json()

      expect(response.status).toBe(200)
      expect(data.providers).toBeDefined()
      expect(Array.isArray(data.providers)).toBe(true)
      expect(data.error).toBeDefined()
    })
  })

  describe('POST /api/providers', () => {
    it('should create a new provider', async () => {
      const newProvider = {
        name: 'test-provider',
        type: 'api',
        enabled: true,
        models: ['gpt-4'],
      }

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...newProvider, id: '123' }),
      } as Response)

      const request = new Request('http://localhost/api/providers', {
        method: 'POST',
        body: JSON.stringify(newProvider),
      })

      const response = await POST(request)
      const data = await response.json()

      expect(response.status).toBe(201)
      expect(data.name).toBe('test-provider')
    })

    it('should return error when creation fails', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Backend error'))

      const request = new Request('http://localhost/api/providers', {
        method: 'POST',
        body: JSON.stringify({ name: 'test' }),
      })

      const response = await POST(request)
      const data = await response.json()

      expect(response.status).toBe(500)
      expect(data.error).toBe('فشل في إنشاء المزود')
    })
  })
})
