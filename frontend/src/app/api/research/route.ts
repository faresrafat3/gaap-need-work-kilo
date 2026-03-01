import { NextRequest, NextResponse } from 'next/server'

const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'
const USE_REAL_BACKEND = process.env.USE_REAL_BACKEND === 'true'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, depth = 3 } = body

    if (!query) {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 })
    }

    // If real backend is enabled, proxy to Python
    if (USE_REAL_BACKEND) {
      return await proxyToPython(query, depth)
    }

    // Otherwise, use mock response
    return mockResearchResponse(query, depth)
  } catch (error) {
    console.error('Research error:', error)
    return NextResponse.json({ error: 'حدث خطأ في البحث' }, { status: 500 })
  }
}

async function proxyToPython(query: string, depth: number) {
  try {
    const response = await fetch(`${PYTHON_API_URL}/api/research/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, depth }),
    })

    if (!response.ok) {
      throw new Error(`Python API error: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json({
      success: true,
      query: data.query || query,
      depth: data.depth || depth,
      sources: data.sources || [],
      summary: data.summary || data.report || '',
      timestamp: new Date().toISOString(),
      totalSources: data.sources?.length || 0,
      avgETS: data.avg_ets || data.avgETS || 0.85,
    })
  } catch (error) {
    console.error('Python research proxy error:', error)
    return mockResearchResponse(query, depth)
  }
}

function mockResearchResponse(query: string, depth: number) {
  const mockSources: Record<number, Array<{ title: string; url: string; snippet: string; ets: number }>> = {
    1: [
      { title: 'مقدمة في الموضوع', url: 'https://example.com/intro', snippet: 'نظرة عامة على الموضوع وأساسياته...', ets: 0.92 },
      { title: 'دليل المبتدئين', url: 'https://guide.example.com', snippet: 'شرح تفصيلي للمفاهيم الأساسية...', ets: 0.85 },
    ],
    2: [
      { title: 'مقدمة في الموضوع', url: 'https://example.com/intro', snippet: 'نظرة عامة على الموضوع وأساسياته...', ets: 0.92 },
      { title: 'دليل المبتدئين', url: 'https://guide.example.com', snippet: 'شرح تفصيلي للمفاهيم الأساسية...', ets: 0.85 },
      { title: 'الأبحاث الحديثة', url: 'https://research.example.com', snippet: 'آخر الدراسات والاكتشافات...', ets: 0.95 },
      { title: 'تحليل معمق', url: 'https://analysis.example.com', snippet: 'دراسة تفصيلية للجوانب المختلفة...', ets: 0.88 },
    ],
    3: [
      { title: 'مقدمة في الموضوع', url: 'https://example.com/intro', snippet: 'نظرة عامة على الموضوع وأساسياته...', ets: 0.92 },
      { title: 'دليل المبتدئين', url: 'https://guide.example.com', snippet: 'شرح تفصيلي للمفاهيم الأساسية...', ets: 0.85 },
      { title: 'الأبحاث الحديثة', url: 'https://research.example.com', snippet: 'آخر الدراسات والاكتشافات...', ets: 0.95 },
      { title: 'تحليل معمق', url: 'https://analysis.example.com', snippet: 'دراسة تفصيلية للجوانب المختلفة...', ets: 0.88 },
      { title: 'دراسات أكاديمية', url: 'https://academic.example.com', snippet: 'أوراق بحثية من مصادر أكاديمية...', ets: 0.97 },
      { title: 'تطبيقات عملية', url: 'https://practical.example.com', snippet: 'أمثلة وتطبيقات واقعية...', ets: 0.91 },
    ],
    4: [
      { title: 'مقدمة في الموضوع', url: 'https://example.com/intro', snippet: 'نظرة عامة على الموضوع وأساسياته...', ets: 0.92 },
      { title: 'دليل المبتدئين', url: 'https://guide.example.com', snippet: 'شرح تفصيلي للمفاهيم الأساسية...', ets: 0.85 },
      { title: 'الأبحاث الحديثة', url: 'https://research.example.com', snippet: 'آخر الدراسات والاكتشافات...', ets: 0.95 },
      { title: 'تحليل معمق', url: 'https://analysis.example.com', snippet: 'دراسة تفصيلية للجوانب المختلفة...', ets: 0.88 },
      { title: 'دراسات أكاديمية', url: 'https://academic.example.com', snippet: 'أوراق بحثية من مصادر أكاديمية...', ets: 0.97 },
      { title: 'تطبيقات عملية', url: 'https://practical.example.com', snippet: 'أمثلة وتطبيقات واقعية...', ets: 0.91 },
      { title: 'مراجع متخصصة', url: 'https://specialized.example.com', snippet: 'مواد متخصصة للمحترفين...', ets: 0.94 },
      { title: 'اتجاهات مستقبلية', url: 'https://future.example.com', snippet: 'توقعات وتطورات مستقبلية...', ets: 0.89 },
    ],
    5: [
      { title: 'مقدمة في الموضوع', url: 'https://example.com/intro', snippet: 'نظرة عامة على الموضوع وأساسياته...', ets: 0.92 },
      { title: 'دليل المبتدئين', url: 'https://guide.example.com', snippet: 'شرح تفصيلي للمفاهيم الأساسية...', ets: 0.85 },
      { title: 'الأبحاث الحديثة', url: 'https://research.example.com', snippet: 'آخر الدراسات والاكتشافات...', ets: 0.95 },
      { title: 'تحليل معمق', url: 'https://analysis.example.com', snippet: 'دراسة تفصيلية للجوانب المختلفة...', ets: 0.88 },
      { title: 'دراسات أكاديمية', url: 'https://academic.example.com', snippet: 'أوراق بحثية من مصادر أكاديمية...', ets: 0.97 },
      { title: 'تطبيقات عملية', url: 'https://practical.example.com', snippet: 'أمثلة وتطبيقات واقعية...', ets: 0.91 },
      { title: 'مراجع متخصصة', url: 'https://specialized.example.com', snippet: 'مواد متخصصة للمحترفين...', ets: 0.94 },
      { title: 'اتجاهات مستقبلية', url: 'https://future.example.com', snippet: 'توقعات وتطورات مستقبلية...', ets: 0.89 },
      { title: 'مصادر دولية', url: 'https://international.example.com', snippet: 'دراسات ومصادر عالمية...', ets: 0.96 },
      { title: 'تحليل البيانات', url: 'https://data.example.com', snippet: 'إحصائيات وتحليلات معمقة...', ets: 0.93 },
    ],
  }

  const sources = mockSources[Math.min(depth, 5) as keyof typeof mockSources] || mockSources[3]

  const summary = `## نتائج البحث: ${query}

⚠️ **ملاحظة**: الـ Python Backend مش شغال - ده رد تجريبي.

تم إجراء بحث بمستوى ${depth} من العمق.

### النتائج الرئيسية:
1. **الأساسيات**: تم جمع معلومات شاملة عن الموضوع
2. **التطورات الحديثة**: رصد آخر التطورات والاتجاهات
3. **التطبيقات**: تحديد التطبيقات العملية والفوائد

### التوصيات:
- متابعة المصادر الموثوقة للحصول على تحديثات مستمرة
- التحقق من المعلومات من مصادر متعددة
- تطبيق المعرفة المكتسبة في مشاريع عملية

**مصادر موثوقة**: ${sources.length} مصدر
**معدل الثقة المعرفية (ETS)**: ${(sources.reduce((acc, s) => acc + s.ets, 0) / sources.length * 100).toFixed(1)}%

---
🚀 لتفعيل البحث الحقيقي، شغّل Python backend:
\`\`\`bash
uvicorn gaap.api.main:app --reload --port 8000
\`\`\``

  return NextResponse.json({
    success: true,
    query,
    depth,
    sources,
    summary,
    timestamp: new Date().toISOString(),
    totalSources: sources.length,
    avgETS: sources.reduce((acc, s) => acc + s.ets, 0) / sources.length,
  })
}
