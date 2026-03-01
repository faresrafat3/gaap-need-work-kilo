'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Loader2, ExternalLink, Shield, TrendingUp, BookOpen, Globe, Clock } from 'lucide-react'
import { useGAAPStore, ResearchResult } from '@/lib/store'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

export function ResearchModule() {
  const { researchResults, isResearching, addResearchResult, setResearching } = useGAAPStore()
  const [query, setQuery] = useState('')
  const [depth, setDepth] = useState(3)

  const handleResearch = async () => {
    if (!query.trim() || isResearching) return

    setResearching(true)

    try {
      const response = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, depth }),
      })

      const data = await response.json()
      
      const result: ResearchResult = {
        id: Date.now().toString(),
        query: data.query,
        depth: data.depth,
        sources: data.sources,
        summary: data.summary,
        timestamp: new Date(data.timestamp),
      }

      addResearchResult(result)
    } catch (error) {
      console.error('Research error:', error)
    } finally {
      setResearching(false)
    }
  }

  const getETSColor = (ets: number) => {
    if (ets >= 0.9) return 'text-green-400'
    if (ets >= 0.8) return 'text-yellow-400'
    return 'text-orange-400'
  }

  const getETSLabel = (ets: number) => {
    if (ets >= 0.95) return 'موثوق جداً'
    if (ets >= 0.9) return 'موثوق'
    if (ets >= 0.8) return 'جيد'
    return 'متوسط'
  }

  const depthLabels = ['سريع', 'أساسي', 'متوسط', 'عميق', 'شامل']

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="gradient-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              وحدة البحث العميق
            </CardTitle>
            <CardDescription>
              إجراء أبحاث معمقة مع تقييم موثوقية المصادر
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Search Input */}
            <div className="flex gap-2">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="أدخل موضوع البحث..."
                className="flex-1 bg-secondary/50"
                onKeyDown={(e) => e.key === 'Enter' && handleResearch()}
                dir="rtl"
              />
              <Button
                onClick={handleResearch}
                disabled={!query.trim() || isResearching}
                className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
              >
                {isResearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
              </Button>
            </div>

            {/* Depth Slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">عمق البحث</span>
                <Badge variant="outline" className="bg-primary/10">
                  {depthLabels[depth - 1]}
                </Badge>
              </div>
              <div className="flex items-center gap-4">
                <Slider
                  value={[depth]}
                  onValueChange={([value]) => setDepth(value)}
                  min={1}
                  max={5}
                  step={1}
                  className="flex-1"
                />
                <span className="text-sm font-medium w-8 text-center">{depth}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                المستوى الأعلى = نتائج أكثر ووقت أطول
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Research Results */}
      <AnimatePresence mode="popLayout">
        {researchResults.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-12 text-center"
          >
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-4">
              <Globe className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-lg font-medium mb-2">ابدأ البحث</h3>
            <p className="text-muted-foreground max-w-sm">
              أدخل موضوعاً للبحث عن معلومات معمقة من مصادر متعددة مع تقييم الموثوقية
            </p>
          </motion.div>
        ) : (
          researchResults.map((result, index) => (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="gradient-border">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{result.query}</CardTitle>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-primary/10">
                        عمق: {result.depth}
                      </Badge>
                      <Badge variant="outline" className="bg-green-500/10 text-green-400">
                        <Shield className="w-3 h-3 mr-1" />
                        ETS: {(((result as any).avgETS || 0) * 100).toFixed(0)}%
                      </Badge>
                    </div>
                  </div>
                  <CardDescription className="flex items-center gap-2">
                    <Clock className="w-3 h-3" />
                    {new Date(result.timestamp).toLocaleString('ar-SA')}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Summary */}
                  <div className="p-4 rounded-lg bg-secondary/50 border border-primary/10">
                    <div className="prose prose-sm prose-invert max-w-none" dir="rtl">
                      <div className="whitespace-pre-wrap text-sm">{result.summary}</div>
                    </div>
                  </div>

                  {/* Sources */}
                  <div>
                    <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-primary" />
                      المصادر ({result.sources.length})
                    </h4>
                    <div className="grid gap-2 max-h-64 overflow-y-auto">
                      {result.sources.map((source, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="p-3 rounded-lg bg-secondary/30 border border-primary/5 hover:border-primary/20 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <a
                                  href={source.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-sm font-medium text-primary hover:underline truncate"
                                >
                                  {source.title}
                                </a>
                                <ExternalLink className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                              </div>
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {source.snippet}
                              </p>
                            </div>
                            <div className="flex items-center gap-1 flex-shrink-0">
                              <Shield className={`w-3 h-3 ${getETSColor(source.ets)}`} />
                              <span className={`text-xs font-medium ${getETSColor(source.ets)}`}>
                                {(source.ets * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                          <div className="mt-2 flex items-center gap-2">
                            <Progress 
                              value={source.ets * 100} 
                              className="h-1 flex-1"
                            />
                            <span className="text-xs text-muted-foreground">
                              {getETSLabel(source.ets)}
                            </span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))
        )}
      </AnimatePresence>
    </div>
  )
}
