'use client'

import { useState } from 'react'
import { Shield, Zap, FlaskConical, Scale, Play, CheckCircle, XCircle, AlertTriangle, Loader2, AlertCircle, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { apiPost } from '@/lib/api'
import { CardSkeleton, ErrorState, EmptyState } from './LoadingStates'

interface ValidationIssue {
  type: string
  severity: string
  message: string
  line?: number
  column?: number
}

interface ValidationResult {
  valid: boolean
  issues?: ValidationIssue[]
  report?: Record<string, unknown>
  warnings?: string[]
  executionTime?: number
}

const validationTypes = [
  { id: 'ast', label: 'AST Guard', icon: Shield, description: 'فحص الأمان والتحليل الثابت' },
  { id: 'performance', label: 'Performance', icon: Zap, description: 'تحليل الأداء وتحسين الكود' },
  { id: 'behavioral', label: 'Behavioral', icon: FlaskConical, description: 'فحص السلوك والتداخلات' },
  { id: 'axiom', label: 'Axiom', icon: Scale, description: 'الالتزام بالقواعد والمبادئ' },
]

export function ValidatorsModule() {
  const [code, setCode] = useState('')
  const [activeTab, setActiveTab] = useState('ast')
  const [results, setResults] = useState<Record<string, ValidationResult | undefined>>({})
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const validate = async (type: string) => {
    const sanitized = code.trim()
    if (!sanitized) {
      setError('يرجى إدخال كود للفحص')
      return
    }

    setError(null)
    setLoading(type)
    setResults(prev => ({ ...prev, [type]: undefined }))
    
    try {
      const data = await apiPost(`/api/validators/${type}`, {
        code: sanitized
      })
      setResults(prev => ({ ...prev, [type]: data }))
    } catch (err: any) {
      console.error(`Validation error: ${type}`, err)
      setError(err.message || 'فشل في الفحص')
      setResults(prev => ({
        ...prev,
        [type]: { valid: false, issues: [{ type: 'error', severity: 'error', message: err.message || 'فشل في الاتصال بالخادم' }] }
      }))
    } finally {
      setLoading(null)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'text-red-500'
      case 'warning': return 'text-yellow-500'
      case 'info': return 'text-blue-500'
      default: return 'text-gray-500'
    }
  }

  const getSeverityBg = (severity: string) => {
    switch (severity) {
      case 'error': return 'bg-red-500/10 border-red-500/20'
      case 'warning': return 'bg-yellow-500/10 border-yellow-500/20'
      case 'info': return 'bg-blue-500/10 border-blue-500/20'
      default: return 'bg-gray-500/10 border-gray-500/20'
    }
  }

  const currentResult = results[activeTab]

  return (
    <Card className="w-full overflow-hidden">
      <CardHeader className="pb-3 border-b">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Shield className="w-5 h-5 text-primary" />
          وحدات الفحص
        </CardTitle>
      </CardHeader>
      
      <CardContent className="p-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="border-b px-4 pt-3">
            <TabsList className="grid w-full grid-cols-4 h-auto p-0 bg-transparent gap-1">
              {validationTypes.map((type) => (
                <TabsTrigger
                  key={type.id}
                  value={type.id}
                  className="flex flex-col items-center gap-1 py-3 px-2 data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-md transition-all"
                >
                  <type.icon className="w-4 h-4" />
                  <span className="text-xs">{type.label}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </div>

          <div className="p-4 space-y-4">
            {error && (
              <div className="flex items-center justify-between p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm">
                <div className="flex items-center gap-2 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => validate(activeTab)}
                  className="h-8 gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  إعادة المحاولة
                </Button>
              </div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-medium">كود الإدخال</label>
              <Textarea
                placeholder="الصق الكود المراد فحصه هنا..."
                value={code}
                onChange={(e) => {
                  setCode(e.target.value)
                  setError(null)
                }}
                className="min-h-[180px] font-mono text-sm resize-none"
              />
            </div>

            {!code.trim() && !currentResult && (
              <div className="flex items-center justify-center p-4 text-sm text-muted-foreground bg-muted/50 rounded-lg">
                يرجى إدخال كود Python للفحص
              </div>
            )}

            <Button
              onClick={() => validate(activeTab)}
              disabled={!code.trim() || loading !== null}
              className="w-full gap-2"
            >
              {loading === activeTab ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              فحص {validationTypes.find(t => t.id === activeTab)?.label}
            </Button>

            {currentResult && (
              <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="flex items-center justify-between p-3 rounded-lg border">
                  <div className="flex items-center gap-2">
                    {currentResult.valid ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-500" />
                    )}
                    <span className="font-medium">
                      {currentResult.valid ? 'ناجح' : 'فاشل'}
                    </span>
                  </div>
                  {currentResult.executionTime && (
                    <Badge variant="outline" className="text-xs">
                      {currentResult.executionTime}ms
                    </Badge>
                  )}
                </div>

                {currentResult.issues && currentResult.issues.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      المشاكل المكتشفة ({currentResult.issues.length})
                    </h4>
                    <ScrollArea className="h-[200px] rounded-md border">
                      <div className="p-3 space-y-2">
                        {currentResult.issues.map((issue, i) => (
                          <div
                            key={i}
                            className={`p-3 rounded-lg border text-sm ${getSeverityBg(issue.severity)}`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className={`font-medium capitalize ${getSeverityColor(issue.severity)}`}>
                                {issue.severity}
                              </span>
                              {issue.line && (
                                <Badge variant="outline" className="text-xs">
                                  سطر {issue.line}
                                  {issue.column && `:${issue.column}`}
                                </Badge>
                              )}
                            </div>
                            <p className="mt-1">{issue.message}</p>
                            {issue.type && (
                              <Badge variant="secondary" className="mt-2 text-xs">
                                {issue.type}
                              </Badge>
                            )}
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                )}

                {currentResult.warnings && currentResult.warnings.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">التحذيرات</h4>
                    <ul className="space-y-1">
                      {currentResult.warnings.map((warning, i) => (
                        <li key={i} className="text-sm text-yellow-600 flex items-center gap-2">
                          <AlertTriangle className="w-3 h-3" />
                          {warning}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {currentResult.report && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">التقرير المفصل</h4>
                    <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto border max-h-[300px] overflow-auto">
                      {JSON.stringify(currentResult.report, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </Tabs>
      </CardContent>
    </Card>
  )
}
