'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, Wrench, Calculator, Loader2, FileCode, Clock, Zap, TrendingDown, AlertCircle, RefreshCw } from 'lucide-react'
import { apiPost } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'

interface DebtItem {
  type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  impact: string
  file: string
  suggestion: string
}

interface ScanResult {
  total_debt: number
  items: DebtItem[]
}

interface RefinanceResult {
  savings: {
    lines_removed: number
    complexity_reduction: number
  }
  results: Array<{
    file: string
    changes: string
  }>
}

interface InterestResult {
  simple_interest: number
  compound_interest: number
  total: number
}

export function MaintenanceModule() {
  const [isScanning, setIsScanning] = useState(false)
  const [isRefinancing, setIsRefinancing] = useState(false)
  const [isCalculating, setIsCalculating] = useState(false)

  const [scanError, setScanError] = useState<string | null>(null)
  const [refinanceError, setRefinanceError] = useState<string | null>(null)
  const [interestError, setInterestError] = useState<string | null>(null)

  const [projectPath, setProjectPath] = useState('')
  const [includeTypes, setIncludeTypes] = useState('all')
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)

  const [debtItems, setDebtItems] = useState('')
  const [optimizationLevel, setOptimizationLevel] = useState<'low' | 'medium' | 'high'>('medium')
  const [refinanceResult, setRefinanceResult] = useState<RefinanceResult | null>(null)

  const [principal, setPrincipal] = useState('')
  const [rate, setRate] = useState('')
  const [timeMonths, setTimeMonths] = useState('')
  const [interestResult, setInterestResult] = useState<InterestResult | null>(null)

  const handleScan = async () => {
    if (!projectPath.trim() || isScanning) return

    setIsScanning(true)
    setScanError(null)
    setScanResult(null)

    try {
      const data = await apiPost<ScanResult>('/api/maintenance/scan', {
        project_path: projectPath,
        include_types: includeTypes || 'all'
      })
      setScanResult(data)
    } catch (error) {
      setScanError(error instanceof Error ? error.message : 'فشل فحص الديون التقنية')
    } finally {
      setIsScanning(false)
    }
  }

  const handleRefinance = async () => {
    if (!debtItems.trim() || isRefinancing) return

    setIsRefinancing(true)
    setRefinanceError(null)
    setRefinanceResult(null)

    try {
      let parsedDebtItems = JSON.parse(debtItems)
      
      const data = await apiPost<RefinanceResult>('/api/maintenance/refinance', {
        debt_items: parsedDebtItems,
        optimization_level: optimizationLevel
      })
      setRefinanceResult(data)
    } catch (error) {
      setRefinanceError(error instanceof Error ? error.message : 'فشل تحسين الكود')
    } finally {
      setIsRefinancing(false)
    }
  }

  const handleInterest = async () => {
    if (!principal || !rate || !timeMonths || isCalculating) return

    setIsCalculating(true)
    setInterestError(null)
    setInterestResult(null)

    try {
      const data = await apiPost<InterestResult>('/api/maintenance/interest', {
        principal: parseFloat(principal),
        rate: parseFloat(rate),
        time_months: parseInt(timeMonths)
      })
      setInterestResult(data)
    } catch (error) {
      setInterestError(error instanceof Error ? error.message : 'فشل حساب الفائدة')
    } finally {
      setIsCalculating(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/10 text-red-400 border-red-500/20'
      case 'high': return 'bg-orange-500/10 text-orange-400 border-orange-500/20'
      case 'medium': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
      case 'low': return 'bg-green-500/10 text-green-400 border-green-500/20'
      default: return 'bg-primary/10 text-primary'
    }
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card className="gradient-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wrench className="w-5 h-5 text-primary" />
              وحدة الصيانة
            </CardTitle>
            <CardDescription>
              فحص الديون التقنية وتحسين الكود وحساب الفائدة
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="scan" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="scan" className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  فحص الديون
                </TabsTrigger>
                <TabsTrigger value="refinance" className="flex items-center gap-2">
                  <Wrench className="w-4 h-4" />
                  تحسين الكود
                </TabsTrigger>
                <TabsTrigger value="interest" className="flex items-center gap-2">
                  <Calculator className="w-4 h-4" />
                  حساب الفائدة
                </TabsTrigger>
              </TabsList>

              <TabsContent value="scan" className="mt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">مسار المشروع</label>
                    <Input
                      value={projectPath}
                      onChange={(e) => setProjectPath(e.target.value)}
                      placeholder="أدخل مسار المشروع..."
                      dir="ltr"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">أنواع الملفات (اختياري)</label>
                    <Input
                      value={includeTypes}
                      onChange={(e) => setIncludeTypes(e.target.value)}
                      placeholder="all أو ts,tsx,js"
                      dir="ltr"
                    />
                  </div>
                </div>
                <Button
                  onClick={handleScan}
                  disabled={!projectPath.trim() || isScanning}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isScanning ? (
                    <Loader2 className="w-4 h-4 animate-spin ml-2" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 ml-2" />
                  )}
                  فحص الديون التقنية
                </Button>

                {scanError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
                  >
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400">{scanError}</span>
                    <Button variant="ghost" size="sm" onClick={handleScan} className="ml-auto">
                      <RefreshCw className="w-3 h-3 ml-1" />
                      إعادة المحاولة
                    </Button>
                  </motion.div>
                )}

                <AnimatePresence>
                  {scanResult && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-4"
                    >
                      <div className="flex items-center gap-2 p-4 rounded-lg bg-secondary/50 border border-primary/10">
                        <AlertTriangle className="w-5 h-5 text-yellow-400" />
                        <span className="font-medium">إجمالي الديون التقنية:</span>
                        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-400">
                          {scanResult.total_debt} عنصر
                        </Badge>
                      </div>

                      <ScrollArea className="h-[400px]">
                        <div className="space-y-3">
                          {scanResult.items.map((item, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.05 }}
                              className="p-4 rounded-lg bg-secondary/30 border border-primary/10 hover:border-primary/30 transition-colors"
                            >
                              <div className="flex items-start justify-between gap-2 mb-2">
                                <div className="flex items-center gap-2">
                                  <FileCode className="w-4 h-4 text-primary" />
                                  <span className="font-medium text-sm">{item.file}</span>
                                </div>
                                <Badge variant="outline" className={getSeverityColor(item.severity)}>
                                  {item.severity}
                                </Badge>
                              </div>
                              <div className="space-y-2 text-sm">
                                <div className="flex items-center gap-2">
                                  <span className="text-muted-foreground">النوع:</span>
                                  <Badge variant="outline" className="bg-primary/10">
                                    {item.type}
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="text-muted-foreground">التأثير:</span>
                                  <span>{item.impact}</span>
                                </div>
                                <div className="p-2 rounded bg-orange-500/10 text-orange-300 text-xs">
                                  <Zap className="w-3 h-3 inline ml-1" />
                                  {item.suggestion}
                                </div>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </ScrollArea>
                    </motion.div>
                  )}
                </AnimatePresence>
              </TabsContent>

              <TabsContent value="refinance" className="mt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">عناصر الديون (JSON)</label>
                    <Input
                      value={debtItems}
                      onChange={(e) => setDebtItems(e.target.value)}
                      placeholder='[{"type": "duplication", "file": "..."}]'
                      dir="ltr"
                      className="h-24 font-mono text-xs"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">مستوى التحسين</label>
                    <div className="flex gap-2">
                      {(['low', 'medium', 'high'] as const).map((level) => (
                        <Button
                          key={level}
                          variant={optimizationLevel === level ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setOptimizationLevel(level)}
                          className="flex-1"
                        >
                          {level === 'low' ? 'منخفض' : level === 'medium' ? 'متوسط' : 'عالي'}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
                <Button
                  onClick={handleRefinance}
                  disabled={!debtItems.trim() || isRefinancing}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isRefinancing ? (
                    <Loader2 className="w-4 h-4 animate-spin ml-2" />
                  ) : (
                    <Wrench className="w-4 h-4 ml-2" />
                  )}
                  تحسين الكود
                </Button>

                {refinanceError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
                  >
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400">{refinanceError}</span>
                    <Button variant="ghost" size="sm" onClick={handleRefinance} className="ml-auto">
                      <RefreshCw className="w-3 h-3 ml-1" />
                      إعادة المحاولة
                    </Button>
                  </motion.div>
                )}

                <AnimatePresence>
                  {refinanceResult && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-4"
                    >
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                          <div className="flex items-center gap-2 mb-1">
                            <TrendingDown className="w-4 h-4 text-green-400" />
                            <span className="text-sm text-green-400">الأسطر المزالة</span>
                          </div>
                          <span className="text-2xl font-bold">{refinanceResult.savings.lines_removed}</span>
                        </div>
                        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                          <div className="flex items-center gap-2 mb-1">
                            <Zap className="w-4 h-4 text-blue-400" />
                            <span className="text-sm text-blue-400">تخفيف التعقيد</span>
                          </div>
                          <span className="text-2xl font-bold">{refinanceResult.savings.complexity_reduction}%</span>
                        </div>
                      </div>

                      <ScrollArea className="h-[300px]">
                        <div className="space-y-3">
                          {refinanceResult.results.map((result, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.05 }}
                              className="p-3 rounded-lg bg-secondary/30 border border-primary/10"
                            >
                              <div className="flex items-center gap-2 mb-1">
                                <FileCode className="w-4 h-4 text-primary" />
                                <span className="font-medium text-sm">{result.file}</span>
                              </div>
                              <p className="text-xs text-muted-foreground">{result.changes}</p>
                            </motion.div>
                          ))}
                        </div>
                      </ScrollArea>
                    </motion.div>
                  )}
                </AnimatePresence>
              </TabsContent>

              <TabsContent value="interest" className="mt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">المبلغ principal</label>
                    <Input
                      type="number"
                      value={principal}
                      onChange={(e) => setPrincipal(e.target.value)}
                      placeholder="أدخل المبلغ..."
                      dir="ltr"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">معدل الفائدة %</label>
                    <Input
                      type="number"
                      value={rate}
                      onChange={(e) => setRate(e.target.value)}
                      placeholder="نسبة الفائدة..."
                      dir="ltr"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">الفترة (أشهر)</label>
                    <Input
                      type="number"
                      value={timeMonths}
                      onChange={(e) => setTimeMonths(e.target.value)}
                      placeholder="عدد الأشهر..."
                      dir="ltr"
                    />
                  </div>
                </div>
                <Button
                  onClick={handleInterest}
                  disabled={!principal || !rate || !timeMonths || isCalculating}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isCalculating ? (
                    <Loader2 className="w-4 h-4 animate-spin ml-2" />
                  ) : (
                    <Calculator className="w-4 h-4 ml-2" />
                  )}
                  حساب الفائدة
                </Button>

                {interestError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20"
                  >
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <span className="text-sm text-red-400">{interestError}</span>
                    <Button variant="ghost" size="sm" onClick={handleInterest} className="ml-auto">
                      <RefreshCw className="w-3 h-3 ml-1" />
                      إعادة المحاولة
                    </Button>
                  </motion.div>
                )}

                <AnimatePresence>
                  {interestResult && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="p-4 rounded-lg bg-secondary/50 border border-primary/10">
                          <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-4 h-4 text-primary" />
                            <span className="text-sm text-muted-foreground">الفائدة البسيطة</span>
                          </div>
                          <span className="text-2xl font-bold">{interestResult.simple_interest.toFixed(2)}</span>
                        </div>
                        <div className="p-4 rounded-lg bg-secondary/50 border border-primary/10">
                          <div className="flex items-center gap-2 mb-2">
                            <TrendingDown className="w-4 h-4 text-green-400" />
                            <span className="text-sm text-muted-foreground">الفائدة المركبة</span>
                          </div>
                          <span className="text-2xl font-bold">{interestResult.compound_interest.toFixed(2)}</span>
                        </div>
                        <div className="p-4 rounded-lg bg-gradient-to-r from-primary/20 to-accent/20 border border-primary/30">
                          <div className="flex items-center gap-2 mb-2">
                            <Calculator className="w-4 h-4 text-accent" />
                            <span className="text-sm">المبلغ الإجمالي</span>
                          </div>
                          <span className="text-2xl font-bold gradient-text">{interestResult.total.toFixed(2)}</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
