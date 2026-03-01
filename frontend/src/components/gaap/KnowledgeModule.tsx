'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileCode, Pickaxe, BookOpen, Download, Loader2, AlertCircle, RefreshCw } from 'lucide-react'
import { apiPost } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'

interface ParseResult {
  classes: string[]
  functions: string[]
  imports: string[]
}

interface Pattern {
  name: string
  description: string
  examples: string[]
}

interface MineResult {
  patterns: Pattern[]
}

interface CheatSheetFunction {
  name: string
  signature: string
  patterns: string[]
}

interface CheatSheetResult {
  library_name: string
  functions: CheatSheetFunction[]
}

interface IngestResult {
  status: string
  files_parsed: number
  functions_found: number
  classes_found: number
}

export function KnowledgeModule() {
  const [parseCode, setParseCode] = useState('')
  const [parseFilePath, setParseFilePath] = useState('')
  const [parseLoading, setParseLoading] = useState(false)
  const [parseResult, setParseResult] = useState<ParseResult | null>(null)

  const [mineCode, setMineCode] = useState('')
  const [targetFunction, setTargetFunction] = useState('')
  const [mineLoading, setMineLoading] = useState(false)
  const [mineResult, setMineResult] = useState<MineResult | null>(null)

  const [libraryName, setLibraryName] = useState('')
  const [cheatFunctions, setCheatFunctions] = useState('')
  const [cheatLoading, setCheatLoading] = useState(false)
  const [cheatResult, setCheatResult] = useState<CheatSheetResult | null>(null)

  const [repoUrl, setRepoUrl] = useState('')
  const [ingestLibraryName, setIngestLibraryName] = useState('')
  const [ingestLoading, setIngestLoading] = useState(false)
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleParse = async () => {
    if (!parseCode.trim()) return
    setParseLoading(true)
    setError(null)
    try {
      const data = await apiPost<ParseResult>('/api/knowledge/parse', {
        code: parseCode,
        file_path: parseFilePath || undefined,
      })
      setParseResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في تحليل الكود'
      setError(message)
      setParseResult(null)
    } finally {
      setParseLoading(false)
    }
  }

  const handleMine = async () => {
    if (!mineCode.trim()) return
    setMineLoading(true)
    setError(null)
    try {
      const data = await apiPost<MineResult>('/api/knowledge/mine', {
        code: mineCode,
        target_function: targetFunction || undefined,
      })
      setMineResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في استخراج الأنماط'
      setError(message)
      setMineResult(null)
    } finally {
      setMineLoading(false)
    }
  }

  const handleCheatSheet = async () => {
    if (!libraryName.trim() || !cheatFunctions.trim()) return
    setCheatLoading(true)
    setError(null)
    try {
      const data = await apiPost<CheatSheetResult>('/api/knowledge/cheatsheet', {
        library_name: libraryName,
        functions: cheatFunctions.split(',').map(f => f.trim()).filter(Boolean),
      })
      setCheatResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في إنشاء الشيت المرجعي'
      setError(message)
      setCheatResult(null)
    } finally {
      setCheatLoading(false)
    }
  }

  const handleIngest = async () => {
    if (!repoUrl.trim() || !ingestLibraryName.trim()) return
    setIngestLoading(true)
    setError(null)
    try {
      const data = await apiPost<IngestResult>('/api/knowledge/ingest', {
        repo_url: repoUrl,
        library_name: ingestLibraryName,
      })
      setIngestResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في استيراد المستودع'
      setError(message)
      setIngestResult(null)
    } finally {
      setIngestLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <Card className="gradient-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              وحدة إدارة المعرفة
            </CardTitle>
            <CardDescription>
              تحليل الكود واستخراج الأنماط وإنشاء شيتات مرجعية
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            {error && (
              <div className="flex items-center justify-between p-3 mx-4 mt-4 text-sm text-red-500 bg-red-500/10 border border-red-500/20 rounded-lg">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  <span>{error}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const tab = document.querySelector('[data-state="active"][role="tab"]')?.getAttribute('value')
                    if (tab === 'parse') handleParse()
                    else if (tab === 'mine') handleMine()
                    else if (tab === 'cheatsheet') handleCheatSheet()
                    else if (tab === 'ingest') handleIngest()
                  }}
                  disabled={parseLoading || mineLoading || cheatLoading || ingestLoading}
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            )}

            <Tabs defaultValue="parse" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="parse" className="flex items-center gap-2">
                  <FileCode className="w-4 h-4" />
                  Parse
                </TabsTrigger>
                <TabsTrigger value="mine" className="flex items-center gap-2">
                  <Pickaxe className="w-4 h-4" />
                  Mine
                </TabsTrigger>
                <TabsTrigger value="cheatsheet" className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  CheatSheet
                </TabsTrigger>
                <TabsTrigger value="ingest" className="flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Ingest
                </TabsTrigger>
              </TabsList>

              <TabsContent value="parse" className="space-y-4 mt-4">
                <div className="space-y-4">
                  <Textarea
                    value={parseCode}
                    onChange={(e) => setParseCode(e.target.value)}
                    placeholder="أدخل الكود للتحليل..."
                    className="min-h-[150px] bg-secondary/50 font-mono text-sm"
                    dir="ltr"
                  />
                  <div className="flex gap-2">
                    <Input
                      value={parseFilePath}
                      onChange={(e) => setParseFilePath(e.target.value)}
                      placeholder="مسار الملف (اختياري)"
                      className="flex-1 bg-secondary/50"
                      dir="ltr"
                    />
                    <Button
                      onClick={handleParse}
                      disabled={!parseCode.trim() || parseLoading}
                      className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                    >
                      {parseLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'تحليل'}
                    </Button>
                  </div>
                </div>

                {parseResult && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-3">
                      <Card className="bg-secondary/30">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Classes ({parseResult.classes.length})</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ScrollArea className="h-[120px]">
                            <div className="space-y-1">
                              {parseResult.classes.map((cls, i) => (
                                <Badge key={i} variant="outline" className="block text-left bg-blue-500/10 text-blue-400">
                                  {cls}
                                </Badge>
                              ))}
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                      <Card className="bg-secondary/30">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Functions ({parseResult.functions.length})</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ScrollArea className="h-[120px]">
                            <div className="space-y-1">
                              {parseResult.functions.map((fn, i) => (
                                <Badge key={i} variant="outline" className="block text-left bg-green-500/10 text-green-400">
                                  {fn}
                                </Badge>
                              ))}
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                      <Card className="bg-secondary/30">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Imports ({parseResult.imports.length})</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <ScrollArea className="h-[120px]">
                            <div className="space-y-1">
                              {parseResult.imports.map((imp, i) => (
                                <Badge key={i} variant="outline" className="block text-left bg-purple-500/10 text-purple-400">
                                  {imp}
                                </Badge>
                              ))}
                            </div>
                          </ScrollArea>
                        </CardContent>
                      </Card>
                    </div>
                  </motion.div>
                )}
              </TabsContent>

              <TabsContent value="mine" className="space-y-4 mt-4">
                <div className="space-y-4">
                  <Textarea
                    value={mineCode}
                    onChange={(e) => setMineCode(e.target.value)}
                    placeholder="أدخل الكود لاستخراج الأنماط..."
                    className="min-h-[150px] bg-secondary/50 font-mono text-sm"
                    dir="ltr"
                  />
                  <div className="flex gap-2">
                    <Input
                      value={targetFunction}
                      onChange={(e) => setTargetFunction(e.target.value)}
                      placeholder="الدالة المستهدفة (اختياري)"
                      className="flex-1 bg-secondary/50"
                      dir="ltr"
                    />
                    <Button
                      onClick={handleMine}
                      disabled={!mineCode.trim() || mineLoading}
                      className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                    >
                      {mineLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'استخراج'}
                    </Button>
                  </div>
                </div>

                {mineResult && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                    {mineResult.patterns.map((pattern, i) => (
                      <Card key={i} className="bg-secondary/30">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">{pattern.name}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <p className="text-xs text-muted-foreground mb-2">{pattern.description}</p>
                          <div className="space-y-1">
                            {pattern.examples.map((ex, j) => (
                              <Badge key={j} variant="outline" className="block text-left bg-primary/10 font-mono text-xs">
                                {ex}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </motion.div>
                )}
              </TabsContent>

              <TabsContent value="cheatsheet" className="space-y-4 mt-4">
                <div className="space-y-4">
                  <div className="grid gap-2 md:grid-cols-2">
                    <Input
                      value={libraryName}
                      onChange={(e) => setLibraryName(e.target.value)}
                      placeholder="اسم المكتبة"
                      className="bg-secondary/50"
                      dir="ltr"
                    />
                    <Input
                      value={cheatFunctions}
                      onChange={(e) => setCheatFunctions(e.target.value)}
                      placeholder="الدوال (مفصولة بفواصل)"
                      className="bg-secondary/50"
                      dir="ltr"
                    />
                  </div>
                  <Button
                    onClick={handleCheatSheet}
                    disabled={!libraryName.trim() || !cheatFunctions.trim() || cheatLoading}
                    className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90"
                  >
                    {cheatLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'إنشاء'}
                  </Button>
                </div>

                {cheatResult && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <Badge className="bg-primary/20 text-primary">{cheatResult.library_name}</Badge>
                    </div>
                    {cheatResult.functions.map((fn, i) => (
                      <Card key={i} className="bg-secondary/30">
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">{fn.name}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <Badge variant="outline" className="mb-2 font-mono text-xs bg-orange-500/10 text-orange-400">
                            {fn.signature}
                          </Badge>
                          <div className="flex flex-wrap gap-1">
                            {fn.patterns.map((p, j) => (
                              <Badge key={j} variant="outline" className="text-xs bg-purple-500/10 text-purple-400">
                                {p}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </motion.div>
                )}
              </TabsContent>

              <TabsContent value="ingest" className="space-y-4 mt-4">
                <div className="space-y-4">
                  <div className="grid gap-2 md:grid-cols-2">
                    <Input
                      value={repoUrl}
                      onChange={(e) => setRepoUrl(e.target.value)}
                      placeholder="رابط المستودع"
                      className="bg-secondary/50"
                      dir="ltr"
                    />
                    <Input
                      value={ingestLibraryName}
                      onChange={(e) => setIngestLibraryName(e.target.value)}
                      placeholder="اسم المكتبة"
                      className="bg-secondary/50"
                    />
                  </div>
                  <Button
                    onClick={handleIngest}
                    disabled={!repoUrl.trim() || !ingestLibraryName.trim() || ingestLoading}
                    className="w-full bg-gradient-to-r from-primary to-accent hover:opacity-90"
                  >
                    {ingestLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'استيراد'}
                  </Button>
                </div>

                {ingestResult && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <Card className="bg-secondary/30">
                      <CardContent className="pt-6">
                        <div className="grid gap-4 md:grid-cols-4">
                          <div className="text-center">
                            <p className="text-sm text-muted-foreground">الحالة</p>
                            <Badge className="mt-1 bg-green-500/20 text-green-400">{ingestResult.status}</Badge>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-muted-foreground">الملفات</p>
                            <p className="text-2xl font-bold">{ingestResult.files_parsed}</p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-muted-foreground">الدوال</p>
                            <p className="text-2xl font-bold">{ingestResult.functions_found}</p>
                          </div>
                          <div className="text-center">
                            <p className="text-sm text-muted-foreground">ال Classes</p>
                            <p className="text-2xl font-bold">{ingestResult.classes_found}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
