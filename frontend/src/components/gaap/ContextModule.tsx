'use client'

import { useState } from 'react'
import { Scissors, Network, Search, Play, FileCode, GitBranch, Hash, Loader2, AlertCircle, RefreshCw } from 'lucide-react'
import { apiPost } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'

interface Chunk {
  id: string
  type: string
  start_line: number
  end_line: number
  content?: string
}

interface ChunkResult {
  chunks: Chunk[]
  file_path?: string
  total_chunks: number
}

interface CallGraphResult {
  nodes: number
  edges: number
  files: number
  graph?: Record<string, unknown>
}

interface SearchResult {
  id: string
  content: string
  score: number
  file_path?: string
}

interface SearchResults {
  results: SearchResult[]
  query: string
}

const contextTypes = [
  { id: 'chunking', label: 'Smart Chunking', icon: Scissors, description: 'تقسيم الكود ذكياً' },
  { id: 'call-graph', label: 'Call Graph', icon: Network, description: 'تحليل الاعتماديات' },
  { id: 'search', label: 'Semantic Search', icon: Search, description: 'البحث الدلالي' },
]

export function ContextModule() {
  const [code, setCode] = useState('')
  const [filePath, setFilePath] = useState('')
  const [projectPath, setProjectPath] = useState('')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState('5')
  const [activeTab, setActiveTab] = useState('chunking')
  const [chunkResult, setChunkResult] = useState<ChunkResult | null>(null)
  const [callGraphResult, setCallGraphResult] = useState<CallGraphResult | null>(null)
  const [searchResults, setSearchResults] = useState<SearchResults | null>(null)
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleChunking = async () => {
    if (!code.trim()) return

    setLoading('chunking')
    setError(null)
    try {
      const data = await apiPost<ChunkResult>('/api/context/chunk', { 
        code, 
        file_path: filePath || undefined 
      })
      setChunkResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في تقسيم الكود'
      setError(message)
      setChunkResult({ chunks: [], total_chunks: 0 })
    }
    setLoading(null)
  }

  const handleCallGraph = async () => {
    if (!projectPath.trim()) return

    setLoading('call-graph')
    setError(null)
    try {
      const data = await apiPost<CallGraphResult>('/api/context/call-graph', { project_path: projectPath })
      setCallGraphResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في بناء الـ Call Graph'
      setError(message)
      setCallGraphResult({ nodes: 0, edges: 0, files: 0 })
    }
    setLoading(null)
  }

  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading('search')
    setError(null)
    try {
      const data = await apiPost<SearchResults>('/api/context/search', { 
        query, 
        top_k: topK ? parseInt(topK) : 5 
      })
      setSearchResults(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'فشل في البحث'
      setError(message)
      setSearchResults({ results: [], query })
    }
    setLoading(null)
  }

  const getChunkTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'function': return 'bg-blue-500/10 border-blue-500/20 text-blue-500'
      case 'class': return 'bg-purple-500/10 border-purple-500/20 text-purple-500'
      case 'import': return 'bg-green-500/10 border-green-500/20 text-green-500'
      case 'variable': return 'bg-yellow-500/10 border-yellow-500/20 text-yellow-500'
      default: return 'bg-gray-500/10 border-gray-500/20 text-gray-500'
    }
  }

  return (
    <Card className="w-full overflow-hidden">
      <CardHeader className="pb-3 border-b">
        <CardTitle className="flex items-center gap-2 text-lg">
          <FileCode className="w-5 h-5 text-primary" />
          سياق الكود
        </CardTitle>
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
                if (activeTab === 'chunking') handleChunking()
                else if (activeTab === 'call-graph') handleCallGraph()
                else if (activeTab === 'search') handleSearch()
              }}
              disabled={loading !== null}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="border-b px-4 pt-3">
            <TabsList className="grid w-full grid-cols-3 h-auto p-0 bg-transparent gap-1">
              {contextTypes.map((type) => (
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

          <TabsContent value="chunking" className="p-4 space-y-4 m-0">
            <div className="space-y-2">
              <label className="text-sm font-medium">كود الإدخال</label>
              <Textarea
                placeholder="الصق الكود المراد تقسيمه هنا..."
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="min-h-[180px] font-mono text-sm resize-none"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">مسار الملف (اختياري)</label>
              <Input
                placeholder="src/utils/helper.py"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
              />
            </div>

            <Button
              onClick={handleChunking}
              disabled={!code.trim() || loading !== null}
              className="w-full gap-2"
            >
              {loading === 'chunking' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Scissors className="w-4 h-4" />
              )}
              تقسيم
            </Button>

            {chunkResult && (
              <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2">
                    <FileCode className="w-5 h-5 text-primary" />
                    <span className="font-medium">النتائج</span>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {chunkResult.total_chunks} chunks
                  </Badge>
                </div>

                {chunkResult.chunks.length > 0 ? (
                  <ScrollArea className="h-[250px] rounded-md border">
                    <div className="p-3 space-y-2">
                      {chunkResult.chunks.map((chunk, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <Badge className={`text-xs ${getChunkTypeColor(chunk.type)}`}>
                              {chunk.type}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              سطر {chunk.start_line}-{chunk.end_line}
                            </Badge>
                          </div>
                          {chunk.content && (
                            <pre className="mt-2 text-xs font-mono text-muted-foreground bg-muted p-2 rounded overflow-x-auto">
                              {chunk.content.slice(0, 100)}...
                            </pre>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    لم يتم العثور على chunks
                  </p>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="call-graph" className="p-4 space-y-4 m-0">
            <div className="space-y-2">
              <label className="text-sm font-medium">مسار المشروع</label>
              <Input
                placeholder="/path/to/project"
                value={projectPath}
                onChange={(e) => setProjectPath(e.target.value)}
              />
            </div>

            <Button
              onClick={handleCallGraph}
              disabled={!projectPath.trim() || loading !== null}
              className="w-full gap-2"
            >
              {loading === 'call-graph' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Network className="w-4 h-4" />
              )}
              بناء
            </Button>

            {callGraphResult && (
              <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-lg border bg-card text-center">
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <GitBranch className="w-4 h-4 text-blue-500" />
                      <span className="text-xs text-muted-foreground">Nodes</span>
                    </div>
                    <span className="text-2xl font-bold">{callGraphResult.nodes}</span>
                  </div>
                  <div className="p-3 rounded-lg border bg-card text-center">
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <Network className="w-4 h-4 text-purple-500" />
                      <span className="text-xs text-muted-foreground">Edges</span>
                    </div>
                    <span className="text-2xl font-bold">{callGraphResult.edges}</span>
                  </div>
                  <div className="p-3 rounded-lg border bg-card text-center">
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <FileCode className="w-4 h-4 text-green-500" />
                      <span className="text-xs text-muted-foreground">Files</span>
                    </div>
                    <span className="text-2xl font-bold">{callGraphResult.files}</span>
                  </div>
                </div>

                {callGraphResult.graph && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">الـ Graph</h4>
                    <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto border max-h-[200px] overflow-auto">
                      {JSON.stringify(callGraphResult.graph, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="search" className="p-4 space-y-4 m-0">
            <div className="space-y-2">
              <label className="text-sm font-medium">استعلام البحث</label>
              <Textarea
                placeholder="ابحث في قاعدة المعرفة..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-h-[80px] resize-none"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">عدد النتائج (اختياري)</label>
              <Input
                placeholder="5"
                type="number"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>

            <Button
              onClick={handleSearch}
              disabled={!query.trim() || loading !== null}
              className="w-full gap-2"
            >
              {loading === 'search' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              بحث
            </Button>

            {searchResults && (
              <div className="space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/50">
                  <div className="flex items-center gap-2">
                    <Search className="w-5 h-5 text-primary" />
                    <span className="font-medium">نتائج البحث</span>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {searchResults.results.length} result{searchResults.results.length !== 1 ? 's' : ''}
                  </Badge>
                </div>

                {searchResults.results.length > 0 ? (
                  <ScrollArea className="h-[280px] rounded-md border">
                    <div className="p-3 space-y-2">
                      {searchResults.results.map((result, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-2 mb-2">
                            <div className="flex items-center gap-2">
                              <Hash className="w-3 h-3 text-muted-foreground" />
                              <span className="text-xs text-muted-foreground">#{i + 1}</span>
                              {result.file_path && (
                                <Badge variant="secondary" className="text-xs">
                                  {result.file_path}
                                </Badge>
                              )}
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {(result.score * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <p className="text-sm">{result.content}</p>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    لم يتم العثور على نتائج
                  </p>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
