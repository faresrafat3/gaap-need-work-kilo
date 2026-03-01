'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, ListTodo, Award, Users, BarChart3, Loader2, CheckCircle, XCircle, Clock, AlertCircle, RefreshCw } from 'lucide-react'
import { apiPost, apiGet } from '@/lib/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface Fractal {
  id: string
  specialization: string
  capabilities: string[]
  status: string
}

interface Task {
  id: string
  task: string
  domain: string
  priority: string
  status: string
  fractal_id: string | null
  result: string | null
}

interface ReputationEntry {
  fractal_id: string
  score: number
}

interface Guild {
  id: string
  name: string
  members: string[]
  avg_score: number
}

interface Metrics {
  total_fractals: number
  active_tasks: number
  completed_tasks: number
  avg_reputation: number
}

export function SwarmModule() {
  const [fractalId, setFractalId] = useState('')
  const [specialization, setSpecialization] = useState('')
  const [capabilities, setCapabilities] = useState('')
  const [registerStatus, setRegisterStatus] = useState<string | null>(null)
  const [isRegistering, setIsRegistering] = useState(false)

  const [taskName, setTaskName] = useState('')
  const [domain, setDomain] = useState('')
  const [priority, setPriority] = useState('medium')
  const [tasks, setTasks] = useState<Task[]>([])
  const [isExecuting, setIsExecuting] = useState(false)

  const [reputation, setReputation] = useState<ReputationEntry[]>([])
  const [isLoadingReputation, setIsLoadingReputation] = useState(false)

  const [guilds, setGuilds] = useState<Guild[]>([])
  const [isLoadingGuilds, setIsLoadingGuilds] = useState(false)

  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false)

  const [error, setError] = useState<string | null>(null)

  const handleRegister = async () => {
    if (!fractalId.trim() || !specialization.trim()) return

    setIsRegistering(true)
    setRegisterStatus(null)
    setError(null)

    try {
      const data = await apiPost<{ success: boolean }>('/api/swarm/fractal/register', {
        fractal_id: fractalId,
        specialization,
        capabilities: capabilities.split(',').map(c => c.trim()).filter(Boolean),
      })

      if (data.success) {
        setRegisterStatus('success')
        setFractalId('')
        setSpecialization('')
        setCapabilities('')
      } else {
        setRegisterStatus('error')
        setError('فشل في تسجيل الوكيل')
      }
    } catch (err) {
      setRegisterStatus('error')
      setError('فشل في تسجيل الوكيل')
    } finally {
      setIsRegistering(false)
    }
  }

  const handleExecuteTask = async () => {
    if (!taskName.trim()) return

    setIsExecuting(true)
    setError(null)

    try {
      const data = await apiPost<{
        status: string
        fractal_id: string | null
        result: string | null
      }>('/api/swarm/task', {
        task: taskName,
        domain,
        priority,
      })

      const newTask: Task = {
        id: Date.now().toString(),
        task: taskName,
        domain,
        priority,
        status: data.status || 'pending',
        fractal_id: data.fractal_id || null,
        result: data.result || null,
      }

      setTasks([newTask, ...tasks])
      setTaskName('')
      setDomain('')
      setPriority('medium')
    } catch (err) {
      console.error('Task error:', err)
      setError('فشل في تنفيذ المهمة')
    } finally {
      setIsExecuting(false)
    }
  }

  const loadReputation = async () => {
    setIsLoadingReputation(true)
    setError(null)

    try {
      const data = await apiGet<{ reputation: ReputationEntry[] }>('/api/swarm/reputation')
      setReputation(data.reputation || [])
    } catch (err) {
      console.error('Reputation error:', err)
      setError('فشل في تحميل السمعة')
    } finally {
      setIsLoadingReputation(false)
    }
  }

  const loadGuilds = async () => {
    setIsLoadingGuilds(true)
    setError(null)

    try {
      const data = await apiGet<{ guilds: Guild[] }>('/api/swarm/guilds')
      setGuilds(data.guilds || [])
    } catch (err) {
      console.error('Guilds error:', err)
      setError('فشل في تحميل المجموعات')
    } finally {
      setIsLoadingGuilds(false)
    }
  }

  const loadMetrics = async () => {
    setIsLoadingMetrics(true)
    setError(null)

    try {
      const data = await apiGet<Metrics>('/api/swarm/metrics')
      setMetrics(data)
    } catch (err) {
      console.error('Metrics error:', err)
      setError('فشل في تحميل الإحصائيات')
    } finally {
      setIsLoadingMetrics(false)
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-500/10 text-red-400'
      case 'medium': return 'bg-yellow-500/10 text-yellow-400'
      case 'low': return 'bg-green-500/10 text-green-400'
      default: return 'bg-primary/10'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400'
      case 'in_progress': return 'text-blue-400'
      case 'failed': return 'text-red-400'
      default: return 'text-yellow-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-3 h-3" />
      case 'in_progress': return <Clock className="w-3 h-3" />
      case 'failed': return <XCircle className="w-3 h-3" />
      default: return <Clock className="w-3 h-3" />
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
              <Bot className="w-5 h-5 text-primary" />
              وحدة السرب (Swarm)
            </CardTitle>
            <CardDescription>
              إدارة الوكلاء والمهام والسمعة والمجموعات
            </CardDescription>
          </CardHeader>
          {error && (
            <div className="mx-6 mt-2 flex items-center justify-between p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setError(null)
                    loadReputation()
                    loadGuilds()
                    loadMetrics()
                  }}
                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                >
                  <RefreshCw className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setError(null)}
                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                >
                  <XCircle className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
          <CardContent>
            <Tabs defaultValue="fractals" className="w-full">
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="fractals" className="flex items-center gap-2">
                  <Bot className="w-4 h-4" />
                  Fractals
                </TabsTrigger>
                <TabsTrigger value="tasks" className="flex items-center gap-2">
                  <ListTodo className="w-4 h-4" />
                  Tasks
                </TabsTrigger>
                <TabsTrigger value="reputation" className="flex items-center gap-2">
                  <Award className="w-4 h-4" />
                  Reputation
                </TabsTrigger>
                <TabsTrigger value="guilds" className="flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Guilds
                </TabsTrigger>
                <TabsTrigger value="metrics" className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Metrics
                </TabsTrigger>
              </TabsList>

              <TabsContent value="fractals" className="mt-4 space-y-4">
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">Fractal ID</label>
                    <Input
                      value={fractalId}
                      onChange={(e) => setFractalId(e.target.value)}
                      placeholder="أدخل معرف الوكيل..."
                      dir="rtl"
                    />
                  </div>
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">التخصص</label>
                    <Input
                      value={specialization}
                      onChange={(e) => setSpecialization(e.target.value)}
                      placeholder="أدخل التخصص..."
                      dir="rtl"
                    />
                  </div>
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">القدرات (مفصولة بفواصل)</label>
                    <Input
                      value={capabilities}
                      onChange={(e) => setCapabilities(e.target.value)}
                      placeholder="coding, research, analysis..."
                      dir="ltr"
                    />
                  </div>
                  <Button
                    onClick={handleRegister}
                    disabled={!fractalId.trim() || !specialization.trim() || isRegistering}
                    className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                  >
                    {isRegistering ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      'تسجيل'
                    )}
                  </Button>
                  {registerStatus && (
                    <div className={`flex items-center gap-2 p-3 rounded-lg ${
                      registerStatus === 'success' 
                        ? 'bg-green-500/10 text-green-400' 
                        : 'bg-red-500/10 text-red-400'
                    }`}>
                      {registerStatus === 'success' ? (
                        <><CheckCircle className="w-4 h-4" /> تم التسجيل بنجاح</>
                      ) : (
                        <><XCircle className="w-4 h-4" /> فشل التسجيل</>
                      )}
                    </div>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="tasks" className="mt-4 space-y-4">
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">المهمة</label>
                    <Input
                      value={taskName}
                      onChange={(e) => setTaskName(e.target.value)}
                      placeholder="أدخل المهمة..."
                      dir="rtl"
                    />
                  </div>
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">النطاق</label>
                    <Input
                      value={domain}
                      onChange={(e) => setDomain(e.target.value)}
                      placeholder="أدخل النطاق..."
                      dir="rtl"
                    />
                  </div>
                  <div className="grid gap-2">
                    <label className="text-sm font-medium">الأولوية</label>
                    <div className="flex gap-2">
                      {['low', 'medium', 'high'].map((p) => (
                        <Button
                          key={p}
                          variant={priority === p ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => setPriority(p)}
                          className={priority === p ? 'bg-primary' : ''}
                        >
                          {p === 'high' ? 'عالية' : p === 'medium' ? 'متوسطة' : 'منخفضة'}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <Button
                    onClick={handleExecuteTask}
                    disabled={!taskName.trim() || isExecuting}
                    className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                  >
                    {isExecuting ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      'تنفيذ'
                    )}
                  </Button>
                </div>

                {tasks.length > 0 && (
                  <ScrollArea className="h-64 mt-4">
                    <div className="space-y-2">
                      {tasks.map((task) => (
                        <Card key={task.id} className="p-4 bg-secondary/30">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1">
                              <p className="font-medium">{task.task}</p>
                              <div className="flex items-center gap-2 mt-1">
                                <Badge className={getPriorityColor(task.priority)}>
                                  {task.priority === 'high' ? 'عالية' : task.priority === 'medium' ? 'متوسطة' : 'منخفضة'}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {task.domain}
                                </span>
                              </div>
                            </div>
                            <div className="flex flex-col items-end gap-1">
                              <span className={`text-sm flex items-center gap-1 ${getStatusColor(task.status)}`}>
                                {getStatusIcon(task.status)}
                                {task.status === 'completed' ? 'مكتمل' : task.status === 'in_progress' ? 'قيد التنفيذ' : task.status === 'failed' ? 'فشل' : 'معلق'}
                              </span>
                              {task.fractal_id && (
                                <span className="text-xs text-muted-foreground">
                                  Agent: {task.fractal_id}
                                </span>
                              )}
                            </div>
                          </div>
                          {task.result && (
                            <div className="mt-2 p-2 rounded bg-primary/10 text-sm">
                              {task.result}
                            </div>
                          )}
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>

              <TabsContent value="reputation" className="mt-4 space-y-4">
                <Button
                  onClick={loadReputation}
                  disabled={isLoadingReputation}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isLoadingReputation ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'عرض'
                  )}
                </Button>

                {reputation.length > 0 && (
                  <ScrollArea className="h-80">
                    <div className="space-y-2">
                      {reputation.map((entry, index) => (
                        <Card key={index} className="p-4 bg-secondary/30">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                                <Bot className="w-5 h-5 text-primary" />
                              </div>
                              <span className="font-medium">{entry.fractal_id}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Award className="w-4 h-4 text-yellow-400" />
                              <span className="text-lg font-bold text-yellow-400">
                                {entry.score.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>

              <TabsContent value="guilds" className="mt-4 space-y-4">
                <Button
                  onClick={loadGuilds}
                  disabled={isLoadingGuilds}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isLoadingGuilds ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'عرض'
                  )}
                </Button>

                {guilds.length > 0 && (
                  <ScrollArea className="h-80">
                    <div className="space-y-2">
                      {guilds.map((guild) => (
                        <Card key={guild.id} className="p-4 bg-secondary/30">
                          <div className="flex items-start justify-between">
                            <div>
                              <h4 className="font-medium flex items-center gap-2">
                                <Users className="w-4 h-4 text-primary" />
                                {guild.name}
                              </h4>
                              <p className="text-sm text-muted-foreground mt-1">
                                الأعضاء: {guild.members.length}
                              </p>
                            </div>
                            <div className="text-left">
                              <p className="text-xs text-muted-foreground">متوسط السمعة</p>
                              <p className="text-lg font-bold text-yellow-400">
                                {guild.avg_score.toFixed(2)}
                              </p>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </TabsContent>

              <TabsContent value="metrics" className="mt-4 space-y-4">
                <Button
                  onClick={loadMetrics}
                  disabled={isLoadingMetrics}
                  className="bg-gradient-to-r from-primary to-accent hover:opacity-90"
                >
                  {isLoadingMetrics ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    'عرض'
                  )}
                </Button>

                {metrics && (
                  <div className="grid grid-cols-2 gap-4">
                    <Card className="p-4 bg-secondary/30">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center">
                          <Bot className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">إجمالي الوكلاء</p>
                          <p className="text-2xl font-bold">{metrics.total_fractals}</p>
                        </div>
                      </div>
                    </Card>

                    <Card className="p-4 bg-secondary/30">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
                          <Clock className="w-6 h-6 text-blue-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">المهام النشطة</p>
                          <p className="text-2xl font-bold text-blue-400">{metrics.active_tasks}</p>
                        </div>
                      </div>
                    </Card>

                    <Card className="p-4 bg-secondary/30">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                          <CheckCircle className="w-6 h-6 text-green-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">المهام المكتملة</p>
                          <p className="text-2xl font-bold text-green-400">{metrics.completed_tasks}</p>
                        </div>
                      </div>
                    </Card>

                    <Card className="p-4 bg-secondary/30">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center">
                          <Award className="w-6 h-6 text-yellow-400" />
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">متوسط السمعة</p>
                          <p className="text-2xl font-bold text-yellow-400">
                            {metrics.avg_reputation.toFixed(2)}
                          </p>
                        </div>
                      </div>
                    </Card>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
