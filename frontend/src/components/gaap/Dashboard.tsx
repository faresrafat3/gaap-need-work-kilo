'use client'

import { motion } from 'framer-motion'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import {
  Activity,
  Clock,
  CheckCircle,
  Globe,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  Zap,
  Layers,
  BarChart3,
  Cpu,
  RefreshCw,
  Loader2,
  WifiOff,
  Server,
} from 'lucide-react'
import { useGAAPStore } from '@/lib/store'
import { OODAVisualization } from './OODAVisualization'
import { ProviderStatusBadge } from './ProviderStatusBadge'
import { useLiveProviders, ProviderInfo } from '@/hooks/useLiveProviders'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useEffect, useMemo } from 'react'

const metrics = [
  {
    key: 'totalRequests',
    label: 'إجمالي الطلبات',
    icon: Layers,
    color: 'from-blue-500 to-cyan-500',
    iconColor: 'text-blue-500',
    bgColor: 'from-blue-500/10 to-cyan-500/10',
    description: 'جميع الطلبات المعالجة',
  },
  {
    key: 'avgLatency',
    label: 'متوسط التأخير (ms)',
    icon: Clock,
    color: 'from-amber-500 to-orange-500',
    iconColor: 'text-amber-500',
    bgColor: 'from-amber-500/10 to-orange-500/10',
    description: 'وقت الاستجابة المتوسط',
  },
  {
    key: 'successRate',
    label: 'معدل النجاح (%)',
    icon: CheckCircle,
    color: 'from-emerald-500 to-green-500',
    iconColor: 'text-emerald-500',
    bgColor: 'from-emerald-500/10 to-green-500/10',
    description: 'نسبة الطلبات الناجحة',
  },
  {
    key: 'activeUsers',
    label: 'المستخدمين النشطين',
    icon: Globe,
    color: 'from-violet-500 to-purple-500',
    iconColor: 'text-violet-500',
    bgColor: 'from-violet-500/10 to-purple-500/10',
    description: 'المستخدمين المتصلين حالياً',
  },
]

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.1,
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94] as const,
    },
  }),
  hover: {
    y: -8,
    scale: 1.02,
    transition: {
      duration: 0.3,
      ease: 'easeOut' as const,
    },
  },
}

const requestsData = [
  { time: '00:00', requests: 120 },
  { time: '04:00', requests: 80 },
  { time: '08:00', requests: 250 },
  { time: '12:00', requests: 420 },
  { time: '16:00', requests: 380 },
  { time: '20:00', requests: 290 },
  { time: '24:00', requests: 150 },
]

export function Dashboard() {
  const {
    systemMetrics,
    budget,
    oodaState,
    setOODAState,
    fetchSessions,
    fetchMetrics,
  } = useGAAPStore()

  const {
    providers,
    loading: providersLoading,
    error: providersError,
    lastUpdated,
    refresh,
    isRefreshing,
  } = useLiveProviders()

  // Calculate live provider stats
  const providerStats = useMemo(() => {
    if (!providers.length) return null

    const activeProviders = providers.filter((p) => p.status === 'active')
    const avgLatency = Math.round(
      providers.reduce((acc, p) => acc + (p.latency_ms || 0), 0) / providers.length
    )
    const avgSuccessRate =
      providers.reduce((acc, p) => acc + p.success_rate, 0) / providers.length

    return {
      activeCount: activeProviders.length,
      totalCount: providers.length,
      avgLatency,
      avgSuccessRate,
    }
  }, [providers])

  // Prepare provider data for chart
  const providerChartData = useMemo(() => {
    return providers.map((p) => ({
      name: p.display_name || p.name,
      requests: Math.round(p.success_rate * 100),
      latency: p.latency_ms || 0,
      success: p.success_rate,
      status: p.status,
    }))
  }, [providers])

  // Fetch sessions and metrics on mount
  useEffect(() => {
    const fetchData = async () => {
      await Promise.all([fetchSessions(), fetchMetrics()])
    }
    fetchData()
  }, [fetchSessions, fetchMetrics])

  // Simulate OODA loop animation
  useEffect(() => {
    const stages = ['observe', 'orient', 'decide', 'act'] as const
    let currentIndex = 0

    const interval = setInterval(() => {
      // Reset all stages
      stages.forEach((stage) => {
        setOODAState({
          [stage]: { status: 'idle', data: '' },
        })
      })

      // Process current stage
      const currentStage = stages[currentIndex]
      setOODAState({
        [currentStage]: { status: 'processing', data: 'جاري المعالجة...' },
      })

      // Complete previous stage
      const prevIndex = (currentIndex - 1 + stages.length) % stages.length
      const prevStage = stages[prevIndex]
      setOODAState({
        [prevStage]: { status: 'complete', data: 'تم بنجاح' },
      })

      currentIndex = (currentIndex + 1) % stages.length
    }, 2000)

    return () => clearInterval(interval)
  }, [setOODAState])

  const budgetPercentage = (budget.used / budget.limit) * 100

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold gradient-text">لوحة التحكم</h1>
          <p className="text-muted-foreground mt-1">نظرة عامة على نظام GAAP</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Last Updated Badge */}
          {lastUpdated && (
            <Badge variant="outline" className="text-xs gap-1.5">
              <Clock className="w-3 h-3" />
              آخر تحديث: {lastUpdated.toLocaleTimeString('ar-SA')}
            </Badge>
          )}
          {/* Refresh Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={refresh}
            disabled={isRefreshing}
            className="gap-2"
          >
            {isRefreshing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            تحديث
          </Button>
          <Badge
            variant="outline"
            className="bg-green-500/10 text-green-400 border-green-500/20"
          >
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
            النظام يعمل
          </Badge>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          const value = systemMetrics[metric.key as keyof typeof systemMetrics]
          return (
            <motion.div
              key={metric.key}
              custom={index}
              initial="hidden"
              animate="visible"
              whileHover="hover"
              variants={cardVariants}
            >
              <Card className="relative overflow-hidden group cursor-pointer border-0 shadow-lg hover:shadow-2xl transition-shadow duration-300">
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${metric.bgColor} opacity-0 group-hover:opacity-100 transition-opacity duration-300`}
                />
                <div
                  className={`absolute inset-0 bg-gradient-to-r ${metric.color} opacity-0 group-hover:opacity-20 transition-opacity duration-300 rounded-lg`}
                />

                <CardContent className="relative z-10 p-6">
                  <div className="flex items-start justify-between">
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                          {metric.label}
                        </p>
                        <p className="text-3xl font-bold mt-2 tracking-tight bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text">
                          {typeof value === 'number'
                            ? value.toLocaleString('ar-SA')
                            : value}
                        </p>
                      </div>
                      <p className="text-xs text-muted-foreground/80 leading-relaxed">
                        {metric.description}
                      </p>
                    </div>
                    <motion.div
                      className={`p-4 rounded-2xl bg-gradient-to-br ${metric.color} shadow-lg`}
                      whileHover={{ rotate: 5, scale: 1.1 }}
                      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
                    >
                      <Icon className="w-6 h-6 text-white" />
                    </motion.div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-border/50">
                    <div className="flex items-center gap-2">
                      <div
                        className={`h-1.5 flex-1 rounded-full bg-gradient-to-r ${metric.color} opacity-60`}
                        style={{ width: '60%' }}
                      />
                      <span className={`text-xs font-semibold ${metric.iconColor}`}>
                        +12%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          <Card className="p-6 border-0 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 shadow-md">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  حركة الطلبات عبر الوقت
                </h3>
                <p className="text-xs text-muted-foreground">تحليل استخدام النظام</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={requestsData}>
                <defs>
                  <linearGradient id="colorRequests" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="hsl(var(--primary))"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="hsl(var(--primary))"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                  opacity={0.5}
                />
                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  stroke="hsl(var(--muted-foreground))"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="requests"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  fill="url(#colorRequests)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          <Card className="p-6 border-0 shadow-lg hover:shadow-xl transition-shadow duration-300">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 shadow-md">
                  <Cpu className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-foreground">
                    أداء المزودين
                  </h3>
                  <p className="text-xs text-muted-foreground">
                    {providerStats ? (
                      <>
                        {providerStats.activeCount} من {providerStats.totalCount} مزود نشط
                        {' • '}
                        متوسط التأخير: {providerStats.avgLatency}ms
                      </>
                    ) : (
                      'جاري التحميل...'
                    )}
                  </p>
                </div>
              </div>
              {isRefreshing && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
            </div>

            {providersLoading && providers.length === 0 ? (
              <div className="h-[220px] flex items-center justify-center">
                <div className="space-y-2 w-full px-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-3/4" />
                  <Skeleton className="h-8 w-1/2" />
                </div>
              </div>
            ) : providersError && providers.length === 0 ? (
              <div className="h-[220px] flex flex-col items-center justify-center gap-2 text-muted-foreground">
                <WifiOff className="w-8 h-8" />
                <p className="text-sm">فشل تحميل البيانات</p>
                <Button variant="outline" size="sm" onClick={refresh} className="gap-2">
                  <RefreshCw className="w-4 h-4" />
                  إعادة المحاولة
                </Button>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={providerChartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                    opacity={0.5}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    stroke="hsl(var(--muted-foreground))"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    stroke="hsl(var(--muted-foreground))"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                    }}
                    formatter={(value: number, name: string) => {
                      if (name === 'latency') return [`${value}ms`, 'التأخير']
                      if (name === 'success') return [`${value.toFixed(1)}%`, 'معدل النجاح']
                      return [value, name]
                    }}
                  />
                  <Bar dataKey="latency" radius={[8, 8, 0, 0]} name="latency">
                    {providerChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.status === 'active'
                            ? entry.latency < 200
                              ? '#10b981'
                              : entry.latency < 500
                              ? '#f59e0b'
                              : '#ef4444'
                            : '#6b7280'
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        </motion.div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* OODA Visualization */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-2"
        >
          <Card className="gradient-border h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary" />
                حلقة OODA
              </CardTitle>
              <CardDescription>مراقبة - توجيه - قرار - تنفيذ</CardDescription>
            </CardHeader>
            <CardContent>
              <OODAVisualization />
            </CardContent>
          </Card>
        </motion.div>

        {/* Budget Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="gradient-border h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="w-5 h-5 text-primary" />
                ميزانية النظام
              </CardTitle>
              <CardDescription>تتبع الاستخدام والتكاليف</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">المستخدم</span>
                  <span className="font-medium">${budget.used.toFixed(2)}</span>
                </div>
                <Progress value={budgetPercentage} className="h-2" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{budgetPercentage.toFixed(1)}%</span>
                  <span>الحد: ${budget.limit.toLocaleString()}</span>
                </div>
              </div>

              {budgetPercentage > 80 && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-2 p-2 rounded-lg bg-yellow-500/10 text-yellow-400 text-sm"
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>تحذير: الميزانية قاربت على النفاد</span>
                </motion.div>
              )}

              <div className="pt-4 border-t border-primary/10">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">إعادة التعيين</span>
                  <span className="font-medium">
                    {Math.ceil(
                      (budget.resetDate.getTime() - Date.now()) /
                        (1000 * 60 * 60 * 24)
                    )}{' '}
                    يوم
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Provider Status Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Card className="gradient-border">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                حالة المزودين
              </CardTitle>
              <CardDescription>مراقبة أداء مزودي الذكاء الاصطناعي</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {lastUpdated && (
                <span className="text-xs text-muted-foreground">
                  تحديث: {lastUpdated.toLocaleTimeString('ar-SA')}
                </span>
              )}
              <Button
                variant="outline"
                size="icon"
                onClick={refresh}
                disabled={isRefreshing}
                className="h-8 w-8"
              >
                {isRefreshing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {providersLoading && providers.length === 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="p-4 rounded-lg border bg-secondary/50 space-y-3">
                    <div className="flex items-center justify-between">
                      <Skeleton className="h-5 w-24" />
                      <Skeleton className="h-5 w-16" />
                    </div>
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                ))}
              </div>
            ) : providersError && providers.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 gap-3">
                <div className="p-3 rounded-full bg-red-500/10">
                  <WifiOff className="w-8 h-8 text-red-500" />
                </div>
                <p className="text-muted-foreground">تعذر تحميل بيانات المزودين</p>
                <Button onClick={refresh} variant="outline" className="gap-2">
                  <RefreshCw className="w-4 h-4" />
                  إعادة المحاولة
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {providers.map((provider, index) => (
                  <motion.div
                    key={provider.name}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                    className={`
                      p-4 rounded-lg bg-secondary/50 border transition-all duration-200
                      hover:shadow-md hover:border-primary/30
                      ${(provider as ProviderInfo & { isStale?: boolean }).status !== 'active' ? 'opacity-75' : ''}
                      ${(provider as ProviderInfo & { isStale?: boolean }).isStale ? 'border-amber-500/30' : 'border-primary/10'}
                    `}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Server className="w-4 h-4 text-muted-foreground" />
                        <span className="font-medium capitalize">
                          {provider.display_name}
                        </span>
                      </div>
                      <ProviderStatusBadge
                        provider={provider}
                        isRefreshing={isRefreshing}
                        size="sm"
                      />
                    </div>

                    {/* Model Info */}
                    <div className="mb-3">
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                        النموذج
                      </p>
                      <code className="text-xs bg-muted px-2 py-1 rounded block truncate">
                        {provider.actual_model}
                      </code>
                    </div>

                    {/* Stats Grid */}
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          التأخير
                        </p>
                        <p
                          className={`font-medium font-mono ${
                            (provider.latency_ms || 0) < 200
                              ? 'text-emerald-500'
                              : (provider.latency_ms || 0) < 500
                              ? 'text-amber-500'
                              : 'text-red-500'
                          }`}
                        >
                          {provider.latency_ms}ms
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground flex items-center gap-1">
                          <CheckCircle className="w-3 h-3" />
                          النجاح
                        </p>
                        <p
                          className={`font-medium font-mono ${
                            provider.success_rate > 95
                              ? 'text-emerald-500'
                              : provider.success_rate > 80
                              ? 'text-amber-500'
                              : 'text-red-500'
                          }`}
                        >
                          {provider.success_rate.toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {/* Stale Warning */}
                    {(provider as ProviderInfo & { isStale?: boolean }).isStale && (
                      <div className="mt-3 pt-3 border-t border-border/50">
                        <p className="text-[10px] text-amber-500 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          البيانات قديمة •{' '}
                          {new Date(provider.last_seen || '').toLocaleTimeString('ar-SA')}
                        </p>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
