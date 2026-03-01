'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Activity, Clock, Zap } from 'lucide-react'
import { perfMonitor } from '@/lib/performance'

interface MetricDisplay {
  name: string
  avg: number
  count: number
  unit: string
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState<MetricDisplay[]>([])
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  useEffect(() => {
    const interval = setInterval(() => {
      const report = perfMonitor.reportMetrics()
      const formatted = Object.entries(report).map(([name, data]) => ({
        name,
        ...data,
      }))
      setMetrics(formatted)
      setLastUpdate(new Date())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (name: string, avg: number): string => {
    if (name.startsWith('api_')) {
      if (avg < 200) return 'bg-green-500'
      if (avg < 500) return 'bg-yellow-500'
      return 'bg-red-500'
    }
    return 'bg-blue-500'
  }

  return (
    <Card className="gradient-border">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Activity className="w-5 h-5 text-primary" />
          مراقبة الأداء
          <Badge variant="outline" className="mr-auto text-xs">
            آخر تحديث: {lastUpdate.toLocaleTimeString('ar-SA')}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {metrics.length === 0 ? (
          <div className="text-center text-muted-foreground py-4">
            جاري جمع البيانات...
          </div>
        ) : (
          <div className="space-y-3">
            {metrics.map((metric) => (
              <div
                key={metric.name}
                className="flex items-center justify-between p-3 rounded-lg bg-secondary/30"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-3 h-3 rounded-full ${getStatusColor(
                      metric.name,
                      metric.avg
                    )}`}
                  />
                  <div>
                    <p className="font-medium text-sm">
                      {metric.name.replace(/_/g, ' ')}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {metric.count} قياس
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-mono font-bold">
                    {metric.avg.toFixed(1)} {metric.unit}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
