'use client'

import { useState, useEffect } from 'react'
import { Bot, Zap, Globe, CheckCircle, XCircle, AlertCircle, Users } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

interface ProviderStatus {
  name: string
  accounts: {
    label: string
    status: 'active' | 'expired' | 'error'
    remainingMinutes: number
  }[]
}

export function ProviderStatus() {
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/providers')
      .then(res => res.json())
      .then(data => {
        const transformed = data.map((p: any) => ({
          name: p.name,
          model: p.model,
          accounts: [
            { 
              label: 'default', 
              status: p.status === 'active' ? 'active' : 'expired', 
              remainingMinutes: p.status === 'active' ? 25000 : 0 
            }
          ]
        }))
        setProviders(transformed)
      })
      .catch(err => {
        console.error('Failed to fetch providers:', err)
        setProviders([])
      })
      .finally(() => setLoading(false))
  }, [])

  const getProviderIcon = (name: string) => {
    switch(name) {
      case 'kimi': return <Bot className="w-4 h-4" />
      case 'deepseek': return <Zap className="w-4 h-4" />
      case 'glm': return <Globe className="w-4 h-4" />
      default: return <Bot className="w-4 h-4" />
    }
  }

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'active': return <CheckCircle className="w-3 h-3 text-green-500" />
      case 'expired': return <XCircle className="w-3 h-3 text-red-500" />
      case 'error': return <AlertCircle className="w-3 h-3 text-yellow-500" />
      default: return <AlertCircle className="w-3 h-3 text-gray-500" />
    }
  }

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">حالة المزودين</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-8 bg-muted rounded"></div>
            <div className="h-8 bg-muted rounded"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm flex items-center gap-2">
          <Users className="w-4 h-4" />
          حالة المزودين والحسابات
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {providers.map((provider) => (
          <div key={provider.name} className="space-y-2">
            <div className="flex items-center gap-2">
              {getProviderIcon(provider.name)}
              <span className="font-medium capitalize">{provider.name}</span>
              <Badge variant="secondary" className="ml-auto">
                {provider.accounts.length} حسابات
              </Badge>
            </div>
            <div className="pl-6 space-y-1">
              {provider.accounts.slice(0, 5).map((account) => (
                <div key={account.label} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1">
                    {getStatusIcon(account.status)}
                    <span className="text-muted-foreground">{account.label}</span>
                  </div>
                  <span className="text-muted-foreground">
                    {account.remainingMinutes > 60 
                      ? `${Math.round(account.remainingMinutes / 60)}h` 
                      : `${account.remainingMinutes}m`}
                  </span>
                </div>
              ))}
              {provider.accounts.length > 5 && (
                <div className="text-xs text-muted-foreground pl-4">
                  +{provider.accounts.length - 5} حسابات أخرى
                </div>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
