'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  History, 
  MessageSquare, 
  Clock, 
  Download, 
  Trash2, 
  Eye,
  Search,
  Filter,
  FileJson,
  FileText,
  Calendar,
  Upload,
  CheckSquare,
  Square,
  X,
  FolderOpen,
  Sparkles,
  MoreVertical,
  Archive,
  AlertTriangle
} from 'lucide-react'
import { useGAAPStore, Session } from '@/lib/store'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

interface SessionWithMessages extends Session {
  messages?: Array<{
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: string
  }>
  messageCount?: number
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      type: 'spring' as const,
      stiffness: 100,
      damping: 15,
    },
  },
  exit: {
    opacity: 0,
    x: -20,
    scale: 0.95,
    transition: { duration: 0.2 },
  },
}

export function SessionsManagement() {
  const { sessions, addSession } = useGAAPStore()
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set())
  const [isBulkMode, setIsBulkMode] = useState(false)
  const [sessionDetail, setSessionDetail] = useState<SessionWithMessages | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [importPreview, setImportPreview] = useState<Session[] | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Enhanced search with multiple fields
  const filteredSessions = sessions.filter((session) => {
    const query = searchQuery.toLowerCase().trim()
    if (!query) {
      const matchesStatus = statusFilter === 'all' || session.status === statusFilter
      return matchesStatus
    }
    
    const matchesSearch = 
      session.name.toLowerCase().includes(query) ||
      session.id.toLowerCase().includes(query) ||
      new Date(session.createdAt).toLocaleDateString('ar-SA').includes(query)
    
    const matchesStatus = statusFilter === 'all' || session.status === statusFilter
    return matchesSearch && matchesStatus
  })

  // Sort sessions by date (newest first)
  const sortedSessions = [...filteredSessions].sort((a, b) => {
    const dateA = a.updatedAt ? new Date(a.updatedAt).getTime() : new Date(a.createdAt).getTime()
    const dateB = b.updatedAt ? new Date(b.updatedAt).getTime() : new Date(b.createdAt).getTime()
    return dateB - dateA
  })

  const handleViewSession = async (session: Session) => {
    try {
      const response = await fetch(`/api/sessions/${session.id}`)
      const data = await response.json()
      setSessionDetail({ ...session, messages: data.messages, messageCount: data.messages?.length || 0 })
    } catch (error) {
      console.error('Failed to fetch session:', error)
    }
  }

  // Export single session
  const handleExportSession = async (session: Session, format: 'json' | 'txt') => {
    try {
      const response = await fetch(`/api/sessions/${session.id}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format }),
      })
      const data = await response.json()
      
      if (data.success) {
        const messageCount = data.messages?.length || data.tasks?.length || 0
        const content = format === 'json' 
          ? JSON.stringify(data, null, 2)
          : `Session: ${session.name}\nDate: ${session.createdAt}\nMessages: ${messageCount}`
        
        const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/plain' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `session-${session.id}.${format}`
        a.click()
        URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error('Export error:', error)
    }
  }

  // Export all sessions
  const exportSessions = useCallback(async () => {
    try {
      const sessionsToExport = selectedSessions.size > 0 
        ? sessions.filter(s => selectedSessions.has(s.id))
        : sessions

      const exportData = {
        version: '1.0',
        exportedAt: new Date().toISOString(),
        sessionsCount: sessionsToExport.length,
        sessions: sessionsToExport.map(session => ({
          ...session,
          exportedAt: new Date().toISOString(),
        })),
      }

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `gaap-sessions-${new Date().toISOString().split('T')[0]}.json`
      a.click()
      URL.revokeObjectURL(url)

      // Clear selection after export
      setSelectedSessions(new Set())
      setIsBulkMode(false)
    } catch (error) {
      console.error('Export sessions error:', error)
    }
  }, [sessions, selectedSessions])

  // Import sessions
  const importSessions = useCallback(async (sessionsToImport: Session[]) => {
    try {
      for (const session of sessionsToImport) {
        const response = await fetch('/api/sessions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            name: session.name,
            description: session.description,
            priority: session.priority,
            tags: session.tags,
            config: session.config,
            metadata: session.metadata,
          }),
        })
        const data = await response.json()
        if (data.id) {
          addSession({
            ...data,
            id: data.id,
            name: data.name,
            description: data.description || '',
            status: data.status || 'pending',
            priority: data.priority || 'normal',
            tags: data.tags || [],
            config: data.config || {},
            metadata: data.metadata || {},
            createdAt: data.created_at || data.createdAt || new Date().toISOString(),
            updatedAt: data.updated_at || data.updatedAt || null,
            startedAt: data.started_at || data.startedAt || null,
            completedAt: data.completed_at || data.completedAt || null,
            progress: data.progress || 0,
            tasksTotal: data.tasks_total || data.tasksTotal || 0,
            tasksCompleted: data.tasks_completed || data.tasksCompleted || 0,
            tasksFailed: data.tasks_failed || data.tasksFailed || 0,
            costUsd: data.cost_usd || data.costUsd || 0,
            tokensUsed: data.tokens_used || data.tokensUsed || 0,
          })
        }
      }
      setShowImportDialog(false)
      setImportPreview(null)
    } catch (error) {
      console.error('Import sessions error:', error)
    }
  }, [addSession])

  const handleFileImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string)
        if (data.sessions && Array.isArray(data.sessions)) {
          setImportPreview(data.sessions)
          setShowImportDialog(true)
        } else if (Array.isArray(data)) {
          setImportPreview(data)
          setShowImportDialog(true)
        } else {
          alert('ملف غير صالح')
        }
      } catch (error) {
        alert('خطأ في قراءة الملف')
      }
    }
    reader.readAsText(file)
    event.target.value = ''
  }

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await fetch(`/api/sessions/${sessionId}`, {
        method: 'DELETE',
      })
    } catch (error) {
      console.error('Delete error:', error)
    }
  }

  const handleBulkDelete = async () => {
    try {
      for (const sessionId of selectedSessions) {
        await fetch(`/api/sessions/${sessionId}`, {
          method: 'DELETE',
        })
      }
      setSelectedSessions(new Set())
      setIsBulkMode(false)
      setShowDeleteConfirm(false)
    } catch (error) {
      console.error('Bulk delete error:', error)
    }
  }

  const handleArchiveSession = async (sessionId: string) => {
    try {
      await fetch(`/api/sessions/${sessionId}/pause`, {
        method: 'POST',
      })
    } catch (error) {
      console.error('Archive error:', error)
    }
  }

  const handleBulkArchive = async () => {
    try {
      for (const sessionId of selectedSessions) {
        await fetch(`/api/sessions/${sessionId}/pause`, {
          method: 'POST',
        })
      }
      setSelectedSessions(new Set())
      setIsBulkMode(false)
    } catch (error) {
      console.error('Bulk archive error:', error)
    }
  }

  const handleCreateSession = async () => {
    try {
      const response = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: 'جلسة جديدة',
          description: '',
        }),
      })
      const data = await response.json()
      if (data.id) {
        addSession({
          ...data,
          id: data.id,
          name: data.name,
          description: data.description || '',
          status: data.status || 'pending',
          priority: data.priority || 'normal',
          tags: data.tags || [],
          config: data.config || {},
          metadata: data.metadata || {},
          createdAt: data.created_at || data.createdAt || new Date().toISOString(),
          updatedAt: data.updated_at || data.updatedAt || null,
          startedAt: data.started_at || data.startedAt || null,
          completedAt: data.completed_at || data.completedAt || null,
          progress: data.progress || 0,
          tasksTotal: data.tasks_total || data.tasksTotal || 0,
          tasksCompleted: data.tasks_completed || data.tasksCompleted || 0,
          tasksFailed: data.tasks_failed || data.tasksFailed || 0,
          costUsd: data.cost_usd || data.costUsd || 0,
          tokensUsed: data.tokens_used || data.tokensUsed || 0,
        })
      }
    } catch (error) {
      console.error('Create session error:', error)
    }
  }

  const toggleSessionSelection = (sessionId: string) => {
    setSelectedSessions(prev => {
      const newSet = new Set(prev)
      if (newSet.has(sessionId)) {
        newSet.delete(sessionId)
      } else {
        newSet.add(sessionId)
      }
      return newSet
    })
  }

  const selectAllSessions = () => {
    if (selectedSessions.size === filteredSessions.length) {
      setSelectedSessions(new Set())
    } else {
      setSelectedSessions(new Set(filteredSessions.map(s => s.id)))
    }
  }

  const clearSelection = () => {
    setSelectedSessions(new Set())
    setIsBulkMode(false)
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
      case 'pending':
        return (
          <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
            نشط
          </Badge>
        )
      case 'completed':
        return (
          <Badge className="bg-blue-500/15 text-blue-400 border-blue-500/20 hover:bg-blue-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mr-1.5" />
            مكتمل
          </Badge>
        )
      case 'paused':
        return (
          <Badge className="bg-slate-500/15 text-slate-400 border-slate-500/20 hover:bg-slate-500/20">
            <Archive className="w-3 h-3 mr-1" />
            مؤرشف
          </Badge>
        )
      case 'failed':
        return (
          <Badge className="bg-red-500/15 text-red-400 border-red-500/20 hover:bg-red-500/20">
            <AlertTriangle className="w-3 h-3 mr-1" />
            فشل
          </Badge>
        )
      case 'cancelled':
        return (
          <Badge className="bg-gray-500/15 text-gray-400 border-gray-500/20 hover:bg-gray-500/20">
            <X className="w-3 h-3 mr-1" />
            ملغي
          </Badge>
        )
      default:
        return null
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
      case 'pending':
        return <Sparkles className="w-4 h-4 text-emerald-400" />
      case 'completed':
        return <CheckSquare className="w-4 h-4 text-blue-400" />
      case 'paused':
        return <Archive className="w-4 h-4 text-slate-400" />
      case 'failed':
        return <AlertTriangle className="w-4 h-4 text-red-400" />
      case 'cancelled':
        return <X className="w-4 h-4 text-gray-400" />
      default:
        return <MessageSquare className="w-4 h-4 text-primary" />
    }
  }

  const formatDate = (date: string | Date | null) => {
    if (!date) return 'غير متوفر'
    const d = new Date(date)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) {
      return `اليوم ${d.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' })}`
    } else if (diffDays === 1) {
      return `أمس ${d.toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' })}`
    } else if (diffDays < 7) {
      return d.toLocaleDateString('ar-SA', { weekday: 'long', hour: '2-digit', minute: '2-digit' })
    }
    
    return d.toLocaleDateString('ar-SA', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div>
            <h1 className="text-2xl font-bold gradient-text">إدارة الجلسات</h1>
            <p className="text-muted-foreground mt-1">عرض وإدارة جلسات المحادثة</p>
          </div>
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileImport}
              className="hidden"
            />
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              className="gap-2"
            >
              <Upload className="w-4 h-4" />
              استيراد
            </Button>
            <Button
              onClick={handleCreateSession}
              className="bg-gradient-to-r from-primary to-accent gap-2"
            >
              <MessageSquare className="w-4 h-4" />
              جلسة جديدة
            </Button>
          </div>
        </motion.div>

        {/* Filters & Bulk Actions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="gradient-border">
            <CardContent className="p-4">
              <div className="flex flex-col gap-4">
                {/* Search Row */}
                <div className="flex flex-col md:flex-row gap-4">
                  <div className="flex-1 relative group">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="بحث في الجلسات (الاسم، المعرف، التاريخ)..."
                      className="pl-10 pr-4 bg-secondary/50 border-secondary focus:border-primary transition-colors"
                    />
                    {searchQuery && (
                      <button
                        onClick={() => setSearchQuery('')}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-full md:w-44 bg-secondary/50">
                      <Filter className="w-4 h-4 mr-2" />
                      <SelectValue placeholder="الحالة" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">جميع الحالات</SelectItem>
                      <SelectItem value="pending">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400" />
                          قيد الانتظار
                        </span>
                      </SelectItem>
                      <SelectItem value="running">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                          نشط
                        </span>
                      </SelectItem>
                      <SelectItem value="completed">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-blue-400" />
                          مكتمل
                        </span>
                      </SelectItem>
                      <SelectItem value="paused">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-slate-400" />
                          متوقف
                        </span>
                      </SelectItem>
                      <SelectItem value="failed">
                        <span className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-red-400" />
                          فشل
                        </span>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Bulk Actions Row */}
                {filteredSessions.length > 0 && (
                  <div className="flex items-center justify-between pt-2 border-t border-border/50">
                    <div className="flex items-center gap-3">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setIsBulkMode(!isBulkMode)}
                        className={cn(
                          "gap-2",
                          isBulkMode && "bg-primary/10 text-primary"
                        )}
                      >
                        {isBulkMode ? <CheckSquare className="w-4 h-4" /> : <Square className="w-4 h-4" />}
                        تحديد متعدد
                      </Button>
                      
                      {isBulkMode && (
                        <motion.div
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          className="flex items-center gap-2"
                        >
                          <Checkbox
                            checked={selectedSessions.size === filteredSessions.length && filteredSessions.length > 0}
                            onCheckedChange={selectAllSessions}
                          />
                          <span className="text-sm text-muted-foreground">
                            {selectedSessions.size === 0 
                              ? 'تحديد الكل' 
                              : `تم تحديد ${selectedSessions.size}`}
                          </span>
                        </motion.div>
                      )}
                    </div>

                    {isBulkMode && selectedSessions.size > 0 && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="flex items-center gap-2"
                      >
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={exportSessions}
                          className="gap-2"
                        >
                          <FileJson className="w-4 h-4" />
                          تصدير
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={handleBulkArchive}
                          className="gap-2"
                        >
                          <Archive className="w-4 h-4" />
                          أرشفة
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => setShowDeleteConfirm(true)}
                          className="gap-2"
                        >
                          <Trash2 className="w-4 h-4" />
                          حذف
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={clearSelection}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </motion.div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Sessions Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <Card className="gradient-border overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <History className="w-5 h-5 text-primary" />
                  قائمة الجلسات
                </CardTitle>
                <Badge variant="secondary" className="font-normal">
                  {filteredSessions.length} جلسة
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-[500px]">
                <AnimatePresence mode="popLayout">
                  {sortedSessions.length === 0 ? (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="flex flex-col items-center justify-center py-16 px-4 text-center"
                    >
                      <div className="relative">
                        <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full" />
                        <div className="relative p-6 rounded-2xl bg-gradient-to-br from-primary/10 to-accent/10 border border-primary/20">
                          {searchQuery ? (
                            <Search className="w-12 h-12 text-primary" />
                          ) : (
                            <FolderOpen className="w-12 h-12 text-primary" />
                          )}
                        </div>
                      </div>
                      <h3 className="text-xl font-semibold mt-6 mb-2">
                        {searchQuery ? 'لا توجد نتائج مطابقة' : 'لا توجد جلسات'}
                      </h3>
                      <p className="text-muted-foreground max-w-sm">
                        {searchQuery 
                          ? 'جرب البحث بكلمات مختلفة أو امسح البحث'
                          : 'ابدأ محادثة جديدة لإنشاء جلسة أو استورد جلسات سابقة'
                        }
                      </p>
                      {searchQuery && (
                        <Button
                          variant="outline"
                          onClick={() => setSearchQuery('')}
                          className="mt-4 gap-2"
                        >
                          <X className="w-4 h-4" />
                          مسح البحث
                        </Button>
                      )}
                    </motion.div>
                  ) : (
                    <div className="p-4 grid gap-3">
                      {sortedSessions.map((session) => (
                        <motion.div
                          key={session.id}
                          variants={itemVariants}
                          layout
                          className={cn(
                            "group relative p-4 rounded-xl border transition-all duration-200",
                            "hover:shadow-lg hover:shadow-primary/5 hover:border-primary/30",
                            selectedSessions.has(session.id) 
                              ? "bg-primary/5 border-primary/40 shadow-md shadow-primary/10"
                              : "bg-card/50 border-border/50 hover:bg-card"
                          )}
                        >
                          <div className="flex items-start gap-4">
                            {/* Checkbox */}
                            <AnimatePresence>
                              {isBulkMode && (
                                <motion.div
                                  initial={{ opacity: 0, scale: 0 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  exit={{ opacity: 0, scale: 0 }}
                                  className="pt-1"
                                >
                                  <Checkbox
                                    checked={selectedSessions.has(session.id)}
                                    onCheckedChange={() => toggleSessionSelection(session.id)}
                                  />
                                </motion.div>
                              )}
                            </AnimatePresence>

                            {/* Icon */}
                            <div className={cn(
                              "p-3 rounded-xl transition-colors shrink-0",
                              session.status === 'running' && "bg-emerald-500/10",
                              session.status === 'pending' && "bg-emerald-500/10",
                              session.status === 'completed' && "bg-blue-500/10",
                              session.status === 'paused' && "bg-slate-500/10",
                              session.status === 'failed' && "bg-red-500/10",
                              session.status === 'cancelled' && "bg-gray-500/10"
                            )}>
                              {getStatusIcon(session.status)}
                            </div>

                            {/* Content */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                  <h4 className="font-semibold text-base truncate">
                                    {session.name}
                                  </h4>
                                  <div className="flex items-center flex-wrap gap-2 mt-2">
                                    {getStatusBadge(session.status)}
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <Clock className="w-3 h-3" />
                                      {formatDate(session.updatedAt || session.createdAt)}
                                    </span>
                                    <span className="flex items-center gap-1 text-xs text-muted-foreground">
                                      <MessageSquare className="w-3 h-3" />
                                      {session.tasksTotal || 0} مهمة
                                    </span>
                                  </div>
                                </div>

                                {/* Actions */}
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button 
                                      variant="ghost" 
                                      size="icon"
                                      className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity"
                                    >
                                      <MoreVertical className="w-4 h-4" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end">
                                    <DropdownMenuItem onClick={() => handleViewSession(session)}>
                                      <Eye className="w-4 h-4 mr-2" />
                                      عرض
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => handleExportSession(session, 'json')}>
                                      <FileJson className="w-4 h-4 mr-2" />
                                      تصدير JSON
                                    </DropdownMenuItem>
                                    <DropdownMenuItem onClick={() => handleExportSession(session, 'txt')}>
                                      <FileText className="w-4 h-4 mr-2" />
                                      تصدير نص
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem onClick={() => handleArchiveSession(session.id)}>
                                      <Archive className="w-4 h-4 mr-2" />
                                      أرشفة
                                    </DropdownMenuItem>
                                    <DropdownMenuItem 
                                      onClick={() => handleDeleteSession(session.id)}
                                      className="text-red-400 focus:text-red-400"
                                    >
                                      <Trash2 className="w-4 h-4 mr-2" />
                                      حذف
                                    </DropdownMenuItem>
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </AnimatePresence>
              </ScrollArea>
            </CardContent>
          </Card>
        </motion.div>

        {/* Session Detail Dialog */}
        <Dialog open={!!sessionDetail} onOpenChange={() => setSessionDetail(null)}>
          <DialogContent className="max-w-3xl max-h-[85vh] p-0 gap-0 overflow-hidden">
            <DialogHeader className="p-6 pb-4 border-b bg-gradient-to-r from-primary/5 to-accent/5">
              <DialogTitle className="flex items-center gap-2 text-xl">
                <MessageSquare className="w-5 h-5 text-primary" />
                {sessionDetail?.name}
              </DialogTitle>
              <DialogDescription className="flex items-center flex-wrap gap-4 mt-2">
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  {sessionDetail && formatDate(sessionDetail.createdAt)}
                </span>
                <span className="flex items-center gap-1">
                  <MessageSquare className="w-3 h-3" />
                  {sessionDetail?.messageCount} رسالة
                </span>
                {sessionDetail && getStatusBadge(sessionDetail.status)}
              </DialogDescription>
            </DialogHeader>
            
            <ScrollArea className="max-h-[50vh] p-6">
              <div className="space-y-4">
                {sessionDetail?.messages?.map((message, index) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className={cn(
                      "p-4 rounded-xl border",
                      message.role === 'user'
                        ? 'bg-primary/5 border-primary/20 ml-12'
                        : 'bg-secondary/50 border-border/50 mr-12'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <div className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium",
                        message.role === 'user' 
                          ? 'bg-primary/20 text-primary'
                          : 'bg-accent/20 text-accent'
                      )}>
                        {message.role === 'user' ? 'أ' : 'G'}
                      </div>
                      <span className="text-xs font-medium text-muted-foreground">
                        {message.role === 'user' ? 'أنت' : 'GAAP'}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {new Date(message.timestamp).toLocaleTimeString('ar-SA')}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed" dir="rtl">{message.content}</p>
                  </motion.div>
                ))}
              </div>
            </ScrollArea>

            <DialogFooter className="p-4 border-t bg-muted/50">
              <Button variant="outline" onClick={() => setSessionDetail(null)}>
                إغلاق
              </Button>
              {sessionDetail && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button className="gap-2">
                      <Download className="w-4 h-4" />
                      تصدير
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => handleExportSession(sessionDetail, 'json')}>
                      <FileJson className="w-4 h-4 mr-2" />
                      تصدير JSON
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExportSession(sessionDetail, 'txt')}>
                      <FileText className="w-4 h-4 mr-2" />
                      تصدير نص
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Bulk Delete Confirmation */}
        <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="w-5 h-5" />
                تأكيد الحذف
              </DialogTitle>
              <DialogDescription>
                هل أنت متأكد من حذف {selectedSessions.size} جلسة محددة؟
                <br />
                لا يمكن التراجع عن هذا الإجراء.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2">
              <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
                إلغاء
              </Button>
              <Button variant="destructive" onClick={handleBulkDelete}>
                <Trash2 className="w-4 h-4 mr-2" />
                حذف
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Import Preview Dialog */}
        <Dialog open={showImportDialog} onOpenChange={setShowImportDialog}>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary" />
                معاينة الاستيراد
              </DialogTitle>
              <DialogDescription>
                سيتم استيراد {importPreview?.length} جلسة
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[300px]">
              <div className="space-y-2 py-2">
                {importPreview?.map((session, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-3 rounded-lg bg-secondary/50 flex items-center gap-3"
                  >
                    <MessageSquare className="w-4 h-4 text-muted-foreground" />
                    <div className="flex-1">
                      <p className="font-medium text-sm truncate">{session.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {session.tasksTotal || 0} مهمة • {formatDate(session.createdAt)}
                      </p>
                    </div>
                    {getStatusBadge(session.status)}
                  </motion.div>
                ))}
              </div>
            </ScrollArea>
            <DialogFooter className="gap-2">
              <Button variant="outline" onClick={() => setShowImportDialog(false)}>
                إلغاء
              </Button>
              <Button onClick={() => importPreview && importSessions(importPreview)}>
                <Upload className="w-4 h-4 mr-2" />
                استيراد
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </TooltipProvider>
  )
}
