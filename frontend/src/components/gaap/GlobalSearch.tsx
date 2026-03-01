'use client'

import { useState, useEffect } from 'react'
import { Search, X, Command } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent } from '@/components/ui/dialog'

const quickActions = [
  { id: 'dashboard', label: 'لوحة التحكم', shortcut: '1' },
  { id: 'chat', label: 'المحادثة', shortcut: '2' },
  { id: 'research', label: 'البحث', shortcut: '3' },
  { id: 'validators', label: 'الفحص', shortcut: '4' },
  { id: 'context', label: 'السياق', shortcut: '5' },
  { id: 'knowledge', label: 'المعرفة', shortcut: '6' },
  { id: 'maintenance', label: 'الصيانة', shortcut: '7' },
  { id: 'swarm', label: 'السرب', shortcut: '8' },
]

export function GlobalSearch() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const router = useRouter()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(true)
      }
      if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const filteredActions = quickActions.filter(action =>
    action.label.toLowerCase().includes(query.toLowerCase())
  )

  const handleSelect = (id: string) => {
    setOpen(false)
    setQuery('')
    window.dispatchEvent(new CustomEvent('navigate', { detail: id }))
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="gap-2 text-muted-foreground"
        onClick={() => setOpen(true)}
      >
        <Search className="w-4 h-4" />
        <span className="hidden sm:inline">بحث...</span>
        <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px]">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="p-0 gap-0 max-w-md">
          <div className="flex items-center border-b px-3">
            <Search className="w-4 h-4 mr-2 text-muted-foreground" />
            <Input
              placeholder="ابحث أو انتقل إلى..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="border-0 focus-visible:ring-0"
            />
            <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
              <X className="w-4 h-4" />
            </Button>
          </div>
          
          <div className="max-h-64 overflow-y-auto p-2">
            {filteredActions.length === 0 ? (
              <p className="p-4 text-center text-sm text-muted-foreground">
                لا توجد نتائج
              </p>
            ) : (
              <div className="space-y-1">
                {filteredActions.map((action) => (
                  <button
                    key={action.id}
                    onClick={() => handleSelect(action.id)}
                    className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-muted text-sm"
                  >
                    <span>{action.label}</span>
                    <kbd className="text-xs text-muted-foreground">{action.shortcut}</kbd>
                  </button>
                ))}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
