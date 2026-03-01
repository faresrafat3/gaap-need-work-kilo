'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Settings,
  Server,
  DollarSign,
  RefreshCw,
  Database,
  Plus,
  Trash2,
  Edit2,
  Check,
  X,
  Play,
  Power,
  AlertCircle,
  CheckCircle,
  Loader2,
  Monitor,
  Cpu,
  Globe,
  Keyboard,
  Sun,
  Moon,
  Palette,
  Bell,
  Shield,
  Zap,
  Clock,
  Save,
  Command,
  Option,
  ArrowRightLeft,
  Search,
  Fullscreen,
  RotateCcw,
  Sparkles,
  FileCode,
  Braces,
  Terminal
} from 'lucide-react'
import { useGAAPStore } from '@/lib/store'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'

interface ShortcutGroup {
  name: string
  shortcuts: { keys: string[]; action: string }[]
}

const keyboardShortcuts: ShortcutGroup[] = [
  {
    name: 'العامة',
    shortcuts: [
      { keys: ['⌘', 'K'], action: 'فتح القائمة السريعة' },
      { keys: ['⌘', '/'], action: 'عرض اختصارات لوحة المفاتيح' },
      { keys: ['⌘', 'B'], action: 'تبديل الشريط الجانبي' },
      { keys: ['ESC'], action: 'إغلاق النافذة/العودة' },
    ],
  },
  {
    name: 'المحادثة',
    shortcuts: [
      { keys: ['⌘', 'Enter'], action: 'إرسال الرسالة' },
      { keys: ['Shift', 'Enter'], action: 'سطر جديد' },
      { keys: ['⌘', 'Shift', 'N'], action: 'محادثة جديدة' },
      { keys: ['⌘', '↑'], action: 'الرد على الرسالة السابقة' },
      { keys: ['⌘', 'L'], action: 'مسح المحادثة' },
    ],
  },
  {
    name: 'التنقل',
    shortcuts: [
      { keys: ['⌘', '1-5'], action: 'التبديل بين التبويبات' },
      { keys: ['⌘', 'T'], action: 'تبويب جديد' },
      { keys: ['⌘', 'W'], action: 'إغلاق التبويب' },
      { keys: ['⌘', '['], action: 'العودة للخلف' },
      { keys: ['⌘', ']'], action: 'التقدم للأمام' },
    ],
  },
  {
    name: 'الإعدادات',
    shortcuts: [
      { keys: ['⌘', ','], action: 'فتح الإعدادات' },
      { keys: ['⌘', 'I'], action: 'معلومات النظام' },
      { keys: ['⌘', 'R'], action: 'إعادة تحميل' },
      { keys: ['F11'], action: 'تبديل ملء الشاشة' },
    ],
  },
]

const themes = [
  { id: 'system', name: 'تلقائي', icon: Monitor, description: 'يتبع إعدادات النظام' },
  { id: 'light', name: 'فاتح', icon: Sun, description: 'وضع النهار' },
  { id: 'dark', name: 'داكن', icon: Moon, description: 'وضع الليل' },
]

const languages = [
  { id: 'ar', name: 'العربية', flag: '🇸🇦' },
  { id: 'en', name: 'English', flag: '🇺🇸' },
  { id: 'fr', name: 'Français', flag: '🇫🇷' },
  { id: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { id: 'tr', name: 'Türkçe', flag: '🇹🇷' },
]

const features = [
  { id: 'autoSave', name: 'الحفظ التلقائي', description: 'حفظ المحادثات تلقائياً', icon: Save },
  { id: 'smartSuggestions', name: 'اقتراحات ذكية', description: 'اقتراحات السياق الذكية', icon: Sparkles },
  { id: 'syntaxHighlight', name: 'تلوين الكود', description: 'تمييز صيغة الكود البرمجي', icon: FileCode },
  { id: 'autoComplete', name: 'إكمال تلقائي', description: 'إكمال الكود والنصوص', icon: Braces },
  { id: 'notifications', name: 'الإشعارات', description: 'إشعارات النظام والتحديثات', icon: Bell },
  { id: 'soundEffects', name: 'المؤثرات الصوتية', description: 'أصوات الإشعارات والتفاعل', icon: Zap },
]

export function ConfigurationPanel() {
  const { config, updateConfig, providers, updateProvider } = useGAAPStore()
  const [testingProvider, setTestingProvider] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null)
  const [editingProvider, setEditingProvider] = useState<(typeof providers)[0] | null>(null)
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [activeTheme, setActiveTheme] = useState('system')
  const [activeLanguage, setActiveLanguage] = useState('ar')
  const [enabledFeatures, setEnabledFeatures] = useState({
    autoSave: true,
    smartSuggestions: true,
    syntaxHighlight: true,
    autoComplete: false,
    notifications: true,
    soundEffects: false,
  })
  const [newProvider, setNewProvider] = useState({
    name: '',
    model: '',
    apiKey: '',
    baseUrl: '',
  })

  const handleToggleFeature = (featureId: string) => {
    setEnabledFeatures(prev => ({
      ...prev,
      [featureId]: !prev[featureId as keyof typeof prev],
    }))
  }

  const handleTestProvider = async (providerId: string) => {
    setTestingProvider(providerId)
    setTestResult(null)

    try {
      const response = await fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'test', providerId }),
      })

      const data = await response.json()
      setTestResult(data)
    } catch (error) {
      setTestResult({ success: false, message: 'حدث خطأ في الاتصال' })
    } finally {
      setTestingProvider(null)
    }
  }

  const handleToggleProvider = async (providerId: string) => {
    try {
      await fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'toggle', providerId }),
      })

      const provider = providers.find(p => p.id === providerId)
      if (provider) {
        updateProvider(providerId, {
          health: provider.health === 'healthy' ? 'unhealthy' : 'healthy'
        })
      }
    } catch (error) {
      console.error('Toggle error:', error)
    }
  }

  const handleAddProvider = async () => {
    try {
      const response = await fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: newProvider.name,
          provider_type: 'chat',
          models: [newProvider.model],
          default_model: newProvider.model,
        }),
      })

      if (response.ok) {
        setIsAddDialogOpen(false)
        setNewProvider({ name: '', model: '', apiKey: '', baseUrl: '' })
      }
    } catch (error) {
      console.error('Add provider error:', error)
    }
  }

  const handleDeleteProvider = async (providerId: string) => {
    try {
      await fetch(`/api/providers/${providerId}`, {
        method: 'DELETE',
      })
    } catch (error) {
      console.error('Delete error:', error)
    }
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Tabs defaultValue="general" className="space-y-4">
          <TabsList className="grid grid-cols-5 w-full">
            <TabsTrigger value="general" className="gap-2">
              <Settings className="w-4 h-4" />
              <span className="hidden sm:inline">العامة</span>
            </TabsTrigger>
            <TabsTrigger value="providers" className="gap-2">
              <Server className="w-4 h-4" />
              <span className="hidden sm:inline">المزودين</span>
            </TabsTrigger>
            <TabsTrigger value="memory" className="gap-2">
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">الذاكرة</span>
            </TabsTrigger>
            <TabsTrigger value="system" className="gap-2">
              <Cpu className="w-4 h-4" />
              <span className="hidden sm:inline">النظام</span>
            </TabsTrigger>
            <TabsTrigger value="shortcuts" className="gap-2">
              <Keyboard className="w-4 h-4" />
              <span className="hidden sm:inline">الاختصارات</span>
            </TabsTrigger>
          </TabsList>

          {/* General Tab */}
          <TabsContent value="general" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              {/* Appearance Section */}
              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Palette className="w-5 h-5 text-primary" />
                    المظهر
                  </CardTitle>
                  <CardDescription>
                    تخصيص مظهر واجهة المستخدم
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <Label>السمة</Label>
                    <div className="grid grid-cols-3 gap-2">
                      {themes.map((theme) => (
                        <button
                          key={theme.id}
                          onClick={() => setActiveTheme(theme.id)}
                          className={`flex flex-col items-center gap-2 p-3 rounded-lg border transition-all ${
                            activeTheme === theme.id
                              ? 'border-primary bg-primary/10'
                              : 'border-border hover:border-primary/50'
                          }`}
                        >
                          <theme.icon className="w-5 h-5" />
                          <span className="text-sm font-medium">{theme.name}</span>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {themes.find(t => t.id === activeTheme)?.description}
                    </p>
                  </div>

                  <Separator />

                  <div className="space-y-3">
                    <Label>اللغة</Label>
                    <div className="grid grid-cols-5 gap-2">
                      {languages.map((lang) => (
                        <button
                          key={lang.id}
                          onClick={() => setActiveLanguage(lang.id)}
                          className={`flex flex-col items-center gap-1 p-2 rounded-lg border transition-all ${
                            activeLanguage === lang.id
                              ? 'border-primary bg-primary/10'
                              : 'border-border hover:border-primary/50'
                          }`}
                        >
                          <span className="text-lg">{lang.flag}</span>
                          <span className="text-xs font-medium">{lang.name}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Features Section */}
              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Zap className="w-5 h-5 text-primary" />
                    الميزات
                  </CardTitle>
                  <CardDescription>
                    تفعيل/تعطيل ميزات النظام
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {features.map((feature) => (
                      <motion.div
                        key={feature.id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-primary/5 hover:border-primary/20 transition-all"
                      >
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-md bg-primary/10">
                            <feature.icon className="w-4 h-4 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium text-sm">{feature.name}</p>
                            <p className="text-xs text-muted-foreground">{feature.description}</p>
                          </div>
                        </div>
                        <Switch
                          checked={enabledFeatures[feature.id as keyof typeof enabledFeatures]}
                          onCheckedChange={() => handleToggleFeature(feature.id)}
                        />
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Budget Card */}
            <Card className="gradient-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <DollarSign className="w-5 h-5 text-primary" />
                  الميزانية والفوترة
                </CardTitle>
                <CardDescription>
                  إدارة حدود الاستخدام والتكاليف
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label>الحد اليومي ($)</Label>
                    <Input
                      type="number"
                      defaultValue={100}
                      className="bg-secondary/50"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>الحد الشهري ($)</Label>
                    <Input
                      type="number"
                      defaultValue={5000}
                      className="bg-secondary/50"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>تنبيه عند (%)</Label>
                    <div className="flex items-center gap-2">
                      <Slider defaultValue={[80]} max={100} step={5} className="flex-1" />
                      <span className="text-sm text-muted-foreground w-10">80%</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/30">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-green-500/10">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    </div>
                    <div>
                      <p className="font-medium">الاستخدام الحالي</p>
                      <p className="text-sm text-muted-foreground">$2,450 من $5,000 هذا الشهر</p>
                    </div>
                  </div>
                  <Badge variant="outline" className="bg-green-500/10 text-green-500">
                    49% مستخدم
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Providers Tab */}
          <TabsContent value="providers" className="space-y-4">
            <Card className="gradient-border">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Server className="w-5 h-5 text-primary" />
                      إدارة المزودين
                    </CardTitle>
                    <CardDescription>
                      إضافة وتعديل واختبار مزودي الذكاء الاصطناعي
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <RotateCcw className="w-4 h-4 mr-2" />
                      تحديث الكل
                    </Button>
                    <Button
                      onClick={() => setIsAddDialogOpen(true)}
                      className="bg-gradient-to-r from-primary to-accent"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      إضافة مزود
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {providers.map((provider, index) => (
                    <motion.div
                      key={provider.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="group relative p-4 rounded-xl bg-card border border-border hover:border-primary/30 transition-all hover:shadow-lg"
                    >
                      <div className="flex items-start gap-4">
                        {/* Status Indicator */}
                        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                          provider.health === 'healthy' ? 'bg-green-500/10' :
                          provider.health === 'unhealthy' ? 'bg-red-500/10' : 'bg-gray-500/10'
                        }`}>
                          <Server className={`w-6 h-6 ${
                            provider.health === 'healthy' ? 'text-green-500' :
                            provider.health === 'unhealthy' ? 'text-red-500' : 'text-gray-500'
                          }`} />
                        </div>

                        {/* Provider Info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-semibold text-base">{provider.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {provider.models[0] || 'N/A'}
                            </Badge>
                            <div className={`w-2 h-2 rounded-full ${
                              provider.health === 'healthy' ? 'bg-green-500 animate-pulse' :
                              provider.health === 'unhealthy' ? 'bg-red-500' : 'bg-gray-500'
                            }`} />
                          </div>
                          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="w-3.5 h-3.5" />
                              {(provider.stats?.latency as number) || 0}ms
                            </span>
                            <span className="flex items-center gap-1">
                              <ArrowRightLeft className="w-3.5 h-3.5" />
                              {(provider.stats?.requests as number) || 0} طلب
                            </span>
                            <span className={`flex items-center gap-1 ${
                              provider.health === 'healthy' ? 'text-green-500' :
                              provider.health === 'unhealthy' ? 'text-red-500' : 'text-gray-500'
                            }`}>
                              <Power className="w-3.5 h-3.5" />
                              {provider.health === 'healthy' ? 'نشط' :
                               provider.health === 'unhealthy' ? 'خطأ' : 'غير نشط'}
                            </span>
                          </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleTestProvider(provider.id)}
                            disabled={testingProvider === provider.id}
                            className="hover:bg-primary/10"
                          >
                            {testingProvider === provider.id ? (
                              <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setEditingProvider(provider)}
                            className="hover:bg-primary/10"
                          >
                            <Edit2 className="w-4 h-4" />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleToggleProvider(provider.id)}
                            className="hover:bg-primary/10"
                          >
                            <Power className={`w-4 h-4 ${
                              provider.health === 'healthy' ? 'text-green-500' : 'text-gray-400'
                            }`} />
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDeleteProvider(provider.id)}
                            className="hover:bg-red-500/10"
                          >
                            <Trash2 className="w-4 h-4 text-red-500" />
                          </Button>
                        </div>
                      </div>

                      {testResult && testingProvider === provider.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className={`mt-3 p-3 rounded-lg flex items-center gap-2 text-sm ${
                            testResult.success
                              ? 'bg-green-500/10 text-green-500 border border-green-500/20'
                              : 'bg-red-500/10 text-red-500 border border-red-500/20'
                          }`}
                        >
                          {testResult.success ? (
                            <CheckCircle className="w-4 h-4" />
                          ) : (
                            <AlertCircle className="w-4 h-4" />
                          )}
                          {testResult.message}
                        </motion.div>
                      )}
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Memory Tab */}
          <TabsContent value="memory" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Database className="w-5 h-5 text-primary" />
                    إدارة الذاكرة
                  </CardTitle>
                  <CardDescription>
                    إعدادات الذاكرة والتخزين المؤقت
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <Label>حد الذاكرة (MB)</Label>
                      <span className="text-sm text-muted-foreground">{config.memoryLimit} MB</span>
                    </div>
                    <Slider
                      value={[config.memoryLimit]}
                      onValueChange={([value]) => updateConfig({ memoryLimit: value })}
                      max={16384}
                      step={256}
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="p-4 rounded-lg bg-secondary/30 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-blue-500" />
                          المستخدم
                        </span>
                        <span className="font-medium">1,024 MB</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-green-500" />
                          المتاح
                        </span>
                        <span className="font-medium text-green-500">3,072 MB</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full bg-purple-500" />
                          الجلسات المخزنة
                        </span>
                        <span className="font-medium">127</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1">
                      <RefreshCw className="w-4 h-4 mr-2" />
                      مسح الذاكرة
                    </Button>
                    <Button variant="outline" className="flex-1 text-red-500 hover:bg-red-500/10">
                      <Trash2 className="w-4 h-4 mr-2" />
                      مسح الجلسات
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <RefreshCw className="w-5 h-5 text-primary" />
                    الإصلاح الذاتي
                  </CardTitle>
                  <CardDescription>
                    إعدادات الإصلاح التلقائي للأخطاء
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div>
                      <Label className="text-base">الإصلاح التلقائي</Label>
                      <p className="text-sm text-muted-foreground">
                        إصلاح الأخطاء تلقائياً عند حدوثها
                      </p>
                    </div>
                    <Switch
                      checked={config.autoHealing}
                      onCheckedChange={(checked) => updateConfig({ autoHealing: checked })}
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <Label>الحد الأقصى للمحاولات</Label>
                      <span className="text-sm text-muted-foreground">{config.maxConcurrent} محاولة</span>
                    </div>
                    <Slider
                      value={[config.maxConcurrent]}
                      onValueChange={([value]) => updateConfig({ maxConcurrent: value })}
                      max={20}
                      step={1}
                    />
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <Label>مهلة الانتظار (ثواني)</Label>
                      <span className="text-sm text-muted-foreground">{config.timeoutMs / 1000} ثانية</span>
                    </div>
                    <Slider
                      value={[config.timeoutMs / 1000]}
                      onValueChange={([value]) => updateConfig({ timeoutMs: value * 1000 })}
                      max={120}
                      step={5}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* System Tab */}
          <TabsContent value="system" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Shield className="w-5 h-5 text-primary" />
                    الأمان والخصوصية
                  </CardTitle>
                  <CardDescription>
                    إعدادات الأمان وحماية البيانات
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div>
                      <p className="font-medium">تشفير المحادثات</p>
                      <p className="text-sm text-muted-foreground">تشفير جميع البيانات</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div>
                      <p className="font-medium">التحقق بخطوتين</p>
                      <p className="text-sm text-muted-foreground">تأمين إضافي للحساب</p>
                    </div>
                    <Switch />
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                    <div>
                      <p className="font-medium">حذف تلقائي</p>
                      <p className="text-sm text-muted-foreground">حذف المحادثات القديمة</p>
                    </div>
                    <Switch />
                  </div>
                </CardContent>
              </Card>

              <Card className="gradient-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Terminal className="w-5 h-5 text-primary" />
                    معلومات النظام
                  </CardTitle>
                  <CardDescription>
                    تفاصيل النظام والإصدار
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-2 rounded-lg bg-secondary/30">
                      <span className="text-sm text-muted-foreground">الإصدار</span>
                      <Badge variant="outline">v2.5.0</Badge>
                    </div>
                    <div className="flex justify-between items-center p-2 rounded-lg bg-secondary/30">
                      <span className="text-sm text-muted-foreground">البيئة</span>
                      <Badge variant="outline" className="bg-green-500/10 text-green-500">Production</Badge>
                    </div>
                    <div className="flex justify-between items-center p-2 rounded-lg bg-secondary/30">
                      <span className="text-sm text-muted-foreground">آخر تحديث</span>
                      <span className="text-sm">2024-01-15</span>
                    </div>
                    <div className="flex justify-between items-center p-2 rounded-lg bg-secondary/30">
                      <span className="text-sm text-muted-foreground">معرف النسخة</span>
                      <code className="text-xs bg-secondary px-2 py-1 rounded">a1b2c3d</code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card className="gradient-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Settings className="w-5 h-5 text-primary" />
                  إجراءات النظام
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-3">
                  <Button variant="outline">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    تحديث النظام
                  </Button>
                  <Button variant="outline">
                    <Database className="w-4 h-4 mr-2" />
                    تصدير البيانات
                  </Button>
                  <Button variant="outline" className="text-red-500 hover:bg-red-500/10">
                    <RotateCcw className="w-4 h-4 mr-2" />
                    إعادة تعيين
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Shortcuts Tab */}
          <TabsContent value="shortcuts" className="space-y-4">
            <Card className="gradient-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Keyboard className="w-5 h-5 text-primary" />
                  اختصارات لوحة المفاتيح
                </CardTitle>
                <CardDescription>
                  قائمة بجميع اختصارات لوحة المفاتيح المتاحة
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px] pr-4">
                  <div className="space-y-6">
                    {keyboardShortcuts.map((group, groupIndex) => (
                      <div key={group.name}>
                        <h4 className="font-semibold mb-3 text-sm text-muted-foreground uppercase tracking-wider">
                          {group.name}
                        </h4>
                        <div className="space-y-2">
                          {group.shortcuts.map((shortcut, shortcutIndex) => (
                            <motion.div
                              key={shortcut.action}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: (groupIndex * 0.1) + (shortcutIndex * 0.05) }}
                              className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                            >
                              <span className="text-sm">{shortcut.action}</span>
                              <div className="flex items-center gap-1">
                                {shortcut.keys.map((key, keyIndex) => (
                                  <span key={key} className="flex items-center">
                                    <kbd className="px-2 py-1 text-xs font-mono bg-background border rounded shadow-sm">
                                      {key}
                                    </kbd>
                                    {keyIndex < shortcut.keys.length - 1 && (
                                      <span className="mx-1 text-muted-foreground">+</span>
                                    )}
                                  </span>
                                ))}
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
          <Button variant="outline">
            <RotateCcw className="w-4 h-4 mr-2" />
            إعادة الافتراضي
          </Button>
          <Button className="bg-gradient-to-r from-primary to-accent">
            <Save className="w-4 h-4 mr-2" />
            حفظ التغييرات
          </Button>
        </div>
      </motion.div>

      {/* Add Provider Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="w-5 h-5" />
              إضافة مزود جديد
            </DialogTitle>
            <DialogDescription>
              أدخل بيانات المزود الجديد للاتصال بخدمة الذكاء الاصطناعي
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>اسم المزود</Label>
              <Input
                value={newProvider.name}
                onChange={(e) => setNewProvider({ ...newProvider, name: e.target.value })}
                placeholder="مثال: OpenAI"
                className="bg-secondary/50"
              />
            </div>
            <div className="space-y-2">
              <Label>النموذج</Label>
              <Input
                value={newProvider.model}
                onChange={(e) => setNewProvider({ ...newProvider, model: e.target.value })}
                placeholder="مثال: gpt-4-turbo"
                className="bg-secondary/50"
              />
            </div>
            <div className="space-y-2">
              <Label>مفتاح API</Label>
              <Input
                type="password"
                value={newProvider.apiKey}
                onChange={(e) => setNewProvider({ ...newProvider, apiKey: e.target.value })}
                placeholder="sk-..."
                className="bg-secondary/50"
              />
            </div>
            <div className="space-y-2">
              <Label>عنوان API (اختياري)</Label>
              <Input
                value={newProvider.baseUrl}
                onChange={(e) => setNewProvider({ ...newProvider, baseUrl: e.target.value })}
                placeholder="https://api.example.com"
                className="bg-secondary/50"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
              إلغاء
            </Button>
            <Button
              onClick={handleAddProvider}
              className="bg-gradient-to-r from-primary to-accent"
            >
              إضافة المزود
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Provider Dialog */}
      <Dialog open={!!editingProvider} onOpenChange={() => setEditingProvider(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit2 className="w-5 h-5" />
              تعديل المزود
            </DialogTitle>
            <DialogDescription>
              تعديل بيانات مزود {editingProvider?.name}
            </DialogDescription>
          </DialogHeader>
          {editingProvider && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>اسم المزود</Label>
                <Input
                  defaultValue={editingProvider.name}
                  className="bg-secondary/50"
                />
              </div>
              <div className="space-y-2">
                <Label>النموذج</Label>
                <Input
                  defaultValue={editingProvider.models[0] || ''}
                  className="bg-secondary/50"
                />
              </div>
              <div className="space-y-2">
                <Label>حالة المزود</Label>
                <div className="flex items-center gap-2 p-3 rounded-lg bg-secondary/30">
                  <div className={`w-3 h-3 rounded-full ${
                    editingProvider.health === 'healthy' ? 'bg-green-500' :
                    editingProvider.health === 'unhealthy' ? 'bg-red-500' : 'bg-gray-500'
                  }`} />
                  <Badge variant="outline" className={
                    editingProvider.health === 'healthy' ? 'bg-green-500/10 text-green-500' :
                    editingProvider.health === 'unhealthy' ? 'bg-red-500/10 text-red-500' :
                    'bg-gray-500/10 text-gray-500'
                  }>
                    {editingProvider.health === 'healthy' ? 'نشط' :
                     editingProvider.health === 'unhealthy' ? 'خطأ' : 'غير نشط'}
                  </Badge>
                </div>
              </div>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingProvider(null)}>
              إلغاء
            </Button>
            <Button className="bg-gradient-to-r from-primary to-accent">
              <Save className="w-4 h-4 mr-2" />
              حفظ التغييرات
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
