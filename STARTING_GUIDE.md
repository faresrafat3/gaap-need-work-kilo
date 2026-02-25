# GAAP User Guide | دليل المستخدم GAAP

## 1. Quick Start | البدء السريع

### 1.1 Installation | التثبيت

```bash
# استنساخ المشروع | Clone the project
git clone https://github.com/gaap-system/gaap.git
cd gaap

# تثبيت Python | Install Python
pip install -e .

# تشغيل الفحص | Verify installation
gaap --help
```

```bash
# تشغيل الواجهة الأمامية | Frontend setup
cd frontend && npm install && npm run dev
```

### 1.2 Basic Usage | الاستخدام الأساسي

```bash
# محادثة سريعة | Quick chat
gaap chat "اكتب دالة للبحث الثنائي" "Write a binary search function"

# تشغيل مهمة | Run a task
gaap run "حل مشكلة البرمجة" "Solve a coding problem"

# البحث العميق | Deep research
gaap research " موضوع البحث" "Research topic"
```

### 1.3 First Command | الأمر الأول

```bash
# اختباري | Test command
gaap doctor
```

النتيجة | Output:
```
GAAP System Diagnostics
========================
System: OK
Providers: OK
Memory: OK
All systems operational!
```

---

## 2. CLI Usage | استخدام CLI

### 2.1 Basic Commands | الأوامر الأساسية

#### gaap run - تشغيل مهمة | Run a task

```bash
# تشغيل مهمة أساسية | Basic task
gaap run "اكتب Hello World" "Write Hello World"

# مع تحديد النموذج | With model
gaap run "Task" --model llama-3.3-70b

# مع ميزانية | With budget
gaap run "Task" --budget 5.0
```

#### gaap chat - محادثة تفاعلية | Interactive chat

```bash
# محادثة سريعة | Quick chat
gaap chat "سؤال" "Question"

# مع تحديد المزود | With provider
gaap chat "سؤال" --provider groq

# JSON output
gaap chat "سؤال" --format json
```

#### gaap config - الإعدادات | Configuration

```bash
# عرض الإعدادات | Show config
gaap config show

# تعيين قيمة | Set value
gaap config set default_budget 20.0
gaap config set system.log_level DEBUG

# تحميل ملف | Load file
gaap config load ./config.yaml
```

#### gaap providers - إدارة المزودين | Provider management

```bash
# عرض المزودين | List providers
gaap providers list

# اختبار مزود | Test provider
gaap providers test groq
gaap providers test --all

# معلومات المزود | Provider info
gaap providers info groq
```

---

### 2.2 Advanced Commands | الأوامر المتقدمة

#### gaap research - وضع البحث | Research mode

```bash
# بحث عميق | Deep research
gaap research "الذكاء الاصطناعي في الطب" "AI in medicine"

# مع تقارير | With reports
gaap research "Topic" --reports

# مع نموذج محدد | With specific model
gaap research "Topic" --model gemini-1.5-pro
```

#### gaap heal - أوامر الإصلاح | Healing commands

```bash
# عرض حالة الإصلاح | Healing status
gaap heal status

# تشغيل الإصلاح التلقائي | Run auto-heal
gaap heal run

# عرض السجل | View logs
gaap heal logs
```

#### gaap memory - إدارة الذاكرة | Memory management

```bash
# عرض حالات الذاكرة | Memory status
gaap memory status

# consolidation يدوي | Manual consolidation
gaap memory consolidate

# عرض إحصائيات | Statistics
gaap memory stats
```

#### gaap swarm - أوامر السرب | Swarm commands

```bash
# تشغيل سرب | Run swarm
gaap swarm run "مهمة" "Task"

# عرض الجلسات النشطة | Active sessions
gaap swarm sessions

# التحكم | Control
gaap swarm pause
gaap swarm resume
```

---

### 2.3 TUI Features | ميزات TUI

#### Fuzzy Search Menus | قوائم البحث الضبابي

- اضغط `Ctrl+K` لفتح البحث | Press `Ctrl+K` to open search
- اكتب للبحث الفوري | Type for instant search
- Arrow keys للتنقل | Arrow keys to navigate
- Enter للاختيار | Enter to select

#### Real-time Progress | التقدم الفوري

- شريط التقدم | Progress bar
- تحديثات الحالة | Status updates
- السجل الحي | Live logs

#### Task Receipts | إيصال المهام

```
┌─────────────────────────────────────┐
│ TASK RECEIPT                        │
├─────────────────────────────────────┤
│ ID:     req_abc123                  │
│ Status: SUCCESS                     │
│ Cost:   $0.023                      │
│ Time:   1.23s                       │
└─────────────────────────────────────┘
```

---

## 3. Web GUI Usage | استخدام واجهة الويب

### 3.1 Starting the Web Interface | تشغيل واجهة الويب

```bash
# تشغيل الخادم الخلفي | Backend
cd /home/fares/Projects/GAAP
uvicorn gaap.api.main:app --reload --port 8000

# تشغيل الواجهة الأمامية | Frontend (new terminal)
cd frontend && npm install && npm run dev
```

**رابط الوصول | Access URL:** http://localhost:3000

---

### 3.2 Pages Overview | نظرة عامة على الصفحات

| الصفحة | Page | الوصف | Description |
|--------|------|-------|-------------|
| Dashboard | / | لوحة التحكم - نظرة عامة على النظام | System overview |
| Config | /config | إعدادات النظام | Configuration management |
| Providers | /providers | إعداد مزودي LLM | LLM provider setup |
| Research | /research | واجهة البحث العميق | Deep research interface |
| Sessions | /sessions | جلسات التنفيذ | Execution sessions |
| Healing | /healing | حالة الإصلاح الذاتي | Self-healing status |
| Memory | /memory | مستويات الذاكرة | Memory tiers |
| Budget | /budget | تتبع التكاليف | Cost tracking |
| Security | /security | إعدادات الأمان | Security settings |
| Debt | /debt | الديون التقنية | Technical debt |

---

### 3.3 Features | الميزات

#### Pause/Resume Steering | التحكم بالإيقاف/الاستئناف

1. انتقل إلى الجلسة | Go to session
2. اضغط Pause لإيقاف | Press Pause to pause
3. أضف توجيه (اختياري) | Add steering (optional)
4. اضغط Resume للاستئناف | Press Resume to resume

#### Real-time Updates via WebSocket | التحديثات الفورية عبر WebSocket

```javascript
// الاتصال | Connect
const ws = new WebSocket('ws://localhost:8000/ws/events');

// استقبال الأحداث | Receive events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};
```

#### Export Functionality | وظيفة التصدير

| الصيغة | Format | الاستخدام | Use Case |
|--------|--------|-----------|----------|
| JSON | البيانات الخام | Raw data, programmatic use |
| Markdown | التوثيق | Documentation, reports |
| CSV | جداول البيانات | Spreadsheet analysis |

---

## 4. Configuration | التكوين

### 4.1 Environment Variables | متغيرات البيئة

```bash
# مفاتيح API | API Keys
export GROQ_API_KEY="your-key"
export CEREBRAS_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"
export GITHUB_TOKEN="your-token"

# إعدادات GAAP | GAAP Settings
export GAAP_ENVIRONMENT=production
export GAAP_LOG_LEVEL=INFO
export GAAP_BUDGET_MONTHLY=5000.0
export GAAP_BUDGET_DAILY=200.0
```

| المتغير | Variable | الوصف | Description |
|---------|----------|-------|-------------|
| `GAAP_ENVIRONMENT` | Environment | بيئة التشغيل | Runtime environment |
| `GAAP_LOG_LEVEL` | Log Level | مستوى السجل | Logging level |
| `GAAP_BUDGET_MONTHLY` | Monthly Budget | الميزانية الشهرية | Monthly budget limit |
| `GROQ_API_KEY` | Groq API Key | مفتاح Groq | Groq API key |
| `CEREBRAS_API_KEY` | Cerebras API Key | مفتاح Cerebras | Cerebras API key |

---

### 4.2 Configuration Files | ملفات التكوين

#### .gaap/config.yaml

```yaml
system:
  name: GAAP-Production
  environment: production
  log_level: INFO

budget:
  monthly_limit: 5000
  daily_limit: 200
  per_task_limit: 10

execution:
  max_parallel_tasks: 10
  genetic_twin_enabled: true
  self_healing_enabled: true

providers:
  - name: groq
    priority: 85
    enabled: true
  - name: cerebras
    priority: 95
    enabled: true
```

#### .gaap/constitution.yaml

```yaml
axioms:
  - name: safety_first
    description: Always prioritize safety
    level: CONSTRAINT
    added_at: '2026-01-01'
    
  - name: transparency
    description: Maintain transparency in decisions
    level: GUIDELINE
    added_at: '2026-01-01'
```

---

## 5. Architecture Overview | نظرة عامة على البنية

### 4-Layer OODA Architecture | بنية OODA ذات 4 طبقات

```
User Input → L0: Observe → L1: Orient → L2: Decide → L3: Act → Learn → Loop
```

| الطبقة | Layer | المكون | Component | المسؤولية | Responsibility |
|--------|-------|--------|------------|-----------|----------------|
| OBSERVE | المراقبة | PromptFirewall | أمن المسح وفهم النية | Security scanning, intent classification |
| ORIENT | التوجيه | StrategicToT | البحث العميق والاستراتيجية | Deep Research, Strategy Generation |
| DECIDE | القرار | TacticalDecomposer | تحليل المهام وبناء الرسم البياني | Task breakdown, DAG construction |
| ACT | التنفيذ | SpecializedExecutors | تنفيذ الإجراءات والتحقق | Action execution, Tool Synthesis |
| LEARN | التعلم | Metacognition | التعلم التأملي وتحديث الذاكرة | Reflective learning, memory storage |

---

### Memory System | نظام الذاكرة

| المستوى | Tier | الاسم | Name | الوصف | Description |
|---------|------|-------|------|-------|-------------|
| L0 | Working | الذاكرة العاملة | Working Memory | السياق النشط | Active context |
| L1 | Episodic | الذاكرة الوقائية | Episodic Memory | التجارب الأخيرة | Recent experiences |
| L2 | Semantic | الذاكرة الدلالية | Semantic Memory | الأنماط والمعرفة | Patterns and knowledge |
| L3 | Long-term | الذاكرة طويلة المدى | Long-term Memory | المبادئ والثوابت | Principles and constants |

---

### Meta-Learning | التعلم الفوقي

- التعلم المستمر منExecutions السابقة | Continuous learning from past executions
- تحديث السمعة | Reputation updates
- توليد axioms جديدة | New axiom generation
- تدقيق السلامة | Integrity auditing

---

## 6. Troubleshooting | استكشاف الأخطاء

### Common Issues | المشاكل الشائعة

| المشكلة | Issue | الحل | Solution |
|--------|-------|-----|----------|
| Connection failed | فشل الاتصال | تحقق من API key | Check API key |
| Rate limited | تحديد المعدل | انتظر وحاول مرة أخرى | Wait and retry |
| Budget exceeded | تجاوز الميزانية | راجع الإعدادات | Check settings |
| Memory full | الذاكرة ممتلئة | اجمع الذاكرة | Consolidate memory |

---

### Debug Mode | وضع التصحيح

```bash
# تفعيل التصحيح | Enable debug
gaap config set system.log_level DEBUG

# تشغيل مع التصحيح | Run with debug
gaap --debug run "Task"
```

---

### Logs Location | موقع السجلات

```bash
# السجلات | Logs
~/.gaap/logs/
├── gaap.log        # السجل العام | General log
├── error.log       # أخطاء | Errors
└── debug.log       # تصحيح | Debug (if enabled)
```

---

## Commands Quick Reference | مرجع سريع للأوامر

| الأمر | Command | الوصف | Description |
|-------|---------|-------|-------------|
| `gaap --help` | المساعدة | Help | عرض المساعدة |
| `gaap doctor` | الفحص | Diagnostics | تشغيل التشخيص |
| `gaap chat` | المحادثة | Chat | محادثة سريعة |
| `gaap run` | التشغيل | Run Task | تشغيل مهمة |
| `gaap research` | البحث | Research | بحث عميق |
| `gaap config` | الإعدادات | Config | إدارة الإعدادات |
| `gaap providers` | المزودون | Providers | إدارة المزودين |
| `gaap status` | الحالة | Status | حالة النظام |
| `gaap web` | الويب | Web | تشغيل واجهة الويب |

---

## Keyboard Shortcuts | اختصارات لوحة المفاتيح

| الاختصار | Shortcut | الإجراء | Action |
|----------|----------|---------|--------|
| `Ctrl+K` | / | فتح البحث | Open search |
| `Ctrl+1` | - | لوحة التحكم | Dashboard |
| `Ctrl+2` | - | الإعدادات | Config |
| `Ctrl+3` | - | المزودون | Providers |
| `Ctrl+4` | - | البحث | Research |
| `Ctrl+B` | - | تبديل الشريط الجانبي | Toggle sidebar |
| `Ctrl+R` | - | تحديث الصفحة | Refresh |
| `Shift+?` | - | مساعدة الاختصارات | Shortcuts help |
| `Escape` | - | إغلاق النافذة | Close modal |

---

**GAAP v0.9.0** | نظام الذكاء الاصطناعي المستقل | Autonomous AI System
