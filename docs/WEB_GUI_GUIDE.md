# GAAP Web Interface User Guide
## دليل واجهة الويب GAAP

---

## 1. Overview | نظرة عامة

The GAAP Web Interface (واجهة الويب GAAP) is a comprehensive control panel for managing and monitoring the GAAP system. It provides real-time visibility into the system's operations, configuration management, and steering capabilities.

**Purpose / الغرض:**
- **Full Control**: Modify all system settings in real-time
- **Real-time Monitoring**: Observe the AI's thought process as it works
- **Multi-Session Management**: Manage multiple research/execution sessions
- **Steering Capabilities**: Pause, resume, or veto running operations
- **Export & Share**: Export results in multiple formats

---

## 2. Getting Started | البدء

### Prerequisites | المتطلبات الأساسية
- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- npm or yarn

### Starting the Backend | تشغيل الخادم الخلفي

```bash
uvicorn gaap.api.main:app --reload --port 8000
```

### Starting the Frontend | تشغيل الواجهة الأمامية

```bash
cd frontend && npm run dev
```

### Access URLs | روابط الوصول

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## 3. Pages Overview | نظرة على الصفحات

The Web GUI consists of 10 main pages:

### 3.1 Dashboard (/)
**لوحة التحكم**

The main landing page providing a system-wide overview.

**Screenshot Placeholder:** *[Dashboard showing system metrics, budget gauge, provider health, and recent events]*

**Features:**
- System status overview
- Active sessions count
- Budget usage visualization
- Provider health status
- Recent events log

---

### 3.2 Config (/config)
**إعدادات النظام**

Manage all system configuration settings.

**Screenshot Placeholder:** *[Config editor with module tabs and JSON editor]*

**Features:**
- Module-specific configuration
- JSON editor with validation
- Configuration validation before saving
- Import/export configurations

---

### 3.3 Providers (/providers)
**إدارة المزودين**

Manage LLM provider connections and settings.

**Screenshot Placeholder:** *[Provider list with status indicators and test buttons]*

**Features:**
- Add/remove providers
- Test provider connections
- View provider health status
- Configure API keys and endpoints

---

### 3.4 Research (/research)
**محرك البحث العميق**

Interface for the Deep Discovery Engine.

**Screenshot Placeholder:** *[Research interface with query input, OODA flow visualization, and results panel]*

**Features:**
- Submit research queries
- Real-time OODA loop visualization
- View research progress and findings
- Export research results

---

### 3.5 Sessions (/sessions)
**إدارة الجلسات**

Manage execution sessions.

**Screenshot Placeholder:** *[Sessions list with status, duration, and action buttons]*

**Features:**
- View all active and past sessions
- Create new sessions
- Pause/resume/terminate sessions
- Export session data

---

### 3.6 Healing (/healing)
**الإصلاح الذاتي**

Monitor self-healing operations.

**Screenshot Placeholder:** *[Healing dashboard with issue history and auto-fix status]*

**Features:**
- View healing history
- Configure healing rules
- Monitor auto-fix operations
- Manual intervention options

---

### 3.7 Memory (/memory)
**الذاكرة**

View memory tiers and consolidation status.

**Screenshot Placeholder:** *[Memory visualization showing L0/L1/L2/L3 tiers and statistics]*

**Features:**
- Memory tier visualization
- Consolidation status
- Memory statistics
- Manual consolidation triggers

---

### 3.8 Budget (/budget)
**الميزانية**

Monitor budget usage and alerts.

**Screenshot Placeholder:** *[Budget dashboard with usage charts and alert configuration]*

**Features:**
- Daily/monthly usage tracking
- Budget alerts configuration
- Usage predictions
- Cost breakdown by provider

---

### 3.9 Security (/security)
**الأمان**

Manage security settings.

**Screenshot Placeholder:** *[Security panel with validation rules and audit logs]*

**Features:**
- Security rule configuration
- Audit log viewer
- Validation settings
- Access control management

---

### 3.10 Debt (/debt)
**الديون التقنية**

Track technical debt and remediation.

**Screenshot Placeholder:** *[Debt tracker with priority matrix and status indicators]*

**Features:**
- Technical debt inventory
- Priority tracking
- Remediation suggestions
- Debt trend analysis

---

## 4. Features | الميزات

### 4.1 Real-time Updates | التحديثات الفورية

The interface uses WebSocket connections for real-time updates:

| Channel | Purpose |
|---------|---------|
| `/ws/events` | System-wide events |
| `/ws/ooda` | OODA loop visualization |
| `/ws/steering` | Steering commands |

**Connection Status Indicator:**
- Green dot = Connected
- Yellow dot = Reconnecting
- Red dot = Disconnected

---

### 4.2 OODA Loop Visualization | تصور حلقة OODA

The OODA (Observe-Orient-Decide-Act-Learn) loop is visualized in real-time during research operations.

**Screenshot Placeholder:** *[OODA flow diagram with animated current phase indicator]*

**Phases:**
1. **OBSERVE** (مراقبة) - Cyan - Gathering information
2. **ORIENT** (توجيه) - Yellow - Analyzing and contextualizing
3. **DECIDE** (اتخاذ قرار) - Purple - Choosing action
4. **ACT** (تنفيذ) - Green - Executing decision
5. **LEARN** (تعلم) - Blue - Updating knowledge

---

### 4.3 Steering Controls | عناصر التحكم

Control running sessions in real-time:

#### Pause Button | زر الإيقاف المؤقت
- Pauses the current operation
- Preserves session state
- Allows intervention

#### Resume Button | زر الاستئناف
- Resumes from paused state
- Optionally accepts new instructions
- Restarts OODA loop if steered

#### Veto Button | زر النقض
- Terminates the current action
- Cannot be undone
- Logs the veto for audit

**Screenshot Placeholder:** *[Steering controls panel with Pause, Resume, Veto, and Steer buttons]*

---

### 4.4 Export Functionality | وظيفة التصدير

Export data in multiple formats:

| Format | Use Case |
|--------|----------|
| JSON | Raw data, programmatic use |
| Markdown | Documentation, reports |
| CSV | Spreadsheet analysis |

**How to Export:**
1. Click the Export button
2. Select desired format
3. File downloads automatically

---

### 4.5 Keyboard Shortcuts | اختصارات لوحة المفاتيح

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` or `/` | Open search |
| `Ctrl+1` | Go to Dashboard |
| `Ctrl+2` | Go to Config |
| `Ctrl+3` | Go to Providers |
| `Ctrl+4` | Go to Research |
| `Ctrl+5` | Go to Sessions |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+R` | Refresh page |
| `Shift+?` | Open keyboard shortcuts help |
| `Escape` | Close modal |

---

### 4.6 Dark Theme (Cyber-Noir) | المظهر الداكن

The interface uses the Cyber-Noir theme with layer-specific colors:

| Layer | Color | Purpose |
|-------|-------|---------|
| L0 (Security) | Dark Purple | Security operations |
| L1 (Strategic) | Purple | Strategic planning |
| L2 (Tactical) | Blue | Tactical decisions |
| L3 (Execution) | Green | Execution actions |
| Healing | Orange | Self-healing |
| Error | Red | Errors and warnings |

---

## 5. Dashboard Widgets | أدوات لوحة التحكم

### 5.1 Active Sessions | الجلسات النشطة

Displays currently running sessions with:
- Session ID
- Status (running, paused, idle)
- Duration
- Current task

**Screenshot Placeholder:** *[Active sessions widget with session cards]*

---

### 5.2 Budget Gauge | مقياس الميزانية

Visual representation of budget usage:
- Circular progress indicator
- Percentage used
- Daily/monthly breakdown
- Warning indicators (>70% warning, >90% critical)

**Screenshot Placeholder:** *[Budget gauge showing 45% usage with green indicator]*

---

### 5.3 Provider Health | صحة المزودين

Status of configured LLM providers:
- Provider name
- Connection status
- Response time
- Last check timestamp

**Screenshot Placeholder:** *[Provider health widget with status badges]*

---

### 5.4 Recent Events | الأحداث الأخيرة

Real-time log of system events:
- Event type
- Timestamp
- Source
- Brief description

**Screenshot Placeholder:** *[Recent events list with event type badges and timestamps]*

---

### 5.5 System Status | حالة النظام

Overall system health indicators:
- Core services status
- Memory usage
- Active connections
- Uptime

**Screenshot Placeholder:** *[System status widget with service indicators]*

---

## 6. Steering Controls | عناصر التحكم

### How to Control Running Sessions | كيفية التحكم في الجلسات

#### Pausing a Session | إيقاف جلسة مؤقتاً
1. Navigate to the session or dashboard
2. Click the **Pause** button
3. The session state is preserved
4. A pause event is logged

#### Resuming a Session | استئناف جلسة
1. Click the **Resume** button on a paused session
2. Optionally enter a steering instruction
3. Click **Apply** to resume
4. The OODA loop continues from where it stopped

#### Providing Steering Instructions | توفير توجيهات
1. Click the **Steer** button
2. Enter your instruction (e.g., "Use PostgreSQL instead of MongoDB")
3. Click **Apply**
4. The session resumes with the new instruction incorporated

#### Vetoing an Action | نقض إجراء
1. Click the **Veto** button
2. Confirm the veto action
3. The current action is terminated
4. Cannot be undone

**Note:** Steering instructions restart the OODA loop with the new guidance incorporated into the decision-making process.

---

## 7. Accessibility Features | ميزات الوصول

### 7.1 Skip to Content | تخطي إلى المحتوى

A "Skip to main content" link is available for keyboard users at the top of every page. Press `Tab` on page load to access it.

---

### 7.2 ARIA Labels | تسميات ARIA

All interactive elements include proper ARIA labels:
- Buttons have descriptive labels
- Forms have associated labels
- Status indicators have live regions
- Navigation has landmarks

---

### 7.3 Screen Reader Support | دعم قارئ الشاشة

The interface supports screen readers through:
- Live announcements for status changes
- Proper heading hierarchy
- Descriptive link text
- Form error announcements

**Announcer System:**
- Polite announcements for non-urgent updates
- Assertive announcements for critical alerts

---

### 7.4 Keyboard Navigation | التنقل بلوحة المفاتيح

Full keyboard navigation support:
- Tab through all interactive elements
- Enter/Space to activate buttons
- Arrow keys for list navigation
- Escape to close modals
- Focus indicators visible on all elements

---

## 8. Troubleshooting | استكشاف الأخطاء

### 8.1 Connection Issues | مشاكل الاتصال

**Symptom:** Frontend shows "Disconnected" status

**Solutions:**
1. Check if backend is running: `curl http://localhost:8000/health`
2. Verify CORS settings in backend configuration
3. Check browser console for errors
4. Ensure correct port (8000) is used

---

### 8.2 WebSocket Disconnection | انقطاع WebSocket

**Symptom:** Real-time updates stop working

**Solutions:**
1. Check network connectivity
2. Verify WebSocket endpoint is accessible
3. Refresh the page to reconnect
4. Check backend logs for WebSocket errors

**Manual Reconnection:**
```javascript
// WebSocket automatically reconnects, but you can refresh the page
window.location.reload();
```

---

### 8.3 Provider Test Failures | فشل اختبار المزود

**Symptom:** Provider test returns error

**Solutions:**
1. Verify API key is correct
2. Check API endpoint URL
3. Ensure sufficient quota/balance
4. Verify network access to provider API
5. Check provider status page for outages

**Common Error Messages:**
| Error | Solution |
|-------|----------|
| "Invalid API key" | Regenerate and update API key |
| "Rate limited" | Wait and retry, or upgrade plan |
| "Network timeout" | Check network connectivity |
| "Insufficient quota" | Add credits to provider account |

---

### 8.4 Export Not Working | التصدير لا يعمل

**Symptom:** Export button doesn't download file

**Solutions:**
1. Check browser download settings
2. Disable popup blockers
3. Check browser console for errors
4. Try a different browser

---

### 8.5 Config Validation Errors | أخطاء التحقق من التكوين

**Symptom:** Config save fails with validation error

**Solutions:**
1. Use the built-in validation before saving
2. Check JSON syntax with a validator
3. Verify all required fields are present
4. Check the API documentation for correct schema

---

## 9. API Quick Reference | مرجع سريع للـ API

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET/PUT | System configuration |
| `/api/providers` | GET/POST | List/add providers |
| `/api/providers/{name}/test` | POST | Test provider |
| `/api/sessions` | GET/POST | List/create sessions |
| `/api/sessions/{id}/pause` | POST | Pause session |
| `/api/sessions/{id}/resume` | POST | Resume session |
| `/api/sessions/{id}/export` | POST | Export session |

### WebSocket Channels

```javascript
// Connect to events channel
const ws = new WebSocket('ws://localhost:8000/ws/events');

// Receive events
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};

// Send ping
ws.send(JSON.stringify({ type: 'ping' }));
```

---

## 10. Support | الدعم

For additional support:
- Check the [API Reference](./API_REFERENCE.md)
- Review the [Architecture Documentation](./ARCHITECTURE.md)
- Consult the [Development Guide](./DEVELOPMENT.md)
- Report issues on the project repository
