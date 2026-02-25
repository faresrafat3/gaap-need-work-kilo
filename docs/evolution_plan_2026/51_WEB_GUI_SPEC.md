# TECHNICAL SPECIFICATION: Web GUI - Complete Control Interface (v1.0)

**Author:** System Architect
**Target:** Full-Stack Development
**Status:** READY FOR IMPLEMENTATION
**Priority:** High

## 1. Overview

Web GUI شامل للتحكم الكامل في GAAP system. يوفر:
- **Full Control**: كل إعدادات النظام قابلة للتعديل
- **Real-time Monitoring**: رؤية الـ thought process في الوقت الحقيقي
- **Multi-Session**: إدارة عدة جلسات بحث/عمل
- **Steering**: Pause/Resume/Veto capabilities
- **Export/Share**: تصدير ومشاركة النتائج

## 2. Architecture

### 2.1 Technology Stack
- **Backend**: FastAPI + WebSocket
- **Frontend**: Next.js 14 + React Flow + TailwindCSS
- **Real-time**: WebSocket for events
- **State**: React Query + Zustand

### 2.2 Layer Architecture
```
Frontend (React)
    │
    ├── WebSocket (Real-time Events)
    │
    └── REST API (CRUD Operations)
           │
        FastAPI Backend
           │
        Event System
           │
        GAAP Core (ConfigManager, SmartRouter, etc.)
```

## 3. API Endpoints

### Config API
- GET/PUT `/api/config` - Full config
- GET/PUT `/api/config/{module}` - Module config
- POST `/api/config/validate` - Validate config

### Providers API
- GET/POST `/api/providers` - List/Add providers
- GET/PUT/DELETE `/api/providers/{name}` - CRUD
- POST `/api/providers/{name}/test` - Test connection

### Sessions API
- GET/POST `/api/sessions` - List/Create
- GET/PUT/DELETE `/api/sessions/{id}` - CRUD
- POST `/api/sessions/{id}/pause` - Pause
- POST `/api/sessions/{id}/resume` - Resume
- POST `/api/sessions/{id}/export` - Export

### WebSocket Channels
- `/ws/events` - System events
- `/ws/ooda` - OODA loop visualization
- `/ws/steering` - Steering commands

## 4. Frontend Pages

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | System overview |
| Config | `/config` | Config editor |
| Providers | `/providers` | Provider management |
| Research | `/research` | Research interface |
| Sessions | `/sessions` | Session manager |
| Healing | `/healing` | Healing monitor |
| Memory | `/memory` | Memory stats |
| Debt | `/debt` | Technical debt |
| Budget | `/budget` | Budget dashboard |
| Security | `/security` | Security panel |

## 5. Implementation Phases

### Phase 1: Backend Foundation
- Event System (`gaap/core/events.py`)
- WebSocket Manager (`gaap/api/websocket.py`)
- Config API (`gaap/api/config.py`)
- Providers API (`gaap/api/providers.py`)
- Main Router (`gaap/api/main.py`)

### Phase 2: Extended APIs
- Healing, Memory, Debt, Budget, Security, Sessions, System APIs

### Phase 3: Frontend Foundation
- Next.js setup
- Layout components
- WebSocket hooks
- API client

### Phase 4: Core Features
- Dashboard
- Config Editor
- Provider Manager
- Research Interface

### Phase 5: Advanced Features
- OODA Visualizer
- Steering Controls
- Session Manager
- Export/Share

## 6. Extension Points

### Adding New API Module
```python
# gaap/api/new_module.py
router = APIRouter(prefix="/api/new-module")

# gaap/api/main.py
app.include_router(new_module_router)
```

### Adding New Event Type
```python
class EventType(Enum):
    NEW_EVENT = "new_event"

event_emitter.emit(EventType.NEW_EVENT, data)
```

### Adding New Frontend Page
```typescript
// frontend/src/app/new-page/page.tsx
export default function NewPage() { ... }
```

## 7. Theme System

Cyber-Noir theme with layer colors:
- L0 (Security): Dark Purple
- L1 (Strategic): Purple
- L2 (Tactical): Blue
- L3 (Execution): Green
- Healing: Orange
- Error: Red

## 8. Files to Create

### Backend (11 files)
- `gaap/core/events.py`
- `gaap/api/main.py`
- `gaap/api/websocket.py`
- `gaap/api/config.py`
- `gaap/api/providers.py`
- `gaap/api/healing.py`
- `gaap/api/memory.py`
- `gaap/api/debt.py`
- `gaap/api/budget.py`
- `gaap/api/security.py`
- `gaap/api/sessions.py`
- `gaap/api/system.py`

### Frontend (~40 files)
- `frontend/src/app/*` (10 pages)
- `frontend/src/components/*` (30+ components)
- `frontend/src/hooks/*` (6 hooks)
- `frontend/src/lib/*` (5 modules)
