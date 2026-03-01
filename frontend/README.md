# GAAP Web Interface

واجهة ويب لمنصة GAAP (Generative Agentic Architecture Platform).

## 🚀 التثبيت

### 1. تثبيت الـ Dependencies

```bash
bun install
# أو
npm install
```

### 2. إعداد Environment Variables

```env
# Python GAAP Backend URL
PYTHON_API_URL=http://localhost:8000

# Enable real backend (true = proxy to Python, false = mock)
USE_REAL_BACKEND=true
```

### 3. تشغيل الـ Python Backend

```bash
cd /path/to/gaap-need-work-kilo
pip install -e .
uvicorn gaap.api.main:app --reload --port 8000
```

### 4. تشغيل الـ Frontend

```bash
bun run dev
# أو
npm run dev
```

## 📁 هيكل المشروع

```
src/
├── app/
│   ├── page.tsx              # الصفحة الرئيسية
│   ├── globals.css           # الأنماط العامة
│   └── api/                  # API Routes
│       ├── chat/             # Chat API
│       ├── research/         # Research API
│       ├── providers/        # Providers API
│       ├── sessions/         # Sessions API
│       └── health/           # Health Check
├── components/
│   ├── gaap/                 # GAAP Components
│   │   ├── Dashboard.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── ResearchModule.tsx
│   │   ├── ConfigurationPanel.tsx
│   │   ├── SessionsManagement.tsx
│   │   └── OODAVisualization.tsx
│   └── ui/                   # shadcn/ui Components
└── lib/
    ├── store.ts              # Zustand Store
    └── utils.ts              # Utilities
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | محادثة مع GAAP |
| `/api/research` | POST | بحث عميق |
| `/api/providers` | GET/POST | إدارة المزودين |
| `/api/sessions` | GET/POST | إدارة الجلسات |
| `/api/health` | GET | فحص حالة النظام |

## 🎨 الميزات

- **Dashboard** - لوحة تحكم مع OODA Visualization
- **Chat Interface** - محادثة تفاعلية مع streaming
- **Research Module** - بحث عميق مع ETS scoring
- **Configuration Panel** - إدارة الإعدادات
- **Sessions Management** - إدارة الجلسات

## 🔧 التقنيات

- Next.js 15 (App Router)
- TypeScript
- Tailwind CSS
- shadcn/ui Components
- Zustand (State Management)
- Framer Motion (Animations)

## 📝 ملاحظات

- لو `USE_REAL_BACKEND=false` أو Python مش شغال، هيستخدم mock data
- لو Python شغال، هيحول كل الـ requests للـ backend

## 📄 License

MIT
