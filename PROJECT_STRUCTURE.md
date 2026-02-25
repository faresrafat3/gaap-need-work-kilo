# GAAP Project Structure / هيكل مشروع GAAP

## 1. Directory Structure / هيكل المجلدات

```
gaap/
├── api/              # FastAPI endpoints / نقاط نهاية FastAPI
├── cli/              # Command line interface / واجهة سطر الأوامر
├── context/         # Context management / إدارة السياق
├── core/             # Core utilities / الأدوات الأساسية
├── healing/         # Self-healing system / نظام الإصلاح الذاتي
├── knowledge/        # Knowledge ingestion / استيعاب المعرفة
├── layers/          # OODA layers (L0-L3) / طبقات OODA
├── mad/             # Multi-agent debug / تصحيح الوكلاء المتعددين
├── maintenance/      # Technical debt / الديون التقنية
├── memory/          # Memory system / نظام الذاكرة
├── meta_learning/   # Meta-learning / التعلم الفوقي
├── observability/    # Tracing, metrics / التتبع والمقاييس
├── providers/       # LLM providers / موفرو LLM
├── research/        # Research engine / محرك البحث
├── routing/         # Smart routing / التوجيه الذكي
├── security/        # Security / الأمان
├── storage/         # Storage / التخزين
├── swarm/           # Swarm intelligence / ذكاء السرب
├── tools/           # Tool system / نظام الأدوات
├── validators/      # Validators / المدققون
└── gaap_engine.py   # Main engine / المحرك الرئيسي

frontend/
├── src/
│   ├── app/         # Next.js pages / صفحات Next.js
│   ├── components/  # React components / مكونات React
│   ├── hooks/       # Custom hooks / hooks مخصصة
│   ├── stores/      # Zustand stores / مخازن Zustand
│   ├── lib/         # Utilities / الأدوات
│   └── __tests__/   # Tests / الاختبارات
├── e2e/            # E2E tests with Playwright / اختبارات E2E
└── tests/           # Additional tests / اختبارات إضافية

tests/
├── unit/            # Unit tests / اختبارات الوحدة
├── integration/     # Integration tests / اختبارات التكامل
├── benchmarks/      # Performance tests / اختبارات الأداء
├── gauntlet/        # E2E gauntlets / اختبارات E2E الشاملة
└── scenarios/       # Test scenarios / سيناريوهات الاختبار
```

## 2. Key Files / الملفات الرئيسية

| File / الملف | Description / الوصف |
|---------------|----------------------|
| `gaap/gaap_engine.py` | Main engine / المحرك الرئيسي |
| `gaap/core/config.py` | Configuration / الإعدادات |
| `gaap/core/events.py` | Event system / نظام الأحداث |
| `gaap/__init__.py` | Package initialization / تهيئة الحزمة |

## 3. Configuration Files / ملفات الإعداد

| File / الملف | Description / الوصف |
|---------------|----------------------|
| `pyproject.toml` | Python project config / إعداد مشروع Python |
| `.env.example` | Environment template / قالب المتغيرات |
| `.gaap/config.yaml` | GAAP configuration / إعدادات GAAP |
| `.gaap_env` | Development environment / بيئة التطوير |
| `docker-compose.yml` | Docker services / خدمات Docker |
| `Dockerfile` | Container image / صورة الحاوية |

## 4. Development / التطوير

### Running Tests / تشغيل الاختبارات

```bash
# All tests / جميع الاختبارات
pytest

# Unit tests only / اختبارات الوحدة فقط
pytest tests/unit/

# Integration tests / اختبارات التكامل
pytest tests/integration/

# With coverage / مع التغطية
pytest --cov=gaap --cov-report=html

# E2E tests (requires frontend) / اختبارات E2E
cd frontend && playwright test
```

### Code Style / نمط الكود

- **Formatter**: Ruff / **المنسق**: Ruff
- **Linter**: Ruff / **المفتش**: Ruff
- **Type Checker**: mypy / **مدقق الأنواع**: mypy
- **Pre-commit Hooks**: Enabled / **خطافات Pre-commit**: مفعلة

```bash
# Format code / تنسيق الكود
ruff format

# Lint / التفتيش
ruff check .

# Type check / التحقق من الأنواع
mypy gaap/
```

### Contributing / المساهمة

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

راجع [CONTRIBUTING.md](./CONTRIBUTING.md) للإرشادات.
